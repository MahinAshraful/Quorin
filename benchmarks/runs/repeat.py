"""Generalized fresh-subprocess orchestrator for tail-latency claims.

Step 7's ``step7_gc_tail_repeat.py`` proved that single-run pytest-benchmark
of rare tail events is statistically useless (one sample of a heavy-tailed
distribution). This script generalizes the pattern: read a bench-spec YAML,
spawn N=20 fresh subprocesses, aggregate per-percentile statistics, write
structured JSON for the README to quote.

What 'fresh' means:
  * Each subprocess is a brand-new Python interpreter via subprocess.run.
  * GC state, page cache, OS scheduler state, Python module-level state — all
    new per subprocess. Variance from these accumulates within a single
    process; a fresh process resets them.
  * NOT fresh: Numba's on-disk JIT cache. Each subprocess's autouse fixture
    in test_assembly_benchmark.py / test_batch_benchmark.py calls
    ``pyforge.assembly.prewarm()`` BEFORE pytest-benchmark's timed loop.
    JIT compile is paid in setup, not measurement.

Run::

    python benchmarks/runs/repeat.py \\
        --bench-spec benchmarks/runs/specs/headline_4_field_warm.yml \\
        --num-runs 20 \\
        --output benchmarks/results/n20/headline_4_field_warm_n20.json

Bench-spec YAML schema::

    name: headline_4_field_warm                # informational
    description: ...                            # informational
    test_path: benchmarks/.../test_x.py::test_x # pytest-style path
    percentiles: [50, 95, 99, 999, 9999]        # which to record per run
    warmup_runs: 1                              # discard first N runs
    timeout_seconds: 600                        # per-subprocess timeout

Aggregation:
  * Per run: ``np.percentile(stats.data, q)`` for each requested q.
  * Across N runs: ``median`` per percentile (the "median run"), ``max(max)``
    for absolute worst, ``stddev`` of p99 to inform headroom decisions.

Environment inheritance (per Step 16 plan, locked):
  Each subprocess inherits the parent's full os.environ via an explicit
  ``env=os.environ.copy()`` kwarg. This is intentional — the bench-spec
  YAML can't carry per-environment data like Redis URLs, and we don't want
  the orchestrator to second-guess what the operator set. To avoid
  accidental PYFORGE_RUN_LARGE_BENCH=1 leakage from a stale parent shell,
  the orchestrator logs all PYFORGE_RUN_* env vars at startup so the
  operator sees what's being passed through.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_NUM_RUNS = 20
DEFAULT_TIMEOUT_SECONDS = 600


def load_bench_spec(spec_path: Path) -> dict[str, Any]:
    """Load + minimally validate a bench-spec YAML. Raises ValueError on missing fields."""
    raw = yaml.safe_load(spec_path.read_text())
    if not isinstance(raw, dict):
        raise ValueError(f"bench spec {spec_path} did not parse as a mapping")
    required = {"name", "test_path", "percentiles"}
    missing = required - set(raw)
    if missing:
        raise ValueError(f"bench spec {spec_path} missing required fields: {sorted(missing)}")
    pcts = raw["percentiles"]
    if not isinstance(pcts, list) or not all(isinstance(p, (int, float)) for p in pcts):
        raise ValueError(f"bench spec {spec_path} percentiles must be a list of numbers")
    raw.setdefault("warmup_runs", 0)
    raw.setdefault("timeout_seconds", DEFAULT_TIMEOUT_SECONDS)
    raw.setdefault("description", "")
    return raw


def run_one_subprocess(
    spec: dict[str, Any],
    json_path: Path,
    *,
    env: dict[str, str],
) -> dict[str, float]:
    """Spawn a fresh pytest subprocess; return per-percentile dict for the bench.

    Each invocation writes its raw round data to ``json_path`` via
    ``--benchmark-json``. We then read that JSON, find the matching bench by
    test_path basename + function-name match, and compute percentiles from
    ``stats.data``.
    """
    cmd = [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "--no-header",
        spec["test_path"],
        "--benchmark-only",
        "--benchmark-save-data",
        f"--benchmark-json={json_path}",
    ]
    res = subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=spec["timeout_seconds"],
        env=env,
        check=False,
    )
    if res.returncode != 0:
        raise RuntimeError(
            f"subprocess failed for {spec['test_path']!r} (rc={res.returncode}):\n"
            f"--- stdout ---\n{res.stdout}\n--- stderr ---\n{res.stderr}"
        )
    if not json_path.exists():
        raise RuntimeError(
            f"subprocess for {spec['test_path']!r} did not write {json_path} "
            "(possible: pytest collected no tests). Check --bench-spec test_path."
        )
    bench_data = json.loads(json_path.read_text())
    benches = bench_data.get("benchmarks") or []
    if not benches:
        raise RuntimeError(f"no benchmarks recorded in {json_path}")
    # If multiple benches matched (parametrize variants), use the first whose
    # name matches the test_path's "::function_name" tail. If unspecified or
    # a single bench, use that.
    target = _select_target_bench(benches, spec["test_path"])
    raw = target["stats"].get("data")
    if raw is None:
        raise RuntimeError(
            f"bench {target['name']!r} in {json_path} missing stats.data; "
            "pytest-benchmark must run with --benchmark-save-data (orchestrator passes it)"
        )
    raw_arr = np.asarray(raw)
    out: dict[str, float] = {}
    for pct in spec["percentiles"]:
        out[f"p{pct}"] = float(np.percentile(raw_arr, _normalize_percentile(pct)))
    out["max"] = float(raw_arr.max())
    out["min"] = float(raw_arr.min())
    out["median"] = float(np.median(raw_arr))
    out["n_samples"] = int(raw_arr.size)
    return out


def _select_target_bench(benches: list[dict[str, Any]], test_path: str) -> dict[str, Any]:
    """Pick the bench matching the test_path's ``::function`` tail, else first."""
    if "::" in test_path:
        target_name = test_path.rsplit("::", 1)[1]
        # Strip any trailing parametrize since pytest-benchmark records the
        # parametrized variant in `name` (e.g. test_x[1000]).
        target_bare = target_name.split("[", 1)[0]
        for b in benches:
            if b["name"] == target_name or b["name"].split("[", 1)[0] == target_bare:
                return b
    return benches[0]


def _normalize_percentile(pct: float) -> float:
    """Convert a YAML-friendly percentile spec to np.percentile's q value.

    YAML convention: 50, 95, 99 are p50/p95/p99. 999 is p99.9, 9999 is p99.99.
    np.percentile takes q in [0, 100], so 999 -> 99.9, 9999 -> 99.99.
    """
    if pct <= 100:
        return float(pct)
    # 999 -> 99.9, 9999 -> 99.99 (drop the leading repeat)
    s = str(int(pct))
    return float(s[0] + "9." + s[2:])


def aggregate(runs: list[dict[str, float]]) -> dict[str, float]:
    """median-of-{percentile, max, median, ...} + stddev(p99) + max-of-max.

    Output is a flat dict so it can land in JSON unchanged. The "median" key
    intentionally collides with the per-run "median" — at the aggregate level
    it means "median of run-medians."
    """
    if not runs:
        raise ValueError("aggregate() requires at least one run")
    keys = list(runs[0].keys())
    out: dict[str, float] = {}
    for k in keys:
        vals = [r[k] for r in runs]
        if k.startswith("p") or k in {"median", "min"}:
            out[f"median_{k}"] = float(np.median(vals))
        elif k == "max":
            out["max_of_max"] = float(np.max(vals))
            out["median_max"] = float(np.median(vals))
        elif k == "n_samples":
            # Sum is the meaningful aggregate (total samples seen across all runs).
            out["total_samples"] = int(np.sum(vals))
    # stddev(p99) is the headroom-decision input.
    p99s = [r["p99"] for r in runs if "p99" in r]
    if p99s:
        out["stddev_p99"] = float(np.std(p99s, ddof=1)) if len(p99s) > 1 else 0.0
    return out


def _log_env_preflight() -> None:
    """Log PYFORGE_RUN_* env state so accidental leakage is visible to the operator."""
    relevant = {k: v for k, v in os.environ.items() if k.startswith("PYFORGE_")}
    if relevant:
        print("orchestrator inheriting PYFORGE_* env (intentional, see module docstring):")
        for k in sorted(relevant):
            print(f"  {k}={relevant[k]}")
    else:
        print("orchestrator: no PYFORGE_* env vars set in parent")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="benchmarks/runs/repeat.py", description=__doc__)
    parser.add_argument("--bench-spec", type=Path, required=True, help="path to a bench-spec YAML")
    parser.add_argument(
        "--num-runs",
        type=int,
        default=DEFAULT_NUM_RUNS,
        help=f"fresh subprocesses to spawn (default: {DEFAULT_NUM_RUNS})",
    )
    parser.add_argument("--output", type=Path, required=True, help="output JSON path")
    args = parser.parse_args(argv)

    spec = load_bench_spec(args.bench_spec)
    print(f"orchestrator: spec={spec['name']} test={spec['test_path']}")
    print(f"orchestrator: num_runs={args.num_runs} timeout/run={spec['timeout_seconds']}s")
    _log_env_preflight()

    # Explicit env=os.environ.copy() per the env-inheritance contract above.
    env = os.environ.copy()

    # Per-subprocess JSON output goes to a tempfile.TemporaryDirectory that
    # auto-cleans on exit (success or error). Each per-run JSON contains the
    # full pytest-benchmark stats.data array (raw round timings) — multi-MB
    # at high round counts. We extract per-run percentile dicts into
    # `runs` (small, ~hundreds of bytes each) and that's what gets written
    # to args.output. Keeping the per-run JSONs around offers no debugging
    # value beyond what pytest-benchmark's stderr already shows on failure.
    runs: list[dict[str, float]] = []
    t0 = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="pyforge_repeat_") as tmp_dir_str:
        tmp_dir = Path(tmp_dir_str)
        for i in range(args.num_runs):
            run_id = f"{i:03d}"
            json_path = tmp_dir / f"run_{run_id}.json"
            try:
                run_metrics = run_one_subprocess(spec, json_path, env=env)
            except RuntimeError as e:
                print(f"  run {run_id} FAILED: {e}", file=sys.stderr)
                return 1
            runs.append(run_metrics)
            elapsed = time.monotonic() - t0
            avg_per_run = elapsed / (i + 1)
            eta = avg_per_run * (args.num_runs - i - 1)
            print(
                f"  run {run_id}: p99={run_metrics.get('p99', 0) * 1e6:.1f}us "
                f"max={run_metrics['max'] * 1e6:.1f}us "
                f"n={run_metrics['n_samples']} "
                f"(elapsed {elapsed:.0f}s, eta {eta:.0f}s)"
            )

    # Trim warmup runs from aggregation but keep them in raw output for traceability.
    warmup = int(spec.get("warmup_runs", 0))
    aggregated_input = runs[warmup:] if warmup < len(runs) else runs
    if warmup >= len(runs):
        print(
            f"  WARN: warmup_runs={warmup} >= num_runs={len(runs)}; aggregating all runs",
            file=sys.stderr,
        )
    agg = aggregate(aggregated_input)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            {
                "spec": spec,
                "num_runs": args.num_runs,
                "warmup_runs_discarded": warmup if warmup < len(runs) else 0,
                "raw_runs": runs,
                "aggregate": agg,
                "elapsed_seconds": time.monotonic() - t0,
            },
            indent=2,
        )
    )
    print(f"\norchestrator: wrote {args.output}")
    print(f"  aggregate p99 (median-of-runs): {agg.get('median_p99', 0) * 1e6:.1f}us")
    if "stddev_p99" in agg:
        print(f"  stddev(p99) across runs:        {agg['stddev_p99'] * 1e6:.1f}us")
    print(f"  max_of_max:                     {agg.get('max_of_max', 0) * 1e6:.1f}us")
    return 0


if __name__ == "__main__":
    sys.exit(main())
