"""Regression-gate enforcement for pytest-benchmark output.

Compares per-bench percentile latencies against thresholds in
``benchmarks/regression/tier1.yml`` (always-run, PR gate) and optionally
``benchmarks/regression/tier2.yml`` (env-gated heavy benches; with
``--include-tier2``).

Run::

    uv run python benchmarks/regression/check.py [--strict] [--include-tier2]

Exit codes:
    0  all gated benches passed (or non-strict and missing benches)
    1  at least one gate breached, OR (with --strict) at least one gate
       missing from the benchmark JSON

Required input:
    pytest-benchmark JSON with ``--benchmark-save-data`` enabled. Without
    that flag, the JSON's per-bench ``stats.data`` array is absent and
    percentile computation is impossible. (B1: ``stats`` does not include
    ``q99`` by default.)

Schema (per gate entry):
    Any subset of ``{p50_seconds, p95_seconds, p99_seconds, p999_seconds,
    p9999_seconds}``. Each present field is gated independently against
    ``np.percentile(stats.data, q)``. Missing fields are skipped.

Name resolution (E2 two-stage lookup):
    1. Exact match against the bench's ``name`` field.
    2. Strip parametrize suffix (everything from ``[`` to end) and retry.

Tier merging (L5 duplicate-key guard):
    Loads tier1, optionally tier2; raises ``ValueError`` if any key
    appears in both. Each bench belongs in exactly one tier.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
TIER1_PATH = REPO_ROOT / "benchmarks" / "regression" / "tier1.yml"
TIER2_PATH = REPO_ROOT / "benchmarks" / "regression" / "tier2.yml"
OUTPUT_PATH = REPO_ROOT / "benchmarks" / "results" / "regression_check.json"

# R1: percentile-field name -> np.percentile q value. A threshold entry
# may carry any subset; each present field is gated independently.
_PERCENTILE_FIELDS: dict[str, float] = {
    "p50_seconds": 50.0,
    "p95_seconds": 95.0,
    "p99_seconds": 99.0,
    "p999_seconds": 99.9,
    "p9999_seconds": 99.99,
}


def _load_thresholds(
    include_tier2: bool,
    *,
    tier1_path: Path = TIER1_PATH,
    tier2_path: Path = TIER2_PATH,
) -> dict[str, dict[str, Any]]:
    """Load and merge tier1.yml (and tier2.yml if requested).

    L5: detects duplicate keys across tiers and raises. dict.update would
    silently let tier2 win on collision, hiding misconfigured gates.

    The path kwargs are for testability; production callers use the defaults.
    """
    t1 = yaml.safe_load(tier1_path.read_text()) or {}
    if not include_tier2:
        return dict(t1)
    t2 = yaml.safe_load(tier2_path.read_text()) or {}
    overlap = set(t1) & set(t2)
    if overlap:
        raise ValueError(
            f"benchmark gates duplicated across tier1.yml and tier2.yml: "
            f"{sorted(overlap)}. Each bench belongs in exactly one tier."
        )
    return {**t1, **t2}


def _resolve_threshold(thresholds: dict[str, dict[str, Any]], name: str) -> dict[str, Any] | None:
    """E2 two-stage lookup: exact match wins; fallback strips parametrize suffix."""
    if name in thresholds:
        return thresholds[name]
    bare = name.split("[", 1)[0]
    if bare != name and bare in thresholds:
        return thresholds[bare]
    return None


def _bench_percentile(bench: dict[str, Any], q: float) -> float:
    """Compute the q-th percentile from raw round data.

    B1: requires ``--benchmark-save-data`` so ``stats.data`` is populated.
    Without it, pytest-benchmark's default JSON has only summary stats
    (min/max/mean/median/q1/q3/iqr) — NO q99 / q999 fields exist.
    """
    data = bench["stats"].get("data")
    if data is None:
        raise RuntimeError(
            f"bench {bench['name']!r} missing stats.data; rerun pytest with --benchmark-save-data"
        )
    return float(np.percentile(np.asarray(data), q))


def _newest_bench_json() -> Path | None:
    """Find the newest pytest-benchmark JSON output (mtime desc).

    pytest-benchmark's storage layout depends on the --benchmark-storage flag.
    With ``--benchmark-storage=benchmarks/results``, JSON is written to
    ``benchmarks/results/<machine>/<run>.json`` (no ``.benchmarks/`` subdir
    because the storage path replaces that). Without the flag, the default
    is ``<rootdir>/.benchmarks/<machine>/<run>.json``. We support both via
    a fallback search.
    """
    base = REPO_ROOT / "benchmarks" / "results"
    candidates: list[Path] = []
    if base.exists():
        # With --benchmark-storage=benchmarks/results
        candidates.extend(base.glob("Linux-CPython-*/*.json"))
        # With pytest-benchmark default (.benchmarks/ subdir under storage)
        candidates.extend(base.glob(".benchmarks/Linux-CPython-*/*.json"))
    if not candidates:
        return None
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def check(
    thresholds: dict[str, dict[str, Any]],
    bench_data: dict[str, Any],
    *,
    strict: bool,
) -> tuple[list[tuple[str, str, float, float]], list[str], int]:
    """Compare bench data to thresholds; return (breaches, missing, n_seen).

    Each breach is (bench_name, percentile_field, actual_seconds, gate_seconds).
    `missing` is gated bench names absent from the JSON.
    `n_seen` is the count of benches seen in the JSON.
    """
    seen: set[str] = set()
    breaches: list[tuple[str, str, float, float]] = []

    for bench in bench_data.get("benchmarks", []):
        name = bench["name"]
        seen.add(name)
        threshold = _resolve_threshold(thresholds, name)
        if threshold is None:
            print(f"  no gate: {name}")
            continue
        gated_fields = [f for f in _PERCENTILE_FIELDS if f in threshold]
        if not gated_fields:
            print(
                f"  WARN: gate entry for {name!r} has no recognized percentile field "
                f"(allowed: {sorted(_PERCENTILE_FIELDS)})",
                file=sys.stderr,
            )
            continue
        for field in gated_fields:
            q = _PERCENTILE_FIELDS[field]
            actual = _bench_percentile(bench, q)
            gate = float(threshold[field])
            status = "PASS" if actual <= gate else "FAIL"
            print(f"  [{status}] {name} {field}: {actual * 1e6:.1f}us vs gate {gate * 1e6:.1f}us")
            if actual > gate:
                breaches.append((name, field, actual, gate))

    # Detect gated benches missing from JSON. Strip parametrize suffix when
    # comparing against `seen` so a parametrized variant counts as covering
    # its bare-name gate.
    missing: list[str] = []
    for gated_name in thresholds:
        # Hit if any seen name matches exactly OR has the gated_name as its bare prefix.
        if gated_name in seen:
            continue
        if any(s == gated_name or s.startswith(gated_name + "[") for s in seen):
            continue
        missing.append(gated_name)
        print(f"  [MISS] {gated_name} not in JSON")

    if strict and missing:
        # In strict mode, missing benches are reportable failures (mode used
        # in CI where every gated bench MUST run).
        pass  # surfaced via missing list; main() decides exit code

    return breaches, missing, len(seen)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail if any gated bench is missing from the JSON.",
    )
    parser.add_argument(
        "--include-tier2",
        action="store_true",
        help="Also load tier2.yml (env-gated heavy benches).",
    )
    args = parser.parse_args(argv)

    # Read paths from module-level attributes at CALL time (not via the default
    # args of _load_thresholds, which are captured at function-definition time).
    # This makes the script monkeypatch-friendly for tests + lets users override
    # via env if ever needed.
    try:
        thresholds = _load_thresholds(
            args.include_tier2,
            tier1_path=TIER1_PATH,
            tier2_path=TIER2_PATH,
        )
    except ValueError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1

    bench_path = _newest_bench_json()
    if bench_path is None:
        print("ERROR: no pytest-benchmark JSON found", file=sys.stderr)
        print(
            "       Did pytest run with --benchmark-autosave + --benchmark-storage?",
            file=sys.stderr,
        )
        return 1
    print(f"Loading bench JSON: {bench_path}")
    bench_data = json.loads(bench_path.read_text())

    try:
        breaches, missing, n_seen = check(thresholds, bench_data, strict=args.strict)
    except RuntimeError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(
        json.dumps(
            {
                "breaches": [
                    {
                        "name": n,
                        "field": f,
                        "actual_seconds": a,
                        "gate_seconds": g,
                    }
                    for n, f, a, g in breaches
                ],
                "missing": missing,
                "total_seen": n_seen,
                "include_tier2": args.include_tier2,
                "strict": args.strict,
            },
            indent=2,
        )
    )

    if breaches:
        print(f"\n{len(breaches)} gate(s) breached", file=sys.stderr)
        return 1
    if args.strict and missing:
        print(
            f"\n{len(missing)} gated bench(es) missing from JSON in --strict mode", file=sys.stderr
        )
        return 1
    print("\nAll gates green.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
