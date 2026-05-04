"""MADV_HUGEPAGE A/B orchestrator (Step 16b).

Cold-cache 200-field assemble p99 across N=20 fresh subprocesses per side
(``hugepage=False`` / ``True``). Decision: SHIP iff
``geo_mean(speedup) >= 1.5x`` AND ``p10(speedup) >= 1.0x`` on the samples
that successfully completed; INCOMPLETE_AB if either side has < 80%
successful samples (i.e. >= 5 failures out of 20). ``REJECT_NOISE_DOMINATED``
when the geo-mean ships but the p10 floor doesn't — the case where a few
high-speedup runs lifted the mean above the line.

Standalone (not pytest-driven). Requires Linux + native L3 sysfs. GitHub
Actions ``ubuntu-latest`` qualifies; WSL2 may run but the result isn't
authoritative — kernel THP is platform-specific and ubuntu-latest's older
Xeon class L3 differs from bare-metal.

Tmpfs THP prerequisites (per ADR-015 §"MADV_HUGEPAGE A/B" / Step 16b plan §2.3):
  * Kernel build: ``CONFIG_TRANSPARENT_HUGEPAGE_SHMEM=y``.
  * Runtime: ``/sys/kernel/mm/transparent_hugepage/shmem_enabled`` ∈
    ``{always, advise, within_size}``.
  * Top-level: ``/sys/kernel/mm/transparent_hugepage/enabled`` ∈
    ``{always, madvise}``.

Even when all three are satisfied, ``madvise(MADV_HUGEPAGE)`` is a kernel
**hint** — it can be declined under fragmentation. The orchestrator records
both sysfs values into ``madv_ab.json["sysfs"]`` so a measured ~1.0x can be
distinguished from "kernel didn't grant hugepages."

The bench uses ``tests/_helpers.py::make_segment`` which calls
``posix_shm.create`` directly — no Redis state, no SegmentRegistry. Per-iter
cleanup is handled by the fixture's ``release_segment`` teardown. This
script's only defensive check is ``glob('/dev/shm/pyforge_*')`` at exit.

Output: ``benchmarks/results/madv_ab.json`` + decision exit code.
  * 0 → SHIP
  * 2 → REJECT or REJECT_NOISE_DOMINATED
  * 3 → INCOMPLETE_AB (re-trigger workflow)
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
from pathlib import Path

import numpy as np

if sys.platform != "linux":
    raise SystemExit("MADV A/B requires Linux")

DEFAULT_N_RUNS_PER_SIDE = 20
SHIP_GEO_MEAN_THRESHOLD = 1.5
SHIP_P10_FLOOR = 1.0  # Rev-6 issue #4: at least 90% of paired ratios >= 1.0x
MIN_SUCCESS_FRACTION = 0.80  # < this on either side -> INCOMPLETE_AB
SUBPROCESS_TIMEOUT_SECONDS = 300

_BENCH_TARGET = (
    "benchmarks/test_assembly_benchmark.py::test_bench_assemble_200_field_cold_cache_numba"
)

# Tmpfs THP prerequisites — captured into madv_ab.json["sysfs"] so a
# measured ~1.0x can be distinguished from "kernel didn't grant hugepages."
_SYSFS_PATHS = (
    "/sys/kernel/mm/transparent_hugepage/enabled",
    "/sys/kernel/mm/transparent_hugepage/shmem_enabled",
)


def _read_sysfs() -> dict[str, str]:
    out: dict[str, str] = {}
    for p in _SYSFS_PATHS:
        try:
            out[p] = Path(p).read_text().strip()
        except (FileNotFoundError, PermissionError, OSError) as e:
            out[p] = f"<read error: {type(e).__name__}>"
    return out


def run_one_side(hugepage: bool, num_runs: int, results_dir: Path) -> tuple[list[float], list[str]]:
    """Spawn num_runs fresh pytest subprocesses; return (per-run p99s, failure_reasons).

    Salvage discipline: a single subprocess failure (timeout, transient
    import error, parse error) does NOT abort the whole A/B run. Failures
    are caught, logged BOTH to stdout (visible in CI log) AND appended to
    the failures list (committed to JSON). Threshold for declaring
    inconclusive is ``>= MIN_SUCCESS_FRACTION`` on each side.

    Rev-9 lesson: when JSON wasn't written (e.g. orchestrator died at
    defensive leak-check before write), failure reasons in the failures
    list became unreachable. Real-time stdout print is the durable channel.
    """
    p99s: list[float] = []
    failures: list[str] = []
    for i in range(num_runs):
        env = os.environ.copy()
        env["PYFORGE_AB_HUGEPAGE"] = "1" if hugepage else "0"
        json_path = results_dir / f"side_{int(hugepage)}_run_{i:02d}.json"
        cmd = [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            _BENCH_TARGET,
            "--benchmark-only",
            "--benchmark-save-data",
            f"--benchmark-json={json_path}",
        ]
        try:
            subprocess.run(
                cmd,
                env=env,
                check=True,
                timeout=SUBPROCESS_TIMEOUT_SECONDS,
                capture_output=True,
                text=True,
            )
            bench = json.loads(json_path.read_text())["benchmarks"][0]
            p99s.append(float(np.percentile(bench["stats"]["data"], 99)))
        except subprocess.TimeoutExpired:
            msg = f"run_{i:02d}: timeout >{SUBPROCESS_TIMEOUT_SECONDS}s"
            failures.append(msg)
            print(f"  FAIL hugepage={int(hugepage)} {msg}", flush=True)
        except subprocess.CalledProcessError as e:
            tail = (e.stderr or "")[-500:]
            msg = f"run_{i:02d}: exit {e.returncode}; stderr_tail={tail!r}"
            failures.append(msg)
            print(
                f"  FAIL hugepage={int(hugepage)} run_{i:02d} "
                f"exit={e.returncode}\n    stderr_tail: {tail.strip()}",
                flush=True,
            )
        except (OSError, KeyError, ValueError, json.JSONDecodeError) as e:
            msg = f"run_{i:02d}: parse {type(e).__name__}: {e}"
            failures.append(msg)
            print(f"  FAIL hugepage={int(hugepage)} {msg}", flush=True)
    return p99s, failures


def main() -> int:
    parser = argparse.ArgumentParser(description="MADV_HUGEPAGE A/B orchestrator")
    parser.add_argument(
        "--num-runs",
        type=int,
        default=DEFAULT_N_RUNS_PER_SIDE,
        help=(
            "Subprocesses per side (default 20). Use --num-runs 2 to smoke-test "
            "the orchestrator plumbing without running the full N=20 measurement."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmarks/results/madv_ab.json"),
        help="Output JSON path (default benchmarks/results/madv_ab.json).",
    )
    args = parser.parse_args()
    n_runs: int = args.num_runs

    # Rev-8: snapshot pre-existing /dev/shm pollution from prior workflow
    # steps. The defensive glob at exit only flags segments NEW since this
    # snapshot — i.e. ones THIS orchestrator's subprocesses leaked. Cross-
    # step pollution (e.g. a SIGBUSed LARGE+RECORD step leaving stale
    # pyforge_* segments behind) is the upstream step's problem and must
    # NOT prevent the MADV A/B from writing its JSON. Run #31 surfaced this:
    # without the snapshot diff, we'd raise SystemExit before write_text().
    pre_existing_shm: set[str] = {str(p) for p in Path("/dev/shm").glob("pyforge_*")}
    if pre_existing_shm:
        print(
            f"NOTE: {len(pre_existing_shm)} pre-existing /dev/shm/pyforge_* segments "
            f"observed at startup (probably leaked by a prior workflow step). "
            f"Snapshotted; only NEW segments will be flagged at exit."
        )

    results_dir = Path("benchmarks/results/madv_ab_runs")
    results_dir.mkdir(parents=True, exist_ok=True)

    print(f"MADV_HUGEPAGE A/B: N={n_runs} per side; bench={_BENCH_TARGET}")
    print("Side hugepage=False ...")
    false_p99s, false_failures = run_one_side(False, n_runs, results_dir)
    print(f"  {len(false_p99s)}/{n_runs} ok, {len(false_failures)} failed")
    print("Side hugepage=True ...")
    true_p99s, true_failures = run_one_side(True, n_runs, results_dir)
    print(f"  {len(true_p99s)}/{n_runs} ok, {len(true_failures)} failed")

    n_false_ok, n_true_ok = len(false_p99s), len(true_p99s)
    threshold_n = int(n_runs * MIN_SUCCESS_FRACTION)
    incomplete = n_false_ok < threshold_n or n_true_ok < threshold_n

    n_pairs = min(n_false_ok, n_true_ok)
    speedups: list[float] = []
    geo_mean: float | None = None
    p10_speedup: float | None = None

    if incomplete:
        decision = "INCOMPLETE_AB"
    else:
        # Pair sorted-by-rank, then geo-mean + p10 floor. Use n_pairs to
        # clip the longer side when failures dropped one side below the
        # other but both passed MIN_SUCCESS_FRACTION.
        speedups = [
            f / t
            for f, t in zip(sorted(false_p99s)[:n_pairs], sorted(true_p99s)[:n_pairs], strict=False)
        ]
        geo_mean = statistics.geometric_mean(speedups)
        p10_speedup = float(np.percentile(speedups, 10))
        # SHIP requires both geo-mean AND p10 floor.
        if geo_mean >= SHIP_GEO_MEAN_THRESHOLD and p10_speedup >= SHIP_P10_FLOOR:
            decision = "SHIP"
        elif geo_mean >= SHIP_GEO_MEAN_THRESHOLD:
            # Mean shipped but tail underperformed — call it out distinctly so
            # ADR-015 rejection can document the noise-dominated case.
            decision = "REJECT_NOISE_DOMINATED"
        else:
            decision = "REJECT"

    output: dict[str, object] = {
        "false_p99s_seconds": false_p99s,
        "true_p99s_seconds": true_p99s,
        "false_failures": false_failures,
        "true_failures": true_failures,
        "n_actual_runs": {"false": n_false_ok, "true": n_true_ok},
        "n_pairs_compared": n_pairs,
        "n_runs_per_side_target": n_runs,
        "min_success_fraction": MIN_SUCCESS_FRACTION,
        "speedups_paired_sorted": speedups,
        "geo_mean_speedup": geo_mean,
        "p10_speedup": p10_speedup,
        "threshold_geo_mean": SHIP_GEO_MEAN_THRESHOLD,
        "threshold_p10": SHIP_P10_FLOOR,
        "sysfs": _read_sysfs(),
        "decision": decision,
    }
    # Rev-9 lesson: write JSON BEFORE the leak check. Run #33 produced a
    # diagnostic output dict (40 failures, INCOMPLETE_AB) but the orchestrator
    # raised SystemExit at the leak check before write_text(), losing all
    # the failure-reason data we needed to debug. Decision: data on disk
    # is the most valuable artifact — leak check becomes a non-fatal warning.
    out_path = args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2))
    print(
        f"\nA/B: false {n_false_ok}/{n_runs} ok, "
        f"true {n_true_ok}/{n_runs} ok; "
        f"comparing {n_pairs} paired-rank ratios"
    )
    if geo_mean is not None and p10_speedup is not None:
        print(f"geo_mean={geo_mean:.2f}x  p10={p10_speedup:.2f}x  → {decision}")
    else:
        print(f"Decision: {decision} (insufficient successful samples; re-trigger workflow)")
    print(f"Output: {out_path}")

    # Defensive leak check (Rev-9 demoted to non-fatal warning). Pre-existing
    # pollution from prior workflow steps was already snapshotted at startup;
    # this only flags segments NEW since this orchestrator started. A leak
    # here means subprocesses crashed before fixture teardown ran (e.g.
    # SIGKILL during the bench). Operator should investigate causes via
    # the failure stderr_tail values printed during run_one_side AND
    # committed to the JSON's false_failures/true_failures lists.
    current_shm: set[str] = {str(p) for p in Path("/dev/shm").glob("pyforge_*")}
    leaked = sorted(current_shm - pre_existing_shm)
    if leaked:
        print(
            f"WARNING: this run added {len(leaked)} /dev/shm/pyforge_* segments "
            f"(probably subprocess crashes before fixture teardown). "
            f"Sample: {leaked[:5]}",
            flush=True,
        )

    # Exit codes: 0 SHIP, 2 REJECT (either flavor), 3 INCOMPLETE_AB.
    return {
        "SHIP": 0,
        "REJECT": 2,
        "REJECT_NOISE_DOMINATED": 2,
        "INCOMPLETE_AB": 3,
    }[decision]


if __name__ == "__main__":
    sys.exit(main())
