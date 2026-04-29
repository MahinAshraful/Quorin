"""Step 7 — repeat-run analysis of GC tail-latency benchmark.

Runs ``benchmarks/test_gc_p999.py`` in fresh subprocesses (N=20 runs per
scenario, 60 fresh subprocesses total) so we can decide the
``start_collector()`` default by data, not by individual lucky/unlucky
runs.

Methodology (locked here, not a knob):

- **Fresh subprocess per run.** GC state, ``gc.freeze`` state, page caches,
  Numba caches, and CPython's adaptive thresholds all carry over within a
  process. Same-process repeats would mask the variance we are trying to
  measure.
- **N=20 runs per config.** p9999 is computed from ~10 organic-GC events
  per 100k iterations; a single run is one sample of that random process.
  N=5 was too small (Signal A flipped between independent batches). N=10
  left the rule AMBIGUOUS for our hardware. N=20 surfaces the rare-event
  spike behavior reliably.
- **Programmatic decision rule + secondary distributional analysis.** The
  binary rule below catches clear-win cases. When the rule returns
  AMBIGUOUS, ADR-006 falls back to comparing spike rates and worst-case
  variance directly.

Decision rule (Signal A and Signal B):
  Signal A: timer's max stays consistently bounded (<= TIMER_MAX_BOUND_US
            across all runs).
  Signal B: freeze-only's max occasionally spikes (>= FREEZE_ONLY_SPIKE_US
            in at least 1 of the runs).

  - Both hold        -> timer is default (gen2_interval_seconds=0.5).
  - Neither holds    -> freeze-only is default.
  - A only           -> freeze-only (timer is bounded, but no spike to fix).
  - B only           -> AMBIGUOUS; consult ADR-006's spike-rate +
                        worst-case-variance secondary analysis.

Usage:
    uv run python benchmarks/runs/step7_gc_tail_repeat.py

Output:
    Tabulated per-run + summary printed to stdout.
    Raw + parsed data saved to benchmarks/results/step7_gc_tail_repeat.txt.
"""

from __future__ import annotations

import argparse
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESULTS_PATH = REPO_ROOT / "benchmarks" / "results" / "step7_gc_tail_repeat.txt"
# Default: 20 runs per config. N=5 was too small (Signal A flipped
# between independent batches). N=10 left the rule AMBIGUOUS. N=20 is
# the canonical decision-grade dataset. CLI ``--num-runs`` overrides for
# targeted investigations (e.g., N=50 anomaly resolution).
DEFAULT_NUM_RUNS = 20
SUBPROCESS_TIMEOUT_SECONDS = 180

# All scenarios the orchestrator knows how to run. The CLI ``--scenarios``
# arg picks a comma-separated subset. Test IDs include parametrize-
# generated entries for the interval sweep.
ALL_SCENARIOS: dict[str, str] = {
    "baseline": "test_p999_no_manager_no_freeze",
    "freeze_only": "test_p999_freeze_only_no_timer",
    "freeze_plus_timer": "test_p999_freeze_plus_timer",
    "freeze_plus_timer_pressure": "test_p999_freeze_plus_timer_with_pressure",
    # Interval sweep on the SAFE no-freeze + callback + timer path. N=50
    # isolation showed freeze + callback together has a measured tail
    # penalty; the recommended opt-in path uses the timer WITHOUT freeze.
    "interval_0.5": "test_p999_no_freeze_timer_interval_sweep[0.5]",
    "interval_1.0": "test_p999_no_freeze_timer_interval_sweep[1.0]",
    "interval_2.0": "test_p999_no_freeze_timer_interval_sweep[2.0]",
    "interval_5.0": "test_p999_no_freeze_timer_interval_sweep[5.0]",
    # ADR-006 anomaly-isolation controls.
    "callback_only_no_freeze": "test_p999_callback_only_no_freeze",
    "freeze_only_no_callback": "test_p999_freeze_only_no_callback",
}

# Default canonical run: the three configs the decision rule was written
# against.
DEFAULT_SCENARIOS = ("baseline", "freeze_only", "freeze_plus_timer")

# Decision-rule thresholds (microseconds). Only applied when both
# ``freeze_only`` and ``freeze_plus_timer`` are in the scenario set.
TIMER_MAX_BOUND_US = 250.0
FREEZE_ONLY_SPIKE_US = 500.0

# Parses lines like ``      p9999_us:    87.96 us``.
_METRIC = re.compile(r"^\s*(p50|p99|p999|p9999|max)_us:\s+([\d.]+)\s+us\s*$")
_HEADER = re.compile(r"^\[gc_p999\] ")


def run_one(test_name: str) -> tuple[str, dict[str, float]]:
    """Invoke pytest for a single test in a fresh Python subprocess.

    Returns the captured stdout and the parsed metrics dict.
    """
    # Use ``sys.executable -m pytest`` rather than ``uv run pytest``: the
    # parent script is already running inside the uv-managed venv, so
    # ``sys.executable`` is the venv Python. Avoids depending on ``uv``
    # being on the subprocess PATH.
    cmd = [
        sys.executable,
        "-m",
        "pytest",
        "-s",
        f"benchmarks/test_gc_p999.py::{test_name}",
    ]
    proc = subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=SUBPROCESS_TIMEOUT_SECONDS,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"pytest failed for {test_name} (rc={proc.returncode}):\n"
            f"--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}"
        )

    # Find the [gc_p999] block and parse metrics from following lines.
    metrics: dict[str, float] = {}
    in_block = False
    for line in proc.stdout.splitlines():
        if _HEADER.match(line):
            in_block = True
            continue
        if in_block:
            m = _METRIC.match(line)
            if m:
                metrics[m.group(1)] = float(m.group(2))
            elif metrics and not line.strip():
                # Blank line after metrics — end of block.
                break
    if not metrics:
        raise RuntimeError(f"failed to parse [gc_p999] metrics from {test_name}:\n{proc.stdout}")
    return proc.stdout, metrics


def _print_and_log(line: str, log: list[str]) -> None:
    print(line)
    log.append(line)


def _collect_runs(
    scenarios: dict[str, str], num_runs: int, log: list[str]
) -> dict[str, list[dict[str, float]]]:
    results: dict[str, list[dict[str, float]]] = {label: [] for label in scenarios}
    total_runs = num_runs * len(scenarios)
    run_counter = 0
    for label, test_name in scenarios.items():
        for run_id in range(1, num_runs + 1):
            run_counter += 1
            print(
                f"[{run_counter}/{total_runs}] {label} run {run_id}/{num_runs}...",
                flush=True,
            )
            stdout, metrics = run_one(test_name)
            results[label].append(metrics)
            log.append(f"--- {label} run {run_id}: {test_name} ---")
            log.append(stdout.rstrip())
            log.append("")
    return results


def _print_per_run_table(results: dict[str, list[dict[str, float]]], log: list[str]) -> None:
    header = f"{'scenario':<22}{'run':>5}{'p50':>9}{'p99':>9}{'p999':>9}{'p9999':>10}{'max':>10}"
    print()
    _print_and_log("=" * 80, log)
    _print_and_log("Per-run results (microseconds)", log)
    _print_and_log("=" * 80, log)
    _print_and_log(header, log)
    for label, runs in results.items():
        for i, m in enumerate(runs, 1):
            row = (
                f"{label:<22}{i:>5}"
                f"{m['p50']:>9.2f}{m['p99']:>9.2f}"
                f"{m['p999']:>9.2f}{m['p9999']:>10.2f}{m['max']:>10.2f}"
            )
            _print_and_log(row, log)
        print()
        log.append("")


def _print_summary(
    results: dict[str, list[dict[str, float]]], log: list[str]
) -> dict[str, dict[str, float]]:
    header = (
        f"{'scenario':<22}{'med_p99':>10}{'med_p999':>10}{'med_p9999':>11}"
        f"{'worst_max':>11}{'sd_max':>10}"
    )
    _print_and_log("=" * 80, log)
    _print_and_log(
        "Summary: median across runs (worst_max = max-of-maxes; sd_max = stddev)",
        log,
    )
    _print_and_log("=" * 80, log)
    _print_and_log(header, log)
    summary: dict[str, dict[str, float]] = {}
    for label, runs in results.items():
        s = {
            "med_p99": statistics.median(r["p99"] for r in runs),
            "med_p999": statistics.median(r["p999"] for r in runs),
            "med_p9999": statistics.median(r["p9999"] for r in runs),
            "worst_max": max(r["max"] for r in runs),
            "sd_max": statistics.stdev(r["max"] for r in runs) if len(runs) > 1 else 0.0,
        }
        summary[label] = s
        row = (
            f"{label:<22}"
            f"{s['med_p99']:>10.2f}{s['med_p999']:>10.2f}{s['med_p9999']:>11.2f}"
            f"{s['worst_max']:>11.2f}{s['sd_max']:>10.2f}"
        )
        _print_and_log(row, log)
    return summary


def _apply_decision_rule(
    results: dict[str, list[dict[str, float]]], num_runs: int, log: list[str]
) -> tuple[str, str] | None:
    """Apply the Signal A / Signal B rule. Returns None and skips if the
    required scenarios aren't both in the results (e.g., the caller is
    running an interval sweep or anomaly-only investigation)."""
    if "freeze_plus_timer" not in results or "freeze_only" not in results:
        log.append("")
        _print_and_log("=" * 80, log)
        _print_and_log(
            "Decision rule skipped: requires both 'freeze_only' and "
            "'freeze_plus_timer' scenarios in the run.",
            log,
        )
        _print_and_log("=" * 80, log)
        return None

    timer_maxs = [r["max"] for r in results["freeze_plus_timer"]]
    freeze_only_maxs = [r["max"] for r in results["freeze_only"]]
    signal_a = all(m <= TIMER_MAX_BOUND_US for m in timer_maxs)
    signal_b = any(m >= FREEZE_ONLY_SPIKE_US for m in freeze_only_maxs)

    log.append("")
    _print_and_log("=" * 80, log)
    _print_and_log("Decision rule", log)
    _print_and_log("=" * 80, log)
    _print_and_log(
        f"Signal A (timer max <= {TIMER_MAX_BOUND_US:.0f} us across all "
        f"{num_runs} runs): {'HOLDS' if signal_a else 'DOES NOT HOLD'}",
        log,
    )
    _print_and_log(f"  timer maxs: {[f'{m:.1f}' for m in timer_maxs]}", log)
    _print_and_log(
        f"Signal B (freeze-only max >= {FREEZE_ONLY_SPIKE_US:.0f} us in any of "
        f"{num_runs} runs): {'HOLDS' if signal_b else 'DOES NOT HOLD'}",
        log,
    )
    _print_and_log(f"  freeze-only maxs: {[f'{m:.1f}' for m in freeze_only_maxs]}", log)

    if signal_a and signal_b:
        return (
            "TIMER (gen2_interval_seconds=0.5)",
            "Both signals hold: timer reliably bounds max AND freeze-only "
            "occasionally spikes. The timer's tail-bounding advantage is "
            "empirical, not theoretical.",
        )
    if not signal_a and not signal_b:
        return (
            "FREEZE-ONLY (current default)",
            "Neither signal holds: timer adds overhead with no observable "
            "tail benefit, and freeze-only's worst case is acceptable. "
            "Simplicity wins.",
        )
    if signal_a and not signal_b:
        return (
            "FREEZE-ONLY (current default)",
            "Timer is bounded but freeze-only doesn't spike either — no "
            "problem to solve. Pay no overhead.",
        )
    return (
        "AMBIGUOUS — re-run with more samples or investigate",
        "Freeze-only spikes but timer doesn't reliably bound max. "
        "Neither design fully solves the tail-latency problem on this "
        "hardware/workload; investigate before shipping a default.",
    )


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Step 7 GC tail-latency repeat-run orchestrator. "
            "Runs benchmarks/test_gc_p999.py in fresh subprocesses."
        ),
    )
    p.add_argument(
        "--scenarios",
        default=",".join(DEFAULT_SCENARIOS),
        help=(
            "Comma-separated scenario keys. Available: "
            f"{', '.join(ALL_SCENARIOS.keys())}. "
            f"Default: {','.join(DEFAULT_SCENARIOS)}."
        ),
    )
    p.add_argument(
        "--num-runs",
        type=int,
        default=DEFAULT_NUM_RUNS,
        help=f"Fresh-subprocess runs per scenario. Default: {DEFAULT_NUM_RUNS}.",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_RESULTS_PATH,
        help=(
            f"Path for raw + parsed log. Default: {DEFAULT_RESULTS_PATH.relative_to(REPO_ROOT)}."
        ),
    )
    return p.parse_args(argv)


def _resolve_scenarios(spec: str) -> dict[str, str]:
    keys = [s.strip() for s in spec.split(",") if s.strip()]
    unknown = [k for k in keys if k not in ALL_SCENARIOS]
    if unknown:
        raise SystemExit(
            f"Unknown scenario keys: {unknown}. Available: {sorted(ALL_SCENARIOS.keys())}"
        )
    return {k: ALL_SCENARIOS[k] for k in keys}


def main() -> int:
    args = _parse_args()
    scenarios = _resolve_scenarios(args.scenarios)
    num_runs = args.num_runs
    output_path: Path = args.output

    log: list[str] = [
        "Step 7 — repeat-run GC tail-latency benchmark",
        f"Methodology: {num_runs} fresh subprocesses per scenario",
        f"Scenarios: {list(scenarios.keys())}",
        f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "=" * 80,
        "",
    ]
    results = _collect_runs(scenarios, num_runs, log)
    _print_per_run_table(results, log)
    _print_summary(results, log)
    decision_result = _apply_decision_rule(results, num_runs, log)
    if decision_result is not None:
        decision, rationale = decision_result
        _print_and_log("", log)
        _print_and_log(f"Decision: {decision}", log)
        _print_and_log(f"Rationale: {rationale}", log)
    # Resolve to absolute so a relative CLI arg (e.g.,
    # ``--output benchmarks/results/foo.txt``) doesn't blow up
    # ``relative_to`` when the cwd happens not to be REPO_ROOT.
    abs_output = output_path.resolve()
    abs_output.parent.mkdir(parents=True, exist_ok=True)
    abs_output.write_text("\n".join(log) + "\n")
    print()
    try:
        display_path = abs_output.relative_to(REPO_ROOT)
    except ValueError:
        display_path = abs_output
    print(f"Raw + parsed log saved to: {display_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
