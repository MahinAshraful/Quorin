"""Integration test for benchmarks/runs/repeat.py — subprocess plumbing dry-run.

Spawns the orchestrator at --num-runs 2 against a fast non-Redis bench
(test_compile_schema_200_fields). Verifies:
  * Orchestrator returns 0.
  * Output JSON has the expected schema (raw_runs, aggregate, spec).
  * Each raw_runs entry has the requested percentiles + max + n_samples.
  * aggregate.median_p99 + aggregate.max_of_max + aggregate.stddev_p99 present.

This test does DOUBLE DUTY:
  1. Schema-contract guard. If pytest-benchmark changes its JSON layout
     (stats.data, the per-bench `name` field, etc.), this fails before
     operators discover it via a broken Tier-2 run mid-Step-16c.
  2. Tier-2 runtime canary. Per-subprocess wall clock here is the same
     plumbing the full Tier-2 N=20 runs use. A future Numba bump / pytest-
     benchmark version drift that adds 5s/subprocess shows up here as
     ~24s -> ~34s — caught in CI, not 60 minutes into the next weekly
     Tier-2 run. Surprise duration jumps signal env regressions worth
     investigating before the canonical headline run.

Marked @pytest.mark.integration so it runs alongside other integration tests
under ``-m integration``. ~24s wall clock (2 fresh subprocesses, each running
pytest + Numba prewarm + the bench).
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.integration

_REPO_ROOT = Path(__file__).resolve().parents[2]


def test_repeat_orchestrator_subprocess_plumbing(tmp_path: Path) -> None:
    """Dry-run repeat.py at N=2 against a fast bench; verify JSON schema."""
    spec_path = tmp_path / "spec.yml"
    spec_path.write_text(
        "name: integration_dry_run\n"
        "description: subprocess plumbing test against a fast non-Redis bench\n"
        "test_path: benchmarks/test_schema_benchmark.py::test_compile_schema_200_fields\n"
        "percentiles: [50, 95, 99]\n"
        "warmup_runs: 0\n"
        "timeout_seconds: 60\n"
    )
    out_path = tmp_path / "n20_output.json"
    cmd = [
        sys.executable,
        str(_REPO_ROOT / "benchmarks" / "runs" / "repeat.py"),
        "--bench-spec",
        str(spec_path),
        "--num-runs",
        "2",
        "--output",
        str(out_path),
    ]
    res = subprocess.run(
        cmd, cwd=_REPO_ROOT, capture_output=True, text=True, timeout=300, check=False
    )
    assert res.returncode == 0, (
        f"orchestrator exited {res.returncode}\nstdout:\n{res.stdout}\nstderr:\n{res.stderr}"
    )
    assert out_path.exists(), "orchestrator did not write the output JSON"
    payload = json.loads(out_path.read_text())

    # Schema spot-checks
    assert payload["num_runs"] == 2
    assert payload["spec"]["name"] == "integration_dry_run"
    assert len(payload["raw_runs"]) == 2
    for run in payload["raw_runs"]:
        # Per-run output schema (run_one_subprocess return value)
        assert "p50" in run and "p95" in run and "p99" in run
        assert "max" in run and "min" in run and "median" in run
        assert run["n_samples"] > 0
        # Sanity: percentile ordering
        assert run["min"] <= run["median"] <= run["max"]
        assert run["p50"] <= run["p95"] <= run["p99"] <= run["max"]

    agg = payload["aggregate"]
    assert "median_p99" in agg, f"aggregate missing median_p99: {sorted(agg)}"
    assert "max_of_max" in agg
    # N=2 -> stddev_p99 is sample stddev (ddof=1) of two values; could be
    # any non-negative float. Just assert presence + non-negativity.
    assert "stddev_p99" in agg
    assert agg["stddev_p99"] >= 0.0
