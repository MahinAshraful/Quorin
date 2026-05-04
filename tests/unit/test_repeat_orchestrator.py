"""Unit tests for benchmarks/runs/repeat.py — the Tier-2 fresh-subprocess orchestrator.

Covers:
  * load_bench_spec validation + defaults.
  * _normalize_percentile YAML-friendly translation (99 -> 99.0, 999 -> 99.9).
  * _select_target_bench picks by function-name suffix; fallback to first.
  * aggregate() math: median-of-percentiles, max-of-max, stddev(p99) with N>=2 vs N=1.

Subprocess plumbing is covered by tests/integration/test_repeat_orchestrator_e2e.py
(integration-marked); these unit tests are pure-function math.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# benchmarks/runs/ isn't a package; import via sys.path
_BENCH_RUNS = Path(__file__).resolve().parents[2] / "benchmarks" / "runs"
sys.path.insert(0, str(_BENCH_RUNS))

import repeat  # noqa: E402, I001


# ---------------------------------------------------------------------------
# load_bench_spec
# ---------------------------------------------------------------------------


def test_load_bench_spec_applies_defaults(tmp_path: Path) -> None:
    spec_path = tmp_path / "spec.yml"
    spec_path.write_text("name: x\ntest_path: a/b.py::test_x\npercentiles: [50, 99]\n")
    spec = repeat.load_bench_spec(spec_path)
    assert spec["name"] == "x"
    assert spec["warmup_runs"] == 0  # default
    assert spec["timeout_seconds"] == repeat.DEFAULT_TIMEOUT_SECONDS
    assert spec["description"] == ""  # default


def test_load_bench_spec_rejects_missing_required(tmp_path: Path) -> None:
    spec_path = tmp_path / "spec.yml"
    spec_path.write_text("name: x\n")  # missing test_path + percentiles
    with pytest.raises(ValueError, match="missing required fields"):
        repeat.load_bench_spec(spec_path)


def test_load_bench_spec_rejects_non_numeric_percentiles(tmp_path: Path) -> None:
    spec_path = tmp_path / "spec.yml"
    spec_path.write_text("name: x\ntest_path: a/b.py::test_x\npercentiles: ['fifty', 99]\n")
    with pytest.raises(ValueError, match="percentiles must be a list of numbers"):
        repeat.load_bench_spec(spec_path)


# ---------------------------------------------------------------------------
# _normalize_percentile
# ---------------------------------------------------------------------------


def test_normalize_percentile_pass_through_for_standard() -> None:
    assert repeat._normalize_percentile(50) == 50.0
    assert repeat._normalize_percentile(95) == 95.0
    assert repeat._normalize_percentile(99) == 99.0


def test_normalize_percentile_handles_999_and_9999() -> None:
    """YAML convention: 999 -> p99.9, 9999 -> p99.99."""
    assert repeat._normalize_percentile(999) == 99.9
    assert repeat._normalize_percentile(9999) == 99.99


# ---------------------------------------------------------------------------
# _select_target_bench
# ---------------------------------------------------------------------------


def test_select_target_bench_matches_function_name_suffix() -> None:
    benches = [
        {"name": "test_other_bench"},
        {"name": "test_assemble_4_field_warm_numba"},
    ]
    target = repeat._select_target_bench(
        benches, "benchmarks/test_assembly_benchmark.py::test_assemble_4_field_warm_numba"
    )
    assert target["name"] == "test_assemble_4_field_warm_numba"


def test_select_target_bench_strips_parametrize_when_matching() -> None:
    benches = [{"name": "test_x[1000]"}, {"name": "test_y"}]
    target = repeat._select_target_bench(benches, "a/b.py::test_x")
    assert target["name"] == "test_x[1000]"


def test_select_target_bench_falls_back_to_first_when_no_match() -> None:
    benches = [{"name": "test_only_one"}]
    target = repeat._select_target_bench(benches, "a/b.py::test_other")
    assert target["name"] == "test_only_one"


# ---------------------------------------------------------------------------
# aggregate
# ---------------------------------------------------------------------------


def test_aggregate_computes_median_of_percentiles_and_max_of_max() -> None:
    runs = [
        {"p50": 1.0, "p99": 5.0, "max": 10.0, "median": 1.0, "min": 0.5, "n_samples": 100},
        {"p50": 2.0, "p99": 6.0, "max": 12.0, "median": 2.0, "min": 0.6, "n_samples": 100},
        {"p50": 3.0, "p99": 7.0, "max": 11.0, "median": 3.0, "min": 0.7, "n_samples": 100},
    ]
    out = repeat.aggregate(runs)
    assert out["median_p50"] == 2.0
    assert out["median_p99"] == 6.0
    assert out["max_of_max"] == 12.0
    assert out["median_max"] == 11.0
    assert out["total_samples"] == 300


def test_aggregate_stddev_p99_with_multiple_runs() -> None:
    """ddof=1 sample stddev. {5, 6, 7} -> mean 6, stddev 1."""
    runs = [
        {"p99": 5.0, "max": 0.0, "median": 0.0, "min": 0.0, "n_samples": 1},
        {"p99": 6.0, "max": 0.0, "median": 0.0, "min": 0.0, "n_samples": 1},
        {"p99": 7.0, "max": 0.0, "median": 0.0, "min": 0.0, "n_samples": 1},
    ]
    out = repeat.aggregate(runs)
    assert out["stddev_p99"] == pytest.approx(1.0)


def test_aggregate_stddev_p99_single_run_is_zero() -> None:
    """Sample stddev is undefined for n=1; aggregate() returns 0.0 for that case."""
    runs = [{"p99": 5.0, "max": 0.0, "median": 0.0, "min": 0.0, "n_samples": 1}]
    out = repeat.aggregate(runs)
    assert out["stddev_p99"] == 0.0


def test_aggregate_empty_runs_raises() -> None:
    with pytest.raises(ValueError, match="at least one run"):
        repeat.aggregate([])
