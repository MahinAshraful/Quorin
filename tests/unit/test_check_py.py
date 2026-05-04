"""Unit tests for benchmarks/regression/check.py.

Covers:
  B1 — raises if stats.data missing; computes percentile when present.
  E2 — two-stage name lookup (exact > parametrize-stripped).
  L5 — duplicate-key guard across tier1 + tier2 raises ValueError.
  R1 — multi-percentile-field gating (p99 only / p99 + p999 / p999 breach
       with p99 pass / threshold with no recognized field).
  Plus tier1-only vs tier1+tier2 inclusion, strict vs non-strict missing
  semantics, and breach-on-any-tier behavior.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest

# benchmarks/regression/ isn't a package; add parent to path for import.
_BENCH_REGRESSION = Path(__file__).resolve().parents[2] / "benchmarks" / "regression"
sys.path.insert(0, str(_BENCH_REGRESSION))

import check  # noqa: E402, I001


# ---------------------------------------------------------------------------
# Helpers — synthetic pytest-benchmark JSON shapes.
# ---------------------------------------------------------------------------


def _bench_with_data(name: str, data: list[float]) -> dict[str, Any]:
    """Build a synthetic per-bench entry with raw round data."""
    return {"name": name, "stats": {"data": data}}


def _bench_without_data(name: str) -> dict[str, Any]:
    """Synthetic per-bench entry missing stats.data (default pytest-benchmark JSON)."""
    return {"name": name, "stats": {"min": 0.0, "max": 1.0, "mean": 0.5}}


def _write_yaml(tmp_path: Path, name: str, content: str) -> Path:
    p = tmp_path / name
    p.write_text(content)
    return p


# ---------------------------------------------------------------------------
# B1 — stats.data must be present (--benchmark-save-data lock)
# ---------------------------------------------------------------------------


def test_bench_percentile_raises_when_data_missing() -> None:
    """B1: clear error when raw round data is absent (default JSON shape)."""
    bench = _bench_without_data("test_x")
    with pytest.raises(RuntimeError, match="--benchmark-save-data"):
        check._bench_percentile(bench, 99.0)


def test_bench_percentile_computes_correctly_with_data() -> None:
    """B1 inverse: with --benchmark-save-data, percentile is computed from raw rounds."""
    # 100 evenly-spaced values 0.001..0.100s; p99 is 0.099s.
    data = [round(i * 0.001, 6) for i in range(1, 101)]
    bench = _bench_with_data("test_x", data)
    p99 = check._bench_percentile(bench, 99.0)
    assert 0.098 <= p99 <= 0.100, f"expected ~0.099, got {p99}"


# ---------------------------------------------------------------------------
# E2 — two-stage name lookup
# ---------------------------------------------------------------------------


def test_resolve_threshold_exact_match_wins() -> None:
    thresholds = {"test_foo": {"p99_seconds": 0.001}, "test_foo[1000]": {"p99_seconds": 0.005}}
    # Exact key wins over the bare-stripped fallback.
    assert check._resolve_threshold(thresholds, "test_foo[1000]") == {"p99_seconds": 0.005}


def test_resolve_threshold_strips_parametrize_suffix() -> None:
    thresholds = {"test_foo": {"p99_seconds": 0.001}}
    # Parametrize variant falls back to the bare name.
    assert check._resolve_threshold(thresholds, "test_foo[42]") == {"p99_seconds": 0.001}


def test_resolve_threshold_returns_none_when_absent() -> None:
    thresholds = {"test_other": {"p99_seconds": 0.001}}
    assert check._resolve_threshold(thresholds, "test_foo") is None
    assert check._resolve_threshold(thresholds, "test_foo[1]") is None


# ---------------------------------------------------------------------------
# L5 — duplicate-key guard across tiers
# ---------------------------------------------------------------------------


def test_load_thresholds_raises_on_tier_collision(tmp_path: Path) -> None:
    """L5: a key in both tier1.yml and tier2.yml raises ValueError, NOT silent overwrite."""
    t1 = _write_yaml(tmp_path, "tier1.yml", "test_foo:\n  p99_seconds: 0.001\n")
    t2 = _write_yaml(tmp_path, "tier2.yml", "test_foo:\n  p99_seconds: 0.002\n")
    with pytest.raises(ValueError, match=r"duplicated across tier1\.yml and tier2\.yml"):
        check._load_thresholds(include_tier2=True, tier1_path=t1, tier2_path=t2)


def test_load_thresholds_tier1_only_does_not_load_tier2(tmp_path: Path) -> None:
    """include_tier2=False ignores tier2.yml entirely (even if it has the colliding key)."""
    t1 = _write_yaml(tmp_path, "tier1.yml", "test_foo:\n  p99_seconds: 0.001\n")
    t2 = _write_yaml(tmp_path, "tier2.yml", "test_foo:\n  p99_seconds: 0.002\n")
    out = check._load_thresholds(include_tier2=False, tier1_path=t1, tier2_path=t2)
    assert out == {"test_foo": {"p99_seconds": 0.001}}


def test_load_thresholds_merges_disjoint_tiers(tmp_path: Path) -> None:
    t1 = _write_yaml(tmp_path, "tier1.yml", "test_a:\n  p99_seconds: 0.001\n")
    t2 = _write_yaml(tmp_path, "tier2.yml", "test_b:\n  p99_seconds: 0.005\n")
    out = check._load_thresholds(include_tier2=True, tier1_path=t1, tier2_path=t2)
    assert out == {
        "test_a": {"p99_seconds": 0.001},
        "test_b": {"p99_seconds": 0.005},
    }


# ---------------------------------------------------------------------------
# R1 — multi-percentile gating
# ---------------------------------------------------------------------------


def test_check_gates_p99_only_when_only_p99_present() -> None:
    """R1: bench gated on p99_seconds only -> only p99 enforced."""
    thresholds = {"test_x": {"p99_seconds": 0.005}}
    bench = _bench_with_data("test_x", [0.001, 0.002, 0.003])  # p99 ~ 0.003
    breaches, missing, n_seen = check.check(thresholds, {"benchmarks": [bench]}, strict=False)
    assert breaches == []
    assert missing == []
    assert n_seen == 1


def test_check_gates_both_p99_and_p999_when_both_present() -> None:
    """R1: bench gated on p99 AND p999 -> both enforced; both breaches reported."""
    thresholds = {"test_x": {"p99_seconds": 0.0001, "p999_seconds": 0.0002}}
    # 1000 samples: p99 = ~0.099, p999 = ~0.099 (both well over 0.0001/0.0002 gates)
    data = [i * 0.0001 for i in range(1, 1001)]
    bench = _bench_with_data("test_x", data)
    breaches, _missing, _n = check.check(thresholds, {"benchmarks": [bench]}, strict=False)
    fields_breached = {b[1] for b in breaches}
    assert fields_breached == {"p99_seconds", "p999_seconds"}


def test_check_p999_breach_with_p99_pass() -> None:
    """R1: p999 fails but p99 passes -> still FAIL with breach line citing p999."""
    thresholds = {"test_x": {"p99_seconds": 0.5, "p999_seconds": 0.0001}}
    # p99 = ~0.099 (under 0.5 gate); p999 = ~0.099 (over 0.0001 gate)
    data = [i * 0.0001 for i in range(1, 1001)]
    bench = _bench_with_data("test_x", data)
    breaches, _, _ = check.check(thresholds, {"benchmarks": [bench]}, strict=False)
    assert len(breaches) == 1
    name, field, actual, gate = breaches[0]
    assert name == "test_x"
    assert field == "p999_seconds"
    assert actual > gate


def test_check_warns_on_threshold_with_no_recognized_field(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """R1: threshold entry with no percentile field -> WARN, no breach, no enforcement."""
    thresholds = {"test_x": {"comment": "I forgot to type the field name"}}
    bench = _bench_with_data("test_x", [0.001, 0.002])
    breaches, _, _ = check.check(thresholds, {"benchmarks": [bench]}, strict=False)
    assert breaches == []
    err = capsys.readouterr().err
    assert "no recognized percentile field" in err


# ---------------------------------------------------------------------------
# Missing-bench semantics: strict vs non-strict
# ---------------------------------------------------------------------------


def test_check_missing_in_non_strict_does_not_breach() -> None:
    """A gated test absent from JSON is logged but not a breach (non-strict)."""
    thresholds = {"test_present": {"p99_seconds": 1.0}, "test_absent": {"p99_seconds": 1.0}}
    bench = _bench_with_data("test_present", [0.001])
    breaches, missing, _ = check.check(thresholds, {"benchmarks": [bench]}, strict=False)
    assert breaches == []
    assert missing == ["test_absent"]


def test_check_missing_when_parametrize_variant_present() -> None:
    """A bare-name gate is satisfied by any parametrize variant in the JSON."""
    thresholds = {"test_foo": {"p99_seconds": 1.0}}
    bench = _bench_with_data("test_foo[1000]", [0.001])
    _breaches, missing, _ = check.check(thresholds, {"benchmarks": [bench]}, strict=False)
    assert missing == []


# ---------------------------------------------------------------------------
# Breach detection in both modes
# ---------------------------------------------------------------------------


def test_check_breach_reported_regardless_of_strict() -> None:
    """A real breach is reported in both strict and non-strict modes."""
    thresholds = {"test_x": {"p99_seconds": 0.0001}}
    bench = _bench_with_data("test_x", [0.001, 0.002, 0.003, 0.005])
    for strict in (False, True):
        breaches, _, _ = check.check(thresholds, {"benchmarks": [bench]}, strict=strict)
        assert len(breaches) == 1, f"expected breach in strict={strict}"


# ---------------------------------------------------------------------------
# main()-level exit-code semantics. Synthesizes bench JSON + tier YAMLs so the
# strict-on-missing exit-code contract has explicit coverage.
# ---------------------------------------------------------------------------


def _setup_main_fixtures(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    tier1_content: str,
    tier2_content: str = "",
    bench_payload: dict[str, Any],
) -> None:
    """Wire tmp_path tier YAMLs + a synthetic pytest-benchmark JSON for main()."""
    import json as _json

    tier1 = tmp_path / "tier1.yml"
    tier1.write_text(tier1_content)
    tier2 = tmp_path / "tier2.yml"
    tier2.write_text(tier2_content)
    monkeypatch.setattr(check, "TIER1_PATH", tier1)
    monkeypatch.setattr(check, "TIER2_PATH", tier2)

    # pytest-benchmark JSON layout under <repo_root>/benchmarks/results/<machine>/.
    bench_dir = tmp_path / "benchmarks" / "results" / "Linux-CPython-3.12-64bit"
    bench_dir.mkdir(parents=True)
    bench_path = bench_dir / "0001_run.json"
    bench_path.write_text(_json.dumps(bench_payload))
    monkeypatch.setattr(check, "REPO_ROOT", tmp_path)
    # OUTPUT_PATH must also live under REPO_ROOT so the write succeeds in tmp_path.
    monkeypatch.setattr(check, "OUTPUT_PATH", tmp_path / "regression_check.json")


def test_main_strict_returns_1_when_gated_bench_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """main(--strict) returns 1 when a gated bench is absent from JSON.

    Locks the contract: --strict in CI must fail the workflow if any gated
    bench didn't run. Without this, partial bench runs would silently pass.
    """
    _setup_main_fixtures(
        tmp_path,
        monkeypatch,
        tier1_content="test_present:\n  p99_seconds: 1.0\ntest_absent:\n  p99_seconds: 1.0\n",
        bench_payload={"benchmarks": [{"name": "test_present", "stats": {"data": [0.001, 0.002]}}]},
    )
    rc = check.main(["--strict"])
    assert rc == 1, "main(--strict) must return 1 when a gated bench is missing"


def test_main_non_strict_returns_0_when_gated_bench_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """main() (non-strict) tolerates missing benches; logs but exits 0."""
    _setup_main_fixtures(
        tmp_path,
        monkeypatch,
        tier1_content="test_present:\n  p99_seconds: 1.0\ntest_absent:\n  p99_seconds: 1.0\n",
        bench_payload={"benchmarks": [{"name": "test_present", "stats": {"data": [0.001, 0.002]}}]},
    )
    rc = check.main([])
    assert rc == 0


def test_main_returns_1_on_breach_in_either_mode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A breach fails both strict and non-strict invocations."""
    _setup_main_fixtures(
        tmp_path,
        monkeypatch,
        tier1_content="test_x:\n  p99_seconds: 0.0001\n",
        bench_payload={
            "benchmarks": [{"name": "test_x", "stats": {"data": [0.001, 0.002, 0.005]}}]
        },
    )
    for argv in ([], ["--strict"]):
        assert check.main(argv) == 1, f"breach must exit 1 with argv={argv}"
