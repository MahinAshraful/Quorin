"""Benchmarks for pyforge.assembly.assemble (Numba kernel).

These run alongside benchmarks/test_serving_benchmark.py (Python oracle) so
the two p99 numbers can be compared directly. ADR-004's adoption gate is
the 200-field warm Numba p99 vs Python p99: >=3x wins adoption as production
default; <3x ships parity-only.

Targets we gate on (see benchmarks/regression/tier1.yml — gate values are
WSL2-tolerant pending Step 16c native-Linux retightening):
  * assemble_4_field_warm_numba:                  p99 (Step 5 + 16 trip-wire)
  * assemble_200_field_warm_numba:                p99 (realistic production)
  * assemble_200_field_cold_cache_numba:          p99 (NEW Step 16, see below)

Step 16 changes:
  * Cold-cache test rewritten to use the session-scoped ``cold_cache_clobber``
    fixture (4x detected L3, capped at 1 GB; 16 MiB fallback on WSL2 / VMs
    that hide cache hierarchy). Replaces the pre-Step-16 hardcoded 32 MiB
    clobber that was median-calibrated and would silently undersize on
    AMD EPYC / Intel Sapphire Rapids hardware.
  * Module-top ``pytestmark`` skipif tightened to require Linux (was
    just non-win32). Cold-cache fixture reads /sys/, so Linux is now
    explicit.
  * Pool head-to-head NOT implemented as a Tier-1 gate — pytest-benchmark
    doesn't model derived metrics (delta between two benches). The pooled
    paths are gated absolutely; the +0.5us-overhead claim from ADR-005
    is a manual diff in 16c review against committed JSON. See
    progress/step16c_review.md for the explicit checklist.
"""

from __future__ import annotations

import contextlib
import sys
import tracemalloc

import numpy as np
import pytest

pytestmark = pytest.mark.skipif(
    sys.platform != "linux",
    reason=(
        "assembly benches require Linux: POSIX shm + /sys/.../index3 for the "
        "cold-cache L3 detection. Step 16 tightened the platform bound from "
        "win32-skip to linux-only."
    ),
)

from _helpers import make_segment, pack_row, release_segment  # noqa: E402
from pyforge.assembly import assemble, prewarm  # noqa: E402
from pyforge.layout import insert  # noqa: E402
from pyforge.schema import FeatureField, FeatureSchema, dtype  # noqa: E402
from pyforge.serving import EntityNotFoundError  # noqa: E402

# ---------------------------------------------------------------------------
# Module-level autouse pre-warm — eliminates first-call compile cost from
# pytest-benchmark's calibration runs.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module", autouse=True)
def _prewarm_numba() -> None:
    prewarm()


# ---------------------------------------------------------------------------
# Schemas — duplicated from test_serving_benchmark.py so both files measure
# against identical schema definitions.
# ---------------------------------------------------------------------------


class _Schema4Field(FeatureSchema):
    """4-field warm-cache scenario. Spec target: ≤5 us p99."""

    version = 1
    fields = [
        FeatureField("age", dtype.int32),
        FeatureField("clicks", dtype.int64),
        FeatureField("ltv", dtype.float64),
        FeatureField("score", dtype.float32),
    ]


def _build_n_field_schema(n: int, with_embedding: bool = False) -> type[FeatureSchema]:
    """Build a schema with N float32 scalar fields, optionally appending one
    128-dim float32 embedding. Used by the crossover scan and the 200-field
    benchmark."""
    fields: list[FeatureField] = [
        FeatureField(f"feat_{i:03d}", dtype.float32) for i in range(n - 1 if with_embedding else n)
    ]
    if with_embedding:
        fields.append(FeatureField("embedding", dtype.float32, shape=(128,)))
    cls_name = f"_SchemaN{n}{'_emb' if with_embedding else ''}"
    return type(cls_name, (FeatureSchema,), {"version": 1, "fields": fields})


_Schema200Field = _build_n_field_schema(200, with_embedding=True)


# ---------------------------------------------------------------------------
# Fixtures.
# ---------------------------------------------------------------------------


@pytest.fixture
def seg_4_field():
    seg = make_segment(_Schema4Field, capacity=64)
    values = {
        "age": np.array([42], dtype=np.int32),
        "clicks": np.array([1_234_567], dtype=np.int64),
        "ltv": np.array([987.65], dtype=np.float64),
        "score": np.array([0.5], dtype=np.float32),
    }
    insert(seg, "u", pack_row(_Schema4Field, values))
    yield seg
    release_segment(seg)


@pytest.fixture
def seg_200_field():
    seg = make_segment(_Schema200Field, capacity=16)
    values: dict[str, np.ndarray] = {
        f.name: np.zeros(f.element_count, dtype=np.float32) for f in _Schema200Field.fields
    }
    insert(seg, "u", pack_row(_Schema200Field, values))
    yield seg
    release_segment(seg)


# ---------------------------------------------------------------------------
# Latency benchmarks. Names exactly match thresholds.yml keys.
# ---------------------------------------------------------------------------


def test_bench_assemble_4_field_warm_numba(benchmark, seg_4_field) -> None:
    """4-field schema, single entity, warm cache. Spec target: ≤5 us p99.
    Gated < 10 us (Python lookup ~3 us is the floor)."""
    seg = seg_4_field
    out = assemble(seg, "u")
    assert out.shape == (4,)

    result = benchmark(assemble, seg, "u")
    assert result.shape == (4,)


def test_bench_assemble_200_field_warm_numba(benchmark, seg_200_field) -> None:
    """200-field schema with 128-dim embedding. Spec target: 10-20 us p99.
    Gated < 50 us. ADR-004 compares against Python's p99 for the same schema."""
    seg = seg_200_field
    out = assemble(seg, "u")
    expected_len = sum(f.element_count for f in _Schema200Field.fields)
    assert out.shape == (expected_len,)

    result = benchmark(assemble, seg, "u")
    assert result.shape == (expected_len,)


def test_bench_assemble_lookup_miss_numba(benchmark, seg_4_field) -> None:
    """Unhappy path: ensure the EntityNotFoundError raise path doesn't quietly
    regress."""
    seg = seg_4_field

    def _miss() -> None:
        with contextlib.suppress(EntityNotFoundError):
            assemble(seg, "ghost")

    benchmark(_miss)


# ---------------------------------------------------------------------------
# Crossover scan — plot Numba vs Python at multiple field counts. Not gated;
# results are recorded for ADR-004's adoption-decision write-up.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n", [4, 10, 20, 50, 100, 200])
def test_bench_assemble_crossover(benchmark, n: int) -> None:
    """Numba assemble at N fields. Companion to a future Python-side scan."""
    schema = _build_n_field_schema(n, with_embedding=False)
    seg = make_segment(schema, capacity=4)
    try:
        values = {f.name: np.zeros(f.element_count, dtype=np.float32) for f in schema.fields}
        insert(seg, "u", pack_row(schema, values))
        out = assemble(seg, "u")
        assert out.shape == (n,)

        result = benchmark(assemble, seg, "u")
        assert result.shape == (n,)
    finally:
        release_segment(seg)


# ---------------------------------------------------------------------------
# Cold-cache benchmark (Step 16 P3 + C3 rewrite).
# Uses the session-scoped ``cold_cache_clobber`` fixture (4x detected L3,
# capped at 1 GB; 16 MiB fallback when sysfs hides cache hierarchy).
# Pedantic mode + rounds=200 + warmup_rounds=0: each round is one assemble
# call preceded by a clobber-array sum, so we measure cold-CPU-cache p99.
# Honest disclosure: this measures cold-CPU-cache, NOT cold-page-cache —
# the segment is still resident in /dev/shm tmpfs. ADR-015 covers the
# methodology in full.
# Step 16 trip-wire: if Tier-2 N=20 native-Linux p99 > 50us AFTER the
# MADV_HUGEPAGE A/B decision (16b), revise the README cold-cache claim
# with the measured number per ADR-015 "Cold-cache reality."
# ---------------------------------------------------------------------------


def test_bench_assemble_200_field_cold_cache_numba(
    benchmark, seg_200_field, cold_cache_clobber
) -> None:
    """200-field cold-CPU-cache assemble; L3-sized clobber via session fixture.

    The clobber array is sized at 4x detected L3 (capped at 1 GB) so the
    sequential .sum() evicts every line of a fully populated L3. On WSL2
    / VMs the fixture trips the 16 MiB fallback and logs WARN (the C3
    fix working as intended; Ubuntu CI runners surface real L3).
    """
    clobber, clobber_bytes = cold_cache_clobber
    benchmark.extra_info["clobber_bytes"] = clobber_bytes
    seg = seg_200_field

    def setup() -> tuple[tuple, dict]:
        _ = clobber.sum()  # forces L1/L2/L3 eviction; numpy 1.26 1D float64 sum is single-threaded
        return (seg, "u"), {}

    benchmark.pedantic(assemble, setup=setup, rounds=200, iterations=1, warmup_rounds=0)


# ---------------------------------------------------------------------------
# Allocation budget — the Numba path should NOT be worse than the Python
# path. Same calibration: 4 * total_element_count + 32 KiB per call.
# ---------------------------------------------------------------------------


def test_assemble_numba_allocation_budget(seg_4_field) -> None:
    """After warmup, 1k Numba assembles should allocate <= the Step 4 budget.
    Numba's compiled kernel allocates nothing; per-call overhead is the
    output buffer + the single np.frombuffer view, same as Python."""
    import gc

    seg = seg_4_field
    target = "u"

    for _ in range(100):
        assemble(seg, target)

    gc.collect()
    tracemalloc.start()
    snap_before = tracemalloc.take_snapshot()

    n_calls = 1_000
    for _ in range(n_calls):
        assemble(seg, target)

    snap_after = tracemalloc.take_snapshot()
    tracemalloc.stop()

    diffs = snap_after.compare_to(snap_before, "lineno")
    total_alloc = sum(d.size_diff for d in diffs)
    per_call = total_alloc / n_calls

    output_bytes = 4 * seg.layout.total_element_count
    budget = output_bytes + 32 * 1024
    assert per_call < budget, (
        f"assemble_numba allocated {per_call:.1f} bytes/call "
        f"(budget = {output_bytes} output + 32 KiB overhead = {budget})"
    )
