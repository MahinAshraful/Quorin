"""Benchmarks for pyforge.serving.assemble.

These run the **pure-Python** oracle. Step 5's Numba kernel will produce
byte-identical output and must beat these baselines (gated by the spec's
crossover-at-20-fields rule). The numbers committed here become the floor.

Targets we gate on (see benchmarks/regression/thresholds.yml):
  * assemble_4_field_warm:    p99 < 50 us  (spec: 10-30 us)
  * assemble_200_field_warm:  p99 < 1 ms   (spec: 100-500 us)
  * allocation budget: < 4 * total_element_count + 32 KiB per call
    (output buffer is unavoidable; +32 KiB absorbs np.frombuffer object
    overhead and allocator headers across N fields)
"""

from __future__ import annotations

import sys
import tracemalloc

import pytest

pytestmark = pytest.mark.skipif(
    sys.platform == "win32",
    reason="serving requires POSIX (Linux/WSL2)",
)

import numpy as np  # noqa: E402

from _helpers import make_segment, pack_row, release_segment  # noqa: E402
from pyforge.layout import insert  # noqa: E402
from pyforge.schema import FeatureField, FeatureSchema, dtype  # noqa: E402
from pyforge.serving import EntityNotFound, assemble  # noqa: E402


# ---------------------------------------------------------------------------
# Schemas covering the two headline scenarios from the spec.
# ---------------------------------------------------------------------------


class _Schema4Field(FeatureSchema):
    """4-field warm-cache scenario. Spec target: 10-30 us p99 (Python)."""

    version = 1
    fields = [
        FeatureField("age", dtype.int32),
        FeatureField("clicks", dtype.int64),
        FeatureField("ltv", dtype.float64),
        FeatureField("score", dtype.float32),
    ]


def _build_200_field_schema() -> type[FeatureSchema]:
    """200 scalar fields + one 128-dim float32 embedding. Matches the spec's
    "realistic-production" schema sizing."""
    fields: list[FeatureField] = [
        FeatureField(f"feat_{i:03d}", dtype.float32) for i in range(199)
    ]
    fields.append(FeatureField("embedding", dtype.float32, shape=(128,)))
    return type(
        "_Schema200Field",
        (FeatureSchema,),
        {"version": 1, "fields": fields},
    )


_Schema200Field = _build_200_field_schema()


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


def test_bench_assemble_4_field_warm(benchmark, seg_4_field) -> None:
    """4-field schema, single entity, warm cache. Spec target: 10-30 us p99
    (Python). Gated < 50 us."""
    seg = seg_4_field
    # Sanity check before measuring.
    out = assemble(seg, "u")
    assert out.shape == (4,)

    result = benchmark(assemble, seg, "u")
    assert result.shape == (4,)


def test_bench_assemble_200_field_warm(benchmark, seg_200_field) -> None:
    """200-field schema with 128-dim embedding. Spec target: 100-500 us p99
    (Python). Gated < 1 ms."""
    seg = seg_200_field
    out = assemble(seg, "u")
    expected_len = sum(f.element_count for f in _Schema200Field.fields)
    assert out.shape == (expected_len,)

    result = benchmark(assemble, seg, "u")
    assert result.shape == (expected_len,)


def test_bench_assemble_lookup_miss(benchmark, seg_4_field) -> None:
    """Unhappy path: ensure the EntityNotFound raise path doesn't quietly
    regress (e.g., someone adds an exception-formatting hot loop)."""
    seg = seg_4_field

    def _miss() -> None:
        try:
            assemble(seg, "ghost")
        except EntityNotFound:
            pass

    benchmark(_miss)


# ---------------------------------------------------------------------------
# Allocation budget — verify assemble's per-call heap allocation scales as
# expected (output buffer + bounded constant overhead).
# ---------------------------------------------------------------------------


def test_assemble_allocation_budget(seg_4_field) -> None:
    """After warmup, 1k assembles should allocate <= (output_bytes + 32 KiB)
    per call. The output buffer itself is unavoidable (4 * total_element_count
    bytes); the +32 KiB absorbs np.frombuffer per-field Python objects and
    allocator headers across the field count.

    Step 6's buffer pool will take this to ``< 32 KiB per call`` (no fresh
    output allocation); Step 5's Numba kernel will further reduce per-field
    overhead. This budget locks the *current* shape so an accidental
    regression — e.g., a list comprehension over fields, or a copy in
    ``np.frombuffer``'s slice path — surfaces here, not in production.
    """
    import gc

    seg = seg_4_field
    target = "u"

    # Warmup: triggers any one-time module imports / NumPy state.
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

    output_bytes = 4 * seg.layout.total_element_count  # float32 buffer
    budget = output_bytes + 32 * 1024
    assert per_call < budget, (
        f"assemble allocated {per_call:.1f} bytes/call "
        f"(budget = {output_bytes} output + 32 KiB overhead = {budget})"
    )
