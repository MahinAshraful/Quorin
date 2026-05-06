"""Benchmarks for quorin.pool.BufferPool.

Three goals:

1. **Net win on the assemble hot path.** With a pre-warmed pool, the
   ``np.empty`` / ``np.zeros`` allocation is gone; ``assemble(..., out=buf)``
   should beat the Step 5 baseline by ~200 ns or more.
2. **Pool primitive cost.** ``checkout`` hit and miss in isolation, so
   regressions in the deque path or the ``buf.fill(0)`` zero-on-return are
   visible.
3. **Pick the zero-on-return implementation empirically.** ``buf.fill(0)``
   vs ``buf[:] = 0``. The spec called this out in "Common failure modes" —
   measure once, document the winner, lock the implementation.
"""

from __future__ import annotations

import sys

import numpy as np
import pytest

pytestmark = pytest.mark.skipif(
    sys.platform == "win32",
    reason="pool benchmarks require POSIX (Linux/WSL2)",
)

from _helpers import make_segment, pack_row, release_segment  # noqa: E402
from quorin.assembly import assemble as assemble_numba  # noqa: E402
from quorin.assembly import prewarm  # noqa: E402
from quorin.layout import insert  # noqa: E402
from quorin.pool import BufferPool  # noqa: E402
from quorin.schema import FeatureField, FeatureSchema, dtype  # noqa: E402
from quorin.serving import assemble as assemble_python  # noqa: E402


@pytest.fixture(scope="module", autouse=True)
def _prewarm_numba() -> None:
    prewarm()


# ---------------------------------------------------------------------------
# Schemas — same shape as test_assembly_benchmark.py for direct comparison.
# ---------------------------------------------------------------------------


class _Schema4Field(FeatureSchema):
    version = 1
    fields = [
        FeatureField("age", dtype.int32),
        FeatureField("clicks", dtype.int64),
        FeatureField("ltv", dtype.float64),
        FeatureField("score", dtype.float32),
    ]


def _build_200_field_schema() -> type[FeatureSchema]:
    fields: list[FeatureField] = [FeatureField(f"feat_{i:03d}", dtype.float32) for i in range(199)]
    fields.append(FeatureField("embedding", dtype.float32, shape=(128,)))
    return type("_Schema200", (FeatureSchema,), {"version": 1, "fields": fields})


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
    values = {f.name: np.zeros(f.element_count, dtype=np.float32) for f in _Schema200Field.fields}
    insert(seg, "u", pack_row(_Schema200Field, values))
    yield seg
    release_segment(seg)


# ---------------------------------------------------------------------------
# Pool primitive cost.
# ---------------------------------------------------------------------------


def test_bench_pool_checkout_hit_only(benchmark) -> None:
    """Checkout-then-immediate-return on a steady-state pool, no assemble.
    Target: < 200 ns. Measures the deque popleft + fill(0) + append cost."""
    pool = BufferPool(_Schema4Field, max_size=128)

    def _cycle() -> None:
        with pool.checkout():
            pass

    benchmark(_cycle)


def test_bench_pool_checkout_hit_no_zero(benchmark) -> None:
    """Same cycle with zero_on_return=False; isolates the buf.fill(0) cost."""
    pool = BufferPool(_Schema4Field, max_size=128, zero_on_return=False)

    def _cycle() -> None:
        with pool.checkout():
            pass

    benchmark(_cycle)


def test_bench_pool_checkout_miss_only(benchmark) -> None:
    """Empty pool, every checkout falls through to ``np.zeros``. Target: < 500 ns."""
    pool = BufferPool(_Schema4Field, max_size=1, zero_on_return=False)
    # Drain so every benchmark iteration falls through. Surplus drops on
    # return, keeping the deque empty.
    drain_ctx = pool.checkout()
    drain_ctx.__enter__()
    try:

        def _cycle() -> None:
            with pool.checkout():
                pass

        benchmark(_cycle)
    finally:
        drain_ctx.__exit__(None, None, None)


# ---------------------------------------------------------------------------
# Assemble + pool — the headline number.
# ---------------------------------------------------------------------------


def test_bench_assemble_4_field_warm_pooled(benchmark, seg_4_field) -> None:
    """4-field assemble with pooled output buffer. Target: beat Step 5's
    4.17 us by >=200 ns by eliminating np.empty."""
    pool = BufferPool(_Schema4Field, max_size=4)
    seg = seg_4_field

    def _pooled_assemble() -> None:
        with pool.checkout() as buf:
            assemble_numba(seg, "u", out=buf)

    benchmark(_pooled_assemble)


def test_bench_assemble_200_field_warm_pooled(benchmark, seg_200_field) -> None:
    """200-field warm assemble, pooled. Target: beat Step 5's 5.37 us."""
    pool = BufferPool(_Schema200Field, max_size=4)
    seg = seg_200_field

    def _pooled_assemble() -> None:
        with pool.checkout() as buf:
            assemble_numba(seg, "u", out=buf)

    benchmark(_pooled_assemble)


def test_bench_assemble_200_field_python_pooled(benchmark, seg_200_field) -> None:
    """Python oracle + pool, for completeness; allocation savings show up
    against test_serving_benchmark's allocation-heavy baseline."""
    pool = BufferPool(_Schema200Field, max_size=4)
    seg = seg_200_field

    def _pooled_assemble() -> None:
        with pool.checkout() as buf:
            assemble_python(seg, "u", out=buf)

    benchmark(_pooled_assemble)


# ---------------------------------------------------------------------------
# Zero-on-return micro-bench: fill(0) vs slice-assign.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("op", ["fill", "slice"])
def test_bench_zero_on_return_op(benchmark, op: str) -> None:
    """Compare ``buf.fill(0)`` vs ``buf[:] = 0`` for a 1300-element float32
    buffer (representative of the 200-field schema). Result documented in
    ADR-005; pool implementation locks the winner."""
    buf = np.zeros(1300, dtype=np.float32)
    buf[:] = 1.0  # dirty it so the zeroing has real work

    if op == "fill":

        def _z() -> None:
            buf.fill(0)
    else:

        def _z() -> None:
            buf[:] = 0

    benchmark(_z)
