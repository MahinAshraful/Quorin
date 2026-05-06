"""Unit tests for quorin.pool.BatchBufferPool (Step 8).

Pool itself is platform-agnostic — it constructs only NumPy arrays and a
``collections.deque``. No POSIX shared memory, no Redis, no segment
lifecycle. These tests run on every platform.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from quorin.metrics import pool_miss_total
from quorin.pool import BatchBufferPool
from quorin.schema import FeatureField, FeatureSchema, dtype


class _OneScalar(FeatureSchema):
    version = 1
    fields = [FeatureField("a", dtype.float32)]


class _OneEmbedding(FeatureSchema):
    version = 1
    fields = [FeatureField("emb", dtype.float32, shape=(128,))]


def _miss_count(pool: BatchBufferPool) -> float:
    return float(pool_miss_total.labels(schema=pool.schema_name)._value.get())


# ---------------------------------------------------------------------------
# Construction & invariants.
# ---------------------------------------------------------------------------


def test_preallocated_at_construction() -> None:
    pool = BatchBufferPool(_OneScalar, batch_size=32, max_size=8)
    assert pool.available == 8
    assert pool.max_size == 8
    assert pool.batch_size == 32
    assert pool.element_count == 1


def test_default_max_size_is_64() -> None:
    pool = BatchBufferPool(_OneScalar, batch_size=4)
    assert pool.max_size == 64


def test_default_zero_on_return_is_false() -> None:
    """Step 8 default: zero_on_return=False (different from BufferPool's True
    because the cost at batch=1000 is ~50-100 us). Safety contract is in
    BatchBufferPool's docstring; assemble_batch's full-overwrite kernel
    makes it safe."""
    pool = BatchBufferPool(_OneScalar, batch_size=4)
    assert pool.zero_on_return is False


def test_batch_size_zero_rejected() -> None:
    with pytest.raises(ValueError, match="batch_size must be >= 1"):
        BatchBufferPool(_OneScalar, batch_size=0)


def test_batch_size_negative_rejected() -> None:
    with pytest.raises(ValueError, match="batch_size must be >= 1"):
        BatchBufferPool(_OneScalar, batch_size=-1)


def test_max_size_zero_rejected() -> None:
    with pytest.raises(ValueError, match="max_size must be >= 1"):
        BatchBufferPool(_OneScalar, batch_size=4, max_size=0)


def test_repr_includes_state() -> None:
    pool = BatchBufferPool(_OneEmbedding, batch_size=10, max_size=4)
    r = repr(pool)
    assert "BatchBufferPool" in r
    assert pool.schema_name in r
    assert "batch_size=10" in r
    assert "max_size=4" in r
    assert "element_count=128" in r
    assert "available=4" in r


def test_schema_label_uses_module_qualname() -> None:
    pool = BatchBufferPool(_OneScalar, batch_size=4)
    expected = f"{_OneScalar.__module__}.{_OneScalar.__qualname__}"
    assert pool.schema_name == expected


# ---------------------------------------------------------------------------
# Checkout shape / dtype / contiguity / writeability.
# ---------------------------------------------------------------------------


def test_checkout_returns_2d_correct_shape() -> None:
    pool = BatchBufferPool(_OneEmbedding, batch_size=32, max_size=2)
    with pool.checkout() as buf:
        assert buf.shape == (32, 128)
        assert buf.dtype == np.float32
        assert buf.flags["C_CONTIGUOUS"]
        assert buf.flags["WRITEABLE"]


def test_distinct_checkouts_are_distinct_views() -> None:
    """Slab-sliced buffers must not alias each other."""
    pool = BatchBufferPool(_OneScalar, batch_size=8, max_size=4)
    ctx1 = pool.checkout()
    buf1 = ctx1.__enter__()
    ctx2 = pool.checkout()
    buf2 = ctx2.__enter__()
    try:
        # Distinct numpy objects.
        assert buf1 is not buf2
        # Different memory ranges (slab[0] vs slab[1] — different rows).
        buf1[:] = 1.0
        assert np.all(buf2 == 0.0), "writing to one buffer corrupted another"
    finally:
        ctx1.__exit__(None, None, None)
        ctx2.__exit__(None, None, None)


def test_checkout_decrements_returns_increments() -> None:
    pool = BatchBufferPool(_OneScalar, batch_size=4, max_size=4)
    assert pool.available == 4
    with pool.checkout():
        assert pool.available == 3
    assert pool.available == 4


# ---------------------------------------------------------------------------
# Zero-on-return behavior.
# ---------------------------------------------------------------------------


def test_zero_on_return_false_keeps_dirty() -> None:
    """Default behavior: returned buffers carry whatever the user wrote."""
    pool = BatchBufferPool(_OneEmbedding, batch_size=4, max_size=1)
    sentinel = 7.0
    with pool.checkout() as buf:
        buf[:] = sentinel
    with pool.checkout() as buf2:
        assert np.all(buf2 == sentinel), "default zero_on_return=False should preserve data"


def test_zero_on_return_true_clears() -> None:
    pool = BatchBufferPool(_OneEmbedding, batch_size=4, max_size=1, zero_on_return=True)
    with pool.checkout() as buf:
        buf[:] = 99.0
    with pool.checkout() as buf2:
        assert np.all(buf2 == 0.0), "zero_on_return=True should clear on return"


# ---------------------------------------------------------------------------
# Exception safety.
# ---------------------------------------------------------------------------


def test_exception_in_with_returns_buffer() -> None:
    pool = BatchBufferPool(_OneScalar, batch_size=4, max_size=2)
    starting = pool.available
    with pytest.raises(RuntimeError, match="boom"), pool.checkout() as buf:
        buf[:] = 99.0
        raise RuntimeError("boom")
    assert pool.available == starting


# ---------------------------------------------------------------------------
# Exhaustion fall-through.
# ---------------------------------------------------------------------------


def test_exhaustion_falls_through_to_fresh_alloc() -> None:
    """20 nested checkouts on a max_size=4 pool: 16 are fresh allocations,
    pool_miss_total ticks 16 times for this pool's schema label."""
    pool = BatchBufferPool(_OneScalar, batch_size=4, max_size=4)
    miss_before = _miss_count(pool)

    held: list[Any] = []
    contexts: list[Any] = []
    try:
        for _ in range(20):
            ctx = pool.checkout()
            buf = ctx.__enter__()
            held.append(buf)
            contexts.append(ctx)

        miss_after = _miss_count(pool)
        assert miss_after - miss_before == 16

        for buf in held:
            assert buf.shape == (4, 1)
            assert buf.dtype == np.float32
        # All 20 ids are distinct.
        assert len({id(b) for b in held}) == 20
    finally:
        for ctx in contexts:
            ctx.__exit__(None, None, None)


def test_surplus_returns_dropped() -> None:
    """A burst grows past max_size via fall-through; on return, only the
    first max_size buffers are re-appended."""
    pool = BatchBufferPool(_OneScalar, batch_size=4, max_size=4)

    contexts = [pool.checkout() for _ in range(8)]
    bufs = [ctx.__enter__() for ctx in contexts]
    assert pool.available == 0
    assert len({id(b) for b in bufs}) == 8

    for ctx in contexts:
        ctx.__exit__(None, None, None)

    assert pool.available == 4


# ---------------------------------------------------------------------------
# Slab-aliveness — slab kept alive by deque holding views.
# ---------------------------------------------------------------------------


def test_slab_alive_after_construction() -> None:
    """The contiguous slab is the backing storage. The deque holds views;
    the slab persists as long as any view does. Drop one view via checkout-
    without-return-cycle and confirm other buffers are still valid."""
    pool = BatchBufferPool(_OneEmbedding, batch_size=4, max_size=8)

    # Take a buffer out and write to it; keep a reference outside the with.
    held_buf = None
    ctx = pool.checkout()
    held_buf = ctx.__enter__()
    held_buf[:] = 5.0
    ctx.__exit__(None, None, None)

    # Pool now has 8 buffers (the same one was returned). A different
    # checkout should yield writeable, non-corrupted memory.
    with pool.checkout() as buf2:
        assert buf2.flags["WRITEABLE"]
        # buf2 may be the same slab row (default zero_on_return=False keeps
        # 5.0 if it's the same row) or a different row (still 0.0). Either
        # way the memory is valid.
        buf2[:] = 11.0
        # The buffer we held earlier shares storage with one slab row.
        # If buf2 is the same physical row as held_buf, held_buf reflects
        # the new value. If different, held_buf still has 5.0. Both prove
        # the slab memory is alive.
        assert held_buf[0, 0] in (5.0, 11.0)
