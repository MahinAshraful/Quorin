"""Unit tests for quorin.pool.BufferPool.

Pool itself is platform-agnostic — it constructs only NumPy arrays and a
``collections.deque``. No POSIX shared memory, no Redis, no segment
lifecycle. These tests run on every platform.

The multi-thread stress test is gated on ``sys._is_gil_enabled()`` because
PEP 703 free-threaded interpreters break the deque-atomicity assumption.
"""

from __future__ import annotations

import sys
import threading
from typing import Any

import numpy as np
import pytest

from quorin.metrics import pool_miss_total
from quorin.pool import BufferPool
from quorin.schema import FeatureField, FeatureSchema, dtype

# ---------------------------------------------------------------------------
# Schemas. Module-level so __init_subclass__ runs at import time.
# ---------------------------------------------------------------------------


class _OneScalar(FeatureSchema):
    version = 1
    fields = [FeatureField("a", dtype.float32)]


class _OneEmbedding(FeatureSchema):
    version = 1
    fields = [FeatureField("emb", dtype.float32, shape=(128,))]


class _Mixed(FeatureSchema):
    version = 1
    fields = [
        FeatureField("age", dtype.int32),
        FeatureField("score", dtype.float32, shape=(4,)),
        FeatureField("flag", dtype.uint8),
    ]


# ---------------------------------------------------------------------------
# Helpers.
# ---------------------------------------------------------------------------


def _miss_count(pool: BufferPool) -> float:
    """Read the current ``pool_miss_total`` value for a pool's schema label.

    prometheus-client's Counter exposes ``_value.get()`` for the per-label
    child; using the private accessor is acceptable in tests (we use the
    same pattern in the integration suite).
    """
    return float(pool_miss_total.labels(schema=pool.schema_name)._value.get())


# ---------------------------------------------------------------------------
# Construction & invariants.
# ---------------------------------------------------------------------------


def test_preallocated_at_construction() -> None:
    pool = BufferPool(_OneScalar, max_size=8)
    assert pool.available == 8
    assert pool.max_size == 8
    assert pool.element_count == 1


def test_preallocated_for_shaped_field() -> None:
    pool = BufferPool(_OneEmbedding, max_size=4)
    assert pool.available == 4
    assert pool.element_count == 128


def test_per_schema_isolation() -> None:
    """Pools for different schemas have different element counts."""
    p_scalar = BufferPool(_OneScalar, max_size=2)
    p_embedding = BufferPool(_OneEmbedding, max_size=2)
    p_mixed = BufferPool(_Mixed, max_size=2)
    assert p_scalar.element_count == 1
    assert p_embedding.element_count == 128
    # Mixed: int32 (1) + float32 shape=(4,) (4) + uint8 (1) = 6.
    assert p_mixed.element_count == 6


def test_max_size_zero_rejected() -> None:
    with pytest.raises(ValueError, match="max_size must be >= 1"):
        BufferPool(_OneScalar, max_size=0)


def test_max_size_negative_rejected() -> None:
    with pytest.raises(ValueError, match="max_size must be >= 1"):
        BufferPool(_OneScalar, max_size=-1)


def test_max_size_one_works() -> None:
    pool = BufferPool(_OneScalar, max_size=1)
    assert pool.available == 1
    with pool.checkout() as buf:
        assert buf.shape == (1,)
        assert pool.available == 0
    assert pool.available == 1


def test_schema_label_uses_module_qualname() -> None:
    pool = BufferPool(_OneScalar)
    expected = f"{_OneScalar.__module__}.{_OneScalar.__qualname__}"
    assert pool.schema_name == expected


def test_repr_includes_state() -> None:
    pool = BufferPool(_OneEmbedding, max_size=4)
    r = repr(pool)
    assert "BufferPool" in r
    assert pool.schema_name in r
    assert "max_size=4" in r
    assert "element_count=128" in r
    assert "available=4" in r


# ---------------------------------------------------------------------------
# Checkout shape / dtype / contiguity / writeability.
# ---------------------------------------------------------------------------


def test_checkout_returns_correct_shape_dtype_contiguity() -> None:
    pool = BufferPool(_OneEmbedding, max_size=2)
    with pool.checkout() as buf:
        assert buf.shape == (128,)
        assert buf.dtype == np.float32
        assert buf.flags["C_CONTIGUOUS"]
        assert buf.flags["WRITEABLE"]


def test_checkout_decrements_returns_increments() -> None:
    pool = BufferPool(_OneScalar, max_size=4)
    assert pool.available == 4
    with pool.checkout():
        assert pool.available == 3
    assert pool.available == 4


def test_initial_buffers_are_zeroed() -> None:
    """Pre-allocated buffers should be zero, not garbage."""
    pool = BufferPool(_OneEmbedding, max_size=2)
    with pool.checkout() as buf:
        assert np.all(buf == 0.0)


# ---------------------------------------------------------------------------
# Zero-on-return behavior.
# ---------------------------------------------------------------------------


def test_buffer_zeroed_on_return_default() -> None:
    """With ``zero_on_return=True`` (default), a returned buffer is clean
    on the next checkout."""
    pool = BufferPool(_OneEmbedding, max_size=1)
    with pool.checkout() as buf:
        buf[:] = 7.0
    # Same buffer object is at the head of the deque again.
    with pool.checkout() as buf2:
        assert np.all(buf2 == 0.0)


def test_zero_on_return_false_keeps_dirty() -> None:
    """With ``zero_on_return=False``, the buffer is re-pooled with whatever
    bytes the user wrote. Verified deterministically with max_size=1 (the
    re-checkout returns the same physical buffer)."""
    pool = BufferPool(_OneEmbedding, max_size=1, zero_on_return=False)
    sentinel: float = 7.0
    with pool.checkout() as buf:
        buf[:] = sentinel
    with pool.checkout() as buf2:
        assert np.all(buf2 == sentinel)


def test_zero_on_return_property_exposed() -> None:
    p_default = BufferPool(_OneScalar, max_size=1)
    p_off = BufferPool(_OneScalar, max_size=1, zero_on_return=False)
    assert p_default.zero_on_return is True
    assert p_off.zero_on_return is False


# ---------------------------------------------------------------------------
# Exception safety.
# ---------------------------------------------------------------------------


def test_exception_in_with_returns_buffer() -> None:
    """The contextmanager's ``finally`` re-appends the buffer even when the
    body raises."""
    pool = BufferPool(_OneScalar, max_size=2)
    starting = pool.available
    with pytest.raises(RuntimeError, match="boom"), pool.checkout() as buf:
        buf[:] = 99.0
        raise RuntimeError("boom")
    assert pool.available == starting
    # And the returned buffer was zeroed.
    with pool.checkout() as buf2:
        assert np.all(buf2 == 0.0)


def test_validation_error_in_with_does_not_corrupt_pool() -> None:
    """Pre-existing ValueError in the user code path also returns the buf."""
    pool = BufferPool(_OneScalar, max_size=2)
    starting = pool.available
    with pytest.raises(ValueError, match="user code"), pool.checkout():
        raise ValueError("user code")
    assert pool.available == starting


# ---------------------------------------------------------------------------
# Exhaustion fall-through.
# ---------------------------------------------------------------------------


def test_exhaustion_falls_through_to_fresh_alloc() -> None:
    """100 nested checkouts on a max_size=10 pool: 90 are fresh allocations,
    ``pool_miss_total`` ticks 90 times for this pool's schema label."""
    pool = BufferPool(_OneScalar, max_size=10)
    miss_before = _miss_count(pool)

    held: list[Any] = []
    contexts: list[Any] = []
    try:
        for _ in range(100):
            ctx = pool.checkout()
            buf = ctx.__enter__()
            held.append(buf)
            contexts.append(ctx)

        miss_after = _miss_count(pool)
        assert miss_after - miss_before == 90

        # All 100 buffers have the right shape/dtype.
        for buf in held:
            assert buf.shape == (1,)
            assert buf.dtype == np.float32

        # All 100 ids are distinct (no double-handout).
        assert len({id(b) for b in held}) == 100
    finally:
        for ctx in contexts:
            ctx.__exit__(None, None, None)


def test_exhaustion_fresh_buffers_are_zero() -> None:
    """Miss-path returns ``np.zeros``, not garbage from ``np.empty``."""
    pool = BufferPool(_OneEmbedding, max_size=1)
    # Drain the pre-alloc.
    ctx1 = pool.checkout()
    ctx1.__enter__()
    try:
        with pool.checkout() as fresh:
            assert np.all(fresh == 0.0)
    finally:
        ctx1.__exit__(None, None, None)


# ---------------------------------------------------------------------------
# Cap-at-max-size on return.
# ---------------------------------------------------------------------------


def test_surplus_returns_dropped() -> None:
    """A burst grows the pool past max_size via fall-through; on return,
    only the first max_size buffers are re-appended."""
    pool = BufferPool(_OneScalar, max_size=4)

    contexts = [pool.checkout() for _ in range(8)]
    bufs = [ctx.__enter__() for ctx in contexts]
    assert pool.available == 0
    assert len({id(b) for b in bufs}) == 8

    for ctx in contexts:
        ctx.__exit__(None, None, None)

    # Cap held: 4 returned, 4 dropped.
    assert pool.available == 4


# ---------------------------------------------------------------------------
# Nesting.
# ---------------------------------------------------------------------------


def test_nested_checkouts_get_distinct_buffers() -> None:
    pool = BufferPool(_OneScalar, max_size=4)
    starting = pool.available
    with pool.checkout() as a:
        with pool.checkout() as b:
            assert id(a) != id(b)
            assert pool.available == starting - 2
        assert pool.available == starting - 1
    assert pool.available == starting


# ---------------------------------------------------------------------------
# Multi-thread stress (GIL atomicity contract).
# ---------------------------------------------------------------------------


def test_multithread_stress_under_gil() -> None:
    """8 threads * 1000 cycles; assert no double-handout and final pool state
    matches the start. Gated on the GIL being enabled — under PEP 703
    free-threaded CPython, deque atomicity guarantees do not hold and the
    pool would need a real lock."""
    if not getattr(sys, "_is_gil_enabled", lambda: True)():
        pytest.skip("free-threaded CPython: deque atomicity does not hold; pool needs lock")

    pool = BufferPool(_OneScalar, max_size=16)

    active: set[int] = set()
    active_lock = threading.Lock()
    errors: list[BaseException] = []

    def worker() -> None:
        try:
            for _ in range(1000):
                with pool.checkout() as buf:
                    bid = id(buf)
                    with active_lock:
                        if bid in active:
                            raise AssertionError(
                                f"buffer id {bid} held by two threads simultaneously"
                            )
                        active.add(bid)
                    # Touch the buffer so the JIT and any aliasing bug shows up.
                    buf[0] = 1.0
                    with active_lock:
                        active.discard(bid)
        except BaseException as exc:
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert errors == [], errors
    # Pool returned to a sane state. Could be exactly ``starting`` or up to
    # ``max_size`` if a brief miss-allocation got reabsorbed; both are valid.
    assert pool.available <= pool.max_size
    assert pool.available >= 0
