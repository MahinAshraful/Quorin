"""End-to-end integration tests for BufferPool.

Real ``SegmentRegistry`` create/close, real Redis. Verifies the full
register-schema, insert, pool-checkout, assemble path that production code
will follow.

The KeyboardInterrupt test uses ``signal.raise_signal(SIGINT)`` to prove
that the ``finally:`` in :meth:`BufferPool.checkout` re-appends the buffer
even when a signal handler injects an exception mid-``with``.
"""

from __future__ import annotations

import signal
import sys

import numpy as np
import pytest

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        sys.platform == "win32",
        reason="pool e2e requires POSIX (Linux/WSL2)",
    ),
]

from _helpers import pack_row  # noqa: E402
from quorin.assembly import assemble as assemble_numba  # noqa: E402
from quorin.assembly import prewarm  # noqa: E402
from quorin.layout import insert  # noqa: E402
from quorin.metrics import pool_miss_total  # noqa: E402
from quorin.pool import BufferPool  # noqa: E402
from quorin.schema import FeatureField, FeatureSchema, dtype  # noqa: E402
from quorin.serving import assemble as assemble_python  # noqa: E402
from quorin.shm import SegmentRegistry  # noqa: E402


@pytest.fixture(scope="module", autouse=True)
def _prewarm_numba() -> None:
    prewarm()


class _UserFeatures(FeatureSchema):
    version = 1
    fields = [
        FeatureField("age", dtype.int32),
        FeatureField("score", dtype.float32),
        FeatureField("embedding", dtype.float32, shape=(128,)),
        FeatureField("flag", dtype.uint8),
    ]


def _miss_count(pool: BufferPool) -> float:
    return float(pool_miss_total.labels(schema=pool.schema_name)._value.get())


def _row(i: int) -> dict[str, np.ndarray]:
    """Deterministic row data for entity ``user_{i}``."""
    emb = np.linspace(float(i), float(i) + 1.0, num=128, dtype=np.float32)
    return {
        "age": np.array([20 + i], dtype=np.int32),
        "score": np.array([0.5 * i], dtype=np.float32),
        "embedding": emb,
        "flag": np.array([i % 256], dtype=np.uint8),
    }


def test_e2e_pool_with_serving(redis_client) -> None:
    """Full path: SegmentRegistry → insert → pool.checkout → serving.assemble."""
    reg = SegmentRegistry(redis_client)
    seg = reg.create(_UserFeatures, capacity=8)
    try:
        for i in range(4):
            insert(seg, f"user_{i}", pack_row(_UserFeatures, _row(i)))

        pool = BufferPool(_UserFeatures, max_size=4)
        for i in range(4):
            with pool.checkout() as buf:
                returned = assemble_python(seg, f"user_{i}", out=buf)
                assert returned is buf
                assert returned.shape == (131,)
                assert returned[0] == np.float32(20 + i)
                assert returned[1] == np.float32(0.5 * i)
                np.testing.assert_array_equal(
                    returned[2:130],
                    np.linspace(float(i), float(i) + 1.0, num=128, dtype=np.float32),
                )
                assert returned[130] == np.float32(i % 256)
        # Pool fully restored after all checkouts.
        assert pool.available == 4
    finally:
        reg.close(seg)


def test_e2e_pool_with_assembly(redis_client) -> None:
    """Same path with the Numba kernel; output must match the Python oracle."""
    reg = SegmentRegistry(redis_client)
    seg = reg.create(_UserFeatures, capacity=8)
    try:
        insert(seg, "user_0", pack_row(_UserFeatures, _row(0)))

        pool = BufferPool(_UserFeatures, max_size=2)
        with pool.checkout() as buf:
            nb_out = assemble_numba(seg, "user_0", out=buf)
            py_out = assemble_python(seg, "user_0")
            np.testing.assert_array_equal(nb_out, py_out)
    finally:
        reg.close(seg)


def test_keyboardinterrupt_during_with_still_returns_buffer(redis_client) -> None:
    """Inject SIGINT inside the ``with`` block. The ``finally:`` must still
    run, the buffer must be re-appended, and ``pool.available`` must restore.
    POSIX-only: signal.raise_signal infrastructure is unreliable on Windows.
    """
    reg = SegmentRegistry(redis_client)
    seg = reg.create(_UserFeatures, capacity=4)
    try:
        insert(seg, "user_0", pack_row(_UserFeatures, _row(0)))

        pool = BufferPool(_UserFeatures, max_size=2)
        starting = pool.available

        with pytest.raises(KeyboardInterrupt), pool.checkout() as buf:
            # Touch the buffer so we know it's live.
            assemble_python(seg, "user_0", out=buf)
            signal.raise_signal(signal.SIGINT)

        assert pool.available == starting
    finally:
        reg.close(seg)


def test_pool_exhaustion_e2e_metric_increments(redis_client) -> None:
    """Drain a small pool by holding more buffers than max_size, doing real
    assembles into each, and verifying ``pool_miss_total`` ticks for the
    fall-through allocations."""
    reg = SegmentRegistry(redis_client)
    seg = reg.create(_UserFeatures, capacity=4)
    try:
        insert(seg, "user_0", pack_row(_UserFeatures, _row(0)))

        pool = BufferPool(_UserFeatures, max_size=4)
        miss_before = _miss_count(pool)

        # Hold 8 simultaneous checkouts. The first 4 are pool hits; the
        # next 4 fall through to fresh allocs and bump the miss counter.
        contexts = [pool.checkout() for _ in range(8)]
        bufs = [c.__enter__() for c in contexts]
        try:
            for buf in bufs:
                returned = assemble_python(seg, "user_0", out=buf)
                assert returned is buf
                assert returned[0] == np.float32(20)
        finally:
            for c in contexts:
                c.__exit__(None, None, None)

        miss_after = _miss_count(pool)
        assert miss_after - miss_before == 4
        # Cap-at-max-size held: only 4 of the 8 returns were re-pooled.
        assert pool.available == 4
    finally:
        reg.close(seg)
