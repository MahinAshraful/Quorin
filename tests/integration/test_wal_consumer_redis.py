"""Integration tests for the WAL consumer against a real Redis stream.

Gated on ``@pytest.mark.integration``; requires the docker-compose Redis
service. The autouse ``_shm_test_isolation`` fixture in conftest scrubs
``pyforge:*`` keys and ``/dev/shm/pyforge_*`` after every test.

Tests exercise the producer + consumer end-to-end with real PEL semantics,
real XACK / XPENDING, and real shm.
"""

from __future__ import annotations

import asyncio
import sys
import time

import pytest

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        sys.platform == "win32",
        reason="WAL consumer requires POSIX shared memory + asyncio Redis",
    ),
]

from typing import cast as _cast  # noqa: E402
from unittest.mock import Mock as _Mock  # noqa: E402

import redis  # noqa: E402
import redis.asyncio  # noqa: E402

from _helpers import make_segment, release_segment  # noqa: E402
from pyforge._internal.pydantic_factory import clear_cache as clear_pydantic_cache  # noqa: E402
from pyforge._internal.row_pack import clear_cache as clear_row_pack_cache  # noqa: E402
from pyforge.layout import lookup  # noqa: E402
from pyforge.schema import FeatureField, FeatureSchema, dtype  # noqa: E402
from pyforge.serving import assemble  # noqa: E402
from pyforge.shm import SegmentRegistry  # noqa: E402
from pyforge.wal import (  # noqa: E402
    DEFAULT_STREAM_KEY,
    PROCESSED_KEY_PREFIX,
    WALProducer,
    WriteSyncTimeoutError,
)
from pyforge.wal_consumer import (  # noqa: E402
    ConsumerNameInUseError,
    NoopOfflineWriter,
    WALConsumer,
)

# Step 15 stub registry — these tests don't trigger the upgrade pause-clear
# branch; a Mock with the right spec is sufficient. See test_evolution_e2e.py
# for tests that exercise the real registry-driven reopen path.
_STUB_REGISTRY: SegmentRegistry = _cast("SegmentRegistry", _Mock(spec=SegmentRegistry))

# ---------------------------------------------------------------------------
# Test schema.
# ---------------------------------------------------------------------------


class _IntCS(FeatureSchema):
    version = 1
    fields = [
        FeatureField("a", dtype.float32),
        FeatureField("b", dtype.int64),
        FeatureField("emb", dtype.float32, shape=(8,)),
    ]


def _values(i: int = 0) -> dict[str, object]:
    return {
        "a": float(i) + 0.5,
        "b": i,
        "emb": [float(i + j) for j in range(8)],
    }


def _pending_count(redis_client: redis.Redis) -> int:
    """xpending count with NOGROUP-tolerance for races with consumer startup."""
    try:
        info = redis_client.xpending(DEFAULT_STREAM_KEY, "pyforge_consumers")
    except redis.exceptions.ResponseError as e:
        if "NOGROUP" in str(e):
            # Group not created yet (consumer hasn't had a chance to run);
            # treat as "everything is still pending" — caller polls again.
            return -1
        raise
    return int(info["pending"])


# ---------------------------------------------------------------------------
# Fixtures.
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _fresh_caches() -> None:
    clear_pydantic_cache()
    clear_row_pack_cache()


@pytest.fixture
def segment():
    seg = make_segment(_IntCS, capacity=256)
    try:
        yield seg
    finally:
        release_segment(seg)


@pytest.fixture
async def async_redis(redis_client: redis.Redis):
    """Async Redis client matching the sync one used by the producer.

    Uses the same URL the conftest's ``redis_client`` does (env or default).
    Closes on teardown so the event-loop's connection pool is drained.
    """
    import os

    url = os.environ.get("PYFORGE_REDIS_URL", "redis://127.0.0.1:6379/0")
    client = redis.asyncio.Redis.from_url(url, decode_responses=False)
    try:
        await client.ping()
    except Exception:
        pytest.skip("Async Redis not reachable")
    yield client
    await client.aclose()


# ---------------------------------------------------------------------------
# 1. Full apply: producer XADDs → consumer drains → shm + side-table + 0 PEL.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_producer_to_consumer_full_drain(
    redis_client: redis.Redis,
    async_redis: redis.asyncio.Redis,
    segment,
) -> None:
    producer = WALProducer(redis_client)
    consumer = WALConsumer(
        async_redis,
        segments={"_IntCS": segment},
        registry=_STUB_REGISTRY,
        consumer_name="int-test-1",
        block_ms=50,
        flush_interval_seconds=10.0,
        max_pending_ack=10,
    )

    # 50 distinct entities.
    msg_ids = [producer.write(_IntCS, f"ent-{i}", _values(i)) for i in range(50)]

    # Run consumer in background until all 50 are XACKed.
    run_task = asyncio.create_task(consumer.run())
    await asyncio.sleep(0)  # yield so consumer.run starts
    try:
        deadline = time.monotonic() + 10.0
        while time.monotonic() < deadline:
            pending = _pending_count(redis_client)
            if pending == 0 and not consumer._pending_ack:
                break
            await asyncio.sleep(0.05)
    finally:
        await consumer.stop()
        await asyncio.wait_for(run_task, timeout=5.0)

    # Each entity is in shm with the right values.
    for i in range(50):
        assert lookup(segment, f"ent-{i}") is not None
        vec = assemble(segment, f"ent-{i}")
        # Vector layout: a (1) + b (1) + emb (8) = 10 elems.
        assert len(vec) == 10

    # All 50 processed-keys exist.
    for mid in msg_ids:
        assert redis_client.exists(PROCESSED_KEY_PREFIX + mid) == 1

    # PEL fully drained.
    assert _pending_count(redis_client) == 0


# ---------------------------------------------------------------------------
# 2. write_sync round-trip with consumer running — returns msg_id.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_write_sync_unblocks_when_consumer_processes(
    redis_client: redis.Redis,
    async_redis: redis.asyncio.Redis,
    segment,
) -> None:
    producer = WALProducer(redis_client)
    consumer = WALConsumer(
        async_redis,
        segments={"_IntCS": segment},
        registry=_STUB_REGISTRY,
        consumer_name="int-test-2",
        block_ms=50,
    )
    run_task = asyncio.create_task(consumer.run())
    try:
        # write_sync blocks the calling thread; we run it in a thread so the
        # event loop can drive the consumer concurrently.
        result: dict[str, object] = {}

        def _do_write() -> None:
            try:
                result["msg_id"] = producer.write_sync(
                    _IntCS, "sync-ent", _values(1), timeout_ms=5000
                )
            except Exception as e:
                result["error"] = e

        import threading

        t = threading.Thread(target=_do_write)
        t.start()
        # Give the consumer time to process while we wait for the thread.
        for _ in range(50):
            if not t.is_alive():
                break
            await asyncio.sleep(0.05)
        t.join(timeout=2.0)
        assert "msg_id" in result, f"write_sync did not unblock: {result}"
        assert lookup(segment, "sync-ent") is not None
    finally:
        await consumer.stop()
        await asyncio.wait_for(run_task, timeout=5.0)


# ---------------------------------------------------------------------------
# 3. write_sync timeout when no consumer is running.
# ---------------------------------------------------------------------------


def test_write_sync_times_out_without_consumer(
    redis_client: redis.Redis,
) -> None:
    producer = WALProducer(redis_client)
    with pytest.raises(WriteSyncTimeoutError):
        producer.write_sync(_IntCS, "no-consumer", _values(0), timeout_ms=200)


# ---------------------------------------------------------------------------
# 4. Consumer-name conflict — second consumer raises.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_consumer_name_conflict_raises(
    async_redis: redis.asyncio.Redis,
    segment,
) -> None:
    c1 = WALConsumer(
        async_redis, segments={"_IntCS": segment}, registry=_STUB_REGISTRY, consumer_name="conflict"
    )
    await c1._acquire_consumer_lock()
    try:
        c2 = WALConsumer(
            async_redis,
            segments={"_IntCS": segment},
            registry=_STUB_REGISTRY,
            consumer_name="conflict",
        )
        with pytest.raises(ConsumerNameInUseError):
            await c2._acquire_consumer_lock()
    finally:
        await c1._release_consumer_lock()


# ---------------------------------------------------------------------------
# 5. Deferred-XACK durability — apply but stop before flush; re-run drains.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_deferred_xack_pel_survives_run_restart(
    redis_client: redis.Redis,
    async_redis: redis.asyncio.Redis,
    segment,
) -> None:
    """Apply N msgs into pending_ack, kill the run task BEFORE the periodic
    flush fires (and BEFORE the final flush would happen on graceful stop).
    Restart and confirm the messages are in PEL — restart's drain path then
    re-applies and XACKs."""

    class StuckOffline(NoopOfflineWriter):
        async def flush(self) -> None:
            # Hang flush so the run task can't complete its final flush
            # within our shutdown window. Will be cancelled.
            await asyncio.sleep(60)

    producer = WALProducer(redis_client)
    for i in range(20):
        producer.write(_IntCS, f"d-{i}", _values(i))

    # First consumer: apply but flush hangs.
    c1 = WALConsumer(
        async_redis,
        segments={"_IntCS": segment},
        registry=_STUB_REGISTRY,
        offline=StuckOffline(),
        consumer_name="durab-1",
        block_ms=50,
        flush_interval_seconds=300.0,
        max_pending_ack=10_000,
    )
    run1 = asyncio.create_task(c1.run())
    # Wait for all 20 to be applied to pending_ack.
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        if len(c1._pending_ack) >= 20:
            break
        await asyncio.sleep(0.02)
    assert len(c1._pending_ack) >= 20

    # Cancel the run task hard. The final flush will hang in StuckOffline.flush;
    # the cancellation propagates and _final_flush_and_ack swallows it. PEL
    # entries are NOT XACKed (because flush never returned).
    run1.cancel()
    import contextlib

    with contextlib.suppress(TimeoutError, asyncio.CancelledError):
        await asyncio.wait_for(run1, timeout=3.0)
    # Lock auto-expires in 60s; for the test we delete it to allow c2 to start.
    await async_redis.delete(c1._lock_key)

    # PEL still has all 20.
    assert _pending_count(redis_client) == 20

    # Restart with a working offline writer.
    c2 = WALConsumer(
        async_redis,
        segments={"_IntCS": segment},
        registry=_STUB_REGISTRY,
        consumer_name="durab-1",  # same name; lock has been released
        block_ms=50,
        flush_interval_seconds=10.0,
        max_pending_ack=5,  # trigger flush quickly
    )
    run2 = asyncio.create_task(c2.run())
    await asyncio.sleep(0)
    try:
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            if _pending_count(redis_client) == 0:
                break
            await asyncio.sleep(0.05)
        assert _pending_count(redis_client) == 0, "PEL did not drain on restart"
        # All entities present in shm.
        for i in range(20):
            assert lookup(segment, f"d-{i}") is not None
    finally:
        await c2.stop()
        await asyncio.wait_for(run2, timeout=5.0)


# ---------------------------------------------------------------------------
# 6. Idempotent re-apply: producer writes the same entity twice; consumer
#    leaves shm with the LAST values.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_same_entity_twice_last_write_wins(
    redis_client: redis.Redis,
    async_redis: redis.asyncio.Redis,
    segment,
) -> None:
    producer = WALProducer(redis_client)
    consumer = WALConsumer(
        async_redis,
        segments={"_IntCS": segment},
        registry=_STUB_REGISTRY,
        consumer_name="last-write",
        block_ms=50,
        max_pending_ack=2,
    )
    producer.write(_IntCS, "same-ent", _values(1))
    producer.write(_IntCS, "same-ent", _values(99))

    run_task = asyncio.create_task(consumer.run())
    await asyncio.sleep(0)
    try:
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            if _pending_count(redis_client) == 0:
                break
            await asyncio.sleep(0.05)
    finally:
        await consumer.stop()
        await asyncio.wait_for(run_task, timeout=5.0)

    vec = assemble(segment, "same-ent")
    # Last write was _values(99); a (first elem) should be 99.5.
    assert vec[0] == pytest.approx(99.5)
