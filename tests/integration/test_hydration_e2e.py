"""End-to-end integration tests for pyforge.hydration.hydrate (Step 13 Commit B).

7 tests covering the full producer -> consumer -> ParquetDatasetStore ->
``hydrate()`` -> new segment pipeline against real Redis.

E1: roundtrip basic (100 entities, all flushed).
E2: hydrate count matches parquet row count, NOT producer count.
E3: hydrate refuses when consumer alive (real liveness key).
E4 (#18): hydrate proceeds after liveness TTL expires.
E5 (#21): hydrate honors as_of_time_ns end-to-end via Parquet.
E6: capacity_factor on a real SegmentRegistry.create.
E7: idempotent — drop, hydrate, drop, hydrate again.

Gated on ``@pytest.mark.integration`` + skip on win32.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import struct
import sys
import time
from pathlib import Path

import pytest

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        sys.platform == "win32",
        reason="Step 13 hydration requires POSIX shm + asyncio Redis",
    ),
]

import pyarrow.dataset as pa_ds  # noqa: E402
import redis  # noqa: E402
import redis.asyncio  # noqa: E402
from _watchdog_helpers import drain_cleanup_queue  # noqa: E402

from pyforge._internal.arrow_schema import clear_cache as clear_arrow_cache  # noqa: E402
from pyforge._internal.pydantic_factory import (  # noqa: E402
    clear_cache as clear_pydantic_cache,
)
from pyforge._internal.row_pack import clear_cache as clear_row_pack_cache  # noqa: E402
from pyforge.hydration import (  # noqa: E402
    HydrationConflictError,
    hydrate,
)
from pyforge.layout import lookup  # noqa: E402
from pyforge.offline import ParquetDatasetStore  # noqa: E402
from pyforge.schema import FeatureField, FeatureSchema, dtype  # noqa: E402
from pyforge.shm import (  # noqa: E402
    SegmentNotFoundError,
    SegmentRegistry,
)
from pyforge.wal import DEFAULT_STREAM_KEY, PROCESSED_KEY_PREFIX, WALProducer  # noqa: E402
from pyforge.wal_consumer import KEY_WAL_CONSUMER_LIVENESS, WALConsumer  # noqa: E402

# ---------------------------------------------------------------------------
# Test schema. Top-level for picklability + reuse across tests.
# ---------------------------------------------------------------------------


class _IntHydrate(FeatureSchema):
    version = 1
    fields = [
        FeatureField("a", dtype.float32),
        FeatureField("b", dtype.int64),
        FeatureField("emb", dtype.float32, shape=(4,)),
    ]


def _values(i: int) -> dict[str, object]:
    return {
        "a": float(i) + 0.5,
        "b": i,
        "emb": [float(i + j) for j in range(4)],
    }


def _pending_count(redis_client: redis.Redis) -> int:
    """Mirrors test_offline_e2e.py:_pending_count."""
    try:
        info = redis_client.xpending(DEFAULT_STREAM_KEY, "pyforge_consumers")
    except redis.exceptions.ResponseError as e:
        if "NOGROUP" in str(e):
            return -1
        raise
    return int(info["pending"])


async def _wait_for_msg_processed(
    redis_client: redis.Redis,
    msg_id: bytes,
    deadline_seconds: float = 5.0,
) -> None:
    """Poll for ``pyforge:processed:{msg_id}`` — set by consumer after apply.

    Deterministic "all applied through msg_id" signal. The consumer
    SETs this side-table key immediately after ``layout.insert``
    succeeds (BEFORE flush). Polling for the LAST msg_id's processed
    key is the cleanest "consumer caught up" check.
    """
    deadline = time.monotonic() + deadline_seconds
    key = PROCESSED_KEY_PREFIX + msg_id
    while time.monotonic() < deadline:
        if redis_client.exists(key):
            return
        await asyncio.sleep(0.01)
    raise AssertionError(f"consumer did not process msg_id={msg_id!r} within {deadline_seconds}s")


async def _drain_consumer(
    consumer: WALConsumer,
    redis_client: redis.Redis,
    deadline_seconds: float = 5.0,
) -> None:
    """Wait until consumer.pending_ack is empty AND PEL is drained.

    Yields control once before polling (consumer.run() has only just
    been scheduled — give it a chance to actually start). Polls every
    50ms (matches test_offline_e2e.py pattern). Without the initial
    yield, the first poll can fire BEFORE the consumer has read any
    messages from the stream — at which point pending=0 AND
    pending_ack=[] are both trivially true and we return prematurely.
    """
    await asyncio.sleep(0)  # yield once so consumer can advance
    deadline = time.monotonic() + deadline_seconds
    while time.monotonic() < deadline:
        pending = _pending_count(redis_client)
        if pending == 0 and not consumer._pending_ack:
            return
        await asyncio.sleep(0.05)
    raise AssertionError(
        f"consumer did not drain within {deadline_seconds}s — "
        f"pending={_pending_count(redis_client)}, "
        f"pending_ack={len(consumer._pending_ack)}"
    )


async def _wait_for_liveness_set(async_redis: redis.asyncio.Redis, timeout: float = 2.0) -> None:
    """Poll for the liveness key. Force-first-refresh sets it during run()'s
    startup awaits (~50-100ms expected); 2s is the safety backstop."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if await async_redis.get(KEY_WAL_CONSUMER_LIVENESS) is not None:
            return
        await asyncio.sleep(0.01)
    raise AssertionError(f"liveness key {KEY_WAL_CONSUMER_LIVENESS!r} not set within {timeout}s")


# ---------------------------------------------------------------------------
# Fixtures.
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _fresh_caches() -> None:
    clear_pydantic_cache()
    clear_row_pack_cache()
    clear_arrow_cache()


@pytest.fixture
def registry(redis_client: redis.Redis) -> SegmentRegistry:
    """Real SegmentRegistry against the test Redis."""
    return SegmentRegistry(redis_client)


@pytest.fixture
async def async_redis(redis_client: redis.Redis):
    """Async Redis paired with the conftest's sync redis_client.

    The unused ``redis_client`` parameter establishes a fixture
    dependency so the sync-side reachability skip applies before the
    async one is built — mirrors test_offline_e2e.py's pattern.
    """
    del redis_client
    url = os.environ.get("PYFORGE_REDIS_URL", "redis://127.0.0.1:6379/0")
    client = redis.asyncio.Redis.from_url(url, decode_responses=False)
    try:
        await client.ping()
    except Exception:
        pytest.skip("Async Redis not reachable")
    yield client
    await client.aclose()


@pytest.fixture
def parquet_store(tmp_path: Path) -> ParquetDatasetStore:
    return ParquetDatasetStore(tmp_path)


def _drop_current(
    registry: SegmentRegistry, redis_client: redis.Redis, schema: type[FeatureSchema]
) -> None:
    """Operator simulation: drop the current segment so a fresh hydrate runs.

    This is the "operator wants to re-hydrate" path. The Step 14 close-Lua
    extension clears ``pyforge:schema:{name}:current`` automatically when
    the closing process is the last holder (refcount-0 path) — this
    eliminates the workaround for the **producer-consumer pipeline**
    cleanup (close + drain after the pipeline already drove refcount to 0).

    But ``hydrate()`` deliberately does NOT call ``registry.close()`` on
    the segment it created — the segment is left alive at refcount=1 for
    serving processes to ``open_current`` and use. ``_drop_current`` is
    simulating an operator who decides "I want to re-hydrate" with no
    serving consumers around: the refcount=1 is a "ghost hold" that
    open-then-close cannot decrement past, so the helper combines the
    Step-14-aware close (drains cleanup_queue) with an explicit DEL of
    ``schema:current`` to mimic the operator's ``redis-cli DEL`` that
    ADR-013's runbook documents.

    Operator runbook (ADR-013) — for unattended workflows, this is what
    the watchdog will do over its 150 s detection ceiling once the
    operator's hydrate process exits. For attended workflows, the
    operator uses ``redis-cli DEL pyforge:schema:X:current`` directly
    after confirming no live serving consumers — same shape as the
    explicit DEL below.
    """
    with contextlib.suppress(SegmentNotFoundError):
        seg = registry.open_current(schema)
        registry.close(seg)
    drain_cleanup_queue(redis_client)
    # Operator-style DEL for hydrate's ghost-hold refcount=1.
    redis_client.delete(f"pyforge:schema:{schema.__name__}:current")


async def _run_producer_consumer_pipeline(
    redis_client: redis.Redis,
    async_redis: redis.asyncio.Redis,
    registry: SegmentRegistry,
    parquet_store: ParquetDatasetStore,
    n_messages: int,
    *,
    consumer_block_ms: int = 50,
    consumer_batch_count: int = 100,
    consumer_max_pending_ack: int = 20,
    consumer_name_suffix: str = "e2e",
) -> list[bytes]:
    """Boilerplate: spin up consumer + segment, write n_messages from a
    producer, drain the consumer, return the producer's emitted msg_ids.
    Caller owns the segment lifecycle (releases via registry.close + drain).
    """
    seg = registry.create(_IntHydrate, capacity=max(2 * n_messages, 16))
    try:
        producer = WALProducer(redis_client)
        consumer = WALConsumer(
            async_redis,  # type: ignore[arg-type]
            segments={"_IntHydrate": seg},
            offline=parquet_store,
            consumer_name=f"hydrate-int-{consumer_name_suffix}",
            block_ms=consumer_block_ms,
            batch_count=consumer_batch_count,
            flush_interval_seconds=0.5,
            max_pending_ack=consumer_max_pending_ack,
        )

        msg_ids = [producer.write(_IntHydrate, f"ent-{i}", _values(i)) for i in range(n_messages)]

        run_task = asyncio.create_task(consumer.run())
        try:
            # Deterministic signal: wait for the LAST message's processed
            # key (set by consumer right after layout.insert). Without
            # this, _drain_consumer can return immediately on the false
            # "drained" condition (pending=0 + pending_ack=[]) BEFORE
            # the consumer has read anything.
            await _wait_for_msg_processed(
                redis_client, msg_ids[-1], deadline_seconds=max(2.0, n_messages * 0.01)
            )
            await _drain_consumer(
                consumer, redis_client, deadline_seconds=max(2.0, n_messages * 0.01)
            )
        finally:
            await consumer.stop()
            await asyncio.wait_for(run_task, timeout=5.0)
        await parquet_store.flush()
    finally:
        registry.close(seg)
        drain_cleanup_queue(redis_client)
        # Step 14: registry.close's extended Lua now clears
        # pyforge:schema:_IntHydrate:current at refcount-0 (rotation-safe).
        # The historical manual `redis_client.delete(_key_current(...))`
        # workaround is gone.

    return msg_ids


# ---------------------------------------------------------------------------
# E1 — Happy-path roundtrip.
# ---------------------------------------------------------------------------


async def test_e1_hydrate_roundtrip_basic(
    redis_client: redis.Redis,
    async_redis: redis.asyncio.Redis,
    registry: SegmentRegistry,
    parquet_store: ParquetDatasetStore,
) -> None:
    """100 entities written, consumed, flushed, hydrated. Round-trip clean."""
    n = 100
    await _run_producer_consumer_pipeline(
        redis_client,
        async_redis,
        registry,
        parquet_store,
        n,
        consumer_name_suffix="e1",
    )

    # Run hydrate against the offline store.
    result = hydrate(_IntHydrate, parquet_store, registry, redis_client=redis_client)

    assert result.entity_count == n, f"hydrated {result.entity_count} entities, expected {n}"

    # Verify every entity is findable in the new segment.
    new_seg = registry.open_current(_IntHydrate)
    try:
        for i in range(n):
            entity_id = f"ent-{i}"
            offset = lookup(new_seg, entity_id)
            assert offset is not None, f"entity {entity_id} missing from new segment"
    finally:
        registry.close(new_seg)
        drain_cleanup_queue(redis_client)


# ---------------------------------------------------------------------------
# E2 — hydrate count = parquet row count, NOT producer count.
# ---------------------------------------------------------------------------


async def test_e2_hydrate_count_matches_parquet_row_count_not_producer_count(
    redis_client: redis.Redis,
    async_redis: redis.asyncio.Redis,
    registry: SegmentRegistry,
    parquet_store: ParquetDatasetStore,
    tmp_path: Path,
) -> None:
    """Locked invariant: hydrate's count = parquet rows, NOT producer count.

    Setup: 100 distinct entity_ids, exactly 1 message per entity.
    `result.entity_count` (post-latest_features dedup) == parquet rows
    only when these invariants hold. A future maintainer who writes 2
    events/entity gets a loud assertion failure here.

    Drive consumer one iter at a time with batch_count=10 so we land
    in a partial state (some applied, some still on the stream). Then
    explicit flush, read parquet directly, compare counts.
    """
    n = 100
    entity_ids = [f"ent-{i}" for i in range(n)]
    assert len(set(entity_ids)) == n, "1 msg/entity invariant broken at setup"
    assert len(entity_ids) == n

    seg = registry.create(_IntHydrate, capacity=256)
    try:
        producer = WALProducer(redis_client)
        # batch_count=10 — default 100 would let one iter grab everything,
        # defeating the partial-state goal. Verified at pyforge/wal_consumer.py:287.
        consumer = WALConsumer(
            async_redis,  # type: ignore[arg-type]
            segments={"_IntHydrate": seg},
            offline=parquet_store,
            consumer_name="hydrate-int-e2",
            block_ms=50,
            batch_count=10,
            flush_interval_seconds=300.0,  # don't auto-flush during the test
            max_pending_ack=10_000,
        )

        # Producer writes all n messages first so they're available on the
        # stream before we start driving iters.
        for entity_id, i in zip(entity_ids, range(n), strict=True):
            producer.write(_IntHydrate, entity_id, _values(i))

        # Acquire consumer's startup state so _run_one_iter() is callable.
        # This mirrors what consumer.run() does in its prelude.
        await consumer._acquire_consumer_lock()
        try:
            await consumer._ensure_group()
            await consumer._drain_pel()  # nothing to drain on a fresh group
            consumer._pid_str = str(os.getpid()).encode("ascii")
            consumer._liveness_last_refresh = time.monotonic()

            # Drive 7 iters. With batch_count=10, applies up to 70 messages.
            # XREADGROUP BLOCK semantics may apply fewer per iter; that's
            # fine, we don't care about the exact count, only that it's
            # > 0 AND < n.
            for _ in range(7):
                await consumer._run_one_iter()

            # Explicit flush — without this, default max_pending_ack=10000
            # never auto-flushes for n=100 and parquet has 0 rows. Test
            # would pass trivially via EmptyDatasetError if we forgot.
            await parquet_store.flush()
        finally:
            await consumer._release_consumer_lock()
    finally:
        registry.close(seg)
        drain_cleanup_queue(redis_client)
        # Step 14: close-Lua clears schema:current at refcount-0.

    # Read parquet directly via pa.dataset to get the authoritative
    # "what's actually flushed" count. This is the contract being locked.
    dataset = pa_ds.dataset(tmp_path, partitioning="hive")
    flushed_count = dataset.count_rows()
    assert 0 < flushed_count < n, (
        f"expected partial state (0 < flushed < {n}), got {flushed_count} — "
        f"either auto-flushed too aggressively or not at all"
    )

    # Now hydrate. The contract: result.entity_count == flushed_count.
    result = hydrate(_IntHydrate, parquet_store, registry, redis_client=redis_client)
    assert result.entity_count == flushed_count, (
        f"hydrate count {result.entity_count} != parquet row count {flushed_count} — "
        f"hydrate is reading something other than the offline store"
    )

    # Cleanup.
    new_seg = registry.open_current(_IntHydrate)
    registry.close(new_seg)
    drain_cleanup_queue(redis_client)


# ---------------------------------------------------------------------------
# E3 — hydrate refuses when consumer alive (real liveness key).
# ---------------------------------------------------------------------------


async def test_e3_hydrate_refuses_when_consumer_alive_real_redis(
    redis_client: redis.Redis,
    async_redis: redis.asyncio.Redis,
    registry: SegmentRegistry,
    parquet_store: ParquetDatasetStore,
) -> None:
    """A live consumer's liveness key blocks hydrate via precondition #2."""
    seg = registry.create(_IntHydrate, capacity=64)
    try:
        consumer = WALConsumer(
            async_redis,  # type: ignore[arg-type]
            segments={"_IntHydrate": seg},
            offline=parquet_store,
            consumer_name="hydrate-int-e3",
            block_ms=50,
            flush_interval_seconds=10.0,
        )

        run_task = asyncio.create_task(consumer.run())
        try:
            # Force-first-refresh sets the liveness key during run()'s
            # startup awaits — poll for it (~50-100ms expected, 2s
            # backstop).
            await _wait_for_liveness_set(async_redis)

            # Drop the registry's current pointer so precondition #1
            # passes; we want to hit precondition #2.
            redis_client.delete("pyforge:schema:_IntHydrate:current")

            with pytest.raises(HydrationConflictError, match="WAL consumer"):
                hydrate(_IntHydrate, parquet_store, registry, redis_client=redis_client)
        finally:
            await consumer.stop()
            await asyncio.wait_for(run_task, timeout=5.0)
    finally:
        # Best-effort cleanup; the consumer's run() should DEL the
        # liveness key on shutdown but be defensive.
        with contextlib.suppress(Exception):
            redis_client.delete(KEY_WAL_CONSUMER_LIVENESS)
        with contextlib.suppress(SegmentNotFoundError):
            current = registry.open_current(_IntHydrate)
            registry.close(current)
        with contextlib.suppress(Exception):
            registry.close(seg)
        drain_cleanup_queue(redis_client)


# ---------------------------------------------------------------------------
# E4 (#18) — hydrate proceeds after liveness TTL expires.
# ---------------------------------------------------------------------------


async def test_e4_hydrate_proceeds_after_liveness_expires(
    redis_client: redis.Redis,
    async_redis: redis.asyncio.Redis,
    registry: SegmentRegistry,
    parquet_store: ParquetDatasetStore,
) -> None:
    """Locked behavior: real Redis TTL expires the liveness key; hydrate then proceeds.

    This is test #18 from Commit A's catalog, deferred because mocked
    Redis can't actually expire TTLs. Real Redis does.
    """
    # Need at least one offline-flushed entity for hydrate to have
    # something to load; otherwise EmptyDatasetError fires before the
    # liveness check matters.
    await _run_producer_consumer_pipeline(
        redis_client,
        async_redis,
        registry,
        parquet_store,
        5,
        consumer_name_suffix="e4-warmup",
    )

    # Set the liveness key with a 1s TTL.
    redis_client.set(KEY_WAL_CONSUMER_LIVENESS, b"99999", ex=1)
    assert redis_client.get(KEY_WAL_CONSUMER_LIVENESS) is not None

    # First call should fail (key still set).
    with pytest.raises(HydrationConflictError, match="WAL consumer"):
        hydrate(_IntHydrate, parquet_store, registry, redis_client=redis_client)

    # Wait for TTL expiry.
    time.sleep(1.2)
    assert redis_client.get(KEY_WAL_CONSUMER_LIVENESS) is None, (
        "liveness key did not expire — Redis TTL semantics broken or test timing flaked"
    )

    # Now hydrate should proceed.
    result = hydrate(_IntHydrate, parquet_store, registry, redis_client=redis_client)
    assert result.entity_count == 5

    # Cleanup.
    new_seg = registry.open_current(_IntHydrate)
    registry.close(new_seg)
    drain_cleanup_queue(redis_client)


# ---------------------------------------------------------------------------
# E5 (#21) — as_of_time_ns honored end-to-end via real Parquet.
# ---------------------------------------------------------------------------


async def test_e5_hydrate_as_of_time_excludes_future_writes(
    redis_client: redis.Redis,
    async_redis: redis.asyncio.Redis,
    registry: SegmentRegistry,
    parquet_store: ParquetDatasetStore,
) -> None:
    """Locked: as_of_time_ns filter applied at row-level end-to-end.

    Two writes for the SAME entity at t1 and t2 (1ms apart, same
    event_date partition). Two explicit flushes produce two files in
    the same partition. Cutoff between t1 and t2 (integer arithmetic)
    must yield t1's value, not t2's.

    Row-level filter test, NOT partition-pruning. Partition-pruning
    across event_date boundaries is a Step 16 concern.
    """
    seg = registry.create(_IntHydrate, capacity=16)
    try:
        producer = WALProducer(redis_client)
        consumer = WALConsumer(
            async_redis,  # type: ignore[arg-type]
            segments={"_IntHydrate": seg},
            offline=parquet_store,
            consumer_name="hydrate-int-e5",
            block_ms=50,
            flush_interval_seconds=300.0,  # don't auto-flush
        )

        # Integer-only timestamps (no float intermediates).
        t1 = 1_700_000_000_000_000_000
        t2 = t1 + 1_000_000  # 1ms later
        cutoff_ns = (t1 + t2) // 2
        assert t1 < cutoff_ns < t2, (
            f"setup error: cutoff {cutoff_ns} not strictly between t1={t1} and t2={t2}"
        )

        run_task = asyncio.create_task(consumer.run())
        try:
            # Write at t1. Producer accepts event_time_ns kwarg.
            # NOTE: with flush_interval_seconds=300 we cannot use
            # _drain_consumer (which waits for pending_ack=[] — only
            # cleared by the consumer's flush task). _wait_for_msg_processed
            # is sufficient: by the time pyforge:processed:{msg_id} is
            # set, the apply order has run layout.insert + offline.append,
            # so the row is already buffered in parquet_store. The explicit
            # flush() then writes it. The pending_ack staying non-empty is
            # fine — it only delays XACK (PEL drain), not parquet content.
            msg_id_1 = producer.write(
                _IntHydrate,
                "user-A",
                {"a": 1.0, "b": 100, "emb": [1.0, 1.0, 1.0, 1.0]},
                event_time_ns=t1,
            )
            await _wait_for_msg_processed(redis_client, msg_id_1, deadline_seconds=5.0)
            await parquet_store.flush()

            # Write at t2 for the same entity.
            msg_id_2 = producer.write(
                _IntHydrate,
                "user-A",
                {"a": 2.0, "b": 200, "emb": [2.0, 2.0, 2.0, 2.0]},
                event_time_ns=t2,
            )
            await _wait_for_msg_processed(redis_client, msg_id_2, deadline_seconds=5.0)
            await parquet_store.flush()
        finally:
            await consumer.stop()
            await asyncio.wait_for(run_task, timeout=5.0)
    finally:
        registry.close(seg)
        drain_cleanup_queue(redis_client)
        # Step 14: close-Lua clears schema:current at refcount-0.

    # Hydrate at the cutoff. Lookback covers both writes; the filter
    # MUST exclude t2 because cutoff < t2.
    # Use a wide lookback to ensure both writes fall within the window;
    # the as_of_time filter is what we're testing, not lookback.
    result = hydrate(
        _IntHydrate,
        parquet_store,
        registry,
        redis_client=redis_client,
        as_of_time_ns=cutoff_ns,
        lookback_days=365,  # ensure t1 is within the window
    )
    assert result.entity_count == 1

    # Verify the segment contains t1's value (a=1.0), NOT t2's (a=2.0).
    new_seg = registry.open_current(_IntHydrate)
    try:
        offset = lookup(new_seg, "user-A")
        assert offset is not None, "user-A missing from new segment"
        # Read the field 'a' bytes via the layout. Field offset comes from
        # the assembly table; we can grab it via the segment's layout.
        layout_obj = new_seg.layout
        # 'a' is the first declared field; assembly_byte_offsets[0] in
        # declaration order.
        a_offset_in_row = int(layout_obj.assembly_byte_offsets[0])
        a_bytes = bytes(new_seg.handle.buf[offset + a_offset_in_row : offset + a_offset_in_row + 4])
        a_value = struct.unpack("<f", a_bytes)[0]
        assert a_value == 1.0, (
            f"as_of_time filter failed: got a={a_value}, expected 1.0 (t1's value); "
            f"2.0 means t2's value leaked through"
        )
    finally:
        registry.close(new_seg)
        drain_cleanup_queue(redis_client)


# ---------------------------------------------------------------------------
# E6 — capacity_factor on real SegmentRegistry.create.
# ---------------------------------------------------------------------------


async def test_e6_hydrate_capacity_factor_real_segment(
    redis_client: redis.Redis,
    async_redis: redis.asyncio.Redis,
    registry: SegmentRegistry,
    parquet_store: ParquetDatasetStore,
) -> None:
    """100 entities x factor 4 -> registry.create called with capacity=400."""
    n = 100
    await _run_producer_consumer_pipeline(
        redis_client,
        async_redis,
        registry,
        parquet_store,
        n,
        consumer_name_suffix="e6",
    )

    result = hydrate(
        _IntHydrate,
        parquet_store,
        registry,
        redis_client=redis_client,
        capacity_factor=4.0,
    )
    assert result.entity_count == n

    new_seg = registry.open_current(_IntHydrate)
    try:
        # capacity 400 = max(int(100 * 4), 16) = 400.
        assert new_seg.layout.capacity == 400, (
            f"capacity {new_seg.layout.capacity} != 400 — capacity_factor not honored"
        )
    finally:
        registry.close(new_seg)
        drain_cleanup_queue(redis_client)


# ---------------------------------------------------------------------------
# E7 — idempotent: hydrate, drop, hydrate again.
# ---------------------------------------------------------------------------


async def test_e7_hydrate_idempotent_after_clean_drop(
    redis_client: redis.Redis,
    async_redis: redis.asyncio.Redis,
    registry: SegmentRegistry,
    parquet_store: ParquetDatasetStore,
) -> None:
    """Hydrate, drop, hydrate again — both runs succeed; segment names differ (UUID)."""
    n = 50
    await _run_producer_consumer_pipeline(
        redis_client,
        async_redis,
        registry,
        parquet_store,
        n,
        consumer_name_suffix="e7",
    )

    # First hydrate.
    result_1 = hydrate(_IntHydrate, parquet_store, registry, redis_client=redis_client)
    assert result_1.entity_count == n

    # Drop the current segment (operator simulation: stop consumer, drop key, drain).
    _drop_current(registry, redis_client, _IntHydrate)

    # Second hydrate — should succeed with a NEW segment name (UUID-distinct).
    result_2 = hydrate(_IntHydrate, parquet_store, registry, redis_client=redis_client)
    assert result_2.entity_count == n
    assert result_2.segment_name != result_1.segment_name, (
        f"segment name reused across hydrate calls: {result_1.segment_name!r} == "
        f"{result_2.segment_name!r}; expected UUID-distinct names"
    )

    # Cleanup.
    new_seg = registry.open_current(_IntHydrate)
    registry.close(new_seg)
    drain_cleanup_queue(redis_client)
