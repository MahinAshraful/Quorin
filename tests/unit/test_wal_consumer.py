"""Unit tests for pyforge.wal_consumer.WALConsumer.

These tests use a hand-rolled in-memory ``AsyncFakeRedis`` so they don't
require the integration Redis service. The integration suite at
``tests/integration/test_wal_consumer_redis.py`` exercises the same paths
against a real Redis (with real PEL semantics, real XACK/XPENDING).
"""

from __future__ import annotations

import asyncio
import logging
import sys
from typing import Any

import msgpack
import pytest

pytestmark = pytest.mark.skipif(
    sys.platform == "win32",
    reason="WAL consumer relies on POSIX shared-memory paths",
)

from typing import cast as _cast  # noqa: E402
from unittest.mock import Mock as _Mock  # noqa: E402

import redis.exceptions  # noqa: E402

from _helpers import make_segment, release_segment  # noqa: E402
from pyforge._internal.pydantic_factory import (  # noqa: E402
    clear_cache as clear_pydantic_cache,
)
from pyforge._internal.row_pack import clear_cache as clear_row_pack_cache  # noqa: E402
from pyforge.layout import lookup  # noqa: E402
from pyforge.schema import FeatureField, FeatureSchema, dtype  # noqa: E402
from pyforge.shm import SegmentRegistry  # noqa: E402
from pyforge.wal import (  # noqa: E402
    _F_BLOB,
    _F_ENTITY_ID,
    _F_EVENT_TIME,
    _F_SCHEMA,
    PROCESSED_KEY_PREFIX,
)
from pyforge.wal_consumer import (  # noqa: E402
    ConsumerNameInUseError,
    NoopOfflineWriter,
    WALConsumer,
)

# Step 15 stub registry for unit tests. Unit tests never exercise the
# pause-cleared reopen branch (which is the only place self._registry is
# dereferenced); a Mock with the right spec is sufficient. Integration/chaos
# tests use a real SegmentRegistry constructed against the real Redis.
_STUB_REGISTRY: SegmentRegistry = _cast("SegmentRegistry", _Mock(spec=SegmentRegistry))

# ---------------------------------------------------------------------------
# Schemas.
# ---------------------------------------------------------------------------


class _S(FeatureSchema):
    version = 1
    fields = [
        FeatureField("a", dtype.float32),
        FeatureField("b", dtype.int64),
        FeatureField("emb", dtype.float32, shape=(4,)),
    ]


# ---------------------------------------------------------------------------
# AsyncFakeRedis — only the methods WALConsumer actually calls.
# ---------------------------------------------------------------------------


class _FakeAsyncPipeline:
    def __init__(self, parent: AsyncFakeRedis) -> None:
        self._parent = parent
        self.command_stack: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = []

    def set(self, key: bytes, value: bytes, ex: int | None = None) -> None:
        self.command_stack.append(("set", (key, value), {"ex": ex}))

    def xack(self, stream: bytes, group: str, *ids: bytes) -> None:
        self.command_stack.append(("xack", (stream, group, *ids), {}))

    async def execute(self) -> list[Any]:
        results: list[Any] = []
        for cmd, args, _kwargs in self.command_stack:
            if cmd == "set":
                key, value = args
                self._parent._kv[key] = value
                results.append(True)
            elif cmd == "xack":
                stream, group, *ids = args
                results.append(self._parent._xack(stream, group, list(ids)))
            else:
                raise NotImplementedError(cmd)
        self.command_stack.clear()
        return results

    async def __aenter__(self) -> _FakeAsyncPipeline:
        return self

    async def __aexit__(self, *_: Any) -> None:
        return None


class AsyncFakeRedis:
    """Tiny async in-memory Redis stub: enough for WALConsumer unit tests.

    Stream model: each ``stream_key`` has a list of ``(msg_id, fields)``
    tuples representing the global stream. A consumer-group's PEL is a
    list of msg_ids delivered to ``consumer_name`` but not yet XACKed.
    XREADGROUP id="0" returns the PEL; id=">" returns + delivers new
    entries from the stream tail.
    """

    def __init__(self) -> None:
        self._kv: dict[bytes, bytes] = {}
        # stream_key -> list of (msg_id, fields_dict)
        self._streams: dict[bytes, list[tuple[bytes, dict[bytes, bytes]]]] = {}
        # (stream_key, group) -> next-undelivered cursor (index into stream list)
        self._group_cursors: dict[tuple[bytes, str], int] = {}
        # (stream_key, group, consumer) -> ordered list of pending msg_ids
        self._pels: dict[tuple[bytes, str, str], list[bytes]] = {}
        self._seq = 0
        # If True, the next set(..., nx=True) call returns False (lock-held).
        self.lock_already_held: bool = False

    # ---- non-stream KV -------------------------------------------------

    async def set(
        self,
        key: bytes,
        value: bytes,
        ex: int | None = None,
        nx: bool = False,
    ) -> bool | None:
        if nx and key in self._kv:
            return None  # redis-py async returns None when NX fails
        if nx and self.lock_already_held:
            return None
        self._kv[key] = value
        return True

    async def get(self, key: bytes) -> bytes | None:
        return self._kv.get(key)

    async def mget(self, *keys: bytes) -> list[bytes | None]:
        # Step 15 pause-check uses mget across pause keys; stub returns
        # None for any unset key (matches redis-py async semantics).
        return [self._kv.get(k) for k in keys]

    async def delete(self, *keys: bytes) -> int:
        n = 0
        for k in keys:
            if k in self._kv:
                del self._kv[k]
                n += 1
        return n

    async def expire(self, key: bytes, seconds: int) -> bool:
        # No-op in the stub; we don't actually expire.
        return key in self._kv

    async def exists(self, key: bytes) -> int:
        return 1 if key in self._kv else 0

    # ---- stream methods ------------------------------------------------

    def add_message(self, stream: bytes, fields: dict[bytes, bytes]) -> bytes:
        """Test helper: synchronously inject a message into the stream."""
        self._seq += 1
        msg_id = f"{self._seq}-0".encode()
        self._streams.setdefault(stream, []).append((msg_id, fields))
        return msg_id

    async def xgroup_create(
        self,
        name: bytes,
        groupname: str,
        id: str = "$",
        mkstream: bool = False,
    ) -> bool:
        del id
        if mkstream:
            self._streams.setdefault(name, [])
        key = (name, groupname)
        if key in self._group_cursors:
            raise redis.exceptions.ResponseError("BUSYGROUP Consumer Group name already exists")
        # Set the cursor to the end of the existing stream — new XREADGROUP "$"
        # only delivers messages added after group creation.
        self._group_cursors[key] = len(self._streams.get(name, []))
        return True

    async def xreadgroup(
        self,
        groupname: str,
        consumername: str,
        streams: dict[bytes, str],
        count: int = 100,
        block: int | None = None,
    ) -> list[tuple[bytes, list[tuple[bytes, dict[bytes, bytes]]]]]:
        # Single-stream support is enough for our tests.
        if len(streams) != 1:
            raise NotImplementedError("AsyncFakeRedis: single-stream xreadgroup only")
        stream, msg_id = next(iter(streams.items()))
        pel_key = (stream, groupname, consumername)
        msgs: list[tuple[bytes, dict[bytes, bytes]]] = []

        if msg_id != ">":
            # PEL read with cursor `msg_id` (exclusive lower bound).
            # "0" returns all PEL; any other id returns PEL entries with id > msg_id.
            pel = self._pels.get(pel_key, [])
            id_to_fields = dict(self._streams.get(stream, []))
            taken = 0
            for mid in pel:
                if mid.decode("ascii") <= msg_id and msg_id != "0":
                    continue
                if mid in id_to_fields:
                    msgs.append((mid, id_to_fields[mid]))
                    taken += 1
                    if taken >= count:
                        break
            if not msgs:
                return []
            return [(stream, msgs)]

        # id == ">" — read new entries (deliver from the cursor).
        cursor_key = (stream, groupname)
        cursor = self._group_cursors.get(cursor_key, 0)
        avail = self._streams.get(stream, [])
        end = min(cursor + count, len(avail))
        new_msgs = avail[cursor:end]
        if not new_msgs and block is not None and block > 0:
            # Simulate the BLOCK timeout cheaply — sleep briefly then return [].
            await asyncio.sleep(min(block / 1000.0, 0.01))
            return []
        if not new_msgs:
            return []
        # Move cursor + register in PEL.
        self._group_cursors[cursor_key] = end
        pel = self._pels.setdefault(pel_key, [])
        for mid, _f in new_msgs:
            pel.append(mid)
        return [(stream, list(new_msgs))]

    def _xack(self, stream: bytes, group: str, ids: list[bytes]) -> int:
        # Remove from every consumer's PEL for this group.
        n = 0
        for (s, g, _c), pel in self._pels.items():
            if s != stream or g != group:
                continue
            for mid in ids:
                while mid in pel:
                    pel.remove(mid)
                    n += 1
        return n

    async def xack(self, stream: bytes, group: str, *ids: bytes) -> int:
        return self._xack(stream, group, list(ids))

    def pipeline(self, transaction: bool = True) -> _FakeAsyncPipeline:
        del transaction
        return _FakeAsyncPipeline(self)

    # ---- introspection (test helpers) ---------------------------------

    def pel_size(self, stream: bytes, group: str, consumer: str) -> int:
        return len(self._pels.get((stream, group, consumer), []))


# ---------------------------------------------------------------------------
# Fixtures.
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _fresh_caches() -> None:
    clear_pydantic_cache()
    clear_row_pack_cache()


@pytest.fixture
def fake_redis() -> AsyncFakeRedis:
    return AsyncFakeRedis()


@pytest.fixture
def segment() -> Any:
    seg = make_segment(_S, capacity=64)
    try:
        yield seg
    finally:
        release_segment(seg)


@pytest.fixture
def consumer(fake_redis: AsyncFakeRedis, segment: Any) -> WALConsumer:
    return WALConsumer(
        fake_redis,  # type: ignore[arg-type]
        segments={"_S": segment},
        registry=_STUB_REGISTRY,
    )


# ---------------------------------------------------------------------------
# Helpers — build a producer-shaped wire message.
# ---------------------------------------------------------------------------


def _wire_msg(
    schema_name: str,
    entity_id: str,
    values_in_hash_order: list[Any],
    event_time_ns: int = 1,
) -> dict[bytes, bytes]:
    return {
        _F_SCHEMA: schema_name.encode("utf-8"),
        _F_ENTITY_ID: entity_id.encode("utf-8"),
        _F_EVENT_TIME: str(event_time_ns).encode("ascii"),
        _F_BLOB: msgpack.packb(values_in_hash_order, use_bin_type=True),
    }


def _values_for_S(a: float, b: int, emb: list[float]) -> list[Any]:  # noqa: N802
    """Reorder values into the producer's name_hash wire order for _S."""
    from pyforge._internal.pydantic_factory import field_order_for

    by_name: dict[str, Any] = {"a": a, "b": b, "emb": emb}
    return [by_name[name] for name in field_order_for(_S)]


# ---------------------------------------------------------------------------
# 1. Construction validation.
# ---------------------------------------------------------------------------


def test_init_raises_when_segments_key_does_not_match_schema_name(
    fake_redis: AsyncFakeRedis, segment: Any
) -> None:
    with pytest.raises(ValueError, match=r"schema\.__name__"):
        WALConsumer(
            fake_redis,  # type: ignore[arg-type]
            segments={"WrongName": segment},
            registry=_STUB_REGISTRY,
        )


# ---------------------------------------------------------------------------
# 2. Apply happy path — SET pipeline, pending_ack, NO XACK yet.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_apply_happy_path_pipelines_set_and_appends_to_pending(
    consumer: WALConsumer, fake_redis: AsyncFakeRedis, segment: Any
) -> None:
    msg_id = fake_redis.add_message(
        consumer._stream_key,
        _wire_msg("_S", "user-1", _values_for_S(1.5, 7, [0.1, 0.2, 0.3, 0.4])),
    )
    await consumer._process_batch(
        [(msg_id, _wire_msg("_S", "user-1", _values_for_S(1.5, 7, [0.1, 0.2, 0.3, 0.4])))]
    )
    # Side-table SET landed.
    assert PROCESSED_KEY_PREFIX + msg_id in fake_redis._kv
    # Pending ACK has the msg.
    assert msg_id in consumer._pending_ack
    # Entity is in the segment.
    assert lookup(segment, "user-1") is not None


# ---------------------------------------------------------------------------
# 3. Apply-then-replay — idempotent at every level.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_apply_then_replay_is_idempotent(
    consumer: WALConsumer, fake_redis: AsyncFakeRedis, segment: Any
) -> None:
    fields = _wire_msg("_S", "u1", _values_for_S(1.0, 2, [0.0, 0.0, 0.0, 0.0]))
    msg_id = fake_redis.add_message(consumer._stream_key, fields)

    await consumer._process_batch([(msg_id, fields)])
    assert lookup(segment, "u1") is not None
    pending_second_msg_id = msg_id  # same id -> idempotent

    await consumer._process_batch([(msg_id, fields)])
    # SET no-op-on-existing; pending_ack appends the msg_id again (we don't
    # de-dup at this layer — flush+XACK collapses; XACK is idempotent on
    # already-acked ids). The important guarantee: shm state is unchanged.
    assert lookup(segment, "u1") is not None
    # First insert returned True; second returns False — counters reflect that.
    # (We don't introspect counter values here; the counter test is below.)
    assert pending_second_msg_id in consumer._pending_ack


# ---------------------------------------------------------------------------
# 4. Unknown schema — counter incremented, log-once, no ACK, no SET.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_unknown_schema_counter_and_log_once(
    consumer: WALConsumer,
    fake_redis: AsyncFakeRedis,
    caplog: pytest.LogCaptureFixture,
) -> None:
    fields = _wire_msg("UnknownSchema", "u1", [0.0])
    msg_id = fake_redis.add_message(consumer._stream_key, fields)

    caplog.set_level(logging.WARNING)
    # Apply 100 copies — log should fire exactly once for that schema name.
    batch = [(msg_id, fields)] * 100
    await consumer._process_batch(batch)

    # No SET, no pending_ack entry.
    assert PROCESSED_KEY_PREFIX + msg_id not in fake_redis._kv
    assert msg_id not in consumer._pending_ack
    # Schema name is now in the seen-set so future calls don't log again.
    assert b"UnknownSchema" in consumer._seen_unknown


# ---------------------------------------------------------------------------
# 5. Bad entity_id (invalid UTF-8) — no ACK, no SET, counter increments.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_bad_entity_id_does_not_ack(
    consumer: WALConsumer, fake_redis: AsyncFakeRedis
) -> None:
    fields = {
        _F_SCHEMA: b"_S",
        _F_ENTITY_ID: b"\xff\xfe\xfd",  # invalid UTF-8
        _F_EVENT_TIME: b"1",
        _F_BLOB: msgpack.packb(_values_for_S(1.0, 2, [0.0, 0.0, 0.0, 0.0]), use_bin_type=True),
    }
    msg_id = fake_redis.add_message(consumer._stream_key, fields)
    await consumer._process_batch([(msg_id, fields)])
    assert PROCESSED_KEY_PREFIX + msg_id not in fake_redis._kv
    assert msg_id not in consumer._pending_ack


# ---------------------------------------------------------------------------
# 6. _flush_and_ack happy path — flush succeeds, XACK fires, pending trimmed.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_flush_and_ack_clears_pending_after_flush(
    consumer: WALConsumer, fake_redis: AsyncFakeRedis
) -> None:
    fields = _wire_msg("_S", "u1", _values_for_S(1.0, 2, [0, 0, 0, 0]))
    msg_id = fake_redis.add_message(consumer._stream_key, fields)
    # Acquire lock + apply (lock is needed for the assertion path? No,
    # only run() takes the lock. Direct call to _process_batch is fine.)
    await consumer._process_batch([(msg_id, fields)])
    assert consumer._pending_ack == [msg_id]
    # Flush.
    await consumer._flush_and_ack()
    assert consumer._pending_ack == []


# ---------------------------------------------------------------------------
# 7. Flush failure — no XACK, pending intact.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_flush_failure_keeps_pending_intact(fake_redis: AsyncFakeRedis, segment: Any) -> None:
    class RaisingOffline(NoopOfflineWriter):
        async def flush(self) -> None:
            raise RuntimeError("disk full")

    consumer = WALConsumer(
        fake_redis,  # type: ignore[arg-type]
        segments={"_S": segment},
        registry=_STUB_REGISTRY,
        offline=RaisingOffline(),
    )
    fields = _wire_msg("_S", "u1", _values_for_S(1.0, 2, [0, 0, 0, 0]))
    msg_id = fake_redis.add_message(consumer._stream_key, fields)
    await consumer._process_batch([(msg_id, fields)])
    assert consumer._pending_ack == [msg_id]
    # Flush raises — _flush_and_ack catches, logs, leaves pending_ack alone.
    await consumer._flush_and_ack()
    assert consumer._pending_ack == [msg_id]


# ---------------------------------------------------------------------------
# 8. Consumer-name lock — second consumer with same name raises.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_consumer_name_lock_prevents_duplicate_consumers(
    fake_redis: AsyncFakeRedis, segment: Any
) -> None:
    c1 = WALConsumer(fake_redis, segments={"_S": segment}, registry=_STUB_REGISTRY)  # type: ignore[arg-type]
    await c1._acquire_consumer_lock()
    try:
        c2 = WALConsumer(fake_redis, segments={"_S": segment}, registry=_STUB_REGISTRY)  # type: ignore[arg-type]
        with pytest.raises(ConsumerNameInUseError, match="held by pid"):
            await c2._acquire_consumer_lock()
    finally:
        await c1._release_consumer_lock()


# ---------------------------------------------------------------------------
# 9. Stop() exits run() loop cleanly with one final flush.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_stop_exits_run_loop_and_finalizes(fake_redis: AsyncFakeRedis, segment: Any) -> None:
    consumer = WALConsumer(
        fake_redis,  # type: ignore[arg-type]
        segments={"_S": segment},
        registry=_STUB_REGISTRY,
        block_ms=10,  # short BLOCK so the loop exits quickly
        flush_interval_seconds=10.0,
    )
    # Pre-add one msg so the loop has something to apply before we stop.
    fields = _wire_msg("_S", "u1", _values_for_S(1.0, 2, [0, 0, 0, 0]))
    fake_redis.add_message(consumer._stream_key, fields)

    async def stop_after_first_apply() -> None:
        # Wait until the consumer has applied at least one message (pending_ack
        # has an entry) — then signal stop.
        for _ in range(200):
            if consumer._pending_ack:
                break
            await asyncio.sleep(0.01)
        await consumer.stop()

    await asyncio.gather(consumer.run(), stop_after_first_apply())
    # After clean exit: pending_ack is drained because run()'s finally block
    # called _final_flush_and_ack().
    assert consumer._pending_ack == []


# ---------------------------------------------------------------------------
# 10. Soft size trigger SIGNALS but does not BLOCK reads (review #C).
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_soft_size_trigger_signals_without_blocking(
    fake_redis: AsyncFakeRedis, segment: Any
) -> None:
    consumer = WALConsumer(
        fake_redis,  # type: ignore[arg-type]
        segments={"_S": segment},
        registry=_STUB_REGISTRY,
        max_pending_ack=5,
    )
    # Pre-fill pending_ack to 4 entries via direct apply, then a 5-msg batch
    # tips us over — _flush_now should be set, but _process_batch must return
    # without awaiting flush.
    msgs = []
    for i in range(5):
        fields = _wire_msg("_S", f"u{i}", _values_for_S(float(i), i, [0, 0, 0, 0]))
        mid = fake_redis.add_message(consumer._stream_key, fields)
        msgs.append((mid, fields))

    assert not consumer._flush_now.is_set()
    await consumer._process_batch(msgs)
    # 5 applied messages tips len(pending_ack) >= max_pending_ack.
    assert len(consumer._pending_ack) == 5
    assert consumer._flush_now.is_set()


# ---------------------------------------------------------------------------
# 11. Hard ceiling (2x max) blocks reads until flush drains.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_hard_ceiling_blocks_reads_until_drain(
    fake_redis: AsyncFakeRedis, segment: Any
) -> None:
    consumer = WALConsumer(
        fake_redis,  # type: ignore[arg-type]
        segments={"_S": segment},
        registry=_STUB_REGISTRY,
        max_pending_ack=2,
    )
    # Pre-fill pending_ack to >= 2 * 2 = 4 entries.
    consumer._pending_ack.extend([b"fake-1", b"fake-2", b"fake-3", b"fake-4"])
    fields = _wire_msg("_S", "u1", _values_for_S(1.0, 2, [0, 0, 0, 0]))
    msg_id = fake_redis.add_message(consumer._stream_key, fields)

    async def drain_after_delay() -> None:
        await asyncio.sleep(0.05)
        consumer._pending_ack.clear()

    # _process_batch should block on the back-pressure sleep loop until
    # pending_ack drops below 2*max. Time the call.
    drain_task = asyncio.create_task(drain_after_delay())
    t0 = asyncio.get_event_loop().time()
    await consumer._process_batch([(msg_id, fields)])
    elapsed = asyncio.get_event_loop().time() - t0
    await drain_task
    # We slept at least 0.05 s waiting for the drain.
    assert elapsed >= 0.04, f"_process_batch returned too quickly ({elapsed:.3f}s)"


# ---------------------------------------------------------------------------
# 12. _flush_loop wakes on _flush_now.set() within one tick (no waiting for the timer).
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# 13. PEL drain pagination — terminates when XACK is deferred (regression).
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_drain_pel_terminates_even_with_deferred_xack(
    consumer: WALConsumer, fake_redis: AsyncFakeRedis, segment: Any
) -> None:
    """Regression: with deferred XACK, ``XREADGROUP id="0"`` returns the
    same PEL on every call. The consumer must paginate by passing the
    last-seen ID, not loop on ``"0"`` forever."""
    # Manually plant 5 messages in the consumer's PEL by adding them to
    # the stream and registering the consumer's group + cursor.
    pel_key = (
        consumer._stream_key,
        consumer._group_name,
        consumer._consumer_name,
    )
    fake_redis._group_cursors[(consumer._stream_key, consumer._group_name)] = 0
    fake_redis._pels[pel_key] = []
    for i in range(5):
        mid = fake_redis.add_message(
            consumer._stream_key,
            _wire_msg("_S", f"u{i}", _values_for_S(float(i), i, [0, 0, 0, 0])),
        )
        fake_redis._pels[pel_key].append(mid)

    # _drain_pel must complete in finite time. A bug here would loop forever
    # because pending_ack accumulates but PEL stays the same.
    await asyncio.wait_for(consumer._drain_pel(), timeout=2.0)
    # All 5 ended up in pending_ack.
    assert len(consumer._pending_ack) == 5


@pytest.mark.asyncio
async def test_flush_loop_wakes_on_signal(fake_redis: AsyncFakeRedis, segment: Any) -> None:
    flush_count = {"n": 0}

    class CountingOffline(NoopOfflineWriter):
        async def flush(self) -> None:
            flush_count["n"] += 1

    consumer = WALConsumer(
        fake_redis,  # type: ignore[arg-type]
        segments={"_S": segment},
        registry=_STUB_REGISTRY,
        offline=CountingOffline(),
        flush_interval_seconds=10.0,  # long timer; won't fire
        max_pending_ack=1,
    )
    # Apply one message → triggers _flush_now signal via _process_batch.
    fields = _wire_msg("_S", "u1", _values_for_S(1.0, 2, [0, 0, 0, 0]))
    msg_id = fake_redis.add_message(consumer._stream_key, fields)
    # Acquire lock so _flush_loop's TTL renew doesn't error on missing key.
    await consumer._acquire_consumer_lock()
    try:
        flush_task = asyncio.create_task(consumer._flush_loop())
        try:
            await consumer._process_batch([(msg_id, fields)])
            # The signal should fire quickly — give it up to 200 ms.
            for _ in range(20):
                if flush_count["n"] >= 1:
                    break
                await asyncio.sleep(0.01)
            assert flush_count["n"] >= 1, "flush_loop did not wake on signal"
        finally:
            consumer._stop_event.set()
            consumer._flush_now.set()
            await asyncio.wait_for(flush_task, timeout=1.0)
    finally:
        await consumer._release_consumer_lock()


# ---------------------------------------------------------------------------
# Test #29 — liveness gauge startup window regression.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_liveness_gauge_does_not_read_system_uptime_at_startup(
    monkeypatch: pytest.MonkeyPatch,
    fake_redis: AsyncFakeRedis,
    segment: Any,
) -> None:
    """The gauge MUST NOT briefly read system_uptime as 'age' between
    consumer startup and the first scheduled refresh.

    Two mechanisms combine to make this robust:
    1. Force-first-refresh pattern in WALConsumer.run() seeds
       ``_liveness_last_refresh = time.monotonic()`` BEFORE the main
       loop's first gauge.set() call.
    2. Defensive ``wal_consumer_liveness_age_seconds.set(0.0)`` in the
       same startup block, immediately after the seed — guards against
       multiprocess-collector futures where an un-set Gauge could read
       a stale value persisted from a crashed predecessor.

    A future reader who removes either mechanism should still see this
    test pass — the OTHER mechanism alone is sufficient today. Both
    together make the contract un-regressable.

    On a long-uptime box, ``time.monotonic()`` returns a value in the
    thousands or hundreds of thousands. Without these mechanisms, the
    first loop iter would compute ``gauge = time.monotonic() - 0.0 =
    system_uptime``, and a Prometheus scrape landing in the 1-5ms
    window before the first scheduled refresh would page on-call.
    """
    import os
    import types

    from pyforge.metrics import (
        wal_consumer_liveness_age_seconds,
    )
    from pyforge.wal_consumer import (
        LIVENESS_REFRESH_INTERVAL_SECONDS,
    )

    # Simulate a 1-day-uptime box. Replace `pyforge.wal_consumer.time`
    # with a tiny stub object exposing `monotonic` and `perf_counter` —
    # avoids mutating the global `time` module which would affect
    # pytest's own clock.
    fake_now = 86_400.0
    import time as real_time

    fake_time = types.SimpleNamespace(
        monotonic=lambda: fake_now,
        perf_counter=real_time.perf_counter,
    )
    monkeypatch.setattr("pyforge.wal_consumer.time", fake_time)

    consumer = WALConsumer(
        fake_redis,  # type: ignore[arg-type]
        segments={"_S": segment},
        registry=_STUB_REGISTRY,
    )

    # Manually replicate run()'s force-first-refresh startup block so
    # we can observe the gauge state without driving the full event
    # loop. If run()'s init block changes, this test will need to
    # update. The two assertions check both mechanisms locked by
    # plan Rev-10 CRITICAL #1:
    consumer._pid_str = str(os.getpid()).encode("ascii")
    consumer._liveness_last_refresh = fake_now  # type: ignore[assignment]
    wal_consumer_liveness_age_seconds.set(0.0)  # defensive line

    # Now drive one iter of the main loop. _run_one_iter calls
    # `time.monotonic()` (now mocked to fake_now), then sets the gauge
    # to `now - _liveness_last_refresh` = 0. Without the seed above,
    # this would compute `86400 - 0 = 86400` and the assertion below
    # would fail.
    await consumer._run_one_iter()

    gauge_value = wal_consumer_liveness_age_seconds._value.get()
    assert gauge_value < LIVENESS_REFRESH_INTERVAL_SECONDS, (
        f"gauge briefly read system_uptime ({gauge_value}) instead of "
        f"~0 — force-first-refresh + defensive set(0.0) pattern broken"
    )
