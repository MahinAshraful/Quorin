"""WAL consumer (Step 10).

Reads from the Redis Stream ``pyforge:wal`` produced by Step 9's
:class:`pyforge.wal.WALProducer`, applies each message to the online
store (shared memory via :func:`pyforge.layout.insert`) and to an
injected offline writer (Step 11's Parquet store; defaults to
:class:`NoopOfflineWriter`), and signals downstream readiness via the
``pyforge:processed:{msg_id}`` side table that ``WALProducer.write_sync``
polls.

Two events the consumer signals are deliberately decoupled:

- ``SET pyforge:processed:{msg_id} EX 86400`` fires immediately after
  ``layout.insert`` succeeds. This is the **online-store** durability
  signal — ``write_sync`` unblocks here. ADR-009 §2.
- ``XACK`` fires only after :meth:`OfflineWriter.flush` returns. This is
  the **offline-store** durability signal. Coupling them would lose up
  to ``flush_interval_seconds`` of offline data on crash because
  ``OfflineWriter.append`` is buffered, not durable.

Locked design (ADR-009):

1. Async via :mod:`asyncio` + :class:`redis.asyncio.Redis`. No
   threadpool, no background-thread wrapper. Caller owns the event
   loop. Sidesteps invariant #14.
2. Single coroutine, single consumer per stream. Multi-consumer would
   violate invariant #3 (single writer per segment); horizontal scaling
   is per-instance hash-shard (CLAUDE.md §1).
3. Borrowed segments — constructor accepts pre-opened
   :class:`pyforge.shm.Segment` instances. The user owns segment
   lifecycle (open + close); the consumer borrows.
4. Apply order: decode → ``layout.insert`` → ``offline.append`` →
   pipelined SETs (one per applied msg) → append to ``pending_ack``.
5. Periodic flush task awaits ``offline.flush()`` then XACKs everything
   in ``pending_ack``. Flush trigger = whichever fires first of:
   ``flush_interval_seconds`` timer, ``max_pending_ack`` size, or
   ``stop()``.
6. The size-trigger is a SIGNAL (``asyncio.Event.set()``), not an
   awaited call — the read loop never blocks behind a slow flush.
   Hard back-pressure ceiling at ``2 * max_pending_ack`` prevents
   unbounded pending_ack growth if flush is wedged.
7. PEL drain on startup always re-applies every message (no
   bulk-EXISTS optimization). With deferred XACK, every PEL message
   is "applied-online but not-yet-flushed-offline" by definition, so
   ``offline.append`` MUST re-fire regardless of the side-table state.
8. Distributed consumer-name lock at startup
   (``pyforge:consumer:lock:{group}:{name}``, 60s TTL renewed each
   flush) prevents two processes with the same consumer name from
   silently scrambling each other's PEL.

Operability notes (CLAUDE.md §8):

- ``WALProducer`` takes a sync ``redis.Redis``; ``WALConsumer`` takes a
  ``redis.asyncio.Redis``. Same-process callers need both. A pool-
  unification helper is deferred to Step 12 (public API). Pickling a
  sync client and using it via asyncio is NOT supported by redis-py.
- Redis connection errors propagate out of :meth:`WALConsumer.run`. The
  caller is responsible for wrapping ``run()`` in a retry/backoff loop.
  Pending_ack does not survive across ``run()`` invocations, but PEL
  state on the Redis side does — re-running drains the unfinished work
  via :meth:`_drain_pel`.
- Worst-case shutdown latency ≈ ``block_ms`` (default 500 ms) because
  ``XREADGROUP`` is not interruptible by an :class:`asyncio.Event`.
  Users with restart-time SLAs override ``block_ms=100``.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import time
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Protocol

import msgpack  # type: ignore[import-untyped]
import redis.exceptions

from pyforge import layout
from pyforge._internal.row_pack import pack_row_from_list
from pyforge.logging import get_logger
from pyforge.metrics import (
    wal_consumer_apply_total,
    wal_consumer_batch_size,
    wal_consumer_flush_seconds,
    wal_consumer_liveness_age_seconds,
    wal_consumer_liveness_refresh_failures,
    wal_consumer_pending_ack_size,
    wal_consumer_unknown_schema_total,
)
from pyforge.schema import FeatureSchema, row_size
from pyforge.wal import (
    _F_BLOB,
    _F_ENTITY_ID,
    _F_EVENT_TIME,
    _F_SCHEMA,
    DEFAULT_STREAM_KEY,
    PROCESSED_KEY_PREFIX,
)

if TYPE_CHECKING:
    import redis.asyncio
    from redis.asyncio.client import Pipeline as AsyncPipeline


logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Constants — tunable via constructor; callers should not import these
# directly except in tests.
# ---------------------------------------------------------------------------

DEFAULT_GROUP_NAME = "pyforge_consumers"
DEFAULT_CONSUMER_NAME = "consumer-1"
DEFAULT_BATCH_COUNT = 100
DEFAULT_BLOCK_MS = 500
DEFAULT_FLUSH_INTERVAL_SECONDS = 60.0
DEFAULT_MAX_PENDING_ACK = 10_000
PROCESSED_KEY_TTL_SECONDS = 86_400  # 24h; ADR-009 §2.

CONSUMER_LOCK_PREFIX = b"pyforge:consumer:lock:"
CONSUMER_LOCK_TTL_SECONDS = 60

# Liveness key — Step 13 cross-step touch. Hydration's precondition #2
# reads this key to refuse a clobber while a consumer is actively writing.
# Single source of truth; `pyforge.hydration` imports the constant here.
KEY_WAL_CONSUMER_LIVENESS = b"pyforge:wal_consumer:liveness"
LIVENESS_REFRESH_INTERVAL_SECONDS = 10
LIVENESS_TTL_SECONDS = 30


# ---------------------------------------------------------------------------
# Exceptions and Protocol.
# ---------------------------------------------------------------------------


class UnknownSchemaError(KeyError):
    """Raised internally (not from the public API) when the consumer reads a
    message whose ``schema`` field is not in the segments map. Callers see
    a ``wal_consumer_unknown_schema_total`` counter increment instead — the
    message stays in PEL until the operator redeploys with the schema."""


class ConsumerNameInUseError(RuntimeError):
    """Raised at :meth:`WALConsumer.run` start if another live consumer is
    holding ``pyforge:consumer:lock:{group}:{name}`` in Redis. Two consumers
    with the same name silently scramble each other's PEL — failing fast
    here prevents a corruption class. The lock auto-expires in 60s if the
    holder dies without releasing it.
    """


class OfflineWriter(Protocol):
    """Contract for the offline (Parquet) sink the consumer fans out to.

    :meth:`append` may be called multiple times for the same ``msg_id``
    across crash/restart (at-least-once). The offline store is responsible
    for deduplicating on read by ``(entity_id, event_time_ns)`` or
    ``msg_id``.

    :meth:`flush` is the durability boundary. After ``flush()`` returns
    successfully, all data appended before ``flush()`` began is on disk.
    The consumer XACKs ``pending_ack`` only after a successful flush.

    :meth:`flush` MUST be cancellation-safe. The consumer cancels the
    flush task during shutdown; the implementation must either complete
    its work or unwind cleanly without leaving partial state.
    """

    async def append(
        self,
        schema: type[FeatureSchema],
        entity_id: str,
        event_time_ns: int,
        values_list: list[Any],
        msg_id: bytes,
    ) -> None: ...

    async def flush(self) -> None: ...

    async def close(self) -> None: ...


class NoopOfflineWriter:
    """Default no-op writer used when no real offline sink is supplied.

    Step 11's Parquet writer replaces this in production. Useful for
    online-only deployments and for testing the consumer's online-store
    crash-safety in isolation."""

    async def append(
        self,
        schema: type[FeatureSchema],
        entity_id: str,
        event_time_ns: int,
        values_list: list[Any],
        msg_id: bytes,
    ) -> None:
        del schema, entity_id, event_time_ns, values_list, msg_id

    async def flush(self) -> None:
        return None

    async def close(self) -> None:
        return None


# ---------------------------------------------------------------------------
# WALConsumer.
# ---------------------------------------------------------------------------


class WALConsumer:
    """Async WAL consumer with deferred-XACK durability.

    Constructor:
        redis_client: A ``redis.asyncio.Redis`` instance. The producer's
            sync ``redis.Redis`` is NOT compatible — see ADR-009 §11.
        segments: Pre-opened ``Segment`` per schema, keyed by
            ``schema.__name__``. The consumer borrows these; the user
            owns lifecycle (close via ``SegmentRegistry.close``).
        offline: An :class:`OfflineWriter` implementation. ``None`` ⇒
            :class:`NoopOfflineWriter` (online-only mode; safe for
            development and online-only deployments).
        stream_key: Redis Stream key. Default matches the producer.
        group_name / consumer_name: Consumer-group identity. Two
            consumers with the same ``consumer_name`` cause silent PEL
            scrambling — guarded by the startup lock (D13).
        batch_count: ``XREADGROUP COUNT`` per call.
        block_ms: ``XREADGROUP BLOCK``. Sets the worst-case shutdown
            latency (D15) because ``stop()`` cannot interrupt an
            in-flight XREADGROUP.
        flush_interval_seconds: Periodic flush trigger.
        max_pending_ack: Soft trigger — when ``len(pending_ack)``
            reaches this, ``_flush_now`` is signaled (non-blocking).
            Hard back-pressure ceiling is ``2 * max_pending_ack``;
            beyond that the read loop pauses until the flush drains.
            Tune for recovery SLA: PEL on crash is bounded by this
            value, and catch-up time = ``max_pending_ack * per-msg-cost``.

    Lifecycle: ``await consumer.run()`` blocks until ``stop()`` is
    called from another task or until an exception propagates out. The
    caller wires SIGTERM / SIGINT handling
    (``loop.add_signal_handler``).
    """

    __slots__ = (
        "_batch_count",
        "_block_ms",
        "_c_apply_bad_id",
        "_c_apply_dup",
        "_c_apply_err",
        "_c_apply_ok",
        "_c_liveness_err",
        "_c_liveness_ok",
        "_consumer_name",
        "_flush_interval_seconds",
        "_flush_now",
        "_group_name",
        "_h_flush_err",
        "_h_flush_ok",
        "_liveness_last_refresh",
        "_lock_key",
        "_max_pending_ack",
        "_offline",
        "_pending_ack",
        "_pid_str",
        "_redis",
        "_row_buffers",
        "_schemas",
        "_seen_unknown",
        "_segments",
        "_stop_event",
        "_stream_key",
    )

    def __init__(
        self,
        redis_client: redis.asyncio.Redis,
        segments: Mapping[str, Any],
        offline: OfflineWriter | None = None,
        *,
        stream_key: bytes = DEFAULT_STREAM_KEY,
        group_name: str = DEFAULT_GROUP_NAME,
        consumer_name: str = DEFAULT_CONSUMER_NAME,
        batch_count: int = DEFAULT_BATCH_COUNT,
        block_ms: int = DEFAULT_BLOCK_MS,
        flush_interval_seconds: float = DEFAULT_FLUSH_INTERVAL_SECONDS,
        max_pending_ack: int = DEFAULT_MAX_PENDING_ACK,
    ) -> None:
        # Validate the segments map: name-vs-class consistency catches the
        # one footgun the API otherwise allows.
        for name, seg in segments.items():
            if seg.schema.__name__ != name:
                raise ValueError(
                    f"segments[{name!r}].schema.__name__ == {seg.schema.__name__!r} "
                    f"— key must equal segment.schema.__name__"
                )

        self._redis = redis_client
        # Wire keys are bytes; key the per-schema dicts on bytes for hot-path
        # zero-allocation lookup.
        self._segments: dict[bytes, Any] = {
            name.encode("utf-8"): seg for name, seg in segments.items()
        }
        self._schemas: dict[bytes, type[FeatureSchema]] = {
            name.encode("utf-8"): seg.schema for name, seg in segments.items()
        }
        self._row_buffers: dict[bytes, bytearray] = {
            name.encode("utf-8"): bytearray(row_size(seg.schema)) for name, seg in segments.items()
        }
        self._offline = offline if offline is not None else NoopOfflineWriter()

        self._stream_key = stream_key
        self._group_name = group_name
        self._consumer_name = consumer_name
        self._batch_count = batch_count
        self._block_ms = block_ms
        self._flush_interval_seconds = flush_interval_seconds
        self._max_pending_ack = max_pending_ack

        self._stop_event = asyncio.Event()
        self._flush_now = asyncio.Event()
        self._pending_ack: list[bytes] = []
        # Rate-limit unknown-schema log spam — log once per name, count always.
        self._seen_unknown: set[bytes] = set()

        self._lock_key = (
            CONSUMER_LOCK_PREFIX
            + self._group_name.encode("ascii")
            + b":"
            + self._consumer_name.encode("ascii")
        )

        # Pre-warm prometheus label children (Step 7 lesson — `Histogram.labels(...)`
        # allocates a tuple key + dict slot on first call).
        self._c_apply_ok = wal_consumer_apply_total.labels(outcome="ok")
        self._c_apply_dup = wal_consumer_apply_total.labels(outcome="duplicate")
        self._c_apply_err = wal_consumer_apply_total.labels(outcome="error")
        self._c_apply_bad_id = wal_consumer_apply_total.labels(outcome="bad_entity_id")
        self._h_flush_ok = wal_consumer_flush_seconds.labels(outcome="ok")
        self._h_flush_err = wal_consumer_flush_seconds.labels(outcome="error")
        self._c_liveness_ok = wal_consumer_liveness_refresh_failures.labels(outcome="ok")
        self._c_liveness_err = wal_consumer_liveness_refresh_failures.labels(outcome="error")

        # Liveness state — `_pid_str` and `_liveness_last_refresh` are
        # re-seeded at run() startup (force-first-refresh pattern) so the
        # gauge never reads system_uptime as "age". See plan §"Boundary
        # risk #1" + run() docstring.
        self._pid_str = b""  # set in run() before any liveness SET
        self._liveness_last_refresh = 0.0

    # ------------------------------------------------------------------
    # Public API.
    # ------------------------------------------------------------------

    async def run(self) -> None:
        """Drain PEL once, then loop XREADGROUP→apply→signal until stop().

        Acquires the consumer-name lock first; raises
        :class:`ConsumerNameInUseError` if held by another process.
        Releases the lock + does a final flush+XACK on exit (whether
        clean or via exception). Issues an initial liveness SET BEFORE
        entering the main loop (force-first-refresh) so the per-iter
        gauge never reads ``time.monotonic()`` as "age" — without this,
        the first iter computes ``gauge = uptime - 0.0`` which on a
        long-uptime box looks like "86400s since last refresh" and
        false-pages on-call.

        Redis connection errors propagate out — caller wraps in retry.
        """
        await self._acquire_consumer_lock()
        try:
            await self._ensure_group()
            await self._drain_pel()

            # Force-first-refresh + monotonic seeding (plan CRITICAL #1).
            # Cache pid_str once; saves str(int) per refresh.
            self._pid_str = str(os.getpid()).encode("ascii")
            self._liveness_last_refresh = time.monotonic()
            # Defensive set-to-zero (plan Rev-10 CRITICAL #1): guards
            # against a multiprocess-collector future where an un-set
            # Gauge could persist a stale value from the previous
            # process incarnation. No-op today (single-process default
            # is 0.0); one line eliminates the race forever.
            wal_consumer_liveness_age_seconds.set(0.0)
            try:
                await self._redis.set(
                    KEY_WAL_CONSUMER_LIVENESS,
                    self._pid_str,
                    ex=LIVENESS_TTL_SECONDS,
                )
                self._c_liveness_ok.inc()
            except Exception:
                self._c_liveness_err.inc()
                # Initial refresh failure is non-fatal; main loop retries
                # every 10s. Hydration's precondition #2 stays "no
                # liveness key" until the first SET succeeds — operator
                # procedure: don't hydrate while a consumer is starting
                # up. Step 14 watchdog will reconcile post-Step-14.

            flush_task = asyncio.create_task(self._flush_loop(), name="pyforge-wal-flush")
            try:
                while not self._stop_event.is_set():
                    await self._run_one_iter()
            finally:
                # Cancel + await the flush task BEFORE the final flush so we
                # never have two flushes in flight at once. ADR-009 §11.
                flush_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await flush_task
                await self._final_flush_and_ack()
                await self._offline.close()
                # Best-effort liveness cleanup so a fast restart doesn't
                # see the stale key (the 30s TTL would expire it eventually).
                with contextlib.suppress(Exception):
                    await self._redis.delete(KEY_WAL_CONSUMER_LIVENESS)
        finally:
            await self._release_consumer_lock()

    async def _run_one_iter(self) -> None:
        """One iteration of the main run loop: XREADGROUP → apply → liveness.

        Extracted from :meth:`run` for testability — Step 13's regression
        test #29 drives a single iter to assert the liveness gauge does
        NOT briefly read ``time.monotonic()`` as "age" between consumer
        startup and the first scheduled refresh. Future chaos /
        integration tests will reuse this affordance.

        Liveness placement: post-XREADGROUP, pre-apply, gated on elapsed
        time. Plan §"sub-step 8" locks this to "post-XREADGROUP, pre-apply,
        unconditional on message count" so quiet streams with long
        ``block_ms`` still see the 10s refresh cadence.
        """
        msgs = await self._read_new_batch()

        # Update age gauge EVERY loop iter so it climbs naturally during
        # Redis outages (when SETs fail and `_liveness_last_refresh` does
        # not advance). Per-loop overhead is ~3-7µs/call (lock + float
        # store) x 50-200 iters/sec = 0.015-0.14% CPU — negligible for
        # the observability gain.
        now = time.monotonic()
        wal_consumer_liveness_age_seconds.set(now - self._liveness_last_refresh)

        if now - self._liveness_last_refresh >= LIVENESS_REFRESH_INTERVAL_SECONDS:
            try:
                await self._redis.set(
                    KEY_WAL_CONSUMER_LIVENESS,
                    self._pid_str,
                    ex=LIVENESS_TTL_SECONDS,
                )
                self._liveness_last_refresh = now
                self._c_liveness_ok.inc()
            except Exception:
                self._c_liveness_err.inc()
                # Liveness drift during a Redis outage is a known
                # correctness compromise — accepted because the
                # alternative is consumer-suicide-on-Redis-blip. After
                # 3 missed refreshes (~30s) the TTL expires; hydration's
                # precondition #2 then passes while a stale consumer
                # still owns the segment write seat. Operators alert
                # on `pyforge_wal_consumer_liveness_refresh_failures_total{outcome="error"}`
                # AND on `pyforge_wal_consumer_liveness_age_seconds > 30`.

        if msgs:
            await self._process_batch(msgs)

    async def stop(self) -> None:
        """Signal the run loop to exit at the next batch boundary.

        Worst-case latency ≈ ``block_ms`` because XREADGROUP is not
        interruptible by an asyncio Event. ADR-009 §12.
        """
        self._stop_event.set()

    # ------------------------------------------------------------------
    # Lifecycle helpers.
    # ------------------------------------------------------------------

    async def _acquire_consumer_lock(self) -> None:
        """SET pyforge:consumer:lock:{group}:{name} <pid> EX 60 NX."""
        ok = await self._redis.set(
            self._lock_key,
            str(os.getpid()).encode("ascii"),
            ex=CONSUMER_LOCK_TTL_SECONDS,
            nx=True,
        )
        if not ok:
            holder = await self._redis.get(self._lock_key)
            raise ConsumerNameInUseError(
                f"consumer name {self._consumer_name!r} held by pid {holder!r}"
            )

    async def _release_consumer_lock(self) -> None:
        """Best-effort. Lock auto-expires in 60s if we crash before this."""
        with contextlib.suppress(Exception):
            await self._redis.delete(self._lock_key)

    async def _ensure_group(self) -> None:
        """Create the consumer group if missing. Idempotent (BUSYGROUP is fine).

        Uses ``id="0"`` so a fresh group drains the entire current stream
        (the WAL semantics — every produced message gets delivered, even
        if the producer wrote before the consumer started). The standard
        Redis-tutorial ``id="$"`` semantics ("only new messages") would
        silently drop pre-existing entries, which is an operability
        footgun for a write-ahead log. ADR-009 records the choice.
        Once the group exists, the cursor lives in Redis and survives
        consumer restarts; subsequent XREADGROUP ``>`` calls deliver
        only what's strictly newer than the cursor.
        """
        try:
            await self._redis.xgroup_create(
                name=self._stream_key,
                groupname=self._group_name,
                id="0",
                mkstream=True,
            )
        except redis.exceptions.ResponseError as e:
            if "BUSYGROUP" not in str(e):
                raise

    # ------------------------------------------------------------------
    # Hot path — PEL drain, read, apply, flush.
    # ------------------------------------------------------------------

    async def _drain_pel(self) -> None:
        """Read this consumer's PEL once, applying each message.

        Always re-applies every PEL message — see ADR-009 §5. With deferred
        XACK, every PEL entry is "applied-online but not-yet-flushed-offline,"
        so :meth:`OfflineWriter.append` MUST re-fire regardless of the
        side-table state. Bulk-EXISTS would optimize zero work.

        Pagination: XREADGROUP id=``0`` returns PEL entries with id > 0.
        Since deferred XACK keeps the PEL populated until the next flush,
        a naive loop on id=``0`` would never terminate (the same messages
        come back every iteration). We advance ``last_id`` past each batch
        so subsequent reads return only PEL entries with id > last_id.
        Loop exits when a read returns no new entries.
        """
        last_id = "0"
        while not self._stop_event.is_set():
            res = await self._redis.xreadgroup(
                groupname=self._group_name,
                consumername=self._consumer_name,
                streams={self._stream_key: last_id},
                count=self._batch_count,
            )
            if not res:
                break
            msgs = res[0][1]
            if not msgs:
                break
            await self._process_batch(msgs)
            # Advance past the last-seen ID — bytes "(<id>" semantics not
            # needed; XREADGROUP id is exclusive lower bound when reading PEL.
            last_id = msgs[-1][0].decode("ascii")

    async def _read_new_batch(
        self,
    ) -> list[tuple[bytes, dict[bytes, bytes]]]:
        res = await self._redis.xreadgroup(
            groupname=self._group_name,
            consumername=self._consumer_name,
            streams={self._stream_key: ">"},
            count=self._batch_count,
            block=self._block_ms,
        )
        if not res:
            return []
        msgs: list[tuple[bytes, dict[bytes, bytes]]] = res[0][1]
        return msgs

    async def _process_batch(self, msgs: list[tuple[bytes, dict[bytes, bytes]]]) -> None:
        """Apply each msg in order; pipeline the SETs at batch end.

        Hard back-pressure: if the flush task is too far behind, pause
        the read loop so pending_ack can't grow without bound.
        """
        wal_consumer_batch_size.observe(len(msgs))

        # Hard ceiling — ADR-009 §6 (review #C).
        while len(self._pending_ack) >= 2 * self._max_pending_ack and not self._stop_event.is_set():
            await asyncio.sleep(0.01)

        applied: list[bytes] = []
        pipe: AsyncPipeline = self._redis.pipeline(transaction=False)
        for msg_id, fields in msgs:
            ok = await self._apply(msg_id, fields)
            if ok:
                pipe.set(
                    PROCESSED_KEY_PREFIX + msg_id,
                    b"1",
                    ex=PROCESSED_KEY_TTL_SECONDS,
                )
                applied.append(msg_id)

        if applied:
            # SETs only — XACK is deferred to the flush task. Catches the
            # bug where someone adds an XACK to the pipeline by accident.
            assert len(pipe.command_stack) == len(applied), (
                f"pipeline command count {len(pipe.command_stack)} != applied count {len(applied)}"
            )
            await pipe.execute()
            self._pending_ack.extend(applied)
            wal_consumer_pending_ack_size.set(len(self._pending_ack))
            # Soft trigger — SIGNAL the flush task; do NOT await flush here.
            # ADR-009 §6: awaiting would cap throughput at ~950 msgs/sec.
            if len(self._pending_ack) >= self._max_pending_ack:
                self._flush_now.set()

    async def _apply(self, msg_id: bytes, fields: dict[bytes, bytes]) -> bool:
        """Decode → row_pack → layout.insert → offline.append.

        Returns ``True`` if SET + pending_ack should follow, ``False`` if
        the message must stay in PEL (unknown schema, bad entity_id, or
        any apply error).
        """
        schema_name = fields.get(_F_SCHEMA)
        if schema_name is None:
            # Pathologically malformed message — no schema field at all.
            logger.error("missing_schema_field", msg_id=msg_id)
            self._c_apply_err.inc()
            return False

        schema = self._schemas.get(schema_name)
        if schema is None:
            if schema_name not in self._seen_unknown:
                self._seen_unknown.add(schema_name)
                logger.warning(
                    "unknown_schema",
                    schema_name=schema_name,
                    msg_id=msg_id,
                )
            wal_consumer_unknown_schema_total.labels(
                schema_name=schema_name.decode("utf-8", "replace")
            ).inc()
            return False

        # Guard the entity_id decode separately from the rest of the apply
        # path so a corrupted UTF-8 doesn't poison the error counter with
        # apply-level errors.
        try:
            entity_id = fields[_F_ENTITY_ID].decode("utf-8")
        except UnicodeDecodeError:
            logger.error(
                "bad_entity_id",
                msg_id=msg_id,
                raw=fields.get(_F_ENTITY_ID),
            )
            self._c_apply_bad_id.inc()
            return False

        try:
            values_list = msgpack.unpackb(fields[_F_BLOB], use_list=True, raw=False)
            row_buffer = self._row_buffers[schema_name]
            pack_row_from_list(schema, values_list, row_buffer)
            # bytearray → memoryview is zero-copy; layout.insert wants
            # bytes | memoryview per its signature.
            new_insert = layout.insert(
                self._segments[schema_name], entity_id, memoryview(row_buffer)
            )
            event_time_ns = int(fields[_F_EVENT_TIME])
            await self._offline.append(schema, entity_id, event_time_ns, values_list, msg_id)
        except Exception:
            logger.exception("apply_error", msg_id=msg_id)
            self._c_apply_err.inc()
            return False

        (self._c_apply_ok if new_insert else self._c_apply_dup).inc()
        return True

    # ------------------------------------------------------------------
    # Flush task — periodic + signal-driven.
    # ------------------------------------------------------------------

    async def _flush_loop(self) -> None:
        """Wake on stop_event, flush_now signal, or the periodic timer.

        Single in-flight flush invariant holds because only this coroutine
        calls :meth:`_flush_and_ack` in steady state.
        """
        while not self._stop_event.is_set():
            stop_task = asyncio.create_task(self._stop_event.wait())
            flush_signal_task = asyncio.create_task(self._flush_now.wait())
            try:
                await asyncio.wait(
                    {stop_task, flush_signal_task},
                    timeout=self._flush_interval_seconds,
                    return_when=asyncio.FIRST_COMPLETED,
                )
            finally:
                for t in (stop_task, flush_signal_task):
                    if not t.done():
                        t.cancel()
                        with contextlib.suppress(asyncio.CancelledError):
                            await t
            if self._stop_event.is_set():
                # Final flush is handled in run()'s finally block.
                return
            self._flush_now.clear()
            # Renew the consumer-name lock TTL while we're awake.
            with contextlib.suppress(Exception):
                await self._redis.expire(self._lock_key, CONSUMER_LOCK_TTL_SECONDS)
            if self._pending_ack:
                await self._flush_and_ack()

    async def _flush_and_ack(self) -> None:
        """Snapshot pending_ack, flush, XACK the snapshot, trim pending_ack.

        Failure handling: if ``offline.flush()`` raises, leave
        ``pending_ack`` intact so the next cycle retries. The XACK is
        skipped — messages stay in PEL.
        """
        snapshot = list(self._pending_ack)
        if not snapshot:
            return

        t0 = time.perf_counter()
        try:
            await self._offline.flush()
        except Exception:
            self._h_flush_err.observe(time.perf_counter() - t0)
            logger.exception("flush_error", count=len(snapshot))
            return
        self._h_flush_ok.observe(time.perf_counter() - t0)

        await self._redis.xack(self._stream_key, self._group_name, *snapshot)
        # Trim the snapshot prefix in place; new entries appended after we
        # took the snapshot stay in pending_ack for the next cycle.
        del self._pending_ack[: len(snapshot)]
        wal_consumer_pending_ack_size.set(len(self._pending_ack))

    async def _final_flush_and_ack(self) -> None:
        """One last flush+XACK on shutdown. Logged-and-swallowed on error
        because we're already in run()'s finally block."""
        if not self._pending_ack:
            return
        try:
            await self._flush_and_ack()
        except Exception:
            logger.exception("final_flush_failed")
