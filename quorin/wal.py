"""WAL producer.

Quorin writes are async-by-default: the caller invokes
:meth:`WALProducer.write` and gets back a Redis-Stream message ID; the
durability boundary lives at the XADD return. Step 10's consumer is the
single writer to shared memory and to the offline Parquet store.

Locked design (ADR-008):

1. **Sync producer, async consumer.** The producer's hot path is small
   (validate → pack → XADD); ``asyncio`` would force the public API to
   ``await`` everywhere for no measurable benefit. The consumer is async
   because *it* fans out to two destinations and benefits from
   coalescing.
2. **Producer never writes to shared memory.** Invariant #3 (single
   writer per segment) lives or dies here. ``write_sync`` polls a side
   table set by the consumer instead of touching the segment itself.
3. **Wire format: msgpack list of values in name_hash order.** The
   schema offset table is sorted by name_hash; the producer extracts
   validated values in that order and packs them as a positional list.
   ~30-50% smaller wire size than a dict-shaped payload at 200 fields,
   ~2x faster encode/decode.
4. **Per-call allocation discipline.** A reusable ``msgpack.Packer()``
   on the instance, pre-encoded XADD field-name keys at module level,
   and per-producer caching of ``schema -> name bytes`` keep the
   per-call allocation count to: one validated pydantic instance, one
   values list (a list comprehension over getattr), one msgpack output
   bytes, one entity-id encode, one event-time render, and one 4-entry
   XADD field dict. None avoidable; everything else is hoisted.
5. **NaN/Inf accepted.** Float fields use ``allow_inf_nan=True`` in the
   pydantic factory. Locked by invariant #12 (NaN bit-pattern
   preservation through ``_assemble_core``) and standard ML
   missing-feature semantics. See
   :mod:`quorin._internal.pydantic_factory` and ADR-008 §5.

Hot-path budget (ADR-008 §2):

==========================  =============
Cost                        Budget
==========================  =============
Redis XADD RTT              ~500 us
Pydantic validate (200 fld) 100-300 us
msgpack pack (200 fld)      20-80 us
Field encode + dispatch     <10 us
Histogram observe + counter <2 us
**Total CPU per call**      **<400 us**
**write p99 wall-clock**    **<2 ms**
==========================  =============
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any, cast

import msgpack  # type: ignore[import-untyped]

from quorin._internal.pydantic_factory import field_order_for, pydantic_model_for
from quorin.metrics import (
    wal_write_latency_seconds,
    wal_write_sync_consumer_lag_seconds,
    wal_write_total,
)

if TYPE_CHECKING:
    import redis

    from quorin.schema import FeatureSchema


# ---------------------------------------------------------------------------
# Module-level constants — pre-encoded so each ``write`` skips str.encode.
# ---------------------------------------------------------------------------

_F_SCHEMA = b"schema"
_F_ENTITY_ID = b"entity_id"
_F_EVENT_TIME = b"event_time_ns"
_F_BLOB = b"blob"

DEFAULT_STREAM_KEY = b"quorin:wal"
"""Default Redis Stream name. Step 10's consumer XREADGROUPs from the same key."""

DEFAULT_MAXLEN = 1_000_000
"""Approximate stream cap. At 10k writes/s, covers 100 s of consumer downtime."""

PROCESSED_KEY_PREFIX = b"quorin:processed:"
"""Side-table prefix written by the consumer; polled by ``write_sync``."""


# ---------------------------------------------------------------------------
# Exceptions.
# ---------------------------------------------------------------------------


class WriteSyncTimeoutError(TimeoutError):
    """Raised by :meth:`WALProducer.write_sync` when the consumer does not
    set ``quorin:processed:{msg_id}`` before ``timeout_ms`` elapses.

    Attributes:
        msg_id: The Redis-assigned message ID that timed out (bytes).
        timeout_ms: The deadline that fired.
    """

    def __init__(self, msg_id: bytes, timeout_ms: int) -> None:
        self.msg_id = msg_id
        self.timeout_ms = timeout_ms
        super().__init__(
            f"WAL write_sync timed out after {timeout_ms} ms waiting for "
            f"consumer to process {msg_id!r}"
        )


# ---------------------------------------------------------------------------
# Producer.
# ---------------------------------------------------------------------------


class WALProducer:
    """Synchronous WAL producer.

    The producer is a thin shim over ``redis.xadd`` that adds:

    - Pydantic validation of ``values`` against the schema.
    - msgpack-list encoding of values in name_hash order (matches
      :func:`quorin.schema.compile_schema` ordering).
    - ``write_sync(...)`` for read-your-own-writes via a polled side table.

    Construction is cheap and should happen once at process startup; the
    ``redis_client`` is reused across every call. Pre-warmed prometheus
    label children are cached on the instance to avoid the
    ``Histogram.labels(...)``-allocates-on-first-use problem (Step 7
    lesson, see CLAUDE.md §8).

    Thread safety: the producer holds a single :class:`msgpack.Packer`
    instance which is **not** thread-safe. Callers that share a producer
    across threads must either serialize calls or construct one producer
    per thread. The Redis client below is independently thread-safe via
    redis-py's connection pool. Spec deems sync-single-thread the
    expected usage; multi-threaded callers can build their own.
    """

    __slots__ = (
        "_c_async_error",
        "_c_async_ok",
        "_c_sync_error",
        "_c_sync_ok",
        "_c_sync_timeout",
        "_h_async",
        "_h_sync",
        "_h_sync_lag",
        "_maxlen",
        "_packer",
        "_redis",
        "_schema_name_cache",
        "_stream_key",
    )

    def __init__(
        self,
        redis_client: redis.Redis,
        *,
        stream_key: bytes = DEFAULT_STREAM_KEY,
        maxlen: int = DEFAULT_MAXLEN,
    ) -> None:
        self._redis = redis_client
        self._stream_key = stream_key
        self._maxlen = maxlen

        # Reusable Packer is ~10-20% faster than msgpack.packb on 200-field
        # schemas because it skips per-call Packer construction.
        self._packer = msgpack.Packer(use_bin_type=True)

        # Per-(producer, schema) encoded-name cache. First write per schema
        # encodes once; subsequent writes look up. Keyed by class identity.
        self._schema_name_cache: dict[type[FeatureSchema], bytes] = {}

        # Pre-warm prometheus label children. Histogram.labels(...) allocates
        # a tuple key + dict slot on first call; doing it here moves that
        # work out of the hot path.
        self._h_async = wal_write_latency_seconds.labels(mode="async")
        self._h_sync = wal_write_latency_seconds.labels(mode="sync")
        self._c_async_ok = wal_write_total.labels(mode="async", outcome="ok")
        self._c_async_error = wal_write_total.labels(mode="async", outcome="error")
        self._c_sync_ok = wal_write_total.labels(mode="sync", outcome="ok")
        self._c_sync_timeout = wal_write_total.labels(mode="sync", outcome="timeout")
        self._c_sync_error = wal_write_total.labels(mode="sync", outcome="error")
        self._h_sync_lag = wal_write_sync_consumer_lag_seconds

    # ------------------------------------------------------------------
    # Public hot path.
    # ------------------------------------------------------------------

    def write(
        self,
        schema: type[FeatureSchema],
        entity_id: str,
        values: dict[str, Any],
        event_time_ns: int | None = None,
    ) -> bytes:
        """Validate, encode, and XADD a single write.

        Returns the Redis-assigned message ID (bytes). The caller can pass
        this ID to :meth:`write_sync` (or to a separate poll) to wait for
        consumer processing — though the standard pattern is to call
        :meth:`write_sync` directly.

        Raises:
            pydantic.ValidationError: ``values`` does not match the schema.
            redis.exceptions.ConnectionError: Redis is unreachable.
            redis.exceptions.ResponseError: Redis rejected the XADD
                (e.g. the stream key holds a non-stream type).
        """
        t0 = time.perf_counter()
        try:
            msg_id = self._do_write(schema, entity_id, values, event_time_ns)
        except Exception:
            self._c_async_error.inc()
            raise
        self._h_async.observe(time.perf_counter() - t0)
        self._c_async_ok.inc()
        return msg_id

    def write_sync(
        self,
        schema: type[FeatureSchema],
        entity_id: str,
        values: dict[str, Any],
        event_time_ns: int | None = None,
        timeout_ms: int = 100,
    ) -> bytes:
        """``write`` plus a poll for the consumer's ``quorin:processed:{id}`` key.

        Polling cadence: 1 ms initial, exponential backoff capped at 10 ms.
        The 1 ms floor is deliberate — tighter wastes Redis RTT cycles
        because the consumer cycle is 5-50 ms; looser bloats p50 by 10 ms.

        **Signal semantics:** unblocks at **online-store** durability — the
        consumer SETs the processed-key immediately after ``layout.insert``
        succeeds, BEFORE the offline (Parquet) flush. Read-your-own-writes
        from :func:`quorin.serving.assemble` is the contract; offline
        durability is a separate, eventually-consistent guarantee owned by
        the consumer's deferred XACK. See ADR-009 §3 for the design.

        The processed-key TTL is 86400 s (24h). Realistic ``timeout_ms``
        values are 100 ms - 10 s; producers stuck polling for >24h indicate
        a separate bug class (the producer process is wedged) and would see
        the side-table entry expire mid-poll.

        Raises:
            WriteSyncTimeoutError: The consumer did not set the processed key
                before the deadline. The XADD itself succeeded; the message
                will be processed eventually unless the consumer is dead.
            pydantic.ValidationError, redis.exceptions.ConnectionError,
            redis.exceptions.ResponseError: as :meth:`write`.
        """
        t0 = time.perf_counter()
        try:
            msg_id = self._do_write(schema, entity_id, values, event_time_ns)
        except Exception:
            self._c_sync_error.inc()
            raise

        # Separate "post-XADD" timestamp so consumer lag excludes our own
        # validate + pack + XADD cost. The full write_sync latency uses t0.
        t_after_xadd = time.perf_counter()
        key = PROCESSED_KEY_PREFIX + msg_id
        deadline = time.monotonic() + timeout_ms / 1000.0
        backoff = 0.001
        while True:
            if self._redis.exists(key):
                now = time.perf_counter()
                self._h_sync.observe(now - t0)
                self._h_sync_lag.observe(now - t_after_xadd)
                self._c_sync_ok.inc()
                return msg_id
            if time.monotonic() >= deadline:
                self._c_sync_timeout.inc()
                raise WriteSyncTimeoutError(msg_id, timeout_ms)
            time.sleep(backoff)
            backoff = min(backoff * 2.0, 0.010)

    # ------------------------------------------------------------------
    # Private — shared validate + encode + XADD path.
    # ------------------------------------------------------------------

    def _do_write(
        self,
        schema: type[FeatureSchema],
        entity_id: str,
        values: dict[str, Any],
        event_time_ns: int | None,
    ) -> bytes:
        model_cls = pydantic_model_for(schema)
        order = field_order_for(schema)

        validated = model_cls.model_validate(values)
        # getattr is ~30 ns; for 200 fields that's ~6 us total. model_dump()
        # would build an intermediate dict at ~50 us — explicit win to skip.
        values_list = [getattr(validated, name) for name in order]
        blob = self._packer.pack(values_list)

        schema_name_bytes = self._encoded_schema_name(schema)
        entity_id_bytes = entity_id.encode("utf-8")
        if event_time_ns is None:
            event_time_ns = time.time_ns()
        event_time_bytes = str(event_time_ns).encode("ascii")

        # redis-py types xadd() as Awaitable | bytes due to its sync/async
        # client unification; the sync client always returns bytes.
        msg_id = cast(
            bytes,
            self._redis.xadd(
                self._stream_key,
                {
                    _F_SCHEMA: schema_name_bytes,
                    _F_ENTITY_ID: entity_id_bytes,
                    _F_EVENT_TIME: event_time_bytes,
                    _F_BLOB: blob,
                },
                maxlen=self._maxlen,
                approximate=True,
            ),
        )
        return msg_id

    def _encoded_schema_name(self, schema: type[FeatureSchema]) -> bytes:
        """Memoized ``schema.__name__.encode("utf-8")``. ~50 ns lookup after first call."""
        if (cached := self._schema_name_cache.get(schema)) is not None:
            return cached
        encoded = schema.__name__.encode("utf-8")
        self._schema_name_cache[schema] = encoded
        return encoded
