"""Hydration (Step 13).

Rebuilds the online store (shared memory) from the offline store (Parquet)
so a process restart or Redis-wipe doesn't take the serving fleet down with
it. Used after a deploy, after a control-plane outage, or when an operator
wants to roll forward to a known-good snapshot.

Public API: :func:`hydrate` (sync) returns a :class:`HydrationResult`. The
helper :func:`_force_drop_orphan` is module-private; it deliberately bypasses
``SegmentRegistry.close → cleanup_queue → watchdog`` because the orchestrator
owns the segment freshly, has not yet released it to other readers, and is
itself alive (so the cleanup_queue's "creator died before close" semantics
do not apply). See the helper's docstring for the four-point justification.

Cancel-safety: every per-step catch in :func:`hydrate` is split into

* ``except (KeyboardInterrupt, SystemExit, asyncio.CancelledError)`` —
  quiet propagate; orphan cleanup runs if a segment exists.
* ``except Exception`` — ``logger.exception(...)`` for forensics; orphan
  cleanup runs if a segment exists.

Operators wrapping ``hydrate`` in ``asyncio.to_thread(...)`` and cancelling
the task get the orphan-safe path without log spam. CLAUDE.md §11 / plan
§"Boundary risk #4" / tests #23d, #23e document the contract.

Concurrent ``hydrate()`` calls on the same schema are NOT supported. Two
operators racing both pass the preconditions, both call ``registry.create()``
with the same deterministic segment name, the second errors with
``FileExistsError``, and (without the locked try/except structure below)
the second's ``_force_drop_orphan`` would unlink the first's segment
mid-write — silent data corruption. The orchestrator guards against this
by only running orphan cleanup when ``registry.create()`` succeeded. See
plan §"Boundary risk #5".
"""

from __future__ import annotations

import asyncio
import contextlib
import math
import os
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

from pyforge._internal import posix_shm
from pyforge._internal.insert_kernel import insert_many, prewarm
from pyforge.logging import get_logger
from pyforge.metrics import hydration_entity_count, hydration_seconds
from pyforge.shm import (
    KEY_SEGMENT_TO_SCHEMA,
    _key_current,
    _key_pid_segments,
    _key_refcount,
)
from pyforge.wal_consumer import KEY_WAL_CONSUMER_LIVENESS

if TYPE_CHECKING:
    import redis

    from pyforge.offline import ParquetDatasetStore
    from pyforge.schema import FeatureSchema
    from pyforge.shm import Segment, SegmentRegistry


logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Public API.
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class HydrationResult:
    """Outcome of a successful :func:`hydrate` call.

    ``frozen=True, slots=True`` matches the codebase convention for
    value types (mirrors ``FeatureField``, ``SegmentLayout``).
    """

    segment_name: str
    entity_count: int
    elapsed_seconds: float


class HydrationError(Exception):
    """Base hydration error."""


class HydrationConflictError(HydrationError):
    """A precondition forbade hydration: a current segment already exists,
    or a WAL consumer is still active. Raised BEFORE the timed section,
    so it does NOT tick :data:`pyforge_hydration_seconds`.
    """


class EmptyDatasetError(HydrationError):
    """No entities have rows within ``lookback_days``. Distinguished from a
    plain "empty dataset" success because hydrating to a zero-row segment
    would silently delete every cached feature on a transient backfill
    issue.
    """


def hydrate(  # noqa: PLR0915 — orchestration logic is intentionally linear for readability
    schema: type[FeatureSchema],
    store: ParquetDatasetStore,
    registry: SegmentRegistry,
    *,
    redis_client: redis.Redis,
    capacity_factor: float = 4.0,
    as_of_time_ns: int | None = None,
    lookback_days: int = 30,
) -> HydrationResult:
    """Rebuild the online store for ``schema`` from the offline store.

    Sync method. Async callers MUST wrap in ``asyncio.to_thread(hydrate, ...)``
    — hydration runs for seconds and would block the event loop otherwise.
    Mirrors the precedent set by ``ParquetDatasetStore.read_point_in_time``
    in Step 12.

    Concurrent ``hydrate()`` calls on the same schema are NOT supported.
    Operators must serialize. See module docstring for the corruption
    failure mode this guards against.

    Parameters
    ----------
    schema : type[FeatureSchema]
        Schema whose dataset partition is read and whose new segment is
        created. ``schema.__name__`` is used in Redis keys + log fields.
        Schema names should not contain customer identifiers — see no-PII
        note below.
    store : ParquetDatasetStore
        Offline store whose ``latest_features(...)`` provides the row set.
    registry : SegmentRegistry
        Shared-memory registry; ``registry.create(...)`` is the segment
        allocator.
    redis_client : redis.Redis
        Sync Redis client. Used for precondition GETs and orphan-cleanup
        DELs/SREMs. Same client should be used by ``registry`` to avoid
        connection-pool divergence.
    capacity_factor : float, default 4.0
        Multiplier on entity count for the new segment's capacity. Must be
        ``>= 2.0`` and finite (NaN/Inf rejected). Lower values risk
        immediate fill; higher values trade RAM for headroom on additional
        writes.
    as_of_time_ns : int | None, default None
        Snapshot timestamp; ``None`` ⇒ ``time.time_ns()`` inside
        ``store.latest_features``. Useful for deterministic tests +
        replaying a known state.
    lookback_days : int, default 30
        Per-query staleness ceiling AND partition-pruning window in
        ``latest_features``.

    Returns
    -------
    HydrationResult

    Raises
    ------
    ValueError
        ``capacity_factor`` not finite OR ``< 2.0``.
    HydrationConflictError
        A current segment already exists for ``schema`` (operator must
        drop it via ``redis-cli DEL`` or Step 15's upgrade path), OR a
        WAL consumer is still alive (stop it via SIGTERM or wait up to
        30s for the liveness key to expire).
    EmptyDatasetError
        Zero entities have rows within ``lookback_days``. Hydrating to
        an empty segment would silently delete every cached feature — we
        refuse and ask the operator to confirm.
    Exception
        Re-raised from ``store.latest_features`` / ``registry.create`` /
        ``insert_many`` after best-effort orphan cleanup.

    Metrics
    -------
    ``pyforge_hydration_seconds{outcome=ok|err}``
        Wall-clock time for runs that reached the timed section.
        Precondition rejections (HydrationConflictError) raised BEFORE
        ``t0 = perf_counter()`` are intentionally NOT recorded here —
        they represent rejected calls, not failed work. Operators
        alerting on hydrate failure rate must combine
        ``rate(pyforge_hydration_seconds_count{outcome="err"}[5m])``
        with the structlog WARNING signal ``hydrate.precondition_*``,
        or accept that precondition rejections are a separate failure
        mode (operator procedure violation, not pipeline failure).
    ``pyforge_hydration_entity_count``
        Entities loaded per successful call.

    Logging (no-PII)
    ----------------
    Logged values deliberately exclude ``entity_id`` values and row data.
    The single hydration-wide ``as_of_time_ns`` snapshot IS logged at
    start (it's a timestamp parameter, not entity-data, and operators
    need it for reproducible failure investigation). Per-entity
    timestamps from the offline store are never logged. ``schema.__name__``
    IS logged; ensure schema names do not contain customer identifiers
    at the schema-design layer (a per-customer
    ``Schema_acme_corp_user_features`` naming pattern would leak via
    support-ticket log greps).

    Race window note
    ----------------
    Precondition #1 (no current segment) is the PRIMARY race guard
    against concurrent writers. Precondition #2 (no WAL consumer
    liveness key) is defense-in-depth for the "current was manually
    dropped while consumer is still running" scenario. Pre-Step-14
    there is a real ~30s race window between the consumer's TTL
    expiring during a Redis outage and its next XREADGROUP returning;
    Step 14 watchdog closes it.
    """
    # NaN/Inf guard: NaN < 2.0 is False, so a bare `< 2.0` check would
    # pass NaN through, then `int(len(latest) * NaN)` would ValueError
    # deep inside the orchestrator with no mention of capacity_factor.
    if not math.isfinite(capacity_factor) or capacity_factor < 2.0:
        raise ValueError(f"capacity_factor must be a finite float >= 2.0, got {capacity_factor!r}")

    logger.info(
        "hydrate.start",
        schema=schema.__name__,
        capacity_factor=capacity_factor,
        lookback_days=lookback_days,
        as_of_time_ns=as_of_time_ns,
    )

    # Precondition 1 — PRIMARY race guard against concurrent writers.
    # `cast` mirrors the pattern in pyforge.shm.SegmentRegistry.open_current —
    # redis-py's stub union with Awaitable forces narrowing for sync usage.
    current = cast("bytes | None", redis_client.get(_key_current(schema)))
    if current is not None:
        current_str = current.decode() if isinstance(current, bytes) else str(current)
        logger.warning(
            "hydrate.precondition_current_segment_exists",
            schema=schema.__name__,
            current_segment=current_str,
        )
        raise HydrationConflictError(
            f"current segment for schema {schema.__name__!r} already "
            f"exists ({current_str!r}). Drop it first via "
            f"`redis-cli DEL pyforge:schema:{schema.__name__}:current` "
            f"+ Step 14 watchdog cleanup, or use Step 15's upgrade path."
        )

    # Precondition 2 — defense-in-depth.
    liveness = cast("bytes | None", redis_client.get(KEY_WAL_CONSUMER_LIVENESS))
    if liveness is not None:
        liveness_str = liveness.decode() if isinstance(liveness, bytes) else str(liveness)
        logger.warning(
            "hydrate.precondition_consumer_alive",
            schema=schema.__name__,
            consumer_pid=liveness_str,
        )
        raise HydrationConflictError(
            f"WAL consumer (pid={liveness_str}) is active. Stop it via "
            f"SIGTERM, or wait up to 30s for the liveness key to expire "
            f"if the consumer is already gone."
        )

    # Pre-warm the Numba kernel BEFORE the timed section so the
    # ``elapsed_seconds`` field reflects steady-state cost, not the
    # ~100-500 ms first-call LLVM compile.
    prewarm()

    t0 = time.perf_counter()
    try:
        latest = store.latest_features(
            schema,
            lookback_days=lookback_days,
            as_of_time_ns=as_of_time_ns,
        )

        if len(latest) == 0:
            logger.warning(
                "hydrate.empty_dataset",
                schema=schema.__name__,
                lookback_days=lookback_days,
            )
            raise EmptyDatasetError(
                f"no entities for schema {schema.__name__!r} have rows "
                f"within lookback_days={lookback_days}; nothing to hydrate"
            )

        capacity = max(int(len(latest) * capacity_factor), 16)

        # `registry.create()` may fail with FileExistsError if a
        # concurrent hydrate raced us. We MUST NOT call
        # _force_drop_orphan in that case — there is no segment we own
        # to clean up, and the other hydrate's segment is mid-write.
        # The differentiation between (KeyboardInterrupt, SystemExit,
        # asyncio.CancelledError) and Exception keeps Ctrl-C / async
        # cancel from emitting noisy tracebacks.
        try:
            segment = registry.create(schema, capacity=capacity)
        except (KeyboardInterrupt, SystemExit, asyncio.CancelledError):
            raise
        except Exception:
            logger.exception(
                "hydrate.registry_create_failed",
                schema=schema.__name__,
            )
            raise

        # From here on, we own ``segment``. Any failure must run orphan
        # cleanup before re-raising. Same Exception/system-exit split
        # to keep async-cancel paths quiet but still safe.
        try:
            inserted = insert_many(segment, latest)
        except (KeyboardInterrupt, SystemExit, asyncio.CancelledError):
            # Without this branch, an asyncio.to_thread(hydrate) caller
            # cancelling its task would leave an orphan segment +
            # populated Redis keys. Clean up FIRST, then propagate.
            _force_drop_orphan(redis_client, schema, segment)
            raise
        except Exception:
            logger.exception(
                "hydrate.force_drop_orphan",
                schema=schema.__name__,
                segment_name=segment.name,
            )
            _force_drop_orphan(redis_client, schema, segment)
            raise

        elapsed = time.perf_counter() - t0
    except BaseException:
        elapsed = time.perf_counter() - t0
        # Observe err-outcome with its own suppress so an instrument
        # failure can't mask hydration's exception. Note: precondition
        # rejections raised BEFORE t0 don't reach here — they're not
        # counted in `pyforge_hydration_seconds` at all. See docstring
        # "Metrics" section.
        with contextlib.suppress(Exception):
            hydration_seconds.labels(outcome="err").observe(elapsed)
        raise
    else:
        # try/else: success path runs ONLY when the try block completed
        # without exception. Cleaner separation of success-observation
        # from failure-cleanup; ``inserted`` and ``segment`` bindings
        # are guaranteed bound here.
        with contextlib.suppress(Exception):
            hydration_seconds.labels(outcome="ok").observe(elapsed)
            hydration_entity_count.observe(inserted)
        logger.info(
            "hydrate.success",
            schema=schema.__name__,
            segment_name=segment.name,
            entity_count=inserted,
            elapsed_seconds=elapsed,
        )
        return HydrationResult(
            segment_name=segment.name,
            entity_count=inserted,
            elapsed_seconds=elapsed,
        )


# ---------------------------------------------------------------------------
# Internal helpers.
# ---------------------------------------------------------------------------


def _force_drop_orphan(
    redis_client: redis.Redis,
    schema: type[FeatureSchema],
    segment: Segment,
) -> None:
    """Atomically drop a freshly-created segment that we own.

    DELIBERATELY BYPASSES the canonical lifecycle (``registry.close →
    pyforge:cleanup_queue → Step 14 watchdog drain``). Justification:

    1. We are the segment's only holder — it was just created in this
       process, no other reader has opened it yet.
    2. The cleanup_queue is for crashed-creator recovery (the creator
       died before close); we are alive and conscious, so we don't
       need it.
    3. We are the creator (``registry.create()`` ran in this process), so
       CLAUDE.md invariant #6 ("only the creator unlinks") permits the
       direct ``posix_shm.unlink`` call below.
    4. **DO NOT** "fix" this to call ``registry.close()``. ``registry.close()``
       would push our own segment name into ``pyforge:cleanup_queue``
       while we still hold the unlink permission — the watchdog would
       later try to unlink an already-unlinked segment, and the
       refcount would have been double-decremented in the interim.

    Best-effort: each block wrapped in its own try/except.
    If Redis is unreachable, the disk side is cleaned anyway; Step 14
    watchdog reconciles. Idempotent — safe to call twice.

    Cancellation edge case: the per-block ``except Exception`` does NOT
    catch ``BaseException``-derived ``asyncio.CancelledError``. If a
    SECOND cancel arrives during the ~1ms cleanup window (e.g.,
    aggressive ``task.cancel()`` retry), it propagates up through this
    helper, skipping remaining cleanup blocks. The segment leaks until
    Step 14 watchdog catches it. Mitigation would require ``except
    BaseException`` everywhere, suppressing legitimate cancels in the
    process. Accepted as a v1 limitation — a future reader who
    "fixes" this back to ``BaseException`` would suppress the very
    cancel signaling we want to honor.

    Logs:
        * ``hydrate.force_drop_orphan_complete`` at INFO if all three
          blocks succeeded — operators querying "did we eventually
          clean up?" should grep INFO, NOT just react to WARNINGs.
        * Per-block WARNINGs (``redis_failed``, ``unlink_failed``,
          ``close_failed``) if any block raised. Operator dashboards
          filtering on level >= WARNING will see failures but not the
          recovery confirmation; both signals are intentional.
        * Log-volume note: in a Redis outage, every hydrate's orphan
          cleanup fails the Redis pipeline block, producing one
          WARNING per failed hydrate. For admin-triggered hydrates this
          is fine. A script accidentally retrying hydrate in a loop
          would generate log spam — operators should rate-limit the
          WARNING in dashboard alerts.
    """
    pid = os.getpid()
    redis_ok = shm_unlink_ok = shm_close_ok = False
    try:
        with redis_client.pipeline(transaction=True) as p:
            p.delete(_key_current(schema))
            p.delete(_key_refcount(segment.name))
            p.srem(_key_pid_segments(pid), segment.name)
            # Step 14: also clear the sidetable so the watchdog doesn't
            # later see a stale segment_to_schema entry pointing at an
            # already-unlinked segment.
            p.hdel(KEY_SEGMENT_TO_SCHEMA, segment.name)
            p.execute()
        redis_ok = True
    except Exception:
        logger.warning(
            "hydrate.force_drop_orphan.redis_failed",
            schema=schema.__name__,
            segment_name=segment.name,
        )
    try:
        posix_shm.unlink(segment.name)
        shm_unlink_ok = True
    except Exception:
        logger.warning(
            "hydrate.force_drop_orphan.unlink_failed",
            schema=schema.__name__,
            segment_name=segment.name,
        )
    try:
        posix_shm.close(segment.handle)
        shm_close_ok = True
    except Exception:
        logger.warning(
            "hydrate.force_drop_orphan.close_failed",
            schema=schema.__name__,
            segment_name=segment.name,
        )
    if redis_ok and shm_unlink_ok and shm_close_ok:
        logger.info(
            "hydrate.force_drop_orphan_complete",
            schema=schema.__name__,
            segment_name=segment.name,
        )
