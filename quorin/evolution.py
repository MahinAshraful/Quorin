"""Schema evolution (Step 15): atomic segment swap with live readers.

Public API: :func:`upgrade_schema` (sync) returns an :class:`UpgradeResult`.
The helper :func:`can_upgrade` is a pure predicate operators / tests can call
without side effects.

The orchestrator's contract:

1. Validate compatibility via :func:`can_upgrade` (no field removals, dtype
   widening only, monotonic version, shape unchanged for existing fields).
2. Acquire ``quorin:upgrade:lock:{safe_name}`` (NX EX 600s).
3. Verify the WAL consumer has caught up (XPENDING == 0) AND no live
   consumer is running (``quorin:wal_consumer:liveness`` absent). XLEN
   is intentionally NOT checked: XACK doesn't decrement XLEN, so a
   healthy long-running stream sits at MAXLEN forever. Only XPENDING
   reflects "consumer drain done." (CR.A.13 / ADR-018.) The supported
   workflow demands this; the pause+reopen logic in
   :class:`quorin.wal_consumer.WALConsumer` is a SAFETY NET, not a
   substitute. See ADR-014 §"Operator runbook".
4. Set ``quorin:upgrade:pause:{safe_name}`` so any consumer that started
   between drain-confirm and now sees the pause and parks.
5. Open the OLD segment (INCR refcount).
6. Allocate the NEW segment via ``registry.create(new_schema, …,
   set_current=False)`` so ``schema:current`` stays pointed at OLD until
   the explicit flip Lua.
7. Per-field VECTORIZED translate from OLD bytes to NEW field arrays;
   build PyArrow table; call :func:`quorin._internal.insert_kernel.insert_many`.
8. Atomically flip ``schema:current`` via :data:`FLIP_SCHEMA_CURRENT_LUA`
   (CAS on the OLD name).
9. Clear the pause key so paused consumers unblock and reopen.
10. Wait for the first consumer to attach to the new segment (refcount > 1)
    so when the orchestrator exits and the watchdog DECRs our refcount,
    the segment survives. Without this wait, refcount=1 → 0 → cleanup
    queue → watchdog unlinks the new segment AND ``schema:current`` gets
    DEL'd by close-Lua's rotation conditional. Set ``wait_for_consumer=False``
    to opt out for fire-and-forget operator workflows.
11. Close the OLD segment handle (DECR; old holders may still keep it
    alive at refcount > 0 until they close).
12. Release lock; emit metrics + structured log.

Cancel-safety: every catch in :func:`upgrade_schema` is split into

* ``except (KeyboardInterrupt, SystemExit, asyncio.CancelledError)`` —
  quiet propagate; orphan cleanup runs if the new segment was created.
* ``except Exception`` — ``logger.exception(...)`` for forensics; orphan
  cleanup runs if the new segment was created.

Concurrent upgrades on the same schema are NOT supported in the sense of
running concurrently; the lock serializes them. Concurrent hydrate during
upgrade is naturally blocked because hydrate's precondition #1 trips on
``schema:current`` existing (it always does during an upgrade).

CRITICAL difference from hydrate's :func:`quorin.hydration._force_drop_orphan`:
:func:`_cleanup_orphan_new_segment` MUST NOT delete ``schema:current``
because Step 15's ``set_current=False`` means the key still points to
OLD throughout the upgrade. Deleting it would orphan all live OLD
readers. Test ``test_orphan_cleanup_does_not_delete_schema_current``
locks this regression. See ADR-014 §"Critical decisions" #10.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import importlib
import math
import os
import sys
import time
import uuid
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Final, cast

import numpy as np
import pyarrow as pa
import redis as redis_lib

from quorin._internal import posix_shm
from quorin._internal.evolution_lua import FLIP_SCHEMA_CURRENT_LUA, RELEASE_LOCK_LUA
from quorin._internal.insert_kernel import insert_many
from quorin.layout import (
    OCCUPIED,
    SegmentLayout,
    _read_metadata,
    _read_string_bytes,
    _slot_table_view,
)
from quorin.logging import get_logger
from quorin.metrics import (
    evolution_consumer_attach_seconds,
    evolution_consumer_pause_seconds,
    evolution_entity_count,
    evolution_seconds,
)
from quorin.schema import (
    DTYPE_TO_NUMPY,
    DType,
    FeatureSchema,
    _hash_name,
)
from quorin.shm import (
    KEY_SEGMENT_TO_SCHEMA,
    Segment,
    SegmentRegistry,
    _key_current,
    _key_pid_segments,
    _key_refcount,
    _safe_class_name,
)
from quorin.wal import DEFAULT_STREAM_KEY
from quorin.wal_consumer import DEFAULT_GROUP_NAME, KEY_WAL_CONSUMER_LIVENESS

if TYPE_CHECKING:
    pass

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# DType widening table — explicit, conservative for v1.
# ---------------------------------------------------------------------------

_DTYPE_WIDENING: Final[frozenset[tuple[DType, DType]]] = frozenset(
    {
        # Identity (always allowed; trivially safe).
        (DType.FLOAT32, DType.FLOAT32),
        (DType.FLOAT64, DType.FLOAT64),
        (DType.INT32, DType.INT32),
        (DType.INT64, DType.INT64),
        (DType.UINT8, DType.UINT8),
        # Widenings (no precision/range loss).
        (DType.FLOAT32, DType.FLOAT64),
        (DType.INT32, DType.INT64),
        (DType.UINT8, DType.INT32),
        (DType.UINT8, DType.INT64),
        # NOTE: cross-family changes (uint8 → float32, int → float, etc.)
        # are deliberately rejected for v1. Lift in Step 16+ if needed.
    }
)


# ---------------------------------------------------------------------------
# Lock / pause TTL constants. 600s is generous (10x the 1M acceptance
# criterion; covers WSL2 dev pace and large-schema operations). Bump via
# a future env-var if a workload regularly exceeds it.
# ---------------------------------------------------------------------------

LOCK_TTL_SECONDS: Final[int] = 600
PAUSE_TTL_SECONDS: Final[int] = 600
DEFAULT_WAIT_FOR_CONSUMER_SECONDS: Final[float] = 60.0


# ---------------------------------------------------------------------------
# Public types — return value + error class hierarchy.
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class UpgradeResult:
    """Outcome of one :func:`upgrade_schema` call.

    ``old_segment_name`` and ``new_segment_name`` are ``None`` only for
    dry-run results; for committed upgrades they hold the actual segment
    names. ``elapsed_seconds`` and ``consumer_pause_seconds`` cover ONLY
    the timed section (preconditions excluded). ``consumer_attach_seconds``
    is 0.0 when ``wait_for_consumer=False`` was passed.
    """

    old_segment_name: str | None
    new_segment_name: str | None
    entity_count: int
    row_size_bytes_old: int
    row_size_bytes_new: int
    elapsed_seconds: float
    consumer_pause_seconds: float
    consumer_attach_seconds: float


class UpgradeError(Exception):
    """Base for all evolution failures."""


class UpgradeConflictError(UpgradeError):
    """Refused before mutation: lock held, WAL not drained, consumer alive,
    or ``schema:current`` changed unexpectedly mid-flight. Idempotent retry
    safe."""


class UpgradeIncompatibleError(UpgradeError):
    """:func:`can_upgrade` returned false. Raised before lock acquisition.
    No state mutated."""


class UpgradeAbortedError(UpgradeError):
    """The atomic flip Lua's CAS lost the race (``schema:current`` is no
    longer the expected old segment). Orphan cleanup ran. Caller can
    retry after re-validating the world state."""


# ---------------------------------------------------------------------------
# can_upgrade — pure predicate, no side effects.
# ---------------------------------------------------------------------------


def can_upgrade(
    old: type[FeatureSchema],
    new: type[FeatureSchema],
) -> tuple[bool, list[str]]:
    """Validate that ``old`` can be upgraded to ``new``.

    Returns ``(ok, reasons)``. ``reasons`` is a list of human-readable
    failure descriptions (empty when ``ok`` is True). Pure function;
    safe to call from anywhere, including the dry-run CLI path.

    Rules (all must hold):

    1. ``new.version > old.version`` (strict monotonic).
    2. Every field in ``old.fields`` is present in ``new.fields`` by name.
    3. For each field present in both: ``new.shape == old.shape``.
    4. For each field present in both: ``(old.dtype, new.dtype)`` is in
       :data:`_DTYPE_WIDENING`.
    5. New fields (in ``new`` but not ``old``) are allowed; they will be
       zero-filled during the copy.

    Field reordering between old and new is allowed because the
    offset table is hash-sorted by name; declaration-order doesn't
    affect correctness. Bytes-on-wire layout will differ, so we cannot
    memcpy raw rows — :func:`_build_translation_table` does per-field
    translation regardless.
    """
    reasons: list[str] = []

    if new.version <= old.version:
        reasons.append(f"new.version ({new.version}) must be > old.version ({old.version})")

    old_by_name = {f.name: f for f in old.fields}
    new_by_name = {f.name: f for f in new.fields}

    removed = set(old_by_name) - set(new_by_name)
    if removed:
        reasons.append(f"fields removed (not allowed): {sorted(removed)}")

    for name, old_f in old_by_name.items():
        if name not in new_by_name:
            continue
        new_f = new_by_name[name]
        if old_f.shape != new_f.shape:
            reasons.append(
                f"field {name!r} shape changed: {old_f.shape} -> {new_f.shape} "
                "(not allowed for existing fields; add a new field with the "
                "new shape and deprecate the old one)"
            )
        if (old_f.dtype, new_f.dtype) not in _DTYPE_WIDENING:
            reasons.append(
                f"field {name!r} dtype change {old_f.dtype.name} -> "
                f"{new_f.dtype.name} is not a permitted widening"
            )

    # CR.A.4 (v0.1.1): 2D-shape upgrades silently break in
    # _build_translation_table — it constructs single-level
    # FixedSizeListArray regardless of shape rank, then insert_kernel's
    # rank-2 path does .flatten().flatten() which AttributeError's on
    # the resulting primitive array. Reject upfront with a clear
    # message until proper nested-list translation lands in v0.2.0.
    #
    # Two cases must be checked, NOT one:
    # (1) Field present in OLD with 2D shape (also catches kept-name
    #     2D fields that the loop above doesn't fault on because shape
    #     is unchanged).
    # (2) NEW added a 2D field that wasn't in OLD. The translation
    #     table builds a column for new fields too (zero-filled for
    #     missing data); insert_kernel's rank-2 path still fires.
    for name, old_f in old_by_name.items():
        if name in new_by_name and (len(old_f.shape) >= 2 or len(new_by_name[name].shape) >= 2):
            reasons.append(
                f"field {name!r}: 2D-shape upgrade not yet supported "
                "(see CR.A.4 / ADR-018; deferred to v0.2.0). Workaround: "
                "introduce a new 1D field and deprecate the 2D one."
            )
    for name, new_f in new_by_name.items():
        if name not in old_by_name and len(new_f.shape) >= 2:
            reasons.append(
                f"field {name!r}: new 2D-shape field not yet supported "
                "in upgrade (must be 1D until v0.2.0). Workaround: "
                "hydrate the new 2D field into a fresh segment instead."
            )

    return (not reasons, reasons)


# ---------------------------------------------------------------------------
# upgrade_schema — orchestrator.
# ---------------------------------------------------------------------------


def upgrade_schema(  # noqa: PLR0912, PLR0915 (orchestrator with sequential phases)
    old_schema: type[FeatureSchema],
    new_schema: type[FeatureSchema],
    registry: SegmentRegistry,
    *,
    redis_client: redis_lib.Redis,
    capacity_factor: float = 1.0,
    dry_run: bool = False,
    wait_for_consumer: bool = True,
    wait_seconds: float = DEFAULT_WAIT_FOR_CONSUMER_SECONDS,
) -> UpgradeResult:
    """Upgrade a schema in place. See module docstring for full contract.

    Args:
        old_schema: The currently-deployed schema class.
        new_schema: The new schema class (must have the same ``__name__``;
            differs in ``version`` and possibly ``fields``).
        registry: Segment registry for both opens.
        redis_client: Sync Redis client (the orchestrator is sync; mirrors
            :func:`quorin.hydration.hydrate`).
        capacity_factor: Multiplier on the OLD segment's row capacity
            for sizing the NEW. Default 1.0 (preserve). Lower-bounded by
            ``occupied_count + 1`` so the upgrade can't overflow mid-copy.
        dry_run: If True, validate compatibility and report projected
            counts without touching ``/dev/shm`` or ``schema:current``.
        wait_for_consumer: Default True — wait for the first consumer to
            ``open_current`` the new segment (refcount > 1) before exiting,
            so the orchestrator's death (and watchdog DECR) doesn't
            destroy the segment. Set False for fire-and-forget operator
            workflows where the operator orchestrates consumer attach
            externally.
        wait_seconds: Timeout for the wait loop (default 60s).

    Returns:
        :class:`UpgradeResult` with timing, segment names, entity count.

    Raises:
        UpgradeIncompatibleError: ``can_upgrade`` returned false, or the
            class names don't match.
        UpgradeConflictError: lock held by another orchestrator, WAL not
            drained, consumer alive, or ``schema:current`` changed
            mid-flight.
        UpgradeAbortedError: Atomic flip lost the CAS race.
        UpgradeError: insert_many returned an unexpected count.
    """
    # Sanity checks BEFORE lock acquisition — no side effects on failure.
    if old_schema.__name__ != new_schema.__name__:
        raise UpgradeIncompatibleError(
            f"schema class names must match: {old_schema.__name__!r} vs "
            f"{new_schema.__name__!r}. Upgrades preserve schema identity; "
            "the version field is what changes."
        )

    # CR.H.5 (v0.1.1): reject capacity_factor that would underflow (<= 0)
    # or be non-finite. Lower-bound is enforced post-occupied-read by
    # ``max(int(... * capacity_factor), occupied_count + 1)`` but a
    # negative / NaN factor would either silently saturate to the
    # occupied bound (swallowing operator intent) or do worse.
    if not math.isfinite(capacity_factor) or capacity_factor <= 0.0:
        raise ValueError(f"capacity_factor must be a finite float > 0, got {capacity_factor!r}")

    ok, reasons = can_upgrade(old_schema, new_schema)
    if not ok:
        raise UpgradeIncompatibleError(
            f"cannot upgrade {old_schema.__name__} v{old_schema.version} -> "
            f"v{new_schema.version}: {'; '.join(reasons)}"
        )

    if dry_run:
        return _dry_run_result(old_schema, new_schema, registry, redis_client)

    # Use _safe_class_name for all Step 15 keys (matches Step 14's
    # quorin:schema:{safe_name}:current convention; avoids collisions if a
    # class name has dots/hyphens that get sanitized).
    safe_name = _safe_class_name(old_schema.__name__)
    schema_name_log = old_schema.__name__  # logging-friendly raw name
    lock_key = f"quorin:upgrade:lock:{safe_name}".encode()
    pause_key = f"quorin:upgrade:pause:{safe_name}".encode()
    lock_token = uuid.uuid4().hex.encode()

    # Register Lua scripts at orchestrator entry (mirrors Step 14's
    # WatchdogState.__init__ pattern; redis-py uses EVALSHA after first call).
    flip_script = redis_client.register_script(FLIP_SCHEMA_CURRENT_LUA)
    release_script = redis_client.register_script(RELEASE_LOCK_LUA)

    # 1. Acquire upgrade lock.
    acquired = bool(redis_client.set(lock_key, lock_token, nx=True, ex=LOCK_TTL_SECONDS))
    if not acquired:
        holder = redis_client.get(lock_key)
        raise UpgradeConflictError(
            f"upgrade already in progress for {schema_name_log!r} (lock held by {holder!r})"
        )

    # 1b. WAL drain precondition — XPENDING + no consumer liveness.
    # CR.A.13: XLEN is intentionally NOT checked. XACK does not decrement
    # XLEN; only XTRIM/XDEL shrink the stream, and the producer's MAXLEN ~
    # trim runs only on XADD with the ~ flag. After any meaningful
    # production traffic, XLEN ≈ MAXLEN forever — so requiring XLEN==0
    # rejects every real upgrade attempt and forces operators to manually
    # XTRIM (undocumented in v0.1.0). PENDING is the correct measure:
    # zero un-ACKed messages means the consumer caught up. See ADR-018.
    pending = 0
    try:
        # XPENDING summary form return shape varies by redis-py version:
        #   * redis-py 5+ wraps the response into a DICT
        #     ``{'pending': N, 'min': ..., 'max': ..., 'consumers': [...]}``.
        #   * Legacy versions / direct Redis-server replies are LISTs
        #     ``[count, min_id, max_id, [(consumer, count), ...]]``.
        # The cast("list[Any]") in v0.1.0 was wrong for redis-py 5; the
        # latent bug was never exercised because the v0.1.0 precondition
        # checked XLEN > 0 FIRST, and any traffic at all made XLEN > 0.
        # CR.A.13 (v0.1.1) removed the XLEN short-circuit and surfaced
        # this; handle both shapes defensively.
        # When the group doesn't exist yet, redis-py raises ResponseError.
        # ResponseError MUST be caught BEFORE RedisError (its parent class) —
        # same hierarchy gotcha as Step 14's ZombieProcess/NoSuchProcess.
        xp_raw = redis_client.xpending(DEFAULT_STREAM_KEY, DEFAULT_GROUP_NAME.encode("ascii"))
        if isinstance(xp_raw, dict):
            pending = int(xp_raw.get("pending", 0))
        elif isinstance(xp_raw, list) and xp_raw:
            pending = int(xp_raw[0])
        else:
            pending = 0
    except redis_lib.ResponseError:
        # Group doesn't exist yet — no pending entries.
        pending = 0
    except redis_lib.RedisError:
        # Connection issue — treat as 0; the liveness check below will
        # surface a truly broken Redis.
        pending = 0
    if pending > 0:
        with contextlib.suppress(Exception):
            release_script(keys=[lock_key], args=[lock_token])
        raise UpgradeConflictError(
            f"WAL not drained: {pending} messages still pending consumer ACK. "
            "Stop producers and wait for the consumer to drain (XPENDING == 0) "
            "before upgrade. See ADR-014 operator runbook."
        )
    consumer_alive = redis_client.get(KEY_WAL_CONSUMER_LIVENESS) is not None
    if consumer_alive:
        with contextlib.suppress(Exception):
            release_script(keys=[lock_key], args=[lock_token])
        raise UpgradeConflictError(
            "WAL consumer is still alive (quorin:wal_consumer:liveness set) — "
            "stop the consumer before upgrade. See ADR-014 operator runbook."
        )

    t0 = time.perf_counter()
    new_seg: Segment | None = None
    old_seg: Segment | None = None
    # CR.A.7: track old_seg's refcount lifecycle so the finally branch
    # can close on any exception path without double-closing on success.
    old_seg_closed = False
    # CR.A.7 + post-flip safety: set True the moment the Lua CAS returns
    # success. Before flip, new_seg is an in-progress orphan that SHOULD
    # be cleaned up on exception. After flip, new_seg is LIVE production
    # state — schema:current points at it, readers attach. Orphan-cleanup
    # post-flip would force every reader's open_current retry to fail with
    # SegmentNotFoundError until manual intervention. The flag is the
    # binary signal that converts "orphan cleanup is safe" to "forbidden."
    flip_completed = False
    consumer_pause_seconds = 0.0
    consumer_attach_seconds = 0.0
    try:
        # 2. Set pause key as a SAFETY NET. The supported workflow already
        #    drained the consumer; this catches an operator who started a
        #    consumer between drain-confirm and now. Value = lock_token
        #    (UUID hex) so operators can `GET quorin:upgrade:pause:{name}`
        #    to identify which orchestrator's token set it.
        redis_client.set(pause_key, lock_token, ex=PAUSE_TTL_SECONDS)

        # 3. Open OLD segment (INCRs old refcount — DECR'd later via close).
        old_seg = registry.open_current(old_schema)

        # Defense-in-depth: schema:current must still match old_seg.name.
        # (Hydrate refuses while schema:current exists, so concurrent hydrate
        # is impossible in practice — but verify anyway.)
        # CR.A.5: handle both bytes (default redis client) and str
        # (decode_responses=True). Prior code's `isinstance(current, bytes)`
        # short-circuited to False on str clients, silently bypassing the
        # mismatch branch.
        current = redis_client.get(_key_current(old_schema))
        current_str = current.decode("utf-8") if isinstance(current, bytes) else current
        if current_str != old_seg.name:
            raise UpgradeConflictError(
                f"schema:current changed under us: expected {old_seg.name!r}, got {current!r}"
            )

        # 4. Compute new capacity (use layout.capacity = row capacity, NOT
        #    layout.slot_capacity = next_pow2(capacity*2)).
        occupied_count, _pool_cursor = _read_metadata(old_seg.handle.buf)
        new_capacity = max(
            int(old_seg.layout.capacity * capacity_factor),
            occupied_count + 1,
        )

        # 5. Create NEW segment with set_current=False so schema:current
        #    stays pointed at OLD throughout the copy. Refcount=1 (ghost
        #    hold on orchestrator).
        new_seg = registry.create(
            new_schema,
            capacity=new_capacity,
            max_id_bytes=old_seg.layout.max_id_bytes,
            set_current=False,
        )

        # 6. Vectorized translate + insert_many. Memory peak ≈ 3-4x row data
        #    size (columnar arrays + PyArrow + insert_many transient).
        table = _build_translation_table(old_seg, old_schema, new_schema)
        inserted = insert_many(new_seg, table)
        if inserted != occupied_count:
            raise UpgradeError(f"insert_many returned {inserted}, expected {occupied_count}")

        # 7. Atomic flip via Lua CAS.
        flip_result = int(
            cast(
                "Any",
                flip_script(
                    keys=[_key_current(old_schema)],
                    args=[old_seg.name.encode("utf-8"), new_seg.name.encode("utf-8")],
                ),
            )
        )
        if flip_result == 0:
            raise UpgradeAbortedError(
                f"flip CAS lost: schema:current changed during copy. "
                f"Old segment {old_seg.name!r} is no longer current."
            )
        if flip_result == -1:
            raise UpgradeConflictError(
                "schema:current was deleted during upgrade (operator interference?)"
            )
        # CR.A.7: flip succeeded. From here on, new_seg is LIVE — exception
        # branches MUST NOT call _cleanup_orphan_new_segment on it.
        flip_completed = True

        # 8. Clear pause key FIRST so consumers can begin reopen.
        consumer_pause_seconds = time.perf_counter() - t0
        with contextlib.suppress(Exception):
            redis_client.delete(pause_key)

        # 9. Wait for first consumer to attach (refcount > 1) so the new
        #    segment survives the orchestrator's exit + watchdog DECR.
        if wait_for_consumer:
            attach_t0 = time.perf_counter()
            attach_deadline = time.monotonic() + wait_seconds
            attached = False
            while time.monotonic() < attach_deadline:
                rc_raw = redis_client.get(_key_refcount(new_seg.name))
                refcount = int(cast("Any", rc_raw)) if rc_raw is not None else 0
                if refcount > 1:  # 1 (orchestrator) + ≥1 (consumer)
                    attached = True
                    break
                time.sleep(0.5)
            consumer_attach_seconds = time.perf_counter() - attach_t0
            evolution_consumer_attach_seconds.observe(consumer_attach_seconds)
            if not attached:
                logger.warning(
                    "evolution.consumer_attach_timeout",
                    new_segment=new_seg.name,
                    wait_seconds=wait_seconds,
                    msg=(
                        "Orchestrator exiting without consumer attach. "
                        "Watchdog may clean up new segment within ~150s. "
                        "Operator must start a consumer or re-run upgrade."
                    ),
                )

        # 10. Close OLD segment handle (DECR; old holders may still keep it
        #     alive). If we held the only open, refcount hits 0 → close-Lua
        #     queues for cleanup → watchdog drains. close-Lua's rotation
        #     conditional sees current=new_name ≠ old_name → does NOT delete
        #     schema:current.
        registry.close(old_seg)
        old_seg_closed = True  # CR.A.7: prevent finally double-close.

        elapsed = time.perf_counter() - t0
        evolution_seconds.labels(outcome="ok").observe(elapsed)
        evolution_entity_count.observe(occupied_count)
        evolution_consumer_pause_seconds.observe(consumer_pause_seconds)

        logger.info(
            "evolution.upgrade_complete",
            schema=schema_name_log,
            old_version=old_schema.version,
            new_version=new_schema.version,
            old_segment=old_seg.name,
            new_segment=new_seg.name,
            entity_count=occupied_count,
            elapsed_seconds=round(elapsed, 4),
            consumer_pause_seconds=round(consumer_pause_seconds, 4),
            consumer_attach_seconds=round(consumer_attach_seconds, 4),
        )

        return UpgradeResult(
            old_segment_name=old_seg.name,
            new_segment_name=new_seg.name,
            entity_count=occupied_count,
            row_size_bytes_old=old_seg.layout.row_size,
            row_size_bytes_new=new_seg.layout.row_size,
            elapsed_seconds=elapsed,
            consumer_pause_seconds=consumer_pause_seconds,
            consumer_attach_seconds=consumer_attach_seconds,
        )

    except (KeyboardInterrupt, SystemExit, asyncio.CancelledError):
        # Quiet propagation. Orphan cleanup before re-raise (no logger.exception
        # — operator-initiated cancel is not an "exception" in the forensic
        # sense). CR.A.7: only orphan-clean PRE-flip; post-flip new_seg is
        # LIVE state.
        if new_seg is not None and not flip_completed:
            _cleanup_orphan_new_segment(new_seg, redis_client, new_schema)
        raise
    except Exception:
        evolution_seconds.labels(outcome="err").observe(time.perf_counter() - t0)
        logger.exception("evolution.upgrade_failed", schema=schema_name_log)
        # CR.A.7: same gate. If the failure landed AFTER the flip (e.g.
        # registry.close(old_seg) raised), do NOT delete new_seg — it's
        # production state. Log instead so operators know to inspect.
        if new_seg is not None and not flip_completed:
            _cleanup_orphan_new_segment(new_seg, redis_client, new_schema)
        elif new_seg is not None and flip_completed:
            logger.error(
                "evolution.post_flip_failure_new_segment_preserved",
                new_segment=new_seg.name,
                schema=schema_name_log,
                msg=(
                    "upgrade flip succeeded but post-flip cleanup raised; "
                    "new segment is LIVE and was NOT orphan-cleaned. "
                    "Inspect old_segment in /dev/shm if it persists after "
                    "live readers should have closed (e.g., minutes after "
                    "the upgrade completed) — that indicates a stuck "
                    "refcount, not the live-readers-still-attached case "
                    "which is expected briefly post-flip."
                ),
            )
        raise
    finally:
        # CR.A.7: close old_seg if not already closed on success. The
        # success-path close is at step 10; this finally handles all
        # exception paths and the rare case where step 10 itself raised.
        # contextlib.suppress because partial state already raising the
        # primary exception — don't mask it with a close failure.
        if old_seg is not None and not old_seg_closed:
            with contextlib.suppress(Exception):
                registry.close(old_seg)
        # Best-effort lock + pause release. Even if cleanup leaves the new
        # segment lingering, the watchdog reaps it in ~150s.
        with contextlib.suppress(Exception):
            release_script(keys=[lock_key], args=[lock_token])
        with contextlib.suppress(Exception):
            redis_client.delete(pause_key)


# ---------------------------------------------------------------------------
# _build_translation_table — per-field VECTORIZED translation.
# ---------------------------------------------------------------------------


def _build_translation_table(  # noqa: PLR0912 (per-field translation has many sequential branches by design)
    old_seg: Segment,
    old_schema: type[FeatureSchema],
    new_schema: type[FeatureSchema],
) -> pa.Table:
    """Read OLD segment row-major bytes; produce a PyArrow table for insert_many.

    Vectorized per-field via numpy strided views; NO per-row Python loop over
    feature data. Per-entity-ID Python loop is the only Python bottleneck;
    at ~1µs/ID it's ~1s at 1M, which is acceptable.

    Operations count: 2 x n_fields vectorized numpy ops (one extract + one
    cast per existing field; one np.zeros per new field). NOT N x n_fields.

    NaN bit-pattern preservation: numpy's standard widening
    (``float32_arr.astype(np.float64)``) preserves NaN payloads through
    IEEE 754 quiet-bit promotion. Locks invariant #12 across the upgrade
    boundary; covered by ``test_float32_to_float64_preserves_nan_bit_pattern``.
    """
    n_rows, _pool_cursor = _read_metadata(old_seg.handle.buf)
    old_layout: SegmentLayout = old_seg.layout
    old_buf = old_seg.handle.buf

    # 1. Build name → row-relative byte-offset map for OLD layout. The
    #    row_offset_table stores name_hash + byte_offset; we map back to
    #    field names via blake2b hashing each schema.fields[i].name.
    old_field_byte_offsets = _row_byte_offsets_by_name(old_schema, old_layout.row_offset_table)

    old_by_name = {f.name: f for f in old_schema.fields}
    new_field_arrays: dict[str, np.ndarray[Any, np.dtype[Any]]] = {}

    if n_rows > 0:
        # 2. Build a (n_rows, row_size) uint8 view of the feature_rows region.
        #    SAFETY: this view holds a mmap reference for the duration of the
        #    function. Per-field astype(copy=True) below releases each per-field
        #    array's hold; the all_rows view itself goes out of scope at function
        #    return.
        all_rows = np.frombuffer(
            old_buf,
            dtype=np.uint8,
            count=n_rows * old_layout.row_size,
            offset=old_layout.feature_rows_offset,
        ).reshape(n_rows, old_layout.row_size)

        # 3. Per-field vectorized extract + cast.
        for new_f in new_schema.fields:
            old_f = old_by_name.get(new_f.name)
            new_dt = DTYPE_TO_NUMPY[new_f.dtype]

            if old_f is None:
                # New field → zero-fill.
                shape: tuple[int, ...] = (
                    (n_rows,) if new_f.shape == () else (n_rows, new_f.element_count)
                )
                new_field_arrays[new_f.name] = np.zeros(shape, new_dt)
                continue

            # Existing field — vectorized strided slice + reinterpret + cast.
            off = old_field_byte_offsets[old_f.name]
            bc = old_f.byte_count
            raw = all_rows[:, off : off + bc]  # (n_rows, byte_count), strided view
            # raw.reshape(-1) forces a contiguous copy because raw's stride
            # is row_size, not byte_count. The copy is the per-field cost
            # baseline (~5-15ms at 1M x 4-byte fields). View as old dtype
            # then astype to new dtype (handles widening + NaN preservation).
            typed = (
                raw.reshape(-1)
                .view(DTYPE_TO_NUMPY[old_f.dtype])
                .reshape(n_rows, old_f.element_count)
            )
            if new_f.shape == ():
                new_field_arrays[new_f.name] = typed[:, 0].astype(new_dt, copy=True)
            else:
                new_field_arrays[new_f.name] = typed.astype(new_dt, copy=True)
    else:
        # n_rows == 0 — initialize empty arrays for the PyArrow build path.
        # (Empty old segment is a legitimate state: hydrate-then-upgrade with
        # no writes in between.)
        for new_f in new_schema.fields:
            new_dt = DTYPE_TO_NUMPY[new_f.dtype]
            if new_f.shape == ():
                new_field_arrays[new_f.name] = np.zeros(0, new_dt)
            else:
                new_field_arrays[new_f.name] = np.zeros((0, new_f.element_count), new_dt)

    # 4. Entity IDs — VECTORIZED slot extraction (NOT iterate_occupied which
    #    would Python-iterate 2 x capacity slot rows).
    if n_rows > 0:
        slot_view = _slot_table_view(old_buf, old_layout)
        occupied_mask = slot_view["flags"] == OCCUPIED  # vectorized boolean
        id_offsets = slot_view["id_offset"][occupied_mask]
        feature_row_indices = slot_view["feature_row_index"][occupied_mask]

        # Sanity: occupied count must match next_free_row_index in v1 (no
        # tombstones).
        if int(occupied_mask.sum()) != n_rows:
            raise UpgradeError(
                f"old segment inconsistency: n_rows={n_rows}, occupied={int(occupied_mask.sum())}"
            )

        # Reorder id_offsets by feature_row_index so the decode loop yields
        # entity_ids[i] aligned with feature_row i (the order the per-field
        # arrays were built in via the row-major bytes view).
        order = np.argsort(feature_row_indices)
        sorted_id_offsets = id_offsets[order]

        entity_ids = [
            _read_string_bytes(old_buf, old_layout, int(off)).decode("utf-8")
            for off in sorted_id_offsets
        ]
    else:
        entity_ids = []

    # 5. Build PyArrow table.
    arrays: dict[str, pa.Array] = {
        "entity_id": pa.array(entity_ids, type=pa.string()),
    }
    for new_f in new_schema.fields:
        arr = new_field_arrays[new_f.name]
        if new_f.shape == ():
            arrays[new_f.name] = pa.array(arr)
        else:
            # FixedSizeListArray for shaped fields. insert_many's column
            # extractor (Step 13 commit B) handles rank dispatch.
            flat = arr.reshape(-1)
            arrays[new_f.name] = pa.FixedSizeListArray.from_arrays(
                pa.array(flat),
                new_f.element_count,
            )

    return pa.table(arrays)


def _row_byte_offsets_by_name(
    schema: type[FeatureSchema],
    row_offset_table: np.ndarray[Any, np.dtype[np.void]],
) -> dict[str, int]:
    """Map ``field_name → row-relative byte offset`` for ``schema``.

    The row_offset_table stores ``name_hash`` + ``byte_offset`` (not
    ``name``). We hash each schema.fields[i].name via blake2b digest_size=8
    (the pinned algorithm; CLAUDE.md invariant #5) and look up the matching
    row in the table.
    """
    hash_to_offset: dict[int, int] = {
        int(row["name_hash"]): int(row["byte_offset"]) for row in row_offset_table
    }
    return {f.name: hash_to_offset[_hash_name(f.name)] for f in schema.fields}


# ---------------------------------------------------------------------------
# _cleanup_orphan_new_segment — CRITICAL: must NOT delete schema:current.
# ---------------------------------------------------------------------------


def _cleanup_orphan_new_segment(
    seg: Segment,
    redis_client: redis_lib.Redis,
    schema: type[FeatureSchema],
) -> None:
    """Best-effort cleanup of a partially-created NEW segment.

    DIFFERENCE FROM :func:`quorin.hydration._force_drop_orphan`:

    Hydrate's :func:`SegmentRegistry.create` sets ``schema:current`` as part
    of its pipeline, so hydrate's orphan cleanup MUST DEL ``schema:current``.

    Step 15's ``create(set_current=False)`` does NOT set ``schema:current``.
    Schema:current still points to OLD throughout the upgrade. **Deleting
    it would orphan all live OLD readers** — their ``open_current`` retries
    would fail with ``SegmentNotFoundError`` and the system would end up
    in ``schema:current undefined`` state.

    DO NOT consolidate this with ``_force_drop_orphan``. The schema:current
    handling is the load-bearing difference.

    Test ``test_orphan_cleanup_does_not_delete_schema_current`` is the
    binding regression for this contract.

    Best-effort: each block wrapped in its own try/except. Idempotent —
    safe to call twice. Watchdog reconciles if Redis is unreachable.
    Caller-side ``if new_seg is not None:`` guards make this safe to call
    in the orchestrator's exception path even when ``registry.create``
    never ran.
    """
    if seg is None:
        return
    del schema  # arg kept for symmetry with hydration._force_drop_orphan
    pid = os.getpid()
    redis_ok = shm_unlink_ok = shm_close_ok = False
    # Block 1: Redis cleanup. NO p.delete(_key_current(schema)).
    try:
        with redis_client.pipeline(transaction=True) as p:
            p.delete(_key_refcount(seg.name))
            p.srem(_key_pid_segments(pid), seg.name)
            p.hdel(KEY_SEGMENT_TO_SCHEMA, seg.name)
            p.execute()
        redis_ok = True
    except Exception:
        logger.warning("evolution.orphan_redis_cleanup_failed", segment=seg.name)

    # Block 2: posix_shm.unlink (creator-only — see CLAUDE.md invariant #6).
    try:
        posix_shm.unlink(seg.name)
        shm_unlink_ok = True
    except FileNotFoundError:
        # Already unlinked — idempotent path.
        shm_unlink_ok = True
    except Exception:
        logger.warning("evolution.orphan_unlink_failed", segment=seg.name)

    # Block 3: posix_shm close handle.
    try:
        posix_shm.close(seg.handle)
        shm_close_ok = True
    except Exception:
        logger.warning("evolution.orphan_close_failed", segment=seg.name)

    if redis_ok and shm_unlink_ok and shm_close_ok:
        logger.info("evolution.orphan_dropped", segment=seg.name)


# ---------------------------------------------------------------------------
# _dry_run_result — diff + projected counts; NO mutations.
# ---------------------------------------------------------------------------


def _dry_run_result(
    old: type[FeatureSchema],
    new: type[FeatureSchema],
    registry: SegmentRegistry,
    redis_client: redis_lib.Redis,
) -> UpgradeResult:
    """Validate compatibility + report projected counts. NO mutations to
    ``/dev/shm`` or ``schema:current``. Acquires NO lock and sets NO pause key.

    NOTE: opens the OLD segment via :func:`SegmentRegistry.open_current`
    which DOES INCR refcount. Always paired with ``registry.close`` in the
    finally block. Net refcount delta = 0; no observable mutation.

    Logs ``evolution.dry_run`` with the diff fields so operators see the
    same diff they'd see if ``--confirm`` were passed.
    """
    del redis_client  # not needed — registry.open_current uses its own client.

    old_seg = registry.open_current(old)
    try:
        n_rows, _ = _read_metadata(old_seg.handle.buf)
        old_row_size = old_seg.layout.row_size
        # New row_size: import compute_layout-equivalent bytes-sum from the
        # schema fields. Cheap; matches what registry.create would allocate.
        new_row_size = sum(f.byte_count for f in new.fields)

        old_by_name = {f.name: f for f in old.fields}
        added = [f.name for f in new.fields if f.name not in old_by_name]
        widened = [
            (f.name, old_by_name[f.name].dtype.name, f.dtype.name)
            for f in new.fields
            if f.name in old_by_name and old_by_name[f.name].dtype != f.dtype
        ]

        logger.info(
            "evolution.dry_run",
            schema=old.__name__,
            old_version=old.version,
            new_version=new.version,
            entity_count=n_rows,
            row_size_bytes_old=old_row_size,
            row_size_bytes_new=new_row_size,
            added_fields=added,
            widened_fields=widened,
        )

        return UpgradeResult(
            old_segment_name=old_seg.name,
            new_segment_name=None,
            entity_count=n_rows,
            row_size_bytes_old=old_row_size,
            row_size_bytes_new=new_row_size,
            elapsed_seconds=0.0,
            consumer_pause_seconds=0.0,
            consumer_attach_seconds=0.0,
        )
    finally:
        registry.close(old_seg)  # pairs the open_current INCR


# ---------------------------------------------------------------------------
# CLI: python -m quorin.evolution upgrade ...
# ---------------------------------------------------------------------------


def _resolve_schema(dotted: str) -> type[FeatureSchema]:
    """Resolve ``module.path:ClassName`` to a FeatureSchema subclass.

    Mirrors entry_points-style dotted-path resolution; same shape as
    common Python CLI tools (gunicorn, uvicorn) so operators don't need
    new conventions.
    """
    if ":" not in dotted:
        raise ValueError(f"--old/--new must be `module.path:ClassName`, got {dotted!r}")
    module_path, class_name = dotted.rsplit(":", 1)
    module = importlib.import_module(module_path)
    klass = getattr(module, class_name)
    if not (isinstance(klass, type) and issubclass(klass, FeatureSchema)):
        raise TypeError(
            f"{dotted!r} resolves to {type(klass).__name__}, not a FeatureSchema subclass"
        )
    return klass


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m quorin.evolution",
        description=(
            "Quorin schema evolution (Step 15): atomic segment swap with "
            "live readers. See ADR-014 for the supported operator workflow "
            "(drain WAL stream + stop consumer BEFORE running upgrade)."
        ),
    )
    sub = parser.add_subparsers(dest="cmd", required=True)
    up = sub.add_parser("upgrade", help="Run an upgrade.")
    up.add_argument("--redis", required=True, help="Redis URL, e.g. redis://localhost:6379/0")
    up.add_argument(
        "--old",
        required=True,
        help="Dotted path to the OLD schema class, e.g. my_pkg.schemas:UserV1",
    )
    up.add_argument(
        "--new",
        required=True,
        help="Dotted path to the NEW schema class, e.g. my_pkg.schemas:UserV2",
    )
    up.add_argument(
        "--capacity-factor",
        type=float,
        default=1.0,
        help="Multiplier on old segment's row capacity. Default 1.0.",
    )
    up.add_argument(
        "--no-wait-for-consumer",
        action="store_true",
        help="Skip wait-for-consumer-attach (fire-and-forget).",
    )
    up.add_argument(
        "--wait-seconds",
        type=float,
        default=DEFAULT_WAIT_FOR_CONSUMER_SECONDS,
        help=f"Wait-for-attach timeout (default: {DEFAULT_WAIT_FOR_CONSUMER_SECONDS}).",
    )
    up.add_argument(
        "--dry-run",
        action="store_true",
        help="Report projected diff + counts WITHOUT mutating anything.",
    )
    up.add_argument(
        "--confirm",
        action="store_true",
        help="Required for non-dry-run execution. Without --confirm, exits after dry-run.",
    )
    up.add_argument(
        "--metrics-port",
        type=int,
        default=None,
        help="If set, expose /metrics via prometheus_client.start_http_server.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """CLI entry for ``python -m quorin.evolution upgrade ...``.

    Parses CLI args, opens a Redis client, optionally starts a Prometheus
    metrics server, and runs :func:`upgrade_schema`. Returns the process exit
    code (0 on success; non-zero on UpgradeError or unhandled exception).
    """
    args = _parse_args(argv)
    redis_client = redis_lib.Redis.from_url(args.redis)
    if args.metrics_port is not None:
        from prometheus_client import start_http_server  # noqa: PLC0415

        from quorin.metrics import registry as quorin_registry  # noqa: PLC0415

        start_http_server(args.metrics_port, registry=quorin_registry)

    old_schema = _resolve_schema(args.old)
    new_schema = _resolve_schema(args.new)
    registry = SegmentRegistry(redis_client)

    # Always run dry-run first; print diff.
    dry = upgrade_schema(
        old_schema,
        new_schema,
        registry,
        redis_client=redis_client,
        capacity_factor=args.capacity_factor,
        dry_run=True,
    )
    print(
        f"DRY RUN: {old_schema.__name__} v{old_schema.version} -> "
        f"v{new_schema.version}\n"
        f"  entity_count: {dry.entity_count}\n"
        f"  row_size_bytes: {dry.row_size_bytes_old} -> {dry.row_size_bytes_new}\n"
    )

    if args.dry_run or not args.confirm:
        if not args.confirm and not args.dry_run:
            print("Pass --confirm to apply the upgrade.")
        return 0

    # Real run.
    result = upgrade_schema(
        old_schema,
        new_schema,
        registry,
        redis_client=redis_client,
        capacity_factor=args.capacity_factor,
        dry_run=False,
        wait_for_consumer=not args.no_wait_for_consumer,
        wait_seconds=args.wait_seconds,
    )
    print(
        f"OK: {old_schema.__name__} v{old_schema.version} -> "
        f"v{new_schema.version}\n"
        f"  old_segment: {result.old_segment_name}\n"
        f"  new_segment: {result.new_segment_name}\n"
        f"  entity_count: {result.entity_count}\n"
        f"  elapsed_seconds: {result.elapsed_seconds:.4f}\n"
        f"  consumer_pause_seconds: {result.consumer_pause_seconds:.4f}\n"
        f"  consumer_attach_seconds: {result.consumer_attach_seconds:.4f}\n"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
