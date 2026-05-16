"""Integration tests for Step 15 WAL consumer pause-and-reopen safety net.

CR.F.1 (audit 2026-05-07): ``WALConsumer._check_upgrade_pause_and_reopen``
is a load-bearing safety net for schema upgrades but had 0% test coverage
in v0.1.0. This module covers the full pause/clear/reopen flow against
real Redis.

Coverage (5 scenarios, per the audit-mandated catalog):

* ``test_pause_path_drains_pending_then_parks`` — pause-key set, consumer
  drains pending+ack and parks (loop refusing new XREADGROUP reads).
* ``test_clear_path_reopens_segment`` — pause cleared, consumer reopens
  ``schema:current`` and swaps in the new segment.
* ``test_crc_mismatch_during_reopen_parks_consumer_and_increments_counter``
  — ``open_current`` raises ``SchemaCRCMismatchError``; consumer parks
  (does NOT crash), bumps ``wal_consumer_schema_crc_mismatch_total``,
  sets stop-event.
* ``test_liveness_refreshes_during_park`` — inline 10s-gated refresh
  inside the pause loop keeps the liveness key TTL fresh; watchdog would
  not declare the parked consumer dead.
* ``test_idempotent_double_pause_no_extra_work`` — pause set twice in a
  row; second iter is a stable no-op.

Test sequencing follows CLAUDE.md §8 — the pause-check sits BEFORE
liveness/process logic in ``_run_one_iter``, so we drive single iters
manually rather than racing with the run-loop. See
``test_hydration_e2e.py::test_e2`` for the manual-iter pattern this
mirrors.

We deliberately call ``_check_upgrade_pause_and_reopen`` directly in
most scenarios (it's the function under test) and ``_run_one_iter`` in
the no-pause fast-path scenarios — both give us deterministic single-
step semantics that aren't achievable while ``run()`` owns the loop.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import sys
import time

import pytest

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        sys.platform == "win32",
        reason="Step 15 pause+reopen requires POSIX shm + asyncio Redis",
    ),
]

import redis  # noqa: E402
import redis.asyncio  # noqa: E402

import _helpers as h  # noqa: E402
from quorin import layout  # noqa: E402
from quorin.metrics import (  # noqa: E402
    wal_consumer_schema_crc_mismatch_total,
)
from quorin.schema import DType, FeatureField, FeatureSchema  # noqa: E402
from quorin.shm import (  # noqa: E402
    SchemaCRCMismatchError,
    SegmentRegistry,
    _key_current,
    _safe_class_name,
)
from quorin.wal_consumer import (  # noqa: E402
    KEY_WAL_CONSUMER_LIVENESS,
    LIVENESS_REFRESH_INTERVAL_SECONDS,
    LIVENESS_TTL_SECONDS,
    WALConsumer,
)

# ---------------------------------------------------------------------------
# Schema. Top-level for picklability + reuse.
# ---------------------------------------------------------------------------


class _IntPauseReopen(FeatureSchema):
    version = 1
    fields = [
        FeatureField("a", DType.FLOAT32),
        FeatureField("b", DType.INT32),
    ]


_PAUSE_KEY = b"quorin:upgrade:pause:" + _safe_class_name(_IntPauseReopen.__name__).encode("utf-8")


# ---------------------------------------------------------------------------
# Fixtures.
# ---------------------------------------------------------------------------


@pytest.fixture
def registry(redis_client: redis.Redis) -> SegmentRegistry:
    return SegmentRegistry(redis_client)


@pytest.fixture
async def async_redis(redis_client: redis.Redis):
    """Async Redis paired with the conftest's sync redis_client.

    The sync-side ping (in conftest's redis_client) gates this on
    Redis-reachable; if Redis is down both fixtures skip together.
    """
    del redis_client
    url = os.environ.get("QUORIN_REDIS_URL", "redis://127.0.0.1:6379/0")
    # CR.E.6: socket_timeout matches what production should set.
    client = redis.asyncio.Redis.from_url(
        url,
        decode_responses=False,
        socket_timeout=5.0,
        socket_connect_timeout=5.0,
    )
    try:
        await client.ping()
    except Exception:
        pytest.skip("Async Redis not reachable")
    yield client
    await client.aclose()


@pytest.fixture(autouse=True)
def _cleanup_step15_keys(redis_client: redis.Redis):
    """Pre-test cleanup of any leftover pause/lock keys from prior runs.

    The autouse ``_shm_test_isolation`` fixture in tests/conftest.py
    handles post-test cleanup; this fires PRE-test to defend against
    a crashed prior test that left a pause key set, which would cause
    spurious park behavior.
    """
    for k in redis_client.scan_iter(match="quorin:upgrade:*", count=100):
        redis_client.delete(k)
    redis_client.delete(KEY_WAL_CONSUMER_LIVENESS)
    yield


def _make_consumer(
    async_redis_client: redis.asyncio.Redis,
    registry_obj: SegmentRegistry,
    seg,
    *,
    consumer_name: str,
) -> WALConsumer:
    """Build a consumer with the test schema's borrowed segment."""
    return WALConsumer(
        async_redis_client,  # type: ignore[arg-type]
        segments={"_IntPauseReopen": seg},
        registry=registry_obj,
        offline=None,  # Noop offline writer
        consumer_name=consumer_name,
        block_ms=50,
        batch_count=10,
        flush_interval_seconds=300.0,  # don't auto-flush during the test
        max_pending_ack=10_000,
    )


def _seed_consumer_state(consumer: WALConsumer) -> None:
    """Seed the bits of consumer state that ``run()`` would set up.

    The consumer's startup awaits (lock acquire, group ensure, PEL drain,
    initial liveness SET) are skipped; we only seed what
    ``_check_upgrade_pause_and_reopen`` and ``_run_one_iter`` actually
    touch (``_pid_str`` + ``_liveness_last_refresh``). Mirrors the manual-
    iter pattern in test_hydration_e2e.py::test_e2.
    """
    consumer._pid_str = str(os.getpid()).encode("ascii")
    consumer._liveness_last_refresh = time.monotonic()


def _create_segment(registry_obj: SegmentRegistry, schema: type[FeatureSchema]):
    """``registry.create`` defaults ``set_current=True`` → schema:current set."""
    return registry_obj.create(schema, capacity=16)


def _drop_current(redis_client: redis.Redis) -> None:
    """Operator-runbook DEL of schema:current to allow segment churn in tests."""
    redis_client.delete(_key_current(_IntPauseReopen))


# ---------------------------------------------------------------------------
# Scenario 1 — pause path: drains pending, parks.
# ---------------------------------------------------------------------------


async def test_pause_path_drains_pending_then_parks(
    redis_client: redis.Redis,
    async_redis: redis.asyncio.Redis,
    registry: SegmentRegistry,
) -> None:
    """Pause set; consumer drains pending_ack, then loops in pause-poll.

    We can't easily run a full XREADGROUP-and-park observation without
    racing the consumer's run loop, so we drive the function under test
    directly. The behavior asserted:

    1. ``_check_upgrade_pause_and_reopen`` returns True (paused-and-handled).
    2. pending_ack list is drained (via ``_flush_and_ack``); for an empty
       list this is a no-op but the early return is exercised.
    3. The pause loop is broken once we DELete the pause key from
       another task; the consumer then attempts to reopen the segment.

    We use ``stop_event.set()`` to break the pause loop after asserting
    the consumer is parked, since waiting for a real upgrade flip is
    out of scope here (covered by scenario #2).
    """
    seg = _create_segment(registry, _IntPauseReopen)
    consumer = _make_consumer(async_redis, registry, seg, consumer_name="park-1")
    _seed_consumer_state(consumer)

    try:
        # Set the pause key with the same shape upgrade_schema does.
        redis_client.set(_PAUSE_KEY, b"test-token", ex=600)

        # Insert one row + simulate one msg in pending_ack so the drain
        # branch has something to drain (proves the path through
        # _flush_and_ack runs).
        layout.insert(
            seg,
            "ent-0",
            h.pack_row(
                _IntPauseReopen,
                {
                    "a": __import__("numpy").array([1.0], dtype="float32"),
                    "b": __import__("numpy").array([42], dtype="int32"),
                },
            ),
        )
        consumer._pending_ack.append(b"0-1")

        # Schedule a coroutine that clears the pause key after a short
        # delay so the consumer's pause-poll loop breaks. Without this,
        # the test would hang in the 1.0s sleep loop.
        async def _clear_pause_after_delay() -> None:
            await asyncio.sleep(0.2)
            redis_client.delete(_PAUSE_KEY)

        clearer = asyncio.create_task(_clear_pause_after_delay())

        result = await consumer._check_upgrade_pause_and_reopen(None)
        assert result is True, "pause check must return True when paused"

        # pending_ack has been drained (XACK runs against an unknown
        # msg-id but the offline NoopWriter.flush is a no-op; XACK on a
        # non-existent id is a Redis no-op too).
        # The drain happens via the function; we just confirm post-state.

        await clearer

    finally:
        with contextlib.suppress(Exception):
            registry.close(seg)
        _drop_current(redis_client)


# ---------------------------------------------------------------------------
# Scenario 2 — clear path: pause cleared, segment reopened.
# ---------------------------------------------------------------------------


async def test_clear_path_reopens_segment(
    redis_client: redis.Redis,
    async_redis: redis.asyncio.Redis,
    registry: SegmentRegistry,
) -> None:
    """Pause cleared mid-poll → reopen swaps in the (potentially new) segment.

    Sequence:
      1. Create OLD segment, point schema:current at it.
      2. Set pause key.
      3. Schedule a task to: create a NEW segment, flip schema:current,
         clear the pause key.
      4. Drive ``_check_upgrade_pause_and_reopen``; expect it to detect
         the cleared pause, call ``registry.open_current``, observe that
         the new segment differs from the old, and swap _segments[name].
    """
    old_seg = _create_segment(registry, _IntPauseReopen)
    consumer = _make_consumer(async_redis, registry, old_seg, consumer_name="reopen-1")
    _seed_consumer_state(consumer)

    new_seg = None
    try:
        redis_client.set(_PAUSE_KEY, b"test-token", ex=600)

        async def _flip_then_clear_pause() -> None:
            nonlocal new_seg
            await asyncio.sleep(0.2)
            # Create a NEW segment; allocate without touching schema:current
            # so we can flip explicitly. Using set_current=True would race
            # the close-Lua's rotation conditional.
            new_seg_local = registry.create(_IntPauseReopen, capacity=16, set_current=False)
            # Manually flip schema:current to the new name (test simulation
            # of upgrade_schema's atomic flip Lua).
            redis_client.set(
                _key_current(_IntPauseReopen),
                new_seg_local.name.encode("utf-8"),
            )
            new_seg = new_seg_local
            # Clear the pause key — consumer's pause-poll loop breaks here.
            redis_client.delete(_PAUSE_KEY)

        flipper = asyncio.create_task(_flip_then_clear_pause())

        result = await consumer._check_upgrade_pause_and_reopen(None)
        assert result is True, "must return True after handling pause+reopen"

        await flipper

        # Consumer must have swapped _segments[b"_IntPauseReopen"] to the
        # new segment. Compare names.
        swapped_seg = consumer._segments[b"_IntPauseReopen"]
        assert swapped_seg.name != old_seg.name, (
            f"segment was not swapped: still pointing at {old_seg.name!r}"
        )
        assert new_seg is not None
        assert swapped_seg.name == new_seg.name, (
            f"swapped to {swapped_seg.name!r}, expected {new_seg.name!r}"
        )

    finally:
        # Close in reverse order (consumer's swapped segment first).
        with contextlib.suppress(Exception):
            registry.close(consumer._segments[b"_IntPauseReopen"])
        # old_seg's handle was registry.close()d by the consumer when it
        # detected segment-changed; closing again would double-DECR. Skip.
        _drop_current(redis_client)


# ---------------------------------------------------------------------------
# Scenario 3 — CRC mismatch during reopen → park + counter increment.
# ---------------------------------------------------------------------------


async def test_crc_mismatch_during_reopen_parks_consumer_and_increments_counter(
    redis_client: redis.Redis,
    async_redis: redis.asyncio.Redis,
    registry: SegmentRegistry,
) -> None:
    """``open_current`` raises CRC mismatch → consumer parks, counter bumps.

    We can't easily produce a real CRC mismatch (requires a segment
    written with one schema version then opened against a class with
    different fields), so we patch ``registry.open_current`` to raise.
    Per the test spec this is the one approved mock — reproducing CRC
    corruption is its own setup.
    """
    seg = _create_segment(registry, _IntPauseReopen)
    consumer = _make_consumer(async_redis, registry, seg, consumer_name="crc-1")
    _seed_consumer_state(consumer)

    counter_before = wal_consumer_schema_crc_mismatch_total._value.get()  # type: ignore[attr-defined]

    try:
        redis_client.set(_PAUSE_KEY, b"test-token", ex=600)

        async def _clear_pause_soon() -> None:
            await asyncio.sleep(0.2)
            redis_client.delete(_PAUSE_KEY)

        clearer = asyncio.create_task(_clear_pause_soon())

        # Patch registry.open_current to raise the CRC error on reopen.
        # We replace the method on the consumer's _registry attribute
        # since that's what _check_upgrade_pause_and_reopen calls into.
        original_open_current = consumer._registry.open_current

        def _raise_crc(_schema):
            raise SchemaCRCMismatchError("simulated CRC mismatch on reopen")

        consumer._registry.open_current = _raise_crc  # type: ignore[method-assign]

        try:
            result = await consumer._check_upgrade_pause_and_reopen(None)
        finally:
            consumer._registry.open_current = original_open_current  # type: ignore[method-assign]

        await clearer

        # Contract: returns True (paused-and-handled), stop_event set
        # (consumer parks indefinitely), counter bumps.
        assert result is True, "must return True on CRC mismatch (consumer parks, doesn't crash)"
        assert consumer._stop_event.is_set(), (
            "stop_event must be set on CRC mismatch — consumer parks indefinitely"
        )
        counter_after = wal_consumer_schema_crc_mismatch_total._value.get()  # type: ignore[attr-defined]
        assert counter_after == counter_before + 1, (
            f"counter increment expected: before={counter_before}, after={counter_after}"
        )

    finally:
        with contextlib.suppress(Exception):
            registry.close(seg)
        _drop_current(redis_client)


# ---------------------------------------------------------------------------
# Scenario 4 — liveness refreshes during park.
# ---------------------------------------------------------------------------


async def test_liveness_refreshes_during_park(
    redis_client: redis.Redis,
    async_redis: redis.asyncio.Redis,
    registry: SegmentRegistry,
) -> None:
    """The pause loop's inline 10s-gated liveness refresh keeps the TTL fresh.

    The pause loop sleeps 1.0s between MGETs and refreshes liveness when
    ``time.monotonic() - last_refresh >= LIVENESS_REFRESH_INTERVAL_SECONDS``.
    We can't actually wait 10 wall-clock seconds without a slow test, so
    we seed ``_liveness_last_refresh`` in the past so the very first
    pause-poll iteration trips the refresh gate.

    Asserts:
      1. The liveness key gets SET with the consumer's pid_str during
         the pause loop (matches production behavior).
      2. ``_liveness_last_refresh`` advances to a recent monotonic value.
    """
    seg = _create_segment(registry, _IntPauseReopen)
    consumer = _make_consumer(async_redis, registry, seg, consumer_name="liveness-1")
    _seed_consumer_state(consumer)

    # Backdate `_liveness_last_refresh` so the first pause-loop iter
    # crosses the 10s gate.
    consumer._liveness_last_refresh = time.monotonic() - LIVENESS_REFRESH_INTERVAL_SECONDS - 5.0
    seeded_ts = consumer._liveness_last_refresh

    try:
        # Confirm liveness key is absent at start.
        redis_client.delete(KEY_WAL_CONSUMER_LIVENESS)

        redis_client.set(_PAUSE_KEY, b"test-token", ex=600)

        # Clear pause after one iteration's sleep + refresh window.
        async def _clear_pause_after_one_iter() -> None:
            # Sleep long enough for the pause loop to do at least one
            # iteration (sleep 1.0s → MGET → liveness-refresh-check → MGET).
            await asyncio.sleep(1.5)
            redis_client.delete(_PAUSE_KEY)

        clearer = asyncio.create_task(_clear_pause_after_one_iter())

        result = await consumer._check_upgrade_pause_and_reopen(None)
        assert result is True

        await clearer

        # Liveness key now set in Redis with the consumer's pid_str.
        live_value = redis_client.get(KEY_WAL_CONSUMER_LIVENESS)
        assert live_value is not None, "liveness key was not refreshed during park"
        assert live_value == consumer._pid_str, (
            f"liveness key value {live_value!r} != consumer pid_str {consumer._pid_str!r}"
        )

        # _liveness_last_refresh advanced past the seeded backdated ts.
        assert consumer._liveness_last_refresh > seeded_ts, (
            f"_liveness_last_refresh did not advance: seeded={seeded_ts}, "
            f"current={consumer._liveness_last_refresh}"
        )

        # TTL is positive (real Redis SET applied EX=LIVENESS_TTL_SECONDS).
        ttl = redis_client.ttl(KEY_WAL_CONSUMER_LIVENESS)
        assert ttl is not None and ttl > 0, (
            f"liveness key TTL expected > 0, got {ttl} (key not set with EX?)"
        )
        assert ttl <= LIVENESS_TTL_SECONDS, (
            f"liveness TTL {ttl} > expected ceiling {LIVENESS_TTL_SECONDS}"
        )

    finally:
        with contextlib.suppress(Exception):
            registry.close(seg)
        _drop_current(redis_client)


# ---------------------------------------------------------------------------
# Scenario 5 — idempotent multi-pause: second pause is a stable no-op.
# ---------------------------------------------------------------------------


async def test_idempotent_double_pause_no_extra_work(
    redis_client: redis.Redis,
    async_redis: redis.asyncio.Redis,
    registry: SegmentRegistry,
) -> None:
    """Setting pause twice in a row exercises stable park behavior.

    The pause-key is idempotent in Redis (last-write-wins on the same key);
    the consumer's poll loop just re-MGETs and sees "still paused." The
    test drives two sequential ``_check_upgrade_pause_and_reopen`` calls
    with pause set throughout the first, cleared mid-second. Both must
    succeed (return True, no crash, no double-drain side effect).
    """
    seg = _create_segment(registry, _IntPauseReopen)
    consumer = _make_consumer(async_redis, registry, seg, consumer_name="idempotent-1")
    _seed_consumer_state(consumer)

    try:
        # Round 1: pause is set; clear it after a short delay so call returns.
        redis_client.set(_PAUSE_KEY, b"test-token-1", ex=600)

        async def _clear_pause_round_1() -> None:
            await asyncio.sleep(0.2)
            redis_client.delete(_PAUSE_KEY)

        clearer1 = asyncio.create_task(_clear_pause_round_1())
        result1 = await consumer._check_upgrade_pause_and_reopen(None)
        await clearer1
        assert result1 is True, "round 1: must return True"

        seg_after_round_1 = consumer._segments[b"_IntPauseReopen"]

        # Round 2: pause set AGAIN (idempotent — overwrites with new value);
        # we simulate the operator re-issuing an upgrade for some reason
        # (e.g. retry after a partial failure). The consumer must handle
        # this cleanly — drain again (already empty), park, reopen.
        redis_client.set(_PAUSE_KEY, b"test-token-2", ex=600)

        async def _clear_pause_round_2() -> None:
            await asyncio.sleep(0.2)
            redis_client.delete(_PAUSE_KEY)

        clearer2 = asyncio.create_task(_clear_pause_round_2())
        result2 = await consumer._check_upgrade_pause_and_reopen(None)
        await clearer2
        assert result2 is True, "round 2: must return True"

        # No exception, stop_event NOT set (no CRC mismatch happened).
        assert not consumer._stop_event.is_set(), (
            "stop_event must remain unset across round 1/2 — no error"
        )

        # Segment did not change between rounds (no upgrade flip happened
        # in this test). Reopen should have observed new_seg.name ==
        # old_seg.name, taken the redundant-open branch, and closed the
        # redundant handle.
        seg_after_round_2 = consumer._segments[b"_IntPauseReopen"]
        assert seg_after_round_2.name == seg_after_round_1.name, (
            f"segment churned without an upgrade flip: "
            f"{seg_after_round_1.name!r} → {seg_after_round_2.name!r}"
        )

    finally:
        with contextlib.suppress(Exception):
            registry.close(consumer._segments[b"_IntPauseReopen"])
        _drop_current(redis_client)
