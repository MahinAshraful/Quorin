"""Unit tests for quorin.watchdog (Step 14).

Tests use real Redis (via the ``redis_client`` fixture) because the Lua
scripts must execute server-side. ``psutil.Process(pid).create_time`` is
mocked per-test to drive the cross-check branches deterministically.

Tests bypass the natural N-tick wait for miss_threshold by setting
``state._tracked[pid].miss_count = state._miss_threshold`` directly
(MEDIUM-Rev3 #4 locks pattern (a)).
"""

from __future__ import annotations

import os
import sys
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import psutil
import pytest
import redis

pytestmark = pytest.mark.skipif(
    sys.platform == "win32",
    reason="watchdog requires POSIX (Linux/WSL2) — depends on posix_shm + psutil",
)


from quorin._internal import heartbeat  # noqa: E402
from quorin.watchdog import (  # noqa: E402
    WatchdogState,
    _parse_heartbeat_value,
    _PidEntry,
)

if TYPE_CHECKING:
    from collections.abc import Iterator


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_segment_in_redis(redis_client: redis.Redis, name: str, pid: int) -> None:
    """Create the Redis state ``SegmentRegistry.create`` would write
    WITHOUT actually allocating a real /dev/shm segment. The watchdog's
    cleanup Lua walks pid_segments, DECRs refcounts, etc. — it never
    touches the actual segment until Python loops over the returned
    names and calls posix_shm.unlink (which we mock in many tests).
    """
    redis_client.set(f"quorin:refcount:{name}", 1)
    redis_client.sadd(f"quorin:pid_segments:{pid}", name)
    redis_client.hset("quorin:segment_to_schema", name, "_FakeSchema")
    redis_client.set("quorin:schema:_FakeSchema:current", name)


def _heartbeat_value(create_ns: int, wall_ns: int) -> bytes:
    return f"{create_ns}:{wall_ns}".encode("ascii")


@pytest.fixture
def fake_pid() -> int:
    """A non-self PID for tracked-PID tests. The watchdog excludes its
    own PID from cleanup candidates; tests use a synthetic PID instead.
    """
    # Pick something well above the typical pid_max for safety.
    return 999_999_998


@pytest.fixture(autouse=True)
def _reset_heartbeat_state() -> Iterator[None]:
    """Each test starts fresh; heartbeat module state is process-global."""
    import contextlib

    yield
    with contextlib.suppress(Exception):
        heartbeat.stop()


# ---------------------------------------------------------------------------
# _parse_heartbeat_value
# ---------------------------------------------------------------------------


def test_parse_heartbeat_value_canonical_shape() -> None:
    create, wall = _parse_heartbeat_value(b"100:200")
    assert create == 100
    assert wall == 200


def test_parse_heartbeat_value_rejects_non_numeric() -> None:
    with pytest.raises(ValueError):
        _parse_heartbeat_value(b"not:numeric")


def test_parse_heartbeat_value_rejects_missing_separator() -> None:
    with pytest.raises(ValueError):
        _parse_heartbeat_value(b"justonecomponent")


# ---------------------------------------------------------------------------
# Liveness scenarios
# ---------------------------------------------------------------------------


def test_heartbeat_advancing_pid_not_declared_dead(
    redis_client: redis.Redis, fake_pid: int
) -> None:
    """wall_time_ns advances every tick → miss_count stays at 0."""
    state = WatchdogState(redis_client)
    redis_client.hset("quorin:heartbeats", str(fake_pid), _heartbeat_value(100, 1000))
    state.run_one_tick()  # initial track

    redis_client.hset("quorin:heartbeats", str(fake_pid), _heartbeat_value(100, 1100))
    state.run_one_tick()  # advanced

    redis_client.hset("quorin:heartbeats", str(fake_pid), _heartbeat_value(100, 1200))
    result = state.run_one_tick()

    assert state._tracked[fake_pid].miss_count == 0
    assert fake_pid not in result.dead_pids


def test_heartbeat_unchanged_psutil_dead_declares_dead(
    redis_client: redis.Redis, fake_pid: int
) -> None:
    """wall_time_ns unchanged across miss_threshold ticks AND psutil
    confirms NoSuchProcess → declare dead, run cleanup Lua, unlink.
    """
    state = WatchdogState(redis_client, miss_threshold=2)
    _make_segment_in_redis(redis_client, "quorin_test_seg_a", fake_pid)
    redis_client.hset("quorin:heartbeats", str(fake_pid), _heartbeat_value(100, 1000))
    state.run_one_tick()  # seed
    state._tracked[fake_pid].miss_count = state._miss_threshold

    with (
        patch("quorin.watchdog.psutil.Process") as mock_process,
        patch("quorin.watchdog.posix_shm.unlink") as mock_unlink,
    ):
        mock_process.side_effect = psutil.NoSuchProcess(fake_pid)
        result = state.run_one_tick()

    assert fake_pid in result.dead_pids
    # Lua queues the segment to cleanup_queue and reports the count;
    # step-4 drain in the SAME tick unlinks it (single canonical syscall site).
    assert result.segments_unlinked_dead == 1
    assert result.segments_unlinked_drain == 1
    mock_unlink.assert_called_once_with("quorin_test_seg_a")
    # Redis state cleaned by Lua.
    assert not redis_client.exists("quorin:refcount:quorin_test_seg_a")
    assert not redis_client.exists("quorin:schema:_FakeSchema:current")
    assert not redis_client.hexists("quorin:segment_to_schema", "quorin_test_seg_a")
    assert not redis_client.hexists("quorin:heartbeats", str(fake_pid))


def test_heartbeat_unchanged_psutil_alive_matching_does_not_declare_dead(
    redis_client: redis.Redis, fake_pid: int
) -> None:
    """Debugger-attach simulation: SIGSTOP'd process — heartbeat stale
    BUT psutil reports alive AND create_time matches → NOT declared dead.
    """
    state = WatchdogState(redis_client, miss_threshold=2)
    redis_client.hset("quorin:heartbeats", str(fake_pid), _heartbeat_value(100, 1000))
    state.run_one_tick()
    state._tracked[fake_pid].miss_count = state._miss_threshold

    with patch("quorin.watchdog.psutil.Process") as mock_process:
        mock_proc_instance = MagicMock()
        # create_time returns float seconds; * 1e9 → 100 (matches stored value).
        mock_proc_instance.create_time.return_value = 100 / 1e9
        mock_process.return_value = mock_proc_instance
        result = state.run_one_tick()

    assert fake_pid not in result.dead_pids
    # _tracked entry still present for next tick.
    assert fake_pid in state._tracked


def test_heartbeat_unchanged_psutil_alive_different_create_time_declares_dead(
    redis_client: redis.Redis, fake_pid: int
) -> None:
    """PID reuse: psutil reports alive but create_time differs from
    stored. Declared dead (the new process is a different PID-reuse
    instance; original is gone). MEDIUM-Rev2 #7 contract test.
    """
    state = WatchdogState(redis_client, miss_threshold=2)
    _make_segment_in_redis(redis_client, "quorin_test_seg_b", fake_pid)
    redis_client.hset("quorin:heartbeats", str(fake_pid), _heartbeat_value(100, 1000))
    state.run_one_tick()
    state._tracked[fake_pid].miss_count = state._miss_threshold

    with (
        patch("quorin.watchdog.psutil.Process") as mock_process,
        patch("quorin.watchdog.posix_shm.unlink") as mock_unlink,
    ):
        mock_proc_instance = MagicMock()
        # Different create_time: 999 ns vs stored 100 ns.
        mock_proc_instance.create_time.return_value = 999 / 1e9
        mock_process.return_value = mock_proc_instance
        result = state.run_one_tick()

    # NOTE: the dead-PID Lua's PID-reuse guard ALSO checks against
    # ARGV[2] = stored create_time = 100. heartbeats[pid] still has
    # create=100, expected=100 → guard passes, cleanup proceeds. The
    # watchdog's own cross-check declared dead via different create
    # in psutil — same conclusion.
    assert fake_pid in result.dead_pids
    assert result.segments_unlinked_dead == 1
    assert result.segments_unlinked_drain == 1
    mock_unlink.assert_called_once_with("quorin_test_seg_b")


# ---------------------------------------------------------------------------
# AccessDenied / ZombieProcess — HIGH-Rev3 #1 conservative branches
# ---------------------------------------------------------------------------


def test_psutil_access_denied_does_not_declare_dead(
    redis_client: redis.Redis, fake_pid: int
) -> None:
    """HIGH-Rev3 #1: AccessDenied (cross-UID without CAP_SYS_PTRACE) is
    conservative — do NOT declare dead. Counter increments.
    """
    state = WatchdogState(redis_client, miss_threshold=2)
    redis_client.hset("quorin:heartbeats", str(fake_pid), _heartbeat_value(100, 1000))
    state.run_one_tick()
    state._tracked[fake_pid].miss_count = state._miss_threshold

    with patch("quorin.watchdog.psutil.Process") as mock_process:
        mock_process.side_effect = psutil.AccessDenied(fake_pid)
        result = state.run_one_tick()

    assert fake_pid not in result.dead_pids
    assert result.unverifiable_count == 1
    # _tracked entry stays so next tick re-checks.
    assert fake_pid in state._tracked


def test_psutil_zombie_process_does_not_declare_dead(
    redis_client: redis.Redis, fake_pid: int
) -> None:
    """HIGH-Rev3 #1: ZombieProcess (un-reaped zombie) is conservative —
    do NOT declare dead. Counter increments under the 'zombie' reason.
    """
    state = WatchdogState(redis_client, miss_threshold=2)
    redis_client.hset("quorin:heartbeats", str(fake_pid), _heartbeat_value(100, 1000))
    state.run_one_tick()
    state._tracked[fake_pid].miss_count = state._miss_threshold

    with patch("quorin.watchdog.psutil.Process") as mock_process:
        mock_process.side_effect = psutil.ZombieProcess(fake_pid)
        result = state.run_one_tick()

    assert fake_pid not in result.dead_pids
    assert result.unverifiable_count == 1


# ---------------------------------------------------------------------------
# Backward jumps + malformed input
# ---------------------------------------------------------------------------


def test_heartbeat_backward_jump_resets_miss_count(
    redis_client: redis.Redis, fake_pid: int
) -> None:
    """NTP correction: wall_time_ns goes backward. ``!=`` check resets
    miss_count + updates last_seen. NOT declared dead.
    """
    state = WatchdogState(redis_client, miss_threshold=2)
    redis_client.hset("quorin:heartbeats", str(fake_pid), _heartbeat_value(100, 5000))
    state.run_one_tick()
    state._tracked[fake_pid].miss_count = 1

    # Backward jump (wall went from 5000 -> 3000)
    redis_client.hset("quorin:heartbeats", str(fake_pid), _heartbeat_value(100, 3000))
    state.run_one_tick()

    assert state._tracked[fake_pid].miss_count == 0
    assert state._tracked[fake_pid].last_wall_time_ns == 3000


def test_malformed_heartbeat_value_skipped_with_counter(
    redis_client: redis.Redis, fake_pid: int
) -> None:
    """MEDIUM-Rev3 #2: malformed heartbeat value is logged + counter
    incremented + skipped. Tick proceeds with valid entries.
    """
    state = WatchdogState(redis_client, miss_threshold=2)
    redis_client.hset("quorin:heartbeats", str(fake_pid), b"garbage_no_colon")
    redis_client.hset("quorin:heartbeats", str(fake_pid + 1), _heartbeat_value(100, 1000))

    result = state.run_one_tick()

    assert result.malformed_count == 1
    # Valid pid still tracked.
    assert fake_pid + 1 in state._tracked
    # Bad pid not tracked.
    assert fake_pid not in state._tracked


def test_malformed_heartbeat_only_one_component(redis_client: redis.Redis, fake_pid: int) -> None:
    """Heartbeat value missing the ``:`` separator is malformed."""
    state = WatchdogState(redis_client)
    redis_client.hset("quorin:heartbeats", str(fake_pid), b"only_one")

    result = state.run_one_tick()

    assert result.malformed_count == 1
    assert fake_pid not in state._tracked


# ---------------------------------------------------------------------------
# Cleanup queue drain
# ---------------------------------------------------------------------------


def test_cleanup_queue_drain_unlinks_each(redis_client: redis.Redis) -> None:
    """Drain SPOPs up to batch_drain_size names from cleanup_queue and
    posix_shm.unlinks each.
    """
    state = WatchdogState(redis_client, batch_drain_size=10)
    redis_client.sadd("quorin:cleanup_queue", "quorin_q_a", "quorin_q_b")

    with patch("quorin.watchdog.posix_shm.unlink") as mock_unlink:
        result = state.run_one_tick()

    assert result.segments_unlinked_drain == 2
    # Set behavior: SPOP is non-deterministic, but both names must be unlinked.
    unlinked = {call.args[0] for call in mock_unlink.call_args_list}
    assert unlinked == {"quorin_q_a", "quorin_q_b"}


def test_cleanup_queue_drain_handles_filenotfound(redis_client: redis.Redis) -> None:
    """POLISH-Rev3 #6: ``posix_shm.unlink`` raising FileNotFoundError
    is debug-logged + treated as success (race with another watchdog
    or test fixture). Loop continues.
    """
    state = WatchdogState(redis_client, batch_drain_size=10)
    redis_client.sadd("quorin:cleanup_queue", "quorin_q_a", "quorin_q_b")

    with patch("quorin.watchdog.posix_shm.unlink") as mock_unlink:
        # First call raises FileNotFoundError; second succeeds.
        mock_unlink.side_effect = [FileNotFoundError(), None]
        result = state.run_one_tick()

    # Both calls happened; success counter increments only for the
    # successful one.
    assert mock_unlink.call_count == 2
    assert result.segments_unlinked_drain == 1


def test_cleanup_queue_drain_propagates_other_oserror(
    redis_client: redis.Redis,
) -> None:
    """POLISH-Rev3 #6: unlink raising OTHER OSError (e.g.
    PermissionError) propagates. Operator must see this.
    """
    state = WatchdogState(redis_client, batch_drain_size=10)
    redis_client.sadd("quorin:cleanup_queue", "quorin_q_a")

    with patch("quorin.watchdog.posix_shm.unlink") as mock_unlink:
        mock_unlink.side_effect = PermissionError(13, "Permission denied")
        with pytest.raises(PermissionError):
            state.run_one_tick()


# ---------------------------------------------------------------------------
# Self-PID exclusion
# ---------------------------------------------------------------------------


def test_self_pid_never_declared_dead(redis_client: redis.Redis) -> None:
    """The watchdog's own PID is never tracked as a dead-PID candidate.

    The construction of ``WatchdogState`` calls ``heartbeat.ensure_started``
    for self_pid → heartbeat hash gets a real entry. Even with synthetic
    miss_count, the watchdog must not declare its own pid dead.
    """
    state = WatchdogState(redis_client)
    # Force a stale heartbeat for self_pid.
    redis_client.hset(
        "quorin:heartbeats",
        str(state._self_pid),
        _heartbeat_value(123, 456),
    )

    # Run several ticks where the value never changes — watchdog should
    # never start tracking self_pid in self._tracked.
    for _ in range(state._miss_threshold + 2):
        state.run_one_tick()

    assert state._self_pid not in state._tracked


# ---------------------------------------------------------------------------
# _tracked pruning + stale entries
# ---------------------------------------------------------------------------


def test_tracked_prunes_pids_no_longer_in_heartbeat_hash(
    redis_client: redis.Redis, fake_pid: int
) -> None:
    """POLISH-Rev3 #7: external HDEL (operator manual cleanup) leaves
    a tracked entry stranded; step 5 prunes it next tick.
    """
    state = WatchdogState(redis_client)
    redis_client.hset("quorin:heartbeats", str(fake_pid), _heartbeat_value(100, 1000))
    state.run_one_tick()
    assert fake_pid in state._tracked

    redis_client.hdel("quorin:heartbeats", str(fake_pid))
    state.run_one_tick()
    assert fake_pid not in state._tracked


# ---------------------------------------------------------------------------
# WatchdogState construction calls heartbeat.ensure_started
# ---------------------------------------------------------------------------


def test_watchdog_state_construction_calls_heartbeat_ensure_started(
    redis_client: redis.Redis,
) -> None:
    """POLISH-Rev3 #12: ensure_started moved to __init__; called once
    at construction, not per-tick.
    """
    with patch("quorin.watchdog.heartbeat.ensure_started") as mock_ensure:
        WatchdogState(redis_client)

    mock_ensure.assert_called_once()
    # First positional arg is the redis client; second is the self_pid.
    args = mock_ensure.call_args[0]
    assert args[0] is redis_client
    assert args[1] == os.getpid()


# ---------------------------------------------------------------------------
# HIGH-Rev4 #1: PID-reuse race in cleanup window
# ---------------------------------------------------------------------------


def test_pid_reuse_race_aborts_cleanup_via_lua_guard(
    redis_client: redis.Redis, fake_pid: int
) -> None:
    """HIGH-Rev4 #1: cleanup Lua's PID-reuse guard returns -1 when
    heartbeats[pid].create_ns differs from ARGV[2].

    The race window is ~1 ms between the watchdog's psutil cross-check
    (NoSuchProcess) and the Lua execution. We can't reproduce the
    timing from a single-threaded test, so we test the Lua's guard
    directly: populate Redis with B's state (create_time=999), invoke
    the cleanup Lua with the watchdog's expected create_time=100 (A's
    cached value pre-race). Lua must return -1 + leave B's state
    intact.
    """
    state = WatchdogState(redis_client)
    # B's state — segment refcounted, pid_segments populated, sidetable.
    _make_segment_in_redis(redis_client, "quorin_test_seg_b_alive", fake_pid)
    redis_client.hset("quorin:heartbeats", str(fake_pid), _heartbeat_value(999, 9999))

    # Watchdog's _tracked thinks A's create_time was 100.
    expected_create_time_ns_for_a = 100

    # Direct Lua call with mismatched expected create_time.
    returned = int(
        state._cleanup_lua(
            keys=[
                f"quorin:pid_segments:{fake_pid}",
                "quorin:heartbeats",
                "quorin:cleanup_queue",
                "quorin:segment_to_schema",
            ],
            args=[str(fake_pid), str(expected_create_time_ns_for_a)],
        )
    )

    # PID-reuse detected → -1 sentinel.
    assert returned == -1
    # B's state preserved across the abort.
    assert redis_client.exists("quorin:refcount:quorin_test_seg_b_alive")
    assert redis_client.hexists("quorin:segment_to_schema", "quorin_test_seg_b_alive")
    assert redis_client.exists("quorin:schema:_FakeSchema:current")
    # Cleanup queue not populated by aborted Lua.
    assert not redis_client.sismember("quorin:cleanup_queue", "quorin_test_seg_b_alive")
    # heartbeats entry preserved.
    assert redis_client.hexists("quorin:heartbeats", str(fake_pid))


def test_pid_reuse_guard_falls_through_when_heartbeats_absent(
    redis_client: redis.Redis, fake_pid: int
) -> None:
    """When heartbeats[pid] is nil (A's atexit HDEL'd, B never wrote),
    the Lua falls through and proceeds with cleanup. This is the
    documented residual-risk path from ADR-013 — combined with B's
    force-first-refresh failure, ~3e-9 probability per dead PID.
    """
    state = WatchdogState(redis_client)
    _make_segment_in_redis(redis_client, "quorin_residual_a", fake_pid)
    # NO heartbeats[pid] entry — A's atexit HDEL'd before any reuse.
    assert not redis_client.hexists("quorin:heartbeats", str(fake_pid))

    returned = int(
        state._cleanup_lua(
            keys=[
                f"quorin:pid_segments:{fake_pid}",
                "quorin:heartbeats",
                "quorin:cleanup_queue",
                "quorin:segment_to_schema",
            ],
            args=[str(fake_pid), "100"],
        )
    )

    # Cleanup proceeded (residual risk path).
    assert returned == 1
    assert redis_client.sismember("quorin:cleanup_queue", "quorin_residual_a")


# ---------------------------------------------------------------------------
# _PidEntry sanity
# ---------------------------------------------------------------------------


def test_pid_entry_initial_miss_count_is_zero() -> None:
    entry = _PidEntry(last_wall_time_ns=100, create_time_ns=50)
    assert entry.miss_count == 0
    assert entry.last_wall_time_ns == 100
    assert entry.create_time_ns == 50


# ---------------------------------------------------------------------------
# Drain + cleanup combined: integration of step 3 + step 4 of run_one_tick
# ---------------------------------------------------------------------------


def test_dead_pid_cleanup_queues_segment_then_drain_unlinks(
    redis_client: redis.Redis, fake_pid: int
) -> None:
    """End-to-end: dead PID has refcount-1 segment → cleanup Lua DECRs
    to 0 + queues for cleanup_queue (Lua returns the count). Step 4 of
    the SAME tick drains the queue and posix_shm.unlinks each.

    Single canonical posix_shm.unlink call site (step-4 drain) — the
    Lua does not unlink directly. Counts in TickResult:
    ``segments_unlinked_dead`` is the Lua's reported count;
    ``segments_unlinked_drain`` is what step 4 actually unlinked.
    """
    state = WatchdogState(redis_client, miss_threshold=2)
    _make_segment_in_redis(redis_client, "quorin_combined_a", fake_pid)
    redis_client.hset("quorin:heartbeats", str(fake_pid), _heartbeat_value(100, 1000))
    state.run_one_tick()
    state._tracked[fake_pid].miss_count = state._miss_threshold

    with (
        patch("quorin.watchdog.psutil.Process") as mock_process,
        patch("quorin.watchdog.posix_shm.unlink") as mock_unlink,
    ):
        mock_process.side_effect = psutil.NoSuchProcess(fake_pid)
        result = state.run_one_tick()

    # Lua queued 1 segment.
    assert result.segments_unlinked_dead == 1
    # Step-4 drain unlinked exactly that 1 segment.
    assert result.segments_unlinked_drain == 1
    mock_unlink.assert_called_once_with("quorin_combined_a")
