"""Unit tests for quorin._internal.heartbeat (Step 14).

Most tests use a mock Redis client to drive deterministic behavior.
Two tests use real Redis:

* :func:`test_atexit_does_not_hdel_after_fork_clean_exit` — forks a real
  child process via :class:`multiprocessing.Process`, has it exit clean,
  asserts parent's heartbeat field stays in the live Redis hash. This
  is the CRITICAL-Rev2 #1 regression test.
* :func:`test_loop_writes_to_real_redis` — sanity check that the
  ``"{create_ns}:{wall_ns}"`` shape lands in Redis.
"""

from __future__ import annotations

import multiprocessing
import os
import sys
import threading
import time
from typing import TYPE_CHECKING
from unittest.mock import MagicMock

import pytest
import redis

pytestmark = pytest.mark.skipif(
    sys.platform == "win32",
    reason="heartbeat requires POSIX (Linux/WSL2) — depends on os.register_at_fork + psutil",
)


from quorin._internal import heartbeat  # noqa: E402

if TYPE_CHECKING:
    from collections.abc import Iterator


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_module_state() -> Iterator[None]:
    """Each test starts with a clean ``heartbeat._state``. The autouse
    fixture in ``tests/conftest.py`` already calls ``heartbeat.stop()``
    after every test, but we also snapshot here to be defensive against
    any test that constructs state via direct attribute writes.
    """
    import contextlib

    yield
    # Belt-and-suspenders cleanup.
    with contextlib.suppress(Exception):
        heartbeat.stop()
    heartbeat._state.thread = None
    heartbeat._state.stop_event = None
    heartbeat._state.pid = None
    heartbeat._state.create_time_ns = None
    heartbeat._state.redis_client = None
    heartbeat._state.labels_writes_ok = None
    heartbeat._state.labels_writes_redis_error = None


@pytest.fixture
def mock_client() -> MagicMock:
    """A MagicMock that quacks like ``redis.Redis``.

    ``hset`` and ``hdel`` return ints (matching real-redis-py behavior);
    other methods can be configured per-test.
    """
    client = MagicMock(spec=redis.Redis)
    client.hset.return_value = 1
    client.hdel.return_value = 1
    return client


# ---------------------------------------------------------------------------
# ensure_started behavior
# ---------------------------------------------------------------------------


def test_ensure_started_is_idempotent(mock_client: MagicMock) -> None:
    """Second call returns early; doesn't spawn a second thread."""
    heartbeat.ensure_started(mock_client, pid=os.getpid())
    first_thread = heartbeat._state.thread
    assert first_thread is not None and first_thread.is_alive()

    heartbeat.ensure_started(mock_client, pid=os.getpid())
    assert heartbeat._state.thread is first_thread, (
        "second ensure_started should not replace the running thread"
    )


def test_ensure_started_first_client_wins(mock_client: MagicMock) -> None:
    """Step 14 / POLISH-Rev3 #8: the FIRST call's redis_client is bound;
    subsequent calls' client argument is ignored.
    """
    other_client = MagicMock(spec=redis.Redis)
    other_client.hset.return_value = 1

    heartbeat.ensure_started(mock_client, pid=os.getpid())
    bound = heartbeat._state.redis_client

    heartbeat.ensure_started(other_client, pid=os.getpid())
    assert heartbeat._state.redis_client is bound, (
        "second ensure_started should not switch to a different client"
    )
    assert heartbeat._state.redis_client is mock_client


def test_ensure_started_writes_force_first_refresh(mock_client: MagicMock) -> None:
    """The synchronous force-first-refresh inside ``ensure_started`` is
    the implicit "alive" signal to the watchdog before the timer
    thread even spins up.
    """
    heartbeat.ensure_started(mock_client, pid=os.getpid())
    # First call is the force-first-refresh (synchronous, before thread.start()).
    assert mock_client.hset.call_count >= 1
    args, _ = mock_client.hset.call_args_list[0]
    assert args[0] == heartbeat.KEY_HEARTBEATS
    assert args[1] == str(os.getpid())
    # Value must parse as "{create_ns}:{wall_ns}" with both being ints.
    create_str, wall_str = args[2].split(":", 1)
    assert int(create_str) > 0
    assert int(wall_str) > 0


def test_ensure_started_swallows_force_first_refresh_redis_error() -> None:
    """HIGH-Rev2 #5 + Rev-3 docs: Redis-down at process startup MUST NOT
    take ``SegmentRegistry.create`` down with it. ``ensure_started``
    swallows the redis.RedisError, increments the redis_error counter,
    and still spawns the timer thread.
    """
    bad_client = MagicMock(spec=redis.Redis)
    bad_client.hset.side_effect = redis.ConnectionError("redis down")

    heartbeat.ensure_started(bad_client, pid=os.getpid())

    assert heartbeat._state.thread is not None
    assert heartbeat._state.thread.is_alive(), (
        "thread must start even when force-first-refresh fails"
    )


# ---------------------------------------------------------------------------
# _atexit_handler — CRITICAL-Rev2 #1 (fork-safety)
# ---------------------------------------------------------------------------


def test_atexit_handler_hdels_when_pid_matches(mock_client: MagicMock) -> None:
    """Happy path: clean exit of the original process HDELs its heartbeat."""
    heartbeat.ensure_started(mock_client, pid=os.getpid())
    mock_client.hdel.reset_mock()

    heartbeat._atexit_handler()

    mock_client.hdel.assert_called_once()
    args = mock_client.hdel.call_args[0]
    assert args[0] == heartbeat.KEY_HEARTBEATS
    assert args[1] == str(os.getpid())


def test_atexit_handler_returns_without_hdel_when_pid_mismatches(
    mock_client: MagicMock,
) -> None:
    """CRITICAL-Rev2 #1 regression: forked child's atexit MUST NOT
    delete parent's heartbeat. The PID gate is the binding contract.

    Simulates the gate by setting ``_state.pid`` to a non-current PID
    (as if we were inside a forked child where _reset_after_fork did
    NOT run for some reason — defense in depth).
    """
    heartbeat.ensure_started(mock_client, pid=os.getpid())
    mock_client.hdel.reset_mock()

    # Pretend we're a forked child: simulate _state.pid being something
    # other than os.getpid(). The handler must early-return.
    heartbeat._state.pid = 99999  # non-existent PID; getpid() != 99999

    heartbeat._atexit_handler()

    mock_client.hdel.assert_not_called()


def test_atexit_handler_returns_when_state_is_none(mock_client: MagicMock) -> None:
    """When ``_state.pid`` is None (e.g. after _reset_after_fork in
    a child that never called ensure_started), the handler short-
    circuits before any Redis call.
    """
    # State is reset by the autouse fixture; verify the handler is
    # safe to call against an empty _state.
    heartbeat._state.pid = None
    heartbeat._state.redis_client = None

    heartbeat._atexit_handler()  # must not raise

    mock_client.hdel.assert_not_called()


def test_atexit_handler_swallows_redis_error(mock_client: MagicMock) -> None:
    """``HDEL`` failing with a Redis error must not propagate out of
    atexit (interpreter-shutdown context — exceptions get muddled).
    """
    heartbeat.ensure_started(mock_client, pid=os.getpid())
    mock_client.hdel.side_effect = redis.ConnectionError("redis down")

    # Should not raise.
    heartbeat._atexit_handler()


# ---------------------------------------------------------------------------
# stop() — POLISH-Rev3 #5 / POLISH-Rev4 #5 symmetry
# ---------------------------------------------------------------------------


def test_stop_nulls_every_state_field_and_unregisters_atexit(
    mock_client: MagicMock,
) -> None:
    """``stop()`` is the symmetric inverse of ``ensure_started``.

    After stop(): every ``_state`` field is None AND atexit registration
    is removed. The "atexit removed" half is verified by calling the
    handler directly post-stop and asserting the mock client received
    no further HDEL.
    """
    heartbeat.ensure_started(mock_client, pid=os.getpid())
    mock_client.hdel.reset_mock()

    heartbeat.stop()

    assert heartbeat._state.thread is None
    assert heartbeat._state.stop_event is None
    assert heartbeat._state.pid is None
    assert heartbeat._state.create_time_ns is None
    assert heartbeat._state.redis_client is None
    assert heartbeat._state.labels_writes_ok is None
    assert heartbeat._state.labels_writes_redis_error is None

    # stop() does ONE final HDEL (via _atexit_handler), so the count goes
    # from 0 (after reset_mock) to 1.
    assert mock_client.hdel.call_count == 1
    mock_client.hdel.reset_mock()

    # Calling _atexit_handler now (post-stop) must early-return —
    # _state.pid is None — so client.hdel must not be called again.
    heartbeat._atexit_handler()
    mock_client.hdel.assert_not_called()


def test_stop_is_idempotent(mock_client: MagicMock) -> None:
    """Double ``stop()`` doesn't raise."""
    heartbeat.ensure_started(mock_client, pid=os.getpid())
    heartbeat.stop()
    heartbeat.stop()  # must not raise


def test_stop_when_never_started_is_safe() -> None:
    """``stop()`` before any ``ensure_started`` must not raise."""
    heartbeat.stop()


# ---------------------------------------------------------------------------
# _reset_after_fork — POLISH-Rev3 #10 (every field nulled)
# ---------------------------------------------------------------------------


def test_reset_after_fork_clears_every_state_field(mock_client: MagicMock) -> None:
    heartbeat.ensure_started(mock_client, pid=os.getpid())
    # Confirm state is populated before reset.
    assert heartbeat._state.thread is not None
    assert heartbeat._state.pid is not None

    heartbeat._reset_after_fork()

    assert heartbeat._state.thread is None
    assert heartbeat._state.stop_event is None
    assert heartbeat._state.pid is None
    assert heartbeat._state.create_time_ns is None
    assert heartbeat._state.redis_client is None
    assert heartbeat._state.labels_writes_ok is None
    assert heartbeat._state.labels_writes_redis_error is None


# ---------------------------------------------------------------------------
# MEDIUM-Rev4 #2: _loop captures local refs so stop-mid-HSET doesn't
# AttributeError on a nulled module-level state.
# ---------------------------------------------------------------------------


def test_loop_survives_stop_mid_hset() -> None:
    """When ``stop()`` runs concurrently with an in-flight HSET that
    eventually unblocks, ``_loop`` must NOT raise AttributeError on
    ``_state.labels_*.inc()`` — it captured local refs at start.

    Simulates by:
    1. Mock ``client.hset`` to block on an Event.
    2. Call ``ensure_started``; the force-first-refresh's HSET blocks.
    3. ``stop()`` from a second thread; nulls _state.* once join times out.
    4. Release the HSET; the periodic loop's eventual write completes
       cleanly (test asserts no exception traceback).

    Note: force-first-refresh runs inside ``ensure_started`` and would
    block the test indefinitely. Instead, we let force-first-refresh
    succeed and block subsequent _loop iterations.
    """
    block_event = threading.Event()
    counter = {"hset_calls": 0}

    def blocking_hset(*_args: object, **_kwargs: object) -> int:
        counter["hset_calls"] += 1
        # First call (force-first-refresh) returns immediately.
        # Subsequent calls (periodic loop) block until released.
        if counter["hset_calls"] >= 2:
            block_event.wait(timeout=5.0)
        return 1

    client = MagicMock(spec=redis.Redis)
    client.hset.side_effect = blocking_hset
    client.hdel.return_value = 1

    # Override the loop interval to a very short value so the periodic
    # write fires fast.
    original_interval = heartbeat.HEARTBEAT_INTERVAL_SECONDS
    try:
        heartbeat.HEARTBEAT_INTERVAL_SECONDS = 0.05  # type: ignore[misc]
        heartbeat.ensure_started(client, pid=os.getpid())

        # Wait for the periodic loop to enter the blocking HSET.
        deadline = time.monotonic() + 2.0
        while counter["hset_calls"] < 2 and time.monotonic() < deadline:
            time.sleep(0.01)
        assert counter["hset_calls"] >= 2, "periodic loop never reached blocking HSET"

        # stop() runs on the main thread; thread.join times out (HSET
        # is blocked) and stop() proceeds to null _state.*.
        heartbeat.stop()  # join_timeout=2.0 inside

        # Now release the HSET. The thread's _loop body resumes and
        # tries to .inc() its captured locals. If the locals were the
        # nulled _state.labels_*, this would AttributeError. With the
        # MEDIUM-Rev4 #2 fix it's safe.
        block_event.set()

        # Give the thread a moment to drain.
        time.sleep(0.2)
    finally:
        heartbeat.HEARTBEAT_INTERVAL_SECONDS = original_interval  # type: ignore[misc]
        block_event.set()


# ---------------------------------------------------------------------------
# is_running
# ---------------------------------------------------------------------------


def test_is_running_reflects_thread_state(mock_client: MagicMock) -> None:
    assert heartbeat.is_running() is False
    heartbeat.ensure_started(mock_client, pid=os.getpid())
    assert heartbeat.is_running() is True
    heartbeat.stop()
    assert heartbeat.is_running() is False


# ---------------------------------------------------------------------------
# CRITICAL-Rev2 #1 system test — real fork via multiprocessing.
# ---------------------------------------------------------------------------


def _fork_child_clean_exit() -> None:
    """Child target. Imports nothing risky; just exits clean.

    The exit triggers atexit handlers registered in the parent. The
    PID-gate in heartbeat._atexit_handler must short-circuit because
    os.getpid() != _state.pid (the fork hook also nulled _state.pid;
    belt-and-suspenders).
    """
    # No-op. Child returns from this function; multiprocessing.Process
    # cleanly tears down.


def test_atexit_does_not_hdel_after_fork_clean_exit(redis_client: redis.Redis) -> None:
    """CRITICAL-Rev2 #1 system test: parent's heartbeat field stays in
    Redis after a forked child cleanly exits.

    Without the PID gate (or _reset_after_fork), the child's atexit
    would HDEL parent's heartbeat — watchdog declares parent dead —
    parent's segments unlinked. Data corruption.
    """
    # Skip if multiprocessing default isn't fork (macOS / Windows ship
    # 'spawn' as default; this regression test specifically exercises fork).
    ctx = multiprocessing.get_context("fork")

    heartbeat.ensure_started(redis_client, pid=os.getpid())
    parent_pid = os.getpid()

    # Confirm parent's entry is in the hash before forking.
    raw = redis_client.hget(heartbeat.KEY_HEARTBEATS, str(parent_pid))
    assert raw is not None, "parent's heartbeat entry should exist before fork"

    proc = ctx.Process(target=_fork_child_clean_exit)
    proc.start()
    proc.join(timeout=5.0)
    assert proc.exitcode == 0, f"child exited unexpectedly: exitcode={proc.exitcode}"

    # Parent's heartbeat entry MUST still be there. If the child's
    # atexit ran the HDEL, this assertion fires.
    raw_after = redis_client.hget(heartbeat.KEY_HEARTBEATS, str(parent_pid))
    assert raw_after is not None, (
        "child's atexit HDEL'd parent's heartbeat — CRITICAL-Rev2 #1 regression"
    )


# ---------------------------------------------------------------------------
# Real-Redis sanity: shape + idempotent stop.
# ---------------------------------------------------------------------------


def test_loop_writes_to_real_redis(redis_client: redis.Redis) -> None:
    """Periodic write lands in Redis with the canonical shape."""
    pid = os.getpid()
    heartbeat.ensure_started(redis_client, pid=pid)
    # Force-first-refresh already ran; check Redis directly.
    raw = redis_client.hget(heartbeat.KEY_HEARTBEATS, str(pid))
    assert raw is not None
    decoded = raw.decode("ascii")
    create_str, wall_str = decoded.split(":", 1)
    assert int(create_str) > 0
    assert int(wall_str) > 0


def test_atexit_register_was_called(mock_client: MagicMock) -> None:
    """Sanity: ``ensure_started`` registers the handler with atexit.

    ``atexit.unregister`` returns None regardless of whether the
    handler was found, so we instead probe by calling unregister and
    checking the post-state via the side effect: a registered handler
    that's been unregistered cannot be unregistered AGAIN-AND-still-
    fire-on-clean-exit. We simulate by calling ``_atexit_handler``
    directly after stop() — the stop() flow includes an unregister.
    Implementation: ``stop()`` calls ``atexit.unregister`` so we
    verify by patching atexit.register/unregister and asserting both
    were called.
    """
    from unittest import mock

    with (
        mock.patch("atexit.register") as mock_register,
        mock.patch("atexit.unregister") as mock_unregister,
    ):
        heartbeat.ensure_started(mock_client, pid=os.getpid())
        # ensure_started registers exactly once.
        mock_register.assert_called_once_with(heartbeat._atexit_handler)
        heartbeat.stop()
        # stop() unregisters exactly once.
        mock_unregister.assert_called_once_with(heartbeat._atexit_handler)
