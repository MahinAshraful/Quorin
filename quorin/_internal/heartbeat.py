"""Per-process heartbeat producer (Step 14).

Each library process that opens a Quorin segment runs a single daemon thread
that ``HSET quorin:heartbeats {pid} "{create_time_ns}:{wall_time_ns}"``
every :data:`HEARTBEAT_INTERVAL_SECONDS`. The watchdog (separate process,
:mod:`quorin.watchdog`) reads this hash and detects dead PIDs via
consecutive missed advances of ``wall_time_ns``.

The thread mirrors :mod:`quorin._internal.gc_manager`'s daemon-thread
pattern (CLAUDE.md invariant #14): daemon=True + ``threading.Event``
shutdown + ``os.register_at_fork(after_in_child=...)`` reset.

CRITICAL fork-safety note (Rev-2 of the Step 14 plan)
=====================================================

Python's ``atexit`` registrations are inherited by forked children.
Without a guard, the standard production deployment shape (gunicorn /
uvicorn fork workers) would have child workers' clean exits delete the
parent's heartbeat — the watchdog would then declare the live parent
dead 150 s later and unlink its segments. Data corruption.

Two-layer defense:

1. :func:`_atexit_handler` PID-gates: ``if os.getpid() != _state.pid: return``.
2. :func:`_reset_after_fork` clears every ``_state`` field to None so a
   child reading state-without-init also fails the gate.

Belt-and-suspenders. See
:func:`tests.unit.test_heartbeat.test_atexit_does_not_hdel_after_fork`
for the regression test.
"""

from __future__ import annotations

import atexit
import contextlib
import os
import threading
import time
from typing import TYPE_CHECKING, Final

import psutil  # type: ignore[import-untyped]
import redis as _redis

from quorin._internal.redis_compat import warn_if_no_socket_timeout
from quorin.logging import get_logger
from quorin.metrics import heartbeat_age_seconds, heartbeat_writes_total

if TYPE_CHECKING:
    import redis as redis_lib

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants. Locked alongside Step 14's plan; matches WAL-consumer liveness.
# ---------------------------------------------------------------------------

KEY_HEARTBEATS: Final[str] = "quorin:heartbeats"
HEARTBEAT_INTERVAL_SECONDS: Final[float] = 10.0


# ---------------------------------------------------------------------------
# Module state. Process-global; only one heartbeat thread per process.
# ---------------------------------------------------------------------------


class _State:
    __slots__ = (
        "create_time_ns",
        "labels_writes_ok",
        "labels_writes_redis_error",
        "pid",
        "redis_client",
        "stop_event",
        "thread",
    )

    def __init__(self) -> None:
        self.thread: threading.Thread | None = None
        self.stop_event: threading.Event | None = None
        self.pid: int | None = None
        self.create_time_ns: int | None = None
        self.redis_client: redis_lib.Redis | None = None
        # Pre-warmed prometheus label children. Populated at ensure_started.
        self.labels_writes_ok: object | None = None
        self.labels_writes_redis_error: object | None = None


_state = _State()
# Module-level lock guarding _state mutation. CR.B.10 (v0.1.1): if a
# parent thread is mid-``ensure_started``'s ``with _state_lock:`` when
# fork() runs, the child inherits a held lock that will never be
# released — child's first ensure_started deadlocks. _reset_after_fork
# replaces this object with a fresh Lock(). Module-level rebinding is
# safe in the fork-hook (fork hooks run before child code resumes).
_state_lock = threading.Lock()


# ---------------------------------------------------------------------------
# Public API.
# ---------------------------------------------------------------------------


def ensure_started(redis_client: redis_lib.Redis, pid: int | None = None) -> None:
    """Idempotent. First call starts the heartbeat thread for this process.

    First-call-wins: the FIRST call's ``redis_client`` is bound for the
    process lifetime; subsequent calls' ``redis_client`` argument is
    ignored. To switch clients, call :func:`stop` first.

    The caller's ``redis_client`` SHOULD have ``socket_timeout`` set to a
    finite value, ideally <= :data:`HEARTBEAT_INTERVAL_SECONDS`. Without
    it, the heartbeat thread can block indefinitely on a network partition
    and :func:`stop` will leak the thread because the join timeout fires
    before the blocked HSET returns. Project-wide concern (the WAL
    producer / consumer have it too); flagged for an invariant doc in
    Step 17.

    On Redis-unreachable at startup: force-first-refresh failure is
    swallowed (logged + counter incremented), the timer thread starts
    anyway. The first successful periodic write (10 s later) is the
    implicit "alive" signal to the watchdog.
    """
    # CR.E.6 (v0.1.1): warn if the caller's redis_client lacks a finite
    # socket_timeout. Heartbeat HSET on a partitioned client can block
    # indefinitely; stop()'s 2s join will time out and leak the thread.
    warn_if_no_socket_timeout(redis_client, source="heartbeat.ensure_started")
    with _state_lock:
        if _state.thread is not None and _state.thread.is_alive():
            return
        pid = pid if pid is not None else os.getpid()
        # Bit-deterministic on the same kernel; psutil reads /proc/{pid}/stat
        # start_time field which is in jiffies. int(float * 1e9) of the same
        # float is bit-identical. PID reuse always differs by >=1 jiffy.
        create_time_ns = int(psutil.Process(pid).create_time() * 1e9)
        # Pre-warm prometheus label children. Same lesson as Step 7's GC
        # callback (ADR-006): allocation inside a callback running under
        # GC pressure can re-enter; allocation here is on the cold path
        # and stays out of the heartbeat hot loop.
        _state.labels_writes_ok = heartbeat_writes_total.labels(outcome="ok")
        _state.labels_writes_redis_error = heartbeat_writes_total.labels(outcome="redis_error")
        # Local-only init, no Redis dependency. Forces the gauge to 0 so a
        # Prometheus scrape between thread start and first successful write
        # doesn't read process-uptime as "age" (same WAL-consumer footgun
        # documented in CLAUDE.md §8).
        heartbeat_age_seconds.set(0.0)
        # Force-first-refresh: swallow Redis errors so transient blips at
        # process startup don't take SegmentRegistry.create down with them.
        try:
            _write_heartbeat(redis_client, pid, create_time_ns)
            _state.labels_writes_ok.inc()
        except _redis.RedisError:
            logger.warning(
                "heartbeat_first_write_failed_redis_unreachable",
                pid=pid,
            )
            _state.labels_writes_redis_error.inc()
        stop_event = threading.Event()
        thread = threading.Thread(
            target=_loop,
            args=(redis_client, pid, create_time_ns, stop_event),
            daemon=True,
            name="quorin-heartbeat",
        )
        _state.thread = thread
        _state.stop_event = stop_event
        _state.pid = pid
        _state.create_time_ns = create_time_ns
        _state.redis_client = redis_client
        # Register atexit AFTER state is fully populated. The handler
        # PID-gates to survive fork+child-clean-exit (CRITICAL fix).
        atexit.register(_atexit_handler)
        thread.start()


def stop() -> None:
    """Symmetric inverse of :func:`ensure_started`.

    After ``stop()``:
      * timer thread joined (best-effort, 2 s timeout)
      * one final HDEL via :func:`_atexit_handler`
      * atexit registration removed (no zombie firing on real exit)
      * every ``_state`` field set to None (matches :func:`_reset_after_fork`)
    """
    with _state_lock:
        if _state.stop_event is not None:
            _state.stop_event.set()
        if _state.thread is not None:
            _state.thread.join(timeout=2.0)
        _atexit_handler()
        with contextlib.suppress(Exception):  # pragma: no cover
            atexit.unregister(_atexit_handler)
        _state.thread = None
        _state.stop_event = None
        _state.pid = None
        _state.create_time_ns = None
        _state.redis_client = None
        _state.labels_writes_ok = None
        _state.labels_writes_redis_error = None


def is_running() -> bool:
    """True iff the heartbeat thread is alive in this process."""
    return _state.thread is not None and _state.thread.is_alive()


# ---------------------------------------------------------------------------
# Internals.
# ---------------------------------------------------------------------------


def _loop(
    redis_client: redis_lib.Redis,
    pid: int,
    create_time_ns: int,
    stop_event: threading.Event,
) -> None:
    """Heartbeat thread main loop.

    Captures label refs as locals BEFORE the polling loop. If :func:`stop`
    runs concurrently and nulls ``_state.labels_*``, an in-flight HSET
    that unblocks after the null would AttributeError on
    ``_state.labels_writes_ok.inc()``. Locals stay valid for the lifetime
    of this stack frame, so the unblocked HSET completes cleanly and the
    next ``stop_event.wait()`` exits the loop.
    """
    labels_ok = _state.labels_writes_ok
    labels_err = _state.labels_writes_redis_error
    while not stop_event.wait(HEARTBEAT_INTERVAL_SECONDS):
        try:
            _write_heartbeat(redis_client, pid, create_time_ns)
            heartbeat_age_seconds.set(0.0)
            if labels_ok is not None:
                labels_ok.inc()  # type: ignore[attr-defined]
        except _redis.RedisError:
            # Don't kill the thread; next tick retries. Gauge climbs
            # naturally (we don't .set(0.0) on the failure path).
            if labels_err is not None:
                labels_err.inc()  # type: ignore[attr-defined]


def _write_heartbeat(redis_client: redis_lib.Redis, pid: int, create_time_ns: int) -> None:
    wall_ns = time.time_ns()
    redis_client.hset(
        KEY_HEARTBEATS,
        str(pid),
        f"{create_time_ns}:{wall_ns}",
    )


def _atexit_handler() -> None:
    """Forked-child's clean exit must NOT delete parent's heartbeat.

    Even though :func:`_reset_after_fork` nulls ``_state.pid``, the gate
    is belt-and-suspenders (covers any code path that corrupted state,
    any fork-hook bypass).
    """
    if _state.pid is None:
        return
    if os.getpid() != _state.pid:
        return  # forked descendant; not our heartbeat to delete
    client = _state.redis_client
    if client is None:
        return
    with contextlib.suppress(_redis.RedisError):
        client.hdel(KEY_HEARTBEATS, str(_state.pid))


def _reset_after_fork() -> None:
    """Child gets fresh state. Parent's thread keeps going in parent.

    Clears EVERY field: any post-fork code reading ``_state.*`` must
    fail loud (None) rather than silently use parent's values. Child's
    first :func:`ensure_started` re-allocates everything.

    CR.B.10 (v0.1.1): also rebind ``_state_lock`` to a fresh
    :class:`threading.Lock`. If the parent was holding the lock at fork
    time (e.g. another thread was mid-``ensure_started``), the child
    would inherit it as held — child's ``with _state_lock:`` would
    deadlock forever. Rebinding here breaks the inheritance. Safe
    because fork hooks run before child code resumes; no thread can
    observe the old vs new lock atomically.
    """
    global _state_lock  # noqa: PLW0603 — module-level lock rebinding is the entire point.
    _state.thread = None
    _state.stop_event = None
    _state.pid = None
    _state.create_time_ns = None
    _state.redis_client = None
    _state.labels_writes_ok = None
    _state.labels_writes_redis_error = None
    _state_lock = threading.Lock()


os.register_at_fork(after_in_child=_reset_after_fork)
