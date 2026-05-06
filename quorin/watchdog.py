"""Quorin watchdog (Step 14).

Runs as a separate process from library producers/consumers. Reads the
``quorin:heartbeats`` hash on a 30 s tick, detects PIDs whose
``wall_time_ns`` field has not advanced for :data:`DEFAULT_MISS_THRESHOLD`
consecutive ticks, cross-checks with :mod:`psutil`, and runs an atomic
Redis Lua cleanup transaction for confirmed-dead PIDs. Also drains
``quorin:cleanup_queue`` (populated by ``SegmentRegistry.close`` when a
live process closes the last reference of a segment).

Run as a CLI:

.. code-block:: bash

    python -m quorin.watchdog --redis redis://localhost:6379/0
    python -m quorin.watchdog --redis redis://localhost:6379/0 \\
        --tick-interval-seconds 30 --miss-threshold 5 --metrics-port 9100

Tests can drive ticks deterministically by constructing
:class:`WatchdogState` directly and calling :meth:`WatchdogState.run_one_tick`.

Detection cadence
=================

Default: heartbeat producers tick at 10 s; watchdog ticks at 30 s; 5
consecutive missed ticks = 150 s detection ceiling. Tolerant of paused
producers (debugger SIGSTOP) per spec § 1339-1340; intolerant of
crashed producers.

Cross-UID watchdog deployments
==============================

If the watchdog runs as a different UID than its producers, the
``psutil.Process(pid).create_time()`` cross-check raises
``psutil.AccessDenied`` for cross-UID PIDs (without ``CAP_SYS_PTRACE``
or root). The cross-check is deliberately conservative: AccessDenied
and ZombieProcess both treat the PID as "alive but unverifiable" — do
NOT unlink. ADR-013 runbook: run the watchdog with
``CAP_SYS_PTRACE`` or root if it must clean up cross-UID producers'
orphans.

PID-reuse race window
=====================

The Lua cleanup script takes ``expected_create_time_ns`` as a second
ARGV. Its first op compares ``heartbeats[pid].create_ns`` against this
value and aborts with an empty list on mismatch — guarding against the
~1 ms window where the kernel could reuse a dead PID for a live new
process between this code's psutil check and the Lua execution. See
:mod:`quorin._internal.watchdog_lua` for the script + residual-risk
discussion.

Multiprocess prometheus
=======================

The watchdog process runs on the default registry. Multi-process
deployments (multiple watchdog instances behind a single metrics
endpoint) configure ``PROMETHEUS_MULTIPROC_DIR`` per
``prometheus_client``'s docs.
"""

from __future__ import annotations

import argparse
import os
import signal
import sys
import threading
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Final, cast

import psutil  # type: ignore[import-untyped]

from quorin._internal import heartbeat, posix_shm
from quorin._internal.watchdog_lua import (
    CLEANUP_DEAD_PID_LUA,
    DRAIN_CLEANUP_QUEUE_LUA,
)
from quorin.logging import get_logger
from quorin.metrics import (
    watchdog_cross_check_unverifiable_total,
    watchdog_dead_pids_total,
    watchdog_malformed_heartbeat_total,
    watchdog_pid_reuse_abort_total,
    watchdog_segments_unlinked_total,
    watchdog_tick_seconds,
)
from quorin.shm import KEY_CLEANUP_QUEUE, _key_pid_segments

if TYPE_CHECKING:
    import redis as redis_lib

import redis as _redis

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Constants. Defaults documented in the module docstring.
# ---------------------------------------------------------------------------

DEFAULT_TICK_INTERVAL_SECONDS: Final[float] = 30.0
DEFAULT_MISS_THRESHOLD: Final[int] = 5
DEFAULT_BATCH_DRAIN_SIZE: Final[int] = 64

KEY_HEARTBEATS: Final[str] = "quorin:heartbeats"
KEY_SEGMENT_TO_SCHEMA: Final[str] = "quorin:segment_to_schema"


# ---------------------------------------------------------------------------
# Per-PID tracking state.
# ---------------------------------------------------------------------------


@dataclass
class _PidEntry:
    """In-memory per-PID watchdog state.

    ``last_wall_time_ns`` is the most recent value seen in the heartbeat
    hash. ``miss_count`` counts consecutive ticks during which the value
    did NOT change. ``create_time_ns`` is read once per PID-first-seen
    and used as the PID-reuse fingerprint.
    """

    last_wall_time_ns: int
    create_time_ns: int
    miss_count: int = 0


@dataclass
class TickResult:
    """Per-tick summary returned by :meth:`WatchdogState.run_one_tick`.

    Useful for tests + operator dashboards reading the watchdog's stdout.
    """

    dead_pids: list[int] = field(default_factory=list)
    segments_unlinked_dead: int = 0
    segments_unlinked_drain: int = 0
    malformed_count: int = 0
    unverifiable_count: int = 0
    pid_reuse_aborts: int = 0


# ---------------------------------------------------------------------------
# WatchdogState. Public for tests; CLI uses run_forever.
# ---------------------------------------------------------------------------


def _parse_heartbeat_value(value: bytes) -> tuple[int, int]:
    """Parse a heartbeat hash value ``b"{create_ns}:{wall_ns}"``.

    Returns ``(create_ns, wall_ns)``. Raises ``ValueError`` on any
    malformation. Caller wraps + counts via
    :data:`quorin_watchdog_malformed_heartbeat_total`.
    """
    decoded = value.decode("ascii")
    create_str, wall_str = decoded.split(":", 1)
    return int(create_str), int(wall_str)


class WatchdogState:
    """In-process watchdog tick driver.

    Tests construct directly and call :meth:`run_one_tick`. The CLI's
    :func:`run_forever` constructs once and calls per tick.
    """

    def __init__(
        self,
        redis_client: redis_lib.Redis,
        *,
        miss_threshold: int = DEFAULT_MISS_THRESHOLD,
        batch_drain_size: int = DEFAULT_BATCH_DRAIN_SIZE,
    ) -> None:
        self._redis = redis_client
        self._miss_threshold = miss_threshold
        self._batch_drain_size = batch_drain_size
        self._cleanup_lua = redis_client.register_script(CLEANUP_DEAD_PID_LUA)
        self._drain_lua = redis_client.register_script(DRAIN_CLEANUP_QUEUE_LUA)
        self._tracked: dict[int, _PidEntry] = {}
        self._self_pid = os.getpid()
        # Pre-warmed prometheus label children for the unverifiable counter.
        # Pre-warming avoids tuple-key + dict-slot allocation at first
        # cross-check failure (ADR-006 lesson).
        self._labels_unverifiable = {
            "AccessDenied": watchdog_cross_check_unverifiable_total.labels(reason="access_denied"),
            "ZombieProcess": watchdog_cross_check_unverifiable_total.labels(reason="zombie"),
        }
        self._labels_unlinked_dead = watchdog_segments_unlinked_total.labels(path="dead_pid")
        self._labels_unlinked_drain = watchdog_segments_unlinked_total.labels(path="cleanup_queue")
        # The watchdog itself heartbeats so a supervisor (systemd /
        # k8s liveness) can monitor IT. Self-PID is excluded from the
        # dead-PID candidate set so the watchdog never declares itself
        # dead. One call at construction; idempotent.
        heartbeat.ensure_started(redis_client, self._self_pid)

    # ----- per-tick driver --------------------------------------------------

    def run_one_tick(self) -> TickResult:  # noqa: PLR0912, PLR0915 - linear orchestration
        """Run one watchdog cycle. Returns a :class:`TickResult` summary.

        Steps:

        1. HGETALL ``quorin:heartbeats``.
        2. For each pid, parse value (skip on ValueError + counter), update
           ``self._tracked``. wall_time_ns ``==`` last seen → miss_count++;
           ``!=`` (including backward jumps) → reset.
        3. For tracked PIDs at threshold: cross-check via psutil. Run the
           Lua for confirmed dead. Empty Lua return = PID-reuse abort.
        4. Drain ``quorin:cleanup_queue`` (up to ``batch_drain_size``).
        5. Prune ``self._tracked`` entries whose pid was NOT in step 1's
           HGETALL (handles external HDEL — operator manual cleanup).
        """
        t0 = time.perf_counter()
        result = TickResult()
        try:
            # Step 1: HGETALL quorin:heartbeats
            # cast: redis-py's stub union with Awaitable forces narrowing.
            raw = cast("dict[Any, Any]", self._redis.hgetall(KEY_HEARTBEATS) or {})
            seen_pids: set[int] = set()

            # Step 2: parse + update _tracked
            for raw_pid, raw_value in raw.items():
                try:
                    pid = int(raw_pid.decode("ascii") if isinstance(raw_pid, bytes) else raw_pid)
                    create_ns, wall_ns = _parse_heartbeat_value(
                        raw_value if isinstance(raw_value, bytes) else raw_value.encode("ascii")
                    )
                except (ValueError, AttributeError, UnicodeDecodeError):
                    logger.warning(
                        "watchdog_malformed_heartbeat",
                        raw_pid=raw_pid,
                        raw_value=raw_value,
                    )
                    watchdog_malformed_heartbeat_total.inc()
                    result.malformed_count += 1
                    continue

                if pid == self._self_pid:
                    seen_pids.add(pid)  # tracked here so step 5 doesn't prune our own entry
                    continue
                seen_pids.add(pid)

                entry = self._tracked.get(pid)
                if entry is None:
                    self._tracked[pid] = _PidEntry(
                        last_wall_time_ns=wall_ns,
                        create_time_ns=create_ns,
                    )
                elif entry.last_wall_time_ns == wall_ns:
                    entry.miss_count += 1
                else:
                    # `!=` handles backward jumps (NTP correction, manual
                    # clock skew). Reset miss_count + update last_seen.
                    entry.last_wall_time_ns = wall_ns
                    entry.miss_count = 0
                    # If create_time changed (rare — would only happen via
                    # operator overwriting heartbeats[pid]), update so the
                    # stored fingerprint matches what the dead-PID Lua will
                    # compare against on a future cleanup.
                    entry.create_time_ns = create_ns

            # Step 3: cross-check + cleanup at threshold
            ready_for_check = [
                pid
                for pid, entry in self._tracked.items()
                if entry.miss_count >= self._miss_threshold
            ]
            for pid in ready_for_check:
                entry = self._tracked[pid]
                declare_dead, unverifiable_kind = self._cross_check(pid, entry.create_time_ns)
                if unverifiable_kind is not None:
                    result.unverifiable_count += 1
                    label = self._labels_unverifiable.get(unverifiable_kind)
                    if label is not None:
                        label.inc()
                    # Leave _tracked entry intact; next tick re-checks.
                    continue
                if not declare_dead:
                    continue

                # Confirmed dead. Run cleanup Lua atomically. Returns:
                #   -1 → PID-reuse race detected (Lua's HGET guard tripped).
                #    N (>=0) → segments whose refcount hit zero;
                #              all were SADDed to cleanup_queue for the
                #              uniform step-4 drain to posix_shm.unlink.
                returned = int(
                    self._cleanup_lua(
                        keys=[
                            _key_pid_segments(pid),
                            KEY_HEARTBEATS,
                            KEY_CLEANUP_QUEUE,
                            KEY_SEGMENT_TO_SCHEMA,
                        ],
                        args=[str(pid), str(entry.create_time_ns)],
                    )
                )
                if returned < 0:
                    logger.warning("watchdog_pid_reuse_abort", pid=pid)
                    watchdog_pid_reuse_abort_total.inc()
                    result.pid_reuse_aborts += 1
                    # Forget the tracked entry; next tick will re-track
                    # under the new (reused) PID's create_time.
                    self._tracked.pop(pid, None)
                    continue

                watchdog_dead_pids_total.inc()
                result.dead_pids.append(pid)
                # The Lua's queued segments will be unlinked by step 4's
                # uniform cleanup_queue drain in this same tick. Step 3
                # does NOT unlink directly — single canonical syscall site.
                # Track count of segments whose refcount hit zero via this
                # path (pre-drain) for observability; actual unlink count
                # is recorded in result.segments_unlinked_drain.
                result.segments_unlinked_dead += returned
                self._labels_unlinked_dead.inc(returned)
                self._tracked.pop(pid, None)

            # Step 4: drain cleanup_queue
            drained = self._drain_lua(
                keys=[KEY_CLEANUP_QUEUE],
                args=[str(self._batch_drain_size)],
            )
            for raw_name in drained or ():
                name = raw_name.decode("utf-8") if isinstance(raw_name, bytes) else str(raw_name)
                try:
                    posix_shm.unlink(name)
                    result.segments_unlinked_drain += 1
                    self._labels_unlinked_drain.inc()
                except FileNotFoundError:
                    logger.debug(
                        "watchdog_unlink_already_gone",
                        segment=name,
                        path="cleanup_queue",
                    )
                except OSError:
                    logger.warning(
                        "watchdog_unlink_failed",
                        segment=name,
                        path="cleanup_queue",
                    )
                    raise

            # Step 5: prune _tracked entries no longer in the heartbeat hash.
            # Step 3 already pop()'d pids it cleaned up; this only catches
            # external HDELs (operator manual cleanup, future control plane).
            for pid in list(self._tracked):
                if pid not in seen_pids:
                    self._tracked.pop(pid, None)

            return result
        finally:
            watchdog_tick_seconds.observe(time.perf_counter() - t0)

    # ----- internals --------------------------------------------------------

    @staticmethod
    def _cross_check(pid: int, expected_create_time_ns: int) -> tuple[bool, str | None]:
        """Return ``(declare_dead, unverifiable_reason)``.

        * ``(True, None)`` — psutil confirms dead OR create_time mismatched.
        * ``(False, None)`` — psutil confirms alive with matching create_time.
        * ``(False, "AccessDenied"|"ZombieProcess")`` — cannot verify;
          conservative skip.
        """
        try:
            actual_create_time_ns = int(psutil.Process(pid).create_time() * 1e9)
        except psutil.ZombieProcess:
            # MUST come before NoSuchProcess — ZombieProcess inherits
            # from it. Conservative: zombie is "alive but unverifiable",
            # do NOT unlink (kernel still tracks the PID; the process
            # entry will go away once the parent reaps it, at which
            # point we'll get NoSuchProcess on a future tick).
            return False, "ZombieProcess"
        except psutil.NoSuchProcess:
            return True, None
        except psutil.AccessDenied:
            return False, "AccessDenied"
        # Exact equality. Same kernel + same /proc parse → bit-identical
        # int(float * 1e9). PID reuse always differs by ≥1 jiffy.
        return actual_create_time_ns != expected_create_time_ns, None


# ---------------------------------------------------------------------------
# CLI entry point.
# ---------------------------------------------------------------------------


def run_forever(
    redis_client: redis_lib.Redis,
    *,
    tick_interval_seconds: float = DEFAULT_TICK_INTERVAL_SECONDS,
    miss_threshold: int = DEFAULT_MISS_THRESHOLD,
    stop_event: threading.Event | None = None,
) -> None:
    """Run watchdog ticks forever (until SIGTERM / SIGINT or stop_event).

    Used by the CLI; tests typically drive :class:`WatchdogState` directly.
    """
    state = WatchdogState(redis_client, miss_threshold=miss_threshold)
    own_stop = stop_event is None
    if stop_event is None:
        stop_event = threading.Event()

    if own_stop:
        # Only install signal handlers if we own the stop_event — otherwise
        # the caller is in charge of shutdown signaling.
        prev_term = signal.signal(signal.SIGTERM, lambda *_: stop_event.set())
        prev_int = signal.signal(signal.SIGINT, lambda *_: stop_event.set())
    else:
        prev_term = None
        prev_int = None

    try:
        while not stop_event.wait(tick_interval_seconds):
            try:
                state.run_one_tick()
            except Exception:
                # Never let one bad tick kill the process — supervisor would
                # just restart us anyway, and a transient Redis blip
                # shouldn't manifest as a process death.
                logger.exception("watchdog_tick_failed")
    finally:
        if own_stop:
            if prev_term is not None:
                signal.signal(signal.SIGTERM, prev_term)
            if prev_int is not None:
                signal.signal(signal.SIGINT, prev_int)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m quorin.watchdog",
        description="Quorin watchdog (Step 14): detects dead PIDs + cleans orphan segments.",
    )
    parser.add_argument(
        "--redis",
        required=True,
        help="Redis URL, e.g. redis://localhost:6379/0",
    )
    parser.add_argument(
        "--tick-interval-seconds",
        type=float,
        default=DEFAULT_TICK_INTERVAL_SECONDS,
        help=f"Watchdog tick interval (default: {DEFAULT_TICK_INTERVAL_SECONDS}).",
    )
    parser.add_argument(
        "--miss-threshold",
        type=int,
        default=DEFAULT_MISS_THRESHOLD,
        help=(
            f"Consecutive missed heartbeats before declaring dead "
            f"(default: {DEFAULT_MISS_THRESHOLD})."
        ),
    )
    parser.add_argument(
        "--metrics-port",
        type=int,
        default=None,
        help=(
            "If set, expose /metrics via prometheus_client.start_http_server "
            "on this port. Default: disabled."
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """CLI entry for ``python -m quorin.watchdog --redis URL [--metrics-port N]``.

    Runs the watchdog forever (blocks); detects dead PIDs via heartbeat
    staleness + ``psutil`` cross-check, decrements their refcounts, drains
    the cleanup queue. Returns the process exit code.
    """
    args = _parse_args(argv)
    redis_client = _redis.Redis.from_url(args.redis)
    if args.metrics_port is not None:
        # Lazy imports — keep them out of the module load path so
        # ``import quorin.watchdog`` (e.g. by tests, or by users
        # constructing WatchdogState directly) doesn't pay
        # prometheus_client's HTTP server cost.
        from prometheus_client import start_http_server  # noqa: PLC0415

        from quorin.metrics import registry as quorin_registry  # noqa: PLC0415

        start_http_server(args.metrics_port, registry=quorin_registry)
    logger.info(
        "watchdog_start",
        tick_interval_seconds=args.tick_interval_seconds,
        miss_threshold=args.miss_threshold,
        metrics_port=args.metrics_port,
    )
    run_forever(
        redis_client,
        tick_interval_seconds=args.tick_interval_seconds,
        miss_threshold=args.miss_threshold,
    )
    logger.info("watchdog_stop")
    return 0


if __name__ == "__main__":
    sys.exit(main())
