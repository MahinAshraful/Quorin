"""GC management — opt-in instrumentation, opt-in timer, opt-in freeze.

Step 7 was supposed to compress the long tail (p999 / p9999) of the assemble
path by combining ``gc.freeze()`` + a 500 ms ``gc.collect(2)`` timer thread
with pause-histogram instrumentation. The build plan and ADR-006's first
draft both treated that combination as the recommended default.

The N=50 repeat-measurement benchmark (``benchmarks/runs/step7_gc_tail_repeat.py``)
revealed an interaction effect:

- ``baseline`` (no callback, no freeze): median max 155 µs, 12% spike rate
  at ≥ 500 µs.
- ``callback_only_no_freeze``: 158 µs / 14% — within noise of baseline.
- ``freeze_only_no_callback``: 161 µs / 12% — within noise of baseline.
- **``callback + freeze`` (no timer): 204 µs / 16%** — 31% worse median max
  and 67% relative increase in 1 ms-spike rate, despite each component
  alone being benign.

Mechanism is unlocalized at ADR-006 close. The module API is therefore
designed so users can pick exactly the components they want and document
the interaction:

1. **`gc.freeze()` is opt-in via :func:`freeze`** — caller invokes
   explicitly. No automatic call from start_collector.
2. **The pause-instrumentation callback is opt-in via the
   ``install_callback`` arg (default True)**. Default-on because the
   callback alone is within measurement noise of baseline; users who
   ALSO call :func:`freeze` should opt out via ``install_callback=False``.
3. **The 500 ms gen-2 timer thread is opt-in via the
   ``gen2_interval_seconds`` arg (default None / off)**. The timer's
   tail-bounding value is hardware-dependent; ADR-006 documents the
   no-freeze interval sweep that justifies the default-off and
   recommends the canonical opt-in interval for users who want it.

Critical invariants (locked by ADR-006):

- **Module-level state.** Python's GC is process-global. There is exactly one
  manager per process; no class layer.
- **Callback bodies must not allocate.** ``prometheus_client.Histogram.labels``
  allocates a tuple key + dict slot on first use; allocation inside a GC
  callback can trigger re-entrant collection. We pre-warm the three label
  children at :func:`start_collector` so the callback only does C-level work.
- **`_pause_starts` is a fixed-size list, not a dict.** Three generations,
  indexed 0/1/2. No allocation, no hash, no defensive ``.pop(gen, None)``.
- **`gc.callbacks.insert(0, ...)`** so our histogram observes the bare
  ``gc_collect_main`` cost, not time spent in peer callbacks (third-party
  tracers, etc).
- **Daemon thread + ``threading.Event`` shutdown.** Pattern that Step 10's
  WAL consumer and Step 14's watchdog will reuse.
- **`os.register_at_fork(after_in_child=...)`** is non-negotiable.
  Forked workers (gunicorn / uvicorn) inherit the parent's ``_thread``
  reference but not the actual thread; without the hook, child processes
  silently run with no scheduled gen-2 sweeps.
- **Stop ordering: remove callback BEFORE setting the stop event.** Trade:
  lose at most one in-flight observation on shutdown for clean state — no
  callback fires after we've started tearing down.
- **Zombie threads on join timeout are tracked, not nulled.** A subsequent
  :func:`start_collector` will not spawn a parallel collector while a zombie
  is still finishing its `gc.collect(2)` — preventing two collectors from
  scrambling ``_pause_starts`` for the same generation.
"""

from __future__ import annotations

import gc
import os
import threading
import time
import warnings
from typing import Any

from quorin.metrics import gc_pause_seconds

# ---------------------------------------------------------------------------
# Module state. Process-global; only one Python GC.
#
# Grouped in :class:`_State` (with ``__slots__``) so the public API can rebind
# fields via attribute access, dodging the ``global`` keyword. ADR-006 keeps
# the API surface module-level (no public class); this is an internal
# implementation detail.
# ---------------------------------------------------------------------------


class _State:
    __slots__ = ("stop_event", "thread")

    def __init__(self) -> None:
        self.thread: threading.Thread | None = None
        self.stop_event: threading.Event | None = None


_state = _State()
_zombies: list[threading.Thread] = []
_lock = threading.Lock()

# Fixed 3-element list keyed by generation index. Beats a dict: no allocation
# per callback, no .pop(gen, None) defensive lookup, no hash. Generations are
# always 0/1/2 in CPython 3.7+.
_pause_starts: list[float | None] = [None, None, None]

# Pre-warmed prometheus children, populated at :func:`start_collector`.
# Empty until the collector runs.
_observers: list[Any] = []


# ---------------------------------------------------------------------------
# Public API.
# ---------------------------------------------------------------------------


def start_collector(
    *,
    install_callback: bool = True,
    gen2_interval_seconds: float | None = None,
) -> None:
    """Idempotent. Register the pause callback (default true) and
    optionally start a gen-2 timer thread (default off).

    Default behavior (no args): install the GC pause callback, no timer
    thread. The callback observes every organic GC pause on
    ``quorin_gc_pause_seconds`` for ops visibility. This is within
    measurement noise of the do-nothing baseline at N=50.

    **Known performance interaction (ADR-006).** Combining
    ``install_callback=True`` with :func:`freeze` produces a measured
    tail-latency penalty (~31% worse median max, ~67% relative increase
    in 1ms-spike rate) that does not appear when either component is
    used alone. Mechanism is unlocalized as of ADR-006. If you call
    :func:`freeze` and want to avoid the interaction, pass
    ``install_callback=False`` here — you lose the pause histogram but
    pay no measurable tail tax.

    Re-startable after :func:`stop_collector`.

    Under :func:`os.fork`, child processes inherit the parent's
    ``_state.thread`` reference but not the actual thread. The fork hook
    resets state, so callers in forked workers (gunicorn / uvicorn) just
    call ``start_collector()`` after fork and it works.

    Args:
        install_callback: when True (default), register the GC pause
            instrumentation callback. Pass False to skip — useful when
            also calling :func:`freeze` (see interaction above).
        gen2_interval_seconds: when not None, start a daemon timer thread
            that runs ``gc.collect(2)`` at that cadence. Default None
            (no timer). The build plan suggested 0.5 s; the no-freeze
            interval sweep in ADR-006 documents the empirical recommendation.
    """
    with _lock:
        callback_already_installed = _gc_callback in gc.callbacks
        thread_already_running = _state.thread is not None and _state.thread.is_alive()

        callback_done = (not install_callback) or callback_already_installed
        timer_done = (gen2_interval_seconds is None) or thread_already_running
        if callback_done and timer_done:
            return

        if install_callback:
            # Pre-warm prometheus label children. Doing this OUTSIDE the
            # callback ensures the dict-insertion path in prometheus_client is
            # never taken during a GC pause — otherwise our gen-0 callback
            # could allocate a tuple key and trigger a re-entrant gen-0
            # collection.
            if not _observers:
                for gen in (0, 1, 2):
                    _observers.append(gc_pause_seconds.labels(generation=gen))

            # Insert at position 0 so our callback runs first — observed pause
            # duration excludes time spent in peer callbacks (third-party
            # tracers, etc).
            if _gc_callback not in gc.callbacks:
                gc.callbacks.insert(0, _gc_callback)

        if gen2_interval_seconds is None:
            # Instrumentation only — no timer thread.
            return

        stop_event = threading.Event()
        _state.stop_event = stop_event
        thread = threading.Thread(
            target=_collector_loop,
            args=(gen2_interval_seconds, stop_event),
            daemon=True,
            name="quorin-gc",
        )
        _state.thread = thread
        thread.start()


def stop_collector(*, join_timeout: float = 2.0) -> None:
    """Idempotent. Stop the timer thread and unregister the callback.

    Order is deliberate: remove the callback BEFORE setting the stop event.
    The in-flight ``gc.collect(2)`` on the timer thread loses its observation
    (one missed pause measurement on shutdown), but state stays clean — no
    callback fires after we've started tearing down.

    If :meth:`threading.Thread.join` times out, the thread becomes a zombie.
    We track zombies and refuse to start_collector() again while one is still
    alive — preventing two collectors from scrambling ``_pause_starts``.

    Args:
        join_timeout: seconds to wait for the timer thread to finish its
            current ``gc.collect(2)`` and exit.
    """
    with _lock:
        # If neither the callback nor a timer thread is installed, nothing
        # to do.
        if _gc_callback not in gc.callbacks and _state.thread is None:
            return
        # Remove callback FIRST so in-flight collects don't observe.
        try:
            if _gc_callback in gc.callbacks:
                gc.callbacks.remove(_gc_callback)
        finally:
            stop_event = _state.stop_event
            thread = _state.thread
            _state.thread = None
            _state.stop_event = None

    # Outside the lock: signaling and joining can take time, and we don't
    # want to block other state-transition calls.
    try:
        if stop_event is not None:
            stop_event.set()
        if thread is not None:
            thread.join(timeout=join_timeout)
            if thread.is_alive():
                # Zombie: ``gc.collect(2)`` was mid-flight and exceeded our
                # SLA. Track it so a subsequent ``start_collector()`` does
                # not spawn a parallel collector.
                _zombies.append(thread)
    finally:
        # Reset accumulator. Even on exception, leave state consistent.
        _pause_starts[0] = None
        _pause_starts[1] = None
        _pause_starts[2] = None
        _observers.clear()


def freeze() -> None:
    """Move all currently-tracked objects into the permanent generation.

    Subsequent ``gc.collect(N)`` walks only objects allocated after this
    call. Idempotent — calling twice just re-freezes anything new.

    Caller's responsibility: ensure all long-lived allocations (segments,
    schemas, pools, layout offset tables) are done before calling. Do NOT
    call from inside a ``gc.callbacks`` handler — behaviour is undefined.

    **Emits UserWarning if the GC instrumentation callback is currently
    installed** (i.e., :func:`start_collector` has been called with
    ``install_callback=True``). The N=50 isolation benchmark in ADR-006
    showed the combination of callback + freeze produces a ~31% tail-
    latency penalty (~67% relative increase in 1 ms-spike rate) that does
    not appear when either component is used alone. Mechanism is
    unlocalized; see ADR-006 for the data and the suspect list.

    Two safe usage patterns:

    1. Call ``freeze()`` BEFORE ``start_collector()``. The freeze runs while
       no callback is installed; subsequent observations cover only the
       post-freeze working set.
    2. Pass ``install_callback=False`` to ``start_collector()`` — you lose
       the pause histogram but pay no interaction tax.
    """
    if _gc_callback in gc.callbacks:
        warnings.warn(
            "quorin._internal.gc_manager.freeze() called while the GC "
            "instrumentation callback is installed. ADR-006 measured a "
            "~31% tail-latency penalty for this combination at N=50. "
            "Safe alternatives: call freeze() BEFORE start_collector(), "
            "or pass install_callback=False to start_collector().",
            UserWarning,
            stacklevel=2,
        )
    gc.freeze()


def unfreeze() -> None:
    """Reverse of :func:`freeze`. Primarily for tests; production code
    rarely needs to undo a freeze."""
    gc.unfreeze()


def is_running() -> bool:
    """True iff the GC pause callback is currently installed (instrumentation
    is active), regardless of whether the gen-2 timer thread is also running.

    For querying the timer thread specifically, see :func:`is_timer_running`.
    """
    return _gc_callback in gc.callbacks


def is_timer_running() -> bool:
    """True iff the gen-2 timer thread is currently alive.

    Distinct from :func:`is_running` because Step 7's default opt-in to
    instrumentation does not start a timer.
    """
    return _state.thread is not None and _state.thread.is_alive()


# ---------------------------------------------------------------------------
# Internals.
# ---------------------------------------------------------------------------


def _collector_loop(interval: float, stop_event: threading.Event) -> None:
    """Sleep ``interval``, run ``gc.collect(2)``, repeat.

    ``Event.wait(timeout)`` returns True when the event is set (early exit)
    or False on timeout. This is the right primitive vs ``time.sleep``,
    which would force waiting the full interval before noticing shutdown.
    """
    while not stop_event.wait(interval):
        gc.collect(2)


def _gc_callback(phase: str, info: dict[str, Any]) -> None:
    """Called by CPython's gc module on every collection start/stop.

    Must not allocate Python objects (re-entrancy risk; see ADR-006). All
    allocation done once at :func:`start_collector` — see ``_observers``.

    On exception: silently swallow. We cannot afford to break GC. A future
    iteration could increment a pre-warmed ``Counter`` for visibility; for
    v1 we accept the blind spot in exchange for guaranteed safety.
    """
    try:
        gen = info.get("generation")
        if gen is None:
            # Never seen on CPython 3.7+; defensive skip beats mislabeling
            # a gen-2 pause as gen-0 in the histogram.
            return
        if phase == "start":
            _pause_starts[gen] = time.perf_counter()
        elif phase == "stop":
            t0 = _pause_starts[gen]
            if t0 is not None:
                _pause_starts[gen] = None
                _observers[gen].observe(time.perf_counter() - t0)
    except Exception:
        # Documented intentional swallow. See ADR-006.
        pass


def _reset_after_fork() -> None:
    """Child-process reset.

    The parent's ``_state.thread`` reference is meaningless after fork —
    the actual thread does not exist in the child. Clear all state so the
    child can call :func:`start_collector` cleanly.
    """
    _state.thread = None
    _state.stop_event = None
    _zombies.clear()
    _pause_starts[0] = None
    _pause_starts[1] = None
    _pause_starts[2] = None
    _observers.clear()
    if _gc_callback in gc.callbacks:
        gc.callbacks.remove(_gc_callback)


# Wire the fork hook at import time so callers in forked workers
# (gunicorn / uvicorn) don't have to think about it.
os.register_at_fork(after_in_child=_reset_after_fork)
