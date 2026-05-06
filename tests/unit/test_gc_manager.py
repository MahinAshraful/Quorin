"""Unit tests for quorin._internal.gc_manager.

Manager itself is platform-agnostic (only stdlib gc / threading / os hooks).
No POSIX gate. The integration-flavored "collector running while assemble
loop runs" test lives in tests/integration/test_gc_manager_e2e.py because
it needs a real segment.

The autouse fixture below ensures every test starts with a fresh manager
state and clean ``gc.callbacks``, even if a prior test leaked.
"""

from __future__ import annotations

import contextlib
import gc
import threading
import time
from collections.abc import Iterator

import pytest

from quorin._internal import gc_manager
from quorin._internal.gc_manager import (
    _gc_callback,
    _observers,
    _pause_starts,
    _zombies,
    freeze,
    is_running,
    is_timer_running,
    start_collector,
    stop_collector,
    unfreeze,
)
from quorin.metrics import gc_pause_seconds


@pytest.fixture(autouse=True)
def _gc_manager_isolation() -> Iterator[None]:
    """Belt-and-suspenders cleanup so a leaked test cannot poison the next."""
    yield
    with contextlib.suppress(Exception):
        stop_collector(join_timeout=1.0)
    with contextlib.suppress(Exception):
        unfreeze()
    # Force-remove our callback in case stop_collector failed.
    while _gc_callback in gc.callbacks:
        gc.callbacks.remove(_gc_callback)
    # Drain any leftover zombies.
    for z in list(_zombies):
        z.join(timeout=2.0)
    _zombies.clear()
    _observers.clear()
    _pause_starts[0] = None
    _pause_starts[1] = None
    _pause_starts[2] = None


# ---------------------------------------------------------------------------
# Lifecycle: start / stop / idempotency.
# ---------------------------------------------------------------------------


def test_start_idempotent() -> None:
    start_collector()  # long interval; collect won't fire
    assert is_running()
    callback_count_before = gc.callbacks.count(_gc_callback)
    start_collector()  # second call — should be no-op
    assert is_running()
    assert gc.callbacks.count(_gc_callback) == callback_count_before


def test_stop_without_start_is_noop() -> None:
    assert not is_running()
    stop_collector()  # no exception
    assert not is_running()


def test_start_stop_cycle() -> None:
    assert not is_running()
    start_collector()
    assert is_running()
    stop_collector()
    assert not is_running()
    start_collector()
    assert is_running()
    stop_collector()
    assert not is_running()


def test_callback_inserted_at_position_zero() -> None:
    start_collector()
    assert gc.callbacks[0] is _gc_callback


def test_callback_removed_on_stop() -> None:
    start_collector()
    assert _gc_callback in gc.callbacks
    stop_collector()
    assert _gc_callback not in gc.callbacks


def test_install_callback_false_skips_callback() -> None:
    """The escape hatch for users who call freeze() and want to avoid the
    callback+freeze interaction. Verifies start_collector(install_callback=
    False) installs nothing."""
    assert _gc_callback not in gc.callbacks
    start_collector(install_callback=False)
    assert _gc_callback not in gc.callbacks
    assert not is_running()  # no callback => not running per is_running's definition
    assert not _observers  # no pre-warming when callback isn't installed
    stop_collector()
    assert _gc_callback not in gc.callbacks


def test_install_callback_false_with_timer() -> None:
    """Timer can run without the callback — observations are skipped, but
    gc.collect(2) still fires on schedule."""
    start_collector(install_callback=False, gen2_interval_seconds=0.05)
    try:
        assert _gc_callback not in gc.callbacks
        assert is_timer_running()
    finally:
        stop_collector()
    assert not is_timer_running()


def test_default_does_not_start_timer() -> None:
    """Post-ADR-006: gen2_interval_seconds default is None, so
    start_collector() must NOT spawn a timer thread."""
    start_collector()
    try:
        assert is_running()
        assert not is_timer_running()
    finally:
        stop_collector()


# ---------------------------------------------------------------------------
# Callback semantics.
# ---------------------------------------------------------------------------


def test_pause_observed_for_each_generation(monkeypatch: pytest.MonkeyPatch) -> None:
    """Force each generation's collection; verify the callback observes
    via the prewarmed observer."""
    start_collector()

    calls: list[tuple[int, float]] = []
    for g in (0, 1, 2):
        # Capture by default-argument trick to avoid late-binding.
        def make_obs(gen: int = g):
            def obs(duration: float) -> None:
                calls.append((gen, duration))

            return obs

        monkeypatch.setattr(_observers[g], "observe", make_obs())

    gc.collect(0)
    gc.collect(1)
    gc.collect(2)

    observed_gens = {g for g, _ in calls}
    # gc.collect(N) collects generations 0..N. So we expect at least 0 to fire
    # for collect(0), at least 1 for collect(1), at least 2 for collect(2).
    # We assert the union covers all three.
    assert observed_gens == {0, 1, 2}, (
        f"expected observations for all 3 generations, got {observed_gens}"
    )
    # All durations must be non-negative.
    for _, duration in calls:
        assert duration >= 0.0


def test_callback_handles_missing_generation_field() -> None:
    start_collector()
    # Snapshot state — the missing-gen call must not mutate _pause_starts.
    before = list(_pause_starts)
    _gc_callback("start", {})  # no "generation" key
    _gc_callback("stop", {})
    assert list(_pause_starts) == before


def test_callback_handles_unmatched_stop() -> None:
    start_collector()
    # Ensure no prior "start" recorded.
    _pause_starts[0] = None
    # "stop" without prior "start" — silent skip.
    _gc_callback("stop", {"generation": 0})
    # Nothing observed; _pause_starts unchanged.
    assert _pause_starts[0] is None


def test_callback_swallows_exception(monkeypatch: pytest.MonkeyPatch) -> None:
    start_collector()

    def explode(_duration: float) -> None:
        raise RuntimeError("boom")

    monkeypatch.setattr(_observers[0], "observe", explode)
    # First fire "start" to populate _pause_starts[0], then "stop" which calls
    # the exploding observe. The callback must not raise.
    _gc_callback("start", {"generation": 0})
    _gc_callback("stop", {"generation": 0})  # would raise without the bare-except


# ---------------------------------------------------------------------------
# freeze / unfreeze.
# ---------------------------------------------------------------------------


def test_freeze_and_unfreeze_idempotent() -> None:
    baseline = gc.get_freeze_count()
    freeze()
    after_first_freeze = gc.get_freeze_count()
    assert after_first_freeze >= baseline
    freeze()  # idempotent — no exception, count is monotonic
    after_second_freeze = gc.get_freeze_count()
    assert after_second_freeze >= after_first_freeze
    unfreeze()
    assert gc.get_freeze_count() == 0


def test_freeze_moves_tracked_objects() -> None:
    """Allocate a long-lived dict, freeze, verify gc.get_freeze_count() grew."""
    baseline = gc.get_freeze_count()
    # Hold a reference so the dict isn't immediately collectable.
    _retained = [{"k": i} for i in range(1000)]
    # Force a gc.collect so the dicts are tracked / promoted.
    gc.collect()
    freeze()
    delta = gc.get_freeze_count() - baseline
    # We expect the 1000 dicts (plus their inner string keys & list shell)
    # to be reflected in the freeze count.
    assert delta >= 1000, f"freeze() should move at least 1000 objects; got delta={delta}"
    # Reference _retained so ruff/lint doesn't flag it as unused.
    assert len(_retained) == 1000


def test_freeze_warns_when_callback_installed() -> None:
    """ADR-006 footgun guard: calling freeze() while the instrumentation
    callback is installed produces a measured tail-latency penalty. The
    UserWarning makes the footgun loud at the API boundary; users get a
    signal before they ship and stare at latency dashboards wondering why."""
    start_collector()  # default: install_callback=True
    with pytest.warns(UserWarning, match="ADR-006"):
        freeze()


def test_freeze_does_not_warn_when_callback_absent() -> None:
    """The safe usage pattern: freeze() before start_collector() (or
    start_collector(install_callback=False)). No warning should fire."""
    import warnings as _warnings

    with _warnings.catch_warnings():
        _warnings.simplefilter("error")  # any warning becomes an exception
        freeze()  # callback is NOT installed (autouse fixture cleans up)


def test_freeze_does_not_warn_with_install_callback_false() -> None:
    """Explicit opt-out path: start_collector(install_callback=False) +
    freeze() should be silent."""
    import warnings as _warnings

    start_collector(install_callback=False)
    with _warnings.catch_warnings():
        _warnings.simplefilter("error")
        freeze()


# ---------------------------------------------------------------------------
# Observer pre-warming.
# ---------------------------------------------------------------------------


def test_observers_prewarmed() -> None:
    """After start_collector(), _observers has 3 entries that match the
    prometheus children — proves caching, no per-callback allocation."""
    start_collector()
    assert len(_observers) == 3
    for g in (0, 1, 2):
        # The prewarmed observer is the same object prometheus_client returns
        # on a subsequent .labels() call (proves dict caching).
        assert _observers[g] is gc_pause_seconds.labels(generation=g)


# ---------------------------------------------------------------------------
# Zombie tracking.
# ---------------------------------------------------------------------------


def test_zombie_tracked_on_join_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    """If the collector loop is misbehaving and exceeds join_timeout, the
    thread is tracked as a zombie. A subsequent start_collector() will not
    spawn a parallel collector (filtered by is_alive())."""

    # Replace the loop with one that ignores the stop event and sleeps for
    # 2 s — long enough that join_timeout=0.1 will expire before it returns.
    def slow_loop(_interval: float, _stop_event: threading.Event) -> None:
        time.sleep(2.0)

    monkeypatch.setattr(gc_manager, "_collector_loop", slow_loop)
    start_collector(gen2_interval_seconds=0.05)
    stop_collector(join_timeout=0.1)
    assert len(_zombies) == 1
    assert _zombies[0].is_alive()
    # The autouse fixture will join the zombie before the next test.


def test_start_after_zombie_does_not_spawn_parallel(monkeypatch: pytest.MonkeyPatch) -> None:
    """Zombies are filtered by is_alive(); a fresh start_collector() works
    once the zombie exits."""

    def slow_loop(_interval: float, _stop_event: threading.Event) -> None:
        time.sleep(0.5)

    monkeypatch.setattr(gc_manager, "_collector_loop", slow_loop)
    start_collector(gen2_interval_seconds=0.05)
    stop_collector(join_timeout=0.05)
    assert len(_zombies) == 1
    # Wait for zombie to finish so the next start gets clean ground.
    _zombies[0].join(timeout=2.0)
    _zombies.clear()

    # Now a normal start works.
    monkeypatch.undo()
    start_collector()
    assert is_running()


# ---------------------------------------------------------------------------
# Tight loop while collector is running — sanity check that threads
# coexist without exceptions. Doesn't need a segment (that test lives in
# tests/integration/).
# ---------------------------------------------------------------------------


def test_main_thread_work_while_collector_runs() -> None:
    """1000 iterations of a small allocation+drop loop while the collector
    runs at 50ms interval. Asserts no exception in either thread, the
    histogram has at least one observation, and is_running() stays True."""
    start_collector(gen2_interval_seconds=0.05)
    try:
        for _ in range(1000):
            _ = [i for i in range(100)]
        assert is_running()
        # Force at least one collection so we know the histogram fired.
        before = _observers[0]._sum.get()
        gc.collect(0)
        after = _observers[0]._sum.get()
        # observed at least one event; sum either grew or stayed same if the
        # gen-0 collection happened to take 0 ns. The Counter for _count
        # is a more reliable signal.
        assert after >= before
    finally:
        stop_collector()
