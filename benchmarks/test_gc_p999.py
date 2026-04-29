"""p999 benchmarks for GC management — the empirical evidence behind ADR-006.

Three distributions of 100k assemble() calls, each measured with
``time.perf_counter_ns`` per iteration so we can compute p50, p99, p999,
p9999 ourselves (pytest-benchmark's reporting collapses to summary
statistics that hide the long tail).

| Scenario | Setup |
|---|---|
| `no_manager_no_freeze` | GC default. The bad baseline. |
| `freeze_plus_timer` | Step 7's design. Freeze long-lived state, gen-2 timer thread. |
| `freeze_plus_timer_with_pressure` | Same plus continuous allocation pressure to flush gen-0/1 frequently. |

ADR-006 quotes the resulting numbers and decides whether the no-``gc.disable``
choice survives or needs revisiting.

NOT marked ``slow``: 100k iterations of a 5 us assemble = ~500 ms work, plus
a 1 GB heap setup. Acceptable for the benchmark suite. Marked manual-only via
``@pytest.mark.benchmark`` so the default ``pytest -q`` doesn't run it; invoke
explicitly with ``pytest --benchmark-only benchmarks/test_gc_p999.py``.
"""

from __future__ import annotations

import gc
import sys
import time
from typing import Any

import numpy as np
import pytest

pytestmark = pytest.mark.skipif(
    sys.platform == "win32",
    reason="benchmarks require POSIX (Linux/WSL2)",
)

from _helpers import make_segment, pack_row, release_segment  # noqa: E402
from pyforge._internal.gc_manager import (  # noqa: E402
    freeze,
    start_collector,
    stop_collector,
    unfreeze,
)
from pyforge.assembly import assemble, prewarm  # noqa: E402
from pyforge.layout import insert  # noqa: E402
from pyforge.schema import FeatureField, FeatureSchema, dtype  # noqa: E402

_ITERS = 100_000


@pytest.fixture(scope="module", autouse=True)
def _prewarm_numba() -> None:
    prewarm()


class _Schema200(FeatureSchema):
    """200-field + 128-dim embedding. Same shape as the assembly benchmark
    for direct comparison; we expect ~5-6 µs warm p99 under the pool."""

    version = 1
    fields = [
        *(FeatureField(f"feat_{i:03d}", dtype.float32) for i in range(199)),
        FeatureField("embedding", dtype.float32, shape=(128,)),
    ]


def _build_long_lived_heap() -> list[Any]:
    """Allocate ~1 M long-lived Python objects so freeze() has something to
    move out of gen-2. Mix of dicts and lists with cross-references; this
    is what a realistic Pyforge process looks like (segment metadata, slot
    tables, schemas, pools, layout offset arrays).
    """
    heap: list[Any] = []
    for i in range(50_000):
        d = {"id": i, "tags": [f"tag_{j}" for j in range(20)]}
        heap.append(d)
    return heap


def _percentiles(samples_ns: np.ndarray) -> dict[str, float]:
    return {
        "p50_us": float(np.percentile(samples_ns, 50)) / 1000.0,
        "p99_us": float(np.percentile(samples_ns, 99)) / 1000.0,
        "p999_us": float(np.percentile(samples_ns, 99.9)) / 1000.0,
        "p9999_us": float(np.percentile(samples_ns, 99.99)) / 1000.0,
        "max_us": float(samples_ns.max()) / 1000.0,
    }


def _run_assemble_loop(seg: Any, n: int) -> np.ndarray:
    """Run `n` assembles with ns-precision timing, return per-call durations."""
    samples = np.empty(n, dtype=np.int64)
    perf = time.perf_counter_ns
    for i in range(n):
        t0 = perf()
        assemble(seg, "u")
        samples[i] = perf() - t0
    return samples


@pytest.fixture
def seg_200():
    seg = make_segment(_Schema200, capacity=16)
    values = {f.name: np.zeros(f.element_count, dtype=np.float32) for f in _Schema200.fields}
    insert(seg, "u", pack_row(_Schema200, values))
    yield seg
    release_segment(seg)


def _print_distribution(label: str, percentiles: dict[str, float]) -> None:
    """Side effect — make the numbers visible in pytest -s output for
    paste-into-ADR-006 by humans."""
    print(f"\n[gc_p999] {label}")
    for k, v in percentiles.items():
        print(f"  {k:>10}: {v:>8.2f} us")


def test_p999_no_manager_no_freeze(seg_200) -> None:
    """Baseline. GC fully default — no freeze, no timer thread."""
    heap = _build_long_lived_heap()
    try:
        gc.collect()  # one-shot to stabilize before measuring
        samples = _run_assemble_loop(seg_200, _ITERS)
        pcts = _percentiles(samples)
        _print_distribution("no_manager_no_freeze (baseline)", pcts)
        # No assertion threshold — this is reference data for the ADR.
        assert pcts["p50_us"] > 0
    finally:
        del heap


def test_p999_callback_only_no_freeze(seg_200) -> None:
    """ADR-006 anomaly investigation: isolate the GC callback's
    observation overhead from gc.freeze()'s effect.

    Installs the pause-instrumentation callback (via start_collector with
    gen2_interval_seconds=None) but does NOT call freeze(). If this matches
    baseline, the callback is innocent and freeze() itself is the culprit.
    If this matches freeze_only, the callback is paying the entire cost.
    """
    heap = _build_long_lived_heap()
    try:
        gc.collect()
        # No freeze(). Just install the callback for observation.
        start_collector(gen2_interval_seconds=None)
        try:
            samples = _run_assemble_loop(seg_200, _ITERS)
            pcts = _percentiles(samples)
            _print_distribution("callback_only_no_freeze", pcts)
            assert pcts["p50_us"] > 0
        finally:
            stop_collector()
    finally:
        del heap


def test_p999_freeze_only_no_callback(seg_200) -> None:
    """ADR-006 anomaly investigation: call gc.freeze() but do NOT install
    the callback.

    If this matches baseline, freeze() is innocent. If this matches
    freeze_only_no_timer (which has BOTH freeze and callback), freeze() is
    the culprit and the callback is innocent.
    """
    heap = _build_long_lived_heap()
    try:
        gc.collect()
        freeze()
        # No start_collector(). No callback installed.
        try:
            samples = _run_assemble_loop(seg_200, _ITERS)
            pcts = _percentiles(samples)
            _print_distribution("freeze_only_no_callback", pcts)
            assert pcts["p50_us"] > 0
        finally:
            unfreeze()
    finally:
        del heap


def test_p999_freeze_only_no_timer(seg_200) -> None:
    """Freeze + instrumentation callback, NO timer thread. The "no timer"
    side of the comparison the repeat-measurement benchmark resolved.

    Used by ``benchmarks/runs/step7_gc_tail_repeat.py`` to measure
    freeze-only's tail behavior across N=20 fresh subprocesses. Per
    ADR-006: freeze-only had a 35% spike rate vs the timer's 15%, and a
    higher worst-case max — hence the timer is now the default."""
    heap = _build_long_lived_heap()
    try:
        gc.collect()
        freeze()
        # Explicitly disable the timer for this scenario (vs default 0.5).
        start_collector(gen2_interval_seconds=None)
        try:
            samples = _run_assemble_loop(seg_200, _ITERS)
            pcts = _percentiles(samples)
            _print_distribution("freeze_only_no_timer", pcts)
            assert pcts["p50_us"] > 0
        finally:
            stop_collector()
            unfreeze()
    finally:
        del heap


def test_p999_freeze_plus_timer(seg_200) -> None:
    """Step 7 default: freeze + 500 ms gen-2 timer thread.

    Per ADR-006's N=20 repeat-measurement benchmark, this is the default
    Pyforge ships. 2.3x lower spike rate than freeze-only and 38% lower
    worst-case max in those measurements. The single-run versions of this
    benchmark are noisy — one run can look terrible (a single forced
    collect during a busy moment) while the next looks great. Use
    ``benchmarks/runs/step7_gc_tail_repeat.py`` for any decision-grade
    tail-latency comparison.
    """
    heap = _build_long_lived_heap()
    try:
        gc.collect()
        freeze()
        start_collector(gen2_interval_seconds=0.5)
        try:
            samples = _run_assemble_loop(seg_200, _ITERS)
            pcts = _percentiles(samples)
            _print_distribution("freeze_plus_timer (Step 7 design)", pcts)
            assert pcts["p50_us"] > 0
        finally:
            stop_collector()
            unfreeze()
    finally:
        del heap


@pytest.mark.parametrize("interval_seconds", [0.5, 1.0, 2.0, 5.0])
def test_p999_no_freeze_timer_interval_sweep(seg_200, interval_seconds: float) -> None:
    """Interval sweep on the SAFE callback+timer path (no freeze).

    Runs with the GC instrumentation callback + a gen-2 timer thread at
    interval_seconds, but without gc.freeze(). This is the combination
    we'd actually recommend to a user who opts into the timer, because
    the N=50 isolation result showed freeze + callback interact badly.

    Sweep range: 0.5, 1.0, 2.0, 5.0. The build plan defaulted to 0.5;
    longer intervals reduce timer overhead but allow gen-2 to grow
    larger between sweeps. The sweep finds the interval that gives the
    best med_p9999 without inflating spike rate.
    """
    heap = _build_long_lived_heap()
    try:
        gc.collect()
        # NO freeze() — we want the timer's effect on a clean callback path.
        start_collector(gen2_interval_seconds=interval_seconds)
        try:
            samples = _run_assemble_loop(seg_200, _ITERS)
            pcts = _percentiles(samples)
            _print_distribution(f"no_freeze_timer_interval_{interval_seconds}", pcts)
            assert pcts["p50_us"] > 0
        finally:
            stop_collector()
    finally:
        del heap


def test_p999_freeze_plus_timer_with_pressure(seg_200) -> None:
    """Same with continuous allocation pressure between iterations to keep
    gen-0/1 churning. The timer-thread design helps more here than on a
    tight low-allocation loop — but is still not the recommended default."""
    heap = _build_long_lived_heap()
    try:
        gc.collect()
        freeze()
        start_collector(gen2_interval_seconds=0.5)
        try:
            n = _ITERS
            samples = np.empty(n, dtype=np.int64)
            perf = time.perf_counter_ns
            for i in range(n):
                # Allocation pressure between iterations: ~200 short-lived
                # dicts per assemble. Stresses gen-0/1 collection scheduling.
                _churn = [{"k": j} for j in range(200)]
                t0 = perf()
                assemble(seg_200, "u")
                samples[i] = perf() - t0
                del _churn
            pcts = _percentiles(samples)
            _print_distribution("freeze_plus_timer + allocation pressure", pcts)
            assert pcts["p50_us"] > 0
        finally:
            stop_collector()
            unfreeze()
    finally:
        del heap
