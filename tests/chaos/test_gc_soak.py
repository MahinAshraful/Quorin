"""Soak test for gc_manager — assert no memory leak under sustained allocation.

10-second soak (3 s warm-up + 7 s sampling at 0.25 s intervals = ~28 RSS
samples). Tight enough to fit the project's <10 s CI budget convention
(Step 2's soak was similarly capped); long enough that a linear-regression
slope on the RSS samples is statistically meaningful.

**Why a linear-regression assertion, not absolute delta:** CI runners share
memory with whatever else is on the host. Absolute RSS at the end of the
soak can wobble by a few MB for reasons that have nothing to do with our
code. The slope is robust — it's measuring the *trend*, not the noise.

A longer 60-second soak is desirable but exceeds the default-suite budget;
to run a deeper soak locally, increase ``WARMUP_SECONDS`` and
``SOAK_SECONDS`` below.
"""

from __future__ import annotations

import sys
import time

import pytest

pytestmark = [
    pytest.mark.slow,
    pytest.mark.chaos,
    pytest.mark.skipif(
        sys.platform == "win32",
        reason="psutil RSS semantics differ on Windows; soak runs on Linux/WSL only",
    ),
]

from quorin._internal.gc_manager import (  # noqa: E402
    freeze,
    start_collector,
    stop_collector,
    unfreeze,
)

WARMUP_SECONDS = 3.0
SOAK_SECONDS = 7.0
SAMPLE_INTERVAL_SECONDS = 0.25


def _churn(n: int) -> None:
    """Allocate-and-drop short-lived objects to exercise gen-0/1.

    The list comprehension is dropped immediately on function return so all
    objects are gen-0 garbage. Repeated invocations let some survive the
    first sweep and promote.
    """
    _ = [{"i": i, "payload": [0.0] * 10} for i in range(n)]


def _linreg_slope(samples: list[int]) -> float:
    """Least-squares slope of `samples` over their indices, in units of
    ``samples-units`` per ``index-step``. Returns 0 for fewer than 2 points.

    For our soak, samples are RSS in bytes and indices are 0.5-second ticks.
    Slope of 100_000 means RSS growing by 100 KB per sample (~200 KB/s).
    """
    n = len(samples)
    if n < 2:
        return 0.0
    sum_x = (n - 1) * n / 2.0
    sum_y = float(sum(samples))
    sum_xx = sum(i * i for i in range(n))
    sum_xy = float(sum(i * y for i, y in enumerate(samples)))
    denominator = n * sum_xx - sum_x * sum_x
    if denominator == 0.0:
        return 0.0
    return (n * sum_xy - sum_x * sum_y) / denominator


def test_gc_manager_no_memory_leak_under_short_soak() -> None:
    """~10-second soak. RSS slope < 100 KB/sample (~400 KB/s at 0.25 s
    intervals).

    A genuine leak (zombie thread accumulation, callback bug retaining
    references) would show a clearly-positive slope; steady-state Python
    allocation is bounded and converges within the warm-up window.
    """
    psutil = pytest.importorskip("psutil")
    process = psutil.Process()

    # Opt into the timer for this soak test specifically — we want to
    # exercise the timer-thread path, not just the callback. Production
    # default for Quorin serving paths is no timer (see ADR-006).
    #
    # Order matters: freeze() BEFORE start_collector() avoids the
    # documented callback+freeze interaction (ADR-006). This also avoids
    # tripping the UserWarning that freeze() emits when called with the
    # callback already installed.
    freeze()
    start_collector(gen2_interval_seconds=0.1)
    try:
        warmup_deadline = time.monotonic() + WARMUP_SECONDS
        while time.monotonic() < warmup_deadline:
            _churn(10_000)

        baseline_rss = process.memory_info().rss
        samples: list[int] = []
        soak_deadline = time.monotonic() + SOAK_SECONDS
        while time.monotonic() < soak_deadline:
            _churn(10_000)
            samples.append(process.memory_info().rss)
            time.sleep(SAMPLE_INTERVAL_SECONDS)

        assert len(samples) > 10, "soak loop must produce enough samples to fit a slope"

        slope_bytes_per_sample = _linreg_slope(samples)
        per_second = slope_bytes_per_sample / SAMPLE_INTERVAL_SECONDS
        # 100 KB/sample = ~400 KB/s growth — anything below this is well
        # within steady-state allocator wobble; anything above is real.
        assert slope_bytes_per_sample < 100_000, (
            f"RSS growing at {slope_bytes_per_sample:.0f} bytes/sample "
            f"(~{per_second / 1024:.0f} KB/s); suspected leak. "
            f"baseline={baseline_rss}, final={samples[-1]}, n_samples={len(samples)}"
        )
    finally:
        unfreeze()
        stop_collector()
