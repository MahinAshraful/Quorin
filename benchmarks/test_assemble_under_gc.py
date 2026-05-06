"""Step 16 P4: assemble p999 under realistic GC pressure.

DIFFERS from ``test_gc_p999.py``: that file measures GC pause durations in
isolation (ADR-006's freeze-vs-no-freeze decision archive). The Step 16
headline claim is "p999 ASSEMBLE LATENCY under GC pressure" — what an
operator sees in production, not the GC pause duration in milliseconds.

This file runs a tight assemble loop while a side thread allocates +
drops short-lived Python lists at a tunable rate (default 50 MB/s),
driving gen-0 collections roughly every 50 ms. We measure the assemble's
percentile distribution.

Methodology locks (per Step 16 plan §3.7):
  C1 — pedantic mode with rounds=10000. Plain ``benchmark(...)`` auto-
       calibrates to ~50-100 rounds; p999 from 100 samples is just max(),
       same statistical uselessness as Step 7's pre-orchestrator GC bench.
       10k rounds x 1 iter = 10k samples per Tier-1 run; Tier-2 N=20 x
       10k = 200k samples per scenario.
  L1 — deadline-based pressure thread. ``time.sleep(interval)`` after the
       burst alloc undershoots target rate (alloc time isn't subtracted).
       Deadline pattern keeps effective rate stable.
  C5 — Linux-only via module-level pytestmark.

Prewarm ordering (locked): the module-level autouse ``_prewarm`` fixture
runs BEFORE any test's GC-pressure thread starts. Numba JIT compile is
paid in module setup, never inside the timed loop. If this file is
refactored, preserve the order — module-scope autouse runs first, then
function-scope fixtures (including the GC-pressure thread inside the
test body itself).
"""

from __future__ import annotations

import sys
import threading
import time
from collections.abc import Iterator

import numpy as np
import pytest

# C5: this bench requires Linux-only deps (POSIX shm via fixtures, /dev/shm).
pytestmark = pytest.mark.skipif(
    sys.platform != "linux",
    reason="GC-under-pressure bench requires POSIX shm + /dev/shm",
)

from _helpers import make_segment, pack_row, release_segment  # noqa: E402
from quorin.assembly import assemble, prewarm  # noqa: E402
from quorin.layout import insert  # noqa: E402
from quorin.schema import FeatureField, FeatureSchema, dtype  # noqa: E402

# ---------------------------------------------------------------------------
# Module-scope autouse prewarm — runs ONCE per pytest module, BEFORE any
# test's GC-pressure thread starts. JIT compile cost is paid here, never
# in the timed loop. Preserve this ordering across refactors.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module", autouse=True)
def _prewarm_numba() -> None:
    prewarm()


class _Schema4Field(FeatureSchema):
    """Same shape as test_assembly_benchmark.py for direct comparison."""

    version = 1
    fields = [
        FeatureField("age", dtype.int32),
        FeatureField("clicks", dtype.int64),
        FeatureField("ltv", dtype.float64),
        FeatureField("score", dtype.float32),
    ]


@pytest.fixture
def seg_4_field() -> Iterator:
    seg = make_segment(_Schema4Field, capacity=64)
    values = {
        "age": np.array([42], dtype=np.int32),
        "clicks": np.array([1_234_567], dtype=np.int64),
        "ltv": np.array([987.65], dtype=np.float64),
        "score": np.array([0.5], dtype=np.float32),
    }
    insert(seg, "u", pack_row(_Schema4Field, values))
    yield seg
    release_segment(seg)


def _gc_pressure_thread(stop_event: threading.Event, alloc_rate_mb_per_sec: int) -> None:
    """Allocate + drop bytearrays at a steady rate to drive gen-0 collections.

    L1 fix: deadline-based loop. ``time.sleep(interval)`` after the burst
    alloc undershoots the target rate because alloc time isn't subtracted.
    Deadline pattern keeps effective rate close to nominal even as system
    load rises.
    """
    chunk_size = 1024  # bytes per element
    elements_per_burst = 1024  # 1 MiB per burst
    bursts_per_sec = alloc_rate_mb_per_sec
    interval = 1.0 / bursts_per_sec
    next_t = time.monotonic() + interval
    while not stop_event.is_set():
        # Allocate then drop; the bytearray comprehension is eligible for
        # gen-0 collection on next loop iteration.
        _ = [bytearray(chunk_size) for _ in range(elements_per_burst)]
        del _
        now = time.monotonic()
        time.sleep(max(0.0, next_t - now))
        next_t += interval
        # Don't accumulate debt if alloc time exceeds interval (system overload).
        if next_t < now:
            next_t = now + interval


def test_bench_assemble_p999_under_gc_pressure_4_field(benchmark, seg_4_field) -> None:
    """Headline claim: p999 assemble latency under realistic GC pressure.

    Tier-1 gates p99 + p999 (both fields in tier1.yml, since rounds=10000
    gives p999 resolution within a single Tier-1 run). Tier-2 N=20 = 200k
    samples per scenario for the README-quote-grade p999 number.

    NOT the same as test_gc_p999 (which measures pause durations directly).
    """
    seg = seg_4_field
    stop = threading.Event()
    pressure = threading.Thread(
        target=_gc_pressure_thread, args=(stop, 50), daemon=True, name="gc-pressure-50mbps"
    )
    pressure.start()
    try:
        # Warmup pressure for 200ms so the GC is in steady state before we time.
        time.sleep(0.2)
        # C1: pedantic mode + rounds=10000 for p999 resolution. The bench file
        # is named *p999* explicitly so the percentile semantic is visible at
        # call site.
        benchmark.pedantic(
            assemble,
            args=(seg, "u"),
            rounds=10000,
            iterations=1,
            warmup_rounds=10,
        )
    finally:
        stop.set()
        pressure.join(timeout=2.0)
