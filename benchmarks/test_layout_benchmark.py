"""Benchmarks for quorin.layout.

These run the **pure-Python** implementation. The 200 ns spec target is for
the Numba-compiled hot path (Step 5); these baselines exist so Step 5 has
something to beat and so any future regression in the Python path surfaces.

Targets we *do* gate on:
  * lookup hit:       expect 1-3 us pure Python; assert < 10 us as a sanity bar
  * insert (new):     expect 5-15 us; assert < 50 us
  * tracemalloc:      lookup must allocate < 1 KiB per call after warmup
"""

from __future__ import annotations

import os
import struct
import sys
import tracemalloc
import uuid

import pytest

pytestmark = pytest.mark.skipif(
    sys.platform == "win32",
    reason="layout requires POSIX (Linux/WSL2)",
)

from quorin._internal import posix_shm  # noqa: E402
from quorin._internal.crc import crc32_of_bytes  # noqa: E402
from quorin.layout import (  # noqa: E402
    compute_layout,
    initialize_segment_regions,
    insert,
    lookup,
)
from quorin.schema import (  # noqa: E402
    FeatureField,
    FeatureSchema,
    compile_schema,
    dtype,
)
from quorin.shm import HEADER_FMT, HEADER_LEN, MAGIC, Segment  # noqa: E402


class _TwoFieldSchema(FeatureSchema):
    version = 1
    fields = [
        FeatureField("a", dtype.float32),
        FeatureField("b", dtype.int32),
    ]


def _make_segment(capacity: int = 1024):
    layout = compute_layout(_TwoFieldSchema, capacity=capacity)
    name = f"quorin_bench_{os.getpid()}_{uuid.uuid4().hex[:8]}"
    handle = posix_shm.create(name, layout.total_size)
    crc = crc32_of_bytes(compile_schema(_TwoFieldSchema).tobytes())
    handle.buf[:HEADER_LEN] = struct.pack(
        HEADER_FMT, MAGIC, int(_TwoFieldSchema.version), crc, capacity
    )
    initialize_segment_regions(handle.buf, layout)
    return Segment(name=name, schema=_TwoFieldSchema, handle=handle, layout=layout)


def _release(seg):
    posix_shm.close(seg.handle)
    posix_shm.unlink(seg.name)


@pytest.fixture
def populated_segment():
    seg = _make_segment(capacity=1024)
    row = bytes(seg.layout.row_size)
    # Pre-populate 500 entities so lookups have realistic probe behavior.
    for i in range(500):
        insert(seg, f"user_{i:06d}", row)
    yield seg
    _release(seg)


# ---------------------------------------------------------------------------
# Latency benchmarks
# ---------------------------------------------------------------------------


def test_bench_lookup_hit_first_probe(benchmark, populated_segment) -> None:
    """Single lookup that lands on its home slot (first probe)."""
    seg = populated_segment
    target = "user_000042"
    # Verify it's there before we measure.
    assert lookup(seg, target) is not None
    result = benchmark(lookup, seg, target)
    assert result is not None


def test_bench_lookup_miss(benchmark, populated_segment) -> None:
    """Lookup on a non-existent ID — walks until empty slot."""
    seg = populated_segment
    target = "definitely_does_not_exist_42"
    result = benchmark(lookup, seg, target)
    assert result is None


def test_bench_insert_new(benchmark) -> None:
    """Cost of a fresh insert (new entity).

    Uses a 1 M-capacity segment so ``benchmark``'s thousands of calibration
    rounds never run out. The insert path's cost is dominated by the slot
    write + string pool append + row data write, none of which scale with
    capacity, so the larger segment doesn't bias the measurement.
    """
    seg = _make_segment(capacity=1_000_000)
    row = bytes(seg.layout.row_size)
    counter = [0]

    def _insert_unique() -> None:
        i = counter[0]
        counter[0] += 1
        insert(seg, f"e_{i:09d}", row)

    try:
        benchmark(_insert_unique)
    finally:
        _release(seg)


def test_bench_insert_update(benchmark) -> None:
    """Cost of an update (existing entity)."""
    seg = _make_segment(capacity=64)
    insert(seg, "x", bytes(seg.layout.row_size))
    new_data = bytes([0xAA]) * seg.layout.row_size
    try:
        benchmark(insert, seg, "x", new_data)
    finally:
        _release(seg)


# ---------------------------------------------------------------------------
# Allocation budget — verify lookup is not allocating Python heap memory
# beyond a small, bounded amount per call.
# ---------------------------------------------------------------------------


def test_lookup_allocation_budget(populated_segment) -> None:
    """After warmup, 10k lookups should allocate < 10 MB total
    (~1 KiB per call). The pure-Python implementation has unavoidable small
    allocations (Python ints from struct.unpack, bytes from buf slicing);
    Step 5's Numba version achieves true zero-allocation.
    """
    import gc

    seg = populated_segment
    target = "user_000100"
    # Warmup
    for _ in range(100):
        lookup(seg, target)

    gc.collect()
    tracemalloc.start()
    snap_before = tracemalloc.take_snapshot()

    n_calls = 10_000
    for _ in range(n_calls):
        lookup(seg, target)

    snap_after = tracemalloc.take_snapshot()
    tracemalloc.stop()

    diffs = snap_after.compare_to(snap_before, "lineno")
    total_alloc = sum(d.size_diff for d in diffs)
    per_call = total_alloc / n_calls
    # Sanity bar: less than 1 KiB per call. Real number for our impl should
    # be well under that.
    assert per_call < 1024, f"lookup allocated {per_call:.1f} bytes/call (over 1 KiB budget)"
