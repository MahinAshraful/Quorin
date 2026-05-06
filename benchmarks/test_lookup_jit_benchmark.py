"""Benchmarks for quorin._internal.lookup_kernel.lookup_jit (Step 16c).

Direct head-to-head against test_layout_benchmark.py's pure-Python
lookup benches. The Step 16c trip-wire bench is
test_bench_assemble_4_field_warm_numba — these benches isolate the
lookup-only cost so a future regression in EITHER the Numba kernel OR
the wrapper's Python overhead surfaces independently of the assemble
kernel.

ADR-017 records the measured speedup. No tier1.yml gate yet — Commit B
adds one if the speedup is meaningful (>= 1.5x).
"""

from __future__ import annotations

import os
import struct
import sys
import uuid

import pytest

pytestmark = pytest.mark.skipif(
    sys.platform == "win32",
    reason="lookup_kernel requires POSIX (Linux/WSL2)",
)

from quorin._internal import posix_shm  # noqa: E402
from quorin._internal.crc import crc32_of_bytes  # noqa: E402
from quorin._internal.lookup_kernel import lookup_jit, prewarm  # noqa: E402
from quorin.layout import (  # noqa: E402
    compute_layout,
    initialize_segment_regions,
    insert,
)
from quorin.schema import (  # noqa: E402
    FeatureField,
    FeatureSchema,
    compile_schema,
    dtype,
)
from quorin.shm import HEADER_FMT, HEADER_LEN, MAGIC, Segment  # noqa: E402


@pytest.fixture(scope="module", autouse=True)
def _prewarm_lookup_kernel() -> None:
    """First-call compile cost is paid in setup, not measurement."""
    prewarm()


class _TwoFieldSchema(FeatureSchema):
    version = 1
    fields = [
        FeatureField("a", dtype.float32),
        FeatureField("b", dtype.int32),
    ]


def _make_segment(capacity: int = 1024) -> Segment:
    layout = compute_layout(_TwoFieldSchema, capacity=capacity)
    name = f"quorin_lookup_jit_bench_{os.getpid()}_{uuid.uuid4().hex[:8]}"
    handle = posix_shm.create(name, layout.total_size)
    crc = crc32_of_bytes(compile_schema(_TwoFieldSchema).tobytes())
    handle.buf[:HEADER_LEN] = struct.pack(
        HEADER_FMT, MAGIC, int(_TwoFieldSchema.version), crc, capacity
    )
    initialize_segment_regions(handle.buf, layout)
    return Segment(name=name, schema=_TwoFieldSchema, handle=handle, layout=layout)


def _release(seg: Segment) -> None:
    posix_shm.close(seg.handle)
    posix_shm.unlink(seg.name)


@pytest.fixture
def populated_segment() -> Segment:
    """Mirrors test_layout_benchmark.py::populated_segment so the head-to-head
    comparison is apples-to-apples."""
    seg = _make_segment(capacity=1024)
    row = bytes(seg.layout.row_size)
    for i in range(500):
        insert(seg, f"user_{i:06d}", row)
    yield seg
    _release(seg)


def test_bench_lookup_jit_hit_first_probe(benchmark, populated_segment: Segment) -> None:
    """Single lookup that lands on its home slot (first probe).

    Direct head-to-head with test_bench_lookup_hit_first_probe (Python).
    """
    seg = populated_segment
    target = "user_000042"
    # Verify it's there before measuring.
    assert lookup_jit(seg, target) is not None
    result = benchmark(lookup_jit, seg, target)
    assert result is not None


def test_bench_lookup_jit_miss(benchmark, populated_segment: Segment) -> None:
    """Lookup on a non-existent ID — walks until empty slot.

    Direct head-to-head with test_bench_lookup_miss (Python).
    """
    seg = populated_segment
    target = "definitely_does_not_exist_42"
    result = benchmark(lookup_jit, seg, target)
    assert result is None
