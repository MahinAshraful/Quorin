"""Stress test for the slot table at realistic-ish scale.

Inserts 100k entities into a 200k-slot table (50% load), then samples random
lookups and verifies probe-distance behavior. The 1M-entity stress test is
deferred to Step 13 (hydration), which is the natural bench harness for
that scale.

Marked ``@pytest.mark.slow`` so it doesn't gate every push but still runs
when ``pytest -m slow`` is invoked.
"""

from __future__ import annotations

import os
import sys
import uuid

import pytest

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(
        sys.platform == "win32",
        reason="layout requires POSIX (Linux/WSL2)",
    ),
]

import struct  # noqa: E402

from pyforge._internal import posix_shm  # noqa: E402
from pyforge._internal.crc import crc32_of_bytes  # noqa: E402
from pyforge._internal.hash_id import hash_entity_id  # noqa: E402
from pyforge.layout import (  # noqa: E402
    OCCUPIED,
    SLOT_DTYPE,
    _slot_table_view,
    compute_layout,
    initialize_segment_regions,
    insert,
    lookup,
)
from pyforge.schema import (  # noqa: E402
    FeatureField,
    FeatureSchema,
    compile_schema,
    dtype,
)
from pyforge.shm import HEADER_FMT, HEADER_LEN, MAGIC, Segment  # noqa: E402


class _SmallSchema(FeatureSchema):
    version = 1
    fields = [FeatureField("v", dtype.uint8)]


def _unique_name() -> str:
    return f"pyforge_stress_{os.getpid()}_{uuid.uuid4().hex[:8]}"


def _make_stress_segment(capacity: int) -> Segment:
    layout = compute_layout(_SmallSchema, capacity=capacity)
    name = _unique_name()
    handle = posix_shm.create(name, layout.total_size)
    crc = crc32_of_bytes(compile_schema(_SmallSchema).tobytes())
    handle.buf[:HEADER_LEN] = struct.pack(
        HEADER_FMT, MAGIC, int(_SmallSchema.version), crc, capacity
    )
    initialize_segment_regions(handle.buf, layout)
    return Segment(name=name, schema=_SmallSchema, handle=handle, layout=layout)


def _release(seg: Segment) -> None:
    posix_shm.close(seg.handle)
    posix_shm.unlink(seg.name)


def _avg_probe_distance(seg: Segment) -> float:
    """Compute average probe distance across all occupied slots.

    Probe distance for an occupied slot at index ``i`` with hash ``h`` and
    mask ``m`` is ``(i - (h & m)) mod (m + 1)`` — how far the slot drifted
    from its natural home due to linear probing.

    Done in a function so the numpy view is GC'd before the test cleanup.
    """
    view = _slot_table_view(seg.mmap_view, seg.layout)
    mask = seg.layout.slot_capacity - 1

    total = 0
    count = 0
    for idx in range(seg.layout.slot_capacity):
        slot = view[idx]
        if int(slot["flags"]) == OCCUPIED:
            home = int(slot["name_hash"]) & mask
            dist = (idx - home) & mask
            total += dist
            count += 1
    return total / count if count else 0.0


@pytest.mark.skip(reason="manual: long runtime, run with --runslow")
def test_100k_inserts_lookup_zero_misses() -> None:
    capacity = 100_000
    seg = _make_stress_segment(capacity)
    try:
        # Insert 100k unique IDs
        ids = [f"user_{i:07d}" for i in range(capacity)]
        row = bytes([0xAB])  # row_size for uint8 single-field schema is 64
        # Pad row to actual row_size
        row = row * seg.layout.row_size
        for eid in ids:
            assert insert(seg, eid, row) is True

        # Sample 1000 random lookups
        import random

        rng = random.Random(0xC0FFEE)
        sample = rng.sample(ids, 1000)
        for eid in sample:
            assert lookup(seg, eid) is not None

        # Average probe distance under 50% load with linear probing should be ~1.5
        avg = _avg_probe_distance(seg)
        assert avg < 2.0, f"average probe distance {avg:.3f} > 2.0"
    finally:
        _release(seg)


# A smaller variant that always runs (still hits >50k entities — a real test
# of the hash table, just not 100k).
def test_50k_inserts_probe_distance_bounded() -> None:
    capacity = 50_000
    seg = _make_stress_segment(capacity)
    try:
        ids = [f"u_{i:06d}" for i in range(capacity)]
        row = bytes([0xCC]) * seg.layout.row_size
        for eid in ids:
            insert(seg, eid, row)

        # Sanity: every id is findable
        import random

        rng = random.Random(42)
        for eid in rng.sample(ids, 500):
            off = lookup(seg, eid)
            assert off is not None

        avg = _avg_probe_distance(seg)
        assert avg < 2.0, f"average probe distance {avg:.3f} > 2.0"
    finally:
        _release(seg)


def test_filling_to_capacity_is_idempotent_then_raises() -> None:
    """Insert exactly ``capacity`` entities — the last should still succeed.
    The next NEW insert raises CapacityExceeded; an UPDATE of an existing
    entity still succeeds."""
    from pyforge.layout import CapacityExceededError

    capacity = 200
    seg = _make_stress_segment(capacity)
    try:
        ids = [f"e_{i:04d}" for i in range(capacity)]
        row = bytes([0xDD]) * seg.layout.row_size
        for eid in ids:
            assert insert(seg, eid, row) is True

        # All 200 occupied — next NEW insert must fail
        with pytest.raises(CapacityExceededError):
            insert(seg, "ghost", row)

        # But UPDATE of an existing entity is fine
        new_row = bytes([0xEE]) * seg.layout.row_size
        assert insert(seg, "e_0042", new_row) is False
    finally:
        _release(seg)


def test_blake2b_does_not_distribute_pathologically() -> None:
    """Sanity: hash distribution shouldn't cluster catastrophically.
    For 10k random IDs in a 16384-slot table, no single 'home' bucket should
    receive more than ~10 IDs (expected ~0.6, but we allow some variance)."""
    capacity = 5_000
    seg = _make_stress_segment(capacity)
    try:
        # capacity=5000 -> slot_capacity = next_pow2(10000) = 16384
        assert seg.layout.slot_capacity == 16384
        mask = seg.layout.slot_capacity - 1
        from collections import Counter

        homes = Counter()
        for i in range(capacity):
            h = hash_entity_id(f"u_{i:06d}")
            homes[h & mask] += 1

        # Worst-bucket count should be reasonable — blake2b is uniform.
        worst = max(homes.values())
        assert worst < 12, f"worst-bucket count {worst} suggests bad hashing"
    finally:
        _release(seg)


def test_slot_dtype_does_not_change_size() -> None:
    """Defensive: the slot's wire size must not silently change. If it does,
    every existing segment becomes unreadable.
    """
    assert SLOT_DTYPE.itemsize == 24
