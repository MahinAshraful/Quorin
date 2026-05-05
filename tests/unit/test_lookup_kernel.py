"""Unit tests for pyforge._internal.lookup_kernel.

Mirror of tests/unit/test_layout.py's `lookup` coverage but exercising
the Numba-compiled kernel + Python wrapper. Step 16c's parity property
test (tests/property/test_lookup_jit_parity.py) does the broader random
comparison; these unit tests are the smaller-grain coverage that's
easier to debug when a specific edge case fails.
"""

from __future__ import annotations

import sys

import numpy as np
import pytest

pytestmark = pytest.mark.skipif(
    sys.platform == "win32",
    reason="lookup_kernel requires POSIX (Linux/WSL2)",
)

from _helpers import make_segment, release_segment  # noqa: E402
from pyforge._internal.lookup_kernel import _lookup_core, lookup_jit, prewarm  # noqa: E402
from pyforge.layout import (  # noqa: E402
    SLOT_BYTES,
    SLOT_FEATURE_ROW_INDEX_OFFSET,
    SLOT_FLAGS_OFFSET,
    SLOT_ID_OFFSET_OFFSET,
    SLOT_NAME_HASH_OFFSET,
    insert,
    lookup,
)
from pyforge.schema import FeatureField, FeatureSchema, dtype  # noqa: E402

# ---------------------------------------------------------------------------
# Module-level prewarm so the first test doesn't pay ~100-200 ms compile cost.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module", autouse=True)
def _prewarm_lookup_kernel() -> None:
    prewarm()


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------


class _TwoFieldSchema(FeatureSchema):
    version = 1
    fields = [
        FeatureField("a", dtype.float32),
        FeatureField("b", dtype.int32),
    ]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_hit_first_probe_returns_correct_row_offset() -> None:
    """Single-entity hit; row_offset matches pyforge.layout.lookup byte-for-byte."""
    seg = make_segment(_TwoFieldSchema, capacity=64)
    try:
        row = bytes([0xAB]) * seg.layout.row_size
        insert(seg, "user_42", row)

        py_offset = lookup(seg, "user_42")
        jit_offset = lookup_jit(seg, "user_42")

        assert py_offset is not None
        assert jit_offset == py_offset
    finally:
        release_segment(seg)


def test_hit_after_collision_returns_correct_row_offset() -> None:
    """Two entities; both findable via byte-compare disambiguation.

    Mirrors test_layout.py's test_hash_collision_path_still_finds_correct_entity:
    forces collisions by using capacity=2 (slot_capacity=4) so probing is
    exercised. Both entities must remain findable.
    """
    seg = make_segment(_TwoFieldSchema, capacity=2)
    try:
        insert(seg, "alpha", bytes([0xA0]) * seg.layout.row_size)
        insert(seg, "beta", bytes([0xB0]) * seg.layout.row_size)

        for eid in ("alpha", "beta"):
            py = lookup(seg, eid)
            jit = lookup_jit(seg, eid)
            assert py is not None, f"Python lookup missed {eid!r}"
            assert jit == py, f"lookup_jit({eid!r}) = {jit} != python lookup {py}"
    finally:
        release_segment(seg)


def test_miss_empty_segment_returns_none() -> None:
    """Wrapper returns None on empty segment (first probed slot is EMPTY)."""
    seg = make_segment(_TwoFieldSchema, capacity=16)
    try:
        assert lookup_jit(seg, "missing") is None
    finally:
        release_segment(seg)


def test_miss_with_occupied_neighbors_returns_none() -> None:
    """Wrapper returns None when probe walks past occupied slots and hits EMPTY."""
    seg = make_segment(_TwoFieldSchema, capacity=64)
    try:
        for i in range(20):
            insert(seg, f"present_{i:04d}", bytes(seg.layout.row_size))
        assert lookup_jit(seg, "definitely_missing_42") is None
    finally:
        release_segment(seg)


def test_miss_walking_full_table_returns_none() -> None:
    """Capacity-cap-at-50% boundary: probe terminates within slot_capacity iters.

    Fill segment to exactly capacity (50% of slot_capacity); a miss for an
    ID that doesn't exist must still terminate via the EMPTY-slot stop
    condition. The bounded probe loop in _lookup_core is the safety net.
    """
    seg = make_segment(_TwoFieldSchema, capacity=8)
    try:
        for i in range(8):
            insert(seg, f"e_{i}", bytes(seg.layout.row_size))
        # 8 entities filled into a 16-slot table -> 8 EMPTY slots remain.
        # Any miss probe terminates within 16 iterations.
        assert lookup_jit(seg, "absent_xyz") is None
    finally:
        release_segment(seg)


def test_empty_entity_id_raises_value_error() -> None:
    """Wrapper raises ValueError BEFORE kernel call. Mirrors Python lookup."""
    seg = make_segment(_TwoFieldSchema, capacity=4)
    try:
        with pytest.raises(ValueError, match="non-empty"):
            lookup_jit(seg, "")
    finally:
        release_segment(seg)


def test_id_at_max_id_bytes_boundary() -> None:
    """64-byte UTF-8 ID succeeds (max_id_bytes default = 64)."""
    seg = make_segment(_TwoFieldSchema, capacity=4)
    try:
        eid = "x" * 64
        insert(seg, eid, bytes(seg.layout.row_size))
        py_off = lookup(seg, eid)
        jit_off = lookup_jit(seg, eid)
        assert jit_off is not None
        assert jit_off == py_off
    finally:
        release_segment(seg)


def test_id_longer_than_max_id_bytes_returns_none() -> None:
    """ID longer than max_id_bytes returns None (length-mismatch path).

    pyforge.layout.lookup walks the slot table; pool entries are length-
    prefixed and <= max_id_bytes, so pool_len == encoded_id_len always
    fails. lookup_jit must match this semantic: walk + return -1.
    """
    seg = make_segment(_TwoFieldSchema, capacity=4)
    try:
        # 100-byte ID; max_id_bytes default = 64. Insert a normal entity
        # so the slot table isn't empty (forces probe walk, not immediate
        # EMPTY-stop).
        insert(seg, "normal", bytes(seg.layout.row_size))
        oversized = "y" * 100

        py_result = lookup(seg, oversized)
        jit_result = lookup_jit(seg, oversized)
        assert py_result is None
        assert jit_result is None
    finally:
        release_segment(seg)


def test_unicode_multibyte_id_round_trip() -> None:
    """Multi-byte UTF-8 entity_id: byte-compare in kernel works because
    both sides encode identically.
    """
    seg = make_segment(_TwoFieldSchema, capacity=8)
    try:
        eid = "日本語_xyz"
        insert(seg, eid, bytes([0xCC]) * seg.layout.row_size)

        py_off = lookup(seg, eid)
        jit_off = lookup_jit(seg, eid)
        assert py_off is not None
        assert jit_off == py_off
    finally:
        release_segment(seg)


def test_kernel_returns_minus_one_sentinel_directly() -> None:
    """Direct _lookup_core call returns -1 on miss (kernel-only contract).

    The wrapper's `None` translation is a Python convenience; the kernel
    itself returns int64 with -1 as the miss sentinel.

    Helper-scoped per CLAUDE.md §7.5 mmap-view-scoping rule: ``seg_u8`` is
    a ``np.frombuffer`` view over the mmap'd buffer. Holding it in the
    test's local frame past the ``finally`` would make ``mmap.close()``
    raise BufferError. Extract into a helper that returns the primitive.
    """

    def _direct_kernel_call(seg) -> int:
        seg_u8 = np.frombuffer(seg.handle.buf, dtype=np.uint8)
        encoded = np.frombuffer(b"missing", dtype=np.uint8).copy()
        layout = seg.layout
        # Kernel signature post Step 16c-d: takes (encoded_id, encoded_id_len)
        # instead of (id_hash, encoded_id) — kernel computes hash internally
        # via blake2b_8.
        return int(
            _lookup_core(
                seg_u8,
                encoded,
                np.int64(len(b"missing")),
                layout.slot_table_offset,
                layout.slot_capacity,
                layout.string_pool_data_offset,
                layout.feature_rows_offset,
                layout.row_size,
                SLOT_BYTES,
                SLOT_NAME_HASH_OFFSET,
                SLOT_ID_OFFSET_OFFSET,
                SLOT_FLAGS_OFFSET,
                SLOT_FEATURE_ROW_INDEX_OFFSET,
            )
        )

    seg = make_segment(_TwoFieldSchema, capacity=4)
    try:
        assert _direct_kernel_call(seg) == -1
    finally:
        release_segment(seg)


def test_hit_against_real_mask_collision() -> None:
    """Two distinct entity_ids whose real BLAKE2b hashes collide on the
    low ``mask`` bits land on the same starting slot. Both must remain
    findable via the kernel's byte-compare disambiguation path.

    Step 16c-d note: full-hash collisions are 2^-64 (Hypothesis can't
    generate them, monkey-patching can't reach inside the @njit kernel
    that now computes BLAKE2b internally). But MASK collisions are
    common (1/slot_capacity ≈ 25% at slot_capacity=4) and exercise the
    same probe-loop + byte-compare path. Brute-force search finds a
    pair in tens of attempts.

    Mirrors Step 3's test_hash_collision_path_still_finds_correct_entity
    in spirit — both still findable through the linear probe.
    """
    from pyforge._internal.hash_id import hash_entity_id

    seg = make_segment(_TwoFieldSchema, capacity=2)  # slot_capacity = 4, mask = 3
    try:
        mask = seg.layout.slot_capacity - 1
        # Find two entity_ids whose hashes collide on mask. Linear search
        # over a small dictionary; ~25% per pair so tens of attempts
        # converges immediately. ASCII-only to keep the search fast.
        candidates = [f"u_{i:04d}" for i in range(50)]
        first_id = None
        second_id = None
        for i, a in enumerate(candidates):
            ha = hash_entity_id(a) & mask
            for b in candidates[i + 1 :]:
                if hash_entity_id(b) & mask == ha:
                    first_id, second_id = a, b
                    break
            if first_id is not None:
                break
        assert first_id is not None and second_id is not None, (
            "expected to find a mask-colliding pair in 50 candidates; this is "
            "a 25% per-pair probability so failure is statistically suspicious"
        )

        insert(seg, first_id, bytes([0x11]) * seg.layout.row_size)
        insert(seg, second_id, bytes([0x22]) * seg.layout.row_size)

        first_off = lookup_jit(seg, first_id)
        second_off = lookup_jit(seg, second_id)
        assert first_off is not None
        assert second_off is not None
        assert first_off != second_off

        first_data = bytes(seg.mmap_view[first_off : first_off + seg.layout.row_size])
        second_data = bytes(seg.mmap_view[second_off : second_off + seg.layout.row_size])
        assert first_data == bytes([0x11]) * seg.layout.row_size
        assert second_data == bytes([0x22]) * seg.layout.row_size
    finally:
        release_segment(seg)
