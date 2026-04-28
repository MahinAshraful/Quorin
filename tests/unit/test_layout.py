"""Unit tests for pyforge.layout.

Most tests bypass Redis by creating a Segment directly from a POSIX shm
handle + a manually-computed SegmentLayout. This lets layout tests run on
any POSIX box without docker-compose. The integration test in
``tests/integration/test_shm_redis.py`` still exercises the full
``SegmentRegistry.create -> initialize_segment_regions`` path.
"""

from __future__ import annotations

import os
import struct
import sys
import uuid

import pytest

pytestmark = pytest.mark.skipif(
    sys.platform == "win32",
    reason="layout requires POSIX (Linux/WSL2)",
)

from pyforge._internal import posix_shm  # noqa: E402
from pyforge._internal.hash_id import hash_entity_id  # noqa: E402
from pyforge.layout import (  # noqa: E402
    DATA_REGION_OFFSET,
    DEFAULT_MAX_ID_BYTES,
    EMPTY,
    METADATA_OFFSET,
    OCCUPIED,
    SLOT_DTYPE,
    STRING_POOL_HEADER_BYTES,
    CapacityExceededError,
    StringPoolExhaustedError,
    _next_pow2,
    _slot_table_view,
    compute_layout,
    initialize_segment_regions,
    insert,
    insert_many,
    iterate_occupied,
    lookup,
    total_segment_size,
)
from pyforge.schema import (  # noqa: E402
    CACHE_LINE_SIZE,
    PAGE_SIZE,
    FeatureField,
    FeatureSchema,
    compute_row_offset_table,
    dtype,
    row_size,
)
from pyforge.shm import HEADER_FMT, HEADER_LEN, MAGIC, Segment  # noqa: E402

# ---------------------------------------------------------------------------
# Test schemas
# ---------------------------------------------------------------------------


class _TwoFieldSchema(FeatureSchema):
    version = 1
    fields = [
        FeatureField("a", dtype.float32),
        FeatureField("b", dtype.int32),
    ]


class _EmbeddingSchema(FeatureSchema):
    version = 1
    fields = [
        FeatureField("x", dtype.float32),
        FeatureField("emb", dtype.float32, shape=(128,)),
    ]


# ---------------------------------------------------------------------------
# Test fixtures — make a Segment without going through SegmentRegistry/Redis.
# ---------------------------------------------------------------------------


def _unique_name(prefix: str = "pyforge_layout_test") -> str:
    return f"{prefix}_{os.getpid()}_{uuid.uuid4().hex[:8]}"


def _make_segment(
    schema: type[FeatureSchema],
    capacity: int,
    *,
    max_id_bytes: int = DEFAULT_MAX_ID_BYTES,
) -> Segment:
    """Allocate and initialize a Segment without involving SegmentRegistry."""
    layout = compute_layout(schema, capacity=capacity, max_id_bytes=max_id_bytes)
    name = _unique_name()
    handle = posix_shm.create(name, layout.total_size)

    # Write the primary header so _verify_header would accept it (some tests
    # don't need this, but keeping it consistent saves friction).
    from pyforge._internal.crc import crc32_of_bytes
    from pyforge.schema import compile_schema

    crc = crc32_of_bytes(compile_schema(schema).tobytes())
    header = struct.pack(HEADER_FMT, MAGIC, int(schema.version), crc, capacity)
    handle.buf[:HEADER_LEN] = header

    initialize_segment_regions(handle.buf, layout)
    return Segment(name=name, schema=schema, handle=handle, layout=layout)


def _release(seg: Segment) -> None:
    posix_shm.close(seg.handle)
    posix_shm.unlink(seg.name)


@pytest.fixture
def two_field_segment():
    seg = _make_segment(_TwoFieldSchema, capacity=64)
    yield seg
    _release(seg)


@pytest.fixture
def embedding_segment():
    seg = _make_segment(_EmbeddingSchema, capacity=16)
    yield seg
    _release(seg)


# ---------------------------------------------------------------------------
# compute_layout / total_segment_size — pure math, no segment needed
# ---------------------------------------------------------------------------


class TestComputeLayout:
    def test_zero_capacity_rejected(self) -> None:
        with pytest.raises(ValueError, match="capacity"):
            compute_layout(_TwoFieldSchema, capacity=0)

    def test_negative_capacity_rejected(self) -> None:
        with pytest.raises(ValueError, match="capacity"):
            compute_layout(_TwoFieldSchema, capacity=-5)

    def test_zero_max_id_bytes_rejected(self) -> None:
        with pytest.raises(ValueError, match="max_id_bytes"):
            compute_layout(_TwoFieldSchema, capacity=10, max_id_bytes=0)

    def test_capacity_overflow_rejected(self) -> None:
        with pytest.raises(ValueError, match="uint32"):
            compute_layout(_TwoFieldSchema, capacity=2**32)

    def test_capacity_one_works(self) -> None:
        layout = compute_layout(_TwoFieldSchema, capacity=1)
        # next_pow2(2) = 2
        assert layout.slot_capacity == 2
        assert layout.capacity == 1

    def test_slot_capacity_is_power_of_two(self) -> None:
        for cap in (1, 2, 7, 100, 1000, 99_999):
            layout = compute_layout(_TwoFieldSchema, capacity=cap)
            n = layout.slot_capacity
            assert n & (n - 1) == 0, f"slot_capacity {n} is not a power of 2"
            assert n >= cap * 2

    def test_all_region_offsets_cache_aligned(self) -> None:
        layout = compute_layout(_EmbeddingSchema, capacity=128)
        for offset in (
            layout.slot_table_offset,
            layout.string_pool_offset,
            layout.feature_rows_offset,
        ):
            assert offset % CACHE_LINE_SIZE == 0

    def test_total_size_is_page_aligned(self) -> None:
        layout = compute_layout(_EmbeddingSchema, capacity=128)
        assert layout.total_size % PAGE_SIZE == 0

    def test_regions_do_not_overlap(self) -> None:
        layout = compute_layout(_EmbeddingSchema, capacity=128)
        # slot_table_end <= string_pool_offset
        assert layout.slot_table_offset + layout.slot_table_bytes <= layout.string_pool_offset
        # string_pool_end <= feature_rows_offset
        string_pool_end = (
            layout.string_pool_offset + STRING_POOL_HEADER_BYTES + layout.string_pool_data_bytes
        )
        assert string_pool_end <= layout.feature_rows_offset
        # feature_rows_end <= total_size
        assert layout.feature_rows_offset + layout.feature_rows_bytes <= layout.total_size

    def test_row_offset_table_is_zero_relative(self) -> None:
        layout = compute_layout(_EmbeddingSchema, capacity=64)
        # First field's row_offset must be 0; others are aligned to 64.
        assert int(layout.row_offset_table["byte_offset"].min()) == 0
        for off in layout.row_offset_table["byte_offset"]:
            assert int(off) % CACHE_LINE_SIZE == 0

    def test_total_segment_size_matches_layout(self) -> None:
        size = total_segment_size(_TwoFieldSchema, capacity=100)
        layout = compute_layout(_TwoFieldSchema, capacity=100)
        assert size == layout.total_size


class TestNextPow2:
    @pytest.mark.parametrize(
        "n,expected",
        [(0, 1), (1, 1), (2, 2), (3, 4), (5, 8), (8, 8), (9, 16), (1023, 1024)],
    )
    def test_next_pow2_values(self, n: int, expected: int) -> None:
        assert _next_pow2(n) == expected


# ---------------------------------------------------------------------------
# Schema row layout (Step 1 extension)
# ---------------------------------------------------------------------------


class TestRowOffsetTable:
    def test_first_field_at_row_offset_zero(self) -> None:
        table = compute_row_offset_table(_TwoFieldSchema)
        assert int(table["byte_offset"].min()) == 0

    def test_columns_match_offset_table(self) -> None:
        table = compute_row_offset_table(_TwoFieldSchema)
        assert table.dtype.names == (
            "name_hash",
            "byte_offset",
            "dtype_code",
            "element_count",
            "byte_count",
        )

    def test_row_size_two_scalars_is_one_cache_line(self) -> None:
        # field a: row_offset=0, byte_count=4
        # field b: row_offset=64 (next cache line), byte_count=4
        # row_size = align_up(64+4, 64) = 128
        assert row_size(_TwoFieldSchema) == 128

    def test_row_size_for_embedding_schema(self) -> None:
        # field x: row_offset=0, byte_count=4
        # field emb: row_offset=64, byte_count=512
        # row_size = align_up(64+512, 64) = 576 + pad to 64 -> 576
        # 576/64 = 9, so 576 is already aligned
        rs = row_size(_EmbeddingSchema)
        assert rs % CACHE_LINE_SIZE == 0
        assert rs >= 64 + 512


# ---------------------------------------------------------------------------
# Insert / lookup happy paths
# ---------------------------------------------------------------------------


def _row_data(seg: Segment, fill_byte: int = 0xAA) -> bytes:
    return bytes([fill_byte]) * seg.layout.row_size


class TestInsertLookup:
    def test_insert_then_lookup_returns_offset(self, two_field_segment) -> None:
        seg = two_field_segment
        new_data = _row_data(seg, 0xAB)

        result = insert(seg, "user_001", new_data)
        assert result is True

        offset = lookup(seg, "user_001")
        assert offset is not None
        # The returned offset points at the start of the row in feature_rows
        assert offset >= seg.layout.feature_rows_offset
        assert offset < seg.layout.feature_rows_offset + seg.layout.feature_rows_bytes

        # And the bytes there match what we wrote
        back = bytes(seg.mmap_view[offset : offset + seg.layout.row_size])
        assert back == new_data

    def test_lookup_missing_returns_none(self, two_field_segment) -> None:
        assert lookup(two_field_segment, "ghost") is None

    def test_insert_many_distinct_findable(self, two_field_segment) -> None:
        seg = two_field_segment
        ids = [f"user_{i:03d}" for i in range(20)]
        for i, eid in enumerate(ids):
            assert insert(seg, eid, bytes([i % 256]) * seg.layout.row_size) is True

        for i, eid in enumerate(ids):
            off = lookup(seg, eid)
            assert off is not None
            back = bytes(seg.mmap_view[off : off + seg.layout.row_size])
            assert back == bytes([i % 256]) * seg.layout.row_size

    def test_insert_same_id_twice_updates_in_place(self, two_field_segment) -> None:
        seg = two_field_segment
        insert(seg, "x", bytes([0x11]) * seg.layout.row_size)
        result = insert(seg, "x", bytes([0x22]) * seg.layout.row_size)
        assert result is False  # update, not new

        # Only one slot should be occupied
        slot_table = _slot_table_view(seg.mmap_view, seg.layout)
        occupied = int((slot_table["flags"] == OCCUPIED).sum())
        assert occupied == 1

        off = lookup(seg, "x")
        back = bytes(seg.mmap_view[off : off + seg.layout.row_size])
        assert back == bytes([0x22]) * seg.layout.row_size

    def test_iterate_occupied_yields_all_inserted(self, two_field_segment) -> None:
        seg = two_field_segment
        ids = {f"e_{i}" for i in range(15)}
        for eid in ids:
            insert(seg, eid, _row_data(seg))

        seen = {entity_id for entity_id, _ in iterate_occupied(seg)}
        assert seen == ids

    def test_iterate_occupied_empty(self, two_field_segment) -> None:
        assert list(iterate_occupied(two_field_segment)) == []


class TestInsertEdgeCases:
    def test_empty_entity_id_rejected(self, two_field_segment) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            insert(two_field_segment, "", _row_data(two_field_segment))

    def test_wrong_row_data_length_rejected(self, two_field_segment) -> None:
        with pytest.raises(ValueError, match="row_size"):
            insert(two_field_segment, "x", b"too short")

    def test_too_long_entity_id_rejected(self, two_field_segment) -> None:
        too_long = "x" * (two_field_segment.layout.max_id_bytes + 1)
        with pytest.raises(StringPoolExhaustedError, match="max_id_bytes"):
            insert(two_field_segment, too_long, _row_data(two_field_segment))

    def test_insert_at_capacity_raises(self) -> None:
        seg = _make_segment(_TwoFieldSchema, capacity=4)
        try:
            for i in range(4):
                insert(seg, f"e_{i}", _row_data(seg))
            with pytest.raises(CapacityExceededError):
                insert(seg, "one_too_many", _row_data(seg))
        finally:
            _release(seg)

    def test_update_at_capacity_still_works(self) -> None:
        # Reaching capacity blocks NEW inserts but updates of existing ones must still succeed.
        seg = _make_segment(_TwoFieldSchema, capacity=4)
        try:
            for i in range(4):
                insert(seg, f"e_{i}", bytes([0]) * seg.layout.row_size)
            # Update e_0 — should not raise
            assert insert(seg, "e_0", bytes([0xFF]) * seg.layout.row_size) is False
        finally:
            _release(seg)


# ---------------------------------------------------------------------------
# Segment internals — verify byte-level correctness
# ---------------------------------------------------------------------------


class TestSegmentInternals:
    def test_metadata_initially_zero(self, two_field_segment) -> None:
        seg = two_field_segment
        next_row, pool_cursor = struct.unpack_from("<QQ", seg.mmap_view, METADATA_OFFSET)
        assert next_row == 0
        assert pool_cursor == 0

    def test_string_pool_header_holds_max_id_bytes(self, two_field_segment) -> None:
        seg = two_field_segment
        max_id, reserved = struct.unpack_from("<II", seg.mmap_view, seg.layout.string_pool_offset)
        assert max_id == seg.layout.max_id_bytes
        assert reserved == 0

    def test_data_region_starts_at_byte_64(self, two_field_segment) -> None:
        # invariant: slot table begins at exactly byte 64 (one cache line in)
        assert two_field_segment.layout.slot_table_offset == DATA_REGION_OFFSET

    def test_initial_slot_table_all_empty(self, two_field_segment) -> None:
        seg = two_field_segment
        slot_table = _slot_table_view(seg.mmap_view, seg.layout)
        assert int((slot_table["flags"] == EMPTY).sum()) == seg.layout.slot_capacity

    def test_insert_advances_cursors_correctly(self, two_field_segment) -> None:
        seg = two_field_segment
        insert(seg, "user_001", _row_data(seg))
        next_row, pool_cursor = struct.unpack_from("<QQ", seg.mmap_view, METADATA_OFFSET)
        assert next_row == 1
        # pool entry: 4 (length prefix) + len("user_001") = 4 + 8 = 12
        assert pool_cursor == 12

    def test_slot_table_view_is_zero_copy(self, two_field_segment) -> None:
        seg = two_field_segment
        view = _slot_table_view(seg.mmap_view, seg.layout)
        # Mutating the view must propagate to the underlying memoryview
        view[0]["flags"] = OCCUPIED
        view2 = _slot_table_view(seg.mmap_view, seg.layout)
        assert int(view2[0]["flags"]) == OCCUPIED

    def test_hash_collision_path_still_finds_correct_entity(self) -> None:
        """Two entities whose hashes collide on the same slot index are
        distinguished by the full string compare. Verify by forcing collision
        via small slot_capacity."""
        seg = _make_segment(_TwoFieldSchema, capacity=2)
        try:
            # Two distinct IDs; one will land on idx0 (or wrap), the other
            # will collide and probe to next slot. Both must remain findable.
            insert(seg, "alpha", bytes([0xA0]) * seg.layout.row_size)
            insert(seg, "beta", bytes([0xB0]) * seg.layout.row_size)
            assert lookup(seg, "alpha") is not None
            assert lookup(seg, "beta") is not None

            a_off = lookup(seg, "alpha")
            b_off = lookup(seg, "beta")
            assert a_off != b_off
            assert (
                bytes(seg.mmap_view[a_off : a_off + seg.layout.row_size])
                == bytes([0xA0]) * seg.layout.row_size
            )
            assert (
                bytes(seg.mmap_view[b_off : b_off + seg.layout.row_size])
                == bytes([0xB0]) * seg.layout.row_size
            )
        finally:
            _release(seg)


# ---------------------------------------------------------------------------
# insert_many
# ---------------------------------------------------------------------------


class TestInsertMany:
    def test_insert_many_returns_new_count(self, two_field_segment) -> None:
        seg = two_field_segment
        rows = [(f"e_{i}", _row_data(seg, i % 256)) for i in range(10)]
        new_count = insert_many(seg, rows)
        assert new_count == 10

    def test_insert_many_counts_only_new(self, two_field_segment) -> None:
        seg = two_field_segment
        rows1 = [(f"e_{i}", _row_data(seg)) for i in range(5)]
        insert_many(seg, rows1)
        # Re-insert same 5 plus 3 new
        rows2 = [(f"e_{i}", _row_data(seg, 0xFF)) for i in range(8)]
        new_count = insert_many(seg, rows2)
        assert new_count == 3


# ---------------------------------------------------------------------------
# Pinned hash regression — entity ID hashes are wire-stable.
# ---------------------------------------------------------------------------


class TestHashPinned:
    def test_known_pinned_hashes(self) -> None:
        # Computed once via blake2b("...", digest_size=8), little-endian uint64.
        # If anyone changes the algorithm, this test fails loudly.
        import hashlib

        for ident in ["user_001", "abc", "550e8400-e29b-41d4-a716-446655440000"]:
            expected = int.from_bytes(
                hashlib.blake2b(ident.encode("utf-8"), digest_size=8).digest(),
                "little",
            )
            assert hash_entity_id(ident) == expected

    def test_unicode_id_hashed_as_utf8(self) -> None:
        import hashlib

        ident = "ユーザー_001"
        expected = int.from_bytes(
            hashlib.blake2b(ident.encode("utf-8"), digest_size=8).digest(),
            "little",
        )
        assert hash_entity_id(ident) == expected


# ---------------------------------------------------------------------------
# Sanity check on SLOT_DTYPE layout
# ---------------------------------------------------------------------------


def test_slot_dtype_is_24_bytes() -> None:
    assert SLOT_DTYPE.itemsize == 24


def test_slot_dtype_has_required_fields() -> None:
    assert SLOT_DTYPE.names == (
        "name_hash",
        "id_offset",
        "flags",
        "feature_row_index",
    )


def test_segment_layout_is_frozen() -> None:
    layout = compute_layout(_TwoFieldSchema, capacity=64)
    with pytest.raises((AttributeError, TypeError)):
        layout.capacity = 999  # type: ignore[misc]
