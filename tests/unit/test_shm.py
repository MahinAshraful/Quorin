"""Unit tests for pyforge.shm.

Uses a real Redis (the ``redis_client`` fixture) rather than a mock because
the close-Lua script needs to execute server-side. These tests still run as
"unit" because they use a small synthetic schema and isolate cleanly via the
autouse fixture.

Linux/WSL2 only (posix_shm requires POSIX).
"""

from __future__ import annotations

import os
import struct
import sys

import pytest

pytestmark = pytest.mark.skipif(
    sys.platform == "win32",
    reason="shm requires POSIX (Linux/WSL2)",
)

from pyforge._internal.crc import crc32_of_bytes  # noqa: E402
from pyforge.schema import (  # noqa: E402
    FeatureField,
    FeatureSchema,
    compile_schema,
    dtype,
)
from pyforge.shm import (  # noqa: E402
    HEADER_FMT,
    HEADER_LEN,
    KEY_CLEANUP_QUEUE,
    KEY_SEGMENT_TO_SCHEMA,
    MAGIC,
    SchemaCRCMismatchError,
    Segment,
    SegmentNotFoundError,
    SegmentRegistry,
    _key_current,
    _key_pid_segments,
    _key_refcount,
)


class _TinySchema(FeatureSchema):
    version = 1
    fields = [
        FeatureField("a", dtype.float32),
        FeatureField("b", dtype.int32),
    ]


class _OtherSchema(FeatureSchema):
    version = 1
    fields = [
        FeatureField("a", dtype.float32),
        FeatureField("c", dtype.int32),  # different name → different CRC
    ]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_int(redis_client, key: str) -> int:
    raw = redis_client.get(key)
    if raw is None:
        return 0
    return int(raw)


# ---------------------------------------------------------------------------
# create
# ---------------------------------------------------------------------------


def test_create_writes_correct_header(redis_client) -> None:
    reg = SegmentRegistry(redis_client)
    seg = reg.create(_TinySchema, capacity=32)
    try:
        magic, version, crc, capacity = struct.unpack(HEADER_FMT, bytes(seg.mmap_view[:HEADER_LEN]))
        assert magic == MAGIC
        assert version == _TinySchema.version
        expected_crc = crc32_of_bytes(compile_schema(_TinySchema).tobytes())
        assert crc == expected_crc
        assert capacity == 32
    finally:
        reg.close(seg)


def test_create_sets_expected_redis_keys(redis_client) -> None:
    reg = SegmentRegistry(redis_client)
    seg = reg.create(_TinySchema, capacity=32)
    try:
        # schema:current points at the new segment
        current = redis_client.get(_key_current(_TinySchema))
        assert current is not None
        assert current.decode() == seg.name

        # refcount is 1 after create
        assert _get_int(redis_client, _key_refcount(seg.name)) == 1

        # pid_segments contains the new segment
        pid_set = redis_client.smembers(_key_pid_segments(os.getpid()))
        assert seg.name.encode() in pid_set
    finally:
        reg.close(seg)


def test_create_handle_size_matches_total_segment_size(redis_client) -> None:
    from pyforge.layout import total_segment_size

    reg = SegmentRegistry(redis_client)
    seg = reg.create(_TinySchema, capacity=32)
    try:
        assert seg.handle.size == total_segment_size(_TinySchema, capacity=32)
    finally:
        reg.close(seg)


# ---------------------------------------------------------------------------
# open_current
# ---------------------------------------------------------------------------


def test_open_current_without_registered_segment_raises(redis_client) -> None:
    reg = SegmentRegistry(redis_client)
    with pytest.raises(SegmentNotFoundError):
        reg.open_current(_TinySchema)


def test_open_current_increments_refcount(redis_client) -> None:
    reg = SegmentRegistry(redis_client)
    seg = reg.create(_TinySchema, capacity=32)
    try:
        assert _get_int(redis_client, _key_refcount(seg.name)) == 1

        seg2 = reg.open_current(_TinySchema)
        try:
            assert _get_int(redis_client, _key_refcount(seg.name)) == 2
        finally:
            reg.close(seg2)
        assert _get_int(redis_client, _key_refcount(seg.name)) == 1
    finally:
        reg.close(seg)


def test_open_current_with_crc_mismatch_raises(redis_client) -> None:
    """Create under one schema, then open under a different schema. CRC
    mismatch surfaces as SchemaCRCMismatchError."""
    reg = SegmentRegistry(redis_client)
    seg = reg.create(_TinySchema, capacity=32)
    try:
        # Point _OtherSchema's :current key at the existing segment.
        redis_client.set(_key_current(_OtherSchema), seg.name)
        with pytest.raises(SchemaCRCMismatchError):
            reg.open_current(_OtherSchema)
    finally:
        reg.close(seg)


def test_open_current_with_corrupted_magic_raises(redis_client) -> None:
    reg = SegmentRegistry(redis_client)
    seg = reg.create(_TinySchema, capacity=32)
    try:
        seg.mmap_view[:4] = b"XXXX"  # clobber magic
        with pytest.raises(SchemaCRCMismatchError):
            reg.open_current(_TinySchema)
    finally:
        # Restore magic so close path + cleanup don't cascade.
        seg.mmap_view[:4] = MAGIC
        reg.close(seg)


# ---------------------------------------------------------------------------
# close
# ---------------------------------------------------------------------------


def test_close_decrements_refcount(redis_client) -> None:
    reg = SegmentRegistry(redis_client)
    seg = reg.create(_TinySchema, capacity=32)
    assert _get_int(redis_client, _key_refcount(seg.name)) == 1

    reg.close(seg)
    # After close, refcount should be 0 (or the key deleted).
    # DECR from 1 leaves 0; the Lua script does not DEL the key.
    assert _get_int(redis_client, _key_refcount(seg.name)) == 0


def test_close_adds_to_cleanup_queue_when_refcount_hits_zero(
    redis_client,
) -> None:
    reg = SegmentRegistry(redis_client)
    seg = reg.create(_TinySchema, capacity=32)
    name = seg.name
    reg.close(seg)

    cleanup_members = {
        m.decode() if isinstance(m, bytes) else m for m in redis_client.smembers(KEY_CLEANUP_QUEUE)
    }
    assert name in cleanup_members


def test_close_does_not_add_to_cleanup_queue_when_others_hold(
    redis_client,
) -> None:
    reg = SegmentRegistry(redis_client)
    seg = reg.create(_TinySchema, capacity=32)
    seg2 = reg.open_current(_TinySchema)
    name = seg.name
    try:
        reg.close(seg2)  # refcount 2 -> 1, not zero
        cleanup_members = {
            m.decode() if isinstance(m, bytes) else m
            for m in redis_client.smembers(KEY_CLEANUP_QUEUE)
        }
        assert name not in cleanup_members
    finally:
        reg.close(seg)


def test_close_removes_from_pid_segments(redis_client) -> None:
    reg = SegmentRegistry(redis_client)
    seg = reg.create(_TinySchema, capacity=32)
    name = seg.name
    reg.close(seg)

    pid_set = {
        m.decode() if isinstance(m, bytes) else m
        for m in redis_client.smembers(_key_pid_segments(os.getpid()))
    }
    assert name not in pid_set


# ---------------------------------------------------------------------------
# Segment
# ---------------------------------------------------------------------------


def test_segment_mmap_view_is_writable(redis_client) -> None:
    reg = SegmentRegistry(redis_client)
    seg = reg.create(_TinySchema, capacity=32)
    try:
        # Write past the header
        seg.mmap_view[HEADER_LEN : HEADER_LEN + 4] = b"DATA"
        assert bytes(seg.mmap_view[HEADER_LEN : HEADER_LEN + 4]) == b"DATA"
    finally:
        reg.close(seg)


def test_segment_is_a_dataclass_with_name_schema_handle() -> None:
    # Basic invariant — later steps reach for .name, .schema, .handle.
    fields = {f for f in Segment.__dataclass_fields__}
    assert {"name", "schema", "handle"}.issubset(fields)


# ---------------------------------------------------------------------------
# Step 14: pyforge:segment_to_schema sidetable + close-Lua extension
# ---------------------------------------------------------------------------


def test_create_writes_sidetable(redis_client) -> None:
    """Step 14: ``SegmentRegistry.create`` writes the sidetable field
    so the watchdog can clear ``schema:current`` without a KEYS scan.
    """
    reg = SegmentRegistry(redis_client)
    seg = reg.create(_TinySchema, capacity=16)
    try:
        # HEXISTS returns 1 if present, 0 if not.
        assert redis_client.hexists(KEY_SEGMENT_TO_SCHEMA, seg.name) == 1
        raw = redis_client.hget(KEY_SEGMENT_TO_SCHEMA, seg.name)
        assert raw is not None
        # Value matches the safe-class-name form used in schema:current keys.
        assert raw.decode() == "_TinySchema"
    finally:
        reg.close(seg)


def test_close_lua_clears_schema_current_at_refcount_zero(redis_client) -> None:
    """Step 14 close-Lua extension: when the closing process is the last
    holder (refcount-0), ``pyforge:schema:{name}:current`` AND the
    sidetable field are cleared atomically. Symmetric with the watchdog
    dead-PID Lua.
    """
    reg = SegmentRegistry(redis_client)
    seg = reg.create(_TinySchema, capacity=16)
    name = seg.name
    # Pre-conditions: schema:current points at this segment, sidetable populated.
    assert redis_client.get(_key_current(_TinySchema)) == name.encode()
    assert redis_client.hexists(KEY_SEGMENT_TO_SCHEMA, name) == 1

    reg.close(seg)
    # Post-conditions: refcount=0 → close-Lua's refcount-0 branch fired.
    assert redis_client.get(_key_current(_TinySchema)) is None, (
        "schema:current should be cleared by close-Lua extension at refcount-0"
    )
    assert redis_client.hexists(KEY_SEGMENT_TO_SCHEMA, name) == 0, (
        "segment_to_schema sidetable should be cleared at refcount-0"
    )
    # Segment queued for posix_shm.unlink by the watchdog/drain.
    assert redis_client.sismember(KEY_CLEANUP_QUEUE, name) == 1


def test_close_lua_rotation_safety_skips_current_when_pointer_advanced(
    redis_client,
) -> None:
    """Step 14 close-Lua extension is rotation-safe: when a newer create
    has flipped ``schema:current`` to a different segment, closing the
    older segment must NOT clobber the newer pointer. The Lua's
    ``GET schema:current; if == name, DEL`` conditional handles this.
    """
    reg = SegmentRegistry(redis_client)
    seg1 = reg.create(_TinySchema, capacity=16)
    name1 = seg1.name
    # Second create rotates schema:current to a fresh segment. The
    # registry doesn't decrement seg1's refcount when this happens
    # (rotation is a separate concept) — but we need TWO holders of
    # seg1 to drive its close to refcount=0 below. Open seg1 again
    # first so the rotation creates a clean state.

    # Rotate: schema:current now points at seg2.
    seg2 = reg.create(_TinySchema, capacity=16)
    name2 = seg2.name
    assert redis_client.get(_key_current(_TinySchema)) == name2.encode()

    # Close seg1 (refcount 1 -> 0). Close-Lua sees current != seg1.name
    # and SKIPs the schema:current DEL.
    reg.close(seg1)
    assert redis_client.get(_key_current(_TinySchema)) == name2.encode(), (
        "close-Lua incorrectly cleared schema:current when it pointed at a different segment"
    )
    # seg1's sidetable entry IS cleared (rotation-safe applies only to
    # schema:current; sidetable lookup is the dispatch and must clean up).
    assert redis_client.hexists(KEY_SEGMENT_TO_SCHEMA, name1) == 0
    # seg2's sidetable entry still intact.
    assert redis_client.hexists(KEY_SEGMENT_TO_SCHEMA, name2) == 1

    # Cleanup.
    reg.close(seg2)


def test_create_segment_to_schema_uses_safe_class_name(redis_client) -> None:
    """Sidetable values store the safe-class-name form so they can be
    string-concatenated into ``pyforge:schema:{value}:current`` keys
    by the watchdog Lua without further escaping.
    """
    reg = SegmentRegistry(redis_client)
    seg = reg.create(_OtherSchema, capacity=16)
    try:
        raw = redis_client.hget(KEY_SEGMENT_TO_SCHEMA, seg.name)
        assert raw is not None
        # No spaces, no special chars (only [A-Za-z0-9_]).
        assert raw.decode() == "_OtherSchema"
    finally:
        reg.close(seg)
