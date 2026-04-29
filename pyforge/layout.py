"""Entity layout: slot table, string pool, feature rows.

Carves a Pyforge shared-memory segment (allocated by :mod:`pyforge.shm` in
Step 2) into four named regions and gives us O(1) ``entity_id -> row_offset``
lookups inside it.

Region layout, computed deterministically from ``(schema, capacity,
max_id_bytes)``:

::

    [0..16)     primary header: magic + version + crc32 + capacity
    [16..24)    next_free_row_index (uint64 LE)         writer cursor
    [24..32)    string_pool_cursor  (uint64 LE)         writer cursor
    [32..64)    reserved zeros (32 bytes)
    [64..A)     slot table — N slots x 24 bytes, N = next_pow2(capacity * 2)
    [A..A+8)    string pool header: max_id_bytes (uint32) + reserved (uint32)
    [A+8..B)    string pool data (length-prefixed UTF-8)
    [B..C)      feature rows — capacity * row_size bytes (B is 64-aligned)
    [C..end)    page-padded to 4 KiB

Three invariants the implementation enforces and every later module respects:

1. **Single writer.** Step 10's WAL consumer (and Step 13's hydration
   coordinator) are the only callers of :func:`insert` and :func:`insert_many`.
   Lookups are lock-free; multi-writer would require CAS and destroy the
   5 us latency claim.
2. **Capacity-cap-at-50%.** The slot table always has at least 2x as many
   slots as ``capacity``; insertion raises :class:`CapacityExceededError` at
   ``next_row == capacity``, well before linear probing degrades.
3. **Cursors advance before slot writes.** :func:`insert` writes the
   ``next_free_row_index`` and ``string_pool_cursor`` updates *before* the
   slot itself becomes :data:`OCCUPIED`. A crash mid-insert leaks a few bytes
   of pool/row storage but cannot corrupt the slot table — replay sees the
   slot as :data:`EMPTY` and allocates fresh storage.
"""

from __future__ import annotations

import struct
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Final

import numpy as np

from pyforge._internal.hash_id import hash_entity_id
from pyforge.schema import (
    CACHE_LINE_SIZE,
    PAGE_SIZE,
    FeatureSchema,
    compute_assembly_table,
    compute_row_offset_table,
    row_size,
    total_element_count,
)

if TYPE_CHECKING:
    from pyforge.shm import Segment

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Slot record dtype. AoS layout — one structured dtype, 24 bytes per slot.
#: Numba ``@njit`` understands structured dtypes natively, so Step 5 can read
#: from this same view without a SoA conversion.
SLOT_DTYPE: Final[np.dtype[np.void]] = np.dtype(
    [
        ("name_hash", np.uint64),
        ("id_offset", np.uint64),
        ("flags", np.uint32),
        ("feature_row_index", np.uint32),
    ]
)
assert SLOT_DTYPE.itemsize == 24, "SLOT_DTYPE must be exactly 24 bytes"

#: Slot ``flags`` values. Tombstone reserved for v2 deletion support; v1 never
#: writes 2 (no deletion). Lookup logic must treat 0 as "stop probing", any
#: non-zero as "check this slot".
EMPTY: Final[int] = 0
OCCUPIED: Final[int] = 1
TOMBSTONE: Final[int] = 2  # reserved; never written in v1

#: Default budget for one entity ID in the string pool. UUIDs in canonical
#: form are 36 bytes; user_*-prefixed UUIDs are ~41 bytes; compound keys can
#: reach ~56 bytes. 64 bytes covers all common cases with room.
DEFAULT_MAX_ID_BYTES: Final[int] = 64

#: Where the writer-cursors metadata region begins (just after the 16-byte
#: primary header from Step 2).
METADATA_OFFSET: Final[int] = 16

#: Where the slot table begins. The bytes between :data:`METADATA_OFFSET` and
#: this offset hold the cursors plus reserved zeros, all read/written via
#: ``struct`` calls below.
DATA_REGION_OFFSET: Final[int] = 64

#: Bytes consumed by the string-pool's own region-local header
#: (``max_id_bytes`` + 4 reserved bytes).
STRING_POOL_HEADER_BYTES: Final[int] = 8

#: Length-prefix size for one string-pool entry. Each pool entry is
#: ``[uint32 length LE][UTF-8 bytes...]``.
_LEN_PREFIX_BYTES: Final[int] = 4


# ---------------------------------------------------------------------------
# Errors.
# ---------------------------------------------------------------------------


class CapacityExceededError(RuntimeError):
    """Raised when an :func:`insert` would push ``next_free_row_index`` past
    the segment's logical capacity. The caller's options are: drop the write,
    rotate to a larger segment via Step 15 schema evolution, or shard."""


class StringPoolExhaustedError(RuntimeError):
    """Raised when an :func:`insert` cannot fit the entity's UTF-8 ID into
    the string pool. Either the ID exceeds ``max_id_bytes`` for the segment,
    or the pool's data area is full (which should not happen if every prior
    insert respected the per-ID budget)."""


# ---------------------------------------------------------------------------
# Internal alignment helper. Mirrored from pyforge.schema._align_up so the
# two modules don't share a private name.
# ---------------------------------------------------------------------------


def _align_up(value: int, alignment: int) -> int:
    return (value + alignment - 1) & ~(alignment - 1)


def _next_pow2(n: int) -> int:
    """Smallest power of two greater than or equal to ``n`` (and >= 1)."""
    if n <= 1:
        return 1
    return 1 << (n - 1).bit_length()


# ---------------------------------------------------------------------------
# Segment layout — pure data; no I/O. Computed once per Segment at
# create/open time and stored on the :class:`pyforge.shm.Segment` instance.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SegmentLayout:
    """All derived offsets and sizes for a multi-entity Pyforge segment."""

    capacity: int  # logical max entities
    slot_capacity: int  # power-of-2 slot count (>= 2*capacity)
    max_id_bytes: int  # max UTF-8 bytes per entity ID
    row_size: int  # bytes per entity row, 64-aligned

    slot_table_offset: int  # absolute byte offset of slot table
    slot_table_bytes: int  # = slot_capacity * 24

    string_pool_offset: int  # absolute byte offset of pool's region header
    string_pool_data_offset: int  # = string_pool_offset + 8
    string_pool_data_bytes: int  # = capacity * (4 + max_id_bytes)

    feature_rows_offset: int  # absolute byte offset of feature rows
    feature_rows_bytes: int  # = capacity * row_size

    total_size: int  # full segment size, page-aligned

    # Hash-sorted, row-relative offsets — lookup uses this (searchsorted).
    row_offset_table: np.ndarray[Any, np.dtype[np.void]]

    # Declaration-order, row-relative offsets — assembly uses this. Output
    # vectors are emitted in declaration order so model.predict(vec) sees
    # fields in the order the user declared them. See ADR-003.
    assembly_table: np.ndarray[Any, np.dtype[np.void]]

    # Length of the float32 vector returned by pyforge.serving.assemble.
    # Sum of element_count across all fields.
    total_element_count: int

    # Flat parallel arrays derived from assembly_table — used by Step 5's
    # Numba kernel (pyforge.assembly._assemble_core). Pre-cast to int64/uint8
    # at compute_layout time so the hot path doesn't pay astype overhead.
    # Order matches assembly_table: declaration order. See ADR-004.
    assembly_byte_offsets: np.ndarray[Any, np.dtype[np.int64]]
    assembly_byte_counts: np.ndarray[Any, np.dtype[np.int64]]
    assembly_dtype_codes: np.ndarray[Any, np.dtype[np.uint8]]
    assembly_element_counts: np.ndarray[Any, np.dtype[np.int64]]


def compute_layout(
    schema: type[FeatureSchema],
    *,
    capacity: int,
    max_id_bytes: int = DEFAULT_MAX_ID_BYTES,
) -> SegmentLayout:
    """Compute the deterministic byte-layout for a ``(schema, capacity,
    max_id_bytes)`` triple.

    Pure function: no segment is opened or read. Useful for sizing before
    allocation and for re-deriving offsets from a segment header on open.
    """
    if capacity <= 0:
        raise ValueError(f"capacity must be >= 1, got {capacity}")
    if max_id_bytes <= 0:
        raise ValueError(f"max_id_bytes must be >= 1, got {max_id_bytes}")
    if capacity > 0xFFFF_FFFF:
        # capacity is stored as uint32 in the header.
        raise ValueError(f"capacity {capacity} exceeds uint32 range")

    slot_capacity = _next_pow2(capacity * 2)
    slot_table_bytes = slot_capacity * SLOT_DTYPE.itemsize
    slot_table_offset = DATA_REGION_OFFSET
    slot_table_end = slot_table_offset + slot_table_bytes

    string_pool_offset = _align_up(slot_table_end, CACHE_LINE_SIZE)
    string_pool_data_offset = string_pool_offset + STRING_POOL_HEADER_BYTES
    # Worst case: every entity uses the full max_id_bytes.
    string_pool_data_bytes = capacity * (_LEN_PREFIX_BYTES + max_id_bytes)
    string_pool_end = string_pool_data_offset + string_pool_data_bytes

    feature_rows_offset = _align_up(string_pool_end, CACHE_LINE_SIZE)
    rs = row_size(schema)
    feature_rows_bytes = capacity * rs
    feature_rows_end = feature_rows_offset + feature_rows_bytes

    total_size = _align_up(feature_rows_end, PAGE_SIZE)

    asm = compute_assembly_table(schema)
    return SegmentLayout(
        capacity=capacity,
        slot_capacity=slot_capacity,
        max_id_bytes=max_id_bytes,
        row_size=rs,
        slot_table_offset=slot_table_offset,
        slot_table_bytes=slot_table_bytes,
        string_pool_offset=string_pool_offset,
        string_pool_data_offset=string_pool_data_offset,
        string_pool_data_bytes=string_pool_data_bytes,
        feature_rows_offset=feature_rows_offset,
        feature_rows_bytes=feature_rows_bytes,
        total_size=total_size,
        row_offset_table=compute_row_offset_table(schema),
        assembly_table=asm,
        total_element_count=total_element_count(schema),
        # Flat arrays for the Numba kernel. ``.astype(np.int64)`` allocates
        # fresh contiguous arrays; ``dtype_code.copy()`` materializes a
        # contiguous uint8 array (a structured-array field view is not
        # guaranteed contiguous, but ``copy()`` ensures the C-contiguous
        # layout the @njit signature requires).
        assembly_byte_offsets=asm["byte_offset"].astype(np.int64),
        assembly_byte_counts=asm["byte_count"].astype(np.int64),
        assembly_dtype_codes=asm["dtype_code"].copy(),
        assembly_element_counts=asm["element_count"].astype(np.int64),
    )


def total_segment_size(
    schema: type[FeatureSchema],
    *,
    capacity: int,
    max_id_bytes: int = DEFAULT_MAX_ID_BYTES,
) -> int:
    """Total bytes needed for a multi-entity segment, rounded to a page boundary.

    Step 2's :class:`pyforge.shm.SegmentRegistry` calls this to size the POSIX
    shm allocation.
    """
    return compute_layout(schema, capacity=capacity, max_id_bytes=max_id_bytes).total_size


def compute_layout_from_segment(
    buf: memoryview,
    schema: type[FeatureSchema],
) -> SegmentLayout:
    """Re-derive the layout for an *already-existing* segment by reading its
    header.

    Used by :meth:`pyforge.shm.SegmentRegistry.open_current` so that opening
    a segment (created by another process, possibly long ago) doesn't require
    out-of-band knowledge of capacity or ``max_id_bytes``.
    """
    capacity = struct.unpack_from("<I", buf, 12)[0]
    if capacity == 0:
        # Either pre-Step-3 segment or a zeroed header — refuse rather than
        # silently building a bogus layout.
        raise ValueError("segment header has capacity=0; not a Step-3 segment")

    # We need slot_capacity to find the string pool offset before we can read
    # max_id_bytes, but slot_capacity is derived from capacity, which we just
    # read. So this is well-defined.
    slot_capacity = _next_pow2(capacity * 2)
    string_pool_offset = _align_up(
        DATA_REGION_OFFSET + slot_capacity * SLOT_DTYPE.itemsize,
        CACHE_LINE_SIZE,
    )
    max_id_bytes = struct.unpack_from("<I", buf, string_pool_offset)[0]
    if max_id_bytes == 0:
        raise ValueError(
            "segment string-pool header has max_id_bytes=0; either uninitialized or corrupt"
        )

    return compute_layout(schema, capacity=capacity, max_id_bytes=max_id_bytes)


# ---------------------------------------------------------------------------
# Internal accessors — narrow, allocation-free reads/writes.
# ---------------------------------------------------------------------------


def _slot_table_view(buf: memoryview, layout: SegmentLayout) -> np.ndarray[Any, np.dtype[np.void]]:
    """Zero-copy structured-array view over the slot table.

    Mutations through the returned array land directly in the underlying
    mmap'd memory (assuming ``buf`` is writable, which it is for our segments).
    """
    return np.frombuffer(
        buf,
        dtype=SLOT_DTYPE,
        count=layout.slot_capacity,
        offset=layout.slot_table_offset,
    )


def _read_metadata(buf: memoryview) -> tuple[int, int]:
    """Return ``(next_free_row_index, string_pool_cursor)``."""
    return struct.unpack_from("<QQ", buf, METADATA_OFFSET)


def _write_metadata(buf: memoryview, next_row: int, pool_cursor: int) -> None:
    struct.pack_into("<QQ", buf, METADATA_OFFSET, next_row, pool_cursor)


def _read_string_bytes(buf: memoryview, layout: SegmentLayout, id_offset: int) -> bytes:
    """Read the length-prefixed UTF-8 entry at ``id_offset`` (pool-data-relative)
    and return its bytes (no decode).
    """
    abs_offset = layout.string_pool_data_offset + id_offset
    length = struct.unpack_from("<I", buf, abs_offset)[0]
    if length > layout.max_id_bytes:
        # Defense-in-depth: a string longer than max_id_bytes means the segment
        # is corrupt. Fail loudly rather than reading junk.
        raise ValueError(
            f"string-pool entry at offset {id_offset} has length {length} "
            f"> max_id_bytes {layout.max_id_bytes}; segment may be corrupt"
        )
    start = abs_offset + _LEN_PREFIX_BYTES
    return bytes(buf[start : start + length])


def _write_pool_header(buf: memoryview, layout: SegmentLayout) -> None:
    """Write the 8-byte string-pool region header (max_id_bytes + reserved)."""
    struct.pack_into("<II", buf, layout.string_pool_offset, layout.max_id_bytes, 0)


# ---------------------------------------------------------------------------
# Public lookup / insert / iterate.
# ---------------------------------------------------------------------------


def lookup(segment: Segment, entity_id: str) -> int | None:
    """Return the byte offset of the entity's row, or ``None`` if not present.

    Lock-free; safe to call from any reader process while the single writer
    is doing inserts. Pure-Python implementation; Step 5 will Numba-jit a
    faster version that operates on raw arrays.

    Raises:
        ValueError: if ``entity_id`` is empty.
    """
    if not entity_id:
        raise ValueError("entity_id must be non-empty")

    layout = segment.layout
    buf = segment.handle.buf
    encoded_id = entity_id.encode("utf-8")
    h = hash_entity_id(entity_id)

    slot_table = _slot_table_view(buf, layout)
    mask = layout.slot_capacity - 1
    idx = h & mask

    # Bounded probe loop — at most slot_capacity iterations.
    for _ in range(layout.slot_capacity):
        slot = slot_table[idx]
        flags = int(slot["flags"])
        if flags == EMPTY:
            return None
        if flags == OCCUPIED and int(slot["name_hash"]) == h:
            stored_id_bytes = _read_string_bytes(buf, layout, int(slot["id_offset"]))
            if stored_id_bytes == encoded_id:
                row_idx = int(slot["feature_row_index"])
                return layout.feature_rows_offset + row_idx * layout.row_size
        idx = (idx + 1) & mask

    return None


def insert(segment: Segment, entity_id: str, row_data: bytes | memoryview) -> bool:
    """Upsert one entity. Returns ``True`` on new insert, ``False`` on update.

    Single-writer contract: the caller is the only writer (Step 10's WAL
    consumer in production; tests may call directly).

    Raises:
        ValueError: if ``entity_id`` is empty or ``row_data`` is the wrong size.
        CapacityExceededError: if the segment has already received ``capacity``
            distinct inserts (no further new entities can be added).
        StringPoolExhaustedError: if the entity's UTF-8 ID exceeds
            ``max_id_bytes``, or the pool's data area is full.
    """
    layout = segment.layout
    buf = segment.handle.buf

    if not entity_id:
        raise ValueError("entity_id must be non-empty")

    if len(row_data) != layout.row_size:
        raise ValueError(f"row_data length {len(row_data)} != row_size {layout.row_size}")

    encoded_id = entity_id.encode("utf-8")
    if len(encoded_id) > layout.max_id_bytes:
        raise StringPoolExhaustedError(
            f"entity_id is {len(encoded_id)} UTF-8 bytes; max is max_id_bytes={layout.max_id_bytes}"
        )

    h = hash_entity_id(entity_id)
    slot_table = _slot_table_view(buf, layout)
    mask = layout.slot_capacity - 1
    idx = h & mask

    for _ in range(layout.slot_capacity):
        slot = slot_table[idx]
        flags = int(slot["flags"])

        if flags == EMPTY:
            return _insert_into_empty_slot(
                buf=buf,
                layout=layout,
                slot_table=slot_table,
                idx=idx,
                hash_value=h,
                encoded_id=encoded_id,
                row_data=row_data,
            )

        if flags == OCCUPIED and int(slot["name_hash"]) == h:
            stored_id = _read_string_bytes(buf, layout, int(slot["id_offset"]))
            if stored_id == encoded_id:
                # Update existing row in place. Idempotent re-write, no slot
                # changes.
                row_idx = int(slot["feature_row_index"])
                row_abs = layout.feature_rows_offset + row_idx * layout.row_size
                buf[row_abs : row_abs + layout.row_size] = row_data
                return False

        idx = (idx + 1) & mask

    # Walked the whole table without finding empty or matching slot. This can
    # only happen at full slot-table occupancy, which we cap-at-50% via
    # ``capacity``. If we reach here, something else is very wrong.
    raise CapacityExceededError("slot table fully scanned without finding empty slot")


def _insert_into_empty_slot(
    *,
    buf: memoryview,
    layout: SegmentLayout,
    slot_table: np.ndarray[Any, np.dtype[np.void]],
    idx: int,
    hash_value: int,
    encoded_id: bytes,
    row_data: bytes | memoryview,
) -> bool:
    """Implementation of the ``flags == EMPTY`` branch of :func:`insert`.

    Extracted so the main probe loop stays scannable. Crash-safety contract:
    cursors advance *before* the slot is marked OCCUPIED, so a crash mid-call
    leaks a few bytes but cannot corrupt the slot table or readers' view of it.
    """
    next_row, pool_cursor = _read_metadata(buf)

    if next_row >= layout.capacity:
        raise CapacityExceededError(
            f"segment capacity {layout.capacity} reached (next_free_row_index={next_row})"
        )

    entry_size = _LEN_PREFIX_BYTES + len(encoded_id)
    if pool_cursor + entry_size > layout.string_pool_data_bytes:
        raise StringPoolExhaustedError(
            f"string pool data area full: cursor={pool_cursor}, "
            f"entry_size={entry_size}, capacity={layout.string_pool_data_bytes}"
        )

    id_off = pool_cursor
    row_idx = next_row

    # 1. Advance cursors first. If we crash before the slot write below, the
    #    slot remains EMPTY, the row index and pool slot are leaked (a tiny
    #    permanent loss in this segment), and replay allocates fresh storage.
    _write_metadata(buf, next_row + 1, pool_cursor + entry_size)

    # 2. Write the string pool entry.
    pool_data_start = layout.string_pool_data_offset
    entry_abs = pool_data_start + id_off
    struct.pack_into("<I", buf, entry_abs, len(encoded_id))
    buf[entry_abs + _LEN_PREFIX_BYTES : entry_abs + entry_size] = encoded_id

    # 3. Write the feature row data.
    row_abs = layout.feature_rows_offset + row_idx * layout.row_size
    buf[row_abs : row_abs + layout.row_size] = row_data

    # 4. Write the slot LAST. The transition EMPTY -> OCCUPIED is what makes
    #    the entity visible to lookup. Until ``flags`` is OCCUPIED, lookups
    #    walking past this slot will treat it as EMPTY and skip onward.
    slot_table[idx]["name_hash"] = hash_value
    slot_table[idx]["id_offset"] = id_off
    slot_table[idx]["feature_row_index"] = row_idx
    slot_table[idx]["flags"] = OCCUPIED

    return True


def insert_many(segment: Segment, rows: Iterable[tuple[str, bytes | memoryview]]) -> int:
    """Bulk upsert. Returns the number of *new* inserts (excludes updates).

    Step 13's hydration is the production caller. For Step 3 this is just
    a tight Python loop over :func:`insert`; Step 13 may add specialized
    fast paths if hydration becomes a bottleneck.
    """
    inserted = 0
    for entity_id, row_data in rows:
        if insert(segment, entity_id, row_data):
            inserted += 1
    return inserted


def iterate_occupied(segment: Segment) -> Iterator[tuple[str, int]]:
    """Yield ``(entity_id, row_offset)`` for every occupied slot, in slot-order.

    Used by Step 15's schema evolution to copy entities from old segment to
    new. Order is the slot table's internal order — *not* insertion order —
    but every occupied entity appears exactly once.
    """
    layout = segment.layout
    buf = segment.handle.buf
    slot_table = _slot_table_view(buf, layout)

    for idx in range(layout.slot_capacity):
        slot = slot_table[idx]
        if int(slot["flags"]) == OCCUPIED:
            id_bytes = _read_string_bytes(buf, layout, int(slot["id_offset"]))
            row_idx = int(slot["feature_row_index"])
            row_offset = layout.feature_rows_offset + row_idx * layout.row_size
            yield id_bytes.decode("utf-8"), row_offset


# ---------------------------------------------------------------------------
# Initialization helper — called by SegmentRegistry.create after the POSIX
# segment is allocated. Writes the writer-cursor metadata (zeros) and the
# string-pool region header.
# ---------------------------------------------------------------------------


def initialize_segment_regions(buf: memoryview, layout: SegmentLayout) -> None:
    """Write the post-header writer cursors and the string-pool region header.

    Called by :class:`pyforge.shm.SegmentRegistry.create` once the segment
    has been allocated and the primary header (magic/version/crc/capacity)
    has been written.
    """
    _write_metadata(buf, next_row=0, pool_cursor=0)
    _write_pool_header(buf, layout)
