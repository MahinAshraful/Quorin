"""Numba-compiled bulk-insert kernel for hydration (Step 13).

Mirror image of Step 8's :func:`pyforge.assembly._assemble_batch_core` —
that kernel reads slot records and feature rows; this one writes them.
Both share the slot byte-layout discipline locked by invariant #15: the
kernel reads/writes raw bytes at offsets passed in as scalar args, never
hardcoded magic numbers.

This module is **isolated** so importing :mod:`pyforge.layout` does not
pay Numba's compile cost. Only callers that explicitly import
``pyforge._internal.insert_kernel`` (the bulk path; Step 13 hydration)
trigger the LLVM/Numba toolchain init.

Design notes:

- **Single uint64-view writes** for ``name_hash`` and ``id_offset``
  (8-byte fields). Sanity check #1 (Rev-3 plan, run pre-impl) confirmed
  Numba 0.60 supports ``buf[a:a+8].view(np.uint64)[0] = h``.
  Fallback path documented in the kernel docstring if ever needed.

- **uint32-view writes** for ``feature_row_index`` (4-byte field) and
  ``flags`` (4-byte field; only the lowest byte semantically matters
  but writing the full uint32 keeps the upper 3 zero deterministically).

- **Slice-assign for byte buffers** (the row data and the entity-ID
  bytes). C-level memcpy on contiguous uint8 arrays — ~10-50x faster
  than per-byte loops at typical row sizes.

- **No invariant-#8 cursor-first ordering.** Hydration's "build new +
  flip" semantics mean partial-write = throwaway segment. We can write
  in whatever order is fastest; per-row we still write ``flags`` last
  as defense in depth (partial writes look EMPTY, not corrupt).

- **fastmath=False** is non-negotiable across all Pyforge Numba
  kernels (invariant #12).

- **Pre-warm is opt-in.** Module load doesn't auto-compile; callers
  (hydration's wrapper, benchmarks) call :func:`prewarm` to amortize
  the ~100-500 ms first-call compile.

- **Probe budget + sentinel return.** Capacity exhaustion can't raise
  from nopython mode, so the kernel returns ``-(i+1)`` to signal the
  Python wrapper which row failed. On success, returns the final
  ``pool_cursor`` (>=0) so the wrapper writes it back to the segment
  header.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numba
import numpy as np
import pyarrow as pa

from pyforge._internal.hash_id import hash_entity_id_bytes
from pyforge.layout import (
    _LEN_PREFIX_BYTES,
    SLOT_BYTES,
    SLOT_FEATURE_ROW_INDEX_OFFSET,
    SLOT_FLAGS_OFFSET,
    SLOT_ID_OFFSET_OFFSET,
    SLOT_NAME_HASH_OFFSET,
    CapacityExceededError,
    StringPoolExhaustedError,
    _read_metadata,
    _write_metadata,
)
from pyforge.schema import DTYPE_TO_NUMPY, FeatureField, FeatureSchema

if TYPE_CHECKING:
    from pyforge.shm import Segment


@numba.njit(  # type: ignore[untyped-decorator]
    "int64(uint8[::1], uint64[::1], int64[::1], uint8[::1],"
    " int64, int64, int64, int64, int64,"
    " int64, int64, int64, int64, int64,"
    " int64[::1], int64[::1], int64[::1], uint8[::1])",
    cache=True,
    boundscheck=False,
    fastmath=False,
)
def _insert_many_core(
    seg_u8: np.ndarray[Any, np.dtype[np.uint8]],
    id_hashes: np.ndarray[Any, np.dtype[np.uint64]],
    id_byte_offsets: np.ndarray[Any, np.dtype[np.int64]],
    encoded_ids_concat: np.ndarray[Any, np.dtype[np.uint8]],
    slot_table_offset: int,
    slot_capacity: int,
    string_pool_data_offset: int,
    feature_rows_offset: int,
    row_size: int,
    slot_bytes: int,
    slot_name_hash_off: int,
    slot_id_off_off: int,
    slot_flags_off: int,
    slot_feature_row_idx_off: int,
    field_byte_offsets_in_row: np.ndarray[Any, np.dtype[np.int64]],
    field_byte_counts: np.ndarray[Any, np.dtype[np.int64]],
    field_data_starts: np.ndarray[Any, np.dtype[np.int64]],
    field_data: np.ndarray[Any, np.dtype[np.uint8]],
) -> int:
    """Bulk-insert ``n_rows`` entities into a freshly-created segment.

    Returns ``pool_cursor`` (>=0) on success — the new value the caller
    must write back to the segment's metadata cursor. On capacity
    exhaustion at row ``i``, returns ``-(i + 1)`` so the wrapper can
    raise ``CapacityExceededError`` with the failing row index.

    Per-field column-major write: for each row, the kernel scatters each
    field's bytes from ``field_data`` (a flat concatenation of all
    fields' column-major data) directly into the segment's row buffer at
    that field's row-relative offset. This avoids materializing a
    redundant 2D ``row_data`` intermediate (which the strided column
    scatter + double memcpy made prohibitive at 100k+ rows x 200
    fields). Sequential row-major writes inside the kernel are
    cache-friendly; the prefetcher streams each row's pages in order.

    ``field_data`` layout:
      * field 0's bytes for all rows: ``field_data[fds[0] : fds[1]]``,
        length ``n * field_byte_counts[0]``, row i at
        ``fds[0] + i * field_byte_counts[0]``.
      * field 1's bytes for all rows: ``field_data[fds[1] : fds[2]]``,
        ... and so on.
      * ``fds == field_data_starts``, length ``n_fields + 1`` (prefix-sum).

    Preconditions (caller-enforced):
        * Target segment is freshly created — all slot ``flags`` bytes
          are zero AND the row buffer is zero-initialized (so padding
          bytes between fields stay zero per-row).
        * ``slot_capacity`` is a power of two.
        * ``id_byte_offsets`` has length ``n_rows + 1``.
        * ``field_data`` is C-contiguous; ``field_data_starts`` has
          length ``n_fields + 1`` (prefix-sum).

    Caller writes the returned ``pool_cursor`` and ``n_rows`` back to
    the segment's metadata header outside the kernel.
    """
    n_rows = id_byte_offsets.shape[0] - 1
    n_fields = field_byte_offsets_in_row.shape[0]
    pool_cursor = 0
    mask = slot_capacity - 1  # slot_capacity is power-of-2

    for i in range(n_rows):
        h = id_hashes[i]
        idx = h & mask

        # Find first EMPTY slot via linear probe.
        found = False
        slot_base = 0
        for _probe in range(slot_capacity):
            slot_base = slot_table_offset + idx * slot_bytes
            f_off = slot_base + slot_flags_off
            slot_flags = seg_u8[f_off : f_off + 4].view(np.uint32)[0]
            if slot_flags == 0:  # EMPTY
                found = True
                break
            idx = (idx + 1) & mask
        if not found:
            return -(i + 1)

        # String pool entry: 4-byte LE length prefix + UTF-8 bytes.
        str_off = id_byte_offsets[i]
        str_len = id_byte_offsets[i + 1] - str_off
        pool_abs = string_pool_data_offset + pool_cursor
        seg_u8[pool_abs : pool_abs + 4].view(np.uint32)[0] = np.uint32(str_len)
        seg_u8[pool_abs + 4 : pool_abs + 4 + str_len] = encoded_ids_concat[
            str_off : str_off + str_len
        ]
        id_off_value = pool_cursor
        pool_cursor += 4 + str_len

        # Feature row data — per-field scatter from column-major source.
        # Direct slice-assigns from field_data into the segment row.
        # PERFORMANCE NOTE: this phase is fundamentally bounded by the
        # cold-page-fault rate of /dev/shm (tmpfs). On WSL2 we measured
        # ~0.8 GB/s during cold writes (vs ~17 GB/s warm); on native
        # Linux it's typically 4-8x faster. No inner-loop optimization
        # (scratch buffer, scalar byte loops, uint32 views) moved the
        # needle past this floor — the cost is in the kernel's mmap
        # fault path, not in our code.
        row_abs = feature_rows_offset + i * row_size
        for f in range(n_fields):
            f_off_in_row = field_byte_offsets_in_row[f]
            f_cnt = field_byte_counts[f]
            f_src_start = field_data_starts[f] + i * f_cnt
            seg_u8[row_abs + f_off_in_row : row_abs + f_off_in_row + f_cnt] = field_data[
                f_src_start : f_src_start + f_cnt
            ]

        # Slot record. Per-field uint64/uint32-view writes.
        # ALIGNMENT NOTE (invariant #15): SLOT_DTYPE places name_hash at
        # offset 0 (uint64), id_offset at offset 8 (uint64),
        # feature_row_index at offset 20 (uint32). All multiples of their
        # field width; .view() requires this. Module-load asserts in
        # pyforge.layout enforce.
        nh_off = slot_base + slot_name_hash_off
        seg_u8[nh_off : nh_off + 8].view(np.uint64)[0] = h
        io_off = slot_base + slot_id_off_off
        seg_u8[io_off : io_off + 8].view(np.uint64)[0] = np.uint64(id_off_value)
        fri_off = slot_base + slot_feature_row_idx_off
        seg_u8[fri_off : fri_off + 4].view(np.uint32)[0] = np.uint32(i)

        # Flags LAST — defense in depth (partial writes look EMPTY).
        f_off = slot_base + slot_flags_off
        seg_u8[f_off : f_off + 4].view(np.uint32)[0] = np.uint32(1)  # OCCUPIED

    return pool_cursor


def prewarm() -> None:
    """Compile :func:`_insert_many_core` ahead of any timed call.

    Mirrors :func:`pyforge.assembly.prewarm`. First call to the kernel
    pays a ~100-500 ms LLVM compile if no ``__pycache__`` hit; calling
    this from setup code (hydration's wrapper does, before its timed
    section) makes that cost a one-time startup tax instead of a
    surprise inside ``elapsed_seconds``.

    Idempotent: subsequent calls are no-ops once the kernel is
    specialized.
    """
    seg_u8 = np.zeros(4096, dtype=np.uint8)
    id_hashes = np.zeros(1, dtype=np.uint64)
    id_byte_offsets = np.zeros(2, dtype=np.int64)
    encoded_ids_concat = np.zeros(0, dtype=np.uint8)
    field_byte_offsets_in_row = np.zeros(1, dtype=np.int64)
    field_byte_counts = np.array([4], dtype=np.int64)
    field_data_starts = np.array([0, 4], dtype=np.int64)
    field_data = np.zeros(4, dtype=np.uint8)
    _insert_many_core(
        seg_u8,
        id_hashes,
        id_byte_offsets,
        encoded_ids_concat,
        # Offsets within seg_u8 — placeholder values that don't actually
        # write anywhere meaningful since n_rows=1 and the slot find
        # immediately succeeds at idx 0.
        np.int64(64),  # slot_table_offset
        np.int64(2),  # slot_capacity (power of 2)
        np.int64(128),  # string_pool_data_offset
        np.int64(256),  # feature_rows_offset
        np.int64(8),  # row_size
        np.int64(24),  # slot_bytes
        np.int64(0),  # slot_name_hash_off
        np.int64(8),  # slot_id_off_off
        np.int64(16),  # slot_flags_off
        np.int64(20),  # slot_feature_row_idx_off
        field_byte_offsets_in_row,
        field_byte_counts,
        field_data_starts,
        field_data,
    )


# ---------------------------------------------------------------------------
# Public wrapper + column preparation. Lives here (not in pyforge.layout) to
# respect invariant #11: pyforge.layout has many non-Numba callers (lookup,
# per-row insert, the WAL consumer's `_apply` hot path) and shouldn't pull
# Numba into its import surface. Mirrors the pyforge.assembly.assemble_batch
# pattern — public wrapper next to the @njit kernel.
# ---------------------------------------------------------------------------


def _extract_field_to_numpy(
    col: pa.Array | pa.ChunkedArray,
    field: FeatureField,
    byte_count: int,
) -> np.ndarray[Any, np.dtype[np.uint8]]:
    """Return a shape ``(n, byte_count)`` uint8 view of one schema field's data.

    Pre-impl sanity check #2 (Rev-3 plan, run for Step 13) verified that
    PyArrow 14's ``to_numpy(zero_copy_only=False)`` on shaped columns
    (rank >=1 fixed-size lists) returns object dtype, not the contiguous
    2D array we want. Workaround: ``.flatten()`` per list nesting level
    to strip down to a 1D primitive array, then reshape.

    Rank dispatch:
      * rank 0 (scalar): ``col.to_numpy()`` -> 1D primitive array.
      * rank 1 (``pa.list_(elem, N)``): need 1x flatten -> 1D primitive.
      * rank 2 (``pa.list_(pa.list_(elem, C), R)``): need 2x flatten -> 1D primitive.
      * rank >=3: ``NotImplementedError``.

    Returns a uint8 view shaped ``(n, byte_count)`` that the caller can
    slice-assign into a row-major buffer with one numpy memcpy per field.
    """
    if isinstance(col, pa.ChunkedArray):
        col = col.combine_chunks()
    n = len(col)
    rank = len(field.shape)
    if rank == 0:
        flat = col.to_numpy(zero_copy_only=False)
    elif rank == 1:
        flat = col.flatten().to_numpy(zero_copy_only=False)
    elif rank == 2:
        flat = col.flatten().flatten().to_numpy(zero_copy_only=False)
    else:
        raise NotImplementedError(
            f"field {field.name!r} rank {rank} not supported (scalar/1D/2D only)"
        )
    if flat.dtype == object:
        # Defense-in-depth: if combine_chunks+flatten still produced object
        # dtype, the source column type wasn't what we expected (variable-
        # length lists, mixed types, etc.). Fail loudly.
        raise TypeError(
            f"field {field.name!r}: extraction returned object dtype, "
            f"expected primitive — verify source column type"
        )
    # Cast to the schema's declared dtype if the caller passed a different
    # one (e.g. float64 column for a float32 schema field). astype with
    # copy=False is no-op when types match.
    target_np_dtype = DTYPE_TO_NUMPY[field.dtype]
    flat = flat.astype(target_np_dtype, copy=False)
    out: np.ndarray[Any, np.dtype[np.uint8]] = flat.view(np.uint8).reshape(n, byte_count)
    return out


def _prepare_insert_columns(
    table: pa.Table,
    schema: type[FeatureSchema],
    layout: Any,  # SegmentLayout — keep Any to avoid importing layout.py's class
) -> tuple[
    np.ndarray[Any, np.dtype[np.uint64]],
    np.ndarray[Any, np.dtype[np.int64]],
    np.ndarray[Any, np.dtype[np.uint8]],
    np.ndarray[Any, np.dtype[np.int64]],
    np.ndarray[Any, np.dtype[np.int64]],
    np.ndarray[Any, np.dtype[np.int64]],
    np.ndarray[Any, np.dtype[np.uint8]],
]:
    """Extract PyArrow columns into kernel-ready column-major buffers.

    Returns a 7-tuple:

    * ``id_hashes`` — uint64 blake2b digests, one per row.
    * ``id_byte_offsets`` — int64 prefix-sum of UTF-8 byte lengths
      (length ``n+1``).
    * ``encoded_ids_concat`` — flat uint8 buffer of all entity_ids
      concatenated (zero-copy view over Arrow's ``StringArray.values``).
    * ``field_byte_offsets_in_row`` — int64 (``n_fields``,) — each
      field's byte offset within a single row.
    * ``field_byte_counts`` — int64 (``n_fields``,) — bytes per field.
    * ``field_data_starts`` — int64 (``n_fields+1``,) prefix-sum
      describing where each field's column-major data lives in
      ``field_data`` (row i of field f at
      ``field_data_starts[f] + i * field_byte_counts[f]``).
    * ``field_data`` — flat uint8 — all fields' bytes concatenated
      column-major. The kernel reads from here per-row.

    Why column-major flat buffer instead of a 2D ``row_data`` of shape
    ``(n, row_size)``: at typical 200-field x ~13 KB row_size scales,
    the strided per-field column scatter into the 2D buffer dominated
    cost (~1.6 s at 100k rows in the v1 design — measured), and the
    kernel's row-by-row memcpy of that buffer into the segment doubled
    the memory traffic. The new layout writes each field once (single
    contiguous slice-assign) and the kernel scatters it directly into
    the segment with sequential row-major writes (cache-friendly).

    ORDERING INVARIANT: ``layout.assembly_byte_offsets[i]`` corresponds
    to ``schema.fields[i]`` (declaration order), NOT name-hash wire
    order. Locked by invariant #15.
    """
    if "entity_id" not in table.column_names:
        raise ValueError("insert_many requires column 'entity_id'")
    missing = [f.name for f in schema.fields if f.name not in table.column_names]
    if missing:
        raise ValueError(f"insert_many missing schema columns: {missing}")

    n = len(table)

    # --- Entity IDs: zero-copy buffer extraction ---
    eid_col = table["entity_id"]
    eid_combined = eid_col.combine_chunks() if isinstance(eid_col, pa.ChunkedArray) else eid_col
    if not isinstance(eid_combined, pa.StringArray):
        raise TypeError(f"entity_id column must be pa.string(), got {eid_combined.type}")
    # StringArray internals via buffers(): [null_bitmap, offsets_i32, values_u8].
    bufs = eid_combined.buffers()
    offsets_i32 = np.frombuffer(bufs[1], dtype=np.int32, count=n + 1)
    encoded_ids_concat = np.ascontiguousarray(
        np.frombuffer(bufs[2], dtype=np.uint8)
        if bufs[2] is not None
        else np.zeros(0, dtype=np.uint8)
    )
    id_byte_offsets = offsets_i32.astype(np.int64)

    # --- Hash entity IDs (per-call Python; ~700 ms-1 s at 1M) ---
    id_hashes = np.empty(n, dtype=np.uint64)
    for i in range(n):
        start = int(id_byte_offsets[i])
        end = int(id_byte_offsets[i + 1])
        id_hashes[i] = hash_entity_id_bytes(encoded_ids_concat[start:end])

    # --- Per-field column-major flat buffer ---
    n_fields = len(schema.fields)
    field_byte_offsets_in_row = np.ascontiguousarray(
        layout.assembly_byte_offsets.astype(np.int64, copy=False)
    )
    field_byte_counts = np.ascontiguousarray(
        layout.assembly_byte_counts.astype(np.int64, copy=False)
    )
    # field_data_starts[f] = byte offset into field_data where field f begins.
    # field_data_starts[-1] = total bytes.
    per_field_total = field_byte_counts * n
    field_data_starts = np.empty(n_fields + 1, dtype=np.int64)
    field_data_starts[0] = 0
    np.cumsum(per_field_total, out=field_data_starts[1:])
    total_field_bytes = int(field_data_starts[-1])

    # np.empty (no zero-init): every byte gets overwritten by the per-field
    # writes below, so zero-init would be wasted work. The kernel uses these
    # bytes directly; the segment's row buffer (which DOES need zero-padding
    # between fields) is zero-initialized by posix_shm.create's mmap.
    field_data = np.empty(total_field_bytes, dtype=np.uint8)
    for f_idx, f in enumerate(schema.fields):
        col = table[f.name]
        if isinstance(col, pa.ChunkedArray):
            col = col.combine_chunks()
        cnt = int(field_byte_counts[f_idx])
        # _extract_field_to_numpy returns shape (n, cnt) uint8. Flatten to 1D
        # for a contiguous slice-assign into field_data — no strided write,
        # no reshape contiguity worry (we control the destination layout).
        np_arr = _extract_field_to_numpy(col, f, cnt)
        start = int(field_data_starts[f_idx])
        end = start + n * cnt
        field_data[start:end] = np_arr.reshape(-1)

    return (
        id_hashes,
        id_byte_offsets,
        encoded_ids_concat,
        field_byte_offsets_in_row,
        field_byte_counts,
        field_data_starts,
        field_data,
    )


def insert_many(segment: Segment, table: pa.Table) -> int:
    """Bulk-insert a PyArrow table into a freshly-created segment.

    Required columns: ``entity_id`` (pa.string()) plus every
    ``schema.fields[i].name`` column with the field's compiled type.
    Other columns (e.g., ``event_time_ns``, ``msg_id_*``, ``label``) are
    ignored.

    Returns the number of rows inserted.

    Preconditions:
        * Segment must be freshly created (zero occupied slots). The
          kernel assumes all slots are EMPTY; calling on a populated
          segment produces silent corruption (kernel skips past existing
          OCCUPIED slots, writes past them).
        * Caller is the only writer to ``segment``. Invariant #3.

    Raises:
        ValueError: if the segment is not fresh, or required columns
            are missing.
        CapacityExceededError: if the slot table fills before all rows
            insert (sized capacity was wrong).
        StringPoolExhaustedError: if the entity_ids' total UTF-8 bytes
            exceeds the segment's pool budget.
    """
    if len(table) == 0:
        return 0

    # Fresh-segment validation. _read_metadata returns (next_row,
    # pool_cursor); both are zero on a freshly-created segment.
    next_row, pool_cursor = _read_metadata(segment.handle.buf)
    if next_row != 0 or pool_cursor != 0:
        raise ValueError(
            f"insert_many requires a fresh segment "
            f"(got next_row={next_row}, pool_cursor={pool_cursor}); "
            f"use layout.insert for non-bulk inserts on populated segments."
        )

    layout = segment.layout
    n = len(table)

    if n > layout.capacity:
        raise CapacityExceededError(
            f"table has {n} rows but segment capacity is {layout.capacity}; "
            f"use a larger capacity_factor or shard."
        )

    (
        id_hashes,
        id_byte_offsets,
        encoded_ids_concat,
        field_byte_offsets_in_row,
        field_byte_counts,
        field_data_starts,
        field_data,
    ) = _prepare_insert_columns(table, segment.schema, layout)

    # Validate per-ID byte budget before invoking the kernel — the kernel
    # writes blindly. Worst-case ID is the longest one in the batch.
    max_id_len = int(np.max(np.diff(id_byte_offsets))) if n > 0 else 0
    if max_id_len > layout.max_id_bytes:
        raise StringPoolExhaustedError(
            f"longest entity_id is {max_id_len} UTF-8 bytes; max is "
            f"max_id_bytes={layout.max_id_bytes}"
        )
    total_pool_bytes = int(id_byte_offsets[-1]) + n * _LEN_PREFIX_BYTES
    if total_pool_bytes > layout.string_pool_data_bytes:
        raise StringPoolExhaustedError(
            f"total string pool bytes needed ({total_pool_bytes}) > "
            f"available ({layout.string_pool_data_bytes})"
        )

    # Materialize a uint8 view of the segment buffer for the kernel.
    seg_u8 = np.frombuffer(segment.handle.buf, dtype=np.uint8)

    result = _insert_many_core(
        seg_u8,
        id_hashes,
        id_byte_offsets,
        encoded_ids_concat,
        np.int64(layout.slot_table_offset),
        np.int64(layout.slot_capacity),
        np.int64(layout.string_pool_data_offset),
        np.int64(layout.feature_rows_offset),
        np.int64(layout.row_size),
        np.int64(SLOT_BYTES),
        np.int64(SLOT_NAME_HASH_OFFSET),
        np.int64(SLOT_ID_OFFSET_OFFSET),
        np.int64(SLOT_FLAGS_OFFSET),
        np.int64(SLOT_FEATURE_ROW_INDEX_OFFSET),
        field_byte_offsets_in_row,
        field_byte_counts,
        field_data_starts,
        field_data,
    )

    if result < 0:
        # Kernel signalled capacity exhaustion at row -(result + 1).
        failed_row = -int(result) - 1
        raise CapacityExceededError(
            f"slot table fully scanned without finding empty slot at row "
            f"{failed_row}; sized capacity was {layout.capacity}, slot "
            f"capacity was {layout.slot_capacity}. This shouldn't happen "
            f"if capacity >= len(table) — check capacity_factor sizing."
        )

    # Write final cursors back to the segment header. _write_metadata
    # takes (next_row, pool_cursor); after inserting n rows, next_row
    # is n; pool_cursor is the kernel's return value.
    _write_metadata(segment.handle.buf, n, int(result))

    return n
