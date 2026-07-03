"""Numba-compiled single-entity lookup kernel for the serving hot path (Step 16c).

Mirror image of Step 13's :mod:`quorin._internal.insert_kernel` for the
read side: that kernel writes slot records + feature rows; this one
walks the slot table to find them. Both share the slot byte-layout
discipline locked by invariant #15 — the kernel reads raw bytes at
offsets passed in as scalar args, never hardcoded magic numbers.

This module is **isolated** so importing :mod:`quorin.layout` does not
pay Numba's compile cost (invariant #11). Only callers that explicitly
import ``quorin._internal.lookup_kernel`` (currently only
:mod:`quorin.assembly`'s single-entity path) trigger LLVM/Numba init.

Why this exists
---------------
Pure-Python :func:`quorin.layout.lookup` is ~2.91 µs hit / ~1.6 µs miss
median (Step 3 measurement). At the 4-field warm assemble path's ~4.2 µs
median, lookup is ~70% of the budget; the Step 5 Numba assemble kernel
is the other ~30%. Step 16c's trip-wire (4-field warm p99 9.24 µs vs
the 5 µs spec) is dominated by the Python lookup's tail variance.
Lookup-jit replaces the Python probe loop + structured-dtype slot reads
+ ``struct.unpack_from``-driven string-pool reads with a single Numba
kernel call (~200 ns kernel + ~150 ns dispatch).

Step 16c-d extension: BLAKE2b is computed **inside the kernel** via
:func:`quorin._internal.hash_kernel.blake2b_8` instead of in the Python
wrapper. Saves the ~1500 ns ``hashlib.blake2b`` per-call cost. Wrapper
now does only encode + ``.copy()`` + ``np.frombuffer`` + kernel dispatch
(~250 ns total); the kernel call does hash + probe + byte-compare in
one Numba dispatch. Closes the 5 µs trip-wire.

Design notes
------------
- **Returns int64, not Optional[int].** ``-1`` sentinel for miss; the
  Python wrapper translates to ``None``. Numba doesn't speak Python's
  ``None``; integer return is the standard pattern (mirrors
  ``insert_kernel``'s sentinel-return convention).

- **All slot byte offsets passed as int64 scalar args** (invariant #15).
  Constants come from :data:`quorin.layout.SLOT_DTYPE`'s field map at
  module load. A future ``SLOT_DTYPE`` reorder would fail loudly at
  ``quorin.layout`` import (where the asserts live), not silently
  inside this kernel returning wrong rows.

- **fastmath=False** (invariant #12). Lookup does no FP arithmetic, but
  the discipline is uniform across all Quorin Numba kernels.

- **Probe-loop shape mirrors :func:`quorin.assembly._assemble_batch_core`'s
  inner probe** (the Step 8 kernel). Same OCCUPIED/EMPTY semantics, same
  byte-compare disambiguation against the string-pool entry, same
  bounded ``slot_capacity`` iterations.

- **Pre-warm is opt-in.** Module load doesn't auto-compile;
  :func:`quorin.assembly.prewarm` calls into :func:`prewarm` here so
  callers don't have to think about per-kernel warmup.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numba
import numpy as np

from quorin._internal.hash_kernel import blake2b_8
from quorin._internal.hash_kernel import prewarm as _hash_prewarm
from quorin.layout import (
    SLOT_BYTES,
    SLOT_FEATURE_ROW_INDEX_OFFSET,
    SLOT_FLAGS_OFFSET,
    SLOT_ID_OFFSET_OFFSET,
    SLOT_NAME_HASH_OFFSET,
)

if TYPE_CHECKING:
    from quorin.shm import Segment


@numba.njit(  # type: ignore[untyped-decorator]
    "int64(uint8[::1], uint8[::1], int64,"
    " int64, int64, int64, int64, int64,"
    " int64, int64, int64, int64, int64, int64)",
    cache=True,
    boundscheck=False,
    fastmath=False,
)
def _lookup_core(
    seg_u8: np.ndarray[Any, np.dtype[np.uint8]],
    encoded_id: np.ndarray[Any, np.dtype[np.uint8]],
    encoded_id_len: int,
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
    max_id_bytes: int,
) -> int:
    """Linear-probe lookup against the AoS slot table.

    Returns ``feature_rows_offset + feature_row_index * row_size`` on
    hit, ``-1`` on miss. Mirrors :func:`quorin.layout.lookup`'s
    probe semantics exactly.

    BLAKE2b is computed inside the kernel via
    :func:`quorin._internal.hash_kernel.blake2b_8` (Step 16c-d). Saves
    ~1300 ns vs paying ``hashlib.blake2b`` in the Python wrapper —
    closes the 4-field warm assemble trip-wire (5 µs spec p99).

    The byte-compare against the string-pool entry mirrors
    :func:`quorin.layout._read_string_bytes`'s read pattern: ``uint32``
    length prefix at ``pool_entry_abs``, followed by ``length`` UTF-8
    bytes. The kernel's per-byte loop is fast enough at typical
    ``encoded_id_len <= 64`` that a vectorized compare doesn't pay.

    Memory ordering: writes to the slot table happen in
    :func:`quorin.layout._insert_into_empty_slot` with ``flags`` written
    LAST (invariant #8). So if this kernel reads ``flags == OCCUPIED``,
    the ``name_hash`` / ``id_offset`` / ``feature_row_index`` fields are
    already valid. Concurrent inserts cannot corrupt our view.

    The ``encoded_id_len`` parameter (vs ``encoded_id.shape[0]``) lets
    the wrapper pass a fixed-size scratch buffer with variable fill
    length — useful if a future refactor reuses ``encoded_id`` across
    calls.
    """
    id_hash = blake2b_8(encoded_id, encoded_id_len)
    mask = slot_capacity - 1
    idx = id_hash & mask

    for _probe in range(slot_capacity):
        slot_base = slot_table_offset + idx * slot_bytes

        f_off = slot_base + slot_flags_off
        slot_flags = seg_u8[f_off : f_off + 4].view(np.uint32)[0]
        if slot_flags == 0:  # EMPTY — probe stops; entity not present
            return -1

        h_off = slot_base + slot_name_hash_off
        slot_hash = seg_u8[h_off : h_off + 8].view(np.uint64)[0]
        if slot_flags == 1 and slot_hash == id_hash:  # OCCUPIED + hash match
            io_off = slot_base + slot_id_off_off
            slot_id_offset = seg_u8[io_off : io_off + 8].view(np.uint64)[0]
            pool_entry_abs = string_pool_data_offset + slot_id_offset
            pool_len = seg_u8[pool_entry_abs : pool_entry_abs + 4].view(np.uint32)[0]
            # CR.B.2: defense-in-depth. A corrupt string-pool entry with
            # ``pool_len > max_id_bytes`` would, with ``boundscheck=False``,
            # let the byte-compare below read past the string-pool region
            # — producing a false-positive hit if those bytes happen to
            # match the query, or a silent false-miss otherwise. Bound the
            # value to the layout-declared ceiling; treat corrupt entries
            # as ``not-found`` (kernel returns -1; Python oracle raises
            # ``ValueError`` instead — documented divergence, kernel is
            # fast-path so degraded-correct on corruption is acceptable).
            if pool_len > max_id_bytes:
                return -1
            if pool_len == encoded_id_len:
                match = True
                for k in range(encoded_id_len):
                    if seg_u8[pool_entry_abs + 4 + k] != encoded_id[k]:
                        match = False
                        break
                if match:
                    fri_off = slot_base + slot_feature_row_idx_off
                    feature_row_idx = seg_u8[fri_off : fri_off + 4].view(np.uint32)[0]
                    # mypy sees np.uint32 arithmetic as Any. Numba's signature
                    # ("int64(...)" above) constrains the actual return type to
                    # int64; the comment is the only mypy can see.
                    return feature_rows_offset + feature_row_idx * row_size  # type: ignore[no-any-return]

        idx = (idx + 1) & mask

    # Walked the whole table — capacity-cap-at-50% (invariant #7) means
    # this is unreachable in normal operation; fall through to "not found".
    return -1


def lookup_jit(segment: Segment, entity_id: str) -> int | None:
    """Numba-compiled equivalent of :func:`quorin.layout.lookup`.

    Byte-identical return value to the Python oracle for any segment
    state. Verified by ``tests/property/test_lookup_jit_parity.py``.

    Args:
        segment: open segment to look up in.
        entity_id: row key (UTF-8 string, must be non-empty).

    Returns:
        Byte offset of the entity's row inside the segment, or
        ``None`` if the entity is not present.

    Raises:
        ValueError: if ``entity_id`` is empty (matches Python lookup).
    """
    if not entity_id:
        raise ValueError("entity_id must be non-empty")

    layout = segment.layout
    encoded = entity_id.encode("utf-8")
    # Numba 0.60's ``uint8[::1]`` dispatch distinguishes writable vs
    # ``readonly array(uint8, 1d, C)`` and rejects the latter (verified
    # empirically — dropping ``.copy()`` raises ``TypeError: No matching
    # definition`` at first call). ``np.frombuffer(bytes, ...)`` returns
    # a read-only view because Python ``bytes`` is immutable; ``.copy()``
    # materializes a writable contiguous array. ~80 ns at max-64-byte IDs.
    #
    # The ``assemble_batch`` path uses a writable padded array because it
    # ASSIGNS frombuffer slices INTO ``query_ids_padded = np.zeros(...)``;
    # the writable flag comes from the target, not the source. Tried in
    # v0.1.2 plan (QW-1), reverted after parity tests went red.
    encoded_arr = np.frombuffer(encoded, dtype=np.uint8).copy()

    seg_u8 = np.frombuffer(segment.handle.buf, dtype=np.uint8)

    # _lookup_core computes the BLAKE2b hash internally via blake2b_8 — saves
    # the ~1500 ns Python hashlib call that previously lived in this wrapper.
    result = _lookup_core(
        seg_u8,
        encoded_arr,
        len(encoded),
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
        layout.max_id_bytes,  # CR.B.2 defense-in-depth bound
    )
    return None if result == -1 else int(result)


def prewarm() -> None:
    """Force compilation of :func:`_lookup_core` (and the BLAKE2b kernel
    it transitively calls) on a tiny synthetic input.

    First call costs ~200-400 ms (Numba's LLVM lowering for both the
    lookup probe and the inlined BLAKE2b compression); subsequent calls
    are no-ops because Numba caches the compiled artifact in
    ``__pycache__/`` (``cache=True``) and in process memory.

    NOT auto-run at module import — callers (tests, benchmark fixtures,
    production startup) invoke explicitly so importing this module stays
    cheap. :func:`quorin.assembly.prewarm` calls into here so a single
    prewarm covers all assembly + lookup + hash kernels.
    """
    # Warm the hash kernel first; _lookup_core calls it transitively but
    # warming it explicitly here ensures a single prewarm() covers BOTH IR
    # objects regardless of Numba's cross-kernel call resolution order.
    _hash_prewarm()

    # Tiny synthetic segment: 256 bytes is enough for the kernel to walk
    # one slot's worth of bytes and return -1 without touching anything
    # interesting. We just need to force the LLVM lowering of the one
    # IR object the @njit decorator produces.
    seg_u8 = np.zeros(256, dtype=np.uint8)
    encoded = np.zeros(1, dtype=np.uint8)
    _lookup_core(
        seg_u8,
        encoded,
        np.int64(1),  # encoded_id_len
        np.int64(64),  # slot_table_offset
        np.int64(2),  # slot_capacity (power of 2)
        np.int64(128),  # string_pool_data_offset
        np.int64(192),  # feature_rows_offset
        np.int64(64),  # row_size
        SLOT_BYTES,
        SLOT_NAME_HASH_OFFSET,
        SLOT_ID_OFFSET_OFFSET,
        SLOT_FLAGS_OFFSET,
        SLOT_FEATURE_ROW_INDEX_OFFSET,
        np.int64(64),  # max_id_bytes
    )
