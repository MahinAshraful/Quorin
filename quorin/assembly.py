"""Numba-compiled assembly kernel.

Step 5 ships a Numba JIT-compiled equivalent of :func:`quorin.serving.assemble`.
The two paths produce **byte-identical output** (verified by
``tests/property/test_assembly_parity.py``); adoption-as-default is gated on the
200-field benchmark showing >=3x speedup over the Python oracle (see ADR-004).

This module is **isolated** so that ``import quorin.serving`` does not pay
Numba's compilation cost. Only callers that explicitly import
``quorin.assembly`` trigger the LLVM/Numba toolchain init.

Design notes (see ADR-004 for the full rationale):

- **Single uint8 view + Numba ``.view(dtype)``.** One ``np.frombuffer`` per call
  reinterprets the segment as ``uint8[::1]``. Inside the @njit kernel, each
  field's slice is reinterpreted via ``.view(np.float32)`` etc. — supported on
  contiguous 1D arrays in nopython mode and alignment-safe because all field
  offsets are 64-byte cache-line aligned (Step 1 invariant).
- **fastmath=False is non-negotiable.** Fastmath licenses LLVM to assume no
  NaN/inf, which would break NaN bit-pattern parity with the Python oracle.
- **Explicit Numba signature** for fast cold compile and a single
  specialization (no implicit type promotion).
- **Pre-warm is opt-in.** Module load doesn't auto-compile; callers (tests,
  benchmark fixtures, production hot-path setup) call :func:`prewarm` to
  amortize the ~100-500 ms first-call compile.
"""

from __future__ import annotations

import sys
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import numba
import numpy as np
from numba import prange

from quorin._internal.hash_kernel import blake2b_8
from quorin._internal.lookup_kernel import lookup_jit
from quorin._internal.lookup_kernel import prewarm as _lookup_prewarm
from quorin.layout import (
    SLOT_BYTES,
    SLOT_FEATURE_ROW_INDEX_OFFSET,
    SLOT_FLAGS_OFFSET,
    SLOT_ID_OFFSET_OFFSET,
    SLOT_NAME_HASH_OFFSET,
)
from quorin.serving import EntityNotFoundError

if TYPE_CHECKING:
    from quorin.shm import Segment


# ---------------------------------------------------------------------------
# Numba kernel.
# ---------------------------------------------------------------------------


@numba.njit(  # type: ignore[untyped-decorator]
    "void(uint8[::1], int64, int64[::1], int64[::1], uint8[::1], int64[::1], float32[::1])",
    cache=True,
    boundscheck=False,
    fastmath=False,
)
def _assemble_core(
    seg_u8: np.ndarray[Any, np.dtype[np.uint8]],
    row_offset: int,
    byte_offsets: np.ndarray[Any, np.dtype[np.int64]],
    byte_counts: np.ndarray[Any, np.dtype[np.int64]],
    dtype_codes: np.ndarray[Any, np.dtype[np.uint8]],
    element_counts: np.ndarray[Any, np.dtype[np.int64]],
    out: np.ndarray[Any, np.dtype[np.float32]],
) -> None:
    """Walk the assembly table in declaration order, casting each field's
    bytes to float32 in the output buffer.

    Inputs are all flat homogeneous arrays (no Python objects, no structured
    dtypes) so Numba can specialize tightly. ``dtype_codes`` carries the
    :class:`quorin.schema.DType` IntEnum values 1-5.

    The branch ladder is one IR object — pre-warming on FLOAT32 alone compiles
    every branch.
    """
    cursor = 0
    n = len(byte_offsets)
    for i in range(n):
        start = row_offset + byte_offsets[i]
        cnt = byte_counts[i]
        ec = element_counts[i]
        code = dtype_codes[i]
        sub = seg_u8[start : start + cnt]
        if code == 1:  # FLOAT32
            v32 = sub.view(np.float32)
            for j in range(ec):
                out[cursor + j] = v32[j]
        elif code == 2:  # FLOAT64
            v64 = sub.view(np.float64)
            for j in range(ec):
                out[cursor + j] = np.float32(v64[j])
        elif code == 3:  # INT32
            vi32 = sub.view(np.int32)
            for j in range(ec):
                out[cursor + j] = np.float32(vi32[j])
        elif code == 4:  # INT64
            vi64 = sub.view(np.int64)
            for j in range(ec):
                out[cursor + j] = np.float32(vi64[j])
        else:  # UINT8 (code == 5)
            for j in range(ec):
                out[cursor + j] = np.float32(sub[j])
        cursor += ec


# ---------------------------------------------------------------------------
# Public wrapper.
# ---------------------------------------------------------------------------


def assemble(
    segment: Segment,
    entity_id: str,
    *,
    out: np.ndarray[Any, np.dtype[np.float32]] | None = None,
) -> np.ndarray[Any, np.dtype[np.float32]]:
    """Numba-compiled equivalent of :func:`quorin.serving.assemble`.

    Output is C-contiguous, 1-D ``np.float32`` of shape
    ``(layout.total_element_count,)`` with elements in the **declaration order**
    of ``segment.schema.fields``. Byte-identical to the Python oracle for any
    valid segment state (verified by Step 5's parity test).

    Args:
        segment: open segment to read from.
        entity_id: row key.
        out: optional caller-supplied buffer. When provided, must be a
            writeable C-contiguous float32 array of shape
            ``(layout.total_element_count,)``; it is filled in place and
            returned. When ``None`` (default), a fresh allocation is
            returned. Step 6's :class:`quorin.pool.BufferPool` is the
            intended source of pooled buffers.

    Raises:
        ValueError: if ``entity_id`` is empty (delegated to
            :func:`quorin._internal.lookup_kernel.lookup_jit`), or if ``out``
            does not match the required shape / dtype / contiguity / writeability.
        EntityNotFoundError: if no row exists for ``entity_id``.
    """
    row_offset = lookup_jit(segment, entity_id)
    if row_offset is None:
        raise EntityNotFoundError(entity_id)

    layout = segment.layout
    n = layout.total_element_count
    if out is None:
        out = np.empty(n, dtype=np.float32)
    elif (
        out.shape != (n,)
        or out.dtype != np.float32
        or not out.flags["C_CONTIGUOUS"]
        or not out.flags["WRITEABLE"]
    ):
        raise ValueError(
            f"out must be a writeable C-contiguous float32 array of "
            f"shape ({n},); got shape={out.shape}, dtype={out.dtype}, "
            f"c_contiguous={out.flags['C_CONTIGUOUS']}, "
            f"writeable={out.flags['WRITEABLE']}"
        )
    seg_u8 = np.frombuffer(segment.handle.buf, dtype=np.uint8)

    _assemble_core(
        seg_u8,
        row_offset,
        layout.assembly_byte_offsets,
        layout.assembly_byte_counts,
        layout.assembly_dtype_codes,
        layout.assembly_element_counts,
        out,
    )
    return out


# ---------------------------------------------------------------------------
# Pre-warm — opt-in, idempotent.
# ---------------------------------------------------------------------------


def prewarm() -> None:
    """Force compilation of both Numba kernels on tiny dummy inputs.

    First call costs ~100-500 ms (Numba's LLVM lowering) per kernel;
    subsequent calls are no-ops because Numba caches the compiled artifact
    in ``__pycache__/`` (cache=True) and in process memory.

    NOT auto-run at module import — callers (tests, benchmark fixtures,
    production startup) invoke explicitly so importing this module stays
    cheap. Pre-warming on the FLOAT32 branch alone compiles the whole
    function; the @njit decorator produces one IR object covering all
    five dtype branches.

    Compiles :func:`_assemble_core` (single-entity, Step 5),
    :func:`_assemble_batch_core` (batch, Step 8), and
    :func:`quorin._internal.lookup_kernel._lookup_core` (single-entity
    lookup, Step 16c). Single entry point so callers don't have to know
    about the kernel split.
    """
    # Lookup kernel first (Step 16c). quorin.assembly.assemble calls into
    # it via lookup_jit, so warming the lookup kernel before the assemble
    # kernel keeps the dispatch order honest if a caller benches `assemble`
    # right after prewarm() returns.
    _lookup_prewarm()
    seg_u8 = np.zeros(64, dtype=np.uint8)
    _assemble_core(
        seg_u8,
        np.int64(0),
        np.array([0], dtype=np.int64),
        np.array([4], dtype=np.int64),
        np.array([1], dtype=np.uint8),  # FLOAT32 branch
        np.array([1], dtype=np.int64),
        np.empty(1, dtype=np.float32),
    )
    # Batch kernel prewarm. Tiny synthetic segment that won't be probed
    # successfully (slot table all zeros => EMPTY, miss-row branch fires).
    # That branch compiles the zero-fill path; the hit branch's assembly
    # ladder is the same code shape as _assemble_core's, so this prewarm
    # plus the single-entity prewarm above covers every IR path.
    batch_seg = np.zeros(256, dtype=np.uint8)
    batch_args = (
        batch_seg,
        np.array([0], dtype=np.uint64),
        np.zeros((1, 1), dtype=np.uint8),
        np.array([1], dtype=np.int64),
        np.int64(64),  # slot_table_offset
        np.int64(2),  # slot_capacity (power of 2)
        np.int64(128),  # string_pool_data_offset
        np.int64(192),  # feature_rows_offset
        np.int64(64),  # row_size
        np.int64(SLOT_BYTES),
        np.int64(SLOT_NAME_HASH_OFFSET),
        np.int64(SLOT_ID_OFFSET_OFFSET),
        np.int64(SLOT_FLAGS_OFFSET),
        np.int64(SLOT_FEATURE_ROW_INDEX_OFFSET),
        np.array([0], dtype=np.int64),
        np.array([4], dtype=np.int64),
        np.array([1], dtype=np.uint8),
        np.array([1], dtype=np.int64),
        np.empty((1, 1), dtype=np.float32),
        np.empty(1, dtype=np.bool_),
    )
    _assemble_batch_core(*batch_args)
    # Parallel kernel separately. Cold compile is more expensive than the
    # serial kernel (~500-1500 ms) because the parallel transform produces
    # additional IR. Subsequent calls hit the Numba cache and the OS
    # thread-pool warmup.
    parallel_args = (
        batch_seg,
        np.array([0], dtype=np.uint64),
        np.zeros((1, 1), dtype=np.uint8),
        np.array([1], dtype=np.int64),
        np.int64(64),
        np.int64(2),
        np.int64(128),
        np.int64(192),
        np.int64(64),
        np.int64(SLOT_BYTES),
        np.int64(SLOT_NAME_HASH_OFFSET),
        np.int64(SLOT_ID_OFFSET_OFFSET),
        np.int64(SLOT_FLAGS_OFFSET),
        np.int64(SLOT_FEATURE_ROW_INDEX_OFFSET),
        np.array([0], dtype=np.int64),
        np.array([4], dtype=np.int64),
        np.array([1], dtype=np.uint8),
        np.array([1], dtype=np.int64),
        np.empty((1, 1), dtype=np.float32),
        np.empty(1, dtype=np.bool_),
    )
    _assemble_batch_core_parallel(*parallel_args)


# ---------------------------------------------------------------------------
# Batch kernel — fused lookup + assemble (Step 8).
#
# Two kernels exist with byte-identical bodies:
#
#   _assemble_batch_core           — serial (range over batch).
#   _assemble_batch_core_parallel  — parallel (prange + parallel=True).
#
# The wrapper :func:`assemble_batch` dispatches to one or the other based on
# ``len(entity_ids) >= PARALLEL_THRESHOLD``. Branched dispatch is necessary
# because the parallel kernel pays thread-pool overhead (~10s of µs) on every
# call regardless of batch size; for small batches the serial path wins.
#
# **The two kernel bodies MUST stay in sync.** Any correctness fix must be
# applied to both. Property tests cover both via ``PARALLEL_THRESHOLD`` monkey-
# patching (see tests/unit/test_batch_assembly.py).
#
# Numba's parallel=True ships with its own threading layer (no new pip dep).
# Free-threaded CPython (PEP 703, post-3.13) is a future revisit; for the
# 3.12 deployment target the GIL-aware threading is correct.
# ---------------------------------------------------------------------------


def _compute_parallel_threshold() -> int:
    """Compute a defensible :data:`PARALLEL_THRESHOLD` from current Numba
    threading config.

    Calibrated from `benchmarks/runs/step8_threshold_sweep.py` on WSL2:

    | NUMBA_NUM_THREADS | Crossover N | Notes |
    |---|---|---|
    | 1                 | (never)     | parallel always loses; serial-only |
    | 2-4               | ~64         | parallel wins ~1.1-1.6x for N >= 64 |
    | 8+ (default)      | ~2048       | thread-spinup dominates; only very large batches amortize |

    The high-thread-count number is the killer: with default
    NUMBA_NUM_THREADS (= os.cpu_count() in many deployments), the parallel
    kernel REGRESSES vs serial at every N from 1 to 1000 in measurement.
    A static low threshold would be a hidden performance footgun for
    unconfigured deployments. Adaptive dispatch from current thread count
    avoids that.

    The threshold is computed once at module load. If you change
    ``numba.set_num_threads()`` at runtime, also reassign
    ``quorin.assembly.PARALLEL_THRESHOLD`` accordingly (or just monkey-patch
    it; it's a normal module attribute).
    """
    n = numba.get_num_threads()
    if n <= 1:
        return sys.maxsize  # never parallel; serial is always faster
    if n <= 4:
        return 64  # parallel wins at N >= 64 with 2-4 threads
    return 2048  # high thread count; only large batches amortize spinup


#: Batch-size crossover at which :func:`assemble_batch` switches from the
#: serial kernel to the parallel kernel. Below this, the serial kernel
#: avoids thread-pool spinup cost; above it, the parallel kernel wins.
#:
#: Computed adaptively from ``numba.get_num_threads()`` at module load
#: (see :func:`_compute_parallel_threshold` for the calibration table).
#: A wrong threshold is a perf bug, not a correctness bug — both kernels
#: produce byte-identical output.
#:
#: To override: ``quorin.assembly.PARALLEL_THRESHOLD = N`` after import.
#: Recommended deployment: set ``NUMBA_NUM_THREADS=4`` in the environment
#: before importing ``quorin`` for best batch perf across N.
PARALLEL_THRESHOLD: int = _compute_parallel_threshold()


@numba.njit(  # type: ignore[untyped-decorator]
    "void(uint8[::1], uint64[::1], uint8[:,::1], int64[::1],"
    " int64, int64, int64, int64, int64,"
    " int64, int64, int64, int64, int64,"
    " int64[::1], int64[::1], uint8[::1], int64[::1],"
    " float32[:,::1], boolean[::1])",
    cache=True,
    boundscheck=False,
    fastmath=False,
)
def _assemble_batch_core(  # noqa: PLR0912, PLR0915
    seg_u8: np.ndarray[Any, np.dtype[np.uint8]],
    id_hashes: np.ndarray[Any, np.dtype[np.uint64]],
    query_ids_padded: np.ndarray[Any, np.dtype[np.uint8]],
    query_id_lens: np.ndarray[Any, np.dtype[np.int64]],
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
    asm_byte_offsets: np.ndarray[Any, np.dtype[np.int64]],
    asm_byte_counts: np.ndarray[Any, np.dtype[np.int64]],
    asm_dtype_codes: np.ndarray[Any, np.dtype[np.uint8]],
    asm_element_counts: np.ndarray[Any, np.dtype[np.int64]],
    out: np.ndarray[Any, np.dtype[np.float32]],
    found_mask: np.ndarray[Any, np.dtype[np.bool_]],
) -> None:
    """Fused batched lookup + assembly.

    For each ``b`` in ``range(len(id_hashes))``:

    1. Linear-probe the slot table for ``id_hashes[b]`` with byte-compare
       disambiguation against ``query_ids_padded[b, :query_id_lens[b]]``.
    2. On hit: walk the assembly table in declaration order, casting each
       field's bytes to float32 in ``out[b, :]``. Set ``found_mask[b] = True``.
    3. On miss: zero ``out[b, :]`` in-place. Set ``found_mask[b] = False``.

    The slot table read uses scalar field offsets (``slot_*_off`` args)
    rather than structured-dtype access, both because Numba's structured-
    field-key handling is fragile in 0.60 and because the AoS slot table
    yields non-contiguous views that violate the ``[::1]`` signature.

    The string-pool entry format mirrors :func:`quorin.layout._read_string_bytes`:
    4-byte LE length at ``string_pool_data_offset + slot_id_offset``,
    followed by ``length`` UTF-8 bytes.

    Miss rows are zero-filled in-kernel so :class:`quorin.pool.BatchBufferPool`'s
    default ``zero_on_return=False`` is safe.
    """
    n_batch = len(id_hashes)
    n_fields = len(asm_byte_offsets)
    elem_count = out.shape[1]
    mask = slot_capacity - 1  # slot_capacity is power-of-2

    for b in range(n_batch):
        h = id_hashes[b]
        qlen = query_id_lens[b]
        idx = h & mask
        row_offset = -1

        # Bounded linear probe — at most slot_capacity iterations.
        for _probe in range(slot_capacity):
            slot_base = slot_table_offset + idx * slot_bytes

            f_off = slot_base + slot_flags_off
            slot_flags = seg_u8[f_off : f_off + 4].view(np.uint32)[0]
            if slot_flags == 0:  # EMPTY
                break

            h_off = slot_base + slot_name_hash_off
            slot_hash = seg_u8[h_off : h_off + 8].view(np.uint64)[0]
            if slot_flags == 1 and slot_hash == h:  # OCCUPIED + hash match
                io_off = slot_base + slot_id_off_off
                slot_id_offset = seg_u8[io_off : io_off + 8].view(np.uint64)[0]
                pool_entry_abs = string_pool_data_offset + slot_id_offset
                pool_len = seg_u8[pool_entry_abs : pool_entry_abs + 4].view(np.uint32)[0]
                if pool_len == qlen:
                    match = True
                    for k in range(qlen):
                        if seg_u8[pool_entry_abs + 4 + k] != query_ids_padded[b, k]:
                            match = False
                            break
                    if match:
                        fri_off = slot_base + slot_feature_row_idx_off
                        feature_row_idx = seg_u8[fri_off : fri_off + 4].view(np.uint32)[0]
                        row_offset = feature_rows_offset + feature_row_idx * row_size
                        break

            idx = (idx + 1) & mask

        if row_offset == -1:
            # MISS: zero the row. Pool default is zero_on_return=False so
            # the buffer is dirty on entry; this branch makes the kernel
            # responsible for full-buffer overwrite (load-bearing for the
            # default to be safe — see BatchBufferPool docstring).
            for j in range(elem_count):
                out[b, j] = np.float32(0.0)
            found_mask[b] = False
        else:
            # HIT: assembly ladder is the same code shape as _assemble_core's
            # so float32 casts are byte-identical to the single-entity Numba
            # path (and therefore to the Python oracle).
            cursor = 0
            for f in range(n_fields):
                start = row_offset + asm_byte_offsets[f]
                cnt = asm_byte_counts[f]
                ec = asm_element_counts[f]
                code = asm_dtype_codes[f]
                sub = seg_u8[start : start + cnt]
                if code == 1:  # FLOAT32
                    v32 = sub.view(np.float32)
                    for j in range(ec):
                        out[b, cursor + j] = v32[j]
                elif code == 2:  # FLOAT64
                    v64 = sub.view(np.float64)
                    for j in range(ec):
                        out[b, cursor + j] = np.float32(v64[j])
                elif code == 3:  # INT32
                    vi32 = sub.view(np.int32)
                    for j in range(ec):
                        out[b, cursor + j] = np.float32(vi32[j])
                elif code == 4:  # INT64
                    vi64 = sub.view(np.int64)
                    for j in range(ec):
                        out[b, cursor + j] = np.float32(vi64[j])
                else:  # UINT8 (5)
                    for j in range(ec):
                        out[b, cursor + j] = np.float32(sub[j])
                cursor += ec
            found_mask[b] = True


@numba.njit(  # type: ignore[untyped-decorator]
    "void(uint8[::1], uint64[::1], uint8[:,::1], int64[::1],"
    " int64, int64, int64, int64, int64,"
    " int64, int64, int64, int64, int64,"
    " int64[::1], int64[::1], uint8[::1], int64[::1],"
    " float32[:,::1], boolean[::1])",
    cache=True,
    boundscheck=False,
    fastmath=False,
    parallel=True,
)
def _assemble_batch_core_parallel(  # noqa: PLR0912, PLR0915
    seg_u8: np.ndarray[Any, np.dtype[np.uint8]],
    id_hashes: np.ndarray[Any, np.dtype[np.uint64]],
    query_ids_padded: np.ndarray[Any, np.dtype[np.uint8]],
    query_id_lens: np.ndarray[Any, np.dtype[np.int64]],
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
    asm_byte_offsets: np.ndarray[Any, np.dtype[np.int64]],
    asm_byte_counts: np.ndarray[Any, np.dtype[np.int64]],
    asm_dtype_codes: np.ndarray[Any, np.dtype[np.uint8]],
    asm_element_counts: np.ndarray[Any, np.dtype[np.int64]],
    out: np.ndarray[Any, np.dtype[np.float32]],
    found_mask: np.ndarray[Any, np.dtype[np.bool_]],
) -> None:
    """Parallel variant of :func:`_assemble_batch_core`.

    Identical body, parallelized across the outer batch dimension via
    ``numba.prange``. Per-iteration writes target ``out[b, :]`` and
    ``found_mask[b]`` — disjoint per ``b`` — so the loop is trivially
    parallelizable with no shared state, no atomics, no locks.

    Pays thread-pool spinup overhead on every call. Use only when
    ``n_batch >= PARALLEL_THRESHOLD`` (the wrapper handles dispatch).

    **MUST stay byte-identical to ``_assemble_batch_core``.** Any change
    to per-row semantics (probe loop, byte compare, dtype cast ladder,
    miss zero-fill) MUST be replicated here. Property tests in
    tests/unit/test_batch_assembly.py monkey-patch ``PARALLEL_THRESHOLD``
    to exercise this kernel against the serial one.
    """
    n_batch = len(id_hashes)
    n_fields = len(asm_byte_offsets)
    elem_count = out.shape[1]
    mask = slot_capacity - 1

    for b in prange(n_batch):
        h = id_hashes[b]
        qlen = query_id_lens[b]
        idx = h & mask
        row_offset = -1

        for _probe in range(slot_capacity):
            slot_base = slot_table_offset + idx * slot_bytes

            f_off = slot_base + slot_flags_off
            slot_flags = seg_u8[f_off : f_off + 4].view(np.uint32)[0]
            if slot_flags == 0:
                break

            h_off = slot_base + slot_name_hash_off
            slot_hash = seg_u8[h_off : h_off + 8].view(np.uint64)[0]
            if slot_flags == 1 and slot_hash == h:
                io_off = slot_base + slot_id_off_off
                slot_id_offset = seg_u8[io_off : io_off + 8].view(np.uint64)[0]
                pool_entry_abs = string_pool_data_offset + slot_id_offset
                pool_len = seg_u8[pool_entry_abs : pool_entry_abs + 4].view(np.uint32)[0]
                if pool_len == qlen:
                    match = True
                    for k in range(qlen):
                        if seg_u8[pool_entry_abs + 4 + k] != query_ids_padded[b, k]:
                            match = False
                            break
                    if match:
                        fri_off = slot_base + slot_feature_row_idx_off
                        feature_row_idx = seg_u8[fri_off : fri_off + 4].view(np.uint32)[0]
                        row_offset = feature_rows_offset + feature_row_idx * row_size
                        break

            idx = (idx + 1) & mask

        if row_offset == -1:
            for j in range(elem_count):
                out[b, j] = np.float32(0.0)
            found_mask[b] = False
        else:
            cursor = 0
            for f in range(n_fields):
                start = row_offset + asm_byte_offsets[f]
                cnt = asm_byte_counts[f]
                ec = asm_element_counts[f]
                code = asm_dtype_codes[f]
                sub = seg_u8[start : start + cnt]
                if code == 1:
                    v32 = sub.view(np.float32)
                    for j in range(ec):
                        out[b, cursor + j] = v32[j]
                elif code == 2:
                    v64 = sub.view(np.float64)
                    for j in range(ec):
                        out[b, cursor + j] = np.float32(v64[j])
                elif code == 3:
                    vi32 = sub.view(np.int32)
                    for j in range(ec):
                        out[b, cursor + j] = np.float32(vi32[j])
                elif code == 4:
                    vi64 = sub.view(np.int64)
                    for j in range(ec):
                        out[b, cursor + j] = np.float32(vi64[j])
                else:
                    for j in range(ec):
                        out[b, cursor + j] = np.float32(sub[j])
                cursor += ec
            found_mask[b] = True


def assemble_batch(
    segment: Segment,
    entity_ids: Sequence[str],
    *,
    out: np.ndarray[Any, np.dtype[np.float32]] | None = None,
    found_mask: np.ndarray[Any, np.dtype[np.bool_]] | None = None,
) -> tuple[
    np.ndarray[Any, np.dtype[np.float32]],
    np.ndarray[Any, np.dtype[np.bool_]],
]:
    """Vectorized feature assembly for ``N`` entities into one 2D buffer.

    For each ``i`` in ``range(len(entity_ids))``:

    - If ``entity_ids[i]`` is found in the segment, ``out[i, :]`` holds its
      assembled float32 vector (declaration order, byte-identical to
      :func:`quorin.serving.assemble`) and ``found_mask[i] == True``.
    - Otherwise ``out[i, :]`` is zero-filled and ``found_mask[i] == False``.

    Args:
        segment: open segment to read from.
        entity_ids: sequence of entity IDs. Empty IDs raise ``ValueError``.
        out: optional caller-supplied 2D buffer. Must be shape
            ``(len(entity_ids), layout.total_element_count)``, dtype float32,
            C-contiguous, writeable. Source from :class:`quorin.pool.BatchBufferPool`
            for the steady-state path. When ``None``, a fresh allocation
            is returned.
        found_mask: optional caller-supplied 1D mask. Must be shape
            ``(len(entity_ids),)``, dtype bool, writeable. When ``None``, a
            fresh allocation is returned.

    Returns:
        ``(out, found_mask)``. The same buffers passed in (or freshly
        allocated when omitted), filled by the kernel.

    Raises:
        ValueError: any of: an entity_id is empty; ``out`` shape/dtype/
            contiguity/writeability mismatch; ``found_mask`` shape/dtype/
            writeability mismatch.
    """
    n_batch = len(entity_ids)
    layout = segment.layout
    elem_count = layout.total_element_count

    if out is None:
        out = np.empty((n_batch, elem_count), dtype=np.float32)
    elif (
        out.shape != (n_batch, elem_count)
        or out.dtype != np.float32
        or not out.flags["C_CONTIGUOUS"]
        or not out.flags["WRITEABLE"]
    ):
        raise ValueError(
            f"out must be a writeable C-contiguous float32 array of shape "
            f"({n_batch}, {elem_count}); got shape={out.shape}, "
            f"dtype={out.dtype}, c_contiguous={out.flags['C_CONTIGUOUS']}, "
            f"writeable={out.flags['WRITEABLE']}"
        )

    if found_mask is None:
        found_mask = np.empty(n_batch, dtype=np.bool_)
    elif (
        found_mask.shape != (n_batch,)
        or found_mask.dtype != np.bool_
        or not found_mask.flags["WRITEABLE"]
    ):
        raise ValueError(
            f"found_mask must be a writeable bool_ array of shape "
            f"({n_batch},); got shape={found_mask.shape}, "
            f"dtype={found_mask.dtype}, writeable={found_mask.flags['WRITEABLE']}"
        )

    if n_batch == 0:
        return out, found_mask

    # Python-side prep: str.encode isn't in Numba; the hash itself runs
    # inside ``blake2b_8`` (Numba kernel, ~150-300 ns vs Python hashlib's
    # ~1500 ns). Pinned-hash invariant #5: output is byte-identical to
    # ``hashlib.blake2b(bytes, digest_size=8)`` — locked by
    # ``tests/unit/test_hash_kernel.py``.
    #
    # ASCII fast-path encode: most production ML entity IDs (UUIDs,
    # integer-as-string, hex hashes) are ASCII. ``str.encode("ascii")`` is
    # ~50-80 ns/call faster than ``encode("utf-8")`` in CPython 3.12 because
    # the ASCII codec has a simpler dispatch path. UnicodeEncodeError fallback
    # preserves correctness for non-ASCII IDs (a small fraction of the work).
    encoded: list[bytes] = []
    max_len = 0
    for i, eid in enumerate(entity_ids):
        if not eid:
            raise ValueError(f"entity_ids[{i}] is empty")
        try:
            b = eid.encode("ascii")
        except UnicodeEncodeError:
            b = eid.encode("utf-8")
        encoded.append(b)
        max_len = max(max_len, len(b))

    # max_len can be 0 only if every id is empty, which we caught above.
    # Pad to (n_batch, max_len) for kernel-friendly indexing. Hash from the
    # padded array's writable contiguous slice (``[::1]``) so Numba accepts
    # it as input without an intermediate ``.copy()`` — recovers ~1.3-1.4x
    # of the lost native-CI batch ratio per ADR-007 §16c amendment.
    query_ids_padded = np.zeros((n_batch, max_len), dtype=np.uint8)
    query_id_lens = np.empty(n_batch, dtype=np.int64)
    id_hashes = np.empty(n_batch, dtype=np.uint64)
    for i, b in enumerate(encoded):
        bl = len(b)
        query_id_lens[i] = bl
        query_ids_padded[i, :bl] = np.frombuffer(b, dtype=np.uint8)
        id_hashes[i] = blake2b_8(query_ids_padded[i, :bl], bl)

    seg_u8 = np.frombuffer(segment.handle.buf, dtype=np.uint8)

    # Branched dispatch: parallel kernel for batches large enough to amortize
    # thread-pool spinup, serial otherwise. Threshold determined empirically;
    # a wrong threshold is a perf bug, not a correctness bug.
    kernel = (
        _assemble_batch_core_parallel if n_batch >= PARALLEL_THRESHOLD else _assemble_batch_core
    )
    kernel(
        seg_u8,
        id_hashes,
        query_ids_padded,
        query_id_lens,
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
        layout.assembly_byte_offsets,
        layout.assembly_byte_counts,
        layout.assembly_dtype_codes,
        layout.assembly_element_counts,
        out,
        found_mask,
    )
    return out, found_mask
