"""Numba-compiled assembly kernel.

Step 5 ships a Numba JIT-compiled equivalent of :func:`pyforge.serving.assemble`.
The two paths produce **byte-identical output** (verified by
``tests/property/test_assembly_parity.py``); adoption-as-default is gated on the
200-field benchmark showing >=3x speedup over the Python oracle (see ADR-004).

This module is **isolated** so that ``import pyforge.serving`` does not pay
Numba's compilation cost. Only callers that explicitly import
``pyforge.assembly`` trigger the LLVM/Numba toolchain init.

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

from typing import TYPE_CHECKING, Any

import numba
import numpy as np

from pyforge.layout import lookup
from pyforge.serving import EntityNotFoundError

if TYPE_CHECKING:
    from pyforge.shm import Segment


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
    :class:`pyforge.schema.DType` IntEnum values 1-5.

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
    """Numba-compiled equivalent of :func:`pyforge.serving.assemble`.

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
            returned. Step 6's :class:`pyforge.pool.BufferPool` is the
            intended source of pooled buffers.

    Raises:
        ValueError: if ``entity_id`` is empty (delegated to
            :func:`pyforge.layout.lookup`), or if ``out`` does not match the
            required shape / dtype / contiguity / writeability.
        EntityNotFoundError: if no row exists for ``entity_id``.
    """
    row_offset = lookup(segment, entity_id)
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
    """Force compilation of :func:`_assemble_core` on a tiny dummy input.

    First call costs ~100-500 ms (Numba's LLVM lowering); subsequent calls
    are no-ops because Numba caches the compiled artifact in
    ``__pycache__/`` (cache=True) and in process memory.

    NOT auto-run at module import — callers (tests, benchmark fixtures,
    production startup) invoke explicitly so importing this module stays
    cheap. Pre-warming on the FLOAT32 branch alone compiles the whole
    function; the @njit decorator produces one IR object covering all
    five dtype branches.
    """
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
