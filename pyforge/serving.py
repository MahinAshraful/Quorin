"""Serving path — assemble a feature vector from a Pyforge segment.

Step 4 ships the pure-Python "oracle" :func:`assemble`. Step 5's Numba kernel
will produce byte-identical output against this function; the property test in
Step 5 calls both and asserts ``np.array_equal``.

**Output ordering is declaration order, not hash order.** Models trained on
``[age, clicks, ltv, embedding]`` call ``model.predict(vec)`` expecting
``vec[0]`` to be ``age``. Hash order is internal scaffolding for
``searchsorted``-based lookup; it must never leak into a user-facing array.
The :class:`pyforge.layout.SegmentLayout` carries two parallel structures:
``row_offset_table`` (hash-sorted, used by :func:`pyforge.layout.lookup`) and
``assembly_table`` (declaration-order, iterated here). See ADR-003.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from pyforge.layout import lookup
from pyforge.schema import DTYPE_TO_NUMPY, DType

if TYPE_CHECKING:
    from pyforge.shm import Segment


class EntityNotFoundError(RuntimeError):
    """Raised by :func:`assemble` when no row exists for the given entity_id.

    Inherits :class:`RuntimeError` to match the existing pyforge convention
    (``SchemaCRCMismatchError``, ``CapacityExceededError``,
    ``StringPoolExhaustedError``). The original ``entity_id`` is preserved on
    the instance for callers that want to log or rethrow with context.
    """

    def __init__(self, entity_id: str) -> None:
        super().__init__(f"entity_id={entity_id!r} not found in segment")
        self.entity_id = entity_id


def assemble(
    segment: Segment,
    entity_id: str,
    *,
    out: np.ndarray[Any, np.dtype[np.float32]] | None = None,
) -> np.ndarray[Any, np.dtype[np.float32]]:
    """Assemble the feature vector for ``entity_id`` as a 1-D float32 array.

    Output is C-contiguous, ``dtype=float32``, ``shape=(layout.total_element_count,)``.
    Elements are laid out in the **declaration order** of
    ``segment.schema.fields`` (NOT the hash order of the offset table — see
    ADR-003). Shaped fields flatten in C order (row-major). Non-float32 fields
    are cast on copy: float64 narrows, ints widen, uint8 promotes — NumPy's
    slice-assignment performs the cast.

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

    .. note::
        The per-field ``np.frombuffer`` views over ``segment.handle.buf``
        are local to this function and drop out of scope at iteration end —
        the mmap-view-scoping rule is satisfied without external bookkeeping.
        Do not return one of these intermediate views from a debugging hack;
        ``mmap.close()`` will start raising ``BufferError``.
    """
    row_offset = lookup(segment, entity_id)
    if row_offset is None:
        raise EntityNotFoundError(entity_id)

    layout = segment.layout
    buf = segment.handle.buf
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

    cursor = 0
    for row in layout.assembly_table:
        # ``DType(code)`` (not ``_DTYPE_TO_NUMPY[code]``) on purpose: the
        # IntEnum constructor raises ValueError for unknown codes, surfacing
        # corrupt-segment data rather than silently KeyError-ing.
        code = int(row["dtype_code"])
        elem_cnt = int(row["element_count"])
        byte_off = int(row["byte_offset"])

        src = np.frombuffer(
            buf,
            dtype=DTYPE_TO_NUMPY[DType(code)],
            count=elem_cnt,
            offset=row_offset + byte_off,
        )
        # Slice-assignment from a non-float32 source triggers an in-place cast
        # into ``out``. ``np.empty`` is C-contiguous; assignment preserves it.
        out[cursor : cursor + elem_cnt] = src
        cursor += elem_cnt

    return out
