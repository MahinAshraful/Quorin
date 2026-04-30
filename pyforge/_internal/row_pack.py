"""Row-buffer packer for the WAL consumer (Step 10).

Step 10's :class:`pyforge.wal_consumer.WALConsumer` decodes the producer's
msgpack-list payload into a list of native Python values in
**name_hash order**, then needs to write those values into a fixed-size
``bytearray`` matching ``pyforge.schema.row_size(schema)`` so that
``pyforge.layout.insert(seg, entity_id, row_buffer)`` lands the row at the
right per-field byte offsets.

This module supplies that packer. It mirrors the test helper
``tests/_helpers.py::pack_row`` in spirit, but the production caller has
a different shape: positional list (not name-keyed dict) and an in-place
bytearray (not a fresh return). Memoization gives us a per-schema "pack
plan" that resolves once and dispatches in microseconds thereafter.

Locked design (ADR-009 §7):

1. **Plan walks** :func:`pyforge.schema.compute_row_offset_table` —
   name_hash-sorted, row-relative offsets. Walking
   :func:`pyforge.schema.compute_assembly_table` (declaration order)
   would silently corrupt rows on every schema where declaration ≠
   name_hash order. The producer's wire format is name_hash order
   (ADR-008); the consumer must match.
2. **One consolidated** :class:`struct.Struct` **for every scalar field**
   in the schema, with ``Nx`` pad-byte format codes for cache-line gaps
   and for shaped-field regions. A 200-field float32 schema becomes
   ``"<f60xf60x...f"`` (200 ``f`` codes, 199 ``60x`` pads); one
   :meth:`struct.Struct.pack_into` call writes all 200 floats at their
   correct cache-line-aligned offsets in a single C pass. Without the
   pads, the consolidator would only catch field 0 (because Pyforge
   aligns every field start to 64 bytes; see
   :func:`pyforge.schema._field_byte_offsets`) and dump 199 fields into
   the slow per-field numpy path — net ~30 µs of dispatch overhead at
   200 fields.
3. **Shaped fields** (``element_count > 1``) are written via direct
   :func:`numpy.frombuffer` views into the output bytearray. The Struct
   skips them via ``Nx`` pads so subsequent scalars still land at the
   correct offset. ``np.copyto(view, src)`` is zero-copy on the
   destination side; the only allocation is ``np.asarray(v, dtype=...)``
   on the source, and that's a no-op when the dtype already matches.
4. **Inter-field padding bytes are not zeroed.** The caller-supplied
   bytearray is reused across calls; trailing bytes from prior packs
   persist in the gap regions. This is safe because
   :func:`pyforge.serving.assemble` walks the offset table — it reads
   only ``(byte_offset, byte_count)`` extents and never touches padding
   bytes. ``pyforge.layout.insert`` copies the full row into shm
   verbatim; the padding bytes in shm are likewise never read.
5. **Memoization is by class identity** (not by name) so Step 15's
   atomic class swap will get a fresh plan automatically.

Estimated per-call cost at 200 scalar fields: ~10-12 µs. Plan cache
lookup ~50 ns after warm-up.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from typing import Any, Final

import numpy as np

from pyforge.schema import (
    DTYPE_TO_NUMPY,
    DType,
    FeatureSchema,
    _hash_name,
    compute_row_offset_table,
    row_size,
)

# ---------------------------------------------------------------------------
# DType → struct format code. Little-endian fixed-size codes only — we never
# mix in the platform-dependent variants (`l`, `n`, etc.) because the wire
# format must round-trip across machines.
# ---------------------------------------------------------------------------

_DTYPE_TO_STRUCT_CODE: Final[dict[DType, str]] = {
    DType.FLOAT32: "f",
    DType.FLOAT64: "d",
    DType.INT32: "i",
    DType.INT64: "q",
    DType.UINT8: "B",
}


# ---------------------------------------------------------------------------
# Pack plan dataclasses. Frozen + slotted; each field is read-only and
# referenced from the hot path so attribute access costs ~30 ns.
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _ShapedSlot:
    """One shaped field's destination in the row buffer."""

    byte_offset: int
    numpy_dtype: np.dtype[Any]
    element_count: int
    byte_count: int


@dataclass(frozen=True, slots=True)
class _PackPlan:
    """Memoized per-schema pack plan. Built once, dispatched many times."""

    scalar_struct: struct.Struct | None
    """Consolidated little-endian struct covering the whole row, with ``Nx``
    pads for cache-line gaps and shaped-field regions. ``None`` only when
    the schema has zero scalar fields (rare; skipping the call is cheaper)."""

    scalar_indices: tuple[int, ...]
    """Indices into the caller's ``values`` list for the scalar args, in the
    same order the consolidated Struct expects them."""

    shaped_slots: tuple[tuple[int, _ShapedSlot], ...]
    """``(idx_in_values, slot)`` for each shaped field. The numpy fill pass
    walks this in order; ordering doesn't affect correctness because writes
    don't overlap."""

    expected_len: int
    """Total field count == ``len(values)``. Validated at every call."""

    row_size: int
    """``pyforge.schema.row_size(schema)``. Validated against ``len(out)`` at
    every call."""


# Module-level cache, process-global. Keyed by class identity so Step 15's
# atomic class swap gets a fresh plan automatically.
_PLAN_CACHE: dict[type[FeatureSchema], _PackPlan] = {}


# ---------------------------------------------------------------------------
# Public API.
# ---------------------------------------------------------------------------


def pack_row_from_list(
    schema: type[FeatureSchema],
    values: list[Any],
    out: bytearray,
) -> None:
    """Pack ``values`` (name_hash-ordered) into ``out`` per ``schema``.

    Args:
        schema: The :class:`FeatureSchema` subclass; first call per class
            builds and caches the pack plan (~1 ms), subsequent calls do a
            ~50 ns dict lookup.
        values: One Python value per field, in **name_hash order** (matches
            the producer's wire format from ADR-008). Length must equal the
            schema's field count.
        out: Caller-supplied bytearray of length ``row_size(schema)``. The
            consumer reuses one buffer per schema across calls; trailing
            bytes from prior packs in unwritten regions are harmless because
            ``assemble`` and ``layout.insert`` ignore padding.

    Raises:
        ValueError: ``len(values) != schema.fields`` count, or
            ``len(out) != row_size(schema)``, or any shaped value's
            ``element_count`` doesn't match the schema.

    The function is the consumer's hot path; expect ~10-12 µs at 200
    scalar fields and ~30-50 µs at 200 fields with one 128-dim shaped
    field.
    """
    plan = _PLAN_CACHE.get(schema)
    if plan is None:
        plan = _build_plan(schema)
        _PLAN_CACHE[schema] = plan

    if len(values) != plan.expected_len:
        raise ValueError(f"row_pack: expected {plan.expected_len} values, got {len(values)}")
    if len(out) != plan.row_size:
        raise ValueError(f"row_pack: out length {len(out)} != row_size {plan.row_size}")

    if plan.scalar_struct is not None:
        # ONE pack_into writes ALL scalars. Pad codes (Nx) skip cache-line
        # gaps and shaped-field regions automatically. The generator
        # materializes to a tuple internally for the *args expansion;
        # cheap at 200 entries.
        plan.scalar_struct.pack_into(
            out,
            0,
            *(values[i] for i in plan.scalar_indices),
        )

    for idx, slot in plan.shaped_slots:
        v = values[idx]
        # Direct write into a numpy view of `out` — no .tobytes() allocation
        # on the destination side. np.asarray() is a no-op if v's dtype
        # already matches; otherwise it produces a converted copy and we
        # then memcpy the bytes via np.copyto.
        view = np.frombuffer(
            out,
            dtype=slot.numpy_dtype,
            count=slot.element_count,
            offset=slot.byte_offset,
        )
        src = np.asarray(v, dtype=slot.numpy_dtype).reshape(-1)
        if src.size != slot.element_count:
            raise ValueError(
                f"row_pack[{idx}]: element_count {src.size} != expected {slot.element_count}"
            )
        np.copyto(view, src)


def clear_cache() -> None:
    """Drop all memoized pack plans. Test-only; do not call from production code."""
    _PLAN_CACHE.clear()


# ---------------------------------------------------------------------------
# Internal builder.
# ---------------------------------------------------------------------------


def _build_plan(schema: type[FeatureSchema]) -> _PackPlan:
    """Build the pack plan for ``schema``.

    The :func:`compute_row_offset_table` rows are sorted by ``name_hash``,
    not by ``byte_offset`` — so the i-th row corresponds to ``values[i]``
    on the consumer side, but its ``byte_offset`` is essentially random.
    To build a single :class:`struct.Struct` that places each value at its
    correct byte offset, we must:

    1. Tag each table row with its **name_hash index** ``i`` (== the
       index in the consumer's ``values`` list).
    2. Sort by ``byte_offset`` (the order the Struct walks the buffer).
    3. Emit the format string in byte-offset order with ``Nx`` pads for
       gaps, recording each scalar's value-index in struct-arg order.

    The result: ``scalar_indices`` maps "the k-th scalar arg the Struct
    consumes" to "its position in the consumer's ``values`` list," so
    ``*(values[i] for i in scalar_indices)`` produces args in the order
    the Struct expects, and each value lands at its correct byte offset.

    For shaped fields (``shape != ()``), byte order doesn't matter
    (numpy writes at explicit offsets); we still emit ``{byte_count}x``
    in the Struct format so subsequent scalars line up correctly.

    Note: ``shape == ()`` (true scalar) and ``shape == (1,)`` both have
    ``element_count == 1``, but the producer's wire format ships a Python
    scalar for the former and a 1-elt list for the latter. The OFFSET
    table only carries ``element_count``, so we cross-reference
    ``schema.fields`` (matched by name_hash) to recover the original
    shape and route correctly.
    """
    table = compute_row_offset_table(schema)

    # name_hash -> shape, looked up while walking the table.
    shape_by_hash: dict[int, tuple[int, ...]] = {
        # Lazy import: pyforge.schema._hash_name is private; we already
        # import the rest of the schema module.
        _hash_name(f.name): f.shape
        for f in schema.fields
    }

    # (byte_off, byte_cnt, dtype_code, elem_cnt, is_true_scalar, value_idx).
    # Sorted by byte_off — the order the Struct format walks.
    rows: list[tuple[int, int, int, int, bool, int]] = sorted(
        (
            (
                int(table[i]["byte_offset"]),
                int(table[i]["byte_count"]),
                int(table[i]["dtype_code"]),
                int(table[i]["element_count"]),
                shape_by_hash[int(table[i]["name_hash"])] == (),
                i,
            )
            for i in range(table.size)
        ),
        key=lambda r: r[0],
    )

    fmt_parts: list[str] = ["<"]
    scalar_indices: list[int] = []
    shaped_slots: list[tuple[int, _ShapedSlot]] = []
    cursor = 0

    for byte_off, byte_cnt, dtype_code, elem_cnt, is_true_scalar, value_idx in rows:
        if byte_off > cursor:
            fmt_parts.append(f"{byte_off - cursor}x")
            cursor = byte_off

        dt_enum = DType(dtype_code)
        np_dt = DTYPE_TO_NUMPY[dt_enum]

        if is_true_scalar:
            # shape == () — wire is a Python scalar; consolidated Struct path.
            fmt_parts.append(_DTYPE_TO_STRUCT_CODE[dt_enum])
            scalar_indices.append(value_idx)
            cursor += np_dt.itemsize
        else:
            # shape != () — wire is a list (possibly length 1); shaped path.
            fmt_parts.append(f"{byte_cnt}x")
            shaped_slots.append((value_idx, _ShapedSlot(byte_off, np_dt, elem_cnt, byte_cnt)))
            cursor += byte_cnt

    scalar_struct = struct.Struct("".join(fmt_parts)) if scalar_indices else None
    return _PackPlan(
        scalar_struct=scalar_struct,
        scalar_indices=tuple(scalar_indices),
        shaped_slots=tuple(shaped_slots),
        expected_len=int(table.size),
        row_size=row_size(schema),
    )
