"""Schema-to-Arrow plan compiler for the offline Parquet writer (Step 11).

Step 11's :class:`pyforge.offline.ParquetDatasetStore` lands each
``OfflineWriter.append`` into a per-``(schema, date)`` columnar buffer
and, on ``flush()``, materializes a :class:`pyarrow.Table` from those
buffers. Building the Arrow schema and the wire-to-declaration permutation
on every call would burn ~1 ms per flush at 200 fields; this module
memoizes both.

Locked design (ADR-010 §6):

1. **Memoization is by class identity** (and by ``include_msg_id``) so
   Step 15's atomic class swap gets a fresh plan automatically and the
   ``include_msg_id=False`` opt-out doesn't poison the default cache.
2. **Column order is declaration order.** Operator-readable; Parquet
   projection is by name so column order has no read-side perf impact.
   The plan carries a ``wire_to_decl`` permutation so per-row append
   does one tuple lookup per field — no name-string hashing in the hot
   path.
3. **Fixed-size lists for shaped fields.** ``shape=(N,)`` →
   ``pa.list_(elem, N)``; ``shape=(R, C)`` → nested fixed-size list.
   Lengths get validated at ``pa.Table.from_pydict`` time. Three- and
   higher-dim shapes raise ``ValueError`` at plan-build time.
4. **Embedded ``msg_id_ms``/``msg_id_seq`` columns at end** when
   ``include_msg_id=True``, never in the middle of the user's fields.
"""

from __future__ import annotations

from dataclasses import dataclass

import pyarrow as pa

from pyforge.schema import DType, FeatureField, FeatureSchema, _hash_name

# ---------------------------------------------------------------------------
# DType -> Arrow leaf type. Module-level so the dict literal is built once.
# ---------------------------------------------------------------------------

_DTYPE_TO_ARROW: dict[DType, pa.DataType] = {
    DType.FLOAT32: pa.float32(),
    DType.FLOAT64: pa.float64(),
    DType.INT32: pa.int32(),
    DType.INT64: pa.int64(),
    DType.UINT8: pa.uint8(),
}

# Column names the writer reserves at the row-schema level. Any user
# schema with a field whose name clashes with one of these would shadow
# the writer's own column (entity_id, event_time_ns, msg_id_*) and the
# hot append path's pre-resolved aliases would silently land the user
# value in a column the producer's wire payload doesn't fill. Hypothesis
# discovered this with a schema field literally named "entity_id".
#
# ``event_date`` is reserved at the *partition* level — it's the hive
# directory key derived from event_time_ns. A user field named
# event_date would write per-row data to the file (typed as the user's
# dtype), then PyArrow's dataset reader would try to merge it with the
# partition column (always pa.string()) and fail with a deep-stack
# ArrowTypeError at read time. Hypothesis P4 caught this in Step 12.
_RESERVED_FIELD_NAMES: frozenset[str] = frozenset(
    {
        "entity_id",
        "event_time_ns",
        "msg_id_ms",
        "msg_id_seq",
        "event_date",
    }
)


def _arrow_type_for(field: FeatureField) -> pa.DataType:
    """Map one ``FeatureField`` to its Arrow data type.

    Scalar -> leaf; 1D shape ``(N,)`` -> fixed-size list of length N;
    2D shape ``(R, C)`` -> nested fixed-size list. Higher dims raise.
    """
    elem = _DTYPE_TO_ARROW[field.dtype]
    rank = len(field.shape)
    if rank == 0:
        return elem
    if rank == 1:
        return pa.list_(elem, field.shape[0])
    if rank == 2:
        return pa.list_(pa.list_(elem, field.shape[1]), field.shape[0])
    raise ValueError(
        f"unsupported shape {field.shape!r} on field {field.name!r}: "
        "Arrow plan supports scalar, 1D, and 2D fields only"
    )


# ---------------------------------------------------------------------------
# Plan dataclass + cache.
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _ArrowPlan:
    """Memoized Arrow plan for one ``(schema, include_msg_id)`` pair.

    Attributes:
        arrow_schema: Declaration-ordered ``pa.Schema`` consumed by
            :meth:`pa.Table.from_pydict` at flush time. Includes the
            two leading ``entity_id`` / ``event_time_ns`` columns and
            (optionally) the trailing ``msg_id_ms`` / ``msg_id_seq``
            columns.
        column_names: Tuple of column names parallel to
            ``arrow_schema.names``. Used to populate the per-bucket
            empty-list dict.
        wire_to_decl: Permutation that maps wire-order field indices
            (name_hash-sorted, matching the producer per ADR-008) to
            their declaration-order index *within* ``column_names``.
            Length is ``len(schema.fields)``; values include the
            ``decl_offset=2`` shift for ``entity_id`` /
            ``event_time_ns``.
        include_msg_id: Whether the plan carries the trailing two
            ``msg_id_*`` columns. Cached separately from ``True`` so
            both layouts can coexist in the cache.
    """

    arrow_schema: pa.Schema
    column_names: tuple[str, ...]
    wire_to_decl: tuple[int, ...]
    include_msg_id: bool


_PLAN_CACHE: dict[tuple[type[FeatureSchema], bool], _ArrowPlan] = {}


def _arrow_plan_for(schema: type[FeatureSchema], *, include_msg_id: bool) -> _ArrowPlan:
    """Return the memoized Arrow plan for a schema class.

    First call for a ``(schema, include_msg_id)`` pair builds the plan
    in ~1 ms; subsequent calls are a single dict lookup (~50 ns).
    """
    key = (schema, include_msg_id)
    cached = _PLAN_CACHE.get(key)
    if cached is not None:
        return cached

    decl_fields = list(schema.fields)
    # Reject user fields whose names collide with the writer's reserved
    # columns. Without this, the bucket's pre-resolved aliases for the
    # reserved columns would silently shadow the user field, landing
    # producer values in the wrong column at flush time.
    collisions = sorted(f.name for f in decl_fields if f.name in _RESERVED_FIELD_NAMES)
    if collisions:
        raise ValueError(
            f"schema {schema.__name__!r} has fields whose names collide with "
            f"writer-reserved columns: {collisions}. Reserved names: "
            f"{sorted(_RESERVED_FIELD_NAMES)}"
        )
    arrow_fields: list[pa.Field] = [
        pa.field("entity_id", pa.string(), nullable=False),
        pa.field("event_time_ns", pa.int64(), nullable=False),
    ]
    for f in decl_fields:
        arrow_fields.append(pa.field(f.name, _arrow_type_for(f), nullable=False))
    if include_msg_id:
        arrow_fields.append(pa.field("msg_id_ms", pa.int64(), nullable=False))
        arrow_fields.append(pa.field("msg_id_seq", pa.int32(), nullable=False))

    arrow_schema = pa.schema(arrow_fields)
    column_names = tuple(arrow_schema.names)

    # Build wire_to_decl: enumerate decl_fields in name_hash order (the
    # producer's wire ordering, ADR-008) and record each one's
    # declaration index within column_names. The +2 offset covers the
    # leading entity_id / event_time_ns columns.
    decl_offset = 2
    name_to_decl = {f.name: decl_offset + i for i, f in enumerate(decl_fields)}
    wire_order = sorted(decl_fields, key=lambda f: _hash_name(f.name))
    wire_to_decl = tuple(name_to_decl[f.name] for f in wire_order)

    plan = _ArrowPlan(
        arrow_schema=arrow_schema,
        column_names=column_names,
        wire_to_decl=wire_to_decl,
        include_msg_id=include_msg_id,
    )
    _PLAN_CACHE[key] = plan
    return plan


def clear_cache() -> None:
    """Drop all cached plans. Test-only; do not call from production code."""
    _PLAN_CACHE.clear()
