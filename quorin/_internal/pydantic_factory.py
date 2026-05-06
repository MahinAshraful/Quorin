"""Pydantic model factory for FeatureSchema validation.

Step 9's :class:`quorin.wal.WALProducer` calls :func:`pydantic_model_for`
on every ``write`` to validate user-supplied values. This module memoizes
the generated model class per :class:`FeatureSchema` subclass — first call
costs ~1 ms (pydantic ``create_model``), every subsequent call is a
~50 ns dict lookup.

Locked design (ADR-008 §4):

- **Internal until Step 12.** Users do not construct pydantic models
  themselves; the producer is the only caller. Keeping this private leaves
  the door open for a hand-rolled validator later if the 100-300 µs
  pydantic cost at 200 fields ever bites the 10k writes/sec budget.
- **Memoized by class identity, not by class name.** Two distinct classes
  with the same ``__name__`` get distinct models. Step 15's atomic class
  swap will provide its own invalidation by giving callers a new class.
- **Field order on the wire is name_hash order**, matching the
  schema-relative offset table from :func:`quorin.schema.compile_schema`.
  :func:`field_order_for` returns that order so the producer can extract
  values via ``getattr`` in the same order the consumer will unpack them.
- **NaN/Inf accepted on every float field** (``allow_inf_nan=True``)
  uniformly. Locked by invariant #12 (NaN bit-pattern preservation through
  ``_assemble_core``) and standard ML missing-feature semantics. No
  per-field toggling. **Do NOT enable global strict mode** — strict would
  reject ``np.float32(nan)`` numpy-scalar input that the assembly path is
  designed to round-trip through. Default coercion + ``allow_inf_nan=True``
  is the right combination.
"""

from __future__ import annotations

from typing import Annotated, Any, cast

import pydantic
from pydantic import BaseModel, ConfigDict, Field

from quorin.schema import (
    DType,
    FeatureField,
    FeatureSchema,
    _hash_name,
    compile_schema,
)

# ---------------------------------------------------------------------------
# Module-level caches. Process-global; one model per FeatureSchema subclass.
# ---------------------------------------------------------------------------

_MODEL_CACHE: dict[type[FeatureSchema], type[BaseModel]] = {}
"""Maps a schema class to its memoized validator model. ~50 ns lookup."""

_FIELD_ORDER_CACHE: dict[type[FeatureSchema], tuple[str, ...]] = {}
"""Maps a schema class to its name_hash-sorted field-name tuple. Producer
extracts values from the validated instance in this order so the consumer
can zip back against ``compile_schema(schema)`` deterministically."""


# Per-dtype Python types and integer-range constraints.
# Floats accept NaN/Inf uniformly (``allow_inf_nan=True``).
# Ints carry ge/le constraints so out-of-range values fail at the producer
# instead of silently overflowing the consumer's numpy view.
_INT64_MIN = -(2**63)
_INT64_MAX = 2**63 - 1
_INT32_MIN = -(2**31)
_INT32_MAX = 2**31 - 1
_UINT8_MIN = 0
_UINT8_MAX = 255


# ---------------------------------------------------------------------------
# Public API.
# ---------------------------------------------------------------------------


def pydantic_model_for(schema: type[FeatureSchema]) -> type[BaseModel]:
    """Return a memoized pydantic model class that validates payloads for
    ``schema``.

    First call per schema costs ~1 ms (pydantic ``create_model``), every
    subsequent call is a ~50 ns dict lookup.

    The returned model:
      * Has one field per :class:`FeatureField` in ``schema.fields``.
      * Rejects unknown keys (``extra="forbid"``).
      * Accepts NaN/Inf on float fields.
      * Enforces ``ge``/``le`` constraints on integer fields.
      * Enforces fixed lengths on shaped fields via ``min_length``/``max_length``.
    """
    if (m := _MODEL_CACHE.get(schema)) is not None:
        return m
    model, order = _build(schema)
    _MODEL_CACHE[schema] = model
    _FIELD_ORDER_CACHE[schema] = order
    return model


def field_order_for(schema: type[FeatureSchema]) -> tuple[str, ...]:
    """Return the schema's field names in name_hash sort order.

    Calling :func:`pydantic_model_for` first is not required — this function
    populates both caches if needed. The producer iterates this tuple to
    extract validated values via ``getattr``, avoiding ``model_dump()``'s
    intermediate-dict allocation.
    """
    if (o := _FIELD_ORDER_CACHE.get(schema)) is not None:
        return o
    pydantic_model_for(schema)
    return _FIELD_ORDER_CACHE[schema]


def clear_cache() -> None:
    """Drop all memoized models. Test-only; do not call from production code."""
    _MODEL_CACHE.clear()
    _FIELD_ORDER_CACHE.clear()


# ---------------------------------------------------------------------------
# Internal builder.
# ---------------------------------------------------------------------------


def _build(schema: type[FeatureSchema]) -> tuple[type[BaseModel], tuple[str, ...]]:
    """Construct the memoized model + the name_hash-ordered field tuple."""
    table = compile_schema(schema)  # already sorted by name_hash

    # Map name_hash -> FeatureField so we can walk the table in order.
    name_to_field = {f.name: f for f in schema.fields}
    hash_to_name: dict[int, str] = {_hash_name(name): name for name in name_to_field}

    order: list[str] = []
    field_definitions: dict[str, tuple[Any, Any]] = {}

    for row in table:
        name = hash_to_name[int(row["name_hash"])]
        f = name_to_field[name]
        annotation, field_info = _field_for(f)
        field_definitions[name] = (annotation, field_info)
        order.append(name)

    # mypy can't reconcile the **dict spread with create_model's overloads;
    # the call is correct at runtime (pydantic accepts (annotation, FieldInfo)
    # tuples per field). Cast keeps the public return type honest.
    model = cast(
        "type[BaseModel]",
        pydantic.create_model(  # type: ignore[call-overload]
            f"{schema.__name__}Validated",
            __config__=ConfigDict(extra="forbid", frozen=True),
            **field_definitions,
        ),
    )
    return model, tuple(order)


def _field_for(f: FeatureField) -> tuple[Any, Any]:
    """Pydantic ``(annotation, FieldInfo)`` for one :class:`FeatureField`.

    Scalars: native ``float``/``int`` with dtype-appropriate constraints.
    Shape ``(N,)``: ``list[float | int]`` with length pin.
    Shape ``(R, C)``: ``list[list[float | int]]`` — outer length pinned by
    ``Field(min_length=R, max_length=R)``; inner length is enforced by an
    annotated inner-list type. Pydantic v2 propagates ``Annotated`` length
    constraints into list-of-list correctly.
    """
    is_float = f.dtype in (DType.FLOAT32, DType.FLOAT64)

    if is_float:
        scalar_type: Any = float
        scalar_kwargs: dict[str, Any] = {"allow_inf_nan": True}
    else:
        scalar_type = int
        scalar_kwargs = _int_constraints(f.dtype)

    if f.shape == ():
        return scalar_type, Field(**scalar_kwargs)

    # Inner constraints (allow_inf_nan / ge / le) live on each element.
    elem_annotation: Any = Annotated[scalar_type, Field(**scalar_kwargs)]

    if len(f.shape) == 1:
        n = f.shape[0]
        return list[elem_annotation], Field(min_length=n, max_length=n)

    if len(f.shape) == 2:
        rows, cols = f.shape
        inner_list_annotation: Any = Annotated[
            list[elem_annotation],
            Field(min_length=cols, max_length=cols),
        ]
        return list[inner_list_annotation], Field(min_length=rows, max_length=rows)

    # FeatureField currently only validates positive 1D / 2D / 3D shapes.
    # We only generate validators for 0D, 1D, 2D — 3D+ is a Step 15 problem.
    raise NotImplementedError(
        f"pydantic_factory: {len(f.shape)}D shapes not yet supported (field {f.name!r})"
    )


def _int_constraints(d: DType) -> dict[str, Any]:
    if d is DType.INT64:
        return {"ge": _INT64_MIN, "le": _INT64_MAX}
    if d is DType.INT32:
        return {"ge": _INT32_MIN, "le": _INT32_MAX}
    if d is DType.UINT8:
        return {"ge": _UINT8_MIN, "le": _UINT8_MAX}
    raise AssertionError(f"_int_constraints called on non-integer dtype {d}")
