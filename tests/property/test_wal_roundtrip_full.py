"""End-to-end property tests: producer wire format → consumer apply path.

Exercises the full chain a real consumer walks for one message:

  1. Producer encodes values with pydantic in name_hash order.
  2. msgpack-pack the validated values list.
  3. Consumer decodes the msgpack blob.
  4. ``pack_row_from_list(schema, values_list, row_buffer)`` writes into
     a row-shaped bytearray.
  5. ``layout.insert(seg, entity_id, row_buffer)`` lands the row.
  6. ``serving.assemble(seg, entity_id)`` reads it back as a flat float32
     vector in declaration order.

The property: the assembled vector matches the same Python-side oracle
the existing Step 4/5 property tests use — the values went in,
they come out.

This is the key regression-guard for the name_hash-vs-declaration order
contract (review #B): the producer packs in name_hash order, the
consumer unpacks in name_hash order, and ``assemble`` returns in
declaration order. A bug at any layer breaks the round-trip.
"""

from __future__ import annotations

import sys
from typing import Any

import msgpack
import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

pytestmark = [
    pytest.mark.skipif(sys.platform == "win32", reason="POSIX shm only"),
    pytest.mark.property,
    # Producer's pydantic factory may emit a UserWarning if Hypothesis-generated
    # field names shadow BaseModel attributes (e.g. "schema"). Informational; no
    # behavior change. Filter scoped to this file.
    pytest.mark.filterwarnings(
        "ignore:Field name .* shadows an attribute in parent .BaseModel.:UserWarning"
    ),
]

from _helpers import (  # noqa: E402
    build_dynamic_schema,
    field_list_strategy,
    make_segment,
    random_value_for,
    release_segment,
)
from pyforge._internal.pydantic_factory import (  # noqa: E402
    clear_cache,
    field_order_for,
    pydantic_model_for,
)
from pyforge._internal.row_pack import (  # noqa: E402
    clear_cache as clear_row_pack_cache,
)
from pyforge.layout import insert as layout_insert  # noqa: E402
from pyforge.schema import DType, FeatureField, FeatureSchema, row_size  # noqa: E402
from pyforge.serving import assemble  # noqa: E402

_RNG = np.random.default_rng(20260429)


def _values_dict_native(fields: list[FeatureField], rng: np.random.Generator) -> dict[str, Any]:
    """Build a Python-native ``{name: value}`` dict for one record."""
    out: dict[str, Any] = {}
    for f in fields:
        arr = random_value_for(f, rng)
        if f.shape == ():
            scalar = arr[0]
            if f.dtype in (DType.FLOAT32, DType.FLOAT64):
                out[f.name] = float(scalar)
            else:
                out[f.name] = int(scalar)
            continue
        flat = arr.tolist()
        if len(f.shape) == 1:
            out[f.name] = flat
        else:
            r, c = f.shape
            out[f.name] = [flat[i * c : (i + 1) * c] for i in range(r)]
    return out


def _oracle_assemble(
    schema: type[FeatureSchema], values_native: dict[str, Any]
) -> np.ndarray[Any, np.dtype[np.float32]]:
    """Python oracle: the float32 vector ``serving.assemble`` should produce."""
    parts: list[np.ndarray[Any, np.dtype[np.float32]]] = []
    for f in schema.fields:
        v = values_native[f.name]
        if f.shape == ():
            parts.append(np.asarray([v], dtype=np.float32))
        else:
            parts.append(np.asarray(v, dtype=np.float32).reshape(-1))
    return np.concatenate(parts).astype(np.float32, copy=False)


@pytest.fixture(autouse=True)
def _fresh_caches() -> None:
    """Property tests build many dynamic schemas; clear caches between runs."""
    clear_cache()
    clear_row_pack_cache()


# ---------------------------------------------------------------------------
# Property 1: full producer→consumer round-trip yields the oracle.
# ---------------------------------------------------------------------------


@given(field_list_strategy())
@settings(
    max_examples=100,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture],
)
def test_producer_to_consumer_roundtrip(fields: list[FeatureField]) -> None:
    schema = build_dynamic_schema(fields)
    values_native = _values_dict_native(fields, _RNG)

    # --- Producer side ---------------------------------------------------
    model_cls = pydantic_model_for(schema)
    validated = model_cls.model_validate(values_native)
    order = field_order_for(schema)
    values_list = [getattr(validated, name) for name in order]
    blob = msgpack.packb(values_list, use_bin_type=True)

    # --- Consumer side ---------------------------------------------------
    from pyforge._internal.row_pack import pack_row_from_list

    decoded = msgpack.unpackb(blob, use_list=True, raw=False)
    seg = make_segment(schema, capacity=4)
    try:
        row_buffer = bytearray(row_size(schema))
        pack_row_from_list(schema, decoded, row_buffer)
        layout_insert(seg, "ent-property", memoryview(row_buffer))

        # --- Read back via assemble + compare to oracle -----------------
        got = assemble(seg, "ent-property")
        oracle = _oracle_assemble(schema, values_native)
        # NaN-aware comparison: NaN != NaN, so use isnan as bit-level mask.
        nan_mask = np.isnan(got) & np.isnan(oracle)
        eq_mask = (got == oracle) | nan_mask
        assert eq_mask.all(), (
            f"mismatch at indices {np.where(~eq_mask)[0].tolist()}: "
            f"got={got[~eq_mask].tolist()} oracle={oracle[~eq_mask].tolist()}"
        )
    finally:
        release_segment(seg)


# ---------------------------------------------------------------------------
# Property 2: ordering invariant — last write per entity_id wins in shm.
# ---------------------------------------------------------------------------


@given(
    fields=field_list_strategy(),
    n_writes=st.integers(min_value=2, max_value=10),
)
@settings(
    max_examples=80,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture],
)
def test_last_write_wins_per_entity(fields: list[FeatureField], n_writes: int) -> None:
    """Multiple writes to the same entity_id in stream order: the LAST one
    is what assemble returns. This is the consumer's per-entity ordering
    contract. Documented in CLAUDE.md and locked by the consumer applying
    msgs in stream order without reordering."""
    schema = build_dynamic_schema(fields)
    model_cls = pydantic_model_for(schema)
    order = field_order_for(schema)

    from pyforge._internal.row_pack import pack_row_from_list

    seg = make_segment(schema, capacity=4)
    try:
        last_oracle: np.ndarray[Any, np.dtype[np.float32]] | None = None
        for _ in range(n_writes):
            values_native = _values_dict_native(fields, _RNG)
            validated = model_cls.model_validate(values_native)
            values_list = [getattr(validated, name) for name in order]
            blob = msgpack.packb(values_list, use_bin_type=True)
            decoded = msgpack.unpackb(blob, use_list=True, raw=False)
            row_buffer = bytearray(row_size(schema))
            pack_row_from_list(schema, decoded, row_buffer)
            layout_insert(seg, "ent-rep", memoryview(row_buffer))
            last_oracle = _oracle_assemble(schema, values_native)

        assert last_oracle is not None
        got = assemble(seg, "ent-rep")
        nan_mask = np.isnan(got) & np.isnan(last_oracle)
        eq_mask = (got == last_oracle) | nan_mask
        assert eq_mask.all()
    finally:
        release_segment(seg)
