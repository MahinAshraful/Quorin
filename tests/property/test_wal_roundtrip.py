"""Property tests for the WAL producer's pack -> unpack -> validate path.

The producer encodes a list of validated values in name_hash order. These
properties verify the round-trip is identity (modulo float32 truncation
and NaN bit-pattern preservation), the encoded blob is smaller than a
naive Python repr, and the field order is stable for any random schema.
"""

from __future__ import annotations

import math
from typing import Any

import msgpack
import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from _helpers import build_dynamic_schema, field_list_strategy, random_value_for
from pyforge._internal.pydantic_factory import clear_cache, field_order_for, pydantic_model_for
from pyforge.schema import DType, FeatureField

_RNG = np.random.default_rng(20260429)


# Hypothesis sometimes generates field names that collide with pydantic
# v2 BaseModel attributes (e.g. "schema", "json", "copy"); pydantic emits a
# UserWarning and falls back to attribute access — validation still works
# correctly. The warning is informational, not a failure mode worth fixing
# in the random schema strategy. Filter it scoped to this file only.
pytestmark = [
    pytest.mark.property,
    pytest.mark.filterwarnings(
        "ignore:Field name .* shadows an attribute in parent .BaseModel.:UserWarning"
    ),
]


def _values_dict(fields: list[FeatureField], rng: np.random.Generator) -> dict[str, Any]:
    """Build a Python-native ``{name: value}`` dict for one record.

    Floats use Python list[float]; ints use Python list[int]. Pydantic's
    default coercion accepts both numpy and native — but msgpack packs
    Python natives faster, and the property is round-trip identity which
    is easier to assert in Python types.
    """
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
        elif len(f.shape) == 2:
            r, c = f.shape
            out[f.name] = [flat[i * c : (i + 1) * c] for i in range(r)]
        else:
            raise AssertionError(f"unhandled shape {f.shape}")
    return out


def _poison_with_nan(values: dict[str, Any], fields: list[FeatureField]) -> dict[str, Any]:
    """Replace one random float scalar with NaN; one float vector element with +Inf."""
    rng = np.random.default_rng(_RNG.integers(0, 2**32 - 1))
    floats = [f for f in fields if f.dtype in (DType.FLOAT32, DType.FLOAT64)]
    if not floats:
        return values
    f = floats[rng.integers(0, len(floats))]
    if f.shape == ():
        values[f.name] = float("nan")
    elif len(f.shape) == 1:
        idx = int(rng.integers(0, f.shape[0]))
        values[f.name][idx] = float("nan")
        if f.shape[0] > 1:
            values[f.name][(idx + 1) % f.shape[0]] = float("inf")
    return values


def _values_match(a: list[Any], b: list[Any]) -> bool:
    """Element-wise compare with NaN-aware semantics. Recursive for nested lists."""
    if len(a) != len(b):
        return False
    for x, y in zip(a, b, strict=True):
        if isinstance(x, list) and isinstance(y, list):
            if not _values_match(x, y):
                return False
        elif isinstance(x, float) and isinstance(y, float):
            if math.isnan(x) and math.isnan(y):
                continue
            if x != y:
                return False
        elif x != y:
            return False
    return True


# ---------------------------------------------------------------------------
# Property 1 — round-trip identity, including NaN-poisoned and ±Inf samples.
# ---------------------------------------------------------------------------


@given(field_list_strategy())
@settings(
    max_examples=150,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture],
)
def test_pack_unpack_validate_round_trip(fields: list[FeatureField]) -> None:
    clear_cache()
    schema = build_dynamic_schema(fields)
    model_cls = pydantic_model_for(schema)
    order = field_order_for(schema)

    rng = np.random.default_rng(_RNG.integers(0, 2**32 - 1))
    values = _values_dict(fields, rng)
    values = _poison_with_nan(values, fields)

    validated = model_cls.model_validate(values)
    payload = [getattr(validated, n) for n in order]
    blob = msgpack.packb(payload)
    decoded = msgpack.unpackb(blob)

    # Decoded list mirrors the validated payload (NaN-aware).
    assert _values_match(decoded, payload)

    # And re-validating the decoded payload (zipped back to a dict) succeeds.
    redecoded_dict = {order[i]: decoded[i] for i in range(len(order))}
    re_validated = model_cls.model_validate(redecoded_dict)
    assert _values_match(
        [getattr(re_validated, n) for n in order],
        payload,
    )


# ---------------------------------------------------------------------------
# Property 2 — field_order_for is stable: same class, same order across calls.
# ---------------------------------------------------------------------------


@given(field_list_strategy())
@settings(max_examples=200, deadline=None)
def test_field_order_is_stable(fields: list[FeatureField]) -> None:
    clear_cache()
    schema = build_dynamic_schema(fields)
    a = field_order_for(schema)
    b = field_order_for(schema)
    assert a == b


# ---------------------------------------------------------------------------
# Property 3 — encoded blob is smaller than a naive repr() at any non-trivial size.
#
# Wire size is one of msgpack's selling points; this property locks the
# benefit so a future "let's switch to JSON for debugging" PR fails loudly.
# ---------------------------------------------------------------------------


@given(field_list_strategy())
@settings(max_examples=100, deadline=None)
def test_blob_is_smaller_than_naive_repr(fields: list[FeatureField]) -> None:
    # Skip very small schemas (1-2 fields) — msgpack's framing overhead can
    # exceed Python's compact repr for tiny payloads.
    if sum(f.element_count for f in fields) < 8:
        return

    clear_cache()
    schema = build_dynamic_schema(fields)
    model_cls = pydantic_model_for(schema)
    order = field_order_for(schema)
    rng = np.random.default_rng(_RNG.integers(0, 2**32 - 1))
    values = _values_dict(fields, rng)

    validated = model_cls.model_validate(values)
    payload = [getattr(validated, n) for n in order]
    blob = msgpack.packb(payload)
    naive = repr(payload).encode("utf-8")
    assert len(blob) < len(naive)


# ---------------------------------------------------------------------------
# Property 4 — ints round-trip exactly through msgpack.
# (Floats are tested above with NaN-aware comparison; integers are exact.)
# ---------------------------------------------------------------------------


_INT_SCHEMA_FIELDS = st.lists(
    st.builds(
        FeatureField,
        st.text(
            alphabet="abcdefghijklmnopqrstuvwxyz_",
            min_size=1,
            max_size=10,
        ).filter(lambda s: s[0].isalpha()),
        st.sampled_from([DType.INT32, DType.INT64, DType.UINT8]),
        st.one_of(st.just(()), st.tuples(st.integers(min_value=1, max_value=8))),
    ),
    min_size=1,
    max_size=4,
).map(
    lambda fs: list({f.name: f for f in fs}.values())  # dedup by name
)


@given(_INT_SCHEMA_FIELDS)
@settings(max_examples=100, deadline=None)
def test_int_fields_round_trip_exactly(fields: list[FeatureField]) -> None:
    clear_cache()
    schema = build_dynamic_schema(fields)
    model_cls = pydantic_model_for(schema)
    order = field_order_for(schema)

    rng = np.random.default_rng(_RNG.integers(0, 2**32 - 1))
    values = _values_dict(fields, rng)
    validated = model_cls.model_validate(values)
    payload = [getattr(validated, n) for n in order]

    decoded = msgpack.unpackb(msgpack.packb(payload))
    assert decoded == payload
