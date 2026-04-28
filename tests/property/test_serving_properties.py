"""Hypothesis-driven property tests for pyforge.serving.assemble.

Generates random schemas + random row payloads. Properties cover the
declaration-order contract, output length, exactness for float32-only
schemas, and call idempotence.
"""

from __future__ import annotations

import string
import sys

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

pytestmark = pytest.mark.skipif(
    sys.platform == "win32",
    reason="serving requires POSIX (Linux/WSL2)",
)

import numpy as np  # noqa: E402

from _helpers import make_segment, pack_row, release_segment  # noqa: E402
from pyforge.layout import insert  # noqa: E402
from pyforge.schema import (  # noqa: E402
    DType,
    FeatureField,
    FeatureSchema,
    DTYPE_TO_NUMPY,
)
from pyforge.serving import assemble  # noqa: E402


_HYPO = settings(
    max_examples=150,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture],
)


# ---------------------------------------------------------------------------
# Strategies.
# ---------------------------------------------------------------------------


# Ascii-only field names so they're trivially valid Python identifiers (we
# don't actually need them to be — schema accepts any non-empty string — but
# Hypothesis shrinks them better when they're simple).
_NAME = st.text(
    alphabet=string.ascii_lowercase + string.digits + "_",
    min_size=1,
    max_size=12,
).filter(lambda s: s[0].isalpha())


_DTYPE_OPTS = st.sampled_from(list(DType))


# Shape: () (scalar) or (1..16,) or (1..4, 1..4). Keeps row sizes manageable.
_SHAPE = st.one_of(
    st.just(()),
    st.tuples(st.integers(min_value=1, max_value=16)),
    st.tuples(st.integers(min_value=1, max_value=4), st.integers(min_value=1, max_value=4)),
)


@st.composite
def _field(draw: st.DrawFn) -> FeatureField:
    return FeatureField(draw(_NAME), draw(_DTYPE_OPTS), draw(_SHAPE))


@st.composite
def _field_list(draw: st.DrawFn) -> list[FeatureField]:
    fields = draw(st.lists(_field(), min_size=1, max_size=6))
    # Enforce unique names — schema rejects duplicates at class-definition.
    seen: set[str] = set()
    deduped: list[FeatureField] = []
    for f in fields:
        if f.name not in seen:
            seen.add(f.name)
            deduped.append(f)
    return deduped


_SCHEMA_VERSION_COUNTER = [0]


def _build_schema(field_list: list[FeatureField]) -> type[FeatureSchema]:
    """Build a FeatureSchema subclass at runtime from a generated field list.

    Each call gets a unique class name so __init_subclass__ is happy and the
    same field list can be re-used across examples without conflict.
    """
    _SCHEMA_VERSION_COUNTER[0] += 1
    cls_name = f"_HypoSchema_{_SCHEMA_VERSION_COUNTER[0]}"
    return type(
        cls_name,
        (FeatureSchema,),
        {"version": 1, "fields": field_list},
    )


def _random_value_for(field: FeatureField, rng: np.random.Generator) -> np.ndarray:
    """A random ndarray matching ``field``'s dtype + shape (post-flatten size)."""
    np_dtype = DTYPE_TO_NUMPY[field.dtype]
    n = field.element_count
    if field.dtype is DType.FLOAT32:
        return rng.standard_normal(n).astype(np.float32)
    if field.dtype is DType.FLOAT64:
        return rng.standard_normal(n).astype(np.float64)
    if field.dtype is DType.INT32:
        return rng.integers(-(1 << 20), 1 << 20, size=n, dtype=np.int32)
    if field.dtype is DType.INT64:
        return rng.integers(-(1 << 40), 1 << 40, size=n, dtype=np.int64)
    if field.dtype is DType.UINT8:
        return rng.integers(0, 256, size=n, dtype=np.uint8)
    raise AssertionError(f"unhandled dtype {field.dtype} → {np_dtype}")


# ---------------------------------------------------------------------------
# Properties.
# ---------------------------------------------------------------------------


@_HYPO
@given(field_list=_field_list(), seed=st.integers(min_value=0, max_value=2**31 - 1))
def test_output_length_equals_sum_element_counts(
    field_list: list[FeatureField], seed: int
) -> None:
    schema = _build_schema(field_list)
    rng = np.random.default_rng(seed)
    seg = make_segment(schema, capacity=2)
    try:
        values = {f.name: _random_value_for(f, rng) for f in schema.fields}
        insert(seg, "u", pack_row(schema, values))

        out = assemble(seg, "u")
        assert out.shape == (sum(f.element_count for f in schema.fields),)
        assert out.dtype == np.float32
    finally:
        release_segment(seg)


@_HYPO
@given(field_list=_field_list(), seed=st.integers(min_value=0, max_value=2**31 - 1))
def test_output_segments_match_declaration_order(
    field_list: list[FeatureField], seed: int
) -> None:
    """For each field i in declaration order, ``out[cursor_i : cursor_i + n_i]``
    equals the field's value cast to float32. Verifies the central contract:
    declaration order, not hash order."""
    schema = _build_schema(field_list)
    rng = np.random.default_rng(seed)
    seg = make_segment(schema, capacity=2)
    try:
        values = {f.name: _random_value_for(f, rng) for f in schema.fields}
        insert(seg, "u", pack_row(schema, values))
        out = assemble(seg, "u")

        cursor = 0
        for f in schema.fields:
            n = f.element_count
            expected = values[f.name].astype(np.float32, copy=False)
            np.testing.assert_array_equal(out[cursor : cursor + n], expected)
            cursor += n
        assert cursor == out.shape[0]
    finally:
        release_segment(seg)


@_HYPO
@given(
    field_list=st.lists(
        st.builds(
            FeatureField,
            _NAME,
            st.just(DType.FLOAT32),
            _SHAPE,
        ),
        min_size=1,
        max_size=6,
    ).map(
        lambda fs: list({f.name: f for f in fs}.values())  # dedupe by name
    ),
    seed=st.integers(min_value=0, max_value=2**31 - 1),
)
def test_float32_only_round_trip_exact(field_list: list[FeatureField], seed: int) -> None:
    """When every field is already float32, assemble's output is bit-exact —
    no cast loss anywhere."""
    schema = _build_schema(field_list)
    rng = np.random.default_rng(seed)
    seg = make_segment(schema, capacity=2)
    try:
        values = {f.name: rng.standard_normal(f.element_count).astype(np.float32) for f in schema.fields}
        insert(seg, "u", pack_row(schema, values))
        out = assemble(seg, "u")

        cursor = 0
        for f in schema.fields:
            n = f.element_count
            np.testing.assert_array_equal(out[cursor : cursor + n], values[f.name])
            cursor += n
    finally:
        release_segment(seg)


@_HYPO
@given(field_list=_field_list(), seed=st.integers(min_value=0, max_value=2**31 - 1))
def test_two_calls_yield_equal_results(field_list: list[FeatureField], seed: int) -> None:
    """assemble is a pure function of segment state. Two calls return equal
    arrays in independent buffers (NaN-safe comparison via assert_equal)."""
    schema = _build_schema(field_list)
    rng = np.random.default_rng(seed)
    seg = make_segment(schema, capacity=2)
    try:
        values = {f.name: _random_value_for(f, rng) for f in schema.fields}
        insert(seg, "u", pack_row(schema, values))

        a = assemble(seg, "u")
        b = assemble(seg, "u")
        assert a is not b
        np.testing.assert_equal(a, b)  # NaN-safe element-wise equality
    finally:
        release_segment(seg)
