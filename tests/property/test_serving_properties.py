"""Hypothesis-driven property tests for pyforge.serving.assemble.

Generates random schemas + random row payloads. Properties cover the
declaration-order contract, output length, exactness for float32-only
schemas, and call idempotence.

Strategies (`field_list_strategy`, `random_value_for`, `build_dynamic_schema`)
live in :mod:`tests._helpers` so Step 5's parity test can reuse them.
"""

from __future__ import annotations

import sys

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

pytestmark = pytest.mark.skipif(
    sys.platform == "win32",
    reason="serving requires POSIX (Linux/WSL2)",
)

from _helpers import (  # noqa: E402
    build_dynamic_schema,
    field_list_strategy,
    make_segment,
    pack_row,
    random_value_for,
    release_segment,
)
from pyforge.layout import insert  # noqa: E402
from pyforge.schema import DType, FeatureField  # noqa: E402
from pyforge.serving import assemble  # noqa: E402

_HYPO = settings(
    max_examples=150,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture],
)


# Local strategies for the float32-only test. The shared strategies in
# _helpers sample from all DTypes; filtering for all-float32 would be
# wasteful, so the float32-only test inlines its own.
_NAME_LOCAL = st.text(
    alphabet="abcdefghijklmnopqrstuvwxyz0123456789_",
    min_size=1,
    max_size=12,
).filter(lambda s: s[0].isalpha())

_SHAPE_LOCAL = st.one_of(
    st.just(()),
    st.tuples(st.integers(min_value=1, max_value=16)),
    st.tuples(st.integers(min_value=1, max_value=4), st.integers(min_value=1, max_value=4)),
)


# ---------------------------------------------------------------------------
# Properties.
# ---------------------------------------------------------------------------


@_HYPO
@given(field_list=field_list_strategy(), seed=st.integers(min_value=0, max_value=2**31 - 1))
def test_output_length_equals_sum_element_counts(field_list: list[FeatureField], seed: int) -> None:
    schema = build_dynamic_schema(field_list)
    rng = np.random.default_rng(seed)
    seg = make_segment(schema, capacity=2)
    try:
        values = {f.name: random_value_for(f, rng) for f in schema.fields}
        insert(seg, "u", pack_row(schema, values))

        out = assemble(seg, "u")
        assert out.shape == (sum(f.element_count for f in schema.fields),)
        assert out.dtype == np.float32
    finally:
        release_segment(seg)


@_HYPO
@given(field_list=field_list_strategy(), seed=st.integers(min_value=0, max_value=2**31 - 1))
def test_output_segments_match_declaration_order(field_list: list[FeatureField], seed: int) -> None:
    """For each field i in declaration order, ``out[cursor_i : cursor_i + n_i]``
    equals the field's value cast to float32. Verifies the central contract:
    declaration order, not hash order."""
    schema = build_dynamic_schema(field_list)
    rng = np.random.default_rng(seed)
    seg = make_segment(schema, capacity=2)
    try:
        values = {f.name: random_value_for(f, rng) for f in schema.fields}
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
        st.builds(FeatureField, _NAME_LOCAL, st.just(DType.FLOAT32), _SHAPE_LOCAL),
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
    schema = build_dynamic_schema(field_list)
    rng = np.random.default_rng(seed)
    seg = make_segment(schema, capacity=2)
    try:
        values = {
            f.name: rng.standard_normal(f.element_count).astype(np.float32) for f in schema.fields
        }
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
@given(field_list=field_list_strategy(), seed=st.integers(min_value=0, max_value=2**31 - 1))
def test_two_calls_yield_equal_results(field_list: list[FeatureField], seed: int) -> None:
    """assemble is a pure function of segment state. Two calls return equal
    arrays in independent buffers (NaN-safe comparison via assert_equal)."""
    schema = build_dynamic_schema(field_list)
    rng = np.random.default_rng(seed)
    seg = make_segment(schema, capacity=2)
    try:
        values = {f.name: random_value_for(f, rng) for f in schema.fields}
        insert(seg, "u", pack_row(schema, values))

        a = assemble(seg, "u")
        b = assemble(seg, "u")
        assert a is not b
        np.testing.assert_equal(a, b)  # NaN-safe element-wise equality
    finally:
        release_segment(seg)
