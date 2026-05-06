"""Parity tests for the ``out=`` parameter on assemble().

Three properties, each x 200 Hypothesis examples (= 600 random schema/row
pairs total):

1. ``serving.assemble(seg, eid, out=pool_buf) == serving.assemble(seg, eid)``
2. ``assembly.assemble(seg, eid, out=pool_buf) == assembly.assemble(seg, eid)``
3. ``serving.assemble(out=pool_buf) == assembly.assemble(out=pool_buf)``

Critical detail: pre-fill the pooled buffer with ``np.nan`` (not zeros)
before the call. Both assemble paths walk every output index in
declaration order, so a correct implementation overwrites every NaN. Zero
poisoning would silently match a "forgot to write index k" bug; NaN
poisoning catches it.
"""

from __future__ import annotations

import sys

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

pytestmark = pytest.mark.skipif(
    sys.platform == "win32",
    reason="assembly requires POSIX (Linux/WSL2)",
)

from _helpers import (  # noqa: E402
    build_dynamic_schema,
    field_list_strategy,
    make_segment,
    pack_row,
    random_value_for,
    release_segment,
)
from quorin.assembly import assemble as assemble_numba  # noqa: E402
from quorin.assembly import prewarm  # noqa: E402
from quorin.layout import insert  # noqa: E402
from quorin.pool import BufferPool  # noqa: E402
from quorin.schema import FeatureField, total_element_count  # noqa: E402
from quorin.serving import assemble as assemble_python  # noqa: E402

_HYPO = settings(
    max_examples=200,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture],
)


@pytest.fixture(scope="module", autouse=True)
def _prewarm_numba() -> None:
    prewarm()


def _nan_buf(n: int) -> np.ndarray:
    """A NaN-poisoned float32 buffer of length ``n``. Sentinel for catching
    'forgot to write position k' bugs."""
    return np.full(n, np.nan, dtype=np.float32)


# ---------------------------------------------------------------------------
# Properties.
# ---------------------------------------------------------------------------


@_HYPO
@given(field_list=field_list_strategy(), seed=st.integers(min_value=0, max_value=2**31 - 1))
def test_serving_pool_out_matches_fresh_alloc(field_list: list[FeatureField], seed: int) -> None:
    """Python oracle with NaN-poisoned ``out=`` matches a fresh-allocation call."""
    schema = build_dynamic_schema(field_list)
    rng = np.random.default_rng(seed)
    n = total_element_count(schema)
    seg = make_segment(schema, capacity=2)
    try:
        values = {f.name: random_value_for(f, rng) for f in schema.fields}
        insert(seg, "u", pack_row(schema, values))

        fresh = assemble_python(seg, "u")
        out = _nan_buf(n)
        returned = assemble_python(seg, "u", out=out)

        assert returned is out
        np.testing.assert_array_equal(returned, fresh)
        assert returned.dtype == np.float32
        assert returned.shape == (n,)
    finally:
        release_segment(seg)


@_HYPO
@given(field_list=field_list_strategy(), seed=st.integers(min_value=0, max_value=2**31 - 1))
def test_assembly_pool_out_matches_fresh_alloc(field_list: list[FeatureField], seed: int) -> None:
    """Numba kernel with NaN-poisoned ``out=`` matches a fresh-allocation call."""
    schema = build_dynamic_schema(field_list)
    rng = np.random.default_rng(seed)
    n = total_element_count(schema)
    seg = make_segment(schema, capacity=2)
    try:
        values = {f.name: random_value_for(f, rng) for f in schema.fields}
        insert(seg, "u", pack_row(schema, values))

        fresh = assemble_numba(seg, "u")
        out = _nan_buf(n)
        returned = assemble_numba(seg, "u", out=out)

        assert returned is out
        np.testing.assert_array_equal(returned, fresh)
    finally:
        release_segment(seg)


@_HYPO
@given(field_list=field_list_strategy(), seed=st.integers(min_value=0, max_value=2**31 - 1))
def test_serving_and_assembly_byte_identical_with_pooled_out(
    field_list: list[FeatureField], seed: int
) -> None:
    """Cross-path: Python oracle and Numba kernel produce byte-identical
    output even when both write into NaN-poisoned pooled buffers."""
    schema = build_dynamic_schema(field_list)
    rng = np.random.default_rng(seed)
    n = total_element_count(schema)
    seg = make_segment(schema, capacity=2)
    try:
        values = {f.name: random_value_for(f, rng) for f in schema.fields}
        insert(seg, "u", pack_row(schema, values))

        py_out = _nan_buf(n)
        nb_out = _nan_buf(n)
        py_returned = assemble_python(seg, "u", out=py_out)
        nb_returned = assemble_numba(seg, "u", out=nb_out)

        np.testing.assert_array_equal(py_returned, nb_returned)
    finally:
        release_segment(seg)


# ---------------------------------------------------------------------------
# Real BufferPool integration (one example each path; the heavy parity work
# is covered above with NaN-poisoned ad-hoc buffers).
# ---------------------------------------------------------------------------


@_HYPO
@given(field_list=field_list_strategy(), seed=st.integers(min_value=0, max_value=2**31 - 1))
def test_real_pool_checkout_parity_serving(field_list: list[FeatureField], seed: int) -> None:
    """The real BufferPool's checkout buffer reproduces fresh-allocation output."""
    schema = build_dynamic_schema(field_list)
    rng = np.random.default_rng(seed)
    seg = make_segment(schema, capacity=2)
    try:
        values = {f.name: random_value_for(f, rng) for f in schema.fields}
        insert(seg, "u", pack_row(schema, values))

        fresh = assemble_python(seg, "u")
        pool = BufferPool(schema, max_size=2)
        with pool.checkout() as buf:
            returned = assemble_python(seg, "u", out=buf)
            np.testing.assert_array_equal(returned, fresh)
    finally:
        release_segment(seg)
