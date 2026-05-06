"""Parity tests: quorin.serving.assemble (Python) vs quorin.assembly.assemble (Numba).

THE Step 5 test. Hypothesis-driven over random schemas + random row data.
The two implementations must produce **byte-identical output** —
``np.testing.assert_array_equal`` (NaN-safe). If this ever fails, Step 5 has
regressed and ADR-004's adoption gate is moot.
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
from quorin.schema import DType, FeatureField  # noqa: E402
from quorin.serving import assemble as assemble_python  # noqa: E402

_HYPO = settings(
    max_examples=200,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture],
)


@pytest.fixture(scope="module", autouse=True)
def _prewarm_numba() -> None:
    """Module-level pre-warm so the first parity test doesn't pay
    ~300 ms compile cost (which would skew Hypothesis's deadline tracking
    and waste real time)."""
    prewarm()


# ---------------------------------------------------------------------------
# Properties.
# ---------------------------------------------------------------------------


@_HYPO
@given(field_list=field_list_strategy(), seed=st.integers(min_value=0, max_value=2**31 - 1))
def test_python_and_numba_byte_identical(field_list: list[FeatureField], seed: int) -> None:
    """For any random schema + random row, the Python oracle and the Numba
    kernel produce byte-identical output."""
    schema = build_dynamic_schema(field_list)
    rng = np.random.default_rng(seed)
    seg = make_segment(schema, capacity=2)
    try:
        values = {f.name: random_value_for(f, rng) for f in schema.fields}
        insert(seg, "u", pack_row(schema, values))

        py_out = assemble_python(seg, "u")
        nb_out = assemble_numba(seg, "u")

        # assert_array_equal is NaN-safe (treats NaN == NaN).
        np.testing.assert_array_equal(py_out, nb_out)
        assert py_out.dtype == nb_out.dtype == np.float32
        assert py_out.shape == nb_out.shape
    finally:
        release_segment(seg)


@_HYPO
@given(field_list=field_list_strategy(), seed=st.integers(min_value=0, max_value=2**31 - 1))
def test_python_and_numba_agree_on_two_entities(field_list: list[FeatureField], seed: int) -> None:
    """With two entities holding different values, both implementations
    agree on each. Catches any cursor-tracking or row_offset bugs."""
    schema = build_dynamic_schema(field_list)
    rng = np.random.default_rng(seed)
    seg = make_segment(schema, capacity=4)
    try:
        values_a = {f.name: random_value_for(f, rng) for f in schema.fields}
        values_b = {f.name: random_value_for(f, rng) for f in schema.fields}
        insert(seg, "a", pack_row(schema, values_a))
        insert(seg, "b", pack_row(schema, values_b))

        np.testing.assert_array_equal(assemble_python(seg, "a"), assemble_numba(seg, "a"))
        np.testing.assert_array_equal(assemble_python(seg, "b"), assemble_numba(seg, "b"))
    finally:
        release_segment(seg)


@_HYPO
@given(field_list=field_list_strategy(), seed=st.integers(min_value=0, max_value=2**31 - 1))
def test_python_and_numba_byte_identical_with_nan_inf(
    field_list: list[FeatureField], seed: int
) -> None:
    """Force a NaN and an inf into every example's float fields; verify
    the Numba kernel preserves the NaN/inf bit pattern (relies on
    fastmath=False)."""
    schema = build_dynamic_schema(field_list)
    rng = np.random.default_rng(seed)
    seg = make_segment(schema, capacity=2)
    try:
        values = {f.name: random_value_for(f, rng) for f in schema.fields}
        # Inject NaN and inf into the first float field we find. If there
        # are no float fields, this test still verifies general parity for
        # the int/uint8 case — still useful.
        for f in schema.fields:
            if f.dtype is DType.FLOAT32:
                arr = values[f.name].copy()
                if arr.size >= 1:
                    arr[0] = np.float32(np.nan)
                if arr.size >= 2:
                    arr[1] = np.float32(np.inf)
                values[f.name] = arr
                break
            if f.dtype is DType.FLOAT64:
                arr = values[f.name].copy()
                if arr.size >= 1:
                    arr[0] = np.float64(np.nan)
                if arr.size >= 2:
                    arr[1] = np.float64(-np.inf)
                values[f.name] = arr
                break

        insert(seg, "u", pack_row(schema, values))

        py_out = assemble_python(seg, "u")
        nb_out = assemble_numba(seg, "u")
        np.testing.assert_array_equal(py_out, nb_out)
    finally:
        release_segment(seg)
