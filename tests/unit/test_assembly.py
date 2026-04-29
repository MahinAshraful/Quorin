"""Unit tests for pyforge.assembly.assemble (Numba kernel).

Mirrors tests/unit/test_serving.py — same schemas, same edge cases — but
exercises the Numba path. Step 5's parity test
(tests/property/test_assembly_parity.py) handles property-based comparison
against the Python oracle; these unit tests are the smaller-grain coverage
that's easier to debug when something specific fails.
"""

from __future__ import annotations

import sys

import numpy as np
import pytest

pytestmark = pytest.mark.skipif(
    sys.platform == "win32",
    reason="assembly requires POSIX (Linux/WSL2)",
)

from _helpers import make_segment, pack_row, release_segment  # noqa: E402
from pyforge.assembly import assemble, prewarm  # noqa: E402
from pyforge.layout import insert  # noqa: E402
from pyforge.schema import (  # noqa: E402
    FeatureField,
    FeatureSchema,
    _hash_name,
    compute_assembly_table,
    compute_row_offset_table,
    dtype,
    total_element_count,
)
from pyforge.serving import EntityNotFoundError  # noqa: E402

# ---------------------------------------------------------------------------
# Module-level pre-warm so the first test doesn't pay ~300 ms compile cost.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module", autouse=True)
def _prewarm_numba() -> None:
    prewarm()


# ---------------------------------------------------------------------------
# Schemas — duplicated from test_serving.py (per-file convention; keeps each
# unit test file self-contained).
# ---------------------------------------------------------------------------


class _OneScalarF32(FeatureSchema):
    version = 1
    fields = [FeatureField("a", dtype.float32)]


class _OneShapedF32(FeatureSchema):
    version = 1
    fields = [FeatureField("emb", dtype.float32, shape=(128,))]


class _Two2DShape(FeatureSchema):
    version = 1
    fields = [FeatureField("matrix", dtype.uint8, shape=(8, 16))]


class _AllDtypes(FeatureSchema):
    version = 1
    fields = [
        FeatureField("f32", dtype.float32),
        FeatureField("f64", dtype.float64),
        FeatureField("i32", dtype.int32),
        FeatureField("i64", dtype.int64),
        FeatureField("u8", dtype.uint8),
    ]


class _ThreeMixed(FeatureSchema):
    version = 1
    fields = [
        FeatureField("age", dtype.int32),
        FeatureField("score", dtype.float32, shape=(4,)),
        FeatureField("flag", dtype.uint8),
    ]


class _PinnedDeclOrder(FeatureSchema):
    """Same locked schema as Step 4's pinned test."""

    version = 1
    fields = [
        FeatureField("zebra", dtype.float32),
        FeatureField("alpha", dtype.int32),
        FeatureField("mango", dtype.uint8),
        FeatureField("beta", dtype.float64),
    ]


# ---------------------------------------------------------------------------
# Tests 1-7: shape/dtype + per-DType round trip.
# ---------------------------------------------------------------------------


def test_returns_float32_contiguous_1d() -> None:
    seg = make_segment(_ThreeMixed, capacity=4)
    try:
        row = pack_row(
            _ThreeMixed,
            {
                "age": np.array([42], dtype=np.int32),
                "score": np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32),
                "flag": np.array([7], dtype=np.uint8),
            },
        )
        insert(seg, "u1", row)

        out = assemble(seg, "u1")

        assert out.dtype == np.float32
        assert out.shape == (total_element_count(_ThreeMixed),)
        assert out.shape == (1 + 4 + 1,)
        assert out.flags["C_CONTIGUOUS"]
        assert out.ndim == 1
    finally:
        release_segment(seg)


def test_single_field_scalar_round_trip() -> None:
    seg = make_segment(_OneScalarF32, capacity=2)
    try:
        row = pack_row(_OneScalarF32, {"a": np.array([3.14], dtype=np.float32)})
        insert(seg, "x", row)

        out = assemble(seg, "x")

        assert out.shape == (1,)
        assert out[0] == np.float32(3.14)
    finally:
        release_segment(seg)


def test_all_dtypes_round_trip_into_float32() -> None:
    seg = make_segment(_AllDtypes, capacity=2)
    try:
        values = {
            "f32": np.array([1.5], dtype=np.float32),
            "f64": np.array([2.5], dtype=np.float64),
            "i32": np.array([-7], dtype=np.int32),
            "i64": np.array([99], dtype=np.int64),
            "u8": np.array([255], dtype=np.uint8),
        }
        insert(seg, "u", pack_row(_AllDtypes, values))

        out = assemble(seg, "u")

        # Output is in declaration order: f32, f64, i32, i64, u8.
        assert out[0] == np.float32(1.5)
        assert out[1] == np.float32(2.5)
        assert out[2] == np.float32(-7)
        assert out[3] == np.float32(99)
        assert out[4] == np.float32(255)
        assert out.dtype == np.float32
    finally:
        release_segment(seg)


def test_lossy_int64_above_2_pow_24() -> None:
    """int64 values > 2^24 lose precision when cast to float32; should not crash
    and should match NumPy's standard cast semantics in the Numba kernel."""
    seg = make_segment(_AllDtypes, capacity=2)
    try:
        big = (1 << 30) + 7
        values = {
            "f32": np.array([0.0], dtype=np.float32),
            "f64": np.array([0.0], dtype=np.float64),
            "i32": np.array([0], dtype=np.int32),
            "i64": np.array([big], dtype=np.int64),
            "u8": np.array([0], dtype=np.uint8),
        }
        insert(seg, "u", pack_row(_AllDtypes, values))

        out = assemble(seg, "u")

        assert out[3] == np.float32(big)
        assert int(out[3]) != big  # precision loss is real
    finally:
        release_segment(seg)


def test_lossy_float64_narrowing() -> None:
    """float64 → float32 narrowing in the Numba kernel matches NumPy."""
    seg = make_segment(_AllDtypes, capacity=2)
    try:
        v64 = 1.0 + 1e-10
        values = {
            "f32": np.array([0.0], dtype=np.float32),
            "f64": np.array([v64], dtype=np.float64),
            "i32": np.array([0], dtype=np.int32),
            "i64": np.array([0], dtype=np.int64),
            "u8": np.array([0], dtype=np.uint8),
        }
        insert(seg, "u", pack_row(_AllDtypes, values))

        out = assemble(seg, "u")
        assert out[1] == np.float32(v64)
        assert out[1] == np.float32(1.0)  # narrowing collapses the +1e-10
    finally:
        release_segment(seg)


def test_negative_int_to_float32() -> None:
    seg = make_segment(_AllDtypes, capacity=2)
    try:
        values = {
            "f32": np.array([0.0], dtype=np.float32),
            "f64": np.array([0.0], dtype=np.float64),
            "i32": np.array([-42], dtype=np.int32),
            "i64": np.array([-(1 << 40)], dtype=np.int64),
            "u8": np.array([0], dtype=np.uint8),
        }
        insert(seg, "u", pack_row(_AllDtypes, values))

        out = assemble(seg, "u")
        assert out[2] == np.float32(-42.0)
        assert out[3] == np.float32(-(1 << 40))
    finally:
        release_segment(seg)


def test_nan_and_inf_round_trip() -> None:
    """fastmath=False on the @njit decorator is what makes this pass."""
    seg = make_segment(_AllDtypes, capacity=2)
    try:
        values = {
            "f32": np.array([np.nan], dtype=np.float32),
            "f64": np.array([np.inf], dtype=np.float64),
            "i32": np.array([0], dtype=np.int32),
            "i64": np.array([0], dtype=np.int64),
            "u8": np.array([0], dtype=np.uint8),
        }
        insert(seg, "u", pack_row(_AllDtypes, values))

        out = assemble(seg, "u")
        assert np.isnan(out[0])
        assert np.isposinf(out[1])

        values["f64"] = np.array([-np.inf], dtype=np.float64)
        insert(seg, "u", pack_row(_AllDtypes, values))
        out = assemble(seg, "u")
        assert np.isneginf(out[1])
    finally:
        release_segment(seg)


# ---------------------------------------------------------------------------
# Tests 8-9: shaped fields.
# ---------------------------------------------------------------------------


def test_shaped_field_flat_in_output() -> None:
    seg = make_segment(_OneShapedF32, capacity=2)
    try:
        emb = np.arange(128, dtype=np.float32) * 0.5
        insert(seg, "u", pack_row(_OneShapedF32, {"emb": emb}))

        out = assemble(seg, "u")

        assert out.shape == (128,)
        np.testing.assert_array_equal(out, emb)
    finally:
        release_segment(seg)


def test_2d_shape_flattens_c_order() -> None:
    seg = make_segment(_Two2DShape, capacity=2)
    try:
        m = np.arange(128, dtype=np.uint8).reshape(8, 16)
        insert(seg, "u", pack_row(_Two2DShape, {"matrix": m}))

        out = assemble(seg, "u")

        assert out.shape == (128,)
        np.testing.assert_array_equal(out, m.ravel().astype(np.float32))
    finally:
        release_segment(seg)


# ---------------------------------------------------------------------------
# Tests 10-12: error paths.
# ---------------------------------------------------------------------------


def test_entity_not_found_raises_same_class() -> None:
    """pyforge.assembly.assemble raises pyforge.serving.EntityNotFoundError
    (the same class — re-imported, not duplicated)."""
    seg = make_segment(_OneScalarF32, capacity=4)
    try:
        with pytest.raises(EntityNotFoundError) as excinfo:
            assemble(seg, "ghost")
        assert "ghost" in str(excinfo.value)
    finally:
        release_segment(seg)


def test_entity_not_found_carries_id_attribute() -> None:
    seg = make_segment(_OneScalarF32, capacity=4)
    try:
        try:
            assemble(seg, "ghost")
        except EntityNotFoundError as e:
            assert e.entity_id == "ghost"
        else:
            pytest.fail("EntityNotFoundError was not raised")
    finally:
        release_segment(seg)


def test_empty_entity_id_raises_value_error() -> None:
    seg = make_segment(_OneScalarF32, capacity=4)
    try:
        with pytest.raises(ValueError):
            assemble(seg, "")
    finally:
        release_segment(seg)


# ---------------------------------------------------------------------------
# Tests 13-16: cross-row, repeat, unicode, post-insert visibility.
# ---------------------------------------------------------------------------


def test_two_entities_no_crosstalk() -> None:
    seg = make_segment(_OneScalarF32, capacity=4)
    try:
        insert(seg, "a", pack_row(_OneScalarF32, {"a": np.array([1.0], dtype=np.float32)}))
        insert(seg, "b", pack_row(_OneScalarF32, {"a": np.array([2.0], dtype=np.float32)}))

        assert assemble(seg, "a")[0] == 1.0
        assert assemble(seg, "b")[0] == 2.0
    finally:
        release_segment(seg)


def test_repeated_assemble_independent_buffers() -> None:
    seg = make_segment(_OneScalarF32, capacity=4)
    try:
        insert(seg, "u", pack_row(_OneScalarF32, {"a": np.array([99.0], dtype=np.float32)}))

        out1 = assemble(seg, "u")
        out2 = assemble(seg, "u")

        assert out1 is not out2
        assert out1.flags["OWNDATA"]
        assert out2.flags["OWNDATA"]
        np.testing.assert_array_equal(out1, out2)

        out1[0] = -1.0
        assert out2[0] == 99.0
    finally:
        release_segment(seg)


def test_unicode_entity_id() -> None:
    seg = make_segment(_OneScalarF32, capacity=4)
    try:
        insert(
            seg,
            "ユーザー_001",
            pack_row(_OneScalarF32, {"a": np.array([7.5], dtype=np.float32)}),
        )

        out = assemble(seg, "ユーザー_001")
        assert out[0] == np.float32(7.5)
    finally:
        release_segment(seg)


def test_visibility_after_insert_in_same_process() -> None:
    seg = make_segment(_OneScalarF32, capacity=8)
    try:
        for i in range(5):
            insert(
                seg,
                f"u{i}",
                pack_row(_OneScalarF32, {"a": np.array([float(i)], dtype=np.float32)}),
            )
            out = assemble(seg, f"u{i}")
            assert out[0] == float(i)
    finally:
        release_segment(seg)


# ---------------------------------------------------------------------------
# Tests 17-19: declaration-order contract + supporting checks.
# ---------------------------------------------------------------------------


def test_declaration_order_pinned() -> None:
    """Numba kernel honors Step 4's locked declaration-order contract."""
    sorted_table = compute_row_offset_table(_PinnedDeclOrder)
    decl_table = compute_assembly_table(_PinnedDeclOrder)
    inverted = any(
        int(sorted_table[i]["name_hash"]) != int(decl_table[i]["name_hash"])
        for i in range(decl_table.size)
    )
    assert inverted, "schema's hash and declaration orders coincide; pick different names"

    for i, f in enumerate(_PinnedDeclOrder.fields):
        assert int(decl_table[i]["name_hash"]) == _hash_name(f.name)

    seg = make_segment(_PinnedDeclOrder, capacity=2)
    try:
        values = {
            "zebra": np.array([2.5], dtype=np.float32),
            "alpha": np.array([777], dtype=np.int32),
            "mango": np.array([3], dtype=np.uint8),
            "beta": np.array([-0.125], dtype=np.float64),
        }
        insert(seg, "u", pack_row(_PinnedDeclOrder, values))
        out = assemble(seg, "u")

        assert out[0] == np.float32(2.5)
        assert out[1] == np.float32(777)
        assert out[2] == np.float32(3)
        assert out[3] == np.float32(-0.125)
    finally:
        release_segment(seg)


def test_prewarm_does_not_crash_idempotent() -> None:
    """prewarm is safe to call multiple times. The autouse module fixture
    already calls it once; this test calls it twice more."""
    prewarm()
    prewarm()


def test_layout_assembly_arrays_match_assembly_table() -> None:
    """The flat parallel arrays on SegmentLayout agree element-by-element
    with the structured assembly_table they're derived from."""
    seg = make_segment(_AllDtypes, capacity=2)
    try:
        layout = seg.layout
        asm = layout.assembly_table
        np.testing.assert_array_equal(
            layout.assembly_byte_offsets,
            asm["byte_offset"].astype(np.int64),
        )
        np.testing.assert_array_equal(
            layout.assembly_byte_counts,
            asm["byte_count"].astype(np.int64),
        )
        np.testing.assert_array_equal(layout.assembly_dtype_codes, asm["dtype_code"])
        np.testing.assert_array_equal(
            layout.assembly_element_counts,
            asm["element_count"].astype(np.int64),
        )
        # And dtypes are exactly what the Numba signature expects.
        assert layout.assembly_byte_offsets.dtype == np.int64
        assert layout.assembly_byte_counts.dtype == np.int64
        assert layout.assembly_dtype_codes.dtype == np.uint8
        assert layout.assembly_element_counts.dtype == np.int64
    finally:
        release_segment(seg)
