"""Unit tests for quorin.serving.assemble.

These tests are the contract Step 5's Numba kernel must reproduce byte-for-byte.
``TestDeclarationOrderPinned::test_known_schema_known_output`` is the pinned
test for Step 5's parity check: any change in ordering or cast semantics
fails it loudly.
"""

from __future__ import annotations

import sys

import numpy as np
import pytest

pytestmark = pytest.mark.skipif(
    sys.platform == "win32",
    reason="serving requires POSIX (Linux/WSL2)",
)

from _helpers import make_segment, pack_row, release_segment  # noqa: E402
from quorin.layout import insert  # noqa: E402
from quorin.schema import (  # noqa: E402
    FeatureField,
    FeatureSchema,
    _hash_name,
    compute_assembly_table,
    compute_row_offset_table,
    dtype,
    total_element_count,
)
from quorin.serving import EntityNotFoundError, assemble  # noqa: E402

# ---------------------------------------------------------------------------
# Schemas used across tests. Defined at module scope so __init_subclass__ runs
# at import time (the same as production schemas).
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


# ---------------------------------------------------------------------------
# Tests 1-6: basic shape, dtype, and per-DType round trip.
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
    and should match NumPy's standard cast semantics."""
    seg = make_segment(_AllDtypes, capacity=2)
    try:
        big = (1 << 30) + 7  # 2^30 + 7; not exactly representable in float32
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
        # Sanity: precision was lost — round-trip via float32 doesn't recover the +7.
        assert int(out[3]) != big
    finally:
        release_segment(seg)


def test_lossy_float64_narrowing() -> None:
    """float64 → float32 narrowing matches NumPy's standard cast."""
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
        # And the narrowing is real: 1.0 + 1e-10 collapses to 1.0 in float32.
        assert out[1] == np.float32(1.0)
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

        # Now -inf via a fresh insert.
        values["f64"] = np.array([-np.inf], dtype=np.float64)
        insert(seg, "u", pack_row(_AllDtypes, values))
        out = assemble(seg, "u")
        assert np.isneginf(out[1])
    finally:
        release_segment(seg)


# ---------------------------------------------------------------------------
# Tests 8-9: shaped fields flatten correctly.
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
    """A (8, 16) field flattens row-major (C order) into the output."""
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


def test_entity_not_found_raises() -> None:
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

        # Mutating one must not affect the other.
        out1[0] = -1.0
        assert out2[0] == 99.0
    finally:
        release_segment(seg)


def test_unicode_entity_id() -> None:
    seg = make_segment(_OneScalarF32, capacity=4)
    try:
        insert(
            seg, "ユーザー_001", pack_row(_OneScalarF32, {"a": np.array([7.5], dtype=np.float32)})
        )

        out = assemble(seg, "ユーザー_001")
        assert out[0] == np.float32(7.5)
    finally:
        release_segment(seg)


def test_visibility_after_insert_in_same_process() -> None:
    """Step 3 invariant: cursors advance before slot is marked OCCUPIED, so a
    successful insert is immediately visible to assemble in the same process."""
    seg = make_segment(_OneScalarF32, capacity=8)
    try:
        for i in range(5):
            insert(
                seg, f"u{i}", pack_row(_OneScalarF32, {"a": np.array([float(i)], dtype=np.float32)})
            )
            out = assemble(seg, f"u{i}")
            assert out[0] == float(i)
    finally:
        release_segment(seg)


# ---------------------------------------------------------------------------
# Tests 17-20: declaration-order contract + supporting helpers.
# ---------------------------------------------------------------------------


# Four field names spanning enough hash space that hash order is virtually
# certain to differ from declaration order at >=1 position (probability of
# total agreement is 1/4! = 1/24 ≈ 4%). The test asserts the inversion
# exists so we know the contract isn't being trivially satisfied.
class _PinnedDeclOrder(FeatureSchema):
    """Declaration order: [zebra, alpha, mango, beta]."""

    version = 1
    fields = [
        FeatureField("zebra", dtype.float32),
        FeatureField("alpha", dtype.int32),
        FeatureField("mango", dtype.uint8),
        FeatureField("beta", dtype.float64),
    ]


class TestDeclarationOrderPinned:
    """The contract Step 5's Numba parity test will lock against."""

    def test_known_schema_known_output(self) -> None:
        sorted_table = compute_row_offset_table(_PinnedDeclOrder)
        decl_table = compute_assembly_table(_PinnedDeclOrder)

        # Sanity: hash order differs from declaration order at >=1 position.
        # If this fires, the chosen field names happen to share both
        # orderings — pick different names to restore the test's
        # discriminative power.
        inverted_positions = [
            i
            for i in range(decl_table.size)
            if int(sorted_table[i]["name_hash"]) != int(decl_table[i]["name_hash"])
        ]
        assert inverted_positions, (
            "_PinnedDeclOrder field names produce identical hash and "
            "declaration orderings; pick names with different blake2b values."
        )

        # Declaration order is preserved in assembly_table (verified by
        # name_hash on each row).
        for i, f in enumerate(_PinnedDeclOrder.fields):
            assert int(decl_table[i]["name_hash"]) == _hash_name(f.name)

        # The contract: assembled output values are in declaration order.
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

            # vec[i] is the i-th DECLARED field, regardless of hash order.
            assert out[0] == np.float32(2.5), "vec[0] must be the FIRST DECLARED field"
            assert out[1] == np.float32(777), "vec[1] must be the SECOND DECLARED field"
            assert out[2] == np.float32(3), "vec[2] must be the THIRD DECLARED field"
            assert out[3] == np.float32(-0.125), "vec[3] must be the FOURTH DECLARED field"
        finally:
            release_segment(seg)


def test_assembly_table_is_declaration_order() -> None:
    table = compute_assembly_table(_ThreeMixed)
    assert int(table[0]["name_hash"]) == _hash_name("age")
    assert int(table[1]["name_hash"]) == _hash_name("score")
    assert int(table[2]["name_hash"]) == _hash_name("flag")


def test_assembly_table_byte_offsets_match_row_offset_table() -> None:
    """Same physical layout, two views — for each field, the byte_offset must
    agree across hash-sorted and declaration-order tables."""
    decl = compute_assembly_table(_AllDtypes)
    sorted_ = compute_row_offset_table(_AllDtypes)

    decl_by_hash = {int(decl[i]["name_hash"]): decl[i] for i in range(decl.size)}
    for i in range(sorted_.size):
        h = int(sorted_[i]["name_hash"])
        assert int(decl_by_hash[h]["byte_offset"]) == int(sorted_[i]["byte_offset"])
        assert int(decl_by_hash[h]["byte_count"]) == int(sorted_[i]["byte_count"])
        assert int(decl_by_hash[h]["element_count"]) == int(sorted_[i]["element_count"])
        assert int(decl_by_hash[h]["dtype_code"]) == int(sorted_[i]["dtype_code"])


def test_total_element_count_helper() -> None:
    assert total_element_count(_OneScalarF32) == 1
    assert total_element_count(_OneShapedF32) == 128
    assert total_element_count(_Two2DShape) == 8 * 16
    assert total_element_count(_AllDtypes) == 5
    assert total_element_count(_ThreeMixed) == 1 + 4 + 1
