"""Unit tests for ``quorin.layout.pack_row`` (Step 17).

Public kwargs-style row packer. Walks ``schema.fields`` in declaration order
and writes each value into a fresh ``bytearray(row_size(schema))`` via numpy
views. Distinct from :func:`tests._helpers.pack_row` (dict-based, strict
dtype-check) and from :func:`quorin._internal.row_pack.pack_row_from_list`
(name-hash-ordered consumer hot path).
"""

from __future__ import annotations

import sys

import numpy as np
import pytest

pytestmark = pytest.mark.skipif(
    sys.platform == "win32",
    reason="pack_row requires posix-only quorin._internal modules pulled by quorin.layout",
)

from _helpers import make_segment, release_segment  # noqa: E402
from quorin.layout import insert, pack_row  # noqa: E402
from quorin.schema import FeatureField, FeatureSchema, dtype  # noqa: E402
from quorin.serving import assemble  # noqa: E402


class _ThreeScalars(FeatureSchema):
    version = 1
    fields = [
        FeatureField("a", dtype.float32),
        FeatureField("b", dtype.int32),
        FeatureField("c", dtype.float32),
    ]


class _ScalarPlusEmbedding(FeatureSchema):
    version = 1
    fields = [
        FeatureField("x", dtype.float32),
        FeatureField("emb", dtype.float32, shape=(8,)),
    ]


class _AllFiveDtypes(FeatureSchema):
    version = 1
    fields = [
        FeatureField("f32", dtype.float32),
        FeatureField("f64", dtype.float64),
        FeatureField("i32", dtype.int32),
        FeatureField("i64", dtype.int64),
        FeatureField("u8", dtype.uint8),
    ]


def test_happy_path_scalars_roundtrip_via_assemble() -> None:
    """pack_row + insert + assemble returns the values we packed."""
    seg = make_segment(_ThreeScalars, capacity=4)
    try:
        row = pack_row(_ThreeScalars, a=0.5, b=42, c=12.3)
        insert(seg, "user_001", row)
        out = assemble(seg, "user_001")
        np.testing.assert_array_equal(out, np.array([0.5, 42.0, 12.3], dtype=np.float32))
    finally:
        release_segment(seg)


def test_happy_path_with_shaped_field() -> None:
    """Shaped field accepts list and ndarray; both round-trip identically."""
    seg = make_segment(_ScalarPlusEmbedding, capacity=4)
    try:
        emb = list(range(8))
        row = pack_row(_ScalarPlusEmbedding, x=1.0, emb=emb)
        insert(seg, "u1", row)
        out = assemble(seg, "u1")
        # Output is concatenated [x, emb...] = 1 + 8 = 9 floats
        expected = np.array([1.0, *emb], dtype=np.float32)
        np.testing.assert_array_equal(out, expected)
    finally:
        release_segment(seg)


def test_missing_field_raises() -> None:
    with pytest.raises(ValueError, match=r"missing fields: \['c'\]"):
        pack_row(_ThreeScalars, a=0.5, b=42)


def test_unknown_extra_field_raises() -> None:
    with pytest.raises(ValueError, match=r"unknown fields: \['z'\]"):
        pack_row(_ThreeScalars, a=0.5, b=42, c=1.0, z=99)


def test_wrong_shape_raises() -> None:
    with pytest.raises(ValueError, match=r"element_count 4 != expected 8"):
        # emb expects 8 elements; pass 4
        pack_row(_ScalarPlusEmbedding, x=1.0, emb=[1, 2, 3, 4])


def test_all_five_dtypes_roundtrip() -> None:
    """pack_row handles every supported dtype (FLOAT32/FLOAT64/INT32/INT64/UINT8)."""
    seg = make_segment(_AllFiveDtypes, capacity=4)
    try:
        row = pack_row(_AllFiveDtypes, f32=1.5, f64=2.5, i32=-7, i64=999_999_999_999, u8=255)
        insert(seg, "u", row)
        out = assemble(seg, "u")
        # assemble concatenates everything as float32 — check it cast roughly right
        # (UINT8 255, INT32 -7, INT64 999B all round-trip via float32 cast lossy
        # on i64 but exact on the others).
        assert out[0] == np.float32(1.5)
        assert out[1] == np.float32(2.5)
        assert out[2] == np.float32(-7)
        # i64 999_999_999_999 → float32 is approximate; just check it's positive + huge
        assert out[3] > 1e11
        assert out[4] == np.float32(255)
    finally:
        release_segment(seg)


def test_python_int_auto_coerced_to_int32() -> None:
    """Python int passed for an int32 field gets np.asarray(value, dtype=int32)'d."""
    seg = make_segment(_ThreeScalars, capacity=4)
    try:
        # Pass plain Python int for the int32 field; pack_row should auto-convert.
        row = pack_row(_ThreeScalars, a=0.0, b=7, c=0.0)
        insert(seg, "u", row)
        out = assemble(seg, "u")
        assert out[1] == np.float32(7)
    finally:
        release_segment(seg)


def test_byte_equality_against_test_helper_pack_row() -> None:
    """pack_row produces bytes identical to tests/_helpers.py::pack_row for equivalent input."""
    from _helpers import pack_row as helper_pack_row

    new_bytes = pack_row(_ThreeScalars, a=0.5, b=42, c=12.3)
    helper_bytes = helper_pack_row(
        _ThreeScalars,
        {
            "a": np.array(0.5, dtype=np.float32),
            "b": np.array(42, dtype=np.int32),
            "c": np.array(12.3, dtype=np.float32),
        },
    )
    assert new_bytes == helper_bytes
