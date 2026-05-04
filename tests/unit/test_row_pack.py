"""Unit tests for pyforge._internal.row_pack."""

from __future__ import annotations

import struct
from typing import Any

import numpy as np
import pytest

from pyforge._internal.row_pack import (
    _DTYPE_TO_STRUCT_CODE,
    _PLAN_CACHE,
    _build_plan,
    clear_cache,
    pack_row_from_list,
)
from pyforge.schema import (
    DTYPE_TO_NUMPY,
    DType,
    FeatureField,
    FeatureSchema,
    _hash_name,
    compute_row_offset_table,
    dtype,
    row_size,
)

# ---------------------------------------------------------------------------
# Schemas used across the file. Class-level so they have stable identity
# (the cache key is class identity, see ADR-009 §7).
# ---------------------------------------------------------------------------


class _ScalarOnly(FeatureSchema):
    version = 1
    fields = [
        FeatureField("a", dtype.float32),
        FeatureField("b", dtype.int64),
        FeatureField("c", dtype.uint8),
    ]


class _OneShape1D(FeatureSchema):
    version = 1
    fields = [
        FeatureField("emb", dtype.float32, shape=(8,)),
    ]


class _OneShape2D(FeatureSchema):
    version = 1
    fields = [
        FeatureField("matrix", dtype.float64, shape=(2, 3)),
    ]


class _Mixed(FeatureSchema):
    version = 1
    fields = [
        FeatureField("a", dtype.float32),
        FeatureField("emb", dtype.float32, shape=(128,)),
        FeatureField("b", dtype.float32),
    ]


class _AllDtypes(FeatureSchema):
    version = 1
    fields = [
        FeatureField("f32", dtype.float32),
        FeatureField("f64", dtype.float64),
        FeatureField("i32", dtype.int32),
        FeatureField("i64", dtype.int64),
        FeatureField("u8", dtype.uint8),
        FeatureField("vec", dtype.int32, shape=(4,)),
    ]


class _OutOfOrderHash(FeatureSchema):
    """Declaration order ≠ name_hash order — built from real hash math.

    >>> hash(zzz) > hash(aaa) > hash(mmm)

    Declaration: ``[zzz (off 0), aaa (off 64), mmm (off 128)]``.
    Sorted by name_hash: ``[mmm (off 128), aaa (off 64), zzz (off 0)]``.

    A consumer that walked the sorted table directly to build the format
    string would emit ``128x f 60x f`` (write at 128, then 132) and put
    every field at the wrong offset. This is the regression-guard
    schema for #B in the plan review.
    """

    version = 1
    fields = [
        FeatureField("zzz", dtype.float32),
        FeatureField("aaa", dtype.float32),
        FeatureField("mmm", dtype.float32),
    ]


class _ManyScalars200F32(FeatureSchema):
    version = 1
    fields = [FeatureField(f"f{i:03d}", dtype.float32) for i in range(200)]


class _NoScalars(FeatureSchema):
    """All fields shaped — exercises the ``scalar_struct is None`` branch."""

    version = 1
    fields = [
        FeatureField("emb1", dtype.float32, shape=(4,)),
        FeatureField("emb2", dtype.int32, shape=(2,)),
    ]


# ---------------------------------------------------------------------------
# Fixtures.
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clear_row_pack_cache() -> None:
    """Each test starts with an empty plan cache so memoization assertions
    aren't poisoned by other tests."""
    clear_cache()


# ---------------------------------------------------------------------------
# Helper — pack a values dict (in name_hash order) and return both
# representations for byte-equality comparison with the test helper.
# ---------------------------------------------------------------------------


def _values_in_hash_order(schema: type[FeatureSchema], by_name: dict[str, Any]) -> list[Any]:
    """Reorder ``by_name`` into the producer's name_hash wire order."""
    table = compute_row_offset_table(schema)
    hash_to_name = {_hash_name(f.name): f.name for f in schema.fields}
    return [by_name[hash_to_name[int(row["name_hash"])]] for row in table]


def _read_field(out: bytearray, schema: type[FeatureSchema], name: str) -> Any:
    """Decode one field's bytes from ``out`` per ``schema``'s row offset table."""
    table = compute_row_offset_table(schema)
    h = _hash_name(name)
    for row in table:
        if int(row["name_hash"]) == h:
            byte_off = int(row["byte_offset"])
            byte_cnt = int(row["byte_count"])
            elem_cnt = int(row["element_count"])
            dt_enum = DType(int(row["dtype_code"]))
            arr = np.frombuffer(
                bytes(out[byte_off : byte_off + byte_cnt]),
                dtype=DTYPE_TO_NUMPY[dt_enum],
            )
            if elem_cnt == 1:
                return arr[0]
            return arr.copy()
    raise KeyError(name)


# ---------------------------------------------------------------------------
# Round-trip: pack → field-extract → match input. Covers all dtypes.
# ---------------------------------------------------------------------------


def test_round_trip_scalar_all_dtypes() -> None:
    by_name: dict[str, Any] = {
        "f32": np.float32(1.5),
        "f64": np.float64(-2.5e10),
        "i32": np.int32(-12345),
        "i64": np.int64(2**40),
        "u8": np.uint8(200),
        "vec": np.array([1, 2, 3, 4], dtype=np.int32),
    }
    out = bytearray(row_size(_AllDtypes))
    pack_row_from_list(_AllDtypes, _values_in_hash_order(_AllDtypes, by_name), out)

    assert _read_field(out, _AllDtypes, "f32") == np.float32(1.5)
    assert _read_field(out, _AllDtypes, "f64") == np.float64(-2.5e10)
    assert _read_field(out, _AllDtypes, "i32") == np.int32(-12345)
    assert _read_field(out, _AllDtypes, "i64") == np.int64(2**40)
    assert _read_field(out, _AllDtypes, "u8") == np.uint8(200)
    assert np.array_equal(
        _read_field(out, _AllDtypes, "vec"), np.array([1, 2, 3, 4], dtype=np.int32)
    )


def test_round_trip_1d_shape() -> None:
    emb = np.arange(8, dtype=np.float32) * 0.5
    out = bytearray(row_size(_OneShape1D))
    pack_row_from_list(_OneShape1D, [emb.tolist()], out)
    assert np.array_equal(_read_field(out, _OneShape1D, "emb"), emb)


def test_round_trip_2d_shape() -> None:
    matrix = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float64)
    # 2D flattened by msgpack as nested lists; row_pack does .reshape(-1).
    out = bytearray(row_size(_OneShape2D))
    pack_row_from_list(_OneShape2D, [matrix.tolist()], out)
    got = _read_field(out, _OneShape2D, "matrix")
    assert np.array_equal(got, matrix.reshape(-1))


def test_round_trip_nan_inf_preserved_bitwise() -> None:
    """Invariant #12 cross-check: NaN bit pattern survives the pack."""
    by_name = {
        "f32": np.float32(np.nan),
        "f64": np.float64(np.inf),
        "i32": np.int32(0),
        "i64": np.int64(0),
        "u8": np.uint8(0),
        "vec": np.array([0, 0, 0, 0], dtype=np.int32),
    }
    out = bytearray(row_size(_AllDtypes))
    pack_row_from_list(_AllDtypes, _values_in_hash_order(_AllDtypes, by_name), out)
    assert np.isnan(_read_field(out, _AllDtypes, "f32"))
    assert np.isposinf(_read_field(out, _AllDtypes, "f64"))


# ---------------------------------------------------------------------------
# Validation errors.
# ---------------------------------------------------------------------------


def test_too_few_values_raises() -> None:
    out = bytearray(row_size(_ScalarOnly))
    with pytest.raises(ValueError, match="expected 3 values"):
        pack_row_from_list(_ScalarOnly, [1.0, 2], out)


def test_too_many_values_raises() -> None:
    out = bytearray(row_size(_ScalarOnly))
    with pytest.raises(ValueError, match="expected 3 values"):
        pack_row_from_list(_ScalarOnly, [1.0, 2, 3, 4], out)


def test_out_length_mismatch_raises() -> None:
    out = bytearray(row_size(_ScalarOnly) + 1)
    with pytest.raises(ValueError, match="out length"):
        pack_row_from_list(_ScalarOnly, [1.0, 2, 3], out)


def test_shaped_element_count_mismatch_raises() -> None:
    out = bytearray(row_size(_OneShape1D))
    # _OneShape1D expects 8 elements, give 7.
    with pytest.raises(ValueError, match="element_count"):
        pack_row_from_list(_OneShape1D, [[1.0] * 7], out)


# ---------------------------------------------------------------------------
# Memoization.
# ---------------------------------------------------------------------------


def test_plan_is_cached_by_class_identity() -> None:
    out = bytearray(row_size(_ScalarOnly))
    by_name: dict[str, Any] = {"a": 1.0, "b": 2, "c": 3}
    values = _values_in_hash_order(_ScalarOnly, by_name)
    pack_row_from_list(_ScalarOnly, values, out)
    plan1 = _PLAN_CACHE[_ScalarOnly]
    pack_row_from_list(_ScalarOnly, values, out)
    plan2 = _PLAN_CACHE[_ScalarOnly]
    assert plan1 is plan2


# ---------------------------------------------------------------------------
# Cache-line alignment fires the consolidated path (#A regression guard).
# ---------------------------------------------------------------------------


def test_200_field_float32_uses_single_struct_with_pad_codes() -> None:
    plan = _build_plan(_ManyScalars200F32)
    assert plan.scalar_struct is not None, "all-scalar schema should consolidate"
    assert len(plan.scalar_indices) == 200
    assert len(plan.shaped_slots) == 0
    # Format: "<" + "f" + "60xf" * 199. One call writes all 200 floats.
    expected_fmt = "<f" + "60xf" * 199
    # struct.Struct may report bytes (Python 3.7+); compare both forms.
    actual_fmt = plan.scalar_struct.format
    if isinstance(actual_fmt, bytes):
        actual_fmt = actual_fmt.decode("ascii")
    assert actual_fmt == expected_fmt
    # Sanity: Struct.size must equal the byte span from offset 0 to the
    # last scalar (4 bytes), not the row_size (which is padded to a
    # cache-line). 200 fields x 64 bytes/field starting at 0 = 12736 + 4 = 12740.
    assert plan.scalar_struct.size == 199 * 64 + 4


def test_200_field_pack_round_trips() -> None:
    """Behavioral check that the consolidated Struct actually works."""
    values = [float(i) * 1.5 for i in range(200)]
    out = bytearray(row_size(_ManyScalars200F32))
    # Build pack plan. Values are in declaration order here (which == our list
    # index), but the consumer's wire is name_hash order — reorder.
    table = compute_row_offset_table(_ManyScalars200F32)
    hash_to_name = {_hash_name(f.name): f.name for f in _ManyScalars200F32.fields}
    name_to_value = {f.name: values[i] for i, f in enumerate(_ManyScalars200F32.fields)}
    values_in_hash_order = [name_to_value[hash_to_name[int(row["name_hash"])]] for row in table]
    pack_row_from_list(_ManyScalars200F32, values_in_hash_order, out)
    for i, f in enumerate(_ManyScalars200F32.fields):
        got = _read_field(out, _ManyScalars200F32, f.name)
        assert got == np.float32(values[i]), f"field {f.name!r} mismatched"


# ---------------------------------------------------------------------------
# Mixed scalar + shaped — Struct skips the shaped region via Nx.
# ---------------------------------------------------------------------------


def test_mixed_scalar_and_shaped_split_correctly() -> None:
    plan = _build_plan(_Mixed)
    # Two scalars + one 128-elem shaped.
    assert len(plan.scalar_indices) == 2
    assert len(plan.shaped_slots) == 1
    # Shaped slot's byte_count is 128 floats x 4 = 512 bytes.
    _, slot = plan.shaped_slots[0]
    assert slot.element_count == 128
    assert slot.byte_count == 512


def test_mixed_round_trip() -> None:
    by_name = {
        "a": np.float32(1.25),
        "b": np.float32(-3.75),
        "emb": np.linspace(0, 1, 128, dtype=np.float32).tolist(),
    }
    out = bytearray(row_size(_Mixed))
    pack_row_from_list(_Mixed, _values_in_hash_order(_Mixed, by_name), out)
    assert _read_field(out, _Mixed, "a") == np.float32(1.25)
    assert _read_field(out, _Mixed, "b") == np.float32(-3.75)
    assert np.allclose(
        _read_field(out, _Mixed, "emb"),
        np.linspace(0, 1, 128, dtype=np.float32),
    )


# ---------------------------------------------------------------------------
# Name-hash vs declaration-order regression (#B).
# ---------------------------------------------------------------------------


def test_values_in_hash_order_land_at_correct_field_offsets() -> None:
    """The plan must walk the sorted table in name-hash order on the
    consumer side and write each value at the BYTE OFFSET corresponding
    to that hash position. A buggy implementation walking declaration
    order would silently swap field values.
    """
    # Verify the schema is actually decl≠hash so this test isn't a no-op.
    decl_order = [f.name for f in _OutOfOrderHash.fields]  # zzz, aaa, mmm
    hash_order = [
        next(f.name for f in _OutOfOrderHash.fields if _hash_name(f.name) == int(row["name_hash"]))
        for row in compute_row_offset_table(_OutOfOrderHash)
    ]
    assert decl_order != hash_order, "test schema must have decl≠hash order"

    # Distinct values for each field — if assignment is wrong, we'll see
    # the wrong value at the wrong offset.
    by_name = {
        "zzz": np.float32(111.0),
        "aaa": np.float32(222.0),
        "mmm": np.float32(333.0),
    }
    values_in_hash = _values_in_hash_order(_OutOfOrderHash, by_name)

    out = bytearray(row_size(_OutOfOrderHash))
    pack_row_from_list(_OutOfOrderHash, values_in_hash, out)

    assert _read_field(out, _OutOfOrderHash, "zzz") == np.float32(111.0)
    assert _read_field(out, _OutOfOrderHash, "aaa") == np.float32(222.0)
    assert _read_field(out, _OutOfOrderHash, "mmm") == np.float32(333.0)


def test_buffer_reuse_does_not_corrupt_subsequent_packs() -> None:
    """Same out buffer used twice with different values — second pack
    must not see ghosts of the first in the field-data regions."""
    out = bytearray(row_size(_AllDtypes))
    by_name_1 = {
        "f32": np.float32(1.0),
        "f64": np.float64(2.0),
        "i32": np.int32(3),
        "i64": np.int64(4),
        "u8": np.uint8(5),
        "vec": np.array([10, 20, 30, 40], dtype=np.int32),
    }
    pack_row_from_list(_AllDtypes, _values_in_hash_order(_AllDtypes, by_name_1), out)
    by_name_2 = {
        "f32": np.float32(99.0),
        "f64": np.float64(88.0),
        "i32": np.int32(77),
        "i64": np.int64(66),
        "u8": np.uint8(55),
        "vec": np.array([100, 200, 300, 400], dtype=np.int32),
    }
    pack_row_from_list(_AllDtypes, _values_in_hash_order(_AllDtypes, by_name_2), out)
    assert _read_field(out, _AllDtypes, "f32") == np.float32(99.0)
    assert _read_field(out, _AllDtypes, "f64") == np.float64(88.0)
    assert _read_field(out, _AllDtypes, "i32") == np.int32(77)
    assert _read_field(out, _AllDtypes, "i64") == np.int64(66)
    assert _read_field(out, _AllDtypes, "u8") == np.uint8(55)
    assert np.array_equal(
        _read_field(out, _AllDtypes, "vec"),
        np.array([100, 200, 300, 400], dtype=np.int32),
    )


# ---------------------------------------------------------------------------
# All-shaped schema — scalar_struct is None branch.
# ---------------------------------------------------------------------------


def test_no_scalars_skips_struct_pack_into() -> None:
    plan = _build_plan(_NoScalars)
    assert plan.scalar_struct is None
    assert plan.scalar_indices == ()
    assert len(plan.shaped_slots) == 2

    out = bytearray(row_size(_NoScalars))
    by_name = {
        "emb1": np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32),
        "emb2": np.array([100, 200], dtype=np.int32),
    }
    pack_row_from_list(_NoScalars, _values_in_hash_order(_NoScalars, by_name), out)
    assert np.array_equal(
        _read_field(out, _NoScalars, "emb1"),
        np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32),
    )
    assert np.array_equal(
        _read_field(out, _NoScalars, "emb2"),
        np.array([100, 200], dtype=np.int32),
    )


# ---------------------------------------------------------------------------
# Cross-check: the production packer agrees with the test helper byte-for-byte.
# ---------------------------------------------------------------------------


def test_byte_equal_with_test_pack_row_helper() -> None:
    """``tests/_helpers.pack_row`` is the test-side oracle. The production
    packer (different signature, different argument order) must produce the
    same bytes for the same input."""
    from _helpers import pack_row as test_pack_row

    by_name_arrays: dict[str, np.ndarray[Any, np.dtype[Any]]] = {
        "a": np.array([7.5], dtype=np.float32),
        "b": np.array([-99], dtype=np.int64),
        "c": np.array([42], dtype=np.uint8),
    }
    expected = test_pack_row(_ScalarOnly, by_name_arrays)

    # Production path: scalar values (not 1-elt arrays) in name_hash order.
    by_name_scalar: dict[str, Any] = {
        "a": 7.5,
        "b": -99,
        "c": 42,
    }
    out = bytearray(row_size(_ScalarOnly))
    pack_row_from_list(_ScalarOnly, _values_in_hash_order(_ScalarOnly, by_name_scalar), out)
    # Production packer doesn't zero padding; test helper does (fresh
    # bytearray). Compare only the data extents — read each field and
    # assert byte-equality.
    for f in _ScalarOnly.fields:
        h = _hash_name(f.name)
        for row in compute_row_offset_table(_ScalarOnly):
            if int(row["name_hash"]) == h:
                bo = int(row["byte_offset"])
                bc = int(row["byte_count"])
                assert bytes(out[bo : bo + bc]) == bytes(expected[bo : bo + bc])
                break


# ---------------------------------------------------------------------------
# DType code coverage — every supported dtype produces a working struct code.
# ---------------------------------------------------------------------------


def test_all_dtype_codes_present() -> None:
    for d in DType:
        code = _DTYPE_TO_STRUCT_CODE[d]
        # Must be a valid little-endian struct format code.
        struct.Struct("<" + code)


# ---------------------------------------------------------------------------
# Shape `(1,)` edge case — element_count is 1 but wire is a 1-elt list.
# Regression guard for the Hypothesis-found bug where the row_pack treated
# any field with elem_cnt==1 as a scalar regardless of original shape.
# ---------------------------------------------------------------------------


class _Shape1(FeatureSchema):
    version = 1
    fields = [FeatureField("v", dtype.float32, shape=(1,))]


def test_shape_1_treated_as_shaped_not_scalar() -> None:
    plan = _build_plan(_Shape1)
    # element_count == 1 but shape is (1,) — must be in shaped_slots, not scalar_indices.
    assert plan.scalar_indices == ()
    assert len(plan.shaped_slots) == 1
    # Wire format is a 1-elt list, not a scalar.
    out = bytearray(row_size(_Shape1))
    pack_row_from_list(_Shape1, [[3.14]], out)
    assert _read_field(out, _Shape1, "v") == np.float32(3.14)


# ---------------------------------------------------------------------------
# Step 15 regression: poison-pill defense depends on length-mismatch raise.
# Locks the consumer-side defense in pyforge.wal_consumer._apply.
# ---------------------------------------------------------------------------


class _LenMismatchSchema(FeatureSchema):
    version = 1
    fields = [
        FeatureField("a", dtype.float32),
        FeatureField("b", dtype.float32),
        FeatureField("c", dtype.float32),
    ]


def test_pack_row_from_list_rejects_length_mismatch() -> None:
    """Step 15 poison-pill defense: a stale producer wrote with OLD schema
    (fewer values); consumer with NEW schema must reject the message
    rather than silently pad.

    If this test ever fails, the WAL consumer's `_apply` poison-pill catch
    will not fire and stale messages will silently corrupt the new
    segment with garbage values. ADR-014 §"Consumer-side defense".
    """
    out = bytearray(row_size(_LenMismatchSchema))
    # 2 values for a 3-field schema — too short.
    with pytest.raises(ValueError, match="expected 3 values, got 2"):
        pack_row_from_list(_LenMismatchSchema, [1.0, 2.0], out)


def test_pack_row_from_list_rejects_too_many_values() -> None:
    """Symmetric to the under-count case — a future producer downgrade
    sending more values than the schema declares should also be rejected."""
    out = bytearray(row_size(_LenMismatchSchema))
    with pytest.raises(ValueError, match="expected 3 values, got 4"):
        pack_row_from_list(_LenMismatchSchema, [1.0, 2.0, 3.0, 4.0], out)
