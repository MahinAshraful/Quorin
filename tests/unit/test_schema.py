"""Unit tests for quorin.schema.

Covers FeatureField value checks, FeatureSchema class-definition-time
validation, compile_schema layout/determinism, total_segment_size page
rounding, and the pinned hash regression guard.
"""

from __future__ import annotations

import hashlib
import inspect
from dataclasses import FrozenInstanceError

import numpy as np
import pytest

import quorin.schema as _schema_module
from quorin.schema import (
    CACHE_LINE_SIZE,
    HEADER_SIZE,
    OFFSET_TABLE_DTYPE,
    PAGE_SIZE,
    DType,
    FeatureField,
    FeatureSchema,
    _hash_name,
    compile_schema,
    dtype,
    total_segment_size,
)

# ---------------------------------------------------------------------------
# A reusable non-trivial schema shared across several tests. Prefixed with
# underscore so pytest does not collect it as a test class.
# ---------------------------------------------------------------------------


class _UserFeatures(FeatureSchema):
    version = 1
    fields = [
        FeatureField("age_normalized", dtype.float32),
        FeatureField("session_count_7d", dtype.int32),
        FeatureField("ltv_score", dtype.float32),
        FeatureField("behavior_embedding", dtype.float32, shape=(128,)),
    ]


# ---------------------------------------------------------------------------
# FeatureField
# ---------------------------------------------------------------------------


def test_scalar_float32_byte_count_is_4() -> None:
    f = FeatureField("x", dtype.float32)
    assert f.byte_count == 4
    assert f.element_count == 1
    assert f.shape == ()


def test_embedding_128_float32_byte_count_is_512() -> None:
    f = FeatureField("embedding", dtype.float32, shape=(128,))
    assert f.byte_count == 512
    assert f.element_count == 128


def test_scalar_has_empty_shape_and_element_count_1() -> None:
    f = FeatureField("x", dtype.int64)
    assert f.shape == ()
    assert f.element_count == 1
    assert f.byte_count == 8


def test_2d_shape_element_count_is_product() -> None:
    f = FeatureField("grid", dtype.uint8, shape=(4, 3))
    assert f.element_count == 12
    assert f.byte_count == 12


def test_shape_with_zero_dim_rejected() -> None:
    with pytest.raises(ValueError, match="positive"):
        FeatureField("x", dtype.float32, shape=(0,))


def test_shape_with_negative_dim_rejected() -> None:
    with pytest.raises(ValueError, match="positive"):
        FeatureField("x", dtype.float32, shape=(-1,))


def test_shape_with_bool_dim_rejected() -> None:
    # bool is a subclass of int — guard against it masquerading as a dim.
    with pytest.raises(TypeError, match="int"):
        FeatureField("x", dtype.float32, shape=(True,))  # type: ignore[arg-type]


def test_shape_as_list_rejected() -> None:
    with pytest.raises(TypeError, match="tuple"):
        FeatureField("x", dtype.float32, shape=[128])  # type: ignore[arg-type]


def test_empty_name_rejected() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        FeatureField("", dtype.float32)


def test_non_dtype_rejected() -> None:
    with pytest.raises(TypeError, match="DType"):
        FeatureField("x", "float32")  # type: ignore[arg-type]


def test_frozen_cannot_reassign_name() -> None:
    f = FeatureField("x", dtype.float32)
    with pytest.raises(FrozenInstanceError):
        f.name = "y"  # type: ignore[misc]


def test_all_five_dtypes_accepted() -> None:
    for d in (DType.FLOAT32, DType.FLOAT64, DType.INT32, DType.INT64, DType.UINT8):
        f = FeatureField("x", d)
        assert f.dtype is d


def test_dtype_namespace_aliases_match_enum() -> None:
    assert dtype.float32 is DType.FLOAT32
    assert dtype.float64 is DType.FLOAT64
    assert dtype.int32 is DType.INT32
    assert dtype.int64 is DType.INT64
    assert dtype.uint8 is DType.UINT8


# ---------------------------------------------------------------------------
# FeatureSchema — validation at class-definition time
# ---------------------------------------------------------------------------


def test_missing_version_rejected() -> None:
    with pytest.raises(TypeError, match="version"):

        class _S(FeatureSchema):
            fields = [FeatureField("x", dtype.float32)]


def test_missing_fields_rejected() -> None:
    with pytest.raises(TypeError, match="fields"):

        class _S(FeatureSchema):
            version = 1


def test_non_int_version_rejected() -> None:
    with pytest.raises(TypeError, match="version"):

        class _S(FeatureSchema):
            version = "1"  # type: ignore[assignment]
            fields = [FeatureField("x", dtype.float32)]


def test_bool_version_rejected() -> None:
    with pytest.raises(TypeError, match="version"):

        class _S(FeatureSchema):
            version = True  # type: ignore[assignment]
            fields = [FeatureField("x", dtype.float32)]


def test_empty_fields_list_rejected() -> None:
    with pytest.raises(ValueError, match="empty"):

        class _S(FeatureSchema):
            version = 1
            fields: list[FeatureField] = []


def test_duplicate_field_names_rejected() -> None:
    with pytest.raises(ValueError, match="duplicate"):

        class _S(FeatureSchema):
            version = 1
            fields = [
                FeatureField("x", dtype.float32),
                FeatureField("x", dtype.float64),
            ]


def test_non_feature_field_in_fields_rejected() -> None:
    with pytest.raises(TypeError, match="FeatureField"):

        class _S(FeatureSchema):
            version = 1
            fields = ["not a field"]  # type: ignore[list-item]


def test_fields_as_tuple_rejected() -> None:
    # We require a list, not a tuple — simpler mental model.
    with pytest.raises(TypeError, match="list"):

        class _S(FeatureSchema):
            version = 1
            fields = (FeatureField("x", dtype.float32),)  # type: ignore[assignment]


def test_valid_schema_defines_cleanly() -> None:
    class UserFeatures(FeatureSchema):
        version = 1
        fields = [
            FeatureField("age_normalized", dtype.float32),
            FeatureField("session_count_7d", dtype.int32),
        ]

    assert UserFeatures.version == 1
    assert len(UserFeatures.fields) == 2
    assert UserFeatures.fields[0].name == "age_normalized"


# ---------------------------------------------------------------------------
# compile_schema
# ---------------------------------------------------------------------------


def test_single_field_table_shape() -> None:
    class _S(FeatureSchema):
        version = 1
        fields = [FeatureField("x", dtype.float32)]

    t = compile_schema(_S)
    assert t.shape == (1,)


def test_table_dtype_matches_constant() -> None:
    t = compile_schema(_UserFeatures)
    assert t.dtype == OFFSET_TABLE_DTYPE


def test_table_has_required_columns() -> None:
    t = compile_schema(_UserFeatures)
    assert t.dtype.names == (
        "name_hash",
        "byte_offset",
        "dtype_code",
        "element_count",
        "byte_count",
    )


def test_first_field_offset_is_header_size_aligned() -> None:
    # HEADER_SIZE=16 aligned up to 64 = 64.
    class _S(FeatureSchema):
        version = 1
        fields = [FeatureField("only", dtype.float32)]

    t = compile_schema(_S)
    assert int(t[0]["byte_offset"]) == 64


def test_two_scalars_second_field_offset_is_128() -> None:
    # First field at 64, byte_count=4, cursor=68 → next offset = align_up(68, 64) = 128.
    class _S(FeatureSchema):
        version = 1
        fields = [
            FeatureField("first", dtype.float32),
            FeatureField("second", dtype.float32),
        ]

    t = compile_schema(_S)
    first_row = t[t["name_hash"] == _hash_name("first")][0]
    second_row = t[t["name_hash"] == _hash_name("second")][0]
    assert int(first_row["byte_offset"]) == 64
    assert int(second_row["byte_offset"]) == 128


def test_all_offsets_multiples_of_64() -> None:
    t = compile_schema(_UserFeatures)
    for off in t["byte_offset"]:
        assert int(off) % CACHE_LINE_SIZE == 0


def test_table_is_sorted_by_name_hash() -> None:
    t = compile_schema(_UserFeatures)
    hashes = t["name_hash"]
    assert np.all(hashes[:-1] <= hashes[1:]), "offset table must be sorted by name_hash"


def test_byte_count_matches_element_count_times_dtype_size() -> None:
    t = compile_schema(_UserFeatures)
    sizes = {
        DType.FLOAT32: 4,
        DType.FLOAT64: 8,
        DType.INT32: 4,
        DType.INT64: 8,
        DType.UINT8: 1,
    }
    for row in t:
        code = DType(int(row["dtype_code"]))
        assert int(row["byte_count"]) == int(row["element_count"]) * sizes[code]


def test_embedding_field_has_correct_byte_count() -> None:
    t = compile_schema(_UserFeatures)
    row = t[t["name_hash"] == _hash_name("behavior_embedding")][0]
    assert int(row["byte_count"]) == 128 * 4
    assert int(row["element_count"]) == 128


def test_dtype_code_column_matches_enum_value() -> None:
    class _S(FeatureSchema):
        version = 1
        fields = [
            FeatureField("a", dtype.float32),
            FeatureField("b", dtype.int64),
        ]

    t = compile_schema(_S)
    a = t[t["name_hash"] == _hash_name("a")][0]
    b = t[t["name_hash"] == _hash_name("b")][0]
    assert int(a["dtype_code"]) == int(DType.FLOAT32)
    assert int(b["dtype_code"]) == int(DType.INT64)


def test_repeated_calls_return_equal_arrays() -> None:
    """Determinism — Step 2's segment CRC32 depends on this."""
    t1 = compile_schema(_UserFeatures)
    t2 = compile_schema(_UserFeatures)
    assert np.array_equal(t1, t2)
    assert t1.tobytes() == t2.tobytes()


def test_hash_uniqueness_enforced(monkeypatch: pytest.MonkeyPatch) -> None:
    """If two field names ever hashed to the same value, compile_schema must raise."""

    def collide(_name: str) -> int:
        return 42

    monkeypatch.setattr("quorin.schema._hash_name", collide)

    class _S(FeatureSchema):
        version = 1
        fields = [
            FeatureField("a", dtype.float32),
            FeatureField("b", dtype.float32),
        ]

    with pytest.raises(ValueError, match="collision"):
        compile_schema(_S)


# ---------------------------------------------------------------------------
# total_segment_size
# ---------------------------------------------------------------------------


def test_total_size_is_multiple_of_4096() -> None:
    assert total_segment_size(_UserFeatures) % PAGE_SIZE == 0


def test_total_size_covers_last_field() -> None:
    total = total_segment_size(_UserFeatures)
    t = compile_schema(_UserFeatures)
    max_end = int((t["byte_offset"] + t["byte_count"]).max())
    assert max_end <= total


def test_tiny_schema_rounds_up_to_one_page() -> None:
    class _Tiny(FeatureSchema):
        version = 1
        fields = [FeatureField("x", dtype.uint8)]

    assert total_segment_size(_Tiny) == PAGE_SIZE


def test_total_size_at_least_header_plus_field_bytes() -> None:
    total = total_segment_size(_UserFeatures)
    assert total >= HEADER_SIZE + sum(f.byte_count for f in _UserFeatures.fields)


# ---------------------------------------------------------------------------
# Hash pinning — catches silent algorithm drift
# ---------------------------------------------------------------------------


def test_hash_is_pinned() -> None:
    """If someone swaps blake2b for xxhash, this breaks loudly."""
    assert _hash_name("age_normalized") == 10066049608243894614
    assert _hash_name("session_count_7d") == 15435207408024440064
    assert _hash_name("ltv_score") == 14946179696997873924
    assert _hash_name("behavior_embedding") == 14736411744753480578


def test_unicode_name_hashed_as_utf8() -> None:
    expected = int.from_bytes(
        hashlib.blake2b("école".encode(), digest_size=8).digest(),
        "little",
    )
    assert _hash_name("école") == expected


# ---------------------------------------------------------------------------
# Module hygiene (acceptance criterion from spec)
# ---------------------------------------------------------------------------


def test_schema_module_has_no_heavy_imports() -> None:
    """Spec acceptance: no numba/redis/pyarrow/pydantic in quorin.schema."""
    forbidden = {"numba", "redis", "pyarrow", "pydantic"}
    src = inspect.getsource(_schema_module)
    for bad in forbidden:
        assert f"import {bad}" not in src, f"{bad} must not be imported in schema.py"
        assert f"from {bad}" not in src, f"{bad} must not be imported in schema.py"
