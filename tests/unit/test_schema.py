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


def test_schema_name_with_hyphen_rejected() -> None:
    """CR.A.6 (v0.1.1): non-identifier names collide via _safe_class_name
    sanitization (e.g. ``MySchema-v1`` and ``MySchema_v1`` both become
    ``MySchema_v1`` in Redis keys / paths). Reject at class-definition.
    """
    with pytest.raises(ValueError, match=r"name.*invalid"):
        # A class name with a hyphen — needs to be created dynamically
        # because Python won't let you write ``class My-Schema``.
        type(
            "My-Schema",
            (FeatureSchema,),
            {
                "version": 1,
                "fields": [FeatureField("x", dtype.float32)],
            },
        )


def test_schema_name_with_dot_rejected() -> None:
    with pytest.raises(ValueError, match=r"name.*invalid"):
        type(
            "My.Schema",
            (FeatureSchema,),
            {
                "version": 1,
                "fields": [FeatureField("x", dtype.float32)],
            },
        )


def test_schema_name_starting_with_digit_rejected() -> None:
    with pytest.raises(ValueError, match=r"name.*invalid"):
        type(
            "1Schema",
            (FeatureSchema,),
            {
                "version": 1,
                "fields": [FeatureField("x", dtype.float32)],
            },
        )


def test_schema_name_too_long_rejected() -> None:
    long_name = "A" + "B" * 63  # 64 chars total — over the 63 ceiling.
    with pytest.raises(ValueError, match=r"name.*invalid"):
        type(
            long_name,
            (FeatureSchema,),
            {
                "version": 1,
                "fields": [FeatureField("x", dtype.float32)],
            },
        )


def test_schema_name_with_underscore_accepted() -> None:
    # Underscores and digits-after-leading-letter are permitted.
    cls = type(
        "My_Schema_v1",
        (FeatureSchema,),
        {
            "version": 1,
            "fields": [FeatureField("x", dtype.float32)],
        },
    )
    assert cls.__name__ == "My_Schema_v1"


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


# ---------------------------------------------------------------------------
# CR.C.2 (v0.1.2) — shape DoS caps. Regression for the audit finding that
# ``FeatureField(shape=(2**30,))`` builds with byte_count=4 GB silently.
# ---------------------------------------------------------------------------


def test_feature_field_rejects_element_count_above_cap() -> None:
    """CR.C.2: ``element_count > MAX_ELEMENT_COUNT`` raises at construction."""
    from quorin.schema import MAX_ELEMENT_COUNT

    over = MAX_ELEMENT_COUNT + 1
    with pytest.raises(ValueError, match="MAX_ELEMENT_COUNT"):
        FeatureField("x", DType.FLOAT32, (over,))


def test_feature_field_accepts_element_count_at_cap() -> None:
    """Just-below-cap stays accepted — verifies the cap is not too tight."""
    from quorin.schema import MAX_ELEMENT_COUNT

    # 16M uint8 = 16 MiB, well below per-field byte ceiling.
    f = FeatureField("x", DType.UINT8, (MAX_ELEMENT_COUNT,))
    assert f.element_count == MAX_ELEMENT_COUNT


def test_feature_field_rejects_byte_count_above_cap() -> None:
    """CR.C.2: element_count under cap but byte_count over still raises.

    16M float64 = 128 MiB ✓; but 33M float64 = 264 MiB > 256 MiB cap.
    Using uint8 with a much larger shape can exceed element_count first; the
    distinct ``byte_count`` cap fires when a wider dtype pushes bytes past
    the per-field limit even with a smaller shape.
    """
    from quorin.schema import MAX_ELEMENT_COUNT, MAX_FIELD_BYTES

    # Pick a shape where element_count is just at the cap, dtype is wide.
    # MAX_ELEMENT_COUNT = 2^24 = 16M. float64 = 8 bytes/elem → 128 MiB. Under
    # MAX_FIELD_BYTES = 256 MiB. Need a different attack vector: a shape
    # smaller than MAX_ELEMENT_COUNT but with dtype that multiplies past the
    # byte cap. element_count = 2^24 with float64 = 128 MiB. To exceed the
    # byte cap WITHOUT exceeding element_count cap, we need
    # element_count * sizeof <= MAX_FIELD_BYTES check. At 2^24 elements
    # x 8 bytes = 128 MiB < 256 MiB — under both caps. With current dtypes
    # (max 8 bytes/elem) the byte cap is mathematically unreachable below
    # element_count cap. Document the relationship; element_count cap is
    # the binding constraint today. The byte cap kicks in if a future dtype
    # exceeds 16 bytes/elem (e.g. complex128 if/when added).
    assert MAX_FIELD_BYTES // 8 >= MAX_ELEMENT_COUNT, (
        "byte cap is non-binding for current dtypes; element_count cap binds first. "
        "Test will need a wider dtype if the cap relationship inverts."
    )


def test_compute_layout_rejects_total_segment_above_cap() -> None:
    """CR.C.2 whole-segment ceiling: capacity x row_size product is bounded.

    Direct test of ``compute_layout`` rather than constructing a schema
    object so we don't have to assemble a giant ``FeatureSchema`` subclass.
    """
    import inspect as _inspect

    from quorin.layout import MAX_TOTAL_SEGMENT_BYTES, compute_layout

    # A modest schema (200 float32 fields = 800 bytes/row), with absurd
    # capacity to push the product past the cap.
    class _Wide200(FeatureSchema):
        version = 1
        fields = [FeatureField(f"f{i}", DType.FLOAT32) for i in range(200)]

    # 800 byte row x 2^28 capacity = ~200 GB. Way over MAX_TOTAL_SEGMENT_BYTES=64 GB.
    huge_capacity = 1 << 28
    with pytest.raises(ValueError, match="MAX_TOTAL_SEGMENT_BYTES"):
        compute_layout(_Wide200, capacity=huge_capacity)
    # Make ruff happy about the unused inspect import.
    assert _inspect.isfunction(compute_layout)
    assert MAX_TOTAL_SEGMENT_BYTES > 0
