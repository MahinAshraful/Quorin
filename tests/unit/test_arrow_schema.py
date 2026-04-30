"""Unit tests for pyforge._internal.arrow_schema."""

from __future__ import annotations

import sys

import pytest

# Step 11's offline writer + arrow plan are POSIX-relevant only because
# the chaos / integration tests rely on /dev/shm and Linux-only fsync
# semantics. The pure plan module itself is OS-agnostic, but we keep
# the marker for consistency with sibling test files.
pytestmark = pytest.mark.skipif(
    sys.platform == "win32",
    reason="Step 11 tests target Linux/WSL2 only",
)

import pyarrow as pa  # noqa: E402

from pyforge._internal.arrow_schema import (  # noqa: E402
    _PLAN_CACHE,
    _arrow_plan_for,
    _arrow_type_for,
    clear_cache,
)
from pyforge.schema import (  # noqa: E402
    DType,
    FeatureField,
    FeatureSchema,
    _hash_name,
    dtype,
)

# ---------------------------------------------------------------------------
# Schemas used across the file. Class-level so they have stable identity
# (the plan cache key is class identity).
# ---------------------------------------------------------------------------


class _Scalars(FeatureSchema):
    version = 1
    fields = [
        FeatureField("a", dtype.float32),
        FeatureField("b", dtype.int64),
        FeatureField("c", dtype.uint8),
    ]


class _Embedding1D(FeatureSchema):
    version = 1
    fields = [FeatureField("emb", dtype.float32, shape=(128,))]


class _Embedding2D(FeatureSchema):
    version = 1
    fields = [FeatureField("mat", dtype.float64, shape=(2, 4))]


class _Mixed(FeatureSchema):
    version = 1
    fields = [
        FeatureField("score", dtype.float32),
        FeatureField("emb", dtype.float32, shape=(8,)),
        FeatureField("count", dtype.int32),
    ]


class _OutOfOrderHash(FeatureSchema):
    """Declaration order ≠ name_hash order. ``hash(zzz) > hash(aaa) > hash(mmm)``."""

    version = 1
    fields = [
        FeatureField("zzz", dtype.float32),
        FeatureField("aaa", dtype.float32),
        FeatureField("mmm", dtype.float32),
    ]


# ---------------------------------------------------------------------------
# Fixtures.
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clear_arrow_plan_cache() -> None:
    """Empty plan cache per-test so memoization assertions aren't poisoned."""
    clear_cache()


# ---------------------------------------------------------------------------
# _arrow_type_for: DType x shape coverage.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("d", "expected"),
    [
        (DType.FLOAT32, pa.float32()),
        (DType.FLOAT64, pa.float64()),
        (DType.INT32, pa.int32()),
        (DType.INT64, pa.int64()),
        (DType.UINT8, pa.uint8()),
    ],
)
def test_arrow_type_for_scalar_all_dtypes(d: DType, expected: pa.DataType) -> None:
    f = FeatureField("x", d)
    assert _arrow_type_for(f) == expected


@pytest.mark.parametrize(
    ("d", "elem"),
    [
        (DType.FLOAT32, pa.float32()),
        (DType.FLOAT64, pa.float64()),
        (DType.INT32, pa.int32()),
        (DType.INT64, pa.int64()),
        (DType.UINT8, pa.uint8()),
    ],
)
def test_arrow_type_for_1d_all_dtypes(d: DType, elem: pa.DataType) -> None:
    f = FeatureField("x", d, shape=(3,))
    assert _arrow_type_for(f) == pa.list_(elem, 3)


@pytest.mark.parametrize(
    ("d", "elem"),
    [
        (DType.FLOAT32, pa.float32()),
        (DType.FLOAT64, pa.float64()),
        (DType.INT32, pa.int32()),
        (DType.INT64, pa.int64()),
        (DType.UINT8, pa.uint8()),
    ],
)
def test_arrow_type_for_2d_all_dtypes(d: DType, elem: pa.DataType) -> None:
    f = FeatureField("x", d, shape=(2, 4))
    assert _arrow_type_for(f) == pa.list_(pa.list_(elem, 4), 2)


def test_arrow_type_for_3d_raises_with_field_name() -> None:
    f = FeatureField("tensor3d", DType.FLOAT32, shape=(2, 3, 4))
    with pytest.raises(ValueError, match="tensor3d"):
        _arrow_type_for(f)


# ---------------------------------------------------------------------------
# _arrow_plan_for — column order + permutation correctness.
# ---------------------------------------------------------------------------


def test_plan_column_order_default() -> None:
    plan = _arrow_plan_for(_Scalars, include_msg_id=True)
    assert plan.column_names == (
        "entity_id",
        "event_time_ns",
        "a",
        "b",
        "c",
        "msg_id_ms",
        "msg_id_seq",
    )


def test_plan_column_order_no_msg_id() -> None:
    plan = _arrow_plan_for(_Scalars, include_msg_id=False)
    assert plan.column_names == ("entity_id", "event_time_ns", "a", "b", "c")
    # The Arrow schema reflects it too.
    assert "msg_id_ms" not in plan.arrow_schema.names
    assert "msg_id_seq" not in plan.arrow_schema.names


def test_arrow_schema_field_types() -> None:
    plan = _arrow_plan_for(_Mixed, include_msg_id=True)
    schema = plan.arrow_schema
    assert schema.field("entity_id").type == pa.string()
    assert schema.field("event_time_ns").type == pa.int64()
    assert schema.field("score").type == pa.float32()
    assert schema.field("emb").type == pa.list_(pa.float32(), 8)
    assert schema.field("count").type == pa.int32()
    assert schema.field("msg_id_ms").type == pa.int64()
    assert schema.field("msg_id_seq").type == pa.int32()


def test_wire_to_decl_permutation_round_trips_out_of_order_hash() -> None:
    """Producer wire order is name_hash-sorted (ADR-008). The
    `wire_to_decl` permutation must map a wire-ordered values list back
    to the declaration-order columns.
    """
    plan = _arrow_plan_for(_OutOfOrderHash, include_msg_id=False)
    # Values are name_hash-sorted at the producer. Build the wire
    # order explicitly here; assert the permutation puts each value
    # back in declaration order.
    decl_fields = list(_OutOfOrderHash.fields)
    name_to_value = {f.name: float(i) for i, f in enumerate(decl_fields)}
    wire_order = sorted(decl_fields, key=lambda f: _hash_name(f.name))
    wire_values = [name_to_value[f.name] for f in wire_order]

    # Apply wire_to_decl manually to reconstruct decl-ordered values.
    decl_values = [0.0] * len(decl_fields)
    for wire_idx, decl_col_idx in enumerate(plan.wire_to_decl):
        # decl_col_idx is the column index in plan.column_names; subtract
        # the leading 2 (entity_id + event_time_ns) to get the
        # decl_fields index.
        decl_values[decl_col_idx - 2] = wire_values[wire_idx]

    expected = [name_to_value[f.name] for f in decl_fields]
    assert decl_values == expected


def test_wire_to_decl_length_matches_field_count() -> None:
    plan = _arrow_plan_for(_Mixed, include_msg_id=True)
    assert len(plan.wire_to_decl) == len(_Mixed.fields)


# ---------------------------------------------------------------------------
# Caching.
# ---------------------------------------------------------------------------


def test_plan_is_memoized_by_class_identity() -> None:
    p1 = _arrow_plan_for(_Scalars, include_msg_id=True)
    p2 = _arrow_plan_for(_Scalars, include_msg_id=True)
    assert p1 is p2


def test_include_msg_id_true_and_false_are_independently_cached() -> None:
    p_with = _arrow_plan_for(_Scalars, include_msg_id=True)
    p_without = _arrow_plan_for(_Scalars, include_msg_id=False)
    assert p_with is not p_without
    assert (_Scalars, True) in _PLAN_CACHE
    assert (_Scalars, False) in _PLAN_CACHE


@pytest.mark.parametrize(
    "name", ["entity_id", "event_time_ns", "msg_id_ms", "msg_id_seq", "event_date"]
)
def test_plan_rejects_schema_with_reserved_field_name(name: str) -> None:
    cls = type(
        f"_Reserved_{name}",
        (FeatureSchema,),
        {"version": 1, "fields": [FeatureField(name, dtype.float32)]},
    )
    with pytest.raises(ValueError, match="reserved"):
        _arrow_plan_for(cls, include_msg_id=True)


def test_clear_cache_drops_entries() -> None:
    _arrow_plan_for(_Scalars, include_msg_id=True)
    _arrow_plan_for(_Scalars, include_msg_id=False)
    assert len(_PLAN_CACHE) >= 2
    clear_cache()
    assert len(_PLAN_CACHE) == 0


# ---------------------------------------------------------------------------
# Embedding round-trips through the plan -> Table -> plan path.
# Sanity check that pa.list_(elem, N) with from_pydict accepts plain
# Python lists.
# ---------------------------------------------------------------------------


def test_plan_accepts_1d_embedding_via_from_pydict() -> None:
    plan = _arrow_plan_for(_Embedding1D, include_msg_id=False)
    cols: dict[str, list[object]] = {name: [] for name in plan.column_names}
    cols["entity_id"].append("ent-0")
    cols["event_time_ns"].append(0)
    cols["emb"].append([float(i) for i in range(128)])
    table = pa.Table.from_pydict(cols, schema=plan.arrow_schema)
    assert table.num_rows == 1
    assert table.column("emb")[0].as_py() == [float(i) for i in range(128)]


def test_plan_accepts_2d_embedding_via_from_pydict() -> None:
    plan = _arrow_plan_for(_Embedding2D, include_msg_id=False)
    cols: dict[str, list[object]] = {name: [] for name in plan.column_names}
    cols["entity_id"].append("ent-0")
    cols["event_time_ns"].append(0)
    cols["mat"].append([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]])
    table = pa.Table.from_pydict(cols, schema=plan.arrow_schema)
    assert table.num_rows == 1
    assert table.column("mat")[0].as_py() == [
        [1.0, 2.0, 3.0, 4.0],
        [5.0, 6.0, 7.0, 8.0],
    ]
