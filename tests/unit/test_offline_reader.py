"""Unit tests for ParquetDatasetStore.read_point_in_time (Step 12).

Covers the ~30 tests called out in
``progress/step12_plan.md``. Property tests live in
``tests/property/test_offline_pit_invariants.py``; the integration
end-to-end is in ``tests/integration/test_offline_e2e.py``.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

pytestmark = pytest.mark.skipif(
    sys.platform == "win32",
    reason="Step 11/12 fsync semantics are POSIX-only",
)

import pyarrow as pa  # noqa: E402

from pyforge._internal.arrow_schema import (  # noqa: E402
    clear_cache as clear_arrow_cache,
)
from pyforge.offline import ParquetDatasetStore  # noqa: E402
from pyforge.schema import (  # noqa: E402
    FeatureField,
    FeatureSchema,
    dtype,
)

_DAY_NS = 86_400_000_000_000


# ---------------------------------------------------------------------------
# Schemas. Class-level so identity is stable across tests / cache hits.
# ---------------------------------------------------------------------------


class _S(FeatureSchema):
    version = 1
    fields = [
        FeatureField("x", dtype.float32),
        FeatureField("y", dtype.int64),
    ]


class _SOnlyX(FeatureSchema):
    version = 1
    fields = [FeatureField("x", dtype.float32)]


class _SDivergent(FeatureSchema):
    version = 1
    fields = [
        FeatureField("x", dtype.float32),
        FeatureField("z", dtype.int32),  # not present in dataset written for _S
    ]


class _SColliding(FeatureSchema):
    """Schema whose field name 'extra' is intended for collision tests."""

    version = 1
    fields = [FeatureField("extra", dtype.float32)]


# ---------------------------------------------------------------------------
# Fixtures.
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clear_arrow_plan_cache() -> None:
    clear_arrow_cache()


def _run(coro: Any) -> Any:
    return asyncio.run(coro)


def _populate(
    store: ParquetDatasetStore,
    schema: type[FeatureSchema],
    rows: list[tuple[str, int, list[Any], bytes]],
) -> None:
    """Append rows + flush in one call. ``rows = [(eid, et_ns, values, msg_id), ...]``."""

    async def _go() -> None:
        for eid, et, values, msg_id in rows:
            await store.append(schema, eid, et, values, msg_id)
        await store.flush()

    _run(_go())


def _q(pairs: list[tuple[str, int]], **extra: Any) -> pa.Table:
    """Build a query table with optional extra columns."""
    cols: dict[str, list[Any]] = {
        "entity_id": [p[0] for p in pairs],
        "as_of_time": [p[1] for p in pairs],
    }
    cols.update({k: list(v) for k, v in extra.items()})
    return pa.table(cols)


# ---------------------------------------------------------------------------
# 1. Basic point-in-time correctness.
# ---------------------------------------------------------------------------


def test_basic_pit_lookup(tmp_path: Path) -> None:
    """Test 1: features at t={100,110,120} for A; query (A,115) → t=110."""
    store = ParquetDatasetStore(tmp_path)
    _populate(
        store,
        _S,
        [
            ("A", 100, [1.0, 100], b"1-0"),
            ("A", 110, [2.0, 200], b"2-0"),
            ("A", 120, [3.0, 300], b"3-0"),
        ],
    )
    result = store.read_point_in_time(_S, _q([("A", 115)]))
    d = result.to_pydict()
    assert d["event_time_ns"] == [110]
    assert d["x"] == [pytest.approx(2.0)]
    assert d["y"] == [200]


def test_query_before_first_event_returns_null(tmp_path: Path) -> None:
    """Test 2: features at t≥200; query (A,100) → feature cols null."""
    store = ParquetDatasetStore(tmp_path)
    _populate(store, _S, [("A", 200, [1.0, 100], b"1-0")])
    result = store.read_point_in_time(_S, _q([("A", 100)]))
    d = result.to_pydict()
    assert d["event_time_ns"] == [None]
    assert d["x"] == [None]
    assert d["y"] == [None]


def test_leakage_regression_no_future_features(tmp_path: Path) -> None:
    """Test 3: features at t=200; query (A,100); event_time_ns is null (no leak)."""
    store = ParquetDatasetStore(tmp_path)
    _populate(store, _S, [("A", 200, [42.0, 999], b"1-0")])
    result = store.read_point_in_time(_S, _q([("A", 100)]))
    d = result.to_pydict()
    # The defining property: NO future leak. event_time_ns is null,
    # and the feature columns are also null.
    assert d["event_time_ns"] == [None]
    assert d["x"] == [None]


# ---------------------------------------------------------------------------
# 2. Dedup variants.
# ---------------------------------------------------------------------------


def test_dedup_msg_id_keeps_one_per_redis_message(tmp_path: Path) -> None:
    """Test 4: same msg_id twice (PEL replay sim) → 1 row in result."""
    store = ParquetDatasetStore(tmp_path)
    # PEL replay: producer wrote the same payload twice with same msg_id.
    _populate(
        store,
        _S,
        [
            ("A", 100, [1.0, 100], b"1-0"),
            ("A", 100, [1.0, 100], b"1-0"),  # duplicate (crash replay)
        ],
    )
    result = store.read_point_in_time(_S, _q([("A", 105)]))
    d = result.to_pydict()
    # Single matched row; dedup happened pre-asof.
    assert d["x"] == [pytest.approx(1.0)]
    assert len(d["x"]) == 1


def test_dedup_msg_id_distinct_keeps_distinct(tmp_path: Path) -> None:
    """Test 5: different msg_ids same entity → both kept (asof picks latest)."""
    store = ParquetDatasetStore(tmp_path)
    _populate(
        store,
        _S,
        [
            ("A", 100, [1.0, 100], b"1-0"),
            ("A", 150, [2.0, 200], b"2-0"),
        ],
    )
    result = store.read_point_in_time(_S, _q([("A", 200)]))
    d = result.to_pydict()
    # The latest before 200 is at t=150.
    assert d["x"] == [pytest.approx(2.0)]
    assert d["event_time_ns"] == [150]


def test_natural_key_dedup_when_no_msg_id(tmp_path: Path) -> None:
    """Test 6: include_msg_id=False, two same (eid, et_ns) → 1 row, last-row-wins."""
    store = ParquetDatasetStore(tmp_path, include_msg_id=False)
    _populate(
        store,
        _S,
        [
            ("A", 100, [1.0, 100], b"1-0"),
            # Same eid+et but different feature values — this is the
            # "silent conflation" the include_msg_id=False docs warn
            # about. Last-row-wins.
            ("A", 100, [2.0, 200], b"2-0"),
        ],
    )
    result = store.read_point_in_time(_S, _q([("A", 105)]))
    d = result.to_pydict()
    assert "msg_id_ms" not in d
    # Exactly one row (no double-count); value is one of the two written.
    assert len(d["x"]) == 1
    assert d["x"][0] in (pytest.approx(1.0), pytest.approx(2.0))


def test_mixed_msg_id_and_no_msg_id_files_raises(tmp_path: Path) -> None:
    """Test 7: manually mismatched files → clearer ValueError."""
    # Write a file WITH msg_id columns via the writer.
    store_with = ParquetDatasetStore(tmp_path, include_msg_id=True)
    _populate(store_with, _S, [("A", 100, [1.0, 100], b"1-0")])
    # Then write a file WITHOUT msg_id columns into the same dataset.
    store_without = ParquetDatasetStore(tmp_path, include_msg_id=False)
    _populate(store_without, _S, [("B", 200, [2.0, 200], b"2-0")])

    reader = ParquetDatasetStore(tmp_path)  # mode irrelevant for read
    with pytest.raises(ValueError, match="mixed include_msg_id state"):
        reader.read_point_in_time(_S, _q([("A", 150)]))


# ---------------------------------------------------------------------------
# 3. Empty / boundary cases.
# ---------------------------------------------------------------------------


def test_zero_files_dataset_returns_null_rows(tmp_path: Path) -> None:
    """Test 8: missing schema=foo/ dir → one null-feature row per query row."""
    store = ParquetDatasetStore(tmp_path)
    result = store.read_point_in_time(_S, _q([("A", 100), ("B", 200)]))
    # Per §J contract: len(result) == len(query); feature cols null.
    assert len(result) == 2
    d = result.to_pydict()
    assert d["entity_id"] == ["A", "B"]
    assert d["as_of_time"] == [100, 200]
    assert d["event_time_ns"] == [None, None]
    assert d["x"] == [None, None]
    assert d["msg_id_ms"] == [None, None]


def test_schema_dir_exists_no_partitions_returns_null_rows(tmp_path: Path) -> None:
    """Test 9: writer constructed, never flushed → null-feature rows aligned with query."""
    store = ParquetDatasetStore(tmp_path)
    # Create the schema dir but no partition subdirs (simulates writer
    # never flushed any rows).
    (tmp_path / f"schema={_S.__name__}").mkdir(parents=True, exist_ok=True)
    result = store.read_point_in_time(_S, _q([("A", 100)]))
    assert len(result) == 1
    d = result.to_pydict()
    assert d["event_time_ns"] == [None]
    assert d["x"] == [None]


def test_empty_query_table(tmp_path: Path) -> None:
    """Test 10: empty query → empty result (all schema columns present, 0 rows)."""
    store = ParquetDatasetStore(tmp_path)
    _populate(store, _S, [("A", 100, [1.0, 100], b"1-0")])
    empty_query = pa.table(
        {"entity_id": pa.array([], type=pa.string()), "as_of_time": pa.array([], type=pa.int64())}
    )
    result = store.read_point_in_time(_S, empty_query)
    assert len(result) == 0
    # Schema still has all columns; type-stable empty.
    assert "x" in result.column_names
    assert "msg_id_ms" in result.column_names


# ---------------------------------------------------------------------------
# 4. Partition pruning.
# ---------------------------------------------------------------------------


def test_partition_pruning_only_reads_window(tmp_path: Path) -> None:
    """Test 11: filter expression contains the event_date partition predicate."""
    store = ParquetDatasetStore(tmp_path)
    _populate(store, _S, [("A", 100, [1.0, 100], b"1-0")])

    # PyArrow Dataset objects have read-only ``to_table``; wrap the
    # dataset in a thin proxy that captures the filter then delegates.
    captured: dict[str, Any] = {}
    real_open = ParquetDatasetStore._open_dataset

    class _CapturingDataset:
        def __init__(self, inner: Any) -> None:
            self._inner = inner

        def __getattr__(self, name: str) -> Any:
            return getattr(self._inner, name)

        def to_table(self, *args: Any, **kwargs: Any) -> Any:
            captured["filter"] = kwargs.get("filter")
            return self._inner.to_table(*args, **kwargs)

    def wrapped_open(self_: ParquetDatasetStore, sch: type[FeatureSchema]) -> Any:
        inner = real_open(self_, sch)
        if inner is None:
            return None
        return _CapturingDataset(inner)

    with patch.object(ParquetDatasetStore, "_open_dataset", wrapped_open):
        store.read_point_in_time(_S, _q([("A", 100)]), lookback_days=30)

    expr_repr = str(captured["filter"])
    assert "event_date" in expr_repr
    assert "event_time_ns" in expr_repr


# ---------------------------------------------------------------------------
# 5. Argument validation.
# ---------------------------------------------------------------------------


def test_lookback_zero_raises(tmp_path: Path) -> None:
    """Test 12: lookback_days=0 → ValueError."""
    store = ParquetDatasetStore(tmp_path)
    with pytest.raises(ValueError, match="lookback_days must be positive"):
        store.read_point_in_time(_S, _q([("A", 100)]), lookback_days=0)


def test_lookback_negative_raises(tmp_path: Path) -> None:
    """Test 13: lookback_days=-1 → ValueError."""
    store = ParquetDatasetStore(tmp_path)
    with pytest.raises(ValueError, match="lookback_days must be positive"):
        store.read_point_in_time(_S, _q([("A", 100)]), lookback_days=-1)


def test_query_table_missing_columns_raises(tmp_path: Path) -> None:
    """Test 14: query without entity_id or as_of_time → ValueError."""
    store = ParquetDatasetStore(tmp_path)
    bad = pa.table({"foo": ["A"], "bar": [100]})
    with pytest.raises(ValueError, match="missing required columns"):
        store.read_point_in_time(_S, bad)


def test_query_table_wrong_eid_type_raises(tmp_path: Path) -> None:
    """Bonus: entity_id as int → ValueError."""
    store = ParquetDatasetStore(tmp_path)
    bad = pa.table({"entity_id": [1, 2], "as_of_time": [100, 200]})
    with pytest.raises(ValueError, match=r"entity_id must be pa\.string"):
        store.read_point_in_time(_S, bad)


def test_query_table_wrong_aot_type_raises(tmp_path: Path) -> None:
    """Bonus: as_of_time as int32 → ValueError."""
    store = ParquetDatasetStore(tmp_path)
    bad = pa.table(
        {
            "entity_id": pa.array(["A"], type=pa.string()),
            "as_of_time": pa.array([100], type=pa.int32()),
        }
    )
    with pytest.raises(ValueError, match=r"as_of_time must be pa\.int64"):
        store.read_point_in_time(_S, bad)


# ---------------------------------------------------------------------------
# 6. Far past / future / duplicates / scale.
# ---------------------------------------------------------------------------


def test_far_future_as_of_time(tmp_path: Path) -> None:
    """Test 15: as_of_time = 2**62; latest within lookback returned."""
    store = ParquetDatasetStore(tmp_path)
    far_future = 2**62
    # Feature event_time within 1 day of the far_future as_of_time so
    # it falls inside the 30-day lookback window.
    feat_t = far_future - _DAY_NS
    _populate(store, _S, [("A", feat_t, [42.0, 7], b"1-0")])
    result = store.read_point_in_time(_S, _q([("A", far_future)]))
    d = result.to_pydict()
    assert d["x"] == [pytest.approx(42.0)]


def test_far_past_as_of_time_returns_null(tmp_path: Path) -> None:
    """Test 16: as_of_time = 0; features in present → null."""
    store = ParquetDatasetStore(tmp_path)
    _populate(store, _S, [("A", 1_700_000_000_000_000_000, [1.0, 1], b"1-0")])
    result = store.read_point_in_time(_S, _q([("A", 0)]))
    d = result.to_pydict()
    assert d["x"] == [None]


def test_duplicate_query_pairs_returned(tmp_path: Path) -> None:
    """Test 17: query [(A,115),(A,115)] → both rows present, both correct."""
    store = ParquetDatasetStore(tmp_path)
    _populate(store, _S, [("A", 110, [2.0, 200], b"1-0")])
    result = store.read_point_in_time(_S, _q([("A", 115), ("A", 115)]))
    d = result.to_pydict()
    assert d["x"] == [pytest.approx(2.0), pytest.approx(2.0)]
    assert d["event_time_ns"] == [110, 110]


def test_single_entity_in_dataset_with_many_others(tmp_path: Path) -> None:
    """Test 18: 1k entities, query 1 → correct row, no perf cliff."""
    store = ParquetDatasetStore(tmp_path)
    rows = [(f"E{i:04d}", 100 + i, [float(i), i], f"{i}-0".encode()) for i in range(1000)]
    _populate(store, _S, rows)
    result = store.read_point_in_time(_S, _q([("E0500", 1000)]))
    d = result.to_pydict()
    assert d["x"] == [pytest.approx(500.0)]
    assert d["y"] == [500]


def test_partition_with_zero_rows_after_filter(tmp_path: Path) -> None:
    """Test 19: filter excludes all rows in a partition; no error."""
    store = ParquetDatasetStore(tmp_path)
    # Two days of writes; query window covers only one of them.
    _populate(
        store,
        _S,
        [
            ("A", _DAY_NS * 100, [1.0, 1], b"1-0"),
            ("A", _DAY_NS * 200, [2.0, 2], b"2-0"),
        ],
    )
    # Query at t=DAY*150; lookback=30 → both partitions outside (one
    # 50 days before, one 50 days after), so result is null.
    result = store.read_point_in_time(_S, _q([("A", _DAY_NS * 150)]), lookback_days=30)
    d = result.to_pydict()
    assert d["x"] == [None]


# ---------------------------------------------------------------------------
# 7. Result schema lock.
# ---------------------------------------------------------------------------


def test_result_row_count_equals_query_row_count(tmp_path: Path) -> None:
    """Test 20: len(result) == len(query_table) for any query."""
    store = ParquetDatasetStore(tmp_path)
    _populate(store, _S, [("A", 100, [1.0, 1], b"1-0")])
    for n in (1, 5, 50):
        query = _q([("A", 200)] * n)
        result = store.read_point_in_time(_S, query)
        assert len(result) == n, f"n={n}: expected {n} rows, got {len(result)}"


def test_result_column_schema_exact(tmp_path: Path) -> None:
    """Test 21: lock the §J column tuple, including extra query columns."""
    store = ParquetDatasetStore(tmp_path)
    _populate(store, _S, [("A", 100, [1.0, 1], b"1-0")])
    query = _q([("A", 200)], label=[1.0], fold=[3])
    result = store.read_point_in_time(_S, query)
    expected = (
        "entity_id",
        "as_of_time",
        "label",
        "fold",
        "event_time_ns",
        "x",
        "y",
        "msg_id_ms",
        "msg_id_seq",
    )
    assert tuple(result.column_names) == expected


def test_result_column_schema_no_msg_id(tmp_path: Path) -> None:
    """Test 21 partner: include_msg_id=False omits the msg_id_* columns."""
    store = ParquetDatasetStore(tmp_path, include_msg_id=False)
    _populate(store, _S, [("A", 100, [1.0, 1], b"1-0")])
    result = store.read_point_in_time(_S, _q([("A", 200)]))
    expected = (
        "entity_id",
        "as_of_time",
        "event_time_ns",
        "x",
        "y",
    )
    assert tuple(result.column_names) == expected


# ---------------------------------------------------------------------------
# 8. asof boundary semantics.
# ---------------------------------------------------------------------------


def test_asof_inclusive_at_upper_boundary(tmp_path: Path) -> None:
    """Test 22: feature t=100; query as_of_time=100 → match (inclusive)."""
    store = ParquetDatasetStore(tmp_path)
    _populate(store, _S, [("A", 100, [42.0, 7], b"1-0")])
    result = store.read_point_in_time(_S, _q([("A", 100)]))
    d = result.to_pydict()
    assert d["event_time_ns"] == [100]
    assert d["x"] == [pytest.approx(42.0)]


def test_asof_exclusive_above_boundary(tmp_path: Path) -> None:
    """Test 23: feature t=100; query as_of_time=99 → null."""
    store = ParquetDatasetStore(tmp_path)
    _populate(store, _S, [("A", 100, [42.0, 7], b"1-0")])
    result = store.read_point_in_time(_S, _q([("A", 99)]))
    d = result.to_pydict()
    assert d["event_time_ns"] == [None]


# ---------------------------------------------------------------------------
# 9. Per-query lookback (single + multi-query regression).
# ---------------------------------------------------------------------------


def test_lookback_excludes_old_features_single_query(tmp_path: Path) -> None:
    """Test 24: feature t = aot - 31d; lookback=30 → null."""
    store = ParquetDatasetStore(tmp_path)
    aot = _DAY_NS * 1000
    feat_t = aot - 31 * _DAY_NS  # 31 days before
    _populate(store, _S, [("A", feat_t, [42.0, 7], b"1-0")])
    result = store.read_point_in_time(_S, _q([("A", aot)]), lookback_days=30)
    d = result.to_pydict()
    assert d["event_time_ns"] == [None]


def test_lookback_includes_features_at_window_edge(tmp_path: Path) -> None:
    """Test 25: feature t = aot - 30d exactly; lookback=30 → match (inclusive)."""
    store = ParquetDatasetStore(tmp_path)
    aot = _DAY_NS * 1000
    feat_t = aot - 30 * _DAY_NS  # exactly at the lower edge
    _populate(store, _S, [("A", feat_t, [42.0, 7], b"1-0")])
    result = store.read_point_in_time(_S, _q([("A", aot)]), lookback_days=30)
    d = result.to_pydict()
    assert d["event_time_ns"] == [feat_t]
    assert d["x"] == [pytest.approx(42.0)]


def test_per_query_lookback_multi_query_regression(tmp_path: Path) -> None:
    """Test 26 (Rev-3 CRITICAL-1): asof primitive enforces per-query lookback.

    Without the inline check inside _asof_join, the global row filter
    (using min(as_of_time) - lookback) loads the feature at 1e9, then
    searchsorted would naively match it to BOTH queries. The fix
    enforces per-query lookback and rejects the second match.

    Asserts BOTH directions per Q1 answer:
      - first query (within 30d) gets event_time_ns=1e9 (no false-null)
      - second query (way past lookback) gets null (no false-match)
    """
    store = ParquetDatasetStore(tmp_path)
    feat_t = 1_000_000_000  # ~1 s into epoch
    _populate(store, _S, [("A", feat_t, [42.0, 7], b"1-0")])
    near = 1_000_000_000_000_000  # ~11.5 days into epoch — within 30d
    far = 5_000_000_000_000_000  # ~57.9 days into epoch — past 30d
    result = store.read_point_in_time(_S, _q([("A", near), ("A", far)]), lookback_days=30)
    d = result.to_pydict()
    assert d["event_time_ns"][0] == feat_t, "first should match (no false-null)"
    assert d["x"][0] == pytest.approx(42.0)
    assert d["event_time_ns"][1] is None, "second should be null (no false-match — THE bug)"
    assert d["x"][1] is None


# ---------------------------------------------------------------------------
# 10. Result shape rules: extra query columns, schema divergence, collisions.
# ---------------------------------------------------------------------------


def test_query_extra_columns_preserved(tmp_path: Path) -> None:
    """Test 27 (Rev-3 HIGH-4): label/fold round-trip in result, in input order."""
    store = ParquetDatasetStore(tmp_path)
    _populate(store, _S, [("A", 100, [1.0, 1], b"1-0")])
    query = _q([("A", 200)], label=[42.5], fold=[3])
    result = store.read_point_in_time(_S, query)
    d = result.to_pydict()
    assert d["label"] == [pytest.approx(42.5)]
    assert d["fold"] == [3]
    # Order: query columns first, in input order
    assert result.column_names[:4] == ["entity_id", "as_of_time", "label", "fold"]


def test_schema_divergence_raises(tmp_path: Path) -> None:
    """Test 28 (Rev-3 HIGH-6): schema declares field absent in dataset → ValueError.

    Builds two same-named classes (``__name__ == "_SVer1"``) so they
    share a partition directory: the v1 producer writes ``[x]``, then
    the test queries with v2 declaring ``[x, z]`` — ``z`` is absent
    from the on-disk dataset, divergence guard fires. Same-named
    classes is the only way to hit the divergence path without writing
    raw Parquet — the partition path is keyed off ``schema.__name__``.
    """
    s_ver1 = type(
        "_SVer1",
        (FeatureSchema,),
        {"version": 1, "fields": [FeatureField("x", dtype.float32)]},
    )
    s_ver2 = type(
        "_SVer1",  # same __name__ -> same hive partition dir
        (FeatureSchema,),
        {
            "version": 2,
            "fields": [
                FeatureField("x", dtype.float32),
                FeatureField("z", dtype.int32),  # absent in v1 dataset
            ],
        },
    )
    store = ParquetDatasetStore(tmp_path)
    _populate(store, s_ver1, [("A", 100, [1.0], b"1-0")])
    with pytest.raises(ValueError, match="Step 15"):
        store.read_point_in_time(s_ver2, _q([("A", 200)]))


def test_query_column_collision_raises(tmp_path: Path) -> None:
    """Test 29 (Rev-3 HIGH-4): query has 'event_time_ns' column → ValueError."""
    store = ParquetDatasetStore(tmp_path)
    _populate(store, _S, [("A", 100, [1.0, 1], b"1-0")])
    bad = pa.table(
        {
            "entity_id": pa.array(["A"], type=pa.string()),
            "as_of_time": pa.array([200], type=pa.int64()),
            "event_time_ns": pa.array([0], type=pa.int64()),  # collides
        }
    )
    with pytest.raises(ValueError, match="collide with result feature columns"):
        store.read_point_in_time(_S, bad)


def test_query_column_collision_with_schema_field_raises(tmp_path: Path) -> None:
    """Test 29 partner: query column matching a schema field name → ValueError."""
    store = ParquetDatasetStore(tmp_path)
    _populate(store, _S, [("A", 100, [1.0, 1], b"1-0")])
    bad = pa.table(
        {
            "entity_id": pa.array(["A"], type=pa.string()),
            "as_of_time": pa.array([200], type=pa.int64()),
            "x": pa.array([99.0], type=pa.float32()),  # collides with schema field
        }
    )
    with pytest.raises(ValueError, match="collide with result feature columns"):
        store.read_point_in_time(_S, bad)


# ---------------------------------------------------------------------------
# 11. Deterministic tiebreak (Rev-3 HIGH-5).
# ---------------------------------------------------------------------------


def test_deterministic_tiebreak_on_distinct_msg_ids_same_eid_event_time(
    tmp_path: Path,
) -> None:
    """Test 30 (Rev-3 HIGH-5): two writes same (eid, et_ns) distinct msg_ids.

    Backfill case (ADR-010 §1) — same event_time_ns is *legal*. The
    msg_id-extended sort in _asof_join makes the asof pick the largest
    msg_id deterministically across multiple reads.
    """
    store = ParquetDatasetStore(tmp_path)
    _populate(
        store,
        _S,
        [
            ("A", 100, [1.0, 100], b"1-0"),
            ("A", 100, [2.0, 200], b"2-0"),  # same et, distinct msg_id
        ],
    )
    # Two reads should produce identical results.
    r1 = store.read_point_in_time(_S, _q([("A", 200)])).to_pydict()
    r2 = store.read_point_in_time(_S, _q([("A", 200)])).to_pydict()
    assert r1 == r2, "two reads must be deterministic"
    # The asof picks the largest msg_id (2-0), which has x=2.0.
    assert r1["msg_id_ms"] == [2]
    assert r1["x"] == [pytest.approx(2.0)]
