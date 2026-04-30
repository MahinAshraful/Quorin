"""Benchmarks for pyforge.offline.ParquetDatasetStore (Step 11).

Five benches, decomposed so a regression in any sub-component shows up
in isolation:

- ``append_per_msg_200_field``  — locks the dict-free hot path
  (Rev-3 #1's bug class). Gate at 20 µs per Rev-4.
- ``flush_10k_rows_50_field``   — full flush including ``from_pydict``
  conversion + zstd + fsync + rename + dir-fsync.
- ``flush_10k_rows_200_field``  — same with a 128-d float32 embedding
  field; biggest realistic flush we'd see in production.
- ``flush_10k_rows_200_field_no_msg_id`` — comparison data point for
  users deciding the ``include_msg_id=False`` flag (NOT gated; just
  recorded to ADR-010 §9 once measured).
- ``arrow_schema_build_200_field`` — cold plan build time
  (``_arrow_plan_for`` with cleared cache).

Gates from Rev-4 §6.8:
  - ``append_per_msg_200_field`` <= 20 us p99
  - ``flush_10k_rows_*`` <= ``<4x WSL2 measured>`` p99 (set after first
    measurement; placeholders in thresholds.yml)
  - ``arrow_schema_build_200_field`` <= 5 ms p99
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Any

import pytest

pytestmark = pytest.mark.skipif(
    sys.platform == "win32",
    reason="Step 11 benchmarks rely on POSIX fsync + atomic rename",
)

from pyforge._internal.arrow_schema import (  # noqa: E402
    _arrow_plan_for,
)
from pyforge._internal.arrow_schema import (  # noqa: E402
    clear_cache as clear_arrow_cache,
)
from pyforge.offline import ParquetDatasetStore, _Bucket  # noqa: E402
from pyforge.schema import FeatureField, FeatureSchema, dtype  # noqa: E402

# ---------------------------------------------------------------------------
# Schemas. Match the shapes used by the WAL benches so end-to-end story
# is comparable.
# ---------------------------------------------------------------------------


def _make_50_field_schema() -> type[FeatureSchema]:
    fs = (
        [FeatureField(f"f{i:02d}", dtype.float32) for i in range(40)]
        + [FeatureField(f"i{i:02d}", dtype.int64) for i in range(8)]
        + [FeatureField("emb", dtype.float32, shape=(16,))]
        + [FeatureField("flags", dtype.uint8, shape=(4,))]
    )
    return type("_OffBench50", (FeatureSchema,), {"version": 1, "fields": fs})


def _make_200_field_schema() -> type[FeatureSchema]:
    fs = (
        [FeatureField(f"f{i:03d}", dtype.float32) for i in range(160)]
        + [FeatureField(f"i{i:03d}", dtype.int64) for i in range(30)]
        + [FeatureField(f"u{i:02d}", dtype.uint8) for i in range(9)]
        + [FeatureField("emb", dtype.float32, shape=(128,))]
    )
    return type("_OffBench200", (FeatureSchema,), {"version": 1, "fields": fs})


def _values_in_wire_order(schema: type[FeatureSchema]) -> list[Any]:
    """Build a representative values list in the same wire (name_hash)
    order the WAL producer emits."""
    from pyforge.schema import _hash_name

    by_name: dict[str, Any] = {}
    for f in schema.fields:
        if f.shape == ():
            by_name[f.name] = 1.5 if f.dtype in (dtype.float32, dtype.float64) else 1
        else:
            n = f.element_count
            by_name[f.name] = [1.5] * n if f.dtype in (dtype.float32, dtype.float64) else [1] * n
    wire_order = sorted(schema.fields, key=lambda f: _hash_name(f.name))
    return [by_name[f.name] for f in wire_order]


def _build_filled_bucket(
    schema: type[FeatureSchema],
    n_rows: int,
    *,
    include_msg_id: bool = True,
) -> _Bucket:
    """Construct a `_Bucket` and fill it with `n_rows` rows ready for
    flush. Used by the flush_* benches so we measure the flush itself,
    not the per-append cost."""
    plan = _arrow_plan_for(schema, include_msg_id=include_msg_id)
    columns: dict[str, list[Any]] = {name: [] for name in plan.column_names}
    wire_lists = tuple(columns[plan.column_names[idx]] for idx in plan.wire_to_decl)
    bucket = _Bucket(
        columns=columns,
        wire_lists=wire_lists,
        entity_id_col=columns["entity_id"],
        event_time_ns_col=columns["event_time_ns"],
        msg_id_ms_col=columns["msg_id_ms"] if include_msg_id else None,
        msg_id_seq_col=columns["msg_id_seq"] if include_msg_id else None,
    )
    values = _values_in_wire_order(schema)
    for i in range(n_rows):
        bucket.entity_id_col.append(f"ent-{i}")
        bucket.event_time_ns_col.append(0)
        for col_list, value in zip(bucket.wire_lists, values, strict=True):
            col_list.append(value)
        if include_msg_id:
            assert bucket.msg_id_ms_col is not None
            assert bucket.msg_id_seq_col is not None
            bucket.msg_id_ms_col.append(1_700_000_000_000 + i)
            bucket.msg_id_seq_col.append(0)
    return bucket


# ---------------------------------------------------------------------------
# 1. append_per_msg_200_field — the dict-free hot path.
# ---------------------------------------------------------------------------


def test_append_per_msg_200_field(benchmark: Any, tmp_path: Path) -> None:
    clear_arrow_cache()
    store = ParquetDatasetStore(tmp_path)
    schema = _make_200_field_schema()
    values = _values_in_wire_order(schema)
    msg_id = b"1700000000000-0"

    # Warm the plan cache + ensure bucket exists with one row.
    asyncio.run(store.append(schema, "ent-warm", 0, values, msg_id))

    def _one_append() -> None:
        # ParquetDatasetStore.append has no await points (all the work
        # is sync Python). Drive the coroutine directly via .send(None)
        # to avoid the ~50-100 us per-call cost of asyncio.run(), which
        # would otherwise dominate the measurement and hide regressions
        # in the actual append hot path.
        coro = store.append(schema, "ent-X", 0, values, msg_id)
        try:
            coro.send(None)
        except StopIteration:
            pass
        else:
            raise AssertionError("append unexpectedly suspended; bench assumes sync-only body")

    benchmark(_one_append)


# ---------------------------------------------------------------------------
# 2-4. flush_10k_rows_* — pre-built bucket, time the flush.
# ---------------------------------------------------------------------------


def _bench_flush(
    benchmark: Any,
    tmp_path: Path,
    schema: type[FeatureSchema],
    *,
    n_rows: int,
    include_msg_id: bool,
) -> None:
    """Build a fresh store + filled bucket per benchmark iteration so
    each measured flush() actually has work to do."""
    clear_arrow_cache()
    store = ParquetDatasetStore(tmp_path, include_msg_id=include_msg_id)

    def _setup() -> tuple[tuple[Any, ...], dict[str, Any]]:
        bucket = _build_filled_bucket(schema, n_rows, include_msg_id=include_msg_id)
        plan = _arrow_plan_for(schema, include_msg_id=include_msg_id)
        store._buffers[(schema, "1970-01-01")] = bucket
        store._plans[schema] = plan
        # Ensure metric label children exist (normally done at first append).
        from pyforge.metrics import (
            offline_bytes_written_total,
            offline_files_written_total,
        )

        store._c_files_by_schema[schema] = offline_files_written_total.labels(
            schema=schema.__name__
        )
        store._c_bytes_by_schema[schema] = offline_bytes_written_total.labels(
            schema=schema.__name__
        )
        return (), {}

    def _flush() -> None:
        asyncio.run(store.flush())

    benchmark.pedantic(_flush, setup=_setup, rounds=10, iterations=1)


def test_flush_10k_rows_50_field(benchmark: Any, tmp_path: Path) -> None:
    _bench_flush(
        benchmark,
        tmp_path,
        _make_50_field_schema(),
        n_rows=10_000,
        include_msg_id=True,
    )


def test_flush_10k_rows_200_field(benchmark: Any, tmp_path: Path) -> None:
    _bench_flush(
        benchmark,
        tmp_path,
        _make_200_field_schema(),
        n_rows=10_000,
        include_msg_id=True,
    )


def test_flush_10k_rows_200_field_no_msg_id(benchmark: Any, tmp_path: Path) -> None:
    """Comparison data point for the include_msg_id=False flag-flip
    decision. Not gated."""
    _bench_flush(
        benchmark,
        tmp_path,
        _make_200_field_schema(),
        n_rows=10_000,
        include_msg_id=False,
    )


# ---------------------------------------------------------------------------
# 5. arrow_schema_build_200_field — cold plan build.
# ---------------------------------------------------------------------------


def test_arrow_schema_build_200_field(benchmark: Any) -> None:
    schema = _make_200_field_schema()

    def _build() -> None:
        clear_arrow_cache()
        _arrow_plan_for(schema, include_msg_id=True)

    benchmark(_build)
