"""Benchmarks for quorin.offline.ParquetDatasetStore.read_point_in_time (Step 12).

Three gated benches + one non-gated record bench (per Step 12 plan,
Rev-3 polish #12). Datasets are generated in a session-scoped fixture
so each repeat runs against a warm-page-cache disk (representative of
repeated training queries).

Gates set at 4x WSL2 measured baseline per ADR-010 §8 methodology.
The record bench tracks the build-plan headline target (10k pairs x
10M rows in <5s native) without failing CI on WSL2 disk variance.
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path
from typing import Any

import pytest

pytestmark = pytest.mark.skipif(
    sys.platform == "win32",
    reason="Step 12 benchmarks rely on POSIX fsync + atomic rename",
)

import numpy as np  # noqa: E402
import pyarrow as pa  # noqa: E402

from quorin._internal.arrow_schema import (  # noqa: E402
    clear_cache as clear_arrow_cache,
)
from quorin.offline import ParquetDatasetStore  # noqa: E402
from quorin.schema import (  # noqa: E402
    FeatureField,
    FeatureSchema,
    _hash_name,
    dtype,
)

_DAY_NS = 86_400_000_000_000


# ---------------------------------------------------------------------------
# Schemas.
# ---------------------------------------------------------------------------


def _make_50_field_schema() -> type[FeatureSchema]:
    fs = (
        [FeatureField(f"f{i:02d}", dtype.float32) for i in range(40)]
        + [FeatureField(f"i{i:02d}", dtype.int64) for i in range(8)]
        + [FeatureField("emb", dtype.float32, shape=(16,))]
        + [FeatureField("flags", dtype.uint8, shape=(4,))]
    )
    return type("_RdrBench50", (FeatureSchema,), {"version": 1, "fields": fs})


def _make_100_field_schema() -> type[FeatureSchema]:
    fs = (
        [FeatureField(f"f{i:03d}", dtype.float32) for i in range(80)]
        + [FeatureField(f"i{i:03d}", dtype.int64) for i in range(15)]
        + [FeatureField(f"u{i:02d}", dtype.uint8) for i in range(4)]
        + [FeatureField("emb", dtype.float32, shape=(32,))]
    )
    return type("_RdrBench100", (FeatureSchema,), {"version": 1, "fields": fs})


def _make_200_field_schema() -> type[FeatureSchema]:
    fs = (
        [FeatureField(f"f{i:03d}", dtype.float32) for i in range(160)]
        + [FeatureField(f"i{i:03d}", dtype.int64) for i in range(30)]
        + [FeatureField(f"u{i:02d}", dtype.uint8) for i in range(9)]
        + [FeatureField("emb", dtype.float32, shape=(128,))]
    )
    return type("_RdrBench200", (FeatureSchema,), {"version": 1, "fields": fs})


def _values_in_wire_order(schema: type[FeatureSchema], rng: np.random.Generator) -> list[Any]:
    """Random values per-field, in name-hash wire order matching the producer."""
    by_name: dict[str, Any] = {}
    for f in schema.fields:
        n = f.element_count
        if f.dtype is dtype.float32:
            arr = rng.standard_normal(n).astype(np.float32)
            flat = [float(x) for x in arr]
        elif f.dtype is dtype.float64:
            arr = rng.standard_normal(n).astype(np.float64)
            flat = [float(x) for x in arr]
        elif f.dtype is dtype.int64:
            flat = [int(x) for x in rng.integers(-(1 << 30), 1 << 30, size=n)]
        elif f.dtype is dtype.int32:
            flat = [int(x) for x in rng.integers(-(1 << 20), 1 << 20, size=n)]
        elif f.dtype is dtype.uint8:
            flat = [int(x) for x in rng.integers(0, 256, size=n)]
        else:
            flat = [0] * n
        if f.shape == ():
            by_name[f.name] = flat[0]
        elif len(f.shape) == 1:
            by_name[f.name] = flat
        else:
            rows, cols = f.shape
            by_name[f.name] = [flat[i * cols : (i + 1) * cols] for i in range(rows)]

    wire_order = sorted(schema.fields, key=lambda f: _hash_name(f.name))
    return [by_name[f.name] for f in wire_order]


def _build_dataset(
    base: Path,
    schema: type[FeatureSchema],
    n_rows: int,
    n_days: int,
    n_entities: int,
    *,
    flush_every: int = 10_000,
    seed: int = 0,
) -> None:
    """Generate a hive-partitioned dataset on disk via the writer.

    Spreads ``n_rows`` rows across ``n_days`` UTC days for partition
    pruning realism. ``n_entities`` distinct entity_ids; rows are
    distributed round-robin so per-entity coverage is uniform.
    """
    clear_arrow_cache()
    store = ParquetDatasetStore(base)
    rng = np.random.default_rng(seed=seed)
    base_event_time = 1_700_000_000_000_000_000

    async def _run() -> None:
        for i in range(n_rows):
            day = i % n_days
            event_time_ns = (
                base_event_time + day * _DAY_NS + (i % 1_000_000) * 1_000_000  # spread within a day
            )
            entity_id = f"e{i % n_entities:06d}"
            values = _values_in_wire_order(schema, rng)
            msg_id = f"{1700_000_000_000 + i}-0".encode()
            await store.append(schema, entity_id, event_time_ns, values, msg_id)
            if (i + 1) % flush_every == 0:
                await store.flush()
        await store.flush()

    asyncio.run(_run())


def _build_query(n_pairs: int, n_entities: int, n_days: int, *, seed: int = 1) -> pa.Table:
    """Random query: 50% within the dataset's time range, 50% before it."""
    rng = np.random.default_rng(seed=seed)
    base_event_time = 1_700_000_000_000_000_000
    eids = [f"e{rng.integers(0, n_entities):06d}" for _ in range(n_pairs)]
    aots: list[int] = []
    for _ in range(n_pairs):
        if rng.random() < 0.5:
            day = int(rng.integers(0, n_days))
            aots.append(int(base_event_time + day * _DAY_NS + rng.integers(0, _DAY_NS)))
        else:
            aots.append(int(base_event_time - rng.integers(1, _DAY_NS * 100)))
    return pa.table(
        {
            "entity_id": pa.array(eids, type=pa.string()),
            "as_of_time": pa.array(aots, type=pa.int64()),
        }
    )


# ---------------------------------------------------------------------------
# Session-scoped dataset fixtures. Generation cost is amortized across all
# repeats of the bench in a single pytest run.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def _dataset_500k_50field(
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[Path, type[FeatureSchema]]:
    base = tmp_path_factory.mktemp("step12_bench_500k_50f")
    schema = _make_50_field_schema()
    _build_dataset(base, schema, n_rows=500_000, n_days=30, n_entities=10_000)
    return base, schema


@pytest.fixture(scope="session")
def _dataset_5m_100field(
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[Path, type[FeatureSchema]]:
    base = tmp_path_factory.mktemp("step12_bench_5m_100f")
    schema = _make_100_field_schema()
    _build_dataset(base, schema, n_rows=5_000_000, n_days=50, n_entities=50_000)
    return base, schema


@pytest.fixture(scope="session")
def _dataset_10m_200field(
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[Path, type[FeatureSchema]]:
    base = tmp_path_factory.mktemp("step12_bench_10m_200f")
    schema = _make_200_field_schema()
    _build_dataset(base, schema, n_rows=10_000_000, n_days=100, n_entities=100_000)
    return base, schema


# ---------------------------------------------------------------------------
# Gated benches.
# ---------------------------------------------------------------------------


def test_read_pit_1k_pairs_500k_rows_50_field(
    benchmark: Any, _dataset_500k_50field: tuple[Path, type[FeatureSchema]]
) -> None:
    """1k pairs x 500k rows x 50 fields. Gate: <8 s p99 WSL2 (4x measured)."""
    base, schema = _dataset_500k_50field
    store = ParquetDatasetStore(base)
    query = _build_query(n_pairs=1_000, n_entities=10_000, n_days=30)

    def _do_read() -> int:
        result = store.read_point_in_time(schema, query, lookback_days=30)
        return len(result)

    n = benchmark(_do_read)
    assert n == 1_000


@pytest.mark.skipif(
    os.environ.get("QUORIN_RUN_LARGE_BENCH") != "1",
    reason="Set QUORIN_RUN_LARGE_BENCH=1 to run the 5M x 100 bench (~15-20 min generation)",
)
def test_read_pit_10k_pairs_5m_rows_100_field(
    benchmark: Any, _dataset_5m_100field: tuple[Path, type[FeatureSchema]]
) -> None:
    """10k pairs x 5M rows x 100 fields. Gate: <60 s p99 WSL2.

    Skipped from routine bench runs because dataset generation takes
    ~15-20 minutes on WSL2 + Docker Desktop (~5M append + flush calls
    through the full async writer path). Opt in with the env var when
    you need a measured number for gate calibration. The
    1k_pairs_500k_rows bench at ~3 s p99 is the routine 2x-regression
    detector.
    """
    base, schema = _dataset_5m_100field
    store = ParquetDatasetStore(base)
    query = _build_query(n_pairs=10_000, n_entities=50_000, n_days=50)

    def _do_read() -> int:
        result = store.read_point_in_time(schema, query, lookback_days=30)
        return len(result)

    n = benchmark(_do_read)
    assert n == 10_000


def test_read_dataset_construction_only(
    benchmark: Any, _dataset_500k_50field: tuple[Path, type[FeatureSchema]]
) -> None:
    """Just _open_dataset + _validate_dataset_uniform on a 30-day dataset.

    No to_table call. Tests the operability cost of the upfront checks.
    Gate: <2 s p99 WSL2.
    """
    base, schema = _dataset_500k_50field
    store = ParquetDatasetStore(base)

    def _do_open() -> int:
        ds_obj = store._open_dataset(schema)
        if ds_obj is None:
            return 0
        store._validate_dataset_uniform(ds_obj)
        return len(ds_obj.schema.names)

    n = benchmark(_do_open)
    assert n > 0


# ---------------------------------------------------------------------------
# Non-gated record bench. Tracks the build-plan headline (10k x 10M x 200)
# without failing CI on WSL2 disk variance.
#
# Skipped by default to keep routine benchmark runs fast. Enable with
# ``QUORIN_RUN_RECORD_BENCH=1`` or invoke directly by name.
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    os.environ.get("QUORIN_RUN_RECORD_BENCH") != "1",
    reason="Set QUORIN_RUN_RECORD_BENCH=1 to run the 10M record bench (~15 min generation)",
)
def test_read_pit_10k_pairs_10m_rows_200_field_record(
    benchmark: Any, _dataset_10m_200field: tuple[Path, type[FeatureSchema]]
) -> None:
    """10k pairs x 10M rows x 200 fields. Build-plan headline target.

    Non-gated; numbers committed to ``benchmarks/results/`` for
    historical comparison only. Build-plan target: <5 s native.
    """
    base, schema = _dataset_10m_200field
    store = ParquetDatasetStore(base)
    query = _build_query(n_pairs=10_000, n_entities=100_000, n_days=100)

    def _do_read() -> int:
        result = store.read_point_in_time(schema, query, lookback_days=30)
        return len(result)

    n = benchmark(_do_read)
    assert n == 10_000
