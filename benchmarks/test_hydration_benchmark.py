"""Benchmarks for pyforge.hydration.hydrate (Step 13).

Four benches:

* ``test_hydrate_10k_50_field_smoke`` — always-on (no env gate).
  Catches order-of-magnitude regressions; finer detection requires
  PYFORGE_RUN_LARGE_BENCH=1 on the larger scales.
* ``test_hydrate_100k_50_field`` — env-gated (PYFORGE_RUN_LARGE_BENCH=1).
* ``test_hydrate_1m_200_field`` — env-gated (PYFORGE_RUN_LARGE_BENCH=1).
* ``test_hydrate_10m_200_field_record`` — env-gated
  (PYFORGE_RUN_RECORD_BENCH=1). Capacity-planning only; no committed gate.

Datasets generated in session-scoped fixtures. Generation is the
dominant cost for the larger scales (~30-90s for 100k, ~10 min for
1M, ~100 min for 10M on WSL2 — see CLAUDE.md §8 cold-page-fault
ceiling).

WARM-CACHE STEADY-STATE measurement: pytest-benchmark runs
``rounds=5`` by default; rounds 2-5 hit warm tmpfs pages because
``posix_shm.unlink`` deallocates the inode but does not flush page
cache. The reported p99 reflects "after-first-call" latency, NOT
first-call cold-cache. Step 16 will add ``test_hydrate_*_cold_round1_only``
variants (rounds=1, fresh dataset per round) alongside py-spy
flamegraphs. For Commit B: warm-cache is the regression-detection
signal we want; cold-cache is the capacity-planning signal that
requires more setup.

Methodology vs Steps 11/12: gates set at NATIVE-LINUX targets, not
"4x WSL2 measured". CI runs on native Linux; that's where the gates
actually block. Steps 11/12 gates are loose by ~4-6x and should be
retroactively tightened in Step 16. See ``thresholds.yml`` top-of-
file methodology block.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import sys
from pathlib import Path
from typing import Any

import pytest

pytestmark = pytest.mark.skipif(
    sys.platform == "win32",
    reason="Step 13 benchmarks rely on POSIX shm + tmpfs",
)

import numpy as np  # noqa: E402

# Reuse Step 12's reader-bench helpers for dataset construction —
# same pattern, same wire-order encoding logic.
from _watchdog_helpers import drain_cleanup_queue  # noqa: E402

from pyforge.hydration import hydrate  # noqa: E402
from pyforge.offline import ParquetDatasetStore  # noqa: E402
from pyforge.schema import (  # noqa: E402
    FeatureField,
    FeatureSchema,
    _hash_name,
    dtype,
)
from pyforge.shm import SegmentNotFoundError, SegmentRegistry, _key_current  # noqa: E402

# ---------------------------------------------------------------------------
# Schemas — top-level so subprocess fixtures (if any) can pickle them.
# ---------------------------------------------------------------------------


def _make_50_field_schema(version: int = 1) -> type[FeatureSchema]:
    fs = (
        [FeatureField(f"f{i:02d}", dtype.float32) for i in range(40)]
        + [FeatureField(f"i{i:02d}", dtype.int64) for i in range(8)]
        + [FeatureField("emb", dtype.float32, shape=(16,))]
        + [FeatureField("flags", dtype.uint8, shape=(4,))]
    )
    return type(
        f"_HydrateBench50_v{version}",
        (FeatureSchema,),
        {"version": version, "fields": fs},
    )


def _make_200_field_schema(version: int = 1) -> type[FeatureSchema]:
    fs = (
        [FeatureField(f"f{i:03d}", dtype.float32) for i in range(160)]
        + [FeatureField(f"i{i:03d}", dtype.int64) for i in range(30)]
        + [FeatureField(f"u{i:02d}", dtype.uint8) for i in range(9)]
        + [FeatureField("emb", dtype.float32, shape=(128,))]
    )
    return type(
        f"_HydrateBench200_v{version}",
        (FeatureSchema,),
        {"version": version, "fields": fs},
    )


# Module-level fixed instances so pickling across pytest fixture cache
# layers works deterministically. Each scale uses a distinct schema
# class so dataset directories don't collide across benches.
_SCHEMA_50_SMOKE = _make_50_field_schema(version=131)
_SCHEMA_50_LARGE = _make_50_field_schema(version=132)
_SCHEMA_200_HUGE = _make_200_field_schema(version=133)
_SCHEMA_200_RECORD = _make_200_field_schema(version=134)


def _values_in_wire_order(schema: type[FeatureSchema], rng: np.random.Generator) -> list[Any]:
    """Random values per field, name-hash wire order matching the producer."""
    by_name: dict[str, Any] = {}
    for f in schema.fields:
        n = f.element_count
        if f.dtype is dtype.float32:
            arr = rng.standard_normal(n).astype(np.float32)
            flat: list[Any] = [float(x) for x in arr]
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
    *,
    n_entities: int,
    flush_every: int = 10_000,
    seed: int = 0,
) -> None:
    """Generate a hive-partitioned dataset on disk via the writer.

    One row per entity (latest_features dedups on read; hydrate's
    "latest per entity within lookback" doesn't need historical
    rows for the bench scenario).

    Timestamps anchored at ``time.time_ns()`` (subtracting a small
    per-row offset) so the dataset falls within hydrate's default
    ``lookback_days=30`` window regardless of when the bench runs.
    Hardcoded epoch values would expire after ~30 days of clock-time
    drift, making benches silently raise EmptyDatasetError instead
    of measuring hydrate cost.
    """
    import time as _time

    store = ParquetDatasetStore(base)
    rng = np.random.default_rng(seed=seed)
    # Anchor at "now"; subtract a small per-row offset so timestamps
    # are strictly increasing yet all within the last few seconds.
    base_event_time_ns = _time.time_ns() - n_entities * 1_000

    async def _run() -> None:
        for i in range(n_entities):
            entity_id = f"e{i:08d}"
            event_time_ns = base_event_time_ns + i * 1_000  # ~1us apart, all recent
            values = _values_in_wire_order(schema, rng)
            msg_id = f"{1_700_000_000_000 + i}-0".encode()
            await store.append(schema, entity_id, event_time_ns, values, msg_id)
            if (i + 1) % flush_every == 0:
                await store.flush()
        await store.flush()
        await store.close()

    asyncio.run(_run())


# ---------------------------------------------------------------------------
# Session-scoped dataset fixtures.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def _hydrate_dataset_10k_50_smoke(
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[Path, type[FeatureSchema]]:
    """10k entities x 50 fields. Generation ~3-5s on WSL2."""
    base = tmp_path_factory.mktemp("step13_bench_10k_50f_smoke")
    _build_dataset(base, _SCHEMA_50_SMOKE, n_entities=10_000)
    return base, _SCHEMA_50_SMOKE


@pytest.fixture(scope="session")
def _hydrate_dataset_100k_50(
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[Path, type[FeatureSchema]]:
    """100k entities x 50 fields. Generation ~30-90s on WSL2."""
    base = tmp_path_factory.mktemp("step13_bench_100k_50f")
    _build_dataset(base, _SCHEMA_50_LARGE, n_entities=100_000)
    return base, _SCHEMA_50_LARGE


@pytest.fixture(scope="session")
def _hydrate_dataset_1m_200(
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[Path, type[FeatureSchema]]:
    """1M entities x 200 fields + 128-emb. Generation ~10 min on WSL2."""
    base = tmp_path_factory.mktemp("step13_bench_1m_200f")
    _build_dataset(base, _SCHEMA_200_HUGE, n_entities=1_000_000)
    return base, _SCHEMA_200_HUGE


@pytest.fixture(scope="session")
def _hydrate_dataset_10m_200(
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[Path, type[FeatureSchema]]:
    """10M entities x 200 fields + 128-emb. Generation ~100 min on WSL2."""
    base = tmp_path_factory.mktemp("step13_bench_10m_200f")
    _build_dataset(base, _SCHEMA_200_RECORD, n_entities=10_000_000)
    return base, _SCHEMA_200_RECORD


# ---------------------------------------------------------------------------
# Benchmark helpers — pre-call cleanup so each round starts from a clean slate.
# ---------------------------------------------------------------------------


def _make_hydrate_runner(
    schema: type[FeatureSchema],
    store: ParquetDatasetStore,
    registry: SegmentRegistry,
    redis_client: Any,
) -> Any:
    """Returns a callable suitable for pytest-benchmark.

    Each call: drop the prior segment if any, drain cleanup_queue,
    then call ``hydrate``. Pre-call cleanup is sub-ms; bundled into
    the bench cost since hydrate runs in seconds. Acceptable for
    the regression-detection signal we want.
    """

    def _do_hydrate() -> int:
        # Drop any prior segment from a previous round.
        # NOTE: registry.close removes refcount + pid_segments + queues for
        # cleanup_queue, but does NOT delete pyforge:schema:{name}:current
        # (production-side, the next registry.create overwrites it). For
        # repeated hydrate calls in a bench loop, we must DEL it manually
        # or precondition #1 trips on round 2+.
        with contextlib.suppress(SegmentNotFoundError):
            seg = registry.open_current(schema)
            registry.close(seg)
        drain_cleanup_queue(redis_client)
        redis_client.delete(_key_current(schema))

        # The actual measured call. Returns HydrationResult.
        result = hydrate(schema, store, registry, redis_client=redis_client)
        return result.entity_count

    return _do_hydrate


# ---------------------------------------------------------------------------
# Bench-0 — always-on smoke variant (10k x 50 fields).
# ---------------------------------------------------------------------------


def test_hydrate_10k_50_field_smoke(
    benchmark: Any,
    _hydrate_dataset_10k_50_smoke: tuple[Path, type[FeatureSchema]],
    redis_client: Any,
) -> None:
    """Always-on smoke. Catches order-of-magnitude regressions only.

    Native target ~50-200ms; gate at 1s (5-20x upper bound). Finer
    regression detection requires PYFORGE_RUN_LARGE_BENCH=1.
    """
    base, schema = _hydrate_dataset_10k_50_smoke
    store = ParquetDatasetStore(base)
    registry = SegmentRegistry(redis_client)

    runner = _make_hydrate_runner(schema, store, registry, redis_client)
    n = benchmark(runner)
    assert n == 10_000

    # Cleanup so the next pytest run doesn't see leftover state.
    with contextlib.suppress(SegmentNotFoundError):
        seg = registry.open_current(schema)
        registry.close(seg)
    drain_cleanup_queue(redis_client)
    redis_client.delete(_key_current(schema))


# ---------------------------------------------------------------------------
# Bench-1 — 100k x 50 fields. Env-gated.
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    os.environ.get("PYFORGE_RUN_LARGE_BENCH") != "1",
    reason="Set PYFORGE_RUN_LARGE_BENCH=1 to run the 100k bench (~30-90s generation)",
)
def test_hydrate_100k_50_field(
    benchmark: Any,
    _hydrate_dataset_100k_50: tuple[Path, type[FeatureSchema]],
    redis_client: Any,
) -> None:
    """100k entities x 50 fields. Gate: 8s p99 native target."""
    base, schema = _hydrate_dataset_100k_50
    store = ParquetDatasetStore(base)
    registry = SegmentRegistry(redis_client)

    runner = _make_hydrate_runner(schema, store, registry, redis_client)
    n = benchmark(runner)
    assert n == 100_000

    with contextlib.suppress(SegmentNotFoundError):
        seg = registry.open_current(schema)
        registry.close(seg)
    drain_cleanup_queue(redis_client)
    redis_client.delete(_key_current(schema))


# ---------------------------------------------------------------------------
# Bench-2 — 1M x 200 fields + 128-emb. Env-gated.
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    os.environ.get("PYFORGE_RUN_LARGE_BENCH") != "1",
    reason="Set PYFORGE_RUN_LARGE_BENCH=1 to run the 1M bench (~10 min generation)",
)
def test_hydrate_1m_200_field(
    benchmark: Any,
    _hydrate_dataset_1m_200: tuple[Path, type[FeatureSchema]],
    redis_client: Any,
) -> None:
    """1M entities x 200 fields + 128-emb. Gate: 45s p99 native target."""
    base, schema = _hydrate_dataset_1m_200
    store = ParquetDatasetStore(base)
    registry = SegmentRegistry(redis_client)

    runner = _make_hydrate_runner(schema, store, registry, redis_client)
    n = benchmark(runner)
    assert n == 1_000_000

    with contextlib.suppress(SegmentNotFoundError):
        seg = registry.open_current(schema)
        registry.close(seg)
    drain_cleanup_queue(redis_client)
    redis_client.delete(_key_current(schema))


# ---------------------------------------------------------------------------
# Bench-3 — 10M x 200 fields. Record-only env gate.
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    os.environ.get("PYFORGE_RUN_RECORD_BENCH") != "1",
    reason="Set PYFORGE_RUN_RECORD_BENCH=1 to run the 10M record bench (~100 min generation)",
)
def test_hydrate_10m_200_field_record(
    benchmark: Any,
    _hydrate_dataset_10m_200: tuple[Path, type[FeatureSchema]],
    redis_client: Any,
) -> None:
    """10M entities x 200 fields. Capacity-planning only; no committed gate."""
    base, schema = _hydrate_dataset_10m_200
    store = ParquetDatasetStore(base)
    registry = SegmentRegistry(redis_client)

    runner = _make_hydrate_runner(schema, store, registry, redis_client)
    n = benchmark(runner)
    assert n == 10_000_000

    with contextlib.suppress(SegmentNotFoundError):
        seg = registry.open_current(schema)
        registry.close(seg)
    drain_cleanup_queue(redis_client)
    redis_client.delete(_key_current(schema))
