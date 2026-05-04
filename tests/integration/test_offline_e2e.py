"""End-to-end integration test: producer → WAL consumer → ParquetDatasetStore.

Stands up the full Step 9 + 10 + 11 pipeline against a real Redis,
writes rows through ``WALProducer``, lets ``WALConsumer`` drain them
into shared memory + a real ``ParquetDatasetStore``, then validates
both stores via ``pa.dataset.dataset(base, partitioning="hive")``.

Gated on ``@pytest.mark.integration``.
"""

from __future__ import annotations

import asyncio
import os
import sys
import time
from pathlib import Path

import pytest

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        sys.platform == "win32",
        reason="Step 11 + WAL consumer require POSIX shm + asyncio Redis",
    ),
]

from typing import cast as _cast  # noqa: E402
from unittest.mock import Mock as _Mock  # noqa: E402

import pyarrow.dataset as ds  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402
import redis  # noqa: E402
import redis.asyncio  # noqa: E402

from _helpers import make_segment, release_segment  # noqa: E402
from pyforge._internal.arrow_schema import clear_cache as clear_arrow_cache  # noqa: E402
from pyforge._internal.pydantic_factory import clear_cache as clear_pydantic_cache  # noqa: E402
from pyforge._internal.row_pack import clear_cache as clear_row_pack_cache  # noqa: E402
from pyforge.offline import ParquetDatasetStore  # noqa: E402
from pyforge.schema import FeatureField, FeatureSchema, dtype  # noqa: E402
from pyforge.shm import SegmentRegistry  # noqa: E402
from pyforge.wal import DEFAULT_STREAM_KEY, WALProducer  # noqa: E402
from pyforge.wal_consumer import WALConsumer  # noqa: E402

# Step 15 stub registry — these tests don't exercise the upgrade pause-clear
# branch (the only place self._registry is dereferenced), so a Mock with the
# right spec is sufficient. Real upgrade tests live in test_evolution_e2e.py.
_STUB_REGISTRY: SegmentRegistry = _cast("SegmentRegistry", _Mock(spec=SegmentRegistry))


class _IntE2E(FeatureSchema):
    version = 1
    fields = [
        FeatureField("a", dtype.float32),
        FeatureField("b", dtype.int64),
        FeatureField("emb", dtype.float32, shape=(4,)),
    ]


def _values(i: int) -> dict[str, object]:
    return {
        "a": float(i) + 0.5,
        "b": i,
        "emb": [float(i + j) for j in range(4)],
    }


def _pending_count(redis_client: redis.Redis) -> int:
    try:
        info = redis_client.xpending(DEFAULT_STREAM_KEY, "pyforge_consumers")
    except redis.exceptions.ResponseError as e:
        if "NOGROUP" in str(e):
            return -1
        raise
    return int(info["pending"])


@pytest.fixture(autouse=True)
def _fresh_caches() -> None:
    clear_pydantic_cache()
    clear_row_pack_cache()
    clear_arrow_cache()


@pytest.fixture
def segment():
    seg = make_segment(_IntE2E, capacity=256)
    try:
        yield seg
    finally:
        release_segment(seg)


@pytest.fixture
async def async_redis(redis_client: redis.Redis):
    url = os.environ.get("PYFORGE_REDIS_URL", "redis://127.0.0.1:6379/0")
    client = redis.asyncio.Redis.from_url(url, decode_responses=False)
    try:
        await client.ping()
    except Exception:
        pytest.skip("Async Redis not reachable")
    yield client
    await client.aclose()


# ---------------------------------------------------------------------------
# 1. Full pipeline: producer → consumer → ParquetDatasetStore.
# ---------------------------------------------------------------------------


async def test_full_pipeline_produces_parquet_files(
    redis_client: redis.Redis,
    async_redis: redis.asyncio.Redis,
    segment,
    tmp_path: Path,
) -> None:
    offline = ParquetDatasetStore(tmp_path)
    producer = WALProducer(redis_client)
    consumer = WALConsumer(
        async_redis,
        segments={"_IntE2E": segment},
        registry=_STUB_REGISTRY,
        offline=offline,
        consumer_name="offline-e2e-1",
        block_ms=50,
        flush_interval_seconds=0.5,
        max_pending_ack=20,
    )

    n = 50
    msg_ids = [producer.write(_IntE2E, f"ent-{i}", _values(i)) for i in range(n)]

    run_task = asyncio.create_task(consumer.run())
    await asyncio.sleep(0)
    try:
        deadline = time.monotonic() + 15.0
        while time.monotonic() < deadline:
            pending = _pending_count(redis_client)
            if pending == 0 and not consumer._pending_ack:
                break
            await asyncio.sleep(0.05)
    finally:
        await consumer.stop()
        await asyncio.wait_for(run_task, timeout=5.0)

    # Hive partition recognition: rebuild the dataset and assert that
    # the partition keys are recognised as such (not as plain string
    # columns).
    dataset = ds.dataset(tmp_path, partitioning="hive")
    assert "schema" in dataset.partitioning.schema.names
    assert "event_date" in dataset.partitioning.schema.names

    table = dataset.to_table()
    assert table.num_rows == n

    # entity_id round-trips.
    assert sorted(table.column("entity_id").to_pylist()) == sorted(f"ent-{i}" for i in range(n))

    # msg_id columns reconstruct to the producer's emitted IDs.
    ms = table.column("msg_id_ms").to_pylist()
    seq = table.column("msg_id_seq").to_pylist()
    reconstructed = sorted(f"{m}-{s}".encode() for m, s in zip(ms, seq, strict=True))
    assert reconstructed == sorted(msg_ids)


# ---------------------------------------------------------------------------
# 2. Consumer restart with fresh ParquetDatasetStore: no double-write.
# ---------------------------------------------------------------------------


async def test_consumer_restart_does_not_double_write(
    redis_client: redis.Redis,
    async_redis: redis.asyncio.Redis,
    segment,
    tmp_path: Path,
) -> None:
    offline_a = ParquetDatasetStore(tmp_path)
    producer = WALProducer(redis_client)
    consumer_a = WALConsumer(
        async_redis,
        segments={"_IntE2E": segment},
        registry=_STUB_REGISTRY,
        offline=offline_a,
        consumer_name="offline-restart-1",
        block_ms=50,
        flush_interval_seconds=0.5,
        max_pending_ack=20,
    )
    n = 30
    for i in range(n):
        producer.write(_IntE2E, f"ent-{i}", _values(i))

    run_a = asyncio.create_task(consumer_a.run())
    await asyncio.sleep(0)
    try:
        deadline = time.monotonic() + 10.0
        while time.monotonic() < deadline:
            if _pending_count(redis_client) == 0 and not consumer_a._pending_ack:
                break
            await asyncio.sleep(0.05)
    finally:
        await consumer_a.stop()
        await asyncio.wait_for(run_a, timeout=5.0)

    files_after_first = sorted(tmp_path.rglob("*.parquet"))
    rows_after_first = sum(pq.read_table(f).num_rows for f in files_after_first)
    assert rows_after_first == n

    # Second consumer with same name and a fresh ParquetDatasetStore.
    offline_b = ParquetDatasetStore(tmp_path)
    consumer_b = WALConsumer(
        async_redis,
        segments={"_IntE2E": segment},
        registry=_STUB_REGISTRY,
        offline=offline_b,
        consumer_name="offline-restart-1",  # same group/consumer name
        block_ms=50,
        flush_interval_seconds=0.5,
        max_pending_ack=20,
    )
    run_b = asyncio.create_task(consumer_b.run())
    await asyncio.sleep(0)
    try:
        # No new producer writes; PEL is empty (everything was XACKed by
        # consumer_a). Nothing to drain.
        await asyncio.sleep(0.5)
    finally:
        await consumer_b.stop()
        await asyncio.wait_for(run_b, timeout=5.0)

    files_after_second = sorted(tmp_path.rglob("*.parquet"))
    rows_after_second = sum(pq.read_table(f).num_rows for f in files_after_second)
    assert rows_after_second == n, (
        f"expected {n} rows after restart, got {rows_after_second} — "
        "consumer is double-writing already-XACKed messages"
    )


# ---------------------------------------------------------------------------
# 3. Step 12 reader: full producer → consumer → reader pipeline.
# ---------------------------------------------------------------------------


async def test_full_pipeline_producer_to_reader(
    redis_client: redis.Redis,
    async_redis: redis.asyncio.Redis,
    segment,
    tmp_path: Path,
) -> None:
    """Producer → consumer → ParquetDatasetStore → read_point_in_time.

    Writes 30 messages spread across 3 entities x 10 distinct
    event_times. Queries each entity at multiple cutoff times. Asserts
    every result matches the producer's last write before that cutoff
    — leak-free, dedup-correct, against real Redis + real fsync + real
    partition layout.
    """
    import pyarrow as pa

    offline = ParquetDatasetStore(tmp_path)
    producer = WALProducer(redis_client)
    consumer = WALConsumer(
        async_redis,
        segments={"_IntE2E": segment},
        registry=_STUB_REGISTRY,
        offline=offline,
        consumer_name="offline-reader-1",
        block_ms=50,
        flush_interval_seconds=0.5,
        max_pending_ack=20,
    )

    # 3 entities, 10 event_times each (spread within a day so all land
    # in one partition; reader exercise stays focused on asof, not
    # partition pruning).
    base_event_time = 1_700_000_000_000_000_000
    write_log: list[tuple[str, int, float]] = []  # (eid, event_time_ns, expected_a)
    for entity_idx in range(3):
        eid = f"ent-{entity_idx}"
        for t_idx in range(10):
            event_time_ns = base_event_time + t_idx * 60_000_000_000  # 60s apart
            i = entity_idx * 10 + t_idx
            producer.write(
                _IntE2E,
                eid,
                _values(i),
                event_time_ns=event_time_ns,
            )
            write_log.append((eid, event_time_ns, _values(i)["a"]))

    run_task = asyncio.create_task(consumer.run())
    await asyncio.sleep(0)
    try:
        deadline = time.monotonic() + 15.0
        while time.monotonic() < deadline:
            if _pending_count(redis_client) == 0 and not consumer._pending_ack:
                break
            await asyncio.sleep(0.05)
    finally:
        await consumer.stop()
        await asyncio.wait_for(run_task, timeout=5.0)

    # Build a query mixing match cases: cutoff at boundary, in-between,
    # before-everything, after-everything.
    query_rows: list[tuple[str, int, float | None]] = []  # (eid, aot, expected_a or None)
    for entity_idx in range(3):
        eid = f"ent-{entity_idx}"
        # Cutoff at the 5th event for this entity → expect the 5th
        # entity write.
        cutoff = base_event_time + 5 * 60_000_000_000
        expected_a = _values(entity_idx * 10 + 5)["a"]
        query_rows.append((eid, cutoff, expected_a))
        # Cutoff before any event → null
        query_rows.append((eid, base_event_time - 1_000_000_000, None))
        # Cutoff after all events → latest (idx=9)
        latest_cutoff = base_event_time + 100 * 60_000_000_000
        latest_a = _values(entity_idx * 10 + 9)["a"]
        query_rows.append((eid, latest_cutoff, latest_a))

    query = pa.table(
        {
            "entity_id": pa.array([r[0] for r in query_rows], type=pa.string()),
            "as_of_time": pa.array([r[1] for r in query_rows], type=pa.int64()),
        }
    )
    result = offline.read_point_in_time(_IntE2E, query, lookback_days=30)
    a_col = result["a"].to_pylist()
    et_col = result["event_time_ns"].to_pylist()
    for i, (_eid, aot, expected_a) in enumerate(query_rows):
        if expected_a is None:
            assert et_col[i] is None, f"row {i}: expected null, got event_time {et_col[i]}"
            assert a_col[i] is None, f"row {i}: expected null, got a={a_col[i]}"
        else:
            assert a_col[i] == pytest.approx(expected_a), (
                f"row {i}: expected a={expected_a}, got {a_col[i]}"
            )
            # Leak-free check: returned event_time must be <= cutoff.
            assert et_col[i] is not None and et_col[i] <= aot, (
                f"row {i}: event_time {et_col[i]} > cutoff {aot} (LEAK)"
            )
