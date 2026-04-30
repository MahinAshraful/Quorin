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
from pyforge.wal import DEFAULT_STREAM_KEY, WALProducer  # noqa: E402
from pyforge.wal_consumer import WALConsumer  # noqa: E402


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
