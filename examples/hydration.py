"""Bulk-load the online store from the offline Parquet store via hydrate().

Use case: warming up a freshly-restarted serving fleet from durable
offline data. The orchestrator reads the latest features for each
entity from the Parquet partition, allocates a fresh shared-memory
segment, and bulk-inserts the rows via the Numba kernel.

Run as a script:

    python examples/hydration.py

Run via pytest:

    pytest examples/hydration.py -v

The example writes a small dataset to a temp directory, hydrates it,
and exits. Hydration is a synchronous operation; async callers should
wrap in ``asyncio.to_thread(hydrate, ...)``.
"""

from __future__ import annotations

import asyncio
import os
import tempfile
import time
from pathlib import Path

# `time` is used to anchor event_time_ns at now-N seconds so the
# 30-day hydrate lookback window includes the seeded rows.
import pytest
import redis

from quorin.hydration import hydrate
from quorin.offline import ParquetDatasetStore
from quorin.schema import FeatureField, FeatureSchema, dtype
from quorin.shm import SegmentRegistry

pytestmark = pytest.mark.requires_redis


class FleetFeatures(FeatureSchema):
    version = 1
    fields = [
        FeatureField("score_a", dtype.float32),
        FeatureField("score_b", dtype.float32),
    ]


def main() -> None:
    redis_url = os.environ.get("QUORIN_REDIS_URL", "redis://127.0.0.1:6379/0")
    redis_client = redis.Redis.from_url(redis_url, socket_timeout=5.0)

    with tempfile.TemporaryDirectory() as tmp:
        # 1. Seed the offline store with a small dataset.
        store = ParquetDatasetStore(Path(tmp))

        # Synchronous bulk-write: build a single PyArrow table directly
        # rather than going through the consumer's append path. (The
        # consumer's path is the production write path; this is a
        # one-shot setup for the example.)
        # In real deployments, the offline store is populated by the
        # WAL consumer over time.
        async def _seed() -> None:
            # Anchor at "now - 100 seconds" so all 100 rows fall inside
            # the default 30-day hydrate lookback window. Using raw
            # ``i * 1_000_000_000`` puts data at 1970-01-01+i seconds
            # — 50+ years outside the window — and hydrate refuses.
            base_ns = time.time_ns() - 100 * 1_000_000_000
            for i in range(100):
                await store.append(
                    FleetFeatures,
                    entity_id=f"ent-{i:04d}",
                    event_time_ns=base_ns + i * 1_000_000_000,
                    values_list=[float(i) / 100, float(i) * 2],
                    msg_id=f"{i}-0".encode(),
                )
            await store.flush()
            await store.close()

        asyncio.run(_seed())

        # 2. Hydrate: precondition is "no segment exists yet for this
        # schema" + "no consumer running." The orchestrator allocates
        # a fresh segment sized to the entity count + headroom.
        registry = SegmentRegistry(redis_client)
        result = hydrate(
            FleetFeatures,
            store,
            registry,
            redis_client=redis_client,
            capacity_factor=4.0,
        )
        print(f"hydrated {result.entity_count} entities")
        print(f"new segment: {result.segment_name}")
        print(f"elapsed: {result.elapsed_seconds:.3f}s")

        # 3. Verify by reading one row.
        seg = registry.open_current(FleetFeatures)
        try:
            import quorin

            out = quorin.assembly.assemble(seg, "ent-0050")
            print(f"ent-0050 features: {out}")
        finally:
            registry.close(seg)


def test_hydration() -> None:
    main()


if __name__ == "__main__":
    main()
