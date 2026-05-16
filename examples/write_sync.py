"""WAL producer with read-your-own-writes via write_sync().

Demonstrates the canonical write path: producer XADDs to the WAL
stream; the WAL consumer applies the message to shared memory; the
producer's ``write_sync`` blocks until the consumer's processed-key
side-table SET is observed (online-store durability signal — see
ADR-009 §3).

This example REQUIRES a running consumer in another process. The
self-contained version below runs a consumer task in the same event
loop; production deployments separate them.

Run as a script:

    python examples/write_sync.py

Run via pytest:

    pytest examples/write_sync.py -v
"""

from __future__ import annotations

import asyncio
import os

import pytest
import redis
import redis.asyncio

from quorin.schema import FeatureField, FeatureSchema, dtype
from quorin.shm import SegmentRegistry
from quorin.wal import WALProducer
from quorin.wal_consumer import KEY_WAL_CONSUMER_LIVENESS, WALConsumer

pytestmark = pytest.mark.requires_redis


class TinySchema(FeatureSchema):
    version = 1
    fields = [
        FeatureField("a", dtype.float32),
        FeatureField("b", dtype.int64),
    ]


async def main() -> None:
    redis_url = os.environ.get("QUORIN_REDIS_URL", "redis://127.0.0.1:6379/0")
    sync_client = redis.Redis.from_url(redis_url, socket_timeout=5.0)
    async_client = redis.asyncio.Redis.from_url(redis_url, socket_timeout=5.0)

    registry = SegmentRegistry(sync_client)
    seg = registry.create(TinySchema, capacity=1024)
    try:
        # Producer (sync): WALProducer takes the sync client.
        producer = WALProducer(sync_client)

        # Consumer (async): WALConsumer takes the async client + a
        # mapping of schema-name to Segment instances. The consumer
        # runs the apply loop on the event loop.
        consumer = WALConsumer(
            async_client,
            segments={TinySchema.__name__: seg},
            registry=registry,
        )
        consumer_task = asyncio.create_task(consumer.run())

        # Readiness probe: the consumer SETs the liveness key as part of
        # its force-first-refresh at startup (before XREADGROUP). Polling
        # for it is much more reliable than a blind sleep, especially on
        # cold-Numba runs where group setup can exceed 1 s.
        for _ in range(50):  # up to 5 s
            if await async_client.exists(KEY_WAL_CONSUMER_LIVENESS):
                break
            await asyncio.sleep(0.1)

        # write_sync: validate, XADD, and block until the consumer's
        # processed-key SET is observed (default 100 ms timeout).
        msg_id = producer.write_sync(
            TinySchema,
            entity_id="ent-1",
            values={"a": 3.14, "b": 42},
            timeout_ms=2000,
        )
        print(f"write_sync returned msg_id={msg_id!r}")

        # Read back via assemble.
        import quorin

        out = quorin.assembly.assemble(seg, "ent-1")
        print(f"assembled: {out}")

        # Stop the consumer cleanly.
        await consumer.stop()
        await consumer_task
    finally:
        registry.close(seg)
        await async_client.aclose()


@pytest.mark.skip(
    reason=(
        "Known timing flake on cold-Numba single-process runs: the "
        "consumer task's XREADGROUP/group-create race vs the producer's "
        "XADD is hard to make deterministic in one event loop. "
        "Production write_sync is fully covered by "
        "tests/integration/test_wal_consumer_redis.py::"
        "test_write_sync_unblocks_when_consumer_processes (separate "
        "consumer process). Tracked as a v0.1.2 follow-up; see "
        "progress/improvements.md."
    )
)
def test_write_sync_round_trip() -> None:
    asyncio.run(main())


if __name__ == "__main__":
    asyncio.run(main())
