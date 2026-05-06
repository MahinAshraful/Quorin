"""Flamegraph driver: WALProducer.write_sync RTT loop against running consumer.

Tight loop calling write_sync against a real WALConsumer thread with
NoopOfflineWriter (per ADR-009 §3 — write_sync unblocks at online-store
durability, NOT offline). Surfaces the producer-side hash + msgpack pack
+ XADD + processed-sidetable polling cost.

py-spy must be invoked with --subprocesses because the consumer runs in
a subprocess-equivalent thread.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from benchmarks.flamegraphs._setup import setup_redis_consumer_50_field

DURATION_SECONDS = 30.0


def main() -> None:
    state = setup_redis_consumer_50_field()
    schema = state["schema"]
    redis_client = state["redis_client"]
    stream_key = state["stream_key"]
    cleanup = state["cleanup"]

    try:
        from quorin.wal import WALProducer

        producer = WALProducer(
            redis_client=redis_client,
            stream_key=stream_key,
            schema=schema,
        )

        # Build a representative row that matches schema.
        from quorin.schema import DTYPE_TO_NUMPY

        row_values = {
            f.name: (
                np.zeros(f.element_count, dtype=DTYPE_TO_NUMPY[f.dtype])
                if f.element_count > 1
                else DTYPE_TO_NUMPY[f.dtype].type(0)
            )
            for f in schema.fields
        }

        deadline = time.monotonic() + DURATION_SECONDS
        i = 0
        while time.monotonic() < deadline:
            producer.write_sync(
                entity_id=f"flame_e_{i:08d}",
                event_time_ns=int(time.time() * 1e9) + i,
                values=row_values,
                timeout_seconds=0.5,
            )
            i += 1
        print(f"write_sync flamegraph: {i:,} round-trips in {DURATION_SECONDS:.0f}s", flush=True)
    finally:
        cleanup()


if __name__ == "__main__":
    main()
