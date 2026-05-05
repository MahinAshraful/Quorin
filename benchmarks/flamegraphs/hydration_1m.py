"""Flamegraph driver: hydrate(...) against a 1M-row Parquet dataset.

OPERATOR-ONLY. ~6 GB peak memory; ubuntu-latest's ~3.5 GB /dev/shm
SIGBUSes during populate. ADR-015 §7 + Step 16b's operator-only
contract apply. Run on a workstation or self-hosted runner with
adequate /dev/shm.

Gating: requires PYFORGE_RUN_LARGE_SHM_BENCH=1.
"""

from __future__ import annotations

import asyncio
import os
import shutil
import sys
import time
from pathlib import Path

if os.environ.get("PYFORGE_RUN_LARGE_SHM_BENCH") != "1":
    raise SystemExit(
        "hydration_1m flamegraph driver requires PYFORGE_RUN_LARGE_SHM_BENCH=1. "
        "ubuntu-latest's ~3.5 GB /dev/shm cannot host the ~6 GB peak. "
        "Run on a workstation or self-hosted runner with adequate /dev/shm. "
        "See ADR-015 §7."
    )

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from benchmarks.flamegraphs._setup import (
    Schema200Field,
    populate_dataset_for_hydration,
)

N_ENTITIES = 1_000_000


def _redis_url() -> str:
    return os.environ.get("PYFORGE_REDIS_URL", "redis://127.0.0.1:6379/0")


def main() -> None:
    import redis

    from pyforge.hydration import hydrate
    from pyforge.offline import ParquetDatasetStore
    from pyforge.shm import SegmentRegistry

    redis_client = redis.Redis.from_url(_redis_url(), decode_responses=False)
    redis_client.ping()
    registry = SegmentRegistry(redis_client)

    safe = Schema200Field.__name__.replace(".", "_")
    redis_client.delete(f"pyforge:schema:{safe}:current".encode())

    print(f"Populating Parquet dataset with {N_ENTITIES:,} entities ...", flush=True)
    t0 = time.monotonic()
    dataset_path = populate_dataset_for_hydration(
        n_entities=N_ENTITIES,
        schema=Schema200Field,
    )
    print(f"  populate done in {time.monotonic() - t0:.1f}s", flush=True)

    try:
        store = ParquetDatasetStore(
            dataset_path=dataset_path,
            schema=Schema200Field,
            flush_interval_seconds=3600,
        )
        try:
            print("Running hydrate() ...", flush=True)
            t1 = time.monotonic()
            result = asyncio.run(
                asyncio.to_thread(
                    hydrate,
                    Schema200Field,
                    store,
                    registry,
                    redis_client=redis_client,
                )
            )
            print(
                f"  hydrate() done in {time.monotonic() - t1:.1f}s "
                f"(entity_count={result.entity_count})",
                flush=True,
            )
        finally:
            asyncio.run(store.close())
    finally:
        try:
            current_key = f"pyforge:schema:{safe}:current".encode()
            seg_name_b = redis_client.get(current_key)
            if seg_name_b:
                import contextlib

                from pyforge._internal import posix_shm

                seg_name = seg_name_b.decode()
                redis_client.delete(current_key)
                with contextlib.suppress(FileNotFoundError):
                    posix_shm.unlink(seg_name)
        finally:
            shutil.rmtree(dataset_path, ignore_errors=True)
            redis_client.close()


if __name__ == "__main__":
    main()
