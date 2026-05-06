"""Chaos tests for ParquetDatasetStore: SIGKILL during flush.

The contract Step 11 must satisfy: every ``.parquet`` file under
``{base}/schema=*/event_date=*/`` is fully written and readable on
restart, regardless of when the consumer was SIGKILL'd. The ``_tmp/``
directory may contain partial files but they must (a) be unreachable
to dataset readers and (b) be cleaned up at the next ``__init__``.

Subprocess pattern mirrors ``test_wal_consumer_crash_safety.py``: fork
a child consumer, SIGKILL once ``pending_ack`` crosses a threshold,
then verify in the parent.
"""

from __future__ import annotations

import asyncio
import multiprocessing
import os
import signal
import sys
import time
from pathlib import Path

import pytest

pytestmark = [
    pytest.mark.chaos,
    pytest.mark.skipif(
        sys.platform == "win32",
        reason="chaos tests use POSIX fork() + SIGKILL",
    ),
]


def _mp_context() -> multiprocessing.context.BaseContext:
    return multiprocessing.get_context("fork")


# Step 15 stub registry for parent-process WALConsumer constructions in
# chaos tests. The pause-clear reopen branch isn't exercised here.
def _stub_registry() -> object:
    from typing import cast as _cast
    from unittest.mock import Mock as _Mock

    from quorin.shm import SegmentRegistry

    return _cast("SegmentRegistry", _Mock(spec=SegmentRegistry))


def _kill_after_n_apply_with_offline_subprocess(
    seg_name: str,
    schema_dotted: str,
    redis_url: str,
    consumer_name: str,
    base_path: str,
    kill_after_n: int,
) -> None:
    """Subprocess body. Module-level so fork() can pickle it."""
    import importlib
    from typing import cast as _cast
    from unittest.mock import Mock as _Mock

    import redis.asyncio

    from quorin._internal import posix_shm
    from quorin.layout import compute_layout_from_segment
    from quorin.offline import ParquetDatasetStore
    from quorin.shm import Segment, SegmentRegistry

    _stub_registry_local = _cast("SegmentRegistry", _Mock(spec=SegmentRegistry))
    from quorin.wal_consumer import WALConsumer

    mod_name, _, cls_name = schema_dotted.rpartition(".")
    schema = getattr(importlib.import_module(mod_name), cls_name)

    handle = posix_shm.open_existing(seg_name)
    layout_obj = compute_layout_from_segment(handle.buf, schema)
    seg = Segment(name=seg_name, schema=schema, handle=handle, layout=layout_obj)

    async def main() -> None:
        client = redis.asyncio.Redis.from_url(redis_url, decode_responses=False)
        offline = ParquetDatasetStore(base_path)
        consumer = WALConsumer(
            client,
            segments={schema.__name__: seg},
            registry=_stub_registry_local,
            offline=offline,
            consumer_name=consumer_name,
            block_ms=50,
            # Short flush interval so flushes actually happen during the
            # run — that way the SIGKILL is much more likely to land
            # mid-flush at least sometimes across the parametrized sweep.
            flush_interval_seconds=0.2,
            max_pending_ack=10_000,
        )

        async def killer() -> None:
            while True:
                if len(consumer._pending_ack) >= kill_after_n:
                    os.kill(os.getpid(), signal.SIGKILL)
                await asyncio.sleep(0.005)

        await asyncio.gather(consumer.run(), killer())

    asyncio.run(main())


# ---------------------------------------------------------------------------
# Top-level schema for the chaos subprocess to import.
# ---------------------------------------------------------------------------


from quorin.schema import FeatureField, FeatureSchema, dtype  # noqa: E402


class _ChaosOff(FeatureSchema):
    version = 1
    fields = [
        FeatureField("a", dtype.float32),
        FeatureField("b", dtype.int64),
    ]


# ---------------------------------------------------------------------------
# Helpers.
# ---------------------------------------------------------------------------


@pytest.fixture
def chaos_redis_url() -> str:
    return os.environ.get("QUORIN_REDIS_URL", "redis://127.0.0.1:6379/0")


def _all_finals_readable(base: Path) -> tuple[int, int]:
    """Return (file_count, total_rows). Raises on any unreadable file."""
    import pyarrow.parquet as pq

    files = sorted(base.rglob("*.parquet"))
    # Exclude any files that happen to be under `_tmp/` (those are
    # legitimately partial or never-finalised).
    files = [f for f in files if "_tmp" not in f.parts]
    total = 0
    for f in files:
        # Reading must not raise — the contract is "atomic rename means
        # what's in finals is fully written."
        table = pq.read_table(f)
        total += table.num_rows
    return len(files), total


def _run_one_offline_chaos_iteration(redis_url: str, base_path: Path, kill_after_n: int) -> None:
    """One iteration: produce → consumer-with-offline → SIGKILL → restart →
    verify final dir contents are all readable + PEL drains."""
    import redis as sync_redis
    import redis.asyncio

    from _helpers import make_segment, release_segment
    from quorin.offline import ParquetDatasetStore
    from quorin.wal import WALProducer
    from quorin.wal_consumer import WALConsumer

    client = sync_redis.Redis.from_url(redis_url, decode_responses=False)
    for k in client.scan_iter(match="quorin:*", count=1000):
        client.delete(k)

    seg = make_segment(_ChaosOff, capacity=128)
    try:
        producer = WALProducer(client)
        n_msgs = 100
        for i in range(n_msgs):
            producer.write(_ChaosOff, f"e-{i}", {"a": float(i), "b": i})

        proc = _mp_context().Process(
            target=_kill_after_n_apply_with_offline_subprocess,
            args=(
                seg.name,
                f"{_ChaosOff.__module__}.{_ChaosOff.__qualname__}",
                redis_url,
                "chaos-offline-consumer",
                str(base_path),
                kill_after_n,
            ),
        )
        proc.start()
        proc.join(timeout=20)
        if proc.is_alive():
            proc.kill()
            proc.join(timeout=5)
            pytest.fail("Subprocess didn't terminate")

        # Phase 1: every file under finals must be readable. The _tmp/
        # may contain orphans from a mid-write SIGKILL.
        _, _ = _all_finals_readable(base_path)

        # Phase 2: restart with a FRESH ParquetDatasetStore at the same
        # base. _cleanup_tmp_dir() runs at __init__ and drops any
        # partial files in _tmp/.
        offline_recover = ParquetDatasetStore(base_path)
        # Tmp dir is now empty (cleanup just ran).
        tmp_dir = base_path / "_tmp"
        leftover_in_tmp = list(tmp_dir.glob("*"))
        assert leftover_in_tmp == [], f"_tmp/ still has files after fresh init: {leftover_in_tmp}"

        # Lock cleanup — subprocess died holding it.
        for k in client.scan_iter(match="quorin:consumer:lock:*", count=100):
            client.delete(k)

        async def recover() -> None:
            ar = redis.asyncio.Redis.from_url(redis_url, decode_responses=False)
            try:
                c = WALConsumer(
                    ar,
                    segments={_ChaosOff.__name__: seg},
                    registry=_stub_registry(),  # type: ignore[arg-type]
                    offline=offline_recover,
                    consumer_name="chaos-offline-consumer",
                    block_ms=50,
                    flush_interval_seconds=0.5,
                    max_pending_ack=20,
                )
                run_task = asyncio.create_task(c.run())
                await asyncio.sleep(0)
                deadline = time.monotonic() + 12.0
                while time.monotonic() < deadline:
                    try:
                        info = await ar.xpending("quorin:wal", "quorin_consumers")
                        if info["pending"] == 0 and not c._pending_ack:
                            break
                    except Exception:
                        pass
                    await asyncio.sleep(0.05)
                await c.stop()
                await asyncio.wait_for(run_task, timeout=5.0)
            finally:
                await ar.aclose()

        asyncio.run(recover())

        # Phase 3: every final file remains readable; total row count
        # >= n_msgs (may exceed due to PEL replay duplicates, which
        # Step 12 dedups on read — that's expected and documented).
        _, total_rows = _all_finals_readable(base_path)
        assert total_rows >= n_msgs, f"recovery left only {total_rows}/{n_msgs} rows in finals"
        # PEL drained.
        info = client.xpending("quorin:wal", "quorin_consumers")
        assert info["pending"] == 0, f"PEL not drained after recovery: {info['pending']} remain"

    finally:
        release_segment(seg)
        client.close()


@pytest.mark.parametrize("kill_after_n", [30, 50, 70])
def test_sigkill_during_offline_apply_converges(
    chaos_redis_url: str, kill_after_n: int, tmp_path: Path
) -> None:
    """SIGKILL the consumer after applying N messages; restart; verify
    finals are all readable, ``_tmp/`` cleans up, and PEL drains."""
    import redis as sync_redis

    try:
        sync_redis.Redis.from_url(chaos_redis_url).ping()
    except Exception:
        pytest.skip("Redis not reachable")

    _run_one_offline_chaos_iteration(chaos_redis_url, tmp_path, kill_after_n)
