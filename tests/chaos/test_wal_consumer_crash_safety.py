"""Chaos tests for the WAL consumer: SIGKILL during apply.

The contract Step 10 must satisfy (ADR-009 §2): SIGKILL at any phase
of the apply path leaves the system in a state from which a restart
converges to "all unique entities in shm, PEL fully drained, no
duplicates."

These tests use ``multiprocessing.fork`` so the subprocess inherits
the parent's POSIX shm segment, the producer's pre-loaded messages
in Redis, and the consumer module's compiled state. SIGKILL is
synchronous (the subprocess kills itself once it has applied N
messages), and the parent measures convergence by running a fresh
consumer to drain whatever the killed one left behind.

Marked ``@pytest.mark.chaos`` so it stays out of the default ``-q``
suite — chaos tests are run on demand and in the benchmark workflow.
"""

from __future__ import annotations

import asyncio
import multiprocessing
import os
import random
import signal
import sys
import time

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


# Step 15 stub registry — chaos tests don't trigger pause-clear reopen.
def _stub_registry() -> object:
    from typing import cast as _cast
    from unittest.mock import Mock as _Mock

    from pyforge.shm import SegmentRegistry

    return _cast("SegmentRegistry", _Mock(spec=SegmentRegistry))


def _kill_after_n_apply_subprocess(
    seg_name: str,
    schema_dotted: str,
    redis_url: str,
    consumer_name: str,
    kill_after_n: int,
) -> None:
    """Subprocess body. Must be a module-level function so fork() can pickle it.

    Opens the existing segment by name, runs the consumer, SIGKILLs once
    pending_ack reaches ``kill_after_n``.
    """
    import importlib
    from typing import cast as _cast
    from unittest.mock import Mock as _Mock

    import redis.asyncio

    from pyforge import layout  # noqa: F401  (ensures import side-effects)
    from pyforge._internal import posix_shm
    from pyforge.layout import compute_layout_from_segment
    from pyforge.shm import HEADER_LEN, Segment
    from pyforge.wal_consumer import WALConsumer

    # Resolve schema.
    mod_name, _, cls_name = schema_dotted.rpartition(".")
    schema = getattr(importlib.import_module(mod_name), cls_name)

    # Attach to segment by name (parent created it).
    handle = posix_shm.open_existing(seg_name)
    layout_obj = compute_layout_from_segment(handle.buf, schema)
    seg = Segment(name=seg_name, schema=schema, handle=handle, layout=layout_obj)
    del HEADER_LEN  # silence import-only warning

    async def main() -> None:
        from pyforge.shm import SegmentRegistry

        _stub_reg = _cast("SegmentRegistry", _Mock(spec=SegmentRegistry))
        client = redis.asyncio.Redis.from_url(redis_url, decode_responses=False)
        consumer = WALConsumer(
            client,
            segments={schema.__name__: seg},
            registry=_stub_reg,
            consumer_name=consumer_name,
            block_ms=50,
            flush_interval_seconds=300.0,
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
# Shared schema for chaos tests. Top-level so the subprocess can import.
# ---------------------------------------------------------------------------


from pyforge.schema import FeatureField, FeatureSchema, dtype  # noqa: E402


class _ChaosCS(FeatureSchema):
    version = 1
    fields = [
        FeatureField("a", dtype.float32),
        FeatureField("b", dtype.int64),
    ]


# ---------------------------------------------------------------------------
# Fixtures.
# ---------------------------------------------------------------------------


@pytest.fixture
def chaos_redis_url() -> str:
    return os.environ.get("PYFORGE_REDIS_URL", "redis://127.0.0.1:6379/0")


# ---------------------------------------------------------------------------
# Baseline chaos test: 100 entities, SIGKILL at random N in [30, 70],
# restart consumer, verify convergence.
# ---------------------------------------------------------------------------


def _run_one_chaos_iteration(redis_url: str, kill_after_n: int) -> None:
    """One iteration: setup + subprocess kill + restart + assert convergence."""
    import redis as sync_redis
    import redis.asyncio

    from _helpers import make_segment, release_segment
    from pyforge.layout import lookup
    from pyforge.wal import WALProducer
    from pyforge.wal_consumer import WALConsumer

    # Fresh stream + group + segment for each iteration.
    client = sync_redis.Redis.from_url(redis_url, decode_responses=False)
    # Scrub previous-iteration leftovers.
    for k in client.scan_iter(match="pyforge:*", count=1000):
        client.delete(k)

    seg = make_segment(_ChaosCS, capacity=128)

    try:
        # Producer side: write 100 distinct entities.
        producer = WALProducer(client)
        for i in range(100):
            producer.write(_ChaosCS, f"e-{i}", {"a": float(i), "b": i})

        # Subprocess consumer: applies up to kill_after_n then SIGKILLs.
        proc = _mp_context().Process(
            target=_kill_after_n_apply_subprocess,
            args=(
                seg.name,
                f"{_ChaosCS.__module__}.{_ChaosCS.__qualname__}",
                redis_url,
                "chaos-consumer",
                kill_after_n,
            ),
        )
        proc.start()
        proc.join(timeout=20)
        # Subprocess SHOULD have died via SIGKILL (or finished if kill timing
        # was at/past the total). Either is acceptable.
        if proc.is_alive():
            proc.kill()
            proc.join(timeout=5)
            pytest.fail("Subprocess didn't terminate")

        # Lock cleanup — subprocess died holding it; clear so the recovery
        # consumer can start.
        for k in client.scan_iter(match="pyforge:consumer:lock:*", count=100):
            client.delete(k)

        # Recovery consumer in parent: drain remaining PEL + new msgs (none).
        async def recover() -> None:
            ar = redis.asyncio.Redis.from_url(redis_url, decode_responses=False)
            try:
                c = WALConsumer(
                    ar,
                    segments={_ChaosCS.__name__: seg},
                    registry=_stub_registry(),  # type: ignore[arg-type]
                    consumer_name="chaos-consumer",
                    block_ms=50,
                    flush_interval_seconds=10.0,
                    max_pending_ack=5,
                )
                run_task = asyncio.create_task(c.run())
                await asyncio.sleep(0)
                deadline = time.monotonic() + 10.0
                while time.monotonic() < deadline:
                    try:
                        info = await ar.xpending("pyforge:wal", "pyforge_consumers")
                        if info["pending"] == 0:
                            break
                    except Exception:
                        pass
                    await asyncio.sleep(0.05)
                await c.stop()
                await asyncio.wait_for(run_task, timeout=5.0)
            finally:
                await ar.aclose()

        asyncio.run(recover())

        # Assertions: all 100 entities in shm; PEL drained.
        for i in range(100):
            assert lookup(seg, f"e-{i}") is not None, (
                f"entity e-{i} missing after kill_after_n={kill_after_n}"
            )
        info = client.xpending("pyforge:wal", "pyforge_consumers")
        assert info["pending"] == 0, f"PEL not drained after recovery: {info['pending']} remain"

    finally:
        release_segment(seg)
        client.close()


@pytest.mark.parametrize("kill_after_n", [30, 50, 70])
def test_sigkill_during_apply_converges(chaos_redis_url: str, kill_after_n: int) -> None:
    """SIGKILL the consumer after applying N messages; restart; verify
    all 100 entities land in shm and PEL drains. Three kill timings
    cover early / mid / late termination."""
    # Skip if Redis unreachable.
    import redis as sync_redis

    try:
        sync_redis.Redis.from_url(chaos_redis_url).ping()
    except Exception:
        pytest.skip("Redis not reachable")

    _run_one_chaos_iteration(chaos_redis_url, kill_after_n)


@pytest.mark.slow
def test_sigkill_repeated_iterations_all_converge(
    chaos_redis_url: str,
) -> None:
    """Repeat the chaos iteration with random kill timings; all converge.

    Spec acceptance: 20/20 runs converge to the same final state. Marked
    slow so the default suite stays fast — this is the contract test
    that validates ADR-009 §2.
    """
    import redis as sync_redis

    try:
        sync_redis.Redis.from_url(chaos_redis_url).ping()
    except Exception:
        pytest.skip("Redis not reachable")

    rng = random.Random(20260429)
    n_runs = 20
    for _run in range(n_runs):
        kill_n = rng.randint(30, 70)
        _run_one_chaos_iteration(chaos_redis_url, kill_n)
