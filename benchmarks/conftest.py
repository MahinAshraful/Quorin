"""pytest-benchmark fixtures.

Session-scoped fixtures land here so heavy setup (cold-cache clobber array,
watchdog subprocess, WAL consumer thread) is paid once per `pytest benchmarks/`
invocation, not per bench. Bench files import via fixture name and let pytest
wire dependency order.

Step 16 added (per the plan):
  * ``cold_cache_clobber`` — L3-sized clobber array for honest cold-CPU-cache
    measurements (P3 + C3).
  * ``running_watchdog`` — background ``pyforge.watchdog`` subprocess that
    drains ``pyforge:cleanup_queue`` so /dev/shm doesn't leak across bench
    runs (E1).
  * ``running_consumer_50_field`` — real ``WALConsumer`` thread for the
    end-to-end ``write_sync`` RTT bench, with ``NoopOfflineWriter`` so we
    measure online-store durability (ADR-009 §3) and not offline flush.
"""

from __future__ import annotations

import asyncio
import contextlib
import gc
import logging
import os
import subprocess
import sys
import threading
import time
from collections.abc import Iterator
from typing import Any

import numpy as np
import pytest
import redis

LOGGER = logging.getLogger(__name__)


@pytest.fixture
def isolate_gc() -> Iterator[None]:
    """Disable GC for the duration of a single benchmark."""
    gc.collect()
    gc.disable()
    try:
        yield
    finally:
        gc.enable()


def _redis_url() -> str:
    return os.environ.get("PYFORGE_REDIS_URL", "redis://127.0.0.1:6379/0")


@pytest.fixture(scope="session")
def redis_client() -> Iterator[redis.Redis]:
    """Live Redis client for Redis-dependent benchmarks (Step 9 WAL).

    Mirrors the fixture in ``tests/conftest.py``. pytest does not share
    conftest fixtures across sibling directories (``tests/`` vs
    ``benchmarks/``), so the definition is duplicated here. Skips
    benchmarks that depend on it if Redis is unreachable.
    """
    client = redis.Redis.from_url(_redis_url(), decode_responses=False)
    try:
        client.ping()
    except redis.ConnectionError:
        pytest.skip("Redis is not reachable — start docker/docker-compose.dev.yml")
    yield client
    client.close()


# ---------------------------------------------------------------------------
# Step 16: cold-cache clobber array (P3 + C3)
#
# Step 16b Rev-4 H1 cascade: ``detect_l3_size_bytes`` lives in
# ``benchmarks/_cache_clobber.py`` so the standalone MADV A/B orchestrator
# (``benchmarks/runs/step16_madvise_ab.py``) can use the same primitive
# without going through pytest fixture machinery.
# ---------------------------------------------------------------------------

from benchmarks._cache_clobber import detect_l3_size_bytes  # noqa: E402


@pytest.fixture(scope="session")
def cold_cache_clobber() -> tuple[np.ndarray, int]:
    """Session-scoped clobber array sized at 4x detected L3 (capped at 1 GB).

    Returned tuple is ``(clobber_array, size_bytes)`` so benches can record the
    actual size in pytest-benchmark's ``extra_info``. Per ADR-015, this measures
    cold-CPU-cache, NOT cold-page-cache (segment is still resident in /dev/shm
    tmpfs).

    On WSL2 / VMs that hide the cache hierarchy from sysfs, this trips the
    16 MiB fallback and logs a WARN. That's the C3 fix working as intended;
    don't suppress the warning. Ubuntu CI runners surface real L3.
    """
    if sys.platform != "linux":
        pytest.skip("cold-cache fixture requires Linux sysfs")
    size_bytes = min(4 * detect_l3_size_bytes(), 1 * 1024 * 1024 * 1024)
    n_doubles = size_bytes // 8
    clobber = np.empty(n_doubles, dtype=np.float64)
    LOGGER.info("cold-cache: clobber size = %.1f MiB", size_bytes / (1024 * 1024))
    return clobber, size_bytes


# ---------------------------------------------------------------------------
# Step 16: background watchdog subprocess (E1)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def running_watchdog(redis_client: redis.Redis) -> Iterator[subprocess.Popen[bytes]]:
    """Background ``pyforge.watchdog`` subprocess for hydration / write_sync benches.

    Without this, /dev/shm leaks across bench runs: refcount=0 segments stay
    in ``pyforge:cleanup_queue`` until something drains them. First run passes;
    subsequent runs may fail with "segment exists" or eventually fill /dev/shm.

    Cadence (LOCKED at production default per ADR-013): ``--tick-interval-seconds 30``
    + ``--miss-threshold 5`` = 150s detection. Earlier Step 16a draft used
    chaos-test cadence (2s/2 = ~5s detection); that broke the evolution
    bench because ``_populate_old`` loops 10k CPU-bound ``layout.insert``
    calls that hold the GIL long enough to starve the heartbeat thread for
    >4s, the watchdog declared the bench-process "dead", and the cleanup-
    Lua DEL'd ``pyforge:schema:{name}:current`` mid-bench. The 150s
    production cadence keeps the watchdog functional for catching real
    crashes between bench files (which is the only realistic dead-PID
    case in a bench session) without spuriously reaping live benches.

    Drain math (the original concern): tick 30s x cleanup_queue batch 100
    = ~3.3 segments/sec drain capacity. Hydration at 100k entities closes
    1 segment per hydrate, so this is still fine; bursts up to 100 segments
    drain in ~30s. Future Tier-2 scenario with >100 segment-closes/30s
    would need cadence retuning AND bench discipline to not collide with
    detection windows.

    C4: ``start_new_session=True`` so a Ctrl-C in pytest doesn't propagate
    SIGINT to the watchdog via shared process group. The fixture's
    ``terminate()`` in cleanup is the only authorized way to stop it.

    C5: skipped on non-Linux (POSIX shm dependency).
    """
    if sys.platform != "linux":
        pytest.skip("running_watchdog fixture requires Linux POSIX shm")
    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "pyforge.watchdog",
            "--redis",
            _redis_url(),
            "--tick-interval-seconds",
            "30",
            "--miss-threshold",
            "5",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,  # C4
    )
    # Watchdog has no readiness probe; 1s of slack is empirically adequate
    # after Step 14 (first tick at 2s interval; we just need ensure_started()
    # + the first HGETALL to land).
    time.sleep(1.0)
    if proc.poll() is not None:
        # Watchdog exited immediately — surface the error.
        raise RuntimeError(f"running_watchdog: process exited rc={proc.returncode}")
    yield proc
    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()


# ---------------------------------------------------------------------------
# Step 16: real WAL consumer thread for end-to-end write_sync RTT bench
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def running_consumer_50_field(
    redis_client: redis.Redis,
    running_watchdog: subprocess.Popen[bytes],
) -> Iterator[dict[str, Any]]:
    """Real ``WALConsumer`` running in a background asyncio thread.

    Uses ``NoopOfflineWriter`` per ADR-009 §3 — write_sync unblocks at
    online-store durability, NOT offline. Wiring a real ``OfflineWriter``
    here would conflate two contracts: the consumer-cycle latency we want
    to measure vs the offline-flush latency that's separately gated.

    Returns a dict with the wiring the bench needs to construct its producer:
        schema:       the 50-field FeatureSchema class
        redis_client: the sync Redis client (caller's responsibility to close)
        segment:      the Segment the consumer is reading INTO (refcount held
                      by the consumer; bench doesn't close)
        registry:     the SegmentRegistry that opened ``segment``

    Lifecycle: the consumer thread starts here, runs until session teardown.
    The bench's producer writes to the same Redis Stream the consumer reads;
    ``write_sync`` polls ``pyforge:processed:{msg_id}`` which the consumer
    sets after ``layout.insert`` completes (i.e. online-store durability).
    """
    if sys.platform != "linux":
        pytest.skip("running_consumer_50_field requires Linux POSIX shm")

    # Defer heavy imports until we know we're on Linux + Redis is reachable.
    import redis.asyncio as redis_async

    from pyforge.schema import FeatureField, FeatureSchema, dtype
    from pyforge.shm import SegmentRegistry
    from pyforge.wal_consumer import NoopOfflineWriter, WALConsumer

    # Distinct schema class so we don't collide with other benches' segments
    # in /dev/shm.
    fields = (
        [FeatureField(f"f{i:02d}", dtype.float32) for i in range(40)]
        + [FeatureField(f"i{i:02d}", dtype.int64) for i in range(8)]
        + [FeatureField("emb", dtype.float32, shape=(16,))]
        + [FeatureField("flags", dtype.uint8, shape=(4,))]
    )
    schema = type("_S50Bench", (FeatureSchema,), {"version": 1, "fields": fields})

    # Distinct stream + group so we don't collide with other tests' WAL state.
    # WALConsumer takes stream_key as bytes but group_name/consumer_name as str.
    bench_stream = b"pyforge:wal:bench:rtt:50f"
    bench_group = "pyforge_bench_rtt_50f_consumers"

    # Drain any stale state from previous bench runs (best-effort).
    redis_client.delete(bench_stream)
    with contextlib.suppress(redis.ResponseError):
        redis_client.execute_command("XGROUP", "DESTROY", bench_stream, bench_group)

    registry = SegmentRegistry(redis_client)
    segment = registry.create(schema, capacity=128)

    # Async consumer needs its own asyncio.Redis client (sync + async clients
    # are NOT pool-shareable per ADR-009 §11).
    loop = asyncio.new_event_loop()
    async_client = redis_async.Redis.from_url(_redis_url(), decode_responses=False)

    consumer = WALConsumer(
        async_client,
        segments={schema.__name__: segment},
        offline=NoopOfflineWriter(),
        registry=registry,
        stream_key=bench_stream,
        group_name=bench_group,
        # Tight cycles for low-latency bench. Default is 5s flush; we want the
        # consumer to drain pending+ack quickly so write_sync's polling doesn't
        # buffer round-trips.
        flush_interval_seconds=0.05,
        block_ms=10,
    )

    schemas = {schema.__name__.encode(): schema}
    consumer._schemas = schemas  # type: ignore[attr-defined] — bench-only test hook

    runner_ready = threading.Event()
    runner_exc: list[BaseException] = []

    def _run_consumer() -> None:
        asyncio.set_event_loop(loop)
        try:
            runner_ready.set()
            loop.run_until_complete(consumer.run())
        except BaseException as e:
            runner_exc.append(e)
        finally:
            loop.run_until_complete(async_client.aclose())
            loop.close()

    thread = threading.Thread(target=_run_consumer, daemon=True, name="bench-consumer")
    thread.start()
    runner_ready.wait(timeout=2.0)
    # Give the consumer one cycle to register the group + write its liveness key.
    time.sleep(0.5)

    yield {
        "schema": schema,
        "redis_client": redis_client,
        "segment": segment,
        "registry": registry,
        "stream_key": bench_stream,
    }

    # Teardown: stop consumer cleanly via its asyncio stop_event.
    # RuntimeError suppressed: loop already closed (shouldn't happen in normal teardown).
    with contextlib.suppress(RuntimeError):
        loop.call_soon_threadsafe(consumer._stop_event.set)
    thread.join(timeout=10)
    if runner_exc:
        # Surface consumer errors but don't fail the session — fixture cleanup
        # runs after benches; the bench has already produced its measurement.
        LOGGER.warning("running_consumer_50_field: consumer thread raised %r", runner_exc[0])
    with contextlib.suppress(Exception):
        registry.close(segment)
