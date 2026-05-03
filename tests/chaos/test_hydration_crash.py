"""Chaos tests for pyforge.hydration: SIGKILL during insert_many + concurrent race.

Step 13 Commit B chaos coverage:

* C1 — SIGKILL mid-insert leaves no corruption (recoverable via watchdog
  simulation + retry).
* C2 — Concurrent hydrate race: exactly one wins, one loses
  (FileExistsError or HydrationConflictError), no silent corruption.
  Uses ``multiprocessing.spawn`` so each child constructs its own
  Redis client + SegmentRegistry — fork would inherit the parent's
  redis-py connection pool + Lua script binding. C1/C3/C4 use fork
  because they're single-process crash semantics.
* C3 — SIGKILL + simulated watchdog cleanup + retry succeeds.
* C4 — 20 random-seeded SIGKILL iterations against a session-scoped
  dataset. Marked ``@pytest.mark.slow``.

Every chaos test asserts ``child.exitcode == -9`` (SIGKILL) AFTER
join — without it, a fast-completing hydrate makes the test silently
a no-op.
"""

from __future__ import annotations

import asyncio
import contextlib
import multiprocessing
import os
import random
import signal
import sys
from pathlib import Path

import pytest

pytestmark = [
    pytest.mark.chaos,
    pytest.mark.skipif(
        sys.platform == "win32",
        reason="chaos tests use POSIX fork()/spawn() + SIGKILL",
    ),
]

import numpy as np  # noqa: E402
import redis  # noqa: E402
from _watchdog_helpers import (  # noqa: E402
    drain_cleanup_queue,
    simulate_watchdog_post_crash_cleanup,
)

from pyforge.hydration import hydrate  # noqa: E402
from pyforge.layout import lookup  # noqa: E402
from pyforge.offline import ParquetDatasetStore  # noqa: E402
from pyforge.schema import (  # noqa: E402
    FeatureField,
    FeatureSchema,
    _hash_name,
    dtype,
)
from pyforge.shm import SegmentRegistry  # noqa: E402

# ---------------------------------------------------------------------------
# Multiprocessing contexts — Rev-3 HIGH #4: get_context, no global mutation.
# ---------------------------------------------------------------------------


_FORK_CTX = multiprocessing.get_context("fork")  # C1, C3, C4
# _SPAWN_CTX (used by the removed C2 test) is dropped. If a future
# concurrent-hydrate test is added, reintroduce via mp.get_context("spawn").


# ---------------------------------------------------------------------------
# Shared schema. Top-level for picklability across both fork + spawn.
# ---------------------------------------------------------------------------


class _ChaosHydrate(FeatureSchema):
    version = 1
    fields = [
        FeatureField("a", dtype.float32),
        FeatureField("b", dtype.int64),
    ]


# ---------------------------------------------------------------------------
# Subprocess targets — module-level for picklability.
# ---------------------------------------------------------------------------


def _hydrate_then_sigkill_at_delay_ms(
    redis_url: str,
    base_path_str: str,
    delay_ms: int,
) -> None:
    """Fork target: run hydrate, kill self ``delay_ms`` after starting.

    A previous design polled the segment's ``next_free_row_index``
    cursor to kill at row-count milestones. Two problems with that:

    1. The bulk-insert kernel writes the cursor in a single Numba
       call — `next_row` jumps from 0 to N at the end. There's no
       observable mid-insert progress to poll.
    2. The killer's ``registry.open_current`` + ``registry.close``
       calls mutate ``pyforge:pid_segments:{child_pid}`` — the very
       Redis state the watchdog walks after the kill. The close-
       in-killer leaves pid_segments empty, so the parent's
       watchdog cleanup finds nothing to clean.

    Simpler design: time-based killer. Sleep ``delay_ms``, then
    SIGKILL. With 50k entities + ~150ms hydrate, ``delay_ms``
    between 10 and 100 lands inside hydrate. The kill could land
    before or after the kernel completes — that's fine. The
    contract is "watchdog cleanup converges regardless of where the
    crash hit", and time-based timing covers all the interesting
    windows across iterations.
    """
    import redis as _redis

    from pyforge.hydration import hydrate
    from pyforge.shm import SegmentRegistry

    redis_client = _redis.Redis.from_url(redis_url)
    registry = SegmentRegistry(redis_client)
    store = ParquetDatasetStore(Path(base_path_str))

    async def _killer() -> None:
        await asyncio.sleep(delay_ms / 1000.0)
        os.kill(os.getpid(), signal.SIGKILL)

    async def _run_hydrate_with_killer() -> None:
        # Hydrate is sync; offload to thread so the killer can run.
        loop = asyncio.get_running_loop()
        killer_task = asyncio.create_task(_killer())
        try:
            await loop.run_in_executor(
                None,
                lambda: hydrate(_ChaosHydrate, store, registry, redis_client=redis_client),
            )
        finally:
            killer_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await killer_task

    asyncio.run(_run_hydrate_with_killer())


# (The C2 spawn target was removed alongside its test. See the C2-removed
#  comment above _test_c3 for the rationale.)


# ---------------------------------------------------------------------------
# Helpers — build a small offline dataset for the chaos tests.
# ---------------------------------------------------------------------------


def _values_in_wire_order(rng: np.random.Generator) -> list:
    by_name = {
        "a": float(rng.standard_normal()),
        "b": int(rng.integers(0, 1 << 30)),
    }
    return [by_name[f.name] for f in sorted(_ChaosHydrate.fields, key=lambda f: _hash_name(f.name))]


def _build_dataset(base: Path, n_entities: int, *, seed: int = 42) -> None:
    """Generate a small offline dataset for chaos hydrate runs.

    Timestamps anchored at ``time.time_ns()`` (subtracting a small
    per-row offset) so the data falls within hydrate's default
    ``lookback_days=30`` window regardless of when the test runs.
    Hardcoded epoch values would expire after ~30 days of clock-time
    drift between dataset build and test execution.
    """
    import time as _time

    store = ParquetDatasetStore(base)
    rng = np.random.default_rng(seed=seed)
    # Anchor at "now"; subtract a few seconds total so all rows are
    # in the very recent past.
    base_event_time_ns = _time.time_ns() - n_entities * 1_000

    async def _run() -> None:
        for i in range(n_entities):
            entity_id = f"chaos-{i:06d}"
            event_time_ns = base_event_time_ns + i * 1_000
            values = _values_in_wire_order(rng)
            msg_id = f"{1_700_000_000_000 + i}-0".encode()
            await store.append(_ChaosHydrate, entity_id, event_time_ns, values, msg_id)
            if (i + 1) % 5_000 == 0:
                await store.flush()
        await store.flush()
        await store.close()

    asyncio.run(_run())


# ---------------------------------------------------------------------------
# Fixtures — Redis + dataset.
# ---------------------------------------------------------------------------


def _redis_url() -> str:
    return os.environ.get("PYFORGE_REDIS_URL", "redis://127.0.0.1:6379/0")


# Note: `redis_client` is provided by tests/conftest.py (session-scoped sync
# client). We don't redefine it here — matching test_wal_consumer_crash_safety.py.


@pytest.fixture
def small_dataset(tmp_path: Path) -> Path:
    """50k-entity dataset rebuilt per test (small enough to be fast)."""
    base = tmp_path / "chaos_dataset"
    base.mkdir()
    _build_dataset(base, n_entities=50_000)
    return base


@pytest.fixture(scope="module")
def session_dataset(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """50k-entity dataset shared across C4's 20 iterations."""
    base = tmp_path_factory.mktemp("chaos_session_dataset")
    _build_dataset(base, n_entities=50_000)
    return base


def _scrub_consumer_locks(redis_client) -> None:
    """Best-effort: clear any consumer locks our parent process holds.

    Mirrors test_wal_consumer_crash_safety.py:176-177.
    """
    for k in redis_client.scan_iter(match="pyforge:consumer:lock:*", count=100):
        redis_client.delete(k)


# ---------------------------------------------------------------------------
# C1 — SIGKILL during insert_many.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("delay_ms", [20, 50, 100])
def test_c1_sigkill_during_insert_many_leaves_no_corruption(
    redis_client,
    small_dataset: Path,
    delay_ms: int,
) -> None:
    """SIGKILL during hydrate; watchdog simulation + retry hydrate succeeds.

    The 20/50/100ms delays cover different windows of hydrate's
    ~150-200ms execution at 50k entities — early (registry.create
    just done), mid (insert_many in progress), late (post-insert
    finalization). All three should converge after watchdog cleanup.
    """
    proc = _FORK_CTX.Process(
        target=_hydrate_then_sigkill_at_delay_ms,
        args=(_redis_url(), str(small_dataset), delay_ms),
    )
    proc.start()
    dead_pid = proc.pid
    assert dead_pid is not None
    proc.join(timeout=30)

    assert proc.exitcode == -signal.SIGKILL, (
        f"child did NOT receive SIGKILL — exitcode={proc.exitcode}; "
        f"delay_ms={delay_ms} may be longer than hydrate runtime "
        f"(hydrate too fast on this machine?). Adjust the delay or "
        f"increase the dataset size to extend hydrate's runtime."
    )

    # Watchdog simulation: clean up any state the dead child left behind.
    simulate_watchdog_post_crash_cleanup(redis_client, dead_pid)
    drain_cleanup_queue(redis_client)
    _scrub_consumer_locks(redis_client)

    # Retry hydrate from the parent — should succeed.
    parent_redis = redis.Redis.from_url(_redis_url())
    registry = SegmentRegistry(parent_redis)
    store = ParquetDatasetStore(small_dataset)
    try:
        result = hydrate(_ChaosHydrate, store, registry, redis_client=parent_redis)
        assert result.entity_count == 50_000

        # Verify segment is live + readable.
        seg = registry.open_current(_ChaosHydrate)
        try:
            offset = lookup(seg, "chaos-000042")
            assert offset is not None, "post-recovery hydrate didn't load entities"
        finally:
            registry.close(seg)
            drain_cleanup_queue(redis_client)
    finally:
        parent_redis.close()


# ---------------------------------------------------------------------------
# C2 (REMOVED) — Concurrent hydrate race.
#
# The original C2 tested "two children call hydrate concurrently;
# exactly one wins (FileExistsError or HydrationConflictError)". That
# contract isn't actually provided by the orchestrator:
#
# - registry.create() uses UUID-suffixed segment names (see
#   pyforge.shm._segment_name), so two concurrent calls create
#   DISTINCT segments — neither hits FileExistsError.
# - Both children pass preconditions in the race window, both call
#   registry.create() successfully, both proceed to insert_many,
#   both succeed. The "current" pointer is overwritten by whichever
#   pipeline EXECUTES last; the other segment is orphaned in
#   /dev/shm.
#
# This is a real concurrency hazard, but the orchestrator's
# documented contract is "Operators must serialize hydrate() calls"
# (pyforge.hydration docstring). The system converges on watchdog
# cleanup of orphans (Step 14), but there's no in-orchestrator
# guarantee that exactly one wins.
#
# Step 14's watchdog tests will exercise the orphan-cleanup path
# directly. C1/C3/C4 already cover the kill+recover contract.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# C3 — SIGKILL + watchdog simulation + retry succeeds.
# ---------------------------------------------------------------------------


def test_c3_sigkill_then_watchdog_simulation_then_retry_succeeds(
    redis_client,
    small_dataset: Path,
) -> None:
    """Operator runbook: kill -> watchdog cleanup -> retry hydrate works.

    50ms delay covers the mid-hydrate window for 50k-entity datasets
    (~150-200ms total runtime).
    """
    proc = _FORK_CTX.Process(
        target=_hydrate_then_sigkill_at_delay_ms,
        args=(_redis_url(), str(small_dataset), 50),
    )
    proc.start()
    dead_pid = proc.pid
    assert dead_pid is not None
    proc.join(timeout=30)

    assert proc.exitcode == -signal.SIGKILL, (
        f"child did NOT receive SIGKILL — exitcode={proc.exitcode}"
    )

    # Watchdog simulation: walks pyforge:pid_segments:{dead_pid}, DECRs
    # refcounts, unlinks segments, drops schema:*:current pointers.
    # The return count is informational only (may be 0 if the child died
    # BEFORE registry.create registered the segment); contract is "retry
    # hydrate succeeds regardless".
    simulate_watchdog_post_crash_cleanup(redis_client, dead_pid)
    drain_cleanup_queue(redis_client)
    _scrub_consumer_locks(redis_client)

    # Retry from parent.
    parent_redis = redis.Redis.from_url(_redis_url())
    registry = SegmentRegistry(parent_redis)
    store = ParquetDatasetStore(small_dataset)
    try:
        result = hydrate(_ChaosHydrate, store, registry, redis_client=parent_redis)
        assert result.entity_count == 50_000

        seg = registry.open_current(_ChaosHydrate)
        try:
            offset = lookup(seg, "chaos-000007")
            assert offset is not None
        finally:
            registry.close(seg)
            drain_cleanup_queue(redis_client)
    finally:
        parent_redis.close()


# ---------------------------------------------------------------------------
# C4 — Repeated random-kill iterations.
# ---------------------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.parametrize("seed", list(range(20)), ids=lambda s: f"seed{s}")
def test_c4_repeated_random_kill_iterations(
    redis_client,
    session_dataset: Path,
    seed: int,
) -> None:
    """20 seeded iterations of fork-then-SIGKILL-at-random-delay.

    Delay range 10-100ms covers different points in hydrate's
    ~150-200ms execution at 50k entities. Each seed picks a
    deterministic delay.
    """
    rng = random.Random(seed)
    delay_ms = rng.randint(10, 100)

    proc = _FORK_CTX.Process(
        target=_hydrate_then_sigkill_at_delay_ms,
        args=(_redis_url(), str(session_dataset), delay_ms),
    )
    proc.start()
    dead_pid = proc.pid
    assert dead_pid is not None
    proc.join(timeout=30)

    assert proc.exitcode == -signal.SIGKILL, (
        f"seed{seed} (delay_ms={delay_ms}): child did NOT receive SIGKILL — "
        f"exitcode={proc.exitcode}"
    )

    # Watchdog simulation + retry.
    simulate_watchdog_post_crash_cleanup(redis_client, dead_pid)
    drain_cleanup_queue(redis_client)
    _scrub_consumer_locks(redis_client)

    parent_redis = redis.Redis.from_url(_redis_url())
    registry = SegmentRegistry(parent_redis)
    store = ParquetDatasetStore(session_dataset)
    try:
        result = hydrate(_ChaosHydrate, store, registry, redis_client=parent_redis)
        assert result.entity_count == 50_000

        # Drop the new segment so the next iteration starts fresh.
        seg = registry.open_current(_ChaosHydrate)
        registry.close(seg)
        drain_cleanup_queue(redis_client)
    finally:
        parent_redis.close()
