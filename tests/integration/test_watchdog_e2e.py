"""Integration tests for the Step 14 watchdog.

End-to-end against real Redis, real fork()ed worker subprocesses, real
heartbeat threads. Driver constructs :class:`WatchdogState` directly and
calls :meth:`run_one_tick` (no subprocess CLI here — that's the chaos
suite). Pattern follows MEDIUM-Rev3 #4: tests force
``_tracked[pid].miss_count = state._miss_threshold`` directly to skip
the natural N-tick wait for threshold.
"""

from __future__ import annotations

import multiprocessing
import os
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
import redis

pytestmark = [
    pytest.mark.skipif(
        sys.platform == "win32",
        reason="watchdog requires POSIX (Linux/WSL2)",
    ),
    pytest.mark.integration,
]


from quorin._internal import heartbeat  # noqa: E402
from quorin.schema import FeatureField, FeatureSchema, dtype  # noqa: E402
from quorin.shm import KEY_SEGMENT_TO_SCHEMA, SegmentRegistry, _key_current  # noqa: E402
from quorin.watchdog import WatchdogState  # noqa: E402

if TYPE_CHECKING:
    from collections.abc import Iterator


# ---------------------------------------------------------------------------
# Test schema. Top-level for picklability + reuse across tests.
# ---------------------------------------------------------------------------


class _WdSchema(FeatureSchema):
    version = 1
    fields = [
        FeatureField("a", dtype.float32),
        FeatureField("b", dtype.int32),
    ]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def registry(redis_client: redis.Redis) -> SegmentRegistry:
    return SegmentRegistry(redis_client)


@pytest.fixture(autouse=True)
def _stop_heartbeat_after_each() -> Iterator[None]:
    import contextlib

    yield
    with contextlib.suppress(Exception):
        heartbeat.stop()


# ---------------------------------------------------------------------------
# E1 — live process never declared dead.
# ---------------------------------------------------------------------------


def test_e1_live_process_never_declared_dead(redis_client: redis.Redis) -> None:
    """Real heartbeat thread in this test process. Drive
    ``run_one_tick`` 6 times back-to-back (no sleep). Test process's
    pid must NOT appear in dead_pids — the heartbeat thread advances
    wall_time_ns faster than ticks consume it.

    Constructing ``WatchdogState`` itself calls
    ``heartbeat.ensure_started`` for self_pid, so the heartbeat hash
    is populated. The watchdog's self_pid exclusion in step 2 prevents
    self-tracking; this test verifies that contract via the test_pid
    being the watchdog's self_pid (single-process integration).
    """
    state = WatchdogState(redis_client, miss_threshold=2)

    for _ in range(6):
        result = state.run_one_tick()
        assert os.getpid() not in result.dead_pids
        # Self-PID is excluded from _tracked so the cross-check never fires.
        assert os.getpid() not in state._tracked


# ---------------------------------------------------------------------------
# E2 — dead PID full cleanup via real fork + SIGKILL.
# ---------------------------------------------------------------------------


def _e2_child_target(redis_url: str, ready_path: str, schema_class: type[FeatureSchema]) -> None:
    """Child process target. Creates a real segment + heartbeat, signals
    the parent it's ready, then sleeps until SIGKILLed.
    """
    import time as _time

    import redis as _redis

    from quorin._internal import heartbeat as _heartbeat
    from quorin.shm import SegmentRegistry as _Registry

    client = _redis.Redis.from_url(redis_url)
    reg = _Registry(client)
    seg = reg.create(schema_class, capacity=16)
    _heartbeat.ensure_started(client, os.getpid())
    Path(ready_path).write_text(seg.name)
    while True:  # SIGKILL terminates this loop.
        _time.sleep(0.5)


def test_e2_dead_pid_full_cleanup(
    redis_client: redis.Redis,
    registry: SegmentRegistry,
    tmp_path: Path,
) -> None:
    """Spawn a child via fork; child creates a real segment + heartbeat,
    signals ready, sleeps. Parent SIGKILLs the child, then constructs
    a watchdog and drives one tick with the child's miss_count forced
    to threshold. Asserts every Redis key + /dev/shm entry the child
    held is cleaned up.
    """
    ready_file = tmp_path / "child_ready"

    ctx = multiprocessing.get_context("fork")
    redis_url = "redis://127.0.0.1:6379/0"
    child = ctx.Process(
        target=_e2_child_target,
        args=(redis_url, str(ready_file), _WdSchema),
    )
    child.start()
    try:
        # Wait for the child to write the segment name to the ready file.
        deadline = time.monotonic() + 5.0
        while not ready_file.exists() and time.monotonic() < deadline:
            time.sleep(0.05)
        assert ready_file.exists(), "child did not signal ready in 5s"
        seg_name = ready_file.read_text().strip()
        child_pid = child.pid
        assert child_pid is not None

        # Confirm child's heartbeat is present.
        deadline = time.monotonic() + 2.0
        while not redis_client.hexists("quorin:heartbeats", str(child_pid)):
            if time.monotonic() > deadline:
                pytest.fail("child's heartbeat never appeared in Redis")
            time.sleep(0.05)

        # SIGKILL — no atexit, child's heartbeat stays in the hash.
        os.kill(child_pid, 9)
        child.join(timeout=5.0)
        assert child.exitcode == -9

        # Construct watchdog AFTER kill so its self_pid != child_pid.
        state = WatchdogState(redis_client, miss_threshold=2)
        # Seed _tracked by reading the heartbeat hash.
        state.run_one_tick()
        assert child_pid in state._tracked, f"child_pid={child_pid} not tracked after seed tick"
        # MEDIUM-Rev3 #4: directly force miss_count to threshold.
        state._tracked[child_pid].miss_count = state._miss_threshold

        # One more tick — cross-check (psutil.NoSuchProcess) + Lua + drain.
        result = state.run_one_tick()

        assert child_pid in result.dead_pids
        assert result.segments_unlinked_dead == 1
        assert result.segments_unlinked_drain == 1

        # Redis state cleaned.
        assert not redis_client.exists(f"quorin:refcount:{seg_name}")
        assert not redis_client.exists(_key_current(_WdSchema))
        assert not redis_client.hexists(KEY_SEGMENT_TO_SCHEMA, seg_name)
        assert not redis_client.exists(f"quorin:pid_segments:{child_pid}")
        assert not redis_client.hexists("quorin:heartbeats", str(child_pid))

        # /dev/shm cleaned. (posix_shm.unlink ran via the drain.)
        shm_path = Path(f"/dev/shm/{seg_name}")
        assert not shm_path.exists(), f"{shm_path} should have been unlinked"
    finally:
        if child.is_alive():
            child.kill()
            child.join(timeout=2.0)


# ---------------------------------------------------------------------------
# E3 — sidetable write on create (real Redis).
# ---------------------------------------------------------------------------


def test_e3_create_writes_sidetable(redis_client: redis.Redis, registry: SegmentRegistry) -> None:
    """``SegmentRegistry.create`` writes the sidetable atomically with
    the rest of its pipeline. Verified against real Redis.
    """
    seg = registry.create(_WdSchema, capacity=16)
    try:
        raw = redis_client.hget(KEY_SEGMENT_TO_SCHEMA, seg.name)
        assert raw is not None
        assert raw.decode() == "_WdSchema"
    finally:
        registry.close(seg)


# ---------------------------------------------------------------------------
# E4 — close-Lua extension at refcount-0 (real Redis).
# ---------------------------------------------------------------------------


def test_e4_close_lua_clears_schema_current_at_refcount_zero(
    redis_client: redis.Redis, registry: SegmentRegistry
) -> None:
    """Live-process close that hits refcount-0 clears schema:current
    AND the sidetable, AND queues the segment for unlink.
    """
    seg = registry.create(_WdSchema, capacity=16)
    name = seg.name
    assert redis_client.get(_key_current(_WdSchema)) == name.encode()

    registry.close(seg)

    assert redis_client.get(_key_current(_WdSchema)) is None
    assert redis_client.hexists(KEY_SEGMENT_TO_SCHEMA, name) == 0
    assert redis_client.sismember("quorin:cleanup_queue", name) == 1


# ---------------------------------------------------------------------------
# E5 — close-Lua rotation safety (real Redis).
# ---------------------------------------------------------------------------


def test_e5_close_lua_rotation_safety(redis_client: redis.Redis, registry: SegmentRegistry) -> None:
    """A creates seg1 (current=seg1), A creates seg2 (current=seg2),
    A closes seg1 → schema:current must STILL point at seg2.
    """
    seg1 = registry.create(_WdSchema, capacity=16)
    seg2 = registry.create(_WdSchema, capacity=16)
    try:
        assert redis_client.get(_key_current(_WdSchema)) == seg2.name.encode()

        registry.close(seg1)

        # Critical assertion: rotation-safe close-Lua extension.
        assert redis_client.get(_key_current(_WdSchema)) == seg2.name.encode()
        assert redis_client.hexists(KEY_SEGMENT_TO_SCHEMA, seg1.name) == 0
        assert redis_client.hexists(KEY_SEGMENT_TO_SCHEMA, seg2.name) == 1
        assert redis_client.sismember("quorin:cleanup_queue", seg1.name) == 1
        assert redis_client.sismember("quorin:cleanup_queue", seg2.name) == 0
    finally:
        registry.close(seg2)


# ---------------------------------------------------------------------------
# E6 — _force_drop_orphan clears schema:current AND sidetable.
# ---------------------------------------------------------------------------


def test_e6_force_drop_orphan_clears_schema_current_and_sidetable(
    redis_client: redis.Redis, registry: SegmentRegistry
) -> None:
    """``_force_drop_orphan`` (used by hydrate's failure path) clears
    schema:current AND the sidetable. The integration test uses real
    Redis to verify the pipeline transaction commits all four keys.
    """
    from quorin.hydration import _force_drop_orphan

    seg = registry.create(_WdSchema, capacity=16)
    name = seg.name

    assert redis_client.get(_key_current(_WdSchema)) == name.encode()
    assert redis_client.hexists(KEY_SEGMENT_TO_SCHEMA, name) == 1

    _force_drop_orphan(redis_client, _WdSchema, seg)

    assert redis_client.get(_key_current(_WdSchema)) is None
    assert redis_client.hexists(KEY_SEGMENT_TO_SCHEMA, name) == 0
    assert not redis_client.exists(f"quorin:refcount:{name}")
    # /dev/shm cleaned by _force_drop_orphan's direct unlink.
    assert not Path(f"/dev/shm/{name}").exists()
