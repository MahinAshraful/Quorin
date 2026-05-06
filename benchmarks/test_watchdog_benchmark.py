"""Benchmarks for quorin.watchdog (Step 14).

One bench: ``test_watchdog_tick_idle_100_pids`` — measures the cost of
a single ``WatchdogState.run_one_tick`` against 100 alive PIDs. The
psutil cross-check is monkeypatched (``pid_exists -> True``,
``create_time -> stored value``) so the bench measures Redis-tick
cost only. Real-psutil cost is parking-lotted for Step 16 if
flamegraphs surface ``psutil.Process`` construction at 100+ candidate
PIDs (POLISH-Rev3 #12).

Gate: 20 ms p99 (4x WSL2 dev-loop tolerance — see ``thresholds.yml``
top-of-file methodology block).
"""

from __future__ import annotations

import sys
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

pytestmark = pytest.mark.skipif(
    sys.platform == "win32",
    reason="Step 14 watchdog requires POSIX",
)

import redis  # noqa: E402

from quorin.watchdog import WatchdogState  # noqa: E402


def _populate_heartbeats(
    redis_client: redis.Redis,
    n_pids: int,
    start_pid: int = 900_000_000,
) -> dict[int, int]:
    """Returns {pid: create_time_ns} so the patched psutil mock can return
    the matching create_time per pid.
    """
    create_times: dict[int, int] = {}
    pipe = redis_client.pipeline(transaction=False)
    for i in range(n_pids):
        pid = start_pid + i
        # Distinct create_time per pid so the cross-check has unique
        # bit patterns to compare. Wall_time advances each fake-tick.
        create_ns = 1_700_000_000_000_000_000 + i  # arbitrary base
        pipe.hset("quorin:heartbeats", str(pid), f"{create_ns}:{1_000 + i}".encode())
        create_times[pid] = create_ns
    pipe.execute()
    return create_times


def _make_psutil_mock(create_times: dict[int, int]) -> Any:
    """Return a fake psutil.Process whose create_time matches stored."""

    def _proc_factory(pid: int) -> Any:
        instance = MagicMock()
        # Convert ns back to float seconds; the watchdog code does
        # int(create_time() * 1e9) again, which is bit-deterministic.
        ct_ns = create_times.get(pid, 0)
        instance.create_time.return_value = ct_ns / 1e9
        return instance

    return _proc_factory


@pytest.fixture
def populated_redis(redis_client: redis.Redis) -> dict[int, int]:
    """Per-bench setup: 100 fake heartbeats. Cleared by the autouse
    cleanup fixture in ``tests/conftest.py``.
    """
    return _populate_heartbeats(redis_client, n_pids=100)


def test_watchdog_tick_idle_100_pids(
    redis_client: redis.Redis,
    populated_redis: dict[int, int],
    benchmark: Any,
) -> None:
    """Idle-mode tick over 100 alive PIDs.

    Setup: psutil monkeypatched so the cross-check ALWAYS reports alive
    + matching create_time → no PID is declared dead. Each tick advances
    every wall_time_ns to keep miss_count at 0 (true idle).

    Gate: 20 ms p99 (4x WSL2 dev-loop tolerance).
    """
    state = WatchdogState(redis_client, miss_threshold=2)

    # Per-tick: bump wall_time_ns for every PID so miss_count stays 0.
    counter = {"tick": 0}

    def _bump_walls() -> None:
        counter["tick"] += 1
        pipe = redis_client.pipeline(transaction=False)
        for pid, ct_ns in populated_redis.items():
            wall = 1000 + counter["tick"] * 1000
            pipe.hset("quorin:heartbeats", str(pid), f"{ct_ns}:{wall}".encode())
        pipe.execute()

    psutil_factory = _make_psutil_mock(populated_redis)

    def _one_tick() -> None:
        _bump_walls()
        with patch("quorin.watchdog.psutil.Process", side_effect=psutil_factory):
            state.run_one_tick()

    # Use pedantic to give pytest-benchmark explicit control over rounds
    # + iterations; the default warmup behavior would skew the first few
    # rounds against the cold connection pool.
    benchmark.pedantic(_one_tick, rounds=20, iterations=1, warmup_rounds=2)

    # Bench docstring covers scope: this is the Redis-tick cost only.
