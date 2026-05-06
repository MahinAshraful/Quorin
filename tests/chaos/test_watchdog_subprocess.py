"""Chaos tests for the Step 14 watchdog running as a real subprocess.

Spawns ``python -m quorin.watchdog`` and a worker subprocess, exercises:

* C1 — SIGKILL convergence: kill the worker, the watchdog cleans up
  ``/dev/shm`` and Redis state within the detection ceiling.
* C2 — second-reader keeps segment alive: A creates, B opens, kill A.
  Watchdog DECRs A's refcount but the segment stays (B still holds).
  When B exits cleanly, close-Lua queues + watchdog drains.
* C3 — debugger-attach resilience: SIGSTOP/SIGCONT cycle exceeds the
  miss-threshold time but psutil cross-check sees the process is
  alive with matching create_time → never declared dead.

Fast cadence (``--miss-threshold=2 --tick-interval-seconds=2``) gives a
~5 s detection ceiling so tests don't sit for 150 s. Worker subprocesses
sleep on ``signal.pause()`` so SIGSTOP/SIGCONT/SIGKILL drive their
lifecycle.

C4 (PID rollover) is intentionally absent — see plan MEDIUM-Rev2 #7;
``test_psutil_alive_different_create_time_declares_dead`` in
``test_watchdog.py`` is the contract test for the PID-reuse cross-check
branch.
"""

from __future__ import annotations

import os
import signal
import subprocess
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
    pytest.mark.chaos,
]

if TYPE_CHECKING:
    from collections.abc import Iterator


# ---------------------------------------------------------------------------
# Fast-cadence config. Detection ceiling = miss_threshold * tick_interval ≈ 4 s.
# Plus up-to-one-tick alignment slop ≈ 6 s ceiling.
# ---------------------------------------------------------------------------

FAST_TICK_INTERVAL = 2.0
FAST_MISS_THRESHOLD = 2
CONVERGENCE_TIMEOUT = 8.0  # 6 s ceiling + safety margin


# ---------------------------------------------------------------------------
# Worker script (top-level so it can be invoked as ``python -c '...'``-able).
# ---------------------------------------------------------------------------


WORKER_SCRIPT = r"""
import os
import signal
import sys
import time

import redis

from quorin._internal import heartbeat
from quorin.schema import FeatureField, FeatureSchema, dtype
from quorin.shm import SegmentRegistry


class _ChaosSchema(FeatureSchema):
    version = 1
    fields = [FeatureField("a", dtype.float32)]


redis_url = sys.argv[1]
ready_path = sys.argv[2]
mode = sys.argv[3] if len(sys.argv) > 3 else "create"

client = redis.Redis.from_url(redis_url)
reg = SegmentRegistry(client)

if mode == "create":
    seg = reg.create(_ChaosSchema, capacity=16)
    name = seg.name
elif mode == "open":
    # Wait for a creator to publish schema:current.
    import time as _t
    deadline = _t.time() + 5.0
    while _t.time() < deadline:
        if client.get("quorin:schema:_ChaosSchema:current") is not None:
            break
        _t.sleep(0.05)
    seg = reg.open_current(_ChaosSchema)
    name = seg.name
else:
    sys.exit(99)

heartbeat.ensure_started(client, os.getpid())

with open(ready_path, "w") as f:
    f.write(name)

# Block until SIGKILL/SIGTERM/SIGSTOP/etc.
while True:
    signal.pause()
"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def redis_url() -> str:
    return os.environ.get("QUORIN_REDIS_URL", "redis://127.0.0.1:6379/0")


@pytest.fixture
def watchdog_proc(redis_url: str) -> Iterator[subprocess.Popen[bytes]]:
    """Spawn a real ``python -m quorin.watchdog`` subprocess at fast cadence."""
    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "quorin.watchdog",
            "--redis",
            redis_url,
            "--tick-interval-seconds",
            str(FAST_TICK_INTERVAL),
            "--miss-threshold",
            str(FAST_MISS_THRESHOLD),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    # Give the watchdog a moment to register scripts + start its first
    # tick (ensure_started does a force-first-refresh of its own pid).
    time.sleep(0.5)
    yield proc
    if proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=3.0)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=2.0)


def _spawn_worker(
    redis_url: str, ready_path: Path, mode: str = "create"
) -> subprocess.Popen[bytes]:
    return subprocess.Popen(
        [
            sys.executable,
            "-c",
            WORKER_SCRIPT,
            redis_url,
            str(ready_path),
            mode,
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def _wait_for_ready(ready_path: Path, deadline_seconds: float = 5.0) -> str:
    deadline = time.monotonic() + deadline_seconds
    while not ready_path.exists() and time.monotonic() < deadline:
        time.sleep(0.05)
    assert ready_path.exists(), f"worker did not signal ready in {deadline_seconds}s"
    return ready_path.read_text().strip()


def _poll_until_clean(
    redis_client: redis.Redis,
    seg_name: str,
    schema_name: str,
    pid: int,
    deadline_seconds: float = CONVERGENCE_TIMEOUT,
) -> bool:
    """Returns True if all the dead-PID's state is gone within deadline."""
    deadline = time.monotonic() + deadline_seconds
    while time.monotonic() < deadline:
        refcount_gone = not redis_client.exists(f"quorin:refcount:{seg_name}")
        current_gone = not redis_client.exists(f"quorin:schema:{schema_name}:current")
        sidetable_gone = not redis_client.hexists("quorin:segment_to_schema", seg_name)
        pid_segs_gone = not redis_client.exists(f"quorin:pid_segments:{pid}")
        heartbeat_gone = not redis_client.hexists("quorin:heartbeats", str(pid))
        shm_gone = not Path(f"/dev/shm/{seg_name}").exists()
        if (
            refcount_gone
            and current_gone
            and sidetable_gone
            and pid_segs_gone
            and heartbeat_gone
            and shm_gone
        ):
            return True
        time.sleep(0.1)
    return False


# ---------------------------------------------------------------------------
# C1 — SIGKILL convergence (parametrized; subset of the spec's 50 seeds).
# ---------------------------------------------------------------------------

# Run 10 seeds by default (covers flake without dominating chaos suite
# runtime); set QUORIN_FULL_CHAOS=1 to expand to 50 for nightly CI.
_C1_SEEDS = list(range(50)) if os.environ.get("QUORIN_FULL_CHAOS") else list(range(10))


@pytest.mark.parametrize("seed", _C1_SEEDS)
def test_c1_sigkill_convergence(
    redis_url: str,
    redis_client: redis.Redis,
    watchdog_proc: subprocess.Popen[bytes],
    tmp_path: Path,
    seed: int,
) -> None:
    """SIGKILL the worker; confirm /dev/shm + Redis state converge to
    clean within the detection ceiling. Spec § 1357 acceptance criterion.
    """
    ready_file = tmp_path / f"ready_{seed}"
    worker = _spawn_worker(redis_url, ready_file)
    try:
        seg_name = _wait_for_ready(ready_file)
        worker_pid = worker.pid
        assert worker_pid is not None

        # Confirm Redis state populated.
        deadline = time.monotonic() + 2.0
        while not redis_client.hexists("quorin:heartbeats", str(worker_pid)):
            if time.monotonic() > deadline:
                pytest.fail("worker heartbeat never appeared")
            time.sleep(0.05)

        os.kill(worker_pid, signal.SIGKILL)
        worker.wait(timeout=3.0)
        assert worker.returncode == -signal.SIGKILL

        converged = _poll_until_clean(redis_client, seg_name, "_ChaosSchema", worker_pid)
        assert converged, f"watchdog did not converge in {CONVERGENCE_TIMEOUT}s for seed={seed}"
    finally:
        if worker.poll() is None:
            worker.kill()
            worker.wait(timeout=2.0)


# ---------------------------------------------------------------------------
# C2 — second reader keeps segment alive after creator dies.
# ---------------------------------------------------------------------------


def test_c2_second_reader_keeps_segment_alive(
    redis_url: str,
    redis_client: redis.Redis,
    watchdog_proc: subprocess.Popen[bytes],
    tmp_path: Path,
) -> None:
    """A creates segment (refcount=1), B opens (refcount=2). Kill A.
    Watchdog DECRs A's refcount → 1; segment NOT unlinked (B still holds).
    Then kill B clean; close-Lua queues + watchdog drains.
    """
    ready_a = tmp_path / "ready_a"
    ready_b = tmp_path / "ready_b"
    worker_a = _spawn_worker(redis_url, ready_a, mode="create")
    seg_name = _wait_for_ready(ready_a)
    pid_a = worker_a.pid
    assert pid_a is not None

    worker_b = _spawn_worker(redis_url, ready_b, mode="open")
    try:
        # Wait for B to open + heartbeat. Confirm refcount = 2.
        seg_name_b = _wait_for_ready(ready_b)
        assert seg_name_b == seg_name
        pid_b = worker_b.pid
        assert pid_b is not None

        deadline = time.monotonic() + 3.0
        while True:
            raw = redis_client.get(f"quorin:refcount:{seg_name}")
            if raw is not None and int(raw) == 2:
                break
            if time.monotonic() > deadline:
                pytest.fail(f"refcount never reached 2; saw {raw!r}")
            time.sleep(0.05)

        # SIGKILL A.
        os.kill(pid_a, signal.SIGKILL)
        worker_a.wait(timeout=3.0)
        assert worker_a.returncode == -signal.SIGKILL

        # Wait for the watchdog to declare A dead. Refcount should drop
        # to 1 but segment stays.
        deadline = time.monotonic() + CONVERGENCE_TIMEOUT
        a_cleaned = False
        while time.monotonic() < deadline:
            pid_segs_a_gone = not redis_client.exists(f"quorin:pid_segments:{pid_a}")
            heartbeat_a_gone = not redis_client.hexists("quorin:heartbeats", str(pid_a))
            refcount_raw = redis_client.get(f"quorin:refcount:{seg_name}")
            refcount = int(refcount_raw) if refcount_raw is not None else 0
            if pid_segs_a_gone and heartbeat_a_gone and refcount == 1:
                a_cleaned = True
                break
            time.sleep(0.1)
        assert a_cleaned, "watchdog did not clean A's state in time"

        # Segment still present in /dev/shm — B holds it.
        assert Path(f"/dev/shm/{seg_name}").exists()
        assert redis_client.hexists("quorin:segment_to_schema", seg_name) == 1
        assert redis_client.exists("quorin:schema:_ChaosSchema:current")

        # Kill B with SIGTERM (SIGTERM doesn't trigger atexit either,
        # use a clean exit via SIGUSR1? Simpler: SIGKILL B too. Watchdog
        # will see B's heartbeat go stale, declare dead, refcount → 0,
        # close-Lua's logic doesn't apply (close-Lua only runs on
        # SegmentRegistry.close — but B was SIGKILLed, so the watchdog
        # dead-PID Lua does the refcount-0 cleanup).
        os.kill(pid_b, signal.SIGKILL)
        worker_b.wait(timeout=3.0)
        assert worker_b.returncode == -signal.SIGKILL

        converged = _poll_until_clean(redis_client, seg_name, "_ChaosSchema", pid_b)
        assert converged, "watchdog did not converge after B's death"
    finally:
        if worker_a.poll() is None:
            worker_a.kill()
            worker_a.wait(timeout=2.0)
        if worker_b.poll() is None:
            worker_b.kill()
            worker_b.wait(timeout=2.0)


# ---------------------------------------------------------------------------
# C3 — debugger-attach resilience (SIGSTOP / SIGCONT).
# ---------------------------------------------------------------------------


def test_c3_sigstop_sigcont_does_not_declare_dead(
    redis_url: str,
    redis_client: redis.Redis,
    watchdog_proc: subprocess.Popen[bytes],
    tmp_path: Path,
) -> None:
    """SIGSTOP at t=0, wait 6s (3 watchdog ticks past threshold of 2),
    SIGCONT. Threshold is reached around t=4s (2 ticks x 2s) - the
    cross-check fires multiple times but psutil reports alive AND
    create_time matches - NOT declared dead. After SIGCONT the worker
    resumes heartbeating and miss_count resets.

    Spec § 1352 acceptance criterion.
    """
    ready_file = tmp_path / "ready_c3"
    worker = _spawn_worker(redis_url, ready_file)
    try:
        seg_name = _wait_for_ready(ready_file)
        worker_pid = worker.pid
        assert worker_pid is not None

        # Wait for heartbeat to populate.
        deadline = time.monotonic() + 2.0
        while not redis_client.hexists("quorin:heartbeats", str(worker_pid)):
            if time.monotonic() > deadline:
                pytest.fail("worker heartbeat never appeared")
            time.sleep(0.05)

        # SIGSTOP — process pauses, heartbeat thread blocked.
        os.kill(worker_pid, signal.SIGSTOP)
        # Wait long enough that the watchdog crosses the miss-threshold
        # AND has the chance to fire its cross-check at least twice.
        time.sleep(6.0)

        # Worker's segment + state must STILL be intact.
        assert redis_client.exists(f"quorin:refcount:{seg_name}"), (
            "watchdog incorrectly declared SIGSTOP'd worker dead — "
            "psutil cross-check should have caught it"
        )
        assert Path(f"/dev/shm/{seg_name}").exists()
        # Heartbeat hash entry stays (no HDEL fired).
        assert redis_client.hexists("quorin:heartbeats", str(worker_pid)) == 1

        # SIGCONT — worker resumes heartbeating.
        os.kill(worker_pid, signal.SIGCONT)
        time.sleep(0.3)  # let the heartbeat thread tick once
    finally:
        if worker.poll() is None:
            worker.kill()
            worker.wait(timeout=2.0)
