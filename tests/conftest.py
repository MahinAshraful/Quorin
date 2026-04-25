"""Shared pytest fixtures.

The ``redis_client`` fixture is a live client against a Redis instance
(normally docker-compose'd at 127.0.0.1:6379). The ``_shm_test_isolation``
autouse fixture scrubs any Pyforge artifacts left behind by each test so a
leak in one test does not cascade into the next.
"""

from __future__ import annotations

import contextlib
import os
import sys
from collections.abc import Iterator
from pathlib import Path

import pytest
import redis


def _redis_url() -> str:
    # WSL2 / Docker Desktop for Windows: always 127.0.0.1, never ``localhost``.
    return os.environ.get("PYFORGE_REDIS_URL", "redis://127.0.0.1:6379/0")


@pytest.fixture(scope="session")
def redis_client() -> Iterator[redis.Redis]:
    """Live Redis client. Tests that need it are gated on it — if Redis is not
    running (CI without the Redis service, or a local run without
    ``./dev-up.sh``), dependent tests skip rather than error.
    """
    client = redis.Redis.from_url(_redis_url(), decode_responses=False)
    try:
        client.ping()
    except redis.ConnectionError:
        pytest.skip("Redis is not reachable — start docker/docker-compose.dev.yml")
    yield client
    client.close()


@pytest.fixture(autouse=True)
def _shm_test_isolation() -> Iterator[None]:
    """Scrub Pyforge artifacts after every test.

    Runs post-test to:
      * Unlink any ``pyforge_*`` entries left in ``/dev/shm``.
      * Delete any ``pyforge:*`` keys from Redis (best-effort; silently skipped
        if Redis isn't reachable).

    This is belt-and-suspenders — tests should clean up after themselves, but
    a crashed test shouldn't poison the suite.
    """
    yield

    # ---- /dev/shm cleanup (no external deps) ------------------------------
    shm_dir = Path("/dev/shm")
    if shm_dir.is_dir() and sys.platform != "win32":
        try:
            from pyforge._internal import posix_shm
        except ImportError:
            posix_shm = None  # type: ignore[assignment]
        if posix_shm is not None:
            for entry in shm_dir.iterdir():
                if entry.name.startswith("pyforge_"):
                    try:
                        posix_shm.unlink(entry.name)
                    except Exception:
                        # Last-ditch: try removing the /dev/shm file directly.
                        with contextlib.suppress(OSError):
                            entry.unlink()

    # ---- Redis cleanup (best-effort) --------------------------------------
    try:
        client = redis.Redis.from_url(
            _redis_url(),
            decode_responses=False,
            socket_connect_timeout=1,
        )
        client.ping()
    except redis.ConnectionError:
        return

    try:
        keys = list(client.scan_iter(match="pyforge:*", count=1000))
        if keys:
            client.delete(*keys)
    finally:
        client.close()
