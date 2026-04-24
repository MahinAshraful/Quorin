"""Shared pytest fixtures.

The Redis + shm-cleanup + gc-isolation fixtures promised in Step 0's layout
arrive in later steps — they are listed here as stubs so every contributor
knows where they will live.
"""

from __future__ import annotations

import os

import pytest
import redis


def _redis_url() -> str:
    # WSL2 / Docker Desktop for Windows: always 127.0.0.1, never ``localhost``.
    return os.environ.get("PYFORGE_REDIS_URL", "redis://127.0.0.1:6379/0")


@pytest.fixture(scope="session")
def redis_client() -> redis.Redis:
    """Live Redis client. Tests that depend on this fixture should be marked
    ``@pytest.mark.integration`` so they only run where Redis is available.
    """
    client = redis.Redis.from_url(_redis_url(), decode_responses=False)
    try:
        client.ping()
    except redis.ConnectionError:
        pytest.skip("Redis is not reachable — start docker/docker-compose.dev.yml")
    yield client
    client.close()
