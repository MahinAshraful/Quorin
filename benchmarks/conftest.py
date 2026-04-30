"""pytest-benchmark fixtures.

Real GC isolation and warm-up logic land alongside Step 5 (Numba) and
Step 7 (GC management). This file exists so ``pytest benchmarks/`` collects
cleanly from day one.
"""

from __future__ import annotations

import gc
import os
from collections.abc import Iterator

import pytest
import redis


@pytest.fixture
def isolate_gc():
    """Disable GC for the duration of a single benchmark. Step 7 will flesh this out."""
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
