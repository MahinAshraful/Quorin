"""Smoke test: docker-compose Redis is reachable."""

from __future__ import annotations

import pytest


@pytest.mark.integration
def test_redis_roundtrip(redis_client) -> None:
    redis_client.set("pyforge:step0:smoke", b"ok", ex=10)
    assert redis_client.get("pyforge:step0:smoke") == b"ok"
