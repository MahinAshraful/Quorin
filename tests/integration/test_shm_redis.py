"""Integration tests: SegmentRegistry against a real Redis instance.

Verifies the full open/close cycle and the Redis key accounting that Step 14
(watchdog) and Step 15 (evolution) will rely on.
"""

from __future__ import annotations

import os
import sys

import pytest

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        sys.platform == "win32",
        reason="shm requires POSIX (Linux/WSL2)",
    ),
]

from quorin.schema import FeatureField, FeatureSchema, dtype  # noqa: E402
from quorin.shm import (  # noqa: E402
    KEY_CLEANUP_QUEUE,
    SegmentRegistry,
    _key_current,
    _key_pid_segments,
    _key_refcount,
)


class _Schema(FeatureSchema):
    version = 1
    fields = [FeatureField("x", dtype.float32)]


def _get_int(redis_client, key: str) -> int:
    raw = redis_client.get(key)
    if raw is None:
        return 0
    return int(raw)


def test_full_create_open_close_cycle_refcount_transitions(redis_client) -> None:
    reg = SegmentRegistry(redis_client)
    seg = reg.create(_Schema, capacity=32)
    name = seg.name
    assert _get_int(redis_client, _key_refcount(name)) == 1

    seg2 = reg.open_current(_Schema)
    assert _get_int(redis_client, _key_refcount(name)) == 2

    reg.close(seg2)
    assert _get_int(redis_client, _key_refcount(name)) == 1

    reg.close(seg)
    assert _get_int(redis_client, _key_refcount(name)) == 0

    # Name ended up in cleanup queue
    members = {
        m.decode() if isinstance(m, bytes) else m for m in redis_client.smembers(KEY_CLEANUP_QUEUE)
    }
    assert name in members


def test_three_concurrent_opens_give_refcount_four(redis_client) -> None:
    reg = SegmentRegistry(redis_client)
    seg = reg.create(_Schema, capacity=32)
    name = seg.name
    opens = [reg.open_current(_Schema) for _ in range(3)]
    try:
        assert _get_int(redis_client, _key_refcount(name)) == 4
    finally:
        for s in opens:
            reg.close(s)
        reg.close(seg)


def test_pid_segments_set_tracks_single_open_lifecycle(redis_client) -> None:
    """The expected usage pattern: one process opens a given segment once.
    After close, that segment name should not be in pid_segments."""
    pid_key = _key_pid_segments(os.getpid())
    assert redis_client.scard(pid_key) == 0

    reg = SegmentRegistry(redis_client)
    seg = reg.create(_Schema, capacity=32)
    members = {m.decode() if isinstance(m, bytes) else m for m in redis_client.smembers(pid_key)}
    assert seg.name in members

    reg.close(seg)
    members = {m.decode() if isinstance(m, bytes) else m for m in redis_client.smembers(pid_key)}
    assert seg.name not in members


def test_schema_current_key_points_at_created_segment(redis_client) -> None:
    reg = SegmentRegistry(redis_client)
    seg = reg.create(_Schema, capacity=32)
    try:
        current = redis_client.get(_key_current(_Schema))
        assert current is not None
        assert current.decode() == seg.name
    finally:
        reg.close(seg)
