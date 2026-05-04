"""Unit tests for Step 15 Lua scripts (real Redis).

Marked ``@pytest.mark.integration`` because the Lua semantics we lock in
(EVALSHA-vs-EVAL, GET vs `false`, atomic CAS) require Redis-side
evaluation — fakeredis Lua compatibility is fragile across versions.

Covers (per plan §5.2):

* ``FLIP_SCHEMA_CURRENT_LUA`` — happy / race / missing.
* ``RELEASE_LOCK_LUA`` — token match / mismatch.
* register_script + EVALSHA path (matches Step 14 convention).
"""

from __future__ import annotations

import sys
import uuid

import pytest

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(sys.platform == "win32", reason="requires Redis service"),
]

import redis  # noqa: E402

from pyforge._internal.evolution_lua import (  # noqa: E402
    FLIP_SCHEMA_CURRENT_LUA,
    RELEASE_LOCK_LUA,
)

# ---------------------------------------------------------------------------
# Test fixtures.
# ---------------------------------------------------------------------------


@pytest.fixture
def flip_script(redis_client: redis.Redis) -> redis.commands.core.Script:
    return redis_client.register_script(FLIP_SCHEMA_CURRENT_LUA)


@pytest.fixture
def release_script(redis_client: redis.Redis) -> redis.commands.core.Script:
    return redis_client.register_script(RELEASE_LOCK_LUA)


def _unique_key(prefix: str) -> bytes:
    return f"{prefix}:{uuid.uuid4().hex[:8]}".encode("ascii")


# ---------------------------------------------------------------------------
# FLIP_SCHEMA_CURRENT_LUA.
# ---------------------------------------------------------------------------


def test_flip_happy_path_returns_1_and_updates_current(
    redis_client: redis.Redis,
    flip_script: redis.commands.core.Script,
) -> None:
    key = _unique_key("pyforge:schema:_FlipTest1:current")
    old_name = b"pyforge_old_v1_aaaa"
    new_name = b"pyforge_new_v2_bbbb"
    redis_client.set(key, old_name)
    try:
        result = flip_script(keys=[key], args=[old_name, new_name])
        assert result == 1
        assert redis_client.get(key) == new_name
    finally:
        redis_client.delete(key)


def test_flip_race_returns_0_when_current_does_not_match(
    redis_client: redis.Redis,
    flip_script: redis.commands.core.Script,
) -> None:
    """Race lost: another orchestrator already flipped to a different name."""
    key = _unique_key("pyforge:schema:_FlipTest2:current")
    other_name = b"pyforge_other_v3_cccc"
    expected_old = b"pyforge_expected_old_v1"
    new_name = b"pyforge_new_v2_dddd"
    redis_client.set(key, other_name)
    try:
        result = flip_script(keys=[key], args=[expected_old, new_name])
        assert result == 0
        # Current pointer untouched.
        assert redis_client.get(key) == other_name
    finally:
        redis_client.delete(key)


def test_flip_missing_returns_minus_1(
    redis_client: redis.Redis,
    flip_script: redis.commands.core.Script,
) -> None:
    """Operator interference: schema:current was DEL'd between
    open_current and flip."""
    key = _unique_key("pyforge:schema:_FlipTest3:current")
    redis_client.delete(key)  # ensure absent
    result = flip_script(keys=[key], args=[b"pyforge_anything", b"pyforge_new_v2"])
    assert result == -1
    assert redis_client.get(key) is None


def test_flip_concurrent_only_one_succeeds(
    redis_client: redis.Redis,
    flip_script: redis.commands.core.Script,
) -> None:
    """Two flip attempts against the same expected_old: first wins, second
    sees a different `current` and returns 0. Atomicity is what makes this
    safe."""
    key = _unique_key("pyforge:schema:_FlipTest4:current")
    old_name = b"pyforge_old_v1"
    new_a = b"pyforge_new_a_v2"
    new_b = b"pyforge_new_b_v2"
    redis_client.set(key, old_name)
    try:
        r1 = flip_script(keys=[key], args=[old_name, new_a])
        r2 = flip_script(keys=[key], args=[old_name, new_b])
        assert r1 == 1
        assert r2 == 0
        assert redis_client.get(key) == new_a
    finally:
        redis_client.delete(key)


# ---------------------------------------------------------------------------
# RELEASE_LOCK_LUA.
# ---------------------------------------------------------------------------


def test_release_lock_token_match_dels_key(
    redis_client: redis.Redis,
    release_script: redis.commands.core.Script,
) -> None:
    key = _unique_key("pyforge:upgrade:lock:_LockTest1")
    token = uuid.uuid4().hex.encode("ascii")
    redis_client.set(key, token)
    try:
        result = release_script(keys=[key], args=[token])
        assert result == 1
        assert redis_client.get(key) is None
    finally:
        redis_client.delete(key)


def test_release_lock_token_mismatch_no_op(
    redis_client: redis.Redis,
    release_script: redis.commands.core.Script,
) -> None:
    """Lock auto-expired and a different orchestrator acquired it; original
    holder's release attempt must be a no-op (token mismatch)."""
    key = _unique_key("pyforge:upgrade:lock:_LockTest2")
    other_token = uuid.uuid4().hex.encode("ascii")
    my_token = uuid.uuid4().hex.encode("ascii")
    assert other_token != my_token
    redis_client.set(key, other_token)
    try:
        result = release_script(keys=[key], args=[my_token])
        assert result == 0
        # Other orchestrator's lock is intact.
        assert redis_client.get(key) == other_token
    finally:
        redis_client.delete(key)


def test_release_lock_absent_key_no_op(
    redis_client: redis.Redis,
    release_script: redis.commands.core.Script,
) -> None:
    key = _unique_key("pyforge:upgrade:lock:_LockTest3")
    redis_client.delete(key)
    result = release_script(keys=[key], args=[uuid.uuid4().hex.encode("ascii")])
    assert result == 0
