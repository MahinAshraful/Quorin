"""Lua scripts for Step 15 schema evolution.

Two scripts:

* :data:`FLIP_SCHEMA_CURRENT_LUA` — atomic CAS on
  ``pyforge:schema:{name}:current``. Without atomicity, two concurrent
  upgrade orchestrators (or a hydrate racing with an upgrade) could
  interleave a "GET old → SET new" sequence and silently lose a flip.
  Returns ``1`` on flipped, ``0`` if the current pointer didn't match the
  expected old name (race lost), ``-1`` if the pointer was missing
  entirely (operator interference).

* :data:`RELEASE_LOCK_LUA` — token-checked DEL of
  ``pyforge:upgrade:lock:{safe_name}``. A simple ``DEL`` would let a
  caller release a lock acquired by a different orchestrator (e.g. after
  a TTL-expiry race). The token-check makes lock release idempotent and
  safe under TTL races.

Both scripts are registered via :meth:`redis.Redis.register_script` at
orchestrator construction (see :func:`pyforge.evolution.upgrade_schema`).
After the first call redis-py uses EVALSHA, matching Step 14's
``pyforge.watchdog.WatchdogState`` convention.
"""

from __future__ import annotations

from typing import Final

# ---------------------------------------------------------------------------
# Atomic flip: schema:current = new IFF current = expected_old.
# ---------------------------------------------------------------------------

FLIP_SCHEMA_CURRENT_LUA: Final[str] = """
-- KEYS:
--   1: pyforge:schema:{safe_name}:current
-- ARGV:
--   1: expected_old_segment_name
--   2: new_segment_name
-- Returns:
--    1  flipped (current was expected_old; current now = new_name)
--    0  not flipped (current != expected_old; pointer unchanged)
--   -1  not flipped (current did not exist)

local current_key = KEYS[1]
local expected_old = ARGV[1]
local new_name = ARGV[2]

local current = redis.call('GET', current_key)
if current == false then
  return -1
end
if current ~= expected_old then
  return 0
end

redis.call('SET', current_key, new_name)
return 1
"""


# ---------------------------------------------------------------------------
# Token-checked lock release.
# ---------------------------------------------------------------------------

RELEASE_LOCK_LUA: Final[str] = """
-- KEYS:
--   1: pyforge:upgrade:lock:{safe_name}
-- ARGV:
--   1: expected_token
-- Returns:
--    1 if released (token matched and DEL succeeded)
--    0 if not released (token mismatch or key absent)

if redis.call('GET', KEYS[1]) == ARGV[1] then
  return redis.call('DEL', KEYS[1])
end
return 0
"""
