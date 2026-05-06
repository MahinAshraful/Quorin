"""Lua scripts shared by the real watchdog (:mod:`quorin.watchdog`) and
the test simulator (:mod:`tests._watchdog_helpers`).

Two scripts:

* :data:`CLEANUP_DEAD_PID_LUA` — atomic dead-PID cleanup transaction.
  Walks ``quorin:pid_segments:{pid}``, DECRs each segment's refcount,
  queues for unlink + clears schema:current via the segment_to_schema
  sidetable when refcount hits zero. Returns the list of segment names
  whose refcount hit zero — caller :func:`posix_shm.unlink`'s each.

* :data:`DRAIN_CLEANUP_QUEUE_LUA` — atomic drain. SPOPs up to
  ``ARGV[1]`` names from ``quorin:cleanup_queue`` and returns them.
  Caller :func:`posix_shm.unlink`'s each. (Lua can't do POSIX syscalls.)

PID-reuse guard
===============

The cleanup script's first op is HGET on ``heartbeats[pid]``. If the
parsed ``create_ns`` differs from the watchdog's expected value
(``ARGV[2]``), the kernel reused the dead PID for a new live process B
between the watchdog's psutil cross-check and Lua execution (~1 ms
window). Aborting with an empty return preserves B's state. If
``heartbeats[pid]`` is nil — A's atexit HDEL'd before reuse AND B's
force-first-refresh failed (Redis blip swallowed) — the script falls
through; combined probability ~3e-9 per dead PID. Documented residual
risk; alert on :data:`quorin_watchdog_pid_reuse_abort_total` AND on
``quorin:cleanup_queue`` size during incidents.
"""

from __future__ import annotations

from typing import Final

CLEANUP_DEAD_PID_LUA: Final[str] = """
-- KEYS:
--   1: quorin:pid_segments:{pid}        (set of segment names held by dead pid)
--   2: quorin:heartbeats                 (hash; for PID-reuse guard)
--   3: quorin:cleanup_queue              (set of segments queued for posix_shm.unlink)
--   4: quorin:segment_to_schema          (hash: segment_name -> schema_name)
-- ARGV:
--   1: pid_str                            (decimal string of the dead pid)
--   2: expected_create_time_ns_str        (watchdog's recorded create_time for that pid)
-- Returns:
--   integer count of segments whose refcount hit zero (and were SADDed to
--   cleanup_queue for the watchdog's step-4 drain to posix_shm.unlink).
--   Special return -1 if PID-reuse was detected (watchdog increments
--   pid_reuse_abort_total counter + drops _tracked entry).
--
-- Why count and not the names?
-- Returning the names AND SADDing to cleanup_queue caused the watchdog's
-- step 3 to unlink each segment, then step 4's drain to SPOP and try to
-- unlink the same name AGAIN (FileNotFoundError → debug log, but
-- 2x syscalls per segment per tick). Since the names are all queued
-- to cleanup_queue regardless, the watchdog's uniform step-4 drain
-- (the one that handles close-Lua's queue too) is the single
-- canonical posix_shm.unlink call site. Step 3 only needs to know
-- "did we successfully clean up, or was this a PID-reuse abort?" —
-- a count is enough.

local pid_segments_key = KEYS[1]
local heartbeats_key   = KEYS[2]
local cleanup_queue    = KEYS[3]
local seg_to_schema    = KEYS[4]
local pid                       = ARGV[1]
local expected_create_time_str  = ARGV[2]

-- PID-reuse guard. Watchdog's cross-check (psutil.NoSuchProcess /
-- create_time mismatch) and this Lua call are separated by ~1 ms. If
-- the kernel reused the dead pid for a new process B, B's heartbeat
-- ensure_started writes heartbeats[pid] with B's create_time. Bail
-- before touching any state.
local hb = redis.call('HGET', heartbeats_key, pid)
if hb then
  local sep = string.find(hb, ':')
  if sep then
    local actual_create = string.sub(hb, 1, sep - 1)
    if actual_create ~= expected_create_time_str then
      return -1
    end
  end
end

local segments = redis.call('SMEMBERS', pid_segments_key)
local zeroed_count = 0

for _, name in ipairs(segments) do
  local refcount_key = 'quorin:refcount:' .. name
  local n = redis.call('DECR', refcount_key)
  if tonumber(n) <= 0 then
    redis.call('SADD', cleanup_queue, name)
    redis.call('DEL', refcount_key)
    local schema = redis.call('HGET', seg_to_schema, name)
    if schema then
      local current_key = 'quorin:schema:' .. schema .. ':current'
      local current = redis.call('GET', current_key)
      if current == name then
        redis.call('DEL', current_key)
      end
      redis.call('HDEL', seg_to_schema, name)
    end
    zeroed_count = zeroed_count + 1
  end
end

redis.call('DEL', pid_segments_key)
redis.call('HDEL', heartbeats_key, pid)

return zeroed_count
"""


DRAIN_CLEANUP_QUEUE_LUA: Final[str] = """
-- KEYS:
--   1: quorin:cleanup_queue
-- ARGV:
--   1: max_drain (int)
-- Returns: list of names (caller posix_shm.unlinks each).
local cleanup_queue = KEYS[1]
local max = tonumber(ARGV[1])
local out = {}
for i = 1, max do
  local name = redis.call('SPOP', cleanup_queue)
  if not name then break end
  table.insert(out, name)
end
return out
"""


__all__ = [
    "CLEANUP_DEAD_PID_LUA",
    "DRAIN_CLEANUP_QUEUE_LUA",
]
