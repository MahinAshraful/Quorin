"""Test-side simulators for the Step 14 watchdog.

Step 14's watchdog ships at :mod:`quorin.watchdog`. These helpers exist
because (a) Step 13 hydration tests pre-date the watchdog and need to
simulate post-crash cleanup deterministically, and (b) some tests want
to drive cleanup outside a subprocess for speed. The helpers now SHARE
the watchdog's Lua scripts (via :mod:`quorin._internal.watchdog_lua`)
so simulation can't drift from production.

Two distinct helpers because the contracts are different:

* :func:`simulate_orphan_cleanup_in_orchestrator` — IN-PROCESS path.
  Wraps :func:`quorin.hydration._force_drop_orphan`.
* :func:`simulate_watchdog_post_crash_cleanup` — POST-CRASH path.
  Calls the real Step 14 cleanup Lua against ``quorin:pid_segments:{dead_pid}``,
  including the PID-reuse guard (requires ``expected_create_time_ns``
  argument; defaults to 0 for tests that don't care about the guard
  and want a fall-through).
"""

from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING

from quorin._internal import posix_shm
from quorin._internal.watchdog_lua import (
    CLEANUP_DEAD_PID_LUA,
    DRAIN_CLEANUP_QUEUE_LUA,
)
from quorin.shm import (
    KEY_CLEANUP_QUEUE,
    KEY_SEGMENT_TO_SCHEMA,
    _key_pid_segments,
)

if TYPE_CHECKING:
    import redis

    from quorin.schema import FeatureSchema
    from quorin.shm import Segment


_KEY_HEARTBEATS = "quorin:heartbeats"


def simulate_orphan_cleanup_in_orchestrator(
    redis_client: redis.Redis,
    schema: type[FeatureSchema],
    segment: Segment,
) -> None:
    """The IN-PROCESS orphan path.

    Wraps :func:`quorin.hydration._force_drop_orphan` for tests that
    need the same semantics outside the orchestrator's exception
    handler. NOT used by chaos tests — those have a dead orchestrator
    process and need the post-crash helper below.
    """
    from quorin.hydration import _force_drop_orphan

    _force_drop_orphan(redis_client, schema, segment)


def simulate_watchdog_post_crash_cleanup(
    redis_client: redis.Redis,
    dead_pid: int,
    expected_create_time_ns: int | None = None,
) -> int:
    """The POST-CRASH watchdog path. Calls the real Step 14 cleanup Lua.

    Atomically:
    1. PID-reuse guard via ``heartbeats[pid]``.
    2. SMEMBERS ``quorin:pid_segments:{dead_pid}``.
    3. Per segment: DECR refcount, SADD-to-cleanup-queue + clear
       schema:current via the sidetable when refcount hits zero.
    4. DEL ``quorin:pid_segments:{dead_pid}``.
    5. HDEL ``quorin:heartbeats {dead_pid}``.

    Then this helper :func:`posix_shm.unlink`'s each returned name
    (Lua can't do POSIX syscalls).

    Returns the number of segments successfully unlinked. Returns 0
    if the PID-reuse guard tripped (caller can detect by checking
    ``quorin:pid_segments:{dead_pid}`` still has members).

    PID-reuse guard handling
    ------------------------

    * ``expected_create_time_ns=None`` (default): helper reads
      ``heartbeats[dead_pid]`` itself, parses the create_ns, and
      passes that value to the Lua. Effectively a no-op guard for
      "I don't care about PID reuse" tests (the common case for
      chaos tests with a SIGKILLed child). If the heartbeats entry
      is absent, passes 0 — Lua falls through (HGET-nil branch).
    * ``expected_create_time_ns=int``: helper passes the given int.
      Used by tests that explicitly validate the guard's mismatch
      branch (set ``heartbeats[pid]`` to one value, pass a different
      value here, assert empty return).
    """
    if expected_create_time_ns is None:
        raw = redis_client.hget(_KEY_HEARTBEATS, str(dead_pid))
        if raw is not None:
            decoded = raw.decode("ascii") if isinstance(raw, bytes) else raw
            try:
                expected_create_time_ns = int(decoded.split(":", 1)[0])
            except (ValueError, IndexError):
                expected_create_time_ns = 0
        else:
            expected_create_time_ns = 0

    cleanup_lua = redis_client.register_script(CLEANUP_DEAD_PID_LUA)
    count = int(
        cleanup_lua(
            keys=[
                _key_pid_segments(dead_pid),
                _KEY_HEARTBEATS,
                KEY_CLEANUP_QUEUE,
                KEY_SEGMENT_TO_SCHEMA,
            ],
            args=[str(dead_pid), str(expected_create_time_ns)],
        )
    )
    if count < 0:
        # PID-reuse abort.
        return 0
    # Lua queued the names to cleanup_queue; drain to actually unlink.
    return drain_cleanup_queue(redis_client)


def drain_cleanup_queue(
    redis_client: redis.Redis,
    *,
    batch: int = 1024,
) -> int:
    """Drain ``quorin:cleanup_queue`` via the real watchdog drain Lua.

    Returns count drained. Used by tests to simulate the watchdog's
    drain pass after a clean ``registry.close`` (e.g. integration
    tests' "operator drops current segment").
    """
    drain_lua = redis_client.register_script(DRAIN_CLEANUP_QUEUE_LUA)
    drained = 0
    while True:
        returned = drain_lua(keys=[KEY_CLEANUP_QUEUE], args=[str(batch)])
        if not returned:
            break
        for raw_name in returned:
            name = raw_name.decode() if isinstance(raw_name, bytes) else str(raw_name)
            with contextlib.suppress(Exception):
                posix_shm.unlink(name)
            drained += 1
    return drained


# Re-export the key helpers so chaos tests don't need to reach into
# quorin.shm's private surface in two places (the helpers above are
# the canonical entry points, but tests that need to introspect Redis
# state directly can use these).
__all__ = [
    "drain_cleanup_queue",
    "simulate_orphan_cleanup_in_orchestrator",
    "simulate_watchdog_post_crash_cleanup",
]
