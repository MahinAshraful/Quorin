"""Test-side simulators for what Step 14's watchdog WILL do.

Step 13 hydration's chaos + integration tests need to simulate the
watchdog's post-crash cleanup behavior to verify hydrate's recovery
path (kill -> watchdog -> retry hydrate). Step 14 hasn't shipped yet,
so this module provides the simulation. When Step 14's real watchdog
lands at ``pyforge/watchdog.py``, the helpers here either (a) wrap
the real watchdog's cleanup function, or (b) get deleted in favor of
calling that function directly. Either way, the contracts these
helpers document are what Step 14 must implement.

Two distinct helpers because the contracts are different:

* :func:`simulate_orphan_cleanup_in_orchestrator` — IN-PROCESS path.
  Wraps :func:`pyforge.hydration._force_drop_orphan`. Used when the
  orchestrator caught its own exception (Commit A's mocked unit
  tests #23, #23b verify this path; Commit B's chaos tests do NOT
  use this — they're testing the post-crash path).
* :func:`simulate_watchdog_post_crash_cleanup` — POST-CRASH path.
  Mirrors the real Step 14 watchdog's semantics: walks
  ``pyforge:pid_segments:{dead_pid}``, DECRs refcounts, unlinks
  segments whose refcount hit zero. Does NOT call ``_force_drop_orphan``
  (different code path entirely — that's same-process cleanup, which
  the dead orchestrator's process can't run).

The helpers live here, not in ``tests/_helpers.py``, because they're
specifically Step 14 territory; isolating them telegraphs the future-
step touch.
"""

from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING

from pyforge._internal import posix_shm
from pyforge.shm import _key_pid_segments, _key_refcount

if TYPE_CHECKING:
    import redis

    from pyforge.schema import FeatureSchema
    from pyforge.shm import Segment


def simulate_orphan_cleanup_in_orchestrator(
    redis_client: redis.Redis,
    schema: type[FeatureSchema],
    segment: Segment,
) -> None:
    """The IN-PROCESS orphan path.

    Wraps :func:`pyforge.hydration._force_drop_orphan` for tests that
    need the same semantics outside the orchestrator's exception
    handler. NOT used by Commit B's chaos tests — those have a dead
    orchestrator process and need the post-crash helper below.

    Reserved for future tests that need same-process cleanup
    simulation (e.g. testing the orchestrator's contract under
    exception classes the unit tests don't already cover).
    """
    from pyforge.hydration import _force_drop_orphan

    _force_drop_orphan(redis_client, schema, segment)


def simulate_watchdog_post_crash_cleanup(
    redis_client: redis.Redis,
    dead_pid: int,
) -> int:
    """The POST-CRASH watchdog path. Mirrors what Step 14 will do.

    Walks ``pyforge:pid_segments:{dead_pid}`` for segments held by the
    dead process. For each: DECR refcount; if it hit zero,
    posix_shm.unlink the segment + DEL the refcount key. Then SREM
    each name from ``pyforge:pid_segments:{dead_pid}`` (drains the
    set; deletes the key when empty).

    Also drops any ``pyforge:schema:*:current`` key whose value points
    at a segment we just unlinked. This uses ``redis.keys()`` which
    is O(N keyspace) — fine for tests with a tiny key set, NEVER
    acceptable in production. Step 14's real watchdog will track
    schema -> segment mapping via a sidetable to avoid this scan.

    Returns the number of segments successfully unlinked.

    Critically: does NOT call :func:`pyforge.hydration._force_drop_orphan`
    — different code path entirely. That helper is the orchestrator's
    in-process cleanup (knows the ``Segment`` object, has the handle,
    can call ``posix_shm.close``). The watchdog runs in a different
    process that didn't create the segment (only knows the name from
    Redis state, has no handle).

    Future-step note: when Step 14's real watchdog ships, this helper
    either (a) wraps ``pyforge.watchdog.cleanup_dead_pid(...)`` or
    similar, or (b) gets deleted in favor of calling that function
    directly. Either way, the contract this helper documents is what
    Step 14 must implement.
    """
    seg_names_key = _key_pid_segments(dead_pid)
    raw_names = redis_client.smembers(seg_names_key) or set()
    unlinked = 0

    for raw_name in raw_names:
        name = raw_name.decode() if isinstance(raw_name, bytes) else raw_name
        new_count = int(redis_client.decr(_key_refcount(name)))
        if new_count <= 0:
            with contextlib.suppress(Exception):
                posix_shm.unlink(name)
                unlinked += 1
            redis_client.delete(_key_refcount(name))

            # Drop schema:*:current pointers that match this name.
            # NOTE: redis.keys() is O(N keyspace) — test-only.
            current_keys = redis_client.keys("pyforge:schema:*:current")
            for ck in current_keys:
                ck_str = ck.decode() if isinstance(ck, bytes) else ck
                v = redis_client.get(ck_str)
                v_str = v.decode() if isinstance(v, bytes) else v
                if v_str == name:
                    redis_client.delete(ck_str)

        # Drain the pid_segments set entry regardless of refcount
        # outcome (we're "absorbing" the dead pid's holdings).
        redis_client.srem(seg_names_key, raw_name)

    # If the set is now empty, the smembers call above returned all
    # there was; the SREMs above drained them. Best-effort delete the
    # set key (Redis auto-deletes empty sets, so this is belt-and-
    # suspenders for environments where that doesn't apply).
    with contextlib.suppress(Exception):
        redis_client.delete(seg_names_key)

    return unlinked


def drain_cleanup_queue(redis_client: redis.Redis) -> int:
    """Drain ``pyforge:cleanup_queue`` set, unlinking each segment.

    Returns count drained.

    The cleanup_queue is ``SADD``-populated by ``SegmentRegistry.close``'s
    Lua script when the refcount hits zero (see
    pyforge/shm.py::_CLOSE_LUA). The real Step 14 watchdog will dequeue
    entries and ``posix_shm.unlink`` each one. Tests use this to
    simulate that drain after a clean ``registry.close`` (e.g. E1's
    "operator drops current segment" simulation).
    """
    drained = 0
    while True:
        name_bytes = redis_client.spop("pyforge:cleanup_queue")
        if name_bytes is None:
            break
        name = name_bytes.decode() if isinstance(name_bytes, bytes) else name_bytes
        with contextlib.suppress(Exception):
            posix_shm.unlink(name)
        drained += 1
    return drained


# Re-export the key helpers so chaos tests don't need to reach into
# pyforge.shm's private surface in two places (the helpers above are
# the canonical entry points, but tests that need to introspect Redis
# state directly can use these).
__all__ = [
    "drain_cleanup_queue",
    "simulate_orphan_cleanup_in_orchestrator",
    "simulate_watchdog_post_crash_cleanup",
]
