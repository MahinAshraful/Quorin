"""Shared Redis-client helpers (CR.E.6 / v0.1.1).

Quorin's producer / consumer / heartbeat all run blocking Redis calls
that can hang indefinitely on a network partition unless the client has
a finite ``socket_timeout``. The library can't enforce this — operators
construct their own ``redis.Redis`` — but we CAN warn at the boundary
so a partition is loud (UserWarning) instead of a silent thread leak.

The check is best-effort: any AttributeError accessing internal client
attributes (different ``redis-py`` versions, custom subclasses, mocks)
is swallowed. That keeps the helper safe to call from every entry-point
without becoming a fragile coupling.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass  # Type annotations use the public ``object`` parameter type;
    # avoiding redis-import here keeps this module Numba-isolation
    # safe (invariant #11) — it gets pulled by quorin.wal_consumer
    # which already imports redis.


def warn_if_no_socket_timeout(client: object, *, source: str) -> None:
    """Emit a :class:`UserWarning` if ``client`` has no finite ``socket_timeout``.

    Called from :class:`quorin.wal.WALProducer.__init__`,
    :class:`quorin.wal_consumer.WALConsumer.__init__`, and
    :func:`quorin._internal.heartbeat.ensure_started`. Each of those
    runs blocking Redis ops on threads or coroutines that can be
    silently stuck for the duration of a network partition without a
    finite timeout. Surfacing a UserWarning at construction time is
    the cheapest signal — operators see it during dev / smoke tests
    rather than a 30-minute-stuck consumer in prod.

    Args:
        client: A ``redis.Redis`` or ``redis.asyncio.Redis`` instance.
        source: Free-form label identifying where the check was triggered
            (e.g. ``"WALProducer"``). Goes into the warning message.

    Best-effort: AttributeError swallowed silently. Different redis-py
    versions store socket_timeout in different places (connection_pool
    or connection_kwargs); this helper checks both common shapes.
    """
    try:
        # redis-py 5.x: client.connection_pool.connection_kwargs
        # holds the constructor kwargs.
        kwargs = getattr(client, "connection_pool", None)
        if kwargs is not None:
            kwargs = getattr(kwargs, "connection_kwargs", None)
        if kwargs is None:
            # No connection pool surface — likely a mock or custom
            # subclass. Don't warn (would be too noisy in tests).
            return
        timeout = kwargs.get("socket_timeout")
    except (AttributeError, TypeError):
        # Connection-pool surface evolved or isn't available; can't
        # check — silently skip.
        return

    if timeout is None:
        warnings.warn(
            f"{source}: redis_client has no socket_timeout configured. "
            "Blocking Redis calls can hang indefinitely on a network "
            "partition; the heartbeat / consumer thread may leak even "
            "after stop() returns. Pass socket_timeout (e.g. 5.0 seconds) "
            "to redis.Redis() or redis.asyncio.Redis() at construction. "
            "(CR.E.6 / ADR-018)",
            UserWarning,
            stacklevel=3,
        )
