"""Structured JSON logging.

Every Quorin subsystem (serving loop, WAL consumer, watchdog) logs through
the same configuration so logs are greppable by ``pid`` and ``schema`` without
post-processing.
"""

from __future__ import annotations

import logging
import os
from typing import Any

import structlog
from structlog.contextvars import bind_contextvars, merge_contextvars
from structlog.stdlib import BoundLogger

# Single-element list instead of a module-level bool so we do not need
# ``global`` (ruff PLW0603). The indirection is cheap and the idempotency
# contract is preserved.
_state: dict[str, bool] = {"configured": False}


def configure(level: str = "INFO") -> None:
    """Configure structlog + stdlib logging once per process.

    Idempotent: subsequent calls are no-ops. Emits JSON to stderr with a
    persistent ``pid`` field and any context bound via :func:`bind`.
    """
    if _state["configured"]:
        return

    logging.basicConfig(
        format="%(message)s",
        level=getattr(logging, level.upper()),
    )

    structlog.configure(
        processors=[
            merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(getattr(logging, level.upper())),
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )
    bind_contextvars(pid=os.getpid())
    _state["configured"] = True


def get_logger(name: str | None = None) -> BoundLogger:
    """Return a structlog ``BoundLogger`` for the given component name.

    Auto-calls :func:`configure` on first use so callers don't have to wire
    structlog setup themselves.
    """
    if not _state["configured"]:
        configure()
    return structlog.get_logger(name)  # type: ignore[no-any-return]


def bind(**kwargs: Any) -> None:
    """Bind context vars visible on every log line from this point on."""
    bind_contextvars(**kwargs)
