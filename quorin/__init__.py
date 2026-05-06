"""Quorin - low-latency ML feature serving via shared memory.

Public API: schema evolution helpers (``upgrade_schema``, ``can_upgrade``,
error classes) are exposed via module ``__getattr__`` so importing any
non-evolution submodule (e.g. ``quorin.metrics`` or ``quorin.serving``)
does NOT transitively pull Numba via ``quorin.evolution``'s
``insert_many`` import. Locks CLAUDE.md invariant #11.

For non-evolution modules, import from the specific module:
``from quorin.schema import FeatureSchema``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

__version__ = "0.1.0"

if TYPE_CHECKING:  # mypy + IDE see the names directly
    # Redundant aliases (`as Name`) silence the F401-in-TYPE_CHECKING-block
    # warning while still exposing the names for type checkers + IDEs.
    from quorin.evolution import UpgradeAbortedError as UpgradeAbortedError
    from quorin.evolution import UpgradeConflictError as UpgradeConflictError
    from quorin.evolution import UpgradeError as UpgradeError
    from quorin.evolution import UpgradeIncompatibleError as UpgradeIncompatibleError
    from quorin.evolution import UpgradeResult as UpgradeResult
    from quorin.evolution import can_upgrade as can_upgrade
    from quorin.evolution import upgrade_schema as upgrade_schema

_EVOLUTION_NAMES = frozenset(
    {
        "UpgradeAbortedError",
        "UpgradeConflictError",
        "UpgradeError",
        "UpgradeIncompatibleError",
        "UpgradeResult",
        "can_upgrade",
        "upgrade_schema",
    }
)


def __getattr__(name: str) -> Any:
    """Lazy resolution of the evolution-module re-exports.

    Triggered on attribute access like ``quorin.upgrade_schema(...)``;
    NOT on ``import quorin.metrics`` (which only runs this module's
    top-level code, not its ``__getattr__``). That is precisely what
    keeps Numba off the import path of every other submodule.
    """
    if name in _EVOLUTION_NAMES:
        from quorin import evolution as _ev  # noqa: PLC0415 - invariant #11

        return getattr(_ev, name)
    raise AttributeError(f"module 'quorin' has no attribute {name!r}")


__all__ = [
    "UpgradeAbortedError",
    "UpgradeConflictError",
    "UpgradeError",
    "UpgradeIncompatibleError",
    "UpgradeResult",
    "__version__",
    "can_upgrade",
    "upgrade_schema",
]
