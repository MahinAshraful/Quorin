"""Module-import hygiene tests — locks invariant #11 (Numba isolation).

A future contributor adds a metric pre-warm or constant import that
naively pulls ``pyforge.hydration`` into a hot-path module, and silently
breaks invariant #11 — every importer of that module pays Numba's ~200ms
LLVM init. The bug surfaces as a 200ms slowdown on ``import pyforge.serving``
six months later and takes two days to localize.

These parametrized tests run a clean import of each module that MUST
stay Numba-free and assert no ``numba.*`` module ended up in
``sys.modules``. Cheap (~10ms per module), locks the invariant globally.

``pyforge.hydration``, ``pyforge.assembly``, ``pyforge._internal.insert_kernel``,
``pyforge._internal.lookup_kernel`` (Step 16c), and
``pyforge._internal.hash_kernel`` (Step 16c-d, Numba BLAKE2b) are the
LEGITIMATE Numba-pulling modules — they are deliberately excluded from
this list. If a future step adds another legitimate Numba module,
append it here and document the rationale.
"""

from __future__ import annotations

import importlib
import sys

import pytest

# Modules whose import paths must NOT transitively pull Numba.
# pyforge.hydration is the legitimate Numba-pulling module on the bulk
# path; everything below stays clean. If any of these starts importing
# hydration (or insert_kernel directly), Numba init costs ~200ms get paid
# by every importer of that module — slaughtering startup.
_NUMBA_FREE_MODULES = (
    "pyforge.metrics",
    "pyforge.layout",
    "pyforge.serving",
    "pyforge.wal_consumer",
    "pyforge.wal",
    "pyforge.shm",
    "pyforge.schema",
    "pyforge.pool",
    "pyforge.offline",
    "pyforge.logging",
)


@pytest.mark.parametrize("module_name", _NUMBA_FREE_MODULES)
def test_module_does_not_pull_numba(module_name: str) -> None:
    """Invariant #11: importing ``module_name`` must not transitively pull numba.

    Snapshots and restores ``sys.modules`` so subsequent tests see the
    original module cache (otherwise the pop + re-import would leave
    fresh module instances behind, breaking tests that assume
    ``pyforge.metrics.registry`` is a singleton).
    """
    saved_modules = dict(sys.modules)
    try:
        # Drop cached pyforge / numba modules so the test sees a clean import.
        for mod in list(sys.modules):
            if mod.startswith(("numba", "pyforge")):
                sys.modules.pop(mod, None)

        importlib.import_module(module_name)

        pulled = sorted(m for m in sys.modules if m.startswith("numba"))
        assert not pulled, (
            f"{module_name} transitively imports numba: {pulled}. "
            f"This forces ~200ms LLVM init on every importer of "
            f"{module_name}. Check for accidental imports of "
            f"`pyforge.hydration`, `pyforge.assembly`, "
            f"`pyforge._internal.insert_kernel`, "
            f"`pyforge._internal.lookup_kernel`, or "
            f"`pyforge._internal.hash_kernel`."
        )
    finally:
        # Restore the original module cache so we don't leave fresh
        # pyforge.* instances behind for the rest of the suite.
        sys.modules.clear()
        sys.modules.update(saved_modules)
