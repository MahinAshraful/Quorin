"""pytest-benchmark fixtures.

Real GC isolation and warm-up logic land alongside Step 5 (Numba) and
Step 7 (GC management). This file exists so ``pytest benchmarks/`` collects
cleanly from day one.
"""

from __future__ import annotations

import gc

import pytest


@pytest.fixture
def isolate_gc():
    """Disable GC for the duration of a single benchmark. Step 7 will flesh this out."""
    gc.collect()
    gc.disable()
    try:
        yield
    finally:
        gc.enable()
