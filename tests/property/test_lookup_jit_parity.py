"""Parity tests: quorin.layout.lookup (Python) vs quorin._internal.lookup_kernel.lookup_jit (Numba).

THE Step 16c parity test. Hypothesis-driven over random schemas + random
entity-id sets. Both implementations must return byte-identical row
offsets for any inserted entity, and both must return None for any
non-inserted entity.

If this test ever fails, lookup-jit has regressed against the Python
oracle and ADR-017's design contract is broken.

Note: tests/property/test_assembly_parity.py provides transitive coverage
because quorin.assembly.assemble now calls lookup_jit internally — any
divergence there would also break the assembly parity test. This file is
the focused single-primitive parity guard, easier to debug when only
lookup is wrong.
"""

from __future__ import annotations

import string
import sys

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

pytestmark = pytest.mark.skipif(
    sys.platform == "win32",
    reason="lookup_kernel requires POSIX (Linux/WSL2)",
)

from _helpers import (  # noqa: E402
    build_dynamic_schema,
    field_list_strategy,
    make_segment,
    release_segment,
)
from quorin._internal.lookup_kernel import lookup_jit, prewarm  # noqa: E402
from quorin.layout import insert, lookup  # noqa: E402
from quorin.schema import FeatureField  # noqa: E402

_HYPO = settings(
    max_examples=200,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture],
)


@pytest.fixture(scope="module", autouse=True)
def _prewarm_lookup_kernel() -> None:
    """Module-level pre-warm so the first parity test doesn't pay
    ~100-200 ms compile cost (which would skew Hypothesis's deadline tracking
    and waste real time)."""
    prewarm()


# ---------------------------------------------------------------------------
# ID generators. Mix ASCII + Unicode + UUIDs to exercise the multi-byte
# UTF-8 byte-compare path in the kernel.
# ---------------------------------------------------------------------------

_ID_TEXT = st.text(
    alphabet=string.ascii_letters + string.digits + "_-",
    min_size=1,
    max_size=32,
)

# A couple of explicit Unicode strings to ensure non-ASCII coverage; Hypothesis's
# default text strategy is ASCII-heavy in practice.
_ID_UNICODE = st.sampled_from(
    [
        "日本語_xyz",
        "café_42",
        "résumé",
        "Spëcial_chårs",
        "user_русский",
    ]
)


_ID = st.one_of(_ID_TEXT, _ID_UNICODE)


@st.composite
def _unique_id_list(draw: st.DrawFn) -> list[str]:
    """A non-empty list of distinct entity IDs (1-30 elements)."""
    raw = draw(st.lists(_ID, min_size=1, max_size=30))
    seen: set[str] = set()
    deduped: list[str] = []
    for s in raw:
        if s not in seen:
            seen.add(s)
            deduped.append(s)
    return deduped


# ---------------------------------------------------------------------------
# Properties.
# ---------------------------------------------------------------------------


@_HYPO
@given(field_list=field_list_strategy(), entity_ids=_unique_id_list())
def test_hit_path_parity(field_list: list[FeatureField], entity_ids: list[str]) -> None:
    """For every inserted entity_id, lookup_jit returns the same row_offset
    as quorin.layout.lookup."""
    schema = build_dynamic_schema(field_list)
    capacity = max(len(entity_ids), 2)
    seg = make_segment(schema, capacity=capacity)
    try:
        row = bytes(seg.layout.row_size)
        for eid in entity_ids:
            insert(seg, eid, row)

        for eid in entity_ids:
            py_off = lookup(seg, eid)
            jit_off = lookup_jit(seg, eid)
            assert py_off is not None, f"Python lookup missed inserted {eid!r}"
            assert jit_off == py_off, f"lookup_jit({eid!r}) = {jit_off} != python lookup {py_off}"
    finally:
        release_segment(seg)


@_HYPO
@given(
    field_list=field_list_strategy(),
    inserted_ids=_unique_id_list(),
    queried_ids=_unique_id_list(),
)
def test_miss_path_parity(
    field_list: list[FeatureField],
    inserted_ids: list[str],
    queried_ids: list[str],
) -> None:
    """For IDs NOT inserted, both implementations return None."""
    schema = build_dynamic_schema(field_list)
    capacity = max(len(inserted_ids), 2)
    seg = make_segment(schema, capacity=capacity)
    try:
        row = bytes(seg.layout.row_size)
        inserted_set = set(inserted_ids)
        for eid in inserted_ids:
            insert(seg, eid, row)

        # Query only IDs that were NOT inserted (subtract the inserted set).
        misses = [q for q in queried_ids if q not in inserted_set]
        for eid in misses:
            py_off = lookup(seg, eid)
            jit_off = lookup_jit(seg, eid)
            assert py_off is None, f"Python lookup unexpectedly hit on {eid!r}"
            assert jit_off is None, f"lookup_jit unexpectedly hit on {eid!r}: {jit_off}"
    finally:
        release_segment(seg)
