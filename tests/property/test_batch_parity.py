"""Parity tests: pyforge.assembly.assemble_batch vs N-loops of serving.assemble.

THE Step 8 test. Hypothesis-driven over random schemas + random batches with
NaN-poisoned output buffers. The batch kernel's per-row output must be
byte-identical to the Python oracle for hit rows, and zero-filled for miss
rows. Mirrors tests/property/test_assembly_parity.py's structure.
"""

from __future__ import annotations

import sys

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

pytestmark = pytest.mark.skipif(
    sys.platform == "win32",
    reason="assembly requires POSIX (Linux/WSL2)",
)

from _helpers import (  # noqa: E402
    build_dynamic_schema,
    field_list_strategy,
    make_segment,
    pack_row,
    random_value_for,
    release_segment,
)
from pyforge.assembly import assemble_batch, prewarm  # noqa: E402
from pyforge.layout import insert  # noqa: E402
from pyforge.schema import FeatureField, total_element_count  # noqa: E402
from pyforge.serving import assemble as assemble_python  # noqa: E402

_HYPO = settings(
    max_examples=200,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture],
)


@pytest.fixture(scope="module", autouse=True)
def _prewarm_numba() -> None:
    prewarm()


# ---------------------------------------------------------------------------
# Property 1: each batch row is byte-identical to single-entity assemble.
# ---------------------------------------------------------------------------


@_HYPO
@given(
    field_list=field_list_strategy(),
    seed=st.integers(min_value=0, max_value=2**31 - 1),
    n_present=st.integers(min_value=1, max_value=8),
    n_missing=st.integers(min_value=0, max_value=4),
)
def test_batch_matches_n_single_calls(
    field_list: list[FeatureField],
    seed: int,
    n_present: int,
    n_missing: int,
) -> None:
    """For random schema + random N entities (some present, some missing):

    - Hit rows: ``out[i]`` is byte-identical to ``serving.assemble(seg, ids[i])``.
    - Miss rows: ``out[i]`` is all-zero.
    - ``mask[i]`` reflects hit/miss correctly.

    NaN-poisons the output buffer before the call so a "forgot to write
    index k" bug fails loudly instead of returning a coincidentally-correct
    leftover.
    """
    schema = build_dynamic_schema(field_list)
    rng = np.random.default_rng(seed)
    elem_count = total_element_count(schema)
    capacity = max(2, n_present + 2)

    seg = make_segment(schema, capacity=capacity)
    try:
        # Insert n_present entities with random data.
        present_ids = [f"present_{i}" for i in range(n_present)]
        present_values: dict[str, dict[str, np.ndarray]] = {}
        for eid in present_ids:
            values = {f.name: random_value_for(f, rng) for f in schema.fields}
            present_values[eid] = values
            insert(seg, eid, pack_row(schema, values))

        # Build query batch interleaving present + missing.
        missing_ids = [f"ghost_{i}" for i in range(n_missing)]
        all_ids = present_ids + missing_ids
        rng.shuffle(all_ids)
        n_batch = len(all_ids)

        # NaN-poison out so any unwritten position fails the comparison.
        out = np.full((n_batch, elem_count), np.nan, dtype=np.float32)
        mask = np.full(n_batch, True, dtype=np.bool_)  # poison mask too

        out_back, mask_back = assemble_batch(seg, all_ids, out=out, found_mask=mask)
        assert out_back is out
        assert mask_back is mask

        for i, eid in enumerate(all_ids):
            if eid in present_values:
                expected = assemble_python(seg, eid)
                np.testing.assert_array_equal(out[i], expected)
                assert mask[i]
            else:
                # Miss row: all zeros.
                np.testing.assert_array_equal(out[i], np.zeros(elem_count, dtype=np.float32))
                assert not mask[i]
    finally:
        release_segment(seg)


# ---------------------------------------------------------------------------
# Property 2: idempotence under repeat call.
# ---------------------------------------------------------------------------


@_HYPO
@given(
    field_list=field_list_strategy(),
    seed=st.integers(min_value=0, max_value=2**31 - 1),
)
def test_batch_idempotent_under_repeat(field_list: list[FeatureField], seed: int) -> None:
    """Two consecutive calls with identical inputs produce byte-identical outputs."""
    schema = build_dynamic_schema(field_list)
    rng = np.random.default_rng(seed)
    seg = make_segment(schema, capacity=4)
    try:
        ids = ["a", "b"]
        for eid in ids:
            values = {f.name: random_value_for(f, rng) for f in schema.fields}
            insert(seg, eid, pack_row(schema, values))

        out1, mask1 = assemble_batch(seg, ids)
        out2, mask2 = assemble_batch(seg, ids)

        np.testing.assert_array_equal(out1, out2)
        np.testing.assert_array_equal(mask1, mask2)
    finally:
        release_segment(seg)


# ---------------------------------------------------------------------------
# Property 3: order independence — permuting input permutes output identically.
# ---------------------------------------------------------------------------


@_HYPO
@given(
    field_list=field_list_strategy(),
    seed=st.integers(min_value=0, max_value=2**31 - 1),
)
def test_batch_order_independence(field_list: list[FeatureField], seed: int) -> None:
    """``assemble_batch([a, b, c])[i]`` matches ``assemble_batch([c, b, a])[order[i]]``
    for any permutation."""
    schema = build_dynamic_schema(field_list)
    rng = np.random.default_rng(seed)
    seg = make_segment(schema, capacity=8)
    try:
        ids = ["x", "y", "z"]
        for eid in ids:
            values = {f.name: random_value_for(f, rng) for f in schema.fields}
            insert(seg, eid, pack_row(schema, values))

        out_orig, _ = assemble_batch(seg, ids)
        permuted = ["z", "x", "y"]
        out_perm, _ = assemble_batch(seg, permuted)

        # out_perm[0] == out_orig[2] (both are "z")
        # out_perm[1] == out_orig[0] (both are "x")
        # out_perm[2] == out_orig[1] (both are "y")
        np.testing.assert_array_equal(out_perm[0], out_orig[2])
        np.testing.assert_array_equal(out_perm[1], out_orig[0])
        np.testing.assert_array_equal(out_perm[2], out_orig[1])
    finally:
        release_segment(seg)
