"""Parity tests: pyforge.assembly.assemble_batch with pooled vs fresh out=.

Asserts the buffer-pool integration produces byte-identical results to a
freshly-allocated output buffer, and that the pool's ``available`` count
stays bounded by ``max_size`` across many checkout cycles.
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
from pyforge.pool import BatchBufferPool  # noqa: E402
from pyforge.schema import FeatureField, FeatureSchema, dtype  # noqa: E402


# Module-level schema for the bounded-availability property test (the test
# itself doesn't need a populated segment; it just exercises checkout cycles).
class _PropertyOneScalar(FeatureSchema):
    version = 1
    fields = [FeatureField("a", dtype.float32)]


_HYPO = settings(
    max_examples=100,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture],
)


@pytest.fixture(scope="module", autouse=True)
def _prewarm_numba() -> None:
    prewarm()


@_HYPO
@given(
    field_list=field_list_strategy(),
    seed=st.integers(min_value=0, max_value=2**31 - 1),
    n_batch=st.integers(min_value=1, max_value=8),
)
def test_pooled_assemble_matches_fresh_assemble(
    field_list: list[FeatureField],
    seed: int,
    n_batch: int,
) -> None:
    """Pooled out= buffer produces byte-identical results to fresh allocation.

    Uses ``zero_on_return=True`` to remove the cross-call data residue
    variable. The kernel always overwrites every position so the default
    ``False`` is also safe; we verify the safer default in the unit tests.
    """
    schema = build_dynamic_schema(field_list)
    rng = np.random.default_rng(seed)
    capacity = max(2, n_batch + 2)

    seg = make_segment(schema, capacity=capacity)
    try:
        ids = [f"id_{i}" for i in range(n_batch)]
        for eid in ids:
            values = {f.name: random_value_for(f, rng) for f in schema.fields}
            insert(seg, eid, pack_row(schema, values))

        # Fresh-allocation reference.
        out_fresh, mask_fresh = assemble_batch(seg, ids)

        # Pooled (zero_on_return=True so a re-checkout is also clean).
        pool = BatchBufferPool(schema, batch_size=n_batch, max_size=2, zero_on_return=True)
        with pool.checkout() as buf:
            out_pooled, mask_pooled = assemble_batch(seg, ids, out=buf)
            assert out_pooled is buf
            np.testing.assert_array_equal(out_pooled, out_fresh)
            np.testing.assert_array_equal(mask_pooled, mask_fresh)
    finally:
        release_segment(seg)


@_HYPO
@given(
    n_batch=st.integers(min_value=1, max_value=4),
    max_size=st.integers(min_value=1, max_value=4),
    cycles=st.integers(min_value=10, max_value=50),
)
def test_pool_available_bounded_by_max_size(n_batch: int, max_size: int, cycles: int) -> None:
    """After many checkout/return cycles, ``pool.available <= max_size`` always."""
    pool = BatchBufferPool(_PropertyOneScalar, batch_size=n_batch, max_size=max_size)
    for _ in range(cycles):
        with pool.checkout():
            pass
    assert pool.available <= pool.max_size
    # Steady-state pool returns to its pre-burst capacity.
    assert pool.available == pool.max_size
