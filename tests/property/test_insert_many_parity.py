"""Hypothesis property tests — bulk insert_many vs single-insert oracle parity.

Two properties:

* **P1 — byte-identical parity.** For any random schema (1-5 fields,
  scalar/1D/2D shapes) x random 1-30 entity IDs x random row data, a
  segment built via the per-row :func:`pyforge.layout.insert` loop is
  byte-identical to a segment built via :func:`pyforge._internal.insert_kernel.insert_many`
  from a PyArrow table of the same data. Locks the bulk kernel against
  any future regression in slot layout, string pool encoding, or
  per-field byte placement.

* **P2 — lookup equivalence.** Same generated input as P1. For every
  entity ID, ``lookup(seg_A, eid) == lookup(seg_B, eid)`` (both the
  row index integer AND the resulting bytes). Cheaper to run than P1's
  byte-equal; doubles as a sanity check that the slot table is
  semantically equivalent even if some reserved/padding bytes ever
  differ.

Memory note: 80 examples x 2 segments x ~1 MB each = ~160 MB transient
``/dev/shm`` usage during the property suite. The autouse
``_shm_test_isolation`` fixture (CLAUDE.md §7.4) cleans between tests.
Future contributors expanding the example count should re-check
``/dev/shm`` headroom on CI workers.
"""

from __future__ import annotations

import string
import sys
from typing import Any

import numpy as np
import pyarrow as pa
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

pytestmark = pytest.mark.skipif(
    sys.platform == "win32",
    reason="insert_many parity requires POSIX (Linux/WSL2)",
)

from _helpers import (  # noqa: E402
    build_dynamic_schema,
    field_list_strategy,
    make_segment,
    pack_row,
    random_value_for,
    release_segment,
)
from pyforge._internal.arrow_schema import _RESERVED_FIELD_NAMES  # noqa: E402
from pyforge._internal.insert_kernel import insert_many  # noqa: E402
from pyforge.layout import insert, lookup  # noqa: E402
from pyforge.schema import (  # noqa: E402
    DType,
    FeatureField,
    FeatureSchema,
)

# ---------------------------------------------------------------------------
# Helpers — arrow column construction for scalar/1D/2D fields.
# ---------------------------------------------------------------------------


_DTYPE_TO_PA = {
    DType.FLOAT32: pa.float32(),
    DType.FLOAT64: pa.float64(),
    DType.INT32: pa.int32(),
    DType.INT64: pa.int64(),
    DType.UINT8: pa.uint8(),
}


def _pa_column_for_field(field: FeatureField, stacked: np.ndarray[Any, np.dtype[Any]]) -> pa.Array:
    """Convert ``stacked`` (shape ``(n_rows, *field.shape)``) to a PyArrow
    array matching ``insert_kernel._extract_field_to_numpy``'s rank
    expectations.

    Rank dispatch:

    * rank 0 (scalar): ``pa.array(stacked)``.
    * rank 1 (``shape=(k,)``): ``pa.FixedSizeListArray`` over flattened values.
    * rank 2 (``shape=(r, c)``): nested ``pa.FixedSizeListArray``.
    """
    rank = len(field.shape)
    pa_elem = _DTYPE_TO_PA[field.dtype]
    if rank == 0:
        # `random_value_for` returns shape (1,) for scalar fields, so
        # `np.stack` produces (n, 1). Flatten to (n,) for pa.array.
        return pa.array(stacked.reshape(-1), type=pa_elem)
    if rank == 1:
        (k,) = field.shape
        flat = pa.array(stacked.reshape(-1), type=pa_elem)
        return pa.FixedSizeListArray.from_arrays(flat, k)
    if rank == 2:
        r, c = field.shape
        flat = pa.array(stacked.reshape(-1), type=pa_elem)
        inner = pa.FixedSizeListArray.from_arrays(flat, c)
        return pa.FixedSizeListArray.from_arrays(inner, r)
    raise NotImplementedError(f"rank {rank} not supported")


def _build_table(
    schema: type[FeatureSchema],
    eids: list[str],
    per_entity_values: list[dict[str, np.ndarray[Any, np.dtype[Any]]]],
) -> pa.Table:
    """Build a PyArrow table conforming to ``schema``."""
    n = len(eids)
    cols: dict[str, pa.Array] = {
        "entity_id": pa.array(eids, type=pa.string()),
    }
    for f in schema.fields:
        # Stack per-entity ndarrays into shape (n, *f.shape).
        stacked = np.stack([per_entity_values[i][f.name] for i in range(n)])
        cols[f.name] = _pa_column_for_field(f, stacked)
    return pa.table(cols)


# ---------------------------------------------------------------------------
# Hypothesis strategies — schema + per-row values, filtered for reserved names.
# ---------------------------------------------------------------------------


# Reserved names the writer (Step 11) and reader (Step 12) refuse. Filter
# at strategy level so we don't waste examples on `assume(...)` rejections.
_RESERVED_PLUS_QUERY: frozenset[str] = _RESERVED_FIELD_NAMES | {"as_of_time"}


@st.composite
def _schema_strategy(draw: st.DrawFn) -> type[FeatureSchema]:
    """A FeatureSchema with 1-5 unique-named fields, no reserved names."""
    fields = draw(field_list_strategy())
    fields = [f for f in fields if f.name not in _RESERVED_PLUS_QUERY]
    if not fields:
        # Hypothesis can leave us with zero fields after filtering; nudge
        # toward a non-empty schema by appending a fixed scalar field.
        fields = [FeatureField("v", DType.FLOAT32, ())]
    return build_dynamic_schema(fields[:5])


_EID_ALPHABET = string.ascii_letters + string.digits + "-_"
_eid_strategy = st.text(alphabet=_EID_ALPHABET, min_size=1, max_size=20)


# ---------------------------------------------------------------------------
# Test runner — shared between P1 and P2.
# ---------------------------------------------------------------------------


def _run_paths(
    schema: type[FeatureSchema], eids: list[str], rng: np.random.Generator
) -> tuple[Any, Any, list[dict[str, np.ndarray[Any, np.dtype[Any]]]]]:
    """Build two segments — one via per-row insert, one via insert_many — and
    return both segments + the per-entity value dicts. Caller releases.
    """
    n = len(eids)

    # Generate per-entity values once; both paths use them.
    per_entity_values: list[dict[str, np.ndarray[Any, np.dtype[Any]]]] = []
    for _ in range(n):
        row_vals: dict[str, np.ndarray[Any, np.dtype[Any]]] = {}
        for f in schema.fields:
            arr = random_value_for(f, rng).reshape(f.shape) if f.shape else random_value_for(f, rng)
            row_vals[f.name] = arr
        per_entity_values.append(row_vals)

    # Sized to give plenty of slot table headroom (slot table = 2 * capacity).
    capacity = max(2 * n, 16)

    seg_loop = make_segment(schema, capacity=capacity)
    seg_bulk = make_segment(schema, capacity=capacity)

    # Path 1: per-row insert via pack_row helper.
    for i, eid in enumerate(eids):
        row_bytes = pack_row(schema, per_entity_values[i])
        insert(seg_loop, eid, row_bytes)

    # Path 2: bulk insert via PyArrow table.
    table = _build_table(schema, eids, per_entity_values)
    insert_many(seg_bulk, table)

    return seg_loop, seg_bulk, per_entity_values


# ---------------------------------------------------------------------------
# P1 — byte-identical parity.
# ---------------------------------------------------------------------------


@given(
    schema=_schema_strategy(),
    eids=st.lists(_eid_strategy, min_size=1, max_size=30, unique=True),
    seed=st.integers(min_value=0, max_value=2**32 - 1),
)
@settings(
    max_examples=80,
    deadline=None,  # /dev/shm IO + Hypothesis can exceed pytest's default deadline
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.data_too_large],
)
def test_insert_many_byte_identical(
    schema: type[FeatureSchema],
    eids: list[str],
    seed: int,
) -> None:
    """P1: bulk insert_many produces a byte-identical segment to per-row insert.

    Locks the kernel against any future regression in slot layout, string
    pool encoding, or per-field byte placement.
    """
    rng = np.random.default_rng(seed)
    seg_loop, seg_bulk, _ = _run_paths(schema, eids, rng)
    try:
        # Compare the entire used region.
        loop_bytes = bytes(seg_loop.handle.buf[: seg_loop.layout.total_size])
        bulk_bytes = bytes(seg_bulk.handle.buf[: seg_bulk.layout.total_size])
        assert loop_bytes == bulk_bytes, (
            f"insert_many produced different bytes than per-row insert "
            f"for schema {schema.__name__} with {len(eids)} entities — "
            f"kernel divergence"
        )
    finally:
        release_segment(seg_loop)
        release_segment(seg_bulk)


# ---------------------------------------------------------------------------
# P2 — lookup equivalence.
# ---------------------------------------------------------------------------


@given(
    schema=_schema_strategy(),
    eids=st.lists(_eid_strategy, min_size=1, max_size=30, unique=True),
    seed=st.integers(min_value=0, max_value=2**32 - 1),
)
@settings(
    max_examples=80,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.data_too_large],
)
def test_insert_many_lookup_equivalent(
    schema: type[FeatureSchema],
    eids: list[str],
    seed: int,
) -> None:
    """P2: lookup(seg_loop, eid) == lookup(seg_bulk, eid) for every eid.

    ``lookup`` returns the byte offset of the entity's row (int) or
    None. Equal offsets + equal row bytes at that offset is a stronger
    check than P1's full-segment byte equality on a per-entity basis;
    catches cases where slot tables semantically agree but reserved
    padding diverges.
    """
    rng = np.random.default_rng(seed)
    seg_loop, seg_bulk, _ = _run_paths(schema, eids, rng)
    try:
        row_size = seg_loop.layout.row_size
        for eid in eids:
            off_loop = lookup(seg_loop, eid)
            off_bulk = lookup(seg_bulk, eid)
            assert off_loop is not None, (
                f"per-row insert: lookup({eid!r}) returned None — insert/lookup contract broken"
            )
            assert off_bulk is not None, (
                f"insert_many: lookup({eid!r}) returned None — bulk path divergence on slot table"
            )
            assert off_loop == off_bulk, (
                f"lookup({eid!r}) returned different byte offsets: "
                f"{off_loop} (per-row) vs {off_bulk} (bulk)"
            )
            row_loop_bytes = bytes(seg_loop.handle.buf[off_loop : off_loop + row_size])
            row_bulk_bytes = bytes(seg_bulk.handle.buf[off_bulk : off_bulk + row_size])
            assert row_loop_bytes == row_bulk_bytes, (
                f"lookup({eid!r}) row bytes differ between per-row and bulk paths"
            )
    finally:
        release_segment(seg_loop)
        release_segment(seg_bulk)
