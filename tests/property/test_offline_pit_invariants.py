"""Hypothesis property tests for ParquetDatasetStore.read_point_in_time (Step 12).

P1 — leak-free + lookback-bounded: every result row's event_time_ns
     is null OR (<= as_of_time AND >= as_of_time - lookback_ns).
P2 — idempotency (include_msg_id=True): two reads return equal results
     after sorting by (entity_id, as_of_time).
P3 — dedup correctness: matched feature row has a (msg_id_ms, msg_id_seq)
     present in the original features for that entity.
P4 — row count: len(result) == len(query_table) always.

P2 is gated to ``include_msg_id=True`` per Rev-3 HIGH-5: the
msg_id-extended sort makes asof tiebreaks deterministic.
"""

from __future__ import annotations

import asyncio
import sys
from typing import Any

import pytest

pytestmark = [
    pytest.mark.skipif(
        sys.platform == "win32",
        reason="Step 11/12 fsync semantics are POSIX-only",
    ),
    pytest.mark.property,
]

import hypothesis.strategies as st  # noqa: E402
import numpy as np  # noqa: E402
import pyarrow as pa  # noqa: E402
from hypothesis import HealthCheck, given, settings  # noqa: E402

from _helpers import build_dynamic_schema, field_list_strategy  # noqa: E402
from pyforge._internal.arrow_schema import (  # noqa: E402
    _RESERVED_FIELD_NAMES,
)
from pyforge._internal.arrow_schema import (  # noqa: E402
    clear_cache as clear_arrow_cache,
)
from pyforge.offline import ParquetDatasetStore  # noqa: E402
from pyforge.schema import DType, FeatureField, _hash_name  # noqa: E402

_DAY_NS = 86_400_000_000_000


def _python_value_for(field: FeatureField, rng: np.random.Generator) -> Any:
    """Generate a JSON-friendly Python value matching the field's shape and dtype.

    No NaN/inf in the property tests — they confuse equality checks
    across the dedup/asof path. The leak-free invariant is the focus
    here, not floating-point bit-pattern preservation (that's covered
    in test_offline_roundtrip).
    """
    n = field.element_count
    flat: list[Any]
    if field.dtype is DType.FLOAT32:
        flat = [float(x) for x in rng.standard_normal(n).astype(np.float32)]
    elif field.dtype is DType.FLOAT64:
        flat = [float(x) for x in rng.standard_normal(n).astype(np.float64)]
    elif field.dtype is DType.INT32:
        flat = [int(x) for x in rng.integers(-(1 << 20), 1 << 20, size=n, dtype=np.int32)]
    elif field.dtype is DType.INT64:
        flat = [int(x) for x in rng.integers(-(1 << 40), 1 << 40, size=n, dtype=np.int64)]
    elif field.dtype is DType.UINT8:
        flat = [int(x) for x in rng.integers(0, 256, size=n, dtype=np.uint8)]
    else:
        raise AssertionError(f"unhandled dtype {field.dtype}")

    if len(field.shape) == 0:
        return flat[0]
    if len(field.shape) == 1:
        return flat
    if len(field.shape) == 2:
        rows, cols = field.shape
        return [flat[i * cols : (i + 1) * cols] for i in range(rows)]
    raise AssertionError(f"unsupported shape {field.shape!r}")


def _populate_dataset(
    base: Any,
    field_list: list[FeatureField],
    n_rows: int,
    n_entities: int,
    seed: int,
    *,
    include_msg_id: bool = True,
) -> tuple[type[Any], list[tuple[str, int, bytes]]]:
    """Materialize a small dataset on disk. Returns (schema, written_keys).

    written_keys is a list of (entity_id, event_time_ns, msg_id) for
    every row written, used by the dedup-correctness invariant.
    """
    clear_arrow_cache()
    store = ParquetDatasetStore(base, include_msg_id=include_msg_id)
    schema = build_dynamic_schema(field_list)
    wire_order = sorted(schema.fields, key=lambda f: _hash_name(f.name))
    rng = np.random.default_rng(seed=seed)

    # event_time_ns spread over ~5 days so partition pruning has work
    # to do. ms_base ensures msg_id is monotonic even within the same
    # event_time_ns.
    base_event_time = 1_700_000_000_000_000_000
    written: list[tuple[str, int, bytes]] = []

    async def _write() -> None:
        for i in range(n_rows):
            entity_id = f"e{i % n_entities:03d}"
            # Vary event_time across rows
            event_time_ns = base_event_time + (i * _DAY_NS // 4)
            per_field = {f.name: _python_value_for(f, rng) for f in schema.fields}
            values_list = [per_field[f.name] for f in wire_order]
            msg_id = f"{1700_000_000_000 + i}-0".encode()
            await store.append(schema, entity_id, event_time_ns, values_list, msg_id)
            written.append((entity_id, event_time_ns, msg_id))
        await store.flush()

    asyncio.run(_write())
    return schema, written


def _build_queries(
    written: list[tuple[str, int, bytes]],
    n_queries: int,
    rng: np.random.Generator,
) -> pa.Table:
    """Build a query table that mixes match cases with nulls and far-past."""
    # Pick query times that include some hits and some misses.
    if not written:
        return pa.table(
            {
                "entity_id": pa.array([], type=pa.string()),
                "as_of_time": pa.array([], type=pa.int64()),
            }
        )
    eids = list({eid for eid, _, _ in written})
    times = [et for _, et, _ in written]
    min_t, max_t = min(times), max(times)

    rows: list[tuple[str, int]] = []
    for _ in range(n_queries):
        eid = eids[rng.integers(0, len(eids))]
        # ~70% chance to query within the dataset's time range; ~30% before.
        if rng.random() < 0.7:
            aot = int(min_t + rng.integers(0, max(1, max_t - min_t + 1)))
        else:
            aot = int(min_t - rng.integers(1, _DAY_NS * 100))
        rows.append((eid, aot))
    return pa.table(
        {
            "entity_id": pa.array([r[0] for r in rows], type=pa.string()),
            "as_of_time": pa.array([r[1] for r in rows], type=pa.int64()),
        }
    )


# ---------------------------------------------------------------------------
# P1 — leak-free + lookback-bounded.
# ---------------------------------------------------------------------------


@settings(
    max_examples=80,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture],
)
@given(
    field_list=field_list_strategy().filter(
        lambda fs: not any(f.name in _RESERVED_FIELD_NAMES or f.name == "as_of_time" for f in fs)
    ),
    n_rows=st.integers(min_value=1, max_value=20),
    n_entities=st.integers(min_value=1, max_value=5),
    n_queries=st.integers(min_value=1, max_value=15),
    lookback_days=st.integers(min_value=1, max_value=30),
    seed=st.integers(min_value=0, max_value=10_000),
)
def test_p1_leak_free_and_lookback_bounded(
    tmp_path_factory: pytest.TempPathFactory,
    field_list: list[FeatureField],
    n_rows: int,
    n_entities: int,
    n_queries: int,
    lookback_days: int,
    seed: int,
) -> None:
    """For every row, event_time_ns is null OR within [aot - lookback, aot]."""
    base = tmp_path_factory.mktemp("step12_p1")
    schema, written = _populate_dataset(base, field_list, n_rows, n_entities, seed)
    rng = np.random.default_rng(seed=seed)
    query = _build_queries(written, n_queries, rng)
    if len(query) == 0:
        return  # vacuously true

    store = ParquetDatasetStore(base)
    result = store.read_point_in_time(schema, query, lookback_days=lookback_days)
    lookback_ns = lookback_days * _DAY_NS

    aot_col = result["as_of_time"].to_pylist()
    et_col = result["event_time_ns"].to_pylist()
    for i, (aot, et) in enumerate(zip(aot_col, et_col, strict=True)):
        if et is None:
            continue
        assert et <= aot, f"row {i}: event_time {et} > as_of_time {aot} (LEAK)"
        assert et >= aot - lookback_ns, (
            f"row {i}: event_time {et} < as_of_time - lookback "
            f"({aot - lookback_ns}) (LOOKBACK VIOLATION)"
        )


# ---------------------------------------------------------------------------
# P2 — idempotency (include_msg_id=True only).
# ---------------------------------------------------------------------------


@settings(
    max_examples=80,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture],
)
@given(
    field_list=field_list_strategy().filter(
        lambda fs: not any(f.name in _RESERVED_FIELD_NAMES or f.name == "as_of_time" for f in fs)
    ),
    n_rows=st.integers(min_value=1, max_value=20),
    n_entities=st.integers(min_value=1, max_value=5),
    n_queries=st.integers(min_value=1, max_value=15),
    seed=st.integers(min_value=0, max_value=10_000),
)
def test_p2_idempotency_with_msg_id(
    tmp_path_factory: pytest.TempPathFactory,
    field_list: list[FeatureField],
    n_rows: int,
    n_entities: int,
    n_queries: int,
    seed: int,
) -> None:
    """Two reads produce equal results after sorting (Rev-3 HIGH-5 determinism)."""
    base = tmp_path_factory.mktemp("step12_p2")
    schema, written = _populate_dataset(
        base, field_list, n_rows, n_entities, seed, include_msg_id=True
    )
    rng = np.random.default_rng(seed=seed)
    query = _build_queries(written, n_queries, rng)
    if len(query) == 0:
        return

    store = ParquetDatasetStore(base)
    r1 = store.read_point_in_time(schema, query, lookback_days=30)
    r2 = store.read_point_in_time(schema, query, lookback_days=30)

    # Sort both by (entity_id, as_of_time) before equality. Within a
    # single read, order matches query input order; across two reads
    # it should also match, but sorting is belt-and-suspenders.
    r1s = r1.sort_by([("entity_id", "ascending"), ("as_of_time", "ascending")])
    r2s = r2.sort_by([("entity_id", "ascending"), ("as_of_time", "ascending")])
    assert r1s.equals(r2s), "two identical reads must produce equal results"


# ---------------------------------------------------------------------------
# P3 — dedup correctness: matched msg_id always belongs to the entity.
# ---------------------------------------------------------------------------


@settings(
    max_examples=80,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture],
)
@given(
    field_list=field_list_strategy().filter(
        lambda fs: not any(f.name in _RESERVED_FIELD_NAMES or f.name == "as_of_time" for f in fs)
    ),
    n_rows=st.integers(min_value=1, max_value=20),
    n_entities=st.integers(min_value=1, max_value=5),
    n_queries=st.integers(min_value=1, max_value=15),
    seed=st.integers(min_value=0, max_value=10_000),
)
def test_p3_matched_msg_id_belongs_to_entity(
    tmp_path_factory: pytest.TempPathFactory,
    field_list: list[FeatureField],
    n_rows: int,
    n_entities: int,
    n_queries: int,
    seed: int,
) -> None:
    """For each matched query, the (msg_id_ms, msg_id_seq) is in features for that entity."""
    base = tmp_path_factory.mktemp("step12_p3")
    schema, written = _populate_dataset(base, field_list, n_rows, n_entities, seed)
    rng = np.random.default_rng(seed=seed)
    query = _build_queries(written, n_queries, rng)
    if len(query) == 0:
        return

    # Build a set of (entity_id, msg_id_ms, msg_id_seq) tuples in
    # written features for membership check.
    valid_per_entity: dict[str, set[tuple[int, int]]] = {}
    for eid, _et, msg_id in written:
        ms_b, _, seq_b = msg_id.partition(b"-")
        valid_per_entity.setdefault(eid, set()).add((int(ms_b), int(seq_b)))

    store = ParquetDatasetStore(base)
    result = store.read_point_in_time(schema, query, lookback_days=30)
    eid_col = result["entity_id"].to_pylist()
    ms_col = result["msg_id_ms"].to_pylist()
    seq_col = result["msg_id_seq"].to_pylist()
    for i, (eid, ms, seq) in enumerate(zip(eid_col, ms_col, seq_col, strict=True)):
        if ms is None:
            assert seq is None, f"row {i}: msg_id_ms null but seq not null"
            continue
        valid = valid_per_entity.get(eid, set())
        assert (ms, seq) in valid, (
            f"row {i} entity {eid}: matched msg_id ({ms}, {seq}) not in features set {valid}"
        )


# ---------------------------------------------------------------------------
# P4 — row count.
# ---------------------------------------------------------------------------


@settings(
    max_examples=80,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture],
)
@given(
    field_list=field_list_strategy().filter(
        lambda fs: not any(f.name in _RESERVED_FIELD_NAMES or f.name == "as_of_time" for f in fs)
    ),
    n_rows=st.integers(min_value=0, max_value=20),
    n_entities=st.integers(min_value=1, max_value=5),
    n_queries=st.integers(min_value=0, max_value=15),
    seed=st.integers(min_value=0, max_value=10_000),
)
def test_p4_row_count_matches_query(
    tmp_path_factory: pytest.TempPathFactory,
    field_list: list[FeatureField],
    n_rows: int,
    n_entities: int,
    n_queries: int,
    seed: int,
) -> None:
    """len(result) == len(query_table) always — even with empty datasets / queries."""
    base = tmp_path_factory.mktemp("step12_p4")
    if n_rows == 0:
        # Skip writing; the dataset directory simply doesn't exist.
        clear_arrow_cache()
        schema = build_dynamic_schema(field_list)
    else:
        schema, _ = _populate_dataset(base, field_list, n_rows, n_entities, seed)
    rng = np.random.default_rng(seed=seed)
    if n_queries == 0:
        query = pa.table(
            {
                "entity_id": pa.array([], type=pa.string()),
                "as_of_time": pa.array([], type=pa.int64()),
            }
        )
    else:
        # Use seeded random entity_ids; they may or may not exist.
        eids = [f"e{rng.integers(0, max(n_entities, 1)):03d}" for _ in range(n_queries)]
        aots = [
            int(1_700_000_000_000_000_000 + int(rng.integers(0, _DAY_NS * 10)))
            for _ in range(n_queries)
        ]
        query = pa.table(
            {
                "entity_id": pa.array(eids, type=pa.string()),
                "as_of_time": pa.array(aots, type=pa.int64()),
            }
        )

    store = ParquetDatasetStore(base)
    result = store.read_point_in_time(schema, query, lookback_days=30)
    assert len(result) == len(query), f"expected {len(query)} result rows, got {len(result)}"
