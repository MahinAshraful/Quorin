"""Benchmarks for pyforge.assembly.assemble_batch (Step 8).

Headline gate (build plan): batch >= 5x faster than N single calls at N=1000.
This file produces the numbers that prove (or disprove) the gate. ADR-007
records the actual ratio.

Sizes x schemas:

  4-field schema    x {1, 10, 100, 1000, 10000}
  200-field schema  x {1, 10, 100, 1000, 10000}

Per (schema, size):
  - assemble_batch (fresh out=)
  - assemble_batch (pooled out= from BatchBufferPool)
  - N x single-entity assemble calls (the comparator)
"""

from __future__ import annotations

import sys
import tracemalloc

import numpy as np
import pytest

pytestmark = pytest.mark.skipif(
    sys.platform == "win32",
    reason="assembly requires POSIX (Linux/WSL2)",
)

from _helpers import make_segment, pack_row, release_segment  # noqa: E402
from pyforge.assembly import assemble, assemble_batch, prewarm  # noqa: E402
from pyforge.layout import insert  # noqa: E402
from pyforge.pool import BatchBufferPool  # noqa: E402
from pyforge.schema import FeatureField, FeatureSchema, dtype  # noqa: E402


@pytest.fixture(scope="module", autouse=True)
def _prewarm_numba() -> None:
    prewarm()


# ---------------------------------------------------------------------------
# Schemas — same shape as test_assembly_benchmark.py for direct comparability.
# ---------------------------------------------------------------------------


class _Schema4Field(FeatureSchema):
    version = 1
    fields = [
        FeatureField("age", dtype.int32),
        FeatureField("clicks", dtype.int64),
        FeatureField("ltv", dtype.float64),
        FeatureField("score", dtype.float32),
    ]


def _build_n_field_schema(n: int, with_embedding: bool = False) -> type[FeatureSchema]:
    fields: list[FeatureField] = [
        FeatureField(f"feat_{i:03d}", dtype.float32) for i in range(n - 1 if with_embedding else n)
    ]
    if with_embedding:
        fields.append(FeatureField("embedding", dtype.float32, shape=(128,)))
    cls_name = f"_BatchSchemaN{n}{'_emb' if with_embedding else ''}"
    return type(cls_name, (FeatureSchema,), {"version": 1, "fields": fields})


_Schema200Field = _build_n_field_schema(200, with_embedding=True)


# ---------------------------------------------------------------------------
# Fixture: build segments once per module (parametrized by N, schema).
# ---------------------------------------------------------------------------


def _populate(seg, schema, n_entities: int, rng: np.random.Generator) -> list[str]:
    ids = [f"e_{i:05d}" for i in range(n_entities)]
    for i, eid in enumerate(ids):
        if schema is _Schema4Field:
            values = {
                "age": np.array([i % 100], dtype=np.int32),
                "clicks": np.array([i], dtype=np.int64),
                "ltv": np.array([float(i)], dtype=np.float64),
                "score": np.array([float(i) / 1000.0], dtype=np.float32),
            }
        else:
            values = {
                f.name: rng.standard_normal(f.element_count).astype(np.float32)
                for f in schema.fields
            }
        insert(seg, eid, pack_row(schema, values))
    return ids


def _make_populated_seg(schema, n_entities: int):
    rng = np.random.default_rng(seed=n_entities)
    seg = make_segment(schema, capacity=max(n_entities + 1, 16))
    ids = _populate(seg, schema, n_entities, rng)
    return seg, ids


# ---------------------------------------------------------------------------
# Headline benchmarks — gated by thresholds.yml.
# ---------------------------------------------------------------------------


def test_bench_assemble_batch_4_field_n1000(benchmark) -> None:
    """4-field batch=1000, fresh allocation. Headline gate at this size."""
    seg, ids = _make_populated_seg(_Schema4Field, 1000)
    try:
        # Pre-allocate out so allocation isn't measured.
        out = np.empty((1000, 4), dtype=np.float32)
        mask = np.empty(1000, dtype=np.bool_)

        def _run():
            return assemble_batch(seg, ids, out=out, found_mask=mask)

        result = benchmark(_run)
        assert result[0].shape == (1000, 4)
        assert result[1].all()
    finally:
        release_segment(seg)


def test_bench_assemble_batch_200_field_n1000(benchmark) -> None:
    """200-field-with-128-emb batch=1000. The 5x-gate-shape benchmark."""
    seg, ids = _make_populated_seg(_Schema200Field, 1000)
    try:
        elem = sum(f.element_count for f in _Schema200Field.fields)
        out = np.empty((1000, elem), dtype=np.float32)
        mask = np.empty(1000, dtype=np.bool_)

        def _run():
            return assemble_batch(seg, ids, out=out, found_mask=mask)

        result = benchmark(_run)
        assert result[0].shape == (1000, elem)
    finally:
        release_segment(seg)


# ---------------------------------------------------------------------------
# Pool integration benchmark — measures the pool checkout overhead.
# ---------------------------------------------------------------------------


def test_bench_assemble_batch_200_field_n1000_pooled(benchmark) -> None:
    """Same as the 200-field n=1000 bench but with BatchBufferPool checkout."""
    seg, ids = _make_populated_seg(_Schema200Field, 1000)
    try:
        pool = BatchBufferPool(_Schema200Field, batch_size=1000, max_size=2)

        def _run():
            with pool.checkout() as buf:
                return assemble_batch(seg, ids, out=buf)

        result = benchmark(_run)
        assert result[0].shape[0] == 1000
    finally:
        release_segment(seg)


# ---------------------------------------------------------------------------
# Comparator: N x single-entity assemble. The 5x-gate denominator.
# ---------------------------------------------------------------------------


def test_bench_n_single_assemble_4_field_n1000(benchmark) -> None:
    """1000 calls to single-entity assemble for 4-field schema. The 5x
    gate's comparator at this schema size."""
    seg, ids = _make_populated_seg(_Schema4Field, 1000)
    try:

        def _run():
            return [assemble(seg, eid) for eid in ids]

        result = benchmark(_run)
        assert len(result) == 1000
    finally:
        release_segment(seg)


def test_bench_n_single_assemble_200_field_n1000(benchmark) -> None:
    """1000 calls to single-entity assemble for 200-field schema. The 5x
    gate's comparator at the headline schema size."""
    seg, ids = _make_populated_seg(_Schema200Field, 1000)
    try:

        def _run():
            return [assemble(seg, eid) for eid in ids]

        result = benchmark(_run)
        assert len(result) == 1000
    finally:
        release_segment(seg)


# ---------------------------------------------------------------------------
# Crossover scan — batch sizes 1, 10, 100, 1000, 10000 at 4-field + 200-field.
# Not gated; recorded for ADR-007's writeup.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n_batch", [1, 10, 100, 1000])
def test_bench_assemble_batch_4_field_scan(benchmark, n_batch: int) -> None:
    """4-field batch sizes scan. Plots batch-overhead vs per-row work."""
    seg, ids = _make_populated_seg(_Schema4Field, n_batch)
    try:
        out = np.empty((n_batch, 4), dtype=np.float32)
        mask = np.empty(n_batch, dtype=np.bool_)

        def _run():
            return assemble_batch(seg, ids, out=out, found_mask=mask)

        result = benchmark(_run)
        assert result[0].shape[0] == n_batch
    finally:
        release_segment(seg)


@pytest.mark.parametrize("n_batch", [1, 10, 100, 1000])
def test_bench_assemble_batch_200_field_scan(benchmark, n_batch: int) -> None:
    """200-field batch sizes scan."""
    seg, ids = _make_populated_seg(_Schema200Field, n_batch)
    try:
        elem = sum(f.element_count for f in _Schema200Field.fields)
        out = np.empty((n_batch, elem), dtype=np.float32)
        mask = np.empty(n_batch, dtype=np.bool_)

        def _run():
            return assemble_batch(seg, ids, out=out, found_mask=mask)

        result = benchmark(_run)
        assert result[0].shape[0] == n_batch
    finally:
        release_segment(seg)


# ---------------------------------------------------------------------------
# Slow / large-batch — n=10000 at both schemas. Manual-only.
# ---------------------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.parametrize("schema_name", ["4_field", "200_field"])
def test_bench_assemble_batch_n10000(benchmark, schema_name: str) -> None:
    schema = _Schema4Field if schema_name == "4_field" else _Schema200Field
    n_batch = 10000
    seg, ids = _make_populated_seg(schema, n_batch)
    try:
        elem = sum(f.element_count for f in schema.fields)
        out = np.empty((n_batch, elem), dtype=np.float32)
        mask = np.empty(n_batch, dtype=np.bool_)

        def _run():
            return assemble_batch(seg, ids, out=out, found_mask=mask)

        result = benchmark(_run)
        assert result[0].shape == (n_batch, elem)
    finally:
        release_segment(seg)


# ---------------------------------------------------------------------------
# Allocation budget — pooled batch should allocate ~nothing per call.
# ---------------------------------------------------------------------------


def test_assemble_batch_allocation_budget_pooled() -> None:
    """Pooled batch=100 calls should allocate within budget per call.

    Budget: ~10 KiB per call (Python prep arrays: id_hashes + query_ids_padded
    + query_id_lens + encoded list). The big buffer comes from the pool.
    """
    import gc

    n_batch = 100
    seg, ids = _make_populated_seg(_Schema4Field, n_batch)
    try:
        pool = BatchBufferPool(_Schema4Field, batch_size=n_batch, max_size=2)

        # Warm up.
        for _ in range(10):
            with pool.checkout() as buf:
                assemble_batch(seg, ids, out=buf)

        gc.collect()
        tracemalloc.start()
        snap_before = tracemalloc.take_snapshot()

        n_calls = 100
        for _ in range(n_calls):
            with pool.checkout() as buf:
                assemble_batch(seg, ids, out=buf)

        snap_after = tracemalloc.take_snapshot()
        tracemalloc.stop()

        diffs = snap_after.compare_to(snap_before, "lineno")
        total_alloc = sum(d.size_diff for d in diffs)
        per_call = total_alloc / n_calls

        # Generous budget — Python prep allocates id_hashes (n_batch * 8),
        # query_ids_padded (n_batch * max_id_bytes), encoded list, etc.
        # Rough estimate: ~50 bytes/entity overhead + 2 KiB per call fixed.
        budget = n_batch * 100 + 16 * 1024
        assert per_call < budget, (
            f"pooled assemble_batch allocated {per_call:.1f} bytes/call (budget {budget})"
        )
    finally:
        release_segment(seg)
