"""End-to-end integration tests for pyforge.assembly.assemble_batch (Step 8).

Goes through the real ``SegmentRegistry`` path. Verifies the new batch
kernel works against segments created with the registry's full lifecycle,
not just the bare ``make_segment`` helper used in unit tests.
"""

from __future__ import annotations

import sys

import numpy as np
import pytest

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        sys.platform == "win32",
        reason="assembly requires POSIX (Linux/WSL2)",
    ),
]

from _helpers import pack_row  # noqa: E402
from pyforge.assembly import assemble, assemble_batch, prewarm  # noqa: E402
from pyforge.layout import insert  # noqa: E402
from pyforge.pool import BatchBufferPool  # noqa: E402
from pyforge.schema import FeatureField, FeatureSchema, dtype  # noqa: E402
from pyforge.shm import SegmentRegistry  # noqa: E402


@pytest.fixture(scope="module", autouse=True)
def _prewarm_numba() -> None:
    prewarm()


class _UserFeatures4(FeatureSchema):
    version = 1
    fields = [
        FeatureField("age", dtype.int32),
        FeatureField("clicks", dtype.int64),
        FeatureField("ltv", dtype.float64),
        FeatureField("score", dtype.float32),
    ]


class _UserFeaturesEmbedding(FeatureSchema):
    version = 1
    fields = [
        FeatureField("age", dtype.int32),
        FeatureField("score", dtype.float32),
        FeatureField("embedding", dtype.float32, shape=(128,)),
        FeatureField("flag", dtype.uint8),
    ]


def test_full_create_insert_batch_4_field_n100(redis_client) -> None:
    reg = SegmentRegistry(redis_client)
    seg = reg.create(_UserFeatures4, capacity=128)
    try:
        for i in range(100):
            values = {
                "age": np.array([i], dtype=np.int32),
                "clicks": np.array([i * 1000], dtype=np.int64),
                "ltv": np.array([float(i) * 1.5], dtype=np.float64),
                "score": np.array([float(i) / 100.0], dtype=np.float32),
            }
            insert(seg, f"user_{i:03d}", pack_row(_UserFeatures4, values))

        ids = [f"user_{i:03d}" for i in range(100)]
        out, mask = assemble_batch(seg, ids)

        assert out.shape == (100, 4)
        assert mask.all(), "all 100 inserted entities should be found"

        for i in range(100):
            # Declaration order: age, clicks, ltv, score.
            assert out[i, 0] == np.float32(i)
            assert out[i, 1] == np.float32(i * 1000)
            assert out[i, 2] == np.float32(float(i) * 1.5)
            assert out[i, 3] == np.float32(float(i) / 100.0)
    finally:
        reg.close(seg)


def test_full_create_insert_batch_200_field_n1000_with_pool(redis_client) -> None:
    """The 5x-gate-shape test: 200-field-with-128-emb schema, batch 1000.

    Exercises the BatchBufferPool integration end-to-end against a real
    registry-managed segment. Doesn't assert performance — that's in
    benchmarks/test_batch_benchmark.py — just correctness at scale.
    """
    reg = SegmentRegistry(redis_client)
    seg = reg.create(_UserFeaturesEmbedding, capacity=1024)
    try:
        rng = np.random.default_rng(42)
        # Insert 1000 entities with deterministic-but-varied data.
        for i in range(1000):
            emb = rng.standard_normal(128).astype(np.float32)
            values = {
                "age": np.array([i % 100], dtype=np.int32),
                "score": np.array([float(i) / 1000.0], dtype=np.float32),
                "embedding": emb,
                "flag": np.array([i & 0xFF], dtype=np.uint8),
            }
            insert(seg, f"u_{i:04d}", pack_row(_UserFeaturesEmbedding, values))

        ids = [f"u_{i:04d}" for i in range(1000)]
        pool = BatchBufferPool(_UserFeaturesEmbedding, batch_size=1000, max_size=2)
        with pool.checkout() as buf:
            out, mask = assemble_batch(seg, ids, out=buf)
            assert out is buf
            assert out.shape == (1000, 1 + 1 + 128 + 1)  # = 131
            assert mask.all()

            # Spot-check correctness against single-entity assemble for a
            # few rows scattered across the batch.
            for i in (0, 1, 100, 500, 999):
                expected = assemble(seg, f"u_{i:04d}")
                np.testing.assert_array_equal(out[i], expected)
    finally:
        reg.close(seg)


def test_mixed_hit_miss_batch(redis_client) -> None:
    """Insert N, query 2N (N known + N synthesized misses). Verify mask."""
    reg = SegmentRegistry(redis_client)
    seg = reg.create(_UserFeatures4, capacity=64)
    try:
        for i in range(50):
            values = {
                "age": np.array([i], dtype=np.int32),
                "clicks": np.array([i], dtype=np.int64),
                "ltv": np.array([float(i)], dtype=np.float64),
                "score": np.array([float(i)], dtype=np.float32),
            }
            insert(seg, f"present_{i}", pack_row(_UserFeatures4, values))

        present = [f"present_{i}" for i in range(50)]
        missing = [f"missing_{i}" for i in range(50)]
        ids = present + missing

        out, mask = assemble_batch(seg, ids)

        assert out.shape == (100, 4)
        np.testing.assert_array_equal(mask[:50], np.ones(50, dtype=np.bool_))
        np.testing.assert_array_equal(mask[50:], np.zeros(50, dtype=np.bool_))

        # Hit rows match single-entity assemble.
        for i in range(50):
            expected = assemble(seg, f"present_{i}")
            np.testing.assert_array_equal(out[i], expected)

        # Miss rows are zero.
        np.testing.assert_array_equal(out[50:], np.zeros((50, 4), dtype=np.float32))
    finally:
        reg.close(seg)
