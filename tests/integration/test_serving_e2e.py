"""End-to-end integration tests for quorin.serving.assemble.

Goes through the real ``SegmentRegistry`` path (not the bare ``make_segment``
helper used in unit tests). Verifies that the new ``SegmentLayout.assembly_table``
populated by ``compute_layout`` survives a registry create + open round trip.
"""

from __future__ import annotations

import sys

import numpy as np
import pytest

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        sys.platform == "win32",
        reason="serving requires POSIX (Linux/WSL2)",
    ),
]

from _helpers import pack_row  # noqa: E402
from quorin.layout import insert  # noqa: E402
from quorin.schema import FeatureField, FeatureSchema, dtype  # noqa: E402
from quorin.serving import assemble  # noqa: E402
from quorin.shm import SegmentRegistry  # noqa: E402


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


def test_full_create_insert_assemble_4_field(redis_client) -> None:
    reg = SegmentRegistry(redis_client)
    seg = reg.create(_UserFeatures4, capacity=16)
    try:
        values = {
            "age": np.array([42], dtype=np.int32),
            "clicks": np.array([1_234_567], dtype=np.int64),
            "ltv": np.array([987.65], dtype=np.float64),
            "score": np.array([0.5], dtype=np.float32),
        }
        insert(seg, "user_001", pack_row(_UserFeatures4, values))

        out = assemble(seg, "user_001")

        # Declaration order: age, clicks, ltv, score.
        assert out.shape == (4,)
        assert out[0] == np.float32(42)
        assert out[1] == np.float32(1_234_567)
        assert out[2] == np.float32(987.65)
        assert out[3] == np.float32(0.5)
        assert out.dtype == np.float32
        assert out.flags["C_CONTIGUOUS"]
    finally:
        reg.close(seg)


def test_full_create_insert_assemble_with_embedding(redis_client) -> None:
    reg = SegmentRegistry(redis_client)
    seg = reg.create(_UserFeaturesEmbedding, capacity=8)
    try:
        emb = np.linspace(-1.0, 1.0, num=128, dtype=np.float32)
        values = {
            "age": np.array([30], dtype=np.int32),
            "score": np.array([3.5], dtype=np.float32),
            "embedding": emb,
            "flag": np.array([1], dtype=np.uint8),
        }
        insert(seg, "user_999", pack_row(_UserFeaturesEmbedding, values))

        out = assemble(seg, "user_999")

        # Declaration order: age (1), score (1), embedding (128), flag (1) = 131.
        assert out.shape == (131,)
        assert out[0] == np.float32(30)
        assert out[1] == np.float32(3.5)
        np.testing.assert_array_equal(out[2:130], emb)
        assert out[130] == np.float32(1)
    finally:
        reg.close(seg)
