"""End-to-end integration tests for pyforge._internal.lookup_kernel.lookup_jit.

Goes through the real ``SegmentRegistry`` path (not the bare ``make_segment``
helper used in unit tests). Verifies that lookup_jit works against a
production-shaped Segment opened via ``registry.open_current``, not just
against ``make_segment``-style fixtures.

The unit + property tests cover correctness comprehensively; this file is
the smaller-grain "real lifecycle" smoke that catches integration issues
the bare-helper path would miss (e.g. layout-recompute drift between
create-time and open-time, mmap_view aliasing through registry, etc.).
"""

from __future__ import annotations

import sys

import pytest

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        sys.platform == "win32",
        reason="lookup_kernel requires POSIX (Linux/WSL2)",
    ),
]

from pyforge._internal.lookup_kernel import lookup_jit, prewarm  # noqa: E402
from pyforge.layout import insert, lookup  # noqa: E402
from pyforge.schema import FeatureField, FeatureSchema, dtype  # noqa: E402
from pyforge.shm import SegmentRegistry  # noqa: E402


@pytest.fixture(scope="module", autouse=True)
def _prewarm_lookup_kernel() -> None:
    prewarm()


class _UserFeatures(FeatureSchema):
    version = 1
    fields = [
        FeatureField("age", dtype.int32),
        FeatureField("score", dtype.float32),
        FeatureField("embedding", dtype.float32, shape=(16,)),
    ]


def test_full_registry_create_insert_lookup_jit(redis_client) -> None:
    """Real SegmentRegistry.create + insert + lookup_jit for 50 entities.

    Verifies the wrapper works end-to-end with the production segment
    lifecycle (Redis bookkeeping, header CRC, layout recompute on open,
    refcount management) — not just the bare ``make_segment`` test path.
    """
    registry = SegmentRegistry(redis_client)
    seg = registry.create(_UserFeatures, capacity=128)
    try:
        row = bytes(seg.layout.row_size)
        for i in range(50):
            insert(seg, f"u_{i:04d}", row)

        # lookup_jit results match Python lookup byte-for-byte.
        for i in range(50):
            eid = f"u_{i:04d}"
            py_off = lookup(seg, eid)
            jit_off = lookup_jit(seg, eid)
            assert py_off is not None
            assert jit_off == py_off, f"lookup_jit({eid!r}) drift: {jit_off} != {py_off}"

        # Misses also agree.
        assert lookup_jit(seg, "u_missing") is None
        assert lookup(seg, "u_missing") is None
    finally:
        registry.close(seg)
