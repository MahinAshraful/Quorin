"""Quickstart: define a schema, create a segment, write one row, read it back.

Run as a script:

    python examples/quickstart.py

Run via pytest (uses the requires_redis marker):

    pytest examples/quickstart.py -v

Prerequisites: a Redis instance reachable at the URL in
``QUORIN_REDIS_URL`` (default ``redis://127.0.0.1:6379/0``). For the
project's dev setup, ``./dev-up.sh`` starts one via docker-compose.
"""

from __future__ import annotations

import os

import pytest
import redis

import quorin

# The explicit ``import quorin.assembly`` below is required so attribute
# access like ``quorin.assembly.assemble(...)`` works — the lazy
# ``__getattr__`` in ``quorin/__init__.py`` only resolves evolution
# names, not submodules.
import quorin.assembly
from quorin.schema import FeatureField, FeatureSchema, dtype
from quorin.shm import SegmentRegistry

pytestmark = pytest.mark.requires_redis


class UserFeatures(FeatureSchema):
    """A small example schema. Real production schemas are larger
    (50-200 fields); the API surface is identical regardless of size.
    """

    version = 1
    fields = [
        FeatureField("age_normalized", dtype.float32),
        FeatureField("session_count_7d", dtype.int32),
        FeatureField("ltv_score", dtype.float32),
        FeatureField("behavior_embedding", dtype.float32, shape=(8,)),
    ]


def main() -> None:
    """End-to-end: insert a row directly via layout.insert, then read it
    back via the production-default Numba assembly path.
    """
    redis_url = os.environ.get("QUORIN_REDIS_URL", "redis://127.0.0.1:6379/0")
    # CR.E.6: socket_timeout is required for production. The library
    # emits a UserWarning if absent.
    redis_client = redis.Redis.from_url(redis_url, socket_timeout=5.0)

    registry = SegmentRegistry(redis_client)
    seg = registry.create(UserFeatures, capacity=1024)
    try:
        # Encode a single row using the public pack_row helper. Field
        # values are passed as kwargs and validated against the schema.
        from quorin.layout import insert, pack_row

        row_bytes = pack_row(
            UserFeatures,
            age_normalized=0.42,
            session_count_7d=7,
            ltv_score=12.5,
            behavior_embedding=[0.1] * 8,
        )
        insert(seg, "user-12345", row_bytes)

        # Read it back. quorin.assembly.assemble is the production
        # default (Numba-compiled); quorin.serving.assemble is the
        # pure-Python oracle. Both produce byte-identical output.
        out = quorin.assembly.assemble(seg, "user-12345")
        print(f"assembled {len(out)} float32 values")
        print(f"first 4: {out[:4]}")

        # Field-by-field if you need named access (assemble returns a
        # raw float32 ndarray in declaration order). ``lookup`` returns
        # the row index (an int) or None — it's a low-level probe, not
        # a byte-getter. Use ``assemble`` for the typed output.
        from quorin.layout import lookup

        row_index = lookup(seg, "user-12345")
        if row_index is not None:
            print(f"row index: {row_index}")
    finally:
        registry.close(seg)


def test_quickstart() -> None:
    """CI gate against future signature drift."""
    main()


if __name__ == "__main__":
    main()
