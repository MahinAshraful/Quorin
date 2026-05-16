"""Schema evolution: upgrade a live segment to a new schema version.

Demonstrates the v1 → v2 upgrade flow. The new schema must have the
same class name (the version field is what changes), and dtype changes
must be in the permitted-widening table (e.g. float32→float64,
int32→int64). 2D-shape upgrades are NOT supported until v0.2.0
(CR.A.4 / ADR-018).

Run as a script:

    python examples/upgrade.py

Run via pytest:

    pytest examples/upgrade.py -v

Operator workflow (production):
1. Stop producers (no more XADDs to the WAL stream).
2. Wait for the consumer to drain (XPENDING == 0). v0.1.1 changed
   the precondition from XLEN to XPENDING — see CR.A.13 / ADR-018.
3. Stop the consumer (its liveness key TTL expires in ~30s).
4. Run upgrade_schema(...).
5. Restart consumers + producers with the new code.
"""

from __future__ import annotations

import os

import pytest
import redis

from quorin.evolution import upgrade_schema
from quorin.schema import FeatureField, FeatureSchema, dtype
from quorin.shm import SegmentRegistry

pytestmark = pytest.mark.requires_redis


# Both versions MUST share the same class __name__. The example uses
# the same Python class name on two distinct subclasses by aliasing
# in different modules in real code; here we use type() to construct
# them inline.
def _make_v1() -> type[FeatureSchema]:
    return type(
        "AccountFeatures",
        (FeatureSchema,),
        {
            "version": 1,
            "fields": [
                FeatureField("balance", dtype.float32),
                FeatureField("days_active", dtype.int32),
            ],
        },
    )


def _make_v2() -> type[FeatureSchema]:
    return type(
        "AccountFeatures",
        (FeatureSchema,),
        {
            "version": 2,
            "fields": [
                # balance: float32 → float64 (widening allowed)
                FeatureField("balance", dtype.float64),
                # days_active: int32 → int64 (widening allowed)
                FeatureField("days_active", dtype.int64),
                # NEW field: zero-filled during the copy.
                FeatureField("score", dtype.float32),
            ],
        },
    )


def main() -> None:
    redis_url = os.environ.get("QUORIN_REDIS_URL", "redis://127.0.0.1:6379/0")
    redis_client = redis.Redis.from_url(redis_url, socket_timeout=5.0)

    v1 = _make_v1()
    v2 = _make_v2()

    registry = SegmentRegistry(redis_client)
    seg = registry.create(v1, capacity=1024)
    try:
        # Insert a couple rows so the upgrade has something to translate.
        from quorin.layout import insert, pack_row

        for i in range(10):
            insert(
                seg,
                f"acc-{i}",
                pack_row(v1, balance=100.0 + i, days_active=30 + i),
            )

        # In a real deployment this is where you'd: stop producers,
        # wait for XPENDING==0, stop the consumer. For the example
        # they're not running.
        result = upgrade_schema(
            v1,
            v2,
            registry=registry,
            redis_client=redis_client,
            wait_for_consumer=False,  # no consumer in this example
        )
        print(f"upgraded {result.entity_count} entities")
        print(f"old segment: {result.old_segment_name}")
        print(f"new segment: {result.new_segment_name}")
        print(f"elapsed: {result.elapsed_seconds:.3f}s")
    finally:
        # Best-effort cleanup. In production the watchdog reaps
        # orphaned segments.
        from contextlib import suppress

        with suppress(Exception):
            registry.close(seg)


def test_upgrade() -> None:
    main()


if __name__ == "__main__":
    main()
