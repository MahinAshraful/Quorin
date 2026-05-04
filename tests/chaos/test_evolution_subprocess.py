"""Chaos tests for Step 15 schema evolution.

Subprocess + SIGKILL pattern from Step 14 (``test_watchdog_subprocess.py``).
Marked ``@pytest.mark.chaos``.

Coverage (per plan §5.4 — focused subset for v1):

* C5 — live-reader thread during upgrade does NOT crash. Acceptance #1.
* Poison-pill: ``pack_row_from_list`` raises on length mismatch (in-process,
  no subprocess needed for this one).

C1/C2/C3/C4/C6 (subprocess SIGKILL variants) require careful watchdog +
heartbeat coordination across processes. Sketched but deferred to Step 16
follow-up if 1M-bench reveals additional failure modes.
"""

from __future__ import annotations

import sys
import threading
import time

import pytest

pytestmark = [
    pytest.mark.chaos,
    pytest.mark.skipif(sys.platform == "win32", reason="POSIX shm + Redis"),
]

import numpy as np  # noqa: E402
import redis  # noqa: E402

import _helpers as h  # noqa: E402
from pyforge import layout  # noqa: E402
from pyforge.evolution import upgrade_schema  # noqa: E402
from pyforge.schema import DType, FeatureField, FeatureSchema  # noqa: E402
from pyforge.serving import assemble  # noqa: E402
from pyforge.shm import SegmentRegistry, _key_current  # noqa: E402


class _C5Schema(FeatureSchema):
    version = 1
    fields = [FeatureField("a", DType.FLOAT32), FeatureField("b", DType.INT32)]


class _C5SchemaV2(FeatureSchema):
    version = 2
    fields = [
        FeatureField("a", DType.FLOAT32),
        FeatureField("b", DType.INT32),
        FeatureField("c", DType.FLOAT32),
    ]


_C5SchemaV2.__name__ = "_C5Schema"


def _populate(registry: SegmentRegistry, schema: type[FeatureSchema], n: int) -> None:
    seg = registry.create(schema, capacity=max(n + 1, 32))
    rng = np.random.default_rng(seed=11)
    for i in range(n):
        values = {
            "a": rng.standard_normal(1).astype(np.float32),
            "b": rng.integers(-100, 100, size=1, dtype=np.int32),
        }
        layout.insert(seg, f"ent-{i:04d}", h.pack_row(schema, values))
    # Leave alive at refcount=1 (ghost hold) for upgrade's open_current.


# ---------------------------------------------------------------------------
# C5 — live-reader thread does NOT crash through an upgrade.
# Acceptance criterion #1 from spec (live-reader test passes).
# ---------------------------------------------------------------------------


def test_c5_live_reader_threads_survive_upgrade(redis_client: redis.Redis) -> None:
    """Spawn N=4 reader threads each looping ``assemble`` on the OLD segment.
    Run upgrade in the middle. NO thread crashes; their `Segment` objects
    are per-process-immutable so the orchestrator's flip doesn't affect
    their reads (CLAUDE.md invariant: lookups are lock-free + handle-stable).
    """
    registry = SegmentRegistry(redis_client)
    _populate(registry, _C5Schema, n=20)

    try:
        # Each reader opens its own handle to the OLD segment (INCRs refcount
        # so close-Lua doesn't DEL schema:current when one closes during
        # the upgrade window).
        reader_segs = [registry.open_current(_C5Schema) for _ in range(4)]
        try:
            stop_event = threading.Event()
            errors: list[BaseException] = []

            def reader_loop(seg: object) -> None:
                try:
                    while not stop_event.is_set():
                        for i in range(20):
                            _ = assemble(seg, f"ent-{i:04d}")  # type: ignore[arg-type]
                except BaseException as e:
                    errors.append(e)

            threads = [
                threading.Thread(target=reader_loop, args=(s,), name=f"reader-{i}")
                for i, s in enumerate(reader_segs)
            ]
            for t in threads:
                t.start()

            # Let readers spin briefly so the upgrade lands mid-read.
            time.sleep(0.2)

            result = upgrade_schema(
                _C5Schema,
                _C5SchemaV2,
                registry,
                redis_client=redis_client,
                wait_for_consumer=False,
            )
            assert result.entity_count == 20

            # Let readers continue post-flip. Their old Segment object is
            # still mapped + readable; assemble keeps returning the OLD
            # field set.
            time.sleep(0.3)

            stop_event.set()
            for t in threads:
                t.join(timeout=5.0)
                assert not t.is_alive(), f"reader {t.name} stuck"

            assert errors == [], f"reader errors: {errors}"
        finally:
            import contextlib as _contextlib

            for s in reader_segs:
                with _contextlib.suppress(Exception):
                    registry.close(s)
    finally:
        # Cleanup schema:current pointer.
        redis_client.delete(_key_current(_C5Schema))
        # Drain any cleanup_queue entries left over.
        for k in redis_client.scan_iter(match="pyforge:upgrade:*", count=100):
            redis_client.delete(k)


# ---------------------------------------------------------------------------
# Poison-pill: a stale producer's WAL message format-mismatches against
# the new schema. The defense is in pyforge.wal_consumer._apply (catches
# ValueError from pack_row_from_list).
# ---------------------------------------------------------------------------


def test_poison_pill_pack_row_from_list_rejects_old_message_format() -> None:
    """In-process verification that the WAL consumer's poison-pill defense
    fires. The end-to-end form (real WAL stream + real consumer) is
    deferred; this locks the underlying invariant.
    """
    from pyforge._internal.row_pack import pack_row_from_list

    new_schema = _C5SchemaV2  # 3 fields
    out = bytearray(h.row_size(new_schema))

    # Stale producer wrote an OLD-shape message (2 values for 3-field schema).
    with pytest.raises(ValueError, match=r"expected 3 values, got 2"):
        pack_row_from_list(new_schema, [1.0, 2], out)
