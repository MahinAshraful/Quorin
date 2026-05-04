"""Benchmarks for Step 15 schema evolution.

Per plan §5.5:

* ``bench_upgrade_10k_50_field`` — smoke (always-on); ~1s native, ~4s WSL2.
* ``bench_upgrade_100k_50_field`` (PYFORGE_RUN_LARGE_BENCH=1) — primary
  perf gate (~5s native).
* ``bench_consumer_pause_overhead`` — per-iter pause-check MGET cost.

Optional record bench (PYFORGE_RUN_RECORD_BENCH=1):

* ``bench_upgrade_1m_50_field`` — spec acceptance check; HARD GATE 10s
  native Linux. If RED, Step 16 Numba translation kernel is on the
  critical path before Step 15 is declared complete.

Native Linux is the source of truth for gate evaluation per Step 13's
methodology (CLAUDE.md §"Bench gates set at native-Linux targets").
"""

from __future__ import annotations

import asyncio
import os
import sys
from typing import Any

import pytest

pytestmark = pytest.mark.skipif(
    sys.platform == "win32",
    reason="Step 15 evolution requires POSIX shared memory",
)

import numpy as np  # noqa: E402

import _helpers as h  # noqa: E402
from pyforge import layout  # noqa: E402
from pyforge.evolution import upgrade_schema  # noqa: E402
from pyforge.schema import DType, FeatureField, FeatureSchema  # noqa: E402
from pyforge.shm import SegmentRegistry, _key_current  # noqa: E402

# ---------------------------------------------------------------------------
# Schemas — 50-field (smoke + primary gate). Realistic ML schema shape:
# 49 float32 scalars + 1 int32, expanded to float64 + int64 in v2.
# ---------------------------------------------------------------------------


def _make_50_field_schema(
    cls_name: str, version: int, *, widen: bool = False
) -> type[FeatureSchema]:
    """Build a 50-field FeatureSchema dynamically. ``widen=True`` widens all
    floats to float64 and ints to int64."""
    fields: list[FeatureField] = []
    for i in range(49):
        fields.append(
            FeatureField(
                f"f{i:02d}",
                DType.FLOAT64 if widen else DType.FLOAT32,
            )
        )
    fields.append(
        FeatureField(
            "label",
            DType.INT64 if widen else DType.INT32,
        )
    )
    return type(cls_name, (FeatureSchema,), {"version": version, "fields": fields})


_BenchOld = _make_50_field_schema("_BenchSchema", version=1, widen=False)
_BenchNew = _make_50_field_schema("_BenchSchema", version=2, widen=True)


# ---------------------------------------------------------------------------
# Helpers.
# ---------------------------------------------------------------------------


def _populate_old(registry: SegmentRegistry, n_rows: int) -> None:
    """Hydrate-style: create OLD segment, populate, leave at refcount=1."""
    seg = registry.create(_BenchOld, capacity=max(n_rows + 1, 32))
    rng = np.random.default_rng(seed=17)
    for i in range(n_rows):
        values: dict[str, np.ndarray[Any, np.dtype[Any]]] = {}
        for j in range(49):
            values[f"f{j:02d}"] = rng.standard_normal(1).astype(np.float32)
        values["label"] = np.array([i % 100], dtype=np.int32)
        layout.insert(seg, f"ent-{i:07d}", h.pack_row(_BenchOld, values))


def _drop(redis_client: Any) -> None:
    redis_client.delete(_key_current(_BenchOld))
    for k in redis_client.scan_iter(match="pyforge:upgrade:*", count=100):
        redis_client.delete(k)


# ---------------------------------------------------------------------------
# Smoke bench (always-on).
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_bench_upgrade_10k_50_field(redis_client: Any, benchmark: Any) -> None:
    """Smoke bench: 10k rows x 50 fields. Threshold ~1s native, ~4s WSL2."""

    def setup() -> tuple[Any, Any]:
        _drop(redis_client)
        registry = SegmentRegistry(redis_client)
        _populate_old(registry, n_rows=10_000)
        return (registry,), {}

    def run_upgrade(registry: SegmentRegistry) -> None:
        upgrade_schema(
            _BenchOld,
            _BenchNew,
            registry,
            redis_client=redis_client,
            wait_for_consumer=False,
        )
        _drop(redis_client)

    benchmark.pedantic(run_upgrade, setup=setup, rounds=3, iterations=1)


# ---------------------------------------------------------------------------
# Primary gate (100k rows x 50 fields), env-gated.
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.skipif(
    os.environ.get("PYFORGE_RUN_LARGE_BENCH") != "1",
    reason="PYFORGE_RUN_LARGE_BENCH=1 to run 100k bench",
)
def test_bench_upgrade_100k_50_field(redis_client: Any, benchmark: Any) -> None:
    """Primary perf gate: 100k rows x 50 fields. Threshold ~5s native (5x
    headroom over the 1M / 10s spec gate at 100k scale).
    """

    def setup() -> tuple[Any, Any]:
        _drop(redis_client)
        registry = SegmentRegistry(redis_client)
        _populate_old(registry, n_rows=100_000)
        return (registry,), {}

    def run_upgrade(registry: SegmentRegistry) -> None:
        upgrade_schema(
            _BenchOld,
            _BenchNew,
            registry,
            redis_client=redis_client,
            wait_for_consumer=False,
        )
        _drop(redis_client)

    benchmark.pedantic(run_upgrade, setup=setup, rounds=2, iterations=1)


# ---------------------------------------------------------------------------
# Optional 1M record bench — spec acceptance check, OPERATOR-VERIFIED.
#
# Step 16b methodology shift (per ADR-014 amendment + ADR-015 §7):
#   * GitHub Actions ``ubuntu-latest`` (~3.5 GB /dev/shm) cannot host this
#     bench (~3.2 GB segment → SIGBUS during populate). The Step 15 plan
#     §5.5 "if RED on native CI" trip-wire is structurally unreachable in
#     CI; the bench is operator-verified, NOT CI-verified.
#   * Measurement of record: WSL2 single-sample 9.91 s (90 ms margin under
#     the 10 s gate). Future operator runs on workstations with adequate
#     /dev/shm append to the record.
#   * Two skipifs gate it: PYFORGE_RUN_RECORD_BENCH (existing) +
#     PYFORGE_RUN_LARGE_SHM_BENCH (Step 16b). The workflow sets the
#     LARGE_SHM var ONLY on ``workflow_dispatch`` — ``schedule`` skips
#     cleanly so the weekly cron stays green.
#   * The 10 s commitment lives HERE (this docstring) + ADR-014, NOT in
#     ``benchmarks/regression/tier2.yml``: a framework gate is incompatible
#     with operator-only running because ``check.py --include-tier2 --strict``
#     would FAIL on schedule (gated bench absent from JSON = MISS = strict
#     FAIL).
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.skipif(
    os.environ.get("PYFORGE_RUN_RECORD_BENCH") != "1",
    reason="PYFORGE_RUN_RECORD_BENCH=1 to run 1M bench (spec acceptance check)",
)
@pytest.mark.skipif(
    os.environ.get("PYFORGE_RUN_LARGE_SHM_BENCH") != "1",
    reason=(
        "needs /dev/shm > 4 GB; ubuntu-latest cannot host on schedule "
        "(ADR-015 §7). Set PYFORGE_RUN_LARGE_SHM_BENCH=1 on operator hosts "
        "or workflow_dispatch."
    ),
)
def test_bench_upgrade_1m_50_field(redis_client: Any, benchmark: Any) -> None:
    """Spec acceptance: 1M rows x 50 fields in <10s. OPERATOR-VERIFIED contract.

    The 10 s gate is operator-evaluated against the bench's raw round timing,
    NOT framework-enforced via check.py. After running locally with
    PYFORGE_RUN_RECORD_BENCH=1 PYFORGE_RUN_LARGE_SHM_BENCH=1, read the
    autosaved JSON and verify ``np.percentile(stats.data, 99) <= 10.0``.

    Measurement of record: WSL2 9.91 s (Step 15 progress entry). Native Linux
    is materially faster on the bandwidth-bound /dev/shm cold-fault path;
    operator workstation runs typically land 1-3 s. If a measurement exceeds
    5 s, escalate to N=10 fresh subprocesses via benchmarks/runs/repeat.py
    and treat the p99 seriously — the parked Numba translation kernel
    (progress/progress.md parking-lot) becomes a candidate."""

    def setup() -> tuple[Any, Any]:
        _drop(redis_client)
        registry = SegmentRegistry(redis_client)
        _populate_old(registry, n_rows=1_000_000)
        return (registry,), {}

    def run_upgrade(registry: SegmentRegistry) -> None:
        upgrade_schema(
            _BenchOld,
            _BenchNew,
            registry,
            redis_client=redis_client,
            wait_for_consumer=False,
        )
        _drop(redis_client)

    benchmark.pedantic(run_upgrade, setup=setup, rounds=1, iterations=1)


# ---------------------------------------------------------------------------
# Consumer-pause-check overhead bench: per-iter MGET cost when no pause is
# set (the fast path through `_check_upgrade_pause_and_reopen`).
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_bench_consumer_pause_overhead(redis_client: Any, benchmark: Any) -> None:
    """Per-iter cost of the pause-check MGET in the no-pause fast path.

    The production consumer holds a persistent ``redis.asyncio.Redis`` client
    + a persistent event loop for its lifetime. The bench mirrors that:
    one event loop, one client, the timed call is just the MGET. Earlier
    revisions accidentally wrapped open + close in the timed path, which
    measured ~1.4 ms (network setup) instead of the ~50-300 µs MGET round
    trip operators actually pay.

    Threshold: 1 ms p99. WSL2 baseline ≈100-300 µs on warm pool; native
    Linux 4-8x faster. Validates "pause check doesn't tank consumer
    throughput when no pause is set."
    """
    del redis_client  # only requested to ensure dev-up.sh Redis is reachable
    import redis.asyncio

    url = os.environ.get("PYFORGE_REDIS_URL", "redis://127.0.0.1:6379/0")

    # Persistent event loop + client. Created ONCE before the timed
    # benchmark loop, closed AFTER. Matches what `WALConsumer` holds in
    # `self._redis` for its entire run.
    loop = asyncio.new_event_loop()
    async_client = redis.asyncio.Redis.from_url(url, decode_responses=False)
    # Force pool warm-up (PING) so the first timed iteration doesn't pay
    # connection-establishment cost.
    loop.run_until_complete(async_client.ping())

    ten_keys = [f"pyforge:upgrade:pause:_Sch{i}".encode("ascii") for i in range(10)]

    async def _do_mget() -> None:
        await async_client.mget(*ten_keys)

    def run() -> None:
        loop.run_until_complete(_do_mget())

    try:
        benchmark(run)
    finally:
        loop.run_until_complete(async_client.aclose())
        loop.close()
