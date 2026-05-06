"""Benchmarks for quorin.wal.WALProducer.

Decomposed so a regression in any sub-component shows up in isolation:

- ``bench_pydantic_validate_*`` — pure pydantic validation cost.
- ``bench_msgpack_pack_*`` — pure msgpack encode cost.
- ``bench_xadd_only`` — pure Redis network cost.
- ``bench_write_*`` — full producer call.
- ``bench_write_sync_*`` — write + polling + consumer-stub round-trip.

If ``write`` p99 drifts and ``xadd_only`` is flat, the regression is in
our code, not the network. Records baseline numbers for ADR-008's
"hot-path budget" table.

These benchmarks are gated on a real Redis (``redis_client`` fixture)
and skip cleanly if it is unreachable. Spec/budget targets:

- ``write`` p99 < 2 ms (50-field and 200-field schemas).
- ``write_sync`` p99 < 100 ms under steady consumer simulation.
- ``pydantic_validate`` p99 < 500 µs at 200 fields (escape-hatch trigger).
- ``msgpack_pack`` p99 < 100 µs at 200 fields.
"""

from __future__ import annotations

import sys
import threading
import time

import msgpack
import pytest

pytestmark = pytest.mark.skipif(
    sys.platform != "linux",
    reason=(
        "WAL benchmarks require Linux: POSIX shm for the running_consumer_50_field "
        "fixture used by test_bench_write_sync_rtt_50_field. Step 16 tightened "
        "the platform bound from win32-skip to linux-only."
    ),
)

from quorin._internal.pydantic_factory import field_order_for, pydantic_model_for  # noqa: E402
from quorin.schema import FeatureField, FeatureSchema, dtype  # noqa: E402
from quorin.wal import PROCESSED_KEY_PREFIX, WALProducer  # noqa: E402

# ---------------------------------------------------------------------------
# Schemas — 50-field and 200-field-with-128-emb match the spec's headline
# scenarios. Naming kept close to test_assembly_benchmark.py.
# ---------------------------------------------------------------------------


def _make_50_field_schema() -> type[FeatureSchema]:
    fs = (
        [FeatureField(f"f{i:02d}", dtype.float32) for i in range(40)]
        + [FeatureField(f"i{i:02d}", dtype.int64) for i in range(8)]
        + [FeatureField("emb", dtype.float32, shape=(16,))]
        + [FeatureField("flags", dtype.uint8, shape=(4,))]
    )
    return type("_S50", (FeatureSchema,), {"version": 1, "fields": fs})


def _make_200_field_schema() -> type[FeatureSchema]:
    fs = (
        [FeatureField(f"f{i:03d}", dtype.float32) for i in range(160)]
        + [FeatureField(f"i{i:03d}", dtype.int64) for i in range(30)]
        + [FeatureField(f"u{i:02d}", dtype.uint8) for i in range(9)]
        + [FeatureField("emb", dtype.float32, shape=(128,))]
    )
    return type("_S200", (FeatureSchema,), {"version": 1, "fields": fs})


def _values_for(schema: type[FeatureSchema]) -> dict:
    out: dict = {}
    for f in schema.fields:
        if f.shape == ():
            out[f.name] = 1.5 if f.dtype is dtype.float32 or f.dtype is dtype.float64 else 1
        else:
            n = f.element_count
            out[f.name] = (
                [1.5] * n if f.dtype is dtype.float32 or f.dtype is dtype.float64 else [1] * n
            )
    return out


# ---------------------------------------------------------------------------
# Sub-component isolation benches (no Redis).
# ---------------------------------------------------------------------------


def test_bench_pydantic_validate_50_field(benchmark) -> None:
    schema = _make_50_field_schema()
    model_cls = pydantic_model_for(schema)
    values = _values_for(schema)
    benchmark.group = "wal_subcomponent"
    benchmark(model_cls.model_validate, values)


def test_bench_pydantic_validate_200_field(benchmark) -> None:
    schema = _make_200_field_schema()
    model_cls = pydantic_model_for(schema)
    values = _values_for(schema)
    benchmark.group = "wal_subcomponent"
    benchmark(model_cls.model_validate, values)


def test_bench_msgpack_pack_50_field(benchmark) -> None:
    schema = _make_50_field_schema()
    model_cls = pydantic_model_for(schema)
    order = field_order_for(schema)
    values = _values_for(schema)
    validated = model_cls.model_validate(values)
    payload = [getattr(validated, n) for n in order]
    packer = msgpack.Packer(use_bin_type=True)
    benchmark.group = "wal_subcomponent"
    benchmark(packer.pack, payload)


def test_bench_msgpack_pack_200_field(benchmark) -> None:
    schema = _make_200_field_schema()
    model_cls = pydantic_model_for(schema)
    order = field_order_for(schema)
    values = _values_for(schema)
    validated = model_cls.model_validate(values)
    payload = [getattr(validated, n) for n in order]
    packer = msgpack.Packer(use_bin_type=True)
    benchmark.group = "wal_subcomponent"
    benchmark(packer.pack, payload)


# ---------------------------------------------------------------------------
# Headline benches — full write() against real Redis.
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_bench_write_50_field(benchmark, redis_client) -> None:
    """WALProducer.write() RTT, 50-field schema. Pedantic mode for deterministic
    p99 — plain ``benchmark()`` auto-calibrates to anywhere from 42 to 373 rounds
    depending on per-call time + warmup state, making p99 essentially
    max(samples) with luck-of-the-draw tail behavior. Step 16a converted both
    write_50/200 + write_sync_rtt to pedantic for consistent methodology.
    Pre-Step-16 plain ``benchmark()`` shape stays in ``test_bench_xadd_only``
    + ``test_bench_write_sync_50_field`` (ADR-006 / ADR-008 archive comparators).
    """
    schema = _make_50_field_schema()
    p = WALProducer(redis_client, stream_key=b"quorin:wal:bench")
    values = _values_for(schema)
    benchmark.group = "wal_write"
    benchmark.pedantic(
        p.write,
        args=(schema, "bench-entity", values),
        rounds=100,
        iterations=1,
        warmup_rounds=5,
    )


@pytest.mark.integration
def test_bench_write_200_field(benchmark, redis_client) -> None:
    """WALProducer.write() RTT, 200-field-with-128-emb schema. See
    test_bench_write_50_field above for the pedantic-mode rationale.
    """
    schema = _make_200_field_schema()
    p = WALProducer(redis_client, stream_key=b"quorin:wal:bench")
    values = _values_for(schema)
    benchmark.group = "wal_write"
    benchmark.pedantic(
        p.write,
        args=(schema, "bench-entity", values),
        rounds=100,
        iterations=1,
        warmup_rounds=5,
    )


@pytest.mark.integration
def test_bench_xadd_only(benchmark, redis_client) -> None:
    """Pure XADD cost. No validation, no msgpack. Establishes the
    network/AOF floor under which our code adds CPU."""
    fields = {
        b"schema": b"_Bench",
        b"entity_id": b"bench-entity",
        b"event_time_ns": b"1700000000000000000",
        b"blob": b"x" * 256,  # ~typical 50-field msgpack blob
    }
    benchmark.group = "wal_write"
    benchmark(
        redis_client.xadd, b"quorin:wal:xadd_only", fields, maxlen=1_000_000, approximate=True
    )


# ---------------------------------------------------------------------------
# write_sync — measures consumer round-trip with a stub setter thread.
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_bench_write_sync_50_field(benchmark, redis_client) -> None:
    schema = _make_50_field_schema()
    bench_stream = b"quorin:wal:bench:write_sync"
    p = WALProducer(redis_client, stream_key=bench_stream)
    values = _values_for(schema)

    stop = threading.Event()

    def stub_consumer() -> None:
        last_id = None
        while not stop.is_set():
            entries = redis_client.xrevrange(bench_stream, count=1)
            if entries:
                msg_id, _ = entries[0]
                if msg_id != last_id:
                    redis_client.set(PROCESSED_KEY_PREFIX + msg_id, b"1", ex=60)
                    last_id = msg_id
            time.sleep(0.001)

    t = threading.Thread(target=stub_consumer, daemon=True)
    t.start()
    try:
        benchmark.group = "wal_write_sync"
        benchmark(p.write_sync, schema, "bench-entity", values, None, 500)
    finally:
        stop.set()
        t.join(timeout=2.0)


# ---------------------------------------------------------------------------
# Step 16 P1: write_sync end-to-end RTT through a REAL WALConsumer.
#
# Differs from test_bench_write_sync_50_field above (which uses a stub-setter
# thread polling XREVRANGE): this one wires the actual production code path
# via the running_consumer_50_field fixture in benchmarks/conftest.py. The
# consumer uses NoopOfflineWriter per ADR-009 §3 — write_sync unblocks at
# online-store durability, and we don't conflate offline-flush latency.
#
# Tier-1 gate is 75ms p99 (math: 1.5 * max_consumer_cycle(50ms) +
# backoff_cap(10ms) ~= 85ms, rounded down with consumer fast-path). Sequential
# single-producer measures one full consumer cycle per call. Concurrent-
# producer "queue-depth" variant is a Step 17 follow-up if real workloads
# need sub-50ms.
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_bench_write_sync_rtt_50_field(benchmark, running_consumer_50_field) -> None:
    """Producer.write_sync() -> real consumer -> processed-flag set; full RTT.

    Pedantic mode + rounds=100 (NOT plain ``benchmark()``). At ~9ms per call,
    plain ``benchmark()`` auto-calibrates to anywhere from 14 to 92 rounds
    depending on system state — same C1 statistical-uselessness shape that
    Step 7's lesson warned about. ``pedantic(rounds=100)`` gives deterministic
    100 samples per run; sample[98] is the meaningful p99 against the 75ms
    Tier-1 gate. ~5 warmup rounds + 100 timed rounds = ~1s total wall clock.
    """
    schema = running_consumer_50_field["schema"]
    redis_client = running_consumer_50_field["redis_client"]
    stream_key = running_consumer_50_field["stream_key"]
    p = WALProducer(redis_client, stream_key=stream_key)
    values = _values_for(schema)
    benchmark.group = "wal_write_sync_rtt"
    # Generous per-call timeout (500ms) so that occasional slow consumer
    # cycles don't WriteSyncTimeout the bench. The Tier-1 gate (75ms) is
    # the realistic ceiling.
    benchmark.pedantic(
        p.write_sync,
        args=(schema, "bench-entity", values, None, 500),
        rounds=100,
        iterations=1,
        warmup_rounds=5,
    )
