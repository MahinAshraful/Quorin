"""Benchmarks for quorin.wal_consumer.WALConsumer + quorin._internal.row_pack.

Decomposed so a regression in any sub-component shows up in isolation:

- ``bench_row_pack_*`` — pure row-pack cost (no Redis, no shm).
- ``bench_consumer_apply_per_msg`` — one message through the consumer's
  full apply path (msgpack-decode + row_pack + layout.insert + offline
  no-op + side-table SET pipeline build).

The row_pack benches do not need Redis. consumer_apply_per_msg requires
Redis and skips cleanly otherwise.

Active gates (ADR-009 §13 + benchmarks/regression/thresholds.yml):
  - ``row_pack_200_field`` ≤ 30 µs p99 (consolidated-Struct path).
  - ``consumer_apply_per_msg_200_field`` ≤ 100 µs p99.

CR.D.14 (v0.1.1): the original ADR-009 §13 also listed
``consumer_throughput_50_field`` ≥ 10 k msgs/sec and
``consumer_lag_e2e_p99`` < 100 ms gates, but the corresponding
benchmarks were never implemented. v0.1.1 retires those gate claims
(the same theme as CR.A.9/A.10's dead-metric removal: don't ship
gates we don't enforce). Implementing real end-to-end throughput +
lag benches is on the v0.2.0 / v0.3.0 perf roadmap.
"""

from __future__ import annotations

import sys
from typing import Any

import pytest

pytestmark = pytest.mark.skipif(
    sys.platform == "win32",
    reason="WAL consumer benchmarks rely on POSIX shared memory paths",
)

from quorin._internal.row_pack import pack_row_from_list  # noqa: E402
from quorin.schema import (  # noqa: E402
    FeatureField,
    FeatureSchema,
    _hash_name,
    compute_row_offset_table,
    dtype,
    row_size,
)

# ---------------------------------------------------------------------------
# Schemas — match the wal-producer benchmark's 50/200 field shapes so apples-
# to-apples comparisons across producer + consumer are easy.
# ---------------------------------------------------------------------------


def _make_50_field_schema() -> type[FeatureSchema]:
    fs = (
        [FeatureField(f"f{i:02d}", dtype.float32) for i in range(40)]
        + [FeatureField(f"i{i:02d}", dtype.int64) for i in range(8)]
        + [FeatureField("emb", dtype.float32, shape=(16,))]
        + [FeatureField("flags", dtype.uint8, shape=(4,))]
    )
    return type("_RPBench50", (FeatureSchema,), {"version": 1, "fields": fs})


def _make_200_field_schema() -> type[FeatureSchema]:
    fs = (
        [FeatureField(f"f{i:03d}", dtype.float32) for i in range(160)]
        + [FeatureField(f"i{i:03d}", dtype.int64) for i in range(30)]
        + [FeatureField(f"u{i:02d}", dtype.uint8) for i in range(9)]
        + [FeatureField("emb", dtype.float32, shape=(128,))]
    )
    return type("_RPBench200", (FeatureSchema,), {"version": 1, "fields": fs})


def _make_200_scalar_only_schema() -> type[FeatureSchema]:
    """Pure scalars, no shaped — exercises the consolidated-Struct path
    in isolation. ADR-009 §7's 10-12 µs estimate is based on this shape."""
    fs = [FeatureField(f"f{i:03d}", dtype.float32) for i in range(200)]
    return type("_RPBench200Scalar", (FeatureSchema,), {"version": 1, "fields": fs})


def _values_in_hash_order(schema: type[FeatureSchema]) -> list[Any]:
    """Build a representative values list in name_hash order."""
    table = compute_row_offset_table(schema)
    name_to_value: dict[str, Any] = {}
    for f in schema.fields:
        if f.shape == ():
            name_to_value[f.name] = (
                1.5 if f.dtype is dtype.float32 or f.dtype is dtype.float64 else 1
            )
        else:
            n = f.element_count
            name_to_value[f.name] = (
                [1.5] * n if f.dtype is dtype.float32 or f.dtype is dtype.float64 else [1] * n
            )
    hash_to_name = {_hash_name(f.name): f.name for f in schema.fields}
    return [name_to_value[hash_to_name[int(row["name_hash"])]] for row in table]


# ---------------------------------------------------------------------------
# Row-pack benches — no Redis, no shm. Pure CPU.
# ---------------------------------------------------------------------------


def test_row_pack_50_field(benchmark: Any) -> None:
    schema = _make_50_field_schema()
    values = _values_in_hash_order(schema)
    out = bytearray(row_size(schema))
    # Warm the plan cache so we measure the steady-state cost, not first-call.
    pack_row_from_list(schema, values, out)
    benchmark(pack_row_from_list, schema, values, out)


def test_row_pack_200_field(benchmark: Any) -> None:
    schema = _make_200_field_schema()
    values = _values_in_hash_order(schema)
    out = bytearray(row_size(schema))
    pack_row_from_list(schema, values, out)
    benchmark(pack_row_from_list, schema, values, out)


def test_row_pack_200_scalar_only(benchmark: Any) -> None:
    """Stress the consolidated-Struct path: 200 scalars, zero shaped fields.
    This is where the 10-12 us ADR-009 section 7 estimate lives."""
    schema = _make_200_scalar_only_schema()
    values = _values_in_hash_order(schema)
    out = bytearray(row_size(schema))
    pack_row_from_list(schema, values, out)
    benchmark(pack_row_from_list, schema, values, out)


# ---------------------------------------------------------------------------
# consumer_apply_per_msg — pure CPU consumer apply path, no Redis I/O.
# Mirrors what _apply does: msgpack-decode + row_pack + layout.insert (NO
# offline.append — Step 11 will add that benchmark separately). Gate at
# 100 us per ADR-009 section 13.
# ---------------------------------------------------------------------------


import msgpack  # noqa: E402

from _helpers import make_segment, release_segment  # noqa: E402
from quorin._internal.pydantic_factory import field_order_for, pydantic_model_for  # noqa: E402
from quorin.layout import insert as layout_insert  # noqa: E402


def _consumer_apply_one(
    schema: type[FeatureSchema], blob: bytes, row_buf: bytearray, seg: Any, eid: str
) -> None:
    values_list = msgpack.unpackb(blob, use_list=True, raw=False)
    pack_row_from_list(schema, values_list, row_buf)
    layout_insert(seg, eid, memoryview(row_buf))


def _make_blob(schema: type[FeatureSchema]) -> bytes:
    by_name: dict[str, Any] = {}
    for f in schema.fields:
        if f.shape == ():
            by_name[f.name] = 1.5 if f.dtype is dtype.float32 or f.dtype is dtype.float64 else 1
        else:
            n = f.element_count
            by_name[f.name] = (
                [1.5] * n if f.dtype is dtype.float32 or f.dtype is dtype.float64 else [1] * n
            )
    model_cls = pydantic_model_for(schema)
    validated = model_cls.model_validate(by_name)
    order = field_order_for(schema)
    values_list = [getattr(validated, name) for name in order]
    return msgpack.packb(values_list, use_bin_type=True)


def test_consumer_apply_per_msg_50_field(benchmark: Any) -> None:
    schema = _make_50_field_schema()
    blob = _make_blob(schema)
    seg = make_segment(schema, capacity=4)
    try:
        row_buf = bytearray(row_size(schema))
        _consumer_apply_one(schema, blob, row_buf, seg, "e-warm")
        benchmark(_consumer_apply_one, schema, blob, row_buf, seg, "e-warm")
    finally:
        release_segment(seg)


def test_consumer_apply_per_msg_200_field(benchmark: Any) -> None:
    schema = _make_200_field_schema()
    blob = _make_blob(schema)
    seg = make_segment(schema, capacity=4)
    try:
        row_buf = bytearray(row_size(schema))
        _consumer_apply_one(schema, blob, row_buf, seg, "e-warm")
        benchmark(_consumer_apply_one, schema, blob, row_buf, seg, "e-warm")
    finally:
        release_segment(seg)
