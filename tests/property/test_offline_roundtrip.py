"""Hypothesis property tests for ParquetDatasetStore (Step 11).

Schema x row-count x random values -> ``append`` x N -> ``flush`` ->
``pq.read_table`` -> assert per-column equality with NaN-aware semantics.

The producer's wire format is name-hash-sorted lists; we mirror that
ordering when calling ``append`` so values land in the right columns.
"""

from __future__ import annotations

import asyncio
import math
import sys
from typing import Any

import pytest

pytestmark = [
    pytest.mark.skipif(
        sys.platform == "win32",
        reason="Step 11 fsync semantics are POSIX-only",
    ),
    pytest.mark.property,
]

import hypothesis.strategies as st  # noqa: E402
import numpy as np  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402
from hypothesis import HealthCheck, given, settings  # noqa: E402

from _helpers import build_dynamic_schema, field_list_strategy  # noqa: E402
from pyforge._internal.arrow_schema import clear_cache as clear_arrow_cache  # noqa: E402
from pyforge.offline import ParquetDatasetStore  # noqa: E402
from pyforge.schema import DType, FeatureField, _hash_name  # noqa: E402


def _python_value_for(field: FeatureField, rng: np.random.Generator) -> Any:
    """Generate a Python value matching field shape + dtype.

    Mirrors the post-msgpack-decode shape that the consumer hands to
    ``OfflineWriter.append``: scalar → number; ``shape=(N,)`` → list of
    N numbers; ``shape=(R, C)`` → list of R lists of C numbers.

    Floats may be NaN/+Inf/-Inf to exercise the round-trip claim made
    by invariant #12 (NaN bit-pattern preservation, ADR-008 §5).
    """
    n = field.element_count
    flat: list[Any]
    if field.dtype is DType.FLOAT32:
        # Mix in NaN/Inf at a low rate.
        roll = rng.random(n)
        nums = rng.standard_normal(n).astype(np.float32)
        nums = np.where(roll < 0.05, np.float32("nan"), nums)
        nums = np.where((roll >= 0.05) & (roll < 0.07), np.float32("inf"), nums)
        nums = np.where((roll >= 0.07) & (roll < 0.09), np.float32("-inf"), nums)
        flat = [float(x) for x in nums]
    elif field.dtype is DType.FLOAT64:
        nums64 = rng.standard_normal(n).astype(np.float64)
        flat = [float(x) for x in nums64]
    elif field.dtype is DType.INT32:
        ints32 = rng.integers(-(1 << 20), 1 << 20, size=n, dtype=np.int32)
        flat = [int(x) for x in ints32]
    elif field.dtype is DType.INT64:
        ints64 = rng.integers(-(1 << 40), 1 << 40, size=n, dtype=np.int64)
        flat = [int(x) for x in ints64]
    elif field.dtype is DType.UINT8:
        u8 = rng.integers(0, 256, size=n, dtype=np.uint8)
        flat = [int(x) for x in u8]
    else:
        raise AssertionError(f"unhandled dtype {field.dtype}")

    if len(field.shape) == 0:
        return flat[0]
    if len(field.shape) == 1:
        return flat
    if len(field.shape) == 2:
        rows, cols = field.shape
        return [flat[i * cols : (i + 1) * cols] for i in range(rows)]
    raise AssertionError(f"unsupported shape {field.shape!r}")


def _values_equal(actual: Any, expected: Any) -> bool:
    """Per-column NaN-aware equality. Recursively walks list-of-list."""
    if isinstance(expected, list):
        if not isinstance(actual, list) or len(actual) != len(expected):
            return False
        return all(_values_equal(a, e) for a, e in zip(actual, expected, strict=True))
    if isinstance(expected, float):
        if math.isnan(expected):
            return isinstance(actual, float) and math.isnan(actual)
        return actual == expected
    return actual == expected


_RESERVED_NAMES = frozenset({"entity_id", "event_time_ns", "msg_id_ms", "msg_id_seq", "event_date"})


@settings(
    max_examples=50,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture],
)
@given(
    field_list=field_list_strategy().filter(
        lambda fs: not any(f.name in _RESERVED_NAMES for f in fs)
    ),
    n_rows=st.integers(min_value=1, max_value=20),
)
def test_random_schema_round_trips_via_parquet(
    tmp_path_factory: pytest.TempPathFactory,
    field_list: list[FeatureField],
    n_rows: int,
) -> None:
    clear_arrow_cache()
    base = tmp_path_factory.mktemp("offline_property")
    store = ParquetDatasetStore(base)

    schema = build_dynamic_schema(field_list)
    # Producer wire order is name-hash-sorted; mirror it for the values list
    # passed to append().
    wire_order = sorted(schema.fields, key=lambda f: _hash_name(f.name))
    rng = np.random.default_rng(seed=12345)

    expected_rows: list[dict[str, Any]] = []
    for i in range(n_rows):
        per_field = {f.name: _python_value_for(f, rng) for f in schema.fields}
        values_list = [per_field[f.name] for f in wire_order]
        msg_id = f"{1700_000_000_000 + i}-0".encode()
        asyncio.run(store.append(schema, f"ent-{i}", 0, values_list, msg_id))
        expected_rows.append({"entity_id": f"ent-{i}", **per_field})

    asyncio.run(store.flush())

    files = sorted(base.rglob("*.parquet"))
    assert len(files) == 1
    table = pq.read_table(files[0])
    assert table.num_rows == n_rows

    # Compare per-column. Order in the file is declaration order, so we
    # walk schema.fields directly.
    for f in schema.fields:
        actual = table.column(f.name).to_pylist()
        expected = [row[f.name] for row in expected_rows]
        for i, (a, e) in enumerate(zip(actual, expected, strict=True)):
            assert _values_equal(a, e), (
                f"row {i} field {f.name!r} mismatch: actual={a!r} expected={e!r}"
            )

    # entity_id round-trips verbatim.
    actual_ids = table.column("entity_id").to_pylist()
    assert actual_ids == [row["entity_id"] for row in expected_rows]
