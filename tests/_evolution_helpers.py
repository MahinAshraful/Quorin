"""Shared helpers for Step 15 schema-evolution tests.

Mirrors :mod:`tests._helpers` (reusable; no fixtures) and
:mod:`tests._watchdog_helpers` (operator-side simulation primitives).
Imported via ``from _evolution_helpers import ...`` thanks to
``pythonpath = ["tests"]`` in ``pyproject.toml``.
"""

from __future__ import annotations

import os
import struct
from typing import Any

import numpy as np

import _helpers as h
from pyforge import layout
from pyforge.schema import (
    DTYPE_TO_NUMPY,
    DType,
    FeatureField,
    FeatureSchema,
)
from pyforge.shm import (
    KEY_SEGMENT_TO_SCHEMA,
    Segment,
    _key_pid_segments,
    _key_refcount,
)


def make_segment_with_random_data(
    schema: type[FeatureSchema],
    n_rows: int,
    *,
    capacity: int | None = None,
    rng_seed: int = 0,
    name_prefix: str = "pyforge_test_evo",
) -> Segment:
    """Create a non-Redis-tracked segment and populate it with ``n_rows``
    randomly-generated entities. Used by unit tests that exercise
    :func:`pyforge.evolution._build_translation_table` directly.

    Returns the segment; caller releases via ``release_segment``.
    """
    cap = capacity if capacity is not None else max(n_rows + 1, 8)
    seg = h.make_segment(schema, capacity=cap, name_prefix=name_prefix)
    rng = np.random.default_rng(rng_seed)
    for i in range(n_rows):
        values: dict[str, np.ndarray[Any, np.dtype[Any]]] = {
            f.name: h.random_value_for(f, rng) for f in schema.fields
        }
        row_bytes = h.pack_row(schema, values)
        layout.insert(seg, f"ent-{i:06d}", row_bytes)
    return seg


def make_segment_with_specific_floats(
    schema: type[FeatureSchema],
    rows: list[tuple[str, dict[str, np.ndarray[Any, np.dtype[Any]]]]],
    *,
    capacity: int | None = None,
    name_prefix: str = "pyforge_test_evo",
) -> Segment:
    """Like :func:`make_segment_with_random_data` but with caller-supplied
    ``(entity_id, values)`` pairs. Used by the NaN bit-pattern test to
    write specific float32 patterns.
    """
    cap = capacity if capacity is not None else max(len(rows) + 1, 8)
    seg = h.make_segment(schema, capacity=cap, name_prefix=name_prefix)
    for entity_id, values in rows:
        row_bytes = h.pack_row(schema, values)
        layout.insert(seg, entity_id, row_bytes)
    return seg


def assert_segment_clean_in_redis(
    redis_client: Any,
    segment_name: str,
) -> None:
    """Assert the segment's Redis bookkeeping was fully cleaned up.

    Used by orphan-cleanup tests to verify ``_cleanup_orphan_new_segment``
    removed refcount + pid_segments + sidetable entries.
    """
    pid = os.getpid()
    assert redis_client.get(_key_refcount(segment_name)) is None, (
        f"refcount key for {segment_name!r} still present after cleanup"
    )
    assert not redis_client.sismember(_key_pid_segments(pid), segment_name), (
        f"pid_segments still references {segment_name!r}"
    )
    assert not redis_client.hexists(KEY_SEGMENT_TO_SCHEMA, segment_name), (
        f"sidetable still references {segment_name!r}"
    )


def schema_pair_for_widen_test(
    *,
    add_field: bool = False,
    widen_dtype: bool = False,
    version_bump: int = 1,
) -> tuple[type[FeatureSchema], type[FeatureSchema]]:
    """Return ``(old, new)`` schema pair for upgrade tests.

    Old has 2 fields; new has those (possibly widened) plus an optional
    new field. Used by both unit tests and the orchestrator happy-path
    test in test_evolution_e2e.py.
    """
    counter = h._SCHEMA_VERSION_COUNTER
    counter[0] += 1
    name = f"_EvoSchema_{counter[0]}"

    old_cls = type(
        name,
        (FeatureSchema,),
        {
            "version": 1,
            "fields": [
                FeatureField("a", DType.FLOAT32),
                FeatureField("b", DType.INT32),
            ],
        },
    )
    new_fields: list[FeatureField] = [
        FeatureField("a", DType.FLOAT64 if widen_dtype else DType.FLOAT32),
        FeatureField("b", DType.INT64 if widen_dtype else DType.INT32),
    ]
    if add_field:
        new_fields.append(FeatureField("c", DType.FLOAT32, shape=(4,)))
    new_cls = type(
        name,  # SAME name (upgrades preserve schema identity)
        (FeatureSchema,),
        {"version": 1 + version_bump, "fields": new_fields},
    )
    return old_cls, new_cls


# Re-export _helpers' release_segment so test files can do a single import.
release_segment = h.release_segment
unique_segment_name = h.unique_segment_name


def write_legacy_format_message_to_wal(
    redis_client: Any,
    schema_name: str,
    entity_id: str,
    legacy_values_count: int,
) -> bytes:
    """Inject a malformed WAL message simulating a stale producer that wrote
    an OLD-schema-shaped message during the upgrade window. Used by C7.

    The blob is a msgpack list of ``legacy_values_count`` zero-floats. When
    the consumer's NEW schema has more fields, ``pack_row_from_list`` raises
    ValueError on length mismatch (verified in
    ``pyforge/_internal/row_pack.py:172-173``).
    """
    import msgpack  # local import — msgpack is a test-time dependency

    blob = msgpack.packb([0.0] * legacy_values_count, use_bin_type=True)
    msg_id = redis_client.xadd(
        b"pyforge:wal",
        {
            b"schema": schema_name.encode("utf-8"),
            b"entity_id": entity_id.encode("utf-8"),
            b"event_time_ns": str(0).encode("ascii"),
            b"blob": blob,
        },
    )
    if isinstance(msg_id, bytes):
        return msg_id
    return msg_id.encode("ascii")  # type: ignore[no-any-return,unreachable]


def pack_test_struct_le(*values_and_codes: tuple[float, str]) -> bytes:
    """Helper: pack values with their struct codes in little-endian.

    Used by NaN bit-pattern tests to construct specific IEEE 754 bit
    patterns deterministically.
    """
    parts: list[bytes] = []
    for value, code in values_and_codes:
        parts.append(struct.pack(f"<{code}", value))
    return b"".join(parts)


__all__ = [
    "DTYPE_TO_NUMPY",
    "assert_segment_clean_in_redis",
    "make_segment_with_random_data",
    "make_segment_with_specific_floats",
    "pack_test_struct_le",
    "release_segment",
    "schema_pair_for_widen_test",
    "unique_segment_name",
    "write_legacy_format_message_to_wal",
]
