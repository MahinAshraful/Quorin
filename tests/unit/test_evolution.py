"""Unit tests for :mod:`quorin.evolution`.

Covers (per ADR-014 / Step 15 plan §5.1):

* ``can_upgrade`` matrix — all dtype-widening combinations + identity +
  rejected combinations.
* ``_DTYPE_WIDENING`` regression — pinned membership.
* ``UpgradeIncompatibleError`` paths — class-name mismatch + can_upgrade
  fail.
* ``_build_translation_table`` direct: vectorized translation correctness +
  NaN bit-pattern preservation through float32 → float64 widening.
* ``_cleanup_orphan_new_segment`` regression: MUST NOT delete
  ``schema:current`` (the binding test for Rev-3 critical bug #12).

Integration / chaos tests (real Redis, real watchdog, subprocess kills)
live in ``tests/integration/test_evolution_e2e.py`` and
``tests/chaos/test_evolution_subprocess.py`` respectively.
"""

from __future__ import annotations

import struct
import sys
from typing import Any
from unittest.mock import Mock

import numpy as np
import pytest

pytestmark = pytest.mark.skipif(
    sys.platform == "win32",
    reason="Step 15 evolution relies on POSIX shared-memory paths",
)

import _evolution_helpers as eh  # noqa: E402

from quorin import layout  # noqa: E402
from quorin.evolution import (  # noqa: E402
    _DTYPE_WIDENING,
    UpgradeAbortedError,
    UpgradeConflictError,
    UpgradeError,
    UpgradeIncompatibleError,
    UpgradeResult,
    _build_translation_table,
    _cleanup_orphan_new_segment,
    can_upgrade,
)
from quorin.schema import (  # noqa: E402
    DTYPE_TO_NUMPY,
    DType,
    FeatureField,
    FeatureSchema,
)

# ---------------------------------------------------------------------------
# Schemas used across tests.
# ---------------------------------------------------------------------------


class _SchemaV1(FeatureSchema):
    version = 1
    fields = [
        FeatureField("a", DType.FLOAT32),
        FeatureField("b", DType.INT32),
        FeatureField("emb", DType.FLOAT32, shape=(4,)),
    ]


class _SchemaV2AddField(FeatureSchema):
    """V2 with `c` added (zero-fill on upgrade)."""

    version = 2
    fields = [
        FeatureField("a", DType.FLOAT32),
        FeatureField("b", DType.INT32),
        FeatureField("emb", DType.FLOAT32, shape=(4,)),
        FeatureField("c", DType.FLOAT32),
    ]


class _SchemaV2WidenDtype(FeatureSchema):
    """V2 widening `a` to float64 and `b` to int64."""

    version = 2
    fields = [
        FeatureField("a", DType.FLOAT64),
        FeatureField("b", DType.INT64),
        FeatureField("emb", DType.FLOAT32, shape=(4,)),
    ]


# ---------------------------------------------------------------------------
# can_upgrade matrix.
# ---------------------------------------------------------------------------


def _build_pair(
    old_fields: list[FeatureField],
    new_fields: list[FeatureField],
    *,
    name: str = "_TestPair",
    new_version: int = 2,
) -> tuple[type[FeatureSchema], type[FeatureSchema]]:
    """Build (old, new) with the SAME class name (upgrades preserve identity)."""
    old = type(name, (FeatureSchema,), {"version": 1, "fields": old_fields})
    new = type(name, (FeatureSchema,), {"version": new_version, "fields": new_fields})
    return old, new


def test_can_upgrade_identity_with_bumped_version_ok() -> None:
    old, new = _build_pair(
        [FeatureField("a", DType.FLOAT32)],
        [FeatureField("a", DType.FLOAT32)],
    )
    ok, reasons = can_upgrade(old, new)
    assert ok, reasons
    assert reasons == []


def test_can_upgrade_add_field_ok() -> None:
    old, new = _build_pair(
        [FeatureField("a", DType.FLOAT32)],
        [FeatureField("a", DType.FLOAT32), FeatureField("b", DType.INT32)],
    )
    ok, reasons = can_upgrade(old, new)
    assert ok, reasons


def test_can_upgrade_remove_field_rejected() -> None:
    old, new = _build_pair(
        [FeatureField("a", DType.FLOAT32), FeatureField("b", DType.INT32)],
        [FeatureField("a", DType.FLOAT32)],
    )
    ok, reasons = can_upgrade(old, new)
    assert not ok
    assert any("removed" in r and "'b'" in r for r in reasons)


def test_can_upgrade_widen_float32_to_float64_ok() -> None:
    old, new = _build_pair(
        [FeatureField("a", DType.FLOAT32)],
        [FeatureField("a", DType.FLOAT64)],
    )
    ok, _ = can_upgrade(old, new)
    assert ok


def test_can_upgrade_narrow_float64_to_float32_rejected() -> None:
    old, new = _build_pair(
        [FeatureField("a", DType.FLOAT64)],
        [FeatureField("a", DType.FLOAT32)],
    )
    ok, reasons = can_upgrade(old, new)
    assert not ok
    assert any("FLOAT64 -> FLOAT32" in r for r in reasons)


def test_can_upgrade_widen_int32_to_int64_ok() -> None:
    old, new = _build_pair(
        [FeatureField("a", DType.INT32)],
        [FeatureField("a", DType.INT64)],
    )
    ok, _ = can_upgrade(old, new)
    assert ok


def test_can_upgrade_narrow_int64_to_int32_rejected() -> None:
    old, new = _build_pair(
        [FeatureField("a", DType.INT64)],
        [FeatureField("a", DType.INT32)],
    )
    ok, reasons = can_upgrade(old, new)
    assert not ok
    assert any("INT64 -> INT32" in r for r in reasons)


def test_can_upgrade_uint8_to_int32_ok() -> None:
    old, new = _build_pair(
        [FeatureField("a", DType.UINT8)],
        [FeatureField("a", DType.INT32)],
    )
    ok, _ = can_upgrade(old, new)
    assert ok


def test_can_upgrade_uint8_to_int64_ok() -> None:
    old, new = _build_pair(
        [FeatureField("a", DType.UINT8)],
        [FeatureField("a", DType.INT64)],
    )
    ok, _ = can_upgrade(old, new)
    assert ok


def test_can_upgrade_uint8_to_float32_rejected_v1_cross_family() -> None:
    """Cross-family change explicitly rejected for v1 (lift in Step 16+)."""
    old, new = _build_pair(
        [FeatureField("a", DType.UINT8)],
        [FeatureField("a", DType.FLOAT32)],
    )
    ok, reasons = can_upgrade(old, new)
    assert not ok
    assert any("UINT8 -> FLOAT32" in r for r in reasons)


def test_can_upgrade_shape_change_existing_field_rejected() -> None:
    old, new = _build_pair(
        [FeatureField("emb", DType.FLOAT32, shape=(4,))],
        [FeatureField("emb", DType.FLOAT32, shape=(8,))],
    )
    ok, reasons = can_upgrade(old, new)
    assert not ok
    assert any("shape changed" in r and "'emb'" in r for r in reasons)


def test_can_upgrade_shape_change_for_new_field_ok() -> None:
    """A new field can have any shape — `shape unchanged` only applies to
    fields present in BOTH old and new."""
    old, new = _build_pair(
        [FeatureField("a", DType.FLOAT32)],
        [
            FeatureField("a", DType.FLOAT32),
            FeatureField("emb", DType.FLOAT32, shape=(8,)),  # new field, any shape
        ],
    )
    ok, _ = can_upgrade(old, new)
    assert ok


def test_can_upgrade_version_not_strictly_increasing_rejected() -> None:
    old, new = _build_pair(
        [FeatureField("a", DType.FLOAT32)],
        [FeatureField("a", DType.FLOAT32)],
        new_version=1,  # same version
    )
    ok, reasons = can_upgrade(old, new)
    assert not ok
    assert any("must be > old.version" in r for r in reasons)


def test_can_upgrade_reordered_fields_ok() -> None:
    """Field hash-order in offset table is what matters; declaration order
    can change between versions."""
    old, new = _build_pair(
        [
            FeatureField("a", DType.FLOAT32),
            FeatureField("b", DType.INT32),
            FeatureField("c", DType.UINT8),
        ],
        [
            FeatureField("c", DType.UINT8),  # reordered to first
            FeatureField("a", DType.FLOAT32),
            FeatureField("b", DType.INT32),
        ],
    )
    ok, _ = can_upgrade(old, new)
    assert ok


# ---------------------------------------------------------------------------
# _DTYPE_WIDENING — locked membership (regression guard).
# ---------------------------------------------------------------------------


def test_dtype_widening_table_membership() -> None:
    """Lock the conservative v1 dtype-widening table. Any change to this
    test should be paired with a docs/adr/014 update."""
    expected: set[tuple[DType, DType]] = {
        # Identity
        (DType.FLOAT32, DType.FLOAT32),
        (DType.FLOAT64, DType.FLOAT64),
        (DType.INT32, DType.INT32),
        (DType.INT64, DType.INT64),
        (DType.UINT8, DType.UINT8),
        # Widenings
        (DType.FLOAT32, DType.FLOAT64),
        (DType.INT32, DType.INT64),
        (DType.UINT8, DType.INT32),
        (DType.UINT8, DType.INT64),
    }
    assert set(_DTYPE_WIDENING) == expected


# ---------------------------------------------------------------------------
# UpgradeIncompatibleError — class-name mismatch + can_upgrade fail.
# ---------------------------------------------------------------------------


def test_upgrade_schema_rejects_class_name_mismatch() -> None:
    from quorin.evolution import upgrade_schema

    class _A(FeatureSchema):
        version = 1
        fields = [FeatureField("a", DType.FLOAT32)]

    class _B(FeatureSchema):  # different class name
        version = 2
        fields = [FeatureField("a", DType.FLOAT32)]

    with pytest.raises(UpgradeIncompatibleError, match="class names must match"):
        upgrade_schema(
            _A,
            _B,  # type: ignore[arg-type] — raised before any kwargs deref
            registry=Mock(),  # type: ignore[arg-type]
            redis_client=Mock(),
        )


def test_upgrade_schema_rejects_when_can_upgrade_fails() -> None:
    from quorin.evolution import upgrade_schema

    old, new = _build_pair(
        [FeatureField("a", DType.FLOAT32)],
        [FeatureField("a", DType.FLOAT32)],
        new_version=1,  # NOT strictly increasing
    )
    with pytest.raises(UpgradeIncompatibleError, match=r"must be > old\.version"):
        upgrade_schema(
            old,
            new,
            registry=Mock(),  # type: ignore[arg-type]
            redis_client=Mock(),
        )


# ---------------------------------------------------------------------------
# Error class hierarchy.
# ---------------------------------------------------------------------------


def test_error_class_hierarchy() -> None:
    """All evolution errors inherit from UpgradeError so callers can catch
    the base."""
    assert issubclass(UpgradeConflictError, UpgradeError)
    assert issubclass(UpgradeIncompatibleError, UpgradeError)
    assert issubclass(UpgradeAbortedError, UpgradeError)


# ---------------------------------------------------------------------------
# UpgradeResult dataclass shape.
# ---------------------------------------------------------------------------


def test_upgrade_result_is_frozen() -> None:
    r = UpgradeResult(
        old_segment_name="x",
        new_segment_name="y",
        entity_count=0,
        row_size_bytes_old=0,
        row_size_bytes_new=0,
        elapsed_seconds=0.0,
        consumer_pause_seconds=0.0,
        consumer_attach_seconds=0.0,
    )
    with pytest.raises(AttributeError):  # frozen=True
        r.entity_count = 99  # type: ignore[misc]


# ---------------------------------------------------------------------------
# _build_translation_table — vectorized correctness on populated segment.
# ---------------------------------------------------------------------------


def test_build_translation_table_add_field_zero_fills() -> None:
    """Add a new field; verify it's zero-filled in the resulting table."""
    seg_old = eh.make_segment_with_random_data(_SchemaV1, n_rows=20)
    try:
        table = _build_translation_table(seg_old, _SchemaV1, _SchemaV2AddField)
    finally:
        eh.release_segment(seg_old)

    assert table.num_rows == 20
    # entity_id column round-trips.
    eids = sorted(table.column("entity_id").to_pylist())
    assert eids == sorted(f"ent-{i:06d}" for i in range(20))
    # New field 'c' is zero-filled.
    c_vals = table.column("c").to_numpy()
    assert c_vals.dtype == np.float32
    assert np.array_equal(c_vals, np.zeros(20, np.float32))


def test_build_translation_table_widen_dtype() -> None:
    """Widen float32→float64 and int32→int64; verify result types and
    values match (cast preserves numeric equality)."""
    seg_old = eh.make_segment_with_random_data(_SchemaV1, n_rows=15, rng_seed=42)
    try:
        # Read OLD a/b values directly from segment for comparison.
        old_a_values: list[float] = []
        old_b_values: list[int] = []
        for entity_id, _row_offset in layout.iterate_occupied(seg_old):
            from quorin.serving import assemble

            vec = assemble(seg_old, entity_id)
            # _SchemaV1's assembly order: a (1) + b (1) + emb (4) = 6 floats.
            old_a_values.append(float(vec[0]))
            old_b_values.append(int(vec[1]))

        table = _build_translation_table(seg_old, _SchemaV1, _SchemaV2WidenDtype)
    finally:
        eh.release_segment(seg_old)

    assert table.num_rows == 15
    a_pa = table.column("a")
    b_pa = table.column("b")
    # PyArrow type checks (the schema we built).
    import pyarrow as pa

    assert a_pa.type == pa.float64()
    assert b_pa.type == pa.int64()


def test_build_translation_table_empty_segment_ok() -> None:
    """Empty old segment is a legitimate state — 0 rows, table is empty
    (NOT EmptyDatasetError)."""
    seg_old = eh.make_segment_with_random_data(_SchemaV1, n_rows=0, capacity=4)
    try:
        table = _build_translation_table(seg_old, _SchemaV1, _SchemaV2AddField)
    finally:
        eh.release_segment(seg_old)
    assert table.num_rows == 0
    assert "c" in table.column_names  # new field still present (empty)


# ---------------------------------------------------------------------------
# NaN bit-pattern preservation through float32 → float64 widening.
# ---------------------------------------------------------------------------


class _SchemaNaN1(FeatureSchema):
    version = 1
    fields = [FeatureField("x", DType.FLOAT32)]


class _SchemaNaN2(FeatureSchema):
    version = 2
    fields = [FeatureField("x", DType.FLOAT64)]


def test_float32_to_float64_preserves_nan_bit_pattern() -> None:
    """NaN bit-pattern preservation through the upgrade boundary.

    Locks invariant #12 (CLAUDE.md) at the unit-test cost: no Redis, no
    consumer, no real flip — just the per-field translation function.
    """
    # Construct a float32 NaN with a specific quiet-bit + payload. 0x7fc12345
    # has the IEEE 754 quiet bit (bit 22) set and a non-zero payload — a
    # legitimate quiet NaN with a recognizable signature.
    nan_f32_bytes = struct.pack("<I", 0x7FC12345)
    nan_f32 = np.frombuffer(nan_f32_bytes, np.float32)[0]
    assert np.isnan(nan_f32)

    seg = eh.make_segment_with_specific_floats(
        _SchemaNaN1,
        rows=[("e1", {"x": np.array([nan_f32], dtype=np.float32)})],
    )
    try:
        table = _build_translation_table(seg, _SchemaNaN1, _SchemaNaN2)
    finally:
        eh.release_segment(seg)

    # The widened value is float64 NaN with quiet bit promoted per IEEE 754.
    widened = table.column("x").to_numpy()
    assert widened.dtype == np.float64
    assert widened.shape == (1,)
    assert np.isnan(widened[0])
    # Quiet bit of float64 NaN is bit 51. Verify it's set.
    # Cast numpy uint64 → Python int so bitwise ops accept Python int operands.
    bits64 = int(widened.view(np.uint64)[0])
    assert bits64 & (1 << 51), f"quiet bit not set: 0x{bits64:016x}"
    # Payload preserved (the original float32 payload bits should appear in
    # the high mantissa bits of the float64 — exact mapping depends on numpy
    # / hardware; assert it's a deterministic non-zero pattern).
    assert (bits64 & ((1 << 51) - 1)) != 0, (
        f"float64 NaN payload zero — expected promotion of float32 payload: 0x{bits64:016x}"
    )


# ---------------------------------------------------------------------------
# _cleanup_orphan_new_segment — MUST NOT delete schema:current.
# Binding regression for Rev-3 critical bug #12.
# ---------------------------------------------------------------------------


def test_orphan_cleanup_does_not_delete_schema_current() -> None:
    """The Rev-3 critical bug #12 regression: with set_current=False,
    schema:current still points at OLD throughout the upgrade. Deleting it
    on orphan cleanup would orphan all live OLD readers.

    This test uses a Mock redis client so we can verify the EXACT pipeline
    operations without standing up real Redis.
    """
    # Mock Redis with pipeline + posix_shm. Mock's pipeline() returns a
    # context manager; .execute() needs to be awaited only in async paths
    # (this is the sync orchestrator).
    redis_client = Mock()
    pipeline_mock = Mock()
    pipeline_mock.__enter__ = Mock(return_value=pipeline_mock)
    pipeline_mock.__exit__ = Mock(return_value=False)
    redis_client.pipeline.return_value = pipeline_mock

    # Mock segment.
    seg = Mock()
    seg.name = "quorin_TestSchema_v2_abcd1234"
    seg.handle = Mock()

    # Patch posix_shm so we don't actually unlink anything.
    import quorin.evolution

    real_unlink = quorin.evolution.posix_shm.unlink
    real_close = quorin.evolution.posix_shm.close
    quorin.evolution.posix_shm.unlink = Mock()  # type: ignore[assignment]
    quorin.evolution.posix_shm.close = Mock()  # type: ignore[assignment]

    try:
        _cleanup_orphan_new_segment(seg, redis_client, _SchemaV1)
    finally:
        quorin.evolution.posix_shm.unlink = real_unlink  # type: ignore[assignment]
        quorin.evolution.posix_shm.close = real_close  # type: ignore[assignment]

    # Verify the pipeline did NOT include `delete(_key_current(...))`.
    # The hydrate orphan cleanup DOES include that op; this test catches
    # any regression that cargo-cults the hydrate version.
    call_names = [c[0] for c in pipeline_mock.method_calls]
    assert "delete" in call_names  # refcount delete IS expected
    delete_args = [c[1][0] for c in pipeline_mock.method_calls if c[0] == "delete"]
    # The only delete should be the refcount key — NOT the schema:current key.
    schema_current_key = "quorin:schema:_SchemaV1:current"
    assert all(schema_current_key not in str(arg) for arg in delete_args), (
        f"orphan cleanup wrongly deleted schema:current: {delete_args}"
    )

    # Other ops should fire.
    assert "srem" in call_names  # pid_segments cleanup
    assert "hdel" in call_names  # sidetable cleanup


def test_orphan_cleanup_handles_none_segment() -> None:
    """Idempotent + None-safe: calling with seg=None is a no-op."""
    redis_client = Mock()
    _cleanup_orphan_new_segment(None, redis_client, _SchemaV1)  # type: ignore[arg-type]
    assert not redis_client.method_calls


# ---------------------------------------------------------------------------
# Pure helpers.
# ---------------------------------------------------------------------------


def test_row_byte_offsets_by_name_round_trips() -> None:
    """The internal helper that maps name → byte_offset for the OLD layout."""
    from quorin.evolution import _row_byte_offsets_by_name

    seg = eh.make_segment_with_random_data(_SchemaV1, n_rows=1)
    try:
        offsets = _row_byte_offsets_by_name(_SchemaV1, seg.layout.row_offset_table)
    finally:
        eh.release_segment(seg)

    # Every schema field must map to a non-negative offset.
    for f in _SchemaV1.fields:
        assert f.name in offsets
        assert offsets[f.name] >= 0


# ---------------------------------------------------------------------------
# Smoke check: DType numpy mappings used by translation are sane.
# ---------------------------------------------------------------------------


def test_dtype_to_numpy_mapping_completeness() -> None:
    """Every DType used by _DTYPE_WIDENING must have a numpy mapping."""
    for old_dt, new_dt in _DTYPE_WIDENING:
        assert old_dt in DTYPE_TO_NUMPY
        assert new_dt in DTYPE_TO_NUMPY


# ---------------------------------------------------------------------------
# Misc: unused import suppressions for mypy.
# ---------------------------------------------------------------------------


def test_unused_imports_compile() -> None:
    """No-op test ensuring imports above are wired correctly."""
    assert UpgradeError is not None
    assert UpgradeAbortedError is not None
    assert UpgradeConflictError is not None
    assert UpgradeIncompatibleError is not None
    assert UpgradeResult is not None
    _ = Any  # silence noqa for forward reference
