"""Unit tests for quorin.assembly.assemble_batch (Step 8).

Mirrors tests/unit/test_assembly.py for the single-entity Numba path. Step 8's
parity test (tests/property/test_batch_parity.py) handles property-based
comparison against the Python oracle (loop over serving.assemble); these unit
tests are smaller-grain coverage that's easier to debug when something specific
fails.
"""

from __future__ import annotations

import sys

import numpy as np
import pytest

pytestmark = pytest.mark.skipif(
    sys.platform == "win32",
    reason="assembly requires POSIX (Linux/WSL2)",
)

from _helpers import make_segment, pack_row, release_segment  # noqa: E402
from quorin.assembly import assemble, assemble_batch, prewarm  # noqa: E402
from quorin.layout import insert  # noqa: E402
from quorin.schema import (  # noqa: E402
    FeatureField,
    FeatureSchema,
    dtype,
    total_element_count,
)


@pytest.fixture(scope="module", autouse=True)
def _prewarm_numba() -> None:
    prewarm()


# ---------------------------------------------------------------------------
# Schemas — duplicated from test_assembly.py per the per-file convention.
# ---------------------------------------------------------------------------


class _OneScalarF32(FeatureSchema):
    version = 1
    fields = [FeatureField("a", dtype.float32)]


class _OneShapedF32(FeatureSchema):
    version = 1
    fields = [FeatureField("emb", dtype.float32, shape=(128,))]


class _Two2DShape(FeatureSchema):
    version = 1
    fields = [FeatureField("matrix", dtype.uint8, shape=(8, 16))]


class _AllDtypes(FeatureSchema):
    version = 1
    fields = [
        FeatureField("f32", dtype.float32),
        FeatureField("f64", dtype.float64),
        FeatureField("i32", dtype.int32),
        FeatureField("i64", dtype.int64),
        FeatureField("u8", dtype.uint8),
    ]


# ---------------------------------------------------------------------------
# Tests 1-7: per-DType round trip + shape contract.
# ---------------------------------------------------------------------------


def test_returns_correct_shape_dtype_contiguity() -> None:
    seg = make_segment(_AllDtypes, capacity=4)
    try:
        values = {
            "f32": np.array([1.5], dtype=np.float32),
            "f64": np.array([2.5], dtype=np.float64),
            "i32": np.array([-7], dtype=np.int32),
            "i64": np.array([99], dtype=np.int64),
            "u8": np.array([255], dtype=np.uint8),
        }
        insert(seg, "u", pack_row(_AllDtypes, values))

        out, mask = assemble_batch(seg, ["u"])

        assert out.dtype == np.float32
        assert out.shape == (1, total_element_count(_AllDtypes))
        assert out.flags["C_CONTIGUOUS"]
        assert out.flags["WRITEABLE"]
        assert mask.dtype == np.bool_
        assert mask.shape == (1,)
        assert mask[0]
    finally:
        release_segment(seg)


def test_all_dtypes_round_trip_into_float32() -> None:
    seg = make_segment(_AllDtypes, capacity=2)
    try:
        values = {
            "f32": np.array([1.5], dtype=np.float32),
            "f64": np.array([2.5], dtype=np.float64),
            "i32": np.array([-7], dtype=np.int32),
            "i64": np.array([99], dtype=np.int64),
            "u8": np.array([255], dtype=np.uint8),
        }
        insert(seg, "u", pack_row(_AllDtypes, values))

        out, mask = assemble_batch(seg, ["u"])

        # Declaration order: f32, f64, i32, i64, u8
        assert out[0, 0] == np.float32(1.5)
        assert out[0, 1] == np.float32(2.5)
        assert out[0, 2] == np.float32(-7)
        assert out[0, 3] == np.float32(99)
        assert out[0, 4] == np.float32(255)
        assert mask[0]
    finally:
        release_segment(seg)


def test_nan_and_inf_round_trip() -> None:
    """fastmath=False on the @njit decorator preserves NaN bit patterns."""
    seg = make_segment(_AllDtypes, capacity=2)
    try:
        values = {
            "f32": np.array([np.nan], dtype=np.float32),
            "f64": np.array([np.inf], dtype=np.float64),
            "i32": np.array([0], dtype=np.int32),
            "i64": np.array([0], dtype=np.int64),
            "u8": np.array([0], dtype=np.uint8),
        }
        insert(seg, "u", pack_row(_AllDtypes, values))

        out, _ = assemble_batch(seg, ["u"])
        assert np.isnan(out[0, 0])
        assert np.isposinf(out[0, 1])
    finally:
        release_segment(seg)


def test_shaped_field_flat_in_output_row() -> None:
    seg = make_segment(_OneShapedF32, capacity=2)
    try:
        emb = np.arange(128, dtype=np.float32) * 0.5
        insert(seg, "u", pack_row(_OneShapedF32, {"emb": emb}))

        out, mask = assemble_batch(seg, ["u"])

        assert out.shape == (1, 128)
        np.testing.assert_array_equal(out[0], emb)
        assert mask[0]
    finally:
        release_segment(seg)


def test_2d_shape_flattens_c_order() -> None:
    seg = make_segment(_Two2DShape, capacity=2)
    try:
        m = np.arange(128, dtype=np.uint8).reshape(8, 16)
        insert(seg, "u", pack_row(_Two2DShape, {"matrix": m}))

        out, _ = assemble_batch(seg, ["u"])

        assert out.shape == (1, 128)
        np.testing.assert_array_equal(out[0], m.ravel().astype(np.float32))
    finally:
        release_segment(seg)


def test_lossy_int64_above_2_pow_24() -> None:
    seg = make_segment(_AllDtypes, capacity=2)
    try:
        big = (1 << 30) + 7
        values = {
            "f32": np.array([0.0], dtype=np.float32),
            "f64": np.array([0.0], dtype=np.float64),
            "i32": np.array([0], dtype=np.int32),
            "i64": np.array([big], dtype=np.int64),
            "u8": np.array([0], dtype=np.uint8),
        }
        insert(seg, "u", pack_row(_AllDtypes, values))

        out, _ = assemble_batch(seg, ["u"])
        assert out[0, 3] == np.float32(big)
        assert int(out[0, 3]) != big
    finally:
        release_segment(seg)


# ---------------------------------------------------------------------------
# Tests 8-12: batch semantics — multi-row, miss handling, mask, duplicates.
# ---------------------------------------------------------------------------


def test_empty_batch_returns_empty_buffers() -> None:
    seg = make_segment(_OneScalarF32, capacity=2)
    try:
        out, mask = assemble_batch(seg, [])
        assert out.shape == (0, total_element_count(_OneScalarF32))
        assert out.dtype == np.float32
        assert mask.shape == (0,)
        assert mask.dtype == np.bool_
    finally:
        release_segment(seg)


def test_single_element_batch_matches_single_assemble() -> None:
    """assemble_batch([eid])[0] is byte-identical to assemble(eid)."""
    seg = make_segment(_AllDtypes, capacity=2)
    try:
        values = {
            "f32": np.array([1.5], dtype=np.float32),
            "f64": np.array([2.5], dtype=np.float64),
            "i32": np.array([-7], dtype=np.int32),
            "i64": np.array([99], dtype=np.int64),
            "u8": np.array([255], dtype=np.uint8),
        }
        insert(seg, "u", pack_row(_AllDtypes, values))

        single = assemble(seg, "u")
        batch_out, mask = assemble_batch(seg, ["u"])

        np.testing.assert_array_equal(batch_out[0], single)
        assert mask[0]
    finally:
        release_segment(seg)


def test_multiple_entities_each_row_correct() -> None:
    seg = make_segment(_OneScalarF32, capacity=8)
    try:
        for i in range(5):
            insert(
                seg,
                f"u{i}",
                pack_row(_OneScalarF32, {"a": np.array([float(i)], dtype=np.float32)}),
            )

        ids = [f"u{i}" for i in range(5)]
        out, mask = assemble_batch(seg, ids)

        for i in range(5):
            assert out[i, 0] == float(i), f"row {i} expected {i}, got {out[i, 0]}"
        assert mask.all()
    finally:
        release_segment(seg)


def test_all_misses_return_zero_rows_and_false_mask() -> None:
    seg = make_segment(_OneScalarF32, capacity=4)
    try:
        # No inserts. Every lookup misses.
        out, mask = assemble_batch(seg, ["a", "b", "c"])

        assert out.shape == (3, 1)
        np.testing.assert_array_equal(out, np.zeros((3, 1), dtype=np.float32))
        assert not mask.any()
    finally:
        release_segment(seg)


def test_mixed_hit_miss_per_row() -> None:
    seg = make_segment(_OneScalarF32, capacity=4)
    try:
        insert(seg, "a", pack_row(_OneScalarF32, {"a": np.array([1.0], dtype=np.float32)}))
        insert(seg, "c", pack_row(_OneScalarF32, {"a": np.array([3.0], dtype=np.float32)}))

        # Pre-fill out with sentinel garbage to prove miss rows get zeroed.
        out = np.full((4, 1), -999.0, dtype=np.float32)
        out_back, mask = assemble_batch(seg, ["a", "missing", "c", "ghost"], out=out)

        assert out_back is out  # buffer was filled in place
        assert out[0, 0] == 1.0
        assert out[1, 0] == 0.0  # miss zeroed
        assert out[2, 0] == 3.0
        assert out[3, 0] == 0.0  # miss zeroed
        np.testing.assert_array_equal(mask, np.array([True, False, True, False]))
    finally:
        release_segment(seg)


def test_duplicate_ids_return_matching_rows() -> None:
    seg = make_segment(_OneScalarF32, capacity=4)
    try:
        insert(seg, "x", pack_row(_OneScalarF32, {"a": np.array([42.0], dtype=np.float32)}))

        out, mask = assemble_batch(seg, ["x", "x", "x"])

        assert out[0, 0] == 42.0
        assert out[1, 0] == 42.0
        assert out[2, 0] == 42.0
        assert mask.all()
    finally:
        release_segment(seg)


# ---------------------------------------------------------------------------
# Test 13: hash-collision probing path (rare in random IDs, must be forced).
# ---------------------------------------------------------------------------


def test_hash_collision_probing_path(monkeypatch: pytest.MonkeyPatch) -> None:
    """Force entity-ID hash collisions and verify the kernel's probe + byte-
    compare path resolves the right rows.

    Hypothesis property tests with random IDs hit collision probability
    ~1/2^64 and would never naturally exercise this path. Mirrors the
    pattern in tests/unit/test_layout.py::test_hash_collision_path.
    """
    # Force every hash call to return the same value. The slot table will
    # then have multiple OCCUPIED slots whose hashes match every query; the
    # byte-compare in the kernel must resolve them.
    #
    # Three call sites need patching:
    #   - quorin._internal.hash_id.hash_entity_id — used by layout.insert
    #   - quorin.layout.hash_entity_id            — used by layout.lookup
    #   - quorin.assembly.blake2b_8                — used by assemble_batch
    #
    # Pre-v0.1.2, assemble_batch called hash_entity_id at the Python level.
    # v0.1.2 §2.1 swapped it for blake2b_8 (Numba kernel) called from a
    # Python loop. The kernel call inside _assemble_batch_core works off
    # the precomputed id_hashes uint64 array, so this Python-level patch
    # is sufficient — the kernel sees whatever hash we wrote into the array.
    import quorin._internal.hash_id as hash_id_mod
    import quorin.assembly as assembly_mod
    import quorin.layout as layout_mod

    fixed_hash = 0xDEADBEEF_CAFEBABE
    monkeypatch.setattr(hash_id_mod, "hash_entity_id", lambda _eid: fixed_hash)
    monkeypatch.setattr(layout_mod, "hash_entity_id", lambda _eid: fixed_hash)
    monkeypatch.setattr(assembly_mod, "blake2b_8", lambda _buf, _n: fixed_hash)

    seg = make_segment(_OneScalarF32, capacity=8)
    try:
        # Insert 4 entities with same forced hash. Slot table has 16 slots
        # (next_pow2(8 * 2)) so all 4 fit via linear probe.
        for i in range(4):
            insert(
                seg,
                f"col{i}",
                pack_row(_OneScalarF32, {"a": np.array([float(i)], dtype=np.float32)}),
            )

        # Query in mixed order. Each query must walk the probe chain and
        # byte-compare to find its row.
        ids = ["col2", "col0", "col3", "col1"]
        out, mask = assemble_batch(seg, ids)

        assert out[0, 0] == 2.0
        assert out[1, 0] == 0.0
        assert out[2, 0] == 3.0
        assert out[3, 0] == 1.0
        assert mask.all()
    finally:
        release_segment(seg)


# ---------------------------------------------------------------------------
# Tests 14-17: out= and found_mask= validation.
# ---------------------------------------------------------------------------


def test_out_wrong_shape_raises() -> None:
    seg = make_segment(_OneScalarF32, capacity=2)
    try:
        bad_out = np.empty((2, 1), dtype=np.float32)  # batch=2 but ids=1
        with pytest.raises(ValueError, match="out must be"):
            assemble_batch(seg, ["a"], out=bad_out)
    finally:
        release_segment(seg)


def test_out_wrong_dtype_raises() -> None:
    seg = make_segment(_OneScalarF32, capacity=2)
    try:
        bad_out = np.empty((1, 1), dtype=np.float64)
        with pytest.raises(ValueError, match="out must be"):
            assemble_batch(seg, ["a"], out=bad_out)
    finally:
        release_segment(seg)


def test_out_non_contiguous_raises() -> None:
    """Non-C-contiguous out= must raise.

    Uses a multi-element schema (n=128) so the stride trick actually
    produces non-C-contiguous memory. A shape-``(1, 1)`` array is always
    C-contiguous regardless of stride because the length-1 inner axis
    doesn't constrain the stride; only schemas with element_count > 1
    can be tested this way.
    """
    seg = make_segment(_OneShapedF32, capacity=2)
    try:
        n = total_element_count(_OneShapedF32)  # 128
        slab = np.empty((1, 2 * n), dtype=np.float32)
        non_contig = slab[:, ::2]
        assert non_contig.shape == (1, n)
        assert not non_contig.flags["C_CONTIGUOUS"]
        with pytest.raises(ValueError, match="out must be"):
            assemble_batch(seg, ["a"], out=non_contig)
    finally:
        release_segment(seg)


def test_out_read_only_raises() -> None:
    seg = make_segment(_OneScalarF32, capacity=2)
    try:
        n = total_element_count(_OneScalarF32)
        ro = np.empty((1, n), dtype=np.float32)
        ro.setflags(write=False)
        with pytest.raises(ValueError, match="out must be"):
            assemble_batch(seg, ["a"], out=ro)
    finally:
        release_segment(seg)


def test_found_mask_wrong_shape_raises() -> None:
    seg = make_segment(_OneScalarF32, capacity=2)
    try:
        bad_mask = np.empty(2, dtype=np.bool_)
        with pytest.raises(ValueError, match="found_mask must be"):
            assemble_batch(seg, ["a"], found_mask=bad_mask)
    finally:
        release_segment(seg)


def test_found_mask_wrong_dtype_raises() -> None:
    seg = make_segment(_OneScalarF32, capacity=2)
    try:
        bad_mask = np.empty(1, dtype=np.uint8)
        with pytest.raises(ValueError, match="found_mask must be"):
            assemble_batch(seg, ["a"], found_mask=bad_mask)
    finally:
        release_segment(seg)


def test_empty_entity_id_raises() -> None:
    seg = make_segment(_OneScalarF32, capacity=2)
    try:
        with pytest.raises(ValueError, match="empty"):
            assemble_batch(seg, ["valid", ""])
    finally:
        release_segment(seg)


# ---------------------------------------------------------------------------
# Tests 18-19: prewarm + slot-byte-layout constants pinning.
# ---------------------------------------------------------------------------


def test_prewarm_idempotent() -> None:
    """prewarm is safe to call multiple times. The autouse module fixture
    already calls it; this calls it twice more."""
    prewarm()
    prewarm()


# ---------------------------------------------------------------------------
# Tests 20-21: parallel kernel parity (forced via PARALLEL_THRESHOLD monkey-patch).
# ---------------------------------------------------------------------------


def test_parallel_kernel_parity_to_serial(monkeypatch: pytest.MonkeyPatch) -> None:
    """Forcing the parallel kernel produces byte-identical results to forcing
    the serial kernel for the same input.

    Both kernels MUST stay in sync (per `_assemble_batch_core_parallel`'s
    docstring). This test catches divergence.

    Uses a batch of 64 entities (default PARALLEL_THRESHOLD) with a
    multi-field schema so both probe + assembly paths are exercised.
    """
    seg = make_segment(_AllDtypes, capacity=128)
    try:
        rng = np.random.default_rng(seed=42)
        n_batch = 64
        ids = []
        for i in range(n_batch):
            values = {
                "f32": rng.standard_normal(1).astype(np.float32),
                "f64": rng.standard_normal(1).astype(np.float64),
                "i32": rng.integers(-1000, 1000, 1, dtype=np.int32),
                "i64": rng.integers(-(1 << 30), 1 << 30, 1, dtype=np.int64),
                "u8": rng.integers(0, 256, 1, dtype=np.uint8),
            }
            eid = f"u_{i:03d}"
            ids.append(eid)
            insert(seg, eid, pack_row(_AllDtypes, values))

        # Force serial.
        monkeypatch.setattr("quorin.assembly.PARALLEL_THRESHOLD", 1_000_000)
        out_serial, mask_serial = assemble_batch(seg, ids)

        # Force parallel.
        monkeypatch.setattr("quorin.assembly.PARALLEL_THRESHOLD", 0)
        out_parallel, mask_parallel = assemble_batch(seg, ids)

        np.testing.assert_array_equal(out_serial, out_parallel)
        np.testing.assert_array_equal(mask_serial, mask_parallel)
        assert mask_serial.all()
    finally:
        release_segment(seg)


def test_parallel_kernel_handles_misses(monkeypatch: pytest.MonkeyPatch) -> None:
    """Mixed hit/miss batch on the parallel path must zero-fill miss rows
    correctly. Independent verification that the parallel kernel doesn't
    skip the miss branch (a plausible bug under prange's scheduling)."""
    seg = make_segment(_OneScalarF32, capacity=64)
    try:
        # Insert 32 entities, query 64 (32 hits + 32 misses).
        for i in range(32):
            insert(
                seg,
                f"hit_{i}",
                pack_row(_OneScalarF32, {"a": np.array([float(i)], dtype=np.float32)}),
            )

        ids = [f"hit_{i}" for i in range(32)] + [f"miss_{i}" for i in range(32)]

        # Force parallel.
        monkeypatch.setattr("quorin.assembly.PARALLEL_THRESHOLD", 0)
        out, mask = assemble_batch(seg, ids)

        # Hits are correct.
        for i in range(32):
            assert out[i, 0] == float(i)
            assert mask[i]
        # Misses are zero.
        for i in range(32, 64):
            assert out[i, 0] == 0.0
            assert not mask[i]
    finally:
        release_segment(seg)


def test_slot_byte_layout_constants_pinned() -> None:
    """The kernel's reads at SLOT_*_OFFSET assume the dtype hasn't drifted.
    A reorder of SLOT_DTYPE fields would silently return wrong rows; this
    test fails loudly first.
    """
    from quorin.layout import (
        SLOT_BYTES,
        SLOT_DTYPE,
        SLOT_FEATURE_ROW_INDEX_OFFSET,
        SLOT_FLAGS_OFFSET,
        SLOT_ID_OFFSET_OFFSET,
        SLOT_NAME_HASH_OFFSET,
    )

    assert SLOT_BYTES == SLOT_DTYPE.itemsize == 24
    fields = SLOT_DTYPE.fields
    assert fields is not None
    assert SLOT_NAME_HASH_OFFSET == fields["name_hash"][1] == 0
    assert SLOT_ID_OFFSET_OFFSET == fields["id_offset"][1] == 8
    assert SLOT_FLAGS_OFFSET == fields["flags"][1] == 16
    assert SLOT_FEATURE_ROW_INDEX_OFFSET == fields["feature_row_index"][1] == 20
