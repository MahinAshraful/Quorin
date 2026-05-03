"""Unit tests for pyforge.hydration.

19 tests with mocked Redis + mocked ParquetDatasetStore. Real
``SegmentRegistry`` is NOT used at unit layer; ``registry.create`` is
mocked to return a segment built via ``tests/_helpers.py::make_segment``.
This bypasses Redis bookkeeping but exercises the orchestrator's
end-to-end interaction with a real ``/dev/shm`` segment + the real
``insert_many`` kernel. Integration tests (Commit B) will exercise the
real registry.

Patch targets are LOCKED at ``pyforge.hydration.{prewarm,insert_many}`` —
the orchestrator imports those names via ``from pyforge._internal.insert_kernel
import insert_many, prewarm`` at module load, so patches at the source
module would not take effect.
"""

from __future__ import annotations

import asyncio
import contextlib
import math
import os
import sys
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pyarrow as pa
import pytest

pytestmark = pytest.mark.skipif(
    sys.platform == "win32",
    reason="hydration requires POSIX (Linux/WSL2)",
)

from _helpers import make_segment, release_segment  # noqa: E402
from pyforge._internal import posix_shm  # noqa: E402
from pyforge.hydration import (  # noqa: E402
    EmptyDatasetError,
    HydrationConflictError,
    HydrationResult,
    _force_drop_orphan,
    hydrate,
)
from pyforge.schema import DType, FeatureField, FeatureSchema  # noqa: E402
from pyforge.shm import _key_current, _key_pid_segments, _key_refcount  # noqa: E402
from pyforge.wal_consumer import KEY_WAL_CONSUMER_LIVENESS  # noqa: E402

# ---------------------------------------------------------------------------
# Test schema + fixtures.
# ---------------------------------------------------------------------------


class _S(FeatureSchema):
    version = 1
    fields = [
        FeatureField("price", DType.FLOAT32, ()),
        FeatureField("quantity", DType.INT32, ()),
    ]


def _make_table(entity_ids: list[str]) -> pa.Table:
    """Build a PyArrow table conforming to ``_S`` for ``entity_ids``."""
    n = len(entity_ids)
    return pa.table(
        {
            "entity_id": pa.array(entity_ids, type=pa.string()),
            "event_time_ns": pa.array([1_000_000 + i for i in range(n)], type=pa.int64()),
            "price": pa.array(np.arange(n, dtype=np.float32) + 1.0, type=pa.float32()),
            "quantity": pa.array(np.arange(n, dtype=np.int32) + 10, type=pa.int32()),
        }
    )


def _kbytes(key: Any) -> bytes:
    """Normalize a Redis key to bytes (matches redis-py's wire format).

    Hydration passes both str (from `_key_*` helpers in pyforge.shm)
    and bytes (from `KEY_WAL_CONSUMER_LIVENESS` in pyforge.wal_consumer);
    the fake unifies them by storing bytes-keyed.
    """
    if isinstance(key, bytes):
        return key
    if isinstance(key, str):
        return key.encode()
    return bytes(key)


class _FakeRedis:
    """Tiny sync Redis stub: only the methods hydrate + _force_drop_orphan
    actually call. Tracks every method call on ``method_calls`` so test
    #23c / #23e can assert the absence of MUTATING operations.

    Keys normalized to bytes internally so str-vs-bytes call-site
    differences (str from ``_key_current``, bytes from
    ``KEY_WAL_CONSUMER_LIVENESS``) don't silently bypass lookups.
    """

    def __init__(self) -> None:
        self._kv: dict[bytes, bytes] = {}
        self._sets: dict[bytes, set[bytes]] = {}
        # Track every (method_name, args, kwargs) tuple so tests can
        # assert which mutating calls happened (or didn't). Args are
        # recorded AS-PASSED (not normalized) so test expectations can
        # match the call site's type.
        self.method_calls: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = []

    def get(self, key: Any) -> bytes | None:
        self.method_calls.append(("get", (key,), {}))
        return self._kv.get(_kbytes(key))

    def set(self, key: Any, value: Any, ex: int | None = None) -> bool:
        self.method_calls.append(("set", (key, value), {"ex": ex}))
        self._kv[_kbytes(key)] = value if isinstance(value, bytes) else str(value).encode()
        return True

    def pipeline(self, transaction: bool = True) -> _FakePipeline:
        self.method_calls.append(("pipeline", (), {"transaction": transaction}))
        return _FakePipeline(self)


class _FakePipeline:
    def __init__(self, parent: _FakeRedis) -> None:
        self._parent = parent
        self._cmds: list[tuple[str, tuple[Any, ...]]] = []

    def __enter__(self) -> _FakePipeline:
        return self

    def __exit__(self, *exc: Any) -> None:
        return None

    def delete(self, key: Any) -> _FakePipeline:
        self._cmds.append(("delete", (key,)))
        return self

    def srem(self, key: Any, member: Any) -> _FakePipeline:
        self._cmds.append(("srem", (key, member)))
        return self

    def execute(self) -> list[Any]:
        results: list[Any] = []
        for cmd, args in self._cmds:
            self._parent.method_calls.append((f"pipeline.{cmd}", args, {}))
            if cmd == "delete":
                self._parent._kv.pop(_kbytes(args[0]), None)
                results.append(1)
            elif cmd == "srem":
                self._parent._sets.get(_kbytes(args[0]), set()).discard(_kbytes(args[1]))
                results.append(1)
        self._cmds.clear()
        return results


@pytest.fixture
def fake_redis() -> _FakeRedis:
    return _FakeRedis()


@pytest.fixture
def real_segment() -> Any:
    """A real /dev/shm segment built via make_segment, released after test.

    Capacity 256 covers all current unit tests with headroom.
    """
    seg = make_segment(_S, capacity=256)
    yield seg
    # Best-effort release; tests that exercise _force_drop_orphan may
    # have already unlinked.
    with contextlib.suppress(Exception):
        release_segment(seg)


@pytest.fixture
def mock_store() -> MagicMock:
    """A mocked ParquetDatasetStore. Tests stub `latest_features` per case."""
    store = MagicMock()
    return store


@pytest.fixture
def mock_registry(real_segment: Any) -> MagicMock:
    """A mocked SegmentRegistry whose `create` returns the real_segment fixture.

    Locks the unit-test pattern from plan §"sub-step 11" — bypasses real
    Redis bookkeeping but exercises the orchestrator's end-to-end
    interaction with a real /dev/shm segment + the real insert_many kernel.
    """
    registry = MagicMock()
    registry.create.return_value = real_segment
    return registry


# ---------------------------------------------------------------------------
# Tests #15-#28.
# ---------------------------------------------------------------------------


# --- #15 happy path -------------------------------------------------------


def test_hydrate_basic(
    fake_redis: _FakeRedis,
    mock_store: MagicMock,
    mock_registry: MagicMock,
    real_segment: Any,
) -> None:
    """100 mock entities → segment populated, HydrationResult correct."""
    eids = [f"entity_{i}" for i in range(100)]
    mock_store.latest_features.return_value = _make_table(eids)

    result = hydrate(_S, mock_store, mock_registry, redis_client=fake_redis)

    assert isinstance(result, HydrationResult)
    assert result.segment_name == real_segment.name
    assert result.entity_count == 100
    assert result.elapsed_seconds >= 0
    mock_registry.create.assert_called_once()


# --- #16 + #17 + #17b precondition refusals -------------------------------


def test_hydrate_refuses_when_current_segment_exists(
    fake_redis: _FakeRedis,
    mock_store: MagicMock,
    mock_registry: MagicMock,
) -> None:
    """Pre-set current key → HydrationConflictError; message mentions recovery."""
    fake_redis._kv[_kbytes(_key_current(_S))] = b"some_segment_name"

    with pytest.raises(HydrationConflictError) as excinfo:
        hydrate(_S, mock_store, mock_registry, redis_client=fake_redis)

    msg = str(excinfo.value)
    assert "current segment" in msg
    # Recovery command must be referenced so operators know what to run.
    assert "DEL" in msg
    assert "Step 15" in msg or "upgrade" in msg
    mock_registry.create.assert_not_called()


def test_hydrate_refuses_when_consumer_alive(
    fake_redis: _FakeRedis,
    mock_store: MagicMock,
    mock_registry: MagicMock,
) -> None:
    """Pre-set liveness key → HydrationConflictError; message mentions expiry."""
    fake_redis._kv[KEY_WAL_CONSUMER_LIVENESS] = b"12345"

    with pytest.raises(HydrationConflictError) as excinfo:
        hydrate(_S, mock_store, mock_registry, redis_client=fake_redis)

    msg = str(excinfo.value)
    assert "consumer" in msg.lower()
    # Operator UX: must reference SIGTERM and the expiry window.
    assert "SIGTERM" in msg
    assert "30s" in msg or "expire" in msg
    mock_registry.create.assert_not_called()


def test_hydrate_precondition_order_locks_current_first(
    fake_redis: _FakeRedis,
    mock_store: MagicMock,
    mock_registry: MagicMock,
) -> None:
    """Both preconditions violated → 'current segment exists' fires first.

    Locks runbook-relevant order. A future refactor that swaps the two
    GETs would silently change which error operators see first.
    """
    fake_redis._kv[_kbytes(_key_current(_S))] = b"some_segment_name"
    fake_redis._kv[KEY_WAL_CONSUMER_LIVENESS] = b"12345"

    with pytest.raises(HydrationConflictError) as excinfo:
        hydrate(_S, mock_store, mock_registry, redis_client=fake_redis)

    assert "current segment" in str(excinfo.value)
    assert "consumer" not in str(excinfo.value).lower() or "current" in str(excinfo.value).lower()


# --- #19 + #26 empty dataset ---------------------------------------------


def test_hydrate_empty_dataset_raises(
    fake_redis: _FakeRedis,
    mock_store: MagicMock,
    mock_registry: MagicMock,
) -> None:
    """Mock store returns 0-row table → EmptyDatasetError; message mentions lookback."""
    mock_store.latest_features.return_value = _make_table([])

    with pytest.raises(EmptyDatasetError) as excinfo:
        hydrate(_S, mock_store, mock_registry, redis_client=fake_redis)

    assert "lookback_days" in str(excinfo.value)
    mock_registry.create.assert_not_called()


def test_hydrate_raises_when_all_rows_outside_lookback(
    fake_redis: _FakeRedis,
    mock_store: MagicMock,
    mock_registry: MagicMock,
) -> None:
    """Mock store returns empty (simulating all rows filtered) → EmptyDatasetError."""
    mock_store.latest_features.return_value = _make_table([])

    with pytest.raises(EmptyDatasetError):
        hydrate(_S, mock_store, mock_registry, redis_client=fake_redis)

    mock_registry.create.assert_not_called()


# --- #20a/b/c/d capacity_factor -------------------------------------------


def test_hydrate_capacity_factor_validation_rejects_below_two(
    fake_redis: _FakeRedis,
    mock_store: MagicMock,
    mock_registry: MagicMock,
) -> None:
    with pytest.raises(ValueError, match="capacity_factor"):
        hydrate(
            _S,
            mock_store,
            mock_registry,
            redis_client=fake_redis,
            capacity_factor=1.5,
        )


def test_hydrate_capacity_factor_minimum_boundary(
    fake_redis: _FakeRedis,
    mock_store: MagicMock,
    mock_registry: MagicMock,
) -> None:
    """capacity_factor=2.0 succeeds; capacity passed to registry.create == 2 * len(latest)."""
    eids = [f"entity_{i}" for i in range(10)]
    mock_store.latest_features.return_value = _make_table(eids)

    hydrate(
        _S,
        mock_store,
        mock_registry,
        redis_client=fake_redis,
        capacity_factor=2.0,
    )

    # Capacity arg from registry.create call: max(int(10 * 2.0), 16) = 20.
    _, kwargs = mock_registry.create.call_args
    assert kwargs["capacity"] == 20


def test_hydrate_capacity_factor_nan_raises(
    fake_redis: _FakeRedis,
    mock_store: MagicMock,
    mock_registry: MagicMock,
) -> None:
    """capacity_factor=NaN → ValueError; message mentions 'finite'."""
    with pytest.raises(ValueError, match="finite"):
        hydrate(
            _S,
            mock_store,
            mock_registry,
            redis_client=fake_redis,
            capacity_factor=float("nan"),
        )


def test_hydrate_capacity_factor_inf_raises(
    fake_redis: _FakeRedis,
    mock_store: MagicMock,
    mock_registry: MagicMock,
) -> None:
    """capacity_factor=Inf → ValueError; message mentions 'finite'."""
    with pytest.raises(ValueError, match="finite"):
        hydrate(
            _S,
            mock_store,
            mock_registry,
            redis_client=fake_redis,
            capacity_factor=math.inf,
        )


# --- #22 lookback subset --------------------------------------------------


def test_hydrate_subset_of_entities_in_lookback(
    fake_redis: _FakeRedis,
    mock_store: MagicMock,
    mock_registry: MagicMock,
) -> None:
    """100 entities, store returns 95 (5 outside lookback) → entity_count==95."""
    eids = [f"entity_{i}" for i in range(95)]
    mock_store.latest_features.return_value = _make_table(eids)

    result = hydrate(_S, mock_store, mock_registry, redis_client=fake_redis)

    assert result.entity_count == 95


# --- #23 / #23b / #23c / #23d / #23e cleanup paths ------------------------


def test_hydrate_unwinds_on_insert_failure(
    fake_redis: _FakeRedis,
    mock_store: MagicMock,
    mock_registry: MagicMock,
    real_segment: Any,
) -> None:
    """Mock insert_many to raise RuntimeError → orphan cleanup fires.

    Simulates a kernel-side error mid-insert. The orchestrator must:
    - Run _force_drop_orphan (Redis pipeline DELs + SREM, posix_shm.unlink).
    - Re-raise the original RuntimeError.

    Note: we do NOT pre-populate `_key_current` because that would trip
    precondition #1 BEFORE the orchestrator ever reaches the timed
    section. The pipeline DEL fires unconditionally inside
    `_force_drop_orphan` (DEL on a missing key is a no-op); we verify
    the cleanup attempt by inspecting `method_calls`.
    """
    eids = [f"entity_{i}" for i in range(50)]
    mock_store.latest_features.return_value = _make_table(eids)

    with (
        patch("pyforge.hydration.insert_many", side_effect=RuntimeError("boom")),
        pytest.raises(RuntimeError, match="boom"),
    ):
        hydrate(_S, mock_store, mock_registry, redis_client=fake_redis)

    # Verify the orphan-cleanup pipeline issued the DEL + SREM commands.
    pipeline_cmds = [name for name, _, _ in fake_redis.method_calls if name.startswith("pipeline.")]
    assert "pipeline.delete" in pipeline_cmds, (
        f"_force_drop_orphan did not issue pipeline DELs; got {pipeline_cmds}"
    )
    assert "pipeline.srem" in pipeline_cmds, (
        f"_force_drop_orphan did not issue pipeline SREM; got {pipeline_cmds}"
    )
    # Segment unlinked from /dev/shm — opening it should fail.
    with pytest.raises(Exception):  # noqa: B017
        posix_shm.open_existing(real_segment.name)


def test_hydrate_force_drop_orphan_proceeds_when_redis_pipeline_fails(
    fake_redis: _FakeRedis,
    mock_store: MagicMock,
    mock_registry: MagicMock,
    real_segment: Any,
) -> None:
    """Redis pipeline raises during orphan cleanup → posix_shm.unlink still runs.

    Locks the contract documented in _force_drop_orphan's docstring:
    Redis-side state may remain stale; the disk side is cleaned anyway.
    """
    eids = [f"entity_{i}" for i in range(10)]
    mock_store.latest_features.return_value = _make_table(eids)

    # Stub pipeline().execute() to raise so the Redis block fails.
    def bad_pipeline(transaction: bool = True) -> Any:
        del transaction
        pipe = MagicMock()
        pipe.__enter__ = MagicMock(return_value=pipe)
        pipe.__exit__ = MagicMock(return_value=None)
        pipe.execute.side_effect = RuntimeError("redis down")
        return pipe

    with (
        patch.object(fake_redis, "pipeline", side_effect=bad_pipeline),
        patch("pyforge.hydration.insert_many", side_effect=RuntimeError("insert boom")),
        pytest.raises(RuntimeError, match="insert boom"),
    ):
        hydrate(_S, mock_store, mock_registry, redis_client=fake_redis)

    # Disk side: segment unlinked despite Redis failure.
    with pytest.raises(Exception):  # noqa: B017
        posix_shm.open_existing(real_segment.name)


def test_hydrate_handles_concurrent_create_failure(
    fake_redis: _FakeRedis,
    mock_store: MagicMock,
    mock_registry: MagicMock,
) -> None:
    """registry.create raises FileExistsError → no orphan cleanup, no MUTATING Redis calls.

    Locks plan CRITICAL #5: a concurrent hydrate that lost the
    registry.create race must NOT clean up — there's no segment we own
    and the winning hydrate's segment is mid-write.
    """
    eids = [f"entity_{i}" for i in range(10)]
    mock_store.latest_features.return_value = _make_table(eids)
    mock_registry.create.side_effect = FileExistsError("race")

    with pytest.raises(FileExistsError, match="race"):
        hydrate(_S, mock_store, mock_registry, redis_client=fake_redis)

    # No MUTATING Redis calls. preconditions' GETs are fine.
    mutating = {"set", "pipeline", "pipeline.delete", "pipeline.srem"}
    actual = {name for name, _, _ in fake_redis.method_calls}
    assert not (actual & mutating), (
        f"unexpected mutating Redis calls after concurrent-create failure: {actual & mutating}"
    )


def test_hydrate_force_drop_orphan_on_cancelled_during_insert(
    fake_redis: _FakeRedis,
    mock_store: MagicMock,
    mock_registry: MagicMock,
    real_segment: Any,
) -> None:
    """insert_many raises CancelledError → orphan cleanup runs, no logger.exception.

    Locks the differentiated BaseException catch contract — cancel is
    in the quiet-propagate branch (no traceback log) but still triggers
    cleanup so an asyncio.to_thread(hydrate) caller cancelling its
    task does not leave an orphan.

    Note: as with #23, we do NOT pre-populate `_key_current`. The
    orphan-cleanup pipeline still fires the DELs (no-op on missing keys);
    we verify the cleanup attempt via `method_calls`.
    """
    eids = [f"entity_{i}" for i in range(10)]
    mock_store.latest_features.return_value = _make_table(eids)

    with (
        patch("pyforge.hydration.insert_many", side_effect=asyncio.CancelledError),
        patch("pyforge.hydration.logger.exception") as mock_logexc,
        pytest.raises(asyncio.CancelledError),
    ):
        hydrate(_S, mock_store, mock_registry, redis_client=fake_redis)

    # Quiet-propagate: NO logger.exception call for the cancel itself.
    for call in mock_logexc.call_args_list:
        assert call.args[0] != "hydrate.force_drop_orphan", (
            "logger.exception fired on the CancelledError path — should be the quiet branch"
        )

    # Cleanup fired despite the quiet path.
    pipeline_cmds = [name for name, _, _ in fake_redis.method_calls if name.startswith("pipeline.")]
    assert "pipeline.delete" in pipeline_cmds
    assert "pipeline.srem" in pipeline_cmds
    with pytest.raises(Exception):  # noqa: B017
        posix_shm.open_existing(real_segment.name)


def test_hydrate_no_orphan_on_cancelled_during_create(
    fake_redis: _FakeRedis,
    mock_store: MagicMock,
    mock_registry: MagicMock,
) -> None:
    """registry.create raises CancelledError → NO orphan cleanup, no logger.exception.

    Sibling to #23d for the create path. There is no segment to clean,
    so cleanup must not run; cancel is quiet so no traceback log.
    """
    eids = [f"entity_{i}" for i in range(10)]
    mock_store.latest_features.return_value = _make_table(eids)
    mock_registry.create.side_effect = asyncio.CancelledError

    with (
        patch("pyforge.hydration.logger.exception") as mock_logexc,
        pytest.raises(asyncio.CancelledError),
    ):
        hydrate(_S, mock_store, mock_registry, redis_client=fake_redis)

    # No logger.exception for registry_create_failed (cancel is quiet).
    for call in mock_logexc.call_args_list:
        assert call.args[0] != "hydrate.registry_create_failed", (
            "logger.exception fired on the CancelledError path — should be the quiet branch"
        )

    # No MUTATING Redis calls (preconditions' GETs are fine).
    mutating = {"set", "pipeline", "pipeline.delete", "pipeline.srem"}
    actual = {name for name, _, _ in fake_redis.method_calls}
    assert not (actual & mutating)


# --- #24 / #25 capacity sizing -------------------------------------------


def test_hydrate_capacity_calculation(
    fake_redis: _FakeRedis,
    mock_store: MagicMock,
    mock_registry: MagicMock,
) -> None:
    """100 entities x factor 4 -> registry.create called with capacity=400."""
    eids = [f"entity_{i}" for i in range(100)]
    mock_store.latest_features.return_value = _make_table(eids)

    hydrate(
        _S,
        mock_store,
        mock_registry,
        redis_client=fake_redis,
        capacity_factor=4.0,
    )

    _, kwargs = mock_registry.create.call_args
    assert kwargs["capacity"] == 400


def test_hydrate_capacity_floor_at_16(
    fake_redis: _FakeRedis,
    mock_store: MagicMock,
    mock_registry: MagicMock,
) -> None:
    """1 entity → capacity=16 (max(4, 16) floor), not 4."""
    eids = ["entity_0"]
    mock_store.latest_features.return_value = _make_table(eids)

    hydrate(
        _S,
        mock_store,
        mock_registry,
        redis_client=fake_redis,
        capacity_factor=4.0,
    )

    _, kwargs = mock_registry.create.call_args
    assert kwargs["capacity"] == 16


# --- #28 prewarm ordering --------------------------------------------------


def test_hydrate_calls_prewarm_before_insert_many(
    fake_redis: _FakeRedis,
    mock_store: MagicMock,
    mock_registry: MagicMock,
) -> None:
    """prewarm() must be called before insert_many() — locks the
    "first-call LLVM compile is excluded from elapsed_seconds" contract.

    Patch target: `pyforge.hydration.{prewarm,insert_many}` (local
    references at module load), NOT `pyforge._internal.insert_kernel.*`.
    """
    eids = [f"entity_{i}" for i in range(10)]
    mock_store.latest_features.return_value = _make_table(eids)

    call_order: list[str] = []

    def record_prewarm() -> None:
        call_order.append("prewarm")

    def record_insert(seg: Any, table: pa.Table) -> int:
        del seg, table
        call_order.append("insert_many")
        return len(eids)

    with (
        patch("pyforge.hydration.prewarm", side_effect=record_prewarm),
        patch("pyforge.hydration.insert_many", side_effect=record_insert),
    ):
        hydrate(_S, mock_store, mock_registry, redis_client=fake_redis)

    assert call_order == ["prewarm", "insert_many"], (
        f"prewarm must precede insert_many; got order {call_order}"
    )


# ---------------------------------------------------------------------------
# Direct _force_drop_orphan tests (covers paths not exercised via hydrate).
# ---------------------------------------------------------------------------


def test_force_drop_orphan_idempotent_when_called_twice(
    fake_redis: _FakeRedis,
    real_segment: Any,
) -> None:
    """_force_drop_orphan must be idempotent — cleanup contract requires
    safety against double-invocation.
    """
    fake_redis._kv[_kbytes(_key_current(_S))] = real_segment.name.encode()

    _force_drop_orphan(fake_redis, _S, real_segment)
    # Second call: Redis keys already gone, segment already unlinked,
    # handle close on an already-closed handle. All blocks must
    # gracefully tolerate this.
    _force_drop_orphan(fake_redis, _S, real_segment)

    # No exception escaped.
    assert _kbytes(_key_current(_S)) not in fake_redis._kv


# Reference test for shm.py imports — pyforge/shm.py exposes both
# pid/pid_segments helpers; we import them in hydration.py. Validate the
# import resolves at this layer.
def test_imports_from_shm_helpers_resolve() -> None:
    pid = os.getpid()
    assert callable(_key_current)
    assert callable(_key_refcount)
    assert callable(_key_pid_segments)
    assert _key_pid_segments(pid).startswith("pyforge:pid_segments:")
