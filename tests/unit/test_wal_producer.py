"""Unit tests for pyforge.wal.WALProducer.

These tests use a local in-memory FakeRedis stub to exercise the producer
without requiring the integration Redis service. The integration suite at
tests/integration/test_wal_producer_redis.py exercises the same paths
against a real Redis.
"""

from __future__ import annotations

import math
import threading
import time
from typing import Any

import msgpack
import numpy as np
import pydantic
import pytest

from pyforge._internal.pydantic_factory import clear_cache, field_order_for
from pyforge.schema import FeatureField, FeatureSchema, dtype
from pyforge.wal import (
    _F_BLOB,
    _F_ENTITY_ID,
    _F_EVENT_TIME,
    _F_SCHEMA,
    DEFAULT_STREAM_KEY,
    PROCESSED_KEY_PREFIX,
    WALProducer,
    WriteSyncTimeoutError,
)

# ---------------------------------------------------------------------------
# FakeRedis — minimal stub of redis.Redis for unit tests.
# ---------------------------------------------------------------------------


class FakeRedis:
    """Tiny in-memory Redis stub: xadd / exists / set are all the producer needs.

    Returns deterministic message IDs of the form ``b"<seq>-0"`` so test
    assertions are stable. Records every xadd call in ``self.calls`` so
    individual fields can be inspected.
    """

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self._seq = 0
        self._kv: dict[bytes, bytes] = {}
        # Optional callback fired on every xadd. Set by tests that want to
        # synthesize the consumer's processed-key write.
        self.on_xadd: Any = None

    def xadd(
        self,
        name: bytes,
        fields: dict[bytes, bytes],
        maxlen: int | None = None,
        approximate: bool = False,
    ) -> bytes:
        self._seq += 1
        msg_id = f"{self._seq}-0".encode()
        self.calls.append(
            {
                "name": name,
                "fields": dict(fields),
                "maxlen": maxlen,
                "approximate": approximate,
                "msg_id": msg_id,
            }
        )
        if self.on_xadd is not None:
            self.on_xadd(msg_id, self)
        return msg_id

    def exists(self, key: bytes) -> int:
        return 1 if key in self._kv else 0

    def set(self, key: bytes, value: bytes, ex: int | None = None) -> bool:
        self._kv[key] = value
        return True


class _S(FeatureSchema):
    version = 1
    fields = [
        FeatureField("a", dtype.float32),
        FeatureField("b", dtype.int64),
        FeatureField("emb", dtype.float32, shape=(4,)),
    ]


@pytest.fixture(autouse=True)
def _fresh_factory_cache():
    clear_cache()
    yield
    clear_cache()


@pytest.fixture
def producer() -> tuple[WALProducer, FakeRedis]:
    fake = FakeRedis()
    return WALProducer(fake), fake  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# 1. write returns a non-empty bytes msg_id.
# ---------------------------------------------------------------------------


def test_write_returns_bytes_msg_id(producer: tuple[WALProducer, FakeRedis]) -> None:
    p, _fake = producer
    msg_id = p.write(_S, "ent-1", {"a": 1.0, "b": 2, "emb": [0.0, 1.0, 2.0, 3.0]})
    assert isinstance(msg_id, bytes)
    assert len(msg_id) > 0


# ---------------------------------------------------------------------------
# 2. write calls XADD with the expected 4-key field dict.
# ---------------------------------------------------------------------------


def test_write_calls_xadd_with_four_field_dict(producer: tuple[WALProducer, FakeRedis]) -> None:
    p, fake = producer
    p.write(_S, "user-42", {"a": 1.5, "b": 7, "emb": [0.1, 0.2, 0.3, 0.4]})
    assert len(fake.calls) == 1
    call = fake.calls[0]
    assert call["name"] == DEFAULT_STREAM_KEY
    assert set(call["fields"].keys()) == {_F_SCHEMA, _F_ENTITY_ID, _F_EVENT_TIME, _F_BLOB}
    assert call["fields"][_F_SCHEMA] == b"_S"
    assert call["fields"][_F_ENTITY_ID] == b"user-42"
    assert call["maxlen"] == 1_000_000
    assert call["approximate"] is True


# ---------------------------------------------------------------------------
# 3. blob is valid msgpack and round-trips to schema-ordered list.
# ---------------------------------------------------------------------------


def test_blob_is_msgpack_list_in_name_hash_order(producer: tuple[WALProducer, FakeRedis]) -> None:
    p, fake = producer
    payload = {"a": 0.5, "b": 99, "emb": [10.0, 20.0, 30.0, 40.0]}
    p.write(_S, "x", payload)
    blob = fake.calls[0]["fields"][_F_BLOB]
    decoded = msgpack.unpackb(blob)
    order = field_order_for(_S)
    assert isinstance(decoded, list)
    assert len(decoded) == len(order)
    # Decoded values should match the input dict, indexed by name_hash order.
    for i, name in enumerate(order):
        assert decoded[i] == payload[name]


# ---------------------------------------------------------------------------
# 4-7. Validation rejects bad inputs.
# ---------------------------------------------------------------------------


def test_unknown_field_rejected(producer: tuple[WALProducer, FakeRedis]) -> None:
    p, fake = producer
    with pytest.raises(pydantic.ValidationError):
        p.write(_S, "x", {"a": 0.0, "b": 0, "emb": [0.0, 0.0, 0.0, 0.0], "extra": 1})
    assert fake.calls == []  # XADD must NOT be called on validation failure


def test_wrong_dtype_rejected(producer: tuple[WALProducer, FakeRedis]) -> None:
    p, _fake = producer
    with pytest.raises(pydantic.ValidationError):
        p.write(_S, "x", {"a": "not a float", "b": 0, "emb": [0.0, 0.0, 0.0, 0.0]})


def test_wrong_shape_rejected(producer: tuple[WALProducer, FakeRedis]) -> None:
    p, _fake = producer
    with pytest.raises(pydantic.ValidationError):
        p.write(_S, "x", {"a": 0.0, "b": 0, "emb": [0.0, 0.0, 0.0]})


def test_unsigned_negative_rejected() -> None:
    cls = type(
        "U8S",
        (FeatureSchema,),
        {"version": 1, "fields": [FeatureField("v", dtype.uint8)]},
    )
    fake = FakeRedis()
    p = WALProducer(fake)  # type: ignore[arg-type]
    with pytest.raises(pydantic.ValidationError):
        p.write(cls, "x", {"v": -1})


# ---------------------------------------------------------------------------
# 8-9. event_time_ns defaults to time.time_ns().
# ---------------------------------------------------------------------------


def test_event_time_defaults_to_now(producer: tuple[WALProducer, FakeRedis]) -> None:
    p, fake = producer
    before = time.time_ns()
    p.write(_S, "x", {"a": 0.0, "b": 0, "emb": [0.0] * 4})
    after = time.time_ns()
    rendered = fake.calls[0]["fields"][_F_EVENT_TIME]
    parsed = int(rendered.decode("ascii"))
    assert before <= parsed <= after


def test_event_time_explicit_value_recorded(producer: tuple[WALProducer, FakeRedis]) -> None:
    p, fake = producer
    p.write(_S, "x", {"a": 0.0, "b": 0, "emb": [0.0] * 4}, event_time_ns=42)
    assert fake.calls[0]["fields"][_F_EVENT_TIME] == b"42"


# ---------------------------------------------------------------------------
# 10. write_sync returns when processed key appears.
# ---------------------------------------------------------------------------


def test_write_sync_returns_when_processed_key_set_synchronously() -> None:
    fake = FakeRedis()

    def setter(msg_id: bytes, redis_stub: FakeRedis) -> None:
        # Synthesize the consumer setting the processed key immediately.
        redis_stub.set(PROCESSED_KEY_PREFIX + msg_id, b"1", ex=86400)

    fake.on_xadd = setter
    p = WALProducer(fake)  # type: ignore[arg-type]
    msg_id = p.write_sync(_S, "x", {"a": 0.0, "b": 0, "emb": [0.0] * 4}, timeout_ms=100)
    assert msg_id == fake.calls[0]["msg_id"]


# ---------------------------------------------------------------------------
# 11. write_sync raises WriteSyncTimeoutError when key never appears.
# ---------------------------------------------------------------------------


def test_write_sync_raises_timeout_when_consumer_silent() -> None:
    fake = FakeRedis()
    p = WALProducer(fake)  # type: ignore[arg-type]
    t0 = time.monotonic()
    with pytest.raises(WriteSyncTimeoutError) as exc_info:
        p.write_sync(_S, "x", {"a": 0.0, "b": 0, "emb": [0.0] * 4}, timeout_ms=20)
    elapsed = time.monotonic() - t0
    # Allow generous slack on slow CI runners.
    assert exc_info.value.timeout_ms == 20
    assert exc_info.value.msg_id == fake.calls[0]["msg_id"]
    assert elapsed >= 0.018  # at least the timeout
    assert elapsed < 0.5  # but not absurd


# ---------------------------------------------------------------------------
# 12. write_sync backoff is bounded <= 10 ms.
# ---------------------------------------------------------------------------


def test_write_sync_backoff_capped_at_10ms(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = FakeRedis()
    p = WALProducer(fake)  # type: ignore[arg-type]
    sleeps: list[float] = []
    real_sleep = time.sleep

    def recording_sleep(d: float) -> None:
        sleeps.append(d)
        real_sleep(0)  # don't actually sleep — fast test

    monkeypatch.setattr(time, "sleep", recording_sleep)
    with pytest.raises(WriteSyncTimeoutError):
        p.write_sync(_S, "x", {"a": 0.0, "b": 0, "emb": [0.0] * 4}, timeout_ms=100)
    assert sleeps, "expected at least one sleep call inside write_sync"
    assert max(sleeps) <= 0.010 + 1e-9


# ---------------------------------------------------------------------------
# 13. WALProducer accepts a custom stream_key.
# ---------------------------------------------------------------------------


def test_custom_stream_key_used_in_xadd() -> None:
    fake = FakeRedis()
    p = WALProducer(fake, stream_key=b"custom:wal", maxlen=500)  # type: ignore[arg-type]
    p.write(_S, "x", {"a": 0.0, "b": 0, "emb": [0.0] * 4})
    assert fake.calls[0]["name"] == b"custom:wal"
    assert fake.calls[0]["maxlen"] == 500


# ---------------------------------------------------------------------------
# 14. WALProducer reuses one msgpack.Packer() across calls.
# ---------------------------------------------------------------------------


def test_msgpack_packer_reused_across_calls() -> None:
    fake = FakeRedis()
    p = WALProducer(fake)  # type: ignore[arg-type]
    packer_before = p._packer
    p.write(_S, "x", {"a": 0.0, "b": 0, "emb": [0.0] * 4})
    p.write(_S, "y", {"a": 1.0, "b": 1, "emb": [1.0] * 4})
    assert p._packer is packer_before


# ---------------------------------------------------------------------------
# 15. Schema-name encoding cache: same schema → encoded once.
# ---------------------------------------------------------------------------


def test_schema_name_encoded_once_per_class() -> None:
    fake = FakeRedis()
    p = WALProducer(fake)  # type: ignore[arg-type]
    p.write(_S, "x", {"a": 0.0, "b": 0, "emb": [0.0] * 4})
    p.write(_S, "y", {"a": 0.0, "b": 0, "emb": [0.0] * 4})
    assert _S in p._schema_name_cache
    assert p._schema_name_cache[_S] == b"_S"
    # Same bytes object reused across calls — identity assertion via the cache.
    cached_first = p._schema_name_cache[_S]
    p.write(_S, "z", {"a": 0.0, "b": 0, "emb": [0.0] * 4})
    cached_second = p._schema_name_cache[_S]
    assert cached_first is cached_second


# ---------------------------------------------------------------------------
# 16-19. NaN / Inf round-trip — invariant #12 contract.
# ---------------------------------------------------------------------------


def _decode_blob(fake: FakeRedis, idx: int = -1) -> list[Any]:
    return list(msgpack.unpackb(fake.calls[idx]["fields"][_F_BLOB]))


def test_float32_nan_accepted_through_write() -> None:
    fake = FakeRedis()
    p = WALProducer(fake)  # type: ignore[arg-type]
    p.write(_S, "x", {"a": np.float32(np.nan), "b": 0, "emb": [0.0] * 4})
    blob = _decode_blob(fake)
    order = field_order_for(_S)
    a_idx = order.index("a")
    assert math.isnan(blob[a_idx])


def test_float64_inf_and_neg_inf_accepted_through_write() -> None:
    cls = type(
        "F64S",
        (FeatureSchema,),
        {"version": 1, "fields": [FeatureField("v", dtype.float64)]},
    )
    fake = FakeRedis()
    p = WALProducer(fake)  # type: ignore[arg-type]
    p.write(cls, "p", {"v": np.float64(np.inf)})
    p.write(cls, "n", {"v": np.float64(-np.inf)})
    pos_blob = _decode_blob(fake, 0)
    neg_blob = _decode_blob(fake, 1)
    assert pos_blob[0] == float("inf")
    assert neg_blob[0] == float("-inf")


def test_nan_inside_1d_shape_field_round_trips() -> None:
    fake = FakeRedis()
    p = WALProducer(fake)  # type: ignore[arg-type]
    emb = [0.0, float("nan"), float("inf"), float("-inf")]
    p.write(_S, "x", {"a": 0.0, "b": 0, "emb": emb})
    blob = _decode_blob(fake)
    order = field_order_for(_S)
    decoded_emb = blob[order.index("emb")]
    assert decoded_emb[0] == 0.0
    assert math.isnan(decoded_emb[1])
    assert decoded_emb[2] == float("inf")
    assert decoded_emb[3] == float("-inf")


def test_strict_mode_regression_guard_numpy_scalar_accepted() -> None:
    """Locks 'no global strict mode' decision. If a future PR sets
    ConfigDict(strict=True), numpy scalar inputs will reject and this
    test fails — reversing the strict-mode flag would also break the
    assemble path's load-bearing numpy round-trip."""
    fake = FakeRedis()
    p = WALProducer(fake)  # type: ignore[arg-type]
    p.write(
        _S,
        "x",
        {
            "a": np.float32(0.5),
            "b": np.int64(42),
            "emb": np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float32).tolist(),
        },
    )
    assert len(fake.calls) == 1


# ---------------------------------------------------------------------------
# Extra: validation failure does not call XADD.
# ---------------------------------------------------------------------------


def test_validation_failure_does_not_call_xadd() -> None:
    fake = FakeRedis()
    p = WALProducer(fake)  # type: ignore[arg-type]
    with pytest.raises(pydantic.ValidationError):
        p.write(_S, "x", {"a": "bad", "b": 0, "emb": [0.0] * 4})
    assert fake.calls == []


def test_write_sync_validation_failure_increments_error_counter() -> None:
    fake = FakeRedis()
    p = WALProducer(fake)  # type: ignore[arg-type]
    before = p._c_sync_error._value.get()
    with pytest.raises(pydantic.ValidationError):
        p.write_sync(_S, "x", {"a": "bad", "b": 0, "emb": [0.0] * 4}, timeout_ms=10)
    after = p._c_sync_error._value.get()
    assert after == before + 1


# ---------------------------------------------------------------------------
# Threaded write_sync: setter from another thread eventually unblocks.
# ---------------------------------------------------------------------------


def test_write_sync_returns_when_setter_runs_on_background_thread() -> None:
    fake = FakeRedis()
    p = WALProducer(fake)  # type: ignore[arg-type]
    target_msg_ids: list[bytes] = []

    def setter() -> None:
        # Wait for the xadd to record a msg_id, then set the processed key.
        for _ in range(100):
            if fake.calls:
                msg_id = fake.calls[0]["msg_id"]
                target_msg_ids.append(msg_id)
                fake.set(PROCESSED_KEY_PREFIX + msg_id, b"1")
                return
            time.sleep(0.001)

    t = threading.Thread(target=setter, daemon=True)
    t.start()
    msg_id = p.write_sync(_S, "x", {"a": 0.0, "b": 0, "emb": [0.0] * 4}, timeout_ms=200)
    t.join(timeout=1.0)
    assert target_msg_ids == [msg_id]
