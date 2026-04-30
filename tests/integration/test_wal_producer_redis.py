"""Integration tests for the WAL producer against a real Redis stream.

Gated on ``@pytest.mark.integration``; requires the docker-compose Redis
service. The autouse ``_shm_test_isolation`` fixture in conftest scrubs
``pyforge:*`` keys after every test, so the stream is fresh between tests.
"""

from __future__ import annotations

import threading
import time

import msgpack
import pytest
import redis

from pyforge._internal.pydantic_factory import clear_cache, field_order_for
from pyforge.schema import FeatureField, FeatureSchema, dtype
from pyforge.wal import DEFAULT_STREAM_KEY, PROCESSED_KEY_PREFIX, WALProducer, WriteSyncTimeoutError

pytestmark = pytest.mark.integration


class _IntS(FeatureSchema):
    version = 1
    fields = [
        FeatureField("a", dtype.float32),
        FeatureField("b", dtype.int64),
        FeatureField("emb", dtype.float32, shape=(8,)),
    ]


@pytest.fixture(autouse=True)
def _fresh_factory_cache():
    clear_cache()
    yield
    clear_cache()


@pytest.fixture
def producer(redis_client: redis.Redis) -> WALProducer:
    return WALProducer(redis_client)


def _values() -> dict[str, object]:
    return {"a": 1.5, "b": 99, "emb": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]}


# ---------------------------------------------------------------------------
# 1. write returns IDs that are strictly monotonic.
# ---------------------------------------------------------------------------


def test_write_returns_monotonic_ids(producer: WALProducer) -> None:
    msg_ids = [producer.write(_IntS, f"e-{i}", _values()) for i in range(50)]

    def _parse(b: bytes) -> tuple[int, int]:
        ms_part, seq_part = b.decode("ascii").split("-")
        return int(ms_part), int(seq_part)

    parsed = [_parse(m) for m in msg_ids]
    assert all(parsed[i] < parsed[i + 1] for i in range(len(parsed) - 1)), parsed


# ---------------------------------------------------------------------------
# 2. Stream length respects MAXLEN approximate trimming.
# ---------------------------------------------------------------------------


def test_stream_length_respects_maxlen_approximate(redis_client: redis.Redis) -> None:
    p = WALProducer(redis_client, stream_key=b"pyforge:wal", maxlen=200)
    for i in range(1500):
        p.write(_IntS, f"e-{i}", _values())
    length = redis_client.xlen(b"pyforge:wal")
    # ``approximate=True`` allows ~10% slack; assert "well under 1500" rather
    # than an exact number. Some Redis builds trim at slightly more.
    assert length < 1500
    assert length >= 100


# ---------------------------------------------------------------------------
# 3. Stream entry has the four expected fields and a valid msgpack blob.
# ---------------------------------------------------------------------------


def test_stream_entry_shape_and_blob(producer: WALProducer, redis_client: redis.Redis) -> None:
    msg_id = producer.write(_IntS, "ent-X", _values(), event_time_ns=1700000000000000000)
    entries = redis_client.xrange(DEFAULT_STREAM_KEY, min=msg_id, max=msg_id)
    assert len(entries) == 1
    entry_id, fields = entries[0]
    assert entry_id == msg_id
    assert fields[b"schema"] == b"_IntS"
    assert fields[b"entity_id"] == b"ent-X"
    assert fields[b"event_time_ns"] == b"1700000000000000000"
    decoded = msgpack.unpackb(fields[b"blob"])
    order = field_order_for(_IntS)
    payload = _values()
    for i, name in enumerate(order):
        assert decoded[i] == payload[name]


# ---------------------------------------------------------------------------
# 4. write_sync succeeds when a stub setter writes the processed key.
# ---------------------------------------------------------------------------


def test_write_sync_returns_when_consumer_stub_sets_key(
    redis_client: redis.Redis, producer: WALProducer
) -> None:
    setter_done = threading.Event()
    setter_started = threading.Event()
    last_id: list[bytes] = []

    def stub_consumer() -> None:
        setter_started.set()
        # Poll the stream for the most recent entry, set its processed key.
        deadline = time.monotonic() + 1.0
        while time.monotonic() < deadline:
            entries = redis_client.xrevrange(DEFAULT_STREAM_KEY, count=1)
            if entries:
                msg_id, _ = entries[0]
                last_id.append(msg_id)
                redis_client.set(PROCESSED_KEY_PREFIX + msg_id, b"1", ex=86400)
                setter_done.set()
                return
            time.sleep(0.001)

    t = threading.Thread(target=stub_consumer, daemon=True)
    t.start()
    setter_started.wait(timeout=1.0)

    msg_id = producer.write_sync(_IntS, "ent-1", _values(), timeout_ms=500)
    setter_done.wait(timeout=1.0)
    t.join(timeout=1.0)
    assert last_id == [msg_id]


# ---------------------------------------------------------------------------
# 5. write_sync raises WriteSyncTimeoutError when no consumer stub runs.
# ---------------------------------------------------------------------------


def test_write_sync_raises_timeout_when_no_consumer(producer: WALProducer) -> None:
    with pytest.raises(WriteSyncTimeoutError) as exc_info:
        producer.write_sync(_IntS, "ent-1", _values(), timeout_ms=50)
    assert exc_info.value.timeout_ms == 50
    assert isinstance(exc_info.value.msg_id, bytes)
