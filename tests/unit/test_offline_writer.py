"""Unit tests for quorin.offline.ParquetDatasetStore.

Covers all Rev-2/3/4 regression items called out in the Step 11 plan.
The integration suite at ``tests/integration/test_offline_e2e.py``
exercises the same paths through a real Redis + WALConsumer.
"""

from __future__ import annotations

import asyncio
import math
import os
import sys
from pathlib import Path
from typing import Any

import pytest

pytestmark = pytest.mark.skipif(
    sys.platform == "win32",
    reason="Step 11 fsync semantics are POSIX-only",
)

import pyarrow.parquet as pq  # noqa: E402

from quorin._internal.arrow_schema import clear_cache as clear_arrow_cache  # noqa: E402
from quorin.metrics import (  # noqa: E402
    offline_bytes_written_total,
    offline_files_written_total,
    offline_flush_seconds,
)
from quorin.offline import ParquetDatasetStore, _Bucket  # noqa: E402
from quorin.schema import (  # noqa: E402
    FeatureField,
    FeatureSchema,
    dtype,
)

# ---------------------------------------------------------------------------
# Schemas. Class-level so identity is stable across tests.
# ---------------------------------------------------------------------------


class _S(FeatureSchema):
    version = 1
    fields = [
        FeatureField("a", dtype.float32),
        FeatureField("b", dtype.int64),
    ]


class _ScalarsOnly(FeatureSchema):
    version = 1
    fields = [
        FeatureField("a", dtype.float32),
        FeatureField("b", dtype.int32),
        FeatureField("c", dtype.uint8),
    ]


class _OnlyEmbedding(FeatureSchema):
    version = 1
    fields = [FeatureField("emb", dtype.float32, shape=(8,))]


class _Has2D(FeatureSchema):
    version = 1
    fields = [FeatureField("mat", dtype.float64, shape=(2, 3))]


class _SchemaA(FeatureSchema):
    version = 1
    fields = [FeatureField("x", dtype.float32)]


class _SchemaB(FeatureSchema):
    version = 1
    fields = [FeatureField("y", dtype.float32)]


# ---------------------------------------------------------------------------
# Fixtures.
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clear_arrow_plan_cache() -> None:
    clear_arrow_cache()


@pytest.fixture
def store(tmp_path: Path) -> ParquetDatasetStore:
    return ParquetDatasetStore(tmp_path)


def _list_parquet_files(base: Path) -> list[Path]:
    return sorted(base.rglob("*.parquet"))


def _histogram_count(metric: Any, **labels: str) -> float:
    """Read the observation count off a labeled Histogram via collect().

    prometheus_client doesn't expose ``._count.get()`` on label children
    (only ``._sum.get()``); the canonical read is to walk ``collect()``
    samples and pick the one whose name ends with ``_count`` and whose
    labels match.
    """
    for fam in metric.collect():
        for s in fam.samples:
            if s.name.endswith("_count") and s.labels == labels:
                return float(s.value)
    return 0.0


def _histogram_sum(metric: Any, **labels: str) -> float:
    for fam in metric.collect():
        for s in fam.samples:
            if s.name.endswith("_sum") and s.labels == labels:
                return float(s.value)
    return 0.0


# ---------------------------------------------------------------------------
# __init__ + _tmp cleanup.
# ---------------------------------------------------------------------------


def test_init_creates_base_and_tmp(tmp_path: Path) -> None:
    target = tmp_path / "fresh"
    assert not target.exists()
    ParquetDatasetStore(target)
    assert target.is_dir()
    assert (target / "_tmp").is_dir()


def test_init_cleans_up_leftover_tmp_files(tmp_path: Path) -> None:
    tmp_dir = tmp_path / "_tmp"
    tmp_dir.mkdir()
    leftover = tmp_dir / "leftover.parquet"
    leftover.write_bytes(b"orphan")
    ParquetDatasetStore(tmp_path)
    assert not leftover.exists()
    assert tmp_dir.is_dir()  # the dir itself stays


# ---------------------------------------------------------------------------
# Append + flush round-trip.
# ---------------------------------------------------------------------------


async def test_append_flush_writes_one_file_in_partition(
    store: ParquetDatasetStore, tmp_path: Path
) -> None:
    await store.append(_S, "ent-0", 0, [1.5, 7], b"100-0")
    await store.flush()
    files = _list_parquet_files(tmp_path)
    assert len(files) == 1
    f = files[0]
    # Hive-partition layout: schema=_S/event_date=YYYY-MM-DD/{uuid}.parquet
    parts = f.parts
    assert "schema=_S" in parts
    assert "event_date=1970-01-01" in parts


async def test_round_trip_columns_match_plan(store: ParquetDatasetStore, tmp_path: Path) -> None:
    await store.append(_S, "ent-0", 0, [1.5, 7], b"100-0")
    await store.flush()
    f = _list_parquet_files(tmp_path)[0]
    table = pq.read_table(f)
    assert table.num_rows == 1
    assert table.column("entity_id")[0].as_py() == "ent-0"
    assert table.column("event_time_ns")[0].as_py() == 0
    assert math.isclose(table.column("a")[0].as_py(), 1.5)
    assert table.column("b")[0].as_py() == 7
    assert table.column("msg_id_ms")[0].as_py() == 100
    assert table.column("msg_id_seq")[0].as_py() == 0


# ---------------------------------------------------------------------------
# _Bucket aliasing — the dict and the wire_lists tuple share list objects.
# ---------------------------------------------------------------------------


async def test_bucket_aliases_share_underlying_lists(
    store: ParquetDatasetStore,
) -> None:
    await store.append(_S, "ent-0", 0, [1.0, 2], b"1-0")
    bucket = next(iter(store._buffers.values()))
    assert isinstance(bucket, _Bucket)
    # The aliases are the same objects as the dict entries.
    assert bucket.entity_id_col is bucket.columns["entity_id"]
    assert bucket.event_time_ns_col is bucket.columns["event_time_ns"]
    assert bucket.msg_id_ms_col is bucket.columns["msg_id_ms"]
    assert bucket.msg_id_seq_col is bucket.columns["msg_id_seq"]
    # wire_lists entries are the data-field columns; appending via wire
    # should be visible in `columns`.
    bucket.wire_lists[0].append(99.0)
    # The underlying schema is name_hash-sorted; we just check that the
    # value landed in some data column.
    data_col_names = [
        n
        for n in bucket.columns
        if n not in {"entity_id", "event_time_ns", "msg_id_ms", "msg_id_seq"}
    ]
    landed = sum(99.0 in bucket.columns[n] for n in data_col_names)
    assert landed == 1


# ---------------------------------------------------------------------------
# Day-quantum cache (Rev-2 #2).
# ---------------------------------------------------------------------------


async def test_date_cache_reuses_strftime_within_one_day(
    store: ParquetDatasetStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    import quorin.offline as offline_mod

    real_dt = offline_mod.datetime
    call_count = {"n": 0}

    class _CountingDatetime:
        @classmethod
        def fromtimestamp(cls, *args: Any, **kwargs: Any) -> Any:
            call_count["n"] += 1
            return real_dt.fromtimestamp(*args, **kwargs)

    monkeypatch.setattr(offline_mod, "datetime", _CountingDatetime)

    # 100 appends, all within the same UTC day.
    for i in range(100):
        await store.append(_S, f"ent-{i}", i, [float(i), i], f"{i}-0".encode())
    assert call_count["n"] == 1
    # One more append a full day later → second strftime call.
    await store.append(_S, "ent-X", 86_400_000_000_000, [0.0, 0], b"0-0")
    assert call_count["n"] == 2
    assert len(store._date_cache) == 2


# ---------------------------------------------------------------------------
# column_encoding + use_dictionary applied (Rev-3 #1).
# ---------------------------------------------------------------------------


async def test_msg_id_columns_use_delta_binary_packed_not_dictionary(
    store: ParquetDatasetStore, tmp_path: Path
) -> None:
    # Append several rows so the dict-encoding heuristic has data to
    # consider (single-row files can be edge-cased).
    for i in range(20):
        await store.append(_S, f"ent-{i}", 0, [float(i), i], f"{1000 + i}-0".encode())
    await store.flush()
    f = _list_parquet_files(tmp_path)[0]
    md = pq.ParquetFile(f).metadata.row_group(0)
    schema_names = [md.column(i).path_in_schema for i in range(md.num_columns)]
    ms_idx = schema_names.index("msg_id_ms")
    seq_idx = schema_names.index("msg_id_seq")
    ms_encs = [str(e) for e in md.column(ms_idx).encodings]
    seq_encs = [str(e) for e in md.column(seq_idx).encodings]
    assert any("DELTA_BINARY_PACKED" in e for e in ms_encs), ms_encs
    assert any("DELTA_BINARY_PACKED" in e for e in seq_encs), seq_encs
    # And critically: dictionary did NOT win the encoding race.
    assert not any("DICTIONARY" in e.upper() for e in ms_encs), ms_encs
    assert not any("DICTIONARY" in e.upper() for e in seq_encs), seq_encs


# ---------------------------------------------------------------------------
# Non-msg_id columns retain dictionary encoding (Rev-4 #3).
# Verifies that PyArrow's use_dictionary={"col": False} dict-form treats
# unlisted columns as default-True (not "False for unlisted" or "True
# only for listed").
# ---------------------------------------------------------------------------


async def test_non_msg_id_columns_retain_dictionary_encoding(
    store: ParquetDatasetStore, tmp_path: Path
) -> None:
    # Deliberate repetition: 10 unique entity_ids x 10 reps each.
    for i in range(100):
        ent = f"ent-{i % 10}"
        await store.append(_S, ent, 0, [float(i), i], f"{1000 + i}-0".encode())
    await store.flush()
    f = _list_parquet_files(tmp_path)[0]
    md = pq.ParquetFile(f).metadata.row_group(0)
    schema_names = [md.column(i).path_in_schema for i in range(md.num_columns)]
    ent_idx = schema_names.index("entity_id")
    encs = [str(e) for e in md.column(ent_idx).encodings]
    assert any("DICTIONARY" in e.upper() for e in encs), (
        f"entity_id should be dict-encoded but got {encs} — "
        f"use_dictionary dict-form may have unexpected semantics"
    )


# ---------------------------------------------------------------------------
# Bytes counter increments by exact file size.
# ---------------------------------------------------------------------------


async def test_bytes_written_total_equals_file_size(
    store: ParquetDatasetStore, tmp_path: Path
) -> None:
    counter = offline_bytes_written_total.labels(schema="_S")
    before = counter._value.get()  # internal but stable in prometheus_client
    for i in range(5):
        await store.append(_S, f"ent-{i}", 0, [float(i), i], f"{i}-0".encode())
    await store.flush()
    f = _list_parquet_files(tmp_path)[0]
    after = counter._value.get()
    assert after - before == f.stat().st_size


# ---------------------------------------------------------------------------
# Per-schema label children are distinct (Rev-3 polish).
# ---------------------------------------------------------------------------


async def test_per_schema_label_children_are_distinct(
    store: ParquetDatasetStore,
) -> None:
    await store.append(_SchemaA, "ent-0", 0, [1.0], b"1-0")
    await store.append(_SchemaB, "ent-0", 0, [2.0], b"2-0")
    assert store._c_files_by_schema[_SchemaA] is not store._c_files_by_schema[_SchemaB]
    assert store._c_bytes_by_schema[_SchemaA] is not store._c_bytes_by_schema[_SchemaB]
    # Sanity-check the underlying Counter children are actually two:
    a_files = offline_files_written_total.labels(schema="_SchemaA")
    b_files = offline_files_written_total.labels(schema="_SchemaB")
    assert a_files is not b_files


# ---------------------------------------------------------------------------
# fsync count per file = 2 (Rev-3 polish).
# ---------------------------------------------------------------------------


async def test_flush_calls_fsync_twice_per_file(
    store: ParquetDatasetStore,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_fsync = os.fsync
    counter = {"n": 0}

    def counting(fd: int) -> None:
        counter["n"] += 1
        real_fsync(fd)

    monkeypatch.setattr(os, "fsync", counting)
    await store.append(_S, "ent-0", 0, [1.0, 2], b"1-0")
    await store.flush()
    assert counter["n"] == 2  # one for the file fd, one for the parent dir


# ---------------------------------------------------------------------------
# Length-mismatch validation (Rev-3 #2).
# ---------------------------------------------------------------------------


async def test_length_mismatch_short_raises_and_does_not_corrupt_bucket(
    store: ParquetDatasetStore,
) -> None:
    await store.append(_S, "ent-0", 0, [1.0, 2], b"1-0")
    bucket = next(iter(store._buffers.values()))
    snapshot_lengths = {n: len(v) for n, v in bucket.columns.items()}
    with pytest.raises(ValueError, match="length"):
        await store.append(_S, "ent-1", 0, [1.0], b"2-0")  # missing one
    after = {n: len(v) for n, v in bucket.columns.items()}
    assert snapshot_lengths == after


async def test_length_mismatch_long_raises_and_does_not_corrupt_bucket(
    store: ParquetDatasetStore,
) -> None:
    await store.append(_S, "ent-0", 0, [1.0, 2], b"1-0")
    bucket = next(iter(store._buffers.values()))
    snapshot_lengths = {n: len(v) for n, v in bucket.columns.items()}
    with pytest.raises(ValueError, match="length"):
        await store.append(_S, "ent-1", 0, [1.0, 2, 99.0], b"2-0")  # too many
    after = {n: len(v) for n, v in bucket.columns.items()}
    assert snapshot_lengths == after


# ---------------------------------------------------------------------------
# Malformed msg_id atomicity (Rev-4 #2).
# ---------------------------------------------------------------------------


async def test_malformed_msg_id_does_not_corrupt_bucket(
    store: ParquetDatasetStore,
) -> None:
    await store.append(_S, "ent-0", 0, [1.0, 2], b"100-0")
    bucket = next(iter(store._buffers.values()))
    snapshot_lengths = {n: len(v) for n, v in bucket.columns.items()}
    with pytest.raises(ValueError):
        await store.append(_S, "ent-1", 0, [1.0, 2], b"not-a-valid-id")
    after = {n: len(v) for n, v in bucket.columns.items()}
    assert snapshot_lengths == after, (
        "bucket corrupted by partial append — msg_id parse must run before any column append"
    )


async def test_msg_id_with_no_dash_raises_atomically(
    store: ParquetDatasetStore,
) -> None:
    with pytest.raises(ValueError):
        await store.append(_S, "ent-0", 0, [1.0, 2], b"")
    # Bucket may or may not have been created; if it was, it must be empty.
    if store._buffers:
        bucket = next(iter(store._buffers.values()))
        for col in bucket.columns.values():
            assert len(col) == 0


# ---------------------------------------------------------------------------
# Empty bucket is not written (Rev-4 polish).
# ---------------------------------------------------------------------------


async def test_empty_bucket_is_not_written_and_no_metric_observation(
    store: ParquetDatasetStore, tmp_path: Path
) -> None:
    flush_ok_count_before = _histogram_count(offline_flush_seconds, outcome="ok")
    # Force a failed first-append (length mismatch) so a bucket is
    # created but empty.
    with pytest.raises(ValueError):
        await store.append(_S, "ent-0", 0, [1.0], b"100-0")  # wrong length
    # Bucket exists but empty.
    assert _S in {schema for schema, _ in store._buffers}
    bucket = next(iter(store._buffers.values()))
    assert len(bucket.entity_id_col) == 0
    # Flush: must not write a 0-row file, must not bump the ok metric.
    await store.flush()
    assert _list_parquet_files(tmp_path) == []
    flush_ok_count_after = _histogram_count(offline_flush_seconds, outcome="ok")
    assert flush_ok_count_after == flush_ok_count_before


# ---------------------------------------------------------------------------
# Empty flush is a no-op.
# ---------------------------------------------------------------------------


async def test_empty_flush_is_noop(store: ParquetDatasetStore, tmp_path: Path) -> None:
    flush_ok_count_before = _histogram_count(offline_flush_seconds, outcome="ok")
    await store.flush()
    assert _list_parquet_files(tmp_path) == []
    flush_ok_count_after = _histogram_count(offline_flush_seconds, outcome="ok")
    assert flush_ok_count_after == flush_ok_count_before


async def test_single_row_flush(store: ParquetDatasetStore, tmp_path: Path) -> None:
    await store.append(_S, "ent-0", 0, [1.0, 2], b"1-0")
    await store.flush()
    files = _list_parquet_files(tmp_path)
    assert len(files) == 1
    table = pq.read_table(files[0])
    assert table.num_rows == 1


# ---------------------------------------------------------------------------
# close() drains pending appends.
# ---------------------------------------------------------------------------


async def test_close_drains_pending_appends(store: ParquetDatasetStore, tmp_path: Path) -> None:
    await store.append(_S, "ent-0", 0, [1.0, 2], b"1-0")
    await store.close()
    files = _list_parquet_files(tmp_path)
    assert len(files) == 1


async def test_close_is_idempotent(store: ParquetDatasetStore, tmp_path: Path) -> None:
    await store.append(_S, "ent-0", 0, [1.0, 2], b"1-0")
    await store.close()
    await store.close()  # must not error
    assert len(_list_parquet_files(tmp_path)) == 1


# ---------------------------------------------------------------------------
# include_msg_id=False.
# ---------------------------------------------------------------------------


async def test_include_msg_id_false_omits_columns_and_skips_parse(
    tmp_path: Path,
) -> None:
    store = ParquetDatasetStore(tmp_path, include_msg_id=False)
    # Pass garbage msg_id to prove the parse is skipped — nothing
    # should raise.
    await store.append(_S, "ent-0", 0, [1.0, 2], b"this-would-explode-on-parse")
    await store.flush()
    f = _list_parquet_files(tmp_path)[0]
    table = pq.read_table(f)
    assert "msg_id_ms" not in table.schema.names
    assert "msg_id_seq" not in table.schema.names


# ---------------------------------------------------------------------------
# Mixed-date flush.
# ---------------------------------------------------------------------------


async def test_mixed_date_flush_writes_one_file_per_partition(
    store: ParquetDatasetStore, tmp_path: Path
) -> None:
    day_ns = 86_400_000_000_000
    await store.append(_S, "ent-0", 0, [1.0, 2], b"1-0")
    await store.append(_S, "ent-1", day_ns, [3.0, 4], b"2-0")
    await store.append(_S, "ent-2", 2 * day_ns, [5.0, 6], b"3-0")
    await store.flush()
    files = _list_parquet_files(tmp_path)
    assert len(files) == 3
    dates = sorted({p.parent.name for p in files})
    assert dates == [
        "event_date=1970-01-01",
        "event_date=1970-01-02",
        "event_date=1970-01-03",
    ]


# ---------------------------------------------------------------------------
# Schema shape coverage.
# ---------------------------------------------------------------------------


async def test_scalars_only_round_trips(tmp_path: Path) -> None:
    from quorin.schema import _hash_name

    store = ParquetDatasetStore(tmp_path)
    # Producer wire order is name-hash-sorted (ADR-008); construct the
    # values list in that order, not declaration order.
    by_name = {"a": 1.5, "b": 7, "c": 200}
    wire_order = sorted(_ScalarsOnly.fields, key=lambda f: _hash_name(f.name))
    values = [by_name[f.name] for f in wire_order]
    await store.append(_ScalarsOnly, "ent-0", 0, values, b"1-0")
    await store.flush()
    f = _list_parquet_files(tmp_path)[0]
    table = pq.read_table(f)
    assert math.isclose(table.column("a")[0].as_py(), 1.5)
    assert table.column("b")[0].as_py() == 7
    assert table.column("c")[0].as_py() == 200


async def test_only_1d_embedding_round_trips(tmp_path: Path) -> None:
    store = ParquetDatasetStore(tmp_path)
    emb = [float(i) for i in range(8)]
    await store.append(_OnlyEmbedding, "ent-0", 0, [emb], b"1-0")
    await store.flush()
    f = _list_parquet_files(tmp_path)[0]
    table = pq.read_table(f)
    assert table.column("emb")[0].as_py() == emb


async def test_2d_embedding_round_trips(tmp_path: Path) -> None:
    store = ParquetDatasetStore(tmp_path)
    mat = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
    await store.append(_Has2D, "ent-0", 0, [mat], b"1-0")
    await store.flush()
    f = _list_parquet_files(tmp_path)[0]
    table = pq.read_table(f)
    assert table.column("mat")[0].as_py() == mat


# ---------------------------------------------------------------------------
# Boundary timestamps.
# ---------------------------------------------------------------------------


async def test_event_time_zero_yields_epoch_partition(
    store: ParquetDatasetStore, tmp_path: Path
) -> None:
    await store.append(_S, "ent-0", 0, [1.0, 2], b"1-0")
    await store.flush()
    f = _list_parquet_files(tmp_path)[0]
    assert "event_date=1970-01-01" in f.parts


async def test_event_time_far_future_yields_valid_partition(
    store: ParquetDatasetStore, tmp_path: Path
) -> None:
    # 2**62 ns ≈ year 2116. Some platforms cap at 2038 for naive
    # fromtimestamp; we use UTC so that's fine on Linux.
    far = 2**62
    await store.append(_S, "ent-0", far, [1.0, 2], b"1-0")
    await store.flush()
    f = _list_parquet_files(tmp_path)[0]
    parent = f.parent.name
    assert parent.startswith("event_date=")
    # Year is somewhere in the 21xx range.
    year_str = parent[len("event_date=") :][:4]
    assert year_str.startswith("21"), parent


# ---------------------------------------------------------------------------
# Cancellation mid-flush.
# ---------------------------------------------------------------------------


async def test_cancellation_during_flush_restores_buffers_and_records_metric(
    store: ParquetDatasetStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    """CR.A.2 (v0.1.1): cancellation mid-flush MUST restore unwritten
    buckets to ``self._buffers`` so the deferred-XACK durability
    contract (ADR-009 §3) holds.

    Pre-v0.1.1 behavior was "buffers cleared on entry, cancellation
    drops everything" — that lost data because XACK fired on online-
    store success but offline data was gone.
    """
    import quorin.offline as offline_mod

    cancelled_before = _histogram_count(offline_flush_seconds, outcome="cancelled")

    def boom(*args: Any, **kwargs: Any) -> None:
        raise asyncio.CancelledError()

    monkeypatch.setattr(offline_mod.pq, "write_table", boom)
    await store.append(_S, "ent-0", 0, [1.0, 2], b"1-0")
    with pytest.raises(asyncio.CancelledError):
        await store.flush()
    # CR.A.2: buffer for the un-written bucket MUST be restored.
    assert store._buffers, "unwritten bucket must be restored on cancellation"
    # Verify the restored bucket has the row we appended.
    only_bucket = next(iter(store._buffers.values()))
    assert only_bucket.entity_id_col == ["ent-0"]
    cancelled_after = _histogram_count(offline_flush_seconds, outcome="cancelled")
    assert cancelled_after - cancelled_before == 1


# ---------------------------------------------------------------------------
# Disk-full simulation.
# ---------------------------------------------------------------------------


async def test_disk_full_oserror_records_metric_and_unlinks_tmp(
    store: ParquetDatasetStore,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import quorin.offline as offline_mod

    err_before = _histogram_count(offline_flush_seconds, outcome="error")

    captured_tmp_paths: list[Path] = []
    real_write = offline_mod.pq.write_table

    def fake_write(table: Any, path: Any, **kwargs: Any) -> None:
        captured_tmp_paths.append(Path(path))
        # Touch the tmp file so we can verify the cleanup branch unlinks
        # it. (Some OSError paths leave a partial file behind.)
        Path(path).write_bytes(b"partial")
        raise OSError("disk full")

    monkeypatch.setattr(offline_mod.pq, "write_table", fake_write)
    await store.append(_S, "ent-0", 0, [1.0, 2], b"1-0")
    with pytest.raises(OSError, match="disk full"):
        await store.flush()
    # CR.A.2 (v0.1.1): unwritten buckets restored to self._buffers so
    # the next flush retries them. Pre-v0.1.1 silently dropped them
    # while online-store XACK still fired — data loss.
    assert store._buffers, "unwritten bucket must be restored on disk-full"
    only_bucket = next(iter(store._buffers.values()))
    assert only_bucket.entity_id_col == ["ent-0"]
    err_after = _histogram_count(offline_flush_seconds, outcome="error")
    assert err_after - err_before == 1
    # The tmp file we touched in the fake should have been unlinked
    # (Rev-4 polish #1).
    assert captured_tmp_paths
    assert all(not p.exists() for p in captured_tmp_paths)
    # Restore the real write_table so other tests aren't affected.
    monkeypatch.setattr(offline_mod.pq, "write_table", real_write)


async def test_partial_flush_failure_restores_only_unwritten_buckets(
    store: ParquetDatasetStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    """CR.A.2 (v0.1.1): when ``_write_table`` succeeds for the first
    K of N buckets and fails for the K-th, only buckets K..N are
    restored to ``self._buffers``. The first K were durable to disk
    and ARE NOT re-buffered (no double-write).
    """
    import quorin.offline as offline_mod

    # Two distinct schemas → two distinct (schema, date_str) buckets.
    await store.append(_SchemaA, "ent-a", 0, [1.5], b"1-0")
    await store.append(_SchemaB, "ent-b", 0, [2.5], b"1-1")
    assert len(store._buffers) == 2, "setup: two buckets seeded"

    # Fake write_table: succeeds for the first call, fails for the second.
    real_write = offline_mod.pq.write_table
    call_count = {"n": 0}

    def selective_write(table: Any, path: Any, **kwargs: Any) -> None:
        call_count["n"] += 1
        if call_count["n"] == 1:
            real_write(table, path, **kwargs)
            return
        # Touch a partial tmp so the cleanup branch fires (mirrors the
        # disk-full test's pattern); then raise OSError.
        Path(path).write_bytes(b"partial")
        raise OSError("disk full on second bucket")

    monkeypatch.setattr(offline_mod.pq, "write_table", selective_write)
    with pytest.raises(OSError, match="disk full on second bucket"):
        await store.flush()
    # The first bucket is durable — must NOT be in _buffers.
    # The second bucket failed — MUST be in _buffers for the next flush.
    assert len(store._buffers) == 1, "exactly the unwritten bucket should be restored"
    surviving_bucket = next(iter(store._buffers.values()))
    # Don't assume which schema dict-iteration ordered — assert the
    # survivor is one of the two and has its row.
    assert surviving_bucket.entity_id_col in (["ent-a"], ["ent-b"])
    monkeypatch.setattr(offline_mod.pq, "write_table", real_write)


async def test_runtime_name_mutation_rejected_at_write(
    store: ParquetDatasetStore,
) -> None:
    """CR.C.1 (v0.1.1): defense-in-depth. ``FeatureSchema.__init_subclass__``
    validates ``cls.__name__`` at class-definition time (CR.A.6), but
    ``cls.__name__`` is writable. A caller that does
    ``Schema.__name__ = "../etc/passwd"`` after class definition would
    slip past CR.A.6 and the runtime name would flow into Parquet path
    construction. Re-validating at the write boundary is the binding
    defense.

    NOTE: we mutate via direct attribute assignment, then restore at end
    so other tests aren't affected by the polluted ``__name__``.
    """

    class _ToBeMutated(FeatureSchema):
        version = 1
        fields = [FeatureField("a", dtype.float32)]

    # Append succeeds (append doesn't read __name__ for path-building).
    await store.append(_ToBeMutated, "ent-0", 0, [1.0], b"1-0")
    saved_name = _ToBeMutated.__name__
    try:
        _ToBeMutated.__name__ = "../etc/passwd"  # path-traversal attempt
        with pytest.raises(ValueError, match="invalid for filesystem path"):
            await store.flush()
    finally:
        _ToBeMutated.__name__ = saved_name


# ---------------------------------------------------------------------------
# Negative event_time_ns is out-of-spec; underlying datetime handles it.
# We don't defensively check, so the test just confirms that *something*
# either raises or rounds. On Linux negative ns rounds backward in time.
# ---------------------------------------------------------------------------


async def test_negative_event_time_ns_either_raises_or_handles(
    store: ParquetDatasetStore,
) -> None:
    # Whatever the platform does, the bucket invariant must hold:
    # success → bucket has the row; failure → bucket unmodified.
    try:
        await store.append(_S, "ent-0", -(10**12), [1.0, 2], b"1-0")
    except (OSError, ValueError, OverflowError):
        # Underlying datetime / strftime can raise on some platforms.
        # We don't add defensive checks — propagation is the contract.
        return
    # If it didn't raise, the bucket has one row.
    bucket = next(iter(store._buffers.values()))
    assert len(bucket.entity_id_col) == 1


# ---------------------------------------------------------------------------
# Non-ASCII entity_id (Rev-3 polish edge case).
# ---------------------------------------------------------------------------


async def test_non_ascii_entity_id_round_trips(store: ParquetDatasetStore, tmp_path: Path) -> None:
    weird = "ent-😀-日本語-Ω"
    await store.append(_S, weird, 0, [1.0, 2], b"1-0")
    await store.flush()
    f = _list_parquet_files(tmp_path)[0]
    table = pq.read_table(f)
    assert table.column("entity_id")[0].as_py() == weird
