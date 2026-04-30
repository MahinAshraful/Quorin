"""Parquet dataset store for the WAL consumer (Step 11).

Implements :class:`ParquetDatasetStore`, an async ``OfflineWriter`` that
satisfies the Protocol declared in
:mod:`pyforge.wal_consumer`. Each ``flush()`` writes one Parquet file per
``(schema, event_date)`` bucket via tmp → fsync → atomic rename. Files
are hive-partitioned. Crash-safe by construction: a partial write either
sits in ``_tmp/`` (GC'd at next ``__init__``) or never exists.

See [`progress/step11_plan.md`](progress/step11_plan.md) and
[`docs/adr/010-parquet-offline-store.md`](docs/adr/010-parquet-offline-store.md)
for the full design lock and rationale.
"""

from __future__ import annotations

import asyncio
import os
import pathlib
import shutil
import time
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Final

import pyarrow as pa
import pyarrow.parquet as pq

from pyforge._internal.arrow_schema import _arrow_plan_for, _ArrowPlan
from pyforge.metrics import (
    offline_bytes_written_total,
    offline_files_written_total,
    offline_flush_rows,
    offline_flush_seconds,
)
from pyforge.schema import FeatureSchema

_DAY_NS: Final[int] = 86_400_000_000_000
"""Nanoseconds per UTC day. Used as the day-quantum key for the
``_date_cache`` so the writer hits the cache once per day instead of
paying ~3-5 µs of ``datetime.fromtimestamp`` + ``strftime`` per append.
"""


@dataclass(slots=True)
class _Bucket:
    """Per-(schema, date) accumulator with pre-resolved column-list refs.

    ``columns`` is the dict consumed by :meth:`pa.Table.from_pydict` at
    flush time. The other attributes are aliases into the same list
    objects so the hot ``append`` path never touches the dict. Lists are
    mutable references; the dict and the wire_lists tuple point at the
    same physical lists.
    """

    columns: dict[str, list[Any]]
    wire_lists: tuple[list[Any], ...]
    entity_id_col: list[str]
    event_time_ns_col: list[int]
    msg_id_ms_col: list[int] | None
    msg_id_seq_col: list[int] | None


class ParquetDatasetStore:
    """Single-coroutine async offline writer for the Pyforge WAL consumer.

    **Concurrency contract:** This class is NOT safe for concurrent
    ``append``/``flush`` from multiple asyncio tasks. The Step 10
    :class:`pyforge.wal_consumer.WALConsumer` guarantees single-coroutine
    access; direct callers (tests, scripts) must wrap in their own
    serialization.

    **Memory contract:** Peak in-memory buffer size scales linearly with
    the consumer's ``max_pending_ack`` x (n_fields x ~50 B Python
    overhead per element). At default ``max_pending_ack=10_000`` and
    200-field schemas, expect ~88 MB peak per writer; at the build-plan
    50k peak, ~440 MB. Operators tuning ``max_pending_ack`` for recovery
    SLA must size memory accordingly.

    **Filesystem contract:** ``base`` and ``base/_tmp`` must be on the
    same filesystem so :func:`os.rename` is atomic. Crossing a mount
    point will raise ``OSError(EXDEV)`` from a flush.

    **Latency contract:** :meth:`flush` is synchronous and blocks the
    asyncio event loop for ~100-150 ms native (~200+ ms WSL2) at
    10k rows x 200 fields. ``pa.Table.from_pydict`` Python->Arrow
    conversion dominates (~80-150 ms); zstd ~30 ms; fsync ~5-15 ms.
    The read coroutine in the consumer does NOT progress during flush;
    Redis buffers messages on the stream side and the next
    ``XREADGROUP`` catches up. Throughput is preserved; per-message
    processing latency exhibits a ~100-200 ms spike on each flush
    boundary. This is the documented baseline, not a regression.
    See ADR-010 §8.

    **entity_id length:** Uses ``pa.string()`` (32-bit offsets, 2 GB
    per column per file). At 36-char UUIDs x 50k rows that's 1.8 MB
    per column - well under the limit. Deployments with very long
    entity IDs (>1 KB) at high row counts may exceed it; switch to
    ``pa.large_string()`` is a future opt-in.
    """

    __slots__ = (
        "_base",
        "_buffers",
        "_c_bytes_by_schema",
        "_c_files_by_schema",
        "_compression",
        "_compression_level",
        "_date_cache",
        "_h_flush_cancelled",
        "_h_flush_err",
        "_h_flush_ok",
        "_include_msg_id",
        "_plans",
        "_tmp_dir",
    )

    def __init__(
        self,
        base: str | os.PathLike[str],
        *,
        include_msg_id: bool = True,
        compression: str = "zstd",
        compression_level: int = 3,
    ) -> None:
        self._base = pathlib.Path(base)
        self._tmp_dir = self._base / "_tmp"
        self._include_msg_id = include_msg_id
        self._compression = compression
        self._compression_level = compression_level
        self._buffers: dict[tuple[type[FeatureSchema], str], _Bucket] = {}
        self._plans: dict[type[FeatureSchema], _ArrowPlan] = {}
        self._date_cache: dict[int, str] = {}
        self._cleanup_tmp_dir()
        # Pre-warm metric label children at constructor time per the
        # Step 7 GC lesson — Histogram.labels(...) allocates a tuple
        # key + dict slot on first use.
        self._h_flush_ok = offline_flush_seconds.labels(outcome="ok")
        self._h_flush_err = offline_flush_seconds.labels(outcome="error")
        self._h_flush_cancelled = offline_flush_seconds.labels(outcome="cancelled")
        # Counter children are populated lazily on first append per
        # schema (schema set is unbounded but small in practice).
        self._c_files_by_schema: dict[type[FeatureSchema], Any] = {}
        self._c_bytes_by_schema: dict[type[FeatureSchema], Any] = {}

    # -- helpers -----------------------------------------------------------

    def _cleanup_tmp_dir(self) -> None:
        """Drop any orphaned tmp files from a prior process and ensure
        the tmp dir exists. Safe at every ``__init__`` because the
        ``_tmp`` dir is private to this writer."""
        shutil.rmtree(self._tmp_dir, ignore_errors=True)
        self._tmp_dir.mkdir(parents=True, exist_ok=True)

    def _get_date_str(self, ns: int) -> str:
        """``yyyy-mm-dd`` for ``event_time_ns``, day-quantum-cached.

        Uncached cost is ~3-5 µs (``datetime.fromtimestamp`` +
        ``strftime``). At 10k/s every append shares a day with the
        previous one ~99.99% of the time, so the cache amortises to
        ~30 ns/call.
        """
        day = ns // _DAY_NS
        cached = self._date_cache.get(day)
        if cached is not None:
            return cached
        s = datetime.fromtimestamp(ns / 1e9, tz=UTC).strftime("%Y-%m-%d")
        self._date_cache[day] = s
        return s

    def _make_bucket(self, plan: _ArrowPlan) -> _Bucket:
        """Allocate a fresh ``_Bucket`` and pre-resolve column-list refs.

        The ``columns`` dict and the ``wire_lists`` tuple share the same
        physical list objects; appending to one is visible via the
        other. This is what keeps the hot ``append`` path dict-lookup-
        free.
        """
        columns: dict[str, list[Any]] = {name: [] for name in plan.column_names}
        wire_lists = tuple(columns[plan.column_names[decl_idx]] for decl_idx in plan.wire_to_decl)
        return _Bucket(
            columns=columns,
            wire_lists=wire_lists,
            entity_id_col=columns["entity_id"],
            event_time_ns_col=columns["event_time_ns"],
            msg_id_ms_col=columns["msg_id_ms"] if plan.include_msg_id else None,
            msg_id_seq_col=columns["msg_id_seq"] if plan.include_msg_id else None,
        )

    # -- OfflineWriter Protocol --------------------------------------------

    async def append(
        self,
        schema: type[FeatureSchema],
        entity_id: str,
        event_time_ns: int,
        values_list: list[Any],
        msg_id: bytes,
    ) -> None:
        plan = self._plans.get(schema)
        if plan is None:
            plan = _arrow_plan_for(schema, include_msg_id=self._include_msg_id)
            self._plans[schema] = plan
            self._c_files_by_schema[schema] = offline_files_written_total.labels(
                schema=schema.__name__
            )
            self._c_bytes_by_schema[schema] = offline_bytes_written_total.labels(
                schema=schema.__name__
            )

        date_str = self._get_date_str(event_time_ns)
        key = (schema, date_str)
        bucket = self._buffers.get(key)
        if bucket is None:
            bucket = self._make_bucket(plan)
            self._buffers[key] = bucket

        # Validate length BEFORE any append. zip(..., strict=True) would
        # raise mid-loop, after partial appends had already mutated some
        # columns, leaving the bucket length-skewed and unrecoverable
        # (Rev-3 #2).
        if len(values_list) != len(bucket.wire_lists):
            raise ValueError(
                f"values_list length {len(values_list)} != "
                f"schema field count {len(bucket.wire_lists)}"
            )

        # Parse msg_id BEFORE any column append, for the same atomicity
        # reason. A malformed msg_id raises ValueError; if it raised
        # AFTER the appends below, entity_id/event_time/wire_lists would
        # be at +1 but msg_id_* at +0 (Rev-4 #2).
        msg_id_ms_val = 0
        msg_id_seq_val = 0
        msg_id_ms_col = bucket.msg_id_ms_col
        msg_id_seq_col = bucket.msg_id_seq_col
        if msg_id_ms_col is not None and msg_id_seq_col is not None:
            ms_b, _, seq_b = msg_id.partition(b"-")
            msg_id_ms_val = int(ms_b)
            msg_id_seq_val = int(seq_b)

        # Hot path from here on — zero dict lookups, all ops infallible.
        bucket.entity_id_col.append(entity_id)
        bucket.event_time_ns_col.append(event_time_ns)
        # strict=True is safe here — pre-loop length check above proves
        # the iterables are equal-length, and avoiding strict avoids the
        # mid-loop atomicity hazard of Rev-3 #2.
        for col_list, value in zip(bucket.wire_lists, values_list, strict=True):
            col_list.append(value)
        if msg_id_ms_col is not None and msg_id_seq_col is not None:
            msg_id_ms_col.append(msg_id_ms_val)
            msg_id_seq_col.append(msg_id_seq_val)

    async def flush(self) -> None:
        if not self._buffers:
            return
        # Filter empty buckets early; if every bucket in the snapshot is
        # empty (only failed appends since last flush — rare but
        # possible after a misconfigured producer), skip the metric
        # observation too. A "successful 0-row flush" is misleading in
        # dashboards (Rev-4 polish #2).
        snapshot = {k: v for k, v in self._buffers.items() if v.entity_id_col}
        # Reset BEFORE any I/O so cancellation has nothing in-flight to
        # clean up (ADR-010 §5).
        self._buffers = {}
        if not snapshot:
            return
        t0 = time.perf_counter()
        try:
            total_rows = 0
            for (schema, date_str), bucket in snapshot.items():
                plan = self._plans[schema]
                table = pa.Table.from_pydict(bucket.columns, schema=plan.arrow_schema)
                self._write_table(schema, date_str, table)
                self._c_files_by_schema[schema].inc()
                total_rows += len(bucket.entity_id_col)
            offline_flush_rows.observe(total_rows)
            self._h_flush_ok.observe(time.perf_counter() - t0)
        except asyncio.CancelledError:
            self._h_flush_cancelled.observe(time.perf_counter() - t0)
            raise
        except Exception:
            self._h_flush_err.observe(time.perf_counter() - t0)
            raise

    def _write_table(self, schema: type[FeatureSchema], date_str: str, table: Any) -> None:
        """Write one ``(schema, date)`` bucket to disk atomically.

        Sequence: ``pq.write_table`` to ``_tmp/{uuid}.parquet`` →
        ``stat().st_size`` → fsync file → ``os.rename`` into final
        partition dir → fsync parent dir. On any failure the tmp file
        is unlinked so ``_tmp/`` can't accumulate orphans across long-
        running deployments (Rev-4 polish #1).
        """
        partition_dir = self._base / f"schema={schema.__name__}" / f"event_date={date_str}"
        partition_dir.mkdir(parents=True, exist_ok=True)
        file_uuid = uuid.uuid4().hex
        tmp_path = self._tmp_dir / f"{file_uuid}.parquet"
        final_path = partition_dir / f"{file_uuid}.parquet"

        write_kwargs: dict[str, Any] = {
            "compression": self._compression,
            "compression_level": self._compression_level,
            "write_statistics": True,
        }
        if self._include_msg_id:
            # column_encoding ALONE is insufficient: PyArrow tries
            # dictionary encoding FIRST, and unique monotonic IDs make
            # the dictionary "fit" as a 1:1 mapping → the encoding hint
            # is silently dropped (Rev-3 #1). Disable dictionary on
            # the msg_id columns to force the delta path.
            #
            # PyArrow 14 enforces: when ``column_encoding`` is set,
            # ``use_dictionary`` must be either ``False`` (global) or a
            # LIST of column names to dict-encode (dict-form mapping
            # ``{col: bool}`` is rejected with "To use 'column_encoding'
            # set 'use_dictionary' to False"). The list form lets us
            # selectively keep dict on every column EXCEPT the two
            # msg_id ones — which is what we want, so entity_id and
            # any other repeated columns retain dict encoding.
            plan = self._plans[schema]
            write_kwargs["column_encoding"] = {
                "msg_id_ms": "DELTA_BINARY_PACKED",
                "msg_id_seq": "DELTA_BINARY_PACKED",
            }
            write_kwargs["use_dictionary"] = [
                name for name in plan.column_names if name not in ("msg_id_ms", "msg_id_seq")
            ]

        try:
            pq.write_table(table, tmp_path, **write_kwargs)
            size = tmp_path.stat().st_size
            self._c_bytes_by_schema[schema].inc(size)
            fd = os.open(tmp_path, os.O_RDONLY)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
            tmp_path.rename(final_path)
            dir_fd = os.open(partition_dir, os.O_RDONLY)
            try:
                os.fsync(dir_fd)
            finally:
                os.close(dir_fd)
        except BaseException:
            # Don't leak partial tmp files between init cleanups
            # (Rev-4 polish #1). missing_ok=True covers the case where
            # pq.write_table raised before the file was created.
            tmp_path.unlink(missing_ok=True)
            raise

    async def close(self) -> None:
        """Defensively drain any pending appends.

        Idempotent on empty buffers. The consumer's lifecycle awaits
        ``_flush_and_ack`` before invoking ``close()``, so the buffer
        is typically empty here — but direct (non-consumer) users get
        the ``file.close()``-style "drain on close" idiom.
        """
        await self.flush()
