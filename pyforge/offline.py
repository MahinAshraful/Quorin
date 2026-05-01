"""Parquet dataset store for the WAL consumer (Step 11) + reader (Step 12).

Implements :class:`ParquetDatasetStore`, an async ``OfflineWriter`` that
satisfies the Protocol declared in
:mod:`pyforge.wal_consumer`. Each ``flush()`` writes one Parquet file per
``(schema, event_date)`` bucket via tmp → fsync → atomic rename. Files
are hive-partitioned. Crash-safe by construction: a partial write either
sits in ``_tmp/`` (GC'd at next ``__init__``) or never exists.

Step 12 adds :meth:`ParquetDatasetStore.read_point_in_time` — a sync
point-in-time reader. Hand-rolls the asof-join primitive via
:func:`numpy.searchsorted` because PyArrow 14 does not expose
``Table.join_asof`` in Python (the C++ Acero ``AsofJoinNodeOptions``
exists but is unbound until PyArrow 16).

See [`progress/step11_plan.md`](progress/step11_plan.md) and
[`docs/adr/010-parquet-offline-store.md`](docs/adr/010-parquet-offline-store.md)
for the writer's full design lock; [`progress/step12_plan.md`](progress/step12_plan.md)
and [`docs/adr/011-point-in-time-reads.md`](docs/adr/011-point-in-time-reads.md)
for the reader's.
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

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds
import pyarrow.parquet as pq

from pyforge._internal.arrow_schema import _arrow_plan_for, _ArrowPlan
from pyforge.metrics import (
    offline_bytes_written_total,
    offline_files_written_total,
    offline_flush_rows,
    offline_flush_seconds,
    offline_read_rows,
    offline_read_seconds,
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
        "_h_read_err",
        "_h_read_ok",
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
        self._h_read_ok = offline_read_seconds.labels(outcome="ok")
        self._h_read_err = offline_read_seconds.labels(outcome="error")
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

    # -- Step 12: point-in-time reader --------------------------------------

    def _open_dataset(self, schema: type[FeatureSchema]) -> ds.Dataset | None:
        """Open the hive-partitioned dataset for ``schema``.

        Returns ``None`` if the schema directory does not exist OR
        exists but contains no Parquet fragments (writer-constructed-
        but-never-flushed case). Both are treated as "empty dataset"
        for the empty-result short-circuit. Genuine data corruption
        (``ArrowInvalid``) propagates as a bug signal — see ADR-011 §K.
        """
        schema_dir = self._base / f"schema={schema.__name__}"
        try:
            dataset = ds.dataset(schema_dir, format="parquet", partitioning="hive")
        except FileNotFoundError:
            return None
        # ``schema_dir`` exists but the writer never flushed any rows —
        # ``dataset.schema`` will be empty / inferred from nothing,
        # which would trip the schema-divergence guard with a
        # confusing message. Treat as empty.
        if not any(True for _ in dataset.get_fragments()):
            return None
        return dataset

    @staticmethod
    def _validate_dataset_uniform(dataset: ds.Dataset) -> None:
        """Surface a clearer error than ArrowInvalid for mixed include_msg_id state.

        Scans every fragment's physical_schema (verified at 65 ms on
        1000 fragments during pre-impl, well under the 100 ms
        threshold). If we see both presence and absence of
        ``msg_id_ms``/``msg_id_seq`` across the dataset, raise a
        message that points at the typical cause (flag flipped
        mid-deployment).
        """
        states: set[bool] = set()
        for frag in dataset.get_fragments():
            names = set(frag.physical_schema.names)
            states.add("msg_id_ms" in names and "msg_id_seq" in names)
            if len(states) > 1:
                raise ValueError(
                    "Dataset has files with mixed include_msg_id state "
                    "(some have msg_id_ms/msg_id_seq columns, some don't). "
                    "This typically happens when include_msg_id was flipped "
                    "mid-deployment without a migration. Use a separate base "
                    "directory per setting, or compact the dataset before "
                    "reading."
                )

    def read_point_in_time(
        self,
        schema: type[FeatureSchema],
        query_table: pa.Table,
        *,
        lookback_days: int = 30,
    ) -> pa.Table:
        """Return one feature row per query row, no future leakage, bounded staleness.

        For each ``(entity_id, as_of_time)`` pair in ``query_table``,
        return a row whose feature columns come from the most recent
        write where BOTH:

        - ``event_time_ns <= as_of_time`` (no future leak; inclusive
          at the boundary)
        - ``event_time_ns >= as_of_time - lookback_days * 86400e9``
          (within per-query lookback; inclusive at lower edge)

        If no feature row satisfies both, the feature columns are
        null. Result rows are returned in **query input order**; all
        ``query_table`` columns are preserved as the leading columns
        of the result.

        **Sync, by design.** Reads are 1-5 s of CPU + IO. Async callers
        must wrap in :func:`asyncio.to_thread` — an async signature
        with no awaits would mislead callers into believing the method
        cooperates with the loop. See ADR-011 §10.

        **Snapshot semantics.** Each call constructs a fresh
        :class:`pyarrow.dataset.Dataset`; concurrent ``flush()``
        activity may or may not be visible. Callers needing
        read-after-write consistency must serialize. ADR-011 §F.

        **Tested up to 10k queries.** Larger queries scale linearly
        via the Python inner loop (~µs/iter); for 100k+ queries
        consider sharding by entity_id across multiple calls.

        Parameters
        ----------
        schema:
            The :class:`FeatureSchema` subclass whose dataset partition
            (``base/schema={name}/``) will be read.
        query_table:
            Required columns: ``entity_id`` (``pa.string()``),
            ``as_of_time`` (``pa.int64()`` nanoseconds). Extra columns
            (``label``, ``fold``, etc.) are preserved in the result in
            input order. Column names must NOT collide with the result's
            feature columns (``event_time_ns``, schema field names,
            ``msg_id_ms``, ``msg_id_seq``).
        lookback_days:
            Per-query staleness ceiling AND partition-pruning window.
            Must be positive.

        Raises
        ------
        ValueError:
            On invalid lookback, missing required query columns,
            mismatched query column types, query column collision with
            result feature columns, schema-divergence vs the on-disk
            dataset (Step 15 evolution is the migration path), or
            mixed-mode dataset detection.
        """
        if lookback_days <= 0:
            raise ValueError(f"lookback_days must be positive, got {lookback_days}")

        needed = {"entity_id", "as_of_time"}
        missing = needed - set(query_table.column_names)
        if missing:
            raise ValueError(f"query_table missing required columns: {sorted(missing)}")

        # Type validation. Mismatched types (pa.large_string,
        # pa.timestamp, pa.int32) silently produce confusing
        # searchsorted behavior; fail fast at the boundary.
        eid_type = query_table.schema.field("entity_id").type
        if eid_type != pa.string():
            raise ValueError(f"query_table.entity_id must be pa.string(), got {eid_type}")
        aot_type = query_table.schema.field("as_of_time").type
        if aot_type != pa.int64():
            raise ValueError(
                f"query_table.as_of_time must be pa.int64() nanoseconds, got {aot_type}"
            )

        t0 = time.perf_counter()
        try:
            dataset = self._open_dataset(schema)
            if dataset is None:
                self._h_read_ok.observe(time.perf_counter() - t0)
                return _empty_result_table(
                    schema,
                    query_table,
                    include_msg_id=self._include_msg_id,
                )

            self._validate_dataset_uniform(dataset)

            dataset_names = set(dataset.schema.names)
            has_msg_id = "msg_id_ms" in dataset_names and "msg_id_seq" in dataset_names

            # Schema divergence guard. KeyError mid-asof would be
            # opaque; surface upfront with a Step 15 pointer.
            missing_fields = [f.name for f in schema.fields if f.name not in dataset_names]
            if missing_fields:
                raise ValueError(
                    f"Schema {schema.__name__!r} declares fields "
                    f"{missing_fields} that are not present in the "
                    f"on-disk dataset. This typically means the dataset "
                    f"was written with a previous schema version. Step "
                    f"15 schema evolution covers the migration path; "
                    f"for now, either roll back the schema or rebuild "
                    f"the dataset."
                )

            # Query column collision guard. entity_id is from the query
            # by contract (single-source); as_of_time is query-only;
            # the rest must not shadow result feature columns.
            forbidden = {"event_time_ns"} | {f.name for f in schema.fields}
            if has_msg_id:
                forbidden |= {"msg_id_ms", "msg_id_seq"}
            collisions = forbidden & set(query_table.column_names)
            if collisions:
                raise ValueError(
                    f"query_table column names {sorted(collisions)} "
                    f"collide with result feature columns. Rename them "
                    f"in query_table before reading."
                )

            if len(query_table) == 0:
                self._h_read_ok.observe(time.perf_counter() - t0)
                return _empty_result_table(schema, query_table, include_msg_id=has_msg_id)

            min_aot = pc.min(query_table["as_of_time"]).as_py()
            max_aot = pc.max(query_table["as_of_time"]).as_py()
            lookback_ns = lookback_days * _DAY_NS
            win_start_ns = min_aot - lookback_ns
            win_end_ns = max_aot

            # event_date is the hive partition column (string) — the
            # filter pushes down to file-listing skip. event_time_ns is
            # the row column — row-group statistics may also prune,
            # otherwise filtered post-load. The row filter is a
            # throughput optimization, not a correctness gate; per-
            # query lookback is enforced inside _asof_join.
            filter_expr = (
                (pc.field("event_date") >= _ns_to_date_str(win_start_ns))
                & (pc.field("event_date") <= _ns_to_date_str(win_end_ns))
                & (pc.field("event_time_ns") >= win_start_ns)
                & (pc.field("event_time_ns") <= win_end_ns)
            )
            features = dataset.to_table(filter=filter_expr)

            # Note: when ``features`` is empty after the filter, we do
            # NOT short-circuit to ``_empty_result_table`` — that helper
            # returns 0 rows, but the contract is "one row per query
            # row." ``_asof_join`` handles the empty-features case
            # cleanly: searchsorted on empty arrays returns 0 for every
            # query, all result_indices stay -1, and the nullable take
            # emits n_query rows of nulls.

            features = _dedup_features(features, has_msg_id=has_msg_id)
            result = _asof_join(
                query_table,
                features,
                schema=schema,
                has_msg_id=has_msg_id,
                lookback_ns=lookback_ns,
            )
            offline_read_rows.observe(len(result))
            self._h_read_ok.observe(time.perf_counter() - t0)
            return result

        except Exception:
            self._h_read_err.observe(time.perf_counter() - t0)
            raise

    # -- Step 13: hydration-side primitives --------------------------------

    def latest_features(
        self,
        schema: type[FeatureSchema],
        *,
        lookback_days: int = 30,
        as_of_time_ns: int | None = None,
    ) -> pa.Table:
        """Return the latest feature row per entity within ``lookback_days``.

        Result schema mirrors the source dataset's columns (``entity_id``,
        ``event_time_ns``, schema fields, optionally ``msg_id_*``). One row
        per entity that has at least one write within the window
        ``[snapshot_ns - lookback_days * 86400e9, snapshot_ns]`` where
        ``snapshot_ns = as_of_time_ns or time.time_ns()``.

        **Result row order is not guaranteed.** PyArrow's ``group_by`` output
        order is implementation-defined. Hydration's slot table is
        hash-indexed and order-agnostic; other callers needing stable order
        should sort post-call.

        Used by Step 13 hydration. For the general
        "as-of-arbitrary-timestamps per query row" case, use
        :meth:`read_point_in_time`. Why a separate primitive: hydration's
        question is simpler ("latest per entity within lookback"), so we
        skip ``read_point_in_time``'s per-query searchsorted machinery and
        Python inner loop. Saves ~3-4 s at 1M rows; eliminates the
        separate :meth:`distinct_entity_ids` scan that the
        ``read_point_in_time`` path would otherwise need.

        Parameters
        ----------
        schema:
            Schema whose dataset partition (``base/schema={name}/``) will
            be read.
        lookback_days:
            Per-query staleness ceiling AND partition-pruning window.
            Must be positive.
        as_of_time_ns:
            Snapshot timestamp; ``time.time_ns()`` if ``None``. Useful
            for deterministic tests and replaying past states.

        Raises
        ------
        ValueError:
            On invalid ``lookback_days``, schema-divergence vs the
            on-disk dataset (Step 15 evolution is the migration path),
            or mixed-mode dataset detection.
        """
        if lookback_days <= 0:
            raise ValueError(f"lookback_days must be positive, got {lookback_days}")

        t0 = time.perf_counter()
        try:
            dataset = self._open_dataset(schema)
            if dataset is None:
                self._h_read_ok.observe(time.perf_counter() - t0)
                return _empty_features_table(schema, include_msg_id=self._include_msg_id)
            self._validate_dataset_uniform(dataset)

            dataset_names = set(dataset.schema.names)
            has_msg_id = "msg_id_ms" in dataset_names and "msg_id_seq" in dataset_names

            # Schema divergence guard (mirrors read_point_in_time §L).
            missing = [f.name for f in schema.fields if f.name not in dataset_names]
            if missing:
                raise ValueError(
                    f"Schema {schema.__name__!r} declares fields {missing} "
                    f"that are not present in the dataset. Step 15 schema "
                    f"evolution is the migration path; rebuild or roll back."
                )

            snapshot_ns = as_of_time_ns if as_of_time_ns is not None else time.time_ns()
            win_start_ns = snapshot_ns - lookback_days * _DAY_NS
            win_end_ns = snapshot_ns
            # Upper bound is required for past-replay correctness: without
            # it, ``as_of_time_ns`` set to a past timestamp would leak future
            # writes via the unbounded filter.
            filter_expr = (
                (pc.field("event_date") >= _ns_to_date_str(win_start_ns))
                & (pc.field("event_date") <= _ns_to_date_str(win_end_ns))
                & (pc.field("event_time_ns") >= win_start_ns)
                & (pc.field("event_time_ns") <= win_end_ns)
            )
            features = dataset.to_table(filter=filter_expr)
            if len(features) == 0:
                self._h_read_ok.observe(time.perf_counter() - t0)
                return _empty_features_table(schema, include_msg_id=has_msg_id)

            # Sort so that __idx_max within an entity-group corresponds to
            # the row with max event_time_ns (with msg_id-extended tiebreak
            # for the backfill case — same determinism contract as
            # read_point_in_time).
            sort_keys: list[tuple[str, str]] = [
                ("entity_id", "ascending"),
                ("event_time_ns", "ascending"),
            ]
            if has_msg_id:
                sort_keys.extend(
                    [
                        ("msg_id_ms", "ascending"),
                        ("msg_id_seq", "ascending"),
                    ]
                )
            features = features.sort_by(sort_keys)
            features = features.append_column(
                "__idx", pa.array(np.arange(len(features), dtype=np.int64))
            )
            grouped = features.group_by("entity_id").aggregate([("__idx", "max")])
            result = features.take(grouped.column("__idx_max")).drop_columns(["__idx"])

            offline_read_rows.observe(len(result))
            self._h_read_ok.observe(time.perf_counter() - t0)
            return result
        except Exception:
            self._h_read_err.observe(time.perf_counter() - t0)
            raise

    def distinct_entity_ids(self, schema: type[FeatureSchema]) -> pa.Array:
        """Return the unique entity IDs across the entire dataset for ``schema``.

        Reads only the ``entity_id`` column via PyArrow's column projection;
        deduplicates via ``pc.unique`` (C-vectorized over the dictionary
        column). Empty array if the schema directory does not exist.

        Memory note: at 100M rows, the entity_id column scan loads ~4 GB
        into memory (UUID strings x dictionary overhead). Sharding by
        schema is the escape hatch for very-large datasets.

        Hydration (Step 13) does NOT use this method — it uses
        :meth:`latest_features`, where ``group_by`` emits unique entities
        as a side effect (one fewer dataset scan). This method ships as a
        public API for ad-hoc operator queries.
        """
        dataset = self._open_dataset(schema)
        if dataset is None:
            return pa.array([], type=pa.string())
        self._validate_dataset_uniform(dataset)
        table = dataset.to_table(columns=["entity_id"])
        return pc.unique(table["entity_id"])


# ---------------------------------------------------------------------------
# Step 12 module-level helpers — pure functions, no instance state.
# ---------------------------------------------------------------------------


def _ns_to_date_str(ns: int) -> str:
    """``yyyy-mm-dd`` for the partition-prune filter.

    Not day-quantum-cached: this is called twice per
    ``read_point_in_time`` call (window start + end), not on a hot
    loop, so the per-call ~3 us cost is amortized across the multi-
    second read.
    """
    return datetime.fromtimestamp(ns / 1e9, tz=UTC).strftime("%Y-%m-%d")


def _dedup_features(table: pa.Table, *, has_msg_id: bool) -> pa.Table:
    """Drop duplicate rows by ``(msg_id_ms, msg_id_seq)`` (or natural key fallback).

    When ``has_msg_id``, crash-replay duplicates are byte-identical
    (the producer wrote the same payload both times into the WAL); any
    tiebreaker is correct, ``min`` is the cheapest pyarrow
    ``group_by`` aggregation. When falling back to
    ``(entity_id, event_time_ns)``, ``max`` (last-row-wins) keeps the
    most-recently-loaded row; users opting into ``include_msg_id=False``
    accept that same-event_time conflation is silent.
    """
    n = len(table)
    if n == 0:
        return table
    table = table.append_column("__idx", pa.array(np.arange(n, dtype=np.int64)))
    if has_msg_id:
        keys = ["msg_id_ms", "msg_id_seq"]
        agg_func = "min"
    else:
        keys = ["entity_id", "event_time_ns"]
        agg_func = "max"
    grouped = table.group_by(keys).aggregate([("__idx", agg_func)])
    indices = grouped.column(f"__idx_{agg_func}")
    return table.take(indices).drop_columns(["__idx"])


def _asof_join(
    query: pa.Table,
    features: pa.Table,
    *,
    schema: type[FeatureSchema],
    has_msg_id: bool,
    lookback_ns: int,
) -> pa.Table:
    """Hand-rolled per-entity asof join via ``numpy.searchsorted``.

    PyArrow 14 doesn't expose ``Table.join_asof``; this is the
    primitive. See ADR-011 §11.

    The per-query lookback check inside the inner loop is THE Rev-3
    CRITICAL-1 fix — the global row filter loads features in
    ``[min(as_of_time) - lookback, max(as_of_time)]`` for throughput,
    which is wider than each individual query's window when the query
    spans multiple as_of_times. Without this check, a feature loaded
    only because some other query's window covered it can match a
    query whose own window ends earlier.
    """
    n_query = len(query)

    # Sort with msg_id keys (when present) for deterministic asof
    # tiebreaks. Backfill duplicates same-(entity_id, event_time_ns)
    # by distinct msg_id are NOT crash-replay (ADR-010 §1); without
    # the msg_id sort keys, take() picks whichever order PyArrow's
    # take saw — non-deterministic across reads.
    sort_keys: list[tuple[str, str]] = [
        ("entity_id", "ascending"),
        ("event_time_ns", "ascending"),
    ]
    if has_msg_id:
        sort_keys.extend(
            [
                ("msg_id_ms", "ascending"),
                ("msg_id_seq", "ascending"),
            ]
        )
    features_sorted = features.sort_by(sort_keys)

    eid_arr = features_sorted["entity_id"].to_numpy(zero_copy_only=False)
    et_arr = features_sorted["event_time_ns"].to_numpy()
    q_eid = query["entity_id"].to_numpy(zero_copy_only=False)
    q_aot = query["as_of_time"].to_numpy()

    lo_arr = np.searchsorted(eid_arr, q_eid, side="left")
    hi_arr = np.searchsorted(eid_arr, q_eid, side="right")

    result_indices = np.full(n_query, -1, dtype=np.int64)
    for i in range(n_query):
        lo = int(lo_arr[i])
        hi = int(hi_arr[i])
        if lo == hi:
            continue  # entity not in features
        # side="right" makes the upper bound INCLUSIVE
        # (event_time_ns == as_of_time matches).
        pos = int(np.searchsorted(et_arr[lo:hi], q_aot[i], side="right"))
        if pos == 0:
            continue  # all event_times in slice > as_of_time
        candidate = lo + pos - 1
        # Per-query lookback check (Rev-3 CRITICAL-1).
        if int(et_arr[candidate]) < int(q_aot[i]) - lookback_ns:
            continue  # candidate older than per-query lookback
        result_indices[i] = candidate

    matched_mask = result_indices >= 0
    safe_indices = np.where(matched_mask, result_indices, 0).astype(np.int64)

    # Single take with nullable indices — PyArrow emits null at masked
    # positions (verified empirically pre-impl on PyArrow 14.0.2). No
    # per-column pc.if_else needed.
    indices_with_nulls = pa.array(safe_indices, type=pa.int64(), mask=~matched_mask)
    features_aligned = features_sorted.take(indices_with_nulls)

    # Assemble per ADR-011 §J: query columns first (in input order),
    # then features columns. entity_id is single-source (from query);
    # we don't include features.entity_id since it equals query's when
    # matched and is null when not.
    out_cols: dict[str, Any] = {col_name: query[col_name] for col_name in query.column_names}
    out_cols["event_time_ns"] = features_aligned["event_time_ns"]
    for f in schema.fields:
        out_cols[f.name] = features_aligned[f.name]
    if has_msg_id:
        out_cols["msg_id_ms"] = features_aligned["msg_id_ms"]
        out_cols["msg_id_seq"] = features_aligned["msg_id_seq"]
    return pa.table(out_cols)


def _empty_result_table(
    schema: type[FeatureSchema],
    query_table: pa.Table,
    *,
    include_msg_id: bool,
) -> pa.Table:
    """Build a "no features matched" result Table — one row per query row.

    Used by the dataset-missing short-circuit. Returns
    ``len(query_table)`` rows: query columns pass through unchanged;
    feature columns are null arrays of length ``len(query_table)``.

    When ``query_table`` itself is empty, this produces a 0-row result
    naturally (no special case). The post-filter "zero features" case
    is handled inside :func:`_asof_join` rather than via this helper —
    nullable take on an empty source emits the right shape.
    """
    plan = _arrow_plan_for(schema, include_msg_id=include_msg_id)
    n = len(query_table)
    out_cols: dict[str, pa.Array | pa.ChunkedArray] = {
        col_name: query_table[col_name] for col_name in query_table.column_names
    }
    arrow_schema = plan.arrow_schema
    out_cols["event_time_ns"] = pa.nulls(n, type=pa.int64())
    for f in schema.fields:
        out_cols[f.name] = pa.nulls(n, type=arrow_schema.field(f.name).type)
    if include_msg_id:
        out_cols["msg_id_ms"] = pa.nulls(n, type=pa.int64())
        out_cols["msg_id_seq"] = pa.nulls(n, type=pa.int32())
    return pa.table(out_cols)


def _empty_features_table(
    schema: type[FeatureSchema],
    *,
    include_msg_id: bool,
) -> pa.Table:
    """Build a 0-row "features-only" Table matching the dataset schema.

    Used by :meth:`ParquetDatasetStore.latest_features` for the
    dataset-missing and post-filter-zero-rows cases. Unlike
    :func:`_empty_result_table` (which preserves query columns from a
    point-in-time read's ``query_table``), this helper has no query
    context — it just emits the dataset's natural columns at zero
    length.
    """
    plan = _arrow_plan_for(schema, include_msg_id=include_msg_id)
    arrow_schema = plan.arrow_schema
    out_cols: dict[str, pa.Array] = {
        "entity_id": pa.array([], type=pa.string()),
        "event_time_ns": pa.array([], type=pa.int64()),
    }
    for f in schema.fields:
        out_cols[f.name] = pa.array([], type=arrow_schema.field(f.name).type)
    if include_msg_id:
        out_cols["msg_id_ms"] = pa.array([], type=pa.int64())
        out_cols["msg_id_seq"] = pa.array([], type=pa.int32())
    return pa.table(out_cols)
