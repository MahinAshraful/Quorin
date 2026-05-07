# ADR-010: Parquet offline store — file-per-flush, hive-partitioned, atomic-rename, dumb-writer

**Status:** Accepted
**Date:** 2026-04-30
**Step:** 11 (Parquet dataset store)

## Decision

Quorin ships [`quorin.offline.ParquetDatasetStore`](../../quorin/offline.py),
an async `OfflineWriter` (per ADR-009's Protocol) that persists every
WAL message to a hive-partitioned Parquet dataset on local disk. Each
`flush()` call writes one file per `(schema, event_date)` bucket via
write-to-`_tmp` → fsync → atomic rename → fsync-parent-dir. Crash-safe
by construction; readers see only fully written files.

Public API:

```python
class ParquetDatasetStore:
    def __init__(
        self,
        base: str | os.PathLike[str],
        *,
        include_msg_id: bool = True,
        compression: str = "zstd",
        compression_level: int = 3,
    ) -> None: ...

    async def append(self, schema, entity_id, event_time_ns, values_list, msg_id) -> None: ...
    async def flush(self) -> None: ...
    async def close(self) -> None: ...
```

The 13 numbered sections below lock the design end to end.

---

## 1. `msg_id` decomposed into `(int64, int32)` columns with explicit DELTA + dictionary opt-out

Each row carries the full Redis-stream `msg_id` it originated from,
decomposed: `msg_id_ms: int64` (the millisecond timestamp portion) +
`msg_id_seq: int32` (the per-millisecond sequence). Step 12's reader
uses these columns for exact dedup; reconstruction back to the original
Redis ID is `f"{ms}-{seq}".encode()`.

Three rejected alternatives:

- **(b) dedup on `(entity_id, event_time_ns)` natural key.** Same-
  `event_time_ns` writes for the same entity are *legal* in the
  producer (backfills using `today_midnight_ns` for everything,
  synthetic test timestamps, second-resolution clocks). Natural-key
  dedup silently conflates "crash-replay duplicate" with "intentional
  distinct write" and loses real data. Correctness bug.
- **(c) in-memory dedup at flush + compaction job.** Splits the dedup
  invariant across writer state + a daemon + the reader. Operational
  debt; a new daemon to monitor / restart / alert on. Step 11's
  "simplest correct thing" remit rejects it.
- **store the raw bytes.** ~10 bytes/row vs ~3 bytes after
  `DELTA_BINARY_PACKED` + zstd. Less compelling: ~0.024% savings at
  200 fields. The deciding factor is dedup speed (int columns dedup
  2-3× faster than string in PyArrow).

**Two encoding hints are required, not one — and PyArrow 14 narrows
the API.** The naive `column_encoding={"msg_id_*": "DELTA_BINARY_PACKED"}`
is silently dropped: the encoding pipeline tries dictionary FIRST, and
unique monotonic IDs make the dictionary "fit" as a 1:1 mapping, so
the hint is bypassed. The writer must also pass `use_dictionary` —
but PyArrow 14 rejects the per-column-mapping form
(`use_dictionary={"msg_id_ms": False}`) when `column_encoding` is set,
with `ValueError: To use 'column_encoding' set 'use_dictionary' to
False`. The dict form is reserved for the no-`column_encoding` path.

The working configuration is the **list form**:
`use_dictionary=[name for name in plan.column_names if name not in
("msg_id_ms", "msg_id_seq")]`. Listed columns get dict encoding;
unlisted columns get plain. Combined with the encoding hint, msg_id_*
gets the delta path while `entity_id` (and other repeating columns)
keeps dict encoding. The metadata-inspection regression tests in
[`tests/unit/test_offline_writer.py`](../../tests/unit/test_offline_writer.py)
assert (a) `DELTA_BINARY_PACKED` is present on msg_id_*, (b)
`PLAIN_DICTIONARY` (or `RLE_DICTIONARY`) is absent on msg_id_*, AND
(c) dictionary encoding survives on `entity_id` when the column has
repetition. Without the list form, either the encoding choice flips
silently OR every column loses dict encoding.

## 2. Writer is dumb. Dedup is at read time.

`append()` writes every row including `msg_id_*` columns and moves on.
No pre-flush dedup; no in-memory hash set. The write path is throughput-
critical (10k/sec steady, 50k/sec peak target); spending cycles on
dedup at write time is the wrong place. Step 12's
`quorin.offline.read(...)` provides dedup-on-`msg_id` at read time —
reads are user-driven and can absorb the millisecond.

## 3. File-per-flush, hive-style partitioning

Each `flush()` call writes one Parquet file per `(schema, event_date)`
bucket. Path layout:

```
{base}/
  _tmp/{uuid}.parquet            # in-flight; cleaned at __init__
  schema={schema_name}/
    event_date=2026-04-29/
      {uuid}.parquet              # sealed; readers see this only
```

Hive-style (`key=value` segments) wins over positional because (a) it
is self-documenting — DuckDB / Spark / Trino / pyarrow.dataset read it
natively without configuration; (b) Step 12's reader pseudocode in
the build plan uses `partitioning="hive"`, so this resolves an
internal inconsistency in the build plan (Step 11's pseudocode had
positional). Operationally identical for Quorin's own reader; better
for ad-hoc DuckDB queries against the dataset.

## 4. Atomic write via tmp + fsync + rename + parent-dir fsync

Sequence in [`_write_table`](../../quorin/offline.py):

1. `pq.write_table(table, _tmp/{uuid}.parquet, ...)`.
2. `tmp_path.stat().st_size` — record bytes for
   `offline_bytes_written_total{schema}`.
3. `os.fsync(file_fd)`.
4. `os.rename(tmp_path, partition_dir / file_name)` — atomic on Linux
   same-FS.
5. `os.fsync(parent_dir_fd)` so the rename is itself durable.

**On any failure, the tmp file is unlinked.** Wrapping the body in
`try / except: tmp_path.unlink(missing_ok=True); raise` prevents
`_tmp/` orphan accumulation in long-running deployments where
`pq.write_table` occasionally raises (disk pressure, OOM during
conversion). At init, `_cleanup_tmp_dir()` runs `shutil.rmtree(_tmp,
ignore_errors=True)` + `mkdir` as belt-and-suspenders for any orphans
the per-call cleanup missed (e.g. SIGKILL between `pq.write_table`
and the `unlink`).

`base` and `_tmp` must be on the same filesystem so `os.rename` is
atomic. Crossing a mount point raises `OSError(EXDEV)`. Documented in
the class docstring.

## 5. Cancellation drops buffers; PEL replay restores

`flush()` snapshots `_buffers` to a local variable and resets the
attribute to `{}` *before* any I/O. From that point on, the snapshot
is local-only; cancellation drops it. The buffered rows correspond to
PEL entries that never XACK'd (ADR-009 §3); restart's PEL drain
re-applies them.

Tmp files left by an interrupted write get cleaned up by
`_cleanup_tmp_dir()` on next `__init__`. Half-completed flush
(bucket 1's file on disk, bucket 2 lost mid-loop) is acceptable
because Step 12's read-time dedup absorbs the bucket-1 duplicates.

Adjacent to atomic-write (§4) because they're two halves of the same
crash-safety story: §4 ensures sealed files are durable; §5 ensures
in-flight buffers don't survive cancellation as half-state.

## 6. `_Bucket` dataclass with pre-resolved column-list refs and atomic-fail validation

The hot `append` path is dict-lookup-free. Each `(schema, date_str)`
pair maps to a `_Bucket` dataclass that holds:

- `columns: dict[str, list]` — for `pa.Table.from_pydict` at flush time.
- `wire_lists: tuple[list, ...]` — parallel to the schema's
  name-hash-sorted wire order.
- Direct attribute aliases: `entity_id_col`, `event_time_ns_col`,
  `msg_id_ms_col`, `msg_id_seq_col`.

The dict and the tuple share the same physical list objects (lists
are mutable references); appending to one is visible via the other.
The dict-to-tuple resolution happens once per `_make_bucket` call,
not per append. At 200 fields, eliminates ~16 µs of pure dict-lookup
overhead per append (Rev-3 #1).

**Atomic fail on any pre-append validation:** length check
(`len(values_list) == len(bucket.wire_lists)`) AND `msg_id` parse
(`int(ms_b)`, `int(seq_b)`) both run before any column is mutated.
Without this, `zip(strict=True)` mid-loop or `int()` mid-append
would leave the bucket length-skewed (entity_id/event_time/wire_lists
at +1 but msg_id_* at +0, or vice versa), corrupting the bucket
permanently and surfacing as opaque `from_pydict` errors at flush
time. Same bug class as Rev-3 #2 (`zip(strict=True)`) and Rev-4 #2
(msg_id parse-after-append).

**Empty buckets are skipped at flush.** A `_make_bucket` call
followed by an `append` that raises leaves an empty bucket in
`_buffers`. The flush loop filters `if not bucket.entity_id_col:
continue` and short-circuits the metric observation if every
bucket is empty — a "successful 0-row flush" is misleading in
dashboards.

## 7. Day-quantum cache for `date_str`

`datetime.fromtimestamp(ns / 1e9, tz=UTC).strftime("%Y-%m-%d")` is
**3-5 µs per call** (float div + `_PyTime_FromSecondsObject` +
datetime alloc + strftime format parsing). At 10k/sec that's
30-50 ms/flush of pure date-formatting waste — bigger than the
zstd budget.

Cache by day-quantum on the store instance:

```python
_DAY_NS = 86_400_000_000_000

def _get_date_str(self, ns: int) -> str:
    day = ns // _DAY_NS
    cached = self._date_cache.get(day)
    if cached is not None:
        return cached
    s = datetime.fromtimestamp(ns / 1e9, tz=timezone.utc).strftime("%Y-%m-%d")
    self._date_cache[day] = s
    return s
```

Hit cost: ~30 ns. Within a single flush window virtually all messages
share the same day → 1 miss + 9999 hits. Cache size is bounded by
deployment retention (a few thousand entries lifetime); no eviction.

## 8. Synchronous `pq.write_table` blocks the event loop

Day-1 native flush p99 is **~100-150 ms** at 10k rows × 200 fields.
Measured on WSL2 + Docker Desktop:

| Bench | Median | Max |
|---|---|---|
| `flush_10k_rows_50_field` | **30.5 ms** | 65.8 ms |
| `flush_10k_rows_200_field` | **114 ms** | 123 ms |
| `flush_10k_rows_200_field_no_msg_id` | 128.9 ms | 199.3 ms |

Phase breakdown (estimated from related Step 8 / Step 10 measurements):

| Phase | Cost |
|---|---|
| `pa.Table.from_pydict` (Python→Arrow conversion) | ~80-150 ms |
| zstd compression | ~30 ms |
| fsync (file + parent dir) | ~5-15 ms native, +20-100 ms WSL2 |

**Synchronous flush blocks the asyncio event loop for the full
duration.** asyncio is single-threaded and only schedules other
coroutines at `await` points; synchronous Python/C work like
`pa.Table.from_pydict` and `pq.write_table` holds the loop for the
full ~100-150 ms native (~200+ ms WSL2). PyArrow may release the GIL
during C-level zstd compression, but a released GIL helps other OS
threads, not other coroutines in the same loop. The read coroutine
in the consumer does NOT progress during flush.

Operationally this means:

1. Flush coroutine starts → event loop is blocked.
2. Read coroutine cannot run; Redis buffers incoming messages on the
   stream side (XADD is fire-and-forget).
3. Flush completes → control returns to the loop → next XREADGROUP
   call returns a larger-than-usual batch.
4. Steady-state throughput is preserved (catch-up is fast);
   per-message processing latency exhibits a **~100-200 ms spike
   on each flush boundary**.

**This is the documented baseline, not a regression.** Tuning
`max_pending_ack` becomes a three-axis tradeoff:

| Direction | Latency spike | Memory | Recovery |
|---|---|---|---|
| Smaller `max_pending_ack` | smaller spike | less memory | faster |
| Larger `max_pending_ack` | larger spike | more memory | slower |

`run_in_executor` would only release the loop during the C-level
zstd phase (~30 ms of the ~100-150 ms total) — the Python→Arrow
conversion phase holds the GIL throughout. Net win is small at the
cost of buffer thread-safety considerations. Not worth it for v1.
**Trigger to revisit: back-pressure events (the consumer's
`2 × max_pending_ack` ceiling firing) OR
`wal_consumer_flush_seconds` p99 > 1s in production (real Linux).**
100-150 ms is the baseline, not the trigger.

Bench gates lock to **4× measured WSL2 baseline** in
[`benchmarks/regression/thresholds.yml`](../../benchmarks/regression/thresholds.yml),
not 4× theoretical, because Docker Desktop fsync via 9P/virtio-fs to
NTFS is ~10× slower than native ext4. The verification step measures
on WSL2 first, then sets gates; the placeholders in `thresholds.yml`
get replaced with measured numbers and recorded back here.

### Future optimizations (deferred, listed for the Step 16 flamegraph debate)

- **Per-column numpy-array buffers for scalar fields.** Replaces
  `list[Any]` in `_Bucket` with sized numpy arrays; eliminates
  `from_pydict`'s Python→Arrow boxing on hot scalar columns.
  Estimated 10× speedup on flush conversion at the cost of
  preallocate-and-grow complexity. Trigger: Step 16 flamegraph
  shows `from_pydict` conversion dominating flush.
- **`pa.binary()` for `entity_id`.** Skips the consumer-decodes-
  then-writer-re-encodes UTF-8 round-trip (~400 ns/row, ~4 ms/sec
  at 10k rps). Loses UTF-8 validation and changes reader semantics —
  probably not worth it.
- **Tuple-key cache for `_buffers`.** Cache the most-recent
  `(schema, date_str) → bucket` on the store; skips ~50-80 ns/append
  of tuple allocation + dict lookup in the common case (same schema,
  same date repeated). Marginal; only if flamegraph identifies it.

## 9. `include_msg_id=False` is opt-in for byte-conscious users

For users who trust their producers' time-distinctness invariant and
want the bytes back, `include_msg_id=False` omits the two `msg_id_*`
columns. Step 12's reader detects the absence and falls back to
`(entity_id, event_time_ns)` natural-key dedup. The trade-off (silent
data conflation on same-`event_time_ns` writes) is the user's to
make, not the default. Measured byte savings will be recorded here
once the `flush_10k_rows_200_field_no_msg_id` bench (not gated)
runs against `flush_10k_rows_200_field`.

**File-count growth at sustained throughput** (compaction is
operator-owned for v1): at 10k/sec with `flush_interval_seconds=60`,
~1 file/min/schema = 1440 files/day = 525k files/year per schema.
Most readers (DuckDB, Spark, Trino) handle this via dataset metadata
caching, but it's a real operational concern. Threshold worth
tracking: ~100k files/schema, or readers showing >10% of query time
in metadata loading. Adding a compaction job at that point is
straightforward; v1 doesn't ship one.

## 10. `close()` defensively calls `flush()`

`close()` runs `await self.flush()` before returning. The consumer's
lifecycle awaits `_flush_and_ack` before invoking `close()`, so the
buffer is typically empty here — `flush()` early-returns and `close()`
is effectively a no-op in that path. But `ParquetDatasetStore` may
also be constructed directly (tests, ad-hoc scripts, future use cases
without a WAL consumer); for those callers the Python `file.close()`
idiom of "flush whatever's pending" is what one expects from reading
the source. Idempotent on empty buffers; cheap; closes a footgun
where a reader of the source would otherwise have to know the
consumer's lifecycle to know whether `close()` is safe.

## 11. `pa.string()` for `entity_id`

32-bit offsets, 2 GB per column per file. At 36-char UUIDs × 50k
rows = 1.8 MB per column — under the limit by 1000×. Deployments
with very long entity IDs (>1 KB) at high row counts may exceed it;
switch to `pa.large_string()` is a future opt-in. The class
docstring documents the limit; v1 doesn't auto-detect.

## 12. Memory contract

Writer peak buffer scales linearly with the consumer's
`max_pending_ack × n_fields × ~50 B` Python overhead. At default
`max_pending_ack=10_000` and 200-field schemas, expect ~88 MB peak
per writer; at the build-plan 50k peak, ~440 MB. Operators tuning
`max_pending_ack` for recovery SLA must size memory accordingly.
Documented in the class docstring and re-stated in §8's three-axis
tradeoff.

## 13. Out of scope

- **Read path** (`pa.dataset` + `asof_join` + dedup-on-msg_id) — Step 12.
- **Hydration** — cold-start backfill from Parquet into shm — Step 13.
- **Periodic compaction** of small files — operator-owned for v1; see §9.
- **Multi-stream sharding** (`quorin:wal:{i}`) — deployment-layer; future.
- **S3 / object-storage backend** — `pathlib.Path` is local-FS-only by design.
- **Per-column dictionary tuning for non-msg_id columns** — let PyArrow
  default hold; revisit if Step 16 flamegraphs identify it. (msg_id
  columns are explicitly `use_dictionary=False`; that's load-bearing,
  not optional — see §1.)
- **`run_in_executor` for `pq.write_table`** — see §8's trigger.
- **`pa.large_string()` for entity_id** — see §11.
- **Per-column numpy-array buffers, `pa.binary()` for entity_id,
  tuple-key cache for `_buffers`** — see §8's future-opts list.

---

## Consequences

- Step 12's reader sees a dataset whose partition keys are
  self-describing (`schema=*/event_date=*/`), files are individually
  fully written, and `msg_id_*` columns are present-by-default for
  fast int-based dedup.
- Step 14's watchdog has nothing to do for the offline path —
  Parquet files don't need lifecycle management.
- Step 15's schema evolution must coexist with file-per-flush:
  schema-version-N files and schema-version-(N+1) files coexist in
  the same `schema={name}/` partition tree. Reader handles
  migration; writer just writes its current schema's columns.
- File-count growth at 525k/year/schema is operator-monitored, not
  quorin-managed. v1 ships no compaction.
- The latency-spike-per-flush baseline (~100-200 ms) is documented
  for downstream alerting setup. Operators alert on >1s, not on
  the baseline.
