# ADR-011: Point-in-time reads — hand-rolled per-entity asof, leak-free, snapshot semantics

**Status:** Accepted
**Date:** 2026-04-30
**Step:** 12 (Point-in-Time Reads)

## Decision

Pyforge ships [`pyforge.offline.ParquetDatasetStore.read_point_in_time`](../../pyforge/offline.py),
a sync method that returns one feature row per query row, with feature
columns null where no feature satisfies BOTH:

- `event_time_ns <= as_of_time` (no future leak; inclusive at boundary)
- `event_time_ns >= as_of_time - lookback_days * _DAY_NS` (per-query
  staleness ceiling; inclusive at lower edge)

Public API:

```python
class ParquetDatasetStore:
    def read_point_in_time(
        self,
        schema: type[FeatureSchema],
        query_table: pa.Table,           # required: entity_id, as_of_time
        *,
        lookback_days: int = 30,
    ) -> pa.Table: ...
```

The 15 numbered sections below lock the design end to end.

---

## 1. `(msg_id_ms, msg_id_seq)` is the dedup primary key

When the dataset has both `msg_id_*` columns, dedup uses
`group_by(["msg_id_ms", "msg_id_seq"]).aggregate([("__idx", "min")])` —
first-row-wins. Crash-replay duplicates are byte-identical (the
producer wrote the same payload both times into the WAL, ADR-009 §2);
any tiebreaker is correct, `min` is the cheapest pyarrow aggregation.

Rejected alternatives:

- **Natural-key dedup as primary** — same-`event_time_ns` writes for
  the same entity are *legal* (backfills using `today_midnight_ns`,
  second-resolution clocks). Conflates "crash replay" with
  "intentional distinct write" → loses real data. ADR-010 §1 was
  explicit.
- **Packed-int64 key (`ms<<24 | seq`) + `pc.unique`** — works while
  `seq < 2^24`. Recorded as the v2 escape hatch if the benchmark gate
  fails; only triggers if real Redis stream sequences ever exceed
  ~16 M per ms (~16k×/µs, well above realistic loads).

## 2. `(entity_id, event_time_ns)` natural-key fallback for `include_msg_id=False`

When the dataset's unified schema lacks both `msg_id_*` columns, dedup
uses `group_by(["entity_id", "event_time_ns"]).aggregate([("__idx", "max")])` —
last-row-wins. Non-deterministic across calls because
`pa.dataset.to_table()` reads files in filesystem-listing order, which
is not stable. Documented as the trade-off `include_msg_id=False`
users opt into (ADR-010 §9).

For determinism, users opt into `include_msg_id=True` (the default).

## 3. Mixed-mode datasets are an error; Step 11 footgun acknowledgment

A dataset where some files have `msg_id_*` and some don't is treated
as corrupted state. `_validate_dataset_uniform` scans every fragment's
`physical_schema` (verified at 65 ms on 1000 fragments during pre-impl
sanity check, well under the 100 ms threshold from the plan; 32-cap
dropped). On the first divergence, raises a `ValueError` with a clear
message identifying the typical cause.

PyArrow's natural error path is a deep-stack `ArrowInvalid` from
`to_table()`; the upfront check surfaces a more actionable message.

**Step 11 footgun acknowledgment.** Flipping `include_msg_id`
mid-deployment without migrating existing files produces a
mixed-schema dataset that this reader rejects upfront. The writer
does NOT validate this; Step 12's reader is the only place the
inconsistency surfaces. **Known gap** — if a Step 11.1 follow-up is
desired, the right place is writer-side validation in
`ParquetDatasetStore.__init__` that scans existing files in
`base/schema=*/` and raises if `include_msg_id` mismatches the
constructor flag. Deferred; Step 12 doesn't add writer-side
validation, only documents the reader-side error.

## 4. `lookback_days = 30` default; conflates partition prune + max staleness

Build-plan default. Acts as **both** the partition-pruning bound
(skips old hive partitions; throughput optimization) AND the
max-staleness ceiling (per-query asof tolerance). Conflating them is
the right move: every workload that wants more partition data also
wants more staleness tolerance.

`lookback_days <= 0` raises `ValueError`. Stale-feature workloads
override (`lookback_days=180` etc.).

## 5. Partition pruning on `event_date` STRING column, not `event_time_ns`

`pa.dataset.dataset(..., partitioning="hive")` exposes `event_date` as
a string column inferred from the directory name. Filters on
`event_date` push down to file-listing skip; filters on
`event_time_ns` push down only to per-file row-group skip (much
weaker).

The reader's filter expression is:

```python
filter_expr = (
    (pc.field("event_date") >= window_start_date_str)
    & (pc.field("event_date") <= window_end_date_str)
    & (pc.field("event_time_ns") >= window_start_ns)
    & (pc.field("event_time_ns") <= window_end_ns)
)
```

Lex compare on `yyyy-mm-dd` strings is correct (zero-padded fields).

## 6. Belt-and-suspenders row filter on `event_time_ns`

The `event_time_ns` halves of the filter expression tighten to the
exact ns window for sub-day precision. Row-group statistics typically
prune this for free; otherwise rows are filtered post-load. Without
this, partition pruning would over-include rows from boundary days
(rows with the same `event_date` partition but `event_time_ns`
outside the window).

## 7. Sort done internally on a copy; deterministic asof tiebreaks

`features.sort_by([eid, et, msg_id_ms, msg_id_seq])` happens inside
`_asof_join` (PyArrow doesn't expose in-place sort; `take` after
`sort_indices` produces an equivalent copy). The msg_id keys are
**load-bearing for determinism**: backfill writes with same
`(entity_id, event_time_ns)` but distinct msg_id are NOT crash-replay
(ADR-010 §1) — two reads of the same dataset would otherwise pick
arbitrary tiebreaks via PyArrow's take order.

The result preserves **query input order** via index-based scatter:
the asof primitive produces `result_indices: np.int64[len(query)]`
keyed by query position; `take(indices_with_nulls)` materializes in
that order. Callers that pre-sort their query for caching or
batching see their order preserved.

## 8. Snapshot semantics — fresh `pa.dataset` per call

Each `read_point_in_time` call constructs a fresh
`pa.dataset.dataset(...)` reflecting the directory listing at function
entry. A `flush()` running concurrently in another coroutine may or
may not be visible. Callers needing read-after-write consistency must
serialize reads after the writer's `flush()` completes.

Why fresh-per-call:

1. Filesystem listing is ~ms-range, dwarfed by the 1-5 s join cost.
2. Caching introduces stale-data footguns ("I just flushed, why isn't
   it in the read?"). Fresh forces correctness.
3. A pre-built dataset hides `_tmp/` orphans correctly because
   `_tmp/` is outside `schema={name}/` (only `schema=*/` is scanned).

A `dataset` parameter for caller-pre-built datasets is a v2
deferred. Chaos test C2 (`test_tmp_dir_files_not_visible_to_reader`)
verifies the `_tmp/` invisibility.

## 9. In-memory v1; sharding escape hatch; sort-copy doubles peak memory

After the 30-day partition filter on a 10M-row/year dataset:

| Window | Rows | Peak mem (200-field, ~1.5 KB/row) |
|---|---|---|
| 30 days (default) | ~821k | ~1.2 GB |
| 90 days | ~2.5 M | ~3.7 GB |
| 365 days | 10 M | ~15 GB |

`asof_join` requires `pa.Table` (not `RecordBatchReader`); full
materialization is mandatory in PyArrow 14. **The sort copy in
`_asof_join` doubles peak memory during the asof phase** — at 1M
filtered rows × 200 fields, peak ≈ 2.4 GB. Operators at 90+ days at
200 fields shard by entity_id (`hash(eid) mod N` per CLAUDE.md §1)
across multiple Pyforge instances or multiple read calls.

Streaming asof (per-partition iteration + per-batch dedup and join)
deferred to v2. Significant complexity; no caller asking yet.

## 10. Sync method on async class

`read_point_in_time` is **synchronous** even though
`ParquetDatasetStore.append`/`flush`/`close` are async. Reads are
1-5 s of CPU + IO, not awaitable work. Async callers wrap in
`asyncio.to_thread(store.read_point_in_time, ...)`.

An async signature with no awaits would mislead callers into
believing the method cooperates with the loop. The async writer is
async because it serializes behind the consumer's event loop; the
reader has no such constraint.

## 11. Hand-rolled asof primitive

PyArrow 14 does NOT expose asof-join in Python. Verified twice
(empirical grep across `.py`/`.pxi`/`.pyx` files in the installed
`pyarrow/` tree returns zero matches for "asof"; `pyarrow.compute.py`
exports `JoinOptions` but no asof variant; C++ Acero header has
`AsofJoinNodeOptions(input_keys, int64_t tolerance)` but the Python
wrapper isn't bound until PyArrow 16+). With `pyarrow>=14.0,<15.0`
pinned in `pyproject.toml`, the build-plan pseudocode using
`pc.asof_join` is unimplementable.

The reader hand-rolls the primitive in numpy + pyarrow:

1. Sort dedup'd features by `(entity_id, event_time_ns, msg_id_ms,
   msg_id_seq)` (or `(entity_id, event_time_ns)` when
   `include_msg_id=False`).
2. For each query row, locate the entity's slice via two
   `np.searchsorted` calls on `eid_arr` (object-dtype string compare —
   the most expensive step at scale, ~50-100 ms / 10k × 1M).
3. Within the slice, find the rightmost row with `event_time_ns <=
   as_of_time` via `np.searchsorted(et_arr[lo:hi], q_aot, side="right") - 1`.
   `side="right"` makes the upper bound INCLUSIVE
   (`event_time_ns == as_of_time` matches).
4. **Per-query lookback check inline:** if the candidate's
   `event_time_ns < as_of_time - lookback_ns`, the candidate is
   outside the per-query lookback window → null. **This is the
   Rev-3 CRITICAL-1 fix.** Without it, a feature loaded by the global
   row filter (which uses `min(as_of_time) - lookback`) can match a
   query whose own per-query window ends earlier. Test
   `test_per_query_lookback_multi_query_regression` is the binary
   check.
5. Materialize via `take(indices_with_nulls)` — single call, PyArrow
   emits null at masked positions (verified empirically pre-impl).

**Tolerance equivalence for future PyArrow 16+ migration:** when
`Table.join_asof` becomes available, our "past-as-of with lookback"
maps to `tolerance = -lookback_days * _DAY_NS` (the C++ docstring
specifies negative = past-as-of; the matching condition is
`right.on - left.on <= tolerance`). One-line function swap when
bindings land.

Rejected alternatives:

- **Bump pyarrow to 16+** — pinning is a contract; 22 source files
  depend on PyArrow 14 behavior including the locked `column_encoding`
  interaction in ADR-010 §1. Step 12 isn't the right pass to bump.
  Revisit in Step 16/17 dep-churn.
- **Add DuckDB** — adds ~50 MB install, new C++ surface, new test
  deps. Single use case doesn't justify it.
- **Use `pa.acero.Declaration("asofjoin", ...)` raw** —
  `AsofJoinNodeOptions` Python wrapper isn't in PyArrow 14; can't
  construct the options object without C++ FFI.

## 12. Result schema is explicit; query columns preserved

Result columns, in this exact order:

```
[<query_table column 0>: <its arrow type>]         # all query columns
[<query_table column 1>: <its arrow type>]
...
[<query_table column M-1>: <its arrow type>]      # M = len(query.column_names)
[event_time_ns: pa.int64()]                        # from features (null if no match)
[<schema field 0 name>: <field 0 arrow type>]      # from features (null if no match)
...
[<schema field N-1 name>: <field N-1 arrow type>]
[msg_id_ms: pa.int64()]                            # from features, only if dataset has it
[msg_id_seq: pa.int32()]                           # from features, only if dataset has it
```

**ALL `query_table` columns are preserved** as the leading columns of
the result, in input order. Real ML callers pass labels, weights, and
metadata in `query_table` (e.g., `pa.table({"entity_id":...,
"as_of_time":..., "label":..., "fold":...})`). Dropping non-required
columns would force a join-back step; pandas `merge_asof` and
PyArrow's future `Table.join_asof` both preserve left columns.

`entity_id` is single-source (from query, not duplicated as
`entity_id_x`/`entity_id_y`). `event_time_ns` is exposed so callers
can audit how recent the matched feature was (training data quality
signal).

Test `test_result_column_schema_exact` locks the exact tuple against
a known schema + known query columns including extras like `label`.

## 13. Schema divergence guard

If any field in `schema.fields` is absent from
`dataset.schema.names`, the reader raises `ValueError` upfront with a
message pointing at Step 15's schema evolution as the migration path.
Without this, the asof loop materialization at
`out_cols[f.name] = features_aligned[f.name]` raises `KeyError`
mid-asof — opaque debugging.

Cheap (~µs); runs after `_open_dataset`, before any I/O on the data
plane.

## 14. Query column collision rule

Query column names MUST NOT collide with the result's feature columns
(`event_time_ns`, schema field names, `msg_id_ms`, `msg_id_seq`).
`entity_id` is required from query and is single-source; `as_of_time`
is query-only. Other query columns must avoid the feature column
namespace.

Reader raises `ValueError` upfront if a collision is found, listing
the offending names. Test `test_query_column_collision_raises` covers
the literal `event_time_ns` collision; a partner test covers schema
field name collisions. The strict rule is by design — relaxing would
just invite confusion (which side wins?). Users wanting renamed
copies of feature columns project after the read.

## 15. Out of scope (deferred)

- **`distinct_entity_ids(schema)` helper** — Step 13 hydration owns
  this with full hydration context.
- **`pyforge.wal.make_clients(url)`** — Step 17 public API
  consolidation.
- **Pydantic factory promotion to public namespace** — Step 17.
- **Column projection (`columns=` parameter)** — v2 enhancement; v1
  returns all columns.
- **Pre-built dataset reuse (`dataset=` parameter)** — v2;
  fresh-per-call is correctness default.
- **Streaming asof via per-partition iteration** — v2; PyArrow 14
  doesn't expose the primitive cleanly.
- **Compaction job** — operator-owned per ADR-010 §9.
- **`pa.large_string()` for entity_id at scale** — per ADR-010 §11.
- **`tolerance=` exposure** — `lookback_days` covers it.
- **Writer-side `include_msg_id` mid-deploy validation** — Step 11.1
  if needed.
- **PyArrow upgrade to 16+ for native `Table.join_asof`** — Step 16/17
  dep-churn pass.
- **Numba JIT'd asof inner loop** — Step 16 if flamegraphs warrant.
- **DuckDB dependency** — same.
- **Vectorized asof primitive (eliminate Python loop)** — Step 16 if
  100k+ queries become a real workload.

---

## Consequences

- Step 13's hydration consumes the §12 result schema and reuses the
  `_open_dataset` / `_dedup_features` / `_asof_join` primitives via
  the same module path; no new public API needed.
- Step 14's watchdog has nothing to do for the read path —
  `read_point_in_time` is sync, doesn't manage shm, can't leak.
- Step 15's schema evolution must coexist with §13's divergence
  guard: the migration-path message points users at it.
- Step 16's flamegraphs identify whether the Python inner loop, the
  string `searchsorted`, or the sort copy dominates. If real users
  hit 100k+ queries / call, the primitives in this ADR are the
  candidates to vectorize.
- The hand-rolled asof primitive is a temporary measure pending a
  PyArrow upgrade; the migration is one function swap (locked by §11
  with the tolerance equivalence formula).
