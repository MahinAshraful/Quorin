# ADR-012: Hydration — orchestrator, liveness gating, watchdog-deferred orphan cleanup

**Status:** Accepted
**Date:** 2026-05-03
**Step:** 13 (Hydration)

## Decision

Pyforge ships [`pyforge.hydration.hydrate`](../../pyforge/hydration.py),
a sync orchestrator that rebuilds the online store (a fresh shared-memory
segment registered as `pyforge:schema:{name}:current`) from the offline
store (a `ParquetDatasetStore`). It runs at startup or after any event
that wipes Redis state — operator-driven recovery, not background
maintenance.

Public API:

```python
def hydrate(
    schema: type[FeatureSchema],
    store: ParquetDatasetStore,
    registry: SegmentRegistry,
    *,
    redis_client: redis.Redis,
    capacity_factor: float = 4.0,
    as_of_time_ns: int | None = None,
    lookback_days: int = 30,
) -> HydrationResult: ...
```

The 14 numbered sections below lock the design end-to-end.

---

## 1. Two preconditions (current-segment + WAL consumer liveness)

Hydrate refuses to run if EITHER:

- `pyforge:schema:{name}:current` exists (a current segment is registered) → `HydrationConflictError`.
- `pyforge:wal_consumer:liveness` exists (a WAL consumer is alive) → `HydrationConflictError`.

The two checks have different roles:

- **Precondition #1 is the PRIMARY race guard.** A live current segment
  means a writer (the WAL consumer) is potentially calling `layout.insert`
  on it. Creating a second segment and flipping `current` to point at
  the new one would race the writer's in-flight insert against the
  reader's view, producing torn slot tables. Operators clear it via
  `redis-cli DEL pyforge:schema:{name}:current` after stopping the
  writer (Step 14 watchdog will automate this once shipped).
- **Precondition #2 is defense-in-depth.** Catches the "operator
  manually dropped current but forgot to stop the consumer" scenario.
  The consumer would notice the missing segment on its next message
  and crash; safer to refuse hydrate upfront.

Both preconditions raise BEFORE `t0 = perf_counter()`, so the
`pyforge_hydration_seconds{outcome="err"}` histogram excludes
precondition rejections. Operators alerting on hydrate failure rate
combine the histogram with the `hydrate.precondition_*` structlog
WARNING signal — rejections are a separate failure mode (operator
procedure violation, not pipeline failure).

## 2. Liveness key uses force-first-refresh + monotonic seeding

The naive implementation — `_liveness_last_refresh = 0.0` at consumer
init, gauge update on each iter as `now - _liveness_last_refresh` — has
a window between init and the first iter's per-loop refresh during
which `_liveness_last_refresh` is still 0.0. A Prometheus scrape
landing in that window reports `now - 0.0`, which on a 1-day-uptime
host is 86400 seconds — paging on-call with "consumer dead 1 day ago"
when the consumer just started.

Fix in [`pyforge.wal_consumer.run`](../../pyforge/wal_consumer.py):

1. Force-first-refresh of the liveness key + `SET pyforge:wal_consumer:liveness ... EX 30` immediately at top of `run()`, before any awaits.
2. `_liveness_last_refresh = time.monotonic()` immediately after the SET.
3. Defensive `wal_consumer_liveness_age_seconds.set(0.0)` (covers multiprocess-collector futures where the gauge's first observation is the source of truth).

Per-iter refreshes are 10s-gated (refresh interval well below the 30s
TTL). The `LIVENESS_TTL_SECONDS = 30` / `LIVENESS_REFRESH_INTERVAL_SECONDS = 10`
ratio gives 3 missed refreshes before a stale-detection signal — tolerant
of a transient Redis hiccup, intolerant of a hung consumer.

## 3. `_force_drop_orphan` deliberately bypasses `cleanup_queue`

When `insert_many` fails mid-bulk, the orchestrator's exception handler
needs to drop the partially-populated segment it just created. The
canonical lifecycle (`registry.close → cleanup_queue → watchdog.unlink`)
is wrong here:

- The orchestrator is the *creator* of this segment (just called
  `registry.create`). Per invariant #6 (only-creator-unlinks),
  `registry.close` would NOT unlink — it would only DECR the refcount
  and queue the name for the watchdog. The watchdog isn't running
  yet (Step 14 unshipped), and even when it ships, hydrate's
  recovery path can't wait for it.
- The orchestrator has the `Segment` handle in scope. Calling
  `posix_shm.unlink(seg.name)` directly is the right shape: same
  process, has the handle, can clean up synchronously.

[`_force_drop_orphan`](../../pyforge/hydration.py) does three things in
explicit order:

1. `posix_shm.unlink(segment.name)` — remove the inode.
2. `redis_client.delete(_key_refcount(segment.name))` — drop the refcount key.
3. `redis_client.srem(_key_pid_segments(os.getpid()), segment.name)` — remove from the per-pid set.

Per-block `try/except` so a Redis failure on step 2 doesn't prevent
step 3 from running (orphan refcount keys are recoverable; orphan
shm inodes leak `/dev/shm` until reboot). INFO log
`hydrate.force_drop_orphan_complete` only when all three blocks
succeeded; per-block WARNINGs otherwise.

The DO-NOT-CHANGE-TO-`registry.close()` warning is in the docstring.
A future refactor that "consolidates" the cleanup paths would
silently leak `/dev/shm` segments on every hydrate failure.

## 4. Concurrent hydrate is operator-serialized; NOT enforced

The plan's Rev-1 chaos suite included a C2 test (concurrent hydrate
race, two children both call hydrate). It was **deleted before shipping**
because the test premise was wrong:

- `_segment_name(schema)` uses a UUID suffix (`pyforge_{schema_name}_v{version}_{uuid_hex}`).
- Each `registry.create` call generates a fresh UUID.
- POSIX `O_CREAT|O_EXCL` only fails on exact-name collisions, which UUID suffixes preclude.
- Both racing children would always succeed at `registry.create`, both would call `_INCR` on different refcount keys, both would `SET pyforge:schema:{name}:current` (last-write-wins), both would exit 0.

There is no enforcement code path to test. The contract is operator
discipline: stop the WAL consumer (or wait for liveness expiry) before
running hydrate, and don't run two hydrates concurrently.

When Step 14 watchdog ships, orphans from a violated contract get
cleaned up automatically. Until then, two-segment leaks require
operator intervention (`redis-cli DEL pyforge:schema:{name}:current` +
manual `posix_shm.unlink` of the orphan name from
`/dev/shm`).

## 5. Differentiated `BaseException` catches

The orchestrator's exception handlers split into two classes:

- `(KeyboardInterrupt, SystemExit, asyncio.CancelledError)` — quiet
  propagate. If a segment was created, run `_force_drop_orphan` first;
  no `logger.exception` (CancelledError is normal during async cancel,
  KeyboardInterrupt is normal during operator shutdown).
- `Exception` — `logger.exception` (full stack), then orphan cleanup,
  then re-raise.

Pre-Rev-7, a bare `except Exception:` would catch CancelledError on
Python 3.7 but not on 3.8+; a bare `except BaseException:` would log
spurious "exception during hydrate" entries on every Ctrl-C. The
split avoids both failure modes.

## 6. NaN/Inf guard on `capacity_factor`

`capacity_factor` is documented as `>= 2.0`. A bare `< 2.0` check passes
NaN through (NaN comparisons are always False). Then
`int(len(latest) * NaN)` raises `ValueError: cannot convert float NaN to integer`
deep inside the orchestrator with no mention of capacity_factor.

Locked: `if not math.isfinite(capacity_factor) or capacity_factor < 2.0: raise ValueError(...)`.
Test #19 in `tests/unit/test_hydration.py` covers NaN, ±Inf, and `< 2.0`.

## 7. Sync method on a mostly-async codebase

`hydrate()` runs for seconds (200ms-15s depending on scale). Async
callers must wrap in `asyncio.to_thread(hydrate, ...)`. An async
signature with no awaits would mislead callers into believing the
method cooperates with the loop; it doesn't.

This mirrors the precedent set by
[`ParquetDatasetStore.read_point_in_time`](../../pyforge/offline.py)
in Step 12 (ADR-011 §10).

## 8. `latest_features` primitive replaces `read_point_in_time(query=now)`

Hydrate needs "the most recent feature row for each entity_id within
lookback_days". Reusing `read_point_in_time` with a synthesized query
table (one row per entity at `now`) would work but pays per-query
`searchsorted` for each entity. For 1M entities that's 1M binary
searches over a sorted-event_time array — pure overhead.

[`ParquetDatasetStore.latest_features`](../../pyforge/offline.py) is
the load-bearing hydration primitive: a single
`group_by(entity_id).aggregate([("event_time_ns", "max"), ...])` over
the dedup'd row set, with the `as_of_time_ns` filter applied at the
row level (not per-query). One pass, no searchsorted.

`distinct_entity_ids` is a separate public method (off the hydration
critical path, useful for operator queries).

## 9. Bulk-insert kernel input is column-major flat (not row-major 2D)

The plan's Rev-1 design materialized a `(n, row_size)` 2D buffer and
passed it to a Numba kernel that memcpy'd row-by-row. Measured cost
at 100k × 200 fields was ~3000ms — twice as bad as expected because
the column-scatter phase did 200 strided writes (each touching 100k
cache lines at 13K stride, ~1630ms alone).

[`_insert_many_core`](../../pyforge/_internal/insert_kernel.py) takes
column-major flat data via 4 array args (`field_byte_offsets_in_row`,
`field_byte_counts`, `field_data_starts`, `field_data`). The kernel
scatters each field directly into the segment's row buffer in
row-major order (cache-friendly; the prefetcher streams pages in
order). Net: 1.76× speedup at 100k × 200, with the column-scatter
phase alone going 4.3× faster (1630 → 379ms).

The byte-identical parity test
(`tests/property/test_insert_many_parity.py::test_insert_many_byte_identical_to_insert_loop`)
locks the kernel against any regression in slot layout, string-pool
encoding, or per-field byte placement.

## 10. Bench gate methodology shift — native targets, not WSL2 measured

Steps 11 and 12 set bench gates at "4× WSL2 measured baseline". The
WSL2 number is dominated by Docker Desktop 9P + virtio-fs cold-page-
fault overhead (~4-6× slower than native ext4 on `/dev/shm` cold
writes; CLAUDE.md §8). CI runs on Ubuntu 24.04 native Linux. A gate
sized at 4× WSL2 for a workload that hits the WSL2 cold-fault floor
is ~6× looser than the native target — a regression that doubles
native latency would still pass.

Step 13 onward: gates set at native-Linux targets. The
[`thresholds.yml`](../../benchmarks/regression/thresholds.yml)
methodology block documents the discipline. WSL2 dev runs may exceed
the gate up to 4× (within tolerance, flag-worthy); above 4× even on
WSL2 is a real regression.

Steps 11/12 gates are loose by ~4-6× and flagged in `thresholds.yml`
for Step 16 retro-tightening:
- `flush_10k_rows_200_field`: 0.6s → ~0.15-0.2s
- `read_pit_1k_pairs_500k_rows_50_field`: 12s → ~3-4s

DO NOT tighten in Commit B — out of scope. Step 16 owns the actual
gate evaluator with WSL2 detection logic.

## 11. Bulk Numba kernel is uninterruptible — chaos tests use coarse timing

C1's first design polled `next_free_row_index` and SIGKILLed when it
crossed a threshold N. Two compounding problems:

- The kernel updates `next_free_row_index` 0→N atomically at the end
  (advance-cursors-before-slot-writes is invariant #8, and the kernel
  honors it by not advancing until the loop completes). There is no
  mid-insert state observable from outside the kernel.
- The poller's `registry.open_current` from the parent SREMs from
  `pyforge:pid_segments:{child_pid}` — sabotaging the watchdog state
  the test needs to verify post-cleanup.

[`tests/chaos/test_hydration_crash.py`](../../tests/chaos/test_hydration_crash.py)
uses a simple time-based killer:
`asyncio.sleep(delay_ms/1000); os.kill(child.pid, SIGKILL)`. Coarse-grained
but deterministic. The asserted `child.exitcode == -9` after `child.join`
ensures the kill landed (otherwise a fast-completing hydrate would make
the test silently a no-op). Delays at {20, 50, 100} ms span the
common kill windows; C4's 20-seed iteration gives ~5 minutes of
chaos-soak per CI run.

## 12. Test sequencing relies on `pyforge:processed:{msg_id}`, not pending counts

Integration tests need to know "consumer has applied through msg_id
N" before flushing the parquet store and calling hydrate. Two
approaches don't work:

- **Polling `consumer._pending_ack=[] AND XPENDING=0`**: at consumer
  startup both are trivially true BEFORE consumer has read anything.
  First poll fires before the consumer task runs, returns immediately,
  parquet flush hits empty buffer, hydrate raises `EmptyDatasetError`.
  Adding `await asyncio.sleep(0)` to yield once doesn't help because
  the consumer's force-first-refresh + XGROUP setup involves multiple
  awaits before XREADGROUP.
- **Polling pending_ack alone with `flush_interval_seconds=300`** (E5's
  case): pending_ack only clears when the consumer's flush task fires,
  which is disabled in this test. Helper waits forever.

[`_wait_for_msg_processed`](../../tests/integration/test_hydration_e2e.py)
polls for `pyforge:processed:{msg_id}` (set by consumer immediately
after `layout.insert + offline.append`). This is the deterministic
"consumer applied through msg_id" signal. The apply order
(layout.insert → offline.append → SET processed_key → pending_ack
extend) means by the time the SET fires, the row is already buffered
in parquet. Subsequent `parquet_store.flush()` writes the file; the
trailing pending_ack staying non-empty (E5 case) is fine because it
only delays XACK (PEL drain), not parquet content.

The pattern is: `wait_for_msg_processed(last_msg_id) → drain_consumer →
flush`. Five integration tests reuse it. Future tests that sequence
producer ↔ consumer ↔ store should follow the same pattern.

## 13. `registry.close` does NOT delete `pyforge:schema:{name}:current`

The Lua script in [`SegmentRegistry.close`](../../pyforge/shm.py)
removes refcount + pid_segments + queues for cleanup_queue, but
deliberately leaves `pyforge:schema:{name}:current` in place.
Production-side, the next `registry.create` overwrites it. For
repeated hydrate calls in tests (setup → hydrate → cleanup → hydrate
again), the orphan `current` pointer trips precondition #1 on round 2.

Five places in Commit B add an explicit
`redis_client.delete(_key_current(schema))` to the cleanup path:
- `tests/integration/test_hydration_e2e.py::_run_producer_consumer_pipeline`
- `tests/integration/test_hydration_e2e.py::_drop_current`
- E2's local cleanup
- E5's local cleanup
- All four bench-file cleanup blocks (`_make_hydrate_runner` + 4 trailing per-test blocks)

A future production caller doing repeated hydrate cycles (e.g. an
operator script that hydrates multiple schemas in sequence) must
follow the same pattern. Step 14's watchdog will eventually own this
deletion as part of its "drop dead PID's segments + their schema
pointers" duty.

## 14. No-PII logging discipline

Logged values deliberately exclude `entity_id` values and row data.
The single hydration-wide `as_of_time_ns` IS logged at start (it's a
timestamp parameter, not entity-data, and operators need it for
reproducible failure investigation). Per-entity timestamps from the
offline store are never logged. `schema.__name__` IS logged; ensure
schema names do not contain customer identifiers at the schema-design
layer (a per-customer `Schema_acme_corp_user_features` naming pattern
would leak via support-ticket log greps).

This matches the discipline established in ADR-009 §11 for the WAL
consumer's apply path; hydration extends it to the orchestrator path.

## 15. Out of scope (deferred)

- **Multi-schema hydrate orchestrator** — Step 14 or 15. `hydrate(schema, ...)` handles one schema; operator scripts loop today.
- **Step 14 watchdog** — automates `_force_drop_orphan` for
  process-crash orphans (this ADR's path is in-process orphans).
  Test-side simulation lives at
  [`tests/_watchdog_helpers.py`](../../tests/_watchdog_helpers.py)
  and documents the contract Step 14 must implement.
- **Schema-evolution + hydrate interplay** — Step 15. Hydrate today
  assumes the offline store's row schema matches the in-memory
  schema; Step 15 will add a column-projection path so hydrate can
  rebuild against an upgraded schema.
- **`MADV_HUGEPAGE` on segment mmap** — Step 16 perf flamegraphs.
  Estimated 3-5× WSL2 / 1.5-2× native cold-write speedup; deferred
  pending native-Linux baseline numbers.
- **Native-Linux hydrate baseline numbers** — CI captures them on
  push; the 200-210ms WSL2 smoke result will likely drop to 30-50ms
  on native Linux. Recorded in `progress/progress.md` once observed.
- **Concurrent-hydrate enforcement** — could be added via a
  `pyforge:hydration:lock` SETNX with a TTL, blocking the second
  caller. Deferred because the operator-serialization contract is
  sufficient for single-node deployments and adds zero RTT to the
  read path. Reconsider if multi-machine orchestration ever lands
  (which is itself out of scope per the spec).

---

## Consequences

- Step 14's watchdog has a clear contract:
  [`tests/_watchdog_helpers.py::simulate_watchdog_post_crash_cleanup`](../../tests/_watchdog_helpers.py)
  documents the post-SIGKILL path that walks
  `pyforge:pid_segments:{dead_pid}`, DECRs refcounts, and unlinks
  segments whose count hit zero. The watchdog must also drop
  `pyforge:schema:*:current` pointers that match unlinked segments
  (test helper does an O(N keyspace) scan; the real watchdog will
  use a sidetable to avoid this).
- Step 15's schema evolution must not break the two-precondition
  contract. The atomic-pointer-flip path will need a concurrent-
  hydrate-vs-evolution mutex or a strict ordering rule.
- Step 16's flamegraphs can tighten Steps 11/12 retroactive gates +
  identify the next bottleneck (likely `/dev/shm` cold-page faults
  per CLAUDE.md §8).
- Operator runbook for hydrate failure: stop WAL consumer (SIGTERM
  or wait 30s for liveness expiry) → confirm precondition #2 cleared →
  `redis-cli DEL pyforge:schema:{name}:current` → run hydrate. If
  hydrate raises `EmptyDatasetError`, inspect `lookback_days` (default
  30) — the offline store may have rotated out all rows.
- The 200-210ms WSL2 smoke result is the regression-detection signal
  for warm-cache hydrate at 10k × 50 fields. Native-Linux CI
  numbers (likely 30-50ms) tighten this in Step 16.
