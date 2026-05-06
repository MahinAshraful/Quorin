# Quorin — API reference

Curated reference for the public modules. Source files are linked; full
docstrings on every public symbol live there. ADRs in [`docs/adr/`](adr/) carry
the design rationale for every load-bearing decision.

> **Module-level imports only.** Per [CLAUDE.md §5 invariant #11](../CLAUDE.md#5-non-negotiable-invariants),
> production code uses module-level imports. Inside a Quorin app, prefer
> `from quorin.X import Y` over `import quorin` — the package's top-level
> `__init__.py` only re-exports a few schema-evolution names lazily via
> `__getattr__` to keep Numba off the import path of every other submodule.

---

## `quorin.schema`

Source: [`quorin/schema.py`](../quorin/schema.py)

Schema definitions and the offset-table compiler. NumPy + stdlib only — no
Numba / Redis / pyarrow / pydantic in this module.

| Symbol | Purpose |
|---|---|
| `FeatureSchema` | Base class for declarative feature schemas. Subclass with `version: int` and `fields: list[FeatureField]`. |
| `FeatureField(name, dtype, shape=())` | Declares one field. Frozen slotted dataclass. |
| `dtype.{float32, float64, int32, int64, uint8}` | Closed enum of supported dtypes. |
| `compile_schema(schema)` | Returns a NumPy structured array (the offset table) sorted by `name_hash`. Cached implicitly via Python class identity. |
| `total_segment_size(schema)` | Bytes the schema needs in shm (header + last field + page-padding). |
| `total_element_count(schema)` | Sum of `element_count` across all fields — the length of the float32 vector returned by `assemble`. |

See [ADR-001](adr/001-posix-shm-over-multiprocessing-shared-memory.md),
[ADR-003](adr/003-output-vector-uses-declaration-order.md).

---

## `quorin.shm`

Source: [`quorin/shm.py`](../quorin/shm.py)

Shared-memory segment lifecycle: create, open, close, refcount via Redis.

| Symbol | Purpose |
|---|---|
| `SegmentRegistry(redis_client)` | Per-process gateway. `create(schema, capacity, set_current=True)` allocates a new segment + registers it; `open_current(schema)` opens the segment Redis says is current. |
| `Segment` | Dataclass with `name`, `handle`, `mmap_view`, `layout`, `compiled_offset_table`. |
| `SchemaCRCMismatchError` | Raised when an opened segment's CRC32 doesn't match the locally-loaded schema. |
| `SegmentNotFoundError` | No segment is currently registered for this schema in Redis. |

Linux/WSL2 only. Importing on native Windows fails (no `posix_ipc` wheel).

See [ADR-001](adr/001-posix-shm-over-multiprocessing-shared-memory.md),
[ADR-002](adr/002-per-open-refcounting.md).

---

## `quorin.layout`

Source: [`quorin/layout.py`](../quorin/layout.py)

Slot table, string pool, feature-row layout. Pure Python; no Numba.

| Symbol | Purpose |
|---|---|
| `insert(seg, entity_id, row_bytes)` | Single-writer upsert. Advances cursors before marking slot occupied (crash-safe per CLAUDE.md invariant #8). |
| `lookup(seg, entity_id)` | Pure-Python single-entity lookup. Used by `quorin.serving` (the parity reference); the Numba JIT path lives at `quorin._internal.lookup_kernel.lookup_jit`. |
| `pack_row(schema, **fields) -> bytes` | **NEW in v0.1.0.** Convenience packer: pass values as keyword arguments, get back `bytes` ready for `insert`. For high-throughput production writes, prefer `quorin.wal.WALProducer`. |
| `iterate_occupied(seg)` | Yields `(entity_id, row_offset)` for every occupied slot. Used by hydration / evolution. |
| `CapacityExceededError` | Raised at `insert` when the segment's capacity is reached (50% slot-table load). |
| `StringPoolExhaustedError` | Raised when entity-ID storage runs out (segment-internal pool). |
| `SegmentLayout`, `compute_layout(...)`, `compute_layout_from_segment(...)` | Derived offsets dataclass + builders. Mostly internal but exposed for advanced callers. |

---

## `quorin.serving`

Source: [`quorin/serving.py`](../quorin/serving.py)

Pure-Python read oracle. Byte-identical to `quorin.assembly.assemble` for any
segment state — the parity reference that locks the Numba kernel's behavior.

| Symbol | Purpose |
|---|---|
| `assemble(seg, entity_id, *, out=None) -> np.ndarray[float32]` | Read one entity's feature vector. `out=` for buffer reuse. |
| `EntityNotFoundError` | Raised when `entity_id` isn't in the segment. |

Importing this module does NOT pull Numba. Use this when you want to skip the
~200 ms LLVM init.

---

## `quorin.assembly`

Source: [`quorin/assembly.py`](../quorin/assembly.py)

Numba-compiled hot path. ~4 µs p99 for the 4-field warm assemble; ~12 µs p99
for 200-field warm. Importing this module triggers Numba's LLVM init.

| Symbol | Purpose |
|---|---|
| `assemble(seg, entity_id, *, out=None) -> np.ndarray[float32]` | Numba JIT single-entity read. Same signature + behavior as `quorin.serving.assemble`. |
| `assemble_batch(seg, entity_ids, *, out=None, found_mask=None) -> tuple[ndarray, ndarray]` | Batch read; returns `(N, total_element_count)` + `(N,)` bool mask. Misses are zero-filled. |
| `prewarm()` | Eagerly compile both kernels (single-entity + batch) + the lookup-jit + Numba BLAKE2b kernels. Opt-in; module load doesn't auto-warm. |
| `PARALLEL_THRESHOLD` | Adaptive batch-size cutoff above which `assemble_batch` uses the parallel kernel. Set automatically at module load based on `numba.get_num_threads()`. |

See [ADR-004](adr/004-numba-adoption-gate.md),
[ADR-007](adr/007-batch-assembly.md),
[ADR-017](adr/017-lookup-jit.md).

---

## `quorin.pool`

Source: [`quorin/pool.py`](../quorin/pool.py)

Pre-allocated buffer pools. Two distinct classes for distinct shapes.

| Symbol | Purpose |
|---|---|
| `BufferPool(schema, max_size=128, zero_on_return=True)` | Single-row 1D float32 buffers. `with pool.checkout() as buf: ...` lifecycle. |
| `BatchBufferPool(schema, batch_size, max_size=64, zero_on_return=False)` | Batch 2D `(batch_size, element_count)` buffers. Pre-allocated as a single contiguous slab. |

Per [ADR-005 §"Step 16c amendment"](adr/005-buffer-pool-lock-free-prealloc-capped.md):
on native CI, `BufferPool` adds **+2-4 µs** to single-entity assemble. Pool is
**default for the batch path** (where amortization wins), **opt-in for
single-entity** workloads. Wins survive (one fewer ndarray allocation per
call, memory ceiling, observability) but the latency cost is honestly disclosed.

---

## `quorin.wal`

Source: [`quorin/wal.py`](../quorin/wal.py)

Write-ahead log producer. Async-by-default (XADD then return); `write_sync`
for read-your-own-writes.

| Symbol | Purpose |
|---|---|
| `WALProducer(redis_client, *, schema_name=None)` | Validates rows via pydantic, msgpack-encodes, XADDs to `quorin:wal`. Reusable msgpack `Packer`; memoized pydantic model class; pre-warmed Prometheus labels. |
| `WALProducer.write(...)` | Async fire-and-forget. ~10k writes/sec target. |
| `WALProducer.write_sync(..., timeout=0.1)` | XADD then poll the consumer's processed-sidetable. Pays one consumer-cycle round trip (5-50 ms typical). |
| `WriteSyncTimeout` | Raised by `write_sync` when the consumer doesn't ack within `timeout`. |

See [ADR-008](adr/008-wal-producer-design.md).

---

## `quorin.wal_consumer`

Source: [`quorin/wal_consumer.py`](../quorin/wal_consumer.py)

WAL consumer — the single writer to the segment.

| Symbol | Purpose |
|---|---|
| `WALConsumer(redis, registry, *, offline_writer=None, ...)` | Async coroutine: read from `quorin:wal` group, validate, `layout.insert` to segment, append to offline writer, defer XACK until offline flush returns. |
| `WALConsumer.run()` | Run forever (cancellation closes cleanly). |
| `OfflineWriter` (Protocol) | Implementations must provide `append(rows)` + `flush()`. |
| `NoopOfflineWriter` | No-op implementation for tests / configurations without an offline store. |

See [ADR-009](adr/009-wal-consumer-design.md).

---

## `quorin.offline`

Source: [`quorin/offline.py`](../quorin/offline.py)

Append-only Parquet store + point-in-time reads + hydration helpers.

| Symbol | Purpose |
|---|---|
| `ParquetDatasetStore(root_dir, schema, *, flush_interval_seconds=5, max_rows_in_memory=10_000, include_msg_id=True, ...)` | Async writer. `await append(rows)` buffers; `await flush()` writes one Parquet file per call. |
| `ParquetDatasetStore.read_point_in_time(queries)` | Synchronous (CPU + IO). Returns features as-of each query timestamp via `searchsorted` asof-join. |
| `ParquetDatasetStore.latest_features(entity_ids)` | Optimized "as-of now" path used by hydration. |
| `ParquetDatasetStore.distinct_entity_ids()` | Returns the unique entity IDs in the dataset. |

See [ADR-010](adr/010-parquet-offline-store.md),
[ADR-011](adr/011-point-in-time-reads.md).

---

## `quorin.hydration`

Source: [`quorin/hydration.py`](../quorin/hydration.py)

Rebuild a segment from the Parquet offline store on cold start.

| Symbol | Purpose |
|---|---|
| `hydrate(redis, schema, store, *, capacity, max_id_bytes=64) -> HydrationResult` | Synchronous (5-10 s for 1M rows on WSL2). Async callers wrap in `asyncio.to_thread`. |
| `HydrationResult` | Dataclass with the populated segment + row count + duration. |
| `HydrationError` (+ `HydrationConflictError`, `EmptyDatasetError`) | Failure modes. |

Preconditions: no `schema:current` set, no live WAL consumer. Operator-
serialized; not enforced via mutex (per [ADR-012](adr/012-hydration.md)).

---

## `quorin.evolution`

Source: [`quorin/evolution.py`](../quorin/evolution.py)

Atomic schema-version flip with operator-verified semantics.

| Symbol | Purpose |
|---|---|
| `upgrade_schema(redis, *, old, new, ...) -> UpgradeResult` | Synchronous orchestrator: build new segment, vectorized translation, atomic CAS flip on `quorin:schema:{name}:current`, wait for consumer attach. |
| `can_upgrade(old, new)` | Pure predicate — checks add-only / dtype-widening rules without touching Redis or shm. |
| `UpgradeResult` | Dataclass with old/new segment names, row count, duration. |
| `UpgradeError` (+ `UpgradeConflictError`, `UpgradeIncompatibleError`, `UpgradeAbortedError`) | Failure modes. |
| `main(argv)` | CLI entry: `python -m quorin.evolution upgrade --redis URL --old PKG:V1 --new PKG:V2 --confirm`. |

Operationally requires draining the WAL stream + stopping the consumer first;
the consumer's pause+reopen safety net catches operators who skip this with a
loud poison-pill failure. See [ADR-014](adr/014-schema-evolution.md).

---

## `quorin.watchdog`

Source: [`quorin/watchdog.py`](../quorin/watchdog.py)

Background process: detects dead PIDs via heartbeat staleness + `psutil`
cross-check; decrements their refcounts; drains the cleanup queue.

| Symbol | Purpose |
|---|---|
| `main(argv)` | CLI entry: `python -m quorin.watchdog --redis URL [--metrics-port N]`. Runs forever. |
| `WatchdogState(redis_client, ...)` | Constructed by tests; the CLI uses `run_forever`. |
| `WatchdogState.run_one_tick()` | Single watchdog cycle: HGETALL heartbeats → cross-check → cleanup Lua → drain. Returns `TickResult`. |
| `run_forever(state, ...)` | Loop forever calling `run_one_tick` at `tick_interval_seconds`. |

See [ADR-013](adr/013-watchdog.md).

---

## `quorin.metrics`

Source: [`quorin/metrics.py`](../quorin/metrics.py)

Prometheus instrumentation. Counters / histograms / gauges register at module
load; values increment in memory whether or not you start the HTTP server.

| Symbol | Purpose |
|---|---|
| `start_metrics_server(port=9100)` | Starts a `prometheus_client` HTTP server on the given port. Optional. |
| `registry` | The `CollectorRegistry` instance — pass to a custom HTTP wrapper if you don't want the built-in server. |
| `read_latency_seconds`, `gc_pause_seconds`, `wal_lag_seconds`, `pool_miss_total`, ... | The instruments themselves. Names are stable; relabel collisions are checked at module load. |

All metric names use the `quorin_*` prefix.

---

## `quorin.logging`

Source: [`quorin/logging.py`](../quorin/logging.py)

structlog JSON configuration.

| Symbol | Purpose |
|---|---|
| `configure(level="INFO", json=True)` | Idempotent. First-call sets the structlog renderer chain. |
| `get_logger(name=None)` | Returns a `BoundLogger`. Auto-calls `configure` on first use. |
| `bind(**kwargs)` | Bind context vars onto the contextvar-backed root logger. Useful for `bind(component="quorin.serving")` at startup. |

---

## Errors at a glance

| Module | Errors |
|---|---|
| `quorin.shm` | `SchemaCRCMismatchError`, `SegmentNotFoundError` |
| `quorin.layout` | `CapacityExceededError`, `StringPoolExhaustedError` |
| `quorin.serving` / `quorin.assembly` | `EntityNotFoundError` (per the `assemble` semantics — typically not raised; `assemble_batch` returns a `found_mask` instead) |
| `quorin.wal` | `WriteSyncTimeout` |
| `quorin.hydration` | `HydrationError`, `HydrationConflictError`, `EmptyDatasetError` |
| `quorin.evolution` | `UpgradeError`, `UpgradeConflictError`, `UpgradeIncompatibleError`, `UpgradeAbortedError` |

---

## Where to look next

- **Why does X exist / why this design?** → [`docs/adr/`](adr/) — 17 ADRs, one
  per load-bearing decision.
- **What does the wire look like?** → ADR-008 (WAL message format), ADR-009
  (consumer apply loop), ADR-014 §"Critical decisions" #6 (poison-pill format
  contract).
- **What's the performance methodology?** → [ADR-015](adr/015-benchmark-methodology.md)
  (venue disclosure, N=20 fresh subprocess discipline, native-CI calibration).
- **What about the cold-cache near-miss?** → ADR-015 §11 (bare-metal
  extrapolation framework), ADR-017 (the trip-wire ratification record).
