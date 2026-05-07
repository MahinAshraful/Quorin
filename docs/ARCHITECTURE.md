# Architecture

A guided tour of how Quorin is built — what each component does, why it exists,
and how the pieces fit together. Written so a software engineer who has never
touched a feature store can follow along, but technical enough to be useful as a
reference if you're going to extend the library.

If you only want to *use* Quorin, see [USAGE.md](USAGE.md). If you want the
load-bearing design rationale for every non-obvious decision, the [ADR archive](adr/)
is the canonical record (17 numbered records, one per decision, written
alongside the step that introduced them).

---

## 1. The problem Quorin exists to solve

When a machine-learning model serves a prediction online, the model itself is
usually fast. A gradient-boosted tree across 200 features (XGBoost, LightGBM)
takes around 200 microseconds. A small neural net is comparable. The model is
not the bottleneck.

The bottleneck is everything *around* the model. A typical production request
flow looks like:

> A request arrives → look up which features the model needs → fetch those
> features from a database (usually Redis) → decode the bytes → assemble them
> into a Python dictionary → pass that dictionary to the model → serialize the
> prediction back to the caller.

Each of those steps allocates Python objects: dictionaries, lists, strings,
small integers. At 50,000 requests per second a serving process allocates
millions of short-lived objects per second. Python's garbage collector
periodically pauses every thread to clean those up, which shows up in the
latency tail as sudden 5–50 millisecond spikes hitting roughly one request in a
thousand. Steady-state CPU is also high because every Redis call is a network
round trip (30–80 microseconds on loopback) and every byte-decode walks a
pure-Python code path.

The result is that the infrastructure around the model takes 5–50 milliseconds
per request. The model itself takes 200 microseconds. The infrastructure is
25–250 times slower than the math.

Quorin's design goal is straightforward: make the infrastructure path fast
enough that it stops dominating the latency. The headline target is **5
microseconds at the 99th percentile** for a small-schema warm-cache read — a
thousand times faster than the 5-millisecond status quo.

---

## 2. The core insight

Two ideas combined make the speed possible:

**Shared memory.** Modern operating systems let multiple processes map the
same physical region of RAM into their own address spaces. Once mapped, every
process can read those bytes directly — no network call, no decode step, no
copy. Quorin allocates one shared-memory segment per feature schema. Every
worker process opens it once at startup. From then on, reads are pure memory
accesses.

**A precomputed offset table.** When you define a schema (a list of typed
fields), Quorin compiles it once into a small NumPy array that says "field
`age_normalized` lives at byte offset 96; field `behavior_embedding` is 128
floats starting at byte offset 192." This lookup table is sorted by a hash of
the field names. At read time, finding a field is a binary search over an
integer array — no Python dictionary lookup, no string comparison, no object
allocation.

A read becomes: hash the entity ID (around 1.5 microseconds in stdlib Python,
much less with the Numba-compiled BLAKE2b kernel Quorin ships); compute the
offsets; copy the bytes into a pre-allocated output buffer; return a NumPy
view. Total time on a 4-field schema with the JIT-compiled kernel is **about
4.5 microseconds at the 99th percentile** on commodity GitHub Actions
hardware. The model can then call `model.predict(features)` directly — the
output is already a contiguous `numpy.float32` array, the exact shape every
ML library expects.

---

## 3. The data flow at a glance

Two paths run concurrently. The fast one (reads) never touches Redis, never
allocates Python objects on the hot path, and never crosses a process
boundary. The slow one (writes) is async by default and routes through Redis
for crash safety.

**The read path** runs in your worker processes. Given an entity ID and a
schema, it hashes the ID into a 64-bit integer, looks up the matching slot in
the segment's slot table via linear-probe search, copies the field bytes into
an output buffer, and returns the buffer as a NumPy array. No allocations
beyond the optional pre-allocated buffer pool. No Redis contact. The Numba
kernel inlines all of this into native machine code.

**The write path** runs anywhere a producer process lives. Producers don't
write to shared memory directly (a single-writer invariant prevents lock-free
reads from becoming lock-required reads). Instead, a producer validates the
incoming row against its pydantic-generated schema model, msgpack-encodes the
field values, and `XADD`s the message to a Redis Stream named
`quorin:wal`. A separate **WAL consumer process** reads from that stream,
applies each message to the shared-memory segment as the segment's only
writer, and appends the row to an append-only Parquet dataset for training
data and crash recovery.

Two background processes round things out. The **watchdog** detects worker
processes that have died without releasing their references and cleans up
their shared-memory segments. The **schema-evolution coordinator** handles
the rare case of upgrading a schema: it allocates a new segment with the new
field layout, copies and translates every row from the old segment, and
atomically flips a Redis pointer so new readers see the new segment while old
readers continue against the old one until they close.

---

## 4. The components, one at a time

### 4.1 Schema and the offset table

A schema is a Python class that names typed fields. Quorin's `compile_schema`
function walks that list once at startup, hashes each field name to a 64-bit
integer using BLAKE2b, computes a byte offset for each field that's aligned
to a 64-byte cache line boundary (so individual reads always hit a single CPU
cache line), and returns a NumPy structured array sorted by name hash. That
sorted array is the offset table. Every later component reads it instead of
re-deriving it. The compile happens once per process; the table is reused for
every request after that.

Field name hashing is pinned forever to BLAKE2b with an 8-byte digest
interpreted as a little-endian unsigned 64-bit integer. Two regression tests
assert known hash outputs for known inputs. Changing the algorithm would
silently break every persisted segment in existence — which is why it's
locked at the invariant level. Hash collisions at 64 bits are vanishingly
rare in practice (10⁻¹⁹ probability per pair); when one does occur, the slot
table's byte-compare disambiguation handles it correctly. A dedicated unit
test forces a collision via `monkeypatch` to verify the path.

Why declare types explicitly instead of inferring them at runtime? Because
training-serving skew — where a model trains on 64-bit floats but serves on
32-bit floats — is a class of bug that's invisible in unit tests and shows up
as silent accuracy regressions in production. Explicit types catch it at
schema definition time.

### 4.2 The shared-memory segment

The segment is a single contiguous region of memory in `/dev/shm` (Linux's
in-RAM filesystem for shared memory). Its layout, computed deterministically
from the schema and a capacity hint, has four named regions:

1. A **header** carrying a magic byte sequence (`PYFG`), the schema version,
   a CRC32 of the compiled offset table, and the segment capacity.
2. A **slot table** — an array of fixed-size records, one per occupied
   entity, each containing the entity ID's hash, a pointer into the string
   pool where the full ID lives, an "occupied" flag, and an index into the
   feature rows region.
3. A **string pool** holding the actual UTF-8 bytes of every entity ID
   (length-prefixed). The slot table only stores hashes for fast lookup; the
   pool stores the originals so that hash collisions can be disambiguated by
   byte comparison.
4. A **feature rows region** — `capacity` rows of fixed byte width, one per
   entity, laid out exactly per the offset table.

Rationale for putting all four in one segment: every read operation needs to
touch the slot table, the string pool, and one feature row. Keeping them
physically adjacent keeps them in nearby cache pages and avoids managing
multiple shared-memory handles per schema.

Why POSIX shared memory directly (via the `posix_ipc` PyPI library) instead
of Python's standard-library `multiprocessing.shared_memory`? Because the
standard library's implementation has a long-standing bug: every reader's
clean exit unregisters the segment from the OS, deleting it out from under
other readers still using it. The bug has been open for years and has no
clean fix at the standard-library level. Quorin's wrapper bypasses it
entirely. See [ADR-001](adr/001-posix-shm-over-multiprocessing-shared-memory.md).

### 4.3 The slot table and lookup

The slot table is sized to roughly twice the segment's nominal capacity (the
nearest power of two above `capacity * 2`), giving it a maximum 50% load
factor. At that load factor, linear probing — the simplest hash-table
collision strategy — averages 1.5 probes per lookup and stays well-behaved
at the tail. Insertion refuses new entities once the segment hits capacity
rather than allowing degenerate probe lengths. See
[ADR-003](adr/003-output-vector-uses-declaration-order.md) for the
field-ordering rationale and the linear-probing decision.

A lookup takes an entity ID, hashes it to 64 bits, and probes the slot table
starting at `hash & (slot_count - 1)`. Each probe checks the slot's flags
(empty means the entity isn't here), compares the stored hash, and on hash
match compares the actual ID bytes from the string pool to disambiguate
collisions. The pure-Python implementation in `quorin.layout.lookup` exists
as a parity reference; the Numba-compiled `_lookup_core` in
`quorin._internal.lookup_kernel` is what production code actually calls. The
two must produce byte-identical results for every input — a property test
covering 200 randomly generated scenarios per run enforces that contract.

### 4.4 The Numba-compiled read path

Numba is a Python library that compiles a subset of Python — basically
loops over numeric arrays — into native machine code via LLVM. Quorin uses
Numba in exactly two places: the single-entity assemble kernel
(`quorin.assembly.assemble`) and the batch assemble kernel
(`quorin.assembly.assemble_batch`). Plus their dependencies: the lookup
kernel and the BLAKE2b hash kernel. Everything else stays pure Python.

The single-entity kernel is roughly: given the segment's byte view, the
pre-computed offset table arrays, and the entity's name hash, do the linear-
probe lookup, then copy each field's bytes from the segment into a
pre-allocated output buffer. The whole thing fits in a few dozen lines of
Numba-compatible Python and runs at native speed. Cold compile cost is around
300 ms the first time; subsequent process starts skip it via Numba's on-disk
cache.

Numba has discipline costs. The kernel can't allocate Python objects, can't
call `hashlib`, can't catch exceptions, and can't read structured-dtype
fields by name from inside `@njit` code (Numba 0.60's structured-field access
is fragile in some builds). Quorin works around all of these: the kernel
takes the segment as a flat `uint8` array plus scalar offset constants
derived from the slot dtype at module load and runtime-asserted. Hardcoded
magic offsets in the kernel are forbidden — invariant #15. See
[ADR-007](adr/007-batch-assembly.md) for the batch kernel design and
[ADR-017](adr/017-lookup-jit.md) for the lookup-jit + Numba BLAKE2b decision.

The Numba dependency is isolated: only `quorin.assembly`, `quorin._internal.lookup_kernel`,
`quorin._internal.hash_kernel`, and `quorin._internal.insert_kernel` import
Numba. Any other Quorin module that imported them transitively would force
every consumer to pay Numba's ~200 ms LLVM initialization cost. Invariant
#11 prevents this; a module-hygiene test enforces it.

### 4.5 The buffer pool

`np.empty(N, dtype=np.float32)` for small N takes roughly 80 nanoseconds.
That's not a lot, but at 50,000 requests per second it's still 50,000
allocations per second feeding the garbage collector. Eliminating that
allocation per request is the buffer pool's job: it pre-allocates a deque of
output buffers at construction time and serves checkouts from the deque
under a lock-free `popleft` / `append` protocol that's safe under CPython's
GIL.

The honest disclosure is that on commodity hardware (GitHub Actions
ubuntu-latest), the pool's per-call protocol overhead — context manager
entry, deque operation, optional zero-on-return — adds **2–4 microseconds**
to a single-entity assemble. That's measured in ADR-005's Step 16c amendment
and contradicts the original "latency-neutral" claim in the build plan. The
pool ships anyway because:

1. It eliminates one ndarray allocation per call, which materially reduces
   garbage-collection pressure even when per-call latency is roughly neutral
   (and on real GC-heavy workloads, fewer allocations means a tighter
   p99.9 / p99.99 tail).
2. It bounds steady-state memory at a known ceiling, useful for capacity
   planning.
3. The batch path's pool (separate class, `BatchBufferPool`) wins decisively
   because allocating a `(1000, 327)` float32 array per batch call costs
   tens of microseconds — the pool's overhead is amortized.

Pool is default for batch; opt-in for single-entity workloads where the
caller has measured the tradeoff. See [ADR-005](adr/005-buffer-pool-lock-free-prealloc-capped.md).

### 4.6 The batch read path

The batch kernel is shaped like the single-entity kernel but loops over many
entities, packing each row's output into a row of a 2D buffer. For batches
above an adaptive threshold (computed at module load from
`numba.get_num_threads()`), Numba's `parallel=True` mode dispatches the row
loop across threads. Below the threshold, the serial kernel wins because
parallel-mode thread-pool spinup overhead dominates for small N.

The honest measured speedup at N=1000 on commodity hardware is **1.5–1.7×**
versus N individual single-call assembles. The original target was 5×; the
gap is explained by ADR-007's Step 16c amendment: the older Xeons in GitHub
Actions runners have ~30 MB of L3 cache shared across cores, so a 1000-row
batch's working set spills to DRAM. Modern desktop CPUs with more L3 should
recover much of the gap. See [ADR-007](adr/007-batch-assembly.md) for the
parking-lot path to higher ratios (Numba-fied hash loop, parallel-threshold
re-tuning, eventual SIMD).

### 4.7 The WAL producer

The WAL — write-ahead log — exists for crash safety. A producer that wrote
directly to shared memory could crash mid-write and leave the segment in a
half-updated state. Routing through a Redis Stream gives every write a
durable ordering: even if the producer dies, the message is committed in
Redis, and the WAL consumer applies it on next read.

`quorin.wal.WALProducer` is the user-facing API. It takes a row as keyword
arguments, validates them against a pydantic model that's memoized per
schema (first call ~1 ms, subsequent ~50 ns dict lookup), msgpack-encodes
the field values into a positional list (sorted by field-name hash to match
the consumer's wire format), and `XADD`s the message. Async by default; an
opt-in `write_sync(timeout=0.1)` polls a Redis side-table that the consumer
sets after applying the message, giving "read your own writes" semantics at
the cost of one consumer-cycle round trip (5–50 ms typical).

Several allocation-discipline choices keep the producer at a 10k-writes-per-
second target: a single reusable msgpack `Packer` instance per producer
(faster than per-call `msgpack.packb`); pre-warmed Prometheus label children
at producer init (avoiding the dictionary-allocation in
`Histogram.labels(...)` on first use); module-level bytes constants for
the four XADD field keys; and `getattr(model, name)` instead of
`model.model_dump()` to avoid an intermediate dict. See
[ADR-008](adr/008-wal-producer-design.md).

### 4.8 The WAL consumer

`quorin.wal_consumer.WALConsumer` is an async coroutine that reads from the
`quorin:wal` Redis Stream as a member of a consumer group, validates each
message against the pydantic model, and calls `quorin.layout.insert` to
apply the row to the shared-memory segment. The consumer is the only writer
to the segment — a single-writer invariant that makes the lock-free read
path possible.

Two durability signals are decoupled. After `layout.insert` succeeds, the
consumer immediately `SET`s a per-message side-table key
(`quorin:processed:{msg_id}`) — this is what `WALProducer.write_sync`
polls. The `XACK` acknowledgment to the consumer group fires later, only
after the per-flush `OfflineWriter.flush()` returns. This means the online
store is durable before `write_sync` returns; the offline store reaches
durability later (eventually consistent). On consumer restart, the unacked
messages in the consumer's pending-entries-list are replayed; `layout.insert`
is idempotent, so re-applying is safe. See [ADR-009](adr/009-wal-consumer-design.md).

### 4.9 The offline store

`quorin.offline.ParquetDatasetStore` is the training-data side of Quorin.
The WAL consumer appends every applied row to it; periodic `flush()` calls
write a Parquet file to disk per flush. The dataset is append-only: no row
is ever rewritten, so a crash mid-flush loses at most one file's worth of
buffered rows.

The store does two non-trivial things. First, it writes the four
"infrastructure" columns (`entity_id`, `event_time_ns`, `msg_id_ms`,
`msg_id_seq`) with carefully chosen Parquet encodings — dictionary for
`entity_id` (high repetition), delta-binary-packed for the monotonic
`msg_id_*` columns (small bit width). Encoding the msg_id columns as
dictionary by mistake (which is what PyArrow tries by default) would inflate
the file size by 5–10×. See [ADR-010](adr/010-parquet-offline-store.md) for
the PyArrow-specific gotcha that locked this in.

Second, it serves **point-in-time reads** for training-data assembly. Given
a list of `(entity_id, as_of_timestamp)` query pairs, the
`read_point_in_time` method returns each entity's most-recent feature values
that existed at or before its `as_of_timestamp`. The implementation is a
hand-rolled asof-join using `numpy.searchsorted` over the sorted
`event_time_ns` column, with an inline per-query lookback check that
prevents leakage when query timestamps span a wide range. See
[ADR-011](adr/011-point-in-time-reads.md).

### 4.10 Hydration

Cold-start recovery: Redis is empty (fresh deployment, full flush), the WAL
stream is empty, but the Parquet dataset still has the latest features for
every entity. `quorin.hydration.hydrate` reads the offline store's "latest
features per entity" view, builds a fresh segment, populates it via a
Numba-compiled bulk-insert kernel, and atomically registers it as the
current segment for the schema. For 1 million entities × 50 fields, this
takes around 2–3 seconds on bare metal (about 10 seconds on WSL2, where the
`/dev/shm` cold-page-fault throughput is the limiting factor). See
[ADR-012](adr/012-hydration.md).

### 4.11 Schema evolution

When you change a schema — add fields, widen a dtype — the in-RAM segment
needs to be rebuilt with the new layout while live readers continue against
the old one. `quorin.evolution.upgrade_schema` orchestrates this: allocate a
new segment with the new schema; vectorized-translate every row from the old
segment into the new one (column by column, using NumPy strided views, no
Python row loop); briefly pause the WAL consumer; atomically flip the
`quorin:schema:{name}:current` Redis pointer via a Lua script that does a
compare-and-swap; clear the pause; wait for at least one consumer to attach
to the new segment.

The atomic flip is the load-bearing primitive. Before the flip, all readers
see OLD; after the flip, new readers see NEW; in-flight readers continue
against OLD until they close (per-process refcounts make this safe). The
WAL consumer's pause + reopen logic is a safety net for the rare case where
an operator runs an evolution without first draining the WAL stream — the
consumer detects schema mismatch as a "poison pill," logs an alert, and
declines to XACK rather than corrupt the new segment with old wire-format
data. See [ADR-014](adr/014-schema-evolution.md).

### 4.12 The watchdog

Worker processes can die without notice — SIGKILL, OOM kill, hardware
failure. When a worker dies mid-read, its per-process refcount on the
segment never gets decremented, so the segment can't be cleaned up even
after every other process closes. Without intervention this leaks
`/dev/shm` segments forever.

`quorin.watchdog` is a separate process that runs forever, polling every
50 milliseconds. It reads a Redis hash where every live worker writes a
heartbeat (PID + monotonic timestamp + create_time), detects stale entries
(consecutive missed advances of `wall_time_ns`), cross-checks each suspected-
dead PID against `psutil.pid_exists()` plus the cached create_time (to
guard against PID reuse), and on confirmed death, runs a Lua script that
atomically decrements every refcount the dead PID held, queues newly-zero
refcounts for cleanup, and removes the heartbeat entry. A separate drain
step pops the cleanup queue and calls `posix_shm.unlink` on each segment
name.

The cross-check uses **exact equality** on `create_time_ns`, not tolerance.
On every Linux kernel `psutil.Process(pid).create_time()` returns
bit-identical floats for two reads of the same PID; PID reuse always
differs by at least one jiffy. Tolerance-based comparison would silently
misclassify on tickless or 1000 Hz kernels where one jiffy is one
millisecond. See [ADR-013](adr/013-watchdog.md).

### 4.13 Metrics and logging

Prometheus instruments register at module load. If you don't start
`prometheus_client.start_http_server`, the counters still increment in
memory and are visible in tests. The instruments cover read latency,
GC pause durations, WAL lag, pool miss rate, watchdog cleanup counts,
schema upgrade durations, and a dozen others. See
[`quorin/metrics.py`](../quorin/metrics.py) for the full list.

Logging uses `structlog` with a JSON renderer by default, so log lines are
structured and machine-parseable. Every Quorin subsystem binds a
`component` field at startup so log queries can filter by subsystem.

---

## 5. Concurrency and crash safety

### 5.1 Single writer per segment

Only the WAL consumer (and the hydration coordinator on cold start) calls
`quorin.layout.insert`. Producers don't write to shared memory directly.
This invariant is what makes the lock-free read path correct: readers never
see a partial write because writes are serialized by the consumer's
single-coroutine event loop.

Concurrent reads are unbounded — any number of processes can `assemble`
simultaneously without coordination. The slot table is read-only during
serving (the only mutating operation, `insert`, runs in the consumer
process). The output buffer pool's deque uses CPython's GIL atomicity for
lock-free `popleft` / `append`.

### 5.2 Per-open refcounting

Refcounts in Redis are incremented exactly once when a process opens a
segment and decremented exactly once when it closes. The hot read path
**never** touches Redis — every per-read Redis call would cost more than
the entire 5-microsecond budget. See [ADR-002](adr/002-per-open-refcounting.md).

### 5.3 Cursors advance before slot writes

Inside `insert`, the writer cursors (`next_free_row_index`,
`string_pool_cursor`) update before the slot is marked occupied. A crash
mid-insert leaks a few bytes of pool / row storage but cannot corrupt the
slot table — a replay sees the slot as empty and allocates fresh storage.
Invariant #8.

### 5.4 The 16-byte primary header is fixed forever

`magic(4) + version(4) + crc32(4) + capacity(4)`. Don't change byte layout.
Adding a Quorin-level field requires bumping the magic to `b"PYFG2"` —
which is a wholly new wire format. This is the one part of the on-disk
representation that's truly locked.

### 5.5 Pinned hashes

BLAKE2b digest_size=8, little-endian uint64, for both schema field names
and entity IDs. Pinned-hash regression tests assert known hash outputs.
Changing the algorithm would silently break every persisted segment.
Invariant #5.

### 5.6 Garbage collection management

CPython's garbage collector pauses every thread when it runs. Quorin's
`gc_manager` (Step 7) provides an instrumented callback that records pause
durations to a Prometheus histogram, plus an opt-in `freeze()` helper that
moves long-lived objects out of generation 0 / 1 so they don't get scanned.
Honest disclosure: combining the callback with `freeze()` shows a measured
+31% interaction effect on tail latency that we haven't been able to
localize. The two features are documented as mutually exclusive at the API
level via a UserWarning. See [ADR-006](adr/006-gc-management.md).

---

## 6. Performance methodology

Every benchmark number Quorin publishes has a venue, methodology, and source
JSON committed in the repo. The headline 4.48 µs p99 is the median of 20
fresh-subprocess runs on GitHub Actions ubuntu-latest at workflow run
25394553451. Single-process pytest-benchmark numbers are gross-regression
detectors, not spec-band enforcers — cold-cache benches in particular have
3–4× run-to-run variance in single-process runs because surrounding bench
activity warms different cache regions. The N=20 fresh-subprocess
orchestrator (`benchmarks/runs/repeat.py`) absorbs this variance and is the
canonical source for any number that goes in user-facing documentation.

The ubuntu-latest Xeons are 1.5–3× slower than modern desktop CPUs on
cache-bound and bandwidth-bound benchmarks. Bare-metal extrapolation is the
right framework for predicting performance on different hardware. README
numbers are floors, not ceilings. See [ADR-015](adr/015-benchmark-methodology.md)
for the full methodology including the cache-clobber strategy, the multi-
percentile gate model, and the calibration discipline that retightened
gates after the canonical Tier-2 N=20 baselines were established.

---

## 7. Where to look next

- **To use the library:** [USAGE.md](USAGE.md) — runnable code examples for
  every public path (sync demo, WAL production flow, batch reads, point-in-
  time reads, hydration, schema upgrade, operations).
- **For the public API surface:** [API.md](API.md) — one section per public
  module with a curated symbol table, links to source, links to relevant
  ADRs.
- **For design rationale on any specific decision:** the [ADR archive](adr/)
  — 17 numbered records, one per load-bearing decision, written alongside
  the step that introduced them.
- **For the project's invariants and gotchas:** [`CLAUDE.md`](../CLAUDE.md)
  in the repo root — the canonical source for the 18 non-negotiable
  invariants and the long list of "we already paid for this bug, don't
  reintroduce it" gotchas.

