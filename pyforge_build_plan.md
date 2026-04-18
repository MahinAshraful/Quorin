# Pyforge — Build Priority & Development Roadmap

## How to Use This Document

This is your Trello/Jira equivalent. Every phase has a clear goal, a definition of "done," and a reason for its priority. Build in order. Do not skip phases. The earlier phases are not "easy warmup" — they are the foundation everything else sits on.

The rule: **A phase is not done until you can write a test that proves it works.** Not "I think it works." A test that fails if it's broken.

---

## Before You Write a Single Line of Code

### Set up your environment

```
Python 3.12+ (required — uses sys.monitoring for lower-overhead tracing)
Redis 7.2+ with AOF enabled
PyArrow 14+
NumPy 1.26+
Numba 0.59+
Pydantic v2
pytest + pytest-benchmark
py-spy (install separately, not via pip in your venv)
psutil
docker + docker-compose
```

### Create your docker-compose.yml first

Redis needs to run with AOF (append-only file) persistence enabled. Without it, Redis loses data on restart. This file should exist before any Python code:

```yaml
version: "3.9"
services:
  redis:
    image: redis:7.2-alpine
    command: redis-server --appendonly yes --appendfsync always
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
volumes:
  redis_data:
```

Run `docker-compose up -d` and keep it running throughout development.

### Create your repo structure now

```
pyforge/
├── pyforge/
│   ├── __init__.py          (empty for now)
│   ├── schema.py            (Phase 1)
│   ├── registry.py          (Phase 2)
│   ├── serving.py           (Phase 3)
│   ├── assembly.py          (Phase 3)
│   ├── parquet.py           (Phase 4)
│   ├── wal.py               (Phase 4)
│   └── watchdog.py          (Phase 5)
├── tests/
│   ├── conftest.py
│   ├── test_schema.py       (Phase 1)
│   ├── test_registry.py     (Phase 2)
│   ├── test_serving.py      (Phase 3)
│   ├── test_parquet.py      (Phase 4)
│   └── test_crash.py        (Phase 5)
├── benchmarks/
│   └── (empty for now)
├── docker-compose.yml
├── pyproject.toml
└── README.md                (start writing this NOW, update as you go)
```

Write the README as you build. Don't leave it for the end. Every component gets a section as you build it.

---

## Phase 1 — Schema Definition and Offset Table Compiler

**Goal:** Be able to define a feature schema in Python and have Pyforge compile it into a pre-computed offset table.

**Why this is first:** Everything else depends on knowing the exact memory layout. The schema system is the foundation. If the offset table is wrong, the serving path reads garbage from wrong memory addresses.

**What to build:**

`pyforge/schema.py` — the only file in this phase.

1. A `FeatureField` class that holds: field name (string), dtype (from a small enum you define: float32, float64, int32, int64), shape (tuple of ints, default is `(1,)` for scalar fields).

2. A `FeatureSchema` base class that validates that all fields have unique names and known dtypes.

3. A `compile_schema(schema_class)` function that returns a NumPy structured array — the offset table — with these columns:
   - `name_hash`: uint64 — a hash of the field name
   - `byte_offset`: uint64 — where this field starts in the shared memory segment
   - `dtype_code`: uint8 — an integer representing the dtype
   - `element_count`: uint32 — total number of elements (shape[0] * shape[1] * ...)
   - `byte_count`: uint32 — total bytes for this field

4. The offsets must be computed correctly, with each field start address aligned to 64 bytes (round up to the next multiple of 64).

5. A `total_segment_size(schema_class)` function that returns the total bytes needed for the segment: 16 bytes for the header plus the sum of all field byte counts (after alignment).

**Tests to write:**

```python
# test_schema.py

def test_field_byte_count_float32():
    field = FeatureField("x", dtype.float32)
    table = compile_schema(...)
    assert table[0]["byte_count"] == 4  # float32 = 4 bytes

def test_field_byte_count_embedding():
    field = FeatureField("embedding", dtype.float32, shape=(128,))
    # 128 * 4 bytes = 512 bytes
    assert ...

def test_offsets_are_64_byte_aligned():
    class Schema(FeatureSchema):
        fields = [
            FeatureField("a", dtype.float32),  # 4 bytes
            FeatureField("b", dtype.float32),  # should start at byte 64, not byte 4
        ]
    table = compile_schema(Schema)
    assert table[1]["byte_offset"] % 64 == 0

def test_duplicate_field_names_raise():
    with pytest.raises(ValueError):
        class Schema(FeatureSchema):
            fields = [
                FeatureField("a", dtype.float32),
                FeatureField("a", dtype.float32),  # duplicate!
            ]

def test_total_segment_size_includes_header():
    class Schema(FeatureSchema):
        fields = [FeatureField("x", dtype.float32)]
    size = total_segment_size(Schema)
    assert size >= 16 + 4  # header + one float32
```

**Definition of done:** All tests pass. `compile_schema` returns a NumPy structured array. You can call it on a schema with mixed dtypes and shapes and get correct byte offsets.

**Time estimate:** 2-3 days.

---

## Phase 2 — Shared Memory Allocation and Registry

**Goal:** Be able to register a schema, have Pyforge allocate a shared memory segment with the right layout, and have multiple processes read from it.

**Why this is second:** The schema system from Phase 1 is useless without the infrastructure that actually creates the shared memory. Phase 2 is where shared memory comes alive.

**What to build:**

`pyforge/registry.py`

1. A `Registry` class that holds: a Redis client, a dict mapping schema names to their compiled offset tables, a dict mapping schema names to open `SharedMemory` objects.

2. A `register(schema_class)` method that:
   - Compiles the schema (using Phase 1)
   - Allocates a `SharedMemory` segment with the right size
   - Writes the 16-byte header (version + CRC32 of schema definition)
   - Stores the segment name in Redis: `SET pyforge:schema:{name}:current_segment {segment_name}`
   - Sets the reference count: `SET pyforge:refcount:{segment_name} 1`
   - Stores the offset table in the registry's local dict

3. A `write(schema_class, entity_id, values: dict)` method that:
   - Looks up the segment for this schema
   - Looks up where in the segment this entity_id's data lives (for now, store one entity per segment — you'll extend this later)
   - Writes each field value to the correct byte offset

4. A `read(schema_class, entity_id)` method that returns a NumPy array view of the entity's feature vector in shared memory — not a copy, a view.

**The reference count increment/decrement:** Every call to `read()` should `INCR` the refcount at the start and `DECR` at the end. Use Python's `contextlib` to implement this cleanly — a context manager that handles the increment/decrement even if an exception is thrown.

**Checkpoint test — the multi-process test:**

This is the most important test in Phase 2. It proves shared memory actually works across processes:

```python
# test_registry.py

import multiprocessing
import numpy as np

def writer_process(schema_name, values):
    # This runs in a separate process
    registry = Registry(redis_url="redis://localhost:6379")
    registry.register(UserFeatures)
    registry.write(UserFeatures, "user_001", values)

def test_shared_memory_cross_process():
    values = {"age_normalized": 0.73, "session_count_7d": 42, "ltv_score": 1.5}
    
    p = multiprocessing.Process(target=writer_process, args=("UserFeatures", values))
    p.start()
    p.join()
    
    # Now read from this process — data should be there
    registry = Registry(redis_url="redis://localhost:6379")
    registry.register(UserFeatures)
    result = registry.read(UserFeatures, "user_001")
    
    assert abs(result["age_normalized"] - 0.73) < 1e-6
    assert result["session_count_7d"] == 42
```

**Definition of done:** The multi-process test passes. You can write from one process and read from another with zero copying.

**Time estimate:** 4-5 days.

---

## Phase 3 — The Lock-Free Serving Path (The Core)

**Goal:** Build the fast serving path — vectorized batch assembly, Numba compilation, pre-allocated buffer pool, GC pause elimination.

**Why this is third:** Phases 1 and 2 give you correctness. Phase 3 gives you performance. Build correctness first, then optimize it.

**What to build:**

`pyforge/serving.py` and `pyforge/assembly.py`

### Part A: The buffer pool (serving.py)

A `BufferPool` class that:
- Pre-allocates N output buffers (NumPy float32 arrays) of a given shape at initialization
- Exposes a context manager `checkout()` that pops a buffer from the pool, yields it, and returns it on exit
- Refills the pool asynchronously when pool size drops below a threshold (use `asyncio` + a background coroutine for this)
- Logs a warning (not an exception) if the pool is exhausted and a fresh allocation is needed

### Part B: The Numba assembly function (assembly.py)

Write this as a standalone module. The function signature:

```python
import numba
import numpy as np

@numba.njit(cache=True)  # cache=True saves compiled code to disk, avoids recompilation on restart
def assemble_vector(
    shm_buffer: np.ndarray,   # uint8 view of the shared memory segment
    byte_offsets: np.ndarray,  # int64 array of field start byte offsets
    byte_counts: np.ndarray,   # int64 array of field byte counts
    output: np.ndarray,        # float32 output buffer (pre-allocated)
) -> None:
    cursor = 0
    for i in range(len(byte_offsets)):
        start = byte_offsets[i]
        count = byte_counts[i]
        for j in range(count):
            output.view(np.uint8)[cursor + j] = shm_buffer[start + j]
        cursor += count
```

Important: write a non-Numba Python version first, test it for correctness, then write the Numba version and verify it produces identical output.

### Part C: Batch assembly (serving.py)

A `get_batch(schema_class, entity_ids, output=None)` method on the Registry class that:
- Accepts a list of entity_ids
- Returns a 2D NumPy array of shape `(len(entity_ids), n_features)` where row i is entity_ids[i]'s feature vector
- Uses the buffer pool for the output buffer
- Calls the Numba assembly function for each entity (or the vectorized version if you can work out the indexing)

### Part D: GC management

In `serving.py`, add a `ServingThread` class that:
- Calls `gc.disable()` at startup
- Registers a `gc.callbacks` handler that records pause durations
- Starts a background thread that calls `gc.collect()` on a 100ms timer

**Benchmark to run at end of Phase 3:**

At the end of this phase, you should run your first benchmark and commit the results:

```bash
py-spy record -o benchmarks/results/hot_path_flamegraph.svg -- python benchmarks/bench_serving.py
pytest benchmarks/bench_serving.py --benchmark-autosave
```

The flamegraph should show the Numba function as the dominant cost in the hot path — not Python overhead.

**Definition of done:** `get_batch` returns correct results. The Numba and Python implementations produce identical output. A flamegraph exists and is committed. p99 latency for single-entity reads is below 50µs (you can tighten this once you have baseline numbers).

**Time estimate:** 1-2 weeks. The Numba compilation issues will take time to debug.

---

## Phase 4 — The Offline Store + WAL

**Goal:** Build append-only Parquet writing, point-in-time reading, and crash-safe writes via Redis Streams WAL.

**Why this is fourth:** The offline store is what makes Pyforge a complete system rather than a toy. Without it, there's no answer to "where do features come from for training?" But it comes after the serving path because the serving path is the project's core value.

**What to build:**

`pyforge/parquet.py` and `pyforge/wal.py`

### Part A: Append-only Parquet writer (parquet.py)

A `ParquetStore` class that:

1. Holds a base directory path and creates one `.parquet` file per schema
2. Has a `write(schema_class, entity_id, values, event_time, processing_time)` method that:
   - Creates a PyArrow `RecordBatch` with columns: `entity_id` (string), `event_time` (timestamp), `processing_time` (timestamp), plus one column per feature field
   - Appends the record batch to the Parquet file using `pyarrow.parquet.ParquetWriter` in append mode
3. Has a `read_point_in_time(schema_class, entity_ids, as_of_time)` method that:
   - Reads the Parquet file for this schema
   - For each entity_id in entity_ids, returns the most recent row with `event_time <= as_of_time`
   - Implements this using `pyarrow.compute` functions (not pandas) for speed

The point-in-time read using PyArrow compute:
```python
# conceptually:
# 1. filter rows where entity_id in entity_ids AND event_time <= as_of_time
# 2. group by entity_id
# 3. take the row with max(event_time) per group
```

4. A `detect_corruption()` method that checks whether the last row group has a valid footer CRC, and a `truncate_to_last_valid()` method that removes corrupted trailing data.

### Part B: The WAL (wal.py)

A `WALConsumer` class that:

1. Runs as an `asyncio` coroutine (not a thread — you want to understand async/await at this depth)
2. Reads messages from a Redis Stream: `XREADGROUP GROUP pyforge_consumers consumer_1 COUNT 10 BLOCK 100 STREAMS pyforge:wal >`
3. For each message:
   - Checks if `message_id <= last_processed_id` (idempotency check). If so, skip and acknowledge.
   - Writes the feature data to Parquet
   - Updates `last_processed_id` in Redis
   - Acknowledges the message: `XACK pyforge:wal pyforge_consumers {message_id}`
4. On startup, calls `XPENDING` to check for unacknowledged messages from a previous run and replays them first

A `WALProducer` mixin that adds a `write_via_wal(schema_class, entity_id, values)` method to the Registry that:
1. Writes to the Redis Stream: `XADD pyforge:wal * entity_id {entity_id} schema {schema_name} values {serialized_values}`
2. Updates the Redis online store
3. Returns immediately (the Parquet write happens asynchronously)

**The crash test — the most important test in Phase 4:**

```python
# test_parquet.py

def test_wal_crash_recovery():
    # Write 10 feature updates via WAL
    for i in range(10):
        registry.write_via_wal(UserFeatures, f"user_{i:03d}", {...})
    
    # Kill the WAL consumer abruptly (simulate crash)
    consumer.kill()
    
    # Verify: Parquet has fewer than 10 rows (some weren't written before crash)
    rows = parquet_store.read_all(UserFeatures)
    assert len(rows) < 10
    
    # Restart consumer
    new_consumer = WALConsumer(...)
    await new_consumer.run_until_caught_up()
    
    # Verify: all 10 rows are now in Parquet, no duplicates
    rows = parquet_store.read_all(UserFeatures)
    assert len(rows) == 10
    assert len(set(row["entity_id"] for row in rows)) == 10  # no duplicates
```

**Definition of done:** The crash recovery test passes. Point-in-time reads return correct values. No data is lost or duplicated on crash + restart.

**Time estimate:** 1-2 weeks. The async WAL consumer and idempotency logic are non-trivial.

---

## Phase 5 — Watchdog Process + Schema Evolution

**Goal:** Handle the two remaining hard questions: what happens when a process crashes (shared memory leaks), and what happens when a schema changes (version evolution).

**Why this is fifth:** This is the "polish" phase — it makes Pyforge robust in adversarial conditions. The core system is fully functional after Phase 4. Phase 5 is what separates a credible project from a toy.

**What to build:**

`pyforge/watchdog.py` and additions to `registry.py`

### Part A: The Watchdog (watchdog.py)

A `Watchdog` class that runs as a separate `multiprocessing.Process`:

1. On startup, creates a Redis consumer group for watchdog messages
2. Loop every 100ms:
   - Read all entries from `pyforge:heartbeats` Redis Hash
   - For each `{pid: last_heartbeat_timestamp}` entry:
     - If `now - last_heartbeat_timestamp > 500ms` AND `not psutil.pid_exists(pid)`:
       - Decrement all refcounts this PID held (stored in `pyforge:pid_segments:{pid}`)
       - Process the cleanup queue: for each segment name in `pyforge:cleanup_queue` with refcount == 0, call `SharedMemory(segment_name).unlink()`
       - Remove the PID from `pyforge:heartbeats`

Every process that opens a shared memory segment must:
- Write `{pid: current_timestamp}` to `pyforge:heartbeats` at startup
- Update the timestamp every 100ms (a background thread)
- Write the segment name to `pyforge:pid_segments:{pid}` (a Redis Set)

### Part B: Schema evolution (additions to registry.py)

A `upgrade_schema(old_schema_class, new_schema_class)` function that:

1. Validates the upgrade is safe (no field removed, dtypes only narrowed if explicitly allowed)
2. Allocates a new shared memory segment for `new_schema_class`
3. Copies current values from the old segment to the new one (for fields that exist in both)
4. Runs the Redis Lua script that atomically:
   - Updates `pyforge:schema:{name}:current_segment` to the new segment name
   - Increments the new segment's refcount
   - Decrements the old segment's refcount
   - Adds the old segment to the cleanup queue if refcount is now 0
5. Returns the new registry

**Schema evolution test:**

```python
def test_schema_evolution_live_readers():
    # Start a reader process that continuously reads from v1
    reader = start_background_reader(UserFeaturesV1)
    
    # While reader is running, upgrade to v2
    registry.upgrade_schema(UserFeaturesV1, UserFeaturesV2)
    
    # Reader should not crash
    time.sleep(0.5)
    assert reader.is_alive()
    
    # Old segment should still exist (reader holds a reference)
    assert old_segment_exists()
    
    # Stop reader
    reader.stop()
    reader.join()
    
    # Now old segment should be cleaned up by watchdog within 500ms
    time.sleep(0.6)
    assert not old_segment_exists()
```

**Definition of done:** Schema evolution test passes. The watchdog test (kill a process and verify the segment is cleaned up) passes. No shared memory leaks under any crash scenario you can construct.

**Time estimate:** 1-2 weeks.

---

## Phase 6 — Benchmarks + Profiling + README

**Goal:** Generate all benchmark numbers, create all flamegraphs, write the complete README with real data.

**This is not optional.** This is 20-30% of the project's value to someone reading your GitHub.

### Benchmarks to run and commit

Run all of these and commit the results to `benchmarks/results/`:

```bash
# Hot path latency — the headline benchmark
pytest benchmarks/bench_serving.py --benchmark-autosave --benchmark-name hot_path

# Batch assembly — shows scaling behavior
pytest benchmarks/bench_batch.py --benchmark-autosave

# GC pause impact
pytest benchmarks/bench_gc.py --benchmark-autosave

# Buffer pool impact
pytest benchmarks/bench_pool.py --benchmark-autosave

# Hydration speed
pytest benchmarks/bench_hydration.py --benchmark-autosave

# Flamegraphs
py-spy record -o benchmarks/results/hot_path.svg -- python benchmarks/run_hot_path.py
py-spy record -o benchmarks/results/batch_assembly.svg -- python benchmarks/run_batch.py
```

### What the README must contain

1. **One paragraph problem statement** — why the infrastructure around ML models is slow
2. **One paragraph solution** — what Pyforge does differently
3. **A table of benchmark results** — p50/p95/p99/p999 for shared memory path vs Redis path vs naive Python
4. **An architecture diagram** — even an ASCII one
5. **A quickstart** — 10 lines of Python that show how to register a schema, write a value, and read it back
6. **The interview questions section** — "Common questions about this project" with real answers. This signals you've thought deeply about the design.
7. **Hardware specs** — exactly what machine the benchmarks ran on (e.g. "MacBook Pro M3 Pro, 18GB RAM, Redis 7.2 running in Docker")

### The "common questions" section in the README

Include these and answer them:

- Why shared memory instead of just using Redis?
- How do you handle process crashes without memory leaks?
- What happens when a schema changes while processes are actively reading?
- Why Numba and not Cython or a C extension?
- What are the durability guarantees?
- Why not just use Feast?

Answering these in the README does two things: it signals you've thought deeply about design tradeoffs, and it prepares you for every interview question about this project.

**Time estimate:** 1 week.

---

## Phase Summary

| Phase | What | Core Skill Demonstrated | Est. Time |
|-------|------|------------------------|-----------|
| 0 | Setup, repo, Docker | Engineering discipline | 1 day |
| 1 | Schema + offset table | NumPy internals, type systems | 2-3 days |
| 2 | Shared memory + registry | OS-level memory management, Redis | 4-5 days |
| 3 | Serving path + Numba + pool | Performance engineering, Numba, asyncio | 1-2 weeks |
| 4 | Parquet + WAL | Async Python, crash safety, idempotency | 1-2 weeks |
| 5 | Watchdog + evolution | Systems thinking, concurrency | 1-2 weeks |
| 6 | Benchmarks + README | Technical communication | 1 week |

**Total: 2-3 months of serious part-time work, or 6-8 weeks full time.**

---

## When to Stop and Reassess

After Phase 3, run your benchmark. If the numbers aren't there — if shared memory isn't significantly faster than Redis for your feature sizes — stop and understand why before proceeding. The whole project rests on the performance thesis being real. If the numbers don't support it, you need to either fix the implementation or reframe the project. Don't build five more phases on top of a broken foundation.

After Phase 4, show it to someone technical. Not for validation — for attack questions. Ask them to try to find holes in the WAL design, the idempotency logic, the crash recovery. Every question they ask that you can't answer is something to fix before the project is done.

---

## Things That Will Go Wrong (and How to Handle Them)

**Numba compilation errors on first run:** Expected. Numba is strict about types. Your first version will probably have type mismatches. Fix them systematically — Numba's error messages are actually helpful.

**Shared memory not cleaning up between test runs:** Add a `conftest.py` fixture that deletes all `pyforge:*` Redis keys and lists all shared memory segments before each test. Otherwise test pollution will drive you insane.

**The WAL consumer falling behind:** If you write features faster than the consumer drains the stream, the stream grows unbounded. Add a `MAXLEN` to the XADD command and document what happens when it fills (backpressure / write rejection). This is a design decision, not a bug.

**PyArrow version incompatibilities:** Pin your PyArrow version in `pyproject.toml`. PyArrow has breaking API changes between major versions. `pyarrow==14.*` is stable.

**GC disable causing memory leaks in tests:** When you `gc.disable()` in tests, Python won't collect circular references. This can cause your test suite to slowly leak memory. Use `gc.enable()` in test teardown or use a separate process for the serving benchmark.
