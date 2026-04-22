# Pyforge — Project Specification

## What This Document Is

This is a complete technical description of the Pyforge project — what it is, why it exists, every component, every design decision, and why each decision was made. When you take this to another chat and ask Claude to explain something, point to the specific section. Nothing here requires prior knowledge. Every concept is explained from first principles.

---

## The One-Sentence Description

Pyforge is a Python library that makes reading machine learning features extremely fast — specifically by bypassing the normal slow path of fetching data from a database, converting it, and copying it into memory, and replacing it with a path that reads directly from memory that multiple processes already share.

---

## The Problem Pyforge Solves

### What is an ML feature?

When a machine learning model makes a prediction, it needs numbers as input. These numbers are called **features**. For example: a model predicting whether a user will click an ad might need:

- How many times the user visited the site in the last 7 days (a number)
- The user's age bracket (a number from 0-5 representing age groups)
- The hour of day (0-23)
- An embedding vector — 128 numbers representing the user's behavior pattern

All of these get assembled into one flat array of numbers, which gets fed to the model. That array is the **feature vector**.

### Where does the slowness come from?

The model itself — the mathematical computation — is fast. XGBoost running on 200 features takes roughly 200 microseconds (0.2 milliseconds). That's fine.

The problem is everything *around* it:

1. You need to go fetch the features from somewhere (usually Redis, a fast in-memory database)
2. Redis stores data as raw bytes — you need to convert those bytes back into numbers
3. You need to assemble those numbers into a contiguous array in memory
4. You call the model
5. You serialize the result back

Steps 1-3 and 5 — the "infrastructure" around the model — routinely take **5 to 50 milliseconds**. That's 25 to 250 times slower than the model itself.

At scale this matters enormously. If you're serving 50,000 predictions per second on one machine, and each request takes 10ms of infrastructure overhead, you need 500 cores just for overhead. Reduce that to 0.5ms and you need 25 cores.

### Why doesn't existing software fix this?

**Feast** (the most popular open-source feature store) was built to solve the problem of managing features for training. Its online serving path looks like this:

```
Request comes in
→ Go to Redis, fetch raw bytes
→ Convert bytes to Python dict
→ Assemble dict values into array
→ Call model
```

Every arrow there involves Python object allocation. Python has a garbage collector, and the garbage collector periodically pauses everything to clean up. At high request rates these pauses show up as sudden latency spikes in your p99 (the slowest 1% of requests).

Pyforge eliminates most of these steps by making features live in **shared memory** — a region of RAM that multiple Python processes can read from simultaneously without any copying or converting.

---

## The Core Insight

When two processes share memory, they can read from the same physical RAM addresses. If you lay out your feature data in that shared memory correctly — as a typed NumPy array with a known, pre-computed layout — then reading a feature vector becomes:

1. Find the memory address of the data (computed once at startup, not on every request)
2. Read the bytes at that address

No network call. No byte conversion. No Python dict. No garbage collector involvement. Just a memory read.

This is what Pyforge is: infrastructure that makes that possible safely — handling the lifecycle of shared memory segments, keeping schemas consistent, knowing what to do when a process crashes.

---

## Component 1: The Schema System

### What a schema is

A schema is a definition of what your features look like. In Pyforge you define it like this:

```python
class UserFeatures(FeatureSchema):
    version = 1
    fields = [
        FeatureField("age_normalized", dtype.float32),
        FeatureField("session_count_7d", dtype.int32),
        FeatureField("ltv_score", dtype.float32),
        FeatureField("behavior_embedding", dtype.float32, shape=(128,)),
    ]
```

This says: a UserFeatures record has four fields. The first three are single numbers, the last is an array of 128 numbers. Every field has an explicit data type (float32 = a 32-bit decimal number, int32 = a 32-bit integer).

Pydantic v2 — a Python validation library that uses Rust under the hood for speed — validates this definition at the moment you write it, not at runtime when you're serving requests.

### The offset table

After the schema is defined, Pyforge compiles it once into an **offset table**. This is a small NumPy array where each row contains:

- A hash of the field name (a number that uniquely identifies the field name quickly)
- The byte offset into the shared memory segment where this field starts
- The data type code
- How many elements this field has

Think of it like a table of contents for the shared memory segment. Instead of looking up "age_normalized" in a Python dict on every request (which involves Python string hashing, dict lookup, Python object creation), you look up a pre-sorted array of integer hashes using a binary search — which is both faster and allocation-free.

This offset table is computed **once when the schema is registered**. On every subsequent request, it's already there, sitting in memory.

### Why data type discipline matters

If you have 200 features and store them as Python floats (64-bit), your feature vector is 200 × 8 bytes = 1,600 bytes. If you store them as float32 (32-bit), it's 800 bytes. That's 800 fewer bytes to copy, 800 fewer bytes the CPU cache has to hold. At 50k requests/second, cache efficiency has a real impact on throughput.

Being explicit about types (float32 vs float64 vs int32) also prevents a class of bugs where you train a model on float64 features but serve with float32, causing subtle numerical differences.

---

## Component 2: Shared Memory + Lifecycle Management

### What shared memory is

Your operating system lets multiple processes share the same physical RAM. Normal Python inter-process communication works like this:

```
Process A has a number
→ Serializes it to bytes
→ Sends bytes through a pipe or socket
→ Process B receives bytes
→ Deserializes bytes back to a number
```

Shared memory skips all of that:

```
Process A and Process B both have a pointer to the same physical RAM address
→ Process A writes a number there
→ Process B reads the number from there
```

No serialization. No network. No copying. Just RAM.

Python's standard library has `multiprocessing.shared_memory.SharedMemory` which lets you create and access named shared memory segments.

### How Pyforge lays out the segment

When you register a schema, Pyforge allocates a shared memory segment with this structure:

```
[16-byte header][field 1 data][field 2 data]...[field N data]
```

The header contains:
- 8 bytes: the schema version number
- 8 bytes: a CRC32 checksum of the schema definition (used to detect corruption)

The fields are laid out in the order they're defined in the schema, with each field's start address aligned to 64-byte boundaries. 64-byte alignment matters because modern CPUs fetch data from RAM in 64-byte "cache lines." If a field straddles a cache line boundary, the CPU has to make two fetches instead of one.

The segment is named with the schema name and version: `pyforge_UserFeatures_v1_<random_id>`. The name is stored in Redis so any process can find it.

### The reference counting system

Here's a problem: imagine Process A has the segment `pyforge_UserFeatures_v1_abc123` open and is reading from it. Meanwhile the schema gets updated to version 2, which allocates `pyforge_UserFeatures_v2_def456`. Can we delete `v1_abc123` immediately?

No — Process A is still reading it. Deleting it while it's being read would crash Process A.

Pyforge uses **reference counting** in Redis to solve this. Every process that opens a segment atomically increments a counter in Redis: `INCR pyforge:refcount:pyforge_UserFeatures_v1_abc123`. When it's done reading, it decrements: `DECR ...`. The segment is only eligible for deletion when the counter reaches zero.

Redis `INCR` and `DECR` are atomic — if two processes increment at the same moment, Redis processes them in order and neither is lost. You get the correct final count.

When a new schema version is registered, Pyforge uses a Redis Lua script (a small program that runs on the Redis server itself, atomically) to:

1. Set the `current_segment` key to the new segment name
2. Increment the new segment's reference count
3. Decrement the old segment's reference count
4. If the old count is now zero, add it to a cleanup list

This all happens in one atomic operation — there's no window where the current segment key is invalid.

### The watchdog process

Reference counting works for normal operation. But what if a process crashes mid-read? The `INCR` was called, but the matching `DECR` never happens. The reference count is permanently stuck above zero. The old segment leaks.

Pyforge runs a **watchdog process** — a separate Python process that does one job: monitor whether the processes that opened segments are still alive.

Every process that opens a shared memory segment writes its PID (process ID) and a timestamp to a Redis hash: `pyforge:heartbeats`. The watchdog loops every 100ms:

1. Reads all entries in `pyforge:heartbeats`
2. For each PID, checks if the process is still alive using `psutil.pid_exists()`
3. If a process is dead and its heartbeat is stale, decrements all refcounts that process held
4. Processes the cleanup queue — calls `shm.unlink()` on any segment whose count is zero

Cleanup happens within one heartbeat interval (100ms) of a crash. This is a concrete answer to "who cleans up shared memory if a process crashes?" with a concrete time bound.

---

## Component 3: The Lock-Free Read Path

### What "lock-free" means

A lock is a mechanism where one process says "I'm using this, nobody else touch it." Other processes have to wait. Under high concurrency, locks become bottlenecks — everyone queuing up, waiting their turn.

A lock-free read path is one where reads can happen simultaneously from many processes without any of them having to wait for each other. This is possible here because:

- The shared memory segment is read-only during serving (only writes go through a separate path)
- NumPy array reads are thread-safe for non-overlapping regions
- The offset table is pre-computed and never changes during serving

No locks means no waiting. Every request proceeds independently.

### The actual hot path

Here is exactly what happens when a request comes in to serve predictions for a single entity:

1. **Schema lookup** (done once at schema registration time, cached in a thread-local variable): get the current segment name from Redis. This Redis call only happens when the schema version changes — not on every request.

2. **Open the segment as a NumPy array**: `np.frombuffer(shm.buf, dtype=np.uint8)`. This creates a NumPy view of the shared memory — zero bytes are copied. The NumPy array and the shared memory point to the same physical RAM.

3. **Use the offset table** to compute the start and end byte indices of each field.

4. **Assemble the output vector**: copy the relevant bytes from the shared memory view into a pre-allocated output buffer. This is where the Numba-compiled function is used.

5. **Return the output buffer** (as a NumPy array) to the caller, who passes it to `model.predict()`.

There are zero Python dict lookups, zero string operations, zero object allocations in steps 2-5. The only allocation is step 4's write into the pre-allocated buffer, which isn't really an allocation — it's a write into memory that was allocated once at startup.

### The Numba-compiled assembly function

Numba is a Python library that compiles Python functions to native machine code using LLVM (the same compiler infrastructure that Clang and Swift use). The key word is "compiles" — the function runs as native code, not interpreted Python.

The assembly operation is:

```python
for each field:
    source_start = offset_table[field].byte_offset
    source_end = source_start + offset_table[field].byte_count
    output[cursor:cursor+count] = shared_memory_view[source_start:source_end]
    cursor += count
```

This is a loop over integers performing memory copies. There are no Python objects created inside the loop. There are no function calls to external libraries. It is a tight numeric loop — exactly the kind of code Numba accelerates.

Numba compiles this on first call (a one-time cost of ~100ms at startup) and then every subsequent call runs at near-C speed, with LLVM potentially vectorizing the inner copy operations using SIMD instructions.

**Important caveat**: Numba is only applied here — not sprinkled throughout the codebase as a performance signal. Every application of Numba in Pyforge is accompanied by a benchmark showing it actually helped. If the benchmark shows Numba doesn't help for small feature counts (it won't — the compilation overhead dominates for tiny arrays), that's documented and Numba is only used above a threshold.

### The pre-allocated buffer pool

Calling `np.empty(200)` on every request allocates a new NumPy array every time. At 50k req/s that's 50,000 allocations per second, which puts pressure on the garbage collector.

Pyforge pre-allocates a pool of output buffers at startup — a fixed number of NumPy arrays of the right shape and dtype for each registered schema. The serving path checks out a buffer from the pool (a `deque.pop()` operation, which is thread-safe in CPython), uses it, and returns it when done:

```python
with registry.get_buffer(UserFeatures, batch_size=1) as buf:
    registry.assemble(entity_id="user_001", output=buf)
    prediction = model.predict(buf)
# buf returned to pool here automatically
```

No allocation on the hot path. The buffer pool refills itself asynchronously when pool size drops below a threshold.

### GC pause elimination

Python's garbage collector periodically stops all threads to clean up circular references. This is called a "stop-the-world pause." It typically takes 1-50ms and happens unpredictably. This shows up as p999 latency spikes — 999 requests are fast, the 1000th suddenly takes 20ms.

Pyforge explicitly disables the GC in the serving thread:

```python
import gc
gc.disable()  # in the serving thread only
```

And runs a dedicated GC thread that calls `gc.collect()` on a 100ms timer, away from the serving path. This way GC still happens (memory doesn't leak), but it happens on a separate thread that isn't processing requests.

Pyforge instruments GC pauses using `gc.callbacks` — callback functions that are called at the start and end of each GC pause — and records the pause duration in a Prometheus histogram. The benchmark suite includes before/after latency distributions showing the p999 improvement.

### Batch assembly

The real-world use case is rarely "serve features for one entity." It's "serve features for 1,000 entities at once" — for recommendation systems, batch scoring jobs, A/B test cohorts.

Naive batch: 1,000 individual reads → 1,000 trips through the assembly logic.

Pyforge batch: one vectorized operation that assembles all 1,000 feature vectors simultaneously using NumPy broadcasting over the entire batch.

The API:

```python
vectors = registry.get_batch(
    schema=UserFeatures,
    entity_ids=["user_001", "user_002", ..., "user_1000"],
)
# returns np.ndarray of shape (1000, 131)  — 131 = total feature count
# contiguous float32, ready for model.predict(vectors)
```

The output is a single contiguous 2D array — row i is entity i's feature vector. This is what scikit-learn, XGBoost, LightGBM, and PyTorch all expect as input. No further conversion needed.

For features stored in Redis (not shared memory), batch assembly uses a single `MGET` command — one Redis round trip for all 1,000 entities — and decodes all 1,000 byte strings in a single vectorized pass.

---

## Component 4: The Offline Store (Simplified and Scoped Correctly)

### Why an offline store exists

The shared memory + Redis serving path is the "online" store — it serves live predictions. But you also need to train your model. Training requires historical data: "what were the features for user_001 at every point in the last year?"

This historical data lives in files on disk — specifically Parquet files, which are a columnar binary format designed for fast analytical reads.

### What the offline store does (and doesn't do)

The offline store in Pyforge is deliberately minimal. It does three things:

1. **Writes feature updates to Parquet** in an append-only fashion
2. **Reads point-in-time correct features** for model training
3. **Hydrates Redis** from Parquet when Redis is wiped (restart/flush)

It does NOT: handle distributed writes, stream processing, schema migration of existing files, or complex event-time watermarking. Those are streaming system problems that belong in Kafka/Flink, not a serving library.

### Append-only Parquet

Pyforge never rewrites existing Parquet data. It only appends new row groups (chunks of rows). Why this matters:

- **No partial-write corruption**: A Parquet row group is written atomically — either the whole row group and its footer are written, or the file looks exactly as it did before. A crash mid-write means at most one row group is lost, detectable by checking the footer CRC.
- **Predictable write performance**: Appending is always O(new data), never O(existing data + new data).

Each row group contains rows with these columns: `entity_id`, `event_time`, `processing_time`, and one column per feature field.

### Point-in-time correctness

For model training, you need features "as they existed at a specific moment in time" — not their current values. If you're building a training dataset and asking "what were user_001's features when they made their purchase at 14:00:00?", you need the feature values that existed at or before 14:00:00, not values that were written to the store after 14:00:00.

Getting this wrong is called "data leakage" — your training data includes information that wouldn't have been available at prediction time. Models trained with leaky data appear to perform well in backtesting but fail in production.

Pyforge's point-in-time query uses `merge_asof` from PyArrow — it takes a list of (entity_id, query_timestamp) pairs and returns the most recent feature value for each entity that existed at or before the query timestamp.

This is implemented using `searchsorted` on the sorted timestamp index of the Parquet file — a binary search that finds the right row in O(log n) time without scanning the whole file.

### WAL (Write-Ahead Log) for crash safety

When Pyforge writes a feature update, it needs to update both Redis (online store) and Parquet (offline store). What if the process crashes after writing to Redis but before writing to Parquet? Redis and Parquet are now inconsistent.

Pyforge uses Redis Streams as a Write-Ahead Log:

1. Write the feature update to the Redis Stream first (this is the "log" step)
2. Update the Redis online store
3. A background async consumer reads from the stream and writes to Parquet
4. Once the Parquet write succeeds, the consumer acknowledges the stream message

If the process crashes after step 1 but before step 3, the stream message is still there. On restart, the consumer replays unacknowledged messages.

**Idempotency** — "what if the same message gets processed twice?" — is handled with sequence IDs. Each stream message has a monotonic integer ID. The consumer stores the last successfully processed ID in Redis. On replay, any message with ID ≤ last_processed_id is skipped. Exact-once semantics implemented in about 30 lines of Python.

### Hydration

If Redis is restarted and all online store data is lost, you need to repopulate it from Parquet. Pyforge's hydration function:

1. Reads the Parquet offline store
2. Applies `merge_asof` with `as_of = now` to get the latest feature value per entity
3. Bulk-loads the results into Redis using pipelines (sending all commands at once, one round trip)

For 1 million entities, this takes roughly 30 seconds. Pyforge documents this number so operators can plan recovery time accordingly.

---

## Component 5: The Benchmark Suite

### Why benchmarks are as important as the code

A project about performance with no measured numbers is not a performance project — it's a performance *claim*. The benchmark suite turns claims into evidence.

Every benchmark in Pyforge is:
- **Reproducible**: `docker-compose up && pytest benchmarks/ --benchmark-autosave`
- **Documented**: hardware specs (CPU model, RAM, Redis version) are in the README
- **Honest**: if a component doesn't improve performance in some regime (e.g. Numba below 50 features), that's shown

### The five benchmarks

**Benchmark 1: Hot path latency**
Measures p50, p95, p99, p999 latency for a single-entity read from shared memory vs Redis vs a naive Python dict. Expected results: shared memory ~5µs p99, Redis ~200µs p99, naive dict ~1ms p99. This is the headline number.

**Benchmark 2: Batch assembly throughput**
Measures entities per second at batch sizes 1, 100, 1000, 10,000. Compares Numba vs NumPy vs Python loop. Shows the crossover point where Numba wins.

**Benchmark 3: GC pause impact**
Measures p999 latency with GC enabled vs GC-disabled-with-separate-thread. Shows the distribution of GC pauses and how they manifest as latency spikes.

**Benchmark 4: Buffer pool impact**
Measures allocation rate and p99 latency with vs without the pre-allocated buffer pool.

**Benchmark 5: Hydration speed**
Measures time to hydrate 100k, 1M, and 10M entities from Parquet to Redis. This is an operational benchmark, not a serving benchmark — it answers "how long does Redis recovery take?"

All benchmark results are committed to the repository as `benchmarks/results/` as JSON files (generated by pytest-benchmark) and as SVG flamegraphs (generated by py-spy). Reviewers can see the actual numbers without running anything.

---

## What Gets Cut (and Why)

### Watermark-based late data correction

The original design included a sophisticated system for handling features that arrive out of order — a "watermark" that tracks how far behind late data can be, and a correction record system that amends the offline store when late data arrives.

**Why it's cut:** This is a streaming systems problem. The correct tools for it are Kafka, Flink, or Spark Streaming. Including it in Pyforge would make Pyforge "a serving engine AND a streaming system," which blurs the project's identity and invites questions Pyforge isn't trying to answer. The offline store instead uses a simple rule: write data with both `event_time` and `processing_time`, and always query by `event_time`. Late data is written normally — Parquet is append-only so it just becomes a new row. Training pipelines that care about late data can filter by `processing_time` to exclude rows that arrived after a training cutoff.

### LLVM IR export

The original design included exporting the LLVM intermediate representation of the Numba-compiled function. Cut — this is resume padding. It adds zero value to the project. Flamegraphs and benchmark tables communicate performance far better.

### Distributed anything

Pyforge runs on one machine. It does not shard across machines, it does not replicate Redis, it does not coordinate across nodes. This is a deliberate scope decision, not a limitation. Single-node performance is the entire point. "Distributed" features would dilute the project's identity and add enormous complexity for no benefit to the core thesis.

---

## The Interview Narrative

When you sit across from an engineer at Two Sigma or D.E. Shaw and they ask about this project, here is the story you tell:

"The bottleneck in ML serving is rarely the model — it's the infrastructure around the model. Feast and similar tools solve this with Redis plus Python dicts, which works but has two problems: Python object allocation on every request creates GC pressure that shows up as p999 spikes, and the serialization/deserialization cycle is pure overhead.

I built Pyforge to eliminate both problems for single-node serving. Hot features live in shared memory as typed NumPy arrays. The read path is a pre-computed offset table lookup followed by a Numba-compiled memory copy — no Python objects allocated, no GC involved. The result is ~5µs p99 vs ~200µs for the Redis path on the same hardware.

The interesting engineering challenges were: managing shared memory segment lifecycle when schemas evolve, handling process crashes without memory leaks via a watchdog, and making the batch assembly path vectorized so 1,000 entities don't mean 1,000 individual reads."

Every sentence of that answer demonstrates concrete engineering knowledge. None of it is theoretical.

---

## The Scope in Plain Terms

To be absolutely clear about what Pyforge is and isn't:

**Pyforge is:** A Python library (~3,000 lines of code) that you `pip install` and use in a single Python process or group of processes on one machine to serve ML model features with low latency.

**Pyforge is not:** A distributed system, a cloud service, a streaming platform, a replacement for Feast in large organizations, or a database.

The target user is an ML engineer building a prediction service on one server who needs features served faster than Redis alone allows, without standing up a large infrastructure project.

