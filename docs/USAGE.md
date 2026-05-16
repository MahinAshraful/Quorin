# Usage

Runnable code examples for every public path. If you want to understand how
the library works internally instead, see [ARCHITECTURE.md](ARCHITECTURE.md).
For the per-module API surface, see [API.md](API.md).

For copy-paste-runnable end-to-end scripts, see the
[`examples/`](../examples/) directory:

| Example | What it shows |
|---|---|
| [`examples/quickstart.py`](../examples/quickstart.py) | Define schema, create segment, insert + assemble |
| [`examples/write_sync.py`](../examples/write_sync.py) | WAL producer + consumer with read-your-own-writes |
| [`examples/hydration.py`](../examples/hydration.py) | Bulk-load online store from Parquet |
| [`examples/upgrade.py`](../examples/upgrade.py) | Schema evolution v1 → v2 |

Every example is exercised by `pytest examples/` in CI (CR.A.12-b /
v0.1.1 future-drift gate); the snippets in this document trace the same
APIs in narrative form. If anything below breaks against the latest
release, file an issue.

---

## Prerequisites

- **Python 3.12+** (Quorin pins this; older Pythons will fail at install).
- **Linux or WSL2** (POSIX shared memory). macOS works but isn't tested in
  CI; native Windows is out of scope.
- **Redis 7.2+** reachable on `127.0.0.1:6379`.

The fastest way to get Redis up locally:

```bash
docker run -d --name quorin-redis -p 6379:6379 redis:7.2-alpine
```

Then `pip install quorin` (or `uv pip install quorin`) into a fresh venv.

---

## 1. Define a schema

A schema is a Python class. No infrastructure needed for this step — pure
typing.

```python
from quorin.schema import FeatureSchema, FeatureField, dtype

class UserFeatures(FeatureSchema):
    version = 1
    fields = [
        FeatureField("age_normalized", dtype.float32),
        FeatureField("session_count_7d", dtype.int32),
        FeatureField("ltv_score", dtype.float32),
        FeatureField("behavior_embedding", dtype.float32, shape=(128,)),
    ]
```

A few rules `FeatureSchema` enforces at class-definition time:

- Every field must have an explicit `dtype`. Supported: `dtype.float32`,
  `dtype.float64`, `dtype.int32`, `dtype.int64`, `dtype.uint8`.
- Field names must be unique within the schema.
- `shape=()` (the default) means a scalar; `shape=(128,)` means a 1D array
  of 128 elements; `shape=(8, 8)` means a 2D 8×8 matrix.
- `version` is required and must be a positive integer. Bump it whenever
  the schema changes.

---

## 2. The synchronous demo (insert + assemble)

The fastest way to see Quorin work end-to-end. Useful for tests, hydration,
demos. Production writes go through the WAL flow (next section); this is the
direct path.

```python
import redis
from quorin.schema import FeatureSchema, FeatureField, dtype
from quorin.shm import SegmentRegistry
from quorin.layout import insert, pack_row
from quorin.assembly import assemble

class UserFeatures(FeatureSchema):
    version = 1
    fields = [
        FeatureField("age_normalized", dtype.float32),
        FeatureField("session_count_7d", dtype.int32),
        FeatureField("ltv_score", dtype.float32),
    ]

r = redis.Redis(host="127.0.0.1", port=6379)
registry = SegmentRegistry(r)
seg = registry.create(UserFeatures, capacity=1000)

row = pack_row(UserFeatures, age_normalized=0.5, session_count_7d=42, ltv_score=12.3)
insert(seg, "user_001", row)

features = assemble(seg, "user_001")
print(features)         # [ 0.5 42.  12.3]
print(features.dtype)   # float32
```

`pack_row` is a convenience helper for the synchronous path. It accepts
keyword arguments matching the schema's declared fields, coerces each value
to the field's declared dtype, and returns the bytes that `insert` expects.
Scalars work for scalar fields; lists, tuples, or numpy arrays work for
shaped fields (with the right element count).

`assemble` returns a contiguous `numpy.float32` array — the exact shape every
ML library expects as model input. You can pass it directly to
`model.predict(features)`.

---

## 3. The production write path (WAL producer + consumer)

In production, writes are async by default. A producer process validates the
row, encodes it, and sends it to a Redis Stream. A separate WAL consumer
process reads from the stream and applies each message to the shared-memory
segment as the segment's only writer. This separation gives you crash safety
(messages are durable in Redis before being applied) and a single-writer
invariant (which is what makes the lock-free read path possible).

### The producer (in your application code)

```python
import time
import redis
from quorin.wal import WALProducer

# Same schema as before
r_sync = redis.Redis(host="127.0.0.1", port=6379, socket_timeout=5.0)
producer = WALProducer(r_sync)  # accepts optional stream_key=, maxlen=

msg_id = producer.write(
    UserFeatures,
    entity_id="user_001",
    values={
        "age_normalized": 0.5,
        "session_count_7d": 42,
        "ltv_score": 12.3,
    },
    event_time_ns=time.time_ns(),
)
```

`producer.write(...)` returns the Redis-assigned message ID immediately
after the `XADD`. The message is durable but not yet visible to readers —
there's a typical 5–50 ms gap before the WAL consumer applies it. For
most production write paths that's fine: writes are eventually
consistent with reads.

### Read your own writes

If you need a write to be visible before continuing, use `write_sync`:

```python
from quorin.wal import WriteSyncTimeoutError

try:
    msg_id = producer.write_sync(
        UserFeatures,
        entity_id="user_001",
        values={
            "age_normalized": 0.5,
            "session_count_7d": 42,
            "ltv_score": 12.3,
        },
        event_time_ns=time.time_ns(),
        timeout_ms=2000,  # milliseconds — default 100
    )
except WriteSyncTimeoutError:
    # The XADD succeeded; the consumer didn't set the processed-key in
    # time. The message will be applied eventually unless the consumer
    # is dead — handle as you would any "best-effort write" timeout.
    pass
```

This blocks until the consumer has applied the message and SET a
per-message side-table key, or raises `WriteSyncTimeoutError` if the
consumer is slow. Costs you one consumer-cycle round trip (~5–50 ms on
a healthy consumer).

### The consumer (in a separate process)

The consumer is an async coroutine. Run it in a dedicated Python process,
typically as a systemd service or container.

```python
import asyncio
import redis.asyncio as aioredis
from quorin.shm import SegmentRegistry
from quorin.wal_consumer import WALConsumer
from quorin.offline import ParquetDatasetStore

async def run_consumer():
    r_async = aioredis.Redis(host="127.0.0.1", port=6379, socket_timeout=5.0)
    r_sync = redis.Redis(host="127.0.0.1", port=6379, socket_timeout=5.0)
    registry = SegmentRegistry(r_sync)
    seg = registry.open_current(UserFeatures)
    store = ParquetDatasetStore("/var/quorin/offline")
    consumer = WALConsumer(
        r_async,
        segments={UserFeatures.__name__: seg},
        offline=store,
        registry=registry,
    )
    await consumer.run()

asyncio.run(run_consumer())
```

The consumer's `segments` mapping is keyed by `schema.__name__` and the
mapped value is a live `Segment` (the consumer borrows the segment;
the caller owns its lifecycle). To consume multiple schemas in one
consumer, populate the mapping with one entry per schema. The
`registry` kwarg is required (the Step-15 pause-and-reopen safety
net depends on it).

Note that the producer takes a synchronous Redis client and the consumer
takes an async one. This is intentional — `redis-py` doesn't unify them.
Same-process callers (e.g. integration tests) need both.

The consumer applies each WAL message to both the shared-memory segment
(immediately) and the Parquet offline store (buffered; flushed periodically).
The two durability signals are decoupled: `write_sync` returns when the
online store has the message; the offline store catches up shortly after.

---

## 4. Reading

Once a segment exists with data in it, every reader process opens it once at
startup and then reads with no further coordination.

```python
from quorin.shm import SegmentRegistry
from quorin.assembly import assemble

# Once at process startup:
registry = SegmentRegistry(r)
seg = registry.open_current(UserFeatures)

# In your request handler:
features = assemble(seg, "user_001")
prediction = model.predict(features.reshape(1, -1))
```

Reads are lock-free, never touch Redis, and complete in microseconds. The
`assemble` call returns a numpy array; reshape if your model expects a 2D
input shape. If the entity isn't in the segment, `assemble` raises
`EntityNotFoundError` — catch it and route to your fallback (a default-row
heuristic, a Redis lookup of cold features, etc.).

### With a buffer pool (reduces GC pressure)

```python
from quorin.pool import BufferPool

# Once at process startup:
pool = BufferPool(UserFeatures, max_size=128)

# In your request handler:
with pool.checkout() as buf:
    assemble(seg, "user_001", out=buf)
    prediction = model.predict(buf.reshape(1, -1))
# buf is automatically returned to the pool here
```

The pool's headline value is **GC-pressure reduction**, not single-call
latency: at the 4.48 µs p99 hot path the pool's checkout adds ~2-4 µs
on native CI (ADR-005 amendments), so for cost-sensitive single-entity
reads the no-pool path is faster. The pool's real wins are batch
assembly (next section) and high-allocation workloads where the
~80-100 ns `np.empty(N, float32)` cost compounds.

---

## 5. Batch reads

For batch scoring or recommendation retrieval, request many entities at once.

```python
from quorin.assembly import assemble_batch

entity_ids = [f"user_{i:06d}" for i in range(1000)]
features, found_mask = assemble_batch(seg, entity_ids)

# features.shape == (1000, total_element_count)
# found_mask.shape == (1000,) bool

# Filter to only entities we actually found
present = features[found_mask]
predictions = model.predict(present)
```

Missing entities (not in the segment) are returned as zero-filled rows; the
returned `found_mask` is a boolean array indicating which rows are valid.

The batch path can use a `BatchBufferPool` to avoid the per-call
output-buffer allocation, which is where buffer pools win decisively
— a single batch buffer at N=1000, 200 fields is ~1.6 MB.

```python
from quorin.pool import BatchBufferPool

batch_pool = BatchBufferPool(UserFeatures, batch_size=1000, max_size=8)

with batch_pool.checkout() as buf:
    # buf is the (batch_size, element_count) output array. Pass it to
    # assemble_batch via the out= kwarg; the kernel zero-fills missing
    # entities so the BatchBufferPool's zero_on_return=False default
    # is safe (see ADR-007). The found_mask is returned separately.
    features, found_mask = assemble_batch(seg, entity_ids, out=buf)
    predictions = model.predict(features[found_mask])
```

For batch sizes above an adaptive threshold (computed from
`numba.get_num_threads()` at import time), the kernel uses Numba's
`parallel=True` mode and dispatches the row loop across threads. For optimal
batch performance set `NUMBA_NUM_THREADS=4` in your environment before
importing `quorin`.

---

## 6. Point-in-time reads (training data)

For building a training dataset, you need feature values "as they were at a
specific moment in the past" — not their current values. Reading current
values into a training set causes data leakage: the model trains on
information it wouldn't have had at prediction time.

```python
import pyarrow as pa
from quorin.offline import ParquetDatasetStore

store = ParquetDatasetStore("/var/quorin/offline")

# Build the query table — Quorin's read_point_in_time takes a
# pa.Table with two columns: entity_id (string) and as_of_time_ns
# (int64). Hand-rolled here; in real workloads the queries come from
# a labeling system or training pipeline.
query_table = pa.table({
    "entity_id": ["user_001", "user_002"],
    "as_of_time_ns": [
        1715040000_000_000_000,  # 2024-05-07 00:00:00 UTC
        1715126400_000_000_000,  # 2024-05-08 00:00:00 UTC
    ],
})

table = store.read_point_in_time(UserFeatures, query_table, lookback_days=30)
# Returns a PyArrow Table with one row per query, populated with each
# entity's most-recent feature values that existed at or before its
# as_of_time_ns. Entities with no rows in the lookback window are
# included with null feature columns.
```

`read_point_in_time` is synchronous (CPU + IO bound, not awaitable). Async
callers should wrap it in `asyncio.to_thread`. The `lookback_days` parameter
caps how far back the query will look — important to set to a sensible value
because it bounds the Parquet scan width.

---

## 7. Hydration (cold start)

If Redis is wiped or you're deploying fresh, the segment doesn't exist yet —
but the Parquet offline store still has all the latest features. `hydrate`
rebuilds the segment from the offline store.

```python
from quorin.hydration import hydrate

result = hydrate(
    UserFeatures,
    store,
    registry,
    redis_client=r,
    capacity_factor=4.0,  # 4x entity count → segment capacity
    lookback_days=30,
)
print(f"Hydrated {result.entity_count} entities in {result.elapsed_seconds:.2f}s")
print(f"new segment: {result.segment_name}")
```

For 1 million entities × 50 fields, this takes ~9.9 s on WSL2 and is
projected at 1.2-2.5 s on bare metal (CLAUDE.md §8 "Step 15 spec
acceptance"). Run it before starting your WAL consumer; the hydrate
function checks for preconditions (no current segment, no live WAL
consumer) and refuses with `HydrationConflictError` if either is
present. `EmptyDatasetError` if the lookback window contains zero
entities.

---

## 8. Schema evolution

Adding fields or widening a dtype requires a schema upgrade — the in-memory
segment needs to be rebuilt with the new layout while live readers continue
against the old one.

```python
from quorin.evolution import upgrade_schema, can_upgrade

class UserFeaturesV2(FeatureSchema):
    version = 2
    fields = [
        FeatureField("age_normalized", dtype.float32),
        FeatureField("session_count_7d", dtype.int32),
        FeatureField("ltv_score", dtype.float32),
        # NEW: added field
        FeatureField("country_code", dtype.uint8),
        # NEW: widened from int32
        FeatureField("session_count_30d", dtype.int64),
    ]

# Pure check, no Redis touched. Returns (bool, list[str]); the list
# is empty when ok is True and human-readable failure descriptions
# otherwise.
ok, reasons = can_upgrade(UserFeatures, UserFeaturesV2)
assert ok, reasons

result = upgrade_schema(
    UserFeatures,
    UserFeaturesV2,
    registry,
    redis_client=r,
)
print(f"Upgraded {result.entity_count} entities in {result.elapsed_seconds:.2f}s")
```

> **v0.1.1 note:** 2D-shape upgrades are deferred to v0.2.0 (CR.A.4 /
> ADR-018). `can_upgrade` returns `(False, ["...2D-shape upgrade not
> yet supported..."])` for any schema with a 2D field; workaround is
> to introduce a new 1D field and deprecate the 2D one. Operators
> attempting an upgrade no longer need to manually `XTRIM` the WAL
> stream first — the precondition is now `XPENDING == 0`, not
> `XLEN == 0` (CR.A.13 / ADR-018).

The upgrade is online: live readers against the OLD segment continue to work
during the entire copy, and atomically flip to NEW when the copy completes.
The WAL consumer is briefly paused during the flip; if you forgot to drain
the WAL stream first (operator discipline), the consumer's pause+reopen
safety net catches the format mismatch as a "poison pill" rather than
corrupt the new segment with old wire-format data.

Supported transitions:

- Adding fields (any new field with a default-zero initial value).
- Widening dtypes: `int32 → int64`, `float32 → float64`. Narrowing is not
  supported (lossy).
- Schema version must increment.

There's also a CLI: `python -m quorin.evolution upgrade --redis URL --old PKG:V1 --new PKG:V2 --confirm`.

---

## 9. Operations

### Run the watchdog

In production, run the watchdog as a separate process (systemd unit, sidecar
container, etc.):

```bash
python -m quorin.watchdog --redis redis://localhost:6379 --metrics-port 9101
```

It runs forever, polling every 50 ms, detecting dead worker processes via
heartbeat staleness, and cleaning up their shared-memory references. Without
the watchdog, segments leak whenever a worker dies without a clean exit.

### Expose Prometheus metrics

```python
from quorin.metrics import start_metrics_server

start_metrics_server(port=9100)
```

Optional. If you don't call it, counters still increment in memory and are
visible to tests; they just aren't scraped. Prometheus scrapes from the
`/metrics` endpoint at the given port.

Available metrics include `quorin_gc_pause_seconds` (per-generation histogram),
`quorin_pool_miss_total` (per-schema counter), `quorin_wal_write_latency_seconds`,
`quorin_wal_consumer_apply_total` (consumer outcome counter),
`quorin_offline_flush_seconds`, watchdog cleanup counters, and schema-upgrade
timing histograms.

> **Note:** v0.1.0 declared `quorin_read_latency_seconds` and
> `quorin_wal_lag_seconds` but never observed them; both were removed in
> v0.1.1 (CR.A.9 / CR.A.10). Hot-path latency observation costs ~30 ns per
> call on the 4.48 µs p99 budget; v0.2.0 will ship opt-in instrumentation
> behind an environment-variable flag.

### Configure structured logging

```python
import quorin.logging

quorin.logging.configure(level="INFO")  # JSON renderer is the default
log = quorin.logging.get_logger("my_service")
log.info("started", schema="UserFeatures", capacity=1_000_000)
```

Output is JSON by default — machine-parseable, suitable for log shippers.
Pass `json=False` for human-readable colored console output.

---

## 10. Common patterns

### Pattern: warm Numba kernels at startup

Numba compiles JIT kernels on first call. By default, that ~300 ms cost is
paid by the first request your service handles. To pay it at startup
instead:

```python
from quorin.assembly import prewarm

prewarm()  # compiles single-entity assemble, batch assemble, lookup-jit, BLAKE2b kernel
```

Subsequent process starts skip the compile via Numba's on-disk cache, but
the first start of any new deployment still pays the cost. `prewarm()` is
opt-in — module load doesn't trigger it, so importing Quorin stays cheap
for callers that don't actually use the Numba path.

### Pattern: handle entity-not-found gracefully

```python
from quorin.serving import EntityNotFoundError

try:
    features = assemble(seg, entity_id)
except EntityNotFoundError:
    # Cold entity — fall back to default heuristics or skip
    features = default_features()
```

The batch path doesn't raise; it returns a `found_mask` bool array.

### Pattern: capacity planning

Pick a `capacity` somewhat above your expected entity count. The slot table
is sized to ~2× capacity, so a 1M-entity segment uses ~48 MB just for the
slot table; the feature region is `capacity * row_size` bytes (around 1 GB
for 1M × 200 fields × 128-dim embedding).

`registry.create` raises if you try to create a segment larger than 50% of
`/dev/shm` to avoid SIGBUS during writes. Plan accordingly: either increase
the `/dev/shm` ceiling (`docker run --shm-size=8gb ...`) or shard
horizontally.

### Anti-pattern: don't write to the segment from a producer process

The single-writer invariant is what makes lock-free reads correct. Always
write through `WALProducer.write` or `write_sync`; never call
`quorin.layout.insert` from anywhere except the WAL consumer (or the
hydration coordinator at startup).

### Anti-pattern: don't share Redis clients between sync and async code

`WALProducer` takes `redis.Redis` (sync). `WALConsumer` takes
`redis.asyncio.Redis` (async). Don't pickle one and use it from the other —
their connection pools are separate and the asyncio one needs an event
loop. Same-process integration tests need both clients constructed
explicitly.

---

## 11. Where to look next

- **For deep technical understanding of why each component exists**:
  [ARCHITECTURE.md](ARCHITECTURE.md)
- **For the per-module API surface**: [API.md](API.md)
- **For design rationale on any specific decision**: the [ADR archive](adr/)
- **For benchmark methodology**: [ADR-015](adr/015-benchmark-methodology.md)
