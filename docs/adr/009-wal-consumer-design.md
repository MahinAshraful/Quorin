# ADR-009: WAL consumer — async, deferred-XACK durability, signal-driven flush, borrowed segments

**Status:** Accepted
**Date:** 2026-04-29
**Step:** 10 (WAL consumer with idempotency)

## Decision

Quorin ships `quorin.wal_consumer.WALConsumer`, a single-coroutine
async consumer for the Redis Stream `quorin:wal` produced by
ADR-008's `WALProducer`. The consumer fans each message out to two
sinks: the online store (POSIX shared memory via
`quorin.layout.insert`) and an injected offline sink
(`OfflineWriter` Protocol; Step 11's Parquet writer is the production
implementation, `NoopOfflineWriter` is the default).

Public API:

```python
class WALConsumer:
    def __init__(
        self,
        redis_client: redis.asyncio.Redis,
        segments: Mapping[str, Segment],
        offline: OfflineWriter | None = None,
        *,
        stream_key: bytes = DEFAULT_STREAM_KEY,
        group_name: str = "quorin_consumers",
        consumer_name: str = "consumer-1",
        batch_count: int = 100,
        block_ms: int = 500,
        flush_interval_seconds: float = 60.0,
        max_pending_ack: int = 10_000,
    ) -> None: ...

    async def run(self) -> None: ...
    async def stop(self) -> None: ...
```

`run()` blocks until `stop()` is called or until an exception
propagates out. Caller owns the event loop. Caller is responsible for
SIGTERM handling and for any retry-on-Redis-error wrapping.

## Why each piece

### 1. Async via `asyncio` + `redis.asyncio.Redis`. No threadpool.

The consumer's blocking points are all Redis I/O (XREADGROUP, XACK,
SET, EXPIRE), which `redis.asyncio` handles natively. `layout.insert`
is fast CPU work (~5 µs at 50-field schemas). A threadpool would only
add lock contention without saving wall-clock time.

Caller owns the event loop. This sidesteps invariant #14 (background
threads need daemon + Event + `os.register_at_fork`) entirely — there
is no Quorin-managed thread, so there is nothing to fork-survive.
Deployments under uvicorn/gunicorn workers fork-then-call `asyncio.run`,
each worker getting its own loop and its own consumer instance.

### 2. Single coroutine, single consumer per stream.

Multi-consumer fan-out across one stream would let multiple processes
write to the same shm segment, violating invariant #3 (single writer
per segment). To keep the lockless 5 µs read budget, we accept
"single consumer" as a hard constraint.

Horizontal scaling answer (CLAUDE.md §1): `hash(entity_id) mod N`
across N independent Quorin instances, each with its own segment,
its own stream `quorin:wal:{i}`, and its own consumer. This is a
deployment-layer concern; Step 10 does not implement it.

### 3. Deferred-XACK durability — SET and XACK signal different things.

The two operations the consumer sends back to Redis are operationally
distinct and intentionally decoupled:

- **`SET quorin:processed:{msg_id} EX 86400`** = "online store has
  this data; `WALProducer.write_sync` may unblock." Fires immediately
  after `layout.insert` succeeds.
- **`XACK`** = "I have durably persisted this message in the offline
  store and no longer need to see it again." Fires only after
  `OfflineWriter.flush()` returns successfully.

Coupling them — for example, XACK at the same time as SET — would
lose up to `flush_interval_seconds` of offline data on crash, because
`OfflineWriter.append` is buffered, not durable. The Parquet writer
specifically buffers rows in memory and seals them into UUID-named
files via atomic-rename on `flush()`. ACKing before the rename means
the producer thinks the message is durable when only the in-memory
buffer holds it.

`write_sync`'s contract is tightened to **online-store** durability
(read-your-own-writes from `assemble`). Offline durability is a
separate, eventually-consistent guarantee. ADR-008 §"write_sync" is
cross-referenced for callers expecting offline durability via
`write_sync` — they don't get it.

### 4. `OfflineWriter` Protocol durability contract.

```python
class OfflineWriter(Protocol):
    async def append(...) -> None: ...   # at-least-once
    async def flush(...) -> None: ...    # durable on return
    async def close(...) -> None: ...
```

`append()` may be called multiple times for the same `msg_id` across
crash/restart. The offline store is responsible for deduplicating on
read by `(entity_id, event_time_ns)` or by `msg_id`.

`flush()` MUST be cancellation-safe. The consumer cancels the flush
task during shutdown; the implementation must either complete its
work or unwind cleanly without leaving partial state. Step 11's
atomic-rename pattern satisfies this naturally — either the rename
landed or it didn't.

### 5. Always re-apply on PEL drain (no bulk-EXISTS optimization).

With deferred XACK, every PEL message is by definition
"applied-online but not-yet-flushed-offline." So `offline.append`
MUST re-fire on replay regardless of the side-table state. A
bulk-EXISTS check would optimize zero work.

PEL drain is **paginated** by passing the last-seen ID as the next
XREADGROUP cursor. A naive loop on `id="0"` would never terminate:
the same PEL entries come back every iteration because XACK is
deferred. The consumer advances `last_id` past each batch and exits
when a read returns no new entries.

Recovery cost is bounded by `max_pending_ack`, not by
`flush_interval × throughput`. At the 10 000 default and ~30 µs
per-msg apply, worst-case catch-up is ~300 ms regardless of sustained
throughput. Operators tune `max_pending_ack` to set their recovery
SLA directly.

### 6. Signal-driven flush trigger; hard back-pressure ceiling.

The size-based flush trigger inside `_process_batch` is an
`asyncio.Event.set()` call, **not** an `await`. Awaiting the flush
from the read loop would cap throughput at
`batch_apply_time / (batch_apply_time + flush_time)` ≈ 950 msgs/sec
once Step 11's flush takes 100 ms — an order of magnitude under the
10 k/s target.

The flush task waits on `asyncio.wait({stop_event, flush_now}, timeout=flush_interval_seconds, return_when=FIRST_COMPLETED)`,
so it wakes on the soonest of: `stop()`, the periodic timer, or the
size-trigger signal. Single in-flight flush invariant holds because
only this coroutine calls `_flush_and_ack`.

Hard back-pressure ceiling at `2 * max_pending_ack`: if the flush task
falls behind and `pending_ack` exceeds twice the soft trigger, the
read loop pauses (`asyncio.sleep(0.01)`) until pending_ack drains.
Bounds memory at ~1 MB worst case (20 000 msg_ids × ~50 bytes each).

### 7. Borrowed segments — never owned.

The consumer constructor accepts pre-opened `Segment` instances. The
caller (the bootstrap code that constructs `WALConsumer`) owns
segment lifecycle: `SegmentRegistry.create` / `open_current`, and
later `SegmentRegistry.close`. The consumer does not call any of
these.

Consequences:
- Invariant #3 stays provable from outside the consumer. The user
  hands segments to one consumer instance, and that's the only writer.
- No hidden Redis calls (`INCR quorin:refcount:*`) on any apply
  path. The producer's cost model still holds.

The constructor validates `seg.schema.__name__ == key` for each
entry in the segments map. Catches the one footgun the API otherwise
allows.

### 8. Internal `row_pack` (NOT the test helper).

`quorin._internal.row_pack.pack_row_from_list(schema, values, out)`
takes the producer's wire-shape values list (positional, name_hash
order) and writes into a caller-supplied bytearray.

Implementation is memoized per schema by class identity. The pack
plan walks `compute_row_offset_table` (name_hash-sorted, row-relative
offsets) — **NOT** `compute_assembly_table` (declaration order,
which would silently corrupt rows because the producer's wire format
is name_hash-ordered).

Single consolidated `struct.Struct.pack_into` call writes ALL scalar
fields in one C pass, with `Nx` pad-byte format codes for
cache-line gaps and for shaped-field regions. Without the pads, the
naive consolidator would only catch the first field (because Quorin
aligns every field start to 64 bytes; see
`quorin.schema._field_byte_offsets`) and dump every other field into
the slow per-field numpy path — net ~30 µs of dispatch overhead at
200 fields. With pads, the format string for a 200-field float32
schema is `"<f60xf60x...f"` and the C call places all 200 floats at
their correct offsets in a single pass.

Critical wrinkle: the sorted-by-name-hash table has byte_offsets
that are NOT monotonically increasing — they're permuted by the
sort. The plan builder sorts table rows by `byte_offset` before
emitting the format string, and records each scalar's value-list
index (in name_hash order) in `scalar_indices` so
`*(values[i] for i in scalar_indices)` produces args in the order
the Struct expects.

Shaped fields (`shape != ()`) write via numpy view directly into
the output bytearray. `np.copyto(view, src)` is zero-copy on the
destination side; the only allocation is `np.asarray(v, dtype=...)`
on the source. `shape == ()` (true scalar) and `shape == (1,)`
both have `element_count == 1` but ship different wire shapes
(scalar vs. 1-elt list); the plan builder cross-references
`schema.fields` by name_hash to recover the original shape and
route correctly.

Inter-field padding bytes in the row buffer are **not zeroed**.
Safe because `assemble` walks the offset table — it reads only
`(byte_offset, byte_count)` extents and never touches padding bytes.

Estimated per-call cost: ~10-12 µs at 200 fields. Measured on
WSL2: see §"Verification" below.

### 9. Pipeline is non-transactional (`transaction=False`).

The consumer's per-batch SET pipeline uses
`redis.pipeline(transaction=False)` — no MULTI/EXEC. Idempotency
makes MULTI unnecessary; partial-pipeline crashes are handled by
replay. Saves the WATCH/MULTI/EXEC round-trip overhead.

`assert len(pipe.command_stack) == len(applied)` before
`pipe.execute()` catches a regression where someone adds an XACK
to the pipeline by accident — XACK is deferred to the flush task
in this design.

### 10. Consumer-name distributed lock.

At `run()` start, the consumer SETs
`quorin:consumer:lock:{group}:{name} <pid> EX 60 NX`. If the SET
fails (another consumer holds the lock), GET returns the holder's
PID and we raise `ConsumerNameInUseError`.

The lock TTL is renewed every flush cycle. On clean shutdown the
key is deleted; on crash it auto-expires in 60s.

Two consumers with the same `consumer_name` would silently scramble
each other's PEL — Redis doesn't reject the conflict. Cheap fail-fast
prevents a corruption class.

### 11. Sync vs async Redis client (operability).

`WALProducer` takes a sync `redis.Redis`; `WALConsumer` takes a
`redis.asyncio.Redis`. Same-process callers (a script that both
produces and consumes; an integration test; the benchmark suite)
need both. `redis-py` does not unify the two client classes —
their connection pools are separate. Pickling a sync client and
using it via asyncio is NOT supported.

A pool-unification helper (`quorin.wal.make_clients(url)`) is
deferred to Step 12 (public API). Cross-listed in CLAUDE.md §8 as
the gotcha lookup.

### 12. Shutdown SLA = `block_ms`.

Worst-case `stop()` latency is `block_ms` (default 500 ms) because
`XREADGROUP block=N` is not interruptible by an `asyncio.Event`.
Users with restart-time SLAs can pass `block_ms=100` (negligible
idle-RPS cost: 10 RPS to Redis when no traffic) or override the
`block_ms` constructor arg.

### 13. XGROUP CREATE id="0", not "$".

Standard Redis-tutorial guidance is `id="$"` ("only deliver new
messages to a fresh group"). For a write-ahead log this is an
operability footgun: a producer that XADDs before any consumer is
running would have its messages silently orphaned. Quorin uses
`id="0"` so a fresh group drains the entire current stream.

Once the group exists, the cursor lives in Redis and survives
consumer restarts. Subsequent XREADGROUP `>` calls deliver only
what's strictly newer than the cursor, so this isn't a "drain
everything every restart" — only a fresh group does that.

## Hot-path budget — measured on WSL2 / Ubuntu / Docker Redis 7.2-alpine, 2026-04-29

Bench file: [`benchmarks/test_wal_consumer_benchmark.py`](../../benchmarks/test_wal_consumer_benchmark.py).

| Bench | Median | Max | Gate (p99) |
|---|---|---|---|
| `row_pack_50_field` | **5.4 µs** | 111 µs | 30 µs |
| `row_pack_200_field` (with 128-emb shaped) | **12.0 µs** | 1.76 ms | 30 µs |
| `row_pack_200_scalar_only` | **7.6 µs** | 380 µs | 30 µs |
| `consumer_apply_per_msg_50_field` | **9.9 µs** | 366 µs | 100 µs |
| `consumer_apply_per_msg_200_field` (with shaped) | **20.7 µs** | 1.26 ms | 100 µs |

The consolidated-Struct path measurably delivers on its design
estimate (8 µs at 200 scalar fields, 12 µs with one shaped
embedding). Per-msg apply is well under the 100 µs gate, leaving
~80 µs of headroom for offline-writer cost and growth.

End-to-end `write_sync` lag and consumer throughput are exercised
by integration / chaos tests rather than absolute benchmarks (the
RTT floor is hardware-dependent; see ADR-008 §"Hot-path budget").

## Verification

```bash
uv run pytest tests/unit/test_row_pack.py tests/unit/test_wal_consumer.py -v
uv run pytest tests/property/test_wal_roundtrip_full.py -v
uv run pytest -m integration tests/integration/test_wal_consumer_redis.py -v
uv run pytest -m chaos tests/chaos/test_wal_consumer_crash_safety.py -v
uv run pytest --benchmark-only benchmarks/test_wal_consumer_benchmark.py
```

Acceptance:
- 19 unit + 2 property + 6 integration + 4 chaos = 31 new tests, all green.
- Chaos `test_sigkill_repeated_iterations_all_converge` runs the SIGKILL+restart
  cycle 20 times with random kill timings — all converge to the same final
  state (100 distinct entities in shm, PEL drained).
- 5 new benchmark cases, all under their threshold gates.
- Static checks (`ruff check`, `ruff format --check`, `mypy quorin`) clean.

## What this rejects (alternatives considered)

- **Coupled SET + XACK** (the original spec idea). Rejected: would lose
  up to `flush_interval_seconds` of offline data on crash. The whole
  point of a WAL is to not lose committed writes.
- **Bulk-EXISTS PEL drain optimization**. Rejected: with deferred XACK,
  PEL entries always need offline.append re-fired, so the EXISTS check
  optimizes zero work and adds a Redis RTT.
- **Synchronous size-trigger flush**. Rejected: caps throughput at
  ~950 msgs/sec because the read loop blocks behind every flush.
- **Multi-consumer fan-out per stream**. Rejected: violates invariant
  #3. Deployment-layer hash sharding is the answer.
- **Threadpool / "run in background" wrapper**. Rejected (for now):
  caller owns the event loop. Future helper deferred to Step 12.
- **`compute_assembly_table` for the row_pack plan**. Rejected: it's
  declaration-order, doesn't match the producer's name_hash wire
  format. Would silently corrupt rows.
- **`XGROUP CREATE id="$"`**. Rejected: operability footgun for a WAL.
- **Background-thread wrapper with `os.register_at_fork`**. Rejected:
  not needed when the caller owns the loop.

## Cross-references

- ADR-008 §4-5 (memoized pydantic factory; NaN/Inf semantics) — the
  consumer trusts producer-side validation.
- ADR-007 §"BatchBufferPool slab is kept alive by the deque's
  slice-views" — same memoization-by-class-identity pattern is reused
  for `_PLAN_CACHE`.
- ADR-006 §"GC callback bodies must not allocate Python objects" — the
  consumer's `_apply` doesn't allocate inside any GC-vulnerable region;
  the metric label children are pre-warmed at `__init__`.
- CLAUDE.md §8 — sync vs async Redis client gotcha entry added.
