# ADR-008: WAL producer — sync XADD, msgpack list payload, internal memoized pydantic factory

**Status:** Accepted
**Date:** 2026-04-29
**Step:** 9 (WAL producer)

## Decision

Pyforge ships `pyforge.wal.WALProducer` with two methods:

- `write(schema, entity_id, values, event_time_ns=None) -> bytes` —
  validates `values`, msgpack-packs them as a list in name_hash order,
  XADDs to `pyforge:wal`, returns the Redis-assigned message ID.
- `write_sync(...)` — same as `write` plus polling the consumer's
  `pyforge:processed:{msg_id}` side table with 1 ms → 10 ms exponential
  backoff. Raises `WriteSyncTimeoutError` on deadline expiry.

Validation goes through `pyforge._internal.pydantic_factory.pydantic_model_for(schema)`,
a memoized factory keyed by class identity. NaN and ±Inf are accepted on
every float field via `Field(allow_inf_nan=True)`. The producer holds a
single `msgpack.Packer()` instance and per-instance caches for
schema-name encoding and pre-warmed prometheus label children.

## Hot-path budget

Measured on WSL2 Ubuntu / Docker Desktop Redis 7.2-alpine,
`appendfsync everysec`, 2026-04-29:

| Cost | Budget | Measured median | Measured max |
|---|---|---|---|
| Redis XADD RTT (`xadd_only` bench) | ~500 µs | **2.82 ms** | 6.68 ms |
| Pydantic validate (200-field) | 100–300 µs | **16.7 µs** | 239 µs |
| msgpack pack (200-field) | 20–80 µs | **6.94 µs** | 143 µs |
| Field encode + dispatch | <10 µs | (rolled into total) | — |
| Histogram observe + counter inc | <2 µs | (rolled into total) | — |
| **Producer CPU above XADD floor** | **<400 µs** | **~120 µs** | — |
| **`write` p50 wall-clock (200-field)** | RTT-bound | **2.94 ms** | 5.07 ms |
| **`write` p99 wall-clock (200-field)** | <5 ms (hardware-dependent) | — | **~5 ms** |
| **`write_sync` p99 wall-clock** | <100 ms | **7.07 ms median** | 9.80 ms |

**The XADD floor is the wall-clock budget at p50 — Pyforge adds <120 µs
on top.** Pydantic at 16.7 µs on a 200-field schema is **30× under the
500 µs escape-hatch trigger**; the parking-lot "swap to hand-rolled
validator" item is solidly closed for this workload.

The original "<2 ms p99" build-plan target was set against a different
Redis configuration (in-memory, no AOF). On WSL2 / Docker Desktop /
`appendfsync everysec`, the realistic ceiling is ~5 ms p99 — the
producer cannot beat the Redis floor. The library hits the **CPU**
budget (<400 µs added) easily; the **wall-clock** number is a
deployment/network knob, not a library knob.

Sub-component thresholds in
[`benchmarks/regression/thresholds.yml`](../../benchmarks/regression/thresholds.yml)
isolate regressions: a drift in `wal_write_*` with flat
`pydantic_validate_*` and `msgpack_pack_*` localizes the regression to
the network or the redis-py serialization layer, not our code.

Sub-component thresholds in
[`benchmarks/regression/thresholds.yml`](../../benchmarks/regression/thresholds.yml)
isolate regressions: a drift in `wal_write_*` with flat
`pydantic_validate_*` and `msgpack_pack_*` localizes the regression to
the network or the redis-py serialization layer, not our code.

## Why sync producer, async consumer

Step 10's consumer is `asyncio`-based because it fans out to two
destinations (shared memory + Parquet) and benefits from coalescing
ACKs and Parquet flushes on independent cadences. The producer's hot
path is a tight validate → pack → XADD path with no I/O fan-out;
making it `async` would force every caller to `await` for no
measurable throughput benefit and would couple the public API to the
caller's event-loop choice.

If a real workload ever needs an `asyncio.Redis` producer, ship it as
`WALProducerAsync` next door, sharing the factory and the msgpack
format. Don't bend the sync producer into an async-shaped hybrid.

## Why `write_sync` is a separate method (preserves invariant #3)

The build-plan correction (build_steps line 962) explicitly notes that
the original design had the producer write to both the WAL **and** the
online store synchronously. That makes the producer a second writer
and reintroduces the inconsistency the WAL exists to prevent. Locked
forever:

> The producer only XADDs. The consumer (Step 10) is the only writer
> to shared memory and to Parquet.

`write_sync` honors this by *polling* the consumer's processed-key
side table — the producer never touches the segment. The cost is the
consumer round-trip latency (5–50 ms typical, 100 ms timeout default),
which is documented at the API boundary. Callers who need
sub-millisecond read-your-own-writes should not use `write_sync` — they
should use the async `write` and accept eventual consistency, or build
a different system.

## Wire format: msgpack list in name_hash order

XADD payload uses four bytes-keyed fields:

| Field | Encoding |
|---|---|
| `b"schema"` | `schema.__name__.encode("utf-8")`, cached per (producer, class) |
| `b"entity_id"` | `entity_id.encode("utf-8")` |
| `b"event_time_ns"` | `str(event_time_ns).encode("ascii")` (Streams stores str values; ASCII int rendering is the cheapest legal encoding) |
| `b"blob"` | msgpack-packed `[v0, v1, ..., vN]` in **name_hash order** |

Three reasons for the list-not-dict shape:

1. **~30–50% smaller wire size at 200 fields.** No key strings means
   the blob fits in fewer Redis Stream entries' AOF append buffers and
   the round-trip costs less.
2. **~2× faster encode/decode at 200 fields.** Skipping per-key
   tokenization on both sides matters when validate + pack is already
   25–60% of the per-call CPU.
3. **Name-hash order is already canonical.**
   `pyforge.schema.compile_schema()` sorts by name_hash; the consumer
   reproduces the order with the same call. No second source of truth.

`schema.__name__` is the wire identifier for the schema. **Locked
assumption (this ADR):** schema class names are globally unique within
a Pyforge deployment. Step 15 (schema evolution) revisits this with
versioning if/when needed.

## Pydantic factory is internal + memoized

`pyforge._internal.pydantic_factory.pydantic_model_for(schema)` is
private (in `_internal/`) for one reason: if pydantic's per-call cost
ever bites the 10k writes/sec budget — measured threshold is **500 µs
at 200 fields**, encoded as the `pydantic_validate_200_field` regression
gate — we can swap the implementation for a hand-rolled validator that
uses `compile_schema(schema)` metadata directly (~5–10 µs/call) without
breaking any public surface.

The factory:

- Memoizes by **class identity** (`type` object), not `__name__`. Two
  distinct classes with the same `__name__` get distinct models. This
  matches Step 15's expected atomic-class-swap pattern.
- Returns `(field_name → pydantic field)` definitions ordered by
  name_hash, exposed as `field_order_for(schema) -> tuple[str, ...]`.
  The producer iterates this tuple to extract validated values via
  `getattr` rather than `model_dump()`, saving ~50 µs at 200 fields by
  avoiding the intermediate dict allocation.
- Sets `model_config = ConfigDict(extra="forbid", frozen=True)`. No
  global strict mode (see §5 below).

## NaN / Inf accepted at validation, not rejected

Every float field is built with `Field(allow_inf_nan=True)`. Locked.
Three reasons in priority order:

1. **Invariant #12 contract.** Step 5's parity tests
   ([`tests/property/test_assembly_parity.py`](../../tests/property/test_assembly_parity.py))
   verify NaN bit patterns round-trip through `_assemble_core`; the
   `fastmath=False` lock exists for exactly this purpose. If the
   producer rejects NaN at write time, the producer would refuse
   inputs the rest of the stack is designed to accept — incoherent
   end-to-end.
2. **NaN-as-missing-feature is standard ML semantics.** XGBoost,
   LightGBM, sklearn HistGradientBoosting all treat NaN as a distinct
   signal. A feature store that strips it at the boundary is broken-
   by-default for the workloads this library exists to serve.
3. **±Inf is rarer but real.** Score-clipping pipelines and rare
   boundary cases produce ±Inf; the assembly kernel handles it
   correctly because `fastmath=False`.

Zero perf cost vs reject — identical pydantic validator instruction
count.

**Trap to avoid: do NOT enable `ConfigDict(strict=True)` globally.**
Strict mode would reject `np.float32(nan)` and any numpy-scalar input,
because numpy scalars aren't `float` instances. Default coercion +
`allow_inf_nan=True` is the correct combination. The regression-guard
test [`test_pydantic_factory.py::test_numpy_float32_scalar_nan_accepted_via_default_coercion`](../../tests/unit/test_pydantic_factory.py)
locks this against future tightening PRs.

## Hot-path allocation discipline

The producer eliminates per-call allocations the standard pydantic /
msgpack idioms would force:

- **Pydantic model class** — memoized; first call ~1 ms, subsequent
  ~50 ns dict lookup.
- **`msgpack.Packer()`** — instance attribute, reused across calls.
  ~10–20% faster than `msgpack.packb(...)` at 200 fields by skipping
  per-call Packer construction.
- **XADD field-name keys** — module-level bytes constants
  (`_F_SCHEMA`, `_F_ENTITY_ID`, `_F_EVENT_TIME`, `_F_BLOB`).
- **Schema-name encoding** — per-`(producer, schema)` cache; first
  write per schema pays one `.encode()`, subsequent writes don't.
- **Prometheus label children** — pre-warmed at construction, cached
  on the instance. Step 7's GC-callback lesson: `Histogram.labels(...)`
  allocates a tuple key + dict slot on first use; doing it eagerly
  moves that work out of the hot path.

The only unavoidable per-call allocations: one validated pydantic
instance, one values list (a list comprehension over getattr), the
msgpack output bytes, the entity-id encode, the event-time render,
and the 4-entry XADD field dict. None can be eliminated without
breaking either pydantic's validation contract or redis-py's
mapping-shaped XADD signature.

## Polling cadence (locked)

`write_sync` polls with **1 ms initial backoff, exponential, 10 ms
cap**. Tighter (100 µs start) wastes Redis RTT cycles because the
consumer cycle is 5–50 ms; looser (10 ms start) bloats p50 by 10 ms.
Locked because both endpoints have asymmetric cost: undershooting
adds Redis load, overshooting adds latency floor.

## Consequences

- The 10k writes/sec target is plausible on a well-tuned machine but
  not guaranteed. The headline number depends on AOF fsync cadence,
  Redis-server NUMA placement, and pydantic's per-call cost on the
  caller's specific schema. Sub-component benchmarks let us localize a
  miss.
- `WriteSyncTimeoutError` is the only public exception type added. It lives
  in `pyforge.wal` per the per-module exception convention
  (`pyforge.shm.SegmentNotFoundError`,
  `pyforge.layout.CapacityExceededError`,
  `pyforge.serving.EntityNotFoundError`). Step 12 may consolidate into
  a `pyforge.exceptions` module if the count crosses ~5; not yet.
- Future flexibility preserved: pydantic can be replaced with a hand-
  rolled validator without breaking any public API; the wire format
  can extend to add a schema-version field by changing the `b"schema"`
  encoding (versioned name) without touching `b"blob"`.

## Open follow-ups

- **Step 10 must accept the wire format.** The consumer reads
  `b"schema"` to look up the FeatureSchema class, then unpacks
  `b"blob"` as a list and zips it back against
  `compile_schema(schema)` (already sorted by name_hash). Don't change
  the wire format in Step 10.
- **Step 15 (schema evolution) decides whether to version the wire
  identifier.** Today's `b"schema"` value is `schema.__name__`; that
  works for one version of one class. Atomic class swap will need
  either a versioned name (`b"User:v3"`) or a separate `b"version"`
  field.
- **Step 16 benchmarks may trigger the pydantic escape hatch.** If
  `pydantic_validate_200_field` consistently exceeds 500 µs on the
  reference hardware, swap the factory implementation for hand-rolled
  validation using `compile_schema(schema)` metadata. Public API stays
  unchanged.
