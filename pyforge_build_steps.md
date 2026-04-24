# Pyforge — Step-by-Step Build Guide

> This document is the executable companion to `pyforge_project_spec.md`. The spec answers "what and why"; this document answers "how, in what order, and how do I know when each piece is done."
>
> What's in here:
>
> - **Background research** for every step, so you understand *why* before you touch code.
> - **Testing discipline** that goes beyond "one test proves it works": unit, integration, property, chaos, and benchmark regression tests at the right layer for each step.
> - **A foundation phase (Step 0)** covering CI, tooling, and observability. These have to exist before any feature work, or you will bolt them on badly later.
> - **Explicit risk triggers** at the end: which benchmark signals would force a rethink of which design call, and what the fallback is.
>
> A handful of steps carry `PLAN CORRECTION` callouts. These document known pitfalls — things a reasonable first-pass design would get wrong (CPython's `resource_tracker` bug, PyArrow's nonexistent append mode, per-read refcounting, etc.) — along with the corrected approach and why.

---

## How to Read This Document

Each step has the same shape:

1. **Goal** — one sentence.
2. **Why now** — what earlier steps unlock this, what later steps depend on it.
3. **Background you need** — the concepts and gotchas from external sources, distilled. Read this *before* writing code.
4. **Design decisions** — the choices available and which one this project takes, with reasoning.
5. **Build** — pseudocode or skeleton code only where it materially helps. Otherwise prose. The goal is to communicate the shape of the code, not dictate every line.
6. **Tests** — broken down by layer. Every step has at least unit tests; some have integration, property, and chaos tests.
7. **Benchmark** — what to measure, what number to beat, and how to guard it in CI.
8. **Acceptance criteria** — binary. A step is either done or not.
9. **Common failure modes** — what will go wrong, and how to recognize it early.

The rule is unchanged from the original plan: **a step is not done until a test proves it is**. Production rigor means that test is also running in CI, with a regression guard on the benchmark.

---

## Environment Assumptions

- Dev + CI target: **Linux** (WSL2 on the dev machine, Debian/Ubuntu base image in Docker and CI). Native Windows is out of scope for correctness — `multiprocessing.shared_memory` has real behavioral differences on Windows that are not worth carrying.
- Python **3.12+** (for `sys.monitoring` / PEP 669 — low-overhead production profiling).
- Redis **7.2+** with AOF always-fsync.
- PyArrow pinned to **14.x** (API stability; PyArrow breaks APIs between majors).
- Numba **0.59+**.

If the development environment doesn't match this list, Step 0 is to fix that — not to start Step 1 anyway.

## Fixed Project Constraints

These are locked-in decisions. The plan is built around them. Revisiting them later is a rewrite, not a tweak.

- **Scale ceiling: 1 M entities, 200 features.** This sizes the slot table (48 MB), feature region (~800 MB), and all benchmarks. Beyond this, the answer is horizontal sharding by entity-id hash range across multiple Pyforge instances — not a bigger single node. The README must state this scope explicitly so "doesn't this not scale?" has a one-sentence answer.
- **Latency targets are conditional, not a single number.**
  - Warm cache, ≤ 50 fields: **5 µs p99**. This is the headline.
  - Cold cache, 200 fields with a 128-dim embedding: **20–50 µs p99**. Still 6–15× faster than Redis; published with the headline as a footnote.
  - Benchmarks report both. The README lists both. Honest beats heroic.
- **Default write path is async.** Writes go through the WAL; the online store is updated by the consumer. Gap between "write confirmed" and "visible online" is ~5–50 ms under normal load.
- **Opt-in `write_sync` for read-your-own-writes.** `WALProducer.write_sync(..., timeout=100)` XADDs, then polls the processed-sidetable until the consumer confirms processing or the timeout fires. Preserves the single-writer-to-shm invariant (Step 10 design); pays the consumer round-trip latency. Documented as an advanced use case, not the default.
- **Linux / WSL2 only.** Native Windows support would require a different shared-memory layer (`CreateFileMapping`) and is a separate project.
- **Single-process trust model.** No auth, no network surface. Pyforge is imported by a trusted process. If it ever gets exposed over a network, that's a new project with a new security design.

---

# Step 0 — Foundation: Tooling, CI, and Observability Skeleton

**Goal:** Stand up everything a production-quality Python library needs *before* writing any feature code: reproducible env, lint, type check, test runner, CI, Docker dev stack, and a logging/metrics skeleton that every later step will emit into.

**Why now:** This is the step the original plan skips. The consequence of skipping it is:
- Phase 3 introduces Numba, but there is no CI that catches Numba compilation regressions.
- Phase 6 writes benchmarks that have never been run in CI, so regressions land silently until the README is being written.
- Every module adds `print()` statements for debugging, and then you retrofit logging at the end.

Doing this first costs one day. Doing it last costs three.

## Background you need

**`uv` vs `pip`/`poetry`:** `uv` (Astral) is the 2024+ standard for fast, reproducible Python envs. It resolves dependencies in seconds, locks to a `uv.lock` file, and wraps `venv` natively. Use it.

**`ruff` vs `black + flake8 + isort`:** `ruff` replaces all three, is ~100x faster, and has a superset of their rules. Use it.

**`mypy` vs `pyright`:** Either is fine. `pyright` is faster and has better incremental mode; `mypy` has the bigger ecosystem. Pick one and commit.

**Structured logging:** `structlog` emits JSON log lines with rich context, which matters when you have a serving process, a WAL consumer, and a watchdog all logging simultaneously. Plain `logging.info("...")` forces you to grep.

**Metrics:** `prometheus_client` exposes a `/metrics` endpoint and a small set of primitives (Counter, Histogram, Gauge). You want histograms from Step 1 onward so that the hot-path latency claim is quantified as you build it, not retrofitted.

**GitHub Actions:** The cheapest CI to set up. One workflow for lint + type + unit tests on every push. A second workflow that runs benchmarks on main and alerts on regression.

## Design decisions

1. **One repo, one package.** `pyforge/` is the library, `tests/` is tests, `benchmarks/` is benchmarks. Do not split into a monorepo yet.
2. **Docker for Redis only.** Python itself runs on the host (in WSL2 or Linux). Running Python inside Docker during dev adds a debugging layer that pays off only at deploy time.
3. **Observability is opt-in but always available.** The library exposes a `pyforge.metrics` module that registers histograms/gauges. If the user doesn't start a `prometheus_client` HTTP server, the counters still increment in memory and are visible in tests.
4. **No `__init__.py` exports until Step 12.** Import from the specific module (`from pyforge.schema import FeatureSchema`). Public API surface is decided at the end, not discovered step by step.

## Build

### Repo layout

```
pyforge/
├── pyforge/
│   ├── __init__.py         # empty until Step 12
│   ├── _internal/          # private helpers; not part of public API
│   ├── metrics.py          # Step 0: Prometheus metric registry
│   ├── logging.py          # Step 0: structlog configuration
│   ├── schema.py           # Step 1
│   ├── shm.py              # Step 2 (shared memory wrapper, not "registry")
│   ├── layout.py           # Step 3 (slot table / entity-id indirection)
│   ├── serving.py          # Steps 4–8
│   ├── assembly.py         # Step 5 (Numba module, isolated for cache reasons)
│   ├── wal.py              # Steps 9–10
│   ├── offline.py          # Steps 11–13
│   ├── watchdog.py         # Step 14
│   └── evolution.py        # Step 15
├── tests/
│   ├── conftest.py         # Redis fixture, shm cleanup fixture, gc isolation
│   ├── unit/
│   ├── integration/
│   ├── property/           # Hypothesis
│   └── chaos/              # kill -9 scenarios
├── benchmarks/
│   ├── conftest.py         # benchmark-specific fixtures (gc.disable, etc.)
│   ├── results/            # committed JSON + SVG outputs
│   └── regression/         # threshold configs consumed by CI
├── .github/
│   └── workflows/
│       ├── ci.yml
│       └── benchmark.yml
├── docker/
│   └── docker-compose.dev.yml
├── pyproject.toml
├── uv.lock
├── ruff.toml
├── mypy.ini
└── README.md
```

### `pyproject.toml` essentials

```toml
[project]
name = "pyforge"
version = "0.1.0"
requires-python = ">=3.12"
dependencies = [
  "numpy>=1.26,<2.0",
  "numba>=0.59,<0.61",
  "pyarrow>=14.0,<15.0",
  "redis>=5.0,<6.0",
  "pydantic>=2.5,<3.0",
  "psutil>=5.9",
  "structlog>=24.0",
  "prometheus-client>=0.19",
]

[project.optional-dependencies]
dev = [
  "pytest>=8.0",
  "pytest-benchmark>=4.0",
  "pytest-asyncio>=0.23",
  "pytest-cov>=5.0",
  "hypothesis>=6.100",
  "ruff>=0.5",
  "mypy>=1.10",
  "py-spy>=0.3",
]
```

### Docker for Redis (WSL2/Linux)

```yaml
# docker/docker-compose.dev.yml
services:
  redis:
    image: redis:7.2-alpine
    command: redis-server --appendonly yes --appendfsync always --maxmemory-policy noeviction
    ports: ["6379:6379"]
    volumes: ["redis_data:/data"]
    # PLAN CORRECTION: Docker defaults /dev/shm to 64 MB inside the container.
    # This doesn't affect Redis, but the pattern must be documented here so
    # we remember it when we containerize the pyforge test runner later.
    shm_size: 2gb
volumes:
  redis_data:
```

### Observability skeleton

`pyforge/logging.py` — configure `structlog` to emit JSON, inject a process-id and a schema-name context variable.

`pyforge/metrics.py` — register three histograms upfront (they'll be filled in later steps):

- `pyforge_read_latency_seconds{schema, path}` — path ∈ {shm, redis}
- `pyforge_gc_pause_seconds` — registered in Step 7
- `pyforge_wal_lag_seconds` — registered in Step 10

Expose a `start_metrics_server(port=9100)` helper that calls `prometheus_client.start_http_server`. It's optional — if not called, the counters still work.

### CI (GitHub Actions)

Two workflows:

**`ci.yml`** — runs on every push/PR:
1. Checkout.
2. `uv sync`.
3. `ruff check .` and `ruff format --check .`.
4. `mypy pyforge/`.
5. Start Redis via `services:` in the job config.
6. `pytest tests/unit tests/integration --cov=pyforge --cov-fail-under=85`.

**`benchmark.yml`** — runs on pushes to main:
1. Same setup.
2. `pytest benchmarks/ --benchmark-autosave --benchmark-compare=<last-main-run>`.
3. Fail if any benchmark regresses by more than the threshold in `benchmarks/regression/thresholds.yml`.

Without the benchmark workflow, performance drifts silently. With it, the headline "5µs p99" claim is defended on every merge.

## Tests

This step's test is: *does the whole stack run end-to-end?*

- `pytest tests/` passes (even with zero real tests, collection works).
- `ruff`, `mypy` pass on the empty package.
- `docker compose -f docker/docker-compose.dev.yml up -d` starts Redis, and a smoke test connects to it.
- `pyforge.metrics.start_metrics_server()` exposes `/metrics` on localhost.

## Acceptance criteria

- `git clone && uv sync && pytest && ruff check . && mypy pyforge/` is green on a fresh checkout.
- CI badge in the README is green.
- A developer can bring up Redis, run a smoke test, and see a Prometheus scrape succeed. All in under 5 minutes.

## Common failure modes

- **"CI is slow so I'll skip it locally."** Pre-commit hooks solve this. Install `pre-commit` and register ruff + mypy. Local feedback in ~2 seconds.
- **Docker-in-WSL2 networking.** On WSL2, `localhost` resolves differently from what Windows sees. Always use `127.0.0.1` in test connection strings.

---

# Step 1 — Schema and Offset Table Compiler

**Goal:** Define feature schemas in Python and compile them, once, into a fixed-layout offset table used by every subsequent step.

**Why now:** Every later step reads or writes data at specific byte offsets in shared memory. If the offset table is wrong, every later test fails in ways that look like a shared-memory bug but are actually a schema bug. Get this right first.

## Background you need

**Cache line alignment (64 bytes):** x86-64 CPUs fetch RAM in 64-byte chunks ("cache lines"). A field that straddles a cache line boundary costs two fetches. Aligning every field start to a 64-byte boundary is a small space cost for a meaningful read-speed gain when the hot path is a tight copy loop. ARM (including Apple Silicon) uses 64-byte or 128-byte cache lines depending on the chip; 64-byte alignment is safe on both.

**NumPy structured arrays as the offset table:** A structured array is a NumPy array whose dtype is a C-like struct. It gives you contiguous, typed, indexable access without Python object overhead. Iterating over it in a Numba function is free; iterating over a list of dicts is not.

**Hashing field names:** Comparing integer hashes is an order of magnitude faster than comparing Python strings. A 64-bit hash of each field name, computed once at compile time, is the fast lookup key. Use `xxhash` or stdlib `hashlib.blake2b(digest_size=8)`. Stdlib is fine and removes a dependency — pick that.

**Pydantic v2 for validation:** Pydantic v2 is written in Rust. Validation of a schema class happens at *definition* time, not at serve time. This is a cost paid once per process startup, not once per request. Don't use v1.

## Design decisions

1. **Fields are declared, not inferred.** No auto-detection. Explicit `dtype` and `shape` on every `FeatureField`. This prevents the class of training/serving skew bugs where a model trains on float64 and serves on float32.
2. **`dtype` is a closed enum** — `float32`, `float64`, `int32`, `int64`, `uint8` (for bitmasks / categorical IDs up to 256). Adding dtypes later is cheap. Adding types now that we never use is clutter.
3. **Variable-length fields are forbidden** in the initial version. A feature with shape `(128,)` is fine. A feature with shape `(None, 128)` is not. Variable shape per entity breaks the contiguous-layout story. If this limitation bites later, revisit — but do not design for it now.
4. **Schema version is part of the class.** `version: int` is a required class attribute. It is baked into the shared-memory segment name and the header CRC. Without it, schema evolution (Step 15) has no anchor.

## Build

The public surface:

```python
from pyforge.schema import FeatureSchema, FeatureField, dtype, compile_schema

class UserFeatures(FeatureSchema):
    version = 1
    fields = [
        FeatureField("age_normalized", dtype.float32),
        FeatureField("session_count_7d", dtype.int32),
        FeatureField("ltv_score", dtype.float32),
        FeatureField("behavior_embedding", dtype.float32, shape=(128,)),
    ]

table = compile_schema(UserFeatures)
# table is a NumPy structured array with columns:
#   name_hash (uint64), byte_offset (uint64), dtype_code (uint8),
#   element_count (uint32), byte_count (uint32)
```

The compile function:

1. Validates uniqueness of `name`s and known dtypes (Pydantic).
2. Hashes each name to `uint64` via `blake2b(name, digest_size=8)`.
3. Walks fields in declaration order, computing byte offsets, **rounding each offset up to the next multiple of 64**.
4. Returns the structured array sorted by `name_hash` so that lookups can use `np.searchsorted` (O(log n), no allocations).

Also expose `total_segment_size(schema)` → header (16 bytes) + last field's offset + last field's byte count, rounded up to a page boundary (4 KiB). Page-aligned segment sizes avoid small surprises when the OS maps memory.

## Tests

**Unit (fast, deterministic):**

- Single-field float32 has byte_count = 4.
- Embedding of shape (128,) float32 has byte_count = 512.
- Two scalar fields: second field's offset is 64, not 4 (alignment).
- Duplicate field names raise `ValueError` at class-definition time.
- Unknown dtype raises at class-definition time.
- Schema with 0 fields raises (degenerate; probably a mistake).
- `total_segment_size` is always a multiple of 4096.
- Offset table is sorted by name_hash (invariant used later).

**Property (Hypothesis):**

- For any generated valid schema (up to 100 fields, random dtypes, random shapes up to (256,)):
  - All byte offsets are multiples of 64.
  - No two fields' byte ranges overlap.
  - `total_segment_size` ≥ header + sum of byte_counts.
  - The final offset + byte_count ≤ `total_segment_size`.
- Hash collisions are astronomically unlikely at this size, but assert uniqueness of `name_hash` values per schema just in case.

## Benchmark

`compile_schema` is not hot-path, but it shouldn't be slow either. Target: < 1 ms for a 200-field schema. Capture it so a later refactor doesn't accidentally make it take 50 ms (which would hurt test suite speed).

## Acceptance criteria

- All unit + property tests pass.
- `mypy` is happy with the public types.
- The schema module has zero runtime imports of `numba`, `redis`, or `pyarrow`. It's a pure data-structures layer.

## Common failure modes

- **Subtle hash implementation change breaks persisted segments.** Pin the hash algorithm (blake2b with a fixed digest size and no key) and write a test that asserts `hash("age_normalized")` has a specific known value. If someone changes the hash function in the future, this test fails loudly instead of silently breaking every persisted segment.
- **Computing offsets in a loop with floating-point math.** Never. Use integer arithmetic only. `(offset + 63) & ~63` is the classic round-up-to-64.

---

# Step 2 — Shared Memory Allocator and Lifecycle

**Goal:** Allocate a shared-memory segment sized from a compiled schema, with a 16-byte header, referenced by other processes via a name stored in Redis. Handle the lifecycle correctly — including the Python `resource_tracker` bug that the original plan does not address.

**Why now:** The schema from Step 1 is inert until a real memory segment exists. This step is also where the biggest known-bad bug in Python's standard library lives: if you don't work around it explicitly, your segments will be destroyed out from under readers.

## Background you need

### ⚠️ PLAN CORRECTION: `multiprocessing.shared_memory.SharedMemory` resource_tracker bug

This is a real, multi-year open issue in CPython ([#82300](https://github.com/python/cpython/issues/82300), [#104291](https://github.com/python/cpython/issues/104291), [#38119](https://bugs.python.org/issue38119)):

> When **any** Python process constructs a `SharedMemory(name=..., create=False)` — i.e., opens an existing segment to read — the `multiprocessing.resource_tracker` subprocess registers that segment for cleanup on process exit. When that process exits, `resource_tracker` calls `shm_unlink()` on the segment, **destroying it for every other process that still has it open**.

The original build plan's Phase 2 describes `read()` as "returns a NumPy array view of the entity's feature vector in shared memory". If that `read` path opens a `SharedMemory` object using the stdlib wrapper, then when the reader process exits, it will unlink the segment and every other reader will get a `FileNotFoundError` or stale data. This is not a rare edge case — it is the default behavior.

**Workarounds, in order of preference:**

1. **Write a thin POSIX wrapper.** Call `posix_ipc` (PyPI library) or directly wrap `shm_open` / `mmap` via `ctypes`. This bypasses `resource_tracker` entirely. Only the segment creator ever calls `shm_unlink`. This is the approach a serious project takes.
2. **Patch `resource_tracker` on the consumer side.** After opening, `multiprocessing.resource_tracker.unregister(shm._name, "shared_memory")`. This is fragile — it depends on a private API that has changed across Python versions.

Pyforge takes option 1: implement a small `pyforge._internal.posix_shm` module with `create`, `open_existing`, `close`, `unlink`. Every later step uses this module, never `multiprocessing.shared_memory`.

### ⚠️ PLAN CORRECTION: reference counting on every `read()` kills the latency claim

The original plan says:

> Every call to `read()` should `INCR` the refcount at the start and `DECR` at the end.

A Redis `INCR` over localhost is ~30–80 µs round-trip. If the headline claim is "5 µs p99 reads", then `INCR`+`DECR` on every read is *20x* the entire latency budget. This design is wrong.

**The correct design:**

- Refcounts are **per-open**, not per-read. A process opens the segment once, `INCR`s once, reads many times, `DECR`s once on close.
- Opens happen at process start (or at schema-registration time). They are not on the hot path.
- The segment is safe to read for the entire time the refcount is held, because the writer-side guarantees the segment's layout doesn't change mid-read. Schema evolution (Step 15) allocates a *new* segment and atomically flips the pointer; the old segment lives until its refcount drops to zero.

This is the invariant that makes the read path lock-free and allocation-free. Respect it.

### Other background

**Redis Lua scripts for atomic multi-key ops.** `INCR ref-new`, `DECR ref-old`, `SET current = new`, and `SADD cleanup-queue old IF ref-old == 0` must happen as one atomic operation. Lua scripts are evaluated server-side in Redis with guaranteed atomicity.

**CRC32 in the segment header.** The 16-byte header contains a 4-byte magic number ("PYFG"), a 4-byte schema version, and an 8-byte CRC32 of the compiled offset table. On open, the reader recomputes the CRC and refuses to use the segment if it doesn't match. This catches the case where two code versions disagree about the schema — a subtle bug that otherwise manifests as reading garbage floats.

**`mmap` vs `shm_open`.** On Linux, POSIX shared memory is implemented on top of `tmpfs` mounted at `/dev/shm`. `shm_open` returns a file descriptor that you `mmap` into the process's address space. They're two steps of the same operation. The `posix_ipc` library wraps this cleanly.

**`/dev/shm` sizing in containers.** Docker gives containers 64 MB of `/dev/shm` by default. A 200-feature × 1 M entities × float32 schema is 800 MB per segment. Always set `shm_size: 4gb` (or larger) on the container running Pyforge.

## Design decisions

1. **Segment naming:** `pyforge_{schema_name}_v{version}_{uuid4().hex[:8]}`. The UUID suffix allows rolling upgrades where two versions of the same schema coexist briefly.
2. **The creator is the only unlinker.** No other process ever calls `unlink`. The watchdog (Step 14) is the only exception — it acts on behalf of a process that crashed.
3. **Header layout is fixed forever:** 4 bytes magic + 4 bytes schema version + 8 bytes CRC32 of the compiled offset table, zero-padded to 16 bytes. If you need more in the header later, add a new magic number and treat it as a schema migration.
4. **No entity data in this step.** This step only allocates the segment and writes the header. Entity storage (the slot table) is Step 3. This keeps each step small and testable.

## Build

`pyforge/_internal/posix_shm.py`:

- `create(name: str, size: int) -> SegmentHandle` — `shm_open(O_CREAT|O_EXCL|O_RDWR)`, `ftruncate`, `mmap`.
- `open_existing(name: str) -> SegmentHandle` — `shm_open(O_RDWR)`, `fstat` for size, `mmap`. **Does not register with any cleanup tracker.**
- `close(handle)` — `munmap` + `close(fd)`.
- `unlink(name: str)` — `shm_unlink`. Only called by the creator or the watchdog.

`pyforge/shm.py`:

- `Segment` class wrapping a `SegmentHandle` plus a memoryview of the mapped region.
- `SegmentRegistry` class (in-memory, not the cross-process "registry" the original plan conflates):
  - `create(schema) -> Segment` — allocates, writes header, registers name in Redis (`SET pyforge:schema:{name}:current {segment_name}`), sets refcount to 1.
  - `open_current(schema) -> Segment` — reads current segment name from Redis, `INCR` refcount (once, not per read), opens via POSIX wrapper.
  - `close(segment)` — `DECR` refcount via Lua script that also adds to cleanup queue if refcount hits 0.

Hydration on open: read the 16-byte header, verify magic, verify CRC matches the locally-compiled offset table's CRC. If mismatch, raise `SchemaCRCMismatch` — do not proceed.

## Tests

**Unit:**

- Create → header written with correct magic, version, CRC.
- Create two segments for the same schema → names differ.
- Open a segment with a CRC that doesn't match the local schema → `SchemaCRCMismatch`.
- Close decrements refcount.

**Integration (requires Redis):**

- End-to-end create → Redis has `pyforge:schema:X:current` pointing at it → open → close. Refcount goes 0 → 1 → 0.

**Cross-process integration — the critical test:**

- Process A creates a segment and writes a sentinel value (e.g., float 3.14) at byte 16.
- Process B opens the segment and reads 3.14 back.
- Process B exits cleanly.
- Process A reads 3.14 again — **still works**. (This is the test that fails without the `resource_tracker` workaround.)
- Process B was killed with SIGKILL instead of exiting cleanly. Process A still reads 3.14. (The watchdog in Step 14 will later handle refcount cleanup for B, but the segment itself must survive.)

**Chaos:**

- Create 1000 segments of different sizes in a loop. Verify they're all unlinked after the test — `/dev/shm` returns to baseline. A fixture in `conftest.py` should enforce this after every test to prevent accumulating test pollution.

## Benchmark

- `create` (allocate 1 MB segment): < 1 ms.
- `open_existing`: < 100 µs. This runs at schema-registration time, not per request, so don't over-optimize it.

## Acceptance criteria

- All tests pass, including the cross-process test where Process B's exit does **not** destroy Process A's segment.
- A soak test (create/open/close 10,000 times in a loop) shows zero growth in `/dev/shm` usage after completion.

## Common failure modes

- **Using `multiprocessing.shared_memory` "temporarily" in tests.** Every time this has been tried it has silently broken the cross-process test in ways that are intermittent and take hours to diagnose. Don't. Use the POSIX wrapper from day one, even in tests.
- **Forgetting to `msync` before crashing tests.** `mmap`-written data is not guaranteed to be on disk until `msync`. For `/dev/shm` (tmpfs) this doesn't matter — tmpfs is RAM. But if you ever switch to `mmap`-on-a-file for experimentation, this matters a lot.

---

# Step 3 — Entity Layout: Slot Table and ID Indirection

**Goal:** Store many entities in one segment. Map `entity_id` (string) to a slot index in the feature array, with O(1) average lookup and no per-read allocation.

**Why now:** The original plan's Phase 2 punts this: "for now, store one entity per segment — you'll extend this later." But the design of entity-to-slot indirection is the hardest layout question in the entire project. Deferring it pretends the hard part doesn't exist. Do it before the serving path so the serving path gets the real shape from day one.

## Background you need

**The core problem:** A request says "give me features for `user_8cf3...`". We need to translate that string into a byte offset in the segment, fast, without allocating, and without locking.

**Approaches:**

1. **External hash table in Redis (`HGET pyforge:slot:{schema}:{entity_id}`).** One Redis round-trip per read. Breaks the 5-µs target.
2. **In-process Python dict.** Fast, but each process has to build it on open and keep it in sync with writes from other processes. Sync is the hard part.
3. **Hash table in shared memory.** No sync problem — all readers see the same table. Requires concurrency control for writes.

Pyforge uses **approach 3** with a specific shape: a **power-of-2 open-addressing hash table in the segment**, with slots that are either empty, tombstoned, or occupied. Writes are serialized through a single writer (Step 9's WAL consumer); reads are lock-free.

**Why open addressing:** A single contiguous array. Cache-friendly. No pointer chasing. Insertion is O(1) amortized with linear probing.

**Why power-of-2 size:** `hash & (size - 1)` instead of `hash % size`. Bitwise AND is meaningfully faster, and matters here because this lookup is on the hot path.

**Load factor:** Keep the table at ≤ 50% load. Above that, clustering makes linear probing pathologically slow. Resizing is handled as a schema-evolution event (Step 15) — allocate a new segment with a bigger table, rehash.

**Entity IDs:** Hashed to 64-bit via blake2b (same family as schema names). The hash is stored alongside the slot so comparison is an integer compare, not a string compare. Full ID is also stored for collision resolution (a 64-bit hash has measurable collisions at 10M entities — birthday paradox).

## Design decisions

1. **Fixed capacity at segment creation.** No resize in place. Growth is a schema-evolution event.
2. **Sized for 1 M entities as the project ceiling.** 2 M slots × 24 bytes = 48 MB for the index. Feature region at 200 float32 fields × 1 M entities = 800 MB. Both fit in RAM on any modern server. Beyond 1 M entities, the answer is multiple Pyforge instances sharded by `hash(entity_id) mod N`, not a bigger single segment — 100 M entities in one segment would be 4.8 GB of slot table alone, and the copy-during-schema-evolution becomes a minutes-long operation.
3. **Slot = {u64 hash, u64 id_offset_in_string_pool, u32 flags, u32 feature_row_index}.** 24 bytes per slot. Power-of-2 sizing for `& mask` indexing.
4. **String pool for entity IDs.** A contiguous append-only region in the segment holds the UTF-8 bytes of all entity IDs, deduplicated. The slot's `id_offset_in_string_pool` points into it. This avoids variable-length data in the fixed slot array.
5. **Feature row storage is dense.** The N-th entity stored (in insertion order) occupies rows in the feature storage region. The slot's `feature_row_index` picks the row. This separates the *index* (hash table) from the *data* (feature rows) — you can rebuild the index without touching the data.
6. **Linear probing, not Robin Hood or Cuckoo.** At ≤ 50% load factor, expected probe distance is ~1.5. Robin Hood would marginally tighten the tail at the cost of tracking displacement per slot and shifting entries on insert. Cuckoo would give O(1) worst-case lookup at the cost of dual hashing and occasional full rehashes. Neither buys enough at this scale to justify the complexity. Revisit only if Step 3's stress test shows probe distances out of line with the expected distribution.

Segment layout becomes:

```
[16B header][slot table — fixed size][string pool — fixed size][feature rows — one row per entity]
```

All four regions are 64-byte aligned (from Step 1's rule).

## Build

`pyforge/layout.py`:

- `SegmentLayout` dataclass: computes the four region offsets from schema + capacity.
- `lookup(segment, entity_id) -> feature_row_offset | None` — linear-probe hash lookup, returns byte offset of the entity's feature row, or None if not present.
- `insert(segment, entity_id, row_data) -> bool` — writer-side only. Called exclusively by the WAL consumer (Step 10).

The lookup must be:
- Allocation-free.
- Branch-predictable (linear probe, tight loop).
- Callable from Numba (Step 5) — so it takes plain buffers and integers, not Python objects.

Pseudocode for lookup:

```
h = blake2b_u64(entity_id)
mask = capacity - 1
idx = h & mask
while True:
    slot = read_slot(segment, idx)
    if slot.flags == EMPTY:
        return None
    if slot.hash == h and read_string(slot.id_offset) == entity_id:
        return feature_row_offset(slot.feature_row_index)
    idx = (idx + 1) & mask
```

The Numba version (Step 5) will work on raw `uint8` views and integer arithmetic only. No Python string comparison. The hash match is checked first; the full string comparison is only needed on hash collision, which is astronomically rare.

## Tests

**Unit:**

- Insert one entity → lookup returns correct offset.
- Insert then lookup a missing entity → None.
- Insert the same entity twice → second insert updates in place, doesn't add a slot.
- Fill the table to 50% → still O(1)-ish (assert probe distance ≤ 8 on average).
- Fill to 90% → we should have rejected the insert. Assert a `CapacityExceeded` error.

**Property (Hypothesis):**

- For random entity IDs up to 50% capacity: every inserted ID is findable, no others are found.
- Insertion order doesn't affect correctness (insert [a,b,c] and [c,b,a] → same reachable set).
- No false positives: inserting N distinct IDs produces exactly N lookups that succeed.

**Stress:**

- Insert 1 M entities into a 2 M-slot table. Lookup 10,000 random ones. Zero false negatives. Average probe distance stays < 2.

## Benchmark

- `lookup` (hot path): target < 200 ns per call when the entity is present and the hash is the first probe. This is the tight inner loop.
- `lookup` for a missing entity: < 500 ns.

These are the numbers that make the 5-µs end-to-end claim possible. If they're not met here, the end-to-end claim is unreachable and the design needs to change.

## Acceptance criteria

- All tests pass.
- Benchmark hits the 200 ns / 500 ns thresholds on the dev machine.
- Layout has zero allocations in the lookup path (verified with `tracemalloc` snapshot around a batch of lookups — growth = 0).

## Common failure modes

- **Hash table at load factor 90% appears to work in tests with 100 entries.** Linear probing collapses at 80%+. Always test at scale (≥ 100 k entries) or you won't see it.
- **Forgetting tombstones on deletion.** We don't delete in the initial version, so this is deferred. Document it. When delete is added, tombstones are required to prevent breaking probe chains.

---

# Step 4 — Serving Path Skeleton (Pure Python, No Numba Yet)

**Goal:** End-to-end read path: given a schema and an entity ID, return the assembled feature vector as a NumPy array. Correctness first, performance later.

**Why now:** Step 5's Numba version has to produce byte-identical output to a known-good Python version. Write the Python one first, let it be your oracle.

## Background you need

**Zero-copy NumPy views:** `np.frombuffer(segment.mmap_view, dtype=np.uint8)` creates a NumPy array that shares memory with the segment. No copy. The same `memoryview` can be sliced into typed views per field without copying.

**Contiguous output:** Models (XGBoost, sklearn, PyTorch) expect a single contiguous C-order array of the right dtype. Assembling into a pre-allocated buffer of that exact shape is what makes the hand-off zero-cost.

**When to copy:** The output buffer is a *copy* of the underlying bytes — that's unavoidable; the model will pass them to a predict function that may hold onto them past this request's lifetime. But it's exactly one copy, into a buffer that's re-used.

## Design decisions

1. **No buffer pool yet.** Allocate the output with `np.empty` for now. The pool is Step 6.
2. **No Numba yet.** Plain Python loop over fields. This step is the correctness oracle for Step 5.
3. **Single-entity path only.** Batch is Step 8. Focus.

## Build

Public API target:

```python
segment = registry.open_current(UserFeatures)
vec = serving.assemble(segment, entity_id="user_001")
# vec is np.ndarray, shape (131,), dtype float32, contiguous
model.predict(vec)
```

`pyforge/serving.py::assemble(segment, entity_id)`:

1. `row_offset = layout.lookup(segment, entity_id)`. If None, raise `EntityNotFound`.
2. Allocate output buffer via `np.empty(total_element_count, dtype=np.float32)`.
3. For each field in the offset table:
   - Compute source slice: `row_offset + field.byte_offset : +field.byte_count`.
   - Copy bytes into the output buffer at the field's cursor position.
4. Return the output.

For fields that aren't float32, **cast on copy**: a float64 feature gets narrowed, an int32 gets widened. This is a deliberate choice — models take a single dtype, and unifying here removes per-model cast code.

## Tests

**Unit:**

- Write three fields directly into a segment (via a test helper that bypasses WAL). Call `assemble`. Output matches expected values within float tolerance.
- `EntityNotFound` raised cleanly for missing entity.
- Works for every dtype (float32, float64, int32, int64, uint8) — including the casts.
- Works for shaped fields (embedding of shape (128,) is laid out contiguously in the output).

**Integration:**

- Full loop: create segment, write entity via `layout.insert` (test helper), assemble, verify.

## Benchmark

Record it but **do not optimize yet**. Expected numbers on the dev machine:

- `assemble` for a 4-field schema: ~10–30 µs (Python loop is the bottleneck).
- `assemble` for a 200-field schema: ~100–500 µs.

These are the baselines Step 5's Numba version will improve upon. Save the `pytest-benchmark` JSON output to `benchmarks/results/step04_baseline.json`.

## Acceptance criteria

- Assembled output matches expected bytes for every dtype.
- Baseline benchmark committed to the repo.

## Common failure modes

- **Silent dtype promotion.** NumPy will happily upcast int32 → float64 if you're not careful, doubling your memory copy. Assert `output.dtype == np.float32` in tests.
- **Contiguity loss.** If you ever use `np.concatenate` on views, the result might not be C-contiguous. Always `np.ascontiguousarray` before returning, or construct the output fresh.

---

# Step 5 — Numba Assembly Path

**Goal:** Replace the Python loop with a Numba-compiled function. Verify it produces byte-identical output. Beat the Step 4 baseline meaningfully, or don't adopt it.

**Why now:** Now that correctness is pinned, optimizing is safe. The oracle from Step 4 will catch any Numba-introduced regression instantly.

## Background you need

**`@njit(cache=True)`:** Numba compiles on first call (~100–500 ms, not measured per-request). With `cache=True` it writes the compiled code to `__pycache__/` on disk, so subsequent Python starts skip compilation. Cache invalidates automatically when Numba version, Python version, or source hash change.

**Numba cannot compile arbitrary Python.** No Python dicts, no strings, no exceptions-as-control-flow. Numba functions take NumPy arrays and scalars and return NumPy arrays and scalars. Everything else is a compilation failure with a long error.

**SIMD in Numba:** Numba passes through LLVM, which auto-vectorizes tight numeric loops. For simple `memcpy`-style loops, Numba will often emit vectorized code automatically. You can't rely on it — verify with `--debug-only=loop-vectorize` or by reading the generated assembly — but for simple cases it usually works.

**Is Numba actually faster here?** Honest answer: a raw byte copy is already a single `memcpy` under the hood if you use NumPy slicing correctly. A Python loop over fields adds Python-interpreter overhead per field (a dict lookup for the method, reference counting, GIL). Numba removes that per-field overhead. For a 4-field schema, the difference might be 5 µs vs 1 µs. For a 200-field schema, it's 400 µs vs 20 µs. The crossover is around 20 fields.

### ⚠️ PLAN CORRECTION: Benchmark gate on adoption

The original plan adopts Numba unconditionally. This step adopts Numba **only if** the benchmark shows it beats the Python oracle on the target schema size. If it doesn't — for small schemas, or for schemas dominated by one big embedding — the project uses the Python path and documents why. "Numba or nothing" is dogma, not engineering.

## Design decisions

1. **Isolate the Numba module.** Put all `@njit` functions in `pyforge/assembly.py` and nothing else there. Numba's compilation cost is per-module; separating it lets you import `pyforge` without paying for Numba unless you actually call `assemble`.
2. **Pre-warm on first registration.** When a schema is registered, call `assemble` once on a dummy entity so compilation happens at startup, not on the first real request.
3. **Both paths stay in the codebase.** Python oracle (Step 4) and Numba path (Step 5) both live on. Tests run against both. A config flag picks which one the production path uses. This way, Numba bugs don't block serving.

## Build

Two functions in `pyforge/assembly.py`:

```python
@numba.njit(cache=True, boundscheck=False)
def _assemble_core(
    segment_bytes: np.ndarray,   # uint8 view of segment
    row_offset: int,
    byte_offsets: np.ndarray,    # int64 array
    byte_counts: np.ndarray,     # int64 array
    output_bytes: np.ndarray,    # uint8 view of output buffer
) -> None:
    cursor = 0
    for i in range(len(byte_offsets)):
        start = row_offset + byte_offsets[i]
        count = byte_counts[i]
        for j in range(count):
            output_bytes[cursor + j] = segment_bytes[start + j]
        cursor += count
```

And a higher-level `assemble_numba(segment, entity_id, output_buffer)` that wraps it with lookup and casting.

The inner loop is a byte copy. For some schemas, replacing it with `output_bytes[cursor:cursor+count] = segment_bytes[start:start+count]` is faster because NumPy's slice-assign is a single `memcpy`. Benchmark both; adopt the winner.

## Tests

**Correctness parity (the critical test):**

- A fuzz test with Hypothesis: for 1000 random schemas and entities, `assemble_python(...)` and `assemble_numba(...)` produce byte-identical output. Not approximately equal — `np.array_equal`.

**Cache behavior:**

- Delete `__pycache__` → first call is slow (> 50 ms). Second call < 10 µs. Third process start (cache warm) has first call < 10 µs.

## Benchmark

Four scenarios, all committed. The four cover both headline conditions (warm cache / small schema, warm cache / large schema) and the harder ones (cold cache):

1. **Warm cache, 4-field schema, single entity.** Python baseline vs Numba. Expected Numba p99: ≤ 5 µs. This is the headline number.
2. **Warm cache, 200-field schema with one 128-dim embedding, single entity.** Expected Numba p99: 10–20 µs.
3. **Cold cache, 200-field schema.** Force a cache flush (`numactl` / explicit cache clobber) between calls. Expected p99: 20–50 µs. This is the honest large-schema number.
4. **Crossover scan:** 4, 10, 20, 50, 100, 200 fields. Shows where Numba starts winning over the Python oracle.

The benchmark runs in CI with a regression threshold: if any scenario degrades by more than 20% relative to the committed baseline, CI fails.

## Acceptance criteria

- Numba output == Python output for 1000 random schemas (Hypothesis).
- On the 200-field benchmark, Numba is at least 3× faster than the Python baseline. If not, either debug or adopt the NumPy-slice path.
- Committed flamegraph (`py-spy record --native`) shows `_assemble_core` as the dominant frame, not Python interpreter overhead. The `--native` flag is required to see into the Numba-compiled code.

## Common failure modes

- **Cache stale after a source edit.** Numba's cache invalidates on source hash, but if you edit imported helpers it sometimes doesn't. If you see weird behavior after an edit, `rm -rf __pycache__`.
- **`boundscheck=False` hides out-of-bounds writes.** Run the tests once with `boundscheck=True` to verify correctness, then flip to `False` for production speed.
- **Numba promotes types silently.** Explicit signatures help: `@numba.njit("(uint8[:], int64, int64[:], int64[:], uint8[:])", cache=True)` forces the types and gives faster compilation.

---

# Step 6 — Buffer Pool

**Goal:** Eliminate the `np.empty` call on the hot path. Pre-allocate N output buffers at schema registration; check one out for each request; return it on completion.

**Why now:** `np.empty(200, dtype=np.float32)` is not free — it's ~500 ns plus GC pressure at 50k RPS. Removing it is the last step of the single-entity hot path before the big wins are all in.

## Background you need

**Thread-safety of `collections.deque`:** `deque.popleft()` and `append()` are atomic in CPython because of the GIL. A pool implemented as a deque is thread-safe without locks — for CPython only. If Python ever gets free-threading (PEP 703), this will need revisiting.

**Context managers for checkout/return:** `with pool.checkout() as buf:` guarantees the buffer returns even on exception. A raw `get`/`put` pair would leak buffers on error paths.

**Pool sizing:** Rule of thumb: pool_size = expected_concurrent_requests × 2. At 50k RPS with 1 ms request latency, you have ~50 in-flight. Pool of 128 is comfortable. Pool of 10 will starve under load.

**Backpressure when exhausted:** Three choices — (a) allocate a fresh buffer and log a warning, (b) block until one's available, (c) raise. Pyforge picks (a): the hot path must not block on concurrency primitives, and raising breaks callers who didn't expect it. The warning is the signal to increase the pool size.

## Design decisions

1. **One pool per (schema, batch_size) pair.** A batch-100 pool and a batch-1 pool are separate. Shared pools across sizes require truncation on return, which is error-prone.
2. **Async refill.** When the pool drops below 25% of capacity, a background task refills it. Refill runs on the event loop, not the hot path.
3. **Buffers are handed out zeroed.** A pool that hands out dirty buffers leaks previous-request data into the next response when a bug writes to the wrong region. Zeroing a 200×4 byte = 800 byte buffer is ~50 ns. Cheap insurance.

## Build

```python
class BufferPool:
    def __init__(self, shape, dtype, capacity, refill_task_runner):
        self._buffers = collections.deque(
            np.zeros(shape, dtype=dtype) for _ in range(capacity)
        )
        self._capacity = capacity
        self._refill_task_runner = refill_task_runner
        self._shape = shape
        self._dtype = dtype

    @contextlib.contextmanager
    def checkout(self):
        try:
            buf = self._buffers.popleft()
        except IndexError:
            metrics.pool_miss.inc()
            buf = np.zeros(self._shape, dtype=self._dtype)
        try:
            yield buf
        finally:
            buf.fill(0)  # zero before return
            if len(self._buffers) < self._capacity:
                self._buffers.append(buf)
            self._maybe_trigger_refill()
```

## Tests

- Checkout + return: pool size returns to initial.
- Checkout that raises: still returns the buffer (context manager guarantee).
- Exhaustion: 100 concurrent checkouts from a 10-buffer pool → 90 are fresh allocations, pool-miss metric increments 90 times.
- Returned buffer is zeroed: write 1s into a checked-out buffer, return, check out again, assert all-zero.

## Benchmark

Single-entity assemble with pool vs without pool:

- Without pool (Step 5 baseline): X µs.
- With pool: X - 500 ns, roughly.

If the pool adds overhead instead of removing it, the implementation is wrong. Investigate before moving on.

## Acceptance criteria

- Pool tests green.
- Benchmark shows measurable improvement.
- Metric `pyforge_pool_miss_total` is emitted on exhaustion.

## Common failure modes

- **Zeroing with `buf[:] = 0` is slower than `buf.fill(0)`.** Benchmark before picking.
- **Checkout from inside a checkout.** Nested checkouts of the same pool under high concurrency can deadlock if the pool is small. Test this.

---

# Step 7 — GC Management

**Goal:** Eliminate stop-the-world GC pauses from the serving thread. Move GC to a dedicated thread. Instrument and expose pause durations as a Prometheus histogram.

**Why now:** This is the step that turns a "fast on average" path into a "fast at p999" path. Without it, 1 request in 1000 will randomly take 10–50 ms because GC fired during it.

## Background you need

**Python's generational GC.** Three generations (0, 1, 2). Most short-lived objects die in gen 0. A full gen-2 collection walks every object in the program and is the expensive one. On a process with 1 GB of long-lived data, gen-2 can take 50+ ms.

**`gc.freeze()`:** Moves all currently-tracked objects into a "permanent" generation that's exempt from collection. Call it after process startup (after all long-lived objects like the offset tables and buffer pools are allocated). Subsequent GCs only walk objects created after `freeze()` — a much smaller set. Instagram famously got a 20% speedup from this one call.

**`gc.disable()` risks:** If your code creates reference cycles (common with async code, less common with pure-numeric code), disabling GC leaks memory. The serving path in Pyforge is deliberately cycle-free — no Python objects captured in closures, no callbacks holding references. So disabling is safe *there*. But the WAL consumer, the watchdog, and the metrics server are Python-object-heavy and need GC enabled.

**`gc.callbacks`:** Python calls these before and after every GC. Registering `start` and `stop` timestamps gives you the pause duration. Emit to a Prometheus histogram.

**`sys.monitoring` (PEP 669, Python 3.12+):** A separate topic — a new API for attaching tracing callbacks at ~5% overhead (vs ~2000% for `sys.settrace`). Not strictly needed for GC, but the right answer for any "I want to trace one thing" debugging in production. Register it but gate it behind a config flag.

## Design decisions

1. **GC stays enabled in the process as a whole.** The WAL consumer and watchdog need it. Pyforge doesn't disable GC globally.
2. **`gc.freeze()` at schema-registration completion.** The long-lived objects (offset tables, pools) are all allocated by then.
3. **Gen-2 collections happen on a dedicated thread on a 500 ms timer.** That thread calls `gc.collect(2)`. The GIL still gets grabbed during the collection — but because the freeze dramatically reduced gen-2 size, the pause is short (target: < 2 ms).
4. **No `gc.disable()` on the serving thread.** The original plan calls for this; Pyforge doesn't. Modern CPython's gen-0/1 pauses are sub-millisecond, and disabling creates the cycle-leak risk for little gain given `gc.freeze()` is already doing the heavy lifting.
5. **Instrument everything.** Every GC pause goes into `pyforge_gc_pause_seconds{generation}` histogram.

## Build

`pyforge/_internal/gc_manager.py`:

- `start()` — registers `gc.callbacks`, starts the background `gc.collect(2)` timer thread.
- `freeze()` — wraps `gc.freeze()`. Called from `SegmentRegistry.register` after all setup.
- `stop()` — cleanup for tests.

Callback:

```python
_pause_start = {}

def _gc_callback(phase, info):
    if phase == "start":
        _pause_start[info.get("generation", 0)] = time.perf_counter()
    elif phase == "stop":
        gen = info.get("generation", 0)
        duration = time.perf_counter() - _pause_start.pop(gen, time.perf_counter())
        metrics.gc_pause_seconds.labels(generation=gen).observe(duration)
```

## Tests

- `freeze()` then allocate 10k short-lived dicts → gen-2 collection count doesn't grow (they live in gen 0/1).
- `gc_pause_seconds` histogram has observations after a forced `gc.collect()`.
- In a multi-threaded test, the background GC thread does not cause `assemble` calls to fail or corrupt output.

## Benchmark

The critical measurement:

- 100,000 `assemble` calls, no GC instrumentation, look at p999.
- Same, with GC freeze + timer thread. p999 should be measurably better.
- Same, with GC fully enabled (no freeze, no timer thread). p999 is the bad baseline.

Commit all three distributions. The "GC freeze + timer" distribution should have a much tighter tail than the "fully enabled" distribution.

## Acceptance criteria

- p999 improvement is real and reproducible.
- `pyforge_gc_pause_seconds` is in the metrics scrape.
- No memory leak after a 10-minute soak run (monitored via `psutil` in the soak test).

## Common failure modes

- **`gc.freeze()` before all long-lived allocations are done.** Objects created after freeze go into regular generations and are walked normally. Call freeze *last* during init.
- **The timer thread's `gc.collect()` runs while a request is in flight.** This is expected; the pauses are measured and bounded by how much work the freeze pruned away. If pauses are still long, investigate allocation patterns, not this design.

---

# Step 8 — Batch Assembly

**Goal:** Assemble features for N entities at once into a single contiguous 2D `(N, feature_count)` array. Not N independent calls — a single vectorized operation.

**Why now:** Real workloads (recsys, batch scoring) serve hundreds to thousands of entities per request. Looping over single-entity assembly is the wrong shape — each iteration pays the Python-call overhead.

## Background you need

**Vectorized lookup:** Hash all N entity IDs in a single pass. For each hash, compute the slot index. Probe until found. All of this in one Numba function that takes an array of IDs and returns an array of row offsets.

**Memory locality of the output:** The output is `(N, feature_count)` row-major float32. Writing row by row is cache-friendly. Writing column by column (assemble field A for all entities, then field B) is often *more* cache-friendly on the *source* side because a single field lives contiguously for all adjacent-row entities. Benchmark both orders.

**What scikit-learn, XGBoost, etc. want:** A C-contiguous float32 2D array. If you hand them something else, they internally copy it. Knowing this lets you produce exactly the right shape and skip the internal copy.

## Design decisions

1. **`get_batch(schema, entity_ids, output=None)` is the public API.** Pass in an `output` buffer from the pool or let the function allocate.
2. **Missing entities return a sentinel row.** Zero-filled, with a boolean mask returned alongside indicating which entities were found. Raising would abort the whole batch for one bad ID — that's not what production workloads want.
3. **Numba function is batched.** One `@njit` function that takes the IDs array and the output array. Not N calls to the single-entity version.

## Build

```python
def get_batch(segment, entity_ids, output=None, found_mask=None):
    n = len(entity_ids)
    if output is None:
        output = buffer_pools[(schema, n)].checkout(...)  # context-managed outside
    if found_mask is None:
        found_mask = np.zeros(n, dtype=np.bool_)

    id_hashes = _hash_ids_numba(entity_ids)  # (N,) uint64
    row_offsets = _lookup_batch_numba(segment_view, id_hashes, ...)  # (N,) int64, -1 for miss
    _assemble_batch_numba(segment_view, row_offsets, byte_offsets,
                           byte_counts, output, found_mask)
    return output, found_mask
```

## Tests

Parity with single-entity version:

- `get_batch([e1, e2, e3])` row i matches `assemble(ei)` for every i.
- Missing entity → row is zero, mask[i] is False.
- Empty batch returns (empty, empty) cleanly.

Hypothesis: random batches of random sizes produce results identical to N single calls.

## Benchmark

Batch sizes: 1, 10, 100, 1_000, 10_000.

Compare:
- N × single-entity assemble.
- `get_batch(N)`.

The batch version should be meaningfully faster at N ≥ 10. At N = 1 it may be slightly slower due to setup overhead — that's fine and expected.

## Acceptance criteria

- Benchmark shows batch version is at least 5× faster than N-single-calls at N = 1000.
- Parity tests green.

---

# Step 9 — WAL Producer

**Goal:** Writes go into a Redis Stream *first*, not directly to Parquet. This is the durability boundary.

**Why now:** Step 10's consumer needs something to consume. Building producer before consumer is the obvious order.

## Background you need

**Redis Streams basics:** `XADD stream * field value ...` appends a message. Returns an auto-generated monotonic ID like `1700000000000-0`. `XREADGROUP` reads via a consumer group with per-consumer delivery tracking.

**`MAXLEN ~ 1000000`:** Caps the stream length with approximate trimming (fast). Without it, the stream grows unbounded. Choose a cap that covers worst-case consumer lag (e.g., consumer down for 10 minutes at 10k writes/s = 6 M messages).

**`NOMKSTREAM`:** Avoid accidentally creating a stream on a typo'd name.

**Serialization format inside the message:** Don't pickle — unsafe across versions and for cross-language consumers. Use a small binary format: msgpack is the pragmatic pick. Keys: `schema`, `entity_id`, `event_time`, `values_blob` (a packed binary of field values in schema order).

### ⚠️ PLAN CORRECTION: Producer does not write to the online store synchronously

The original plan has the producer do XADD *and* update the Redis online store (HSET or similar). That makes the producer a two-writer and reintroduces the inconsistency the WAL is meant to prevent.

**The correct design:** the producer only XADDs. The consumer updates both the online store (shared memory via `layout.insert`) and the offline store (Parquet). One writer, two destinations, one ordering.

## Design decisions

1. **Producer is sync for simplicity.** The XADD call blocks until Redis confirms. This is the durability boundary — callers get a write confirmation back.
2. **Acknowledgement strategy:** the caller gets the message ID back. They can query processing status later if needed (rarely needed in practice).
3. **Schema-validated at produce time.** Invalid inputs are caught here, not in the consumer. The consumer assumes validated payloads.
4. **`write_sync` is the opt-in read-your-own-writes path.** Default `write` is fire-and-forget (durability, not visibility). `write_sync(..., timeout=100)` XADDs then polls `pyforge:processed:{msg_id}` until the consumer confirms or the timeout fires. This preserves the "consumer is the only writer to shm" invariant from Step 10 — the alternative (letting the producer write shm directly) would require cross-process locking in the hot write path for a use case that most callers don't need.

## Build

```python
class WALProducer:
    def write(self, schema, entity_id, values, event_time=None):
        validated = pydantic_model_for(schema).model_validate(values)
        blob = msgpack.packb(validated.model_dump())
        msg_id = redis.xadd(
            "pyforge:wal",
            {
                "schema": schema_name,
                "entity_id": entity_id,
                "event_time": str(event_time or time.time_ns()),
                "blob": blob,
            },
            maxlen=1_000_000,
            approximate=True,  # "~" prefix
        )
        return msg_id

    def write_sync(self, schema, entity_id, values, event_time=None, timeout_ms=100):
        msg_id = self.write(schema, entity_id, values, event_time)
        deadline = time.monotonic() + timeout_ms / 1000
        backoff = 0.001  # 1 ms start, cap at 10 ms
        while time.monotonic() < deadline:
            if redis.exists(f"pyforge:processed:{msg_id}"):
                return msg_id
            time.sleep(backoff)
            backoff = min(backoff * 2, 0.010)
        raise WriteSyncTimeout(msg_id, timeout_ms)
```

Expected `write_sync` latency: ~consumer cycle time + 1 ms poll jitter. Under steady load that's 10–50 ms. Document this explicitly — callers who expect sub-millisecond should use the async `write` instead.

## Tests

- `write` produces a monotonic ID.
- Stream length respects MAXLEN after a burst.
- Invalid values (wrong dtype, missing field) → Pydantic error before XADD.
- `write_sync` returns the msg_id after the consumer processes it (integration test with real consumer running).
- `write_sync` raises `WriteSyncTimeout` if the consumer is stopped.

## Benchmark

- `write` writes/sec sustained: target 10k/s on the dev machine.
- `write` p99 latency: target < 2 ms (dominated by Redis RTT + AOF fsync).
- `write_sync` p99: target < 100 ms under steady load; records the consumer-processing tail.

## Acceptance criteria

- Tests green.
- Benchmarks recorded for both `write` and `write_sync`.

---

# Step 10 — WAL Consumer with Idempotency

**Goal:** An async consumer that reads from the Redis Stream, updates the shared-memory segment (online) and the Parquet dataset (offline), and acknowledges. Survives crashes without losing or duplicating data.

**Why now:** Correctness of the offline store depends entirely on the consumer's crash-safety. Build it before the Parquet store so the Parquet store has a real caller from day one.

## Background you need

**PEL (Pending Entries List):** When `XREADGROUP` delivers a message, Redis moves it into the PEL for that consumer. It stays there until `XACK`ed. On restart, the consumer reads its PEL first (with ID `0`), *then* new messages (with ID `>`). This is the two-phase consume pattern documented in Redis's own consumer-group docs.

**At-least-once by default.** Redis Streams give at-least-once delivery via the PEL + XACK model. Exactly-once comes from the consumer being idempotent — applying a duplicate message produces the same result as applying it once.

### ⚠️ PLAN CORRECTION: idempotency via "last processed ID" is wrong

The original plan says:

> Checks if `message_id <= last_processed_id` (idempotency check). If so, skip and acknowledge.

This does not work. Redis Streams IDs are globally monotonic across all consumers — but the PEL can have gaps where one consumer in a group ACKed message `N+1` before another ACKed `N`. A single "last processed" scalar isn't enough.

**The correct design:** idempotency is per-message, via a *side table*. For each message ID, the consumer writes `SET pyforge:processed:{msg_id} 1 EX 86400` after successful processing. Before processing, it checks this key. The 24-hour TTL is fine — PEL messages rarely sit pending that long, and Redis Streams IDs are never reused.

Even better: make the *write itself* idempotent (writing the same entity_id + values at the same event_time is a no-op if the online store already reflects it). Then the side table is belt-and-suspenders.

## Design decisions

1. **Async consumer via `asyncio`.** Not threads. Redis client has native async support (`redis.asyncio`).
2. **Parse once; apply to both stores.** Decode the msgpack blob once, update shm via `layout.insert`, then append to the Parquet dataset, then ACK. Ordering matters: if the Parquet write fails after the shm update, the XACK is not issued, and on restart the message is reprocessed. `layout.insert` must be idempotent (writing the same row twice is safe — the slot is keyed by entity_id).
3. **Two independent batching knobs.** ACK cadence and Parquet flush cadence are decoupled because they optimize different things:
   - **ACK batch:** every 100 messages or every 500 ms, whichever comes first. Optimizes Redis round-trip throughput.
   - **Parquet flush:** every 10k–50k rows or every 60 s, whichever comes first. Optimizes file size (too-small Parquet files inflate read latency, metadata overhead, and inode count). Coupling these would mean either tiny Parquet files or slow ACKs — both bad.
4. **Startup drains PEL first.** On boot, read with ID `0` until the PEL is empty. Then switch to `>`.
5. **Emit `pyforge:processed:{msg_id}` after successful apply.** This key is what `WALProducer.write_sync` (Step 9) polls on. TTL 24 h — Redis Streams IDs are never reused.

## Build

```python
class WALConsumer:
    async def run(self):
        await self._drain_pel()
        while self._running:
            msgs = await self._read_new_batch()
            for msg_id, payload in msgs:
                if await self._already_processed(msg_id):
                    await self._ack(msg_id)
                    continue
                await self._apply(payload)
                await self._mark_processed(msg_id)
                await self._ack(msg_id)

    async def _drain_pel(self):
        while True:
            msgs = await self.redis.xreadgroup(
                groupname=GROUP, consumername=self.name,
                streams={STREAM: "0"},  # read PEL
                count=100,
            )
            if not msgs or not msgs[0][1]:
                break
            for msg_id, payload in msgs[0][1]:
                if not await self._already_processed(msg_id):
                    await self._apply(payload)
                    await self._mark_processed(msg_id)
                await self._ack(msg_id)
```

## Tests

**Unit:**

- `_apply` with the same message twice → shm state identical to one apply.

**Integration:**

- Produce 100 messages. Run consumer. All 100 end up in shm and Parquet. Zero duplicates.

**Chaos (the critical test):**

- Produce 100 messages. Start consumer. Kill it with SIGKILL at a random moment between message 30 and 70. Check: shm + Parquet contain some prefix (exact count varies). Restart consumer. After catch-up: shm + Parquet contain exactly 100 entries, no duplicates.
- Repeat 20 times with different kill timings. All 20 converge to the same final state.

## Benchmark

- Sustained throughput: target 10k messages/second.
- End-to-end lag (time between XADD and XACK): target p99 < 100 ms under steady load.
- `pyforge_wal_lag_seconds` gauge exposed.

## Acceptance criteria

- Chaos test passes 20/20 runs.
- Throughput meets target.
- Lag metric exposed and reasonable.

## Common failure modes

- **Consumer group not created before first read.** `XGROUP CREATE pyforge:wal pyforge_consumers $ MKSTREAM` has to happen once, idempotently, at startup. Use a distributed lock or rely on Redis's error-on-exists behavior.
- **PEL drain hanging forever.** If the consumer's claimed messages outnumber what XREADGROUP with ID=0 returns in one shot (COUNT limit), you need to loop. Cover this in a test with 1000 PEL messages.
- **Clock skew between client and Redis for message IDs.** Redis assigns the IDs, not the client — this is fine. But if you *ever* generate IDs client-side (explicit ID in XADD), clock skew will break monotonicity. Don't.

---

# Step 11 — Parquet Dataset Store

**Goal:** Persist feature writes to disk in a columnar, crash-safe, queryable format.

**Why now:** The WAL consumer needs a destination on disk. Without this step, Step 10's "write to Parquet" is a TODO.

## Background you need

### ⚠️ PLAN CORRECTION: `ParquetWriter` has no real "append mode"

The original plan says:

> Appends the record batch to the Parquet file using `pyarrow.parquet.ParquetWriter` in append mode.

This is not how PyArrow works. [PyArrow docs](https://arrow.apache.org/docs/python/parquet.html) and [JIRA ARROW-18171](https://issues.apache.org/jira/browse/ARROW-18171) are explicit:

> Once the writer is closed, it's not possible to append new row groups to a parquet file.

Options:

1. **Keep one `ParquetWriter` open for the lifetime of the process.** Works for a long-running consumer. Breaks on restart — the file being written when the process died has an invalid footer. You'd need to rewrite it from the beginning.
2. **Write a new Parquet file per batch** (or per minute, or per N messages) and treat the directory as a `pyarrow.dataset.Dataset`. This is what Delta Lake, Iceberg, and every serious columnar store do.

Pyforge uses **option 2**: the offline store is a directory of Parquet files, read via `pyarrow.dataset`. Each batch from the WAL consumer becomes one file with a UUID-based filename. No rewriting. Crash-safe by construction — a file is either fully written and moved into place, or never exists.

### Atomicity of a file write

The sequence:

1. Write to `tmp/{uuid}.parquet` in the same filesystem as the final directory.
2. `fsync` the file.
3. `rename` to `final/{schema}/{yyyy-mm-dd}/{uuid}.parquet`.

`rename` within the same filesystem is atomic on Linux. The reader never sees a partial file.

### Partitioning by date

`yyyy-mm-dd` subdirectories on `event_time`. This lets readers skip files outside their time window without opening them — PyArrow's dataset partition pruning.

## Design decisions

1. **File-per-Parquet-flush, not file-per-ACK-batch.** The Parquet flush cadence from Step 10 (10k–50k rows or 60 s) is what triggers file creation, **not** the ACK batch (100 messages). At 10k writes/sec with a 10k-row flush threshold, that's 1 file/second — manageable. With the ACK-batch cadence you'd get 100 files/second, which is small-file hell. Keep these knobs separate.
2. **Periodic compaction.** Once a day, a compaction job (not on the hot write path) merges small files in older partitions into larger ones. This is a future-work item if file count actually becomes a problem; initial version just writes.
3. **Schema stored in each Parquet file.** PyArrow embeds the schema in the file. Readers don't need external metadata.

## Build

`pyforge/offline.py`:

```python
class ParquetDatasetStore:
    def write_batch(self, schema, rows: list[dict]):
        table = pa.Table.from_pylist(rows, schema=self._arrow_schema(schema))
        date_str = datetime.fromtimestamp(rows[0]["event_time"]).strftime("%Y-%m-%d")
        final_dir = self.base / schema.__name__ / date_str
        final_dir.mkdir(parents=True, exist_ok=True)
        tmp_path = self.tmp_dir / f"{uuid4().hex}.parquet"
        pq.write_table(table, tmp_path, compression="zstd")
        with open(tmp_path, "rb") as f:
            os.fsync(f.fileno())
        final_path = final_dir / tmp_path.name
        os.rename(tmp_path, final_path)
        return final_path
```

## Tests

- Write batch → file exists in final dir, not tmp dir.
- Kill process mid-write (chaos) → tmp has a partial file; final doesn't. On restart, partial tmp is garbage-collected.
- Read back the file → rows match what was written.
- Partition structure: files land in the correct date directory.

## Benchmark

- Write throughput: target 50k rows/second with zstd compression.
- File size: 10k-row file is roughly 200 KB – 1 MB depending on embedding dimensions.

## Acceptance criteria

- All tests pass.
- Partial-file cleanup works in chaos test.

---

# Step 12 — Point-in-Time Reads

**Goal:** Given a list of (entity_id, as_of_time) pairs, return the most recent feature value per entity that was known by `as_of_time`.

**Why now:** Model training requires this. Without it, the offline store is write-only.

## Background you need

**Data leakage:** If a training example at `t=100` uses features updated at `t=110`, the model learns from information that wouldn't have been available at prediction time. Models trained with leaks look great in backtest and fail in production.

**`pyarrow.compute.asof_join`:** PyArrow has a primitive for exactly this. Takes a left table (the query) and a right table (the features), joins on entity_id with a "nearest value ≤ timestamp" semantic. Zero-copy, vectorized, implemented in C++.

**Partition pruning:** If `as_of_time` is June 15, you don't need to open July's Parquet files. PyArrow's dataset layer handles this automatically when partitions are date-based.

## Design decisions

1. **`asof_join` as the primitive.** Don't hand-roll a `merge_asof` unless it proves insufficient. PyArrow's version is the reference implementation.
2. **Input is itself a PyArrow Table.** The user supplies (entity_ids, as_of_times) as a PyArrow Table. Avoids the Pandas-in-the-hot-path trap.

## Build

```python
class ParquetDatasetStore:
    def read_point_in_time(self, schema, query_table):
        # query_table: pa.Table with columns (entity_id, as_of_time)
        min_t = pc.min(query_table["as_of_time"]).as_py()
        max_t = pc.max(query_table["as_of_time"]).as_py()
        dataset = pa.dataset.dataset(
            self.base / schema.__name__,
            format="parquet",
            partitioning="hive",
        )
        features = dataset.to_table(
            filter=(pc.field("event_time") <= max_t) & (pc.field("event_time") >= min_t - 30 * 86400),
        )
        joined = pa.compute.asof_join(
            query_table.sort_by("as_of_time"),
            features.sort_by("event_time"),
            left_on="as_of_time", right_on="event_time",
            left_by="entity_id", right_by="entity_id",
            tolerance=None,  # any lookback
        )
        return joined
```

The 30-day lookback in the filter is a heuristic — if a feature hasn't been updated in 30 days, it's probably not the relevant one. Configurable.

## Tests

- Write features at t=100, 110, 120 for entity A. Query with as_of_time=115 → returns the t=110 value.
- Query at as_of_time=90 → no row for A (entity didn't exist yet).
- Batch queries: 1000 entities, 1000 different as_of times. All correct.

**Leakage regression test:**

- Deliberately construct a case where a naive implementation (no time filter) would leak. Verify Pyforge's answer is leak-free.

## Benchmark

- 10k entity-timestamp pairs against a 10 M-row dataset: target < 5 seconds.

## Acceptance criteria

- Leakage test green.
- Benchmark recorded.

---

# Step 13 — Hydration

**Goal:** Rebuild the online store (shared memory) from the offline store (Parquet). Used after a restart or when the online store is wiped.

**Why now:** The system is now write-safe but not restart-safe — wipe Redis and all online data disappears. Hydration closes that gap.

## Background you need

**`asof_join` with as_of_time=now:** Gets the latest value for every entity. This is the hydration query.

**Bulk shm writes:** Hydration writes millions of entities into `layout.insert`. The insert function must be fast in bulk. A batched `insert_many` that does one pass over the hash table is the right shape.

## Design decisions

1. **Hydration is explicit, not automatic.** Called from an admin CLI. Automatic hydration on a cold cache is a foot-gun — startup time becomes unbounded.
2. **Hydration is idempotent.** Running it twice produces the same state.

## Build

```python
def hydrate(schema, parquet_store, segment_registry):
    now = time.time_ns()
    entity_ids = parquet_store.distinct_entity_ids(schema)
    query = pa.table({"entity_id": entity_ids, "as_of_time": [now] * len(entity_ids)})
    latest = parquet_store.read_point_in_time(schema, query)
    segment = segment_registry.create(schema, capacity=len(entity_ids) * 2)
    layout.insert_many(segment, latest)
```

## Tests

- Write 1000 entities, hydrate, verify online state matches.
- Wipe online, hydrate, verify recovery.

## Benchmark

- 1 M entities: target < 30 seconds end-to-end.

---

# Step 14 — Watchdog

**Goal:** A separate process that detects dead processes and cleans up the refcounts they held. Guarantees no shared-memory leaks under any crash scenario.

**Why now:** The WAL + offline store is a complete serving system by now. The watchdog closes the last integrity gap.

## Background you need

**Heartbeats:** Each process writes `HSET pyforge:heartbeats {pid} {timestamp_ns}` every 100 ms. The watchdog reads this hash in a 100 ms loop and compares to `time.time_ns()`.

**`psutil.pid_exists`:** Checks if a PID is alive. On Linux, this is a `/proc/{pid}` stat. Very fast. PID reuse is possible — a new process might have the same PID as a dead one. Mitigate by also checking process start time (`psutil.Process(pid).create_time()`). Stored in the heartbeat.

**Cleanup transaction:** When a process is declared dead, the watchdog must: (1) find all segments it held (`SMEMBERS pyforge:pid_segments:{pid}`), (2) for each, DECR refcount, (3) if refcount hit 0, unlink the segment, (4) remove the PID from heartbeats. All in a Redis Lua script for atomicity.

## Design decisions

1. **Watchdog is its own process.** Not a thread. A thread dies with the process; a separate process survives.
2. **Watchdog is itself monitored.** It writes its own heartbeat. A supervisor (systemd, k8s liveness) restarts it if missed.
3. **Staleness is measured in missed heartbeats, not wall-clock delta.** A process is declared dead after **5 consecutive missed heartbeats** (5 × 100 ms = 2.5 s of total silence). Wall-clock-delta logic sounds equivalent but fails under two common conditions: a debugger attaching via `SIGSTOP` (process is paused, not dead — you don't want to unlink its segments), and a briefly backlogged GC pause on the *watchdog* (it would read stale timestamps and false-positive). Consecutive-miss logic is robust to both: the watchdog only counts misses it observed itself, and a paused process resumes heartbeating before the count threshold if it wasn't actually dying.
4. **Heartbeat payload includes process start time.** From `psutil.Process(pid).create_time()`. Before declaring a PID dead, the watchdog cross-checks that the PID still exists *and* still has the same start time. Guards against PID reuse on fast-recycling systems.

## Build

Skeleton only — straightforward once the rules above are clear.

## Tests

**Chaos:**

- Start process A, register a schema, hold a segment, SIGKILL A. Within ~3 s (5 missed heartbeats): watchdog has decremented refcount, segment is unlinked (since refcount hit 0), `/dev/shm` is clean.
- Same but A's segment has another reader B. After A dies, segment stays. After B exits cleanly, segment unlinks.
- **Debugger attach resilience:** Start process A. Send SIGSTOP (simulating `gdb attach`). Wait 4 s — longer than the 2.5 s staleness window if it were wall-clock-based. Send SIGCONT. A resumes heartbeating. Watchdog did not declare A dead. Its segments are intact.
- **PID reuse:** kill A, spawn a new process that happens to get the same PID. Watchdog correctly identifies A as dead (different start time).

## Acceptance criteria

- 50 consecutive random-kill tests all converge to zero-leak state.
- Debugger-attach chaos test passes (no false positive on SIGSTOP/SIGCONT cycle).

---

# Step 15 — Schema Evolution

**Goal:** Upgrade a schema (add fields, widen dtypes) while active readers are mid-request. No reader crashes. Old segment is cleaned up after all readers release it.

**Why now:** The last piece. Everything else works for a static schema; this makes the system survive a deploy.

## Background you need

**Atomic pointer flip:** A Redis Lua script that updates `pyforge:schema:{name}:current` to the new segment, increments new's refcount, decrements old's refcount, all in one operation.

**Validating upgrade safety:** No field removals. Dtype widening only (float32 → float64 OK; reverse not). Shape unchanged for existing fields.

**Copying data:** The new segment is allocated; every existing entity's row is copied from old to new, with any new fields zero-filled. This is a batch operation, not per-request.

## Design decisions

1. **Upgrades are offline-admin actions, not automatic.** Run from a CLI with dry-run + confirm.
2. **Old segment stays until refcount == 0.** Processes holding the old segment continue reading it. They pick up the new segment on their next `open_current` (typically next request).

## Tests

- Live-reader test: a thread is continuously calling `assemble` in a loop. Run `upgrade_schema`. Reader doesn't crash. After reader exits, watchdog cleans up the old segment.

## Acceptance criteria

- Live-reader test passes.
- Upgrade of a 1 M-entity segment completes in < 10 seconds.

---

# Step 16 — Benchmarks and Flamegraphs

**Goal:** Everything from Steps 1–15 has a committed benchmark. This step is about producing the final set, with flamegraphs, on the machine whose specs are in the README.

## The headline benchmarks

1. **Hot-path latency — warm cache, small schema.** 4-field schema, entity in L2 after prior access. p50/p95/p99/p999 across three paths: shm, Redis baseline, naive Python dict baseline. This is the 5 µs p99 claim.
2. **Hot-path latency — warm cache, large schema.** 200-field schema with a 128-dim embedding. Same three paths. Expected shm p99: 10–20 µs. This is the realistic-production number.
3. **Hot-path latency — cold cache, large schema.** Same schema as (2), but flush CPU caches between calls (via `numactl` or explicit cache-clobbering array traversal). Expected p99: 20–50 µs. This is the honest worst-case number that goes in the README with a footnote.
4. **Batch throughput.** Entities/second at batch sizes 1, 10, 100, 1k, 10k. Compared against N × single-entity.
5. **GC pause impact.** p999 with and without `gc.freeze()` + background-thread strategy.
6. **Buffer pool impact.** Allocation rate and p99 with vs without pool.
7. **`write_sync` latency distribution.** p50/p95/p99 for the full producer → consumer → processed-flag round-trip under steady load.
8. **Hydration time.** 100k, 1M, 10M entities.

Every one of these has committed JSON (`pytest-benchmark` output) and, where applicable, an SVG flamegraph.

## Flamegraphs

```bash
py-spy record --native --rate 500 -o benchmarks/results/hot_path.svg -- python benchmarks/run_hot_path.py
py-spy record --native --rate 500 --subprocesses -o benchmarks/results/batch.svg -- python benchmarks/run_batch.py
```

`--native` is mandatory to see into Numba. `--subprocesses` is mandatory when the watchdog is running.

## Regression guard in CI

`benchmarks/regression/thresholds.yml` sets per-benchmark absolute thresholds. CI fails if the current run exceeds them.

## Acceptance criteria

- All five benchmarks' JSON outputs committed.
- All flamegraphs committed as SVG.
- CI regression gate active.

---

# Step 17 — Documentation and Release

**Goal:** Produce the artifacts that turn a working library into a credible project: README with real numbers, API reference, Architecture Decision Records, and a published package.

## README must contain

1. Problem statement (one paragraph).
2. Solution (one paragraph).
3. Architecture diagram — ASCII is fine.
4. Benchmark table with actual numbers — exact hardware specs listed below the table.
5. 10-line quickstart.
6. The "Common Questions" FAQ (see original spec).
7. Install and dev instructions.
8. CI + benchmark badges.

## ADRs (Architecture Decision Records)

Short markdown files in `docs/adr/`, **written alongside each step, not batched at the end**. An ADR written three months after the decision is a reconstruction, not a record — the alternatives you actually rejected fade, and what's left reads as post-hoc justification. Writing the ADR before you move on to the next step forces you to articulate the tradeoff while it's fresh, and catches the "wait, I don't actually remember why I chose this" cases early.

Format per ADR: 1–2 paragraphs. What decision. What alternatives were considered. Why this one. One ADR per file, numbered in decision order.

Mandatory set, mapped to the steps they're written during:

- **001** (Step 2): Why POSIX shared memory instead of `multiprocessing.shared_memory`.
- **002** (Step 2): Why per-open refcounting instead of per-read.
- **003** (Step 3): Why a 1 M-entity ceiling and horizontal sharding beyond it, rather than a single-segment design sized for 100 M+.
- **004** (Step 3): Why linear probing over Robin Hood / Cuckoo hashing.
- **005** (Step 5): Why Numba is gated on a per-schema benchmark threshold instead of used unconditionally.
- **006** (Step 7): Why `gc.freeze()` + background thread instead of `gc.disable()`.
- **007** (Step 9): Why `write_sync` is a separate method preserving single-writer-to-shm, instead of a "producer writes everything" fast path.
- **008** (Step 11): Why file-per-flush Parquet dataset instead of `ParquetWriter` append mode.
- **009** (Step 14): Why missed-heartbeat-count instead of wall-clock-delta for watchdog staleness.
- **010** (Step 17): Why no distributed features, no auth, Linux-only.

These are what an interviewer reads to see how you think about tradeoffs. They're also what you'll read in 6 months when you've forgotten why something looks the way it does.

## API reference

Auto-generate with `pdoc` or `mkdocs + mkdocstrings`. Docstrings on every public symbol.

## Publishing

- PyPI release: `uv build && uv publish`. Even if nobody installs it, it's the proof that the package is complete.
- GitHub release with tagged version and auto-generated notes.

## Acceptance criteria

- Someone clicking into the repo in GitHub can read the README and understand what Pyforge is in 2 minutes, see the performance claim substantiated, and find the install command in 30 seconds.

---

# Known Risks and What Will Trigger a Rethink

Every design call in this document is intentional, but some will only be validated by the benchmarks in Step 16. This section names the ones that could still force a revisit, and states the exact signal that would do it. If you hit one of these signals, stop and reconsider — don't patch around it.

- **Linear probing in Step 3.** Expected probe distance at 50% load factor is ~1.5. **Trigger to revisit:** Step 3 stress test (1 M entities) shows average probe distance > 3 or p99 probe distance > 20. Fallback: Robin Hood hashing. Not Cuckoo — the rehash cost during schema evolution would be prohibitive.
- **`gc.freeze()` over `gc.disable()` in Step 7.** Expected p999 improvement is meaningful but not dramatic. **Trigger to revisit:** Step 7 benchmark shows `freeze` + timer thread fails to materially tighten the p999 tail relative to the fully-enabled baseline. Fallback: selective `gc.disable()` on the serving thread only, accepting the cycle-leak risk and mitigating with periodic explicit collections.
- **Async-by-default writes in Step 10.** Expected 5–50 ms gap between `write` return and online-store visibility. **Trigger to revisit:** a real workload needs sub-10 ms read-your-own-writes at high rates — in which case `write_sync` with polling isn't fast enough, and the design needs a dedicated synchronous path (with cross-process locking in the slot table).
- **File-per-flush Parquet in Step 11.** Expected file count is manageable at 10k-row flush granularity. **Trigger to revisit:** write rate > 100k rows/sec sustained, producing > 10 files/sec even at 10k-row flushes. Fallback: compaction job becomes mandatory and runs every hour instead of daily.
- **Numba adoption gate in Step 5.** Expected to win at ≥ 20 fields. **Trigger to revisit:** Numba fails to meaningfully beat the NumPy-slice path at *any* schema size. Fallback: drop Numba; rely on NumPy's internal `memcpy` via slice assignment. This is the rare case where removing a dependency is the right call.

Any trigger firing is new information, not a failure. The plan is built to absorb a rethink at that point without cascading through every later step.

---

# Appendix A — Research Sources

Key external references consulted while writing this plan:

- CPython shared memory resource_tracker issues: GitHub #82300, #104291, #38119, #91577.
- PyArrow append-mode limitations: JIRA ARROW-18171, Apache Arrow Python docs.
- Redis Streams consumer-group patterns: Redis official docs, `redis.antirez.com/fundamental/streams-consumer-patterns.html`.
- PEP 669 (sys.monitoring) overhead profile: peps.python.org/pep-0669/.
- `gc.freeze()` production impact: Instagram engineering blog ("Dismissing Python Garbage Collection at Instagram"), Thumbtack engineering post.
- Numba caching and SIMD: numba.readthedocs.io/en/stable/user/faq.html, numba compilation developer docs.
- py-spy with `--native`: github.com/benfred/py-spy.
- Docker `/dev/shm` sizing: Docker official docs on `--shm-size`.
- pytest-benchmark pedantic mode: pytest-benchmark.readthedocs.io.

---

# Appendix B — Summary Table

| Step | What | Real risk / value |
|------|------|-------------------|
| 0 | CI, tooling, observability | Everything later assumes it exists |
| 1 | Schema + offset table | Foundation — wrong here, everything downstream is wrong |
| 2 | Shared memory lifecycle | The CPython resource_tracker bug is a real hazard |
| 3 | Slot table | The hardest layout decision in the project |
| 4 | Python serving skeleton | Correctness oracle for Numba |
| 5 | Numba assembly | Benchmark-gated; adopt only if it helps |
| 6 | Buffer pool | Removes the last hot-path allocation |
| 7 | GC management | Gates the p999 claim |
| 8 | Batch assembly | Unlocks real workloads |
| 9 | WAL producer | Durability boundary |
| 10 | WAL consumer + idempotency | The hardest crash-safety code |
| 11 | Parquet dataset store | File-per-batch, not append-mode |
| 12 | Point-in-time reads | Leakage regression test is critical |
| 13 | Hydration | Closes the restart gap |
| 14 | Watchdog | Zero-leak invariant |
| 15 | Schema evolution | Deploy-safety |
| 16 | Benchmarks | Turns claims into evidence |
| 17 | Docs + release | Turns code into a project |

Total: 8–12 weeks of focused work, up to ~16 weeks part-time with the production rigor this plan demands.
