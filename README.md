# Quorin

**Low-latency ML feature serving for one machine. ~5 µs p99 reads from shared memory.**

[![CI](https://github.com/MahinAshraful/Quorin/actions/workflows/ci.yml/badge.svg)](https://github.com/MahinAshraful/Quorin/actions/workflows/ci.yml)
[![Benchmark](https://github.com/MahinAshraful/Quorin/actions/workflows/benchmark.yml/badge.svg)](https://github.com/MahinAshraful/Quorin/actions/workflows/benchmark.yml)
[![PyPI](https://img.shields.io/pypi/v/quorin.svg)](https://pypi.org/project/quorin/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> v0.1.0 — feature-complete; 758 tests passing; 5 µs p99 substantiated on
> GitHub Actions ubuntu-latest at N=20 fresh subprocesses (median_p99 =
> **4.48 µs** for the 4-field warm assemble path; see [Benchmarks](#benchmarks)).

---

## What is this

Machine-learning serving has a structural latency floor that the model itself
doesn't cause. A typical online-prediction request looks like:

```
fetch features from Redis  →  decode bytes  →  build Python dict  →  call model  →  return
```

Steps 1–3 cost **5–50 ms** on a healthy box. The model's `predict()` call is
often **~200 µs**. The infrastructure around the model is the bottleneck —
not the math. At 50,000 RPS that's a 250-core overhead just to shuttle bytes.

Quorin replaces the slow path with a **shared-memory + precomputed-offset-table**
read: features live as typed bytes in a POSIX shm segment that every worker
process already has mapped. A read becomes "compute the offset, copy the bytes,
return a `numpy.float32` array" — ~4 µs p99 on commodity hardware, zero Python
object allocations, zero Redis calls on the hot path.

It is **deliberately single-node**. No distribution, no replication, no
cross-node coordination. Beyond ~1M entities the answer is horizontal sharding
by `hash(entity_id) mod N` across multiple Quorin instances. See
[FAQ](#faq) for the explicit scope discipline.

---

## Schema preview

This is what defining a feature schema looks like — pure Python, no
infrastructure:

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

A `FeatureSchema` subclass compiles, once at process start, into a
NumPy-backed offset table. Lookups are `searchsorted` on a sorted hash array;
reads are a Numba-compiled memory copy.

---

## Benchmarks

Numbers measured on **GitHub Actions `ubuntu-latest`** (`ubuntu-24.04`)
via the N=20 fresh-subprocess orchestrator
([`benchmarks/runs/repeat.py`](benchmarks/runs/repeat.py)),
workflow run 25394553451, commit `4818ea4`.

| Scenario                                | median p50  | median p99    | Spec band   | Source JSON |
|---|---|---|---|---|
| 4-field warm assemble (the headline)    | 4.14 µs     | **4.48 µs ✅** | ≤ 5 µs       | [`headline_4_field_warm_n20.json`](benchmarks/results/n20/headline_4_field_warm_n20.json) |
| 200-field warm assemble                 | 7.59 µs     | 11.66 µs ✅    | 10–20 µs     | [`headline_200_field_warm_n20.json`](benchmarks/results/n20/headline_200_field_warm_n20.json) |
| 200-field cold assemble                 | 31.28 µs    | 66.14 µs †    | 20–50 µs     | [`headline_200_field_cold_n20.json`](benchmarks/results/n20/headline_200_field_cold_n20.json) |
| 4-field assemble under GC pressure p999 | —           | 22.44 µs (p999) | (informational) | [`gc_p999_pressure_n20.json`](benchmarks/results/n20/gc_p999_pressure_n20.json) |
| `write_sync` end-to-end RTT             | 1.93 ms     | 2.18 ms        | ≤ 75 ms gate | [`write_sync_rtt_n20.json`](benchmarks/results/n20/write_sync_rtt_n20.json) |

> **† Cold-cache 66 µs p99 is over the 20–50 µs spec band by ~30%** on
> ubuntu-latest's older Xeon CPUs (~30 MB L3 per socket). Per
> [ADR-015 §11](docs/adr/015-benchmark-methodology.md) bare-metal extrapolation
> (modern desktop CPUs are 1.5–3× faster than ubuntu-latest on this
> bandwidth-bound bench), the projected bare-metal range is **~22–44 µs**, inside
> the spec band. Re-measure on your own hardware if the cold-cache number
> matters — single-process cold-cache p99 is heavy-tailed (3-4× run-to-run
> variance per ADR-015 §4); N=20 fresh-subprocess aggregation is the meaningful
> measurement.

**Methodology (full detail in [ADR-015](docs/adr/015-benchmark-methodology.md)):**
each scenario runs N=20 fresh Python subprocesses; per-process pytest-benchmark
captures raw round timings; the orchestrator aggregates `median(p99)` across
runs (not max-of-max). 30+ regression gates are enforced in CI on every PR via
[`tier1.yml`](benchmarks/regression/tier1.yml).

---

## Architecture

```
                   ┌─────────────────────────────────────────┐
   client          │  multiple worker processes              │
   request   ─────►│  (web server / batch job / notebook)    │
                   └──────────────────┬──────────────────────┘
                                      │ assemble(seg, entity_id)  ~4 µs p99
                                      ▼
   ┌──────────────────────────────────────────────────────────────┐
   │  POSIX shared memory segment in /dev/shm                     │
   │                                                              │
   │  [16 B header][48 B metadata][slot table][string pool][rows] │
   │                                                              │
   │  Read path is allocation-free Numba-compiled copy.           │
   │  Read path NEVER touches Redis (per ADR-002).                │
   └──────────────────────────────────────────────────────────────┘
                                      ▲
                                      │ insert(seg, entity_id, row_bytes)
                                      │  (single writer; WAL consumer)
                                      │
                   ┌──────────────────┴──────────────────────┐
                   │  WAL consumer (single writer per seg)   │
                   │  reads from Redis Stream "quorin:wal"   │
                   └─────────────────────────────────────────┘
                                      ▲
                                      │ XADD (async by default;
                                      │  write_sync available)
                                      │
                   ┌──────────────────┴──────────────────────┐
                   │  WALProducer  (in user processes)       │
                   └─────────────────────────────────────────┘

   Cross-cutting:
   - Redis (control plane only): segment names, refcounts, WAL stream.
     Reads do NOT touch Redis.
   - quorin.watchdog: detects dead PIDs via heartbeat, drains cleanup queue.
   - quorin.evolution: atomic pointer flip on schema upgrade.
   - quorin.offline (Parquet): training-data store + point-in-time reads
     + Redis hydration on cold start.
```

---

## Quickstart

**Prereq:** Redis 7.2+ on `127.0.0.1:6379`. Quorin ships a docker-compose for
local dev:

```bash
docker compose -f docker/docker-compose.dev.yml up -d
```

Then:

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
print(features)            # [ 0.5 42.  12.3]
print(features.dtype)      # float32
```

**What just happened:**

- Defined a schema; allocated a shared-memory segment named `quorin_UserFeatures_<uuid>`.
- Packed one row's bytes via `pack_row` (kwargs API; coerces to declared dtypes).
- Wrote the row via the synchronous `insert` path; read it back as a
  `numpy.float32` array via `assemble`.
- The `assemble` call is the headline ~4 µs p99 path on warm cache.

**Production writes** go through [`quorin.wal.WALProducer`](quorin/wal.py) —
async write to a Redis Stream + a separate WAL consumer applies it to the
segment with crash-safety semantics. The synchronous `insert` shown here is
the testing / hydration / demo path. See [docs/API.md](docs/API.md) for the
full surface.

---

## Install

```bash
pip install quorin
```

Requires Python 3.12+, Linux or WSL2 (POSIX shared memory). Redis 7.2+ for the
control plane.

### Dev setup

```bash
git clone https://github.com/MahinAshraful/Quorin.git
cd Quorin
uv sync --all-extras
docker compose -f docker/docker-compose.dev.yml up -d
uv run pytest          # 758 tests, ~4 min on WSL2
```

---

## FAQ

**Why single-node?**
Single-node is the design thesis, not a limitation. The 5 µs p99 claim depends
on every reader having the segment mmapped in their own address space; that
breaks the moment you cross a machine boundary. Beyond ~1M entities, shard
horizontally by `hash(entity_id) mod N` across multiple Quorin instances.

**Why Linux-only?**
POSIX `shm_open`. macOS has [`posix_ipc`](https://pypi.org/project/posix-ipc/)
support but Quorin's CI doesn't test it; native Windows is out of scope
(different syscall surface — `CreateFileMapping` would be a separate project).

**Why Redis on the control plane?**
Per-process refcounts, segment-name resolution, the WAL stream, watchdog
heartbeats. Redis is on the *control* path; the *read* path **never** touches
it (per [ADR-002](docs/adr/002-per-open-refcounting.md)). Hot-path RPCs to
Redis would blow the latency budget in a single round trip (~30-80 µs over
loopback).

**How does this compare to Feast?**
Different scope. [Feast](https://github.com/feast-dev/feast) is a feature
*store* (training + serving + lineage); Quorin is a feature *server* (read
path only) optimized for one machine. Quorin could plug into a Feast deployment
as the online-serving layer; the comparison is "Feast's online layer vs Quorin,"
not "Feast vs Quorin."

**Does the buffer pool always help?**
**No.** Per the [ADR-005](docs/adr/005-buffer-pool-lock-free-prealloc-capped.md)
Step 16c amendment: on native CI, the pool adds **+2-4 µs** of latency to the
single-entity assemble path. Pool wins are real but indirect (eliminates one
ndarray allocation per call → less GC pressure; bounds memory ceiling) — the
direct latency cost is honest and disclosed. Pool is default for the **batch**
path (where amortization wins) and opt-in for single-entity calls.

**How much faster is batch?**
**1.5-1.7× at N=1000 on ubuntu-latest** (per [ADR-007](docs/adr/007-batch-assembly.md)
Step 16c amendment). The original spec target was 5×; the older Xeons in
GitHub Actions are cache-bound on this workload (~30 MB L3 spills to DRAM).
Bare-metal modern CPUs (more L3, higher clocks) should lift the ratio
meaningfully — re-measure on your own hardware.

**What about late data / out-of-order writes?**
Append-only Parquet with `event_time` and `processing_time` columns; query
by `event_time` for point-in-time-correct training reads. Stream-system
concerns (watermarks, exactly-once across nodes) are out of scope — those
belong in Kafka / Flink upstream.

**Why no auth?**
Single-process trust model. Quorin is imported by a trusted process; if exposed
over a network, that's a different project with a different security design.

**Is this production-ready?**
v0.1.0 means "feature-complete library; 758 tests pass; 5 µs p99 substantiated
on native CI; no real-world deployments yet." API may evolve based on user
feedback before v1.0.0. Performance regression gates run on every PR.

**Why is the codebase named `quorin` but the docs say `Pyforge` in places?**
`Pyforge` was the internal-development codename. The published package is
`quorin`. The codename survives in the [ADR archive](docs/adr/) (timestamped
historical decision records — they reference the codename current at decision
time, same shape as a git commit message), in [`CLAUDE.md`](CLAUDE.md) (the
internal Claude Code tooling document), and in git history. Functionally
identical.

---

## What's in the box

Public modules — full API surface in [`docs/API.md`](docs/API.md).

| Module | Purpose |
|---|---|
| [`quorin.schema`](quorin/schema.py) | `FeatureSchema`, `FeatureField`, `dtype`, `compile_schema` |
| [`quorin.shm`](quorin/shm.py) | `SegmentRegistry` — POSIX shm lifecycle + Redis bookkeeping |
| [`quorin.layout`](quorin/layout.py) | `insert`, `lookup`, `pack_row`, slot-table + string-pool primitives |
| [`quorin.serving`](quorin/serving.py) | `assemble` — pure-Python read oracle (parity reference) |
| [`quorin.assembly`](quorin/assembly.py) | `assemble`, `assemble_batch` — Numba JIT read path |
| [`quorin.pool`](quorin/pool.py) | `BufferPool`, `BatchBufferPool` — pre-allocated output buffers |
| [`quorin.wal`](quorin/wal.py) | `WALProducer` — async writes to Redis Stream |
| [`quorin.wal_consumer`](quorin/wal_consumer.py) | `WALConsumer` — applies WAL messages to the segment |
| [`quorin.offline`](quorin/offline.py) | `ParquetDatasetStore` — training-data writes + point-in-time reads |
| [`quorin.hydration`](quorin/hydration.py) | `hydrate` — rebuild segment from Parquet on cold start |
| [`quorin.evolution`](quorin/evolution.py) | `upgrade_schema` — atomic schema-version flip |
| [`quorin.watchdog`](quorin/watchdog.py) | Background process: detects dead PIDs, cleans up segments |
| [`quorin.metrics`](quorin/metrics.py) | Prometheus histograms + `start_metrics_server` |
| [`quorin.logging`](quorin/logging.py) | structlog JSON config |

---

## License

MIT — see [LICENSE](LICENSE).

## Acknowledgments

Built on numpy, numba, pyarrow, redis-py, pydantic, posix-ipc, structlog,
prometheus-client. Thanks to all upstream maintainers.
