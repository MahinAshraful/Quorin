# Quorin

**Low-latency ML feature serving for one machine. ~5 µs p99 reads from shared memory.**

[![CI](https://github.com/MahinAshraful/Quorin/actions/workflows/ci.yml/badge.svg)](https://github.com/MahinAshraful/Quorin/actions/workflows/ci.yml)
[![Benchmark](https://github.com/MahinAshraful/Quorin/actions/workflows/benchmark.yml/badge.svg)](https://github.com/MahinAshraful/Quorin/actions/workflows/benchmark.yml)
[![PyPI](https://img.shields.io/pypi/v/quorin.svg)](https://pypi.org/project/quorin/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> v0.1.0 — feature-complete; 767 tests passing; 5 µs p99 substantiated on
> GitHub Actions ubuntu-latest at N=20 fresh subprocesses (median p99 =
> **4.48 µs** for the warm-cache 4-field assemble path).

---

## The problem

When a machine-learning model serves a prediction online, the model itself is
usually fast. A gradient-boosted tree across 200 features takes around 200
microseconds. The model is not the bottleneck.

The bottleneck is everything around the model. A typical request flow goes
out to a database (usually Redis), pulls back raw bytes, decodes them into
Python objects, builds a dictionary, and finally hands that to the model.
Each step allocates short-lived Python objects. At 50,000 requests per
second a serving process allocates millions of those per second, which
keeps the garbage collector busy. Periodically it pauses every thread to
clean up, and that shows up in the latency tail as 5–50 millisecond spikes
hitting roughly one request in a thousand. Steady-state CPU is also high
because every Redis call is a network round trip and every byte-decode
walks pure Python.

The combined infrastructure cost lands around 5–50 milliseconds per request
— which is 25 to 250 times slower than the model itself. At scale that's a
direct multiplier on how many machines you need just to shuttle bytes
around the model. Quorin's design goal is to make the infrastructure stop
being the bottleneck: the headline target is 5 microseconds at the 99th
percentile for a small-schema warm read.

## What Quorin does

Quorin replaces the slow path with a path that reads features directly from
a shared-memory region that every worker process has already mapped.
"Shared memory" here means a region of RAM the operating system lets
multiple processes see at the same time — like a whiteboard everyone in a
room can read from without copying it. Once a worker has the segment
mapped, reading a feature vector becomes a few microseconds of pointer math
and memory copies. There is no network call, no serialization, no Python
dictionary, and no garbage-collected object created on the hot path. The
output is a contiguous `numpy.float32` array — the exact shape every ML
library wants as input — ready to hand straight to `model.predict(...)`.

The library is **deliberately single-node**. It does not distribute across
machines, does not replicate Redis, and does not coordinate across nodes.
Beyond roughly one million entities, the answer is to shard horizontally
by hashing the entity ID across multiple Quorin instances. This is a scope
discipline, not a limitation — single-node is what makes the 5-microsecond
target reachable, and a distributed system would dilute the entire
project's identity.

## Architecture

Quorin organizes work into a few cooperating components, each with a
distinct responsibility.

A **schema compiler** turns your typed feature definitions into a small
sorted lookup table of byte offsets, computed once at process start.
Looking up "where does field X live in memory?" becomes a binary search
over an integer array — no string hashing, no dictionary, no allocation.

A **shared-memory segment** holds the actual feature bytes plus a slot
table mapping entity IDs to row positions. The whole layout is one
contiguous region, sized once when the schema is registered. Every worker
process maps the same physical RAM at startup; from then on, reads are
free.

A **read path** does the lookup and the byte-copy in a small function
compiled to native machine code by Numba (a Python library that JIT-
compiles numeric Python via LLVM). The read kernel never allocates Python
objects, never touches Redis, and never copies more than one feature
vector's worth of bytes.

A **write path** is intentionally separate from the read path. Producer
processes don't write to shared memory directly — they validate the row,
encode it, and append a message to a Redis Stream. A dedicated **WAL
consumer** process is the segment's only writer; it reads from the stream
and applies each message. This separation gives crash safety (writes are
durable in Redis before they're applied) and a single-writer invariant,
which is what makes the lock-free read path correct.

An **offline store** captures every applied row to append-only Parquet
files for training data. It also serves point-in-time-correct reads, which
matter when building training datasets — you need each entity's feature
values "as they were at a specific moment in the past," not their current
values, otherwise your model trains on data it wouldn't have had at
prediction time.

A **watchdog** background process monitors worker liveness via Redis
heartbeats and cleans up shared-memory references when a worker crashes
without releasing them. Without it, segments would leak whenever a process
died ungracefully.

A **schema-evolution coordinator** handles the rare case of upgrading a
schema. It allocates a new segment with the new layout, copies and
translates every row from the old segment, and atomically flips a Redis
pointer so new readers see the new segment while in-flight readers
continue against the old one until they close.

For the deep technical explanation of how these components are
implemented and why, see [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Benchmarks

Numbers measured on **GitHub Actions `ubuntu-latest`** (`ubuntu-24.04`)
via the N=20 fresh-subprocess orchestrator, workflow run 25394553451,
commit `4818ea4`. Source JSONs are committed under
[`benchmarks/results/n20/`](benchmarks/results/n20/).

| Scenario                                | median p50  | median p99    | Spec band   |
|---|---|---|---|
| 4-field warm assemble (the headline)    | 4.14 µs     | **4.48 µs**   | ≤ 5 µs       |
| 200-field warm assemble                 | 7.59 µs     | 11.66 µs      | 10–20 µs     |
| 200-field cold assemble                 | 31.28 µs    | 66.14 µs †    | 20–50 µs     |
| GC pressure (4-field) p999              | —           | 22.44 µs      | informational |
| `write_sync` end-to-end RTT             | 1.93 ms     | 2.18 ms       | ≤ 75 ms gate |

> **† Cold-cache 66 µs p99 is over the 20–50 µs spec band by ~30%** on
> ubuntu-latest's older Xeon CPUs (~30 MB L3 per socket). Modern desktop
> CPUs are 1.5–3× faster than ubuntu-latest on this bandwidth-bound
> bench, projecting bare-metal cold-cache p99 to **~22–44 µs** — back
> inside the spec band. Re-measure on your own hardware if cold-cache
> latency matters; the methodology is documented in
> [ADR-015 §11](docs/adr/015-benchmark-methodology.md).

Every benchmark is reproducible (`gh workflow run benchmark.yml`). 30+
regression gates are enforced in CI on every pull request. The full
methodology — venue disclosure, fresh-subprocess discipline, calibration
rules — lives in [ADR-015](docs/adr/015-benchmark-methodology.md).

A few measured results we publish honestly even though they're worse than
originally claimed: the buffer pool adds 2–4 µs of overhead on the
single-entity path on commodity hardware (the win is reduced GC pressure,
not direct latency); batch reads at N=1000 are 1.5–1.7× faster than N
single calls on the same hardware (the original 5× target is realistic
only on bare-metal CPUs with substantially more L3 cache). Both numbers
are documented in their respective ADR amendments —
[ADR-005](docs/adr/005-buffer-pool-lock-free-prealloc-capped.md) and
[ADR-007](docs/adr/007-batch-assembly.md).

## Install

```
pip install quorin
```

Requires Python 3.12 or newer, Linux or WSL2 (POSIX shared memory), and
Redis 7.2+ for the control plane. The hot read path itself never touches
Redis; Redis is used at process startup for segment-name resolution and
for the write-side WAL stream.

For local development, the repo includes a `docker-compose.dev.yml` that
brings up a Redis container on `127.0.0.1:6379` — enough to run the full
demo end-to-end.

## Documentation

| Where to look | What's there |
|---|---|
| [docs/USAGE.md](docs/USAGE.md) | Runnable code examples for every public path: defining schemas, the synchronous demo, the production WAL flow, batch reads, point-in-time reads for training data, hydration on cold start, schema upgrades, operations (watchdog, metrics). |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | Deep technical walk-through of every component — what it does, why it exists, how it fits together. Written for engineers extending the library. |
| [docs/API.md](docs/API.md) | Hand-curated reference for the public API surface, organized by module. Each section names the symbols, links to source, and links to the relevant ADR(s). |
| [docs/adr/](docs/adr/) | Architecture Decision Records — 17 numbered records, one per load-bearing decision, written alongside the step that introduced them. The canonical record of why anything in the codebase looks the way it does. |
| [CHANGELOG.md](CHANGELOG.md) | Per-release summary of what shipped. |

## FAQ

**Why single-node?**
Single-node is the design thesis, not a limitation. The 5 µs p99 target
depends on every reader having the segment mapped in their own address
space; that breaks the moment you cross a machine boundary. Beyond roughly
one million entities, the answer is to shard horizontally — run multiple
Quorin instances and route each entity ID to one of them by hash.

**Why Linux-only?**
POSIX shared memory (`shm_open`). macOS has the underlying `posix_ipc`
support but Quorin doesn't test on it in CI; native Windows is out of
scope (it would need a different shared-memory layer entirely). WSL2 on
Windows works as a Linux box.

**Why Redis on the control plane?**
Per-process refcounts, segment-name resolution, the WAL stream, and
watchdog heartbeats. Redis is on the *control* path; the *read* path
**never** touches it. Per-read Redis calls would blow the latency budget
in a single round trip — Redis loopback measures 30–80 microseconds, more
than the entire 5-microsecond budget.

**How does this compare to Feast?**
Different scope. Feast is a feature *store* — it manages training data,
serving, lineage, integrations with cloud providers. Quorin is a feature
*server* — the read path only, optimized for one machine. Quorin could
plug into a Feast deployment as the online-serving layer; the comparison
is "Feast's online layer vs Quorin," not "Feast vs Quorin."

**What about late data and out-of-order writes?**
The offline Parquet store records both `event_time` and `processing_time`
on every row. Training reads query by `event_time` for point-in-time
correctness. Stream-system concerns like watermarks and exactly-once
semantics across nodes are out of scope — those belong upstream in Kafka
or Flink.

**Why no auth?**
Single-process trust model. Quorin is imported by a trusted process; if
exposed over a network, that's a different project with a different
security design.

**Is this production-ready?**
v0.1.0 means "feature-complete library, 767 tests passing, 5 µs p99
substantiated on native CI, no real-world deployments yet." The API may
evolve based on user feedback before v1.0.0. Performance regression gates
run on every pull request, so the perf claim is defended automatically.

**What's the catch on the cold-cache number?**
Disclosed openly in the benchmark table footnote: 66 µs p99 on ubuntu-
latest is over the 20–50 µs spec band by about 30%, attributable to the
older Xeon CPUs in GitHub Actions runners having less shared L3 cache
than modern desktops. Bare-metal extrapolation puts the number at 22–44
µs, back inside the spec band. The ADR archive
([ADR-015 §11](docs/adr/015-benchmark-methodology.md)) explains the
methodology that supports the extrapolation. Honest beats heroic — we
don't quote the number we wish we measured.

## License

MIT — see [LICENSE](LICENSE).

## Acknowledgments

Built on numpy, numba, pyarrow, redis-py, pydantic, posix-ipc, structlog,
prometheus-client. Thanks to all upstream maintainers — Quorin would not
exist without these libraries doing their jobs well.
