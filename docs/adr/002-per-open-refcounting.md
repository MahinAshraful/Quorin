# ADR-002: Segment refcounting is per-open, not per-read

**Status:** Accepted
**Date:** 2026-04-24
**Step:** 2 (shared memory and lifecycle)

## Decision

Segment reference counts in Redis are incremented exactly once when a
process opens a segment (`SegmentRegistry.open_current`) and decremented
exactly once when it closes (`SegmentRegistry.close`). Individual
`read()` calls on the mmap'd region do **not** touch Redis.

## Context

The spec's headline performance claim is **5 µs p99 for a single-entity
read** on the warm-cache path. The read path must therefore not perform
any operation that is slower than the entire budget. Redis localhost
round-trip on a loopback interface measures **30–80 µs** — six to
sixteen times the whole budget for a single network call.

An early reading of the spec suggested that every `read()` should `INCR`
the segment's refcount at the start and `DECR` at the end, to prevent the
segment from being unlinked while the read is in flight. Doing so
inserts **two** Redis round trips into every read — destroying the latency
claim outright.

## Analysis

Reference counting exists to answer one question: "is it safe to unlink
this segment?" Safe means no process is currently mapping it. The
question needs to be answered at *administrative* boundaries (process
start, schema upgrade, watchdog run), not at each byte read.

The layout guarantee that makes the lock-free read path sound is:

> The segment's layout is immutable while any reader holds an open handle.

Writes happen through a single writer (the WAL consumer, Step 10) to
regions that readers are not concurrently reading. Schema evolution
(Step 15) allocates a **new** segment and atomically flips the
`pyforge:schema:{name}:current` pointer — the old segment continues to
live until its refcount drops to zero. The old segment's bytes never
change under a reader.

Given this, the relevant safety question is "has any process currently
opened this segment?" — not "is any process currently mid-read?". A
per-open refcount answers the relevant question correctly.

## The design

- **`open_current`** does one `INCR pyforge:refcount:{segment_name}` and
  one `SADD pyforge:pid_segments:{pid} {segment_name}`, in a pipelined
  transaction.
- **`close`** runs a single Lua script that does
  `DECR pyforge:refcount:{segment_name}`; if the result is zero, it adds
  the segment name to `pyforge:cleanup_queue`; and unconditionally
  `SREM`s the segment from this process's `pid_segments` set.
- **`read()`** — there is no read method on `Segment`. Callers access
  `segment.mmap_view` (a zero-copy memoryview) and slice it directly.
  No Redis call, no Python object allocation.

## Handling reader crashes

A reader that crashes with SIGKILL never calls `close`. Its refcount
stays inflated. Two mechanisms catch this:

1. **Watchdog (Step 14)** reads `pyforge:heartbeats` at 100 ms intervals,
   detects dead PIDs, and for each one iterates
   `SMEMBERS pyforge:pid_segments:{pid}`, `DECR`ing each segment's
   refcount and queuing for cleanup if the count hits zero.
2. **Segment survival under SIGKILL** is tested by
   `tests/integration/test_shm_lifecycle.py::test_reader_sigkill_does_not_destroy_segment`.
   The segment itself (which lives in `/dev/shm`) remains readable even
   when the refcount is temporarily stuck above zero.

The cost of this design is that cleanup is **deferred** rather than
immediate — a crashed reader's refcount hangs around for up to one
watchdog cycle before being reclaimed. For a single-node system with
~100 concurrent openers, this is acceptable.

## Consequences

- **Positive:** the hot read path is Redis-free. The 5 µs latency target
  is achievable.
- **Positive:** `open_current` pays two Redis round trips (INCR + SADD)
  once per process — negligible at process-start time.
- **Negative:** refcount accuracy relies on the watchdog for crash
  recovery. Delayed cleanup means `/dev/shm` can temporarily hold
  orphaned segments for up to one watchdog cycle (~2.5 s per the Step 14
  design).
- **Negative:** a process that calls `open_current` twice (for the same
  schema) adds two refs but only one `pid_segments` entry. On crash, the
  watchdog DECRs once, leaking one ref. The fix is a process-level
  discipline (open once per schema) rather than a registry-level
  invariant; this is documented in the tests.

## References

- Redis localhost latency characterization: numerous benchmarks show
  ~30-80 µs p99 for a single INCR over loopback on modern hardware.
- Pyforge spec, section "Latency targets are conditional, not a single
  number": 5 µs p99 warm-cache / small-schema.
- ADR-001 on why the segment lives in `/dev/shm` at all.

## Validation

`tests/integration/test_shm_redis.py::test_full_create_open_close_cycle_refcount_transitions`
verifies refcount transitions 1 → 2 → 1 → 0 across sequential operations.
`tests/integration/test_shm_redis.py::test_three_concurrent_opens_give_refcount_four`
verifies concurrent opens accumulate correctly.
