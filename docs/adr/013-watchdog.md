# ADR-013: Watchdog (Step 14)

## Status

Accepted (Step 14 ships at this commit).

## Context

Step 13 (Hydration) closed the cold-restart loop: parquet → bulk-insert →
segment ready. The remaining integrity gap is the **hot-restart** loop. A
process that crashes mid-flight leaves three pieces of state nobody else
will clean up:

* `pyforge:refcount:{name}` — initial value of 1, never decremented.
* `pyforge:pid_segments:{pid}` — set of segment names held; the `close`
  Lua's SREM never runs for the dead PID.
* `/dev/shm/{name}` — the actual shared-memory inode; per invariant #6,
  only the creator may unlink, and the creator is dead.

Without an external janitor, every uncaught crash leaks one or more
`/dev/shm` segments forever. Step 14 adds that janitor.

## Decision

Three deliverables, all in one commit (`feat: step 14 — watchdog`).

### 1. Heartbeat producer (`pyforge._internal.heartbeat`)

Every library process that opens a Pyforge segment runs a single daemon
thread that periodically writes:

```
HSET pyforge:heartbeats {pid} "{create_time_ns}:{wall_time_ns}"
```

Mirrors `gc_manager`'s daemon-thread pattern (CLAUDE.md invariant #14):
`daemon=True` + `threading.Event` shutdown + `os.register_at_fork(after_in_child=...)`
state reset.

Public API: `ensure_started(redis_client, pid=None)` (idempotent) +
`stop()` + `is_running()`. Called from `SegmentRegistry.create` and
`SegmentRegistry.open_current` at method top (before any state mutation).

### 2. Watchdog process (`pyforge.watchdog`)

Separate process (`python -m pyforge.watchdog --redis ...`) that:

1. HGETALLs `pyforge:heartbeats` every `tick_interval_seconds` (default 30s).
2. Per pid: tracks `wall_time_ns` across ticks. Unchanged → `miss_count++`;
   changed (including backward jumps from NTP correction) → reset.
3. After `miss_threshold` consecutive misses (default 5), cross-checks
   via `psutil.Process(pid).create_time()` with EXACT equality against
   the stored `create_time_ns`.
4. Confirmed dead → atomic Lua transaction (DECRs refcounts, queues to
   `pyforge:cleanup_queue`, clears `schema:current` via the sidetable).
5. Drains `pyforge:cleanup_queue` — the canonical posix_shm.unlink call site.

Tests construct `WatchdogState` directly and call `run_one_tick()`
deterministically; chaos tests spawn the real subprocess.

### 3. Sidetable + back-touches (`pyforge:segment_to_schema`)

A hash mapping `segment_name → schema_name`. Written by
`SegmentRegistry.create` in the existing pipelined transaction. Read by
both the watchdog's dead-PID Lua AND the close-Lua extension at
refcount-0 — eliminates the O(N keyspace) `KEYS pyforge:schema:*:current`
scan that test helpers had been doing in lieu of production. Symmetric
across the two cleanup paths (watchdog dead-PID + live-process close).

## Detection cadence

| Knob | Default | Rationale |
|---|---|---|
| Heartbeat tick | 10 s | Matches WAL-consumer liveness refresh. |
| Watchdog tick | 30 s | ≥ 2× heartbeat (Nyquist). |
| Miss threshold | 5 ticks | 150 s of total silence before declaring dead. Tolerant of paused producers (debugger SIGSTOP) per spec § 1339-1340. |
| Detection ceiling | ~150-180 s | 5 ticks × 30 s + up-to-one-tick alignment slop. |

Spec § 1329's "100 ms tick / 5 missed = 2.5 s" is overridden because (a)
debugger-pause tolerance requires a longer threshold and (b) 30 s tick
matches Step 13's WAL-consumer liveness TTL — operators reason about
one number.

### Cadence asymmetry: WAL liveness 30 s vs watchdog 150 s

Hydrate has two preconditions cleared on different timescales after a
worker crash:

* **Precondition #2** (no WAL consumer liveness) clears via Redis TTL
  at 30 s.
* **Precondition #1** (no `schema:current`) clears via watchdog's
  dead-PID Lua at 150 s.

An operator running `hydrate()` at t+60 s sees "consumer not alive" but
"current segment exists" → confused. Documented in the runbook below
(see "Operator runbook"); not addressed by tuning the watchdog because
that would lose debugger-pause tolerance.

## Critical decisions

### #1. atexit handlers PID-gate to survive fork

Python's `atexit` registrations are inherited by forked children. The
standard production deployment shape (gunicorn / uvicorn fork workers)
without a guard:

```
parent (pid=100) calls ensure_started → atexit fires HDEL on parent's PID
parent forks → child (pid=101) inherits the atexit registration
child exits clean → atexit fires in child → HDELs heartbeats[100]
parent is now silent in the heartbeat hash but very much alive
150 s later → watchdog declares parent dead → unlinks parent's segments
```

That's data corruption on the canonical deployment shape.

Two-layer defense:

1. `_atexit_handler` checks `if os.getpid() != _state.pid: return`.
2. `_reset_after_fork` clears every `_state` field to None so the
   handler also early-returns via `if _state.pid is None: return`.

Belt-and-suspenders. Regression test:
[`test_atexit_does_not_hdel_after_fork_clean_exit`](../../tests/unit/test_heartbeat.py)
forks via `multiprocessing.Process`, has the child exit clean, asserts
parent's heartbeat field stays intact.

### #2. Cross-check uses EXACT equality, not tolerance

`psutil.Process(pid).create_time()` returns float seconds derived from
`/proc/{pid}/stat`'s `start_time` field (jiffies). Two reads of the
same PID on the same kernel return bit-identical floats; PID reuse
always differs by ≥1 jiffy. No tolerance needed; `==` is correct.

A "±1 ms tolerance" approach would silently misclassify on tickless or
1000 Hz kernels (jiffy = 1 ms, within tolerance).

### #3. AccessDenied + ZombieProcess are conservative (don't declare dead)

`psutil.Process(pid).create_time()` raises three relevant exceptions:

| Exception | Semantic | Action |
|---|---|---|
| `NoSuchProcess` | PID gone | Declare dead. |
| `AccessDenied` | Cross-UID without CAP_SYS_PTRACE; can't read `/proc/{pid}/stat` | Conservative — do NOT declare dead. |
| `ZombieProcess` | Un-reaped zombie | Conservative — do NOT declare dead (kernel still tracks PID; will become NoSuchProcess once parent reaps). |

`ZombieProcess` MUST be caught BEFORE `NoSuchProcess` in the except
chain because it inherits from `NoSuchProcess` in psutil's class
hierarchy.

Counter: `pyforge_watchdog_cross_check_unverifiable_total{reason}` with
reason ∈ {`access_denied`, `zombie`}. Operators alert on >0 — non-zero
means the watchdog is partially blind to some PIDs.

### #4. PID-reuse race window guarded by Lua

Between the watchdog's psutil cross-check (says "NoSuchProcess") and
the Lua execution (~1 ms later), the kernel can reuse the dead PID for
a new live process B. B's `SegmentRegistry.create` calls
`heartbeat.ensure_started`, which writes `heartbeats[pid]` with B's
create_time (force-first-refresh; synchronous before the pipeline).

Without a guard, the cleanup Lua reads B's `pid_segments`, DECRs B's
refcounts, queues B's segments for unlink → silent data corruption.

The cleanup Lua's first op is HGET on `heartbeats[pid]`; parses
`create_ns` from the stored value; compares to `expected_create_time_ns`
(passed by the watchdog as ARGV[2] from its cached `_PidEntry`).
Mismatch → return `-1` (sentinel) → watchdog increments
`pyforge_watchdog_pid_reuse_abort_total` and drops the `_tracked` entry
(next tick re-tracks under B's create_time).

**Residual risk**: if A's heartbeat was HDEL'd via atexit before the
reuse AND B's force-first-refresh failed (Redis blip swallowed),
`heartbeats[pid]` is nil → Lua falls through → cleanup proceeds against
whatever's in `pid_segments`. Combined probability ~3e-9 per dead PID
(~1 ms reuse window × ~1e-3 force-first-refresh failure rate / ~327 s
PID-reuse interval at default `pid_max=32768`). Operators alert on the
counter AND on `pyforge:cleanup_queue` size during incidents.

### #5. Single canonical posix_shm.unlink call site

The dead-PID Lua SADDs zero-refcount segments to `pyforge:cleanup_queue`
and returns the **count** (not the names). The watchdog's `run_one_tick`
step 4 drains `pyforge:cleanup_queue` and calls `posix_shm.unlink`.

The dead-PID Lua does NOT return names. An earlier draft did, leading
to step-3 unlinking the names AND step-4's drain SPOPing them and
trying again — 2x syscalls per segment per tick (FileNotFoundError on
the second, debug-logged but real overhead).

If the watchdog process crashes between the dead-PID Lua and the drain,
the names are still in `pyforge:cleanup_queue` — next tick's drain (or
next watchdog instance's first tick) picks them up.

### #6. Close-Lua extension symmetric with dead-PID Lua

Live-process closes that hit refcount-0 via `SegmentRegistry.close`
clear `schema:current` AND the sidetable atomically inside the close
Lua. Rotation-safe — the conditional `GET schema:current; if == name,
DEL` skips when a newer create has flipped the pointer.

This eliminates Step 13's `_drop_current` workaround in
`tests/integration/test_hydration_e2e.py`'s pipeline cleanup blocks.
The helper `_drop_current` retains an explicit DEL for the **hydrate
ghost-hold** case: hydrate's `registry.create` sets refcount=1 and
never closes; open-then-close cannot reach refcount-0; the operator
DEL is the realistic recovery path.

## Operator runbook

### Post-crash hydrate recovery

If a hydrate process crashed mid-run, the operator wants to re-hydrate.
Two paths:

**Unattended (automatic)**:
1. Wait for WAL-consumer liveness key to expire (≤ 30 s).
2. Wait for watchdog dead-PID detection (≤ 150 s).
3. Re-run `hydrate()` — both preconditions clear.

**Attended (instant recovery)** — use when no live serving consumers
hold the schema:
1. Confirm via `redis-cli HGETALL pyforge:heartbeats` that no live
   process is heartbeating against the dead PID.
2. `redis-cli DEL pyforge:schema:X:current`
3. Re-run `hydrate()`.

### Cross-UID watchdog deployments

If the watchdog runs as a different UID than its monitored producers
(e.g. dedicated `pyforge-watchdog` service user, root-owned producer
pods), `psutil.Process(pid).create_time()` raises `AccessDenied` for
cross-UID PIDs. The watchdog can't distinguish "stuck alive" from "PID
reuse" → conservative; does NOT clean up.

**Recommendation**: run the watchdog with `CAP_SYS_PTRACE` or as root
if it must clean up cross-UID producers. Alert on
`pyforge_watchdog_cross_check_unverifiable_total{reason="access_denied"}` > 0.

### Pre-Step-14 sidetable migration

Producers that were running before the Step 14 deploy did NOT write
`pyforge:segment_to_schema` entries. When those producers eventually
die post-upgrade, the watchdog's HGET returns nil → the
`schema:current` cleanup branch is skipped → operator falls back to
the manual `redis-cli DEL pyforge:schema:X:current`.

**Mitigation**: drain producers before upgrading. Strict-acceptance is
also fine — new segments created post-upgrade self-clean from day 1.

## Consequences

* Every uncaught crash now self-heals within 150 s without operator
  intervention.
* Every library process pays one Redis HSET / 10 s — negligible at any
  scale Pyforge targets.
* The watchdog's tick is observable end-to-end via
  `pyforge_watchdog_*` metrics; a `--metrics-port` CLI flag exposes
  `/metrics` via `prometheus_client.start_http_server` for operator
  scraping.
* Multiple watchdog instances against the same Redis is benign double
  work (Lua atomicity + DECR floor + SPOP semantics prevent
  double-unlink corruption). Operationally undesirable but not a
  correctness gap.

## Out of scope (deferred)

* **Watchdog HA / leader election** — Step 17 README documents
  "single watchdog per Redis cluster, restarted by supervisor."
* **Periodic /dev/shm orphan scan** — pre-existing failure mode where
  a process SIGKILL'd between `posix_shm.create` and the Redis pipeline
  EXEC leaves a `/dev/shm/pyforge_*` entry with no Redis state.
  Watchdog can't see it. Step 16 parking-lot.
* **Lua cleanup chunking** — pathological `pid_segments` with 10k+
  entries blocks Redis during the dead-PID Lua. Realistic <100;
  Step 16 if flamegraphs surface.
* **Real-psutil bench at scale** — current bench monkeypatches psutil
  to measure Redis-tick cost only. Step 16 if `psutil.Process`
  construction at 100+ candidate PIDs ever surfaces in flamegraphs.
* **Project-wide socket_timeout invariant** — heartbeat producer + WAL
  producer + WAL consumer all share the same shape (Redis client
  without `socket_timeout` blocks indefinitely on partition).
  `ensure_started`'s docstring documents the contract for heartbeat;
  Step 17's CLAUDE.md §5 invariant covers all three.
* **Real-PID-rollover chaos test** — flaky on Linux's typical
  `pid_max` settings. Mocked-psutil unit test
  ([`test_heartbeat_unchanged_psutil_alive_different_create_time_declares_dead`](../../tests/unit/test_watchdog.py))
  is the contract binding.

## Alternatives considered

* **Tick counter via HINCRBY instead of wall_time_ns**: cleaner
  conceptually (no clock-skew concerns) but requires HINCRBY +
  separate HSET (two ops per tick instead of one). Rejected;
  `!=`-based wall_time_ns comparison is already robust to NTP backward
  jumps (the `else` branch in step-2's miss-counting handles all
  inequalities, including backward jumps, by resetting miss_count).

* **Drop the SADD-to-cleanup-queue from dead-PID Lua and have the
  watchdog Python unlink the returned names directly**: simpler step-3
  flow but loses the "watchdog crashed mid-unlink" recovery path.
  Rejected; the SADD is cheap and the recovery property is operator
  insurance.

* **Tuning watchdog cadence to 30 s for symmetry with WAL liveness**:
  loses debugger-pause tolerance (4 s threshold under fast cadence
  would declare a SIGSTOP'd process dead before the operator could
  attach). Rejected; the asymmetry is documented in the runbook.

## References

* Spec: [`pyforge_build_steps.md` § "Step 14 — Watchdog"](../../pyforge_build_steps.md#L1321-L1359)
* Plan: [`progress/step14_plan.md`](../../progress/step14_plan.md)
* Heartbeat module: [`pyforge/_internal/heartbeat.py`](../../pyforge/_internal/heartbeat.py)
* Watchdog module: [`pyforge/watchdog.py`](../../pyforge/watchdog.py)
* Lua scripts: [`pyforge/_internal/watchdog_lua.py`](../../pyforge/_internal/watchdog_lua.py)
* CLAUDE.md invariant #14 (background threads): [`CLAUDE.md` §5](../../CLAUDE.md)
* CLAUDE.md invariant #6 (creator-only-unlinks): [`CLAUDE.md` §5](../../CLAUDE.md)
* Step 13's `_force_drop_orphan` justification: [ADR-012 §3](./012-hydration.md)
