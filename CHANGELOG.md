# Changelog

All notable changes to Quorin will be documented in this file.

## v0.1.2 — 2026-05-21

Performance + final audit cleanup patch. Closes the remaining
producer/consumer hardening items deferred from v0.1.1's audit, lands
two batch-path performance wins (~1.3-1.4x batch ratio recovery
toward the original 5x claim), and adds defense-in-depth bounds in
the Numba lookup kernel. **No public-API additions; no breaking
behavior changes for valid input.** Three new `ValueError`s fire at
producer/schema boundaries for inputs that previously DoS'd silently.

### Performance

- **Batch hash migration** — `assemble_batch` now hashes each row's
  entity_id via the Numba `blake2b_8` kernel instead of Python
  `hashlib.blake2b`. At N=1000 batch, the Python prep floor drops
  from ~1.5 ms to ~150-300 us. Byte-identical output per pinned-hash
  invariant #5; verified by `test_batch_parity.py`. Native-CI batch
  ratio expected to recover from 1.46x/1.73x (4-field/200-field) toward
  ~2.0-2.4x. See [`quorin/assembly.py`](quorin/assembly.py).
- **`_F` compression function inlined** — saves a function-call
  boundary per `blake2b_8` invocation (~10-20 ns). Compounds across
  batch hashing. Pinned-hash tests verify byte-identity.

### Hardening (DoS class closure)

- **CR.C.2** — `FeatureField` shape caps: `element_count <= 2^24`
  (16M), per-field bytes `<= 2^28` (256 MiB), whole-segment bytes
  `<= 2^36` (64 GiB). Defends against `shape=(2**30,)`-class inputs
  that would previously OOM or SIGBUS at first write. New
  `quorin.schema.MAX_ELEMENT_COUNT`, `MAX_FIELD_BYTES`,
  `quorin.layout.MAX_TOTAL_SEGMENT_BYTES`.
- **CR.C.3** — Producer-side `entity_id` length cap (256 bytes,
  `MAX_ENTITY_ID_BYTES`). Without it, a single oversize ID landed in
  the Redis Stream + msgpack + consumer memory before the consumer
  eventually rejected it, leaving the message PEL-stuck. Now rejected
  before XADD.
- **CR.C.4** — Consumer unknown-schema label cardinality cap
  (1024 distinct names; further unknowns increment the `<other>`
  bucket). Closes the Prometheus DoS + `_seen_unknown` memory-leak
  shape.
- **CR.B.2** — Numba lookup kernel validates `pool_len <= max_id_bytes`
  before the byte-compare. With `boundscheck=False`, a corrupt pool
  entry could previously read past the string-pool region; now the
  kernel returns -1 (miss) for corrupt entries. ~1-2 ns added per
  probe; locked by `test_lookup_kernel.py`.

### Operability

- **CR.B.14** — `WALConsumer` NOGROUP recovery. Operator actions
  like `XGROUP DESTROY` or `DEL quorin:wal` previously crashed the
  consumer; now the consumer logs `wal_consumer.nogroup_recreating`,
  recreates the group via `_ensure_group()` (`id="0"` semantics), and
  resumes.
- **CR.B.13** — Wedged-flush back-pressure escalation. When
  `pending_ack` sits at `2x max_pending_ack` for >= 60 s (full disk
  / NFS stuck / deterministic offline failure), the consumer logs
  ERROR `wal_consumer.backpressure_exceeded` and increments a new
  `quorin_wal_consumer_backpressure_alerted_total` counter. Operators
  alert on rate > 0.
- **CR.E.5** — `WALProducer.write_sync` docstring documents the
  `timeout_ms x flush_interval_seconds x max_pending_ack` interaction
  explicitly. The default ~1.6% timeout rate at `timeout_ms=100` was
  previously undocumented.

### Closed without implementation (WON'T-DO / verified-already-closed)

- **QW-1 (lookup-jit `.copy()` drop)** — Attempted, reverted. Numba
  0.60 dispatch distinguishes writable from read-only `uint8[::1]`
  and rejects the read-only variant for `_lookup_core`'s registered
  signature. The `.copy()` is load-bearing; comment in
  [`quorin/_internal/lookup_kernel.py`](quorin/_internal/lookup_kernel.py)
  records the empirical finding to prevent re-litigation.
- **CR.B.8 (Redis-blip post-flip outage)** — Verified already closed
  by v0.1.1's CR.A.7 `flip_completed` flag at
  [`quorin/evolution.py:531`](quorin/evolution.py#L531). Post-flip
  exception branch preserves `new_seg` and logs
  `evolution.post_flip_failure_new_segment_preserved` instead of
  orphan-cleaning. Moved from "deferred" to DONE.
- **CR.B.1 (Numba UINT8 `else` branch)** — Dropped. v0.1.1's CR.A.1
  poison-pill widening + segment-open CRC verification means corrupt
  dtype codes never reach the kernel in practice; the kernel's
  fast-path `else: UINT8` is acceptable degraded-correct behavior.

### Tests + lint + types

- 10 new regression tests across `test_schema.py`, `test_wal_producer.py`,
  `test_wal_consumer.py` (3 for C.2, 3 for C.3, 2 for B.14, 1 for C.4,
  1 fixup for the batch hash swap).
- **681 tests passing** (was 671 at v0.1.1).
- mypy strict clean (30 source files).
- ruff check + format clean (107 files).

## v0.1.1 — 2026-05-08

Production-readiness patch. Closes 13 confirmed durability /
correctness bugs surfaced by the 2026-05-07 multi-agent audit, scrubs
documentation drift, and hardens trust-boundary inputs. **No new
public-API additions; no functional behavior changes outside the
documented bug fixes.** See [`docs/adr/018-v0_1_1-production-readiness.md`](docs/adr/018-v0_1_1-production-readiness.md)
for the full design narrative.

### Breaking-by-design (one item)

- **CR.A.6 / Schema name validation** — `FeatureSchema` subclasses now
  reject class names that don't match `^[A-Za-z_][A-Za-z0-9_]{0,62}$`
  at class-definition time. Prior behavior silently sanitized
  hyphens/dots via `_safe_class_name`, which created multi-tenant
  collision risk in Redis keys + Parquet partition paths. **If you
  have schemas named with hyphens, dots, or other non-identifier
  characters, they will now fail to import.** Workaround: rename to
  alphanumeric+underscore, or pin to `quorin==0.1.0` and file an
  issue. We considered an env-var escape hatch
  (`QUORIN_PERMISSIVE_SCHEMA_NAMES=1`) but deferred it pending real
  user feedback — env vars create permanent compat surface that's
  painful to remove pre-1.0.

### Durability fixes

- **CR.A.1** — Widened the WAL consumer's poison-pill catch from
  `ValueError` to `(ValueError, OverflowError, struct.error)` so
  numeric-overflow messages from non-pydantic producers surface as
  `wal_consumer_poison_pill_total` instead of silent PEL bloat.
- **CR.A.2** — `ParquetDatasetStore.flush()` now restores unwritten
  buckets to `_buffers` on partial failure or cancellation, instead
  of silently dropping them while online-store XACK fired (data-loss
  fix; deferred-XACK durability contract preserved).
- **CR.A.5** — Pre-flip `schema:current` defense-in-depth guard now
  handles both `bytes` and `str` returns from Redis (fixes silent
  bypass when caller uses `decode_responses=True`).
- **CR.A.7 + B.8-bis** — Schema upgrade adds a `flip_completed` flag
  so post-flip exception paths log instead of orphan-cleaning the
  LIVE new segment. Old-segment refcount also closed via try/finally;
  prior behavior leaked +1 refcount on every aborted upgrade.
- **CR.A.8** — `_CLOSE_LUA` now `DEL`s the refcount key at
  refcount-0, mirroring the dead-PID Lua. Prior behavior leaked
  zero-valued `quorin:refcount:*` keys forever.
- **CR.A.13** — WAL drain precondition is now `XPENDING == 0` (not
  `XLEN == 0`). XACK doesn't decrement XLEN; only XTRIM/XDEL do. The
  prior precondition rejected every production upgrade (XLEN sits at
  MAXLEN forever after meaningful traffic). Operators no longer need
  to manually `XTRIM` before upgrading.
- **CR.B.10** — `quorin._internal.heartbeat._reset_after_fork` now
  rebinds `_state_lock` to a fresh `threading.Lock`. Prior behavior
  had child workers deadlock on first `ensure_started` if the parent
  was mid-`with _state_lock:` at fork time (standard gunicorn /
  uvicorn deployment shape).

### Operability + supply chain

- **CR.A.4** — `can_upgrade` upfront-rejects 2D-shape upgrades
  (`shape=(R, C)`) instead of letting `_build_translation_table` die
  with a cryptic `AttributeError` mid-upgrade. Both old-existing and
  new-only 2D fields are caught. Full 2D translation deferred to
  v0.2.0.
- **CR.A.11** — CI coverage gate raised from `--cov-fail-under=0` to
  `--cov-fail-under=75`. Prior threshold meant any future PR could
  delete tests without CI noticing.
- **CR.C.1** — Defense-in-depth schema-name re-validation at the
  Parquet write boundary (`offline.py`). Closes the runtime
  `cls.__name__` mutation bypass of CR.A.6.
- **CR.C.5 + C.6** — GitHub Actions pinned to commit SHAs (not
  floating tags); `permissions: contents: read` on all workflows.
- **CR.D.5** — `SegmentRegistry.create` now refuses allocations that
  would exceed 50% of free `/dev/shm` with `OSError(errno=ENOSPC)`.
  Surfaces disk-pressure issues before the next mmap fault SIGBUSes
  the process.
- **CR.E.6** — `WALProducer.__init__`, `WALConsumer.__init__`, and
  `heartbeat.ensure_started` emit a `UserWarning` if the supplied
  Redis client lacks `socket_timeout`. On partition, blocking ops
  hang indefinitely; the warning surfaces the issue at construction
  rather than 30 minutes into a stuck consumer.
- **CR.H.4** — `compute_layout` rejects `max_id_bytes >
  MAX_ID_BYTES_CEILING` (64 KiB); defends against corrupt segment
  headers triggering huge allocations on open.
- **CR.H.5** — `upgrade_schema(...)` rejects non-finite or
  non-positive `capacity_factor`. Prior behavior silently saturated
  to `occupied_count + 1` via the `max(...)` floor.
- **CR.H.8** — `quorin/py.typed` marker now explicitly listed in
  `pyproject.toml`'s wheel `include`. Downstream `mypy --strict`
  picks up our types.
- **CR.B.11** — `psutil` upper bound tightened to `<6.0` (watchdog's
  PID-reuse correctness depends on `Process.create_time()` semantics
  staying stable). **Anyone with `psutil>=6.0` resolved in their
  lockfile must downgrade or pin in their own constraint.**

### Documentation + observability cleanup

- **CR.A.9 / A.10** — Removed `quorin_read_latency_seconds` and
  `quorin_wal_lag_seconds` from the Prometheus registry. Both were
  declared at v0.1.0 but never observed. Wiring `.observe()` on the
  hot path costs ~30 ns; opt-in instrumentation is planned for
  v0.2.0 behind an env-var flag. **Operators tracking these in
  Grafana saw zeros.**
- **CR.A.12** — Full rewrite of `docs/USAGE.md` and `docs/API.md`
  against current code signatures. Prior versions had ~12 fictional
  signatures (e.g. `WriteSyncTimeout` vs the real
  `WriteSyncTimeoutError`) that produced `NameError` on copy-paste.
- **CR.A.12-b** — New runnable `examples/` directory
  (`quickstart.py`, `write_sync.py`, `hydration.py`, `upgrade.py`)
  with `pytest examples/` as the future-drift gate. Replaces
  docs-as-doctest (which doesn't parse fenced ```python blocks).
- **CR.D.1 / D.2** — Reconciled "17 ADRs" / "767 tests" claims
  against actual file counts and `pytest --collect-only` output.
- **CR.D.14** — Removed never-implemented `consumer_throughput_*`
  and `consumer_lag_e2e_*` gate references from
  `docs/adr/009-wal-consumer-design.md` §13 and the bench module's
  docstring.
- **New: `docs/operations.md`** — Ops runbook covering Redis
  AUTH+TLS, the `redis-py` fork-unsafe-pool footgun (gunicorn
  pattern), watchdog deployment, DR scenarios, and Prometheus alert
  thresholds.
- **New: `SECURITY.md`** — Vulnerability reporting policy, threat
  model, in-scope vs out-of-scope.

### Tests

- **CR.F.1** — New integration test
  [`tests/integration/test_step15_pause_reopen.py`](tests/integration/test_step15_pause_reopen.py)
  with 5 scenarios covering Step 15's WAL consumer
  pause-and-reopen safety net (`_check_upgrade_pause_and_reopen`),
  which had 0% coverage at v0.1.0.
- Per-fix regression tests for CR.A.1, A.2, A.4, A.5, A.6, A.7, A.8,
  A.13, B.10, C.1, D.5, H.4, H.5.

### Items deliberately not done in v0.1.1

- **DROPPED** (annotated as won't-do in `progress/improvements.md`):
  CR.G.4, CR.I.4 (xxh3 entity-ID hash — investment vs ~6% latency
  payoff is poor), CR.C.12 (subsumed by CR.A.6 + C.1).
- **Deferred to v0.1.2 or v0.2.0**: most CR.B.* "suspicious" items,
  CR.A.3 retry budget, full 2D-shape upgrade translation,
  embedded-mode (no Redis), Numba-jit batch hash (T2.1), macOS port,
  sharding helper, Grafana dashboards, opt-in observability via
  env-var.

### Acknowledgments

The 2026-05-07 multi-agent adversarial audit (recorded in
`progress/improvements.md` §"Critical review") surfaced every
durability bug closed in this release. Without it, several silent
data-loss paths would have shipped to production users.

---

## v0.1.0 — 2026-05-06

Initial release. Feature-complete; **5 µs p99 spec MET on native CI** at the
warm-cache 4-field assemble path.

### Highlights

- **5 µs p99 substantiated** on GitHub Actions ubuntu-latest at N=20
  fresh-subprocesses (`headline_4_field_warm` median_p99 = **4.48 µs**;
  see [`benchmarks/results/n20/headline_4_field_warm_n20.json`](benchmarks/results/n20/headline_4_field_warm_n20.json)).
- 17 numbered build steps shipped (Steps 0 — 16c-d).
- 758 tests passing across unit / property / integration / chaos / benchmark
  layers.
- 16 ADRs ([`docs/adr/`](docs/adr/)) documenting every load-bearing design
  decision (numbered 001-015 + 017; ADR-016 was rolled into 015 + 017
  during Step 16's authoring).
- 30+ regression gates enforced in CI on every PR via
  [`benchmarks/regression/tier1.yml`](benchmarks/regression/tier1.yml).
- New public helper: [`quorin.layout.pack_row`](quorin/layout.py) — kwargs API
  for the synchronous insert path.

### Known scope boundaries

- Single-node only. Beyond ~1M entities, shard horizontally by
  `hash(entity_id) mod N` across multiple Quorin instances.
- Linux / WSL2. macOS untested in CI; native Windows out of scope (POSIX
  shared memory required).
- Requires Redis 7.2+ on the control path. The hot read path never touches
  Redis (per [ADR-002](docs/adr/002-per-open-refcounting.md)).

### Note on planning artifacts

Pre-implementation planning documents were consolidated into the ADR
archive at v0.1.0. The canonical record of every load-bearing design
decision lives in `docs/adr/` (17 numbered ADRs, one per decision); the
value-proposition content is in the README. Git history preserves the
original planning artifacts. (ADR-016 number was retired during Step 16
authoring; net count is 16 ADRs.)

### Acknowledgments

Built on numpy, numba, pyarrow, redis-py, pydantic, posix-ipc, structlog,
prometheus-client. Thanks to all upstream maintainers.
