# ADR-018 — v0.1.1 production-readiness patch

**Status:** Accepted (v0.1.1, 2026-05-08).

## Context

Quorin v0.1.0 shipped to PyPI on 2026-05-05 (Step 17). On 2026-05-07 a
multi-agent adversarial audit walked the codebase + docs end-to-end and
filed `progress/improvements.md` with **13 confirmed durability /
correctness bugs** (CR.A.* class) and ~30 documentation drifts (CR.D.*).
Independent re-verification on 2026-05-08 against `main` at commit
`cb59c68` confirmed every CR.A.* finding still reproduced.

v0.1.1 is the focused production-readiness patch that closes the
confirmed bugs, hardens the supply chain, and rewrites the quickstart so
it actually runs. **No new public API. No new features.** The only
breaking-by-design change is CR.A.6 schema-name validation — and the
prior behavior was already silently corrupt, so failing loudly is the
correct fix per ADR-014's design philosophy ("convert silent corruption
into loud failure").

This ADR captures the load-bearing decisions. The full per-fix table
lives in [`progress/v0_1_1_plan.md`](../../progress/v0_1_1_plan.md);
this document records *why* each call went the way it did so the next
audit cycle doesn't re-litigate the same trade-offs.

## Decisions locked

### 1. Patch (not minor) bump

Pre-1.0 semver permits patch for any non-breaking change. v0.1.1
contains zero new public functions, zero new public classes, and zero
new public exception types. Every signature in `quorin.shm`,
`quorin.layout`, `quorin.schema`, `quorin.evolution`, `quorin.hydration`,
`quorin.wal`, `quorin.wal_consumer`, and `quorin.offline` is
byte-identical to v0.1.0.

The single breaking-by-design change is **CR.A.6 schema-name
validation**: a `FeatureSchema` subclass with hyphens, dots, or other
non-`[A-Za-z0-9_]` characters now raises `ValueError` at
`__init_subclass__`. We accept this as a patch-level break because:

- The v0.1.0 behavior was already corrupt (`_safe_class_name`
  silently sanitized non-conforming names → multi-tenant Redis-key /
  Parquet-path collision).
- Failing loudly at class-definition time is strictly better than
  letting the corruption ship to production.
- Any subclass that hits this check today was already producing
  collisions; no working workload regresses.

The CHANGELOG calls the break out in bold. The reactive escape hatch
(`QUORIN_PERMISSIVE_SCHEMA_NAMES=1`) is **deferred to v0.1.2 if user
feedback warrants** — env vars create permanent compat surface that
becomes painful to remove pre-1.0. We'd rather ship the strict default
and add the escape hatch only if the issue tracker demands it.

### 2. Schema-name validation at `__init_subclass__` (CR.A.6)

```python
# quorin/schema.py
_SCHEMA_NAME_PATTERN = re.compile(r"^[A-Za-z][A-Za-z0-9_]{0,62}$")

class FeatureSchema:
    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        if not _SCHEMA_NAME_PATTERN.match(cls.__name__):
            raise ValueError(
                f"FeatureSchema subclass name {cls.__name__!r} is invalid: "
                "must match ^[A-Za-z][A-Za-z0-9_]{0,62}$. "
                "(CR.A.6 / ADR-018)"
            )
```

The 63-char ceiling sits well under POSIX `NAME_MAX=255` even after the
`quorin_{name}_v{n}_{uuid8}` prefix overhead used in `_segment_name`.
Generous-but-bounded.

**Why class-definition rather than runtime sanitize?** Multi-tenant
collision risk in Redis keys, `/dev/shm` paths, and Parquet partition
directories. v0.1.0's `_safe_class_name` mapped both `MySchema-v1` and
`MySchema_v1` to the same sanitized identifier — silent
namespace-collision in any deployment with even one operator who used
hyphens. The fix has to either (a) make the sanitization
collision-resistant (out of scope for a patch — needs a hash suffix
that breaks segment-name compatibility) or (b) reject the input. We
took (b).

### 3. CR.C.1 defense-in-depth at the Parquet boundary

`__init_subclass__` only catches **definition-time** names. `cls.__name__`
is a writable attribute — `Foo.__name__ = "../etc/passwd"` would slip
past CR.A.6 and the mutated value would flow into the partition path
construction in `quorin.offline._write_table` (and `read_point_in_time`).

The Parquet write is the **one out-of-process trust-boundary crossing**
in Quorin's single-machine model. Everything else lives in `/dev/shm`
or Redis, which are process-controlled by definition. The Parquet
output writes named directories under a user-supplied `base_path` that
might be (a) a shared dataset volume, (b) a network mount, (c) symlinked
into a downstream training pipeline. Path traversal there is the only
file-system escape vector.

Re-validation at the boundary, in `quorin/offline.py`:

```python
def _validate_schema_name_for_path(schema: type[FeatureSchema]) -> str:
    name = schema.__name__
    if not _SCHEMA_NAME_PATTERN.match(name):
        raise ValueError(
            f"schema.__name__ {name!r} is invalid for filesystem path "
            "construction: ... typically indicates cls.__name__ was "
            "mutated after class definition (CR.A.6 / CR.C.1 / ADR-018)."
        )
    return name
```

This is defense-in-depth, **not** a security guarantee. ADR-018
documents the trust model explicitly: we trust schema authors not to
alias `__name__` post-hoc; the Parquet sanitizer makes accidental
breakage loud. Runtime mutation by a hostile in-process actor is
out-of-scope (an attacker with that level of access can already
`shutil.rmtree` the dataset directory directly).

### 4. Upfront-reject 2D-shape upgrades in `can_upgrade` (CR.A.4)

v0.1.0's `_build_translation_table` builds a single-level
`FixedSizeListArray` regardless of `len(field.shape)`. The
`insert_kernel`'s rank-2 path then does `.flatten().flatten()` on the
PyArrow array, which `AttributeError`s on the resulting primitive
array — silent silent crash mid-upgrade with `schema:current` flipped
into an inconsistent state.

Full nested-`FixedSizeList` support is an ~3-day effort across
`_build_translation_table`, `_arrow_plan_for`, and `insert_kernel`'s
rank-2 dispatch. **Deferred to v0.2.0.** v0.1.1 closes the silent
breakage by rejecting the case upfront in `can_upgrade`.

The check is **two-loop**, not one. Both code paths fail the same way
in `insert_kernel`:

- **Loop 1** (existing-name intersect): for every field shared between
  OLD and NEW, reject if either side has `len(shape) >= 2`. Catches
  "kept its 2D shape" + "widened from 1D to 2D".
- **Loop 2** (NEW-only fields): reject if NEW added a 2D field that
  wasn't in OLD. The translation table reads OLD bytes for shared
  fields, but new-only fields are populated with default zero values
  via `insert_kernel` — and `insert_kernel`'s rank-2 path is exactly
  what blows up.

A single intersect-only loop would miss the new-only path; that was
caught in plan review (Rev-2). The error message names CR.A.4 + ADR-018
so operators have a thread to pull on.

### 5. Remove dead metrics, don't wire them (CR.A.9 / A.10 / D.14)

v0.1.0 declared two Prometheus metrics that **no production code path
ever observed**:

- `quorin_read_latency_seconds` — declared in `quorin/metrics.py`,
  documented in `docs/USAGE.md` and `docs/API.md`. Zero `.observe()`
  call sites.
- `quorin_wal_lag_seconds` — same shape: declared, documented, never
  observed.

Operator dashboards relying on these scraped 0 forever — the metric
existed in `/metrics` but reported "everything is instant." That's
strictly worse than the metric being absent (then they'd know they
need to instrument).

Two options:

- **(a) Wire them up.** Adding `.observe()` calls on the assemble +
  consumer hot paths costs ~30 ns per call (Histogram observe with
  pre-warmed labels — see [ADR-006](006-gc-management.md)'s cost
  finding). On the **4.48 µs p99 hot path** that's a 0.7% measured
  regression — bench-significant per [ADR-015](015-benchmark-methodology.md)
  Tier-1 gates. Doing this *correctly* (opt-in via env var, gated
  prewarm, multiprocess-collector compatibility, a tested off-path)
  is real engineering work.
- **(b) Delete the declarations.** Honest answer for v0.1.1 — operators
  see the metric is gone, the README + USAGE doc no longer claim
  observability we don't deliver, v0.2.0 ships proper opt-in.

We took (b). Same logic applies to **CR.D.14**: ADR-009 §13 referenced
`consumer_throughput_*` and `consumer_lag_e2e_*` regression gates, but
the underlying benches don't exist; the gate-check fail-soft to no-op.
We removed the gate references from ADR-009 and the bench module
docstring rather than ship the benches in a patch release. Same theme:
no false claims.

### 6. WAL precondition: PENDING, not XLEN (CR.A.13)

v0.1.0's `quorin.evolution.upgrade_schema` precondition was:

```python
if redis_client.xlen(DEFAULT_STREAM_KEY) > 0:
    raise UpgradeConflictError("WAL not drained")
```

This is wrong. **`XACK` does not decrement `XLEN`**; only `XTRIM` /
`XDEL` shrink the stream. The `WALProducer` writes with `MAXLEN ~ N`
(approximate trim), so after even a single hour of production traffic
`XLEN` ≈ MAXLEN forever. The v0.1.0 check effectively required
operators to manually `XTRIM 0` before every upgrade — undocumented,
unintuitive, and the `XTRIM` itself could race with in-flight producer
writes.

The correct measure is **un-ACKed messages** (consumer lag), reported
by `XPENDING`:

```python
xp = redis_client.xpending(DEFAULT_STREAM_KEY, DEFAULT_GROUP_NAME.encode("ascii"))
pending = int(xp[0]) if xp else 0
if pending > 0:
    raise UpgradeConflictError(
        f"WAL not drained: {pending} messages still pending consumer ACK..."
    )
```

PENDING goes to zero when the consumer drains, which is the actual
"safe to upgrade" condition. ADR-014's operator runbook is amended to
match. Operators no longer need to XTRIM before upgrading.

### 7. Restore `_buffers` on flush failure (CR.A.2)

v0.1.0's `ParquetDatasetStore.flush()` reset `self._buffers = {}` **at
the top**, before any I/O. Rationale (per the original ADR-010
comment): "cancellation atomicity — if `CancelledError` lands mid-flush,
the buckets that were being written are gone but no half-written state
is observable." That trade-off was wrong: it lost data on any
partial-failure path (disk full mid-write of bucket K of N → buckets K
through N silently dropped → `_flush_and_ack` proceeds → consumer
XACKs messages whose offline-store data was never written).

The deferred-XACK invariant ([ADR-009 §3](009-wal-consumer-design.md))
gates XACK on `flush()` returning successfully. Honoring that contract
requires "successfully" to mean "every bucket made it to disk", not
"the in-memory buffer is empty regardless of what landed."

The v0.1.1 design uses pop-as-we-go + restore-via-`setdefault`:

```python
async def flush(self) -> None:
    snapshot = {k: v for k, v in self._buffers.items() if v.entity_id_col}
    self._buffers = {}
    if not snapshot:
        return
    try:
        for key in list(snapshot.keys()):
            ...write_table(snapshot[key])...
            snapshot.pop(key)  # mark as durable
    except (asyncio.CancelledError, Exception):
        for key, bucket in snapshot.items():
            self._buffers.setdefault(key, bucket)
        raise
```

`setdefault` is the load-bearing call: if a defensive caller appended a
fresh bucket during the await window of `_write_table`, restore
preserves the new entry without clobbering. Both `CancelledError` and
`Exception` paths restore unwritten buckets — cancellation atomicity
**and** partial-failure durability are now both preserved. Chaos test
`test_buckets_restored_on_mid_flush_exception` is the binding
regression.

### 8. Post-flip safety via `flip_completed` flag (CR.A.7 + B.8-bis)

v0.1.0's `upgrade_schema` had a latent outage path: any exception
raised **after** `FLIP_SCHEMA_CURRENT_LUA` returned success — for
example, `registry.close(old_seg)` raising on a transient Redis
hiccup — would land in the `except Exception:` branch and call
`_cleanup_orphan_new_segment(new_seg, ...)`. By that point `new_seg`
was the LIVE production segment; "orphan cleanup" would `posix_shm.unlink`
it from under live readers.

Pre-existing bug in 0.1.0; auditor flagged as B.8-bis. v0.1.1 adds a
single boolean:

```python
flip_completed = False
try:
    ...
    flip_result = flip_script(keys=[...], args=[...])
    if flip_result == 1:
        flip_completed = True  # LIVE state from this point
    ...
except Exception:
    if new_seg is not None and not flip_completed:
        _cleanup_orphan_new_segment(new_seg, redis_client, new_schema)
    elif new_seg is not None and flip_completed:
        logger.error("evolution.post_flip_failure_new_segment_preserved", ...)
    raise
```

Free roll-up under CR.A.7's primary fix (the `try/finally` on `old_seg`
to prevent leaking the OPEN refcount on upgrade failure). Both fixes
landed in the same edit because the safety analysis touches the same
exception handler. Test `test_old_seg_closed_on_upgrade_failure` is
the binding check on A.7; the post-flip preservation case has its
own test.

### Other fixes (1-2 lines each)

- **CR.A.1 — widened poison-pill catch.** v0.1.0 caught `ValueError`
  from `pack_row_from_list` length mismatch. v0.1.1 widens to
  `(ValueError, OverflowError, struct.error)` to also catch numeric
  out-of-range values (e.g. a float64 producer enqueuing `1e40` for a
  float32 field — `struct.pack_into('<f', ...)` raises `OverflowError`).
  We **did not** add producer-side range validation: pydantic's `lt`/`gt`
  on `Field(allow_inf_nan=True)` rejects NaN, breaking
  invariant #12 (NaN bit-pattern preservation through the assemble
  oracle). Catching wider on the consumer is the right side of the
  asymmetry — producer stays permissive, consumer stays loud.

- **CR.A.5 — pre-flip guard accepts both `bytes` and `str`.** v0.1.0's
  `isinstance(current, bytes)` branch was silently bypassed by clients
  using `decode_responses=True`. v0.1.1 accepts both shapes.

- **CR.A.8 — `_CLOSE_LUA` DELs the refcount key.** At refcount-zero
  the close-Lua was leaving an orphan `quorin:refcount:{name}` key
  with value `0`. Mirrors `CLEANUP_DEAD_PID_LUA`'s existing `DEL`
  pattern. Five-line fix in `quorin/shm.py`.

- **CR.B.10 — fork-lock rebinding.** `heartbeat._state_lock` is
  module-level; a fork while a parent thread holds it inherits a
  pre-locked lock in the child, deadlocking the child's first
  `ensure_started`. v0.1.1's `_reset_after_fork` rebinds the global to
  a fresh `threading.Lock()`. The 10-min fix; the harder ~1 hr was the
  test that itself doesn't deadlock when verifying the fix.

- **CR.B.11 — psutil pin tightened to `>=5.9,<6.0`.** psutil 6.0
  introduced API changes that haven't been verified against our
  cross-check assertions (invariant #4 + ADR-013's exact-equality
  PID-create-time compare). Tightening is the conservative pre-1.0
  default; bumping the upper bound is a separate planned exercise
  with its own A/B verification.

- **CR.D.5 — `/dev/shm` 50% capacity guard with `FileNotFoundError`
  fallback.** `SegmentRegistry.create` reads `os.statvfs("/dev/shm")`
  and raises `OSError(errno=ENOSPC)` if the requested allocation
  exceeds 50% of free space. Surfaces disk-pressure issues at create()
  rather than letting `posix_shm.create` succeed and the next mmap
  fault SIGBUS the process. On systems without `/dev/shm` (rare;
  sandboxes, non-Linux POSIX) we emit `UserWarning` and skip — the
  next `posix_shm.create` will fail with its own clearer error per
  invariant #9 (Linux/WSL2 only). Don't mask the wrong-platform
  signal.

- **CR.H.4 / H.5 — input caps.** `compute_layout_from_segment` rejects
  unreasonable `max_id_bytes` values (defense against a corrupted
  segment header). `upgrade_schema` and `hydrate` reject
  `capacity_factor=0` (would produce a segment with zero slots —
  silent upgrade-completes-but-every-lookup-misses).

- **CR.E.6 — `socket_timeout` UserWarning.** `WALProducer.write`,
  `WALConsumer.run`, and `heartbeat.ensure_started` now emit a
  `UserWarning` if the supplied `redis.Redis` / `redis.asyncio.Redis`
  client lacks a finite `socket_timeout`. A network partition can
  block the heartbeat HSET indefinitely; `heartbeat.stop()` joins
  with a 2 s timeout, so a blocked HSET leaks the thread on shutdown.
  The warning at construction time gives operators the flag at the
  API boundary; ignoring it is a deliberate choice. Project-wide
  concern (all three call sites share the shape).

## Consequences

**Positive:**

- All 13 confirmed CR.A.* durability bugs closed. The "no silent
  corruption" theme is now substantiated; `progress/improvements.md`'s
  CR.A section moves to DONE on 2026-05-08.
- The hot-path 5 µs p99 claim is preserved — none of the v0.1.1 fixes
  add observe() calls or extra Redis round-trips on the assemble path.
- The supply chain is hardened: SHA-pinned GitHub Actions
  ([CR.C.5](../../.github/workflows/ci.yml)), `permissions: contents: read`
  block on every workflow (CR.C.6), `SECURITY.md` reporting address
  (CR.C.9), Redis AUTH/TLS guidance in `docs/operations.md` (CR.C.10).
- The quickstart actually runs. `examples/*.py` is the new
  future-drift gate; `pytest examples/` lands as a CI step. Runnable
  scripts beat docs-as-doctest for the reasons in
  `progress/v0_1_1_plan.md` §2 commit 3.

**Negative:**

- Users with non-conforming `FeatureSchema` subclass names hit
  `ValueError` at import. Mitigation: bold CHANGELOG note + workaround
  (rename or pin `quorin==0.1.0` + file an issue).
- Operators whose runbook relied on `XLEN > 0` upgrade rejection
  ("the stream is alive") need to update — `XLEN` is no longer the
  precondition. ADR-014's runbook + CHANGELOG document the change.
- `quorin_read_latency_seconds` and `quorin_wal_lag_seconds` are gone
  from the registry. Anyone with a Grafana panel querying them sees
  empty. CHANGELOG points to v0.2.0's planned opt-in instrumentation
  and suggests `wal_consumer_apply_total` rate as an interim proxy.

## Out of scope (deferred)

### To v0.1.2 (if needed before v0.2.0)

- All CR.B.* "suspicious" items (CR.B.1, B.2, B.3, B.4, B.5, B.6, B.8,
  B.9, B.12, B.13, B.14) — needs case-by-case investigation.
- **CR.A.3** — retry budget for offline-failure-only path. The
  CR.A.2 fix preserves un-flushed buckets, but a permanently-failing
  offline write currently grows `_buffers` without bound.
- **CR.E.* operational hardening** beyond `socket_timeout` warning:
  E.1 TTLs, E.2 DR docs, E.3 watchdog-of-watchdog, E.4
  `appendfsync`, E.5 `write_sync × flush_interval` interaction,
  E.7 multiproc Prometheus metrics.
- All **CR.F.* test gaps** that aren't F.1 (Step 15 pause-and-reopen
  shipped in v0.1.1).
- `QUORIN_PERMISSIVE_SCHEMA_NAMES=1` reactive escape hatch for
  CR.A.6 — only if user feedback demands it.

### To v0.2.0 (theme: performance + adoption)

- **CR.A.4 full fix**: nested `FixedSizeListArray` for proper 2D-shape
  upgrade (~3 days across translation, arrow plan, and rank-2 insert
  kernel).
- **CR.A.9 / A.10 wire-up via opt-in env var**: properly designed
  observability with prewarmed labels + multiproc-collector
  compatibility. The hot-path cost is real engineering, not a
  one-line `.observe()` add.
- **T2.1 Numba-jit batch hash** (recovers batch speedup ratio per
  ADR-007 amendment).
- **T4.5 embedded mode**, **T6.4 macOS port**, **T6.5 sharding helper**.
- **T6.2 Numba 0.61+** / **T6.3 PyArrow 16+** asof-join migration.
- **CR.G.* API ergonomics** not in v0.1.1.

### To v0.3.0+ (strategic)

All CR.I.* strategic items: PyO3 Rust core, msgspec swap-out, gRPC
sidecar, MAP_HUGETLB, NUMA-aware placement, SwissTable, etc. Plus
T1.1 (C-extension pool overhead per ADR-005 §7c amendment), T1.5
(GC interaction class), T2.2-T2.4 deeper perf engineering.

### Items DROPPED (won't-do)

- **CR.G.4** — `assemble_batch` `AttributeError` on `bytes` input.
  Current behavior is correct per the docstring (`Sequence[str]`);
  adding runtime input-type validation would cost ~50 ns on the hot
  path with no UX gain over the existing exception. The exception
  message is the validation.
- **CR.I.4** — xxh3 entity-ID hash. Saves ~270 ns of 4480 ns p99
  (~6%) at the cost of bumping the segment magic ([invariant
  #4](../../CLAUDE.md#5-non-negotiable-invariants)) and introducing
  adversarial-collision risk for entity IDs that could be
  attacker-influenced. Investment-vs-payoff is poor; pinned-hash
  invariant #5 is load-bearing for persisted-segment compatibility.
- **CR.C.12** — trust-boundary inconsistency. Subsumed by the
  CR.A.6 + CR.C.1 unified validator. Annotated WON'T-DO in
  improvements.md.
- **CR.A.9 wire-up path-b** — see Decision #5 above.

## References

- [`progress/v0_1_1_plan.md`](../../progress/v0_1_1_plan.md) — the full
  per-fix table, decision matrix, and 4-commit structure.
- `progress/improvements.md` — the 2026-05-07 audit findings with
  CR.A.* / CR.B.* / CR.C.* / CR.D.* / CR.F.* / CR.H.* identifiers
  referenced throughout this ADR.
- [ADR-009](009-wal-consumer-design.md) — deferred-XACK durability
  invariant (Decision #7 above honors it).
- [ADR-010](010-parquet-offline-store.md) — Parquet writer design;
  Decision #7 amends the flush ordering documented there.
- [ADR-014](014-schema-evolution.md) — schema evolution + the
  operator runbook that Decision #6 amends (PENDING-not-XLEN).
