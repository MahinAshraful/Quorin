# ADR-016: MADV_HUGEPAGE on POSIX shm segments — SHIP

> **Status: DRAFT (Step 16b — pre-canonical-A/B).** This file is staged for the **SHIP outcome**. If the canonical A/B run on a venue with tmpfs THP enabled lands `geo_mean_speedup ≥ 1.5×` AND `p10_speedup ≥ 1.0×`, the operator slots in canonical numbers from `benchmarks/results/madv_ab.json` (TBD markers below), keeps this file, and DELETES the staged ADR-015 §rejection amendment. If the A/B rejects, the operator DELETES this file and keeps the ADR-015 amendment instead. See `progress/step16b_plan.md` §10 for the commit cadence.

## Status

Accepted (Step 16b ships at this commit).

## Context

Step 13's `insert_many` is `/dev/shm` cold-page-fault-bound. ADR-012 §10 / Step 13 progress measured 0.88 GB/s effective on WSL2 tmpfs vs. 4–8 GB/s on native Linux. Step 16's plan §2.3 forecast a 1.5–2× native speedup from `MADV_HUGEPAGE` on the segment mmap (cuts page-fault count by 512× when tmpfs huge-page allocation lands; 4 KB → 2 MB pages). The plan's "bench-then-ship" rule: ship iff measured `geo_mean(speedup) ≥ 1.5×` on the cold-cache 200-field path, with a `p10(speedups) ≥ 1.0×` floor against the noise-dominated edge of the prior.

Tmpfs THP has explicit kernel-config prerequisites (per [Step 16b plan §2.3](../../progress/step16b_plan.md)):

1. Kernel build: `CONFIG_TRANSPARENT_HUGEPAGE_SHMEM=y`.
2. Runtime tunable: `/sys/kernel/mm/transparent_hugepage/shmem_enabled` ∈ `{always, advise, within_size}`.
3. Top-level: `/sys/kernel/mm/transparent_hugepage/enabled` ∈ `{always, madvise}`.

Even when all three are satisfied, MADV_HUGEPAGE is a kernel **hint** — fragmentation can cause the kernel to decline the request. The orchestrator records both sysfs values into `madv_ab.json["sysfs"]` so a measured ≈1.0× can be distinguished from "kernel didn't grant hugepages."

## Decision

`pyforge/_internal/posix_shm.py::create()` calls `m.madvise(mmap.MADV_HUGEPAGE)` unconditionally on Linux, with `(AttributeError, OSError)` swallowed as a non-fatal hint-rejection path. Workers on serving hosts inherit base 4 KB pages on their own VMA unless they too madvise; production correctness lives in the call sites that matter (`SegmentRegistry.open_current` is unconditional too).

Ship-state diff vs. the bench-state Step 16b commit: the `hugepage` kwarg is **removed** from `posix_shm.create` / `open_existing` / `SegmentRegistry.create` / `open_current` / `tests/_helpers.py::make_segment`. The `m.madvise(...)` call becomes unconditional with the same try/except. The bench fixture toggle (`PYFORGE_AB_HUGEPAGE` env var) is also removed since both sides of the A/B are now the same code path.

## Measured A/B (canonical CI run)

**Venue:** TBD (e.g., GitHub Actions `ubuntu-latest` ubuntu-24.04, run ID TBD, commit TBD).
**N:** 20 fresh subprocesses per side. **Bench:** `test_bench_assemble_200_field_cold_cache_numba` (cold-cache, L3-clobbered between calls).

**Sysfs state at A/B time:**
- `/sys/kernel/mm/transparent_hugepage/enabled`: TBD (must be `[always]` or `[madvise]` for SHIP).
- `/sys/kernel/mm/transparent_hugepage/shmem_enabled`: TBD (must NOT be `[never]` / `[deny]` / `[force]`).

**Result:**

| Metric | Value | Threshold | Pass? |
|---|---|---|---|
| `geo_mean(speedup)` | TBD× | ≥ 1.5× | TBD |
| `p10(speedup)` | TBD× | ≥ 1.0× | TBD |
| `n_actual_runs` (false / true) | TBD / TBD | both ≥ 16 | TBD |
| `n_pairs_compared` | TBD | — | — |

Full per-side per-run p99s + paired-rank speedups + failure reasons: see `benchmarks/results/madv_ab.json`.

## Alternatives considered

- **Always-on without an A/B (faith-based ship).** Rejected: forum-post estimates were 1.5–2× native but unverified against this codebase. The plan's "bench-then-ship" discipline (Step 16 plan §2.3) is the project's standard for performance-suspicious changes.
- **Per-segment opt-in via a kwarg in production.** Rejected: complicates the call-site contract for negligible benefit (operators almost always want the speedup; the kernel already handles the rare reject path internally). The `hugepage` kwarg existed only during the A/B bench window.
- **Migrate `/dev/shm` mount options at deploy time.** Rejected: out of scope for a Python library — `mount -o huge=advise` is a host-admin task, not Pyforge's. The library asks the kernel via `MADV_HUGEPAGE`; honoring it is the host's job.

## Consequences

- **Workers must `open_current` to inherit hugepages.** Step 4's serving path calls `open_current`, which calls `posix_shm.open_existing` — both `madvise(MADV_HUGEPAGE)` unconditionally. Per-VMA THP semantics: each opening process gets its own madvise, regardless of what the creator did.
- **Hosts without `CONFIG_TRANSPARENT_HUGEPAGE_SHMEM=y` are unaffected.** The `(AttributeError, OSError)` swallow keeps the segment alive at base 4 KB pages.
- **Kernel may decline under fragmentation.** A long-running host accumulating fragmentation in tmpfs may stop granting hugepages mid-day; the segment stays alive at base 4 KB pages, no crash. Operators see the same write throughput as pre-Step-16b in that case.
- **Bare-metal results may differ.** Canonical A/B is on ubuntu-latest's older Xeon class L3 (~30 MB shared). 2024-era desktop CPUs have larger L3 (~100 MB) and may see a different speedup ratio — could be larger or smaller depending on whether the bench's working set already fit in L3 pre-MADV. Operators on bare metal should measure for their workload; the SHIP decision is venue-specific to ubuntu-latest.

## Validation

- `madv_ab.json["decision"] == "SHIP"` (gates `geo_mean ≥ 1.5×` AND `p10 ≥ 1.0×`).
- `madv_ab.json["sysfs"]` confirms tmpfs THP was actually granted (not `[never]`).
- `tests/unit/test_posix_shm.py::test_hugepage_kwarg_no_error` retired (kwarg removed); the unconditional `m.madvise(...)` is exercised by every existing posix_shm test.
- All Tier-1 + Tier-2 gates green on native CI for the post-A/B-unwind state.
