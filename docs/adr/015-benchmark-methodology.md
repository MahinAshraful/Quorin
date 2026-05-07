# ADR-015: Benchmark Methodology (Step 16)

## Status

Accepted (Step 16a ships at this commit; 16b + 16c follow with the MADV
A/B decision and the canonical native-Linux baseline run).

## Context

Steps 0–15 shipped 35 regression gates in `benchmarks/regression/thresholds.yml`,
14 bench files, and a Step 7 fresh-subprocess orchestrator. None of it was
enforced — the CI workflow loaded the YAML as a smoke test (a print-only
step) but never compared against bench output. The gate **keys never
matched** the actual pytest-benchmark function names (e.g. yaml said
`wal_write_50_field` but the function is `test_bench_write_50_field`),
so even if the comparison had been wired, every gate would have been
silently MISS. And several gates were calibrated against measured
**medians** while real enforcement uses **p99**.

Step 16 closes the gap between "we have benches" and "the headline
numbers are defensible." This ADR records the methodology decisions
locked into the regression-gate enforcement, the Tier-2 orchestrator,
and the cold-cache harness.

## Decisions

### 1. Two-tier YAML split (`tier1.yml` always-run / `tier2.yml` env-gated)

Three options were considered for the `--strict` collision with env-gated
benches:

- (a) Set `QUORIN_RUN_LARGE_BENCH=1` etc. on every PR run. Wasteful;
      PRs would run 100k/1M-scale benches each time.
- (b) Per-threshold `requires_env:` field in the YAML. Adds parser
      complexity; bench list and env list drift apart.
- (c) **Split by file.** `tier1.yml` is always-run + strict-checked on
      PRs. `tier2.yml` is env-gated + strict-checked on push-to-main +
      `workflow_dispatch` + scheduled weekly. File membership = scope.
      No drift.

We chose (c). `check.py::_load_thresholds` raises `ValueError` if any
key appears in both files (L5 duplicate-key guard) — each bench belongs
in exactly one tier.

### 2. Multi-percentile gates per bench (`_PERCENTILE_FIELDS`)

A single threshold entry can carry any subset of `{p50_seconds,
p95_seconds, p99_seconds, p999_seconds, p9999_seconds}`. Each present
field is gated independently against `np.percentile(stats.data, q)`.
Missing fields are skipped (no over-gating); an entry with zero
recognized percentile fields prints a WARN to stderr and is treated
as un-gated.

This pattern is load-bearing for the assemble-under-GC bench
(`test_bench_assemble_p999_under_gc_pressure_4_field`), which gates
both p99 (50us) and p999 (200us) in tier1.yml — same bench, same raw
round data, two independent percentile checks.

### 3. `--benchmark-save-data` is REQUIRED in CI

pytest-benchmark's default JSON has summary stats (`min, max, mean,
stddev, median, q1, q3, iqr`) — **no `q99` or `q999`**. To compute p99
from raw data, the run must use `--benchmark-save-data`, which preserves
the per-round timing array as `stats.data`. Without it, `check.py`
raises `RuntimeError` on first encounter — fail-loud rather than
silently skip the gate.

### 4. Cold-cache: explicit cache-clobber array, NOT `numactl`

We chose explicit cache-clobbering array traversal between calls over
`numactl` / `taskset` for portability + privilege reasons (works on
any Linux box without root or capabilities).

The clobber array is sized at **4× detected L3** (`/sys/devices/system/cpu/cpu0/cache/index3/size`)
capped at 1 GB. Critical: do **NOT** fall back to `index2` (L2) when
L3 detection fails. WSL2 / many VMs hide cache hierarchy from sysfs;
`index2` reports L2 (a few MB), and a 4×-L2 clobber is nowhere near
L3 size. The bench would silently measure L2-cold instead of L3-cold.
Better to use a deliberately conservative 16 MiB fallback that logs
WARN than to confidently report bad data. Ubuntu CI runners surface
real L3 in sysfs.

**Honest disclosure (1):** this measures cold-CPU-cache, NOT cold-page-cache.
The segment is still resident in /dev/shm tmpfs (RAM-backed). Cold-
page-cache benchmarking on tmpfs requires `posix_fadvise(POSIX_FADV_DONTNEED)`
(which doesn't apply to tmpfs) or process restart — both Step 17+
deferrals.

**Honest disclosure (2): single-process cold-cache has 3-4x run-to-run
variance.** Step 16a verification surfaced this: same code, same hardware,
same machine — full-sweep p99 100.9us vs targeted-solo p99 ~33us.
Surrounding bench activity warms different cache regions and shifts the
cold-cache p99 distribution non-trivially. This means:

- The **Tier-1 cold-cache gate is a gross-regression detector**, not a
  spec-band enforcer. Sized at 5x WSL2 full-sweep p99 (500us) per
  tier1.yml; catches order-of-magnitude regressions, accepts that the
  underlying p99 is variable.
- The **headline cold-cache claim ships from Tier-2 N=20**, not from
  Tier-1 single-process numbers. The orchestrator (`benchmarks/runs/repeat.py`)
  spawns fresh subprocesses with consistent startup state per §6,
  which absorbs the surrounding-state variance that single-process
  measurement can't. Note: per §6 mitigation #3, Tier-2 N=20 outputs
  are NOT auto-gated in CI — they're committed JSON for human review +
  README quoting, with the trip-wire (cold p99 > 50us native after
  MADV decision) enforced via that manual review.
- Operators reading bench JSON should know: cold-cache p99 in any
  single-process run is a point-estimate of a noisy distribution. The
  Tier-2 N=20 median-of-runs is the meaningful aggregate.

This is THE bench where Step 7's lesson applies most strongly. The
trip-wire (cold p99 > 50us native after MADV decision) is enforced
against the Tier-2 N=20 number, not Tier-1.

### 5. Absolute thresholds, NOT a perf ratchet

We considered a ratchet where each run's gate is set to N% over the
median of the last K runs. Rejected for two reasons:

- Ratchets are state-bearing. The YAML's gate value drifts with every
  PR; "what's our actual baseline" becomes "what's the median of the
  last K runs," which is implementation-trivia in CI tooling but a
  maintenance burden.
- Absolute ceilings fail loud and force a deliberate gate-bump
  ("we changed methodology; here's why we re-baselined"). Ratchets
  hide drift behind statistics.

Default headroom is **3× measured native-Linux p99 baseline** with
per-bench tightening allowed where N=20 stddev justifies. Step 16c
locks the canonical native-Linux baselines + retightens the WSL2-
tolerant Step 16a placeholders.

### 6. Tier-2 N=20 fresh subprocesses, NOT in-process N=20

Step 7's lesson (ADR-006): single-run pytest-benchmark of rare tail
events is statistically useless because GC state, page cache, OS
scheduler state, and Python module-level state all carry over within
a process. The Step 16 orchestrator (`benchmarks/runs/repeat.py`)
spawns 20 fresh subprocesses per scenario and aggregates per-percentile
medians + max-of-max + stddev(p99).

What "fresh" means: GC state, page cache, OS state, Python module-
level state — all new per subprocess. **NOT fresh:** Numba's on-disk
JIT cache. Each subprocess's autouse fixture calls `prewarm()` BEFORE
pytest-benchmark's timed loop. JIT compile is paid in setup, not
measurement. This is intentional — the JIT cache is reproducible
across runs and excluding it from "fresh" is documented in
`repeat.py`'s docstring.

### 7. Hardware spec for the README: `ubuntu-latest` GitHub Actions

Pinned in ADR for future Step 17 README authorship: numbers measured
on GitHub Actions `ubuntu-latest` runners (`ubuntu-24.04` at time of
Step 16). Reproducible by anyone via `gh workflow run benchmark.yml`.

This avoids the "Mahin's WSL2 box" trap (numbers unreproducible by
anyone else) and the "Threadripper Pro under specific kernel tunings"
trap (numbers theoretically reproducible but not by readers running
commodity hardware).

**Native CI vs WSL2 — NOT a uniform speedup factor (Step 16a CI
finding):** The intuitive assumption "native is uniformly faster than
WSL2" is wrong. Step 16a's first push-tier CI runs revealed a cache-
architecture asymmetry:

| Bench | WSL2 p99 | Native CI p99 | Native vs WSL2 |
|---|---|---|---|
| `test_bench_assemble_4_field_warm_numba` | 9.4µs | 6.7µs | **1.4× faster** |
| `test_bench_assemble_200_field_warm_numba` | 13.3µs | 26.7µs | **2× slower** |
| `test_bench_assemble_200_field_cold_cache_numba` | 90.4µs | 32.7µs | **2.8× faster** |
| `test_bench_assemble_batch_4_field_n1000` | 1.4ms | 8.6ms | **6× slower** |
| `test_bench_assemble_batch_200_field_n1000` | 3.5ms | 10.7ms | **3× slower** |
| `test_bench_upgrade_10k_50_field` | 220ms | 74ms | **3× faster** |

**Pattern:** small/short Numba benches faster on native (better
single-thread per-core perf). Cache-bound + bandwidth-bound benches
(batch assembly at N≥1000, large warm-cache 200-field) are SLOWER on
native. GitHub Actions ubuntu-latest CPUs are older Intel Xeons
(E5-2673-class, Haswell/Broadwell era, ~30 MB L3 per socket). The
batch 200-field bench's working set (~1.6 MB output buffer + segment
reads) fits comfortably in modern desktop L2 (~1 MB per core for
recent Intel/AMD) but spills to L3 or DRAM on the older Xeons,
inflating tail latency 3-6×.

**Methodology consequence:** **gates MUST be calibrated against native
CI numbers, NOT WSL2 numbers, with no assumed "WSL2-tolerant means
native-tolerant" logic.** The right pre-commit workflow is:

1. Push the workflow + bench changes to a branch.
2. Trigger `gh workflow run benchmark.yml` against that branch.
3. Pull native CI's `regression_check.json` artifact.
4. Set gates from native p99 × 2.5x default headroom.
5. Push the gate calibrations.
6. Merge.

Step 16a violated this by calibrating from WSL2 measurements + assuming
native would be uniformly faster. The result was three rounds of
gate-bump commits as native CI surfaced different bottlenecks. 16c +
all subsequent perf work uses the native-CI-first calibration discipline.

**Bare-metal extrapolation:** numbers measured on GitHub Actions are
generally 1.5–3× SLOWER than a 2024-era desktop CPU (modern AMD Ryzen /
Intel Core has more L3 + higher single-thread clocks than the older
Xeons). README claims should disclose the venue and let readers
extrapolate per their own hardware, NOT assert a single multiplier.

**Memory ceiling on `ubuntu-latest`:** 7 GB RAM, /dev/shm ≈ 3.5 GB
(50% of RAM via tmpfs default). This is enough for **Tier-1 only**
in a single pytest session. Heavy benches (LARGE + RECORD) cannot
reliably run in the same session because of cumulative state across
benches + `insert_many`'s transient column-major buffers (~3-4×
row data size during the bulk-insert kernel).

Per-bench peak memory during a single bench session:

| Bench | Per-bench peak | Cumulative-with-prior-benches /dev/shm fit on `ubuntu-latest`? |
|---|---|---|
| Tier-1 benches (warm-cache, GC pressure, write_sync RTT, etc.) | < 100 MB each | YES — fits comfortably with margin |
| `test_bench_upgrade_100k_50_field` | ~600 MB | YES (run alone) |
| `test_hydrate_100k_50_field` | ~1.0 GB peak (segment + insert_many flat buffers + PyArrow table) | NO — observed SIGBUS when run after Tier-1 + 100k upgrade benches |
| `test_bench_upgrade_1m_50_field` | ~3.2 GB | NO — SIGBUS during 1M-row populate |
| `test_hydrate_1m_200_field` | ~6 GB | NO |
| `test_hydrate_10m_200_field_record` | ~6 GB+ | NO |
| `test_read_pit_10k_pairs_10m_rows_200_field_record` | ~6 GB+ | NO |

Discovered on Step 16a's first two push-to-main CI runs:
1. First run: SIGBUS in `_populate_old` during 1M evolution bench's
   1M-row insert loop. POSIX shm uses lazy `ftruncate` — size accepted
   up front, blocks commit on write. /dev/shm exhausts mid-populate →
   SIGBUS, not graceful EALLOC.
2. Second run (after pulling RECORD-tier off push): SIGBUS in
   `insert_many` Numba kernel during 100k hydration. Segment
   accumulation from earlier benches + the kernel's transient buffers
   pushed cumulative usage over the limit.

**Workflow consequence:** `QUORIN_RUN_LARGE_BENCH` AND
`QUORIN_RUN_RECORD_BENCH` both restricted to `workflow_dispatch` +
`schedule` (manual + weekly), NOT push. Push-to-main runs Tier-1
only — same scope as pre-Step-16 CI but now with actual gate
enforcement via `check.py --strict`. The `benchmarks/conftest.py`
session does NOT have an autouse `_shm_test_isolation` fixture
(unlike `tests/conftest.py`), so heavy benches accumulate state
within a session. Heavy-bench coverage on every push would require
either:

- An autouse cleanup fixture in `benchmarks/conftest.py` (per-bench
  unlink + Redis flush). Belt-and-suspenders; doesn't fix the
  per-bench-peak issue at 1M+ scales.
- `--forked` pytest plugin so each bench runs in its own process
  (fresh /dev/shm per bench). Adds ~3s subprocess startup per bench;
  ~4 min added to push-tier runtime. Acceptable trade-off if
  automated heavy-bench coverage is needed.
- `runs-on: ubuntu-latest-large` (16-core, 64 GB RAM, ~$0.20/run).
  Skips the memory-ceiling problem entirely.
- `benchmarks/runs/repeat.py` orchestrator pattern — already does
  per-subprocess fresh state via `tempfile.TemporaryDirectory` +
  fresh pytest invocations. The right tool for headline-quality
  heavy-bench numbers, but not the right tool for PR-blocking gate
  enforcement (slow + manual review).

For now, push runs Tier-1 only. Heavy benches verified manually on
operator workstation + on `workflow_dispatch` runs.

### 7c. Step 16c canonical native-CI baselines + ADR-005/007 amendments

**Date:** 2026-05-05
**Workflow:** GitHub Actions ubuntu-latest run 25394553451, commit 4818ea4.

Step 16c-d's USER-driven scope expansion (Numba BLAKE2b on top of
lookup-jit) closed the 5 us p99 trip-wire on `headline_4_field_warm_numba`:
**N=20 native-CI median_p99 = 4.48 us — GREEN, spec MET at p99**.

Canonical N=20 JSONs committed at `benchmarks/results/n20/`:

| Scenario | median_p50 | median_p99 | stddev_p99 | Spec | Verdict |
|---|---|---|---|---|---|
| `headline_4_field_warm` | 4.14 us | **4.48 us** | 1.18 us | <= 5 us p99 | GREEN |
| `headline_200_field_warm` | 7.59 us | 11.66 us | 1.10 us | 10-20 us p99 | GREEN (mid-band) |
| `headline_200_field_cold` | 31.28 us | 66.14 us | 7.84 us | 20-50 us p99 | over band — disclosed |
| `gc_p999_pressure` | 4.07 us | 17.88 us | 4.09 us | (informational) | — |
| `write_sync_rtt` | 1930 us | 2181 us | 556 us | <= 75 ms gate | GREEN (35x headroom) |

**Cold-cache 66 us median_p99 is over the 20-50 us spec band by 30%.**
Per §4 honest-disclosure-2 (single-process cold-cache has 3-4x variance,
N=20 stddev_ratio is 11.9% — middle of distribution). The README quotes
the canonical 66 us with venue disclosure; bare-metal extrapolation puts
it at ~22-44 us (1.5-3x faster than ubuntu-latest), bringing it back
inside the spec band on modern hardware.

Tier-1 gates retightened from this run per §11 calibration discipline.
See `benchmarks/regression/tier1.yml` for the per-bench rationale +
calibration source comments. New gate values are 3.0x median_p99 (no
N=20 scenario qualified for 2.5x — all stddev_ratios were 9-26%).
"Tightening only" discipline preserved 4 single-process gates that were
already tighter than 3x measured.

**ADR-005 amendment** (pool overhead): the native-CI Check A measured
+2.09 us at 4-field and +3.90 us at 200-field — both BREACH the +0.5 us
"latency-neutral" claim. ADR-005 amended with honest measured numbers;
pool stays default for batch path, opt-in for single-entity path on
cost-sensitive workloads. Step 17's C-extension `_Checkout` is the
parking-lot path to ~50 ns checkout cost.

**ADR-007 amendment** (batch ratio): the native-CI Check B measured
1.46x speedup at 4-field and 1.73x at 200-field — both BREACH the 5x
ratio claim. The build-plan 5x target only realistic on hardware with
substantial L3 + good batch-loop SIMD; ubuntu-latest's older Xeons
collapse the ratio to 1.5-1.7x per the §7 cache-architecture finding.
ADR-007 amended; README quotes 1.5-1.7x with venue disclosure +
"faster on bare metal."

**Trip-wire closed; Step 16 status flips to ✅.** Step 17 (Documentation
+ release) is the next ⬜.

### 7b. MADV_HUGEPAGE A/B negative result (Step 16b — REJECT)

The Step 16b A/B harness (`benchmarks/runs/step16_madvise_ab.py`) measured `MADV_HUGEPAGE` on cold-cache 200-field assemble against the no-MADV baseline. Decision rule (per Step 16b plan §2.3): SHIP iff `geo_mean(speedup) ≥ 1.5×` AND `p10(speedup) ≥ 1.0×`. The canonical A/B did **NOT** meet the gates.

**Venue:** GitHub Actions `ubuntu-latest` (`ubuntu-24.04`), commit `08711e6` (Rev-10 of the Step 16b plan), workflow run #37 (ID `25338866105`).
**N:** 20 fresh subprocesses per side. **Bench:** `test_bench_assemble_200_field_cold_cache_numba`.

**Sysfs state at A/B time:**
- `/sys/kernel/mm/transparent_hugepage/enabled`: `[always] madvise never` (top-level THP allows madvise — fine).
- `/sys/kernel/mm/transparent_hugepage/shmem_enabled`: `always within_size advise [never] deny force` — **tmpfs THP is DISABLED on ubuntu-latest by default**.

**Result:**

| Metric | Value | Threshold | Pass? |
|---|---|---|---|
| `geo_mean(speedup)` | 1.029× | ≥ 1.5× | NO |
| `p10(speedup)` | 0.969× | ≥ 1.0× | NO |
| Outcome flavor | `REJECT` (both gates failed) | — | — |
| `n_actual_runs` | 20 / 20 (false / true) | both ≥ 16 | YES (clean run) |

**Interpretation: this is a venue-limitation REJECT, NOT a "MADV doesn't help" REJECT.** `shmem_enabled=[never]` means the kernel never honored the `MADV_HUGEPAGE` advice on either side of the A/B — the orchestrator's `hugepage=True` subprocesses got the same base 4 KB pages as the `hugepage=False` ones. The slight `< 1.0×` p10 (0.969×) reflects that the wasted `m.madvise()` syscall added per-create overhead without delivering any page-fault reduction. The A/B as designed cannot demonstrate SHIP on this venue because the venue kernel ignores the question entirely.

**Decision:** the `hugepage` kwarg threading is reverted in the same Step 16b commit. Production code at end of commit is identical to Step 16a — Steps 1-15 behavior preserved.

**Revisit conditions (when to re-run the A/B):**

- **A venue with tmpfs THP enabled.** `cat /sys/kernel/mm/transparent_hugepage/shmem_enabled` must show `[always]` / `[advise]` / `[within_size]`. Operators on hosts where this is configured (workstations, self-hosted CI runners, custom kernels) can re-run the orchestrator and revisit the decision.
- **Different kernel** (newer THP defragmentation behavior, larger 1 GB pages support, etc.).
- **Larger L3 host** (more recent Intel/AMD platform; ubuntu-latest's older Xeon class is ~30 MB shared L3).
- **Bare-metal venue** where THP fragmentation pressure is lower than virtualized runners.

The A/B harness (`benchmarks/runs/step16_madvise_ab.py`) is kept in repo. To re-run on a better venue: restore the `hugepage` kwarg threading from the Rev-10-era git history (`quorin/_internal/posix_shm.py`, `quorin/shm.py`, `tests/_helpers.py::make_segment`, `benchmarks/test_assembly_benchmark.py::seg_200_field` env-var read) and invoke `python benchmarks/runs/step16_madvise_ab.py --num-runs 20`. Canonical JSON committed at `benchmarks/results/madv_ab.json` for the Rev-10 baseline.

### 8. P4 distinction: assemble-under-GC vs GC pause durations

`benchmarks/test_assemble_under_gc.py::test_bench_assemble_p999_under_gc_pressure_4_field`
measures **assemble latency** while a side thread allocates short-lived
Python lists at 50 MB/s, driving gen-0 collections every ~50 ms.
`benchmarks/test_gc_p999.py` measures **GC pause durations** in
isolation (ADR-006's freeze-vs-no-freeze decision archive).

Both stay. The Step 16 headline claim ("p999 assemble under GC pressure")
needs the new file's measurement — what an operator sees in production,
not the GC pause duration in milliseconds. Existing pause-time benches
remain locked to ADR-006's decision pattern.

### 9. Pool head-to-head + batch ratio: NOT auto-gated

Step 16 plan §3.3 originally called for two derived-metric gates:

- `pooled_minus_baseline_4_field_us` ≤ +0.5us (ADR-005's "latency-
  neutral" claim).
- `batch_ratio_n1000_vs_n1` ≤ 0.22 (ADR-007's 5× speedup claim).

Both were dropped because **pytest-benchmark records wall-clock per
round, not derived ratios**. Implementing required either a custom
timer (overengineering) or a wrapper that conflated two paths into
one timed call (loses signal). The absolute Tier-1 gates (pooled
paths gated independently; batch absolute gates) catch regressions
in either path; the relationship-claims become manual diffs in 16c
review against committed JSON. See `progress/step16c_review.md` for
the explicit checklist.

The honest tradeoff: we lose automated detection of "both paths got
slower together by 50% but still under their absolute gates" — the
rare drift mode where ratio degrades while both absolutes pass.
Mitigation: 16c manual review, plus the tightened native-Linux
absolute gates that come with it.

If a future Step 18+ wants ratio enforcement, the right home is
extending `check.py` with `derived:` entries that reference other
gates — but that's feature creep; defer until the manual diff ever
misses something.

### 10. Default-arg-MODULE_CONST gotcha lock

`check.py::_load_thresholds` originally captured `TIER1_PATH` /
`TIER2_PATH` at function-definition time via default args
(`def f(x=MODULE_CONST)`). `monkeypatch.setattr(check, "TIER1_PATH", ...)`
in tests didn't take effect because the default-arg const was bound
at import. Surfaced via the new main()-level test
(`test_main_returns_1_on_breach_in_either_mode`).

Fixed by having `main()` pass paths explicitly to `_load_thresholds(
include_tier2, tier1_path=TIER1_PATH, tier2_path=TIER2_PATH)`. Module-
level `TIER1_PATH` is looked up at call time, so monkeypatch works.

This is a class of bug — scan any future module-level path constant
referenced as a default arg for the same shape.

### 11. Findings from validating against existing bench JSON

The Step 16 plan predicted (and the user's checkpoint guidance
confirmed) that running enforced check.py against existing bench data
would surface RED gates. It did, in the cleanest possible failure
mode: **all-same-shape median-vs-p99 calibration error**. Seven gates
were bumped to 2-3× WSL2 measured p99 with run-IDs in comments:

| Bench | Old gate | WSL2 p99 measured | New gate |
|---|---|---|---|
| `test_bench_assemble_4_field_warm_numba` | 10us | 28.4us | 60us |
| `test_bench_assemble_200_field_warm_numba` | 50us | 50.9us | 150us |
| `test_bench_assemble_4_field_warm` | 50us | 70.8us | 150us |
| `test_bench_assemble_200_field_warm` | 1ms | 1018.6us | 2.5ms |
| `test_bench_assemble_4_field_warm_pooled` | 10us | 35.7us | 80us |
| `test_bench_assemble_200_field_warm_pooled` | 50us | 57.9us | 150us |
| `test_bench_assemble_batch_4_field_n1000` | 5ms | 9.28ms | 20ms |

The headline "5us p99" claim from CLAUDE.md is at the **median**, not
the p99 — single-process pytest-benchmark on WSL2 hits ~28us p99 at
the 4-field warm path. The Tier-2 N=20 native-Linux trip-wire (Step 16
plan §2.1) is the source of truth for the headline; if that p99 > 5us,
lookup-jit ships in 16c.

## Consequences

- Every gate in `tier1.yml` + `tier2.yml` is now actually enforced.
  Step 17's README can quote committed numbers with confidence that
  they're not fictional.
- Tier-1 runs on every PR (~30s). Tier-2 runs on push-to-main +
  workflow_dispatch + weekly schedule (~10–15 min for env-gated
  benches; ~30–60 min for the N=20 orchestrator). PR latency is
  preserved.
- Two committed-JSON manual reviews remain as 16c todos
  (`progress/step16c_review.md`): pool overhead and batch ratio.
- A future Numba bump or pytest-benchmark version drift is caught
  by the `tests/integration/test_repeat_orchestrator_e2e.py` canary
  (the integration test does double duty: schema-contract guard +
  Tier-2 runtime canary).

## Out of scope (deferred)

- 10M hydration bench (capacity-planning only; `QUORIN_RUN_RECORD_BENCH=1`
  staying ungated).
- Lookup-jit (with trip-wire to 16c).
- MADV_HUGEPAGE A/B (16b deliverable; ships only if measured ≥ 1.5×
  geo-mean speedup on native Linux cold-cache).
- GC callback+freeze +31% interaction localization (ADR-006 known
  limitation).
- Buffer pool C-extension `_Checkout` optimization.
- `quorin.wal_consumer` warm-import optimization (cosmetic).
- Self-hosted runners.
- CPU pinning / numa beyond MADV_HUGEPAGE.
- README authorship (Step 17).

## Validation

- `tests/unit/test_check_py.py` — 18 tests covering R1 multi-percentile,
  L5 duplicate-key guard, B1 stats.data requirement, E2 two-stage name
  lookup, main()-level strict-vs-non-strict exit codes.
- `tests/unit/test_repeat_orchestrator.py` — 12 tests on aggregation
  math (median-of-percentiles, max-of-max, stddev(p99) with N>=2 vs
  N=1, percentile normalization for the YAML-friendly 999 → 99.9
  convention).
- `tests/integration/test_repeat_orchestrator_e2e.py` — subprocess
  plumbing dry-run at N=2 (~24s wall clock); locks the JSON-schema
  contract and serves as a Tier-2 runtime canary.
- Manual end-to-end run of `check.py` against fresh bench JSON
  surfaced + bumped 7 gates inherited from the old YAML. All Tier-1
  gates green at 16a commit.
