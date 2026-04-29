# ADR-007: Batch assembly — fused Numba kernel + separate `BatchBufferPool`

**Status:** Accepted
**Date:** 2026-04-29
**Step:** 8 (Batch assembly)

## Decision

Pyforge ships `pyforge.assembly.assemble_batch(seg, entity_ids, *, out, found_mask)`
returning a `(N, total_element_count)` float32 buffer plus an `(N,)` bool mask.
Implementation is:

1. **Single fused Numba kernel** `_assemble_batch_core(...)` that does both
   the per-entity slot-table probe and the per-row feature assembly in one
   pass. Avoids an intermediate `int64[N]` row-offsets allocation, one
   Numba-dispatch overhead, and keeps `row_offset` and the found-flag in
   registers across the per-row work.
2. **Raw `uint8[::1]` segment view + scalar slot-byte-layout constants**
   (`SLOT_BYTES`, `SLOT_NAME_HASH_OFFSET`, `SLOT_ID_OFFSET_OFFSET`,
   `SLOT_FLAGS_OFFSET`, `SLOT_FEATURE_ROW_INDEX_OFFSET`) passed as kernel
   args. Avoids both the AoS-stride trap (structured-dtype field views are
   non-contiguous and violate `[::1]`) and Numba 0.60's fragile structured-
   field-key support.
3. **Slot-byte constants are derived from `SLOT_DTYPE.fields` at module load**
   in `pyforge/layout.py` and pinned via runtime asserts. A future dtype
   reorder fails at import time, not silently inside a kernel returning
   wrong rows. Hardcoded magic offsets in the kernel are forbidden — the
   failure mode (incorrect row data, no exception) is exactly the kind of
   bug parity tests catch slowly.
4. **Miss rows are zero-filled in-kernel.** Per-row branch: hit -> assembly
   ladder, miss -> zero-fill. Bandwidth-equivalent to a bulk memset because
   we walk every row anyway.
5. **Python-side ID prep stays in Python.** Strings + blake2b are not in
   Numba (invariant #5 pins the hash algorithm). The wrapper UTF-8 encodes
   each ID, blake2b-hashes via `pyforge._internal.hash_id.hash_entity_id`,
   and packs query bytes into a padded `(N, max_query_len)` uint8 array
   for the kernel.
6. **Separate `BatchBufferPool` class** in `pyforge/pool.py`, NOT an
   extension of Step 6's `BufferPool`. Different defaults
   (`max_size=64`, `zero_on_return=False`, `batch_size` required), 2D
   output instead of 1D, different memory-budget shape. Forcing one class
   would require default-flipping on a parameterized constructor and
   create a footgun about 1D vs 2D output.
7. **No module-level pool registry.** YAGNI for Step 8's API
   (`assemble_batch(out=pool.checkout())`). Globals add test-cleanup,
   thread-safety, and schema-evolution surface. Step 12 is the right
   place if it's still needed.
8. **Single contiguous slab pre-allocation** in `BatchBufferPool.__init__`:
   one `np.zeros((max_size, batch_size, element_count), dtype=np.float32)`
   sliced into the deque. One allocator call instead of `max_size`,
   lower allocator-metadata overhead, future huge-page coalescing
   friendly. (Not for cross-checkout cache locality — at batch >= ~256
   each buffer exceeds L2 and is cold on every checkout regardless.)
9. **Class-based `_BatchCheckout` with `__slots__`** mirroring Step 6's
   `_Checkout`. Same `@contextlib.contextmanager` lesson — generator-
   protocol overhead (~700 ns/call) is the difference between net-positive
   and net-negative on a sub-microsecond hot path. Two classes share an
   implementation but are not related by inheritance — their abstractions
   are different beasts.
10. **`prewarm()` extends to compile both kernels.** Single opt-in entry
    point; callers don't need to know about the kernel split. Module
    import does not auto-warm (preserves cheap `import pyforge.assembly`).

## Context

Step 5 ships single-entity Numba assembly at ~5.37 µs p99 for a 200-field
schema. Real ML serving workloads (recsys retrieval, batch scoring) request
hundreds-to-thousands of entities per call; looping `assemble()` N times
pays Python-call overhead per row.

The build plan calls out:

> `get_batch(schema, entity_ids, output=None)` is the public API. Pass in
> an `output` buffer from the pool or let the function allocate. Missing
> entities return a sentinel row, with a boolean mask returned alongside
> indicating which entities were found.

Acceptance criterion: **batch >= 5x faster than N single calls at N=1000.**

## The pool: separate class vs extending `BufferPool`

Decision: separate `BatchBufferPool`. The two pools are 80% the same code
but different beasts:

| Aspect | `BufferPool` (Step 6) | `BatchBufferPool` (Step 8) |
|---|---|---|
| Output shape | `(n,)` 1D | `(batch_size, n)` 2D |
| Buffer size | ~5 KB | ~800 KB at batch=1000 |
| `fill(0)` cost on return | ~30 ns | ~50-100 us |
| Right `zero_on_return` default | `True` (cheap insurance) | `False` (cost too high) |
| Right `max_size` default | 128 | 64 (memory budget) |
| Pool wins net (latency)? | No (-450 ns) | Yes (+4-50 us) |

The defaults need to differ because the cost structure differs.
`zero_on_return=True` on a single-row pool is ~30 ns and saves you from
data-leak bugs. On batch, it's ~50-100 us — enough to erase the entire
pool win on the very call that should benefit most. Different class lets
each pick the right default without surprising users.

Rejected alternatives:

- **Extend `BufferPool` with optional `batch_size`.** The "is-a"
  relationship doesn't hold — a single-row pool is not "a batch pool with
  batch_size=1, dimension squeezed." They return different shapes.
  Forcing one class means callers can't tell from `pool.checkout()`
  whether they're getting a vector or a matrix without checking how the
  pool was constructed. Footgun. Also, default-flipping in a parameterized
  class is ugly: hard-code the dependency or always pass it (defeats the
  "simpler" argument).

- **`BatchBufferPool(BufferPool)` inheritance.** The protocol is
  ~15 lines of `__enter__` / `__exit__`; copying it is cleaner than
  inheriting the wrong abstraction.

- **Module-level `buffer_pools[(schema, batch_size)]` registry.** YAGNI
  for Step 8. Globals add test-cleanup, thread-safety on get-or-create,
  memory growth as different batch sizes accumulate, and schema-evolution
  interaction (Step 15). Step 12 is the right place if it's still needed.

## Default `zero_on_return=False` safety contract

Safe for `assemble_batch` because the kernel **guarantees full-buffer
overwrite**: every row is either filled with feature data (hit path) or
explicitly zeroed (miss path). Prior buffer contents are never read by the
consumer.

**Load-bearing invariant:** if the kernel ever changes such that a row
could be partially written, this default becomes a data-leak vector. The
`BatchBufferPool` docstring documents this; if you reuse pool buffers for
any other purpose (debug print, alternate kernel), set
`zero_on_return=True` to defend against prior-data leakage.

## Kernel design — fused vs split

Considered both. **Fused wins** because:

- Avoids an intermediate `int64[N]` row-offsets allocation. At N=10000
  that's 80 KB of malloc + free per call, real cost.
- One Numba dispatch instead of two — saves ~µs of dispatch overhead.
- Keeps `row_offset` and the found-flag in CPU registers across the
  per-row work, no roundtrip to memory.
- Only ~30-40 lines longer than two split kernels combined.

The lookup probe is bounded (worst case slot_capacity iterations); the
assembly ladder is byte-identical in code shape to `_assemble_core`'s,
which means parity to the Python oracle is structural, not testimonial.

## Hash collisions

Hypothesis with random IDs hits collision probability ~1/2^64 — would
never naturally exercise the kernel's probe + byte-compare path. Step 8
includes a dedicated unit test
(`tests/unit/test_batch_assembly.py::test_hash_collision_probing_path`)
that monkey-patches `hash_entity_id` to return a constant, inserts 4
colliding entities, and asserts the kernel resolves them via byte
comparison. Mirrors the existing pattern in
`tests/unit/test_layout.py::test_hash_collision_path`.

## The 5x gate — what we shipped

### Initial measurement (serial-only)

Back-of-envelope at 200 fields predicted ~3.5-4.5x. Measured (WSL2 Ubuntu):

| Schema | N=1000 batch median | N x single median | Ratio |
|---|---|---|---|
| 4-field | 1,353 µs | 4,728 µs | **3.49x** |
| 200-field | 2,632 µs | 6,150 µs | **2.34x** |

Per-entity batch cost was stable across N (~1.35 µs at 4-field, ~2.6 µs at
200-field), confirming Python prep + memcpy dominate. Both missed the 5x
gate honestly.

### Tier 3: parallel kernel + branched dispatch + ADAPTIVE threshold

Single-core 200-field assembly is memory-bandwidth-bound at ~10 GB/s; aggregate
machine bandwidth on a multi-core box is 30-50 GB/s. With 800 KB writes per
batch=1000 we're nowhere near saturation on one core. `@numba.njit(parallel=True)`
+ `prange` over the outer batch dimension is the natural lever: per-row
writes target disjoint `out[b, :]` and `found_mask[b]` slices, so the loop
parallelizes with no shared state, no atomics, no locks.

The catch: parallel kernels pay thread-pool spinup cost (~10s of µs) on every
call regardless of batch size. For small batches the serial kernel wins.
**Branched dispatch** at the wrapper level resolves the small-N case:

```python
kernel = (
    _assemble_batch_core_parallel if n_batch >= PARALLEL_THRESHOLD
    else _assemble_batch_core
)
```

But there's a second, larger catch: **the optimal `PARALLEL_THRESHOLD` is
strongly thread-count-dependent**. The sweep on WSL2 with the 200-field
schema (327 elements/row) produced this calibration table:

| `NUMBA_NUM_THREADS` | N=1 | N=10 | N=32 | N=64 | N=128 | N=256 | N=1000 | N=10000 |
|---|---|---|---|---|---|---|---|---|
| **default (~8)** serial µs | 6.1 | 29.3 | 86.8 | 169.1 | 335.0 | 671.9 | 2682.2 | 29352.4 |
| **default (~8)** parallel µs | 95.7 | 32.0 | 205.0 | 254.5 | 433.6 | 797.2 | 3390.3 | 23217.4 |
| **default (~8)** speedup | **0.06x** | 0.91x | **0.42x** | **0.66x** | **0.77x** | **0.84x** | **0.79x** | **1.26x** |
| **2** speedup | 0.80x | 1.16x | 0.98x | 1.34x | 1.25x | 1.35x | 1.43x | 0.97x |
| **4** speedup | 1.26x | 1.01x | 1.30x | 1.13x | **1.61x** | 1.49x | 1.14x | 1.47x |

**The default-thread-count regression is the killer.** With Numba's default
thread count (`os.cpu_count()`, often 8-16 in production), the parallel kernel
*regresses* vs serial for every N from 1 to 1000. A static low threshold
(say, 64) would silently degrade performance for the most common deployment
mode.

**Solution: adaptive threshold computed at module load.**

```python
def _compute_parallel_threshold() -> int:
    n = numba.get_num_threads()
    if n <= 1:
        return sys.maxsize  # never parallel; serial-only
    if n <= 4:
        return 64           # parallel wins at N >= 64 with 2-4 threads
    return 2048             # high-thread; only large batches amortize spinup

PARALLEL_THRESHOLD: int = _compute_parallel_threshold()
```

Thread counts of 2-4 (deliberately constrained for batch-bound workloads)
get the meaningful wins. Default-thread deployments use the parallel kernel
only for very large batches where the win is robust. Single-thread machines
never use it.

**Recommended deployment**: set `NUMBA_NUM_THREADS=4` in the environment
before importing `pyforge`. With 4 threads:
- parallel wins at every N >= 32 (best at N=128: 1.61x)
- 200-field N=1000 batch: ~2327 µs vs 2940 µs serial (1.26x kernel speedup)
- combined with ~6150 µs single-call comparator: **~2.64x** end-to-end ratio

### Tiered optimization summary

| Tier | Optimization | Status |
|---|---|---|
| 1 | ASCII-fast-path encode (`try id.encode("ascii"); except UnicodeEncodeError: id.encode("utf-8")`) | **Shipped.** ~50-80 µs win at N=1000. ~70%+ of production ML entity IDs are ASCII. Zero new dependencies. |
| 2 | Concatenated bytes + offsets array instead of padded 2D | **Skipped.** ~50-100 µs marginal win, more complex kernel indexing — not worth the maintenance cost. |
| 3 | `@numba.njit(parallel=True)` + `prange` + branched dispatch | **Shipped.** Numba ships its own threading layer (no new pip dep). Two kernels with byte-identical bodies (`_assemble_batch_core`, `_assemble_batch_core_parallel`); per-row writes are disjoint per `b`, so trivially parallelizable. |

### Two-kernel maintenance contract

The serial and parallel kernels MUST stay byte-identical in their per-row
behavior. Any correctness fix to one MUST be replicated in the other. Two
unit tests in
[`tests/unit/test_batch_assembly.py`](../../tests/unit/test_batch_assembly.py)
monkey-patch `PARALLEL_THRESHOLD` to force each kernel and assert
`np.array_equal` of outputs:

- `test_parallel_kernel_parity_to_serial`
- `test_parallel_kernel_handles_misses`

The property tests in
[`tests/property/test_batch_parity.py`](../../tests/property/test_batch_parity.py)
do not naturally exercise the parallel kernel because their batch sizes
(1-12) stay below `PARALLEL_THRESHOLD`. The parallel-path coverage comes
exclusively from the two unit tests above. If `PARALLEL_THRESHOLD` is ever
tuned upward, these tests must remain at a forced batch size that exceeds
the new threshold.

### Free-threaded CPython caveat (deferred)

PEP 703's free-threaded interpreter (Python 3.13+ experimental, 3.14+
stable) removes the GIL. Numba's parallel mode currently assumes the GIL
is acquired by the calling thread; behavior under free-threaded CPython
is unverified. Pyforge's deployment target is 3.12 (pinned in
`.python-version`), so this is a 2027+ concern. Document and revisit at
that point — not avoid the optimization now.

### Final ratios — measured

At NUMBA_NUM_THREADS=4 (recommended deployment), 200-field N=1000:
- Single-entity Numba comparator: 6,124 µs measured
- Parallel batch kernel: 1,936 µs measured
- **Ratio: 3.16x** (vs build-plan target 5x)

At default thread count (8), 200-field N=1000:
- Single-entity Numba comparator: 6,486 µs measured
- Serial batch kernel (dispatch picked serial, threshold=2048): 2,872 µs
- **Ratio: 2.26x**

The post-Tier-3 measurement beat my pre-impl estimate (2.64x) — likely because
the in-suite benchmark uses pre-allocated `out=` buffers, eliminating the
allocation cost the standalone sweep included.

**The 5x gate is honestly missed on this hardware.** The bottlenecks:

1. **Python-side prep is a hard floor** (~1 ms at N=1000). blake2b + UTF-8
   encode + tuple builds run at C speed in CPython but the dispatch overhead
   per call sets a floor. Tier 1 (ASCII fast-path) shaved ~50-80 µs; not
   transformative.
2. **Memory bandwidth doesn't scale linearly with thread count.** At 4
   threads we get 1.14-1.61x kernel speedup, not the 2-4x predicted. WSL2
   memory subsystem may saturate at ~2-3 threads on this machine.
3. **Numba thread-pool spinup is non-trivial.** ~10s of µs per call. For
   workloads that aren't 100% in the kernel, this is a real tax.

What would push it to 5x (deferred to Step 16):
- **Numba-fied lookup**: replace the ~3 µs/call Python lookup wrapper that
  still dominates the single-call comparator. If the comparator drops from
  ~6 µs/call to ~3 µs/call, the batch ratio collapses (denominator shrinks).
  But that's a Step 16 optimization that improves ALL paths, not a batch-
  specific gain.
- **C-extension blake2b batch**: eliminate the 1 ms Python prep floor.
  Days of work, no telemetry justification yet.
- **SIMD via SVML or vectorized intrinsics inside the assembly ladder**:
  uses AVX-512 or NEON depending on hardware. Numba supports this via
  `fastmath`-equivalent flags but invariant #12 forbids fastmath.

### Echoes the Step 7 lesson

Throughput / tail-latency claims need real measurement, not estimates. The
actual measured ratio (2.34-2.64x) is honestly less than the 5x build-plan
estimate. The threshold sweep data lives in
`benchmarks/results/step8_threshold_sweep.txt` for reproducibility. The
batch is still meaningfully faster than naive single-call (real-world win),
just not the headline-paper number.

## Files affected

- `pyforge/assembly.py` — added `_assemble_batch_core` (serial Numba),
  `_assemble_batch_core_parallel` (parallel + prange), and `assemble_batch`
  (Python wrapper with ASCII fast-path encode + branched dispatch on
  `PARALLEL_THRESHOLD`). Extended `prewarm()` to compile all three Numba
  kernels.
- `benchmarks/runs/step8_threshold_sweep.py` — fresh-subprocess sweep
  orchestrator that forces each kernel via `PARALLEL_THRESHOLD` monkey-
  patching and tabulates median µs/call across N ∈ {1, 10, 32, 64, 128,
  256, 1000, 10000}. Output committed to
  `benchmarks/results/step8_threshold_sweep.txt`.
- `pyforge/pool.py` — added `_BatchCheckout` and `BatchBufferPool`
  alongside existing `_Checkout` / `BufferPool`.
- `pyforge/layout.py` — added `SLOT_BYTES` and four `SLOT_*_OFFSET`
  constants derived from `SLOT_DTYPE.fields`, with runtime asserts
  pinning the values.
- `tests/unit/test_batch_assembly.py` — 19 unit tests (all 5 dtypes,
  shaped fields, miss zero-fill, mask correctness, empty batch,
  validation, hash-collision probing, slot-constant pinning).
- `tests/unit/test_batch_pool.py` — 16 unit tests (construction,
  validation, checkout shape, slab-aliveness, exhaustion, exception
  safety).
- `tests/property/test_batch_parity.py` — 3 properties x 200 examples
  = 600 random schema/batch pairs; NaN-poisoned out= catches "forgot
  to write index k" bugs.
- `tests/property/test_batch_pool_parity.py` — 2 properties x 100
  examples = 200; pooled vs fresh byte-identical, pool-bounded.
- `tests/integration/test_batch_e2e.py` — 3 e2e tests via real
  `SegmentRegistry` (4-field N=100, 200-field N=1000 with pool, mixed
  hit/miss N=100).
- `benchmarks/test_batch_benchmark.py` — sizes [1, 10, 100, 1000] x
  {4-field, 200-field}, single-call comparator, pooled vs fresh,
  allocation budget.
- `benchmarks/regression/thresholds.yml` — added
  `assemble_batch_4_field_n1000` (5 ms) and
  `assemble_batch_200_field_n1000` (10 ms).

## Consequences

- `pyforge.assembly` now compiles two Numba kernels at import. Cold
  compile cost ~600 ms total (was ~300 ms in Step 5); cached after.
- `pyforge.pool` now exports two pool classes. Public-API consumers
  need to choose `BufferPool` vs `BatchBufferPool` based on shape
  needs.
- The slot-byte-layout constants in `pyforge.layout` become a public
  contract — the kernel reads at fixed offsets, and any future
  `SLOT_DTYPE` reorder must update both the constants and the runtime
  asserts together.
- Step 12's public-API stabilization will decide whether to add a
  module-level `BatchBufferPool` registry. Until then, callers
  construct one pool per `(schema, batch_size)` tuple at startup.
- The fused kernel's two-responsibility shape (lookup + assembly) is a
  trade for performance. If correctness ever drifts and one
  responsibility regresses, the parity test catches it row-by-row but
  the failure surfaces in `assemble_batch` rather than in a
  `lookup_batch` we can't isolate. ~150 LOC kernel; manageable.
