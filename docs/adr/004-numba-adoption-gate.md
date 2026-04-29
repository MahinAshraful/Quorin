# ADR-004: Numba assembly is gated on a per-schema benchmark threshold

**Status:** Accepted
**Date:** 2026-04-28
**Step:** 5 (Numba assembly path)

## Decision

Pyforge ships **two** assembly implementations: the pure-Python oracle
`pyforge.serving.assemble` (Step 4) and the Numba-compiled
`pyforge.assembly.assemble` (Step 5). They produce **byte-identical output**,
verified by `tests/property/test_assembly_parity.py` over 200 Hypothesis-
generated random schemas. Adoption of the Numba path as the production default
is **gated on the 200-field benchmark showing ≥3× speedup** over the Python
baseline; this ADR records the measured ratio. Either way, no Step-5
public-API change — users explicitly select by import (`from pyforge.serving
import assemble` for Python, `from pyforge.assembly import assemble` for
Numba). Step 12 wires the unified production entry once benchmark numbers
across the build are stable.

## Context

The build plan's original phrasing ("replace the Python loop with Numba")
treats Numba adoption as unconditional. That is dogma, not engineering. NumPy's
slice-assign is already a single `memcpy` for byte-identical-dtype copies; the
gain from Numba is removing per-field Python-interpreter overhead (dict
lookup for the cast, refcount, GIL ops). Estimated crossover is around 20
fields. Below that, a Python-FFI roundtrip into the JIT'd function may cost
more than it saves. Above that, Numba dominates: a 200-field schema with one
128-dim embedding is ~10-20× faster on the JIT path because the Python loop's
per-iteration overhead compounds.

The cost of carrying both paths is one extra ~100-line module
(`pyforge/assembly.py`), one parity test, and a 5-line ADR — finite and bounded.
The cost of carrying *only* Numba and discovering that small-schema latency
got worse is a cross-step bug hunt and an unwind. Both paths is the cheap
insurance, and the parity test ensures they cannot silently diverge.

## The design

- **`pyforge/assembly.py` is isolated.** `pyforge.serving` does not import it,
  so `import pyforge.serving` does not pay Numba's ~200 ms LLVM init or
  compilation cost. Only callers that explicitly `import pyforge.assembly`
  trigger the toolchain.
- **Single uint8 segment view + Numba `.view(dtype)` per field.** One
  `np.frombuffer(seg.handle.buf, dtype=np.uint8)` per assemble call. Inside
  the @njit kernel, each field's slice is reinterpreted via
  `.view(np.float32)` etc. — supported on contiguous 1D arrays in nopython
  mode. Alignment-safe by construction: every field offset is 64-byte
  cache-line aligned (Step 1 invariant).
- **`fastmath=False` is non-negotiable.** Fastmath licenses LLVM to assume no
  NaN/inf, which would break NaN bit-pattern parity with the Python oracle.
  Memory-bound assembly wouldn't benefit from fastmath anyway.
- **Explicit Numba signature** for fast cold compile and a single
  specialization. `[::1]` (C-contiguous) gives Numba freedom to vectorize
  inner loops.
- **`SegmentLayout` carries flat parallel arrays** (`assembly_byte_offsets`,
  `assembly_byte_counts`, `assembly_dtype_codes`, `assembly_element_counts`)
  pre-cast to `int64`/`uint8` at `compute_layout` time. The hot path doesn't
  pay `.astype()` overhead — that ~800 ns is 16% of the 4-field budget.

## Consequences

- **Positive:** the parity test pins the Numba kernel against the Python
  oracle. Any divergence (silent type promotion, ordering bug, NaN
  mishandling) fails the build immediately.
- **Positive:** Python users who don't want a Numba dependency at runtime can
  use `pyforge.serving.assemble` exclusively; Numba never imports.
- **Positive:** an honest crossover point can be measured and documented per
  schema. Step 12's public API can dispatch based on schema size, not on
  blind dogma.
- **Negative:** every reviewer of new assembly code must remember there are
  two paths. The naming carries the intent (`pyforge.serving` vs
  `pyforge.assembly`), and the parity test is the safety net.
- **Negative:** `boundscheck=False` in production speed mode hides
  out-of-bounds writes. The parity test (correctness check) doesn't run with
  bounds checking, but the test suite includes a separately-jitted
  `boundscheck=True` variant for confidence.

## The 4-field schema budget

The build plan targets ≤5 µs p99 for warm 4-field assemble. Step 5's
threshold is **10 µs (2× headroom)** because Step 5 only Numba-fies assembly,
not lookup. `pyforge.layout.lookup` remains pure-Python at ~3 µs/call —
dominating the 4-field budget. Numba-fying lookup is deferred to a later
step; it would close the gap. For 200-field schemas, lookup overhead is a
small fraction of total cost and Numba's gains dominate.

## Validation

`tests/property/test_assembly_parity.py` is the binary agreement check
between the two paths over Hypothesis-generated schemas. The benchmark
ratio (Numba vs Python on the 200-field warm scenario) is recorded in
`progress/progress.md` alongside the Step 5 completion entry.

### Measured numbers (post-Step-5 benchmark, WSL2 Ubuntu)

| Scenario | Python p99 (median) | Numba p99 (median) | Ratio |
|----------|---------------------|---------------------|-------|
| 4-field warm  | 11.94 us | 4.17 us | 2.86x |
| 200-field warm + 128-emb | 372.55 us | 5.37 us | **69.4x** |
| Lookup-miss path | 2.43 us | 2.45 us | 1.00x (parity) |

**Decision: Numba is ADOPTED as the production default.** The 200-field
warm ratio (69.4x) clears the 3x gate by an order of magnitude. Two
data points worth flagging:

- **4-field warm Numba is 4.17 us** — under the spec's headline 5 us
  target, without Numba-fying lookup. The 2.86x ratio at 4 fields is
  below the gate, but the gate is specifically for 200-field; small
  schemas were never the expected win case (Python's slice-assign is
  already a `memcpy`).
- **The Numba kernel is essentially flat** across [4, 10, 20, 50, 100,
  200] fields (4.17 us -> 5.30 us). The marginal cost per field is
  ~5 ns. Lookup overhead (~3 us Python) dominates at small N; the
  embedding bytes dominate at large N.

Step 12 will wire `pyforge.assembly.assemble` as the default for
`pyforge.serving.assemble` when the production API stabilizes. Until
then, both paths remain importable and the parity test guards against
divergence.

## References

- Pyforge spec, Step 5 section ("⚠️ PLAN CORRECTION: Benchmark gate on
  adoption").
- Build plan estimate: ~20-field crossover; 4-field 5 µs vs 1 µs (20% gain
  too small); 200-field 400 µs vs 20 µs (20× gain).
- Numba `.view(dtype)` support: documented in nopython mode for contiguous
  arrays since 0.39.
- ADR-003 on declaration-order output (the ordering Numba must honor).
