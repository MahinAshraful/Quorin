# ADR-017: Lookup-jit — Numba-compiled single-entity lookup + BLAKE2b

**Status:** Accepted
**Date:** 2026-05-05
**Step:** 16c (Lookup-jit + flamegraphs + Tier-1 native-CI retighten + Numba BLAKE2b for trip-wire closure)

## Decision

Quorin ships a **Numba-compiled single-entity lookup** at
`quorin._internal.lookup_kernel.lookup_jit`, used by
`quorin.assembly.assemble` to resolve `entity_id -> row_offset` on the
serving hot path. The kernel includes a **Numba-compiled BLAKE2b-8** at
`quorin._internal.hash_kernel.blake2b_8` — byte-identical to
`hashlib.blake2b(input, digest_size=8)` per pinned-hash invariant #5.
The pure-Python `quorin.layout.lookup` remains unchanged — it's still
the canonical reader for `quorin.serving.assemble` (the Python oracle
/ parity reference) and any caller that wants to skip Numba init.

The kernel lives in **`quorin/_internal/lookup_kernel.py`**, NOT in
`quorin.layout`, because invariant #11 forbids `quorin.layout` from
pulling Numba (it would force every `quorin.serving` / `quorin.shm` /
test importer to pay ~200 ms LLVM init). Mirrors Step 13's
`quorin/_internal/insert_kernel.py` precedent (the bulk-insert kernel
has the same isolation contract).

Byte-identical-to-Python contract is the load-bearing correctness
property: `lookup_jit(seg, eid) == quorin.layout.lookup(seg, eid)` for
any segment state. Verified by
`tests/property/test_lookup_jit_parity.py` (200 Hypothesis examples per
property, hit + miss paths) and by transitive coverage from
`tests/property/test_assembly_parity.py` (Step 5's load-bearing parity
test, which now exercises lookup_jit through `assemble_numba`).

## Context

Step 16c's trip-wire fired: the n=5 native-CI smoke for
`test_bench_assemble_4_field_warm_numba` measured `median_p99 = 9.24 µs`,
~1.85× over the 5 µs spec headline. Per CLAUDE.md gotcha "lookup is still
~3 µs Python — dominates the 4-field assemble budget," the prescribed
remedy was to Numba-jit the lookup primitive (parking-lot item since
Step 5).

The Step 3 measurement records pure-Python `lookup` at ~2.91 µs hit /
~1.6 µs miss median. Of that 2.91 µs:
- ~1500 ns: `hash_entity_id` (Python `hashlib.blake2b` call).
- ~100 ns: `_slot_table_view` (`np.frombuffer` over memoryview).
- ~500 ns: probe-loop iteration with structured-dtype slot reads.
- ~400 ns: `_read_string_bytes` (struct.unpack_from + slicing + bytes()).
- ~50 ns: bytes equality + return.
- ~360 ns: function call + arg unpacking overhead.

Lookup-jit replaces the latter ~1.4 µs of Python-interpreter work with
a single Numba kernel call (~200 ns kernel + ~150 ns dispatch overhead,
pre-warmed). Hash cost (~1500 ns) is unchanged — the kernel takes a
pre-hashed `uint64` so the wrapper is the only one paying the Python
blake2b cost. Numba-jit'ing blake2b itself is parking-lot work
(`progress/progress.md:2286`), deferred to Step 17 if flamegraphs
surface it as the next bottleneck.

## The design

- **Kernel signature: 13 scalar args + 3 array args.** Mirrors
  `quorin.assembly._assemble_batch_core`'s probe-loop shape — same
  byte-compare disambiguation, same EMPTY-slot stop condition, same
  `slot_capacity`-bounded probe budget. Slot byte offsets passed as
  `int64` scalar args (invariant #15 — no magic offsets in Numba kernels).

- **Returns `int64`, not `Optional[int]`.** -1 sentinel for miss; the
  Python wrapper translates to `None`. Numba doesn't speak Python's
  `None`; integer return is the standard pattern. Mirrors
  `insert_kernel`'s sentinel-return convention.

- **`fastmath=False`** (invariant #12). Lookup does no FP arithmetic, but
  the discipline is uniform across all Quorin Numba kernels.

- **`cache=True`** writes compiled artifact to `__pycache__/`. Subsequent
  process starts skip the ~100-200 ms compile. Same as `_assemble_core`.

- **Wrapper `lookup_jit` does the Python-side prep:**
  1. Empty-string check (raise `ValueError` to match Python semantics).
  2. UTF-8 encode + writable-array materialization
     (`np.frombuffer(bytes, ...).copy()` — Numba 0.60's `uint8[::1]`
     historically requires writable arrays; the `.copy()` is ~80 ns at
     max-64-byte IDs, acceptable overhead).
  3. `hash_entity_id` call (~1500 ns Python blake2b).
  4. Single `np.frombuffer(segment.handle.buf, dtype=np.uint8)` view.
  5. Kernel invocation.
  6. -1 → None translation.

- **Prewarm extension.** `quorin.assembly.prewarm()` calls
  `quorin._internal.lookup_kernel.prewarm()` first so a single
  prewarm covers all assembly + lookup kernels. Module load doesn't
  auto-warm — opt-in only.

## Consequences

- **Positive:** The byte-identical-to-Python contract means the existing
  `tests/property/test_assembly_parity.py` (Step 5) extends transitively
  to cover lookup_jit through the assemble wrapper. If lookup_jit ever
  returns a different row_offset than Python lookup, the assembly parity
  test (200 random scenarios) catches it before any commit lands.

- **Positive:** Module-level isolation preserves invariant #11. The
  module-hygiene test (`tests/unit/test_module_hygiene.py`) confirms
  `quorin.serving`, `quorin.layout`, `quorin.metrics`, etc. don't pull
  Numba via `quorin._internal.lookup_kernel`. Adding lookup_kernel to
  the test's "legitimate Numba" exclusion list (alongside `quorin.assembly`
  and `quorin._internal.insert_kernel`) keeps the contract honest.

- **Positive:** ~1 µs saved per assemble call at the 4-field warm path.
  Projected median: 4.2 µs → ~3.0 µs. Projected p99: 9.24 µs → ~6.5-7.0 µs.

- **Negative (likely-RED on ubuntu-latest):** the projected ~6.5-7.0 µs
  p99 still misses the 5 µs spec on ubuntu-latest by 30-40%. Per ADR-015
  §11's bare-metal extrapolation (CI is 1.5-3× slower than 2024-era
  desktop CPUs), this projects to ~3-4 µs on bare metal — meeting spec.
  The README discloses the venue per ADR-015 §7's discipline. RED is the
  honest engineering response, not a failure: bundling additional
  optimization (C-extension `_Checkout`, Numba-jit blake2b) into 16c
  would expand surface 3-4× and delay trip-wire resolution. Those are
  Step 17 (or a hypothetical "Step 16d") deferrals.

- **Negative:** every reviewer of `quorin.assembly.assemble` must
  remember it now calls into `quorin._internal.lookup_kernel`, not
  `quorin.layout.lookup`. The naming carries the intent
  (`_internal.lookup_kernel`) and the parity test is the safety net.

## The byte-identical contract

`lookup_jit(seg, eid)` returns the same `row_offset` (or `None`) as
`quorin.layout.lookup(seg, eid)` for any:

- valid `entity_id` string (UTF-8 encodable, non-empty, any length);
- segment state (empty, partially populated, capacity-cap-at-50%);
- hash collision arrangement (probed via byte-compare disambiguation).

Two test layers lock the contract:
1. **Property test** (`tests/property/test_lookup_jit_parity.py`) — 200
   Hypothesis examples per property × 2 properties (hit-path + miss-path).
2. **Unit test** (`tests/unit/test_lookup_kernel.py`) — 11 explicit cases
   including a hash-collision case via `monkeypatch.setattr` of
   `hash_entity_id` (Hypothesis won't generate collisions at 2^-64
   probability).

The hash-collision regression test mirrors Step 3's
`test_hash_collision_path_still_finds_correct_entity` — forces two IDs
to share the same hash, asserts both findable via the kernel's
byte-compare path.

## Validation — canonical native-CI numbers (Commit B, run 25394553451)

**Trip-wire ratification: GREEN.** GitHub Actions ubuntu-latest N=20
fresh-subprocess orchestrator on `headline_4_field_warm`:

| Metric | Native CI N=20 | Spec | Verdict |
|---|---|---|---|
| `median_p50` | **4.14 us** | <= 5 us p99 | GREEN at median |
| `median_p99` | **4.48 us** | <= 5 us | **GREEN at p99 — SPEC MET** |
| `median_p999` | 10.19 us | (informational) | — |
| `stddev_p99` | 1.18 us (26%) | (informational) | — |
| `max_of_max` | 56.01 us | (informational) | one OS-scheduler outlier |

The Numba BLAKE2b kernel was the load-bearing change. Pre-Numba-BLAKE2b
projection was ~6.5-7 us p99 (RED). Post-Numba-BLAKE2b actual = 4.48 us
p99 (GREEN with 12% headroom).

**README quotes:** "5 us p99 substantiated on GitHub Actions
ubuntu-latest at 4-field warm-cache assemble (N=20 fresh subprocesses,
2026-05-05). 200-field warm-cache p99 = 11.66 us. Cold-cache 200-field
p99 = 66 us — over the 20-50 us spec band on the older Xeon CPUs, but
within band on modern desktop hardware per ADR-015 §11 bare-metal
extrapolation."

## Validation — WSL2 reference numbers (informational)

WSL2 single-process pytest-benchmark, autouse prewarm fixture (cumulative
post-Step-5 / post-Step-16c / post-Step-16c-d):

WSL2 single-process pytest-benchmark, autouse prewarm fixture (cumulative
post-Step-5 / post-Step-16c / post-Step-16c-d):

| Bench | Step 5 (Python lookup) | Step 16c (lookup-jit + Python blake2b) | Step 16c-d (lookup-jit + Numba blake2b) | Cumulative Δ |
|---|---|---|---|---|
| `lookup_miss` (Python) | 1.59 µs | 1.62 µs | **1.58 µs** | parity (no change) |
| `lookup_jit_miss` | n/a | 2.50 µs | **1.95 µs** | -22% vs 16c |
| `lookup_hit_first_probe` (Python) | 2.95 µs | 2.98 µs | **2.92 µs** | parity (no change) |
| `lookup_jit_hit_first_probe` | n/a | 2.57 µs | **1.98 µs** | -23% vs 16c, **1.47x faster than Python** |
| **`assemble_4_field_warm_numba`** ← TRIP-WIRE | **4.17 µs** | 3.71 µs | **3.15 µs** | **-24% cumulative** |
| `assemble_200_field_warm_numba` | 5.37 µs | n/a | **4.39 µs** | -18% vs Step 5 |

**The Step 16c-d Numba BLAKE2b drop is what closed the trip-wire.** Before
the hash-jit, lookup_jit was paying ~1500 ns per call to Python's
`hashlib.blake2b`. The kernel call itself was already fast (~200 ns); the
hashlib call was the dominant per-call cost and it was Python overhead
that no amount of Numba-fying the probe loop could fix.

After Step 16c-d:
- WSL2 4-field warm Numba **median** = 3.15 µs (target: ≤5 µs at p99).
- WSL2 stddev = 1.54 µs; min = 3.01 µs; max = 170 µs (one OS-scheduler outlier).
- Native CI is typically 1.4× faster than WSL2 on this Numba-warm bench
  per ADR-015 §7. Projected native CI median: ~2.2 µs. Projected p99
  (median + 3×stddev scaled by 1.4): ~5.5 µs — **GREEN at p99 likely on
  native CI** (was projected RED at 6.5-7 µs without Numba blake2b).

Commit B replaces these projections with the canonical N=20 native-CI
numbers from `benchmarks/results/n20/headline_4_field_warm_n20.json`
and ratifies the trip-wire GREEN/RED verdict.

### Hash kernel head-to-head (informational)

`hashlib.blake2b(s, digest_size=8)` Python: ~1500 ns per call (measured
via decomposition; the per-call cost dominates pure-Python lookup hits).

`quorin._internal.hash_kernel.blake2b_8` Numba: ~150-300 ns per call
in-kernel context (the Numba dispatch overhead is amortized inside
`_lookup_core` since BLAKE2b is called from inside another @njit
function — no Python boundary).

### Pinned-hash invariant #5 — unchanged

`tests/unit/test_hash_kernel.py` covers:
- 3 pinned ASCII hashes matching `tests/unit/test_layout.py::TestHashPinned`.
- 1 Unicode pinned hash.
- 6 boundary cases (empty input, single byte, 128, 129, 256 bytes, fixed-size buffer with variable n).
- 1 Hypothesis property test × 200 examples × random byte strings 0-300 bytes.

All pass byte-for-byte against `hashlib.blake2b`. The algorithm was
re-implemented from RFC 7693, NOT changed.

## References

- ADR-004: Numba assembly is gated on a per-schema benchmark threshold
  (precedent: `quorin.serving` vs `quorin.assembly` dual-path shape).
- ADR-007: Batch assembly (precedent for `_assemble_batch_core`'s probe
  loop, which lookup_jit's kernel mirrors byte-for-byte).
- ADR-015 §7 / §11: native-CI venue limitations, bare-metal extrapolation,
  the framework that makes "likely-RED is honest" the right call.
- CLAUDE.md invariant #11 (Numba isolation) and invariant #15 (no magic
  byte offsets in Numba kernels) — both upheld by lookup_kernel's design.
- `progress/step16c_plan.md` (Rev-4 TERMINAL): the locked design rationale.
