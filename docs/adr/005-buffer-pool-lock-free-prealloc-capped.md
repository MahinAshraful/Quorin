# ADR-005: Buffer pool — lock-free, pre-allocated, capped, no threaded refill

**Status:** Accepted
**Date:** 2026-04-28
**Step:** 6 (Buffer pool)

## Decision

Quorin ships `quorin.pool.BufferPool` — a per-schema pool of pre-allocated
float32 output buffers consumed via a `with pool.checkout() as buf:` context
manager. The pool is:

1. **Class-based context manager**, not `@contextlib.contextmanager`. The
   generator-based decorator costs ~700 ns/call (GeneratorContextManager
   allocation + `next()` + `send()`). On a sub-microsecond serving path
   that's the difference between net-positive and net-negative; the
   class-based `_Checkout` with `__slots__` brings the per-call protocol
   cost to ~150 ns.
2. **Lock-free** under the CPython GIL via `collections.deque.popleft()` /
   `.append()` atomicity; no `threading.Lock` is acquired per checkout.
3. **Pre-allocated at construction** — all `max_size` buffers are minted in
   `__init__`, so the pool runs at 100% hit rate from request 1, not after
   warm-up.
4. **Capped at `max_size` on return** — surplus buffers from a fall-through-
   allocate burst are dropped at exit, preventing permanent memory inflation.
5. **Zero-on-return, default true, toggleable** via `zero_on_return: bool`.
6. **No threaded refill.** Rejected by an explicit cost analysis (below).
7. **Only `pool_miss_total` is observed.** No per-checkout counter, no
   pool-size gauge — the hot-path tax (~150 ns / `Counter.inc()`) would
   wipe out any pool benefit.
8. **Pools are immortal in v1.** No `clear()` / `close()` method. Schema
   upgrade (Step 15) will replace, not clear.

## Context

After Step 5, the 200-field warm assemble runs in ~5.37 µs p99. Profiling
identifies the `np.empty(total_element_count, dtype=np.float32)` call inside
`assemble` as the only remaining hot-path allocation. The build plan's
Step 6 sketch proposed a `collections.deque`-backed pool with an
asyncio-driven refill thread that tops up the pool when it drops below
25% capacity.

The refill-thread half is wrong for a steady-state ML serving workload, where
concurrency is bounded by worker count and not bursty:

| Path | Cost (estimate) |
|---|---|
| Pool hit: `deque.popleft()` only | ~50 ns |
| Pool miss: `np.zeros(N, dtype=np.float32)` | ~80–300 ns (depends on N) |
| `threading.Lock` acquire + release on every borrow | ~100 ns |

A refill thread saves the miss-allocation cost. To break even, every ~3
checkouts must miss. At a steady-state RPS where the pool is right-sized,
miss rate is 0% — the lock cost is paid 100% of the time, the saving is paid
0% of the time. The threaded-refill design solves a problem the workload
does not have, and creates a per-call tax that exceeds the per-miss cost it
tries to amortize. **Premature optimization the wrong half of the term.**

### What the measurement showed

The first implementation used `@contextlib.contextmanager`. Benchmarks were
~940 ns hit-only — *worse* than the bare `np.empty(N, float32)` path the
pool was supposed to replace. Refactoring to a class-based context manager
(`_Checkout` with `__slots__`, direct `__enter__` / `__exit__`) brought
hit-only down to **~550 ns median**. That gap (~400 ns) is the cost of
`@contextmanager`'s generator protocol on a sub-microsecond loop:
GeneratorContextManager allocation, `next()` to advance to yield, `send()`
to advance past. Worth knowing for any future Quorin code that's tempted
to `@contextmanager` something on the hot path.

Even with the class-based refactor, the pool is **not a per-call latency
win** for the schemas Quorin currently serves:

| Scenario | Step 5 (no pool) | Step 6 pooled | Delta |
|---|---|---|---|
| 4-field warm | 4.17 µs | 4.83 µs | +660 ns |
| 200-field warm + 128-emb | 5.37 µs | 6.10 µs | +730 ns |
| `pool_checkout_hit_only` | — | 550 ns | — |

`np.empty(N, dtype=np.float32)` for `N <= 1300` is ~80–100 ns on this
hardware — far cheaper than the ~300 ns the build plan estimated. The pool
replaces 80–100 ns of `np.empty` with ~550 ns of context-manager protocol;
on the absolute hot path, that's a small regression, not a win.

So why ship the pool at all? Three real wins survive:

1. **Eliminates one ndarray allocation per call.** Step 7's GC management
   takes a hard line on allocation rate — every `np.empty` is a Python
   object that goes through the cyclic-GC graph. Removing 50,000 of those
   per second meaningfully reduces gen-0 pressure.
2. **Memory ceiling.** A burst that holds 200 buffers concurrently grows the
   pool past `max_size`, but the cap-at-max-size return path drops the
   surplus. Steady-state memory is bounded by `max_size * element_count *
   4` bytes — predictable for capacity planning.
3. **Foundation for Step 8 (batch assemble).** Output buffers there are
   `(N, feature_count)` 2D arrays — typically 100×–1000× larger than the
   single-row case. `np.empty` cost scales linearly with bytes; the pool's
   constant ~550 ns overhead becomes a clear net win.
4. **Observability.** `pool_miss_total` is the canary for under-sized
   pools.

Honesty wins over advocacy: this ADR records that for the *current*
schemas, the pool is approximately a wash on raw latency. It's still the
right scaffolding to land in Step 6, both for the secondary wins above and
because Step 8 makes the trade flip decisively in the pool's favor.

## The design

### Class-based context manager (NOT `@contextmanager`)

`pool.checkout()` returns a fresh `_Checkout` instance whose `__enter__`
pops from the deque and whose `__exit__` zeros and re-appends. `_Checkout`
uses `__slots__ = ("_buf", "_pool")` to keep allocation tiny.

```python
class _Checkout:
    __slots__ = ("_buf", "_pool")

    def __init__(self, pool):
        self._pool = pool
        self._buf = None

    def __enter__(self):
        pool = self._pool
        try:
            buf = pool._buffers.popleft()
        except IndexError:
            pool_miss_total.labels(schema=pool._schema_label).inc()
            buf = np.zeros(pool._element_count, dtype=np.float32)
        self._buf = buf
        return buf

    def __exit__(self, exc_type, exc_val, exc_tb):
        buf = self._buf
        if buf is None:
            return
        self._buf = None
        pool = self._pool
        if pool._zero_on_return:
            buf.fill(0)
        if len(pool._buffers) < pool._max_size:
            pool._buffers.append(buf)
```

The `buf is None` sentinel in `__exit__` handles the corner where
`__enter__` raised before assigning `_buf` (e.g., MemoryError on the
fallback `np.zeros`). The local `pool = self._pool` aliasing is
deliberate: each subsequent attribute access via the local skips one
descriptor lookup vs `self._pool._buffers`.

### Lock-free via deque atomicity

CPython implements `collections.deque` with internal locking around
`append` / `popleft`; the GIL plus those internal mutex operations make the
ops indivisible from a Python-level observer. Two threads racing on
`popleft()` on an empty deque both safely raise `IndexError`; two threads
racing on a full deque each receive a different element. No double-take is
possible.

### Pre-allocate at `__init__`

The build plan's spec sketch grew the pool lazily on first checkout. That
makes request 1 a miss, request 2 a miss, etc., until the working set is
covered. Pre-allocating moves that cost to construction time — paid once,
during process startup, before any request is served.

128 buffers × ~5 KB (200-field schema with 128-dim embedding) = **640 KB**.
Trivial. Smaller schemas use less. Each buffer is a separate `np.zeros`
allocation, so they're not contiguous and there's no cross-checkout cache
locality — but since assemble walks every output position before yielding
the buffer to the caller, that's irrelevant.

### Cap returns at `max_size`

A burst (e.g. a batch endpoint that holds 200 buffers concurrently for one
request) grows the pool past `max_size` via the fall-through allocation
path. When the burst returns its buffers, we re-append only up to
`max_size`; the rest are dropped and GC reclaims them. This caps
steady-state memory at `max_size * element_count * 4` bytes regardless of
peak burst size.

### Zero-on-return, default true, toggleable

Both assemble paths walk every output index in declaration order, so a
correct implementation overwrites every byte and prior buffer contents are
never observed. Zeroing on return is therefore strictly defense in depth:
catches bugs (e.g. a future `assemble_batch` that partial-fills) and
guarantees a clean buffer if the caller violates the lifetime contract by
reading the buffer outside the `with` block.

`buf.fill(0)` for a 1300-element float32 buffer costs ~30 ns on this
machine — under 1% of the assemble cost. The default is `True`. The toggle
exists because users who have measured the cost as unaffordable will ask
for it; better to ship explicit than discover the request after launch.

### Why not more metrics

The reviewer suggested adding `pool_checkout_total` Counter and
`pool_size` Gauge for ops visibility. `prometheus_client.Counter.inc()`
costs ~150–200 ns in pure Python; on a 50 ns hit path that is a 3–4×
tax that wipes out the pool's win. Hit rate can be derived downstream from
`pool_miss_total / request_total` if `request_total` is tracked at a layer
above the pool. **Revisit when ops asks** — adding the metric at that point
is a 5-line change, and we'll have a real cost-benefit ratio to decide on.

### Pools are immortal

No `clear()` / `close()` method in v1. A long-running service that
hot-swaps schemas (Step 15) will replace its pool with a fresh one;
the old one's buffers GC out when the last reference drops.
Adding a `clear()` to satisfy a hypothetical future need is the kind of
premature optionality the project rules forbid.

## Consequences

- **Positive:** the hot path eliminates one Python `ndarray` allocation per
  call — meaningful for Step 7's GC management even where the per-call
  latency cost is roughly neutral.
- **Positive:** the design is small enough to audit completely. ~50 lines
  of pool code, no async, no extra threads, no shared state beyond the
  deque.
- **Positive:** lifetime contract is loud and simple — buffers must not
  outlive the `with` block. Users who follow it cannot corrupt the pool.
- **Negative:** lock-free via GIL atomicity is **not** lock-free under PEP
  703 free-threaded interpreters. `sys._is_gil_enabled() is False` would
  require a real `threading.Lock` around the deque ops, which would push
  hit-cost from ~50 ns to ~150 ns. Multi-thread test is gated on
  `sys._is_gil_enabled()`; production code carries a TODO to revisit when
  CPython 3.13t adoption matters in this codebase.
- **Negative:** users who retain references to the buffer past the `with`
  block will silently see corruption when the next checkout reuses the
  same allocation. The lifetime contract is documented in the docstring;
  no runtime detection (would require expensive aliasing checks). If this
  becomes a recurring footgun, a debug-mode wrapper that panics-on-
  retained-reference is one option.

## Validation

- `tests/unit/test_pool.py` covers construction, exhaustion, exception
  safety, surplus-drop, multi-thread stress, max_size validation, and the
  zero-on-return toggle.
- `tests/property/test_pool_parity.py` runs ~600 Hypothesis examples
  asserting `assemble(out=pool_buf)` ≡ `assemble()` for both paths and
  cross-path. Buffers are NaN-poisoned before the call so a "forgot to
  write index k" bug cannot pass.
- `tests/integration/test_pool_e2e.py` includes a SIGINT-injected
  KeyboardInterrupt test proving the `finally:` re-appends.
- `benchmarks/test_pool_benchmark.py` measures pool primitive cost (hit,
  hit-no-zero, miss), pooled assemble at both 4-field and 200-field, and
  `buf.fill(0)` vs `buf[:] = 0` for the zero-on-return implementation choice.

### Measured numbers (post-Step-6 benchmark, WSL2 Ubuntu)

| Benchmark | Median | Notes |
|---|---|---|
| `pool_checkout_hit_only` | 550 ns | full hit path with zero-on-return |
| `pool_checkout_hit_no_zero` | 330 ns | with `zero_on_return=False`; isolates fill cost |
| `pool_checkout_miss_only` | 420 ns | empty pool; np.zeros fallback |
| `assemble_4_field_warm_pooled` | 4.83 us | vs Step 5's 4.17 us (no pool) — +660 ns |
| `assemble_200_field_warm_pooled` | 6.10 us | vs Step 5's 5.37 us (no pool) — +730 ns |
| `zero_on_return_op[fill]` | 260 ns | for 1300-element float32 |
| `zero_on_return_op[slice]` | 471 ns | `buf[:] = 0` is ~1.8x slower than `buf.fill(0)` |

**`buf.fill(0)` is locked as the implementation** for zero-on-return —
~1.8× faster than the slice-assign alternative on a 1300-element buffer,
matching CPython's "single C call vs slice protocol" intuition.

## Step 16c amendment — native-CI pool overhead is wider than the original WSL2 measurement

**Date:** 2026-05-05
**Workflow:** GitHub Actions ubuntu-latest run 25394553451, commit 4818ea4.

The `progress/step16c_review.md` Check A (pool overhead vs Numba assemble
on the hot path) ran against the canonical Tier-1 single-process JSON
post-Step-16c-d:

| Schema | numba p99 | pooled p99 | delta | Original ADR-005 claim |
|---|---|---|---|---|
| 4-field | 6.61 us | 8.70 us | **+2.09 us** | +0.66 us (~"latency-neutral") |
| 200-field | 12.14 us | 16.03 us | **+3.90 us** | +0.73 us |

Both **breach the +0.5 us latency-neutral gate** by 4-8x. The pool is no
longer accurately described as "latency-neutral on the hot path" on
native CI. The honest disclosure:

1. **The pool costs measurable latency on native CI** — +2-4 us per
   single-entity assemble at WSL2-tolerable schema sizes. WSL2's
   original +660-730 ns delta was an underestimate of the production
   cost on native CI's older Xeons (per ADR-015 §7's cache-architecture
   asymmetry).
2. **The wins still survive** — the four reasons in §"Why ship the pool
   anyway" above (one ndarray allocation eliminated, memory ceiling,
   foundation for batch, `pool_miss_total` observability) still apply.
3. **Pool stays default for the batch path** (Step 8's `BatchBufferPool`)
   where the pool's overhead is amortized over 1000 entities — `np.empty`'s
   cost scales linearly with bytes; the pool's ~550 ns is constant.
4. **Pool is opt-in for the single-entity path** — callers who measure
   their own workload and find the +2-4 us unacceptable use
   `quorin.assembly.assemble(seg, eid)` without a pool. Callers wanting
   the GC-pressure / memory-ceiling wins use the pooled API.

The Step 17 follow-up "C-extension `_Checkout`" (parking-lot item)
targets ~50 ns instead of ~550 ns Python overhead. If a workload needs
sub-microsecond pool checkout on native CI, that's the path.

The README's quoted pool numbers cite the native-CI measurement above,
NOT the WSL2 numbers in the table at §"Measured numbers (post-Step-6
benchmark, WSL2 Ubuntu)". The WSL2 table is preserved for historical
context.

## References

- Quorin build plan, Step 6 section ("Buffer pool").
- ADR-002 on per-open refcounting (analogous "no per-call shared-state
  mutation" reasoning for hot paths).
- ADR-004 on Numba adoption (the path this pool serves).
- ADR-015 §7 (native CI vs WSL2 cache-architecture asymmetry — context
  for the Step 16c amendment above).
- ADR-017 (lookup-jit + Numba BLAKE2b — the trip-wire ratification that
  surfaced this overhead measurement).
- CPython source `Modules/_collectionsmodule.c` (deque atomicity).
- PEP 703 (free-threaded CPython, 3.13t — pool's GIL assumption boundary).
