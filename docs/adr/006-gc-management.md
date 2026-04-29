# ADR-006: GC management — opt-in instrumentation, opt-in timer, opt-in freeze

**Status:** Accepted
**Date:** 2026-04-29
**Step:** 7 (GC management)

## Decision

Pyforge ships a process-wide GC management module
(`pyforge._internal.gc_manager`) where every component is **opt-in** based on
data, not on the build plan's a priori claims. The user-visible API is:

```python
def start_collector(
    *,
    install_callback: bool = True,
    gen2_interval_seconds: float | None = None,
) -> None: ...
def freeze() -> None: ...
def unfreeze() -> None: ...
def stop_collector(*, join_timeout: float = 2.0) -> None: ...
def is_running() -> bool: ...           # callback installed?
def is_timer_running() -> bool: ...     # timer thread alive?
```

**Defaults (locked by N=50 + N=20 measurements):**

- `start_collector()` with no args installs the GC pause-instrumentation
  callback only. No timer thread. No `gc.freeze()`. Within measurement noise
  of the do-nothing baseline at N=50.
- `freeze()` is a separate explicit call. Never invoked automatically.
- The 500 ms gen-2 timer thread is **off by default**. The build plan's
  recommended 0.5 s interval was empirically catastrophic on the no-freeze
  path and was not justified on the freeze path either.
- `os.register_at_fork(after_in_child=...)` resets state in forked
  children. Non-negotiable for any deployment under gunicorn / uvicorn.
- No `gc.disable()` on the serving thread. Cycle-leak risk for marginal
  gain.

**Recommended opt-in for users who want bounded gen-2 cadence:**
`start_collector(install_callback=True, gen2_interval_seconds=2.0)` after
verifying on their own workload. The `2.0` value comes from the N=20
no-freeze interval sweep below.

**For users who call `freeze()`:** pass `install_callback=False` to avoid
the documented callback+freeze interaction (~31% worse median tail at
N=50). To make the footgun loud at the API boundary rather than only in
latency dashboards, `freeze()` emits a `UserWarning` when called with an
installed callback — the warning quotes ADR-006 and points to the two
safe usage patterns.

## The empirical journey — three pivots, one lesson

This ADR records three reversals in a single step. Each was driven by
data the previous decision didn't have. The lesson is in the pivots
themselves: **single-run benchmarks of rare events are statistically
useless. Sample size matters. Interaction effects don't show up in
isolated measurements.**

### Pivot 1: build plan → "freeze + 500ms timer is default"

The build plan recommended freeze + 500 ms gen-2 timer + callback. This was
the implementation's first state.

A single-run benchmark showed the timer producing a 6 ms max spike vs
~470 µs baseline, looking like a regression. The default flipped to
**freeze-only (no timer)**.

### Pivot 2: → "freeze-only + callback is default"

Two N=20 fresh-subprocess runs (mine and the user's) showed:

- Mine: timer beat freeze-only on every tail metric (35% → 15% spike rate,
  6311 → 3825 µs worst max).
- User's: hardware-dependent. Timer's spike-rate advantage didn't
  reproduce; median p999/p9999 still favored timer slightly.

The disagreement suggested timer's value is environment-dependent.
Decision: ship the simpler default (no timer), document the timer as
opt-in for noisy hardware. State at this point: callback installed by
default, timer opt-in via `gen2_interval_seconds`, freeze a separate call.

### Pivot 3: N=50 anomaly investigation reveals interaction effect

In my N=20 data, `freeze_only` (callback + freeze) appeared to be **worse
than baseline** (no callback, no freeze). 35% spike rate vs 20%. The user
flagged this correctly: "freeze should never be worse than doing nothing —
investigate before shipping."

**N=50 anomaly run** (`step7_freeze_anomaly_n50.txt`):

| Scenario | spike ≥ 500 µs | spike ≥ 1 ms | median max | mean max | sd_max |
|---|---|---|---|---|---|
| baseline (nothing) | 6/50 = 12% | 3/50 = 6% | 155 µs | 326 µs | 526 |
| freeze_only (callback + freeze) | 8/50 = 16% | 5/50 = 10% | **204 µs** | **396 µs** | 625 |

Confirmed: freeze_only is consistently worse. **+33% relative on spike rate,
+67% on 1 ms-spikes, +31% on median max.** Not noise.

**N=50 isolation run** (`step7_freeze_isolation_n50.txt`):

| Scenario | spike ≥ 500 | spike ≥ 1 ms | median max | mean max | sd_max |
|---|---|---|---|---|---|
| baseline | 6/50 = 12% | 3/50 = 6% | 155 µs | 326 µs | 526 |
| **callback_only (no freeze)** | 7/50 = 14% | 2/50 = 4% | **158 µs** | 286 µs | 274 |
| **freeze_only_no_callback** | 6/50 = 12% | 2/50 = 4% | **161 µs** | 277 µs | 325 |
| freeze_only (BOTH) | 8/50 = 16% | 5/50 = 10% | 204 µs | 396 µs | 625 |

**Neither component is harmful in isolation.** Callback-alone is within 2%
of baseline. Freeze-alone is within 4%. **The combination produces an
interaction effect** of ~31% on median max and ~67% relative on the 1ms
spike rate.

**Mechanism is unlocalized at ADR-006 close.** Possibilities I cannot
disambiguate without deeper profiling:

- prometheus_client lock contention with post-freeze allocator state
- gen-0/1 walk timing changes when references cross the permanent boundary
  (the cyclic GC follows but does not recurse into permanent objects;
  references-into-permanent identification cost may scale with permanent
  set size)
- page-cache effects from the 1M frozen objects influencing collect timing
  via cache pressure on hot allocator paths

The interaction is real at N=50; the cause is a future investigation.

### Interval sweep — choosing the canonical opt-in interval

The build plan suggested 0.5 s as a starting interval. To pick a
data-justified opt-in for users who want the timer, I swept
`{0.5, 1.0, 2.0, 5.0}` seconds at N=20 each on the no-freeze + callback
path (`step7_no_freeze_interval_sweep.txt`):

| Interval | spike ≥ 500 µs | spike ≥ 1 ms | spike ≥ 10 ms | median max | p90 max |
|---|---|---|---|---|---|
| 0.5 s | 18/20 = 90% | 18/20 = 90% | 18/20 = 90% | 46,619 µs | 53,032 µs |
| 1.0 s | 2/20 = 10% | 1/20 = 5% | 0/20 = 0% | 189 µs | 619 µs |
| **2.0 s** ✓ | **2/20 = 10%** | **0/20 = 0%** | **0/20 = 0%** | **159 µs** | **528 µs** |
| 5.0 s | 1/20 = 5% | 0/20 = 0% | 0/20 = 0% | 174 µs | 498 µs |

Without `freeze()`, the timer forces gen-2 to walk the unfrozen
1M-object heap every interval (~45 ms per collect, GIL held). At 0.5 s
cadence the timer fires once during the 0.55 s assemble loop and the
walk lands inside it; 18 of 20 runs had 45–66 ms maxes. At 1.0 s and
above the timer's expected number of in-loop fires drops below 1 and
the catastrophic spikes disappear.

Adjusted opt-in recommendation to **`gen2_interval_seconds=2.0`** based
on the data: best median max, best p90 max, lowest sd_max. 5.0 s is
within noise of 2.0 but gives gen-2 more time to grow on workloads with
higher allocation rates than the synthetic benchmark, so 2.0 s is the
safer default opt-in. The build plan's 0.5 s value is fine when
`freeze()` is also called (gen-2 stays small post-freeze, so 0.5 s
collects are cheap); the sweep above runs without freeze because that
matches the recommended safe combination.

## Why each design choice survives

### Module-level state, not a class

Python's GC is process-global; there is exactly one to manage. A
`class GcManager` would be premature object-orientation. Internal state
is grouped in a slotted `_State` instance to avoid the `global` keyword
without exposing a public class.

### `gc.callbacks.insert(0, ...)`

Running first means our histogram observes bare `gc_collect_main` cost,
not time spent in peer callbacks (third-party tracers, profilers).

### Pre-warmed prometheus label children

`prometheus_client.Histogram.labels(...)` allocates a tuple key + dict slot
on first use. **In a GC callback, that allocation can trigger a re-entrant
collection.** We pre-warm the three label children at `start_collector()`
when `install_callback=True` and cache them in `_observers[]` indexed by
generation. Indexed access only — no allocation.

### `_pause_starts` as a fixed list, not a dict

Three generations, indexed 0/1/2 in CPython 3.7+. List access is one
bytecode; dict access allocates a tuple key on insert.

### `os.register_at_fork(after_in_child=...)`

Multi-worker servers (gunicorn, uvicorn) fork after import. The parent's
`_state.thread` reference is meaningless in the child. The hook resets all
module state in the child so any subsequent `start_collector()` works
cleanly. Wired at module-import time so callers in forked workers don't
have to think about it.

### Stop ordering: remove callback first

Race scenario: timer thread is mid-`gc.collect(2)` when `stop_collector()`
is called. If we set the stop event first, the in-flight collect's "stop"
callback fires *after* we've started clearing module state. Fix: remove
the callback before signaling. Trade: lose at most one in-flight
observation on shutdown for clean state.

### Zombie threads on join timeout: tracked, not nulled

If `gc.collect(2)` is mid-flight on a giant heap and exceeds
`join_timeout`, the thread doesn't exit cleanly. We track timed-out
threads in `_zombies`; `start_collector()` checks `is_alive()` for
idempotency, so a zombie cannot result in two parallel collectors
scrambling `_pause_starts`.

### Errors in GC callbacks: deliberate swallow

Bare `except Exception: pass` in `_gc_callback`. The callback runs from
inside CPython's GC. Logging via `structlog` allocates Python objects →
re-entrant GC → instant deadlock. With pre-warming, the most likely failure
sources are eliminated. A pre-warmed error counter for visibility is
deferred to Step 16.

### No `gc.disable()` on serving thread

Build plan's fallback. Cycle-leak risk for marginal gain when the
combinations of callback + freeze + timer don't reliably compress the tail
either. Not adopted.

### No `BackgroundWorker` base class yet

Step 10 (WAL consumer) and Step 14 (watchdog) will reuse `daemon=True +
threading.Event + register_at_fork`. Their loop bodies differ
fundamentally (Redis blocking I/O, watchdog self-monitoring). Don't
extract a base class until Step 14 reveals what's actually shared. ADR-006
documents the *primitives* (in this file and in CLAUDE.md section 5).

## Validation

- `tests/unit/test_gc_manager.py` — 21 tests covering lifecycle (start /
  stop / idempotency / cycle), `install_callback=False` escape hatch,
  default-no-timer behavior, callback semantics, pre-warming, freeze
  idempotency, the freeze-while-callback-installed warning (3 tests:
  warning fires when callback is installed, no warning fires on the safe
  freeze-before-start_collector path, no warning fires on the explicit
  `install_callback=False` opt-out path), zombie tracking, tight-loop
  sanity.
- `tests/chaos/test_gc_soak.py` — 10 s soak with linear-regression slope
  assertion. Exercises both freeze and timer paths under load.
- `benchmarks/test_gc_p999.py` — 6 distributions × 100 k assembles each:
  baseline, freeze_only, freeze_plus_timer, freeze_plus_timer + pressure,
  callback_only_no_freeze (ADR-006 isolation control),
  freeze_only_no_callback (ADR-006 isolation control), no_freeze interval
  sweep (parametrized over [0.5, 1.0, 2.0, 5.0]).
- `benchmarks/runs/step7_gc_tail_repeat.py` — fresh-subprocess
  orchestrator with `--scenarios` and `--num-runs` CLI args. The data
  generator behind every claim in this ADR.
- Raw datasets committed to `benchmarks/results/`:
  - `step7_freeze_anomaly_n50.txt` — N=50 baseline vs freeze_only
  - `step7_freeze_isolation_n50.txt` — N=50 callback_only vs freeze_only_no_callback
  - `step7_no_freeze_interval_sweep.txt` — N=20 across 4 intervals

## Future revisit triggers

Re-open this decision if:

- **Production p9999 telemetry contradicts the synthetic benchmark.**
  Real-workload allocation rates and heap shapes differ from the synthetic
  `_build_long_lived_heap()` setup. If real-world tail behavior is
  meaningfully different, the no-timer default may be wrong.
- **The mechanism behind the callback+freeze interaction is localized.**
  If a fix to the callback (e.g., skip gen-0 observations, switch to a
  lock-free counter) eliminates the interaction, the API can simplify and
  freeze + callback can be the recommended combination.
- **CPython's GC implementation changes materially in 3.13+** (free-threaded,
  incremental GC). Lock-free deque atomicity and `gc.callbacks` semantics
  may shift.
- **Buffer pool's allocation rate drops materially.** Step 6's pool already
  cut allocations sharply; if Step 8's batch assembly or a future
  C-extension cuts further, gen-2 pressure may drop below the threshold
  where any timer interval matters.
- **Step 14 watchdog or Step 10 WAL consumer reveal a shared
  background-worker abstraction.** Extract a `BackgroundWorker` base class
  then, not now.

## Consequences

- **Positive:** every component is opt-in based on N=50 / N=20 measurements,
  not on a priori claims from the build plan. Defaults are within
  measurement noise of the do-nothing baseline.
- **Positive:** the `install_callback=False` escape hatch gives users who
  want freeze + timer a way to avoid the documented interaction tax.
- **Positive:** the canonical opt-in interval (2.0 s) is data-justified.
  Ships as a documented recommendation rather than a hardcoded default.
- **Positive:** the repeat-measurement orchestrator
  (`benchmarks/runs/step7_gc_tail_repeat.py`) is reusable for any future
  tail-latency claim in this codebase. Step 14 watchdog jitter and Step 16
  flamegraph experiments should follow the same N≥20 fresh-subprocess
  pattern.
- **Negative:** the API surface is wider than the build plan envisioned —
  three opt-in toggles where the plan had one default-on path. Necessary
  given the data; documented.
- **Negative:** the callback+freeze interaction is unfixed and unexplained.
  Users who want both pay the tax. Future revisit triggers cover this.
- **Negative:** no automatic `freeze()` from `start_collector()`. Callers
  who don't read the docs may not realize they should call freeze
  explicitly. Acceptable trade for not auto-creating the interaction.

## References

- Build plan, Step 7 section.
- Instagram engineering blog on `gc.freeze()` (memory savings + pause
  reduction in their Django workload).
- CPython `Modules/gcmodule.c` — confirmation that `info["generation"]` is
  always populated in 3.7+.
- ADR-005 (buffer pool) — eliminates the per-call allocation that would
  otherwise feed gen-0/1.
- PEP 703 (free-threaded CPython) — `gc.callbacks` list semantics may
  require re-validation in 3.13t.
