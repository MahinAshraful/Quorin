"""Step 8 — threshold sweep for ``PARALLEL_THRESHOLD`` in quorin.assembly.

Sweeps batch sizes for both the serial and parallel batch kernels at the
200-field schema (the headline schema where the parallelism win matters most).
Output is a per-N median time for each kernel; the crossover point sets
``PARALLEL_THRESHOLD``.

The sweep also produces multi-thread scaling numbers via NUMBA_NUM_THREADS:
the parallel-kernel medians are recorded per thread count, so scaling
characteristics can be documented in ADR-007 (linear up to N cores? saturates
at 2x? plateau at 4?). On WSL2 the user's available core count constrains
the upper bound.

Usage:
    uv run python benchmarks/runs/step8_threshold_sweep.py

    # Or with custom thread counts:
    NUMBA_NUM_THREADS=4 uv run python benchmarks/runs/step8_threshold_sweep.py

Output:
    Tabulated medians per (batch_size, kernel) printed to stdout.
    Also written to benchmarks/results/step8_threshold_sweep.txt.

The user runs this once; we set PARALLEL_THRESHOLD based on the data, then
re-run the regular benchmark suite for headline ratio numbers.
"""

from __future__ import annotations

import math
import os
import statistics
import sys
import time
from pathlib import Path

import numpy as np

# Add tests/ to sys.path so we can import _helpers directly.
_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "tests"))

from _helpers import make_segment, pack_row, release_segment  # noqa: E402
from quorin import assembly as assembly_mod  # noqa: E402
from quorin.assembly import assemble_batch, prewarm  # noqa: E402
from quorin.layout import insert  # noqa: E402
from quorin.schema import FeatureField, FeatureSchema, dtype  # noqa: E402

OUTPUT = _REPO / "benchmarks" / "results" / "step8_threshold_sweep.txt"

#: Batch sizes to sweep. Anchored on the user-specified set so the crossover
#: lands somewhere in the middle.
BATCH_SIZES = [1, 10, 32, 64, 128, 256, 1000, 10000]

#: Per-(N, kernel) iteration count - runs N x K assembly calls and reports
#: the median per-call wall time. Higher = more stable median, but takes
#: longer for large N.
ITERATIONS_PER_CONFIG = 20

#: Number of inner repetitions per config to derive the median from.
WARMUP = 3


def _build_200_field_schema() -> type[FeatureSchema]:
    fields: list[FeatureField] = [FeatureField(f"feat_{i:03d}", dtype.float32) for i in range(199)]
    fields.append(FeatureField("embedding", dtype.float32, shape=(128,)))
    return type("_SweepSchema200", (FeatureSchema,), {"version": 1, "fields": fields})


def _populate(seg, schema: type[FeatureSchema], n: int) -> list[str]:
    rng = np.random.default_rng(seed=n)
    ids = [f"e_{i:05d}" for i in range(n)]
    for eid in ids:
        values = {
            f.name: rng.standard_normal(f.element_count).astype(np.float32) for f in schema.fields
        }
        insert(seg, eid, pack_row(schema, values))
    return ids


def _measure_one(seg, ids: list[str], out: np.ndarray, mask: np.ndarray, iterations: int) -> float:
    """Run iterations, return median per-call wall time in microseconds."""
    times: list[float] = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        assemble_batch(seg, ids, out=out, found_mask=mask)
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1e6)
    return statistics.median(times)


def _sweep(force_threshold: int, label: str, schema, capacity: int) -> dict[int, float]:
    """Force a single kernel via PARALLEL_THRESHOLD and sweep batch sizes."""
    assembly_mod.PARALLEL_THRESHOLD = force_threshold
    results: dict[int, float] = {}
    elem_count = sum(f.element_count for f in schema.fields)

    for n in BATCH_SIZES:
        if n > capacity:
            results[n] = float("nan")
            continue
        seg = make_segment(schema, capacity=capacity)
        try:
            ids = _populate(seg, schema, n)
            out = np.empty((n, elem_count), dtype=np.float32)
            mask = np.empty(n, dtype=np.bool_)

            # Warmup (paid once per (n, kernel) — Numba threading-pool spinup,
            # page cache for the segment, etc.).
            for _ in range(WARMUP):
                assemble_batch(seg, ids, out=out, found_mask=mask)

            median_us = _measure_one(seg, ids, out, mask, ITERATIONS_PER_CONFIG)
            results[n] = median_us
        finally:
            release_segment(seg)
    return results


def main() -> None:
    print("=" * 80)
    print("Step 8 — threshold sweep: PARALLEL_THRESHOLD calibration")
    print("=" * 80)
    print(f"NUMBA_NUM_THREADS = {os.environ.get('NUMBA_NUM_THREADS', '<default>')}")
    print("Schema: 200-field with 128-dim embedding (327 elements / row)")
    print(f"Iterations per (N, kernel): {ITERATIONS_PER_CONFIG} after {WARMUP} warmup(s)")
    print()

    print("Pre-warming both kernels...", end=" ", flush=True)
    prewarm()
    print("done.")
    print()

    schema = _build_200_field_schema()
    capacity = max(BATCH_SIZES) + 16

    # Force serial for all sizes.
    print("Sweeping serial kernel...")
    serial_results = _sweep(
        force_threshold=1_000_000, label="serial", schema=schema, capacity=capacity
    )

    # Force parallel for all sizes.
    print("Sweeping parallel kernel...")
    parallel_results = _sweep(force_threshold=0, label="parallel", schema=schema, capacity=capacity)

    # Tabulate.
    lines: list[str] = []
    lines.append("")
    lines.append(
        f"{'N':>8} {'serial (us)':>14} {'parallel (us)':>16} {'speedup':>10} {'crossover':>12}"
    )
    lines.append("-" * 64)
    for n in BATCH_SIZES:
        s = serial_results[n]
        p = parallel_results[n]
        if math.isnan(s) or math.isnan(p):
            continue
        speedup = s / p if p > 0 else float("nan")
        marker = "" if speedup < 1.0 else "<- parallel wins"
        lines.append(f"{n:>8} {s:>14.2f} {p:>16.2f} {speedup:>10.2f}x  {marker:>12}")
    lines.append("")
    lines.append("Recommendation: set PARALLEL_THRESHOLD to the smallest N where")
    lines.append("the speedup ratio first exceeds 1.0 (and stays >1.0 for larger N).")
    lines.append("If no N shows speedup > 1.0, the parallel kernel is not viable on")
    lines.append("this machine; document and ship serial-only.")

    output = "\n".join(lines)
    print(output)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w") as f:
        f.write(f"NUMBA_NUM_THREADS = {os.environ.get('NUMBA_NUM_THREADS', '<default>')}\n")
        f.write(output)
    print(f"\nWritten to: {OUTPUT}")


if __name__ == "__main__":
    main()
