"""Flamegraph driver: warm-cache assemble of a 4-field schema.

Tight loop calling pyforge.assembly.assemble. Run via:

    py-spy record --native --rate 500 --duration 30 \\
      --output benchmarks/results/flamegraphs/assemble_4_field_warm.svg \\
      -- python benchmarks/flamegraphs/assemble_4_field_warm.py

Step 16c trip-wire scenario. The flamegraph should surface lookup_jit
(post-Step-16c) as a small fraction of the per-iter cost; hash_entity_id's
blake2b (~1.5 us Python) is the next-largest box and the candidate for
Step 17 Numba-jit work.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from benchmarks.flamegraphs._setup import Schema4Field, make_warm_segment
from pyforge.assembly import assemble, prewarm

DURATION_SECONDS = 60.0  # generous; py-spy's --duration usually wins first


def main() -> None:
    prewarm()
    seg, cleanup = make_warm_segment(Schema4Field, capacity=64)
    try:
        deadline = time.monotonic() + DURATION_SECONDS
        out = None
        while time.monotonic() < deadline:
            for _ in range(10_000):
                out = assemble(seg, "u")
        # Reference out so the optimizer doesn't elide the call.
        if out is None or out.shape != (4,):
            raise RuntimeError("assemble produced unexpected output")
    finally:
        cleanup()


if __name__ == "__main__":
    main()
