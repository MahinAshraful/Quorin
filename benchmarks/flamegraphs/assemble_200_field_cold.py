"""Flamegraph driver: cold-cache assemble of a 200-field + 128-emb schema.

Traverses an L3-sized clobber array between assemble calls so each
assemble starts with cold CPU caches. Spec target 20-50 us p99.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from benchmarks.flamegraphs._setup import (
    Schema200Field,
    make_clobber_array,
    make_warm_segment,
)
from pyforge.assembly import assemble, prewarm

DURATION_SECONDS = 60.0


def main() -> None:
    prewarm()
    seg, cleanup = make_warm_segment(Schema200Field, capacity=16)
    clobber = make_clobber_array()
    try:
        deadline = time.monotonic() + DURATION_SECONDS
        out = None
        # Touch every cache line of the clobber, then do one assemble.
        # Repeats fast enough to give py-spy plenty of samples inside assemble
        # while the surrounding clobber traverses keep the segment evicted.
        while time.monotonic() < deadline:
            # 4x stride-by-1 to ensure cache lines are touched (each float64
            # is 8 bytes; cache line is 64; touching every 8th element evicts
            # everything resident).
            clobber[::8].fill(0)
            out = assemble(seg, "u")
        if out is None:
            raise RuntimeError("assemble produced unexpected output")
    finally:
        cleanup()


if __name__ == "__main__":
    main()
