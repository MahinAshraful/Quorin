"""Flamegraph driver: warm-cache assemble of a 200-field + 128-emb schema.

The "realistic-production" headline scenario. Spec target 10-20 us p99.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from benchmarks.flamegraphs._setup import Schema200Field, make_warm_segment
from quorin.assembly import assemble, prewarm

DURATION_SECONDS = 60.0


def main() -> None:
    prewarm()
    seg, cleanup = make_warm_segment(Schema200Field, capacity=16)
    try:
        deadline = time.monotonic() + DURATION_SECONDS
        out = None
        while time.monotonic() < deadline:
            for _ in range(5_000):
                out = assemble(seg, "u")
        if out is None:
            raise RuntimeError("assemble produced unexpected output")
    finally:
        cleanup()


if __name__ == "__main__":
    main()
