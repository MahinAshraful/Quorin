"""Flamegraph driver: assemble_batch(N=1000) on a 200-field schema.

Per-call cost dominated by Python prep (str.encode + hash) before kernel
dispatch. ADR-007's 5x-vs-N-singles claim is the manual-review check
landing in 16c.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from benchmarks.flamegraphs._setup import Schema200Field, _make_segment, _pack_row
from quorin.assembly import assemble_batch, prewarm
from quorin.layout import insert

DURATION_SECONDS = 60.0
N_BATCH = 1000


def main() -> None:
    prewarm()
    seg = _make_segment(Schema200Field, capacity=N_BATCH * 2)
    try:
        # Populate N_BATCH entities so the lookup probe lands on real slots.
        zero_values = {
            f.name: np.zeros(f.element_count, dtype=np.float32) for f in Schema200Field.fields
        }
        for i in range(N_BATCH):
            insert(seg, f"u_{i:06d}", _pack_row(Schema200Field, zero_values))

        ids = [f"u_{i:06d}" for i in range(N_BATCH)]
        deadline = time.monotonic() + DURATION_SECONDS
        out = None
        mask = None
        while time.monotonic() < deadline:
            for _ in range(50):
                out, mask = assemble_batch(seg, ids)
        if out is None or mask is None or not mask.all():
            raise RuntimeError("assemble_batch produced unexpected output / not all-hits")
    finally:
        from benchmarks.flamegraphs._setup import _release_segment  # local import; cleanup-only

        _release_segment(seg)


if __name__ == "__main__":
    main()
