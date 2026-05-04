"""Shared L3-detection helper for cold-cache benches.

Extracted from ``benchmarks/conftest.py`` (Step 16b Rev-4 H1 cascade) so the
standalone MADV A/B orchestrator at ``benchmarks/runs/step16_madvise_ab.py``
can size its clobber array without going through pytest fixture machinery.

The conftest fixture ``cold_cache_clobber`` imports from here. Don't put the
fixture itself in this module — pytest fixtures aren't usable outside pytest's
collection.
"""

from __future__ import annotations

import logging
from pathlib import Path

LOGGER = logging.getLogger(__name__)


def detect_l3_size_bytes() -> int:
    """Read L3 cache size from sysfs. Returns 16 MiB fallback if L3 unavailable.

    Critical (C3): do NOT fall back to ``/sys/.../index2/size``. On WSL2 / VMs
    that hide cache hierarchy, ``index2`` reports L2 (a few MB). A 4x-L2
    clobber is nowhere near L3 size and the bench measures L2-cold instead
    of L3-cold, silently. Better to use a deliberately conservative 16 MiB
    fallback that the bench logs WARN about than to confidently report
    bad data.
    """
    p = Path("/sys/devices/system/cpu/cpu0/cache/index3/size")
    try:
        raw = p.read_text().strip()
        # format: "32K", "8M", "105M"
        unit = raw[-1].upper()
        num = int(raw[:-1])
        return num * (1024 if unit == "K" else 1024 * 1024 if unit == "M" else 1)
    except (FileNotFoundError, ValueError, OSError):
        LOGGER.warning(
            "cold-cache: /sys/.../index3/size unavailable; using 16 MiB fallback "
            "(4x clobber = 64 MiB). On hosts with L3 > 16 MiB this measures "
            "L2/L3-partial-cold, not L3-cold. Run on bare metal for honest L3-cold."
        )
        return 16 * 1024 * 1024
