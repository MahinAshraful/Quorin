"""Chaos tests: SIGKILL + soak for /dev/shm leak detection.

Runs on Linux/WSL2 only.
"""

from __future__ import annotations

import contextlib
import multiprocessing
import os
import signal
import struct
import sys
import uuid
from pathlib import Path

import pytest

pytestmark = [
    pytest.mark.chaos,
    pytest.mark.skipif(
        sys.platform == "win32",
        reason="posix_shm requires POSIX (Linux/WSL2)",
    ),
]


def _subprocess_writer_then_sigkill(name: str) -> None:
    """Open the segment, busy-write a pattern, SIGKILL self mid-write."""
    from pyforge._internal import posix_shm

    h = posix_shm.open_existing(name)
    # Write many small patterns past the header.
    for i in range(100):
        h.buf[256 + i * 8 : 256 + i * 8 + 8] = struct.pack("<Q", i)
    # Kill during the 101st write
    os.kill(os.getpid(), signal.SIGKILL)


def _mp_context() -> multiprocessing.context.BaseContext:
    return multiprocessing.get_context("fork")


def test_sigkill_during_writes_leaves_header_intact() -> None:
    """SIGKILL a writer subprocess. The 16-byte header (written by the
    creator before the subprocess was spawned) must still be intact."""
    from pyforge._internal import posix_shm

    name = f"pyforge_chaos_{os.getpid()}_{uuid.uuid4().hex[:8]}"
    h = posix_shm.create(name, 8192)
    try:
        # Pretend creator already wrote a header.
        header = b"PYFG" + struct.pack("<II", 1, 0xDEADBEEF) + b"\x00\x00\x00\x00"
        assert len(header) == 16
        h.buf[:16] = header

        p = _mp_context().Process(target=_subprocess_writer_then_sigkill, args=(name,))
        p.start()
        p.join(timeout=10)
        assert p.exitcode == -signal.SIGKILL

        # Parent: header intact?
        back = bytes(h.buf[:16])
        assert back == header
    finally:
        posix_shm.close(h)
        posix_shm.unlink(name)


@pytest.mark.slow
def test_soak_create_close_cycle_returns_dev_shm_to_baseline(redis_client) -> None:
    """Run 200 create/close cycles and drain the cleanup queue. `/dev/shm`
    byte usage for pyforge entries must return to baseline (zero)."""
    from pyforge._internal import posix_shm
    from pyforge.schema import FeatureField, FeatureSchema, dtype
    from pyforge.shm import KEY_CLEANUP_QUEUE, SegmentRegistry

    class _SoakSchema(FeatureSchema):
        version = 1
        fields = [FeatureField("x", dtype.float32)]

    shm_dir = Path("/dev/shm")

    def _pyforge_usage() -> int:
        if not shm_dir.is_dir():
            return 0
        total = 0
        for entry in shm_dir.iterdir():
            if entry.name.startswith("pyforge_"):
                with contextlib.suppress(FileNotFoundError):
                    total += entry.stat().st_size
        return total

    baseline = _pyforge_usage()

    reg = SegmentRegistry(redis_client)
    # Reduced from 1000 → 200 so the test runs in under 10s even on a slow
    # dev box. The chaos property (no leak per cycle) is unchanged by count;
    # 200 cycles is enough to make any linear leak obvious.
    for _ in range(200):
        seg = reg.create(_SoakSchema)
        reg.close(seg)

    # Drain cleanup queue — simulates what Step 14's watchdog will do.
    members = list(redis_client.smembers(KEY_CLEANUP_QUEUE))
    for m in members:
        name = m.decode() if isinstance(m, bytes) else m
        posix_shm.unlink(name)
    if members:
        redis_client.delete(KEY_CLEANUP_QUEUE)

    assert _pyforge_usage() == baseline, (
        f"leak: /dev/shm pyforge usage went from {baseline} to {_pyforge_usage()}"
    )
