"""⚡ THE CRITICAL CROSS-PROCESS TEST ⚡

If ``test_reader_clean_exit_does_not_destroy_segment`` fails, something in
the open path is registering the segment with
``multiprocessing.resource_tracker``. That is the exact bug the POSIX
wrapper exists to avoid. Find and remove the offending registration.

If ``test_reader_sigkill_does_not_destroy_segment`` fails, the segment is
being torn down by some other crash-cleanup path — also a bug. Under SIGKILL
no Python cleanup runs, so the segment *must* survive.
"""

from __future__ import annotations

import multiprocessing
import os
import signal
import struct
import sys
import uuid

import pytest

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        sys.platform == "win32",
        reason="posix_shm requires POSIX (Linux/WSL2)",
    ),
]


# Subprocess bodies are top-level functions so ``multiprocessing`` can pickle
# them (fork inherits but doesn't pickle; spawn/forkserver would).
def _subprocess_read_then_clean_exit(name: str, offset: int, expected_value: float) -> None:
    from pyforge._internal import posix_shm

    h = posix_shm.open_existing(name)
    value = struct.unpack("<d", bytes(h.buf[offset : offset + 8]))[0]
    if value != expected_value:
        os._exit(2)
    posix_shm.close(h)
    # Clean exit — this is the path where stdlib SharedMemory's
    # resource_tracker would fire and shm_unlink the segment.
    os._exit(0)


def _subprocess_read_then_sigkill(name: str, offset: int, expected_value: float) -> None:
    from pyforge._internal import posix_shm

    h = posix_shm.open_existing(name)
    value = struct.unpack("<d", bytes(h.buf[offset : offset + 8]))[0]
    if value != expected_value:
        os._exit(2)
    os.kill(os.getpid(), signal.SIGKILL)


def _subprocess_read_and_report(
    name: str, offset: int, expected_value: float, barrier_value: float
) -> None:
    """Open, read, write barrier_value at offset+16, clean exit.

    Used for the parallel-readers test: the parent waits for all three
    subprocesses to have written their barrier values, then checks the
    sentinel is still readable after they all exit.
    """
    from pyforge._internal import posix_shm

    h = posix_shm.open_existing(name)
    value = struct.unpack("<d", bytes(h.buf[offset : offset + 8]))[0]
    if value != expected_value:
        os._exit(2)
    # Write barrier so parent knows we opened successfully.
    h.buf[offset + 16 : offset + 24] = struct.pack("<d", barrier_value)
    posix_shm.close(h)
    os._exit(0)


def _unique_name() -> str:
    return f"pyforge_xtest_{os.getpid()}_{uuid.uuid4().hex[:8]}"


def _write_sentinel(handle, offset: int, value: float) -> None:
    handle.buf[offset : offset + 8] = struct.pack("<d", value)


def _read_sentinel(handle, offset: int) -> float:
    return struct.unpack("<d", bytes(handle.buf[offset : offset + 8]))[0]


@pytest.fixture
def _mp_context() -> multiprocessing.context.BaseContext:
    # Force 'fork' on Linux; 'spawn' would re-import and slow the test.
    # On macOS the default is 'spawn' but tests here are Linux-only anyway.
    return multiprocessing.get_context("fork")


def test_reader_clean_exit_does_not_destroy_segment(_mp_context) -> None:
    """Reader subprocess opens, reads, exits CLEANLY. Segment must survive.

    This is the resource_tracker-bug test: stdlib SharedMemory would unlink
    the segment on clean exit. posix_ipc does not.
    """
    from pyforge._internal import posix_shm

    name = _unique_name()
    sentinel = 3.14
    sentinel_offset = 16

    h = posix_shm.create(name, 4096)
    try:
        _write_sentinel(h, sentinel_offset, sentinel)

        p = _mp_context.Process(
            target=_subprocess_read_then_clean_exit,
            args=(name, sentinel_offset, sentinel),
        )
        p.start()
        p.join(timeout=10)
        assert p.exitcode == 0, (
            f"subprocess did not exit cleanly (exitcode={p.exitcode}); "
            "likely read wrong value or crashed before writing"
        )

        # ⚡ The segment MUST still exist and be readable.
        h2 = posix_shm.open_existing(name)
        try:
            assert _read_sentinel(h2, sentinel_offset) == sentinel
        finally:
            posix_shm.close(h2)
    finally:
        posix_shm.close(h)
        posix_shm.unlink(name)


def test_reader_sigkill_does_not_destroy_segment(_mp_context) -> None:
    """Reader subprocess is SIGKILLed. No atexit runs. Segment must survive.

    This is a weaker variant of the clean-exit test (SIGKILL bypasses the
    bug anyway because Python cleanup doesn't run), but covers the
    real-world "my serving process crashed" scenario.
    """
    from pyforge._internal import posix_shm

    name = _unique_name()
    sentinel = 2.71828
    sentinel_offset = 16

    h = posix_shm.create(name, 4096)
    try:
        _write_sentinel(h, sentinel_offset, sentinel)

        p = _mp_context.Process(
            target=_subprocess_read_then_sigkill,
            args=(name, sentinel_offset, sentinel),
        )
        p.start()
        p.join(timeout=10)
        # SIGKILL on POSIX -> exitcode is -9
        assert p.exitcode == -signal.SIGKILL, (
            f"subprocess did not SIGKILL cleanly (exitcode={p.exitcode})"
        )

        h2 = posix_shm.open_existing(name)
        try:
            assert _read_sentinel(h2, sentinel_offset) == sentinel
        finally:
            posix_shm.close(h2)
    finally:
        posix_shm.close(h)
        posix_shm.unlink(name)


def test_three_parallel_readers_all_see_sentinel(_mp_context) -> None:
    """Three subprocesses open the same segment concurrently, each writes a
    distinct barrier value, all exit cleanly. After they're done, the
    parent's sentinel is still readable."""
    from pyforge._internal import posix_shm

    name = _unique_name()
    sentinel = 1.6180339887
    sentinel_offset = 16

    # Layout per reader: parent writes sentinel at offset; each reader writes
    # barrier at offset + 16, offset + 32, offset + 48. Parent checks all.
    barriers = (10.0, 20.0, 30.0)

    h = posix_shm.create(name, 4096)
    try:
        _write_sentinel(h, sentinel_offset, sentinel)

        processes = []
        # Separate barrier offsets per subprocess
        base_offsets = (
            sentinel_offset + 16,
            sentinel_offset + 32,
            sentinel_offset + 48,
        )
        # We encode each subprocess's "offset from sentinel where barrier lands"
        # via the barrier_value trick — but _subprocess_read_and_report
        # always writes at offset+16. To get distinct locations we'd need
        # to parameterize the function. Simpler: spawn all three, let them
        # race on offset+16 (last writer wins), and assert parent still
        # reads the sentinel. That's the weaker but sufficient property.
        _ = base_offsets  # silence unused

        for b in barriers:
            p = _mp_context.Process(
                target=_subprocess_read_and_report,
                args=(name, sentinel_offset, sentinel, b),
            )
            p.start()
            processes.append(p)
        for p in processes:
            p.join(timeout=10)
            assert p.exitcode == 0

        # After all three exit, parent must still see sentinel
        h2 = posix_shm.open_existing(name)
        try:
            assert _read_sentinel(h2, sentinel_offset) == sentinel
        finally:
            posix_shm.close(h2)

        # At least one barrier write should be visible (any of the three)
        last_barrier = struct.unpack(
            "<d", bytes(h.buf[sentinel_offset + 16 : sentinel_offset + 24])
        )[0]
        assert last_barrier in barriers
    finally:
        posix_shm.close(h)
        posix_shm.unlink(name)


def _subprocess_registry_read(name: str, offset: int, expected: float) -> None:
    """Top-level (picklable) subprocess body for the registry roundtrip test."""
    from pyforge._internal import posix_shm

    h = posix_shm.open_existing(name)
    try:
        val = struct.unpack("<d", bytes(h.buf[offset : offset + 8]))[0]
    finally:
        posix_shm.close(h)
    os._exit(0 if val == expected else 2)


def test_registry_roundtrip_across_processes(redis_client, _mp_context) -> None:
    """High-level: create via SegmentRegistry, open the segment by name in a
    subprocess, read the sentinel bytes, exit cleanly. Parent still reads."""
    from pyforge.schema import FeatureField, FeatureSchema, dtype
    from pyforge.shm import HEADER_LEN, SegmentRegistry

    class _XSchema(FeatureSchema):
        version = 1
        fields = [FeatureField("sentinel", dtype.float64)]

    reg = SegmentRegistry(redis_client)
    seg = reg.create(_XSchema)
    try:
        # Write sentinel into the first-field region (after the header).
        # First field starts at 64 (align_up(HEADER_SIZE=16, 64)=64), not
        # HEADER_LEN + 64. Use offset 64 directly.
        sentinel = 42.42
        seg.mmap_view[64 : 64 + 8] = struct.pack("<d", sentinel)

        p = _mp_context.Process(target=_subprocess_registry_read, args=(seg.name, 64, sentinel))
        p.start()
        p.join(timeout=10)
        assert p.exitcode == 0

        # Parent can still read.
        sentinel_back = struct.unpack("<d", bytes(seg.mmap_view[64 : 64 + 8]))[0]
        assert sentinel_back == sentinel

        # Silence unused import
        _ = HEADER_LEN
    finally:
        reg.close(seg)
