"""Unit tests for pyforge._internal.posix_shm.

These are low-level tests of the POSIX wrapper — no Redis, no Segment, no
schema. They prove the wrapper correctly bypasses CPython's resource_tracker
and that create/open/close/unlink behave as documented.

Linux/macOS only; the module fails to import on native Windows.
"""

from __future__ import annotations

import os
import struct
import sys
import uuid

import pytest

pytestmark = pytest.mark.skipif(
    sys.platform == "win32",
    reason="posix_shm requires POSIX (Linux/WSL2/macOS)",
)

# Imports that require posix_ipc are placed inside the conditional guard so
# collection on Windows doesn't explode.
from pyforge._internal import posix_shm  # noqa: E402


def _unique_name(prefix: str = "pyforge_test") -> str:
    return f"{prefix}_{os.getpid()}_{uuid.uuid4().hex[:8]}"


class TestCreate:
    def test_create_returns_handle_with_requested_size(self) -> None:
        name = _unique_name()
        h = posix_shm.create(name, 4096)
        try:
            assert h.name == name
            assert h.size == 4096
            assert len(h.buf) == 4096
        finally:
            posix_shm.close(h)
            posix_shm.unlink(name)

    def test_newly_created_segment_is_zero_filled(self) -> None:
        name = _unique_name()
        h = posix_shm.create(name, 4096)
        try:
            assert bytes(h.buf[:128]) == b"\x00" * 128
        finally:
            posix_shm.close(h)
            posix_shm.unlink(name)

    def test_create_duplicate_name_raises(self) -> None:
        import posix_ipc

        name = _unique_name()
        h1 = posix_shm.create(name, 4096)
        try:
            with pytest.raises(posix_ipc.ExistentialError):
                posix_shm.create(name, 4096)
        finally:
            posix_shm.close(h1)
            posix_shm.unlink(name)


class TestOpenExisting:
    def test_open_nonexistent_raises_file_not_found(self) -> None:
        with pytest.raises(FileNotFoundError):
            posix_shm.open_existing(_unique_name("pyforge_test_missing"))

    def test_open_then_read_written_bytes(self) -> None:
        name = _unique_name()
        h1 = posix_shm.create(name, 4096)
        try:
            h1.buf[0:8] = struct.pack("<Q", 0xDEADBEEFCAFEBABE)
            h2 = posix_shm.open_existing(name)
            try:
                value = struct.unpack("<Q", bytes(h2.buf[0:8]))[0]
                assert value == 0xDEADBEEFCAFEBABE
            finally:
                posix_shm.close(h2)
        finally:
            posix_shm.close(h1)
            posix_shm.unlink(name)

    def test_open_reports_correct_size(self) -> None:
        name = _unique_name()
        h1 = posix_shm.create(name, 8192)
        try:
            h2 = posix_shm.open_existing(name)
            try:
                assert h2.size == 8192
                assert len(h2.buf) == 8192
            finally:
                posix_shm.close(h2)
        finally:
            posix_shm.close(h1)
            posix_shm.unlink(name)


class TestCloseSeparateFromUnlink:
    def test_close_does_not_unlink_segment(self) -> None:
        """The whole point of this module: closing a handle must not destroy
        the segment. Another caller must still be able to open it."""
        name = _unique_name()
        h1 = posix_shm.create(name, 4096)
        try:
            h1.buf[0:4] = b"TEST"
            posix_shm.close(h1)

            # Segment should still exist — open it again.
            h2 = posix_shm.open_existing(name)
            try:
                assert bytes(h2.buf[0:4]) == b"TEST"
            finally:
                posix_shm.close(h2)
        finally:
            posix_shm.unlink(name)

    def test_close_is_idempotent(self) -> None:
        name = _unique_name()
        h = posix_shm.create(name, 4096)
        try:
            posix_shm.close(h)
            posix_shm.close(h)  # second call must not raise
        finally:
            posix_shm.unlink(name)


class TestUnlink:
    def test_unlink_removes_the_segment(self) -> None:
        name = _unique_name()
        h = posix_shm.create(name, 4096)
        posix_shm.close(h)
        posix_shm.unlink(name)
        with pytest.raises(FileNotFoundError):
            posix_shm.open_existing(name)

    def test_unlink_on_missing_name_is_idempotent(self) -> None:
        # Must not raise — Step 14's watchdog relies on this.
        posix_shm.unlink(_unique_name("pyforge_test_ghost"))
