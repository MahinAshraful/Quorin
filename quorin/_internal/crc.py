"""CRC32 helper.

Wraps :func:`zlib.crc32` with explicit unsigned masking so the returned value
matches what we write into the segment header (uint32 little-endian).
"""

from __future__ import annotations

import zlib

_UINT32_MASK = 0xFFFF_FFFF


def crc32_of_bytes(buf: bytes | memoryview | bytearray) -> int:
    """Return the unsigned 32-bit CRC32 of ``buf``."""
    return zlib.crc32(buf) & _UINT32_MASK
