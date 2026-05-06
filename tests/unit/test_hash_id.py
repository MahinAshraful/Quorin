"""Tests for ``quorin._internal.hash_id``.

The pinned-hash regression test (asserting literal blake2b output for known
strings) lives at ``tests/unit/test_layout.py::TestHashPinned::test_known_pinned_hashes``;
this file covers the bytes-input parity contract added for Step 13's bulk
hash loop.
"""

from __future__ import annotations

import numpy as np

from quorin._internal.hash_id import hash_entity_id, hash_entity_id_bytes


class TestHashEntityIdBytesParity:
    """``hash_entity_id_bytes`` must produce identical output to
    ``hash_entity_id`` for any input that round-trips through UTF-8."""

    def test_ascii_strings(self) -> None:
        for s in ["", "a", "abc", "user-12345", "x" * 256]:
            assert hash_entity_id_bytes(s.encode("utf-8")) == hash_entity_id(s)

    def test_unicode_strings(self) -> None:
        # Multi-byte UTF-8: the bytes-input variant should still match the
        # str-input variant since the str-input variant encodes internally.
        # The Greek letters are intentional non-ASCII test inputs.
        for s in ["café", "emoji-🔥", "日本語", "mixed-α-β-γ"]:  # noqa: RUF001
            assert hash_entity_id_bytes(s.encode("utf-8")) == hash_entity_id(s)

    def test_accepts_memoryview(self) -> None:
        s = "user-42"
        encoded = s.encode("utf-8")
        assert hash_entity_id_bytes(memoryview(encoded)) == hash_entity_id(s)

    def test_accepts_bytearray(self) -> None:
        s = "user-42"
        assert hash_entity_id_bytes(bytearray(s, "utf-8")) == hash_entity_id(s)

    def test_accepts_numpy_uint8_slice(self) -> None:
        # The hydration bulk-hash loop slices a flat utf-8 buffer pulled
        # from PyArrow's StringArray.values. Verify a numpy uint8 slice
        # works without an explicit conversion.
        s = "user-42"
        encoded = s.encode("utf-8")
        # Embed the encoded bytes in a larger buffer with surrounding noise
        # to verify the slice is interpreted correctly.
        buf = np.zeros(100, dtype=np.uint8)
        buf[10 : 10 + len(encoded)] = np.frombuffer(encoded, dtype=np.uint8)
        sliced = buf[10 : 10 + len(encoded)]
        assert hash_entity_id_bytes(sliced) == hash_entity_id(s)

    def test_returns_unsigned_64bit(self) -> None:
        # Hashes must fit in uint64 for the slot table.
        for s in ["a", "abc", "user-12345"]:
            h = hash_entity_id_bytes(s.encode("utf-8"))
            assert 0 <= h < 2**64

    def test_empty_buffer_does_not_raise(self) -> None:
        # blake2b on empty input is well-defined; we just verify it
        # produces some uint64 value matching the str variant.
        assert hash_entity_id_bytes(b"") == hash_entity_id("")
