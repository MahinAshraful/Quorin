"""Unit tests for quorin._internal.hash_kernel.blake2b_8.

The kernel must produce byte-identical output to ``hashlib.blake2b(input,
digest_size=8)`` for any input. Pinned-hash invariant #5 locks this:
changing the algorithm silently breaks every persisted segment.

Two layers of coverage:
1. **Pinned hashes** — assert literal uint64 output for known inputs that
   match ``tests/unit/test_layout.py::TestHashPinned``'s reference values.
   This is the same set the Python ``hash_entity_id`` is locked against.
2. **Parity property** — Hypothesis-generated random byte strings 0-256
   bytes; both implementations must agree.
"""

from __future__ import annotations

import hashlib
import sys

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

pytestmark = pytest.mark.skipif(
    sys.platform == "win32",
    reason="hash_kernel requires POSIX (Linux/WSL2) for the lookup-jit hot path",
)

from quorin._internal.hash_kernel import blake2b_8, prewarm  # noqa: E402

# ---------------------------------------------------------------------------
# Pre-warm so the first test doesn't pay ~100-200 ms compile cost.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module", autouse=True)
def _prewarm_hash_kernel() -> None:
    prewarm()


# ---------------------------------------------------------------------------
# Pinned hashes (mirror tests/unit/test_layout.py::TestHashPinned).
# ---------------------------------------------------------------------------


def _ref_hash(s: str) -> int:
    return int.from_bytes(
        hashlib.blake2b(s.encode("utf-8"), digest_size=8).digest(),
        "little",
        signed=False,
    )


def _kernel_hash(s: str) -> int:
    encoded = s.encode("utf-8")
    arr = np.frombuffer(encoded, dtype=np.uint8).copy()
    return int(blake2b_8(arr, len(encoded)))


@pytest.mark.parametrize(
    "ident",
    ["user_001", "abc", "550e8400-e29b-41d4-a716-446655440000"],
)
def test_known_pinned_hashes_ascii(ident: str) -> None:
    """Match the same locked set as test_layout.py::test_known_pinned_hashes."""
    assert _kernel_hash(ident) == _ref_hash(ident)


def test_unicode_id_hashed_as_utf8() -> None:
    """UTF-8 multi-byte input matches Python's hashlib output."""
    ident = "ユーザー_001"
    assert _kernel_hash(ident) == _ref_hash(ident)


def test_empty_input() -> None:
    """Edge case: empty input still returns a valid hash (one F() call with all-zero block)."""
    expected = int.from_bytes(hashlib.blake2b(b"", digest_size=8).digest(), "little", signed=False)
    arr = np.zeros(0, dtype=np.uint8)
    assert int(blake2b_8(arr, 0)) == expected


def test_single_byte() -> None:
    arr = np.array([0xAB], dtype=np.uint8)
    expected = int.from_bytes(
        hashlib.blake2b(bytes([0xAB]), digest_size=8).digest(), "little", signed=False
    )
    assert int(blake2b_8(arr, 1)) == expected


def test_exactly_128_bytes() -> None:
    """Boundary: 128 bytes = exactly one block; finalize on the only block."""
    data = bytes(range(128))
    arr = np.frombuffer(data, dtype=np.uint8).copy()
    expected = int.from_bytes(hashlib.blake2b(data, digest_size=8).digest(), "little", signed=False)
    assert int(blake2b_8(arr, 128)) == expected


def test_129_bytes_first_multiblock() -> None:
    """Boundary: 129 bytes triggers the multi-block path (one full block + 1-byte tail)."""
    data = bytes(range(129)) if False else bytes([i & 0xFF for i in range(129)])
    arr = np.frombuffer(data, dtype=np.uint8).copy()
    expected = int.from_bytes(hashlib.blake2b(data, digest_size=8).digest(), "little", signed=False)
    assert int(blake2b_8(arr, 129)) == expected


def test_256_bytes_two_blocks() -> None:
    """Two full blocks; second block is also the finalize block."""
    data = bytes([i & 0xFF for i in range(256)])
    arr = np.frombuffer(data, dtype=np.uint8).copy()
    expected = int.from_bytes(hashlib.blake2b(data, digest_size=8).digest(), "little", signed=False)
    assert int(blake2b_8(arr, 256)) == expected


def test_n_param_independent_of_buffer_length() -> None:
    """``n`` controls how many bytes are hashed, even if the buffer is larger.

    Locks the contract that lookup_jit can pass a max_id_bytes-sized scratch
    buffer with variable fill length without poisoning the hash.
    """
    buf = np.zeros(64, dtype=np.uint8)
    buf[:5] = np.frombuffer(b"hello", dtype=np.uint8)
    expected = int.from_bytes(
        hashlib.blake2b(b"hello", digest_size=8).digest(), "little", signed=False
    )
    assert int(blake2b_8(buf, 5)) == expected


# ---------------------------------------------------------------------------
# Parity property — Hypothesis-generated random byte strings.
# ---------------------------------------------------------------------------


_HYPO = settings(
    max_examples=200,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture],
)


@_HYPO
@given(data=st.binary(min_size=0, max_size=4096))
def test_parity_against_hashlib(data: bytes) -> None:
    """For any random input ≤ 4 KB (covers 0, single-block,
    multi-block, and realistic-max-entity-id paths), the Numba kernel
    matches hashlib.blake2b byte-for-byte.

    QW-2 (v0.1.1): bumped from 300 → 4096. Real entity_ids can reach
    max_id_bytes (default 64, configurable up to MAX_ID_BYTES_CEILING
    = 64 KiB per CR.H.4); 4 KB covers compound-key + UUID-prefix +
    namespace tenants without making the test slow.
    """
    arr = np.frombuffer(data, dtype=np.uint8).copy() if data else np.zeros(0, dtype=np.uint8)
    expected = int.from_bytes(hashlib.blake2b(data, digest_size=8).digest(), "little", signed=False)
    actual = int(blake2b_8(arr, len(data)))
    assert actual == expected, (
        f"blake2b_8 disagrees with hashlib at len={len(data)}: "
        f"expected {expected:#018x}, got {actual:#018x}"
    )
