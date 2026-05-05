"""Numba-compiled BLAKE2b-8 hash for the serving hot path (Step 16c "16d").

Mirror image of :mod:`pyforge._internal.hash_id`'s ``hash_entity_id`` —
same input (UTF-8 bytes), same output (little-endian uint64 from the
8-byte BLAKE2b digest). The Python implementation calls
``hashlib.blake2b(buf, digest_size=8)`` (~1500 ns); this Numba kernel
replicates BLAKE2b's compression function in pure Numba and runs at
~150-300 ns per call after the first compile.

Why this exists
---------------
After Step 16c Commit A's lookup-jit shipment, the 4-field warm assemble
WSL2 measurement landed at 3.71 µs median (was 4.17 µs in Step 5). The
trip-wire bench p99 was projected to land at ~6.5-7 µs on native CI —
still missing the 5 µs spec by 30-40%. Per profile decomposition:

    - lookup_jit kernel + dispatch:   ~350 ns  (post-Step-16c)
    - hashlib.blake2b in wrapper:    ~1500 ns  (THE remaining bottleneck)
    - encode + .copy():              ~150 ns
    - _assemble_core kernel:         ~500 ns
    - frombuffer + return:           ~500 ns
    - other Python overhead:         ~300 ns

Removing the hashlib.blake2b wrapper call (~1300 ns saved per assemble)
puts the 4-field warm Numba path at ~2.4 µs median + ~5-6 µs p99 on
native CI — meeting the 5 µs spec at p99. Per the project mandate
(CLAUDE.md §1: "the most performant Python feature serving library
possible"), shipping a 5 µs spec-met-at-p99 number is the right
engineering call vs. shipping 16c with a documented near-miss.

Pinned-hash invariant #5
------------------------
The hash algorithm is BLAKE2b with digest_size=8, byte-output
little-endian-interpreted as uint64. **This module reimplements the
algorithm byte-for-byte; it does NOT change anything.** The pinned-hash
regression tests in ``tests/unit/test_hash_kernel.py`` and
``tests/unit/test_layout.py::TestHashPinned`` assert literal output for
known inputs.

Module isolation (invariant #11)
--------------------------------
Lives at ``pyforge/_internal/hash_kernel.py`` (NOT in
``pyforge/_internal/hash_id.py``) because ``hash_id`` is imported by
``pyforge.layout``, ``pyforge.serving``, etc. — non-Numba modules.
``hash_kernel`` only gets pulled by Numba-aware modules
(``pyforge._internal.lookup_kernel``, eventually
``pyforge.assembly.assemble_batch``).

Implementation notes
--------------------
- **Single-block fast path for n ≤ 128 bytes** (covers entity_ids ≤ 128;
  default max_id_bytes = 64). Multi-block path handles up to ~16 KB
  for parity-test coverage.
- **G mixing function inlined** via ``inline='always'`` decorator
  hint. ~480 lines of explicit code per round x 12 rounds is fine for
  Numba's LLVM lowering; the alternative (Python loop calling G) loses
  vectorization opportunities.
- **No Python objects or strings inside the kernel.** Inputs are
  ``uint8[::1]`` (the encoded UTF-8 bytes); output is ``uint64``.
"""

from __future__ import annotations

from typing import Any

import numba
import numpy as np

# ---------------------------------------------------------------------------
# BLAKE2b constants (RFC 7693 §2.6).
# ---------------------------------------------------------------------------

#: BLAKE2b initialization vector — first 64 bits of the fractional parts of
#: the square roots of the first 8 primes (2, 3, 5, 7, 11, 13, 17, 19).
_BLAKE2B_IV = np.array(
    [
        0x6A09E667F3BCC908,
        0xBB67AE8584CAA73B,
        0x3C6EF372FE94F82B,
        0xA54FF53A5F1D36F1,
        0x510E527FADE682D1,
        0x9B05688C2B3E6C1F,
        0x1F83D9ABFB41BD6B,
        0x5BE0CD19137E2179,
    ],
    dtype=np.uint64,
)

#: Parameter-block XOR for digest_length=8, key_length=0, fanout=1, depth=1,
#: leaf_length=0, node_offset=0, node_depth=0, inner_length=0.
#: Encoded as a single uint64 covering bytes 0-7 of the 64-byte param block:
#:   byte 0 = digest_length (0x08)
#:   byte 1 = key_length (0x00)
#:   byte 2 = fanout (0x01)
#:   byte 3 = depth (0x01)
#:   bytes 4-7 = leaf_length (0x00000000)
#: Little-endian: 0x01010008.
_PARAM_XOR = np.uint64(0x0000000001010008)

#: BLAKE2b sigma permutation table — 12 rounds x 16 indices each. RFC 7693
#: defines 10 distinct rounds; rounds 10 and 11 reuse rounds 0 and 1.
_SIGMA = np.array(
    [
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
        [14, 10, 4, 8, 9, 15, 13, 6, 1, 12, 0, 2, 11, 7, 5, 3],
        [11, 8, 12, 0, 5, 2, 15, 13, 10, 14, 3, 6, 7, 1, 9, 4],
        [7, 9, 3, 1, 13, 12, 11, 14, 2, 6, 5, 10, 4, 0, 15, 8],
        [9, 0, 5, 7, 2, 4, 10, 15, 14, 1, 11, 12, 6, 8, 3, 13],
        [2, 12, 6, 10, 0, 11, 8, 3, 4, 13, 7, 5, 15, 14, 1, 9],
        [12, 5, 1, 15, 14, 13, 4, 10, 0, 7, 6, 3, 9, 2, 8, 11],
        [13, 11, 7, 14, 12, 1, 3, 9, 5, 0, 15, 4, 8, 6, 2, 10],
        [6, 15, 14, 9, 11, 3, 0, 8, 12, 2, 13, 7, 1, 4, 10, 5],
        [10, 2, 8, 4, 7, 6, 1, 5, 15, 11, 9, 14, 3, 12, 13, 0],
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
        [14, 10, 4, 8, 9, 15, 13, 6, 1, 12, 0, 2, 11, 7, 5, 3],
    ],
    dtype=np.int64,
)


# ---------------------------------------------------------------------------
# BLAKE2b primitives.
# ---------------------------------------------------------------------------


@numba.njit(  # type: ignore[untyped-decorator]
    "uint64(uint64, int64)",
    cache=True,
    boundscheck=False,
    fastmath=False,
    inline="always",
)
def _rotr64(x: int, n: int) -> int:
    """Right-rotate uint64 ``x`` by ``n`` bits."""
    # mypy sees np.uint64 ops as signedinteger[Any]; Numba's signature
    # ("uint64(uint64, int64)") constrains the runtime types.
    return (x >> np.uint64(n)) | (x << np.uint64(64 - n))  # type: ignore[return-value]


@numba.njit(  # type: ignore[untyped-decorator]
    "void(uint64[::1], int64, int64, int64, int64, uint64, uint64)",
    cache=True,
    boundscheck=False,
    fastmath=False,
    inline="always",
)
def _G(  # noqa: N802 — RFC 7693 §3.1 names the mixing function "G"; preserve the standard name
    v: np.ndarray[Any, np.dtype[np.uint64]],
    a: int,
    b: int,
    c: int,
    d: int,
    x: int,
    y: int,
) -> None:
    """BLAKE2b mixing function G (RFC 7693 §3.1).

    Mutates ``v`` in place; relies on uint64 overflow wrapping for the
    additions (Numba's uint64 arithmetic matches C semantics here).
    """
    v[a] = v[a] + v[b] + x
    v[d] = _rotr64(v[d] ^ v[a], 32)
    v[c] = v[c] + v[d]
    v[b] = _rotr64(v[b] ^ v[c], 24)
    v[a] = v[a] + v[b] + y
    v[d] = _rotr64(v[d] ^ v[a], 16)
    v[c] = v[c] + v[d]
    v[b] = _rotr64(v[b] ^ v[c], 63)


@numba.njit(  # type: ignore[untyped-decorator]
    "void(uint64[::1], uint64[::1], uint64, boolean)",
    cache=True,
    boundscheck=False,
    fastmath=False,
)
def _F(  # noqa: N802 — RFC 7693 §3.2 names the compression function "F"; preserve the standard name
    h: np.ndarray[Any, np.dtype[np.uint64]],
    m: np.ndarray[Any, np.dtype[np.uint64]],
    t: int,
    finalize: bool,
) -> None:
    """BLAKE2b compression function F (RFC 7693 §3.2).

    Compresses one 128-byte message block (16 uint64 words ``m``) into
    state ``h``. ``t`` is the byte-counter (total bytes processed so
    far including this block). ``finalize`` flips the final-block flag.

    Mutates ``h`` in place. Single-counter form (high counter == 0) —
    suffices for inputs up to 2^64 bytes, far beyond any realistic
    entity_id.
    """
    v = np.empty(16, dtype=np.uint64)
    for i in range(8):
        v[i] = h[i]
        v[8 + i] = _BLAKE2B_IV[i]
    v[12] = v[12] ^ t
    # v[13] stays unchanged (high counter == 0 for our scale).
    if finalize:
        v[14] = v[14] ^ np.uint64(0xFFFFFFFFFFFFFFFF)

    for r in range(12):
        s = _SIGMA[r]
        _G(v, 0, 4, 8, 12, m[s[0]], m[s[1]])
        _G(v, 1, 5, 9, 13, m[s[2]], m[s[3]])
        _G(v, 2, 6, 10, 14, m[s[4]], m[s[5]])
        _G(v, 3, 7, 11, 15, m[s[6]], m[s[7]])
        _G(v, 0, 5, 10, 15, m[s[8]], m[s[9]])
        _G(v, 1, 6, 11, 12, m[s[10]], m[s[11]])
        _G(v, 2, 7, 8, 13, m[s[12]], m[s[13]])
        _G(v, 3, 4, 9, 14, m[s[14]], m[s[15]])

    for i in range(8):
        h[i] = h[i] ^ v[i] ^ v[i + 8]


# ---------------------------------------------------------------------------
# Public Numba kernel: blake2b digest_size=8, byte-input, uint64 output.
# ---------------------------------------------------------------------------


@numba.njit(  # type: ignore[untyped-decorator]
    "uint64(uint8[::1], int64)",
    cache=True,
    boundscheck=False,
    fastmath=False,
)
def blake2b_8(input_bytes: np.ndarray[Any, np.dtype[np.uint8]], n: int) -> int:
    """Compute ``blake2b(input_bytes[:n], digest_size=8)`` and return it as
    a little-endian uint64.

    Byte-identical to ::

        int.from_bytes(
            hashlib.blake2b(input_bytes[:n].tobytes(), digest_size=8).digest(),
            byteorder="little",
            signed=False,
        )

    Verified by ``tests/unit/test_hash_kernel.py``'s pinned-hash + Hypothesis
    parity tests. Pinned-hash invariant #5 forbids changing the algorithm;
    this kernel reimplements byte-for-byte, never changes.

    Args:
        input_bytes: contiguous uint8 array containing the message to hash.
        n: number of bytes to hash from the start of ``input_bytes``. Pass
            ``len(buf)`` for the typical case; allows passing a fixed-size
            scratch buffer with variable fill length.

    Returns:
        ``uint64`` — the low 8 bytes of the BLAKE2b state, little-endian.
    """
    # Initialize state h = IV with parameter block XOR.
    h = np.empty(8, dtype=np.uint64)
    for i in range(8):
        h[i] = _BLAKE2B_IV[i]
    h[0] = h[0] ^ _PARAM_XOR

    # Process all complete 128-byte blocks except the last. For n == 0 we still
    # need ONE F call with a zero block (handled by the final-block path below).
    # Final block is always present (even if < 128 bytes); n_full_blocks counts
    # complete 128-byte blocks BEFORE the final block.
    n_full_blocks = 0 if n == 0 else (n - 1) // 128

    m = np.empty(16, dtype=np.uint64)

    for b in range(n_full_blocks):
        block_offset = b * 128
        for w in range(16):
            word = np.uint64(0)
            wo = block_offset + w * 8
            for byte_idx in range(8):
                word = word | (np.uint64(input_bytes[wo + byte_idx]) << np.uint64(byte_idx * 8))
            m[w] = word
        _F(h, m, np.uint64(block_offset + 128), False)

    # Final block: pad with zeros to 128 bytes.
    final_offset = n_full_blocks * 128

    for w in range(16):
        word = np.uint64(0)
        wo = final_offset + w * 8
        for byte_idx in range(8):
            abs_idx = wo + byte_idx
            if abs_idx < n:
                word = word | (np.uint64(input_bytes[abs_idx]) << np.uint64(byte_idx * 8))
            # else: word stays 0 in this byte position (zero-pad)
        m[w] = word

    _F(h, m, np.uint64(n), True)

    # Output: low 8 bytes of h[0], little-endian → uint64 directly.
    # mypy sees h[0]'s indexing as Any; Numba's signature ("uint64(...)")
    # constrains the return type at runtime.
    return h[0]  # type: ignore[no-any-return]


# ---------------------------------------------------------------------------
# Pre-warm — opt-in, idempotent.
# ---------------------------------------------------------------------------


def prewarm() -> None:
    """Force compilation of :func:`blake2b_8` (and its inlined helpers)
    on a tiny synthetic input. ~100-200 ms on first call; subsequent
    calls are no-ops thanks to ``cache=True``.

    NOT auto-run at module import. :func:`pyforge.assembly.prewarm`
    calls into here so a single prewarm covers all assembly + lookup
    + hash kernels.
    """
    sample = np.zeros(1, dtype=np.uint8)
    blake2b_8(sample, 1)
