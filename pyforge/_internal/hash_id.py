"""Entity-ID hashing.

Pyforge uses ``blake2b(digest_size=8)`` for both schema field names (Step 1,
``pyforge.schema._hash_name``) and entity IDs (this module). They live in
separate functions to keep the namespace boundaries clean — name-hash collisions
and entity-id-hash collisions are unrelated failure modes — and so that
Step 5's Numba-compiled lookup can import a tight, dependency-free helper.

Pinned forever: changing the algorithm or digest size would silently break
every persisted segment, so the pinned-hash test in
``tests/unit/test_layout.py`` asserts known outputs for known inputs.
"""

from __future__ import annotations

import hashlib


def hash_entity_id(entity_id: str) -> int:
    """64-bit unsigned hash of an entity ID.

    UTF-8 encodes the input, runs ``blake2b`` with an 8-byte digest, and
    interprets the result as a little-endian unsigned 64-bit integer.

    Returns:
        An ``int`` in ``[0, 2**64)``. Stored as ``np.uint64`` in the slot
        table; equality comparisons are integer-fast.
    """
    return int.from_bytes(
        hashlib.blake2b(entity_id.encode("utf-8"), digest_size=8).digest(),
        byteorder="little",
        signed=False,
    )
