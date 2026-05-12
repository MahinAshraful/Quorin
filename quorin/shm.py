"""Shared-memory segment lifecycle: create, open, close, refcount.

A :class:`Segment` is a typed wrapper around a POSIX shm region containing:

::

    bytes  0..3  : magic = b"PYFG"
    bytes  4..7  : schema version (uint32 LE)
    bytes  8..11 : CRC32 of the compiled offset table (uint32 LE)
    bytes 12..15 : capacity (uint32 LE)         <- repurposed in Step 3
    bytes 16..   : metadata + slot table + string pool + feature rows  (Step 3)

The :class:`SegmentRegistry` maintains Redis bookkeeping so that multiple
processes can share segments without leaking them. Refcounts are
**per-open**, not per-read — one ``INCR`` at process start, one ``DECR`` at
process end. The hot read path never touches Redis.

Linux/WSL2 only. Importing this module on native Windows fails because
``posix_ipc`` has no Windows wheel.
"""

from __future__ import annotations

import contextlib
import errno
import os
import re
import struct
import uuid
import warnings
from dataclasses import dataclass
from typing import Final, cast

import redis

from quorin._internal import heartbeat, posix_shm
from quorin._internal.crc import crc32_of_bytes
from quorin.layout import (
    DEFAULT_MAX_ID_BYTES,
    SegmentLayout,
    compute_layout,
    compute_layout_from_segment,
    initialize_segment_regions,
)
from quorin.schema import FeatureSchema, compile_schema

# ---------------------------------------------------------------------------
# Header format — 16 bytes, fixed forever. Every Quorin segment starts with
# these bytes; mismatches refuse to open rather than read garbage.
#
# Step 3 repurposed the trailing 4-byte zero pad as ``capacity`` (uint32 LE).
# The CRC32 still covers only the compiled offset table, so this change
# doesn't alter the verification semantics.
# ---------------------------------------------------------------------------

MAGIC: Final[bytes] = b"PYFG"

# little-endian: magic[4] + version u32 + crc32 u32 + capacity u32
HEADER_FMT: Final[str] = "<4sIII"
HEADER_LEN: Final[int] = 16

assert struct.calcsize(HEADER_FMT) == HEADER_LEN, "header format must be 16 bytes"


# ---------------------------------------------------------------------------
# Errors.
# ---------------------------------------------------------------------------


class SchemaCRCMismatchError(RuntimeError):
    """Raised when an opened segment's stored CRC32 (or magic, or version)
    doesn't match what the locally-loaded schema would produce. Indicates two
    software versions disagree about the layout — refuse to read further."""


class SegmentNotFoundError(RuntimeError):
    """No segment is currently registered for this schema in Redis."""


# ---------------------------------------------------------------------------
# Redis key helpers — single source of truth for namespacing. Step 14 and 15
# read these same keys; do not change without updating both.
# ---------------------------------------------------------------------------


def _safe_class_name(name: str) -> str:
    """Sanitize a Python class name for safe use in a POSIX shm path."""
    return re.sub(r"[^A-Za-z0-9_]", "_", name)


def _segment_name(schema: type[FeatureSchema]) -> str:
    return f"quorin_{_safe_class_name(schema.__name__)}_v{schema.version}_{uuid.uuid4().hex[:8]}"


def _key_current(schema: type[FeatureSchema]) -> str:
    return f"quorin:schema:{_safe_class_name(schema.__name__)}:current"


def _key_refcount(segment_name: str) -> str:
    return f"quorin:refcount:{segment_name}"


def _key_pid_segments(pid: int) -> str:
    return f"quorin:pid_segments:{pid}"


KEY_CLEANUP_QUEUE: Final[str] = "quorin:cleanup_queue"

# CR.D.5 (v0.1.1): refuse to allocate a segment that would consume more
# than this fraction of currently-free /dev/shm. Operators expect this
# guard from the README's "fail loud, not at runtime" theme; it surfaces
# disk-pressure issues at create() rather than letting posix_shm.create
# succeed and the next mmap fault crash the process. Set to 0.5 = 50%.
_DEV_SHM_RESERVE_FRACTION: Final[float] = 0.5
_DEV_SHM_PATH: Final[str] = "/dev/shm"


def _check_dev_shm_capacity(total_size_bytes: int) -> None:
    """Refuse the allocation if it would exceed 50% of free /dev/shm.

    Raises :class:`OSError` with ``errno=ENOSPC`` and a clear message
    when the request is too large. Matches POSIX shm kernel semantics
    (``shm_open`` would have raised ENOSPC anyway, just at a less
    legible spot — typically a SIGBUS on the next mmap fault).

    On systems without ``/dev/shm`` (rare; non-Linux POSIX, container
    sandboxes), :func:`os.statvfs` raises :class:`FileNotFoundError`.
    We catch and emit a :class:`UserWarning`, then return — the
    subsequent ``posix_shm.create`` will fail with its own clearer
    error per invariant #9 (Linux/WSL2 only). Don't mask the
    wrong-platform signal by raising here.
    """
    try:
        st = os.statvfs(_DEV_SHM_PATH)
    except FileNotFoundError:
        warnings.warn(
            f"{_DEV_SHM_PATH} not found; skipping capacity guard. "
            "Quorin requires Linux/WSL2 per invariant #9. The next "
            "posix_shm.create call will fail with the platform-level "
            "error.",
            UserWarning,
            stacklevel=3,
        )
        return
    # f_bavail counts blocks available to non-privileged users; this is
    # the right measure for "what can we allocate." f_bfree includes
    # reserved-for-root blocks which we wouldn't get.
    free_bytes = int(st.f_bavail) * int(st.f_frsize)
    cap = int(free_bytes * _DEV_SHM_RESERVE_FRACTION)
    if total_size_bytes > cap:
        raise OSError(
            errno.ENOSPC,
            (
                f"Refusing to allocate {total_size_bytes:,} bytes in "
                f"{_DEV_SHM_PATH}: would exceed "
                f"{int(_DEV_SHM_RESERVE_FRACTION * 100)}% of free space "
                f"({free_bytes:,} bytes free; cap "
                f"{cap:,} bytes). Free more /dev/shm space, reduce "
                "segment capacity, or shard across multiple Quorin "
                "instances."
            ),
        )


# Step 14: sidetable so the watchdog can clear ``quorin:schema:{name}:current``
# pointers without an O(N keyspace) ``KEYS`` scan. Written by
# :meth:`SegmentRegistry.create`, read + cleared by the watchdog's dead-PID
# Lua and the close-Lua extension at refcount-0.
KEY_SEGMENT_TO_SCHEMA: Final[str] = "quorin:segment_to_schema"


# ---------------------------------------------------------------------------
# Segment + registry.
# ---------------------------------------------------------------------------


@dataclass
class Segment:
    """A live, mapped Quorin segment.

    The ``layout`` field is computed once at create/open time and cached so
    that hot-path lookups don't recompute slot-table offsets per call.
    """

    name: str
    schema: type[FeatureSchema]
    handle: posix_shm.SegmentHandle
    layout: SegmentLayout

    @property
    def mmap_view(self) -> memoryview:
        """Direct memoryview of the mmap'd region (zero-copy)."""
        return self.handle.buf


# Atomic close: DECR refcount, SADD-to-cleanup-queue if it hit zero, SREM
# from this process's pid_segments set. One Redis round trip, atomic.
#
# Step 14 extension at refcount-0: clear the sidetable + the schema:current
# pointer (only if it still equals this segment name — rotation-safe). This
# keeps live-process closes that hit refcount-0 symmetric with the watchdog's
# dead-PID Lua. KEYS[4] = quorin:segment_to_schema. ARGV[1] = segment name.
_CLOSE_LUA: Final[str] = """
local n = redis.call('DECR', KEYS[1])
if tonumber(n) <= 0 then
    redis.call('SADD', KEYS[3], ARGV[1])
    local schema = redis.call('HGET', KEYS[4], ARGV[1])
    if schema then
        local current_key = 'quorin:schema:' .. schema .. ':current'
        local current = redis.call('GET', current_key)
        if current == ARGV[1] then
            redis.call('DEL', current_key)
        end
        redis.call('HDEL', KEYS[4], ARGV[1])
    end
    redis.call('DEL', KEYS[1])
end
redis.call('SREM', KEYS[2], ARGV[1])
return n
"""


class SegmentRegistry:
    """Cross-process registry for Quorin shm segments.

    A single instance per process is enough; multiple are harmless. The
    registry holds no in-memory state about who-has-what — Redis is the
    source of truth.
    """

    def __init__(self, redis_client: redis.Redis) -> None:
        self._redis = redis_client
        self._close_script = redis_client.register_script(_CLOSE_LUA)

    # ----- create -----------------------------------------------------------

    def create(
        self,
        schema: type[FeatureSchema],
        *,
        capacity: int,
        max_id_bytes: int = DEFAULT_MAX_ID_BYTES,
        set_current: bool = True,
    ) -> Segment:
        """Allocate a new multi-entity segment for ``schema`` with the given
        ``capacity`` and per-ID byte budget.

        Args:
            set_current: If ``True`` (default; matches Step 1-14 behavior),
                the create pipeline writes ``quorin:schema:{safe_name}:current``
                to the new segment's name, making the segment immediately
                visible to ``open_current``. Step 15 schema evolution passes
                ``False`` so it can populate the new segment via
                ``insert_many`` and only then atomically flip ``schema:current``
                via the CAS Lua script — without a window where consumers see
                the new segment empty. All other pipeline ops (refcount,
                pid_segments SADD, sidetable HSET) run unchanged.

        On any failure after :func:`posix_shm.create` succeeds, the segment is
        released and unlinked so ``/dev/shm`` does not leak.
        """
        # Step 14: heartbeat thread for this process. Idempotent — first
        # call wins, subsequent calls return immediately.
        heartbeat.ensure_started(self._redis, os.getpid())
        layout = compute_layout(schema, capacity=capacity, max_id_bytes=max_id_bytes)
        # CR.D.5 (v0.1.1): refuse before posix_shm.create if /dev/shm
        # is too full. Surfaces disk-pressure issues at create() rather
        # than letting the segment allocation succeed and SIGBUS on the
        # next mmap fault. Raises OSError(ENOSPC) — no new public
        # exception class.
        _check_dev_shm_capacity(layout.total_size)
        name = _segment_name(schema)
        handle = posix_shm.create(name, layout.total_size)
        try:
            offset_table = compile_schema(schema)
            crc = crc32_of_bytes(offset_table.tobytes())
            header = struct.pack(
                HEADER_FMT,
                MAGIC,
                int(schema.version),
                crc,
                capacity,
            )
            handle.buf[:HEADER_LEN] = header

            # Step 3 metadata + string-pool region header.
            initialize_segment_regions(handle.buf, layout)

            pid = os.getpid()
            with self._redis.pipeline(transaction=True) as p:
                # Step 15: set_current=False omits the schema:current write so
                # an upgrade orchestrator can populate the new segment before
                # exposing it. All other ops are unchanged so refcount /
                # pid_segments / sidetable invariants hold either way.
                if set_current:
                    p.set(_key_current(schema), name)
                p.set(_key_refcount(name), 1)
                p.sadd(_key_pid_segments(pid), name)
                # Step 14: sidetable so the watchdog (and the close-Lua
                # extension) can clear schema:current at refcount-0 without
                # an O(N keyspace) KEYS scan.
                p.hset(KEY_SEGMENT_TO_SCHEMA, name, _safe_class_name(schema.__name__))
                p.execute()
        except BaseException:
            with contextlib.suppress(Exception):
                posix_shm.close(handle)
            with contextlib.suppress(Exception):
                posix_shm.unlink(name)
            raise

        return Segment(name=name, schema=schema, handle=handle, layout=layout)

    # ----- open -------------------------------------------------------------

    def open_current(self, schema: type[FeatureSchema]) -> Segment:
        """Open the schema's currently-active segment.

        Verifies header magic, schema version, and CRC32 against the local
        schema; reads ``capacity`` and ``max_id_bytes`` out of the segment to
        derive a fresh :class:`SegmentLayout`. INCRs refcount and adds the
        segment to this PID's held set.

        Raises:
            SegmentNotFoundError: when no current segment is registered.
            SchemaCRCMismatchError: when stored magic/version/CRC differ from local.
        """
        # Step 14: heartbeat thread for this process. Idempotent.
        heartbeat.ensure_started(self._redis, os.getpid())
        raw = cast("bytes | None", self._redis.get(_key_current(schema)))
        if raw is None:
            raise SegmentNotFoundError(
                f"no current segment registered for schema {schema.__name__}"
            )
        name: str = raw.decode("utf-8") if isinstance(raw, bytes) else raw
        handle = posix_shm.open_existing(name)
        try:
            self._verify_header(handle, schema)
            layout = compute_layout_from_segment(handle.buf, schema)

            pid = os.getpid()
            with self._redis.pipeline(transaction=True) as p:
                p.incr(_key_refcount(name))
                p.sadd(_key_pid_segments(pid), name)
                p.execute()
        except BaseException:
            with contextlib.suppress(Exception):
                posix_shm.close(handle)
            raise

        return Segment(name=name, schema=schema, handle=handle, layout=layout)

    # ----- close ------------------------------------------------------------

    def close(self, segment: Segment) -> None:
        """Release this process's mapping; DECR refcount.

        If the refcount drops to zero, the segment name is added to the
        ``quorin:cleanup_queue`` set for the watchdog (Step 14) to unlink.
        We never unlink directly here — that would violate the
        creator-only-unlinks invariant.
        """
        name = segment.name
        pid = os.getpid()
        try:
            self._close_script(
                keys=[
                    _key_refcount(name),
                    _key_pid_segments(pid),
                    KEY_CLEANUP_QUEUE,
                    KEY_SEGMENT_TO_SCHEMA,
                ],
                args=[name],
            )
        finally:
            posix_shm.close(segment.handle)

    # ----- internals --------------------------------------------------------

    @staticmethod
    def _verify_header(
        handle: posix_shm.SegmentHandle,
        schema: type[FeatureSchema],
    ) -> None:
        magic, version, crc, _capacity = struct.unpack(HEADER_FMT, bytes(handle.buf[:HEADER_LEN]))
        if magic != MAGIC:
            raise SchemaCRCMismatchError(
                f"segment {handle.name!r}: bad magic {magic!r} (expected {MAGIC!r})"
            )
        if version != schema.version:
            raise SchemaCRCMismatchError(
                f"segment {handle.name!r}: stored version {version} "
                f"!= schema {schema.__name__} version {schema.version}"
            )
        expected_crc = crc32_of_bytes(compile_schema(schema).tobytes())
        if crc != expected_crc:
            raise SchemaCRCMismatchError(
                f"segment {handle.name!r}: stored CRC32 0x{crc:08x} "
                f"!= local schema CRC32 0x{expected_crc:08x}"
            )
