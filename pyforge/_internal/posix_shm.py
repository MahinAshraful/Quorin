"""POSIX shared memory wrapper, bypassing CPython's resource_tracker.

Stdlib's ``multiprocessing.shared_memory.SharedMemory`` automatically registers
opened segments with ``multiprocessing.resource_tracker`` for cleanup on
process exit. The consequence: a *reader* exiting cleanly will ``shm_unlink``
the segment, breaking every other reader. This module avoids that by going
through ``posix_ipc`` directly — it never registers anything with the
resource tracker.

Three invariants every caller must respect:

1. **Only the creator calls** :func:`unlink`. Readers never. The watchdog
   (Step 14) is the only proxy-unlinker, acting on behalf of a dead creator.
2. :func:`close` and :func:`unlink` are **separate** operations. ``close``
   releases this process's mapping; ``unlink`` removes the segment name from
   the system so no new process can open it. Conflating them is the bug
   we're avoiding.
3. :func:`open_existing` **does not** register with any cleanup tracker. That
   is the entire reason this module exists.

Linux/macOS only. ``posix_ipc`` has no native-Windows wheel; the spec scopes
Pyforge to Linux/WSL2.
"""

from __future__ import annotations

import contextlib
import mmap
from dataclasses import dataclass, field
from typing import Final

import posix_ipc

_DEFAULT_MODE: Final[int] = 0o600


@dataclass
class SegmentHandle:
    """Opaque handle to an open POSIX shm segment.

    The internal ``_mmap`` keeps the underlying mapping alive so that the
    public ``buf`` memoryview remains valid. Do not access ``_mmap`` from
    outside this module.
    """

    name: str
    size: int
    buf: memoryview
    # Backing mmap. Releasing it invalidates ``buf``. Excluded from repr to
    # keep test failures readable.
    _mmap: mmap.mmap = field(repr=False)


def create(name: str, size: int, *, hugepage: bool = False) -> SegmentHandle:
    """Allocate a brand-new POSIX shm segment of ``size`` bytes.

    Uses ``O_CREX`` so an existing name with the same id raises rather than
    silently sharing memory. If the post-create ``mmap`` fails, the segment
    is unlinked before re-raising — no leak in ``/dev/shm``.

    When ``hugepage=True``, ``MADV_HUGEPAGE`` is requested on the mapping.
    Tmpfs THP requires kernel ``CONFIG_TRANSPARENT_HUGEPAGE_SHMEM=y`` plus
    ``/sys/kernel/mm/transparent_hugepage/shmem_enabled`` ∈ ``{always,
    advise, within_size}``; even then the kernel may decline under
    fragmentation. A failure to honor the advice is non-fatal — the
    ``(AttributeError, OSError)`` swallow keeps the segment alive at the
    base 4 KB page granularity.

    Raises:
        posix_ipc.ExistentialError: if a segment with this name already exists.
        OSError: if ``mmap`` fails after ``shm_open`` succeeded.
    """
    shm = posix_ipc.SharedMemory(
        name,
        flags=posix_ipc.O_CREX,
        size=size,
        mode=_DEFAULT_MODE,
    )
    try:
        m = mmap.mmap(shm.fd, size)
    except BaseException:
        # mmap failed — clean up the segment we just created.
        with contextlib.suppress(OSError):
            shm.close_fd()
        with contextlib.suppress(posix_ipc.ExistentialError):
            posix_ipc.unlink_shared_memory(name)
        raise
    if hugepage:
        with contextlib.suppress(AttributeError, OSError):
            m.madvise(mmap.MADV_HUGEPAGE)
    # mmap succeeded; the fd is no longer needed (mapping survives).
    shm.close_fd()
    return SegmentHandle(name=name, size=size, buf=memoryview(m), _mmap=m)


def open_existing(name: str, *, hugepage: bool = False) -> SegmentHandle:
    """Open an existing POSIX shm segment for read+write.

    Critical property: this does **not** register the segment with
    ``multiprocessing.resource_tracker``, so when this process exits, the
    segment is **not** unlinked. That contract is what makes Pyforge readable
    by multiple processes safely.

    The ``hugepage`` kwarg works as in :func:`create`. Kernel THP is per-VMA,
    not per-fd: a worker process opening the same segment without this kwarg
    gets base 4 KB pages on its own faults regardless of whether the creator
    requested hugepages.

    Raises:
        FileNotFoundError: if the segment does not exist.
    """
    try:
        shm = posix_ipc.SharedMemory(name)
    except posix_ipc.ExistentialError as e:
        raise FileNotFoundError(f"shm segment not found: {name}") from e
    try:
        m = mmap.mmap(shm.fd, shm.size)
    except BaseException:
        with contextlib.suppress(OSError):
            shm.close_fd()
        raise
    if hugepage:
        with contextlib.suppress(AttributeError, OSError):
            m.madvise(mmap.MADV_HUGEPAGE)
    size = shm.size
    shm.close_fd()
    return SegmentHandle(name=name, size=size, buf=memoryview(m), _mmap=m)


def close(handle: SegmentHandle) -> None:
    """Release this process's mapping. Does NOT unlink the segment name.

    Safe to call multiple times — second call is a no-op.
    """
    with contextlib.suppress(BufferError, ValueError):
        handle.buf.release()
    with contextlib.suppress(ValueError):
        handle._mmap.close()


def unlink(name: str) -> None:
    """Remove the segment name from the system. Idempotent.

    Should be called only by the **creator** of the segment, or by the
    watchdog (Step 14) on behalf of a dead creator. Calling it from any
    reader violates the cross-process invariant and is the bug
    ``multiprocessing.shared_memory`` introduces silently.
    """
    with contextlib.suppress(posix_ipc.ExistentialError):
        posix_ipc.unlink_shared_memory(name)
