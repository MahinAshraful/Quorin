# ADR-001: POSIX shared memory via `posix_ipc`, not `multiprocessing.shared_memory`

**Status:** Accepted
**Date:** 2026-04-24
**Step:** 2 (shared memory and lifecycle)

## Decision

Quorin's POSIX shared-memory layer (`quorin/_internal/posix_shm.py`) uses the
`posix_ipc` PyPI library as its sole underlying dependency for
`shm_open` / `mmap` / `shm_unlink`. The stdlib `multiprocessing.shared_memory`
module is **not** used.

## Context

Shared memory segments need three operations for Quorin's use case:
allocate (`shm_open` + `ftruncate` + `mmap`), open-existing, and
unlink. There are three viable Python-level approaches:

1. **`multiprocessing.shared_memory.SharedMemory`** (stdlib).
2. **`ctypes` wrapper over `libc`** (no new dependencies).
3. **`posix_ipc`** (mature PyPI library, maintained since 2010, >1 M monthly downloads).

## The problem with the stdlib

When *any* Python process opens a segment with `SharedMemory(name=..., create=False)`,
CPython's internal `multiprocessing.resource_tracker` subprocess registers
the segment for cleanup on process exit. When that process exits
(cleanly — via `atexit` or normal interpreter shutdown), `resource_tracker`
calls `shm_unlink(name)`.

This means **every reader is a potential destroyer**. A worker that opens
the segment, reads a value, and exits cleanly will unlink the segment out
from under every other process still using it. Subsequent reads in those
processes return stale bytes or raise `FileNotFoundError`.

This is not an edge case; it is the **default** behavior and has been an
open CPython issue since 2019:
[bpo-38119](https://bugs.python.org/issue38119),
[cpython#82300](https://github.com/python/cpython/issues/82300),
[cpython#104291](https://github.com/python/cpython/issues/104291),
[cpython#91577](https://github.com/python/cpython/issues/91577).

Workarounds exist (monkey-patching `resource_tracker.unregister` after each
open) but depend on a private API and break across Python minor versions.
A public API to disable resource-tracker registration does not exist.

## The problem with the ctypes path

Writing `shm_open` / `mmap` / `munmap` / `shm_unlink` directly via ctypes is
the dependency-free path, but carries three categories of risk that
`posix_ipc` has already solved:

- **File-descriptor lifecycle.** After `mmap` succeeds, the fd returned from
  `shm_open` must be closed (the mapping persists independently). Leaking
  the fd leaks kernel resources. Closing it in the wrong error path leaks
  the segment name in `/dev/shm`. `posix_ipc.SharedMemory.close_fd()`
  handles this.
- **`mmap` return value.** `mmap` returns `MAP_FAILED` (which is
  `(void *)-1`, **not** `NULL`) on error. In ctypes this shows up as a
  pointer to address `0xFFFFFFFFFFFFFFFF`. Missing the explicit check
  causes segfaults with no Python traceback. `posix_ipc` checks this.
- **Platform differences.** Linux treats `/dev/shm` as tmpfs; macOS's
  `shm_open` is a true syscall with stricter name rules (must start with
  `/`, may not contain additional `/`); FreeBSD differs again. `posix_ipc`
  normalizes these. A ctypes wrapper written on Linux has silent bugs on
  macOS that CI (Linux-only) will not catch.

None of these risks are interesting engineering for this project. The
interesting engineering is the offset table, the Numba hot path, the WAL
idempotency, and the watchdog. Every hour spent debugging a ctypes
fd-lifecycle mistake is an hour not spent on those.

## The choice

`posix_ipc` is a mature, audited wrapper that:

- Does **not** touch `multiprocessing.resource_tracker` on open.
- Handles fd lifecycle, MAP_FAILED checks, and platform differences correctly.
- Has been in production use for ~15 years.

The Quorin wrapper (`quorin/_internal/posix_shm.py`) is a ~100-line
adapter that presents a clean internal interface
(`create` / `open_existing` / `close` / `unlink`). If a future Python
version fixes the resource_tracker bug cleanly or if `posix_ipc` ever
needs to be swapped out, there is exactly one file to change.

## Three invariants the wrapper enforces

1. **Only the creator calls `unlink`.** Readers never. The watchdog
   (Step 14) is the only proxy-unlinker, acting on behalf of a dead creator.
2. **`close` and `unlink` are separate operations.** `close` releases this
   process's mapping; `unlink` removes the segment name from the system.
   They are not the same call.
3. **`open_existing` does not register with any cleanup tracker.** That is
   the entire reason this wrapper exists.

## Consequences

- **Positive:** segments survive reader exits. Cross-process integration
  tests pass deterministically. No silent data loss from
  `resource_tracker`.
- **Positive:** one dependency (`posix_ipc`) is a small price for avoiding
  the ctypes risk categories.
- **Negative:** `posix_ipc` has no native-Windows wheel, which reinforces
  the already-in-scope "Linux/WSL2 only" constraint.
- **Negative:** if `posix_ipc` ever goes unmaintained, we own the
  replacement. Mitigated by the thin-wrapper design.

## References

- [posix_ipc PyPI](https://pypi.org/project/posix-ipc/) — maintained by
  Osvaldo Santana since 2010.
- [cpython#82300](https://github.com/python/cpython/issues/82300) — original
  bug report for the resource_tracker shared-memory behavior.
- [cpython#104291](https://github.com/python/cpython/issues/104291) — more
  recent discussion; still open as of writing.

## Validation

The integration test `tests/integration/test_shm_lifecycle.py::test_reader_clean_exit_does_not_destroy_segment`
is the binary validator. If it passes, the resource_tracker bypass is
working. If it fails, something on the open path is registering with
`multiprocessing.resource_tracker` and needs to be found and removed.
