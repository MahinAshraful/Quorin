"""Buffer pool — pre-allocated, lock-free, capped float32 output buffers.

Step 6 removes the last hot-path allocation (``np.empty(total_element_count,
dtype=np.float32)``) from :func:`pyforge.serving.assemble` and
:func:`pyforge.assembly.assemble`. A :class:`BufferPool` is constructed once
per schema; callers grab a buffer with ``with pool.checkout() as buf:`` and
pass it to ``assemble(..., out=buf)``.

Design (locked by ADR-005):

- **Class-based context manager**, NOT ``@contextlib.contextmanager``. The
  generator-based decorator costs ~700 ns/call (generator allocation +
  ``next()`` + ``send()``). On a sub-microsecond hot path that is the
  difference between a net win and a net regression. :class:`_Checkout`
  uses ``__slots__`` and direct ``__enter__`` / ``__exit__`` methods —
  ~150 ns of total overhead.
- **Pre-allocate at __init__.** All ``max_size`` buffers are minted at pool
  construction so the pool is at 100% hit rate from request 1, not after
  warm-up. Cold-start cost is paid once, off the request path.
- **Lock-free via deque.** ``collections.deque.popleft()`` and ``.append()``
  are atomic under CPython's GIL. No ``threading.Lock`` is acquired per
  checkout. (See PEP 703 / free-threaded caveat in ADR-005.)
- **Cap returns at max_size.** A burst that grew the pool past ``max_size``
  via fall-through allocations should not permanently inflate memory; surplus
  returned buffers are dropped and GC reclaims them.
- **Zero-on-return, default true, toggleable.** Defense in depth at ~30 ns
  per return for small schemas. Callers that have measured the cost as
  unaffordable can pass ``zero_on_return=False``.
- **No threaded refill.** A refill thread would add per-call lock cost
  greater than the per-miss allocation cost it tries to amortize for a
  workload whose concurrency is bounded by worker count. Revisit if
  Step 16 flamegraphs justify; otherwise YAGNI.

Lifetime contract for callers
-----------------------------
The ndarray yielded by :meth:`BufferPool.checkout` is invalidated when the
``with`` block exits. Do not retain references, store views, or pass the
buffer to async callbacks that outlive the block. To return data to a
caller, copy with ``vec.copy()`` inside the ``with``.
"""

from __future__ import annotations

import collections
from types import TracebackType
from typing import Any

import numpy as np

from pyforge.metrics import pool_miss_total
from pyforge.schema import FeatureSchema, total_element_count


class _Checkout:
    """One-shot context manager for a single :meth:`BufferPool.checkout` call.

    Class-based (not ``@contextlib.contextmanager``-decorated) because the
    generator protocol's per-call overhead (~700 ns: GeneratorContextManager
    allocation, ``next()`` to advance to yield, ``send()`` to advance past)
    is large enough on a sub-microsecond hot path to swamp the np.empty
    savings the pool exists to capture.

    ``__slots__`` keeps the per-call allocation tiny; the only attributes
    touched in __enter__ / __exit__ are ``_pool`` and ``_buf``, both
    pre-declared.

    Direct field access on the pool (``self._pool._buffers`` etc.) bypasses
    the property-getter dispatch path, saving ~50 ns per attribute read.
    """

    __slots__ = ("_buf", "_pool")

    def __init__(self, pool: BufferPool) -> None:
        self._pool = pool
        self._buf: np.ndarray[Any, np.dtype[np.float32]] | None = None

    def __enter__(self) -> np.ndarray[Any, np.dtype[np.float32]]:
        pool = self._pool
        try:
            buf = pool._buffers.popleft()
        except IndexError:
            pool_miss_total.labels(schema=pool._schema_label).inc()
            buf = np.zeros(pool._element_count, dtype=np.float32)
        self._buf = buf
        return buf

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        buf = self._buf
        # ``buf is None`` only when ``__enter__`` raised before the
        # assignment (e.g., MemoryError on the fallback np.zeros). Nothing
        # to return; let the original exception propagate.
        if buf is None:
            return
        # Drop the local reference so a re-use of this _Checkout (which
        # we don't support, but defend against anyway) cannot double-return.
        self._buf = None
        pool = self._pool
        if pool._zero_on_return:
            buf.fill(0)
        if len(pool._buffers) < pool._max_size:
            pool._buffers.append(buf)


class BufferPool:
    """Lock-free pool of float32 output buffers sized for one schema.

    All buffers are 1-D, C-contiguous, ``dtype=np.float32``,
    ``shape=(total_element_count(schema),)``. The pool itself is GIL-atomic
    (deque ops); concurrent checkouts from multiple threads work without
    explicit locking under CPython.
    """

    __slots__ = (
        "_buffers",
        "_element_count",
        "_max_size",
        "_schema_label",
        "_zero_on_return",
    )

    def __init__(
        self,
        schema: type[FeatureSchema],
        *,
        max_size: int = 128,
        zero_on_return: bool = True,
    ) -> None:
        if max_size < 1:
            raise ValueError(f"max_size must be >= 1, got {max_size}")
        # ``f"{module}.{qualname}"`` disambiguates same-name schemas in
        # different modules (e.g., ``a.MySchema`` vs ``b.MySchema``).
        self._schema_label = f"{schema.__module__}.{schema.__qualname__}"
        self._element_count = total_element_count(schema)
        self._max_size = max_size
        self._zero_on_return = zero_on_return
        # Pre-allocate every buffer at construction. ``np.zeros`` (vs
        # ``np.empty``) so the very first checkout-without-write returns a
        # clean buffer — defence in depth that costs O(max_size) once.
        self._buffers: collections.deque[np.ndarray[Any, np.dtype[np.float32]]] = collections.deque(
            np.zeros(self._element_count, dtype=np.float32) for _ in range(max_size)
        )

    def checkout(self) -> _Checkout:
        """Return a one-shot context manager that yields a pooled buffer.

        Pool hit (~150 ns total: _Checkout allocation + ``__enter__`` +
        ``__exit__`` + ``buf.fill(0)``): a pre-allocated buffer is popped
        from the deque and yielded.

        Pool miss: a fresh ``np.zeros`` is allocated and ``pool_miss_total``
        is incremented. The fresh buffer is still returned at ``__exit__``
        and joins the pool if there is capacity, naturally growing the
        pool toward steady-state.

        On ``__exit__`` the buffer is zeroed (when ``zero_on_return=True``)
        and re-appended to the deque iff ``len(deque) < max_size``. Surplus
        buffers (from a fall-through-allocate burst) are dropped, capping
        steady-state memory.

        ``__exit__`` runs even when the ``with`` body raises (including
        ``KeyboardInterrupt``), so a signal injected mid-block cannot leak
        the buffer.
        """
        return _Checkout(self)

    @property
    def available(self) -> int:
        """Buffers currently in the pool. Read-only; race-free under GIL."""
        return len(self._buffers)

    @property
    def max_size(self) -> int:
        return self._max_size

    @property
    def schema_name(self) -> str:
        """``f"{module}.{qualname}"`` of the schema this pool was built for.

        Used as the ``schema`` label on :data:`pyforge.metrics.pool_miss_total`.
        """
        return self._schema_label

    @property
    def element_count(self) -> int:
        """Length of every buffer in the pool (== ``total_element_count(schema)``)."""
        return self._element_count

    @property
    def zero_on_return(self) -> bool:
        return self._zero_on_return

    def __repr__(self) -> str:
        return (
            f"BufferPool(schema={self._schema_label!r}, "
            f"max_size={self._max_size}, "
            f"element_count={self._element_count}, "
            f"available={len(self._buffers)})"
        )


# ---------------------------------------------------------------------------
# Batch buffer pool — pre-allocated 2D float32 buffers for assemble_batch.
# ---------------------------------------------------------------------------


class _BatchCheckout:
    """One-shot context manager for a single :meth:`BatchBufferPool.checkout`.

    Same protocol-cost discipline as :class:`_Checkout` (Step 6): class-based
    rather than ``@contextlib.contextmanager`` to avoid the ~700 ns generator
    overhead. ``__slots__`` keeps the per-call allocation minimal.
    """

    __slots__ = ("_buf", "_pool")

    def __init__(self, pool: BatchBufferPool) -> None:
        self._pool = pool
        self._buf: np.ndarray[Any, np.dtype[np.float32]] | None = None

    def __enter__(self) -> np.ndarray[Any, np.dtype[np.float32]]:
        pool = self._pool
        try:
            buf = pool._buffers.popleft()
        except IndexError:
            pool_miss_total.labels(schema=pool._schema_label).inc()
            buf = np.zeros((pool._batch_size, pool._element_count), dtype=np.float32)
        self._buf = buf
        return buf

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        buf = self._buf
        if buf is None:
            return
        self._buf = None
        pool = self._pool
        if pool._zero_on_return:
            # ~50-100 us at batch=1000, opt-in only. See class docstring
            # for the full-buffer-overwrite invariant that makes False safe.
            buf.fill(0)
        if len(pool._buffers) < pool._max_size:
            pool._buffers.append(buf)


class BatchBufferPool:
    """Pool of ``(batch_size, total_element_count)`` float32 buffers.

    Sized for a single ``(schema, batch_size)`` pair. Distinct from
    :class:`BufferPool` because:

    - Output shape is 2D, not 1D.
    - ``zero_on_return`` defaults to ``False`` (cost at batch=1000 is
      ~50-100 us, would erase the entire pool win on the call that should
      benefit most).
    - ``max_size`` defaults to ``64`` (memory ceiling).
    - ``batch_size`` is required (no implicit dimension).

    Memory budget::

        max_size * batch_size * total_element_count(schema) * 4 bytes

    Worked example: ``BatchBufferPool(my_200_field_schema, batch_size=10000)``
    with ``max_size=64`` (the default) and a 200-field-with-128-emb schema
    (total_element_count ~= 327) holds::

        64 * 10000 * 327 * 4 bytes = ~838 MiB

    Inverse - for an X MiB memory budget::

        max_size = (X * 1024 * 1024) / (batch_size * element_count * 4)

    Tune ``max_size`` to your worker count and request burst pattern. The
    default of 64 is appropriate for moderate batch sizes (100-1000);
    explicitly lower it for batch >= 5000.

    **Default ``zero_on_return=False`` safety contract**

    Safe for :func:`pyforge.assembly.assemble_batch` because the kernel
    guarantees full-buffer overwrite - every row is either filled with
    feature data (hit) or explicitly zeroed (miss). Prior buffer contents
    are never read by the consumer.

    *Load-bearing invariant*: if the kernel ever changes such that a row
    could be partially written, this default becomes a data-leak vector.
    If you reuse pool buffers for any other purpose (debug print, copy to
    caller, alternate kernel), set ``zero_on_return=True`` to defend
    against prior-data leakage. Cost at batch=1000: ~50-100 us per return.

    Construction allocates a single contiguous ``(max_size, batch_size,
    element_count)`` slab once and slices it into the deque. One allocator
    call instead of ``max_size``, lower allocator-metadata overhead, future
    huge-page coalescing friendly. The slab ndarray stays alive as long as
    any of its slice-views is referenced; the deque holds them all.
    """

    __slots__ = (
        "_batch_size",
        "_buffers",
        "_element_count",
        "_max_size",
        "_schema_label",
        "_zero_on_return",
    )

    def __init__(
        self,
        schema: type[FeatureSchema],
        *,
        batch_size: int,
        max_size: int = 64,
        zero_on_return: bool = False,
    ) -> None:
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}")
        if max_size < 1:
            raise ValueError(f"max_size must be >= 1, got {max_size}")
        self._schema_label = f"{schema.__module__}.{schema.__qualname__}"
        self._element_count = total_element_count(schema)
        self._batch_size = batch_size
        self._max_size = max_size
        self._zero_on_return = zero_on_return
        # Single contiguous slab. The slab is kept alive by the deque holding
        # views into it; when the deque drops a view, that row's memory is
        # still pinned by every other view, so the slab persists until the
        # pool itself is collected.
        slab = np.zeros((max_size, batch_size, self._element_count), dtype=np.float32)
        self._buffers: collections.deque[np.ndarray[Any, np.dtype[np.float32]]] = collections.deque(
            slab[i] for i in range(max_size)
        )

    def checkout(self) -> _BatchCheckout:
        """Return a one-shot context manager that yields a pooled 2D buffer.

        Buffer shape is ``(batch_size, element_count)``, dtype ``float32``,
        C-contiguous, writeable.

        Pool hit: a pre-allocated buffer is popped from the deque.

        Pool miss: a fresh ``np.zeros`` is allocated and ``pool_miss_total``
        is incremented. The fresh buffer joins the pool on return if there
        is capacity, growing toward steady-state.

        ``__exit__`` runs even when the ``with`` body raises (including
        ``KeyboardInterrupt``), so a signal injected mid-block cannot leak
        the buffer.
        """
        return _BatchCheckout(self)

    @property
    def available(self) -> int:
        """Buffers currently in the pool. Read-only; race-free under GIL."""
        return len(self._buffers)

    @property
    def batch_size(self) -> int:
        return self._batch_size

    @property
    def max_size(self) -> int:
        return self._max_size

    @property
    def element_count(self) -> int:
        return self._element_count

    @property
    def schema_name(self) -> str:
        """``f"{module}.{qualname}"`` of the schema this pool was built for."""
        return self._schema_label

    @property
    def zero_on_return(self) -> bool:
        return self._zero_on_return

    def __repr__(self) -> str:
        return (
            f"BatchBufferPool(schema={self._schema_label!r}, "
            f"batch_size={self._batch_size}, "
            f"max_size={self._max_size}, "
            f"element_count={self._element_count}, "
            f"available={len(self._buffers)})"
        )
