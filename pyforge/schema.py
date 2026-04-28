"""Schema definitions and offset-table compiler.

A Pyforge schema is a declarative Python class that names the typed fields of
a feature record. ``compile_schema`` turns it, once, into a NumPy structured
array (the "offset table") used by every later step as the single source of
truth for the segment's byte layout.

The module is deliberately dependency-light: numpy plus stdlib only. No
numba, redis, pyarrow, or pydantic imports live here.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from enum import IntEnum
from typing import Any, ClassVar, Final

import numpy as np

# ---------------------------------------------------------------------------
# Constants — single source of truth. Step 2 re-imports these.
# ---------------------------------------------------------------------------

HEADER_SIZE: Final[int] = 16
"""Bytes reserved at the start of every Pyforge segment for magic + version + CRC32."""

CACHE_LINE_SIZE: Final[int] = 64
"""x86-64 / ARM cache-line size. Every field start is aligned to this."""

PAGE_SIZE: Final[int] = 4096
"""OS page size. Total segment size is rounded up to this."""


# ---------------------------------------------------------------------------
# DType — closed enum of supported dtypes + lowercase alias namespace.
# ---------------------------------------------------------------------------


class DType(IntEnum):
    """Closed set of supported dtypes.

    Integer values are wire-stable: the ``dtype_code`` column of the offset
    table carries these numbers. Do not renumber once a segment exists on disk.
    """

    FLOAT32 = 1
    FLOAT64 = 2
    INT32 = 3
    INT64 = 4
    UINT8 = 5


@dataclass(frozen=True, slots=True)
class _DTypeAliases:
    """Lowercase namespace so user code can write ``dtype.float32`` as in the spec.

    A frozen dataclass gives us typed attribute access (mypy-friendly) without
    exposing a mutable object.
    """

    float32: DType = DType.FLOAT32
    float64: DType = DType.FLOAT64
    int32: DType = DType.INT32
    int64: DType = DType.INT64
    uint8: DType = DType.UINT8


dtype: Final[_DTypeAliases] = _DTypeAliases()


_DTYPE_BYTE_SIZE: Final[dict[DType, int]] = {
    DType.FLOAT32: 4,
    DType.FLOAT64: 8,
    DType.INT32: 4,
    DType.INT64: 8,
    DType.UINT8: 1,
}


DTYPE_TO_NUMPY: Final[dict[DType, np.dtype[Any]]] = {
    DType.FLOAT32: np.dtype(np.float32),
    DType.FLOAT64: np.dtype(np.float64),
    DType.INT32: np.dtype(np.int32),
    DType.INT64: np.dtype(np.int64),
    DType.UINT8: np.dtype(np.uint8),
}
"""DType IntEnum -> NumPy dtype object. ``pyforge.serving.assemble`` uses this
to build a typed ``np.frombuffer`` view over each field's bytes before slice-
assigning into the float32 output. Public because it crosses module
boundaries; ``_DTYPE_BYTE_SIZE`` stays private because it's used only inside
``FeatureField.byte_count``.
"""


# ---------------------------------------------------------------------------
# FeatureField — one typed field in a schema.
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class FeatureField:
    """One typed field in a FeatureSchema.

    ``shape=()`` means a scalar. Shapes with zero or negative dims are rejected.
    Variable-length shapes (e.g. ``(None, 128)``) are not supported in v1.
    """

    name: str
    dtype: DType
    shape: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError(f"FeatureField.name must be a non-empty string, got {self.name!r}")
        if not isinstance(self.dtype, DType):
            raise TypeError(f"FeatureField.dtype must be a DType, got {type(self.dtype).__name__}")
        if not isinstance(self.shape, tuple):
            raise TypeError(f"FeatureField.shape must be a tuple, got {type(self.shape).__name__}")
        for d in self.shape:
            # bool is a subclass of int — explicitly reject.
            if not isinstance(d, int) or isinstance(d, bool):
                raise TypeError(f"shape dimensions must be int, got {type(d).__name__}")
            if d <= 0:
                raise ValueError(f"shape dimensions must be positive, got {self.shape}")

    @property
    def element_count(self) -> int:
        count = 1
        for d in self.shape:
            count *= d
        return count

    @property
    def byte_count(self) -> int:
        return self.element_count * _DTYPE_BYTE_SIZE[self.dtype]


# ---------------------------------------------------------------------------
# FeatureSchema — base class. Subclasses validated at class-definition time.
# ---------------------------------------------------------------------------


class FeatureSchema:
    """Base class for all Pyforge schemas.

    Subclasses MUST declare:
      * ``version: int`` — baked into the shared-memory segment name.
      * ``fields: list[FeatureField]`` — non-empty, unique-named.

    Validation runs in ``__init_subclass__`` so errors surface at class
    definition time rather than when the schema is first compiled.
    """

    version: ClassVar[int]
    fields: ClassVar[list[FeatureField]]

    def __init_subclass__(cls, **kwargs: object) -> None:
        super().__init_subclass__(**kwargs)

        if "version" not in cls.__dict__:
            raise TypeError(f"{cls.__name__} must define a class attribute `version: int`")
        version = cls.__dict__["version"]
        # bool is an int subclass — reject it explicitly.
        if not isinstance(version, int) or isinstance(version, bool):
            raise TypeError(f"{cls.__name__}.version must be int, got {type(version).__name__}")

        if "fields" not in cls.__dict__:
            raise TypeError(
                f"{cls.__name__} must define a class attribute `fields: list[FeatureField]`"
            )
        field_list = cls.__dict__["fields"]
        if not isinstance(field_list, list):
            raise TypeError(
                f"{cls.__name__}.fields must be a list, got {type(field_list).__name__}"
            )
        if not field_list:
            raise ValueError(f"{cls.__name__}.fields must not be empty")
        for i, f in enumerate(field_list):
            if not isinstance(f, FeatureField):
                raise TypeError(
                    f"{cls.__name__}.fields[{i}] must be a FeatureField, got {type(f).__name__}"
                )
        names = [f.name for f in field_list]
        if len(set(names)) != len(names):
            dupes = sorted({n for n in names if names.count(n) > 1})
            raise ValueError(f"{cls.__name__}.fields has duplicate names: {dupes}")


# ---------------------------------------------------------------------------
# Offset-table structured dtype — Numba-readable primitives only.
# ---------------------------------------------------------------------------

OFFSET_TABLE_DTYPE: Final[np.dtype[np.void]] = np.dtype(
    [
        ("name_hash", np.uint64),
        ("byte_offset", np.uint64),
        ("dtype_code", np.uint8),
        ("element_count", np.uint32),
        ("byte_count", np.uint32),
    ]
)


# ---------------------------------------------------------------------------
# Private helpers.
# ---------------------------------------------------------------------------


def _hash_name(name: str) -> int:
    """64-bit hash of a field name.

    Pinned forever: ``blake2b(UTF-8 bytes, digest_size=8)`` read as little-endian
    unsigned. Changing this silently breaks every persisted segment, so the
    pinned test ``test_hash_is_pinned`` asserts known outputs for known inputs.
    """
    return int.from_bytes(
        hashlib.blake2b(name.encode("utf-8"), digest_size=8).digest(),
        byteorder="little",
        signed=False,
    )


def _align_up(value: int, alignment: int) -> int:
    """Round ``value`` up to the next multiple of ``alignment`` (power of two)."""
    return (value + alignment - 1) & ~(alignment - 1)


def _field_byte_offsets(
    field_list: list[FeatureField], *, base: int = HEADER_SIZE
) -> list[int]:
    """Walk ``field_list`` in declaration order, returning each field's start
    byte offset.

    Every field is rounded up to :data:`CACHE_LINE_SIZE` before its bytes are
    written, so no two fields share a cache line. ``base`` is the starting
    cursor — :data:`HEADER_SIZE` for segment-absolute offsets used by
    :func:`compile_schema`, or 0 for row-relative offsets used by callers like
    :func:`compute_assembly_table` that operate within a single entity row.
    """
    cursor = base
    offsets: list[int] = []
    for f in field_list:
        cursor = _align_up(cursor, CACHE_LINE_SIZE)
        offsets.append(cursor)
        cursor += f.byte_count
    return offsets


# ---------------------------------------------------------------------------
# Public functions: compile_schema + total_segment_size.
# ---------------------------------------------------------------------------


def compile_schema(schema: type[FeatureSchema]) -> np.ndarray[Any, np.dtype[np.void]]:
    """Compile a FeatureSchema subclass into its offset table.

    Layout:
      * The 16-byte header occupies bytes 0-15.
      * The first field starts at offset ``_align_up(HEADER_SIZE, 64) = 64``.
      * Each subsequent field is packed in declaration order, with its start
        rounded up to a 64-byte cache-line boundary.

    The returned structured array is sorted by ``name_hash`` so a lookup can
    use ``np.searchsorted`` in O(log n) with zero allocation. Field order on
    the wire is therefore hash order, not declaration order.
    """
    field_list = schema.fields
    n = len(field_list)
    table = np.zeros(n, dtype=OFFSET_TABLE_DTYPE)

    offsets = _field_byte_offsets(field_list, base=HEADER_SIZE)

    # Populate the table.
    for i, f in enumerate(field_list):
        table[i] = (
            _hash_name(f.name),
            offsets[i],
            int(f.dtype),
            f.element_count,
            f.byte_count,
        )

    # Sort by name_hash for searchsorted lookups downstream.
    table.sort(order="name_hash")

    # Sanity: reject any 64-bit hash collision within a single schema.
    # Astronomically unlikely at our scale (≤200 fields), but cheap to check.
    if np.unique(table["name_hash"]).size != n:
        by_hash: dict[int, list[str]] = {}
        for f in field_list:
            by_hash.setdefault(_hash_name(f.name), []).append(f.name)
        collisions = {h: names for h, names in by_hash.items() if len(names) > 1}
        raise ValueError(f"blake2b 64-bit hash collision in schema {schema.__name__}: {collisions}")

    return table


def total_segment_size(schema: type[FeatureSchema]) -> int:
    """Total bytes a *single-entity* segment needs for this schema.

    Computed as ``HEADER_SIZE + sum(aligned field slots)``, rounded up to a
    4 KiB page boundary.

    .. note::
        This is the **single-entity** sizing used by Step 1's tests and the
        original Step 2 placeholder. The real Pyforge segment is multi-entity
        and uses :func:`pyforge.layout.total_segment_size` (which takes
        ``capacity`` and accounts for slot table + string pool + feature rows).
    """
    cursor = HEADER_SIZE
    for f in schema.fields:
        cursor = _align_up(cursor, CACHE_LINE_SIZE)
        cursor += f.byte_count
    return _align_up(cursor, PAGE_SIZE)


def compute_row_offset_table(
    schema: type[FeatureSchema],
) -> np.ndarray[Any, np.dtype[np.void]]:
    """Like :func:`compile_schema`, but ``byte_offset`` is row-relative (0-based).

    The offset table from :func:`compile_schema` is *segment*-relative — its
    first field sits at byte 64 (where the data area begins after the header
    is aligned up to a cache line). For multi-entity layouts, what matters is
    the offset within a single row, where the first field is at byte 0.

    This function returns the same structured-array shape with ``byte_offset``
    rebased to row-zero. ``name_hash``, ``dtype_code``, ``element_count``, and
    ``byte_count`` are unchanged.
    """
    table = compile_schema(schema).copy()
    if table.size > 0:
        first = int(table["byte_offset"].min())
        # ``first`` is align_up(HEADER_SIZE, CACHE_LINE_SIZE) by construction.
        # Subtracting it places the first field at row offset 0.
        table["byte_offset"] = table["byte_offset"] - first
    return table


def row_size(schema: type[FeatureSchema]) -> int:
    """Bytes of one entity's row, padded so the *next* row is cache-aligned.

    Computed from :func:`compute_row_offset_table` as ``align_up(max(offset +
    byte_count), CACHE_LINE_SIZE)``. A row therefore always starts and ends on
    a cache-line boundary, so neighbouring rows never share a cache line.
    """
    table = compute_row_offset_table(schema)
    if table.size == 0:
        return 0
    end = int((table["byte_offset"] + table["byte_count"]).max())
    return _align_up(end, CACHE_LINE_SIZE)


def compute_assembly_table(
    schema: type[FeatureSchema],
) -> np.ndarray[Any, np.dtype[np.void]]:
    """Like :func:`compute_row_offset_table`, but rows are in DECLARATION ORDER.

    The hash-sorted offset table powers ``searchsorted``-based lookups; this
    one drives output-vector assembly. Each row's ``byte_offset`` is row-
    relative (matches :func:`compute_row_offset_table`) so a reader indexes it
    as ``buf[row_offset + byte_offset : row_offset + byte_offset + byte_count]``.

    Returned dtype is :data:`OFFSET_TABLE_DTYPE` so Step 5's Numba kernel can
    iterate this and the lookup table with the same kernel — only the order
    differs. See ADR-003 for why output ordering is declaration, not hash.
    """
    field_list = schema.fields
    n = len(field_list)
    table = np.zeros(n, dtype=OFFSET_TABLE_DTYPE)

    # Row-relative offsets: the first field sits at byte 0 within a row.
    offsets = _field_byte_offsets(field_list, base=0)

    for i, f in enumerate(field_list):
        table[i] = (
            _hash_name(f.name),
            offsets[i],
            int(f.dtype),
            f.element_count,
            f.byte_count,
        )

    # Deliberately NOT sorted — declaration order is the contract.
    return table


def total_element_count(schema: type[FeatureSchema]) -> int:
    """Sum of ``element_count`` across all fields. Length of the float32 vector
    returned by :func:`pyforge.serving.assemble`.
    """
    return sum(f.element_count for f in schema.fields)
