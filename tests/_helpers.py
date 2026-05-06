"""Shared test helpers.

Imported via ``from _helpers import ...`` thanks to ``pythonpath = ["tests"]``
in ``pyproject.toml``. Keep this module dependency-light (numpy + hypothesis +
quorin only) so it can be used from unit, property, integration, and
benchmark tests without dragging in pytest fixtures or Redis.

Conventions: top-of-file imports of POSIX-only modules require an explicit
``sys.platform`` skip in the importing test file. This module is itself
imported under that gate, so it's safe to import freely from POSIX-gated
test modules.
"""

from __future__ import annotations

import os
import string
import struct
import uuid
from typing import Any

import numpy as np
from hypothesis import strategies as st

from quorin._internal import posix_shm
from quorin._internal.crc import crc32_of_bytes
from quorin.layout import (
    DEFAULT_MAX_ID_BYTES,
    compute_layout,
    initialize_segment_regions,
)
from quorin.schema import (
    DTYPE_TO_NUMPY,
    DType,
    FeatureField,
    FeatureSchema,
    compile_schema,
    compute_assembly_table,
    row_size,
)
from quorin.shm import HEADER_FMT, HEADER_LEN, MAGIC, Segment


def unique_segment_name(prefix: str = "quorin_test") -> str:
    """Process-local unique segment name. Tests use it to avoid /dev/shm
    collisions across parallel runs and prior leaks."""
    return f"{prefix}_{os.getpid()}_{uuid.uuid4().hex[:8]}"


def make_segment(
    schema: type[FeatureSchema],
    capacity: int,
    *,
    max_id_bytes: int = DEFAULT_MAX_ID_BYTES,
    name_prefix: str = "quorin_test",
) -> Segment:
    """Allocate and initialize a Segment without touching SegmentRegistry / Redis.

    Mirrors the per-file ``_make_segment`` pattern in
    ``tests/unit/test_layout.py`` etc. Centralized here so Step 4's three new
    test files don't all duplicate it.
    """
    layout = compute_layout(schema, capacity=capacity, max_id_bytes=max_id_bytes)
    name = unique_segment_name(name_prefix)
    handle = posix_shm.create(name, layout.total_size)

    crc = crc32_of_bytes(compile_schema(schema).tobytes())
    header = struct.pack(HEADER_FMT, MAGIC, int(schema.version), crc, capacity)
    handle.buf[:HEADER_LEN] = header
    initialize_segment_regions(handle.buf, layout)

    return Segment(name=name, schema=schema, handle=handle, layout=layout)


def release_segment(seg: Segment) -> None:
    """Close + unlink a test-owned segment. Mirrors what
    ``SegmentRegistry.close`` would do in production minus the Redis bookkeeping.
    """
    posix_shm.close(seg.handle)
    posix_shm.unlink(seg.name)


def pack_row(
    schema: type[FeatureSchema], values: dict[str, np.ndarray[Any, np.dtype[Any]]]
) -> bytes:
    """Pack a ``{field_name: ndarray}`` dict into ``row_size(schema)`` bytes.

    Walks the schema's row offset table; for each field, asserts the provided
    ndarray has the correct dtype and ``element_count`` (post-flatten), then
    writes its bytes at the row-relative byte offset. The result is exactly
    ``row_size(schema)`` bytes — suitable for direct passing to
    ``layout.insert(seg, entity_id, row)``.

    Raises ValueError on shape/dtype mismatch so test bugs surface loudly.
    """
    rs = row_size(schema)
    out = bytearray(rs)
    table = compute_assembly_table(schema)  # declaration-order; matches schema.fields

    provided = set(values.keys())
    declared = {f.name for f in schema.fields}
    if provided != declared:
        missing = declared - provided
        extra = provided - declared
        raise ValueError(
            f"pack_row values keys {sorted(provided)} do not match "
            f"schema fields {sorted(declared)} (missing={sorted(missing)}, "
            f"extra={sorted(extra)})"
        )

    for i, f in enumerate(schema.fields):
        row = table[i]
        elem_cnt = int(row["element_count"])
        byte_off = int(row["byte_offset"])
        byte_cnt = int(row["byte_count"])

        arr = values[f.name]
        expected_dtype = DTYPE_TO_NUMPY[DType(int(row["dtype_code"]))]
        if arr.dtype != expected_dtype:
            raise ValueError(
                f"pack_row[{f.name!r}]: dtype {arr.dtype} != expected {expected_dtype}"
            )
        flat = np.ascontiguousarray(arr).reshape(-1)
        if flat.size != elem_cnt:
            raise ValueError(
                f"pack_row[{f.name!r}]: element_count {flat.size} != expected {elem_cnt}"
            )

        raw = flat.tobytes()
        # raw length is guaranteed == byte_cnt because dtype.itemsize * elem_cnt
        # equals byte_cnt by FeatureField construction. No defensive check.
        out[byte_off : byte_off + byte_cnt] = raw

    return bytes(out)


# ---------------------------------------------------------------------------
# Hypothesis strategies — shared across property tests (Step 4 + Step 5).
# Lifted here from tests/property/test_serving_properties.py so Step 5's
# parity test can import them without cross-test-module fragility.
# ---------------------------------------------------------------------------


_NAME = st.text(
    alphabet=string.ascii_lowercase + string.digits + "_",
    min_size=1,
    max_size=12,
).filter(lambda s: s[0].isalpha())


_DTYPE_OPTS = st.sampled_from(list(DType))


# Shape: () (scalar) or (1..16,) or (1..4, 1..4). Keeps row sizes manageable.
_SHAPE = st.one_of(
    st.just(()),
    st.tuples(st.integers(min_value=1, max_value=16)),
    st.tuples(st.integers(min_value=1, max_value=4), st.integers(min_value=1, max_value=4)),
)


@st.composite
def _field(draw: st.DrawFn) -> FeatureField:
    return FeatureField(draw(_NAME), draw(_DTYPE_OPTS), draw(_SHAPE))


@st.composite
def field_list_strategy(draw: st.DrawFn) -> list[FeatureField]:
    """Hypothesis strategy yielding a non-empty list of unique-named
    :class:`FeatureField` objects (1-6 fields, dedup-by-name)."""
    fields = draw(st.lists(_field(), min_size=1, max_size=6))
    seen: set[str] = set()
    deduped: list[FeatureField] = []
    for f in fields:
        if f.name not in seen:
            seen.add(f.name)
            deduped.append(f)
    return deduped


_SCHEMA_VERSION_COUNTER = [0]


def build_dynamic_schema(field_list: list[FeatureField]) -> type[FeatureSchema]:
    """Build a :class:`FeatureSchema` subclass at runtime from a generated
    field list. Each call gets a unique class name so ``__init_subclass__``
    is happy and the same field list can be re-used across examples without
    conflict."""
    _SCHEMA_VERSION_COUNTER[0] += 1
    cls_name = f"_HypoSchema_{_SCHEMA_VERSION_COUNTER[0]}"
    return type(
        cls_name,
        (FeatureSchema,),
        {"version": 1, "fields": field_list},
    )


def random_value_for(field: FeatureField, rng: np.random.Generator) -> np.ndarray:
    """A random ndarray matching ``field``'s dtype + shape (post-flatten size).

    Uses moderate ranges so casts to float32 don't produce uninformative
    results (e.g. int64 is restricted to ±2^40, well above float32's
    24-bit mantissa for verifying lossy casts).
    """
    n = field.element_count
    if field.dtype is DType.FLOAT32:
        return rng.standard_normal(n).astype(np.float32)
    if field.dtype is DType.FLOAT64:
        return rng.standard_normal(n).astype(np.float64)
    if field.dtype is DType.INT32:
        return rng.integers(-(1 << 20), 1 << 20, size=n, dtype=np.int32)
    if field.dtype is DType.INT64:
        return rng.integers(-(1 << 40), 1 << 40, size=n, dtype=np.int64)
    if field.dtype is DType.UINT8:
        return rng.integers(0, 256, size=n, dtype=np.uint8)
    raise AssertionError(f"unhandled dtype {field.dtype}")
