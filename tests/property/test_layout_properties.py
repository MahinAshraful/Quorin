"""Hypothesis-driven property tests for quorin.layout.

Generates random valid entity-ID sets, inserts them, and asserts the spec
invariants hold for every generated case — the kind of edge-case coverage
hand-written tests can't realistically achieve.
"""

from __future__ import annotations

import os
import string
import struct
import sys
import uuid

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

pytestmark = pytest.mark.skipif(
    sys.platform == "win32",
    reason="layout requires POSIX (Linux/WSL2)",
)

from quorin._internal import posix_shm  # noqa: E402
from quorin._internal.crc import crc32_of_bytes  # noqa: E402
from quorin.layout import (  # noqa: E402
    OCCUPIED,
    _slot_table_view,
    compute_layout,
    initialize_segment_regions,
    insert,
    iterate_occupied,
    lookup,
)
from quorin.schema import (  # noqa: E402
    FeatureField,
    FeatureSchema,
    compile_schema,
    dtype,
)
from quorin.shm import HEADER_FMT, HEADER_LEN, MAGIC, Segment  # noqa: E402


class _Schema(FeatureSchema):
    version = 1
    fields = [FeatureField("v", dtype.int32)]


def _unique_name() -> str:
    return f"quorin_proptest_{os.getpid()}_{uuid.uuid4().hex[:8]}"


def _make_segment(capacity: int) -> Segment:
    layout = compute_layout(_Schema, capacity=capacity)
    name = _unique_name()
    handle = posix_shm.create(name, layout.total_size)
    crc = crc32_of_bytes(compile_schema(_Schema).tobytes())
    handle.buf[:HEADER_LEN] = struct.pack(HEADER_FMT, MAGIC, int(_Schema.version), crc, capacity)
    initialize_segment_regions(handle.buf, layout)
    return Segment(name=name, schema=_Schema, handle=handle, layout=layout)


def _release(seg: Segment) -> None:
    posix_shm.close(seg.handle)
    posix_shm.unlink(seg.name)


def _count_occupied(seg: Segment) -> int:
    """Count OCCUPIED slots without leaving a numpy view bound in the caller's
    frame.

    A structured-array view created via ``np.frombuffer(memoryview, ...)``
    keeps the underlying buffer exported, which prevents ``mmap.close()`` from
    succeeding. By doing the count inside a function, the view is local to
    that frame and refcounted to zero on return — which is the production
    pattern anyway (views are short-lived, segments are long-lived).
    """
    return int((_slot_table_view(seg.mmap_view, seg.layout)["flags"] == OCCUPIED).sum())


# Hypothesis strategies — bounded to keep the per-case cost low.
_id_chars = string.ascii_letters + string.digits + "_-"
_entity_ids = st.text(alphabet=_id_chars, min_size=1, max_size=24)


@st.composite
def _id_set_and_capacity(draw: st.DrawFn) -> tuple[set[str], int]:
    capacity = draw(st.integers(min_value=2, max_value=128))
    # Cap inserted size at capacity so we don't trip CapacityExceeded.
    n_to_insert = draw(st.integers(min_value=1, max_value=capacity))
    ids = draw(st.sets(_entity_ids, min_size=n_to_insert, max_size=n_to_insert))
    return ids, capacity


_HYPO = settings(
    max_examples=150,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture],
)


@_HYPO
@given(payload=_id_set_and_capacity())
def test_every_inserted_id_is_findable(payload) -> None:
    ids, capacity = payload
    seg = _make_segment(capacity)
    try:
        row_data = bytes(seg.layout.row_size)
        for eid in ids:
            insert(seg, eid, row_data)
        for eid in ids:
            assert lookup(seg, eid) is not None
    finally:
        _release(seg)


@_HYPO
@given(payload=_id_set_and_capacity())
def test_no_false_positive_for_unrelated_id(payload) -> None:
    ids, capacity = payload
    seg = _make_segment(capacity)
    try:
        row_data = bytes(seg.layout.row_size)
        for eid in ids:
            insert(seg, eid, row_data)
        # Pick a string that's definitely not in the set.
        unrelated = "ZZZNEVER_INSERTED_" + uuid.uuid4().hex[:8]
        assert unrelated not in ids
        assert lookup(seg, unrelated) is None
    finally:
        _release(seg)


@_HYPO
@given(payload=_id_set_and_capacity())
def test_iterate_occupied_matches_insertion_set(payload) -> None:
    ids, capacity = payload
    seg = _make_segment(capacity)
    try:
        row_data = bytes(seg.layout.row_size)
        for eid in ids:
            insert(seg, eid, row_data)
        seen = {entity_id for entity_id, _ in iterate_occupied(seg)}
        assert seen == ids
    finally:
        _release(seg)


@_HYPO
@given(payload=_id_set_and_capacity())
def test_distinct_ids_produce_distinct_row_offsets(payload) -> None:
    ids, capacity = payload
    seg = _make_segment(capacity)
    try:
        row_data = bytes(seg.layout.row_size)
        for eid in ids:
            insert(seg, eid, row_data)
        offsets = {lookup(seg, eid) for eid in ids}
        # Every inserted id has a unique row offset (no aliasing).
        assert len(offsets) == len(ids)
        assert None not in offsets
    finally:
        _release(seg)


@_HYPO
@given(payload=_id_set_and_capacity())
def test_n_distinct_ids_produce_n_occupied_slots(payload) -> None:
    ids, capacity = payload
    seg = _make_segment(capacity)
    try:
        row_data = bytes(seg.layout.row_size)
        for eid in ids:
            insert(seg, eid, row_data)
        assert _count_occupied(seg) == len(ids)
    finally:
        _release(seg)


@_HYPO
@given(payload=_id_set_and_capacity())
def test_insertion_order_does_not_affect_findability(payload) -> None:
    ids, capacity = payload
    ids_list = list(ids)
    if len(ids_list) < 2:
        return  # nothing to permute meaningfully

    # Insert in original order
    seg_a = _make_segment(capacity)
    # Insert in reversed order
    seg_b = _make_segment(capacity)
    try:
        row_data = bytes(seg_a.layout.row_size)
        for eid in ids_list:
            insert(seg_a, eid, row_data)
        for eid in reversed(ids_list):
            insert(seg_b, eid, row_data)

        # Both segments must find every id.
        for eid in ids_list:
            assert lookup(seg_a, eid) is not None
            assert lookup(seg_b, eid) is not None
    finally:
        _release(seg_a)
        _release(seg_b)


@_HYPO
@given(payload=_id_set_and_capacity())
def test_re_insert_does_not_change_occupied_count(payload) -> None:
    ids, capacity = payload
    seg = _make_segment(capacity)
    try:
        row_data_v1 = bytes([0xAA]) * seg.layout.row_size
        row_data_v2 = bytes([0xBB]) * seg.layout.row_size

        for eid in ids:
            insert(seg, eid, row_data_v1)
        first_count = _count_occupied(seg)

        # Update every entity (upsert). Slot count must not change.
        for eid in ids:
            insert(seg, eid, row_data_v2)
        second_count = _count_occupied(seg)

        assert first_count == second_count == len(ids)

        # And the latest data is what's returned.
        for eid in ids:
            off = lookup(seg, eid)
            assert bytes(seg.mmap_view[off : off + seg.layout.row_size]) == row_data_v2
    finally:
        _release(seg)
