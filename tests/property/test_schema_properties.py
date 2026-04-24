"""Hypothesis-driven property tests for pyforge.schema.

Each test generates random valid schemas (1-100 unique-named fields, random
dtypes and shapes) and asserts an invariant that must hold for every possible
schema — not just the hand-picked examples in ``tests/unit/test_schema.py``.
"""

from __future__ import annotations

import string

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from pyforge.schema import (
    CACHE_LINE_SIZE,
    HEADER_SIZE,
    DType,
    FeatureField,
    FeatureSchema,
    compile_schema,
    total_segment_size,
)

# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

_NAME_ALPHABET = string.ascii_letters + string.digits + "_"

_names = st.text(alphabet=_NAME_ALPHABET, min_size=1, max_size=32)

_dtypes = st.sampled_from(list(DType))

_shapes = st.one_of(
    st.just(()),
    st.lists(st.integers(min_value=1, max_value=256), min_size=1, max_size=3).map(tuple),
)


@st.composite
def _feature_fields(draw: st.DrawFn) -> list[FeatureField]:
    n = draw(st.integers(min_value=1, max_value=100))
    names = draw(st.lists(_names, min_size=n, max_size=n, unique=True))
    fields: list[FeatureField] = []
    for nm in names:
        fields.append(FeatureField(nm, draw(_dtypes), shape=draw(_shapes)))
    return fields


def _make_schema(field_list: list[FeatureField]) -> type[FeatureSchema]:
    """Dynamically build a FeatureSchema subclass for a given list of fields."""
    return type(
        "TestSchema",
        (FeatureSchema,),
        {"version": 1, "fields": field_list},
    )


# ---------------------------------------------------------------------------
# Invariants
# ---------------------------------------------------------------------------

_HYPO = settings(
    max_examples=200,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)


@_HYPO
@given(fields=_feature_fields())
def test_all_byte_offsets_multiples_of_64(fields: list[FeatureField]) -> None:
    t = compile_schema(_make_schema(fields))
    assert all(int(o) % CACHE_LINE_SIZE == 0 for o in t["byte_offset"])


@_HYPO
@given(fields=_feature_fields())
def test_field_byte_ranges_do_not_overlap(fields: list[FeatureField]) -> None:
    t = compile_schema(_make_schema(fields))
    pairs = sorted(zip(t["byte_offset"].tolist(), t["byte_count"].tolist(), strict=True))
    for i in range(len(pairs) - 1):
        start_i, count_i = pairs[i]
        start_next, _ = pairs[i + 1]
        assert start_i + count_i <= start_next


@_HYPO
@given(fields=_feature_fields())
def test_total_segment_size_covers_every_field(fields: list[FeatureField]) -> None:
    schema = _make_schema(fields)
    t = compile_schema(schema)
    total = total_segment_size(schema)
    ends = (t["byte_offset"] + t["byte_count"]).astype("int64")
    assert int(ends.max()) <= total


@_HYPO
@given(fields=_feature_fields())
def test_total_size_at_least_header_plus_sum_byte_counts(
    fields: list[FeatureField],
) -> None:
    schema = _make_schema(fields)
    total = total_segment_size(schema)
    assert total >= HEADER_SIZE + sum(f.byte_count for f in fields)


@_HYPO
@given(fields=_feature_fields())
def test_name_hashes_unique_within_a_schema(fields: list[FeatureField]) -> None:
    t = compile_schema(_make_schema(fields))
    assert len({int(h) for h in t["name_hash"]}) == len(fields)


@_HYPO
@given(fields=_feature_fields())
def test_offset_table_sorted_by_name_hash(fields: list[FeatureField]) -> None:
    t = compile_schema(_make_schema(fields))
    hashes = t["name_hash"]
    for i in range(len(hashes) - 1):
        assert int(hashes[i]) <= int(hashes[i + 1])


@_HYPO
@given(fields=_feature_fields())
def test_compile_is_deterministic(fields: list[FeatureField]) -> None:
    schema = _make_schema(fields)
    t1 = compile_schema(schema)
    t2 = compile_schema(schema)
    assert t1.tobytes() == t2.tobytes()
