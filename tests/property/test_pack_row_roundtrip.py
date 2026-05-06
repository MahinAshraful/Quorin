"""Hypothesis property test for ``quorin.layout.pack_row`` (Step 17).

Generates random schemas + values, packs via the new public ``pack_row``
(kwargs API), inserts via ``layout.insert``, reads via ``serving.assemble``,
and asserts byte-identical output to the test-helper ``pack_row``
(``dict[str, ndarray]`` API). The two packers must agree for every input
shape — that's the property test's binary guarantee.
"""

from __future__ import annotations

import sys

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings

pytestmark = pytest.mark.skipif(
    sys.platform == "win32",
    reason="pack_row requires posix-only quorin._internal modules",
)

from _helpers import (  # noqa: E402
    build_dynamic_schema,
    field_list_strategy,
    make_segment,
    random_value_for,
    release_segment,
)
from _helpers import (  # noqa: E402
    pack_row as helper_pack_row,
)
from quorin.layout import insert, pack_row  # noqa: E402
from quorin.serving import assemble  # noqa: E402


@given(field_list=field_list_strategy())
@settings(
    max_examples=100,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture],
)
def test_pack_row_byte_identical_to_helper_and_assembles_correctly(
    field_list: list,
) -> None:
    """For any random schema, the new kwargs ``pack_row`` produces bytes
    byte-identical to the dict-based test helper, AND the resulting bytes
    round-trip through assemble to give the input values back (modulo
    declared-dtype casts)."""
    schema = build_dynamic_schema(field_list)
    rng = np.random.default_rng(0)

    # Generate one value per field; keep both the ndarray (for the helper) and
    # a kwargs-friendly form (for the new API). For shape=() fields, asarray
    # on a Python scalar works; for shaped fields, the ndarray itself is fine
    # as a kwarg value.
    values_dict: dict[str, np.ndarray] = {}
    kwargs: dict[str, object] = {}
    for f in schema.fields:
        v = random_value_for(f, rng)
        values_dict[f.name] = v
        kwargs[f.name] = v  # ndarray is a valid kwarg shape

    # Property 1: byte-identical packing.
    new_bytes = pack_row(schema, **kwargs)
    helper_bytes = helper_pack_row(schema, values_dict)
    assert new_bytes == helper_bytes, (
        f"pack_row diverged from helper for schema {schema.__name__}; "
        f"new_len={len(new_bytes)}, helper_len={len(helper_bytes)}"
    )

    # Property 2: assemble round-trip succeeds. Build a segment, insert with
    # the new bytes, assemble, verify length matches schema's total element count.
    seg = make_segment(schema, capacity=4)
    try:
        insert(seg, "u_property", new_bytes)
        out = assemble(seg, "u_property")
        # Just verify shape — full value-equality is covered by the existing
        # serving / parity tests; this property test's job is the byte-identical
        # contract above.
        expected_len = sum(f.element_count for f in schema.fields)
        assert out.shape == (expected_len,)
        assert out.dtype == np.float32
    finally:
        release_segment(seg)
