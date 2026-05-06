"""Unit tests for quorin._internal.pydantic_factory."""

from __future__ import annotations

import math

import numpy as np
import pydantic
import pytest

from quorin._internal.pydantic_factory import (
    _MODEL_CACHE,
    _build,
    clear_cache,
    field_order_for,
    pydantic_model_for,
)
from quorin.schema import (
    FeatureField,
    FeatureSchema,
    _hash_name,
    compile_schema,
    dtype,
)


class _ScalarSchema(FeatureSchema):
    version = 1
    fields = [
        FeatureField("a", dtype.float32),
        FeatureField("b", dtype.int64),
        FeatureField("c", dtype.uint8),
    ]


class _Mixed1DSchema(FeatureSchema):
    version = 1
    fields = [
        FeatureField("emb", dtype.float32, shape=(8,)),
        FeatureField("flags", dtype.uint8, shape=(4,)),
        FeatureField("score", dtype.float64),
    ]


class _Schema2D(FeatureSchema):
    version = 1
    fields = [
        FeatureField("matrix", dtype.float32, shape=(2, 3)),
        FeatureField("scalar", dtype.int32),
    ]


@pytest.fixture(autouse=True)
def _clear_factory_cache():
    """Each factory test starts on a clean cache so memoization assertions
    aren't poisoned by other tests in the file."""
    clear_cache()
    yield
    clear_cache()


# ---------------------------------------------------------------------------
# Memoization.
# ---------------------------------------------------------------------------


def test_pydantic_model_for_returns_same_class_on_repeat() -> None:
    m1 = pydantic_model_for(_ScalarSchema)
    m2 = pydantic_model_for(_ScalarSchema)
    assert m1 is m2


def test_field_order_for_populates_cache_lazily() -> None:
    # Cold: nothing cached
    assert _ScalarSchema not in _MODEL_CACHE
    order = field_order_for(_ScalarSchema)
    # Warm after the call
    assert _ScalarSchema in _MODEL_CACHE
    assert isinstance(order, tuple)


def test_field_order_matches_name_hash_sort() -> None:
    order = field_order_for(_Mixed1DSchema)
    table = compile_schema(_Mixed1DSchema)
    expected = tuple(
        {_hash_name(f.name): f.name for f in _Mixed1DSchema.fields}[int(h)]
        for h in table["name_hash"]
    )
    assert order == expected


def test_distinct_classes_with_same_name_get_distinct_models() -> None:
    a_cls = type(
        "DupName", (FeatureSchema,), {"version": 1, "fields": [FeatureField("x", dtype.float32)]}
    )
    b_cls = type(
        "DupName", (FeatureSchema,), {"version": 1, "fields": [FeatureField("x", dtype.float32)]}
    )
    m_a = pydantic_model_for(a_cls)
    m_b = pydantic_model_for(b_cls)
    assert m_a is not m_b


# ---------------------------------------------------------------------------
# Validation: scalars.
# ---------------------------------------------------------------------------


def test_scalar_floats_and_ints_validate() -> None:
    m = pydantic_model_for(_ScalarSchema)
    instance = m.model_validate({"a": 1.5, "b": 42, "c": 200})
    assert instance.a == 1.5
    assert instance.b == 42
    assert instance.c == 200


def test_extra_field_rejected_by_extra_forbid() -> None:
    m = pydantic_model_for(_ScalarSchema)
    with pytest.raises(pydantic.ValidationError):
        m.model_validate({"a": 1.0, "b": 1, "c": 1, "junk": 0})


def test_missing_field_rejected() -> None:
    m = pydantic_model_for(_ScalarSchema)
    with pytest.raises(pydantic.ValidationError):
        m.model_validate({"a": 1.0, "b": 1})


def test_int_out_of_range_rejected() -> None:
    m = pydantic_model_for(_ScalarSchema)
    # uint8 capped at 255
    with pytest.raises(pydantic.ValidationError):
        m.model_validate({"a": 0.0, "b": 0, "c": 256})
    # uint8 cannot be negative
    with pytest.raises(pydantic.ValidationError):
        m.model_validate({"a": 0.0, "b": 0, "c": -1})


def test_int64_min_max_accepted() -> None:
    m = pydantic_model_for(_ScalarSchema)
    m.model_validate({"a": 0.0, "b": -(2**63), "c": 0})
    m.model_validate({"a": 0.0, "b": 2**63 - 1, "c": 0})


def test_int32_overflow_rejected() -> None:
    cls = type(
        "Int32S",
        (FeatureSchema,),
        {"version": 1, "fields": [FeatureField("v", dtype.int32)]},
    )
    m = pydantic_model_for(cls)
    with pytest.raises(pydantic.ValidationError):
        m.model_validate({"v": 2**31})


# ---------------------------------------------------------------------------
# Validation: shaped fields.
# ---------------------------------------------------------------------------


def test_1d_shape_length_pinned() -> None:
    m = pydantic_model_for(_Mixed1DSchema)
    payload = {
        "emb": [0.0] * 8,
        "flags": [1, 2, 3, 4],
        "score": 1.5,
    }
    m.model_validate(payload)
    # Wrong length
    payload_short = dict(payload, emb=[0.0] * 7)
    with pytest.raises(pydantic.ValidationError):
        m.model_validate(payload_short)
    payload_long = dict(payload, emb=[0.0] * 9)
    with pytest.raises(pydantic.ValidationError):
        m.model_validate(payload_long)


def test_2d_shape_outer_and_inner_length_pinned() -> None:
    m = pydantic_model_for(_Schema2D)
    good = {"matrix": [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], "scalar": 0}
    m.model_validate(good)
    # Wrong outer
    with pytest.raises(pydantic.ValidationError):
        m.model_validate({"matrix": [[1.0, 2.0, 3.0]], "scalar": 0})
    # Wrong inner
    with pytest.raises(pydantic.ValidationError):
        m.model_validate({"matrix": [[1.0, 2.0], [3.0, 4.0]], "scalar": 0})


# ---------------------------------------------------------------------------
# NaN / Inf — invariant #12 contract (locked policy).
# ---------------------------------------------------------------------------


def test_float32_nan_accepted_on_scalar_field() -> None:
    m = pydantic_model_for(_ScalarSchema)
    instance = m.model_validate({"a": float("nan"), "b": 0, "c": 0})
    assert math.isnan(instance.a)


def test_float64_inf_and_neg_inf_accepted_on_scalar_field() -> None:
    cls = type(
        "F64S",
        (FeatureSchema,),
        {"version": 1, "fields": [FeatureField("v", dtype.float64)]},
    )
    m = pydantic_model_for(cls)
    assert m.model_validate({"v": float("inf")}).v == float("inf")
    assert m.model_validate({"v": float("-inf")}).v == float("-inf")


def test_nan_accepted_inside_1d_shape() -> None:
    m = pydantic_model_for(_Mixed1DSchema)
    emb = [0.0, float("nan"), 0.0, 0.0, float("inf"), 0.0, float("-inf"), 0.0]
    instance = m.model_validate({"emb": emb, "flags": [0, 0, 0, 0], "score": 1.0})
    assert math.isnan(instance.emb[1])
    assert instance.emb[4] == float("inf")
    assert instance.emb[6] == float("-inf")


def test_numpy_float32_scalar_nan_accepted_via_default_coercion() -> None:
    """Regression guard for the 'no global strict mode' decision.

    Strict mode would reject ``np.float32(nan)`` because numpy scalars
    aren't ``float`` instances. Default coercion accepts them; the
    assemble path is designed to round-trip the bit pattern. If a future
    PR adds ``ConfigDict(strict=True)``, this test fails loudly.
    """
    m = pydantic_model_for(_ScalarSchema)
    instance = m.model_validate({"a": np.float32(np.nan), "b": np.int64(7), "c": np.uint8(3)})
    assert math.isnan(instance.a)
    assert instance.b == 7
    assert instance.c == 3


# ---------------------------------------------------------------------------
# _build internals.
# ---------------------------------------------------------------------------


def test_build_returns_order_in_name_hash_sort() -> None:
    model, order = _build(_Mixed1DSchema)
    table = compile_schema(_Mixed1DSchema)
    hash_to_name = {_hash_name(f.name): f.name for f in _Mixed1DSchema.fields}
    expected = tuple(hash_to_name[int(h)] for h in table["name_hash"])
    assert order == expected
    # And the model is round-trippable
    payload = {"emb": [0.0] * 8, "flags": [0, 0, 0, 0], "score": 0.0}
    model.model_validate(payload)


def test_3d_shape_raises_not_implemented() -> None:
    cls = type(
        "Schema3D",
        (FeatureSchema,),
        {"version": 1, "fields": [FeatureField("cube", dtype.float32, shape=(2, 2, 2))]},
    )
    with pytest.raises(NotImplementedError):
        pydantic_model_for(cls)


def test_clear_cache_drops_both_caches() -> None:
    pydantic_model_for(_ScalarSchema)
    assert _ScalarSchema in _MODEL_CACHE
    clear_cache()
    assert _ScalarSchema not in _MODEL_CACHE
