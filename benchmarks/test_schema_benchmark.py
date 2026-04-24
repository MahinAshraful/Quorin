"""Benchmarks for pyforge.schema.

Spec target: ``compile_schema`` completes a 200-field schema in under 1 ms
on the dev machine. Not on the hot path, but slow compile here would drag
the test suite and mask later regressions.
"""

from __future__ import annotations

from pyforge.schema import (
    DType,
    FeatureField,
    FeatureSchema,
    _hash_name,
    compile_schema,
    dtype,
    total_segment_size,
)


def _make_large_schema(n: int) -> type[FeatureSchema]:
    """Build an n-field schema cycling through four dtypes with periodic shapes."""
    rotating: list[DType] = [
        dtype.float32,
        dtype.float64,
        dtype.int32,
        dtype.int64,
    ]
    fields = [
        FeatureField(
            name=f"field_{i:04d}",
            dtype=rotating[i % 4],
            shape=() if i % 5 else (16,),
        )
        for i in range(n)
    ]
    return type("BigSchema", (FeatureSchema,), {"version": 1, "fields": fields})


def test_compile_schema_200_fields(benchmark) -> None:
    schema = _make_large_schema(200)
    t = benchmark(compile_schema, schema)
    assert t.shape == (200,)


def test_compile_schema_10_fields(benchmark) -> None:
    schema = _make_large_schema(10)
    benchmark(compile_schema, schema)


def test_total_segment_size_200_fields(benchmark) -> None:
    schema = _make_large_schema(200)
    size = benchmark(total_segment_size, schema)
    assert size % 4096 == 0


def test_hash_name_single_call(benchmark) -> None:
    benchmark(_hash_name, "session_count_7d")
