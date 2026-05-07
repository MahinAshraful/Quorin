# ADR-003: Assembled output vectors use declaration order, not hash order

**Status:** Accepted
**Date:** 2026-04-28
**Step:** 4 (serving path skeleton — pure-Python oracle)

## Decision

`quorin.serving.assemble(segment, entity_id)` returns a 1-D float32 NumPy
array whose elements are laid out in the **declaration order** of
`segment.schema.fields` — the order the user wrote the fields in their
`FeatureSchema` subclass. The hash-sorted order used by `compile_schema` and
`compute_row_offset_table` is internal scaffolding for `searchsorted`-based
lookups and never leaks into a user-facing array.

To support both orderings without an indirection in the inner loop,
`SegmentLayout` carries two parallel structured arrays of the same
`OFFSET_TABLE_DTYPE` shape:

- `row_offset_table` — hash-sorted, used by `quorin.layout.lookup` for
  O(log n) `searchsorted` against an entity ID's `name_hash`.
- `assembly_table` — declaration-order, iterated by `quorin.serving.assemble`
  (and Step 5's Numba kernel) in a tight linear walk.

Both are computed once at `compute_layout` time and stored frozen on the
`SegmentLayout` instance.

## Context

The user trains a model on a fixed feature ordering — typically the order
they declared the fields in. The hand-off from `assemble` to `model.predict`
is positional: `predict(vec)` reads `vec[0]`, `vec[1]`, …, in order. If
`vec[0]` were the embedding's first byte (because "embedding" hashed smaller
than "age") instead of the user's first declared feature, the model would
silently produce wrong predictions. There is no exception, no warning — just
worse business metrics.

Hash order was introduced in Step 1 because `searchsorted` over a
hash-sorted array is the cheapest way to map a field name to its byte
offset on the read path: O(log n) with zero allocations. That's a real
optimization for `lookup`, where every microsecond counts under the 5 µs
budget. But `assemble` doesn't need lookup — it walks every field
sequentially. Both orderings can serve their respective callers; we just
need to keep them straight.

## What we considered

1. **One table, hash order, sort/index at output time.** Rejected — would
   require a per-field translation in the assembly inner loop, exactly the
   indirection that defeats Numba auto-vectorization in Step 5.
2. **One table, declaration order, linear scan for lookup.** Rejected —
   linear `lookup` is O(n) per call, against a 5 µs budget. Unacceptable
   for schemas with more than a handful of fields.
3. **Two tables.** The chosen design. Accepts a small duplicate (n_fields ×
   25 bytes) in exchange for both paths staying optimal.

## What it costs

- **Memory:** one extra `OFFSET_TABLE_DTYPE` array per segment open. For a
  200-field schema, 200 × 25 = 5 KiB per opener. Negligible.
- **Conceptual:** every later step that touches feature data must pick the
  right table. We address this by naming (`row_offset_table` for lookup,
  `assembly_table` for assembly) and by Step 5's Numba kernel iterating
  `assembly_table` exclusively — hash order is never visible at the
  output-producing layer.

## Consequences

- **Positive:** `model.predict(vec)` works as the user expects. The
  positional contract is the user's mental model; the data structure obeys
  it.
- **Positive:** Step 5's Numba kernel iterates a contiguous structured
  array in declaration order with no per-field indirection — the SIMD-
  friendly form.
- **Negative:** every reviewer of new code in `quorin` must remember
  there are two tables. The naming carries the intent, and the pinned
  unit test catches accidental crosses.

## Validation

`tests/unit/test_serving.py::TestDeclarationOrderPinned::test_known_schema_known_output`
constructs a schema with two fields chosen so that hash order ≠ declaration
order, asserts the inversion is real, then asserts that `assemble`'s output
is in declaration order regardless. This test is the contract Step 5's
Numba parity test will validate against.

## References

- Quorin spec, "Public API target" example showing
  `vec = serving.assemble(...); model.predict(vec)`.
- Step 1's `compile_schema` — the hash-sort that ADR-003 explicitly does
  not undo.
- Step 5 (forthcoming): Numba kernel iterates `assembly_table` exclusively.
