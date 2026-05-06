# Changelog

All notable changes to Quorin will be documented in this file.

## v0.1.0 — 2026-05-06

Initial release. Feature-complete; **5 µs p99 spec MET on native CI** at the
warm-cache 4-field assemble path.

### Highlights

- **5 µs p99 substantiated** on GitHub Actions ubuntu-latest at N=20
  fresh-subprocesses (`headline_4_field_warm` median_p99 = **4.48 µs**;
  see [`benchmarks/results/n20/headline_4_field_warm_n20.json`](benchmarks/results/n20/headline_4_field_warm_n20.json)).
- 17 numbered build steps shipped (Steps 0 — 16c-d).
- 758 tests passing across unit / property / integration / chaos / benchmark
  layers.
- 17 ADRs ([`docs/adr/`](docs/adr/)) documenting every load-bearing design
  decision.
- 30+ regression gates enforced in CI on every PR via
  [`benchmarks/regression/tier1.yml`](benchmarks/regression/tier1.yml).
- New public helper: [`quorin.layout.pack_row`](quorin/layout.py) — kwargs API
  for the synchronous insert path.

### Known scope boundaries

- Single-node only. Beyond ~1M entities, shard horizontally by
  `hash(entity_id) mod N` across multiple Quorin instances.
- Linux / WSL2. macOS untested in CI; native Windows out of scope (POSIX
  shared memory required).
- Requires Redis 7.2+ on the control path. The hot read path never touches
  Redis (per [ADR-002](docs/adr/002-per-open-refcounting.md)).

### Project history note

Quorin's internal-development codename was **Pyforge**. The ADR archive
(`docs/adr/`), CLAUDE.md, and gitignored `progress/` journal continue to
refer to the codebase by the codename for historical continuity. The
published package is `quorin`. Functionally identical.

The pre-implementation `pyforge_project_spec.md` and `pyforge_build_steps.md`
planning artifacts were deleted at v0.1.0; their value-prop content lives in
the README, and their design-decision content lives in the ADRs (which are
the canonical record). Git history preserves both files for archeology.

### Acknowledgments

Built on numpy, numba, pyarrow, redis-py, pydantic, posix-ipc, structlog,
prometheus-client. Thanks to all upstream maintainers.
