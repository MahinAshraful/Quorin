# Pyforge

Low-latency ML feature serving via shared memory. Target: **5 µs p99** on the
warm-cache hot path; **20–50 µs p99** for 200-field schemas with a 128-dim
embedding, cold cache. Numbers are published alongside the exact hardware used.

See [`pyforge_project_spec.md`](pyforge_project_spec.md) for the design and
[`pyforge_build_steps.md`](pyforge_build_steps.md) for the step-by-step build.

## Status

Step 0 — foundation scaffolding. No feature code yet.

## Scope

- **Target OS:** Linux or WSL2. Native Windows is not supported (POSIX
  `shm_open` only).
- **Python:** 3.12+.
- **Scale ceiling:** 1 M entities × 200 features per segment. Beyond that,
  shard horizontally by `hash(entity_id) mod N`.

## Dev setup

```bash
# 1. Install uv: https://docs.astral.sh/uv/
uv sync --all-extras

# 2. Start Redis (Docker Desktop or WSL2 + docker)
docker compose -f docker/docker-compose.dev.yml up -d

# 3. Verify
uv run ruff check .
uv run mypy pyforge
uv run pytest
```

## Layout

```
pyforge/      library source
tests/        unit / integration / property / chaos
benchmarks/   pytest-benchmark cases + committed results
docker/       docker-compose.dev.yml (Redis)
.github/      ci.yml + benchmark.yml
```

## Metrics

Pyforge registers Prometheus histograms on import. If you never call
`pyforge.metrics.start_metrics_server()`, counters still increment in memory
and are visible in tests:

- `pyforge_read_latency_seconds{schema, path}` — hot-path latency
- `pyforge_gc_pause_seconds{generation}` — GC impact (Step 7)
- `pyforge_wal_lag_seconds` — producer→consumer lag (Step 10)
- `pyforge_pool_miss_total{schema}` — buffer-pool exhaustions (Step 6)
