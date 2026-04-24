"""Placeholder benchmark so ``pytest benchmarks/`` has at least one case to run.

Real benchmarks (hot-path latency, batch throughput, GC impact, buffer pool,
hydration) are added as their steps land. Each gets a baseline JSON committed
to ``benchmarks/results/`` and a regression threshold in
``benchmarks/regression/thresholds.yml``.
"""

from __future__ import annotations


def test_noop_benchmark(benchmark) -> None:
    benchmark(lambda: 1 + 1)
