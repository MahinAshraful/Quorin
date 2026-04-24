"""Prometheus metric registry.

Histograms and gauges are registered at import time so every later step can
``from pyforge.metrics import read_latency_seconds`` and ``.observe()`` without
worrying about registration order. If no HTTP server is started, counters
still increment in memory and are visible via :func:`collect` (used in tests).
"""

from __future__ import annotations

from prometheus_client import (
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    start_http_server,
)

# Dedicated registry so the library never collides with an application's
# default registry. ``start_metrics_server`` exposes this one explicitly.
registry = CollectorRegistry()

# Read-path latency, labelled by the path taken (shm vs redis) and by schema.
# Buckets target the 5 us - 1 ms range; anything slower than 1 ms is a bug.
_LATENCY_BUCKETS = (
    1e-6,
    2e-6,
    5e-6,
    1e-5,
    2e-5,
    5e-5,
    1e-4,
    2e-4,
    5e-4,
    1e-3,
    2e-3,
    5e-3,
    1e-2,
)

read_latency_seconds = Histogram(
    "pyforge_read_latency_seconds",
    "Per-call read latency on the serving hot path.",
    labelnames=("schema", "path"),
    buckets=_LATENCY_BUCKETS,
    registry=registry,
)

# GC pause distribution. Populated by pyforge._internal.gc_manager (Step 7).
gc_pause_seconds = Histogram(
    "pyforge_gc_pause_seconds",
    "Duration of each Python GC pause observed on the serving process.",
    labelnames=("generation",),
    buckets=(1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1),
    registry=registry,
)

# Lag between WAL XADD and the consumer's processed-sidetable write. (Step 10.)
wal_lag_seconds = Gauge(
    "pyforge_wal_lag_seconds",
    "Seconds between WAL producer XADD and consumer acknowledgement.",
    registry=registry,
)

# Incremented whenever a buffer-pool checkout has to allocate a fresh buffer
# because the pool was empty. (Step 6.)
pool_miss_total = Counter(
    "pyforge_pool_miss_total",
    "Buffer pool checkouts that fell through to a fresh allocation.",
    labelnames=("schema",),
    registry=registry,
)


def start_metrics_server(port: int = 9100) -> None:
    """Expose the Pyforge registry on ``/metrics``.

    This is optional. Callers who do not start the server still get in-memory
    counters that can be scraped by other means or inspected in tests.
    """
    start_http_server(port, registry=registry)
