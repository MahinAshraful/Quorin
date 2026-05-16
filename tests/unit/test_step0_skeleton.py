"""Step-0 smoke tests: package imports, metrics registry works, logging configures."""

from __future__ import annotations

from prometheus_client import generate_latest

import quorin
from quorin import (
    assembly,  # imported to prove the stub exists
    evolution,
    layout,
    metrics,
    offline,
    schema,
    serving,
    shm,
    wal,
    watchdog,
)
from quorin import logging as quorin_logging


def test_package_version_is_defined() -> None:
    # Sanity: __version__ is wired up. Exact value is verified by
    # importlib.metadata in CI (CR.A.12 / v0.1.1 — single source of
    # truth via importlib.metadata.version("quorin")).
    assert isinstance(quorin.__version__, str)
    assert quorin.__version__.count(".") >= 1


def test_metrics_registry_has_expected_series() -> None:
    text = generate_latest(metrics.registry).decode()
    # CR.A.9 / CR.A.10 (v0.1.1): read_latency_seconds and wal_lag_seconds
    # were removed (declared but never observed in v0.1.0). The remaining
    # smoke check covers a metric that IS actually populated by production
    # code — gc_pause_seconds (Step 7 GC manager).
    assert "quorin_gc_pause_seconds" in text
    assert "quorin_pool_miss_total" in text


def test_counter_increments_without_http_server() -> None:
    metrics.pool_miss_total.labels(schema="smoke").inc()
    text = generate_latest(metrics.registry).decode()
    assert 'quorin_pool_miss_total{schema="smoke"}' in text


def test_histogram_observes() -> None:
    """Smoke check that the Histogram observe() round-trips through the
    registry. Uses gc_pause_seconds (gen=2) since it's a real production
    metric, not a Step-0 placeholder.
    """
    metrics.gc_pause_seconds.labels(generation="2").observe(5e-3)
    text = generate_latest(metrics.registry).decode()
    assert 'quorin_gc_pause_seconds_bucket{generation="2"' in text


def test_logging_configures_idempotently() -> None:
    quorin_logging.configure()
    quorin_logging.configure()  # second call must not raise
    log = quorin_logging.get_logger("smoke")
    log.info("step0_smoke", check=True)


def test_expected_stub_modules_importable() -> None:
    # Every module named in the Step-0 layout must at least import - that's
    # the signal that later steps have a home to land in.
    for mod in (assembly, evolution, layout, offline, schema, serving, shm, wal, watchdog):
        assert mod is not None
