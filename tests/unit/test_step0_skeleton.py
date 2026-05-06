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
    assert quorin.__version__ == "0.1.0"


def test_metrics_registry_has_expected_series() -> None:
    text = generate_latest(metrics.registry).decode()
    assert "quorin_read_latency_seconds" in text
    assert "quorin_gc_pause_seconds" in text
    assert "quorin_wal_lag_seconds" in text
    assert "quorin_pool_miss_total" in text


def test_counter_increments_without_http_server() -> None:
    metrics.pool_miss_total.labels(schema="smoke").inc()
    text = generate_latest(metrics.registry).decode()
    assert 'quorin_pool_miss_total{schema="smoke"}' in text


def test_histogram_observes() -> None:
    metrics.read_latency_seconds.labels(schema="smoke", path="shm").observe(5e-6)
    text = generate_latest(metrics.registry).decode()
    assert 'quorin_read_latency_seconds_bucket{le="5e-06",path="shm",schema="smoke"}' in text


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
