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

# WAL producer (Step 9).
# Buckets cover the 100 us - 1 s range: write() p99 target is 2 ms but
# write_sync() can run up to ~100 ms under steady consumer load.
_WAL_LATENCY_BUCKETS = (
    1e-4,
    2e-4,
    5e-4,
    1e-3,
    2e-3,
    5e-3,
    1e-2,
    2e-2,
    5e-2,
    1e-1,
    2e-1,
    5e-1,
    1.0,
)

wal_write_latency_seconds = Histogram(
    "pyforge_wal_write_latency_seconds",
    "Wall-clock latency of WAL producer writes (XADD round-trip + consumer ack for write_sync).",
    labelnames=("mode",),  # async | sync
    buckets=_WAL_LATENCY_BUCKETS,
    registry=registry,
)

wal_write_total = Counter(
    "pyforge_wal_write_total",
    "WAL producer writes by mode and outcome.",
    labelnames=("mode", "outcome"),  # mode: async|sync; outcome: ok|timeout|error
    registry=registry,
)

# Lag observed by write_sync between XADD and the consumer's processed-key
# write. Distinct from pyforge_wal_lag_seconds (a gauge sampled by the
# consumer); this one measures the producer-observed round-trip.
wal_write_sync_consumer_lag_seconds = Histogram(
    "pyforge_wal_write_sync_consumer_lag_seconds",
    "Time from WAL producer XADD to consumer processed-key set, observed by write_sync().",
    buckets=_WAL_LATENCY_BUCKETS,
    registry=registry,
)

# WAL consumer (Step 10).
wal_consumer_apply_total = Counter(
    "pyforge_wal_consumer_apply_total",
    "WAL consumer message apply outcomes.",
    # ok = new insert; duplicate = existing entity_id overwritten;
    # error = apply raised (msgpack/pack/insert/append); unknown_schema =
    # schema name not in segments map (no ACK); bad_entity_id = entity_id
    # not valid UTF-8 (no ACK).
    labelnames=("outcome",),
    registry=registry,
)

wal_consumer_batch_size = Histogram(
    "pyforge_wal_consumer_batch_size",
    "Number of messages per consumer apply batch (XREADGROUP COUNT bound).",
    buckets=(1, 2, 5, 10, 20, 50, 100, 200, 500),
    registry=registry,
)

wal_consumer_unknown_schema_total = Counter(
    "pyforge_wal_consumer_unknown_schema_total",
    "Messages received for a schema name not in the consumer's segments map.",
    labelnames=("schema_name",),
    registry=registry,
)

# Pending-ACK queue depth — depth grows when offline.flush is slow or
# stuck. Operators alert on this exceeding `max_pending_ack`.
wal_consumer_pending_ack_size = Gauge(
    "pyforge_wal_consumer_pending_ack_size",
    "Messages applied online but awaiting offline.flush() before XACK.",
    registry=registry,
)

# Distribution of OfflineWriter.flush() durations. The consumer waits
# synchronously for each flush before XACK; a slow flush is the most
# likely operability surprise.
wal_consumer_flush_seconds = Histogram(
    "pyforge_wal_consumer_flush_seconds",
    "Wall-clock duration of OfflineWriter.flush() calls.",
    labelnames=("outcome",),  # ok | error
    buckets=_WAL_LATENCY_BUCKETS,
    registry=registry,
)

# ParquetDatasetStore (Step 11).
offline_flush_seconds = Histogram(
    "pyforge_offline_flush_seconds",
    "ParquetDatasetStore.flush() wall time, by outcome.",
    labelnames=("outcome",),  # ok | error | cancelled
    buckets=_WAL_LATENCY_BUCKETS,
    registry=registry,
)

offline_flush_rows = Histogram(
    "pyforge_offline_flush_rows",
    "Rows written per ParquetDatasetStore.flush() call.",
    buckets=(1, 10, 100, 500, 1000, 5000, 10000, 50000),
    registry=registry,
)

offline_files_written_total = Counter(
    "pyforge_offline_files_written_total",
    "Parquet files written, labeled by schema.",
    labelnames=("schema",),
    registry=registry,
)

offline_bytes_written_total = Counter(
    "pyforge_offline_bytes_written_total",
    "Compressed Parquet bytes written, labeled by schema.",
    labelnames=("schema",),
    registry=registry,
)

# Read-side metrics for ParquetDatasetStore.read_point_in_time (Step 12).
# offline_read_files_scanned was considered in the plan and dropped: PyArrow
# does not expose post-prune file count cleanly, and offline_read_rows
# already covers volume tracking.
offline_read_seconds = Histogram(
    "pyforge_offline_read_seconds",
    "ParquetDatasetStore.read_point_in_time() wall time, by outcome.",
    labelnames=("outcome",),  # ok | error
    buckets=(
        1e-3,
        5e-3,
        1e-2,
        5e-2,
        1e-1,
        5e-1,
        1.0,
        2.0,
        5.0,
        10.0,
        30.0,
        60.0,
    ),
    registry=registry,
)

offline_read_rows = Histogram(
    "pyforge_offline_read_rows",
    "Rows returned per ParquetDatasetStore.read_point_in_time() call.",
    buckets=(1, 10, 100, 1000, 10_000, 100_000, 1_000_000),
    registry=registry,
)

# Hydration (Step 13).
# Precondition rejections (HydrationConflictError raised before the timed
# section) are NOT counted in `hydration_seconds` — they represent rejected
# calls, not failed work. The 'err' bucket counts only failures that reached
# the timed section. See `pyforge.hydration.hydrate` docstring "Metrics".
hydration_seconds = Histogram(
    "pyforge_hydration_seconds",
    "Wall-clock time per hydrate() call that reached the timed section. "
    "Excludes precondition rejections.",
    labelnames=("outcome",),  # ok | err
    buckets=(0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0),
    registry=registry,
)

hydration_entity_count = Histogram(
    "pyforge_hydration_entity_count",
    "Entities loaded per successful hydrate() call.",
    buckets=(1, 10, 100, 1_000, 10_000, 100_000, 1_000_000),
    registry=registry,
)

# WAL consumer liveness (Step 13 cross-step touch).
# Hydration's precondition #2 reads `pyforge:wal_consumer:liveness` to refuse
# a clobber while a consumer is actively writing. The consumer SETs this key
# every 10s with a 30s TTL; if Redis is reachable, age stays near zero. During
# a Redis outage the gauge climbs (see WALConsumer.run for the per-loop
# update). Operators alert on `liveness_age_seconds > 30` AND on the failure
# counter's error rate.
wal_consumer_liveness_age_seconds = Gauge(
    "pyforge_wal_consumer_liveness_age_seconds",
    "Seconds since the WAL consumer's last successful liveness-key SET. "
    "Climbs above LIVENESS_REFRESH_INTERVAL_SECONDS during Redis outages.",
    registry=registry,
)

wal_consumer_liveness_refresh_failures = Counter(
    "pyforge_wal_consumer_liveness_refresh_failures_total",
    "WAL consumer liveness-key SET attempts by outcome.",
    labelnames=("outcome",),  # ok | error
    registry=registry,
)

# Heartbeat producer (Step 14).
# Per-process daemon thread writes pyforge:heartbeats {pid} -> "{create_ns}:{wall_ns}"
# every HEARTBEAT_INTERVAL_SECONDS. Watchdog reads the hash and detects dead PIDs
# via consecutive missed advances of wall_ns. See pyforge._internal.heartbeat.
heartbeat_writes_total = Counter(
    "pyforge_heartbeat_writes_total",
    "Heartbeat HSET attempts by outcome.",
    labelnames=("outcome",),  # ok | redis_error
    registry=registry,
)

heartbeat_age_seconds = Gauge(
    "pyforge_heartbeat_age_seconds",
    "Seconds since the heartbeat thread's last successful HSET. "
    "Climbs above HEARTBEAT_INTERVAL_SECONDS during Redis outages.",
    registry=registry,
)

# Watchdog (Step 14). Counters/histograms fed by WatchdogState.run_one_tick().
watchdog_dead_pids_total = Counter(
    "pyforge_watchdog_dead_pids_total",
    "PIDs declared dead by the watchdog (cross-check confirmed dead).",
    registry=registry,
)

watchdog_segments_unlinked_total = Counter(
    "pyforge_watchdog_segments_unlinked_total",
    "Segments unlinked by the watchdog, by source path.",
    labelnames=("path",),  # dead_pid | cleanup_queue
    registry=registry,
)

watchdog_tick_seconds = Histogram(
    "pyforge_watchdog_tick_seconds",
    "Wall-clock duration of a single watchdog tick (HGETALL + cleanup + drain).",
    buckets=(1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1, 5e-1, 1.0, 5.0),
    registry=registry,
)

watchdog_cross_check_unverifiable_total = Counter(
    "pyforge_watchdog_cross_check_unverifiable_total",
    "Cross-check could not verify liveness; conservative skip (do NOT unlink). "
    "Reasons: access_denied (psutil.AccessDenied — typically cross-UID without "
    "CAP_SYS_PTRACE), zombie (psutil.ZombieProcess).",
    labelnames=("reason",),  # access_denied | zombie
    registry=registry,
)

watchdog_malformed_heartbeat_total = Counter(
    "pyforge_watchdog_malformed_heartbeat_total",
    "Heartbeat hash entries that failed to parse as '{create_ns}:{wall_ns}'. "
    "Likely cause: operator HSET to pyforge:heartbeats with a non-conforming "
    "value. Defensive skip — tick proceeds.",
    registry=registry,
)

watchdog_pid_reuse_abort_total = Counter(
    "pyforge_watchdog_pid_reuse_abort_total",
    "Dead-PID cleanup Lua aborted because heartbeats[pid].create_time mismatched "
    "the watchdog's expected value — kernel reused PID for a new process between "
    "cross-check and Lua execution. Non-zero count means at least one race was "
    "caught; alert on rate >0.",
    registry=registry,
)

# Pre-warm Histogram/Counter label children at module load so the first
# observe() inside a timed/locked section doesn't pay the tuple-key + dict-slot
# allocation cost (Step 7 GC lesson — see CLAUDE.md §8 "GC callback bodies
# must not allocate"). Hydration is multi-second so this is academic for
# `hydration_seconds`, but the liveness counter is touched inside the WAL
# consumer's hot loop where it matters.
hydration_seconds.labels(outcome="ok")
hydration_seconds.labels(outcome="err")
wal_consumer_liveness_refresh_failures.labels(outcome="ok")
wal_consumer_liveness_refresh_failures.labels(outcome="error")
heartbeat_writes_total.labels(outcome="ok")
heartbeat_writes_total.labels(outcome="redis_error")
watchdog_segments_unlinked_total.labels(path="dead_pid")
watchdog_segments_unlinked_total.labels(path="cleanup_queue")
watchdog_cross_check_unverifiable_total.labels(reason="access_denied")
watchdog_cross_check_unverifiable_total.labels(reason="zombie")


def start_metrics_server(port: int = 9100) -> None:
    """Expose the Pyforge registry on ``/metrics``.

    This is optional. Callers who do not start the server still get in-memory
    counters that can be scraped by other means or inspected in tests.
    """
    start_http_server(port, registry=registry)
