"""Prometheus metric registry.

Histograms and gauges are registered at import time so every later step can
``from quorin.metrics import gc_pause_seconds`` and ``.observe()`` without
worrying about registration order. If no HTTP server is started, counters
still increment in memory and are visible via :func:`collect` (used in tests).

CR.A.9 / CR.A.10 (v0.1.1): two metrics declared at v0.1.0 that no
production code ever observes were removed:

* ``quorin_read_latency_seconds`` — would have measured per-call hot-path
  read latency. Wiring ``.observe()`` on every assemble call costs
  ~30 ns on the 4.48 µs p99 budget (~0.7%); v0.1.1 cannot honestly
  carry an unobserved metric AND keep the bench claim. Deferred to
  v0.2.0 as opt-in instrumentation behind an env var.
* ``quorin_wal_lag_seconds`` — would have been a gauge sampled by the
  consumer; the consumer ships ``quorin_wal_write_sync_consumer_lag_seconds``
  (a histogram) instead, and never sampled the gauge.

Operators tracking these in v0.1.0 dashboards saw zeros. v0.2.0 will
ship proper opt-in observability. See ADR-018.
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

# GC pause distribution. Populated by quorin._internal.gc_manager (Step 7).
gc_pause_seconds = Histogram(
    "quorin_gc_pause_seconds",
    "Duration of each Python GC pause observed on the serving process.",
    labelnames=("generation",),
    buckets=(1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1),
    registry=registry,
)

# Incremented whenever a buffer-pool checkout has to allocate a fresh buffer
# because the pool was empty. (Step 6.)
pool_miss_total = Counter(
    "quorin_pool_miss_total",
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
    "quorin_wal_write_latency_seconds",
    "Wall-clock latency of WAL producer writes (XADD round-trip + consumer ack for write_sync).",
    labelnames=("mode",),  # async | sync
    buckets=_WAL_LATENCY_BUCKETS,
    registry=registry,
)

wal_write_total = Counter(
    "quorin_wal_write_total",
    "WAL producer writes by mode and outcome.",
    labelnames=("mode", "outcome"),  # mode: async|sync; outcome: ok|timeout|error
    registry=registry,
)

# Lag observed by write_sync between XADD and the consumer's processed-key
# write. The producer-observed round-trip (the consumer's own lag
# instrumentation, originally planned as quorin_wal_lag_seconds, was
# removed in v0.1.1 per CR.A.10 — never wired up at v0.1.0; deferred to
# v0.2.0 opt-in observability).
wal_write_sync_consumer_lag_seconds = Histogram(
    "quorin_wal_write_sync_consumer_lag_seconds",
    "Time from WAL producer XADD to consumer processed-key set, observed by write_sync().",
    buckets=_WAL_LATENCY_BUCKETS,
    registry=registry,
)

# WAL consumer (Step 10).
wal_consumer_apply_total = Counter(
    "quorin_wal_consumer_apply_total",
    "WAL consumer message apply outcomes.",
    # ok = new insert; duplicate = existing entity_id overwritten;
    # error = apply raised (msgpack/pack/insert/append); unknown_schema =
    # schema name not in segments map (no ACK); bad_entity_id = entity_id
    # not valid UTF-8 (no ACK).
    labelnames=("outcome",),
    registry=registry,
)

wal_consumer_batch_size = Histogram(
    "quorin_wal_consumer_batch_size",
    "Number of messages per consumer apply batch (XREADGROUP COUNT bound).",
    buckets=(1, 2, 5, 10, 20, 50, 100, 200, 500),
    registry=registry,
)

wal_consumer_unknown_schema_total = Counter(
    "quorin_wal_consumer_unknown_schema_total",
    "Messages received for a schema name not in the consumer's segments map.",
    labelnames=("schema_name",),
    registry=registry,
)

# CR.B.13 (v0.1.2): increments each time the consumer sits at the
# ``2x max_pending_ack`` hard ceiling for ``_BACKPRESSURE_ALERT_SECONDS``.
# Indicates a wedged offline-writer flush (full disk, NFS stuck, deterministic
# failure on every retry). Operators alert on rate > 0 over 5 minutes.
wal_consumer_backpressure_alerted_total = Counter(
    "quorin_wal_consumer_backpressure_alerted_total",
    "Consumer back-pressure hit the hard ceiling for >= 60s (wedged flush).",
    registry=registry,
)

# Pending-ACK queue depth — depth grows when offline.flush is slow or
# stuck. Operators alert on this exceeding `max_pending_ack`.
wal_consumer_pending_ack_size = Gauge(
    "quorin_wal_consumer_pending_ack_size",
    "Messages applied online but awaiting offline.flush() before XACK.",
    registry=registry,
)

# Distribution of OfflineWriter.flush() durations. The consumer waits
# synchronously for each flush before XACK; a slow flush is the most
# likely operability surprise.
wal_consumer_flush_seconds = Histogram(
    "quorin_wal_consumer_flush_seconds",
    "Wall-clock duration of OfflineWriter.flush() calls.",
    labelnames=("outcome",),  # ok | error
    buckets=_WAL_LATENCY_BUCKETS,
    registry=registry,
)

# ParquetDatasetStore (Step 11).
offline_flush_seconds = Histogram(
    "quorin_offline_flush_seconds",
    "ParquetDatasetStore.flush() wall time, by outcome.",
    labelnames=("outcome",),  # ok | error | cancelled
    buckets=_WAL_LATENCY_BUCKETS,
    registry=registry,
)

offline_flush_rows = Histogram(
    "quorin_offline_flush_rows",
    "Rows written per ParquetDatasetStore.flush() call.",
    buckets=(1, 10, 100, 500, 1000, 5000, 10000, 50000),
    registry=registry,
)

offline_files_written_total = Counter(
    "quorin_offline_files_written_total",
    "Parquet files written, labeled by schema.",
    labelnames=("schema",),
    registry=registry,
)

offline_bytes_written_total = Counter(
    "quorin_offline_bytes_written_total",
    "Compressed Parquet bytes written, labeled by schema.",
    labelnames=("schema",),
    registry=registry,
)

# Read-side metrics for ParquetDatasetStore.read_point_in_time (Step 12).
# offline_read_files_scanned was considered in the plan and dropped: PyArrow
# does not expose post-prune file count cleanly, and offline_read_rows
# already covers volume tracking.
offline_read_seconds = Histogram(
    "quorin_offline_read_seconds",
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
    "quorin_offline_read_rows",
    "Rows returned per ParquetDatasetStore.read_point_in_time() call.",
    buckets=(1, 10, 100, 1000, 10_000, 100_000, 1_000_000),
    registry=registry,
)

# Hydration (Step 13).
# Precondition rejections (HydrationConflictError raised before the timed
# section) are NOT counted in `hydration_seconds` — they represent rejected
# calls, not failed work. The 'err' bucket counts only failures that reached
# the timed section. See `quorin.hydration.hydrate` docstring "Metrics".
hydration_seconds = Histogram(
    "quorin_hydration_seconds",
    "Wall-clock time per hydrate() call that reached the timed section. "
    "Excludes precondition rejections.",
    labelnames=("outcome",),  # ok | err
    buckets=(0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0),
    registry=registry,
)

hydration_entity_count = Histogram(
    "quorin_hydration_entity_count",
    "Entities loaded per successful hydrate() call.",
    buckets=(1, 10, 100, 1_000, 10_000, 100_000, 1_000_000),
    registry=registry,
)

# WAL consumer liveness (Step 13 cross-step touch).
# Hydration's precondition #2 reads `quorin:wal_consumer:liveness` to refuse
# a clobber while a consumer is actively writing. The consumer SETs this key
# every 10s with a 30s TTL; if Redis is reachable, age stays near zero. During
# a Redis outage the gauge climbs (see WALConsumer.run for the per-loop
# update). Operators alert on `liveness_age_seconds > 30` AND on the failure
# counter's error rate.
wal_consumer_liveness_age_seconds = Gauge(
    "quorin_wal_consumer_liveness_age_seconds",
    "Seconds since the WAL consumer's last successful liveness-key SET. "
    "Climbs above LIVENESS_REFRESH_INTERVAL_SECONDS during Redis outages.",
    registry=registry,
)

wal_consumer_liveness_refresh_failures = Counter(
    "quorin_wal_consumer_liveness_refresh_failures_total",
    "WAL consumer liveness-key SET attempts by outcome.",
    labelnames=("outcome",),  # ok | error
    registry=registry,
)

# Heartbeat producer (Step 14).
# Per-process daemon thread writes quorin:heartbeats {pid} -> "{create_ns}:{wall_ns}"
# every HEARTBEAT_INTERVAL_SECONDS. Watchdog reads the hash and detects dead PIDs
# via consecutive missed advances of wall_ns. See quorin._internal.heartbeat.
heartbeat_writes_total = Counter(
    "quorin_heartbeat_writes_total",
    "Heartbeat HSET attempts by outcome.",
    labelnames=("outcome",),  # ok | redis_error
    registry=registry,
)

heartbeat_age_seconds = Gauge(
    "quorin_heartbeat_age_seconds",
    "Seconds since the heartbeat thread's last successful HSET. "
    "Climbs above HEARTBEAT_INTERVAL_SECONDS during Redis outages.",
    registry=registry,
)

# Watchdog (Step 14). Counters/histograms fed by WatchdogState.run_one_tick().
watchdog_dead_pids_total = Counter(
    "quorin_watchdog_dead_pids_total",
    "PIDs declared dead by the watchdog (cross-check confirmed dead).",
    registry=registry,
)

watchdog_segments_unlinked_total = Counter(
    "quorin_watchdog_segments_unlinked_total",
    "Segments unlinked by the watchdog, by source path.",
    labelnames=("path",),  # dead_pid | cleanup_queue
    registry=registry,
)

watchdog_tick_seconds = Histogram(
    "quorin_watchdog_tick_seconds",
    "Wall-clock duration of a single watchdog tick (HGETALL + cleanup + drain).",
    buckets=(1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1, 5e-1, 1.0, 5.0),
    registry=registry,
)

watchdog_cross_check_unverifiable_total = Counter(
    "quorin_watchdog_cross_check_unverifiable_total",
    "Cross-check could not verify liveness; conservative skip (do NOT unlink). "
    "Reasons: access_denied (psutil.AccessDenied — typically cross-UID without "
    "CAP_SYS_PTRACE), zombie (psutil.ZombieProcess).",
    labelnames=("reason",),  # access_denied | zombie
    registry=registry,
)

watchdog_malformed_heartbeat_total = Counter(
    "quorin_watchdog_malformed_heartbeat_total",
    "Heartbeat hash entries that failed to parse as '{create_ns}:{wall_ns}'. "
    "Likely cause: operator HSET to quorin:heartbeats with a non-conforming "
    "value. Defensive skip — tick proceeds.",
    registry=registry,
)

watchdog_pid_reuse_abort_total = Counter(
    "quorin_watchdog_pid_reuse_abort_total",
    "Dead-PID cleanup Lua aborted because heartbeats[pid].create_time mismatched "
    "the watchdog's expected value — kernel reused PID for a new process between "
    "cross-check and Lua execution. Non-zero count means at least one race was "
    "caught; alert on rate >0.",
    registry=registry,
)

# Schema evolution (Step 15). Precondition rejections (UpgradeConflictError /
# UpgradeIncompatibleError raised before the timed section) are NOT counted in
# `evolution_seconds` — they represent rejected calls, not failed work. Same
# "ok | err" labelling as hydration.
evolution_seconds = Histogram(
    "quorin_evolution_seconds",
    "Wall-clock time per upgrade_schema() call that reached the timed section. "
    "Excludes precondition rejections.",
    labelnames=("outcome",),  # ok | err
    buckets=(0.1, 0.5, 1.0, 5.0, 10.0, 30.0, 60.0, 300.0),
    registry=registry,
)

evolution_entity_count = Histogram(
    "quorin_evolution_entity_count",
    "Entities translated per successful upgrade_schema() call.",
    buckets=(1, 100, 10_000, 100_000, 1_000_000, 10_000_000),
    registry=registry,
)

evolution_consumer_pause_seconds = Histogram(
    "quorin_evolution_consumer_pause_seconds",
    "Wall-clock duration the WAL consumer was parked due to a Step 15 upgrade "
    "pause-key. Operators alert on p99 > 30s — long pauses suggest a stuck "
    "orchestrator or insufficient WAL drain before upgrade.",
    buckets=(0.1, 0.5, 1.0, 5.0, 30.0, 60.0),
    registry=registry,
)

evolution_consumer_attach_seconds = Histogram(
    "quorin_evolution_consumer_attach_seconds",
    "Wall-clock duration the upgrade orchestrator waited for the first consumer "
    "to attach to the new segment after the flip. Bucketed up to wait_seconds "
    "default (60s); a count near the 60s ceiling means the orchestrator timed "
    "out and the new segment is at risk of being reaped by the watchdog.",
    buckets=(0.1, 1.0, 5.0, 30.0, 60.0),
    registry=registry,
)

# WAL consumer Step 15 cross-step touches. Both fire from
# `quorin.wal_consumer._run_one_iter` (segment reopen on pause-clear) and
# `_apply` (poison-pill on stale producer message).
wal_consumer_schema_crc_mismatch_total = Counter(
    "quorin_wal_consumer_schema_crc_mismatch_total",
    "WAL consumer's reopen after upgrade-pause-clear failed because its "
    "cached schema class is stale (CRC mismatch on new segment). Operator "
    "must restart the consumer with new schema code. Alert on > 0.",
    registry=registry,
)

wal_consumer_poison_pill_total = Counter(
    "quorin_wal_consumer_poison_pill_total",
    "WAL message rejected by `pack_row_from_list` because a stale producer "
    "wrote with old schema during the upgrade window. Message stays in PEL "
    "(not XACKed) so XPENDING reflects it. Alert on > 0 correlated with "
    "non-empty PEL — operator restarts producer with new schema code.",
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
evolution_seconds.labels(outcome="ok")
evolution_seconds.labels(outcome="err")


def start_metrics_server(port: int = 9100) -> None:
    """Expose the Quorin registry on ``/metrics``.

    This is optional. Callers who do not start the server still get in-memory
    counters that can be scraped by other means or inspected in tests.
    """
    start_http_server(port, registry=registry)
