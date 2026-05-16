# Operations Guide

This is the operator runbook for `quorin`. It complements
[`USAGE.md`](USAGE.md) (which covers the public API) with the
deployment-side concerns: securing Redis, monitoring the watchdog,
recovering from common failure modes, and the redis-py
fork-unsafe-pool footgun every gunicorn deployment hits.

> v0.1.1 introduces this file. Earlier versions had ops content
> scattered across the README, USAGE.md, and ADRs; consolidating
> here gives operators a single source of truth (CR.C.10 / CR.C.11 /
> CR.E.4 / CR.E.6 / CR.E.8).

---

## 1. Redis security: AUTH + TLS

`quorin` does **not** ship Redis authentication or TLS. The library
hands off the `redis.Redis` / `redis.asyncio.Redis` client your
application constructed; what's secured is up to you.

### Production checklist

- [ ] Redis is **not** reachable from outside the VPC / cluster
      network. Bind to `127.0.0.1` (single-host) or a private
      subnet (multi-host); never `0.0.0.0` on a public address.
- [ ] `requirepass` is set in `redis.conf` to a long random string
      (256 bits / 32 bytes / base64 ~43 chars). Rotate quarterly.
- [ ] Producer + consumer + watchdog all construct their Redis
      client with `password=` and `username=` (Redis 6+ ACL).
- [ ] `bind` directive is restrictive (`bind 127.0.0.1 ::1` for
      single-host).
- [ ] If Redis is on a different host than `quorin`, use TLS:
      `tls-port 6379`, `tls-cert-file`, `tls-key-file`,
      `tls-ca-cert-file`. Construct the client with
      `redis.Redis(..., ssl=True, ssl_certfile=..., ssl_keyfile=...,
      ssl_ca_certs=...)`.
- [ ] `appendonly yes` for durability of the WAL stream — see §3.

### `socket_timeout` is REQUIRED for production

```python
import redis

# WRONG — no socket_timeout. WALProducer / WALConsumer / heartbeat
# will all emit a UserWarning at construction; on partition the
# heartbeat thread leaks (stop()'s 2s join times out before the
# blocked HSET returns). v0.1.1 added the warning; v0.1.0 had no
# signal at all.
client = redis.Redis(host="127.0.0.1", port=6379)

# RIGHT — finite socket_timeout. 5 seconds is a reasonable default
# for a healthy LAN; tune based on your Redis-side latency p99.
client = redis.Redis(
    host="127.0.0.1",
    port=6379,
    socket_timeout=5.0,
    socket_connect_timeout=5.0,
)
```

### `appendfsync` durability

The WAL stream backs `write_sync`'s read-your-own-writes contract
AND is the only signal Step 15's upgrade orchestrator uses to detect
"consumer caught up." Recovery semantics depend on Redis durability:

- `appendfsync always` — every WAL write fsync'd. ~10x write
  amplification but zero data loss on power-cut. Recommend for
  workloads where `write_sync` is the durability story.
- `appendfsync everysec` (default) — up to 1s of WAL loss on
  power-cut. Acceptable when the offline Parquet store is the
  durability source of truth and the WAL is "queue, not log."
- `appendfsync no` — never. Don't.

---

## 2. The redis-py fork-unsafe-pool footgun

**This bites every gunicorn / uvicorn fork-worker deployment.** It is
not specific to `quorin`; it's a property of `redis-py`'s connection
pool. We document it here because `quorin` exposes the failure mode
as the symptom (heartbeat thread leak, consumer hang) without
pointing at the root cause.

### The problem

`redis.Redis` lazily constructs a `ConnectionPool` on first call.
When the pre-fork master process makes ANY Redis call before forking
its worker pool, the pool's open sockets get inherited by every fork
worker. Two workers writing to the same socket at once corrupts the
Redis protocol state — symptoms range from `ResponseError` to
`InvalidResponse` to silent stale data.

### The fix

Construct the `redis.Redis` client **inside each worker**, after
fork, **never** in the master process. The standard gunicorn pattern:

```python
# app.py
import redis
import quorin

_redis_client = None
_registry = None

def _post_fork_init() -> None:
    """Called by gunicorn's post_fork hook OR by lazy-init on first
    request. NEVER touch redis from module import time / master.
    """
    global _redis_client, _registry
    _redis_client = redis.Redis(
        host="...", port=6379, socket_timeout=5.0, password="...",
    )
    _registry = quorin.shm.SegmentRegistry(_redis_client)


def get_features(entity_id: str) -> dict:
    if _registry is None:
        _post_fork_init()
    seg = _registry.open_current(MySchema)
    out = quorin.assembly.assemble(seg, entity_id)
    return out
```

### gunicorn config

```python
# gunicorn.conf.py
def post_fork(server, worker):
    """Called in the worker process AFTER fork. Construct Redis here."""
    from app import _post_fork_init
    _post_fork_init()
```

### Symptoms of getting it wrong

- Consumer: `redis.exceptions.InvalidResponse` mid-XREADGROUP.
- Producer: `BUSYGROUP` errors that shouldn't occur.
- Heartbeat: silent thread leak (stop's join times out).
- Watchdog: false dead-PID detection because heartbeats look stale.

If you see ANY of the above clustered around fork events, audit
when your Redis client is constructed.

---

## 3. The watchdog: who watches the watchdog?

`quorin.watchdog` is a single-process daemon that detects dead
producer / consumer PIDs and reaps their segments. **It is itself a
single point of failure.** The recommended deployment shape:

- Run the watchdog as a **systemd unit** with `Restart=always` and
  `RestartSec=5s`. A crash-then-restart cycle takes ~5s; segments
  pile up until restart but are reaped on the next tick.
- Run the watchdog as a **dedicated user** with `CAP_SYS_PTRACE` (or
  as root) if your producers/consumers run under different UIDs.
  Without ptrace, `psutil.Process(pid).create_time()` raises
  `AccessDenied` for cross-UID PIDs and the watchdog conservatively
  skips them — those segments don't get reaped.
- Alert on `quorin_watchdog_cross_check_unverifiable_total{reason=
  "access_denied"} > 0` if you have multi-UID deployments.
- Alert on the absence of `quorin_watchdog_dead_pids_total`
  increments AND `quorin_watchdog_tick_seconds_count` increments
  for >5 minutes (means the watchdog is hung).

### What if the watchdog is down for a while?

`/dev/shm` segments accumulate. Production impact:

- New `registry.create` may hit the v0.1.1 50% capacity guard
  (CR.D.5) and refuse with `OSError(ENOSPC)`.
- Eventually `posix_shm.create` fails with the kernel-level ENOSPC
  if `/dev/shm` fills.
- Live segments keep working; reads/writes against
  already-attached segments are unaffected.

Recovery: start the watchdog. Its first tick scans the cleanup
queue + checks heartbeats; segments orphaned during the outage are
reaped within ~30s.

---

## 4. Disaster recovery: known scenarios

### Scenario: process crashed mid-upgrade

Symptom: `schema:current` points at the new segment but no consumer
is attached; the watchdog will eventually reap it, leaving
`schema:current` undefined.

Recovery path:
1. Stop any remaining consumers (`SIGTERM`, wait for liveness key
   to expire ~30s).
2. Wait the full watchdog interval (~150s) for orphan cleanup.
3. Re-run `upgrade_schema(...)` from a fresh orchestrator process.

If urgent: `redis-cli DEL quorin:schema:{name}:current`, then
`upgrade_schema(...)` will not find a current segment to upgrade
from. You'll need to re-hydrate from the offline store instead.

### Scenario: Redis appendfsync everysec lost the last second of WAL

Symptom: post-restart, `XPENDING` is lower than it should be; some
messages that producers thought were durable are gone.

Recovery: the offline Parquet store is the source of truth. Replay
from there via `hydrate(...)` to repopulate the online store. Some
client `write_sync` calls that returned success may have been lost;
the application layer is responsible for retry semantics if this
matters for your workload.

### Scenario: 2D-shape upgrade attempted

Symptom: v0.1.1 rejects with `UpgradeIncompatibleError("field 'X':
2D-shape upgrade not yet supported")` from `can_upgrade`.

Recovery: 2D upgrades are deferred to v0.2.0. Workaround: introduce
a new 1D field with the same data and deprecate the 2D one, OR
hydrate to a fresh segment from the offline store.

---

## 5. Common Prometheus alerts

| Alert | Threshold | Reason |
|---|---|---|
| `quorin_wal_consumer_pending_ack_size > 5000` | 5 min | Consumer is falling behind; offline flush is slow or stuck. |
| `quorin_wal_consumer_poison_pill_total > 0` | any | Schema-mismatched producer is writing; PEL grows. |
| `quorin_wal_consumer_schema_crc_mismatch_total > 0` | any | A consumer has stale schema code post-upgrade; restart with new code. |
| `quorin_evolution_consumer_pause_seconds > 30` | p99 | Long upgrade pauses suggest stuck orchestrator. |
| `quorin_watchdog_dead_pids_total` rate ≈ 0 AND `quorin_watchdog_tick_seconds_count` rate ≈ 0 | 5 min | Watchdog hung. |
| `quorin_watchdog_pid_reuse_abort_total > 0` | any | At least one race was caught — alert on rate, investigate if increasing. |
| `quorin_offline_flush_seconds{outcome="error"}` | any | Disk or Parquet failure; CR.A.2 ensures retry but persistent failures need intervention. |
| `quorin_heartbeat_writes_total{outcome="redis_error"}` rate > 0 | 5 min | Heartbeat thread is stuck on Redis; watchdog may declare host dead. |

---

## 6. Other operational items

- **Port-in-use**: `start_metrics_server(port=9100)` raises a raw
  `OSError` if the port is busy. v0.1.1 catches and re-raises with
  a clearer message; prefer running on a unique port per process.
- **`pid_max`**: Watchdog's PID-reuse defense (`expected_create_time_ns`
  CAS in the dead-PID Lua) has residual probability ~3e-9 per
  reuse. ADR-013 §"Critical decisions" #4 has the calculation if
  your kernel uses non-default `pid_max`.
- **multiprocess metrics**: `prometheus-client` doesn't natively
  aggregate counters across fork-workers. Single-process metrics
  endpoint per worker is the v0.1.1 deployment shape; v0.2.0 adds
  multiprocess collector documentation.

---

*Last updated alongside v0.1.1 (CR.C.10 / CR.C.11 / CR.E.4 / CR.E.6
/ CR.E.8 / CR.E.9). Operators: file issues against this doc when a
real outage path isn't covered.*
