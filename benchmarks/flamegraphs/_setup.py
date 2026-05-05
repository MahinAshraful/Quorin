"""Shared setup helpers for benchmarks/flamegraphs/* drivers.

py-spy records a Python process's CPU profile and emits an SVG flamegraph.
Each driver script in this directory exercises a production hot-path in a
tight loop so py-spy can sample inside it; the SVGs land in
``benchmarks/results/flamegraphs/`` for the README to embed.

This module mirrors what pytest fixtures (``benchmarks/conftest.py``,
``tests/_helpers.py``) provide, but standalone — drivers run as plain
``python script.py`` so py-spy's process-attachment doesn't have to navigate
pytest's runner indirection.

Functions:
    * :func:`make_warm_segment` — populated Segment for assemble drivers.
    * :func:`make_clobber_array` — L3-sized cache clobber for cold-cache.
    * :func:`setup_redis_consumer_50_field` — running WAL consumer (write_sync
      driver).
    * :func:`populate_dataset_for_hydration` — Parquet dataset (hydration drivers).

The module is **NOT** a flamegraph driver itself — the workflow YAML's
glob excludes ``_setup.py`` (leading underscore convention).
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import struct
import sys
import tempfile
import threading
import time
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

if sys.platform != "linux":
    raise SystemExit("flamegraph drivers require Linux POSIX shm + /sys")

from benchmarks._cache_clobber import detect_l3_size_bytes
from pyforge._internal import posix_shm
from pyforge._internal.crc import crc32_of_bytes
from pyforge.layout import (
    DEFAULT_MAX_ID_BYTES,
    compute_layout,
    initialize_segment_regions,
    insert,
)
from pyforge.schema import (
    DTYPE_TO_NUMPY,
    FeatureField,
    FeatureSchema,
    compile_schema,
    compute_assembly_table,
    dtype,
    row_size,
)
from pyforge.shm import HEADER_FMT, HEADER_LEN, MAGIC, Segment

# ---------------------------------------------------------------------------
# Schemas — mirror benchmarks/test_assembly_benchmark.py.
# ---------------------------------------------------------------------------


class Schema4Field(FeatureSchema):
    """4-field warm-cache scenario. Spec target: ≤5 us p99."""

    version = 1
    fields = [
        FeatureField("age", dtype.int32),
        FeatureField("clicks", dtype.int64),
        FeatureField("ltv", dtype.float64),
        FeatureField("score", dtype.float32),
    ]


def _build_n_field_schema(n: int, with_embedding: bool = False) -> type[FeatureSchema]:
    fields: list[FeatureField] = [
        FeatureField(f"feat_{i:03d}", dtype.float32) for i in range(n - 1 if with_embedding else n)
    ]
    if with_embedding:
        fields.append(FeatureField("embedding", dtype.float32, shape=(128,)))
    cls_name = f"_FlameSchemaN{n}{'_emb' if with_embedding else ''}"
    return type(cls_name, (FeatureSchema,), {"version": 1, "fields": fields})


Schema200Field = _build_n_field_schema(200, with_embedding=True)


# ---------------------------------------------------------------------------
# Segment helpers — mirror tests/_helpers.py::make_segment but standalone.
# ---------------------------------------------------------------------------


def _unique_segment_name(prefix: str = "pyforge_flame") -> str:
    return f"{prefix}_{os.getpid()}_{uuid.uuid4().hex[:8]}"


def _make_segment(
    schema: type[FeatureSchema],
    capacity: int,
    *,
    max_id_bytes: int = DEFAULT_MAX_ID_BYTES,
) -> Segment:
    layout = compute_layout(schema, capacity=capacity, max_id_bytes=max_id_bytes)
    name = _unique_segment_name()
    handle = posix_shm.create(name, layout.total_size)
    crc = crc32_of_bytes(compile_schema(schema).tobytes())
    handle.buf[:HEADER_LEN] = struct.pack(HEADER_FMT, MAGIC, int(schema.version), crc, capacity)
    initialize_segment_regions(handle.buf, layout)
    return Segment(name=name, schema=schema, handle=handle, layout=layout)


def _release_segment(seg: Segment) -> None:
    posix_shm.close(seg.handle)
    posix_shm.unlink(seg.name)


def _pack_row(
    schema: type[FeatureSchema], values: dict[str, np.ndarray[Any, np.dtype[Any]]]
) -> bytes:
    rs = row_size(schema)
    out = bytearray(rs)
    table = compute_assembly_table(schema)
    for i, f in enumerate(schema.fields):
        row = table[i]
        byte_off = int(row["byte_offset"])
        byte_cnt = int(row["byte_count"])
        arr = values[f.name]
        flat = np.ascontiguousarray(arr).reshape(-1)
        out[byte_off : byte_off + byte_cnt] = flat.tobytes()
    return bytes(out)


def make_warm_segment(
    schema: type[FeatureSchema], *, capacity: int = 64
) -> tuple[Segment, Callable[[], None]]:
    """Allocate a Segment + insert one entity ``"u"`` with zero-filled values.

    Returns ``(segment, cleanup_fn)``. Caller invokes cleanup_fn after the
    flamegraph capture completes.
    """
    seg = _make_segment(schema, capacity=capacity)
    if schema is Schema4Field:
        values = {
            "age": np.array([42], dtype=np.int32),
            "clicks": np.array([1_234_567], dtype=np.int64),
            "ltv": np.array([987.65], dtype=np.float64),
            "score": np.array([0.5], dtype=np.float32),
        }
    else:
        values = {
            f.name: np.zeros(f.element_count, dtype=DTYPE_TO_NUMPY[f.dtype]) for f in schema.fields
        }
    insert(seg, "u", _pack_row(schema, values))

    def _cleanup() -> None:
        _release_segment(seg)

    return seg, _cleanup


# ---------------------------------------------------------------------------
# Cold-cache clobber.
# ---------------------------------------------------------------------------


def make_clobber_array() -> np.ndarray[Any, np.dtype[np.float64]]:
    """L3-sized clobber array (4x detected L3, capped at 1 GB).

    Mirrors ``benchmarks/conftest.py::cold_cache_clobber`` but standalone.
    Drivers traverse the array between assemble calls to evict the segment
    from CPU cache.
    """
    size_bytes = min(4 * detect_l3_size_bytes(), 1 * 1024 * 1024 * 1024)
    n_doubles = size_bytes // 8
    return np.empty(n_doubles, dtype=np.float64)


# ---------------------------------------------------------------------------
# Hydration: small pre-built Parquet dataset.
# ---------------------------------------------------------------------------


def populate_dataset_for_hydration(*, n_entities: int, schema: type[FeatureSchema]) -> Path:
    """Write a Parquet dataset suitable for hydration to read.

    Uses :class:`pyforge.offline.ParquetDatasetStore.append` for the write
    path (matches the production-shaped offline store layout). Returns the
    path the driver passes to :func:`pyforge.hydration.hydrate`.
    """
    # Deferred import: pyforge.offline pulls pyarrow (~50 ms) which we don't
    # want at module-import time of this helper.
    from pyforge.offline import ParquetDatasetStore

    tmp_dir = tempfile.mkdtemp(prefix="pyforge_flame_hydrate_")
    dataset_path = Path(tmp_dir)
    store = ParquetDatasetStore(
        dataset_path=dataset_path,
        schema=schema,
        flush_interval_seconds=3600,  # never auto-flush; we flush at end
    )

    async def _populate() -> None:
        now_ns = int(time.time() * 1e9)
        for i in range(n_entities):
            row_values = {
                f.name: np.zeros(f.element_count, dtype=DTYPE_TO_NUMPY[f.dtype])
                for f in schema.fields
            }
            await store.append(
                entity_id=f"flame_e_{i:08d}",
                event_time_ns=now_ns + i,
                row_values=row_values,
            )
        await store.flush()
        await store.close()

    asyncio.run(_populate())
    return dataset_path


# ---------------------------------------------------------------------------
# WAL consumer for write_sync flamegraph.
# ---------------------------------------------------------------------------


def setup_redis_consumer_50_field() -> dict[str, Any]:
    """Mirror benchmarks/conftest.py::running_consumer_50_field but standalone.

    Returns a dict with ``schema``, ``redis_client``, ``segment``, ``registry``,
    ``stream_key``, plus a ``cleanup`` callable. The driver uses ``schema`` +
    ``redis_client`` + ``stream_key`` to construct a ``WALProducer``.
    """
    # Deferred imports: heavy + Redis-dependent.
    import redis
    import redis.asyncio as redis_async

    from pyforge.shm import SegmentRegistry
    from pyforge.wal_consumer import NoopOfflineWriter, WALConsumer

    redis_url = os.environ.get("PYFORGE_REDIS_URL", "redis://127.0.0.1:6379/0")
    redis_client = redis.Redis.from_url(redis_url, decode_responses=False)
    redis_client.ping()  # fail fast if Redis is unreachable

    fields = (
        [FeatureField(f"f{i:02d}", dtype.float32) for i in range(40)]
        + [FeatureField(f"i{i:02d}", dtype.int64) for i in range(8)]
        + [FeatureField("emb", dtype.float32, shape=(16,))]
        + [FeatureField("flags", dtype.uint8, shape=(4,))]
    )
    schema = type("_FlameS50", (FeatureSchema,), {"version": 1, "fields": fields})

    bench_stream = b"pyforge:wal:flame:rtt:50f"
    bench_group = "pyforge_flame_rtt_50f_consumers"
    redis_client.delete(bench_stream)
    with contextlib.suppress(Exception):
        redis_client.execute_command("XGROUP", "DESTROY", bench_stream, bench_group)

    registry = SegmentRegistry(redis_client)
    segment = registry.create(schema, capacity=128)

    loop = asyncio.new_event_loop()
    async_client = redis_async.Redis.from_url(redis_url, decode_responses=False)

    consumer = WALConsumer(
        async_client,
        segments={schema.__name__: segment},
        offline=NoopOfflineWriter(),
        registry=registry,
        stream_key=bench_stream,
        group_name=bench_group,
        flush_interval_seconds=0.05,
        block_ms=10,
    )
    consumer._schemas = {schema.__name__.encode(): schema}  # type: ignore[attr-defined]

    runner_ready = threading.Event()

    def _run_consumer() -> None:
        asyncio.set_event_loop(loop)
        try:
            runner_ready.set()
            loop.run_until_complete(consumer.run())
        except BaseException:
            pass
        finally:
            loop.run_until_complete(async_client.aclose())
            loop.close()

    thread = threading.Thread(target=_run_consumer, daemon=True, name="flame-consumer")
    thread.start()
    runner_ready.wait(timeout=2.0)
    time.sleep(0.5)  # consumer registers group + writes liveness key

    def _cleanup() -> None:
        with contextlib.suppress(RuntimeError):
            loop.call_soon_threadsafe(consumer._stop_event.set)
        thread.join(timeout=10)
        with contextlib.suppress(Exception):
            registry.close(segment)
        with contextlib.suppress(Exception):
            redis_client.close()

    return {
        "schema": schema,
        "redis_client": redis_client,
        "segment": segment,
        "registry": registry,
        "stream_key": bench_stream,
        "cleanup": _cleanup,
    }
