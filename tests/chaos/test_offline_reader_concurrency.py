"""Chaos tests for ParquetDatasetStore.read_point_in_time concurrency (Step 12).

C1 — concurrent ``flush()`` and ``read_point_in_time``: read completes,
     result reflects pre-flush OR post-flush state (no error, no
     partial state).
C2 — ``_tmp/`` invisibility: a uuid file present in ``base/_tmp/``
     during a read is NOT visible to the dataset (only files under
     ``schema=*/event_date=*/`` are).
C3 — dataset directory deleted mid-read: reader either completes with
     what it loaded OR raises ``FileNotFoundError``; no segfault, no
     infinite loop.

The reader's snapshot semantics (ADR-011 §F) make C1/C3 well-defined:
each call constructs a fresh ``pa.dataset.dataset()`` at function
entry, so concurrent writer activity may or may not be visible.
"""

from __future__ import annotations

import asyncio
import shutil
import sys
import threading
import uuid
from pathlib import Path

import pytest

pytestmark = [
    pytest.mark.skipif(
        sys.platform == "win32",
        reason="Step 11/12 fsync semantics are POSIX-only",
    ),
    pytest.mark.chaos,
]

import pyarrow as pa  # noqa: E402

from pyforge._internal.arrow_schema import (  # noqa: E402
    clear_cache as clear_arrow_cache,
)
from pyforge.offline import ParquetDatasetStore  # noqa: E402
from pyforge.schema import FeatureField, FeatureSchema, dtype  # noqa: E402


class _S(FeatureSchema):
    version = 1
    fields = [
        FeatureField("x", dtype.float32),
        FeatureField("y", dtype.int64),
    ]


@pytest.fixture(autouse=True)
def _clear_arrow_plan_cache() -> None:
    clear_arrow_cache()


def _q(eid: str, aot: int) -> pa.Table:
    return pa.table(
        {
            "entity_id": pa.array([eid], type=pa.string()),
            "as_of_time": pa.array([aot], type=pa.int64()),
        }
    )


# ---------------------------------------------------------------------------
# C1. Concurrent flush + read_point_in_time.
# ---------------------------------------------------------------------------


def test_concurrent_flush_and_read_completes_without_error(
    tmp_path: Path,
) -> None:
    """C1: a flush running concurrently with reads doesn't crash either side.

    The reader is sync; the flush is async. Run flush in a background
    thread driving its own asyncio event loop while the main thread
    issues reads. The flush eventually completes; reads either see
    pre-flush or post-flush state.
    """
    store = ParquetDatasetStore(tmp_path)

    # Pre-populate so the dataset exists and the reader has work.
    async def _seed() -> None:
        for i in range(200):
            await store.append(_S, f"e{i}", 100 + i, [float(i), i], f"{i}-0".encode())
        await store.flush()

    asyncio.run(_seed())

    # Stage another batch in the buffer to flush concurrently.
    async def _stage_buffer() -> None:
        for i in range(200, 400):
            await store.append(_S, f"e{i}", 100 + i, [float(i), i], f"{i}-0".encode())

    asyncio.run(_stage_buffer())

    # Run the flush in a background thread; meanwhile, hammer the reader.
    flush_done = threading.Event()
    flush_error: list[BaseException] = []

    def _do_flush() -> None:
        try:
            asyncio.run(store.flush())
        except BaseException as e:
            flush_error.append(e)
        finally:
            flush_done.set()

    flusher = threading.Thread(target=_do_flush)
    flusher.start()

    read_results: list[int] = []
    for i in range(20):
        # Each read constructs a fresh dataset (snapshot semantics).
        result = store.read_point_in_time(_S, _q(f"e{i % 200}", 200), lookback_days=30)
        read_results.append(len(result))

    flusher.join(timeout=10.0)
    assert flush_done.is_set(), "flush thread did not finish in 10 s"
    assert not flush_error, f"flush raised: {flush_error}"
    # Every read returned the right number of rows (1 per query row).
    assert all(n == 1 for n in read_results)


# ---------------------------------------------------------------------------
# C2. _tmp/ directory invisibility.
# ---------------------------------------------------------------------------


def test_tmp_dir_files_not_visible_to_reader(tmp_path: Path) -> None:
    """C2: an in-flight uuid file in ``base/_tmp/`` does not affect reads."""
    store = ParquetDatasetStore(tmp_path)

    async def _populate() -> None:
        await store.append(_S, "A", 100, [1.0, 100], b"1-0")
        await store.flush()

    asyncio.run(_populate())

    # Plant a file in _tmp/ that, if the reader scanned it, would
    # confuse the dataset (different schema). Use a marker file that's
    # not even valid Parquet — the reader should never see it.
    tmp_marker = tmp_path / "_tmp" / f"{uuid.uuid4().hex}.parquet"
    tmp_marker.write_bytes(b"NOT A VALID PARQUET FILE - SHOULD NEVER BE READ")

    # Reader operates on schema={S.__name__}/, not _tmp/. The marker
    # file is invisible.
    result = store.read_point_in_time(_S, _q("A", 200))
    d = result.to_pydict()
    assert d["x"] == [pytest.approx(1.0)]
    assert d["event_time_ns"] == [100]


# ---------------------------------------------------------------------------
# C3. Dataset deleted mid-read.
# ---------------------------------------------------------------------------


def test_dataset_deleted_after_open_completes_or_raises_cleanly(
    tmp_path: Path,
) -> None:
    """C3: dataset dir deleted between _open_dataset and to_table.

    The pa.dataset object holds file paths captured at construction; if
    the underlying files vanish, ``to_table`` may raise
    ``FileNotFoundError`` (or a wrapped ArrowInvalid). Either is
    acceptable — what matters is no segfault, no infinite loop.

    We trigger the race deterministically by patching ``_open_dataset``
    to delete the partition tree before returning the dataset.
    """
    from unittest.mock import patch

    store = ParquetDatasetStore(tmp_path)

    async def _populate() -> None:
        await store.append(_S, "A", 100, [1.0, 100], b"1-0")
        await store.flush()

    asyncio.run(_populate())

    schema_dir = tmp_path / f"schema={_S.__name__}"
    assert schema_dir.exists()

    real_open = ParquetDatasetStore._open_dataset

    def open_then_delete(self_, sch):  # type: ignore[no-untyped-def]
        ds = real_open(self_, sch)
        # Delete the data files between dataset construction and to_table.
        shutil.rmtree(schema_dir)
        return ds

    with patch.object(ParquetDatasetStore, "_open_dataset", open_then_delete):
        # Either the read completes (PyArrow read enough metadata up
        # front) or it raises an OSError-flavored exception. Both are
        # acceptable — no segfault, no infinite loop.
        try:
            result = store.read_point_in_time(_S, _q("A", 200))
            # Completed cleanly; result has the expected shape.
            assert len(result) == 1
        except (FileNotFoundError, OSError, pa.lib.ArrowInvalid):
            pass  # acceptable per snapshot contract
