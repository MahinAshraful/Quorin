"""pytest configuration for examples/ — registers the requires_redis marker.

Skip mechanism (CR.A.12-b / v0.1.1):

* Examples that need a live Redis instance use ``pytestmark =
  pytest.mark.requires_redis``.
* The basic CI job runs ``pytest examples/ -m 'not requires_redis'``
  to skip them on hosts without Redis.
* The integration CI job (which already starts a Redis service for
  ``tests/integration/``) runs ``pytest examples/ -m requires_redis``.
* Local devs with Redis up can run ``pytest examples/`` (no marker
  flag) to exercise everything.

The naive ``pytest.importorskip("redis")`` does NOT work here because
``redis`` is a runtime dep of ``quorin`` and is always importable; the
``requires_redis`` marker is the right primitive.
"""

from __future__ import annotations

import pytest


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "requires_redis: example requires a live Redis instance to run "
        "(skip via -m 'not requires_redis' in CI without Redis)",
    )
