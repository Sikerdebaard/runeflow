# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2024-2026 Thomas Phil — runeflow
# See LICENSE and COMMERCIAL-LICENSE.md for licensing details.

"""Simple in-process TTL cache for memoising expensive API calls within a run."""

from __future__ import annotations

import threading
import time
from typing import Any


class InProcessTTLCache:
    """Thread-safe dict-backed TTL cache.

    Entries expire after *ttl_seconds* (default 3600 s).  The cache is
    purely in-memory and lives only for the lifetime of the process, making
    it ideal for eliminating redundant API calls within a single pipeline run
    (e.g. the same ENTSO-E / EnergyZero / NED data fetched once per provider
    during ``make dev-build-site``).
    """

    def __init__(self, ttl_seconds: float = 3600.0) -> None:
        self._ttl = ttl_seconds
        self._store: dict[Any, tuple[Any, float]] = {}
        self._lock = threading.Lock()

    def get(self, key: Any) -> tuple[bool, Any]:
        """Return *(hit, value)*.  Returns ``(False, None)`` when absent or stale."""
        with self._lock:
            entry = self._store.get(key)
            if entry is None:
                return False, None
            value, expiry = entry
            if time.monotonic() > expiry:
                del self._store[key]
                return False, None
            return True, value

    def set(self, key: Any, value: Any) -> None:
        """Store *value* under *key*; expires after the configured TTL."""
        expiry = time.monotonic() + self._ttl
        with self._lock:
            self._store[key] = (value, expiry)

    def clear(self) -> None:
        """Evict all entries."""
        with self._lock:
            self._store.clear()
