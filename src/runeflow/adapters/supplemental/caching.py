# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2024-2026 Thomas Phil — runeflow
# See LICENSE and COMMERCIAL-LICENSE.md for licensing details.

"""CachingSupplementalAdapter — in-process TTL cache decorator for SupplementalDataPort.

Wraps any :class:`~runeflow.ports.supplemental.SupplementalDataPort` and
memoises ``download`` and ``download_forecast`` calls for the lifetime of the
process (default TTL: 1 hour).  This prevents the NED API from being called
once per provider during a ``make dev-build-site`` run.

Example::

    from runeflow.adapters.supplemental.caching import CachingSupplementalAdapter

    inner = NedAdapter(api_key=ned_key)
    cached = CachingSupplementalAdapter(inner)
    binder.bind(SupplementalDataPort, cached)
"""

from __future__ import annotations

import datetime
from typing import cast

import pandas as pd
from loguru import logger

from runeflow.adapters._ttl_cache import InProcessTTLCache
from runeflow.ports.supplemental import SupplementalDataPort

_DEFAULT_TTL = 3600.0  # seconds


class CachingSupplementalAdapter(SupplementalDataPort):
    """Transparent in-memory caching decorator for any ``SupplementalDataPort``."""

    def __init__(self, inner: SupplementalDataPort, ttl_seconds: float = _DEFAULT_TTL) -> None:
        self._inner = inner
        self._cache: InProcessTTLCache = InProcessTTLCache(ttl_seconds=ttl_seconds)

    def download(
        self,
        zone: str,
        start: datetime.date,
        end: datetime.date,
    ) -> pd.DataFrame | None:
        key = ("download", zone, str(start), str(end))
        hit, cached = self._cache.get(key)
        if hit:
            logger.debug(f"[CachingSupplemental] Cache hit: download {zone} {start}→{end}")
            return cast(pd.DataFrame | None, cached)

        result = self._inner.download(zone, start, end)
        self._cache.set(key, result)
        return result

    def supports_zone(self, zone: str) -> bool:
        return self._inner.supports_zone(zone)

    def download_forecast(self, zone: str) -> pd.DataFrame | None:
        key = ("forecast", zone)
        hit, cached = self._cache.get(key)
        if hit:
            logger.debug(f"[CachingSupplemental] Cache hit: forecast {zone}")
            return cast(pd.DataFrame | None, cached)

        result = self._inner.download_forecast(zone)
        self._cache.set(key, result)
        return result
