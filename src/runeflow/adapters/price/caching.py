# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2024-2026 Thomas Phil — runeflow
# See LICENSE and COMMERCIAL-LICENSE.md for licensing details.

"""CachingPriceAdapter — in-process TTL cache decorator for PricePort.

Wraps any :class:`~runeflow.ports.price.PricePort` and memoises
``download_historical`` and ``download_day_ahead`` calls for the lifetime of
the process (default TTL: 1 hour).  This prevents the same date-range from
being fetched multiple times during a single ``make dev-build-site`` run when
several providers share the same price data.

Example::

    from runeflow.adapters.price.caching import CachingPriceAdapter

    inner = FallbackPriceAdapter([entsoe, energyzero])
    cached = CachingPriceAdapter(inner)
    binder.bind(PricePort, cached)
"""

from __future__ import annotations

import datetime
from typing import cast

from loguru import logger

from runeflow.adapters._ttl_cache import InProcessTTLCache
from runeflow.domain.price import PriceSeries
from runeflow.ports.price import PricePort

_DEFAULT_TTL = 3600.0  # seconds


class CachingPriceAdapter(PricePort):
    """Transparent in-memory caching decorator for any ``PricePort``."""

    def __init__(self, inner: PricePort, ttl_seconds: float = _DEFAULT_TTL) -> None:
        self._inner = inner
        self._cache: InProcessTTLCache = InProcessTTLCache(ttl_seconds=ttl_seconds)

    @property
    def name(self) -> str:
        return f"Caching({self._inner.name})"

    def supports_zone(self, zone: str) -> bool:
        return self._inner.supports_zone(zone)

    def download_historical(
        self,
        zone: str,
        start: datetime.date,
        end: datetime.date,
    ) -> PriceSeries:
        key = ("historical", zone, str(start), str(end))
        hit, cached = self._cache.get(key)
        if hit:
            logger.debug(f"[CachingPrice] Cache hit: historical {zone} {start}→{end}")
            return cast(PriceSeries, cached)

        result = self._inner.download_historical(zone, start, end)
        self._cache.set(key, result)
        return result

    def download_day_ahead(self, zone: str) -> PriceSeries | None:
        key = ("day_ahead", zone)
        hit, cached = self._cache.get(key)
        if hit:
            logger.debug(f"[CachingPrice] Cache hit: day-ahead {zone}")
            return cast(PriceSeries | None, cached)

        result = self._inner.download_day_ahead(zone)
        self._cache.set(key, result)
        return result
