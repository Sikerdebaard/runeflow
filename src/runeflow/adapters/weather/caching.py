# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2024-2026 Thomas Phil — runeflow
# See LICENSE and COMMERCIAL-LICENSE.md for licensing details.

"""CachingWeatherAdapter — transparent caching decorator for WeatherPort.

Wraps any :class:`~runeflow.ports.weather.WeatherPort` and intercepts
``download_forecast`` and ``download_ensemble_forecast`` calls with a
pluggable :class:`~runeflow.adapters.weather.strategies.CachingStrategy`.

The adapter is completely transparent to callers: it implements the same
``WeatherPort`` interface as the wrapped adapter, so services like
``InferenceService`` need no caching awareness at all.

Example — wire up a 3-hour TTL cache in the binder::

    from runeflow.adapters.weather.caching import CachingWeatherAdapter
    from runeflow.adapters.weather.strategies import TTLCachingStrategy
    import datetime

    inner = OpenMeteoAdapter(...)
    strategy = TTLCachingStrategy(ttl=datetime.timedelta(hours=3))
    weather = CachingWeatherAdapter(inner, store, zone, strategy=strategy)
    binder.bind(WeatherPort, weather)
"""

from __future__ import annotations

import datetime
import logging

from runeflow.adapters.weather.openmeteo import (
    DEFAULT_HOURLY_VARS,
    ENSEMBLE_VARS,
)
from runeflow.adapters.weather.strategies import (
    CachingStrategy,
    TTLCachingStrategy,
)
from runeflow.domain.weather import WeatherLocation, WeatherSeries
from runeflow.ports.store import DataStore
from runeflow.ports.weather import WeatherPort

_DEFAULT_N_ENSEMBLE_MEMBERS = 51

logger = logging.getLogger(__name__)


class CachingWeatherAdapter(WeatherPort):
    """Decorator that adds cache read/write around any ``WeatherPort``.

    Parameters
    ----------
    inner:
        The real weather adapter to delegate downloads to.
    store:
        Data-store used for reading and writing cached weather files.
    zone:
        ENTSO-E zone code (e.g. ``"NL"``).  Used as the cache key.
    strategy:
        Caching strategy that controls freshness checks and persistence.
        Defaults to :class:`TTLCachingStrategy` with a 3-hour TTL.
    det_vars:
        Expected column names for deterministic forecasts.  An existing
        cache file is considered stale if it is missing any of these columns.
        Defaults to :data:`~runeflow.adapters.weather.openmeteo.DEFAULT_HOURLY_VARS`.
    ens_vars:
        Expected column names for ensemble members.  Defaults to
        :data:`~runeflow.adapters.weather.openmeteo.ENSEMBLE_VARS`.
    n_ensemble_members:
        Maximum number of ensemble members to cache / retrieve.
    """

    def __init__(
        self,
        inner: WeatherPort,
        store: DataStore,
        zone: str,
        *,
        strategy: CachingStrategy | None = None,
        det_vars: list[str] | None = None,
        ens_vars: list[str] | None = None,
        n_ensemble_members: int = _DEFAULT_N_ENSEMBLE_MEMBERS,
    ) -> None:
        self._inner = inner
        self._store = store
        self._zone = zone
        self._strategy: CachingStrategy = strategy or TTLCachingStrategy()
        self._det_cols = sorted(det_vars or DEFAULT_HOURLY_VARS)
        self._ens_cols = sorted(ens_vars or ENSEMBLE_VARS)
        self._n_members = n_ensemble_members

    # ── WeatherPort ───────────────────────────────────────────────────────────

    def download_historical(
        self,
        locations: list[WeatherLocation],
        start: datetime.date,
        end: datetime.date,
    ) -> WeatherSeries:
        """Historical data is not cached — delegate directly to inner."""
        return self._inner.download_historical(locations, start, end)

    def download_forecast(
        self,
        locations: list[WeatherLocation],
        horizon_days: int = 9,
    ) -> WeatherSeries:
        """Return a cached deterministic forecast, or download and cache one."""
        if self._strategy.is_fresh(self._store, self._zone, None, self._det_cols):
            cached = self._store.load_forecast_weather(self._zone)
            if cached is not None:
                logger.info(
                    "Weather cache hit: deterministic forecast for zone=%s (strategy=%s)",
                    self._zone,
                    type(self._strategy).__name__,
                )
                return cached

        logger.info("Downloading deterministic forecast weather for zone=%s...", self._zone)
        data = self._inner.download_forecast(locations, horizon_days=horizon_days)
        self._strategy.on_downloaded(self._store, data, self._zone, None)
        return data

    def download_ensemble_forecast(
        self,
        locations: list[WeatherLocation],
        horizon_days: int = 9,
    ) -> list[WeatherSeries]:
        """Return cached ensemble members, or download and cache all of them.

        Member 0 is used as a proxy for the freshness check; if it is fresh,
        all members are loaded from the store.  If any member is missing,
        the full ensemble is re-downloaded.
        """
        # Use member 0 as an availability/freshness proxy for the whole batch.
        if self._strategy.is_fresh(self._store, self._zone, 0, self._ens_cols):
            members = self._load_all_cached_members()
            if members is not None:
                logger.info(
                    "Weather cache hit: %d ensemble members for zone=%s (strategy=%s)",
                    len(members),
                    self._zone,
                    type(self._strategy).__name__,
                )
                return members

        logger.info(
            "Downloading ensemble forecast weather (%d members) for zone=%s...",
            self._n_members,
            self._zone,
        )
        downloaded = self._inner.download_ensemble_forecast(locations, horizon_days=horizon_days)
        batch = downloaded[: self._n_members]
        for i, member in enumerate(batch):
            self._strategy.on_downloaded(self._store, member, self._zone, i)
        if batch:
            logger.info(
                "Cached %d/%d ensemble members for zone=%s",
                len(batch),
                len(downloaded),
                self._zone,
            )
        return batch

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _load_all_cached_members(self) -> list[WeatherSeries] | None:
        """Load all *n_members* from the store; return *None* if any is missing."""
        members: list[WeatherSeries] = []
        for i in range(self._n_members):
            m = self._store.load_forecast_weather_ensemble(self._zone, i)
            if m is None:
                return None
            members.append(m)
        return members
