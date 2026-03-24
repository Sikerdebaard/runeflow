# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2024-2026 Thomas Phil — runeflow
# See LICENSE and COMMERCIAL-LICENSE.md for licensing details.

"""Caching strategies for the CachingWeatherAdapter.

A strategy encapsulates two decisions:
  1. ``is_fresh``     — should a cached copy be returned instead of downloading?
  2. ``on_downloaded`` — what to persist (if anything) after a fresh download?

Built-in strategies
-------------------
TTLCachingStrategy
    Re-use the cached copy while it is younger than *ttl* **and** its stored
    column schema is a superset of *expected_cols*.  Persists every download.
NoCachingStrategy
    Always download; never read from or write to the store.
ReadOnlyCachingStrategy
    Same freshness check as ``TTLCachingStrategy`` but never writes back —
    useful when the cache volume is read-only or managed externally.

Custom strategies implement the :class:`CachingStrategy` ABC.
"""
from __future__ import annotations

import datetime
from abc import ABC, abstractmethod

from runeflow.domain.weather import WeatherSeries
from runeflow.ports.store import DataStore


class CachingStrategy(ABC):
    """Interface that governs read/write decisions for cached weather data."""

    @abstractmethod
    def is_fresh(
        self,
        store: DataStore,
        zone: str,
        member: int | None,
        expected_cols: list[str],
    ) -> bool:
        """Return *True* if the cached copy may be used without downloading."""

    @abstractmethod
    def on_downloaded(
        self,
        store: DataStore,
        data: WeatherSeries,
        zone: str,
        member: int | None,
    ) -> None:
        """Called after a successful download; persist *data* if desired."""


# ---------------------------------------------------------------------------

class TTLCachingStrategy(CachingStrategy):
    """Re-use cache within *ttl*; persist every download.

    Freshness is determined by the store's own ``is_forecast_weather_fresh``
    method, which checks both the age of the cached file **and** whether its
    stored column schema is a superset of *expected_cols*.  If a new feature
    column is added, stale data will be automatically re-fetched.
    """

    def __init__(self, ttl: datetime.timedelta = datetime.timedelta(hours=3)) -> None:
        self._ttl = ttl

    def is_fresh(
        self,
        store: DataStore,
        zone: str,
        member: int | None,
        expected_cols: list[str],
    ) -> bool:
        return store.is_forecast_weather_fresh(zone, self._ttl, expected_cols, member)

    def on_downloaded(
        self,
        store: DataStore,
        data: WeatherSeries,
        zone: str,
        member: int | None,
    ) -> None:
        store.save_forecast_weather(data, zone, member=member)


# ---------------------------------------------------------------------------

class NoCachingStrategy(CachingStrategy):
    """Bypass cache entirely — always download, never persist."""

    def is_fresh(
        self,
        store: DataStore,
        zone: str,
        member: int | None,
        expected_cols: list[str],
    ) -> bool:
        return False

    def on_downloaded(
        self,
        store: DataStore,
        data: WeatherSeries,
        zone: str,
        member: int | None,
    ) -> None:
        pass  # intentionally a no-op


# ---------------------------------------------------------------------------

class ReadOnlyCachingStrategy(CachingStrategy):
    """Return cached copy when fresh, but never write new downloads back.

    Useful when the cache volume is managed externally (e.g. mounted
    read-only, or populated by a separate pre-cache job).
    """

    def __init__(self, ttl: datetime.timedelta = datetime.timedelta(hours=3)) -> None:
        self._ttl = ttl

    def is_fresh(
        self,
        store: DataStore,
        zone: str,
        member: int | None,
        expected_cols: list[str],
    ) -> bool:
        return store.is_forecast_weather_fresh(zone, self._ttl, expected_cols, member)

    def on_downloaded(
        self,
        store: DataStore,
        data: WeatherSeries,
        zone: str,
        member: int | None,
    ) -> None:
        pass  # read-only — do not persist