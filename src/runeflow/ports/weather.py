# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2024-2026 Thomas Phil — runeflow
# See LICENSE and COMMERCIAL-LICENSE.md for licensing details.

"""WeatherPort — abstract interface for weather data adapters."""

from __future__ import annotations

import datetime
from abc import ABC, abstractmethod

from runeflow.domain.weather import WeatherLocation, WeatherSeries


class WeatherPort(ABC):
    """Download historical observations and forecasts."""

    @abstractmethod
    def download_historical(
        self,
        locations: list[WeatherLocation],
        start: datetime.date,
        end: datetime.date,
    ) -> WeatherSeries:
        """Download historical hourly weather for *locations* over *start..end*."""

    @abstractmethod
    def download_forecast(
        self,
        locations: list[WeatherLocation],
        horizon_days: int = 9,
    ) -> WeatherSeries:
        """Download a deterministic single-member forecast."""

    @abstractmethod
    def download_ensemble_forecast(
        self,
        locations: list[WeatherLocation],
        horizon_days: int = 9,
    ) -> list[WeatherSeries]:
        """
        Download a multi-member ensemble forecast.

        Returns one WeatherSeries per member (e.g. 51 ECMWF members).
        """
