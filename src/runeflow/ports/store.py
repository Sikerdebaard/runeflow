# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2024-2026 Thomas Phil — runeflow
# See LICENSE and COMMERCIAL-LICENSE.md for licensing details.

"""DataStore — abstract interface for persistent storage."""

from __future__ import annotations

import datetime
from abc import ABC, abstractmethod

import pandas as pd

from runeflow.domain.forecast import ForecastResult
from runeflow.domain.generation import GenerationSeries
from runeflow.domain.price import PriceSeries
from runeflow.domain.weather import WeatherSeries


class DataStore(ABC):
    """Persist and retrieve time series and model artifacts."""

    # ── Price ─────────────────────────────────────────────────────────────────

    @abstractmethod
    def save_prices(self, data: PriceSeries) -> None: ...

    @abstractmethod
    def load_prices(
        self,
        zone: str,
        start: datetime.date | None = None,
        end: datetime.date | None = None,
    ) -> PriceSeries | None: ...

    # ── Weather ───────────────────────────────────────────────────────────────

    @abstractmethod
    def save_weather(self, data: WeatherSeries, zone: str) -> None: ...

    @abstractmethod
    def save_forecast_weather(
        self, data: WeatherSeries, zone: str, member: int | None = None
    ) -> None: ...

    @abstractmethod
    def load_weather(
        self,
        zone: str,
        start: datetime.date | None = None,
        end: datetime.date | None = None,
    ) -> WeatherSeries | None: ...

    @abstractmethod
    def load_forecast_weather(self, zone: str) -> WeatherSeries | None: ...

    @abstractmethod
    def load_forecast_weather_ensemble(self, zone: str, member: int) -> WeatherSeries | None: ...

    @abstractmethod
    def is_forecast_weather_fresh(
        self,
        zone: str,
        ttl: datetime.timedelta,
        expected_cols: list[str],
        member: int | None = None,
    ) -> bool: ...

    @abstractmethod
    def is_historical_weather_fresh(
        self,
        zone: str,
        expected_cols: list[str],
    ) -> bool:
        """Return *True* if cached historical weather exists with a schema
        that is a superset of *expected_cols* (no TTL — eternal cache)."""

    # ── Generation ────────────────────────────────────────────────────────────

    @abstractmethod
    def save_generation(self, data: GenerationSeries) -> None: ...

    @abstractmethod
    def load_generation(
        self,
        zone: str,
        start: datetime.date | None = None,
        end: datetime.date | None = None,
    ) -> GenerationSeries | None: ...

    # ── Supplemental ──────────────────────────────────────────────────────────

    @abstractmethod
    def save_supplemental(self, df: pd.DataFrame, zone: str, key: str) -> None: ...

    @abstractmethod
    def load_supplemental(self, zone: str, key: str) -> pd.DataFrame | None: ...

    # ── Model Artifacts ───────────────────────────────────────────────────────

    @abstractmethod
    def save_model(self, model_bytes: bytes, zone: str, model_name: str) -> None: ...

    @abstractmethod
    def load_model(self, zone: str, model_name: str) -> bytes | None: ...

    # ── Forecast Results ──────────────────────────────────────────────────────

    @abstractmethod
    def save_forecast(self, result: ForecastResult) -> None: ...

    @abstractmethod
    def load_latest_forecast(self, zone: str) -> ForecastResult | None: ...

    # ── Warmup Cache ──────────────────────────────────────────────────────────

    @abstractmethod
    def save_warmup_cache(self, df: pd.DataFrame, zone: str) -> None: ...

    @abstractmethod
    def load_warmup_cache(self, zone: str) -> pd.DataFrame | None: ...
