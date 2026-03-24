# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2024-2026 Thomas Phil — runeflow
# See LICENSE and COMMERCIAL-LICENSE.md for licensing details.

"""
UpdateDataService — downloads and caches prices, weather, generation, supplemental.

All ports are injected via @inject.autoparams().
"""

from __future__ import annotations

import logging
from datetime import date

import inject
import pandas as pd

from runeflow.ports.generation import GenerationPort
from runeflow.ports.price import PricePort
from runeflow.ports.store import DataStore
from runeflow.ports.supplemental import SupplementalDataPort
from runeflow.ports.validator import DataValidator
from runeflow.ports.weather import WeatherPort
from runeflow.zones.config import ZoneConfig

logger = logging.getLogger(__name__)


class UpdateDataService:
    """Downloads and persists all data sources for a zone."""

    @inject.autoparams()
    def __init__(
        self,
        zone_cfg: ZoneConfig = inject.attr("zone_config"),  # type: ignore[assignment]  # noqa: B008
        price_port: PricePort = inject.attr(PricePort),  # type: ignore[assignment]  # noqa: B008
        weather_port: WeatherPort = inject.attr(WeatherPort),  # type: ignore[assignment]  # noqa: B008, E501
        store: DataStore = inject.attr(DataStore),  # type: ignore[assignment]  # noqa: B008
        validator: DataValidator = inject.attr(DataValidator),  # type: ignore[assignment]  # noqa: B008, E501
    ) -> None:
        self._zone_cfg = zone_cfg
        self._price_port = price_port
        self._weather_port = weather_port
        self._store = store
        self._validator = validator
        # Optional ports — may not be bound
        try:
            self._generation_port: GenerationPort | None = inject.instance(GenerationPort)  # type: ignore[assignment]
        except Exception:
            self._generation_port = None
        try:
            self._supplemental_port: SupplementalDataPort | None = inject.instance(
                SupplementalDataPort
            )  # type: ignore[assignment]
        except Exception:
            self._supplemental_port = None

    # ------------------------------------------------------------------
    def run(self, years: tuple[int, ...] | None = None) -> None:
        """Download all data for the zone and persist to the store."""
        zone = self._zone_cfg.zone
        year_list = list(years or self._zone_cfg.historical_years)
        logger.info("UpdateData starting for zone=%s years=%s", zone, year_list)

        self._update_prices(zone, year_list)
        self._update_weather(zone, year_list)
        self._update_generation(zone, year_list)
        self._update_supplemental(zone, year_list)
        logger.info("UpdateData complete for zone=%s", zone)

    # ------------------------------------------------------------------
    def _update_prices(self, zone: str, years: list[int]) -> None:
        start = pd.Timestamp(f"{years[0]}-01-01", tz="UTC")
        end = pd.Timestamp(f"{years[-1]}-12-31 23:00:00", tz="UTC")
        logger.info("Downloading prices %s – %s", start.date(), end.date())

        existing = self._store.load_prices(zone)
        if existing is not None:
            existing_end = existing.date_range()[1]  # type: ignore[index]
            # Only fetch missing period — ensure tz-aware comparison
            gap_start = pd.Timestamp(existing_end, tz="UTC") + pd.Timedelta(hours=1)
            if gap_start >= end:
                logger.info("Prices up-to-date, skipping download")
                return
            start = gap_start

        series = self._price_port.download_historical(zone, start, end)
        df = series.to_dataframe()
        result = self._validator.validate(df, {"zone": zone, "check": "prices"})  # type: ignore[arg-type]
        if not result.passed:
            logger.warning("Price validation warnings: %s", result.warnings)
        self._store.save_prices(series)

    # ------------------------------------------------------------------
    def _update_weather(self, zone: str, years: list[int]) -> None:
        locations = list(self._zone_cfg.weather_locations)
        start = date(years[0], 1, 1)
        end = date(years[-1], 12, 31)
        logger.info("Downloading historical weather (%d locations)", len(locations))

        series = self._weather_port.download_historical(locations, start, end)
        self._store.save_weather(series, zone)

        logger.info("Downloading weather forecast")
        forecast = self._weather_port.download_forecast(locations, horizon_days=9)
        self._store.save_forecast_weather(forecast, zone)

        # Ensemble (optional — best-effort)
        try:
            members = self._weather_port.download_ensemble_forecast(locations)
            for i, wm in enumerate(members):
                self._store.save_forecast_weather(wm, zone, member=i)
        except Exception:
            logger.debug("Ensemble weather forecast not available")

    # ------------------------------------------------------------------
    def _update_generation(self, zone: str, years: list[int]) -> None:
        if self._generation_port is None:
            return
        start = pd.Timestamp(f"{years[0]}-01-01", tz="UTC")
        end = pd.Timestamp(f"{years[-1]}-12-31 23:00:00", tz="UTC")
        logger.info("Downloading generation data")
        series = self._generation_port.download_generation(zone, start, end)
        if series is not None:
            self._store.save_generation(series)

    # ------------------------------------------------------------------
    def _update_supplemental(self, zone: str, years: list[int]) -> None:
        if self._supplemental_port is None:
            return
        start = pd.Timestamp(f"{years[0]}-01-01", tz="UTC")
        end = pd.Timestamp(f"{years[-1]}-12-31 23:00:00", tz="UTC")
        logger.info("Downloading supplemental data")
        df = self._supplemental_port.download(zone, start, end)
        if df is not None and not df.empty:
            self._store.save_supplemental(df, zone, "historical")

        forecast_df = self._supplemental_port.download_forecast(zone)
        if forecast_df is not None and not forecast_df.empty:
            self._store.save_supplemental(forecast_df, zone, "forecast")
