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

from runeflow.ports.commodity import CommodityPricePort
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
        try:
            self._commodity_port: CommodityPricePort | None = inject.instance(CommodityPricePort)  # type: ignore[assignment]
        except Exception:
            self._commodity_port = None

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
        self._update_commodity(zone, year_list)
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

        # Skip only when the cache actually covers the full requested year range.
        # "any data exists" is not sufficient — the cache end date must be checked.
        existing = self._store.load_weather(zone)
        needs_download = True
        if existing is not None and len(existing.df) > 0:
            cached_last = existing.df.index[-1]
            # Normalise to UTC for comparison.
            cached_last_utc = (
                pd.Timestamp(cached_last).tz_convert("UTC")
                if cached_last.tzinfo
                else pd.Timestamp(cached_last, tz="UTC")
            )
            # Archive data can only cover up to today, so compare against today
            # not against end-of-year — otherwise the check always fails and we
            # re-download on every run (infinite loop).
            today_utc = pd.Timestamp.now("UTC").normalize()
            if cached_last_utc >= today_utc - pd.Timedelta(days=7):
                needs_download = False
                logger.info(
                    "Historical weather up-to-date for zone=%s (cached through %s)",
                    zone,
                    cached_last_utc.date(),
                )
            else:
                logger.info(
                    "Historical weather cache for zone=%s ends at %s, re-downloading through %s",
                    zone,
                    cached_last_utc.date(),
                    end,
                )

        if needs_download:
            series = self._weather_port.download_historical(locations, start, end)
            self._store.save_weather(series, zone)

        logger.info("Downloading weather forecast")
        forecast = self._weather_port.download_forecast(locations, horizon_days=9)
        self._store.save_forecast_weather(forecast, zone)

        # Ensemble weather is intentionally NOT downloaded here.
        # At ~60 MB per zone (91 members × 9 days × hourly), it times out silently
        # and blocks the pipeline for hours.  InferenceService downloads and caches
        # it at inference time when it is actually needed.

    # ------------------------------------------------------------------
    def _update_generation(self, zone: str, years: list[int]) -> None:
        if self._generation_port is None:
            return
        end = pd.Timestamp(f"{years[-1]}-12-31 23:00:00", tz="UTC")

        # Incremental: only download the missing tail rather than the full history.
        existing = self._store.load_generation(zone)
        if existing is not None and len(existing.df) > 0:
            cached_last = existing.df.index[-1]
            cached_last_utc = (
                pd.Timestamp(cached_last).tz_convert("UTC")
                if cached_last.tzinfo
                else pd.Timestamp(cached_last, tz="UTC")
            )
            today_utc = pd.Timestamp.now("UTC").normalize()
            if cached_last_utc >= today_utc - pd.Timedelta(days=3):
                logger.info(
                    "Generation data up-to-date for zone=%s (cached through %s), skipping",
                    zone,
                    cached_last_utc.date(),
                )
                return
            # Re-download from one week before the cache end to catch any revisions.
            start = (cached_last_utc - pd.Timedelta(days=7)).floor("D")
            logger.info(
                "Generation data cache ends at %s; downloading %s – %s",
                cached_last_utc.date(),
                start.date(),
                end.date(),
            )
        else:
            start = pd.Timestamp(f"{years[0]}-01-01", tz="UTC")
            logger.info("Downloading generation data %s – %s", start.date(), end.date())

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

    # ------------------------------------------------------------------
    def _update_commodity(self, zone: str, years: list[int]) -> None:
        if self._commodity_port is None:
            return
        start = pd.Timestamp(f"{years[0]}-01-01", tz="UTC").date()
        end = pd.Timestamp(f"{years[-1]}-12-31", tz="UTC").date()
        logger.info("Downloading commodity price data (oil, gas, coal)")
        df = self._commodity_port.download(start, end)
        if df is not None and not df.empty:
            self._store.save_supplemental(df, zone, "commodity")
