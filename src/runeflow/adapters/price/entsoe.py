# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2024-2026 Thomas Phil — runeflow
# See LICENSE and COMMERCIAL-LICENSE.md for licensing details.

"""ENTSO-E price adapter — wraps entsoe-py."""

from __future__ import annotations

import datetime

import pandas as pd
from entsoe import EntsoePandasClient
from entsoe.mappings import Area
from loguru import logger

from runeflow.domain.price import PriceSeries
from runeflow.exceptions import AuthenticationError, DataUnavailableError, DownloadError
from runeflow.ports.price import PricePort

# Mapping from ENTSO-E zone code → Area enum
_AREA_MAP: dict[str, Area] = {area.name: area for area in Area}

# Additional common aliases
_ZONE_ALIASES: dict[str, str] = {
    "DE_LU": "DE_LU",
    "DE": "DE_LU",
    # Ireland's day-ahead market uses the Single Electricity Market area code
    "IE": "IE_SEM",
}


def _zone_to_area(zone: str) -> Area:
    """Convert zone code to ENTSO-E Area enum."""
    canonical = _ZONE_ALIASES.get(zone.upper(), zone.upper())
    if canonical in _AREA_MAP:
        return _AREA_MAP[canonical]
    raise DataUnavailableError(f"Zone '{zone}' not recognised by entsoe-py Area enum.")


class EntsoePriceAdapter(PricePort):
    """Download day-ahead electricity prices from ENTSO-E Transparency Platform."""

    def __init__(self, api_key: str) -> None:
        if not api_key:
            raise AuthenticationError("ENTSO-E API key is required.")
        self._client = EntsoePandasClient(api_key=api_key)

    @property
    def name(self) -> str:
        return "ENTSO-E"

    def supports_zone(self, zone: str) -> bool:
        try:
            _zone_to_area(zone)
            return True
        except DataUnavailableError:
            return False

    def download_historical(
        self,
        zone: str,
        start: datetime.date,
        end: datetime.date,
    ) -> PriceSeries:
        """Download day-ahead spot prices for *zone* from *start* to *end* (inclusive)."""
        area = _zone_to_area(zone)
        # ENTSO-E API uses timezone-aware pd.Timestamp
        ts_start = pd.Timestamp(start)
        ts_start = (
            ts_start.tz_localize("UTC") if ts_start.tzinfo is None else ts_start.tz_convert("UTC")
        )
        ts_end = pd.Timestamp(end)
        ts_end = ts_end.tz_localize("UTC") if ts_end.tzinfo is None else ts_end.tz_convert("UTC")
        ts_end += pd.Timedelta("23h")

        logger.info(f"[ENTSO-E] Downloading prices for {zone} ({start} → {end})…")
        try:
            series: pd.Series = self._client.query_day_ahead_prices(
                country_code=area.name,
                start=ts_start,
                end=ts_end,
            )
        except Exception as exc:
            raise DownloadError(f"ENTSO-E download failed for {zone}: {exc}") from exc

        if series is None or series.empty:
            raise DataUnavailableError(f"ENTSO-E returned no data for {zone} ({start} → {end}).")

        # Convert Series to DataFrame (index → 'date', values → 'Price_EUR_MWh')
        df = series.rename("Price_EUR_MWh").rename_axis("date").reset_index()
        df["date"] = pd.to_datetime(df["date"], utc=True)

        return PriceSeries.from_dataframe(df, zone=zone, source=self.name)

    def download_day_ahead(self, zone: str) -> PriceSeries | None:
        """Download tomorrow's day-ahead prices (published at ~13:00 CET)."""
        tomorrow = datetime.date.today() + datetime.timedelta(days=1)
        try:
            return self.download_historical(zone, tomorrow, tomorrow)
        except (DataUnavailableError, DownloadError) as exc:
            logger.warning(f"[ENTSO-E] Day-ahead prices not yet available: {exc}")
            return None

    def get_supported_zones(self) -> set[str]:
        """Return all ENTSO-E Area names — for compatibility / zone discovery."""
        return {area.name for area in Area}
