# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2024-2026 Thomas Phil — runeflow
# See LICENSE and COMMERCIAL-LICENSE.md for licensing details.

"""Nordpool price adapter — Nordic/Baltic zones via nordpool Python library."""

from __future__ import annotations

import datetime
import time

import pandas as pd
from loguru import logger

from runeflow.domain.price import PriceSeries
from runeflow.exceptions import DataUnavailableError, DownloadError
from runeflow.ports.price import PricePort

# Maps runeflow zone codes → Nordpool area codes
_ZONE_TO_AREA: dict[str, str] = {
    "DK_1": "DK1",
    "DK_2": "DK2",
    "FI": "FI",
    "NO_1": "Oslo",
    "NO_2": "Kr.sand",
    "NO_3": "Trondheim",
    "NO_4": "Tromsø",
    "NO_5": "Bergen",
    "SE_1": "SE1",
    "SE_2": "SE2",
    "SE_3": "SE3",
    "SE_4": "SE4",
    "EE": "EE",
    "LV": "LV",
    "LT": "LT",
}

_SUPPORTED_ZONES = set(_ZONE_TO_AREA.keys())

# Polite delay between per-day fetch calls (seconds)
_CHUNK_DELAY = 0.5


class NordpoolPriceAdapter(PricePort):
    """Download day-ahead electricity prices from Nordpool (Nordic/Baltic)."""

    def __init__(self) -> None:
        from nordpool import elspot  # type: ignore[import-untyped]

        self._api = elspot.Prices(currency="EUR")

    @property
    def name(self) -> str:
        return "Nordpool"

    def supports_zone(self, zone: str) -> bool:
        return zone.upper() in _SUPPORTED_ZONES

    def download_historical(
        self,
        zone: str,
        start: datetime.date,
        end: datetime.date,
    ) -> PriceSeries:
        zone_upper = zone.upper()
        if zone_upper not in _SUPPORTED_ZONES:
            raise DataUnavailableError(f"Nordpool adapter does not support zone '{zone}'.")

        area = _ZONE_TO_AREA[zone_upper]
        rows: list[dict] = []
        current = start

        while current <= end:
            logger.info(f"[Nordpool] Fetching {zone_upper} ({area}) for {current}…")
            try:
                day_data = self._api.fetch(end_date=current, areas=[area])
            except Exception as exc:
                raise DownloadError(f"Nordpool fetch failed for {area} {current}: {exc}") from exc

            area_values = day_data.get("areas", {}).get(area, {}).get("values", [])
            for entry in area_values:
                value = entry.get("value")
                if value is None:
                    continue
                ts = pd.Timestamp(entry["start"]).tz_localize(None).tz_localize("UTC")
                rows.append({"date": ts, "Price_EUR_MWh": float(value)})

            current += datetime.timedelta(days=1)
            if current <= end:
                time.sleep(_CHUNK_DELAY)

        if not rows:
            raise DataUnavailableError(
                f"Nordpool returned no data for {zone_upper} ({start} → {end})."
            )

        df = pd.DataFrame(rows)
        df.drop_duplicates(subset=["date"], keep="first", inplace=True)
        df.sort_values("date", inplace=True)
        df.reset_index(drop=True, inplace=True)
        return PriceSeries.from_dataframe(df, zone=zone_upper, source=self.name)

    def download_day_ahead(self, zone: str) -> PriceSeries | None:
        if not self.supports_zone(zone):
            return None
        zone_upper = zone.upper()
        area = _ZONE_TO_AREA[zone_upper]
        try:
            day_data = self._api.fetch(areas=[area])
        except Exception as exc:
            logger.warning(f"[Nordpool] Day-ahead fetch failed for {area}: {exc}")
            return None

        area_values = day_data.get("areas", {}).get(area, {}).get("values", [])
        rows: list[dict] = []
        for entry in area_values:
            value = entry.get("value")
            if value is None:
                continue
            ts = pd.Timestamp(entry["start"]).tz_localize(None).tz_localize("UTC")
            rows.append({"date": ts, "Price_EUR_MWh": float(value)})

        if not rows:
            logger.warning(f"[Nordpool] No day-ahead prices available for {area}")
            return None

        df = pd.DataFrame(rows)
        df.drop_duplicates(subset=["date"], keep="first", inplace=True)
        df.sort_values("date", inplace=True)
        df.reset_index(drop=True, inplace=True)
        return PriceSeries.from_dataframe(df, zone=zone_upper, source=self.name)
