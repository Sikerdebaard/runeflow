# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2024-2026 Thomas Phil — runeflow
# See LICENSE and COMMERCIAL-LICENSE.md for licensing details.

"""EnergyZero price adapter — free NL-only API, no authentication required."""

from __future__ import annotations

import datetime
import time

import pandas as pd
import requests
from loguru import logger

from runeflow.domain.price import PriceSeries
from runeflow.exceptions import DataUnavailableError, DownloadError
from runeflow.ports.price import PricePort

_BASE_URL = "https://api.energyzero.nl/v1/energyprices"
_SUPPORTED_ZONES = {"NL"}

# EnergyZero returns EUR/kWh; convert to EUR/MWh
_ENERGYZERO_TO_MWH = 1000.0

# Maximum days per single API request
_CHUNK_SIZE_DAYS = 90

# Polite delay between chunked requests (seconds)
_CHUNK_DELAY = 0.5


class EnergyZeroPriceAdapter(PricePort):
    """Download Dutch day-ahead electricity prices from EnergyZero."""

    @property
    def name(self) -> str:
        return "EnergyZero"

    def supports_zone(self, zone: str) -> bool:
        return zone.upper() in _SUPPORTED_ZONES

    def download_historical(
        self,
        zone: str,
        start: datetime.date,
        end: datetime.date,
    ) -> PriceSeries:
        if not self.supports_zone(zone):
            raise DataUnavailableError(
                f"EnergyZero only supports NL; zone '{zone}' is not supported."
            )

        # Split into ≤90-day chunks
        chunks: list[pd.DataFrame] = []
        chunk_start = start
        while chunk_start <= end:
            chunk_end = min(chunk_start + datetime.timedelta(days=_CHUNK_SIZE_DAYS - 1), end)
            logger.info(f"[EnergyZero] Downloading {chunk_start} → {chunk_end}…")
            chunk_df = self._fetch_chunk(chunk_start, chunk_end)
            if chunk_df is not None and not chunk_df.empty:
                chunks.append(chunk_df)
            chunk_start = chunk_end + datetime.timedelta(days=1)
            if chunk_start <= end:
                time.sleep(_CHUNK_DELAY)

        if not chunks:
            raise DataUnavailableError(f"EnergyZero returned no data for NL ({start} → {end}).")

        df = pd.concat(chunks, ignore_index=True)
        df.drop_duplicates(subset=["date"], keep="first", inplace=True)
        df.sort_values("date", inplace=True)
        df.reset_index(drop=True, inplace=True)
        return PriceSeries.from_dataframe(df, zone="NL", source=self.name)

    def download_day_ahead(self, zone: str) -> PriceSeries | None:
        if not self.supports_zone(zone):
            return None
        tomorrow = datetime.date.today() + datetime.timedelta(days=1)
        try:
            return self.download_historical(zone, tomorrow, tomorrow)
        except (DataUnavailableError, DownloadError) as exc:
            logger.warning(f"[EnergyZero] Day-ahead not yet available: {exc}")
            return None

    # ── Internal ─────────────────────────────────────────────────────────────

    def _fetch_chunk(self, start: datetime.date, end: datetime.date) -> pd.DataFrame | None:
        params = {
            "fromDate": start.strftime("%Y-%m-%dT00:00:00.000Z"),
            "tillDate": end.strftime("%Y-%m-%dT23:59:59.000Z"),
            "interval": 4,  # Hourly
            "usageType": 1,  # Electricity
            "inclBtw": "false",  # Excl. VAT
        }
        try:
            resp = requests.get(_BASE_URL, params=params, timeout=30)  # type: ignore[arg-type]
            resp.raise_for_status()
            data = resp.json()
        except requests.RequestException as exc:
            raise DownloadError(f"EnergyZero request failed: {exc}") from exc

        prices = data.get("Prices", [])
        if not prices:
            logger.warning(f"[EnergyZero] No prices in response for {start} → {end}")
            return None

        rows = []
        for item in prices:
            ts = pd.to_datetime(item.get("readingDate"), utc=True)
            price_kwh = float(item.get("price", 0.0))
            rows.append({"date": ts, "Price_EUR_MWh": price_kwh * _ENERGYZERO_TO_MWH})

        return pd.DataFrame(rows)
