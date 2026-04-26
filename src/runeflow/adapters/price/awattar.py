# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2024-2026 Thomas Phil — runeflow
# See LICENSE and COMMERCIAL-LICENSE.md for licensing details.

"""aWATTar price adapter — free DE/AT API, no authentication required."""

from __future__ import annotations

import datetime
import time

import pandas as pd
import requests
from loguru import logger

from runeflow.domain.price import PriceSeries
from runeflow.exceptions import DataUnavailableError, DownloadError
from runeflow.ports.price import PricePort

_SUPPORTED_ZONES = {"DE_LU", "AT"}
_BASE_URLS: dict[str, str] = {
    "DE_LU": "https://api.awattar.de/v1/marketdata",
    "AT": "https://api.awattar.at/v1/marketdata",
}

# Maximum days per single API request
_CHUNK_SIZE_DAYS = 30

# Polite delay between chunked requests (seconds)
_CHUNK_DELAY = 0.5


class AwattarPriceAdapter(PricePort):
    """Download day-ahead electricity prices from aWATTar (DE/AT)."""

    @property
    def name(self) -> str:
        return "aWATTar"

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
            raise DataUnavailableError(
                f"aWATTar only supports DE_LU and AT; zone '{zone}' is not supported."
            )

        chunks: list[pd.DataFrame] = []
        chunk_start = start
        while chunk_start <= end:
            chunk_end = min(chunk_start + datetime.timedelta(days=_CHUNK_SIZE_DAYS - 1), end)
            logger.info(f"[aWATTar] Downloading {zone_upper} {chunk_start} → {chunk_end}…")
            chunk_df = self._fetch_chunk(zone_upper, chunk_start, chunk_end)
            if chunk_df is not None and not chunk_df.empty:
                chunks.append(chunk_df)
            chunk_start = chunk_end + datetime.timedelta(days=1)
            if chunk_start <= end:
                time.sleep(_CHUNK_DELAY)

        if not chunks:
            raise DataUnavailableError(
                f"aWATTar returned no data for {zone_upper} ({start} → {end})."
            )

        df = pd.concat(chunks, ignore_index=True)
        df.drop_duplicates(subset=["date"], keep="first", inplace=True)
        df.sort_values("date", inplace=True)
        df.reset_index(drop=True, inplace=True)
        return PriceSeries.from_dataframe(df, zone=zone_upper, source=self.name)

    def download_day_ahead(self, zone: str) -> PriceSeries | None:
        if not self.supports_zone(zone):
            return None
        tomorrow = datetime.date.today() + datetime.timedelta(days=1)
        try:
            return self.download_historical(zone, tomorrow, tomorrow)
        except (DataUnavailableError, DownloadError) as exc:
            logger.warning(f"[aWATTar] Day-ahead not yet available: {exc}")
            return None

    # ── Internal ─────────────────────────────────────────────────────────────

    def _fetch_chunk(
        self,
        zone: str,
        start: datetime.date,
        end: datetime.date,
    ) -> pd.DataFrame | None:
        base_url = _BASE_URLS[zone]

        # aWATTar expects epoch-milliseconds
        start_ms = int(
            datetime.datetime(start.year, start.month, start.day, tzinfo=datetime.UTC).timestamp()
            * 1000
        )
        # end is inclusive, so fetch up to midnight of the next day
        end_dt = datetime.datetime(
            end.year, end.month, end.day, tzinfo=datetime.UTC
        ) + datetime.timedelta(days=1)
        end_ms = int(end_dt.timestamp() * 1000)

        params = {"start": start_ms, "end": end_ms}

        try:
            resp = requests.get(base_url, params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()
        except requests.RequestException as exc:
            raise DownloadError(f"aWATTar request failed: {exc}") from exc

        items = data.get("data", [])
        if not items:
            logger.warning(f"[aWATTar] No data in response for {zone} {start} → {end}")
            return None

        rows = []
        for item in items:
            ts = pd.to_datetime(item["start_timestamp"], unit="ms", utc=True)
            price_mwh = float(item["marketprice"])  # already EUR/MWh
            rows.append({"date": ts, "Price_EUR_MWh": price_mwh})

        return pd.DataFrame(rows)
