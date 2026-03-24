# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2024-2026 Thomas Phil — runeflow
# See LICENSE and COMMERCIAL-LICENSE.md for licensing details.

"""NED (NL) supplemental data adapter — historical utilization + 9-day forecast."""
from __future__ import annotations

import datetime
import time
from datetime import datetime as dt, timedelta

import pandas as pd
import requests
from loguru import logger

from runeflow.exceptions import AuthenticationError, DataUnavailableError, DownloadError
from runeflow.ports.supplemental import SupplementalDataPort

_BASE_URL = "https://api.ned.nl/v1/utilizations"
_SUPPORTED_ZONES = {"NL"}


class NedAdapter(SupplementalDataPort):
    """NED.nl historical electricity utilization and 9-day consumption forecast."""

    def __init__(self, api_key: str) -> None:
        if not api_key:
            raise AuthenticationError("NED API key is required. Set the NED environment variable.")
        self._api_key = api_key
        self._headers = {"X-AUTH-TOKEN": api_key, "Accept": "application/ld+json"}

    def supports_zone(self, zone: str) -> bool:
        return zone.upper() in _SUPPORTED_ZONES

    # ── SupplementalDataPort interface ────────────────────────────────────────

    def download(
        self, zone: str, start: datetime.date, end: datetime.date
    ) -> pd.DataFrame | None:
        if not self.supports_zone(zone):
            return None
        data = self._download_historical(start, end)
        if data is None or data.empty:
            return None
        # Rename volume → ned_utilization_kwh and set validfrom as index
        data["validfrom"] = pd.to_datetime(data["validfrom"])
        data = data.set_index("validfrom")
        if "volume" in data.columns:
            data = data[["volume"]].rename(columns={"volume": "ned_utilization_kwh"})
        return data

    def download_forecast(self, zone: str) -> pd.DataFrame | None:
        if not self.supports_zone(zone):
            return None
        data = self._download_forecasted()
        if data is None or data.empty:
            return None
        data["validfrom"] = pd.to_datetime(data["validfrom"])
        data = data.set_index("validfrom")
        if "forecast_kWh" in data.columns:
            data = data[["forecast_kWh"]].rename(columns={"forecast_kWh": "ned_forecast_kwh"})
        return data

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _download_historical(
        self, start: datetime.date, end: datetime.date
    ) -> pd.DataFrame | None:
        """Download historical hourly utilization, paged monthly."""
        start_dt = dt(start.year, start.month, start.day)
        end_dt = dt(end.year, end.month, end.day, 23, 59, 59)

        all_records: list[dict] = []
        for chunk_start, chunk_end in self._month_chunks(start_dt, end_dt):
            params = {
                "point": 0,
                "type": 1,
                "granularity": 5,
                "granularitytimezone": 1,
                "activity": 1,
                "classification": 2,
                "validfrom[after]": chunk_start.strftime("%Y-%m-%d"),
                "validfrom[strictly_before]": chunk_end.strftime("%Y-%m-%d"),
                "itemsPerPage": 1000,
            }
            try:
                resp = requests.get(
                    _BASE_URL, headers=self._headers, params=params, timeout=30
                )
                resp.raise_for_status()
                records = resp.json().get("hydra:member", [])
                all_records.extend(records)
            except requests.RequestException as exc:
                raise DownloadError(f"NED historical request failed: {exc}") from exc
            time.sleep(0.3)

        if not all_records:
            return None

        df = pd.DataFrame(all_records)
        if "validfrom" in df.columns:
            df["validfrom"] = pd.to_datetime(df["validfrom"])
            df = (
                df.sort_values("validfrom")
                .drop_duplicates(subset=["validfrom"])
                .reset_index(drop=True)
            )
        if "volume" in df.columns:
            df["volume"] = pd.to_numeric(df["volume"], errors="coerce")
        return df

    def _download_forecasted(self) -> pd.DataFrame | None:
        """Download 9-day consumption forecast from NED."""
        now = dt.now()
        params = {
            "point": 0,
            "type": 59,
            "granularity": 5,
            "granularitytimezone": 1,
            "activity": 2,
            "classification": 1,
            "validfrom[after]": now.strftime("%Y-%m-%d"),
            "validfrom[strictly_before]": (now + timedelta(days=9)).strftime("%Y-%m-%d"),
            "itemsPerPage": 200,
        }
        try:
            resp = requests.get(
                _BASE_URL, headers=self._headers, params=params, timeout=30
            )
            resp.raise_for_status()
            data = resp.json().get("hydra:member", [])
        except requests.RequestException as exc:
            raise DownloadError(f"NED forecast request failed: {exc}") from exc

        if not data:
            return None
        df = pd.DataFrame(data)
        df["validfrom"] = pd.to_datetime(df["validfrom"])
        df = df.sort_values("validfrom").reset_index(drop=True)
        df["forecast_kWh"] = pd.to_numeric(df.get("volume"), errors="coerce")
        return df[["validfrom", "forecast_kWh"]]

    @staticmethod
    def _month_chunks(
        start: dt, end: dt
    ) -> list[tuple[dt, dt]]:
        chunks = []
        cur = dt(start.year, start.month, 1)
        while cur < end:
            if cur.month == 12:
                nxt = dt(cur.year + 1, 1, 1)
            else:
                nxt = dt(cur.year, cur.month + 1, 1)
            chunks.append((cur, nxt))
            cur = nxt
        return chunks