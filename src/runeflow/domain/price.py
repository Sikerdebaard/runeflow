# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2024-2026 Thomas Phil — runeflow
# See LICENSE and COMMERCIAL-LICENSE.md for licensing details.

"""Price domain types."""

from __future__ import annotations

import datetime
from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class PriceRecord:
    """Single hourly electricity price."""

    timestamp: pd.Timestamp  # UTC, timezone-aware
    price_eur_mwh: float  # EPEX spot price in EUR/MWh


@dataclass(frozen=True)
class PriceSeries:
    """Time-indexed collection of price records."""

    zone: str
    records: tuple[PriceRecord, ...]
    source: str  # e.g. "entsoe", "energyzero"
    fetched_at: pd.Timestamp  # When the data was downloaded

    def to_dataframe(self) -> pd.DataFrame:
        """Convert to DataFrame with DatetimeIndex (UTC), column Price_EUR_MWh."""
        if not self.records:
            return pd.DataFrame(columns=["date", "Price_EUR_MWh"])
        df = pd.DataFrame(
            {
                "date": [r.timestamp for r in self.records],
                "Price_EUR_MWh": [r.price_eur_mwh for r in self.records],
            }
        )
        df = df.set_index("date")
        df.index = pd.DatetimeIndex(df.index)
        return df

    @classmethod
    def from_dataframe(
        cls,
        df: pd.DataFrame,
        zone: str,
        source: str,
        fetched_at: pd.Timestamp | None = None,
    ) -> PriceSeries:
        """
        Create a PriceSeries from a DataFrame.

        Expected columns: ``date`` (or DatetimeIndex) + ``Price_EUR_MWh``.
        """
        if fetched_at is None:
            fetched_at = pd.Timestamp.now("UTC")

        if "date" in df.columns:
            idx = pd.DatetimeIndex(df["date"])
            prices = df["Price_EUR_MWh"].tolist()
        else:
            idx = pd.DatetimeIndex(df.index)
            col = "Price_EUR_MWh" if "Price_EUR_MWh" in df.columns else df.columns[0]
            prices = df[col].tolist()

        records = tuple(
            PriceRecord(timestamp=ts, price_eur_mwh=float(p))
            for ts, p in zip(idx, prices, strict=False)
        )
        return cls(zone=zone, records=records, source=source, fetched_at=fetched_at)

    def __len__(self) -> int:
        return len(self.records)

    def date_range(self) -> tuple[datetime.date, datetime.date] | None:
        if not self.records:
            return None
        return (
            self.records[0].timestamp.date(),
            self.records[-1].timestamp.date(),
        )
