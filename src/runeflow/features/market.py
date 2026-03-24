# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2024-2026 Thomas Phil — runeflow
# See LICENSE and COMMERCIAL-LICENSE.md for licensing details.

"""Market structure and temporal interaction feature group."""

from __future__ import annotations

import pandas as pd

from runeflow.zones.config import ZoneConfig

from .base import FeatureGroup


class MarketStructureFeatures(FeatureGroup):
    """
    Hour×DoW interactions, season×hour patterns, weather-volatility index.
    """

    name = "market_structure"

    @property
    def requires(self) -> tuple[str, ...]:
        return ("hour_of_day", "day_of_week", "month", "is_weekend", "is_peak_hour")

    def transform(self, df: pd.DataFrame, zone_cfg: ZoneConfig) -> pd.DataFrame:
        df = self._copy(df)

        if "hour_of_day" not in df.columns or "day_of_week" not in df.columns:
            return df

        df["hour_dow_mon_evening"] = (
            (df["hour_of_day"].isin([17, 18, 19])) & (df["day_of_week"] == 0)
        ).astype(int)
        df["hour_dow_fri_evening"] = (
            (df["hour_of_day"].isin([17, 18, 19])) & (df["day_of_week"] == 4)
        ).astype(int)
        df["hour_dow_weekend"] = df["is_weekend"] * df["hour_of_day"]

        if "month" in df.columns and "is_peak_hour" in df.columns:
            df["winter_peak_hour"] = (
                df["month"].isin([12, 1, 2]) & df["is_peak_hour"].astype(bool)
            ).astype(int)
            df["summer_peak_hour"] = (
                df["month"].isin([6, 7, 8]) & df["is_peak_hour"].astype(bool)
            ).astype(int)
            df["winter_evening"] = (
                df["month"].isin([12, 1, 2]) & df["hour_of_day"].isin([17, 18, 19])
            ).astype(int)

        # Weather volatility index
        weather_features = [
            c
            for c in df.columns
            if any(x in c.lower() for x in ["temp", "wind", "precip", "cloud"])
        ]
        if len(weather_features) >= 3:
            weather_data = df[weather_features]
            normalised_changes = weather_data.diff().abs() / (weather_data.std() + 1e-8)
            df["weather_volatility_24h"] = (
                normalised_changes.mean(axis=1).rolling(24, min_periods=12).mean()
            )

        return df
