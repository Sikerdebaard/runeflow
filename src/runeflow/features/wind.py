# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2024-2026 Thomas Phil — runeflow
# See LICENSE and COMMERCIAL-LICENSE.md for licensing details.

"""Wind feature group."""
from __future__ import annotations

import pandas as pd

from runeflow.zones.config import ZoneConfig
from .base import FeatureGroup


class WindFeatures(FeatureGroup):
    """Wind power potential, extremeness, scarcity, drought indicators."""

    name = "wind"

    def transform(self, df: pd.DataFrame, zone_cfg: ZoneConfig) -> pd.DataFrame:
        df = self._copy(df)

        wind_cols = [c for c in df.columns if "wind" in c.lower() and "speed" in c.lower()]
        if not wind_cols:
            return df

        wind_col = wind_cols[0]

        df["wind_power_potential"] = (df[wind_col] ** 3).clip(0, None)

        rolling_p90 = df[wind_col].shift(1).rolling(168, min_periods=72).quantile(0.9)
        df["wind_extremeness"] = (df[wind_col] / (rolling_p90 + 1e-8)).clip(0, 2)

        df["wind_gust_rate"] = df[wind_col].diff(1).rolling(3, min_periods=2).max()

        df["wind_scarcity"] = 1.0 / (df["wind_power_potential"] + 1.0)

        low_threshold = df[wind_col].quantile(0.25) if len(df) > 0 else 5.0
        df["wind_drought_hours"] = (
            (df[wind_col] < low_threshold).astype(int).rolling(72, min_periods=24).sum()
        )

        if "is_peak_hour" in df.columns:
            df["peak_hour_wind_scarcity"] = df["is_peak_hour"] * df["wind_scarcity"]
            df["peak_hour_wind_drought"] = df["is_peak_hour"] * df["wind_drought_hours"]

        return df