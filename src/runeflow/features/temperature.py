# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2024-2026 Thomas Phil — runeflow
# See LICENSE and COMMERCIAL-LICENSE.md for licensing details.

"""Temperature feature group."""
from __future__ import annotations

import numpy as np
import pandas as pd

from runeflow.zones.config import ZoneConfig
from .base import FeatureGroup

_HEATING_BASE = 15.5
_COOLING_BASE = 20.0


class TemperatureFeatures(FeatureGroup):
    """Temperature extremeness, HDD/CDD, rapid-change indicators."""

    name = "temperature"

    def transform(self, df: pd.DataFrame, zone_cfg: ZoneConfig) -> pd.DataFrame:
        df = self._copy(df)

        temp_cols = [c for c in df.columns if "temp" in c.lower() and "dew" not in c.lower()]
        if not temp_cols:
            return df

        temp_col = temp_cols[0]
        shifted_temp = df[temp_col].shift(1)

        rolling_p10 = shifted_temp.rolling(168, min_periods=72).quantile(0.1)
        rolling_p90 = shifted_temp.rolling(168, min_periods=72).quantile(0.9)
        temp_range = rolling_p90 - rolling_p10 + 1e-8

        below_normal = (rolling_p10 - df[temp_col]) / temp_range
        above_normal = (df[temp_col] - rolling_p90) / temp_range
        df["temp_extremeness"] = np.maximum(below_normal, above_normal).clip(0, 1)

        df["temp_change_1h"] = df[temp_col].diff(1).abs()
        df["temp_change_24h"] = df[temp_col].diff(24).abs()
        df["temp_change_rate"] = df["temp_change_1h"].rolling(6, min_periods=3).mean()

        df["hdd"] = np.maximum(0, _HEATING_BASE - df[temp_col])
        df["cdd"] = np.maximum(0, df[temp_col] - _COOLING_BASE)
        df["hdd_24h"] = df["hdd"].rolling(24, min_periods=12).sum()
        df["cdd_24h"] = df["cdd"].rolling(24, min_periods=12).sum()
        df["hdd_7d"] = df["hdd"].rolling(168, min_periods=72).sum()
        df["cdd_7d"] = df["cdd"].rolling(168, min_periods=72).sum()

        return df