# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2024-2026 Thomas Phil — runeflow
# See LICENSE and COMMERCIAL-LICENSE.md for licensing details.

"""Cross-border / interconnector feature group."""
from __future__ import annotations

import numpy as np
import pandas as pd

from runeflow.zones.config import ZoneConfig
from .base import FeatureGroup


class CrossBorderFeatures(FeatureGroup):
    """
    German wind index and French nuclear cooling-risk proxy.

    Driven by neighbor weather columns (prefixed by location name).
    """

    name = "cross_border"

    def transform(self, df: pd.DataFrame, zone_cfg: ZoneConfig) -> pd.DataFrame:
        df = self._copy(df)

        # German wind power index from multiple locations
        lower_saxony = [c for c in df.columns if "lower_saxony" in c.lower() and "wind_speed" in c.lower()]
        brandenburg = [c for c in df.columns if "brandenburg" in c.lower() and "wind_speed" in c.lower()]
        schleswig = [c for c in df.columns if "schleswig" in c.lower() and "wind_speed" in c.lower()]

        components = [
            (df[c] ** 3).clip(0, None)
            for col_list in [lower_saxony, brandenburg, schleswig]
            for c in col_list[:1]
        ]

        if components:
            df["german_wind_power_index"] = sum(components) / len(components)

            gw_p10 = (
                df["german_wind_power_index"].shift(1).rolling(168, min_periods=72).quantile(0.1)
            )
            df["german_wind_drought"] = (
                (df["german_wind_power_index"] < gw_p10).fillna(0).astype(int)
            )
            is_evening_peak = df.index.hour.isin([17, 18, 19, 20]).astype(int)
            df["german_wind_drought_peak"] = df["german_wind_drought"] * is_evening_peak

            # Interconnector stress proxy
            wind_shortfall = np.maximum(
                0.0, gw_p10 - df["german_wind_power_index"]
            )
            df["interconnector_stress_proxy"] = (
                wind_shortfall / (gw_p10 + 1e-8)
            ).clip(0, 3)
            df["interconnector_stress_peak"] = df["interconnector_stress_proxy"] * is_evening_peak

        # French nuclear cooling risk (river temperature > 30°C → curtailment)
        france_temp_cols = [
            c for c in df.columns
            if any(loc in c.lower() for loc in ["normandy", "rhone", "grand_est"])
            and "temperature" in c.lower()
        ]
        if france_temp_cols:
            df["french_max_temp"] = df[france_temp_cols].max(axis=1)
            df["french_nuclear_cooling_risk"] = np.maximum(
                0.0, df["french_max_temp"] - 30.0
            )

        return df