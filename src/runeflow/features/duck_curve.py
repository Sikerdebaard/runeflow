# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2024-2026 Thomas Phil — runeflow
# See LICENSE and COMMERCIAL-LICENSE.md for licensing details.

"""Duck-curve severity feature group."""

from __future__ import annotations

import numpy as np
import pandas as pd

from runeflow.zones.config import ZoneConfig

from .base import FeatureGroup


class DuckCurveFeatures(FeatureGroup):
    """
    Evening ramp severity = solar_ramp_down × heating_demand × wind_scarcity.

    Captures the core mechanism behind solar-induced evening spikes.
    """

    name = "duck_curve"

    @property
    def requires(self) -> tuple[str, ...]:
        return ("solar_ramp_down", "wind_scarcity", "hdd")

    def transform(self, df: pd.DataFrame, zone_cfg: ZoneConfig) -> pd.DataFrame:
        df = self._copy(df)

        has_solar_ramp = "solar_ramp_down" in df.columns
        has_wind_scarcity = "wind_scarcity" in df.columns
        has_hdd = "hdd" in df.columns

        if not (has_solar_ramp and has_wind_scarcity and has_hdd):
            return df

        df["evening_ramp_severity"] = (
            df["solar_ramp_down"]
            * np.maximum(df["hdd"], 1.0)
            * (1.0 + df["wind_scarcity"].clip(0, 10.0))
        )

        if "is_solar_cliff" in df.columns:
            df["evening_ramp_severity_peak"] = df["is_solar_cliff"] * df["evening_ramp_severity"]

        return df
