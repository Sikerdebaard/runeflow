# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2024-2026 Thomas Phil — runeflow
# See LICENSE and COMMERCIAL-LICENSE.md for licensing details.

"""Peak interaction feature group."""

from __future__ import annotations

import pandas as pd

from runeflow.zones.config import ZoneConfig

from .base import FeatureGroup


class PeakInteractionFeatures(FeatureGroup):
    """
    Peak-hour interactions with temperature and spike indicators.
    Requires temperature, hdd/cdd, and spike features to be computed first.
    """

    name = "interaction"

    def transform(self, df: pd.DataFrame, zone_cfg: ZoneConfig) -> pd.DataFrame:
        df = self._copy(df)

        if "is_peak_hour" not in df.columns:
            return df

        # Temperature × peak
        if "hdd_24h" in df.columns and "temp_extremeness" in df.columns:
            df["peak_hour_cold_interaction"] = df["is_peak_hour"] * df["hdd_24h"]
            df["peak_hour_cold_extreme"] = (
                df["is_peak_hour"] * df["temp_extremeness"] * df["hdd_24h"]
            )

        if "cdd_24h" in df.columns:
            df["peak_hour_heat_interaction"] = df["is_peak_hour"] * df["cdd_24h"]

        return df
