# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2024-2026 Thomas Phil — runeflow
# See LICENSE and COMMERCIAL-LICENSE.md for licensing details.

"""Cloud cover and radiation trend feature group."""

from __future__ import annotations

import pandas as pd

from runeflow.zones.config import ZoneConfig

from .base import FeatureGroup


class CloudRadiationFeatures(FeatureGroup):
    """Cloud-change rate, cloud duration, solar deficit."""

    name = "cloud"

    def transform(self, df: pd.DataFrame, zone_cfg: ZoneConfig) -> pd.DataFrame:
        df = self._copy(df)

        cloud_cols = [c for c in df.columns if "cloud" in c.lower()]
        solar_cols = [c for c in df.columns if "radiation" in c.lower() or "solar" in c.lower()]

        if cloud_cols:
            cloud_col = cloud_cols[0]
            df["cloud_change_rate"] = df[cloud_col].diff(1).abs().rolling(3, min_periods=2).mean()
            df["cloud_duration"] = (df[cloud_col] > 70).astype(int).rolling(24, min_periods=1).sum()
            df["clear_sky_duration"] = (
                (df[cloud_col] < 30).astype(int).rolling(24, min_periods=1).sum()
            )

        if solar_cols:
            solar_col = solar_cols[0]
            col_max = df[solar_col].max()
            if col_max and col_max > 0:
                df["solar_deficit"] = (100 - df[solar_col] / col_max * 100).clip(lower=0)
            else:
                df["solar_deficit"] = 0

        return df
