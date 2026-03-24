# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2024-2026 Thomas Phil — runeflow
# See LICENSE and COMMERCIAL-LICENSE.md for licensing details.

"""Precipitation and snowfall feature group."""

from __future__ import annotations

import pandas as pd

from runeflow.zones.config import ZoneConfig

from .base import FeatureGroup


class PrecipitationFeatures(FeatureGroup):
    """
    Precipitation accumulation with exponential decay.

    Snow accumulation is vectorised via ``ewm`` to avoid row-level loops.
    """

    name = "precipitation"

    def transform(self, df: pd.DataFrame, zone_cfg: ZoneConfig) -> pd.DataFrame:
        df = self._copy(df)

        precip_cols = [c for c in df.columns if "precip" in c.lower() or "rain" in c.lower()]
        _snow_cols = [c for c in df.columns if "snow" in c.lower()]
        _temp_cols = [c for c in df.columns if "temp" in c.lower() and "dew" not in c.lower()]

        if precip_cols:
            precip_col = precip_cols[0]
            # Exponential decay (half-life = 48 h), multiply by 10 to scale up
            df["precip_accumulated"] = (
                df[precip_col].shift(1).ewm(halflife=48, min_periods=1).mean() * 10
            )
            df["heavy_rain_recent"] = (
                (df[precip_col] > 2.0).shift(1).astype(float).rolling(6, min_periods=1).sum()
            )

        # NOTE: snow_accumulated removed to match production feature set.
        # Production model (215 features) does not include snow_accumulated.

        return df
