# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2024-2026 Thomas Phil — runeflow
# See LICENSE and COMMERCIAL-LICENSE.md for licensing details.

"""Spike detection and clustering feature group."""
from __future__ import annotations

import numpy as np
import pandas as pd

from runeflow.zones.config import ZoneConfig
from .base import FeatureGroup


class SpikeMomentumFeatures(FeatureGroup):
    """Z-score, spike count, max/min rolling, momentum indicators."""

    name = "spike_momentum"

    @property
    def requires(self) -> tuple[str, ...]:
        return ("Price_EUR_MWh",)

    def transform(self, df: pd.DataFrame, zone_cfg: ZoneConfig) -> pd.DataFrame:
        df = self._copy(df)
        col = "Price_EUR_MWh"
        if col not in df.columns:
            return df

        shifted = df[col].shift(1)
        rolling_mean_24 = shifted.rolling(24, min_periods=12).mean()
        rolling_std_24 = shifted.rolling(24, min_periods=12).std()

        df[f"{col}_zscore_24h"] = (shifted - rolling_mean_24) / (rolling_std_24 + 1e-8)

        is_spike = (shifted > rolling_mean_24 + 2 * rolling_std_24).astype(float)
        df[f"{col}_spike_count_24h"] = is_spike.rolling(24, min_periods=12).sum()

        df[f"{col}_max_24h"] = shifted.rolling(24, min_periods=12).max()
        df[f"{col}_max_168h"] = shifted.rolling(168, min_periods=72).max()
        df[f"{col}_min_24h"] = shifted.rolling(24, min_periods=12).min()

        return df


class SpikeRiskFeatures(FeatureGroup):
    """
    Hours-since-last-spike (vectorised) + spike clustering.

    Uses cumsum trick to avoid Python-level loops.
    """

    name = "spike_risk"

    @property
    def requires(self) -> tuple[str, ...]:
        return ("Price_EUR_MWh",)

    def transform(self, df: pd.DataFrame, zone_cfg: ZoneConfig) -> pd.DataFrame:
        df = self._copy(df)
        col = "Price_EUR_MWh"
        if col not in df.columns:
            return df

        major_spike_threshold = df[col].quantile(0.95)
        is_spike = (df[col] > major_spike_threshold).astype(int)

        # Vectorised hours-since-last-spike via cumsum groups
        # For each spike occurrence, assign an incrementing group id
        spike_group = is_spike.cumsum()
        # Within each group, count rows since group boundary
        df["hours_since_last_spike"] = (
            df.groupby(spike_group).cumcount().where(~is_spike.astype(bool), 0)
        ).astype(float)

        df["spike_frequency_7d"] = is_spike.rolling(168, min_periods=1).sum()

        # Clustering
        df["spike_cluster_active"] = (df["spike_frequency_7d"] >= 3).astype(int)
        df["spike_cluster_intensity"] = df["spike_frequency_7d"] / 7.0

        df["days_since_spike"] = df["hours_since_last_spike"] / 24.0
        df["days_since_spike_inv"] = 1.0 / (df["days_since_spike"] + 1.0)

        if "is_peak_hour" in df.columns:
            df["recent_spike_peak_hour"] = df["days_since_spike_inv"] * df["is_peak_hour"]

        df["spike_acceleration_7d"] = (
            df["spike_frequency_7d"].diff(24).rolling(168, min_periods=72).mean()
        )

        # Risk composite
        risk_parts: list[pd.Series] = []
        if "is_peak_hour" in df.columns:
            risk_parts.append(df["is_peak_hour"].astype(float))
        if "temp_extremeness" in df.columns:
            risk_parts.append(df["temp_extremeness"])
        if "wind_scarcity" in df.columns:
            ws = df["wind_scarcity"]
            risk_parts.append(((ws - ws.min()) / (ws.max() - ws.min() + 1e-8)))
        if "spike_cluster_active" in df.columns:
            risk_parts.append(df["spike_cluster_active"].astype(float))

        if len(risk_parts) >= 2:
            df["spike_risk_composite"] = sum(risk_parts) / len(risk_parts)
            if len(risk_parts) >= 3:
                df["spike_risk_extreme"] = (df["spike_risk_composite"] > 0.6).astype(int)

        return df