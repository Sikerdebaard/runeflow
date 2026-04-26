# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2024-2026 Thomas Phil — runeflow
# See LICENSE and COMMERCIAL-LICENSE.md for licensing details.

"""Forecast domain types."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class ForecastPoint:
    """Single forecast hour with uncertainty bounds."""

    timestamp: pd.Timestamp
    prediction: float  # Central estimate (EUR/MWh)
    lower: float  # Combined lower bound (weather ensemble ∪ model quantiles)
    upper: float  # Combined upper bound (weather ensemble ∪ model quantiles)
    uncertainty: float  # upper - lower
    model_agreement: float  # 0–1 score
    lower_static: float = 0.0  # Model-quantile-only lower bound (no weather spread)
    upper_static: float = 0.0  # Model-quantile-only upper bound (no weather spread)
    ensemble_p50: float = 0.0  # Median of weather-ensemble member predictions
    ensemble_p25: float = 0.0  # P25 of ensemble members (IQR lower)
    ensemble_p75: float = 0.0  # P75 of ensemble members (IQR upper)


@dataclass(frozen=True)
class ForecastResult:
    """Complete forecast output."""

    zone: str
    points: tuple[ForecastPoint, ...]
    ensemble_members: pd.DataFrame  # N-column DataFrame, one per weather scenario
    model_predictions: dict[str, pd.Series]  # Individual model outputs
    created_at: pd.Timestamp
    model_version: str

    def to_dataframe(self) -> pd.DataFrame:
        """Flat DataFrame: DatetimeIndex, columns prediction/lower/upper/uncertainty."""
        if not self.points:
            return pd.DataFrame(
                columns=["prediction", "lower", "upper", "uncertainty", "model_agreement"]
            )
        return pd.DataFrame(
            {
                "prediction": [p.prediction for p in self.points],
                "lower": [p.lower for p in self.points],
                "upper": [p.upper for p in self.points],
                "uncertainty": [p.uncertainty for p in self.points],
                "model_agreement": [p.model_agreement for p in self.points],
                "lower_static": [p.lower_static for p in self.points],
                "upper_static": [p.upper_static for p in self.points],
                "ensemble_p50": [p.ensemble_p50 for p in self.points],
                "ensemble_p25": [p.ensemble_p25 for p in self.points],
                "ensemble_p75": [p.ensemble_p75 for p in self.points],
            },
            index=pd.DatetimeIndex([p.timestamp for p in self.points]),
        )

    def __len__(self) -> int:
        return len(self.points)
