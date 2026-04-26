# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2024-2026 Thomas Phil — runeflow
# See LICENSE and COMMERCIAL-LICENSE.md for licensing details.

"""Model performance domain types."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class ForecastAccuracy:
    """Accuracy metrics for a single archived forecast compared against actuals."""

    forecast_created_at: pd.Timestamp
    model_version: str
    n_comparable_hours: int  # Hours where both forecast and actual exist
    mae: float  # Mean Absolute Error (EUR/MWh)
    rmse: float  # Root Mean Squared Error (EUR/MWh)
    mape: float | None  # Mean Absolute % Error (None when actuals contain zeros)
    directional_accuracy: float  # Fraction of hours where direction (up/down) was correct
    mean_model_agreement: float  # Average model_agreement score for compared hours


@dataclass(frozen=True)
class HorizonMetrics:
    """Accuracy bucketed by forecast horizon band."""

    horizon_label: str  # e.g. "1-6h", "6-12h", "12-24h", "24-48h", "48h+"
    n_hours: int
    mae: float
    rmse: float


@dataclass(frozen=True)
class ZonePerformance:
    """Complete performance profile for a single zone."""

    zone: str
    zone_name: str

    # Training history from the sidecar (one row per training run)
    training_history: list[dict]  # [{model_version, mae, r2, coverage, trained_at}]

    # Retrospective accuracy per archived forecast
    forecast_accuracies: list[ForecastAccuracy]
    horizon_metrics: list[HorizonMetrics]

    # Aggregate stats across all archived forecasts
    overall_mae: float | None
    overall_rmse: float | None
    ensemble_coverage_pct: float | None  # % of actuals within the P25–P75 ensemble band
    n_archived_forecasts: int
    n_comparable_hours: int

    generated_at: pd.Timestamp


@dataclass(frozen=True)
class GlobalPerformance:
    """Cross-zone performance comparison."""

    zones: dict[str, ZonePerformance]
    rankings: list[dict]  # [{zone, mae, rmse, rank}] sorted ascending by MAE
    generated_at: pd.Timestamp
