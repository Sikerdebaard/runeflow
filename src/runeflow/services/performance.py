# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2024-2026 Thomas Phil — runeflow
# See LICENSE and COMMERCIAL-LICENSE.md for licensing details.

"""PerformanceService — compute retrospective forecast accuracy."""

from __future__ import annotations

import logging
import math
from typing import Any

import inject
import pandas as pd

from runeflow.domain.performance import (
    ForecastAccuracy,
    HorizonMetrics,
    ZonePerformance,
)
from runeflow.ports.store import DataStore
from runeflow.zones.config import ZoneConfig

logger = logging.getLogger(__name__)

# Horizon bands: (label, lower_h_exclusive, upper_h_inclusive)
_HORIZON_BANDS = [
    ("1-6h", 0, 6),
    ("6-12h", 6, 12),
    ("12-24h", 12, 24),
    ("24-48h", 24, 48),
    ("48h+", 48, float("inf")),
]


class PerformanceService:
    """Compute retrospective forecast accuracy for a single zone."""

    @inject.autoparams()
    def __init__(
        self,
        zone_cfg: ZoneConfig = inject.attr("zone_config"),  # type: ignore[assignment]  # noqa: B008
        store: DataStore = inject.attr(DataStore),  # type: ignore[assignment]  # noqa: B008
    ) -> None:
        self._zone_cfg = zone_cfg
        self._store = store

    # ------------------------------------------------------------------
    def compute_zone_performance(self) -> ZonePerformance:
        zone = self._zone_cfg.zone
        zone_name = self._zone_cfg.name

        train_history = self._load_train_history(zone)

        archives = self._store.load_forecast_archive(zone, days_back=30)
        actuals = self._store.load_prices(zone)

        accuracies: list[ForecastAccuracy] = []
        all_horizon_data: list[tuple[float, float, float, float]] = []
        # each entry: (horizon_hours, abs_error, actual_price, predicted_price)

        if actuals is not None and archives:
            actual_df = actuals.to_dataframe()
            # Ensure UTC datetime index
            if isinstance(actual_df.index, pd.DatetimeIndex):
                if actual_df.index.tz is None:
                    actual_df.index = actual_df.index.tz_localize("UTC")
                else:
                    actual_df.index = actual_df.index.tz_convert("UTC")

            for forecast in archives:
                acc, horizon_data = self._evaluate_forecast(forecast, actual_df)
                if acc is not None:
                    accuracies.append(acc)
                    all_horizon_data.extend(horizon_data)

        horizon_metrics = self._compute_horizon_metrics(all_horizon_data)
        ensemble_coverage = self._compute_ensemble_coverage(archives, actuals)

        overall_mae: float | None = None
        overall_rmse: float | None = None
        total_hours = sum(a.n_comparable_hours for a in accuracies)
        if accuracies and total_hours > 0:
            overall_mae = sum(a.mae * a.n_comparable_hours for a in accuracies) / total_hours
            overall_rmse = math.sqrt(
                sum((a.rmse**2) * a.n_comparable_hours for a in accuracies) / total_hours
            )

        return ZonePerformance(
            zone=zone,
            zone_name=zone_name,
            training_history=train_history,
            forecast_accuracies=accuracies,
            horizon_metrics=horizon_metrics,
            overall_mae=overall_mae,
            overall_rmse=overall_rmse,
            ensemble_coverage_pct=ensemble_coverage,
            n_archived_forecasts=len(archives),
            n_comparable_hours=total_hours,
            generated_at=pd.Timestamp.now("UTC"),
        )

    # ------------------------------------------------------------------
    def _evaluate_forecast(
        self,
        forecast: Any,
        actual_df: pd.DataFrame,
    ) -> tuple[ForecastAccuracy | None, list[tuple[float, float, float, float]]]:
        """Compare one archived forecast against actual prices.

        Returns ``(ForecastAccuracy, horizon_data)`` where *horizon_data* is a
        list of ``(horizon_hours, abs_error, actual, predicted)`` tuples.
        """
        errors: list[float] = []
        horizon_data: list[tuple[float, float, float, float]] = []
        agreements: list[float] = []
        directions_correct = 0
        directions_total = 0

        # Build a fast lookup: UTC timestamp → actual price
        price_col = "Price_EUR_MWh"
        if price_col not in actual_df.columns:
            return None, []

        prev_actual: float | None = None
        prev_predicted: float | None = None

        for pt in forecast.points:
            ts = pt.timestamp
            ts = ts.tz_localize("UTC") if ts.tz is None else ts.tz_convert("UTC")

            if ts not in actual_df.index:
                continue

            actual = actual_df.loc[ts, price_col]
            if pd.isna(actual):
                continue

            actual = float(actual)
            pred = float(pt.prediction)
            error = abs(pred - actual)
            errors.append(error)

            horizon_h = (ts - forecast.created_at).total_seconds() / 3600
            horizon_data.append((horizon_h, error, actual, pred))

            if 0.0 < pt.model_agreement <= 1.0:
                agreements.append(float(pt.model_agreement))

            # Directional accuracy vs previous hour
            if prev_actual is not None and prev_predicted is not None:
                actual_dir = actual - prev_actual
                pred_dir = pred - prev_predicted
                if (actual_dir >= 0) == (pred_dir >= 0):
                    directions_correct += 1
                directions_total += 1

            prev_actual = actual
            prev_predicted = pred

        if not errors:
            return None, []

        n = len(errors)
        mae = sum(errors) / n
        rmse = math.sqrt(sum(e**2 for e in errors) / n)

        nonzero = [(d[3], d[2]) for d in horizon_data if d[2] != 0.0]
        mape: float | None = (
            sum(abs((pred - act) / act) for pred, act in nonzero) / len(nonzero) * 100
            if nonzero
            else None
        )

        return ForecastAccuracy(
            forecast_created_at=forecast.created_at,
            model_version=forecast.model_version,
            n_comparable_hours=n,
            mae=mae,
            rmse=rmse,
            mape=mape,
            directional_accuracy=directions_correct / directions_total if directions_total else 0.0,
            mean_model_agreement=sum(agreements) / len(agreements) if agreements else 0.0,
        ), horizon_data

    # ------------------------------------------------------------------
    def _compute_horizon_metrics(
        self,
        data: list[tuple[float, float, float, float]],
    ) -> list[HorizonMetrics]:
        """Bucket ``(horizon_h, abs_error, actual, pred)`` records by horizon band."""
        buckets: dict[str, list[float]] = {label: [] for label, *_ in _HORIZON_BANDS}

        for horizon_h, error, _actual, _pred in data:
            for label, lo, hi in _HORIZON_BANDS:
                if lo < horizon_h <= hi or (lo == 0 and horizon_h <= hi):
                    buckets[label].append(error)
                    break

        result = []
        for label, _lo, _hi in _HORIZON_BANDS:
            errs = buckets[label]
            if not errs:
                continue
            n = len(errs)
            result.append(
                HorizonMetrics(
                    horizon_label=label,
                    n_hours=n,
                    mae=sum(errs) / n,
                    rmse=math.sqrt(sum(e**2 for e in errs) / n),
                )
            )
        return result

    # ------------------------------------------------------------------
    def _compute_ensemble_coverage(self, archives: Any, actuals: Any) -> float | None:
        """Return percentage of actual prices that fell within the P25–P75 ensemble band."""
        if actuals is None or not archives:
            return None

        actual_df = actuals.to_dataframe()
        if isinstance(actual_df.index, pd.DatetimeIndex):
            if actual_df.index.tz is None:
                actual_df.index = actual_df.index.tz_localize("UTC")
            else:
                actual_df.index = actual_df.index.tz_convert("UTC")

        price_col = "Price_EUR_MWh"
        if price_col not in actual_df.columns:
            return None

        inside = 0
        total = 0
        for forecast in archives:
            for pt in forecast.points:
                ts = pt.timestamp
                ts = ts.tz_localize("UTC") if ts.tz is None else ts.tz_convert("UTC")

                if ts not in actual_df.index:
                    continue
                actual = actual_df.loc[ts, price_col]
                if pd.isna(actual):
                    continue

                # Use ensemble IQR band (P25–P75)
                if pt.ensemble_p25 <= float(actual) <= pt.ensemble_p75:
                    inside += 1
                total += 1

        return (inside / total * 100) if total > 0 else None

    # ------------------------------------------------------------------
    def _load_train_history(self, zone: str) -> list[dict]:
        sidecar = self._store.load_supplemental(zone, "train_result")
        if sidecar is None or sidecar.empty:
            return []
        return sidecar.to_dict("records")
