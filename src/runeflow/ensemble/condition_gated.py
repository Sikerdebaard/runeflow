# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2024-2026 Thomas Phil — runeflow
# See LICENSE and COMMERCIAL-LICENSE.md for licensing details.

"""
Condition-gated ensemble strategy.

Combines XGBoostQuantileModel (P50 + P1/P99 intervals) with
ExtremeHigh / ExtremeLow models based on the time-of-day condition
detected from feature columns in the input DataFrame.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from runeflow.ports.ensemble import EnsembleStrategy

logger = logging.getLogger(__name__)


class ConditionGatedStrategy(EnsembleStrategy):
    """
    Ensemble blending logic:

    - **Spike condition** (evening_peak | solar_cliff): blend P50 + ExtremeHigh
    - **Dip condition** (solar_midday | night_valley): blend P50 + ExtremeLow
    - **Neutral**: P50 only

    Model agreement is used to widen or narrow the confidence interval.
    An optional *weather_uncertainty_factor* per-timestep scalar further
    widens the interval to reflect compounding autoregressive error.
    """

    def __init__(
        self,
        xgboost_weight: float = 0.70,
        extreme_weight: float = 0.30,
    ) -> None:
        self._xgb_w = xgboost_weight
        self._ext_w = extreme_weight

    # ------------------------------------------------------------------
    @property
    def name(self) -> str:
        return "condition_gated"

    # ------------------------------------------------------------------
    def combine(
        self,
        predictions: dict[str, pd.DataFrame],
        features: pd.DataFrame,
        weather_uncertainty_factor: float | np.ndarray = 1.0,
    ) -> pd.DataFrame:
        """
        Combine component model predictions.

        Parameters
        ----------
        predictions:
            Mapping of model name → DataFrame with at minimum a ``prediction``
            column.  ``xgboost_quantile`` should also have ``lower`` and ``upper``.
        features:
            The feature DataFrame used for condition detection.
        weather_uncertainty_factor:
            Scalar or 1-D array of length N to widen uncertainty intervals.

        Returns
        -------
        DataFrame with columns:
            prediction, lower, upper, uncertainty, model_agreement
        """
        xgb = predictions.get("xgboost_quantile")
        ext_high = predictions.get("extreme_high")
        ext_low = predictions.get("extreme_low")

        if xgb is None:
            raise ValueError("'xgboost_quantile' predictions are required")

        p50 = xgb["prediction"].to_numpy()
        lower = xgb["lower"].to_numpy() if "lower" in xgb.columns else p50.copy()
        upper = xgb["upper"].to_numpy() if "upper" in xgb.columns else p50.copy()

        n = len(p50)

        # ── Condition detection ─────────────────────────────────────────
        is_spike = np.zeros(n, dtype=bool)
        is_dip = np.zeros(n, dtype=bool)

        cols = features.columns
        if "is_evening_peak" in cols:
            is_spike |= features["is_evening_peak"].to_numpy().astype(bool)
        if "is_solar_cliff" in cols:
            is_spike |= features["is_solar_cliff"].to_numpy().astype(bool)
        if "is_solar_midday" in cols:
            is_dip |= features["is_solar_midday"].to_numpy().astype(bool)
        if "is_night_valley" in cols:
            is_dip |= features["is_night_valley"].to_numpy().astype(bool)

        has_condition = is_spike | is_dip

        # ── Extreme model selection ─────────────────────────────────────
        # Spike takes precedence when overlapping conditions occur
        # (e.g. hour 15 is both is_solar_midday and is_solar_cliff).
        ext_high_arr = ext_high["prediction"].to_numpy() if ext_high is not None else p50
        ext_low_arr = ext_low["prediction"].to_numpy() if ext_low is not None else p50
        ext_pred = np.where(
            is_spike, ext_high_arr,
            np.where(is_dip, ext_low_arr, p50),
        )

        # ── Model agreement ─────────────────────────────────────────────
        # Use a 3-element stack [p50, p50, ext_pred] to match the original's
        # [p50, knn, extreme] stabiliser (knn≈p50, knn_weight=0 in blend).
        stack = np.stack([p50, p50, ext_pred], axis=0)
        pred_std = np.std(stack, axis=0)
        pred_mean = np.mean(stack, axis=0)
        relative_std = pred_std / (np.abs(pred_mean) + 1e-8)
        model_agreement = np.clip(1 - relative_std * 5, 0, 1)

        # ── Adaptive weighting ─────────────────────────────────────────
        xgb_w = np.where(has_condition, self._xgb_w, 1.0)
        ext_w = np.where(has_condition, self._ext_w, 0.0)

        # Extra weight to extreme during disagreement at condition hours
        disagree_cond = (model_agreement < 0.5) & has_condition
        xgb_w = np.where(disagree_cond, 0.5, xgb_w)
        ext_w = np.where(disagree_cond, 0.5, ext_w)

        total_w = xgb_w + ext_w
        xgb_w = xgb_w / total_w
        ext_w = ext_w / total_w

        prediction = xgb_w * p50 + ext_w * ext_pred

        # ── Interval widening ───────────────────────────────────────────
        disagree_factor = 1 + 0.5 * (1 - model_agreement)
        centre = (upper + lower) / 2
        half = (upper - lower) / 2 * disagree_factor

        wu = np.asarray(weather_uncertainty_factor, dtype=float)
        half = half * wu

        lower_adj = centre - half
        upper_adj = centre + half
        uncertainty = upper_adj - lower_adj

        return pd.DataFrame(
            {
                "prediction": prediction,
                "lower": lower_adj,
                "upper": upper_adj,
                "uncertainty": uncertainty,
                "model_agreement": model_agreement,
            },
            index=xgb.index,
        )