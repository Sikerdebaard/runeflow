# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2024-2026 Thomas Phil — runeflow
# See LICENSE and COMMERCIAL-LICENSE.md for licensing details.

"""Simple fixed-weight ensemble strategy (fallback / baseline)."""
from __future__ import annotations

import pandas as pd
import numpy as np

from runeflow.ports.ensemble import EnsembleStrategy


class SimpleWeightedStrategy(EnsembleStrategy):
    """
    Weighted average of all provided predictions.

    Uses the ``prediction`` column from each model; intervals come from
    xgboost_quantile if available, otherwise derived from prediction spread.
    """

    def __init__(self, weights: dict[str, float] | None = None) -> None:
        self._weights = weights or {}

    @property
    def name(self) -> str:
        return "simple_weighted"

    def combine(
        self,
        predictions: dict[str, pd.DataFrame],
        features: pd.DataFrame,
        weather_uncertainty_factor: float | np.ndarray = 1.0,
    ) -> pd.DataFrame:
        if not predictions:
            raise ValueError("No predictions to combine")

        preds = {k: v["prediction"].to_numpy() for k, v in predictions.items()}
        names = list(preds.keys())
        weights = np.array(
            [self._weights.get(n, 1.0) for n in names], dtype=float
        )
        weights = weights / weights.sum()

        combined = sum(w * preds[n] for w, n in zip(weights, names))

        # Intervals from xgboost_quantile or spread
        if "xgboost_quantile" in predictions:
            xgb = predictions["xgboost_quantile"]
            lower = xgb["lower"].to_numpy() if "lower" in xgb.columns else combined
            upper = xgb["upper"].to_numpy() if "upper" in xgb.columns else combined
        else:
            stack = np.stack(list(preds.values()), axis=0)
            std = np.std(stack, axis=0)
            lower = combined - 1.96 * std
            upper = combined + 1.96 * std

        wu = np.asarray(weather_uncertainty_factor, dtype=float)
        centre = (upper + lower) / 2
        half = (upper - lower) / 2 * wu
        lower = centre - half
        upper = centre + half

        index = next(iter(predictions.values())).index
        return pd.DataFrame(
            {
                "prediction": combined,
                "lower": lower,
                "upper": upper,
                "uncertainty": upper - lower,
                "model_agreement": np.ones(len(combined)),
            },
            index=index,
        )