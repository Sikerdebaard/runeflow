# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2024-2026 Thomas Phil — runeflow
# See LICENSE and COMMERCIAL-LICENSE.md for licensing details.

"""Extreme-High (spike) model — XGBoost quantile α=0.90."""
from __future__ import annotations

import logging
import pickle
from typing import Any

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error

from runeflow.exceptions import ModelNotTrainedError
from runeflow.ports.model import ModelPort
from runeflow.ports.store import DataStore

logger = logging.getLogger(__name__)


class ExtremeHighModel(ModelPort):
    """
    XGBoost quantile (α=0.90) biased toward the upper tail.

    Sample weights are scaled up for the top *extreme_percentile* %
    of prices so that the model becomes particularly good at predicting
    price spikes.
    """

    QUANTILE_ALPHA = 0.90

    PARAMS: dict[str, Any] = {
        "n_estimators": 300,
        "learning_rate": 0.05,
        "max_depth": 8,
        "gamma": 0.1,
        "min_child_weight": 3,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "tree_method": "hist",
    }

    def __init__(
        self,
        extreme_percentile: float = 12.0,
        max_weight: float = 15.0,
    ) -> None:
        self._extreme_percentile = extreme_percentile
        self._max_weight = max_weight
        self._model: xgb.XGBRegressor | None = None
        self._trained = False
        self._metrics: dict[str, Any] = {}

    @property
    def name(self) -> str:
        return "extreme_high"

    @property
    def is_trained(self) -> bool:
        return self._trained

    def _compute_weights(self, y: np.ndarray) -> np.ndarray:
        weights = np.ones(len(y))
        threshold = np.percentile(y, 100 - self._extreme_percentile)
        spike_mask = y >= threshold
        if spike_mask.any():
            top = y.max()
            weights[spike_mask] = 1 + (self._max_weight - 1) * (
                (y[spike_mask] - threshold) / (top - threshold + 1e-8)
            )
        return weights

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame | None = None,
        y_val: pd.Series | None = None,
        sample_weight: pd.Series | None = None,
    ) -> dict[str, Any]:
        logger.info("Training ExtremeHighModel (α=%.2f)…", self.QUANTILE_ALPHA)

        spike_weights = self._compute_weights(y_train.to_numpy())
        if sample_weight is not None:
            combined = spike_weights * sample_weight.to_numpy()
        else:
            combined = spike_weights

        self._model = xgb.XGBRegressor(
            objective="reg:quantileerror",
            quantile_alpha=self.QUANTILE_ALPHA,
            n_jobs=-1,
            random_state=42,
            early_stopping_rounds=30 if X_val is not None else None,
            **self.PARAMS,
        )

        fit_kw: dict[str, Any] = {"sample_weight": combined}
        if X_val is not None and len(X_val) > 0:
            self._model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False, **fit_kw)
        else:
            self._model.fit(X_train, y_train, verbose=False, **fit_kw)

        if X_val is not None and y_val is not None:
            pred = self._model.predict(X_val)
            self._metrics = {"mae": float(mean_absolute_error(y_val, pred))}
            logger.info("  ExtremeHigh MAE=%.4f", self._metrics["mae"])

        self._trained = True
        return self._metrics

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self._trained or self._model is None:
            raise ModelNotTrainedError("ExtremeHighModel not trained")
        pred = self._model.predict(X)
        return pd.DataFrame({"prediction": pred}, index=X.index)

    def save(self, store: DataStore, zone: str) -> None:
        store.save_model(pickle.dumps({"model": self._model, "metrics": self._metrics}), zone, self.name)

    def load(self, store: DataStore, zone: str) -> bool:
        raw = store.load_model(zone, self.name)
        if raw is None:
            return False
        try:
            payload = pickle.loads(raw)
            self._model = payload["model"]
            self._metrics = payload.get("metrics", {})
            self._trained = True
            return True
        except Exception:
            logger.exception("Failed to load %s", self.name)
            return False