# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2024-2026 Thomas Phil — runeflow
# See LICENSE and COMMERCIAL-LICENSE.md for licensing details.

"""XGBoost quantile regression model (P1 / P50 / P99 + conformal calibration)."""
from __future__ import annotations

import logging
import pickle
from typing import Any

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, r2_score

from runeflow.exceptions import ModelNotTrainedError
from runeflow.ports.model import ModelPort
from runeflow.ports.store import DataStore

logger = logging.getLogger(__name__)


class XGBoostQuantileModel(ModelPort):
    """
    Three quantile regressors: P1 (lower), P50 (central), P99 (upper).

    After training, conformal calibration is applied on the validation set
    to achieve a target coverage (default 95 %).
    """

    _BASE_QUANTILE_LOWER = 0.01
    _BASE_QUANTILE_UPPER = 0.99
    _TARGET_COVERAGE = 0.95

    # Hyperparameters for the P50 (median) model
    BEST_PARAMS: dict[str, Any] = {
        "n_estimators": 200,
        "learning_rate": 0.05,
        "max_depth": 6,
        "gamma": 0.1,
        "min_child_weight": 3,
        "tree_method": "hist",
    }

    # Deeper/wider params for the P1 and P99 tail models
    TAIL_PARAMS: dict[str, Any] = {
        "n_estimators": 400,
        "learning_rate": 0.05,
        "max_depth": 8,
        "gamma": 0.1,
        "min_child_weight": 3,
        "tree_method": "hist",
    }

    def __init__(self) -> None:
        self._model_lower: xgb.XGBRegressor | None = None
        self._model_p50: xgb.XGBRegressor | None = None
        self._model_upper: xgb.XGBRegressor | None = None
        self._conf_adj_lower: float = 0.0
        self._conf_adj_upper: float = 0.0
        self._trained = False
        self._metrics: dict[str, Any] = {}

    # ------------------------------------------------------------------
    @property
    def name(self) -> str:
        return "xgboost_quantile"

    @property
    def is_trained(self) -> bool:
        return self._trained

    # ------------------------------------------------------------------
    def _train_single(
        self,
        alpha: float,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame | None,
        y_val: pd.Series | None,
        sample_weight: pd.Series | None,
        params: dict[str, Any],
    ) -> xgb.XGBRegressor:
        model = xgb.XGBRegressor(
            objective="reg:quantileerror",
            quantile_alpha=alpha,
            n_jobs=-1,
            random_state=42,
            early_stopping_rounds=30 if X_val is not None else None,
            **params,
        )
        fit_kw: dict[str, Any] = {}
        if sample_weight is not None:
            sw = sample_weight.to_numpy() if hasattr(sample_weight, "to_numpy") else np.asarray(sample_weight)
            fit_kw["sample_weight"] = sw

        if X_val is not None and len(X_val) > 0:
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False, **fit_kw)
        else:
            model.fit(X_train, y_train, verbose=False, **fit_kw)
        return model

    def _conformalize(
        self,
        actual: np.ndarray,
        pred_lower: np.ndarray,
        pred_upper: np.ndarray,
        safety_margin: float = 0.03,
    ) -> tuple[float, float]:
        n = len(actual)
        lower_scores = pred_lower - actual
        upper_scores = actual - pred_upper

        effective_cov = min(self._TARGET_COVERAGE + safety_margin, 0.99)
        alpha = 1 - effective_cov
        q_level = min((1 - alpha) * (1 + 1 / n), 1.0)

        adj_lower = max(0.0, float(np.quantile(lower_scores, q_level)))
        adj_upper = max(0.0, float(np.quantile(upper_scores, q_level)))

        for _ in range(5):
            covered = (
                (actual >= pred_lower - adj_lower) & (actual <= pred_upper + adj_upper)
            )
            if covered.mean() >= self._TARGET_COVERAGE:
                break
            gap = self._TARGET_COVERAGE - covered.mean()
            mult = 1.0 + gap + 0.05
            adj_lower *= mult
            adj_upper *= mult

        return adj_lower, adj_upper

    # ------------------------------------------------------------------
    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame | None = None,
        y_val: pd.Series | None = None,
        sample_weight: pd.Series | None = None,
    ) -> dict[str, Any]:
        logger.info("Training XGBoostQuantileModel (P1 / P50 / P99)…")

        self._model_lower = self._train_single(
            self._BASE_QUANTILE_LOWER, X_train, y_train, X_val, y_val, sample_weight,
            self.TAIL_PARAMS,
        )
        self._model_p50 = self._train_single(
            0.50, X_train, y_train, X_val, y_val, sample_weight, self.BEST_PARAMS
        )
        self._model_upper = self._train_single(
            self._BASE_QUANTILE_UPPER, X_train, y_train, X_val, y_val, sample_weight,
            self.TAIL_PARAMS,
        )

        if X_val is not None and y_val is not None:
            p50_pred = self._model_p50.predict(X_val)
            pred_lower = self._model_lower.predict(X_val)
            pred_upper = self._model_upper.predict(X_val)

            self._conf_adj_lower, self._conf_adj_upper = self._conformalize(
                y_val.to_numpy(), pred_lower, pred_upper
            )
            pred_lower_cal = pred_lower - self._conf_adj_lower
            pred_upper_cal = pred_upper + self._conf_adj_upper
            coverage = (
                (y_val.to_numpy() >= pred_lower_cal) & (y_val.to_numpy() <= pred_upper_cal)
            ).mean() * 100

            self._metrics = {
                "mae": float(mean_absolute_error(y_val, p50_pred)),
                "r2": float(r2_score(y_val, p50_pred)),
                "coverage": float(coverage),
                "conf_adj_lower": self._conf_adj_lower,
                "conf_adj_upper": self._conf_adj_upper,
            }
            logger.info(
                "  MAE=%.4f  R²=%.4f  Coverage=%.1f%%",
                self._metrics["mae"], self._metrics["r2"], coverage,
            )

        self._trained = True
        return self._metrics

    # ------------------------------------------------------------------
    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self._trained:
            raise ModelNotTrainedError("XGBoostQuantileModel is not trained")

        assert self._model_p50 is not None
        assert self._model_lower is not None
        assert self._model_upper is not None

        p50 = self._model_p50.predict(X)
        lower = self._model_lower.predict(X) - self._conf_adj_lower
        upper = self._model_upper.predict(X) + self._conf_adj_upper

        return pd.DataFrame(
            {"prediction": p50, "lower": lower, "upper": upper},
            index=X.index,
        )

    # ------------------------------------------------------------------
    def save(self, store: DataStore, zone: str) -> None:
        payload = {
            "model_lower": self._model_lower,
            "model_p50": self._model_p50,
            "model_upper": self._model_upper,
            "conf_adj_lower": self._conf_adj_lower,
            "conf_adj_upper": self._conf_adj_upper,
            "metrics": self._metrics,
        }
        store.save_model(pickle.dumps(payload), zone, self.name)

    def load(self, store: DataStore, zone: str) -> bool:
        raw = store.load_model(zone, self.name)
        if raw is None:
            return False
        try:
            payload = pickle.loads(raw)
            self._model_lower = payload["model_lower"]
            self._model_p50 = payload["model_p50"]
            self._model_upper = payload["model_upper"]
            self._conf_adj_lower = payload.get("conf_adj_lower", 0.0)
            self._conf_adj_upper = payload.get("conf_adj_upper", 0.0)
            self._metrics = payload.get("metrics", {})
            self._trained = True
            return True
        except Exception:
            logger.exception("Failed to load %s", self.name)
            return False

    # ------------------------------------------------------------------
    def get_feature_importance(self) -> pd.DataFrame:
        if self._model_p50 is None:
            return pd.DataFrame()
        scores = self._model_p50.get_booster().get_score(importance_type="weight")
        df = pd.DataFrame(list(scores.items()), columns=["feature", "importance"])
        return df.sort_values("importance", ascending=False)