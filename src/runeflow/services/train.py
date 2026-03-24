# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2024-2026 Thomas Phil — runeflow
# See LICENSE and COMMERCIAL-LICENSE.md for licensing details.

"""
TrainService — assembles training data and fits the ensemble.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

import inject
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer

from runeflow.domain.training import TrainResult
from runeflow.features import build_pipeline, GENERATION_COLUMNS_HISTORICAL_ONLY
from runeflow.models.xgboost_quantile import XGBoostQuantileModel
from runeflow.models.extreme_high import ExtremeHighModel
from runeflow.models.extreme_low import ExtremeLowModel
from runeflow.ports.store import DataStore
from runeflow.zones.config import ZoneConfig

logger = logging.getLogger(__name__)

TARGET_COLUMN = "Price_EUR_MWh"
# Validation fraction (most recent data held out for conformal calibration)
VALIDATION_FRACTION = 0.20


class TrainService:
    """
    Loads historical data from the store, engineers features, and trains
    the three-model ensemble (XGBoostQuantile + ExtremeHigh + ExtremeLow).
    """

    @inject.autoparams()
    def __init__(
        self,
        zone_cfg: ZoneConfig = inject.attr("zone_config"),
        store: DataStore = inject.attr(DataStore),
    ) -> None:
        self._zone_cfg = zone_cfg
        self._store = store

    # ------------------------------------------------------------------
    def run(self) -> TrainResult:
        zone = self._zone_cfg.zone
        logger.info("TrainService starting for zone=%s", zone)

        df = self._assemble_training_frame(zone)
        df, features = self._engineer_and_select_features(df)

        X = df[features]
        y = df[TARGET_COLUMN]

        # Chronological train / val split
        n_val = max(1, int(len(df) * VALIDATION_FRACTION))
        X_train, X_val = X.iloc[:-n_val], X.iloc[-n_val:]
        y_train, y_val = y.iloc[:-n_val], y.iloc[-n_val:]

        # Recent-years sample weight (double-weight last 2 years)
        sample_weight = self._compute_sample_weights(y_train)

        # Impute NaN features
        imputer = SimpleImputer(strategy="mean")
        X_train_imp = pd.DataFrame(
            imputer.fit_transform(X_train), columns=features, index=X_train.index
        )
        X_val_imp = pd.DataFrame(
            imputer.transform(X_val), columns=features, index=X_val.index
        )

        # Train three models
        xgb_model = XGBoostQuantileModel()
        ext_high = ExtremeHighModel()
        ext_low = ExtremeLowModel()

        xgb_metrics = xgb_model.train(X_train_imp, y_train, X_val_imp, y_val, sample_weight)
        ext_high_metrics = ext_high.train(X_train_imp, y_train, X_val_imp, y_val, sample_weight)
        ext_low_metrics = ext_low.train(X_train_imp, y_train, X_val_imp, y_val, sample_weight)

        # Save all three
        for model in (xgb_model, ext_high, ext_low):
            model.save(self._store, zone)

        # Persist imputer + feature list
        import pickle, json
        self._store.save_model(pickle.dumps(imputer), zone, "imputer")
        self._store.save_model(json.dumps(features).encode(), zone, "features")

        metrics: dict[str, Any] = {
            "xgboost_quantile": xgb_metrics,
            "extreme_high": ext_high_metrics,
            "extreme_low": ext_low_metrics,
        }

        data_range = (
            pd.Timestamp(y.index.min()),
            pd.Timestamp(y.index.max()),
        )

        quality = self._assess_quality(metrics, n_train=len(X_train))

        result = TrainResult(
            zone=zone,
            features=tuple(features),
            metrics=metrics,
            quality_assessment=quality,
            trained_at=datetime.now(timezone.utc),
            model_version="1.0",
            data_range=data_range,
        )
        logger.info(
            "Training complete: MAE=%.4f  R²=%.4f  coverage=%.1f%%",
            xgb_metrics.get("mae", float("nan")),
            xgb_metrics.get("r2", float("nan")),
            xgb_metrics.get("coverage", float("nan")),
        )
        return result

    # ------------------------------------------------------------------
    def _assemble_training_frame(self, zone: str) -> pd.DataFrame:
        price_series = self._store.load_prices(zone)
        weather_series = self._store.load_weather(zone)
        if price_series is None or weather_series is None:
            raise RuntimeError(
                f"Missing price or weather data for zone={zone}. Run 'update-data' first."
            )

        df_prices = price_series.to_dataframe()
        idx = pd.DatetimeIndex(df_prices.index)
        df_prices.index = idx.tz_localize("UTC") if idx.tz is None else idx.tz_convert("UTC")
        df_prices = df_prices[["Price_EUR_MWh"]]

        df_weather = weather_series.df.astype("float64")
        idx = pd.DatetimeIndex(df_weather.index)
        df_weather.index = idx.tz_localize("UTC") if idx.tz is None else idx.tz_convert("UTC")

        df = df_weather.join(df_prices, how="left")

        # Supplemental (NED etc.)
        supp = self._store.load_supplemental(zone, "historical")
        if supp is not None and not supp.empty:
            idx = pd.DatetimeIndex(supp.index)
            supp.index = idx.tz_localize("UTC") if idx.tz is None else idx.tz_convert("UTC")
            df = df.join(supp, how="left")

        df.sort_index(inplace=True)
        df = df[~df.index.duplicated(keep="first")]
        return df

    def _engineer_and_select_features(
        self, df: pd.DataFrame
    ) -> tuple[pd.DataFrame, list[str]]:
        pipeline = build_pipeline(self._zone_cfg)
        df = pipeline.transform(df, self._zone_cfg)

        # Drop rows with missing target
        df = df.dropna(subset=[TARGET_COLUMN])
        df = df.dropna(thresh=4)

        # Remove all-NaN columns
        df = df.dropna(axis=1, how="all")

        # Remove historical-only generation columns
        hist_cols = [
            c for c in df.columns
            if any(p in c.lower() for p in GENERATION_COLUMNS_HISTORICAL_ONLY)
        ]
        df = df.drop(columns=hist_cols, errors="ignore")

        # Deduplicate
        df = df.groupby(df.index).mean()

        features = [c for c in df.columns if c != TARGET_COLUMN]
        return df, features

    @staticmethod
    def _compute_sample_weights(y_train: pd.Series) -> pd.Series:
        """Exponential-decay recency weighting (half-life 180 days).

        Matches the original runeflow weighting scheme so that older,
        lower-price data is down-weighted relative to more recent market
        conditions.  Weights are normalised so their mean ≈ 1.
        """
        HALF_LIFE_DAYS = 180
        latest_ts = y_train.index.max()
        days_ago = (latest_ts - y_train.index).total_seconds().to_numpy() / 86_400
        weights = np.exp(-np.log(2) * days_ago / HALF_LIFE_DAYS)
        weights = weights / weights.mean()          # normalise mean → 1
        return pd.Series(weights, index=y_train.index, name="sample_weight")

    @staticmethod
    def _assess_quality(metrics: dict[str, Any], n_train: int) -> dict[str, Any]:
        xgb = metrics.get("xgboost_quantile", {})
        mae = xgb.get("mae", float("nan"))
        r2 = xgb.get("r2", float("nan"))
        coverage = xgb.get("coverage", float("nan"))

        def grade(v: float, good: float, bad: float, higher_is_better: bool) -> str:
            if higher_is_better:
                return "good" if v >= good else ("ok" if v >= bad else "poor")
            return "good" if v <= good else ("ok" if v <= bad else "poor")

        return {
            "mae_grade": grade(mae, 5.0, 15.0, higher_is_better=False),
            "r2_grade": grade(r2, 0.85, 0.70, higher_is_better=True),
            "coverage_grade": grade(coverage, 93.0, 85.0, higher_is_better=True),
            "n_training_samples": n_train,
        }