# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2024-2026 Thomas Phil — runeflow
# See LICENSE and COMMERCIAL-LICENSE.md for licensing details.

"""
TrainService — assembles training data and fits the ensemble.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Any

import inject
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer

from runeflow.domain.training import TrainResult
from runeflow.features import GENERATION_COLUMNS_HISTORICAL_ONLY, build_pipeline
from runeflow.features.flags import enabled_flag_labels
from runeflow.models.extreme_high import ExtremeHighModel
from runeflow.models.extreme_low import ExtremeLowModel
from runeflow.models.xgboost_quantile import XGBoostQuantileModel
from runeflow.ports.store import DataStore
from runeflow.zones.config import ZoneConfig

logger = logging.getLogger(__name__)

TARGET_COLUMN = "Price_EUR_MWh"
# Validation fraction (most recent data held out for conformal calibration)
VALIDATION_FRACTION = 0.20
# Temporal gap (in days) between the training and validation windows. Without
# it the validation set is regime-adjacent to the recency-weighted training
# tail, so the validation MAE looks healthy even when the model is overfitting
# the most recent period. The gap forces a true out-of-sample measurement.
VALIDATION_GAP_DAYS = 7

# Promotion guardrail: a freshly trained model is NOT written to the store when
# its validation MAE is materially worse than the currently deployed model's
# last recorded MAE. This is a cheap safety net (validation windows differ per
# run); the ablation backtest remains the authoritative comparison.
PROMOTION_MAX_REGRESSION_PCT = 0.15

# Recency weighting parameters (timestamp-only; never derived from the target).
# These are shared between the weighting function and the training log so the
# two can never drift apart.
WEIGHT_HALF_LIFE_DAYS = 180.0
WEIGHT_RECENT_WINDOW_DAYS = 30.0
WEIGHT_RECENT_BOOST = 1.5
WEIGHT_MAX_RECENT_SHARE = 0.30


class TrainService:
    """
    Loads historical data from the store, engineers features, and trains
    the three-model ensemble (XGBoostQuantile + ExtremeHigh + ExtremeLow).
    """

    @inject.autoparams()
    def __init__(
        self,
        zone_cfg: ZoneConfig = inject.attr("zone_config"),  # type: ignore[assignment]  # noqa: B008
        store: DataStore = inject.attr(DataStore),  # type: ignore[assignment]  # noqa: B008
    ) -> None:
        self._zone_cfg = zone_cfg
        self._store = store

    # ------------------------------------------------------------------
    def run(self, *, force: bool = False) -> TrainResult:
        zone = self._zone_cfg.zone
        logger.info("TrainService starting for zone=%s", zone)

        df = self._assemble_training_frame(zone)
        df, features = self._engineer_and_select_features(df)

        logger.info(
            "Training bundle: %d features active; regime flags=%s",
            len(features),
            self._summarize_feature_flags(),
        )

        X = df[features]
        y = df[TARGET_COLUMN]

        # Chronological train / val split with a temporal gap
        n_val = max(1, int(len(df) * VALIDATION_FRACTION))
        gap_rows = int(VALIDATION_GAP_DAYS * 24)
        if n_val + gap_rows >= len(df):
            gap_rows = 0  # fall back to no gap for very short datasets
        X_train, X_val = X.iloc[: -(n_val + gap_rows)], X.iloc[-n_val:]
        y_train, y_val = y.iloc[: -(n_val + gap_rows)], y.iloc[-n_val:]

        # Recency-weighted training (timestamp-only; no target leakage)
        sample_weight = self._compute_sample_weights(y_train)
        logger.info(
            "Training recency weighting: half-life=%s days, recent-window boost<=%sx, "
            "recent-share cap=%s%% (timestamp-only)",
            WEIGHT_HALF_LIFE_DAYS,
            WEIGHT_RECENT_BOOST,
            WEIGHT_MAX_RECENT_SHARE * 100,
        )

        # Impute NaN features
        imputer = SimpleImputer(strategy="mean")
        X_train_imp = pd.DataFrame(
            imputer.fit_transform(X_train), columns=features, index=X_train.index
        )
        X_val_imp = pd.DataFrame(imputer.transform(X_val), columns=features, index=X_val.index)

        # Train three models
        xgb_model = XGBoostQuantileModel()
        ext_high = ExtremeHighModel()
        ext_low = ExtremeLowModel()

        xgb_metrics = xgb_model.train(X_train_imp, y_train, X_val_imp, y_val, sample_weight)
        ext_high_metrics = ext_high.train(X_train_imp, y_train, X_val_imp, y_val, sample_weight)
        ext_low_metrics = ext_low.train(X_train_imp, y_train, X_val_imp, y_val, sample_weight)

        metrics: dict[str, Any] = {
            "xgboost_quantile": xgb_metrics,
            "extreme_high": ext_high_metrics,
            "extreme_low": ext_low_metrics,
        }

        trained_at = datetime.now(UTC)
        model_version = trained_at.strftime("%Y%m%d%H%M")

        quality = self._assess_quality(metrics, n_train=len(X_train))
        promotion = self._check_promotion(xgb_metrics, zone, force=force)
        quality.update(promotion)

        if promotion["promoted"]:
            # Save all three models + artifacts
            for model in (xgb_model, ext_high, ext_low):
                model.save(self._store, zone)

            import json
            import pickle

            self._store.save_model(pickle.dumps(imputer), zone, "imputer")
            self._store.save_model(json.dumps(features).encode(), zone, "features")
            self._store.save_model(model_version.encode(), zone, "model_version")

            # Save training metrics sidecar for ExportQualityService
            xgb = metrics.get("xgboost_quantile", {})
            sidecar_df = pd.DataFrame(
                [
                    {
                        "model_version": model_version,
                        "mae": xgb.get("mae", float("nan")),
                        "r2": xgb.get("r2", float("nan")),
                        "coverage": xgb.get("coverage", float("nan")),
                        "trained_at": trained_at.isoformat(),
                    }
                ]
            )
            self._store.save_supplemental(sidecar_df, zone, "train_result")
            logger.info(
                "Training complete: MAE=%.4f  R²=%.4f  coverage=%.1f%%",
                xgb_metrics.get("mae", float("nan")),
                xgb_metrics.get("r2", float("nan")),
                xgb_metrics.get("coverage", float("nan")),
            )
        else:
            logger.warning(
                "Promotion rejected for zone=%s: %s",
                zone,
                promotion["promotion_reason"],
            )

        data_range = (
            pd.Timestamp(y.index.min()),
            pd.Timestamp(y.index.max()),
        )

        return TrainResult(
            zone=zone,
            features=tuple(features),
            metrics=metrics,
            quality_assessment=quality,
            trained_at=trained_at,  # type: ignore[arg-type]
            model_version=model_version,
            data_range=data_range,
        )

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

        # Commodity prices (oil, gas, coal)
        commodity = self._store.load_supplemental(zone, "commodity")
        if commodity is not None and not commodity.empty:
            idx = pd.DatetimeIndex(commodity.index)
            commodity.index = idx.tz_localize("UTC") if idx.tz is None else idx.tz_convert("UTC")
            new_cols = [c for c in commodity.columns if c not in df.columns]
            if new_cols:
                df = df.join(commodity[new_cols], how="left")

        df.sort_index(inplace=True)
        df = df[~df.index.duplicated(keep="first")]
        return df

    def _engineer_and_select_features(self, df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
        pipeline = build_pipeline(self._zone_cfg)
        df = pipeline.transform(df, self._zone_cfg)

        # Drop rows with missing target
        df = df.dropna(subset=[TARGET_COLUMN])
        df = df.dropna(thresh=4)

        # Remove all-NaN columns
        df = df.dropna(axis=1, how="all")

        # Remove historical-only generation columns
        hist_cols = [
            c for c in df.columns if any(p in c.lower() for p in GENERATION_COLUMNS_HISTORICAL_ONLY)
        ]
        df = df.drop(columns=hist_cols, errors="ignore")

        # Remove legacy diesel columns (the US GASDESW proxy was dropped in the
        # PR #18 post-mortem). These may still appear in older commodity caches
        # and must never leak into the feature set.
        diesel_cols = [c for c in df.columns if c.startswith("commodity_diesel")]
        df = df.drop(columns=diesel_cols, errors="ignore")

        # Deduplicate
        df = df.groupby(df.index).mean()

        features = [c for c in df.columns if c != TARGET_COLUMN]
        return df, features

    @staticmethod
    def _summarize_feature_flags() -> str:
        enabled = enabled_flag_labels()
        return ", ".join(enabled) if enabled else "none"

    @staticmethod
    def _compute_sample_weights(y_train: pd.Series) -> pd.Series:
        """Timestamp-only recency weighting (no target leakage).

        Exponential decay with a long half-life plus a modest, capped boost on
        the most recent window so the model tracks the latest regime without
        overfitting it. Weights depend only on the index — never on the target
        values — so they cannot bias the model toward elevated prices.

        Guardrails:
        * half-life ``WEIGHT_HALF_LIFE_DAYS`` (restores the pre-PR-#18 scheme)
        * recent-window boost capped at ``WEIGHT_RECENT_BOOST``
        * the recent window never exceeds ``WEIGHT_MAX_RECENT_SHARE`` of total
          weight (protects against very short training histories)
        """
        latest_ts = y_train.index.max()
        days_ago = (latest_ts - y_train.index).total_seconds().to_numpy() / 86_400.0

        base = np.exp(-np.log(2) * days_ago / WEIGHT_HALF_LIFE_DAYS)
        recent_mask = (days_ago <= WEIGHT_RECENT_WINDOW_DAYS).astype(float)

        weights = base * (1.0 + (WEIGHT_RECENT_BOOST - 1.0) * recent_mask)

        # Cap the recent window's share of total weight.
        base_recent = float((base * recent_mask).sum())
        base_other = float((base * (1.0 - recent_mask)).sum())
        if base_recent > 0.0:
            share = (base_recent * WEIGHT_RECENT_BOOST) / (
                base_recent * WEIGHT_RECENT_BOOST + base_other
            )
            if share > WEIGHT_MAX_RECENT_SHARE and base_other > 0.0:
                # Solve for the effective boost k that hits the cap:
                #   base_recent * k / (base_recent * k + base_other) == cap
                #   => k = cap * base_other / (base_recent * (1 - cap))
                k = (WEIGHT_MAX_RECENT_SHARE * base_other) / (
                    base_recent * (1.0 - WEIGHT_MAX_RECENT_SHARE)
                )
                k = max(1.0, k)
                weights = base * (1.0 + (k - 1.0) * recent_mask)

        weights = weights / weights.mean()  # normalise mean → 1
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

    # ------------------------------------------------------------------
    def _check_promotion(
        self,
        xgb_metrics: dict[str, Any],
        zone: str,
        *,
        force: bool = False,
    ) -> dict[str, Any]:
        """Compare the new model against the currently deployed model.

        Rejects promotion when the new validation MAE is materially worse than
        the deployed model's last recorded MAE. ``force`` bypasses the check
        (used for manual overrides / rollbacks). This is a cheap guardrail; the
        ablation backtest (purged holdout) remains the authoritative comparison.
        """
        if force:
            return {"promoted": True, "promotion_reason": "forced"}

        current_mae = self._current_deployed_mae(zone)
        new_mae = xgb_metrics.get("mae")
        if current_mae is None or new_mae is None:
            return {"promoted": True, "promotion_reason": "no_comparable_baseline"}

        threshold = current_mae * (1.0 + PROMOTION_MAX_REGRESSION_PCT)
        if new_mae > threshold:
            return {
                "promoted": False,
                "promotion_reason": (
                    f"new_mae={new_mae:.2f} worse than current_mae={current_mae:.2f} "
                    f"by >{PROMOTION_MAX_REGRESSION_PCT:.0%}"
                ),
            }
        return {
            "promoted": True,
            "promotion_reason": f"new_mae={new_mae:.2f} <= current_mae={current_mae:.2f}",
        }

    def _current_deployed_mae(self, zone: str) -> float | None:
        """Return the last recorded validation MAE of the deployed model, if any."""
        sidecar = self._store.load_supplemental(zone, "train_result")
        if sidecar is None or sidecar.empty:
            return None
        if "mae" not in sidecar.columns:
            return None
        last = sidecar["mae"].dropna()
        if last.empty:
            return None
        return float(last.iloc[-1])
