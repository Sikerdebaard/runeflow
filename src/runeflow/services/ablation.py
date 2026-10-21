# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2024-2026 Thomas Phil — runeflow
# See LICENSE and COMMERCIAL-LICENSE.md for licensing details.

"""AblationService — cheap-first E0..E7 experiment runner.

Each experiment re-trains the model **in memory** under a specific feature-flag
profile and evaluates point-forecast accuracy (MAE/RMSE/bias) on a held-out,
recent window of *actual* prices.

The runner is deliberately side-effect free with respect to the production
store: training never calls ``save_model`` and no forecast is written to the
archive. Evaluation uses a temporal holdout, not the forecast archive, so the
reported numbers measure the *experiment's* model — not the previously
deployed one.
"""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error

from runeflow.models.xgboost_quantile import XGBoostQuantileModel
from runeflow.services.train import (
    TARGET_COLUMN,
    VALIDATION_FRACTION,
    VALIDATION_GAP_DAYS,
    TrainService,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _StageConfig:
    days_back: int  # holdout window (days of actuals to score against)


_STAGE_CONFIGS: dict[str, _StageConfig] = {
    "A": _StageConfig(days_back=7),
    "B": _StageConfig(days_back=30),
    "C": _StageConfig(days_back=60),
    "D": _StageConfig(days_back=90),
}


@dataclass(frozen=True)
class _Profile:
    key: str
    description: str
    flags: dict[str, bool]


_PROFILES: dict[str, _Profile] = {
    # Legacy-ish baseline (new feature blocks disabled)
    "E0": _Profile(
        key="E0",
        description="baseline_without_new_blocks",
        flags={
            "RUNEFLOW_ENABLE_CYCLICAL_TEMPORAL": False,
            "RUNEFLOW_ENABLE_SPREAD_FEATURES": False,
            "RUNEFLOW_ENABLE_MARKET_STRESS": False,
            "RUNEFLOW_ENABLE_DATACENTER_PROXY": False,
        },
    ),
    "E1": _Profile(
        key="E1",
        description="seasonality_only",
        flags={
            "RUNEFLOW_ENABLE_CYCLICAL_TEMPORAL": True,
            "RUNEFLOW_ENABLE_SPREAD_FEATURES": False,
            "RUNEFLOW_ENABLE_MARKET_STRESS": False,
            "RUNEFLOW_ENABLE_DATACENTER_PROXY": False,
        },
    ),
    "E2": _Profile(
        key="E2",
        description="spread_only",
        flags={
            "RUNEFLOW_ENABLE_CYCLICAL_TEMPORAL": False,
            "RUNEFLOW_ENABLE_SPREAD_FEATURES": True,
            "RUNEFLOW_ENABLE_MARKET_STRESS": False,
            "RUNEFLOW_ENABLE_DATACENTER_PROXY": False,
        },
    ),
    "E3": _Profile(
        key="E3",
        description="stress_only",
        flags={
            "RUNEFLOW_ENABLE_CYCLICAL_TEMPORAL": False,
            "RUNEFLOW_ENABLE_SPREAD_FEATURES": True,
            "RUNEFLOW_ENABLE_MARKET_STRESS": True,
            "RUNEFLOW_ENABLE_DATACENTER_PROXY": False,
        },
    ),
    "E4": _Profile(
        key="E4",
        description="datacenter_only",
        flags={
            "RUNEFLOW_ENABLE_CYCLICAL_TEMPORAL": False,
            "RUNEFLOW_ENABLE_SPREAD_FEATURES": False,
            "RUNEFLOW_ENABLE_MARKET_STRESS": False,
            "RUNEFLOW_ENABLE_DATACENTER_PROXY": True,
        },
    ),
    "E5": _Profile(
        key="E5",
        description="seasonality_plus_spread",
        flags={
            "RUNEFLOW_ENABLE_CYCLICAL_TEMPORAL": True,
            "RUNEFLOW_ENABLE_SPREAD_FEATURES": True,
            "RUNEFLOW_ENABLE_MARKET_STRESS": False,
            "RUNEFLOW_ENABLE_DATACENTER_PROXY": False,
        },
    ),
    "E6": _Profile(
        key="E6",
        description="seasonality_plus_spread_plus_stress",
        flags={
            "RUNEFLOW_ENABLE_CYCLICAL_TEMPORAL": True,
            "RUNEFLOW_ENABLE_SPREAD_FEATURES": True,
            "RUNEFLOW_ENABLE_MARKET_STRESS": True,
            "RUNEFLOW_ENABLE_DATACENTER_PROXY": False,
        },
    ),
    "E7": _Profile(
        key="E7",
        description="all_new_feature_blocks_enabled",
        flags={
            "RUNEFLOW_ENABLE_CYCLICAL_TEMPORAL": True,
            "RUNEFLOW_ENABLE_SPREAD_FEATURES": True,
            "RUNEFLOW_ENABLE_MARKET_STRESS": True,
            "RUNEFLOW_ENABLE_DATACENTER_PROXY": True,
        },
    ),
}


@contextmanager
def _env_flags(flags: dict[str, bool]) -> Iterator[None]:
    old: dict[str, str | None] = {k: os.getenv(k) for k in flags}
    try:
        for key, value in flags.items():
            os.environ[key] = "1" if value else "0"
        yield
    finally:
        for key, prev in old.items():
            if prev is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = prev


class AblationService:
    """Run predefined ablation experiments under staged holdout windows."""

    @staticmethod
    def available_profiles() -> list[str]:
        return list(_PROFILES.keys())

    @staticmethod
    def available_stages() -> list[str]:
        return list(_STAGE_CONFIGS.keys())

    # ------------------------------------------------------------------
    def _assemble_features(self) -> tuple[pd.DataFrame, list[str]]:
        """Assemble + engineer the feature frame (read-only wrt the store)."""
        train_svc: TrainService = TrainService()
        zone = train_svc._zone_cfg.zone
        df = train_svc._assemble_training_frame(zone)
        return train_svc._engineer_and_select_features(df)

    # ------------------------------------------------------------------
    def _backtest(self, stage_cfg: _StageConfig) -> dict[str, Any]:
        """Train in memory on the historical frame and score a recent holdout."""
        df, features = self._assemble_features()
        X = df[features]
        y = df[TARGET_COLUMN]

        holdout_hours = stage_cfg.days_back * 24
        if len(df) <= holdout_hours + 24:
            raise RuntimeError(
                f"Not enough historical data for a {stage_cfg.days_back}-day holdout "
                f"(have {len(df)} rows)"
            )

        X_train, X_hold = X.iloc[:-holdout_hours], X.iloc[-holdout_hours:]
        y_train, y_hold = y.iloc[:-holdout_hours], y.iloc[-holdout_hours:]

        imputer = SimpleImputer(strategy="mean")
        X_train_imp = pd.DataFrame(
            imputer.fit_transform(X_train), columns=features, index=X_train.index
        )
        X_hold_imp = pd.DataFrame(imputer.transform(X_hold), columns=features, index=X_hold.index)

        # Internal train/val split (with a temporal gap) for early stopping and
        # conformal calibration — mirrors the production training path.
        n_val = max(1, int(len(X_train_imp) * VALIDATION_FRACTION))
        gap_rows = int(VALIDATION_GAP_DAYS * 24)
        if n_val + gap_rows >= len(X_train_imp):
            gap_rows = 0
        X_tr = X_train_imp.iloc[: -(n_val + gap_rows)]
        X_va = X_train_imp.iloc[-n_val:]
        y_tr = y_train.iloc[: -(n_val + gap_rows)]
        y_va = y_train.iloc[-n_val:]

        sample_weight = TrainService._compute_sample_weights(y_tr)

        model = XGBoostQuantileModel()
        train_metrics = model.train(X_tr, y_tr, X_va, y_va, sample_weight)

        preds = model.predict(X_hold_imp)
        y_pred = preds["prediction"].to_numpy(dtype=float)
        y_true = y_hold.to_numpy(dtype=float)

        mae = float(mean_absolute_error(y_true, y_pred))
        rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
        bias = float(np.mean(y_pred - y_true))

        return {
            "n_train_hours": int(len(X_train_imp)),
            "holdout_hours": holdout_hours,
            "train_val_mae": train_metrics.get("mae"),
            "train_val_r2": train_metrics.get("r2"),
            "train_val_coverage": train_metrics.get("coverage"),
            "holdout_mae": mae,
            "holdout_rmse": rmse,
            "holdout_bias": bias,
        }

    # ------------------------------------------------------------------
    def run(
        self,
        *,
        zone: str,
        stage: str,
        experiments: list[str],
        output_path: Path,
    ) -> dict[str, Any]:
        stage = stage.upper()
        if stage not in _STAGE_CONFIGS:
            expected = ", ".join(_STAGE_CONFIGS)
            raise ValueError(f"Unknown stage '{stage}'. Expected one of: {expected}")

        stage_cfg = _STAGE_CONFIGS[stage]
        now = datetime.now(UTC)

        report: dict[str, Any] = {
            "zone": zone,
            "stage": stage,
            "stage_config": {"days_back": stage_cfg.days_back},
            "started_at": now.isoformat(),
            "experiments": [],
        }

        for key in experiments:
            profile = _PROFILES.get(key)
            if profile is None:
                report["experiments"].append(
                    {"key": key, "status": "skipped", "reason": "unknown_experiment_key"}
                )
                continue

            logger.info("Ablation %s (%s): starting", profile.key, profile.description)
            exp_rec: dict[str, Any] = {
                "key": profile.key,
                "description": profile.description,
                "flags": profile.flags,
            }

            try:
                with _env_flags(profile.flags):
                    backtest = self._backtest(stage_cfg)
                exp_rec.update({"status": "ok", "backtest": backtest})
            except Exception as exc:
                logger.exception("Ablation %s failed", profile.key)
                exp_rec.update({"status": "failed", "error": str(exc)})

            report["experiments"].append(exp_rec)

        report["finished_at"] = datetime.now(UTC).isoformat()

        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
        logger.info("Ablation report written to %s", output_path)
        return report
