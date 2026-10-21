# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2024-2026 Thomas Phil — runeflow
# See LICENSE and COMMERCIAL-LICENSE.md for licensing details.

"""Deterministic model-performance regression gate.

Trains the production P50 model on a small synthetic dataset with a strong,
learnable signal and asserts that it beats a naive baseline on a purged holdout
and that its prediction bias stays bounded.

This is a smoke-level guard that catches catastrophic breakage (e.g. a broken
training/save wiring or a target-leakage weighting scheme that destroys
learning). It is **not** a substitute for the ablation backtest, which is the
authoritative comparison on real data.
"""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error

from runeflow.models.xgboost_quantile import XGBoostQuantileModel


def _make_synthetic(n: int = 1500, seed: int = 0) -> pd.DataFrame:
    """Synthetic hourly data with a strong, learnable temperature/seasonal signal."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2023-01-01", periods=n, freq="h", tz="UTC")
    hour = idx.hour.to_numpy().astype(float)
    doy = idx.dayofyear.to_numpy().astype(float)
    temperature = (
        10.0 + 8.0 * np.sin(2 * np.pi * doy / 365.25) + 3.0 * np.sin(2 * np.pi * hour / 24.0)
    )
    price = (
        40.0 + 3.0 * temperature + 12.0 * np.sin(2 * np.pi * hour / 24.0) + rng.normal(0.0, 2.0, n)
    )
    return pd.DataFrame(
        {"temperature_2m": temperature, "hour_of_day": hour, "Price_EUR_MWh": price},
        index=idx,
    )


def test_p50_model_beats_naive_baseline_and_bias_bounded() -> None:
    df = _make_synthetic()
    features = ["temperature_2m", "hour_of_day"]
    X = df[features]
    y = df["Price_EUR_MWh"]

    # Purged chronological split: hold out the most recent 15% for evaluation.
    holdout = max(1, int(len(df) * 0.15))
    X_train, X_hold = X.iloc[:-holdout], X.iloc[-holdout:]
    y_train, y_hold = y.iloc[:-holdout], y.iloc[-holdout:]

    model = XGBoostQuantileModel()
    # No early stopping: we evaluate on a separate held-out tail, not the
    # training-time validation set.
    model.train(X_train, y_train)

    preds = model.predict(X_hold)
    y_pred = preds["prediction"].to_numpy(dtype=float)
    y_true = y_hold.to_numpy(dtype=float)

    mae = float(mean_absolute_error(y_true, y_pred))
    bias = float(np.mean(y_pred - y_true))

    # Naive baseline: predict the training mean.
    naive_mae = float(mean_absolute_error(y_true, np.full(len(y_true), y_train.mean())))

    assert mae < naive_mae * 0.5, (
        f"model MAE {mae:.2f} is not meaningfully better than naive baseline {naive_mae:.2f}"
    )
    assert abs(bias) < 5.0, f"prediction bias {bias:.2f} exceeds tolerance"


def test_ablation_backtest_produces_real_metrics() -> None:
    """The ablation backtest must train a real model and score a holdout end-to-end."""
    from runeflow.services.ablation import AblationService, _StageConfig

    svc = AblationService()
    df = _make_synthetic()
    features = ["temperature_2m", "hour_of_day"]

    with patch.object(svc, "_assemble_features", return_value=(df, features)):
        result = svc._backtest(_StageConfig(days_back=7))

    assert result["holdout_hours"] == 7 * 24
    assert result["train_val_mae"] is not None
    assert np.isfinite(result["holdout_mae"])
    assert np.isfinite(result["holdout_rmse"])
    assert np.isfinite(result["holdout_bias"])
    assert result["holdout_mae"] > 0
