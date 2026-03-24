# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2024-2026 Thomas Phil — runeflow
# See LICENSE and COMMERCIAL-LICENSE.md for licensing details.

"""Tests for ConditionGatedStrategy and SimpleWeightedStrategy."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from runeflow.ensemble.condition_gated import ConditionGatedStrategy
from runeflow.ensemble.simple_weighted import SimpleWeightedStrategy
from runeflow.ensemble.registry import ENSEMBLE_REGISTRY


# ── Helpers ───────────────────────────────────────────────────────────────────

def _pred_df(n: int, value: float, idx=None) -> pd.DataFrame:
    """Return a predictions DataFrame with prediction/lower/upper columns."""
    if idx is None:
        idx = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")
    return pd.DataFrame(
        {
            "prediction": value,
            "lower": value - 10.0,
            "upper": value + 10.0,
        },
        index=idx,
    )


def _feature_df(n: int, evening_peak: bool = False, solar_midday: bool = False) -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")
    return pd.DataFrame(
        {
            "is_evening_peak": int(evening_peak),
            "is_solar_cliff": 0,
            "is_solar_midday": int(solar_midday),
            "is_night_valley": 0,
        },
        index=idx,
    )


# ── ConditionGatedStrategy ────────────────────────────────────────────────────

class TestConditionGatedStrategy:
    def test_name(self):
        assert ConditionGatedStrategy().name == "condition_gated"

    def test_requires_xgboost_quantile(self):
        strategy = ConditionGatedStrategy()
        n = 5
        with pytest.raises(ValueError, match="xgboost_quantile"):
            strategy.combine(
                predictions={"extreme_high": _pred_df(n, 80.0)},
                features=_feature_df(n),
            )

    def test_returns_dataframe_with_required_cols(self):
        n = 24
        strategy = ConditionGatedStrategy()
        preds = {
            "xgboost_quantile": _pred_df(n, 55.0),
            "extreme_high": _pred_df(n, 80.0),
            "extreme_low": _pred_df(n, 20.0),
        }
        out = strategy.combine(preds, _feature_df(n))
        assert "prediction" in out.columns
        assert "lower" in out.columns
        assert "upper" in out.columns
        assert "uncertainty" in out.columns
        assert "model_agreement" in out.columns

    def test_output_length_matches_input(self):
        n = 48
        strategy = ConditionGatedStrategy()
        preds = {"xgboost_quantile": _pred_df(n, 55.0)}
        out = strategy.combine(preds, _feature_df(n))
        assert len(out) == n

    def test_index_preserved(self):
        n = 12
        idx = pd.date_range("2024-06-01", periods=n, freq="h", tz="UTC")
        strategy = ConditionGatedStrategy()
        preds = {"xgboost_quantile": _pred_df(n, 55.0, idx)}
        features = pd.DataFrame(
            {"is_evening_peak": 0, "is_solar_midday": 0,
             "is_solar_cliff": 0, "is_night_valley": 0},
            index=idx,
        )
        out = strategy.combine(preds, features)
        assert out.index.equals(idx)

    def test_neutral_conditions_use_only_xgb(self):
        """When no condition columns are present, prediction == P50."""
        n = 10
        p50 = 55.0
        strategy = ConditionGatedStrategy()
        preds = {
            "xgboost_quantile": _pred_df(n, p50),
        }
        # Empty features (no condition columns)
        features = pd.DataFrame(index=pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC"))
        out = strategy.combine(preds, features)
        # All rows neutral: prediction should equal P50
        assert np.allclose(out["prediction"].values, p50)

    def test_evening_peak_blends_extreme_high(self):
        """During evening_peak, extreme_high should influence the prediction (blend > p50)."""
        n = 10
        p50, ext_high = 55.0, 90.0
        strategy = ConditionGatedStrategy(xgboost_weight=0.70, extreme_weight=0.30)
        preds = {
            "xgboost_quantile": _pred_df(n, p50),
            "extreme_high": _pred_df(n, ext_high),
        }
        features = _feature_df(n, evening_peak=True)
        out = strategy.combine(preds, features)
        # Prediction should be blended between p50 and ext_high (exclusive)
        assert out["prediction"].mean() > p50, "Evening peak should blend toward ext_high"
        assert out["prediction"].mean() <= ext_high, "Blended prediction should not exceed ext_high"
        n = 10
        p50, ext_low = 55.0, 15.0
        strategy = ConditionGatedStrategy(xgboost_weight=0.70, extreme_weight=0.30)
        preds = {
            "xgboost_quantile": _pred_df(n, p50),
            "extreme_low": _pred_df(n, ext_low),
        }
        features = _feature_df(n, solar_midday=True)
        out = strategy.combine(preds, features)
        # Prediction should be blended toward ext_low (< p50)
        assert out["prediction"].mean() < p50, "Solar midday should blend toward ext_low"
        assert out["prediction"].mean() >= ext_low, "Blended prediction should not go below ext_low"

    def test_model_agreement_in_0_1(self):
        n = 24
        strategy = ConditionGatedStrategy()
        preds = {
            "xgboost_quantile": _pred_df(n, 55.0),
            "extreme_high": _pred_df(n, 80.0),
        }
        out = strategy.combine(preds, _feature_df(n, evening_peak=True))
        assert (out["model_agreement"] >= 0).all()
        assert (out["model_agreement"] <= 1).all()

    def test_uncertainty_positive(self):
        n = 24
        strategy = ConditionGatedStrategy()
        preds = {"xgboost_quantile": _pred_df(n, 55.0)}
        out = strategy.combine(preds, _feature_df(n))
        assert (out["uncertainty"] >= 0).all()

    def test_weather_uncertainty_factor_widens_interval(self):
        n = 24
        strategy = ConditionGatedStrategy()
        preds = {"xgboost_quantile": _pred_df(n, 55.0)}
        features = _feature_df(n)
        out_1x = strategy.combine(preds, features, weather_uncertainty_factor=1.0)
        out_2x = strategy.combine(preds, features, weather_uncertainty_factor=2.0)
        # Uncertainty should be larger with higher factor
        assert out_2x["uncertainty"].mean() >= out_1x["uncertainty"].mean()


# ── SimpleWeightedStrategy ────────────────────────────────────────────────────

class TestSimpleWeightedStrategy:
    @pytest.fixture()
    def strategy(self):
        from runeflow.ensemble.simple_weighted import SimpleWeightedStrategy
        return SimpleWeightedStrategy(
            weights={"xgboost_quantile": 0.6, "extreme_high": 0.4}
        )

    def test_name(self, strategy):
        assert strategy.name == "simple_weighted"

    def test_output_has_prediction(self, strategy):
        n = 12
        preds = {
            "xgboost_quantile": _pred_df(n, 55.0),
            "extreme_high": _pred_df(n, 80.0),
        }
        out = strategy.combine(preds, _feature_df(n))
        assert "prediction" in out.columns

    def test_weighted_blend(self, strategy):
        n = 5
        p50, ext = 60.0, 80.0
        preds = {
            "xgboost_quantile": _pred_df(n, p50),
            "extreme_high": _pred_df(n, ext),
        }
        out = strategy.combine(preds, _feature_df(n))
        expected = 0.6 * p50 + 0.4 * ext
        assert np.isclose(out["prediction"].mean(), expected, atol=0.01)

    def test_output_length(self, strategy):
        n = 30
        preds = {
            "xgboost_quantile": _pred_df(n, 55.0),
            "extreme_high": _pred_df(n, 80.0),
        }
        out = strategy.combine(preds, _feature_df(n))
        assert len(out) == n


# ── Ensemble Registry ─────────────────────────────────────────────────────────

class TestEnsembleRegistry:
    def test_contains_condition_gated(self):
        assert "condition_gated" in ENSEMBLE_REGISTRY

    def test_contains_simple_weighted(self):
        assert "simple_weighted" in ENSEMBLE_REGISTRY

    def test_factories_return_strategy(self):
        for name, factory in ENSEMBLE_REGISTRY.items():
            obj = factory()
            assert hasattr(obj, "combine")
            assert hasattr(obj, "name")
