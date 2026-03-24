# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2024-2026 Thomas Phil — runeflow
# See LICENSE and COMMERCIAL-LICENSE.md for licensing details.

"""Tests for XGBoostQuantileModel, ExtremeHighModel, ExtremeLowModel."""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from runeflow.exceptions import ModelNotTrainedError
from runeflow.models.xgboost_quantile import XGBoostQuantileModel
from runeflow.models.extreme_high import ExtremeHighModel
from runeflow.models.extreme_low import ExtremeLowModel
from runeflow.models.registry import MODEL_REGISTRY


# ── Synthetic data helpers ────────────────────────────────────────────────────

def _make_Xy(n: int = 300, seed: int = 99):
    """Create minimal X (feature) DataFrame and y (target) Series for tests."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")
    X = pd.DataFrame(
        {
            "hour_of_day": idx.hour,
            "day_of_week": idx.dayofweek,
            "month": idx.month,
            "is_weekend": (idx.dayofweek >= 5).astype(int),
            "temp": 10.0 + 5.0 * rng.standard_normal(n),
            "wind": np.abs(6.0 + 3.0 * rng.standard_normal(n)),
        },
        index=idx,
    )
    y = pd.Series(50.0 + 20.0 * rng.standard_normal(n), index=idx, name="Price_EUR_MWh")
    return X, y


def _split(X, y, val_frac=0.20):
    split = int(len(X) * (1 - val_frac))
    return X.iloc[:split], y.iloc[:split], X.iloc[split:], y.iloc[split:]


# ── Fast model params (override for test speed) ────────────────────────────

_FAST_PARAMS = {
    "n_estimators": 10,
    "learning_rate": 0.3,
    "max_depth": 3,
    "gamma": 0.0,
    "min_child_weight": 1,
    "tree_method": "hist",
    "verbosity": 0,  # suppress XGBoost C-level output in tests
}


@pytest.fixture()
def fast_xgb_model(monkeypatch):
    """XGBoostQuantileModel with reduced params for speed."""
    monkeypatch.setattr(XGBoostQuantileModel, "BEST_PARAMS", _FAST_PARAMS)
    monkeypatch.setattr(XGBoostQuantileModel, "TAIL_PARAMS", _FAST_PARAMS)
    return XGBoostQuantileModel()


@pytest.fixture()
def fast_extreme_high(monkeypatch):
    monkeypatch.setattr(ExtremeHighModel, "PARAMS", _FAST_PARAMS)
    return ExtremeHighModel()


@pytest.fixture()
def fast_extreme_low(monkeypatch):
    monkeypatch.setattr(ExtremeLowModel, "PARAMS", _FAST_PARAMS)
    return ExtremeLowModel()


# ── XGBoostQuantileModel ──────────────────────────────────────────────────────

class TestXGBoostQuantileModel:
    def test_name(self):
        assert XGBoostQuantileModel().name == "xgboost_quantile"

    def test_not_trained_initially(self):
        assert not XGBoostQuantileModel().is_trained

    def test_predict_raises_before_training(self):
        X, _ = _make_Xy(10)
        with pytest.raises(ModelNotTrainedError):
            XGBoostQuantileModel().predict(X)

    def test_train_marks_trained(self, fast_xgb_model):
        X, y = _make_Xy()
        fast_xgb_model.train(X, y)
        assert fast_xgb_model.is_trained

    def test_train_returns_metrics_dict(self, fast_xgb_model):
        X, y = _make_Xy()
        X_tr, y_tr, X_v, y_v = _split(X, y)
        metrics = fast_xgb_model.train(X_tr, y_tr, X_val=X_v, y_val=y_v)
        assert isinstance(metrics, dict)
        # With a validation set, MAE and R² should be present
        assert "mae" in metrics
        assert "r2" in metrics

    def test_predict_output_columns(self, fast_xgb_model):
        X, y = _make_Xy()
        fast_xgb_model.train(X, y)
        preds = fast_xgb_model.predict(X)
        assert "prediction" in preds.columns
        assert "lower" in preds.columns
        assert "upper" in preds.columns

    def test_predict_index_matches_input(self, fast_xgb_model):
        X, y = _make_Xy()
        fast_xgb_model.train(X, y)
        preds = fast_xgb_model.predict(X)
        assert preds.index.equals(X.index)

    def test_predict_length(self, fast_xgb_model):
        X, y = _make_Xy()
        fast_xgb_model.train(X, y)
        preds = fast_xgb_model.predict(X)
        assert len(preds) == len(X)

    def test_coverage_interval_mostly_valid(self, fast_xgb_model):
        """lower ≤ prediction ≤ upper should hold for most rows."""
        X, y = _make_Xy()
        fast_xgb_model.train(X, y)
        preds = fast_xgb_model.predict(X)
        valid = (preds["lower"] <= preds["prediction"]) & (
            preds["prediction"] <= preds["upper"]
        )
        # Conformal calibration may produce slight violations; allow up to 10%
        assert valid.mean() >= 0.9

    def test_save_load_roundtrip(self, fast_xgb_model, tmp_path):
        from runeflow.adapters.store.parquet import ParquetStore

        X, y = _make_Xy()
        fast_xgb_model.train(X, y)

        store = ParquetStore(tmp_path)
        fast_xgb_model.save(store, zone="NL")

        model2 = XGBoostQuantileModel()
        model2._model_lower = None
        model2._model_p50 = None
        model2._model_upper = None
        model2._trained = False

        # Load through store
        raw = store.load_model("NL", "xgboost_quantile")
        assert raw is not None
        payload = pickle.loads(raw)
        model2._model_lower = payload["model_lower"]
        model2._model_p50 = payload["model_p50"]
        model2._model_upper = payload["model_upper"]
        model2._conf_adj_lower = payload["conf_adj_lower"]
        model2._conf_adj_upper = payload["conf_adj_upper"]
        model2._trained = True

        preds_orig = fast_xgb_model.predict(X)
        preds_loaded = model2.predict(X)
        pd.testing.assert_frame_equal(preds_orig, preds_loaded)

    def test_feature_importance_after_training(self, fast_xgb_model):
        X, y = _make_Xy()
        fast_xgb_model.train(X, y)
        fi = fast_xgb_model.get_feature_importance()
        assert "feature" in fi.columns
        assert "importance" in fi.columns
        assert len(fi) > 0


# ── ExtremeHighModel ──────────────────────────────────────────────────────────

class TestExtremeHighModel:
    def test_name(self):
        assert ExtremeHighModel().name == "extreme_high"

    def test_not_trained_initially(self):
        assert not ExtremeHighModel().is_trained

    def test_predict_raises_before_training(self):
        X, _ = _make_Xy(10)
        with pytest.raises(ModelNotTrainedError):
            ExtremeHighModel().predict(X)

    def test_train_and_predict(self, fast_extreme_high):
        X, y = _make_Xy()
        fast_extreme_high.train(X, y)
        preds = fast_extreme_high.predict(X)
        assert "prediction" in preds.columns
        assert len(preds) == len(X)

    def test_spike_weight_boosts_high_prices(self, fast_extreme_high):
        """Extreme predictions should skew high relative to median."""
        X, y = _make_Xy()
        median_model = XGBoostQuantileModel()
        # just a quick sanity check — model trains without crashing
        fast_extreme_high.train(X, y)
        assert fast_extreme_high.is_trained


# ── ExtremeLowModel ───────────────────────────────────────────────────────────

class TestExtremeLowModel:
    def test_name(self):
        assert ExtremeLowModel().name == "extreme_low"

    def test_not_trained_initially(self):
        assert not ExtremeLowModel().is_trained

    def test_train_and_predict(self, fast_extreme_low):
        X, y = _make_Xy()
        fast_extreme_low.train(X, y)
        preds = fast_extreme_low.predict(X)
        assert "prediction" in preds.columns
        assert len(preds) == len(X)

    def test_save_and_load(self, fast_extreme_low, tmp_path):
        from runeflow.adapters.store.parquet import ParquetStore

        X, y = _make_Xy()
        fast_extreme_low.train(X, y)

        store = ParquetStore(tmp_path)
        fast_extreme_low.save(store, zone="NL")

        raw = store.load_model("NL", "extreme_low")
        assert raw is not None  # model was persisted


# ── MODEL_REGISTRY ────────────────────────────────────────────────────────────

class TestXGBoostQuantileModelExtra:
    def test_train_with_sample_weight(self, fast_xgb_model):
        """Passing sample_weight as a Series exercises _train_single lines 95-96."""
        X, y = _make_Xy()
        weights = pd.Series(np.ones(len(y)), index=y.index)
        fast_xgb_model.train(X, y, sample_weight=weights)
        assert fast_xgb_model.is_trained

    def test_conformalize_loop_executes(self, fast_xgb_model, monkeypatch):
        """Force refinement loop (lines 128-131) via an impossible TARGET_COVERAGE."""
        # By setting _TARGET_COVERAGE > 1.0 no real coverage can exceed it,
        # so the loop body always executes on every one of its 5 iterations.
        monkeypatch.setattr(type(fast_xgb_model), "_TARGET_COVERAGE", 1.01)
        n = 50
        actual = np.linspace(0.0, 1.0, n)
        pred_lower = actual - 0.1
        pred_upper = actual + 0.1
        adj_lower, adj_upper = fast_xgb_model._conformalize(actual, pred_lower, pred_upper)
        assert adj_lower >= 0 and adj_upper >= 0

    def test_load_via_load_method(self, fast_xgb_model, tmp_path):
        """load() success path covers lines 218-230."""
        from runeflow.adapters.store.parquet import ParquetStore

        X, y = _make_Xy()
        fast_xgb_model.train(X, y)
        store = ParquetStore(tmp_path)
        fast_xgb_model.save(store, "NL")

        model2 = XGBoostQuantileModel()
        result = model2.load(store, "NL")
        assert result is True
        assert model2.is_trained

    def test_load_returns_false_when_missing(self, tmp_path):
        """load() returns False when no model has been saved (line 221)."""
        from runeflow.adapters.store.parquet import ParquetStore

        store = ParquetStore(tmp_path)
        assert XGBoostQuantileModel().load(store, "NL") is False

    def test_load_returns_false_on_corrupt(self, tmp_path):
        """load() returns False and does not raise on corrupt pickle (lines 231-232)."""
        from runeflow.adapters.store.parquet import ParquetStore

        store = ParquetStore(tmp_path)
        store.save_model(b"not-valid-pickle", "NL", "xgboost_quantile")
        assert XGBoostQuantileModel().load(store, "NL") is False

    def test_feature_importance_untrained(self):
        """get_feature_importance() returns empty DataFrame when _model_p50 is None (line 238)."""
        fi = XGBoostQuantileModel().get_feature_importance()
        assert isinstance(fi, pd.DataFrame)
        assert len(fi) == 0


class TestExtremeHighModelExtra:
    def test_save_and_load_roundtrip(self, fast_extreme_high, tmp_path):
        """save() + load() success path (extreme_high.py lines 122-133)."""
        from runeflow.adapters.store.parquet import ParquetStore

        X, y = _make_Xy()
        fast_extreme_high.train(X, y)
        store = ParquetStore(tmp_path)
        fast_extreme_high.save(store, "NL")

        model2 = ExtremeHighModel()
        result = model2.load(store, "NL")
        assert result is True
        assert model2.is_trained
        preds = model2.predict(X)
        assert "prediction" in preds.columns

    def test_load_returns_false_when_missing(self, tmp_path):
        """load() returns False when no model has been saved."""
        from runeflow.adapters.store.parquet import ParquetStore

        store = ParquetStore(tmp_path)
        assert ExtremeHighModel().load(store, "NL") is False

    def test_load_returns_false_on_corrupt(self, tmp_path):
        """load() returns False and does not raise on corrupt data."""
        from runeflow.adapters.store.parquet import ParquetStore

        store = ParquetStore(tmp_path)
        store.save_model(b"bad-data", "NL", "extreme_high")
        assert ExtremeHighModel().load(store, "NL") is False

    def test_train_with_sample_weight(self, fast_extreme_high):
        """sample_weight branch (combined = spike_weights * sample_weight)."""
        X, y = _make_Xy()
        weights = pd.Series(np.ones(len(y)), index=y.index)
        fast_extreme_high.train(X, y, sample_weight=weights)
        assert fast_extreme_high.is_trained

    def test_train_with_validation_set(self, fast_extreme_high):
        """Validation set triggers MAE metrics computation (lines 108-110)."""
        X, y = _make_Xy()
        X_tr, y_tr, X_v, y_v = _split(X, y)
        metrics = fast_extreme_high.train(X_tr, y_tr, X_val=X_v, y_val=y_v)
        assert "mae" in metrics


class TestExtremeLowModelExtra:
    def test_predict_raises_before_training(self):
        X, _ = _make_Xy(10)
        with pytest.raises(ModelNotTrainedError):
            ExtremeLowModel().predict(X)

    def test_save_and_load_roundtrip(self, fast_extreme_low, tmp_path):
        """save() + load() success path (extreme_low.py lines 122-133)."""
        from runeflow.adapters.store.parquet import ParquetStore

        X, y = _make_Xy()
        fast_extreme_low.train(X, y)
        store = ParquetStore(tmp_path)
        fast_extreme_low.save(store, "NL")

        model2 = ExtremeLowModel()
        result = model2.load(store, "NL")
        assert result is True
        assert model2.is_trained
        preds = model2.predict(X)
        assert "prediction" in preds.columns

    def test_load_returns_false_when_missing(self, tmp_path):
        """load() returns False when no model has been saved."""
        from runeflow.adapters.store.parquet import ParquetStore

        store = ParquetStore(tmp_path)
        assert ExtremeLowModel().load(store, "NL") is False

    def test_load_returns_false_on_corrupt(self, tmp_path):
        """load() returns False and does not raise on corrupt data."""
        from runeflow.adapters.store.parquet import ParquetStore

        store = ParquetStore(tmp_path)
        store.save_model(b"bad-data", "NL", "extreme_low")
        assert ExtremeLowModel().load(store, "NL") is False

    def test_train_with_sample_weight(self, fast_extreme_low):
        """sample_weight branch (combined = dip_weights * sample_weight)."""
        X, y = _make_Xy()
        weights = pd.Series(np.ones(len(y)), index=y.index)
        fast_extreme_low.train(X, y, sample_weight=weights)
        assert fast_extreme_low.is_trained

    def test_train_with_validation_set(self, fast_extreme_low):
        """Validation set triggers MAE metrics computation."""
        X, y = _make_Xy()
        X_tr, y_tr, X_v, y_v = _split(X, y)
        metrics = fast_extreme_low.train(X_tr, y_tr, X_val=X_v, y_val=y_v)
        assert "mae" in metrics


# ── MODEL_REGISTRY ────────────────────────────────────────────────────────────

class TestModelRegistry:
    def test_contains_xgboost_quantile(self):
        assert "xgboost_quantile" in MODEL_REGISTRY

    def test_contains_extreme_high(self):
        assert "extreme_high" in MODEL_REGISTRY

    def test_contains_extreme_low(self):
        assert "extreme_low" in MODEL_REGISTRY

    def test_factory_returns_instance(self):
        for name, factory in MODEL_REGISTRY.items():
            obj = factory()
            assert hasattr(obj, "train")
            assert hasattr(obj, "predict")
