# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tests for services/plot.py — helper methods and full run with mocked deps."""
from __future__ import annotations

import datetime
import pickle
from unittest.mock import MagicMock, PropertyMock, patch

import matplotlib
matplotlib.use("Agg")  # headless backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from runeflow.domain.forecast import ForecastPoint, ForecastResult


# ---------------------------------------------------------------------------
# PlotService static helpers
# ---------------------------------------------------------------------------


class TestStyleAxis:
    def test_style_axis_does_not_raise(self):
        import matplotlib.dates as mdates
        from runeflow.services.plot import _style_axis

        fig, ax = plt.subplots()
        date_fmt = mdates.DateFormatter("%a %d %b %H:%M")
        _style_axis(ax, date_fmt)
        plt.close(fig)


class TestAnnotateExtremes:
    def test_annotate_marks_peak_and_trough(self):
        from runeflow.services.plot import _annotate_extremes

        fig, ax = plt.subplots()
        idx = pd.date_range("2024-01-01", periods=5, freq="h", tz="UTC")
        df = pd.DataFrame({"price": [1.0, 3.0, 2.0, 0.5, 2.5]}, index=idx)
        _annotate_extremes(ax, df, "price", "#000000")
        # 2 annotations (peak + trough)
        assert len(ax.texts) == 2
        plt.close(fig)

    def test_annotate_empty_series_noop(self):
        from runeflow.services.plot import _annotate_extremes

        fig, ax = plt.subplots()
        df = pd.DataFrame({"price": pd.Series([], dtype=float)})
        _annotate_extremes(ax, df, "price", "#000000")
        assert len(ax.texts) == 0
        plt.close(fig)

    def test_annotate_single_value_noop(self):
        from runeflow.services.plot import _annotate_extremes

        fig, ax = plt.subplots()
        idx = pd.date_range("2024-01-01", periods=1, freq="h", tz="UTC")
        df = pd.DataFrame({"price": [5.0]}, index=idx)
        _annotate_extremes(ax, df, "price", "#000000")
        assert len(ax.texts) == 0
        plt.close(fig)


# ---------------------------------------------------------------------------
# _composite_grade
# ---------------------------------------------------------------------------

class TestCompositeGrade:
    def test_excellent_score(self):
        from runeflow.services.plot import PlotService

        train = {"xgboost": {"mae": 4.0, "r2": 0.90, "coverage": 95.0}}
        live = {
            "model_agreement": 0.8,
            "ensemble_spread": 5.0,
            "live_mae": 0.01,
        }
        score, max_score, label = PlotService._composite_grade(train, live)
        assert label == "Excellent"
        assert max_score == 10

    def test_good_score(self):
        from runeflow.services.plot import PlotService

        # Need 7-9 pts for "Good": mae<=8→2, r2>=0.85→2, cov>=93→2, agreement=0.5→1 = 7
        train = {"xgboost": {"mae": 7.0, "r2": 0.90, "coverage": 95.0}}
        live = {
            "model_agreement": 0.5,
            "ensemble_spread": 5.0,
        }
        score, max_score, label = PlotService._composite_grade(train, live)
        assert label == "Good"

    def test_poor_score(self):
        from runeflow.services.plot import PlotService

        train = {"xgboost": {"mae": 20.0, "r2": 0.2, "coverage": 50.0}}
        live = {
            "model_agreement": 0.1,
            "ensemble_spread": 30.0,
        }
        score, max_score, label = PlotService._composite_grade(train, live)
        assert label == "Poor"
        assert score <= 3.0

    def test_directional_accuracy_high(self):
        from runeflow.services.plot import PlotService

        train = {"xgboost": {"mae": 4.0, "r2": 0.90, "coverage": 95.0}}
        live = {
            "model_agreement": 0.8,
            "ensemble_spread": 5.0,
            "directional_accuracy": 70.0,
        }
        score, _, label = PlotService._composite_grade(train, live)
        assert label == "Excellent"

    def test_empty_metrics(self):
        from runeflow.services.plot import PlotService

        score, _, label = PlotService._composite_grade({}, {})
        assert label == "Poor"


# ---------------------------------------------------------------------------
# _compute_live_metrics
# ---------------------------------------------------------------------------

class TestComputeLiveMetrics:
    def test_with_actual_data_calculates_mae(self):
        from runeflow.services.plot import PlotService

        idx = pd.date_range("2024-01-01", periods=10, freq="h", tz="UTC")
        df = pd.DataFrame({
            "ensemble_p50": np.linspace(50, 60, 10),
            "lower": np.linspace(40, 50, 10),
            "upper": np.linspace(60, 70, 10),
            "model_agreement": [0.8] * 10,
        }, index=idx)

        df_actual = pd.Series(np.linspace(50, 60, 10) + 1.0, index=idx)

        forecast = MagicMock()
        forecast.ensemble_members = pd.DataFrame(
            {"m0": [50.0] * 10, "m1": [55.0] * 10}, index=idx
        )

        m = PlotService._compute_live_metrics(df, df_actual, forecast)

        assert "live_mae" in m
        assert "live_bias" in m
        assert "directional_accuracy" in m
        assert "ensemble_spread" in m
        assert "model_agreement" in m
        assert "mean_band_width" in m
        assert m["horizon_hours"] == 10

    def test_without_actual(self):
        from runeflow.services.plot import PlotService

        idx = pd.date_range("2024-01-01", periods=5, freq="h", tz="UTC")
        df = pd.DataFrame({
            "ensemble_p50": [50.0] * 5,
            "lower": [40.0] * 5,
            "upper": [60.0] * 5,
            "model_agreement": [0.9] * 5,
        }, index=idx)

        forecast = MagicMock()
        forecast.ensemble_members = None

        m = PlotService._compute_live_metrics(df, None, forecast)
        assert "live_mae" not in m
        assert m["horizon_hours"] == 5

    def test_no_ensemble_no_agreement(self):
        from runeflow.services.plot import PlotService

        idx = pd.date_range("2024-01-01", periods=3, freq="h", tz="UTC")
        df = pd.DataFrame({"x": [1, 2, 3]}, index=idx)

        forecast = MagicMock()
        forecast.ensemble_members = pd.DataFrame()

        m = PlotService._compute_live_metrics(df, None, forecast)
        assert "model_agreement" not in m
        assert m["horizon_hours"] == 3


# ---------------------------------------------------------------------------
# _load_training_metrics
# ---------------------------------------------------------------------------

class TestLoadTrainingMetrics:
    def _make_service(self):
        from runeflow.services.plot import PlotService

        svc = PlotService.__new__(PlotService)
        svc._store = MagicMock()
        svc._zone_cfg = MagicMock()
        svc._zone_cfg.zone = "NL"
        svc._price_port = MagicMock()
        return svc

    def test_loads_xgb_and_extreme_metrics(self):
        svc = self._make_service()

        xgb_payload = {"metrics": {"mae": 5.0, "r2": 0.88}}
        eh_payload = {"metrics": {"mae": 8.0}}
        el_payload = {"metrics": {"mae": 7.0}}

        svc._store.load_model.side_effect = lambda zone, name: {
            "xgboost_quantile": pickle.dumps(xgb_payload),
            "extreme_high": pickle.dumps(eh_payload),
            "extreme_low": pickle.dumps(el_payload),
        }.get(name)

        metrics = svc._load_training_metrics("NL")
        assert metrics["xgboost"]["mae"] == 5.0
        assert metrics["extreme_high"]["mae"] == 8.0
        assert metrics["extreme_low"]["mae"] == 7.0

    def test_handles_missing_models(self):
        svc = self._make_service()
        svc._store.load_model.return_value = None

        metrics = svc._load_training_metrics("NL")
        assert metrics == {}

    def test_handles_exception_in_load(self):
        svc = self._make_service()
        svc._store.load_model.side_effect = RuntimeError("corrupt")

        metrics = svc._load_training_metrics("NL")
        assert metrics == {}


# ---------------------------------------------------------------------------
# _render_scorecard
# ---------------------------------------------------------------------------

class TestRenderScorecard:
    def test_render_scorecard_all_badges(self):
        from runeflow.services.plot import PlotService

        forecast = MagicMock()
        forecast.created_at = "2024-01-01T00:00:00+00:00"
        forecast.model_version = "1.0"

        train = {"xgboost": {"mae": 5.0, "r2": 0.88, "coverage": 95.0}}
        live = {
            "model_agreement": 0.8,
            "ensemble_spread": 5.0,
            "mean_band_width": 0.01,
            "horizon_hours": 216,
            "live_mae": 0.015,
            "live_bias": -0.002,
            "n_actual_hours": 24,
            "directional_accuracy": 70.0,
        }

        for score, label in [(9.0, "Excellent"), (6.0, "Good"), (4.0, "Fair"), (1.0, "Poor")]:
            fig, ax = plt.subplots()
            PlotService._render_scorecard(ax, train, live, score, 10, label, forecast)
            plt.close(fig)

    def test_render_scorecard_extreme_model_metrics(self):
        from runeflow.services.plot import PlotService

        forecast = MagicMock()
        forecast.created_at = "2024-01-01"
        forecast.model_version = "1.0"

        train = {
            "xgboost": {"mae": 5.0},
            "extreme_high": {"mae": 8.0},
            "extreme_low": {"mae": 7.0},
        }
        live = {"horizon_hours": 48}

        fig, ax = plt.subplots()
        PlotService._render_scorecard(ax, train, live, 5.0, 10, "Good", forecast)
        plt.close(fig)


# ---------------------------------------------------------------------------
# PlotService.run() — full integration with mocked deps
# ---------------------------------------------------------------------------

class TestPlotServiceRun:
    def _make_service_and_deps(self):
        from runeflow.services.plot import PlotService
        from runeflow.zones.config import ZoneConfig
        from runeflow.domain.tariff import TariffFormula

        wholesale = TariffFormula(
            provider_id="wholesale",
            country="NL",
            label="Wholesale",
            apply=lambda price, date: price,
        )

        svc = PlotService.__new__(PlotService)
        svc._zone_cfg = MagicMock(spec=ZoneConfig)
        svc._zone_cfg.zone = "NL"
        svc._zone_cfg.timezone = "Europe/Amsterdam"
        svc._zone_cfg.tariff_formulas = {
            "wholesale": wholesale,
        }

        svc._store = MagicMock()
        svc._price_port = MagicMock()

        return svc

    def _make_forecast(self, n=48, with_ensemble=True, with_model_preds=True):
        start = pd.Timestamp.now(tz="UTC").normalize()
        idx = pd.date_range(start, periods=n, freq="h", tz="UTC")
        points = []
        for ts in idx:
            points.append(ForecastPoint(
                timestamp=ts,
                prediction=50.0,
                lower=40.0,
                upper=60.0,
                uncertainty=20.0,
                model_agreement=0.9,
                lower_static=38.0,
                upper_static=62.0,
                ensemble_p50=50.0,
                ensemble_p25=45.0,
                ensemble_p75=55.0,
            ))

        if with_ensemble:
            ens = pd.DataFrame(
                {"member_00": [50.0] * n, "member_01": [52.0] * n},
                index=idx,
            )
        else:
            ens = pd.DataFrame(index=idx)

        model_preds = {}
        if with_model_preds:
            model_preds = {
                "xgboost_p50": pd.Series([50.0] * n, index=idx),
                "xgboost_p10": pd.Series([40.0] * n, index=idx),
                "xgboost_p90": pd.Series([60.0] * n, index=idx),
                "extreme_high": pd.Series([65.0] * n, index=idx),
                "extreme_low": pd.Series([35.0] * n, index=idx),
            }

        return ForecastResult(
            zone="NL",
            points=tuple(points),
            ensemble_members=ens,
            model_predictions=model_preds,
            created_at=pd.Timestamp.now(tz="UTC"),
            model_version="1.0",
        )

    def test_run_renders_full_chart(self, tmp_path):
        svc = self._make_service_and_deps()
        forecast = self._make_forecast()
        svc._store.load_latest_forecast.return_value = forecast
        svc._store.load_model.return_value = None
        svc._price_port.download_historical.side_effect = RuntimeError("no data")

        out = tmp_path / "chart.png"
        result = svc.run(output_path=out, provider="wholesale")

        assert result == out
        assert out.exists()
        assert out.stat().st_size > 0

    def test_run_no_forecast_raises(self, tmp_path):
        svc = self._make_service_and_deps()
        svc._store.load_latest_forecast.return_value = None

        with pytest.raises(RuntimeError, match="No forecast found"):
            svc.run(output_path=tmp_path / "x.png")

    def test_run_invalid_provider_raises(self, tmp_path):
        svc = self._make_service_and_deps()
        forecast = self._make_forecast()
        svc._store.load_latest_forecast.return_value = forecast

        with pytest.raises(ValueError, match="Provider 'nonexistent' not found"):
            svc.run(output_path=tmp_path / "x.png", provider="nonexistent")

    def test_run_without_ensemble_or_models(self, tmp_path):
        svc = self._make_service_and_deps()
        forecast = self._make_forecast(with_ensemble=False, with_model_preds=False)
        svc._store.load_latest_forecast.return_value = forecast
        svc._store.load_model.return_value = None
        svc._price_port.download_historical.side_effect = RuntimeError("no data")

        out = tmp_path / "simple.png"
        result = svc.run(output_path=out, provider="wholesale")
        assert result == out
        assert out.exists()

    def test_run_with_actual_prices(self, tmp_path):
        svc = self._make_service_and_deps()
        forecast = self._make_forecast(n=48)
        svc._store.load_latest_forecast.return_value = forecast
        svc._store.load_model.return_value = None

        # Mock price_port to return actual prices
        from runeflow.domain.price import PriceSeries, PriceRecord

        start = pd.Timestamp.now(tz="UTC").normalize()
        idx = pd.date_range(start, periods=24, freq="h", tz="UTC")
        records = [PriceRecord(timestamp=ts, price_eur_mwh=50.0 + i) for i, ts in enumerate(idx)]
        price_series = PriceSeries(zone="NL", source="test", records=tuple(records),
                                   fetched_at=pd.Timestamp.now(tz="UTC"))
        svc._price_port.download_historical.return_value = price_series

        out = tmp_path / "with_actual.png"
        result = svc.run(output_path=out, provider="wholesale")
        assert result == out
        assert out.exists()

    def test_run_defaults_output_path(self, tmp_path, monkeypatch):
        """When output_path is None, defaults to forecast_nl.png in cwd."""
        import os
        monkeypatch.chdir(tmp_path)

        svc = self._make_service_and_deps()
        forecast = self._make_forecast(n=24, with_ensemble=False, with_model_preds=False)
        svc._store.load_latest_forecast.return_value = forecast
        svc._store.load_model.return_value = None
        svc._price_port.download_historical.side_effect = RuntimeError("nope")

        result = svc.run(output_path=None, provider="wholesale")
        assert result.name == "forecast_nl.png"
        assert result.resolve().exists()

    def test_run_with_tz_naive_forecast(self, tmp_path):
        """Cover tz-naive branches: lines 142, 172, 240-242."""
        from runeflow.services.plot import PlotService
        from runeflow.zones.config import ZoneConfig
        from runeflow.domain.tariff import TariffFormula
        from runeflow.domain.forecast import ForecastResult, ForecastPoint

        # Build a tz-NAIVE forecast (no tz on timestamps)
        start = pd.Timestamp.now().normalize()  # no tz
        idx = pd.date_range(start, periods=48, freq="h")
        points = tuple(
            ForecastPoint(
                timestamp=ts, prediction=50.0, lower=40.0, upper=60.0,
                uncertainty=20.0, model_agreement=0.9,
                lower_static=38.0, upper_static=62.0,
                ensemble_p50=50.0, ensemble_p25=45.0, ensemble_p75=55.0,
            )
            for ts in idx
        )
        ens = pd.DataFrame({"member_00": [50.0] * 48}, index=idx)
        forecast = ForecastResult(
            zone="NL", points=points, ensemble_members=ens,
            model_predictions={}, created_at=pd.Timestamp.now(),
            model_version="1.0",
        )

        wholesale = TariffFormula(
            provider_id="wholesale", country="NL", label="Wholesale",
            apply=lambda price, date: price,
        )
        svc = PlotService.__new__(PlotService)
        svc._zone_cfg = MagicMock(spec=ZoneConfig)
        svc._zone_cfg.zone = "NL"
        svc._zone_cfg.timezone = "Europe/Amsterdam"
        svc._zone_cfg.tariff_formulas = {"wholesale": wholesale}
        svc._store = MagicMock()
        svc._store.load_latest_forecast.return_value = forecast
        svc._store.load_model.return_value = None
        svc._price_port = MagicMock()
        svc._price_port.download_historical.side_effect = RuntimeError("no data")

        out = tmp_path / "naive.png"
        result = svc.run(output_path=out, provider="wholesale")
        assert result == out
        assert out.exists()

    def test_run_tz_naive_with_actual_prices(self, tmp_path):
        """Cover line 172: tz-naive forecast + successful actual price fetch."""
        from runeflow.services.plot import PlotService
        from runeflow.zones.config import ZoneConfig
        from runeflow.domain.tariff import TariffFormula
        from runeflow.domain.forecast import ForecastResult, ForecastPoint
        from runeflow.domain.price import PriceSeries, PriceRecord

        # tz-NAIVE forecast
        start = pd.Timestamp.now().normalize()
        idx = pd.date_range(start, periods=24, freq="h")
        points = tuple(
            ForecastPoint(
                timestamp=ts, prediction=50.0, lower=40.0, upper=60.0,
                uncertainty=20.0, model_agreement=0.9,
                lower_static=38.0, upper_static=62.0,
                ensemble_p50=50.0, ensemble_p25=45.0, ensemble_p75=55.0,
            )
            for ts in idx
        )
        forecast = ForecastResult(
            zone="NL", points=points, ensemble_members=pd.DataFrame(index=idx),
            model_predictions={}, created_at=pd.Timestamp.now(),
            model_version="1.0",
        )

        wholesale = TariffFormula(
            provider_id="wholesale", country="NL", label="Wholesale",
            apply=lambda price, date: price,
        )
        svc = PlotService.__new__(PlotService)
        svc._zone_cfg = MagicMock(spec=ZoneConfig)
        svc._zone_cfg.zone = "NL"
        svc._zone_cfg.timezone = "Europe/Amsterdam"
        svc._zone_cfg.tariff_formulas = {"wholesale": wholesale}
        svc._store = MagicMock()
        svc._store.load_latest_forecast.return_value = forecast
        svc._store.load_model.return_value = None

        # Return actual prices (tz-aware UTC) — triggers line 172 tz_localize(None)
        price_idx = pd.date_range(start, periods=12, freq="h", tz="UTC")
        records = tuple(
            PriceRecord(timestamp=ts, price_eur_mwh=50.0 + i)
            for i, ts in enumerate(price_idx)
        )
        price_series = PriceSeries(
            zone="NL", source="test", records=records,
            fetched_at=pd.Timestamp.now(tz="UTC"),
        )
        svc._price_port = MagicMock()
        svc._price_port.download_historical.return_value = price_series

        out = tmp_path / "naive_actual.png"
        result = svc.run(output_path=out, provider="wholesale")
        assert result == out
        assert out.exists()

    def test_run_with_timestamp_column_in_df(self, tmp_path):
        """Cover line 135: df has 'timestamp' column instead of DatetimeIndex."""
        svc = self._make_service_and_deps()
        forecast = self._make_forecast(n=24, with_ensemble=False, with_model_preds=False)

        # Patch to_dataframe to return a df with 'timestamp' as a column
        orig_df = forecast.to_dataframe()
        reset_df = orig_df.reset_index().rename(columns={"index": "timestamp"})
        mock_forecast = MagicMock()
        mock_forecast.to_dataframe.return_value = reset_df
        mock_forecast.model_predictions = {}
        mock_forecast.ensemble_members = pd.DataFrame()

        svc._store.load_latest_forecast.return_value = mock_forecast
        svc._store.load_model.return_value = None
        svc._price_port.download_historical.side_effect = RuntimeError("nope")

        out = tmp_path / "ts_col.png"
        result = svc.run(output_path=out, provider="wholesale")
        assert result == out
        assert out.exists()

    def test_run_date_filter_exception(self, tmp_path):
        """Cover lines 144-145: exception during date filtering."""
        svc = self._make_service_and_deps()
        forecast = self._make_forecast(n=24, with_ensemble=False, with_model_preds=False)
        svc._store.load_latest_forecast.return_value = forecast
        svc._store.load_model.return_value = None
        svc._price_port.download_historical.side_effect = RuntimeError("nope")

        # Make timezone property raise to trigger the except branch
        type(svc._zone_cfg).timezone = PropertyMock(side_effect=TypeError("bad tz"))

        out = tmp_path / "exc.png"
        result = svc.run(output_path=out, provider="wholesale")
        assert result == out

    def test_init_via_inject(self):
        """Cover lines 105-107: __init__ through inject."""
        import inject
        from runeflow.services.plot import PlotService
        from runeflow.zones.config import ZoneConfig

        zone_cfg = MagicMock(spec=ZoneConfig)
        store = MagicMock()
        price_port = MagicMock()

        def _binder(binder):
            binder.bind("zone_config", zone_cfg)
            binder.bind(ZoneConfig, zone_cfg)
            from runeflow.ports.store import DataStore
            from runeflow.ports.price import PricePort
            binder.bind(DataStore, store)
            binder.bind(PricePort, price_port)

        try:
            inject.configure(_binder, allow_override=True)
            svc = PlotService()
            assert svc._zone_cfg is zone_cfg
            assert svc._store is store
            assert svc._price_port is price_port
        finally:
            inject.clear()
