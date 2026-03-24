# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2024-2026 Thomas Phil — runeflow
# See LICENSE and COMMERCIAL-LICENSE.md for licensing details.

"""Tests for the service layer (WarmupService, UpdateDataService,
ExportTariffsService, TrainService static helpers)."""
from __future__ import annotations

import datetime
import json
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

import numpy as np
import pandas as pd
import pytest

from runeflow.domain.price import PriceRecord, PriceSeries
from runeflow.domain.weather import WeatherSeries, WeatherLocation
from runeflow.domain.forecast import ForecastPoint, ForecastResult
from runeflow.domain.tariff import TariffFormula
from runeflow.zones.registry import ZoneRegistry


# ---------------------------------------------------------------------------
# Helpers / Fixtures
# ---------------------------------------------------------------------------

def _price_series(zone: str = "NL", n: int = 48) -> PriceSeries:
    ts = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")
    records = tuple(
        PriceRecord(timestamp=t, price_eur_mwh=float(50 + i % 20))
        for i, t in enumerate(ts)
    )
    return PriceSeries(
        zone=zone, records=records, source="test",
        fetched_at=pd.Timestamp.now("UTC"),
    )


def _weather_series(n: int = 48, zone: str = "NL") -> WeatherSeries:
    idx = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")
    df = pd.DataFrame(
        {
            "temperature_2m": np.ones(n) * 12.0,
            "wind_speed_10m": np.ones(n) * 5.0,
        },
        index=idx,
    )
    return WeatherSeries(
        locations=("de_bilt",),
        df=df,
        source="test",
        fetched_at=pd.Timestamp.now("UTC"),
    )


def _forecast_result(zone: str = "NL", n: int = 24) -> ForecastResult:
    ts = pd.date_range("2024-06-01", periods=n, freq="h", tz="UTC")
    points = tuple(
        ForecastPoint(
            timestamp=t, prediction=60.0, lower=40.0, upper=80.0,
            uncertainty=40.0, model_agreement=0.9,
        )
        for t in ts
    )
    members = pd.DataFrame(
        {f"member_{i}": [60.0] * n for i in range(3)},
        index=ts,
    )
    return ForecastResult(
        zone=zone,
        points=points,
        ensemble_members=members,
        model_predictions={},
        created_at=pd.Timestamp.now("UTC"),
        model_version="1.0",
    )


@pytest.fixture(scope="session")
def real_zone_cfg_nl():
    return ZoneRegistry.get("NL")


# ---------------------------------------------------------------------------
# Services __init__ import smoke-test
# ---------------------------------------------------------------------------

class TestServicesInit:
    def test_import_all_services(self):
        """Importing services.__init__ should expose all public service classes."""
        import runeflow.services as svcs  # noqa: F401 – covers __init__ lines

        assert hasattr(svcs, "WarmupService")
        assert hasattr(svcs, "UpdateDataService")
        assert hasattr(svcs, "TrainService")
        assert hasattr(svcs, "WarmupService")
        assert hasattr(svcs, "InferenceService")
        assert hasattr(svcs, "ExportTariffsService")
        assert hasattr(svcs, "PlotService")


# ---------------------------------------------------------------------------
# WarmupService
# ---------------------------------------------------------------------------

class TestWarmupService:
    def _make_store(self) -> MagicMock:
        store = MagicMock()
        store.load_warmup_cache.return_value = None
        store.load_prices.return_value = _price_series()
        store.load_weather.return_value = _weather_series()
        store.load_supplemental.return_value = None
        store.save_warmup_cache.return_value = None
        return store

    def test_returns_cached_when_available(self, real_zone_cfg_nl, tmp_path):
        from runeflow.services.warmup import WarmupService

        cached_df = pd.DataFrame(
            {"Price_EUR_MWh": [50.0] * 10},
            index=pd.date_range("2024-01-01", periods=10, freq="h", tz="UTC"),
        )
        store = MagicMock()
        store.load_warmup_cache.return_value = cached_df

        svc = WarmupService(zone_cfg=real_zone_cfg_nl, store=store)
        result = svc.run(force=False)

        assert result is cached_df
        store.save_warmup_cache.assert_not_called()

    def test_skips_cache_when_force(self, real_zone_cfg_nl):
        from runeflow.services.warmup import WarmupService

        store = self._make_store()
        svc = WarmupService(zone_cfg=real_zone_cfg_nl, store=store)
        result = svc.run(force=True)

        # Cache should be refreshed regardless of existing entry
        store.save_warmup_cache.assert_called_once()
        assert isinstance(result, pd.DataFrame)

    def test_builds_warmup_frame_without_supplemental(self, real_zone_cfg_nl):
        from runeflow.services.warmup import WarmupService

        store = self._make_store()
        svc = WarmupService(zone_cfg=real_zone_cfg_nl, store=store)
        df = svc.run()

        store.save_warmup_cache.assert_called_once()
        assert not df.empty

    def test_raises_when_prices_missing(self, real_zone_cfg_nl):
        from runeflow.services.warmup import WarmupService

        store = self._make_store()
        store.load_prices.return_value = None
        svc = WarmupService(zone_cfg=real_zone_cfg_nl, store=store)

        with pytest.raises(RuntimeError, match="Missing data"):
            svc.run()

    def test_raises_when_weather_missing(self, real_zone_cfg_nl):
        from runeflow.services.warmup import WarmupService

        store = self._make_store()
        store.load_weather.return_value = None
        svc = WarmupService(zone_cfg=real_zone_cfg_nl, store=store)

        with pytest.raises(RuntimeError, match="Missing data"):
            svc.run()

    def test_includes_supplemental_when_available(self, real_zone_cfg_nl):
        from runeflow.services.warmup import WarmupService

        store = self._make_store()
        supp_idx = pd.date_range("2024-01-01", periods=48, freq="h", tz="UTC")
        store.load_supplemental.return_value = pd.DataFrame(
            {"ned_utilization_kwh": np.ones(48) * 12000.0},
            index=supp_idx,
        )
        svc = WarmupService(zone_cfg=real_zone_cfg_nl, store=store)
        df = svc.run()

        assert "ned_utilization_kwh" in df.columns

    def test_empty_cached_df_triggers_rebuild(self, real_zone_cfg_nl):
        from runeflow.services.warmup import WarmupService

        store = self._make_store()
        store.load_warmup_cache.return_value = pd.DataFrame()
        svc = WarmupService(zone_cfg=real_zone_cfg_nl, store=store)
        svc.run(force=False)

        store.save_warmup_cache.assert_called_once()


# ---------------------------------------------------------------------------
# UpdateDataService
# ---------------------------------------------------------------------------

class TestUpdateDataService:
    def _make_ports(self, zone: str = "NL"):
        zone_cfg = ZoneRegistry.get(zone)

        price_port = MagicMock()
        price_port.download_historical.return_value = _price_series(zone)

        weather_series = _weather_series(zone=zone)
        weather_port = MagicMock()
        weather_port.download_historical.return_value = weather_series
        weather_port.download_forecast.return_value = weather_series
        weather_port.download_ensemble_forecast.return_value = [weather_series]

        store = MagicMock()
        store.load_prices.return_value = None  # No existing data

        validator = MagicMock()
        val_result = MagicMock()
        val_result.passed = True
        val_result.warnings = []
        validator.validate.return_value = val_result

        return zone_cfg, price_port, weather_port, store, validator

    def test_run_downloads_all_sources(self):
        from runeflow.services.update_data import UpdateDataService

        zone_cfg, price_port, weather_port, store, validator = self._make_ports()
        svc = UpdateDataService(
            zone_cfg=zone_cfg,
            price_port=price_port,
            weather_port=weather_port,
            store=store,
            validator=validator,
        )
        svc.run(years=(2024,))

        price_port.download_historical.assert_called_once()
        weather_port.download_historical.assert_called_once()
        store.save_prices.assert_called_once()
        store.save_weather.assert_called_once()

    def test_skips_price_download_when_up_to_date(self):
        from runeflow.services.update_data import UpdateDataService

        zone_cfg, price_port, weather_port, store, validator = self._make_ports()
        # Price series is up-to-date
        existing = _price_series(n=8760)
        existing_end = existing.records[-1].timestamp
        store.load_prices.return_value = existing

        # Make existing end after any requested end (2024-12-31)
        # Use a very late date so gap_start >= end
        late_records = (
            PriceRecord(
                timestamp=pd.Timestamp("2026-12-31 23:00:00", tz="UTC"),
                price_eur_mwh=50.0,
            ),
        )
        store.load_prices.return_value = PriceSeries(
            zone="NL", records=late_records, source="test",
            fetched_at=pd.Timestamp.now("UTC"),
        )

        svc = UpdateDataService(
            zone_cfg=zone_cfg,
            price_port=price_port,
            weather_port=weather_port,
            store=store,
            validator=validator,
        )
        svc.run(years=(2024,))

        price_port.download_historical.assert_not_called()

    def test_validation_warnings_do_not_raise(self):
        from runeflow.services.update_data import UpdateDataService

        zone_cfg, price_port, weather_port, store, validator = self._make_ports()
        val_result = MagicMock()
        val_result.passed = False
        val_result.warnings = ["Some warning"]
        validator.validate.return_value = val_result

        svc = UpdateDataService(
            zone_cfg=zone_cfg,
            price_port=price_port,
            weather_port=weather_port,
            store=store,
            validator=validator,
        )
        # Should complete without raising
        svc.run(years=(2024,))
        store.save_prices.assert_called_once()

    def test_generation_port_used_when_provided(self):
        from runeflow.services.update_data import UpdateDataService
        from runeflow.domain.generation import GenerationSeries

        zone_cfg, price_port, weather_port, store, validator = self._make_ports()
        gen_port = MagicMock()
        gen_series = MagicMock(spec=GenerationSeries)
        gen_port.download_generation.return_value = gen_series

        svc = UpdateDataService(
            zone_cfg=zone_cfg,
            price_port=price_port,
            weather_port=weather_port,
            store=store,
            validator=validator,
        )
        svc._generation_port = gen_port
        svc.run(years=(2024,))

        gen_port.download_generation.assert_called_once()
        store.save_generation.assert_called_once_with(gen_series)

    def test_generation_port_none_skipped(self):
        from runeflow.services.update_data import UpdateDataService

        zone_cfg, price_port, weather_port, store, validator = self._make_ports()
        svc = UpdateDataService(
            zone_cfg=zone_cfg,
            price_port=price_port,
            weather_port=weather_port,
            store=store,
            validator=validator,
        )
        svc._generation_port = None
        svc.run(years=(2024,))
        store.save_generation.assert_not_called()

    def test_supplemental_port_used_when_provided(self):
        from runeflow.services.update_data import UpdateDataService

        zone_cfg, price_port, weather_port, store, validator = self._make_ports()
        idx = pd.date_range("2024-01-01", periods=24, freq="h", tz="UTC")
        supp_df = pd.DataFrame({"ned_utilization_kwh": np.ones(24)}, index=idx)

        supp_port = MagicMock()
        supp_port.download.return_value = supp_df
        supp_port.download_forecast.return_value = supp_df

        svc = UpdateDataService(
            zone_cfg=zone_cfg,
            price_port=price_port,
            weather_port=weather_port,
            store=store,
            validator=validator,
        )
        svc._supplemental_port = supp_port
        svc.run(years=(2024,))

        supp_port.download.assert_called_once()
        assert store.save_supplemental.call_count == 2  # historical + forecast

    def test_ensemble_weather_exception_logged_not_raised(self):
        from runeflow.services.update_data import UpdateDataService

        zone_cfg, price_port, weather_port, store, validator = self._make_ports()
        weather_port.download_ensemble_forecast.side_effect = Exception("Not available")

        svc = UpdateDataService(
            zone_cfg=zone_cfg,
            price_port=price_port,
            weather_port=weather_port,
            store=store,
            validator=validator,
        )
        # Should not raise — ensemble is best-effort
        svc.run(years=(2024,))
        store.save_weather.assert_called_once()

    def test_partial_existing_prices_sets_gap_start(self):
        """_update_prices: existing data with gap triggers start=gap_start (line 83)."""
        from runeflow.services.update_data import UpdateDataService

        zone_cfg, price_port, weather_port, store, validator = self._make_ports()

        # Existing prices cover only 2024-01-01 .. 2024-01-02 (48 hours)
        partial_series = _price_series(n=48)  # ends 2024-01-02 23:00
        store.load_prices.return_value = partial_series

        svc = UpdateDataService(
            zone_cfg=zone_cfg,
            price_port=price_port,
            weather_port=weather_port,
            store=store,
            validator=validator,
        )
        # years=(2024,) → end = 2024-12-31 23:00, gap_start = 2024-01-03 00:00 < end
        svc.run(years=(2024,))

        # gap_start was set → download was called with adjusted start
        price_port.download_historical.assert_called_once()


# ---------------------------------------------------------------------------
# ExportTariffsService
# ---------------------------------------------------------------------------

class TestExportTariffsService:
    def _make_service(self, tmp_path, zone: str = "NL"):
        from runeflow.services.export_tariffs import ExportTariffsService

        zone_cfg = ZoneRegistry.get(zone)
        store = MagicMock()
        store.load_latest_forecast.return_value = _forecast_result(zone)
        # load_prices returns None (no actual-price splice)
        store.load_prices.return_value = None

        svc = ExportTariffsService(zone_cfg=zone_cfg, store=store)
        svc._price_port = None
        return svc, zone_cfg, store

    def test_run_writes_json(self, real_zone_cfg_nl, tmp_path):
        svc, _, _ = self._make_service(tmp_path)
        out = tmp_path / "tariffs.json"
        slots = svc.run(provider="vattenfall", output_path=out)

        assert out.exists()
        payload = json.loads(out.read_text())
        assert payload["type"] == "fixed"
        assert len(payload["zones"]) == 24

    def test_run_returns_slots(self, real_zone_cfg_nl, tmp_path):
        svc, _, _ = self._make_service(tmp_path)
        out = tmp_path / "t.json"
        slots = svc.run(provider="tibber", output_path=out)
        assert len(slots) == 24

    def test_run_default_output_path(self, real_zone_cfg_nl, tmp_path, monkeypatch):
        svc, _, _ = self._make_service(tmp_path)
        monkeypatch.chdir(tmp_path)
        slots = svc.run(provider="wholesale")
        # A file should be produced in cwd
        assert (tmp_path / "tariffs_nl.json").exists()

    def test_raises_when_no_forecast(self, tmp_path):
        from runeflow.services.export_tariffs import ExportTariffsService

        zone_cfg = ZoneRegistry.get("NL")
        store = MagicMock()
        store.load_latest_forecast.return_value = None
        svc = ExportTariffsService(zone_cfg=zone_cfg, store=store)
        svc._price_port = None

        with pytest.raises(RuntimeError, match="No forecast found"):
            svc.run(provider="vattenfall", output_path=tmp_path / "out.json")

    def test_raises_for_unknown_provider(self, tmp_path):
        svc, _, _ = self._make_service(tmp_path)
        with pytest.raises(ValueError, match="Provider 'no_such_provider' not found"):
            svc.run(provider="no_such_provider", output_path=tmp_path / "out.json")

    def test_slot_price_is_float(self, tmp_path):
        svc, _, _ = self._make_service(tmp_path)
        out = tmp_path / "t.json"
        slots = svc.run(provider="zonneplan", output_path=out)
        for slot in slots:
            assert isinstance(slot.price, float)

    def test_actual_prices_splice_from_store(self, tmp_path):
        """Prices loaded from store should override forecast values."""
        from runeflow.services.export_tariffs import ExportTariffsService

        zone_cfg = ZoneRegistry.get("NL")
        # Use timestamps that will survive the "yesterday" filter
        now_utc = pd.Timestamp.now("UTC").floor("h")
        ts = pd.date_range(now_utc - pd.Timedelta(hours=23), periods=24, freq="h", tz="UTC")
        forecast_points = tuple(
            ForecastPoint(
                timestamp=t, prediction=60.0, lower=40.0, upper=80.0,
                uncertainty=40.0, model_agreement=0.9,
            )
            for t in ts
        )
        forecast = ForecastResult(
            zone="NL", points=forecast_points,
            ensemble_members=pd.DataFrame(index=ts),
            model_predictions={},
            created_at=pd.Timestamp.now("UTC"),
            model_version="1.0",
        )
        actual_prices_df = pd.DataFrame({"Price_EUR_MWh": [100.0] * 24}, index=ts)
        actual_series = PriceSeries.from_dataframe(
            actual_prices_df, zone="NL", source="entsoe"
        )
        store = MagicMock()
        store.load_latest_forecast.return_value = forecast
        store.load_prices.return_value = actual_series

        svc = ExportTariffsService(zone_cfg=zone_cfg, store=store)
        svc._price_port = None
        out = tmp_path / "t.json"
        slots = svc.run(provider="wholesale", output_path=out)
        # Slots from the actual-price window should have the actual wholesale value
        # 100 EUR/MWh → 0.1 EUR/kWh wholesale
        for s in slots:
            assert abs(s.price - 0.1) < 0.01

    def test_day_ahead_prices_from_price_port(self, tmp_path):
        """If a PricePort is injected, day-ahead prices are spliced in."""
        from runeflow.services.export_tariffs import ExportTariffsService

        zone_cfg = ZoneRegistry.get("NL")
        forecast = _forecast_result(zone="NL", n=24)
        ts = pd.date_range("2024-06-01", periods=24, freq="h", tz="UTC")
        da_df = pd.DataFrame({"Price_EUR_MWh": [200.0] * 24}, index=ts)
        da_series = PriceSeries.from_dataframe(da_df, zone="NL", source="da")

        store = MagicMock()
        store.load_latest_forecast.return_value = forecast
        store.load_prices.return_value = None

        price_port = MagicMock()
        price_port.download_day_ahead.return_value = da_series

        svc = ExportTariffsService(zone_cfg=zone_cfg, store=store)
        svc._price_port = price_port
        out = tmp_path / "t.json"
        slots = svc.run(provider="wholesale", output_path=out)
        # All 24 slots should use day-ahead price 200 EUR/MWh → 0.2 EUR/kWh
        for s in slots:
            assert abs(s.price - 0.2) < 0.01

    def test_day_ahead_exception_is_swallowed(self, tmp_path):
        """If price_port.download_day_ahead raises, run still succeeds."""
        from runeflow.services.export_tariffs import ExportTariffsService

        zone_cfg = ZoneRegistry.get("NL")
        store = MagicMock()
        store.load_latest_forecast.return_value = _forecast_result("NL", 24)
        store.load_prices.return_value = None

        price_port = MagicMock()
        price_port.download_day_ahead.side_effect = Exception("API down")

        svc = ExportTariffsService(zone_cfg=zone_cfg, store=store)
        svc._price_port = price_port
        out = tmp_path / "t.json"
        # Should not raise
        slots = svc.run(provider="wholesale", output_path=out)
        assert len(slots) == 24
    def test_prices_from_store_with_tz_naive_index(self, tmp_path):
        """PriceSeries with tz-naive index hits the tz_localize branch (line 129)."""
        from runeflow.services.export_tariffs import ExportTariffsService

        zone_cfg = ZoneRegistry.get("NL")
        # Hardcoded tz-naive date range (no tz= arg → index.tz is None)
        ts_naive = pd.date_range("2024-01-01", periods=24, freq="h")
        records = tuple(
            PriceRecord(timestamp=t, price_eur_mwh=50.0)
            for t in ts_naive
        )
        naive_series = PriceSeries(
            zone="NL", records=records, source="test", fetched_at=pd.Timestamp.now("UTC")
        )

        # Forecast can be anything — we just need the service to process the store prices
        forecast = _forecast_result(zone="NL", n=24)
        store = MagicMock()
        store.load_latest_forecast.return_value = forecast
        store.load_prices.return_value = naive_series

        svc = ExportTariffsService(zone_cfg=zone_cfg, store=store)
        svc._price_port = None
        slots = svc.run(provider="wholesale", output_path=tmp_path / "t.json")
        assert len(slots) == 24

    def test_day_ahead_with_tz_naive_index(self, tmp_path):
        """Day-ahead PriceSeries with tz-naive index hits the tz_localize branch (line 146)."""
        from runeflow.services.export_tariffs import ExportTariffsService

        zone_cfg = ZoneRegistry.get("NL")
        forecast = _forecast_result(zone="NL", n=24)

        # Day-ahead data with tz-naive timestamps
        ts_naive = pd.date_range("2024-06-01", periods=24, freq="h")
        records = tuple(
            PriceRecord(timestamp=t, price_eur_mwh=200.0)
            for t in ts_naive
        )
        da_series = PriceSeries(
            zone="NL", records=records, source="da", fetched_at=pd.Timestamp.now("UTC")
        )

        store = MagicMock()
        store.load_latest_forecast.return_value = forecast
        store.load_prices.return_value = None

        price_port = MagicMock()
        price_port.download_day_ahead.return_value = da_series

        svc = ExportTariffsService(zone_cfg=zone_cfg, store=store)
        svc._price_port = price_port
        slots = svc.run(provider="wholesale", output_path=tmp_path / "t.json")
        assert len(slots) == 24


# ---------------------------------------------------------------------------
# TrainService static helpers
# ---------------------------------------------------------------------------

class TestTrainServiceHelpers:
    def test_compute_sample_weights_mean_one(self):
        from runeflow.services.train import TrainService

        idx = pd.date_range("2023-01-01", periods=365 * 24, freq="h", tz="UTC")
        y = pd.Series(np.ones(len(idx)) * 50.0, index=idx)
        weights = TrainService._compute_sample_weights(y)

        assert abs(weights.mean() - 1.0) < 0.05  # mean ≈ 1

    def test_compute_sample_weights_recent_higher(self):
        from runeflow.services.train import TrainService

        idx = pd.date_range("2022-01-01", periods=730 * 24, freq="h", tz="UTC")
        y = pd.Series(50.0, index=idx)
        weights = TrainService._compute_sample_weights(y)

        # Last 30 days should have higher weight than first 30 days
        w_recent = weights.iloc[-30 * 24:].mean()
        w_old = weights.iloc[:30 * 24].mean()
        assert w_recent > w_old

    def test_compute_sample_weights_monotone_increasing(self):
        from runeflow.services.train import TrainService

        idx = pd.date_range("2023-01-01", periods=100, freq="h", tz="UTC")
        y = pd.Series(50.0, index=idx)
        weights = TrainService._compute_sample_weights(y)

        # Weights should be monotonically non-decreasing
        assert (weights.diff().dropna() >= 0).all()

    def test_assess_quality_good_metrics(self):
        from runeflow.services.train import TrainService

        metrics = {
            "xgboost_quantile": {"mae": 3.0, "r2": 0.92, "coverage": 95.0}
        }
        result = TrainService._assess_quality(metrics, n_train=10000)
        assert result["mae_grade"] == "good"
        assert result["r2_grade"] == "good"
        assert result["coverage_grade"] == "good"

    def test_assess_quality_poor_metrics(self):
        from runeflow.services.train import TrainService

        metrics = {
            "xgboost_quantile": {"mae": 25.0, "r2": 0.2, "coverage": 60.0}
        }
        result = TrainService._assess_quality(metrics, n_train=100)
        assert result["mae_grade"] == "poor"
        assert result["r2_grade"] == "poor"
        assert result["coverage_grade"] == "poor"

    def test_assess_quality_ok_metrics(self):
        from runeflow.services.train import TrainService

        metrics = {
            "xgboost_quantile": {"mae": 7.0, "r2": 0.75, "coverage": 87.0}
        }
        result = TrainService._assess_quality(metrics, n_train=5000)
        assert result["mae_grade"] in ("ok", "good")
        assert result["r2_grade"] in ("ok", "good")

    def test_assess_quality_nan_metrics(self):
        from runeflow.services.train import TrainService

        metrics: dict = {}
        result = TrainService._assess_quality(metrics, n_train=1000)
        # Should not raise — NaN grades are handled gracefully
        assert "mae_grade" in result

    def test_assess_quality_contains_n_train(self):
        from runeflow.services.train import TrainService

        metrics = {"xgboost_quantile": {"mae": 5.0, "r2": 0.85, "coverage": 90.0}}
        result = TrainService._assess_quality(metrics, n_train=12345)
        assert result.get("n_training_samples") == 12345


class TestTrainServiceRun:
    """Test TrainService.__init__ and run() to cover train.py lines 46-180."""

    def _make_training_data(self, n: int = 2500) -> tuple:
        """Return (price_series_mock, weather_series_mock) with n rows."""
        idx = pd.date_range("2022-06-01", periods=n, freq="h", tz="UTC")

        price_df = pd.DataFrame(
            {"Price_EUR_MWh": np.random.default_rng(42).uniform(20, 150, n)},
            index=idx,
        )
        price_mock = MagicMock()
        price_mock.to_dataframe.return_value = price_df

        weather_df = pd.DataFrame(
            {
                "temperature_2m": np.random.default_rng(1).uniform(-5, 30, n),
                "shortwave_radiation": np.random.default_rng(2).uniform(0, 800, n),
                "wind_speed_10m": np.random.default_rng(3).uniform(0, 20, n),
            },
            index=idx,
        )
        weather_mock = MagicMock()
        weather_mock.df = weather_df

        return price_mock, weather_mock

    def _make_service(self, price_mock, weather_mock):
        from runeflow.services.train import TrainService
        from runeflow.zones.registry import ZoneRegistry

        zone_cfg = ZoneRegistry.get("NL")
        store = MagicMock()
        store.load_prices.return_value = price_mock
        store.load_weather.return_value = weather_mock
        store.load_supplemental.return_value = None
        store.save_model = MagicMock()

        svc = TrainService.__new__(TrainService)
        svc._zone_cfg = zone_cfg
        svc._store = store
        return svc

    def test_run_returns_train_result(self):
        """TrainService.run() with mocked models returns a TrainResult."""
        from runeflow.services.train import TrainService
        from runeflow.domain.training import TrainResult
        from runeflow.models.xgboost_quantile import XGBoostQuantileModel
        from runeflow.models.extreme_high import ExtremeHighModel
        from runeflow.models.extreme_low import ExtremeLowModel

        price_mock, weather_mock = self._make_training_data()
        svc = self._make_service(price_mock, weather_mock)

        fake_metrics = {"mae": 5.0, "r2": 0.90, "coverage": 94.0}
        with (
            patch.object(XGBoostQuantileModel, "train", return_value=fake_metrics),
            patch.object(XGBoostQuantileModel, "save"),
            patch.object(ExtremeHighModel, "train", return_value=fake_metrics),
            patch.object(ExtremeHighModel, "save"),
            patch.object(ExtremeLowModel, "train", return_value=fake_metrics),
            patch.object(ExtremeLowModel, "save"),
        ):
            result = svc.run()

        assert isinstance(result, TrainResult)
        assert result.zone == "NL"
        assert len(result.features) > 0
        assert result.metrics["xgboost_quantile"]["mae"] == 5.0

    def test_assemble_frame_with_supplemental(self):
        """_assemble_training_frame joins supplemental data when present (line 147)."""
        from runeflow.services.train import TrainService
        from runeflow.zones.registry import ZoneRegistry

        zone_cfg = ZoneRegistry.get("NL")
        n = 500
        idx = pd.date_range("2023-01-01", periods=n, freq="h", tz="UTC")

        price_df = pd.DataFrame({"Price_EUR_MWh": np.ones(n) * 50.0}, index=idx)
        price_mock = MagicMock()
        price_mock.to_dataframe.return_value = price_df

        weather_df = pd.DataFrame({"temperature_2m": np.ones(n) * 10.0}, index=idx)
        weather_mock = MagicMock()
        weather_mock.df = weather_df

        supp_df = pd.DataFrame({"solar_output_mw": np.ones(n) * 200.0}, index=idx)

        store = MagicMock()
        store.load_prices.return_value = price_mock
        store.load_weather.return_value = weather_mock
        store.load_supplemental.return_value = supp_df

        svc = TrainService.__new__(TrainService)
        svc._zone_cfg = zone_cfg
        svc._store = store

        df = svc._assemble_training_frame("NL")
        assert "solar_output_mw" in df.columns

    def test_assemble_frame_missing_prices_raises(self):
        """_assemble_training_frame raises RuntimeError when prices are None."""
        from runeflow.services.train import TrainService
        from runeflow.zones.registry import ZoneRegistry

        zone_cfg = ZoneRegistry.get("NL")
        store = MagicMock()
        store.load_prices.return_value = None
        store.load_weather.return_value = MagicMock()

        svc = TrainService.__new__(TrainService)
        svc._zone_cfg = zone_cfg
        svc._store = store

        with pytest.raises(RuntimeError, match="Missing price or weather"):
            svc._assemble_training_frame("NL")

    def test_init_via_inject(self, tmp_path):
        """TrainService.__init__ (lines 46-47) via inject.configure."""
        import inject
        from runeflow.services.train import TrainService
        from runeflow.zones.registry import ZoneRegistry
        from runeflow.zones.config import ZoneConfig
        from runeflow.ports.store import DataStore

        zone_cfg = ZoneRegistry.get("NL")
        store_mock = MagicMock(spec=DataStore)

        def _binder(binder):
            binder.bind(ZoneConfig, zone_cfg)       # class-based lookup
            binder.bind("zone_config", zone_cfg)    # string-based lookup
            binder.bind(DataStore, store_mock)

        inject.configure(_binder, allow_override=True)
        try:
            svc = TrainService()
            assert svc._zone_cfg is zone_cfg
            assert svc._store is store_mock
        finally:
            inject.clear()


# ---------------------------------------------------------------------------
# InferenceService
# ---------------------------------------------------------------------------


class TestInferenceService:
    """Tests for services/inference.py — helper methods and build_result."""

    def _make_service(self, zone=None, supplemental_port=None):
        from unittest.mock import MagicMock

        from runeflow.services.inference import InferenceService
        from runeflow.zones.config import ZoneConfig

        zone_cfg = MagicMock(spec=ZoneConfig)
        zone_cfg.zone = zone or "NL"
        zone_cfg.weather_locations = []

        store_mock = MagicMock()
        weather_mock = MagicMock()

        svc = InferenceService.__new__(InferenceService)
        svc._zone_cfg = zone_cfg
        svc._store = store_mock
        svc._weather_port = weather_mock
        svc._supplemental_port = supplemental_port
        return svc

    # ── _build_result ──────────────────────────────────────────────────────

    def test_build_result_no_members(self):
        """_build_result with no ensemble members produces ForecastResult with static bounds."""
        import pandas as pd

        from runeflow.domain.forecast import ForecastResult

        svc = self._make_service()
        ts = pd.Timestamp("2024-07-01T12:00:00", tz="UTC")
        timestamps = pd.date_range(ts, periods=3, freq="h", tz="UTC")
        det_results = {
            t: {
                "prediction": 50.0,
                "lower": 40.0,
                "upper": 60.0,
                "uncertainty": 20.0,
                "model_agreement": 0.9,
                "xgboost_p50": 50.0,
                "xgboost_p10": 40.0,
                "xgboost_p90": 60.0,
                "extreme_high": None,
                "extreme_low": None,
            }
            for t in timestamps
        }

        result = svc._build_result("NL", timestamps, det_results, member_results=[])

        assert isinstance(result, ForecastResult)
        assert len(result.points) == 3
        assert result.zone == "NL"
        # No ensemble → bounds come from det model static bounds
        assert result.points[0].lower == 40.0
        assert result.points[0].upper == 60.0

    def test_build_result_with_members(self):
        """_build_result with ensemble members uses percentile-based bounds."""
        import pandas as pd

        svc = self._make_service()
        ts = pd.Timestamp("2024-07-01T12:00:00", tz="UTC")
        timestamps = pd.date_range(ts, periods=2, freq="h", tz="UTC")
        det_results = {
            t: {
                "prediction": 50.0,
                "lower": 40.0,
                "upper": 60.0,
                "uncertainty": 20.0,
                "model_agreement": 0.9,
                "xgboost_p50": 50.0,
                "xgboost_p10": 40.0,
                "xgboost_p90": 60.0,
                "extreme_high": 65.0,
                "extreme_low": 35.0,
            }
            for t in timestamps
        }
        # 5 member results
        member_results = [
            {t: {"prediction": 50.0 + i} for t in timestamps}
            for i in range(5)
        ]

        result = svc._build_result("NL", timestamps, det_results, member_results)

        assert len(result.points) == 2
        assert result.points[0].lower < result.points[0].upper
        assert len(result.ensemble_members.columns) == 5

    def test_build_result_empty_det_results(self):
        """_build_result with empty det_results returns ForecastResult with no points."""
        import pandas as pd

        from runeflow.domain.forecast import ForecastResult

        svc = self._make_service()
        timestamps = pd.date_range("2024-07-01", periods=3, freq="h", tz="UTC")

        result = svc._build_result("NL", timestamps, det_results={}, member_results=[])

        assert isinstance(result, ForecastResult)
        assert len(result.points) == 0

    def test_build_result_extreme_model_preds_included(self):
        """model_predictions series includes extreme_high/low when not None."""
        import pandas as pd

        svc = self._make_service()
        ts = pd.Timestamp("2024-07-01T00:00:00", tz="UTC")
        timestamps = pd.date_range(ts, periods=1, freq="h", tz="UTC")
        det_results = {
            ts: {
                "prediction": 50.0,
                "lower": 40.0,
                "upper": 60.0,
                "uncertainty": 20.0,
                "model_agreement": 0.9,
                "xgboost_p50": 50.0,
                "xgboost_p10": 40.0,
                "xgboost_p90": 60.0,
                "extreme_high": 70.0,
                "extreme_low": 25.0,
            }
        }

        result = svc._build_result("NL", timestamps, det_results, member_results=[])

        assert "extreme_high" in result.model_predictions
        assert "extreme_low" in result.model_predictions
        assert "xgboost_p50" in result.model_predictions

    # ── _load_supplemental_forecast ────────────────────────────────────────

    def test_load_supplemental_no_port_returns_none(self):
        """No supplemental port → falls through to store → store returns None → None."""
        svc = self._make_service(supplemental_port=None)
        svc._store.load_supplemental.return_value = None

        result = svc._load_supplemental_forecast("NL")

        assert result is None

    def test_load_supplemental_port_returns_data(self):
        """Supplemental port returns a DataFrame → renamed and returned."""
        from unittest.mock import MagicMock

        import pandas as pd

        port = MagicMock()
        port.supports_zone.return_value = True
        idx = pd.date_range("2024-01-01", periods=3, freq="h", tz="UTC")
        port.download_forecast.return_value = pd.DataFrame(
            {"ned_forecast_kwh": [1000.0, 1100.0, 1200.0]}, index=idx
        )

        svc = self._make_service(supplemental_port=port)
        result = svc._load_supplemental_forecast("NL")

        assert result is not None
        assert "ned_utilization_kwh" in result.columns

    def test_load_supplemental_port_download_fails_falls_back_to_cache(self):
        """Port download fails → falls back to store.load_supplemental."""
        from unittest.mock import MagicMock

        import pandas as pd

        port = MagicMock()
        port.supports_zone.return_value = True
        port.download_forecast.side_effect = RuntimeError("network error")

        idx = pd.date_range("2024-01-01", periods=3, freq="h", tz="UTC")
        cached = pd.DataFrame({"ned_forecast_kwh": [900.0, 950.0, 1000.0]}, index=idx)

        svc = self._make_service(supplemental_port=port)
        svc._store.load_supplemental.return_value = cached

        result = svc._load_supplemental_forecast("NL")

        assert result is not None
        assert "ned_utilization_kwh" in result.columns

    def test_load_supplemental_cache_non_datetime_index_returns_none(self):
        """Cache with non-DateTime index logs warning and returns None."""
        import pandas as pd

        svc = self._make_service(supplemental_port=None)
        # RangeIndex, not DatetimeIndex
        cached = pd.DataFrame({"ned_forecast_kwh": [100.0, 200.0]})
        svc._store.load_supplemental.return_value = cached

        result = svc._load_supplemental_forecast("NL")

        assert result is None

    def test_load_supplemental_cache_load_exception(self):
        """store.load_supplemental raises → returns None."""
        svc = self._make_service(supplemental_port=None)
        svc._store.load_supplemental.side_effect = RuntimeError("disk error")

        result = svc._load_supplemental_forecast("NL")

        assert result is None

    def test_load_supplemental_port_returns_none_uses_cache(self):
        """Port returns None DataFrame → falls back to store cache."""
        from unittest.mock import MagicMock
        import pandas as pd

        port = MagicMock()
        port.supports_zone.return_value = True
        port.download_forecast.return_value = None

        idx = pd.date_range("2024-01-01", periods=2, freq="h", tz="UTC")
        cached = pd.DataFrame({"ned_forecast_kwh": [500.0, 600.0]}, index=idx)
        svc = self._make_service(supplemental_port=port)
        svc._store.load_supplemental.return_value = cached

        result = svc._load_supplemental_forecast("NL")

        assert result is not None

    def test_load_supplemental_port_returns_empty_df_uses_cache(self):
        """Port returns empty DataFrame → falls back to store cache."""
        from unittest.mock import MagicMock
        import pandas as pd

        port = MagicMock()
        port.supports_zone.return_value = True
        port.download_forecast.return_value = pd.DataFrame()

        idx = pd.date_range("2024-01-01", periods=2, freq="h", tz="UTC")
        cached = pd.DataFrame({"ned_forecast_kwh": [500.0, 600.0]}, index=idx)
        svc = self._make_service(supplemental_port=port)
        svc._store.load_supplemental.return_value = cached

        result = svc._load_supplemental_forecast("NL")

        assert result is not None

    def test_load_supplemental_unsupported_zone_skips_port_uses_cache(self):
        """Zone not supported by port → skip port entirely, use cache."""
        from unittest.mock import MagicMock
        import pandas as pd

        port = MagicMock()
        port.supports_zone.return_value = False

        idx = pd.date_range("2024-01-01", periods=2, freq="h", tz="UTC")
        cached = pd.DataFrame({"ned_forecast_kwh": [500.0, 600.0]}, index=idx)
        svc = self._make_service(supplemental_port=port)
        svc._store.load_supplemental.return_value = cached

        result = svc._load_supplemental_forecast("NL")

        port.download_forecast.assert_not_called()
        assert result is not None

    # ── _load_models ───────────────────────────────────────────────────────

    def test_load_models_xgb_not_found_raises(self):
        """_load_models raises RuntimeError if XGBoost model not found."""
        from unittest.mock import patch

        svc = self._make_service()
        svc._store.load_model.return_value = None

        with patch("runeflow.services.inference.XGBoostQuantileModel") as mock_xgb_cls:
            mock_xgb_cls.return_value.load.return_value = False
            with pytest.raises(RuntimeError, match="XGBoost model not found"):
                svc._load_models("NL")

    def test_load_models_success(self):
        """_load_models loads all models and returns correct tuple."""
        import json
        import pickle
        from sklearn.impute import SimpleImputer
        from unittest.mock import MagicMock, patch

        svc = self._make_service()

        real_imputer = SimpleImputer(strategy="mean")
        imputer_bytes = pickle.dumps(real_imputer)
        features_json = json.dumps(["feature_a", "feature_b"]).encode()

        svc._store.load_model.side_effect = lambda zone, name: (
            imputer_bytes if name == "imputer" else features_json if name == "features" else None
        )

        with patch("runeflow.services.inference.XGBoostQuantileModel") as mock_xgb, \
             patch("runeflow.services.inference.ExtremeHighModel") as mock_high, \
             patch("runeflow.services.inference.ExtremeLowModel") as mock_low:
            mock_xgb.return_value.load.return_value = True
            mock_high.return_value.load.return_value = True
            mock_low.return_value.load.return_value = True

            xgb, ext_high, ext_low, imputer, features = svc._load_models("NL")

        assert features == ["feature_a", "feature_b"]

    def test_load_models_no_imputer_uses_default(self):
        """If store has no imputer, a default SimpleImputer is created."""
        import json
        from unittest.mock import patch

        svc = self._make_service()

        features_json = json.dumps([]).encode()
        svc._store.load_model.side_effect = lambda zone, name: (
            None if name == "imputer" else features_json
        )

        with patch("runeflow.services.inference.XGBoostQuantileModel") as mock_xgb, \
             patch("runeflow.services.inference.ExtremeHighModel"), \
             patch("runeflow.services.inference.ExtremeLowModel"):
            mock_xgb.return_value.load.return_value = True

            xgb, ext_high, ext_low, imputer, features = svc._load_models("NL")

        from sklearn.impute import SimpleImputer
        assert isinstance(imputer, SimpleImputer)
        assert features == []

    # ── __init__ via inject ────────────────────────────────────────────────

    def test_init_via_inject_no_supplemental_port(self):
        """__init__ with inject.instance failing for SupplementalDataPort → None."""
        import inject
        from unittest.mock import MagicMock, patch

        from runeflow.services.inference import InferenceService
        from runeflow.zones.config import ZoneConfig
        from runeflow.ports.store import DataStore
        from runeflow.ports.weather import WeatherPort

        zone_cfg = MagicMock(spec=ZoneConfig)
        zone_cfg.zone = "NL"
        store_mock = MagicMock(spec=DataStore)
        weather_mock = MagicMock(spec=WeatherPort)

        def _binder(binder):
            binder.bind(ZoneConfig, zone_cfg)
            binder.bind("zone_config", zone_cfg)
            binder.bind(DataStore, store_mock)
            binder.bind(WeatherPort, weather_mock)

        inject.configure(_binder, allow_override=True)
        try:
            svc = InferenceService()
            assert svc._zone_cfg is zone_cfg
            assert svc._store is store_mock
            assert svc._supplemental_port is None  # inject.instance raises for SupplementalDataPort
        finally:
            inject.clear()


class TestRunForecastWorker:
    """Tests for the module-level _run_forecast_worker function."""

    def test_worker_single_timestamp_returns_prediction(self):
        """Worker processes a single future timestamp and returns results dict."""
        from unittest.mock import MagicMock, patch

        import numpy as np
        import pandas as pd
        from sklearn.impute import SimpleImputer

        from runeflow.services.inference import _run_forecast_worker

        ts = pd.Timestamp("2024-07-01T00:00:00", tz="UTC")
        timestamps = pd.DatetimeIndex([ts])
        weather_df = pd.DataFrame(
            {"temperature_2m": [10.0], "wind_speed_10m": [5.0]}, index=timestamps
        )

        warmup_idx = pd.date_range("2024-06-15", periods=24, freq="h", tz="UTC")
        warmup = pd.DataFrame(
            {"temperature_2m": [8.0] * 24, "Price_EUR_MWh": [50.0] * 24}, index=warmup_idx
        )

        features_list = ["temperature_2m", "wind_speed_10m"]
        imputer = SimpleImputer(strategy="mean")
        imputer.fit(pd.DataFrame({"temperature_2m": [10.0], "wind_speed_10m": [5.0]}))

        xgb = MagicMock()
        xgb._model_lower = xgb._model_p50 = xgb._model_upper = None
        xgb.predict.return_value = pd.DataFrame(
            {"prediction": [50.0], "lower": [40.0], "upper": [60.0]}, index=[ts]
        )

        ext_high = MagicMock()
        ext_high._model = None
        ext_high.is_trained = False

        ext_low = MagicMock()
        ext_low._model = None
        ext_low.is_trained = False

        zone_cfg = MagicMock()
        zone_cfg.zone = "NL"

        mock_pipeline = MagicMock()
        mock_pipeline.transform.return_value = pd.DataFrame(
            {"temperature_2m": [10.0], "wind_speed_10m": [5.0]}, index=timestamps
        )

        mock_ensemble = MagicMock()
        mock_ensemble.combine.return_value = pd.DataFrame(
            {
                "prediction": [50.0],
                "lower": [40.0],
                "upper": [60.0],
                "uncertainty": [20.0],
                "model_agreement": [0.9],
            },
            index=[ts],
        )

        with patch("runeflow.services.inference.build_pipeline", return_value=mock_pipeline):
            with patch("runeflow.services.inference.ConditionGatedStrategy", return_value=mock_ensemble):
                results = _run_forecast_worker(
                    weather_df, warmup, timestamps,
                    xgb, ext_high, ext_low, imputer, features_list,
                    zone_cfg, label="test",
                )

        assert ts in results
        assert results[ts]["prediction"] == 50.0
        assert results[ts]["xgboost_p50"] == 50.0
        assert results[ts]["extreme_high"] is None

    def test_worker_with_extreme_models(self):
        """Worker calls extreme models when they are trained."""
        from unittest.mock import MagicMock, patch

        import pandas as pd
        from sklearn.impute import SimpleImputer

        from runeflow.services.inference import _run_forecast_worker

        ts = pd.Timestamp("2024-07-01T00:00:00", tz="UTC")
        timestamps = pd.DatetimeIndex([ts])
        weather_df = pd.DataFrame({"temperature_2m": [10.0]}, index=timestamps)

        warmup_idx = pd.date_range("2024-06-15", periods=24, freq="h", tz="UTC")
        warmup = pd.DataFrame({"temperature_2m": [8.0] * 24, "Price_EUR_MWh": [50.0] * 24}, index=warmup_idx)

        imputer = SimpleImputer(strategy="mean")
        imputer.fit(pd.DataFrame({"temperature_2m": [10.0]}))

        xgb = MagicMock()
        xgb._model_lower = xgb._model_p50 = xgb._model_upper = None
        xgb.predict.return_value = pd.DataFrame({"prediction": [50.0], "lower": [40.0], "upper": [60.0]}, index=[ts])

        ext_high = MagicMock()
        ext_high._model = None
        ext_high.is_trained = True
        ext_high.predict.return_value = pd.DataFrame({"prediction": [65.0], "lower": [60.0], "upper": [70.0]}, index=[ts])

        ext_low = MagicMock()
        ext_low._model = None
        ext_low.is_trained = True
        ext_low.predict.return_value = pd.DataFrame({"prediction": [35.0], "lower": [30.0], "upper": [40.0]}, index=[ts])

        zone_cfg = MagicMock()
        mock_pipeline = MagicMock()
        mock_pipeline.transform.return_value = pd.DataFrame({"temperature_2m": [10.0]}, index=timestamps)

        mock_ensemble = MagicMock()
        mock_ensemble.combine.return_value = pd.DataFrame({
            "prediction": [50.0], "lower": [35.0], "upper": [65.0],
            "uncertainty": [30.0], "model_agreement": [0.8],
        }, index=[ts])

        with patch("runeflow.services.inference.build_pipeline", return_value=mock_pipeline):
            with patch("runeflow.services.inference.ConditionGatedStrategy", return_value=mock_ensemble):
                results = _run_forecast_worker(
                    weather_df, warmup, timestamps,
                    xgb, ext_high, ext_low, imputer, ["temperature_2m"],
                    zone_cfg, label="test_extreme",
                )

        assert results[ts]["extreme_high"] == 65.0
        assert results[ts]["extreme_low"] == 35.0
        ext_high.predict.assert_called_once()
        ext_low.predict.assert_called_once()

    def test_worker_already_known_timestamp(self):
        """If ts is already in warmup, the worker still produces a result."""
        from unittest.mock import MagicMock, patch

        import pandas as pd
        from sklearn.impute import SimpleImputer

        from runeflow.services.inference import _run_forecast_worker

        ts = pd.Timestamp("2024-06-15T00:00:00", tz="UTC")
        timestamps = pd.DatetimeIndex([ts])
        weather_df = pd.DataFrame({"temperature_2m": [99.0]}, index=timestamps)

        warmup = pd.DataFrame(
            {"temperature_2m": [8.0], "Price_EUR_MWh": [50.0]}, index=[ts]
        )

        imputer = SimpleImputer(strategy="mean")
        imputer.fit(pd.DataFrame({"temperature_2m": [10.0]}))

        xgb = MagicMock()
        xgb._model_lower = xgb._model_p50 = xgb._model_upper = None
        xgb.predict.return_value = pd.DataFrame({"prediction": [50.0], "lower": [40.0], "upper": [60.0]}, index=[ts])

        ext_high = MagicMock()
        ext_high._model = None
        ext_high.is_trained = False
        ext_low = MagicMock()
        ext_low._model = None
        ext_low.is_trained = False
        zone_cfg = MagicMock()

        mock_pipeline = MagicMock()
        mock_pipeline.transform.return_value = pd.DataFrame({"temperature_2m": [8.0]}, index=[ts])

        mock_ensemble = MagicMock()
        mock_ensemble.combine.return_value = pd.DataFrame({
            "prediction": [50.0], "lower": [40.0], "upper": [60.0],
            "uncertainty": [20.0], "model_agreement": [0.9],
        }, index=[ts])

        with patch("runeflow.services.inference.build_pipeline", return_value=mock_pipeline):
            with patch("runeflow.services.inference.ConditionGatedStrategy", return_value=mock_ensemble):
                results = _run_forecast_worker(
                    weather_df, warmup, timestamps,
                    xgb, ext_high, ext_low, imputer, ["temperature_2m"],
                    zone_cfg, label="test_known",
                )

        assert ts in results

    def test_worker_empty_features_returns_no_results(self):
        """If pipeline.transform returns empty, timestamp is skipped."""
        from unittest.mock import MagicMock, patch

        import pandas as pd
        from sklearn.impute import SimpleImputer

        from runeflow.services.inference import _run_forecast_worker

        ts = pd.Timestamp("2024-07-01T00:00:00", tz="UTC")
        timestamps = pd.DatetimeIndex([ts])
        weather_df = pd.DataFrame({"temperature_2m": [10.0]}, index=timestamps)

        warmup_idx = pd.date_range("2024-06-15", periods=24, freq="h", tz="UTC")
        warmup = pd.DataFrame({"temperature_2m": [8.0] * 24}, index=warmup_idx)

        xgb = MagicMock()
        xgb._model_lower = xgb._model_p50 = xgb._model_upper = None
        ext_high = MagicMock()
        ext_high._model = None
        ext_high.is_trained = False
        ext_low = MagicMock()
        ext_low._model = None
        ext_low.is_trained = False
        zone_cfg = MagicMock()

        mock_pipeline = MagicMock()
        mock_pipeline.transform.return_value = pd.DataFrame()  # empty

        imputer = SimpleImputer(strategy="mean")
        imputer.fit(pd.DataFrame({"temperature_2m": [10.0]}))

        with patch("runeflow.services.inference.build_pipeline", return_value=mock_pipeline):
            with patch("runeflow.services.inference.ConditionGatedStrategy"):
                results = _run_forecast_worker(
                    weather_df, warmup, timestamps,
                    xgb, ext_high, ext_low, imputer, [],
                    zone_cfg, label="test_empty",
                )

        assert len(results) == 0

    def test_worker_ts_not_in_weather_skips(self):
        """If a future timestamp is not in weather_df, it's skipped."""
        from unittest.mock import MagicMock, patch

        import pandas as pd
        from sklearn.impute import SimpleImputer

        from runeflow.services.inference import _run_forecast_worker

        ts = pd.Timestamp("2024-07-01T00:00:00", tz="UTC")
        timestamps = pd.DatetimeIndex([ts])
        other_ts = pd.Timestamp("2024-07-02T00:00:00", tz="UTC")
        weather_df = pd.DataFrame({"temperature_2m": [10.0]}, index=[other_ts])

        warmup_idx = pd.date_range("2024-06-15", periods=24, freq="h", tz="UTC")
        warmup = pd.DataFrame({"temperature_2m": [8.0] * 24}, index=warmup_idx)

        xgb = MagicMock()
        xgb._model_lower = xgb._model_p50 = xgb._model_upper = None
        ext_high = MagicMock()
        ext_high._model = None
        ext_low = MagicMock()
        ext_low._model = None
        zone_cfg = MagicMock()

        imputer = SimpleImputer(strategy="mean")
        imputer.fit(pd.DataFrame({"temperature_2m": [10.0]}))

        with patch("runeflow.services.inference.build_pipeline"):
            with patch("runeflow.services.inference.ConditionGatedStrategy"):
                results = _run_forecast_worker(
                    weather_df, warmup, timestamps,
                    xgb, ext_high, ext_low, imputer, [],
                    zone_cfg, label="test_skip",
                )

        assert len(results) == 0

    def test_worker_set_params_on_model_internals(self):
        """Cover lines 83-86: set_params(nthread=1) on internal model objects."""
        from unittest.mock import MagicMock, patch

        import pandas as pd
        from sklearn.impute import SimpleImputer

        from runeflow.services.inference import _run_forecast_worker

        ts = pd.Timestamp("2024-07-01T00:00:00", tz="UTC")
        timestamps = pd.DatetimeIndex([ts])
        weather_df = pd.DataFrame({"temperature_2m": [10.0]}, index=timestamps)

        warmup_idx = pd.date_range("2024-06-15", periods=24, freq="h", tz="UTC")
        warmup = pd.DataFrame({"temperature_2m": [8.0] * 24, "Price_EUR_MWh": [50.0] * 24}, index=warmup_idx)

        imputer = SimpleImputer(strategy="mean")
        imputer.fit(pd.DataFrame({"temperature_2m": [10.0]}))

        inner_model_lower = MagicMock()
        inner_model_p50 = MagicMock()
        inner_model_upper = MagicMock()
        inner_ext_high = MagicMock()
        inner_ext_low = MagicMock()

        xgb = MagicMock()
        xgb._model_lower = inner_model_lower
        xgb._model_p50 = inner_model_p50
        xgb._model_upper = inner_model_upper
        xgb.predict.return_value = pd.DataFrame(
            {"prediction": [50.0], "lower": [40.0], "upper": [60.0]}, index=[ts]
        )

        ext_high = MagicMock()
        ext_high._model = inner_ext_high
        ext_high.is_trained = False
        ext_low = MagicMock()
        ext_low._model = inner_ext_low
        ext_low.is_trained = False
        zone_cfg = MagicMock()

        mock_pipeline = MagicMock()
        mock_pipeline.transform.return_value = pd.DataFrame(
            {"temperature_2m": [10.0]}, index=timestamps
        )
        mock_ensemble = MagicMock()
        mock_ensemble.combine.return_value = pd.DataFrame({
            "prediction": [50.0], "lower": [40.0], "upper": [60.0],
            "uncertainty": [20.0], "model_agreement": [0.9],
        }, index=[ts])

        with patch("runeflow.services.inference.build_pipeline", return_value=mock_pipeline):
            with patch("runeflow.services.inference.ConditionGatedStrategy", return_value=mock_ensemble):
                results = _run_forecast_worker(
                    weather_df, warmup, timestamps,
                    xgb, ext_high, ext_low, imputer, ["temperature_2m"],
                    zone_cfg, label="test_setparams",
                )

        inner_model_lower.set_params.assert_called_once_with(nthread=1)
        inner_model_p50.set_params.assert_called_once_with(nthread=1)
        inner_model_upper.set_params.assert_called_once_with(nthread=1)
        inner_ext_high.set_params.assert_called_once_with(nthread=1)
        inner_ext_low.set_params.assert_called_once_with(nthread=1)
        assert ts in results

    def test_worker_set_params_exception_is_swallowed(self):
        """Cover lines 85-86: set_params raises → except passes silently."""
        from unittest.mock import MagicMock, patch

        import pandas as pd
        from sklearn.impute import SimpleImputer

        from runeflow.services.inference import _run_forecast_worker

        ts = pd.Timestamp("2024-07-01T00:00:00", tz="UTC")
        timestamps = pd.DatetimeIndex([ts])
        weather_df = pd.DataFrame({"temperature_2m": [10.0]}, index=timestamps)

        warmup_idx = pd.date_range("2024-06-15", periods=24, freq="h", tz="UTC")
        warmup = pd.DataFrame({"temperature_2m": [8.0] * 24, "Price_EUR_MWh": [50.0] * 24}, index=warmup_idx)

        imputer = SimpleImputer(strategy="mean")
        imputer.fit(pd.DataFrame({"temperature_2m": [10.0]}))

        # Model internals that RAISE on set_params
        inner_model = MagicMock()
        inner_model.set_params.side_effect = TypeError("nope")

        xgb = MagicMock()
        xgb._model_lower = inner_model
        xgb._model_p50 = None
        xgb._model_upper = None
        xgb.predict.return_value = pd.DataFrame(
            {"prediction": [50.0], "lower": [40.0], "upper": [60.0]}, index=[ts]
        )
        ext_high = MagicMock(); ext_high._model = None; ext_high.is_trained = False
        ext_low = MagicMock(); ext_low._model = None; ext_low.is_trained = False
        zone_cfg = MagicMock()

        mock_pipeline = MagicMock()
        mock_pipeline.transform.return_value = pd.DataFrame({"temperature_2m": [10.0]}, index=timestamps)
        mock_ensemble = MagicMock()
        mock_ensemble.combine.return_value = pd.DataFrame({
            "prediction": [50.0], "lower": [40.0], "upper": [60.0],
            "uncertainty": [20.0], "model_agreement": [0.9],
        }, index=[ts])

        with patch("runeflow.services.inference.build_pipeline", return_value=mock_pipeline):
            with patch("runeflow.services.inference.ConditionGatedStrategy", return_value=mock_ensemble):
                # Should NOT raise despite set_params failing
                results = _run_forecast_worker(
                    weather_df, warmup, timestamps,
                    xgb, ext_high, ext_low, imputer, ["temperature_2m"],
                    zone_cfg, label="test_setparams_exc",
                )

        assert ts in results

    def test_worker_tz_naive_weather_df(self):
        """Cover line 95: weather_df has no tz → tz_localize branch."""
        from unittest.mock import MagicMock, patch

        import pandas as pd
        from sklearn.impute import SimpleImputer

        from runeflow.services.inference import _run_forecast_worker

        ts = pd.Timestamp("2024-07-01T00:00:00", tz="UTC")
        timestamps = pd.DatetimeIndex([ts])
        # tz-NAIVE weather_df → triggers tz_localize (line 95)
        weather_df = pd.DataFrame(
            {"temperature_2m": [10.0]},
            index=pd.DatetimeIndex([pd.Timestamp("2024-07-01T00:00:00")])  # no tz
        )

        warmup_idx = pd.date_range("2024-06-15", periods=24, freq="h", tz="UTC")
        warmup = pd.DataFrame({"temperature_2m": [8.0] * 24, "Price_EUR_MWh": [50.0] * 24}, index=warmup_idx)

        imputer = SimpleImputer(strategy="mean")
        imputer.fit(pd.DataFrame({"temperature_2m": [10.0]}))

        xgb = MagicMock()
        xgb._model_lower = xgb._model_p50 = xgb._model_upper = None
        xgb.predict.return_value = pd.DataFrame(
            {"prediction": [50.0], "lower": [40.0], "upper": [60.0]}, index=[ts]
        )
        ext_high = MagicMock(); ext_high._model = None; ext_high.is_trained = False
        ext_low = MagicMock(); ext_low._model = None; ext_low.is_trained = False
        zone_cfg = MagicMock()

        mock_pipeline = MagicMock()
        mock_pipeline.transform.return_value = pd.DataFrame({"temperature_2m": [10.0]}, index=timestamps)
        mock_ensemble = MagicMock()
        mock_ensemble.combine.return_value = pd.DataFrame({
            "prediction": [50.0], "lower": [40.0], "upper": [60.0],
            "uncertainty": [20.0], "model_agreement": [0.9],
        }, index=[ts])

        with patch("runeflow.services.inference.build_pipeline", return_value=mock_pipeline):
            with patch("runeflow.services.inference.ConditionGatedStrategy", return_value=mock_ensemble):
                results = _run_forecast_worker(
                    weather_df, warmup, timestamps,
                    xgb, ext_high, ext_low, imputer, ["temperature_2m"],
                    zone_cfg, label="test_tz",
                )

        assert ts in results

    def test_worker_missing_features_and_imputation(self):
        """Cover lines 122, 130, 137: ts not in index, missing features, imputer."""
        from unittest.mock import MagicMock, patch

        import numpy as np
        import pandas as pd
        from sklearn.impute import SimpleImputer

        from runeflow.services.inference import _run_forecast_worker

        ts = pd.Timestamp("2024-07-01T00:00:00", tz="UTC")
        timestamps = pd.DatetimeIndex([ts])
        weather_df = pd.DataFrame({"temperature_2m": [10.0]}, index=timestamps)

        warmup_idx = pd.date_range("2024-06-15", periods=24, freq="h", tz="UTC")
        warmup = pd.DataFrame({"temperature_2m": [8.0] * 24, "Price_EUR_MWh": [50.0] * 24}, index=warmup_idx)

        features_list = ["temperature_2m", "wind_speed_10m", "extra_feature"]

        imputer = SimpleImputer(strategy="mean")
        imputer.fit(pd.DataFrame({
            "temperature_2m": [10.0, 12.0],
            "wind_speed_10m": [5.0, 6.0],
            "extra_feature": [1.0, 2.0],
        }))

        xgb = MagicMock()
        xgb._model_lower = xgb._model_p50 = xgb._model_upper = None
        xgb.predict.return_value = pd.DataFrame(
            {"prediction": [50.0], "lower": [40.0], "upper": [60.0]}, index=[ts]
        )
        ext_high = MagicMock(); ext_high._model = None; ext_high.is_trained = False
        ext_low = MagicMock(); ext_low._model = None; ext_low.is_trained = False
        zone_cfg = MagicMock()

        other_ts = pd.Timestamp("2024-06-30T23:00:00", tz="UTC")
        mock_pipeline = MagicMock()
        mock_pipeline.transform.return_value = pd.DataFrame(
            {"temperature_2m": [np.nan]},
            index=pd.DatetimeIndex([other_ts]),
        )

        mock_ensemble = MagicMock()
        mock_ensemble.combine.return_value = pd.DataFrame({
            "prediction": [50.0], "lower": [40.0], "upper": [60.0],
            "uncertainty": [20.0], "model_agreement": [0.9],
        }, index=[ts])

        with patch("runeflow.services.inference.build_pipeline", return_value=mock_pipeline):
            with patch("runeflow.services.inference.ConditionGatedStrategy", return_value=mock_ensemble):
                results = _run_forecast_worker(
                    weather_df, warmup, timestamps,
                    xgb, ext_high, ext_low, imputer, features_list,
                    zone_cfg, label="test_missing",
                )

        assert ts in results


class TestInferenceServiceRun:
    """Tests for InferenceService.run() — the orchestration method (lines 234-321)."""

    def _make_service(self):
        from runeflow.services.inference import InferenceService

        svc = InferenceService.__new__(InferenceService)
        svc._zone_cfg = MagicMock()
        svc._zone_cfg.zone = "NL"
        svc._zone_cfg.weather_locations = [MagicMock()]
        svc._store = MagicMock()
        svc._weather_port = MagicMock()
        svc._supplemental_port = MagicMock()
        return svc

    def test_run_deterministic_only(self):
        """Cover run() with no ensemble members."""
        from unittest.mock import patch

        svc = self._make_service()

        xgb = MagicMock(); xgb.is_trained = True
        ext_high = MagicMock(); ext_high.is_trained = False
        ext_low = MagicMock(); ext_low.is_trained = False
        imputer = MagicMock()
        features = ["temperature_2m"]
        svc._store.load_warmup_cache.return_value = pd.DataFrame(
            {"temperature_2m": [8.0] * 24, "Price_EUR_MWh": [50.0] * 24},
            index=pd.date_range("2024-06-01", periods=24, freq="h", tz="UTC"),
        )

        det_weather = MagicMock()
        det_weather.df = pd.DataFrame({"temperature_2m": [10.0]}, index=pd.date_range("2024-07-01", periods=1, freq="h", tz="UTC"))
        det_weather.source = "test"
        det_weather.locations = []
        det_weather.fetched_at = pd.Timestamp.now("UTC")
        svc._weather_port.download_forecast.return_value = det_weather
        svc._weather_port.download_ensemble_forecast.side_effect = RuntimeError("no ensemble")
        svc._supplemental_port.supports_zone.return_value = False

        ts = pd.Timestamp.now("UTC").floor("h")
        worker_result = {
            ts: {
                "prediction": 50.0, "lower": 40.0, "upper": 60.0,
                "uncertainty": 20.0, "model_agreement": 0.9,
                "lower_static": 38.0, "upper_static": 62.0,
                "xgboost_p50": 50.0, "xgboost_p10": 40.0, "xgboost_p90": 60.0,
                "extreme_high": None, "extreme_low": None,
            }
        }

        with patch.object(svc, "_load_models", return_value=(xgb, ext_high, ext_low, imputer, features)):
            with patch.object(svc, "_load_supplemental_forecast", return_value=None):
                with patch("runeflow.services.inference._run_forecast_worker", return_value=worker_result):
                    result = svc.run()

        assert result.zone == "NL"
        assert len(result.points) > 0

    def test_run_with_ensemble_and_supplemental(self):
        """Cover run() with ensemble members and supplemental data."""
        from unittest.mock import patch

        svc = self._make_service()

        xgb = MagicMock(); xgb.is_trained = True
        ext_high = MagicMock(); ext_high.is_trained = True
        ext_low = MagicMock(); ext_low.is_trained = True
        imputer = MagicMock()
        features = ["temperature_2m"]
        svc._store.load_warmup_cache.return_value = pd.DataFrame(
            {"temperature_2m": [8.0] * 24, "Price_EUR_MWh": [50.0] * 24},
            index=pd.date_range("2024-06-01", periods=24, freq="h", tz="UTC"),
        )

        det_weather = MagicMock()
        det_weather.df = pd.DataFrame({"temperature_2m": [10.0]}, index=pd.date_range("2024-07-01", periods=1, freq="h", tz="UTC"))
        det_weather.source = "test"
        det_weather.locations = []
        det_weather.fetched_at = pd.Timestamp.now("UTC")
        svc._weather_port.download_forecast.return_value = det_weather

        ens_member = MagicMock()
        ens_member.df = pd.DataFrame({"temperature_2m": [11.0]}, index=pd.date_range("2024-07-01", periods=1, freq="h", tz="UTC"))
        svc._weather_port.download_ensemble_forecast.return_value = [ens_member] * 3

        ned_df = pd.DataFrame(
            {"ned_forecast_kwh": [1000.0]},
            index=pd.DatetimeIndex([pd.Timestamp("2024-07-01", tz="UTC")])
        )

        ts = pd.Timestamp.now("UTC").floor("h")
        worker_result = {
            ts: {
                "prediction": 50.0, "lower": 40.0, "upper": 60.0,
                "uncertainty": 20.0, "model_agreement": 0.9,
                "lower_static": 38.0, "upper_static": 62.0,
                "xgboost_p50": 50.0, "xgboost_p10": 40.0, "xgboost_p90": 60.0,
                "extreme_high": 65.0, "extreme_low": 35.0,
            }
        }

        with patch.object(svc, "_load_models", return_value=(xgb, ext_high, ext_low, imputer, features)):
            with patch.object(svc, "_load_supplemental_forecast", return_value=ned_df):
                with patch("runeflow.services.inference._run_forecast_worker", return_value=worker_result):
                    with patch("runeflow.services.inference.joblib") as mock_joblib:
                        mock_joblib.Parallel.return_value = lambda gen: list(gen)
                        mock_joblib.delayed = lambda f: f
                        result = svc.run()

        assert result.zone == "NL"
        assert len(result.points) > 0

    def test_run_no_warmup_raises(self):
        """Cover warmup check in run()."""
        from unittest.mock import patch

        svc = self._make_service()

        xgb = MagicMock(); xgb.is_trained = True
        ext_high = MagicMock(); ext_high.is_trained = False
        ext_low = MagicMock(); ext_low.is_trained = False
        imputer = MagicMock()
        features = []

        svc._store.load_warmup_cache.return_value = None

        with patch.object(svc, "_load_models", return_value=(xgb, ext_high, ext_low, imputer, features)):
            with pytest.raises(RuntimeError, match="No warmup cache"):
                svc.run()


class TestInferenceSupplementalTzConvert:
    """Cover tz_localize branches in _load_supplemental_forecast (lines 337, 355)."""

    def _make_service(self):
        from runeflow.services.inference import InferenceService

        svc = InferenceService.__new__(InferenceService)
        svc._zone_cfg = MagicMock()
        svc._zone_cfg.zone = "NL"
        svc._store = MagicMock()
        svc._supplemental_port = MagicMock()
        return svc

    def test_port_returns_tz_naive_data(self):
        """Cover line 337: port data index has no tz → tz_localize('UTC')."""
        svc = self._make_service()
        svc._supplemental_port.supports_zone.return_value = True

        # tz-NAIVE index → triggers tz_localize path (line 337)
        naive_idx = pd.date_range("2024-07-01", periods=5, freq="h")
        df = pd.DataFrame({"ned_forecast_kwh": [100.0] * 5}, index=naive_idx)
        svc._supplemental_port.download_forecast.return_value = df

        result = svc._load_supplemental_forecast("NL")
        assert result is not None
        assert "ned_utilization_kwh" in result.columns
        assert result.index.tz is not None

    def test_cache_returns_tz_naive_data(self):
        """Cover line 355: cached data index has no tz → tz_localize('UTC')."""
        svc = self._make_service()
        svc._supplemental_port.supports_zone.return_value = False

        # tz-NAIVE index → triggers tz_localize path (line 355)
        naive_idx = pd.date_range("2024-07-01", periods=5, freq="h")
        cached = pd.DataFrame({"ned_forecast_kwh": [200.0] * 5}, index=naive_idx)
        svc._store.load_supplemental.return_value = cached

        result = svc._load_supplemental_forecast("NL")
        assert result is not None
        assert "ned_utilization_kwh" in result.columns
        assert result.index.tz is not None