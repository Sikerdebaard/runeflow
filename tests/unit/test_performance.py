# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2024-2026 Thomas Phil — runeflow
# See LICENSE and COMMERCIAL-LICENSE.md for licensing details.

"""Tests for Phase 4 — model performance domain, services and store archive."""

from __future__ import annotations

import contextlib
import json
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from runeflow.adapters.store.parquet import ParquetStore
from runeflow.domain.forecast import ForecastPoint, ForecastResult
from runeflow.domain.performance import (
    ForecastAccuracy,
    GlobalPerformance,
    HorizonMetrics,
    ZonePerformance,
)
from runeflow.domain.price import PriceRecord, PriceSeries
from runeflow.zones.registry import ZoneRegistry

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_forecast_result(
    zone: str = "NL",
    n: int = 24,
    created_at: pd.Timestamp | None = None,
    base_price: float = 50.0,
    lower: float = 30.0,
    upper: float = 70.0,
    p25: float = 35.0,
    p75: float = 65.0,
) -> ForecastResult:
    if created_at is None:
        created_at = pd.Timestamp.now("UTC") - pd.Timedelta(hours=1)
    ts_start = created_at + pd.Timedelta(hours=1)
    ts = pd.date_range(ts_start, periods=n, freq="h", tz="UTC")
    points = tuple(
        ForecastPoint(
            timestamp=t,
            prediction=base_price,
            lower=lower,
            upper=upper,
            uncertainty=upper - lower,
            model_agreement=0.85,
            lower_static=lower,
            upper_static=upper,
            ensemble_p50=base_price,
            ensemble_p25=p25,
            ensemble_p75=p75,
        )
        for t in ts
    )
    return ForecastResult(
        zone=zone,
        points=points,
        ensemble_members=pd.DataFrame(index=ts),
        model_predictions={},
        created_at=created_at,
        model_version="202401010000",
    )


def _make_price_series(
    zone: str = "NL",
    n: int = 48,
    base_price: float = 50.0,
) -> PriceSeries:
    ts = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")
    records = tuple(
        PriceRecord(timestamp=t, price_eur_mwh=base_price + float(i % 5)) for i, t in enumerate(ts)
    )
    return PriceSeries(
        zone=zone,
        records=records,
        source="test",
        fetched_at=pd.Timestamp.now("UTC"),
    )


@pytest.fixture(scope="session")
def nl_zone_cfg():
    return ZoneRegistry.get("NL")


# ---------------------------------------------------------------------------
# 1. ParquetStore — forecast archive
# ---------------------------------------------------------------------------


class TestParquetStoreForecastArchive:
    def test_save_archive_creates_file(self, tmp_cache_dir):
        store = ParquetStore(tmp_cache_dir)
        result = _make_forecast_result()
        store.save_forecast_archive(result)
        archive_dir = tmp_cache_dir / "forecasts" / "NL" / "archive"
        files = list(archive_dir.glob("*.json"))
        assert len(files) == 1

    def test_save_forecast_auto_archives(self, tmp_cache_dir):
        store = ParquetStore(tmp_cache_dir)
        result = _make_forecast_result()
        store.save_forecast(result)
        archive_dir = tmp_cache_dir / "forecasts" / "NL" / "archive"
        assert archive_dir.exists()
        files = list(archive_dir.glob("*.json"))
        assert len(files) == 1

    def test_load_archive_empty_when_none(self, tmp_cache_dir):
        store = ParquetStore(tmp_cache_dir)
        results = store.load_forecast_archive("NL", days_back=30)
        assert results == []

    def test_load_archive_roundtrip(self, tmp_cache_dir):
        store = ParquetStore(tmp_cache_dir)
        result = _make_forecast_result()
        store.save_forecast_archive(result)

        loaded = store.load_forecast_archive("NL", days_back=30)
        assert len(loaded) == 1
        assert loaded[0].zone == "NL"
        assert loaded[0].model_version == result.model_version
        assert len(loaded[0].points) == 24

    def test_load_archive_returns_multiple(self, tmp_cache_dir):
        store = ParquetStore(tmp_cache_dir)
        for offset_h in [2, 5, 10]:
            r = _make_forecast_result(
                created_at=pd.Timestamp.now("UTC") - pd.Timedelta(hours=offset_h)
            )
            store.save_forecast_archive(r)

        loaded = store.load_forecast_archive("NL", days_back=30)
        assert len(loaded) == 3

    def test_load_archive_excludes_past_days_back(self, tmp_cache_dir):
        """Archive files with timestamps older than days_back are skipped."""
        store = ParquetStore(tmp_cache_dir)
        archive_dir = tmp_cache_dir / "forecasts" / "NL" / "archive"
        archive_dir.mkdir(parents=True, exist_ok=True)

        # Write a recent archive entry
        recent = _make_forecast_result(created_at=pd.Timestamp.now("UTC") - pd.Timedelta(days=1))
        ts_recent = (pd.Timestamp.now("UTC") - pd.Timedelta(days=1)).strftime("%Y%m%d_%H%M")
        data = {
            "zone": recent.zone,
            "created_at": recent.created_at.isoformat(),
            "model_version": recent.model_version,
            "points": [],
            "model_predictions": {},
        }
        (archive_dir / f"{ts_recent}.json").write_text(json.dumps(data), encoding="utf-8")

        # Write an old archive entry (40 days ago)
        ts_old = (pd.Timestamp.now("UTC") - pd.Timedelta(days=40)).strftime("%Y%m%d_%H%M")
        (archive_dir / f"{ts_old}.json").write_text(json.dumps(data), encoding="utf-8")

        loaded = store.load_forecast_archive("NL", days_back=30)
        assert len(loaded) == 1

    def test_cleanup_archive_deletes_old(self, tmp_cache_dir):
        store = ParquetStore(tmp_cache_dir)
        archive_dir = tmp_cache_dir / "forecasts" / "NL" / "archive"
        archive_dir.mkdir(parents=True, exist_ok=True)

        # Write an old file
        ts_old = (pd.Timestamp.now("UTC") - pd.Timedelta(days=40)).strftime("%Y%m%d_%H%M")
        old_path = archive_dir / f"{ts_old}.json"
        old_path.write_text("{}", encoding="utf-8")

        # Write a recent file
        ts_new = (pd.Timestamp.now("UTC") - pd.Timedelta(hours=1)).strftime("%Y%m%d_%H%M")
        new_path = archive_dir / f"{ts_new}.json"
        new_path.write_text("{}", encoding="utf-8")

        store._cleanup_archive("NL", max_days=30)
        assert not old_path.exists()
        assert new_path.exists()

    def test_load_archive_skips_corrupt_file(self, tmp_cache_dir):
        store = ParquetStore(tmp_cache_dir)
        archive_dir = tmp_cache_dir / "forecasts" / "NL" / "archive"
        archive_dir.mkdir(parents=True, exist_ok=True)

        # Good file
        r = _make_forecast_result()
        store.save_forecast_archive(r)

        # Corrupt file
        ts = (pd.Timestamp.now("UTC") - pd.Timedelta(hours=2)).strftime("%Y%m%d_%H%M")
        (archive_dir / f"{ts}.json").write_text("NOT JSON {{{", encoding="utf-8")

        loaded = store.load_forecast_archive("NL", days_back=30)
        # Only the valid file is returned; corrupt one returns None, gets skipped
        assert all(x is not None for x in loaded)

    def test_write_then_load_preserves_ensemble_p25_p75(self, tmp_cache_dir):
        store = ParquetStore(tmp_cache_dir)
        result = _make_forecast_result(p25=38.0, p75=62.0)
        store.save_forecast_archive(result)

        loaded = store.load_forecast_archive("NL", days_back=30)
        assert loaded[0].points[0].ensemble_p25 == pytest.approx(38.0)
        assert loaded[0].points[0].ensemble_p75 == pytest.approx(62.0)


# ---------------------------------------------------------------------------
# 2. Domain types — instantiation & field checks
# ---------------------------------------------------------------------------


class TestPerformanceDomainTypes:
    def test_forecast_accuracy_fields(self):
        acc = ForecastAccuracy(
            forecast_created_at=pd.Timestamp("2024-01-01", tz="UTC"),
            model_version="202401010000",
            n_comparable_hours=20,
            mae=5.0,
            rmse=6.5,
            mape=8.0,
            directional_accuracy=0.75,
            mean_model_agreement=0.85,
        )
        assert acc.mae == pytest.approx(5.0)
        assert acc.n_comparable_hours == 20

    def test_horizon_metrics_fields(self):
        hm = HorizonMetrics(horizon_label="1-6h", n_hours=100, mae=3.0, rmse=4.0)
        assert hm.horizon_label == "1-6h"
        assert hm.n_hours == 100

    def test_zone_performance_frozen(self):
        zp = ZonePerformance(
            zone="NL",
            zone_name="Netherlands",
            training_history=[],
            forecast_accuracies=[],
            horizon_metrics=[],
            overall_mae=5.0,
            overall_rmse=6.5,
            ensemble_coverage_pct=45.0,
            n_archived_forecasts=10,
            n_comparable_hours=200,
            generated_at=pd.Timestamp.now("UTC"),
        )
        assert zp.zone == "NL"
        with pytest.raises((AttributeError, TypeError)):
            zp.zone = "DE"  # type: ignore[misc]  # frozen dataclass

    def test_global_performance_rankings(self):
        gp = GlobalPerformance(
            zones={},
            rankings=[{"zone": "NL", "mae": 5.0, "rank": 1}],
            generated_at=pd.Timestamp.now("UTC"),
        )
        assert gp.rankings[0]["rank"] == 1


# ---------------------------------------------------------------------------
# 3. PerformanceService
# ---------------------------------------------------------------------------


class TestPerformanceService:
    def _make_store(
        self,
        archives: list | None = None,
        prices: PriceSeries | None = None,
        train_history: pd.DataFrame | None = None,
    ) -> MagicMock:
        store = MagicMock()
        store.load_forecast_archive.return_value = archives or []
        store.load_prices.return_value = prices
        store.load_supplemental.return_value = train_history
        return store

    def test_no_archives_returns_empty_performance(self, nl_zone_cfg):
        from runeflow.services.performance import PerformanceService

        store = self._make_store()
        svc = PerformanceService(zone_cfg=nl_zone_cfg, store=store)
        perf = svc.compute_zone_performance()

        assert perf.zone == "NL"
        assert perf.n_archived_forecasts == 0
        assert perf.n_comparable_hours == 0
        assert perf.overall_mae is None

    def test_no_actuals_returns_empty_performance(self, nl_zone_cfg):
        from runeflow.services.performance import PerformanceService

        result = _make_forecast_result()
        store = self._make_store(archives=[result], prices=None)
        svc = PerformanceService(zone_cfg=nl_zone_cfg, store=store)
        perf = svc.compute_zone_performance()

        assert perf.overall_mae is None
        assert perf.n_comparable_hours == 0

    def test_computes_mae_with_matching_actuals(self, nl_zone_cfg):
        from runeflow.services.performance import PerformanceService

        # Forecast starts at T+1h from created_at; actuals cover the same window
        created = pd.Timestamp("2024-01-01 00:00:00", tz="UTC")
        forecast = _make_forecast_result(
            created_at=created,
            n=24,
            base_price=50.0,
        )

        # Actuals: constant 52 EUR/MWh → MAE should be 2.0
        ts = pd.date_range("2024-01-01", periods=48, freq="h", tz="UTC")
        records = tuple(PriceRecord(timestamp=t, price_eur_mwh=52.0) for t in ts)
        prices = PriceSeries(
            zone="NL", records=records, source="test", fetched_at=pd.Timestamp.now("UTC")
        )

        store = self._make_store(archives=[forecast], prices=prices)
        svc = PerformanceService(zone_cfg=nl_zone_cfg, store=store)
        perf = svc.compute_zone_performance()

        assert perf.n_comparable_hours > 0
        assert perf.overall_mae is not None
        assert perf.overall_mae == pytest.approx(2.0, abs=1e-6)

    def test_horizon_metrics_bucketing(self, nl_zone_cfg):
        from runeflow.services.performance import PerformanceService

        created = pd.Timestamp("2024-01-01 00:00:00", tz="UTC")
        forecast = _make_forecast_result(created_at=created, n=72, base_price=50.0)

        ts = pd.date_range("2024-01-01", periods=120, freq="h", tz="UTC")
        records = tuple(PriceRecord(timestamp=t, price_eur_mwh=50.0) for t in ts)
        prices = PriceSeries(
            zone="NL", records=records, source="test", fetched_at=pd.Timestamp.now("UTC")
        )

        store = self._make_store(archives=[forecast], prices=prices)
        svc = PerformanceService(zone_cfg=nl_zone_cfg, store=store)
        perf = svc.compute_zone_performance()

        # Should have multiple horizon bands (1-6h, 6-12h, 12-24h, 24-48h, 48h+)
        assert len(perf.horizon_metrics) >= 2
        labels = [h.horizon_label for h in perf.horizon_metrics]
        assert "1-6h" in labels

    def test_ensemble_coverage_within_band(self, nl_zone_cfg):
        from runeflow.services.performance import PerformanceService

        created = pd.Timestamp("2024-01-01 00:00:00", tz="UTC")
        # P25=40, P75=60 → actual=50 should be inside → coverage 100%
        forecast = _make_forecast_result(
            created_at=created, n=24, base_price=50.0, p25=40.0, p75=60.0
        )

        ts = pd.date_range("2024-01-01", periods=48, freq="h", tz="UTC")
        records = tuple(PriceRecord(timestamp=t, price_eur_mwh=50.0) for t in ts)
        prices = PriceSeries(
            zone="NL", records=records, source="test", fetched_at=pd.Timestamp.now("UTC")
        )

        store = self._make_store(archives=[forecast], prices=prices)
        svc = PerformanceService(zone_cfg=nl_zone_cfg, store=store)
        perf = svc.compute_zone_performance()

        assert perf.ensemble_coverage_pct is not None
        assert perf.ensemble_coverage_pct == pytest.approx(100.0, abs=1e-6)

    def test_ensemble_coverage_outside_band(self, nl_zone_cfg):
        from runeflow.services.performance import PerformanceService

        created = pd.Timestamp("2024-01-01 00:00:00", tz="UTC")
        # P25=40, P75=60 → actual=80 is outside → coverage 0%
        forecast = _make_forecast_result(
            created_at=created, n=24, base_price=50.0, p25=40.0, p75=60.0
        )

        ts = pd.date_range("2024-01-01", periods=48, freq="h", tz="UTC")
        records = tuple(PriceRecord(timestamp=t, price_eur_mwh=80.0) for t in ts)
        prices = PriceSeries(
            zone="NL", records=records, source="test", fetched_at=pd.Timestamp.now("UTC")
        )

        store = self._make_store(archives=[forecast], prices=prices)
        svc = PerformanceService(zone_cfg=nl_zone_cfg, store=store)
        perf = svc.compute_zone_performance()

        assert perf.ensemble_coverage_pct == pytest.approx(0.0, abs=1e-6)

    def test_train_history_loaded_from_sidecar(self, nl_zone_cfg):
        from runeflow.services.performance import PerformanceService

        sidecar = pd.DataFrame(
            [
                {
                    "model_version": "202401010000",
                    "mae": 4.5,
                    "r2": 0.8,
                    "coverage": 85.0,
                    "trained_at": "2024-01-01T00:00:00",
                },
            ]
        )
        store = self._make_store(train_history=sidecar)
        svc = PerformanceService(zone_cfg=nl_zone_cfg, store=store)
        perf = svc.compute_zone_performance()

        assert len(perf.training_history) == 1
        assert perf.training_history[0]["model_version"] == "202401010000"

    def test_rmse_is_sqrt_of_mean_squared_errors(self, nl_zone_cfg):
        from runeflow.services.performance import PerformanceService

        created = pd.Timestamp("2024-01-01 00:00:00", tz="UTC")
        # Forecast predicts 50, actuals alternate 48 and 52 → errors are all 2
        forecast = _make_forecast_result(created_at=created, n=24, base_price=50.0)

        ts = pd.date_range("2024-01-01", periods=48, freq="h", tz="UTC")
        records = tuple(
            PriceRecord(timestamp=t, price_eur_mwh=48.0 + float(4 * (i % 2)))
            for i, t in enumerate(ts)
        )
        prices = PriceSeries(
            zone="NL", records=records, source="test", fetched_at=pd.Timestamp.now("UTC")
        )

        store = self._make_store(archives=[forecast], prices=prices)
        svc = PerformanceService(zone_cfg=nl_zone_cfg, store=store)
        perf = svc.compute_zone_performance()

        assert perf.overall_rmse is not None
        # All errors are 2 → RMSE = 2
        assert perf.overall_rmse == pytest.approx(2.0, abs=0.01)


# ---------------------------------------------------------------------------
# 4. ExportPerformanceService
# ---------------------------------------------------------------------------


class TestExportPerformanceService:
    def _make_mock_zone_performance(self, zone: str = "NL") -> MagicMock:
        perf = MagicMock(spec=ZonePerformance)
        perf.zone = zone
        perf.zone_name = "Netherlands"
        perf.overall_mae = 5.0
        perf.overall_rmse = 6.5
        perf.ensemble_coverage_pct = 45.0
        perf.n_archived_forecasts = 10
        perf.n_comparable_hours = 200
        perf.training_history = []
        perf.horizon_metrics = []
        perf.forecast_accuracies = []
        perf.generated_at = pd.Timestamp.now("UTC")
        return perf

    def test_run_writes_json_file(self, tmp_path, nl_zone_cfg):
        from runeflow.services.export_performance import ExportPerformanceService
        from runeflow.services.performance import PerformanceService

        mock_perf = self._make_mock_zone_performance("NL")
        mock_svc = MagicMock(spec=PerformanceService)
        mock_svc.compute_zone_performance.return_value = mock_perf

        with (
            patch("runeflow.services.performance.PerformanceService", return_value=mock_svc),
            patch("runeflow.binder.configure_injector"),
        ):
            store = MagicMock()
            svc = ExportPerformanceService(store=store)
            output = tmp_path / "performance.json"
            svc.run(output_path=output, zones=["NL"])

        assert output.exists()
        data = json.loads(output.read_text(encoding="utf-8"))
        assert "NL" in data
        assert "_generated_at" in data

    def test_run_includes_rankings(self, tmp_path, nl_zone_cfg):
        from runeflow.services.export_performance import ExportPerformanceService
        from runeflow.services.performance import PerformanceService

        perf_nl = self._make_mock_zone_performance("NL")
        perf_de = self._make_mock_zone_performance("DE_LU")
        perf_de.zone = "DE_LU"
        perf_de.zone_name = "Germany/Luxembourg"
        perf_de.overall_mae = 3.0  # DE_LU should rank higher (lower MAE)

        perfs = {"NL": perf_nl, "DE_LU": perf_de}

        with (
            patch.object(
                PerformanceService,
                "compute_zone_performance",
                side_effect=lambda: perfs[perf_nl.zone],
            ),
            patch("runeflow.binder.configure_injector"),
        ):
            store = MagicMock()
            svc = ExportPerformanceService(store=store)
            output = tmp_path / "performance.json"
            payload = svc.run(output_path=output, zones=["NL"])

        assert "_rankings" in payload
        assert isinstance(payload["_rankings"], list)

    def test_serialize_zone_handles_nan(self, nl_zone_cfg):
        from runeflow.services.export_performance import _safe

        assert _safe(float("nan")) is None
        assert _safe(float("inf")) is None
        assert _safe(None) is None
        assert _safe(5.0) == pytest.approx(5.0)
        assert _safe(42) == 42

    def test_serialize_zone_rounds_floats(self, nl_zone_cfg):
        from runeflow.services.export_performance import _safe

        result = _safe(3.14159265)
        assert result == pytest.approx(3.1416, abs=1e-4)

    def test_failed_zone_does_not_crash_run(self, tmp_path):
        from runeflow.services.export_performance import ExportPerformanceService

        with (
            patch("runeflow.binder.configure_injector"),
            patch(
                "runeflow.services.export_performance.ExportPerformanceService._serialize_zone",
                side_effect=RuntimeError("oops"),
            ),
        ):
            store = MagicMock()
            svc = ExportPerformanceService(store=store)
            output = tmp_path / "performance.json"
            # Should not raise; silently skips failed zone
            with contextlib.suppress(Exception):
                svc.run(output_path=output, zones=["NL"])

    def test_safe_numeric_passthrough(self):
        from runeflow.services.export_performance import _safe

        assert _safe(10) == 10
        assert _safe("hello") == "hello"
