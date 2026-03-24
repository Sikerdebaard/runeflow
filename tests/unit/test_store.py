# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2024-2026 Thomas Phil — runeflow
# See LICENSE and COMMERCIAL-LICENSE.md for licensing details.

"""Tests for ParquetStore (price / model save-load, TTL, atomic writes)."""
from __future__ import annotations

import datetime
from pathlib import Path

import pandas as pd
import pytest

from runeflow.adapters.store.parquet import ParquetStore
from runeflow.domain.price import PriceRecord, PriceSeries


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_price_series(zone: str = "NL", n: int = 48) -> PriceSeries:
    ts = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")
    records = tuple(
        PriceRecord(timestamp=t, price_eur_mwh=float(50 + i % 20))
        for i, t in enumerate(ts)
    )
    return PriceSeries(
        zone=zone,
        records=records,
        source="test",
        fetched_at=pd.Timestamp.now("UTC"),
    )


# ── Save / Load Prices ────────────────────────────────────────────────────────

class TestParquetStorePrices:
    def test_save_and_load_roundtrip(self, tmp_cache_dir):
        store = ParquetStore(tmp_cache_dir)
        ps = _make_price_series()
        store.save_prices(ps)

        start = datetime.date(2024, 1, 1)
        end = datetime.date(2024, 1, 3)
        loaded = store.load_prices("NL", start, end)
        assert loaded is not None
        assert len(loaded) > 0

    def test_load_missing_returns_none(self, tmp_cache_dir):
        store = ParquetStore(tmp_cache_dir)
        result = store.load_prices("NL", datetime.date(2020, 1, 1), datetime.date(2020, 1, 2))
        assert result is None

    def test_load_out_of_range_returns_none(self, tmp_cache_dir):
        store = ParquetStore(tmp_cache_dir)
        ps = _make_price_series()
        store.save_prices(ps)

        # Request a date range well before the stored data
        result = store.load_prices("NL", datetime.date(2010, 1, 1), datetime.date(2010, 1, 2))
        assert result is None

    def test_save_merges_with_existing(self, tmp_cache_dir):
        """Saving two non-overlapping chunks should merge them."""
        store = ParquetStore(tmp_cache_dir)

        ts_a = pd.date_range("2024-01-01", periods=24, freq="h", tz="UTC")
        ts_b = pd.date_range("2024-01-02", periods=24, freq="h", tz="UTC")

        recs_a = tuple(PriceRecord(t, 50.0) for t in ts_a)
        recs_b = tuple(PriceRecord(t, 60.0) for t in ts_b)

        ps_a = PriceSeries(zone="NL", records=recs_a, source="x", fetched_at=pd.Timestamp.now("UTC"))
        ps_b = PriceSeries(zone="NL", records=recs_b, source="y", fetched_at=pd.Timestamp.now("UTC"))

        store.save_prices(ps_a)
        store.save_prices(ps_b)

        loaded = store.load_prices("NL", datetime.date(2024, 1, 1), datetime.date(2024, 1, 2))
        assert loaded is not None
        assert len(loaded) == 48

    def test_save_deduplicates_on_merge(self, tmp_cache_dir):
        """Saving the same data twice should not create duplicates."""
        store = ParquetStore(tmp_cache_dir)
        ps = _make_price_series(n=24)
        store.save_prices(ps)
        store.save_prices(ps)  # second save of same data

        loaded = store.load_prices("NL", datetime.date(2024, 1, 1), datetime.date(2024, 1, 2))
        assert loaded is not None
        assert len(loaded) == 24  # no duplicates

    def test_different_zones_isolated(self, tmp_cache_dir):
        store = ParquetStore(tmp_cache_dir)
        ps_nl = _make_price_series(zone="NL")
        ps_de = _make_price_series(zone="DE_LU")
        store.save_prices(ps_nl)
        store.save_prices(ps_de)

        loaded_nl = store.load_prices("NL", datetime.date(2024, 1, 1), datetime.date(2024, 1, 3))
        loaded_de = store.load_prices("DE_LU", datetime.date(2024, 1, 1), datetime.date(2024, 1, 3))
        assert loaded_nl is not None
        assert loaded_de is not None


# ── Save / Load Model Artifacts ───────────────────────────────────────────────

class TestParquetStoreModels:
    def test_save_and_load_bytes_roundtrip(self, tmp_cache_dir):
        store = ParquetStore(tmp_cache_dir)
        payload = b"fake-model-bytes-1234"
        store.save_model(payload, zone="NL", model_name="xgboost_quantile")
        loaded = store.load_model("NL", "xgboost_quantile")
        assert loaded == payload

    def test_load_missing_model_returns_none(self, tmp_cache_dir):
        store = ParquetStore(tmp_cache_dir)
        result = store.load_model("NL", "nonexistent_model")
        assert result is None

    def test_overwrite_model(self, tmp_cache_dir):
        store = ParquetStore(tmp_cache_dir)
        store.save_model(b"v1", zone="NL", model_name="extreme_high")
        store.save_model(b"v2", zone="NL", model_name="extreme_high")
        assert store.load_model("NL", "extreme_high") == b"v2"

    def test_different_zones_isolated(self, tmp_cache_dir):
        store = ParquetStore(tmp_cache_dir)
        store.save_model(b"nl-model", zone="NL", model_name="xgboost_quantile")
        store.save_model(b"de-model", zone="DE_LU", model_name="xgboost_quantile")
        assert store.load_model("NL", "xgboost_quantile") == b"nl-model"
        assert store.load_model("DE_LU", "xgboost_quantile") == b"de-model"


# ── is_stale ──────────────────────────────────────────────────────────────────

class TestParquetStoreIsStale:
    def test_missing_path_is_stale(self, tmp_cache_dir):
        import datetime
        store = ParquetStore(tmp_cache_dir)
        phantom = tmp_cache_dir / "does_not_exist.parquet"
        assert store.is_stale(phantom, ttl=datetime.timedelta(hours=24)) is True

    def test_freshly_written_is_not_stale(self, tmp_cache_dir):
        import datetime
        store = ParquetStore(tmp_cache_dir)
        ps = _make_price_series()
        store.save_prices(ps)
        # Locate the written file
        price_files = list((tmp_cache_dir / "prices").glob("**/*.parquet"))
        assert price_files, "No price parquet written"
        assert store.is_stale(price_files[0], ttl=datetime.timedelta(hours=24)) is False


# ── Atomic write ──────────────────────────────────────────────────────────────

class TestAtomicWrite:
    def test_write_creates_file(self, tmp_cache_dir):
        store = ParquetStore(tmp_cache_dir)
        ps = _make_price_series()
        store.save_prices(ps)
        price_files = list((tmp_cache_dir / "prices").rglob("*.parquet"))
        assert len(price_files) >= 1


# ── Save / Load Weather ──────────────────────────────────────────────────────

import numpy as np
from runeflow.domain.weather import WeatherSeries


def _make_weather_series(n: int = 48, zone: str = "NL") -> WeatherSeries:
    idx = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")
    idx.name = "date"
    df = pd.DataFrame(
        {
            "temperature_2m": np.ones(n) * 12.0,
            "wind_speed_10m": np.ones(n) * 5.0,
        },
        index=idx,
    )
    return WeatherSeries(
        locations=("nl",),
        df=df,
        source="test",
        fetched_at=pd.Timestamp.now("UTC"),
    )


class TestParquetStoreWeather:
    def test_save_and_load_roundtrip(self, tmp_cache_dir):
        store = ParquetStore(tmp_cache_dir)
        ws = _make_weather_series()
        store.save_weather(ws, zone="NL")

        loaded = store.load_weather("NL")
        assert loaded is not None
        assert "temperature_2m" in loaded.df.columns or not loaded.df.empty

    def test_load_missing_returns_none(self, tmp_cache_dir):
        store = ParquetStore(tmp_cache_dir)
        assert store.load_weather("NL") is None

    def test_save_merges_with_existing(self, tmp_cache_dir):
        store = ParquetStore(tmp_cache_dir)
        ws1 = _make_weather_series(n=24)
        idx2 = pd.date_range("2024-01-02", periods=24, freq="h", tz="UTC")
        idx2.name = "date"
        df2 = pd.DataFrame({"temperature_2m": [10.0] * 24, "wind_speed_10m": [3.0] * 24}, index=idx2)
        ws2 = WeatherSeries(locations=("nl",), df=df2, source="test", fetched_at=pd.Timestamp.now("UTC"))
        store.save_weather(ws1, zone="NL")
        store.save_weather(ws2, zone="NL")

        loaded = store.load_weather("NL")
        assert loaded is not None
        assert len(loaded.df) >= 48

    def test_load_with_date_range(self, tmp_cache_dir):
        store = ParquetStore(tmp_cache_dir)
        ws = _make_weather_series(n=72)
        store.save_weather(ws, zone="NL")
        start = datetime.date(2024, 1, 1)
        end = datetime.date(2024, 1, 2)
        loaded = store.load_weather("NL", start=start, end=end)
        assert loaded is not None

    def test_load_date_range_returns_none_when_empty(self, tmp_cache_dir):
        store = ParquetStore(tmp_cache_dir)
        ws = _make_weather_series(n=24)
        store.save_weather(ws, zone="NL")
        # Filter to a range with no data
        loaded = store.load_weather("NL", start=datetime.date(2020, 1, 1), end=datetime.date(2020, 1, 2))
        assert loaded is None


class TestParquetStoreForecastWeather:
    def test_save_deterministic_and_load(self, tmp_cache_dir):
        store = ParquetStore(tmp_cache_dir)
        ws = _make_weather_series()
        store.save_forecast_weather(ws, zone="NL")

        loaded = store.load_forecast_weather("NL")
        assert loaded is not None

    def test_load_forecast_weather_missing_returns_none(self, tmp_cache_dir):
        store = ParquetStore(tmp_cache_dir)
        assert store.load_forecast_weather("NL") is None

    def test_save_ensemble_member_and_load(self, tmp_cache_dir):
        store = ParquetStore(tmp_cache_dir)
        ws = _make_weather_series()
        store.save_forecast_weather(ws, zone="NL", member=0)

        loaded = store.load_forecast_weather_ensemble("NL", member=0)
        assert loaded is not None

    def test_load_ensemble_member_missing_returns_none(self, tmp_cache_dir):
        store = ParquetStore(tmp_cache_dir)
        assert store.load_forecast_weather_ensemble("NL", member=0) is None

    def test_is_forecast_weather_fresh_when_missing(self, tmp_cache_dir):
        store = ParquetStore(tmp_cache_dir)
        fresh = store.is_forecast_weather_fresh("NL", datetime.timedelta(hours=6), ["temperature_2m"])
        assert fresh is False

    def test_is_forecast_weather_fresh_with_schema_match(self, tmp_cache_dir):
        store = ParquetStore(tmp_cache_dir)
        ws = _make_weather_series()
        store.save_forecast_weather(ws, zone="NL")
        # Both columns are in the schema
        fresh = store.is_forecast_weather_fresh(
            "NL", datetime.timedelta(hours=6), ["temperature_2m"]
        )
        assert fresh is True

    def test_is_forecast_weather_fresh_schema_mismatch(self, tmp_cache_dir):
        store = ParquetStore(tmp_cache_dir)
        ws = _make_weather_series()
        store.save_forecast_weather(ws, zone="NL")
        # Request a column that was NOT saved → schema mismatch → stale
        fresh = store.is_forecast_weather_fresh(
            "NL", datetime.timedelta(hours=6), ["nonexistent_col"]
        )
        assert fresh is False

    def test_is_forecast_weather_fresh_ensemble_member(self, tmp_cache_dir):
        store = ParquetStore(tmp_cache_dir)
        ws = _make_weather_series()
        store.save_forecast_weather(ws, zone="NL", member=1)
        fresh = store.is_forecast_weather_fresh(
            "NL", datetime.timedelta(hours=6), ["temperature_2m"], member=1
        )
        assert fresh is True

    def test_is_forecast_weather_stale_old_file(self, tmp_cache_dir):
        """A file with a very old fetched_at timestamp should be stale."""
        import json
        store = ParquetStore(tmp_cache_dir)
        ws = _make_weather_series()
        store.save_forecast_weather(ws, zone="NL")
        # Overwrite meta to make it look old
        meta_path = store._weather_forecast_path("NL").with_suffix(".meta.json")
        old_ts = (pd.Timestamp.now("UTC") - pd.Timedelta(days=30)).isoformat()
        meta_path.write_text(json.dumps({"fetched_at": old_ts, "weather_schema": ["temperature_2m"]}))
        fresh = store.is_forecast_weather_fresh("NL", datetime.timedelta(hours=6), ["temperature_2m"])
        assert fresh is False


# ── Save / Load Generation ────────────────────────────────────────────────────

from runeflow.domain.generation import GenerationSeries


def _make_generation_series(n: int = 48, zone: str = "NL") -> GenerationSeries:
    idx = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")
    idx.name = "date"
    df = pd.DataFrame(
        {
            "load_forecast_mw": np.ones(n) * 10000.0,
            "solar_forecast_mw": np.abs(np.random.default_rng(0).standard_normal(n)) * 1000,
        },
        index=idx,
    )
    return GenerationSeries(
        zone=zone,
        df=df,
        source="test",
        fetched_at=pd.Timestamp.now("UTC"),
    )


class TestParquetStoreGeneration:
    def test_save_and_load_roundtrip(self, tmp_cache_dir):
        store = ParquetStore(tmp_cache_dir)
        gs = _make_generation_series()
        store.save_generation(gs)
        loaded = store.load_generation("NL")
        assert loaded is not None
        assert "load_forecast_mw" in loaded.df.columns or not loaded.df.empty

    def test_load_missing_returns_none(self, tmp_cache_dir):
        store = ParquetStore(tmp_cache_dir)
        assert store.load_generation("NL") is None

    def test_save_merges_with_existing(self, tmp_cache_dir):
        store = ParquetStore(tmp_cache_dir)
        gs1 = _make_generation_series(n=24)
        idx2 = pd.date_range("2024-01-02", periods=24, freq="h", tz="UTC")
        idx2.name = "date"
        df2 = pd.DataFrame({"load_forecast_mw": [9000.0] * 24}, index=idx2)
        gs2 = GenerationSeries(zone="NL", df=df2, source="test", fetched_at=pd.Timestamp.now("UTC"))
        store.save_generation(gs1)
        store.save_generation(gs2)
        loaded = store.load_generation("NL")
        assert loaded is not None
        assert len(loaded.df) >= 48

    def test_load_with_date_range(self, tmp_cache_dir):
        store = ParquetStore(tmp_cache_dir)
        gs = _make_generation_series(n=72)
        store.save_generation(gs)
        loaded = store.load_generation("NL", start=datetime.date(2024, 1, 1), end=datetime.date(2024, 1, 2))
        assert loaded is not None

    def test_load_date_range_no_data_returns_none(self, tmp_cache_dir):
        store = ParquetStore(tmp_cache_dir)
        gs = _make_generation_series(n=24)
        store.save_generation(gs)
        loaded = store.load_generation("NL", start=datetime.date(2025, 1, 1), end=datetime.date(2025, 1, 2))
        assert loaded is None


# ── Save / Load Supplemental ──────────────────────────────────────────────────

class TestParquetStoreSupplemental:
    def _make_supp_df(self, n: int = 48) -> pd.DataFrame:
        idx = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")
        return pd.DataFrame({"solar_mw": np.ones(n) * 200.0}, index=idx)

    def test_save_and_load_roundtrip(self, tmp_cache_dir):
        store = ParquetStore(tmp_cache_dir)
        df = self._make_supp_df()
        store.save_supplemental(df, zone="NL", key="ned")
        loaded = store.load_supplemental("NL", "ned")
        assert loaded is not None
        assert len(loaded) > 0

    def test_load_missing_returns_none(self, tmp_cache_dir):
        store = ParquetStore(tmp_cache_dir)
        assert store.load_supplemental("NL", "ned") is None

    def test_save_merges_and_deduplicates(self, tmp_cache_dir):
        store = ParquetStore(tmp_cache_dir)
        df = self._make_supp_df(n=24)
        store.save_supplemental(df, zone="NL", key="ned")
        store.save_supplemental(df, zone="NL", key="ned")  # same data again
        loaded = store.load_supplemental("NL", "ned")
        assert loaded is not None
        # No duplicates
        assert len(loaded) == 24


# ── Save / Load Forecast ──────────────────────────────────────────────────────

from runeflow.domain.forecast import ForecastPoint, ForecastResult


def _make_forecast_result(zone: str = "NL", n: int = 24) -> ForecastResult:
    ts = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")
    points = tuple(
        ForecastPoint(
            timestamp=t,
            prediction=50.0,
            lower=30.0,
            upper=70.0,
            uncertainty=40.0,
            model_agreement=0.9,
        )
        for t in ts
    )
    return ForecastResult(
        zone=zone,
        points=points,
        ensemble_members=pd.DataFrame(index=ts),
        model_predictions={},
        created_at=pd.Timestamp.now("UTC"),
        model_version="1.0",
    )


class TestParquetStoreForecast:
    def test_save_and_load_roundtrip(self, tmp_cache_dir):
        store = ParquetStore(tmp_cache_dir)
        result = _make_forecast_result()
        store.save_forecast(result)

        loaded = store.load_latest_forecast("NL")
        assert loaded is not None
        assert len(loaded.points) == 24
        assert loaded.zone == "NL"

    def test_load_missing_returns_none(self, tmp_cache_dir):
        store = ParquetStore(tmp_cache_dir)
        assert store.load_latest_forecast("NL") is None

    def test_roundtrip_preserves_predictions(self, tmp_cache_dir):
        store = ParquetStore(tmp_cache_dir)
        result = _make_forecast_result()
        store.save_forecast(result)
        loaded = store.load_latest_forecast("NL")
        assert loaded.points[0].prediction == pytest.approx(50.0)
        assert loaded.points[0].lower == pytest.approx(30.0)
        assert loaded.points[0].upper == pytest.approx(70.0)

    def test_save_with_ensemble_members(self, tmp_cache_dir):
        """save_forecast with non-empty ensemble_members hits the ensemble branch."""
        store = ParquetStore(tmp_cache_dir)
        ts = pd.date_range("2024-01-01", periods=12, freq="h", tz="UTC")
        points = tuple(
            ForecastPoint(timestamp=t, prediction=50.0, lower=30.0, upper=70.0,
                          uncertainty=40.0, model_agreement=0.9) for t in ts
        )
        ens = pd.DataFrame({"m0": [50.0] * 12, "m1": [52.0] * 12}, index=ts)
        result = ForecastResult(
            zone="NL", points=points, ensemble_members=ens,
            model_predictions={"xgboost": pd.Series([50.0] * 12, index=ts)},
            created_at=pd.Timestamp.now("UTC"), model_version="1.0",
        )
        store.save_forecast(result)
        loaded = store.load_latest_forecast("NL")
        assert loaded is not None
        assert len(loaded.points) == 12


# ── Save / Load Warmup Cache ──────────────────────────────────────────────────

class TestParquetStoreWarmupCache:
    def test_save_and_load_roundtrip(self, tmp_cache_dir):
        store = ParquetStore(tmp_cache_dir)
        idx = pd.date_range("2024-01-01", periods=48, freq="h", tz="UTC")
        df = pd.DataFrame({"feature_a": np.ones(48), "feature_b": np.arange(48)}, index=idx)
        df.index.name = "date"
        store.save_warmup_cache(df, zone="NL")

        loaded = store.load_warmup_cache("NL")
        assert loaded is not None
        assert len(loaded) == 48

    def test_load_missing_returns_none(self, tmp_cache_dir):
        store = ParquetStore(tmp_cache_dir)
        assert store.load_warmup_cache("NL") is None

    def test_save_without_named_index(self, tmp_cache_dir):
        """Covers the branch where df.index.name is falsy → no reset_index."""
        store = ParquetStore(tmp_cache_dir)
        idx = pd.date_range("2024-01-01", periods=10, freq="h", tz="UTC")
        df = pd.DataFrame({"x": range(10)}, index=idx)
        # No index name → df.index.name is None (falsy)
        assert df.index.name is None
        store.save_warmup_cache(df, zone="DE_LU")
        loaded = store.load_warmup_cache("DE_LU")
        assert loaded is not None


# ── Staleness Edge Cases ──────────────────────────────────────────────────────

class TestParquetStoreIsStaleEdgeCases:
    def test_stale_when_meta_missing_fetched_at_key(self, tmp_cache_dir):
        """Meta file exists but has no fetched_at → exception → returns True (stale)."""
        import json
        store = ParquetStore(tmp_cache_dir)
        ps = _make_price_series()
        store.save_prices(ps)
        # Find the meta file and overwrite with corrupt data (no fetched_at key)
        meta_files = list((tmp_cache_dir / "prices").rglob("*.meta.json"))
        assert meta_files
        meta_files[0].write_text(json.dumps({"zone": "NL", "source": "test"}))
        assert store.is_stale(meta_files[0].with_suffix(".parquet"), datetime.timedelta(hours=1)) is True

    def test_not_stale_for_fresh_weather(self, tmp_cache_dir):
        store = ParquetStore(tmp_cache_dir)
        ws = _make_weather_series()
        store.save_forecast_weather(ws, zone="NL")
        path = store._weather_forecast_path("NL")
        assert store.is_stale(path, datetime.timedelta(hours=12)) is False


# ── Internal Helpers ──────────────────────────────────────────────────────────

class TestParquetStoreInternals:
    def test_read_parquet_corrupt_file_returns_none(self, tmp_cache_dir):
        """_read_parquet returns None and logs a warning when file is unreadable."""
        store = ParquetStore(tmp_cache_dir)
        bad_parquet = tmp_cache_dir / "bad.parquet"
        bad_parquet.write_bytes(b"not a parquet file at all")
        result = store._read_parquet(bad_parquet)
        assert result is None

    def test_read_meta_corrupt_json_returns_none(self, tmp_cache_dir):
        """_read_meta returns None when the .meta.json is corrupt."""
        store = ParquetStore(tmp_cache_dir)
        meta = tmp_cache_dir / "bad.meta.json"
        meta.write_text("{not valid json}")
        result = store._read_meta(tmp_cache_dir / "bad.parquet")
        assert result is None

    def test_atomic_write_cleans_up_on_failure(self, tmp_cache_dir):
        """_atomic_write removes the temp file and re-raises when write_fn raises."""
        from runeflow.adapters.store.parquet import _atomic_write
        target = tmp_cache_dir / "target.txt"

        def _failing_write(tmp):
            tmp.write_text("partial")
            raise RuntimeError("deliberate failure")

        with pytest.raises(RuntimeError, match="deliberate failure"):
            _atomic_write(target, _failing_write)

        # Target should NOT exist (write was not completed)
        assert not target.exists()

    def test_filter_by_date_with_datetime_index(self, tmp_cache_dir):
        """_filter_by_date covers the DatetimeIndex branch (lines 462-466)."""
        store = ParquetStore(tmp_cache_dir)
        idx = pd.date_range("2024-01-01", periods=72, freq="h", tz="UTC")
        df = pd.DataFrame({"value": range(72)}, index=idx)
        start = datetime.date(2024, 1, 2)
        end = datetime.date(2024, 1, 3)
        filtered = store._filter_by_date(df, start, end)
        # Should contain only rows from Jan 2 and Jan 3
        assert len(filtered) > 0
        assert len(filtered) < 72

    def test_filter_by_date_empty_df(self, tmp_cache_dir):
        """_filter_by_date on an empty DataFrame returns empty immediately."""
        store = ParquetStore(tmp_cache_dir)
        df = pd.DataFrame({"value": []})
        result = store._filter_by_date(df, datetime.date(2024, 1, 1), datetime.date(2024, 1, 2))
        assert result.empty

    def test_is_stale_no_meta_file(self, tmp_cache_dir):
        """is_stale returns True when meta file does not exist alongside file."""
        import json
        store = ParquetStore(tmp_cache_dir)
        # Create the parquet file but no .meta.json
        p = tmp_cache_dir / "orphan.parquet"
        p.write_bytes(b"x")
        assert store.is_stale(p, datetime.timedelta(hours=1)) is True



# ── Additional coverage for parquet.py missing lines ──────────────────────────

class TestParquetAdditionalCoverage:
    """Targeted tests for parquet.py lines 154, 158, 376-377, 466."""

    @pytest.fixture
    def store(self, tmp_path):
        return ParquetStore(tmp_path)

    # ── Lines 154 & 158: is_forecast_weather_fresh edge-case branches ─────────

    def test_is_forecast_weather_fresh_branches_154_and_158(self, tmp_path):
        """Lines 154 and 158 triggered by patching is_stale to return False.

        When is_stale returns False (fresh), is_forecast_weather_fresh reads
        the meta.  Two sub-cases:
          - meta is None → line 154 (return False)
          - meta has empty weather_schema → line 158 (return False)
        """
        import json
        from unittest.mock import patch

        store2 = ParquetStore(tmp_path / "fresh")
        zone = "NL"
        path = store2._weather_forecast_path(zone)
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"temperature_2m": [10.0]}).to_parquet(path, index=False)

        # --- Sub-case 1: meta returns None (line 154) ---
        with patch.object(type(store2), "is_stale", return_value=False):
            with patch.object(store2, "_read_meta", return_value=None):
                r1 = store2.is_forecast_weather_fresh(
                    zone, ["temperature_2m"], datetime.timedelta(hours=1)
                )
        assert r1 is False  # line 154: if meta is None: return False

        # --- Sub-case 2: meta has empty schema (line 158) ---
        empty_schema_meta = {
            "fetched_at": pd.Timestamp.now("UTC").isoformat(),
            "weather_schema": [],
        }
        with patch.object(type(store2), "is_stale", return_value=False):
            with patch.object(store2, "_read_meta", return_value=empty_schema_meta):
                r2 = store2.is_forecast_weather_fresh(
                    zone, ["temperature_2m"], datetime.timedelta(hours=1)
                )
        assert r2 is False  # line 158: if not cached_schema: return False

    # ── Line 158: is_forecast_weather_fresh → empty cached_schema ─────────────

    # ── Lines 376-377: is_stale → except Exception: return True ───────────────

    def test_is_stale_malformed_fetched_at_returns_true(self, store, tmp_path):
        """Lines 376-377: fetched_at is not a valid timestamp → exception → return True."""
        import json
        from unittest.mock import patch

        zone = "NL"
        path = store._weather_forecast_path(zone)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"x")  # file must exist

        # Meta with malformed fetched_at
        bad_meta = {"fetched_at": "not-a-real-timestamp-xyz"}

        with patch.object(store, "_read_meta", return_value=bad_meta):
            result = store.is_stale(path, datetime.timedelta(hours=1))

        assert result is True  # exception branch (line 376-377) returns True

    # ── Line 466: _filter_by_date fall-through (no date col, no DatetimeIndex) ─

    def test_filter_by_date_no_date_column_no_datetimeindex(self, store):
        """Line 466: _filter_by_date with plain RangeIndex returns df unchanged."""
        import datetime as dt

        df = pd.DataFrame({"value": [1, 2, 3]})  # RangeIndex, no "date" column
        start = dt.date(2024, 1, 1)
        end = dt.date(2024, 12, 31)

        result = store._filter_by_date(df, start, end)

        # Fall-through: returns df unfiltered
        assert len(result) == 3
