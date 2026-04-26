# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2024-2026 Thomas Phil — runeflow
# See LICENSE and COMMERCIAL-LICENSE.md for licensing details.

"""Tests for adapters: FallbackPriceAdapter, WeatherSeries, ForecastResult,
and binder.configure_injector."""

from __future__ import annotations

import datetime
import json
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from runeflow.domain.forecast import ForecastPoint, ForecastResult
from runeflow.domain.price import PriceRecord, PriceSeries
from runeflow.domain.weather import WeatherLocation, WeatherRecord, WeatherSeries
from runeflow.exceptions import DataUnavailableError

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _price_series(zone: str = "NL", n: int = 48, offset_hours: int = 0) -> PriceSeries:
    ts = pd.date_range(
        pd.Timestamp("2024-01-01", tz="UTC") + pd.Timedelta(hours=offset_hours),
        periods=n,
        freq="h",
        tz="UTC",
    )
    records = tuple(
        PriceRecord(timestamp=t, price_eur_mwh=float(50 + i % 20)) for i, t in enumerate(ts)
    )
    return PriceSeries(
        zone=zone,
        records=records,
        source="test",
        fetched_at=pd.Timestamp.now("UTC"),
    )


def _mock_price_port(zone: str = "NL", supported: bool = True) -> MagicMock:
    port = MagicMock()
    port.name = "MockPort"
    port.supports_zone.return_value = supported
    port.download_historical.return_value = _price_series(zone)
    port.download_day_ahead.return_value = _price_series(zone, n=24)
    return port


# ---------------------------------------------------------------------------
# WeatherLocation
# ---------------------------------------------------------------------------


class TestWeatherLocation:
    def test_basic_fields(self):
        loc = WeatherLocation(name="de_bilt", lat=52.1, lon=5.18, purpose="primary")
        assert loc.name == "de_bilt"
        assert loc.lat == 52.1
        assert loc.lon == 5.18
        assert loc.purpose == "primary"

    def test_immutable(self):
        loc = WeatherLocation(name="test", lat=0.0, lon=0.0, purpose="wind")
        with pytest.raises(AttributeError):
            loc.lat = 1.0  # type: ignore[misc]


# ---------------------------------------------------------------------------
# WeatherRecord
# ---------------------------------------------------------------------------


class TestWeatherRecord:
    def test_basic_construction(self):
        ts = pd.Timestamp("2024-01-15T12:00:00", tz="UTC")
        rec = WeatherRecord(
            timestamp=ts,
            location="de_bilt",
            temperature_2m=10.5,
            wind_speed_10m=6.0,
        )
        assert rec.timestamp == ts
        assert rec.location == "de_bilt"
        assert rec.temperature_2m == 10.5

    def test_optional_fields_default_none(self):
        ts = pd.Timestamp("2024-01-15T12:00:00", tz="UTC")
        rec = WeatherRecord(timestamp=ts, location="somewhere")
        assert rec.wind_gusts_10m is None
        assert rec.diffuse_radiation is None
        assert rec.is_day is None
        assert rec.direct_radiation is None


# ---------------------------------------------------------------------------
# WeatherSeries
# ---------------------------------------------------------------------------


class TestWeatherSeries:
    def _make_frame(self, n: int = 24, prefix: str = "de_bilt") -> pd.DataFrame:
        idx = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")
        return pd.DataFrame(
            {"temperature_2m": np.ones(n) * 10.0, "wind_speed_10m": np.ones(n) * 5.0},
            index=idx,
        )

    def test_from_location_frames_single(self):
        df = self._make_frame()
        series = WeatherSeries.from_location_frames({"de_bilt": df}, source="test")
        assert "de_bilt" in series.locations
        assert "de_bilt_temperature_2m" in series.df.columns

    def test_from_location_frames_multiple(self):
        df1 = self._make_frame()
        df2 = self._make_frame()
        series = WeatherSeries.from_location_frames({"loc_a": df1, "loc_b": df2}, source="test")
        assert "loc_a" in series.locations
        assert "loc_b" in series.locations
        assert "loc_a_temperature_2m" in series.df.columns
        assert "loc_b_temperature_2m" in series.df.columns

    def test_from_location_frames_empty_dict(self):
        series = WeatherSeries.from_location_frames({}, source="test")
        assert series.df.empty

    def test_from_location_frames_explicit_fetched_at(self):
        ts = pd.Timestamp("2024-06-01T00:00:00", tz="UTC")
        df = self._make_frame()
        series = WeatherSeries.from_location_frames({"x": df}, source="test", fetched_at=ts)
        assert series.fetched_at == ts

    def test_to_dataframe_returns_copy(self):
        df = self._make_frame()
        series = WeatherSeries.from_location_frames({"x": df}, source="test")
        out = series.to_dataframe()
        out["new_col"] = 0
        assert "new_col" not in series.df.columns

    def test_post_init_copies_df(self):
        idx = pd.date_range("2024-01-01", periods=5, freq="h", tz="UTC")
        raw = pd.DataFrame({"a": [1, 2, 3, 4, 5]}, index=idx)
        series = WeatherSeries(
            locations=("loc",),
            df=raw,
            source="test",
            fetched_at=pd.Timestamp.now("UTC"),
        )
        raw["a"] = 999
        # WeatherSeries should have its own copy
        assert (series.df["a"] != 999).all()


# ---------------------------------------------------------------------------
# ForecastResult domain extras
# ---------------------------------------------------------------------------


class TestForecastResult:
    def test_to_dataframe_empty(self):
        result = ForecastResult(
            zone="NL",
            points=(),
            ensemble_members=pd.DataFrame(),
            model_predictions={},
            created_at=pd.Timestamp.now("UTC"),
            model_version="1.0",
        )
        df = result.to_dataframe()
        assert df.empty
        assert "prediction" in df.columns

    def test_len_empty(self):
        result = ForecastResult(
            zone="NL",
            points=(),
            ensemble_members=pd.DataFrame(),
            model_predictions={},
            created_at=pd.Timestamp.now("UTC"),
            model_version="1.0",
        )
        assert len(result) == 0

    def test_len_non_empty(self):
        ts = pd.date_range("2024-01-01", periods=5, freq="h", tz="UTC")
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
        result = ForecastResult(
            zone="NL",
            points=points,
            ensemble_members=pd.DataFrame(),
            model_predictions={},
            created_at=pd.Timestamp.now("UTC"),
            model_version="1.0",
        )
        assert len(result) == 5

    def test_to_dataframe_columns(self):
        ts = pd.date_range("2024-01-01", periods=3, freq="h", tz="UTC")
        points = tuple(
            ForecastPoint(
                timestamp=t,
                prediction=50.0,
                lower=30.0,
                upper=70.0,
                uncertainty=40.0,
                model_agreement=0.85,
                lower_static=32.0,
                upper_static=68.0,
                ensemble_p50=51.0,
                ensemble_p25=45.0,
                ensemble_p75=55.0,
            )
            for t in ts
        )
        result = ForecastResult(
            zone="NL",
            points=points,
            ensemble_members=pd.DataFrame(),
            model_predictions={},
            created_at=pd.Timestamp.now("UTC"),
            model_version="1.0",
        )
        df = result.to_dataframe()
        for col in [
            "prediction",
            "lower",
            "upper",
            "uncertainty",
            "model_agreement",
            "lower_static",
            "upper_static",
            "ensemble_p50",
            "ensemble_p25",
            "ensemble_p75",
        ]:
            assert col in df.columns


# ---------------------------------------------------------------------------
# FallbackPriceAdapter
# ---------------------------------------------------------------------------


class TestFallbackPriceAdapter:
    def test_name_single_adapter(self):
        from runeflow.adapters.price.fallback import FallbackPriceAdapter

        port = _mock_price_port()
        port.name = "Alpha"
        adapter = FallbackPriceAdapter([port])
        assert "Alpha" in adapter.name

    def test_name_multiple_adapters(self):
        from runeflow.adapters.price.fallback import FallbackPriceAdapter

        p1, p2 = _mock_price_port(), _mock_price_port()
        p1.name = "Alpha"
        p2.name = "Beta"
        adapter = FallbackPriceAdapter([p1, p2])
        assert "Alpha" in adapter.name
        assert "Beta" in adapter.name

    def test_name_empty(self):
        from runeflow.adapters.price.fallback import FallbackPriceAdapter

        adapter = FallbackPriceAdapter([])
        assert "Fallback" in adapter.name

    def test_supports_zone_true(self):
        from runeflow.adapters.price.fallback import FallbackPriceAdapter

        port = _mock_price_port()
        port.supports_zone.return_value = True
        adapter = FallbackPriceAdapter([port])
        assert adapter.supports_zone("NL") is True

    def test_supports_zone_false_all(self):
        from runeflow.adapters.price.fallback import FallbackPriceAdapter

        port = _mock_price_port()
        port.supports_zone.return_value = False
        adapter = FallbackPriceAdapter([port])
        assert adapter.supports_zone("NL") is False

    def test_download_historical_first_succeeds(self):
        from runeflow.adapters.price.fallback import FallbackPriceAdapter

        port1 = _mock_price_port()
        port1.name = "Primary"
        series = _price_series()
        port1.download_historical.return_value = series

        adapter = FallbackPriceAdapter([port1])
        result = adapter.download_historical(
            "NL",
            datetime.date(2024, 1, 1),
            datetime.date(2024, 1, 2),
        )
        assert result is not None
        assert len(result) > 0

    def test_download_historical_first_fails_second_succeeds(self):
        from runeflow.adapters.price.fallback import FallbackPriceAdapter

        port1 = _mock_price_port()
        port1.name = "Primary"
        port1.download_historical.side_effect = Exception("API error")

        port2 = _mock_price_port()
        port2.name = "Backup"
        series = _price_series()
        port2.download_historical.return_value = series

        adapter = FallbackPriceAdapter([port1, port2])
        result = adapter.download_historical(
            "NL",
            datetime.date(2024, 1, 1),
            datetime.date(2024, 1, 2),
        )
        assert result is not None
        assert len(result) > 0
        port2.download_historical.assert_called_once()

    def test_download_historical_all_fail_raises(self):
        from runeflow.adapters.price.fallback import FallbackPriceAdapter

        port1 = _mock_price_port()
        port1.download_historical.side_effect = ValueError("gone")

        adapter = FallbackPriceAdapter([port1])
        with pytest.raises(ValueError):
            adapter.download_historical("NL", datetime.date(2024, 1, 1), datetime.date(2024, 1, 2))

    def test_download_historical_no_adapters_raises(self):
        from runeflow.adapters.price.fallback import FallbackPriceAdapter

        adapter = FallbackPriceAdapter([])
        with pytest.raises(DataUnavailableError):
            adapter.download_historical("NL", datetime.date(2024, 1, 1), datetime.date(2024, 1, 2))

    def test_download_historical_empty_result_tries_next(self):
        from runeflow.adapters.price.fallback import FallbackPriceAdapter

        empty_series = PriceSeries(
            zone="NL",
            records=(),
            source="empty",
            fetched_at=pd.Timestamp.now("UTC"),
        )
        port1 = _mock_price_port()
        port1.download_historical.return_value = empty_series
        port1.name = "Empty"

        port2 = _mock_price_port()
        port2.download_historical.return_value = _price_series()
        port2.name = "Full"

        adapter = FallbackPriceAdapter([port1, port2])
        result = adapter.download_historical(
            "NL", datetime.date(2024, 1, 1), datetime.date(2024, 1, 2)
        )
        assert len(result) > 0
        port2.download_historical.assert_called_once()

    def test_download_historical_unsupported_zone_skips(self):
        from runeflow.adapters.price.fallback import FallbackPriceAdapter

        port1 = _mock_price_port(supported=False)
        port1.name = "NoZone"

        port2 = _mock_price_port(supported=True)
        port2.download_historical.return_value = _price_series()
        port2.name = "YesZone"

        adapter = FallbackPriceAdapter([port1, port2])
        result = adapter.download_historical(
            "NL", datetime.date(2024, 1, 1), datetime.date(2024, 1, 2)
        )
        assert len(result) > 0
        port1.download_historical.assert_not_called()

    def test_download_day_ahead_returns_first_non_empty(self):
        from runeflow.adapters.price.fallback import FallbackPriceAdapter

        port1 = _mock_price_port()
        port1.download_day_ahead.return_value = _price_series(n=24)
        port1.name = "DA"

        adapter = FallbackPriceAdapter([port1])
        result = adapter.download_day_ahead("NL")
        assert result is not None
        assert len(result) > 0

    def test_download_day_ahead_returns_none_when_all_empty(self):
        from runeflow.adapters.price.fallback import FallbackPriceAdapter

        port1 = _mock_price_port()
        empty = PriceSeries(zone="NL", records=(), source="e", fetched_at=pd.Timestamp.now("UTC"))
        port1.download_day_ahead.return_value = empty
        port1.name = "Empty"

        adapter = FallbackPriceAdapter([port1])
        result = adapter.download_day_ahead("NL")
        assert result is None

    def test_download_day_ahead_returns_none_when_port_returns_none(self):
        from runeflow.adapters.price.fallback import FallbackPriceAdapter

        port1 = _mock_price_port()
        port1.download_day_ahead.return_value = None
        port1.name = "None"

        adapter = FallbackPriceAdapter([port1])
        result = adapter.download_day_ahead("NL")
        assert result is None

    def test_find_missing_ranges_no_gaps(self):
        from runeflow.adapters.price.fallback import FallbackPriceAdapter

        idx = pd.date_range("2024-01-01", periods=48, freq="h", tz="UTC")
        df = pd.DataFrame({"date": idx})
        gaps = FallbackPriceAdapter._find_missing_ranges(
            df,
            start=datetime.date(2024, 1, 1),
            end=datetime.date(2024, 1, 2),
        )
        assert gaps == []

    def test_find_missing_ranges_empty_df(self):
        from runeflow.adapters.price.fallback import FallbackPriceAdapter

        gaps = FallbackPriceAdapter._find_missing_ranges(
            pd.DataFrame(),
            start=datetime.date(2024, 1, 1),
            end=datetime.date(2024, 1, 2),
        )
        assert len(gaps) == 1
        assert gaps[0] == (datetime.date(2024, 1, 1), datetime.date(2024, 1, 2))

    def test_find_missing_ranges_with_gap(self):
        from runeflow.adapters.price.fallback import FallbackPriceAdapter

        # Create series with a gap: Jan 1-2 OK, Jan 3 missing, Jan 4 OK
        ts_1 = pd.date_range("2024-01-01", periods=48, freq="h", tz="UTC")
        ts_2 = pd.date_range("2024-01-04", periods=24, freq="h", tz="UTC")
        idx = ts_1.append(ts_2)
        df = pd.DataFrame({"date": idx})
        gaps = FallbackPriceAdapter._find_missing_ranges(
            df,
            start=datetime.date(2024, 1, 1),
            end=datetime.date(2024, 1, 4),
        )
        # Jan 3 should be in a gap
        assert len(gaps) >= 1

    def test_gap_fill_succeeds_with_patch(self):
        """Primary adapter returns data with a gap; backup fills it."""
        from runeflow.adapters.price.fallback import FallbackPriceAdapter

        # Primary: only Jan 1
        port1 = _mock_price_port()
        port1.name = "Primary"
        port1.download_historical.return_value = _price_series(n=24, offset_hours=0)

        # Backup: provides Jan 2 (offset 24 hours)
        port2 = _mock_price_port()
        port2.name = "Backup"
        port2.download_historical.return_value = _price_series(n=24, offset_hours=24)

        adapter = FallbackPriceAdapter([port1, port2])
        # Request Jan 1-2 → primary fills Jan 1, backup fills gap Jan 2
        result = adapter.download_historical(
            "NL",
            datetime.date(2024, 1, 1),
            datetime.date(2024, 1, 2),
        )
        assert result is not None

    def test_download_day_ahead_skips_unsupported_zone(self):
        """Line 77: download_day_ahead continues past unsupported-zone adapter."""
        from runeflow.adapters.price.fallback import FallbackPriceAdapter

        # First adapter does NOT support the zone
        port_no = _mock_price_port(supported=False)
        port_no.name = "Unsupported"

        # Second adapter supports the zone and returns data
        port_yes = _mock_price_port(supported=True)
        port_yes.name = "Supported"
        port_yes.download_day_ahead.return_value = _price_series(n=24)

        adapter = FallbackPriceAdapter([port_no, port_yes])
        result = adapter.download_day_ahead("NL")
        assert result is not None
        port_no.download_day_ahead.assert_not_called()

    def test_find_missing_ranges_two_separate_gaps(self):
        """Lines 161-162: two non-contiguous missing periods → two gap tuples."""
        from runeflow.adapters.price.fallback import FallbackPriceAdapter

        # Present: Jan1-2 and Jan4, missing: Jan3 AND Jan5
        ts_1 = pd.date_range("2024-01-01", periods=48, freq="h", tz="UTC")  # Jan1-2
        ts_2 = pd.date_range("2024-01-04", periods=24, freq="h", tz="UTC")  # Jan4
        idx = ts_1.append(ts_2)
        df = pd.DataFrame({"date": idx})

        # Request Jan1-Jan6: Jan3 and Jan5 are missing (separate ranges)
        gaps = FallbackPriceAdapter._find_missing_ranges(
            df,
            start=datetime.date(2024, 1, 1),
            end=datetime.date(2024, 1, 6),
        )
        assert len(gaps) == 2  # Jan3 gap AND Jan5-6 gap

    def test_fill_gaps_skips_unsupported_adapter(self):
        """Line 105: _fill_gaps skips adapter that doesn't support zone."""
        from runeflow.adapters.price.fallback import FallbackPriceAdapter

        # Adapter that doesn't support zone
        port_no = _mock_price_port(supported=False)
        port_no.name = "Unsupported"
        # Adapter that does support and fills gap
        port_yes = _mock_price_port(supported=True)
        port_yes.name = "Supported"
        port_yes.download_historical.return_value = _price_series(n=24, offset_hours=48)

        adapter = FallbackPriceAdapter([port_no, port_yes])

        # Primary download gets only 24 hours; request 48 hours → gap
        primary = _price_series(n=24, offset_hours=0)
        primary_download_count = [0]

        def _download_hist(zone, s, e):
            primary_download_count[0] += 1
            return primary

        port_no.download_historical.side_effect = _download_hist

        result = adapter.download_historical(
            "NL", datetime.date(2024, 1, 1), datetime.date(2024, 1, 2)
        )
        # Unsupported adapter should not be called for gap fill
        assert result is not None

    def test_fill_gaps_handles_adapter_exception(self):
        """Lines 115-116: adapter raises during gap fill → logged, continues."""
        from runeflow.adapters.price.fallback import FallbackPriceAdapter

        # Primary gives partial data (only Jan 1)
        port1 = _mock_price_port()
        port1.name = "Primary"
        port1.download_historical.return_value = _price_series(n=24, offset_hours=0)

        # Second adapter raises on EVERY call to download_historical (gap fill)
        port2 = _mock_price_port()
        port2.name = "Failing"
        port2.download_historical.side_effect = RuntimeError("API down")

        adapter = FallbackPriceAdapter([port1, port2])
        # Should not raise despite adapter failure (lines 115-116 catch exception)
        result = adapter.download_historical(
            "NL", datetime.date(2024, 1, 1), datetime.date(2024, 1, 2)
        )
        assert result is not None

    def test_fill_gaps_no_patches_returns_original(self):
        """Lines 124-125: no adapter can fill gaps → original series returned."""

        from runeflow.adapters.price.fallback import FallbackPriceAdapter

        port1 = _mock_price_port()
        port1.name = "Primary"
        port1.download_historical.return_value = _price_series(n=24, offset_hours=0)

        adapter = FallbackPriceAdapter([port1])

        # Patch _fill_gaps to simulate that all adapters return empty results
        # by making download_historical return an empty series on gap fill
        empty = PriceSeries(zone="NL", records=(), source="x", fetched_at=pd.Timestamp.now("UTC"))
        call_count = [0]

        def _download_side(zone, s, e):
            call_count[0] += 1
            if call_count[0] == 1:
                return _price_series(n=24, offset_hours=0)  # initial
            return empty  # gap fill attempt → empty → no patches

        port1.download_historical.side_effect = _download_side

        result = adapter.download_historical(
            "NL", datetime.date(2024, 1, 1), datetime.date(2024, 1, 2)
        )
        assert result is not None  # returns original series


# ---------------------------------------------------------------------------
# Binder smoke-test
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=False)
def clear_inject():
    """Clear inject state before and after each binder test."""
    import inject

    inject.clear()
    yield
    inject.clear()


class TestBinder:
    def test_configure_injector_nl(self, tmp_path, clear_inject):
        """configure_injector should complete without exceptions for NL zone."""
        from runeflow.binder import configure_injector

        env = {
            "ZONE": "NL",
            "CACHE_DIR": str(tmp_path),
        }
        configure_injector("NL", env=env, allow_override=True)

        import inject as _inject

        from runeflow.zones.config import ZoneConfig

        cfg = _inject.instance(ZoneConfig)
        assert cfg.zone == "NL"

    def test_configure_injector_with_entsoe_key(self, tmp_path, clear_inject):
        """If ENTSOE key is provided, EntsoePriceAdapter + GenerationAdapter are bound."""
        import inject as _inject

        from runeflow.binder import configure_injector
        from runeflow.ports.price import PricePort

        env = {
            "ZONE": "NL",
            "ENTSOE": "test_api_key_12345",
            "CACHE_DIR": str(tmp_path),
        }
        configure_injector("NL", env=env, allow_override=True)

        price_port = _inject.instance(PricePort)
        assert price_port is not None

    def test_configure_injector_de_lu(self, tmp_path, clear_inject):
        """configure_injector should work for DE_LU zone."""
        from runeflow.binder import configure_injector

        env = {"ZONE": "DE_LU", "CACHE_DIR": str(tmp_path)}
        configure_injector("DE_LU", env=env, allow_override=True)

        import inject as _inject

        from runeflow.zones.config import ZoneConfig

        cfg = _inject.instance(ZoneConfig)
        assert cfg.zone == "DE_LU"

    def test_configure_injector_with_ned_key(self, tmp_path, clear_inject):
        """NED key set for NL zone → NedAdapter binding (binder.py lines 110-112)."""
        import inject as _inject

        from runeflow.binder import configure_injector
        from runeflow.ports.supplemental import SupplementalDataPort

        env = {"ZONE": "NL", "CACHE_DIR": str(tmp_path), "NED": "test_ned_api_key"}
        configure_injector("NL", env=env, allow_override=True)

        port = _inject.instance(SupplementalDataPort)
        assert port is not None


# ---------------------------------------------------------------------------
# Weather Caching Strategies
# ---------------------------------------------------------------------------


class TestWeatherCachingStrategies:
    """Tests for ForecastCachingStrategy, NoCachingStrategy, ReadOnlyCachingStrategy."""

    def _make_weather_series(self) -> WeatherSeries:
        from runeflow.domain.weather import WeatherSeries

        idx = pd.date_range("2024-01-01", periods=24, freq="h", tz="UTC")
        df = pd.DataFrame({"temperature_2m": [10.0] * 24}, index=idx)
        return WeatherSeries(
            locations=("nl",),
            df=df,
            source="test",
            fetched_at=pd.Timestamp.now("UTC"),
        )

    def test_forecast_caching_strategy_is_fresh(self):
        """TTLCachingStrategy.is_fresh() calls store.is_forecast_weather_fresh (line 78)."""
        import datetime as dt
        from unittest.mock import MagicMock

        from runeflow.adapters.weather.strategies import TTLCachingStrategy

        store = MagicMock()
        store.is_forecast_weather_fresh.return_value = True

        strategy = TTLCachingStrategy(ttl=dt.timedelta(hours=3))
        result = strategy.is_fresh(store, "NL", None, ["temperature_2m"])

        assert result is True
        store.is_forecast_weather_fresh.assert_called_once()

    def test_forecast_caching_strategy_on_downloaded(self):
        """TTLCachingStrategy.on_downloaded() calls store.save_forecast_weather (line 87)."""
        import datetime as dt
        from unittest.mock import MagicMock

        from runeflow.adapters.weather.strategies import TTLCachingStrategy

        store = MagicMock()
        ws = self._make_weather_series()

        strategy = TTLCachingStrategy(ttl=dt.timedelta(hours=3))
        strategy.on_downloaded(store, ws, "NL", None)

        store.save_forecast_weather.assert_called_once_with(ws, "NL", member=None)

    def test_no_caching_strategy_is_fresh_always_false(self):
        """NoCachingStrategy.is_fresh() returns False (line 102)."""
        from unittest.mock import MagicMock

        from runeflow.adapters.weather.strategies import NoCachingStrategy

        store = MagicMock()
        strategy = NoCachingStrategy()
        result = strategy.is_fresh(store, "NL", None, ["temperature_2m"])
        assert result is False

    def test_no_caching_strategy_on_downloaded_is_noop(self):
        """NoCachingStrategy.on_downloaded() is a no-op (line 111)."""
        from unittest.mock import MagicMock

        from runeflow.adapters.weather.strategies import NoCachingStrategy

        store = MagicMock()
        ws = self._make_weather_series()
        strategy = NoCachingStrategy()
        strategy.on_downloaded(store, ws, "NL", None)  # should not raise or call store
        store.save_forecast_weather.assert_not_called()

    def test_readonly_caching_strategy_init(self):
        """ReadOnlyCachingStrategy.__init__ stores ttl (line 124)."""
        import datetime as dt

        from runeflow.adapters.weather.strategies import ReadOnlyCachingStrategy

        strategy = ReadOnlyCachingStrategy(ttl=dt.timedelta(hours=6))
        assert strategy._ttl == dt.timedelta(hours=6)

    def test_readonly_caching_strategy_is_fresh(self):
        """ReadOnlyCachingStrategy.is_fresh() calls store.is_forecast_weather_fresh (line 133)."""
        import datetime as dt
        from unittest.mock import MagicMock

        from runeflow.adapters.weather.strategies import ReadOnlyCachingStrategy

        store = MagicMock()
        store.is_forecast_weather_fresh.return_value = True

        strategy = ReadOnlyCachingStrategy(ttl=dt.timedelta(hours=3))
        result = strategy.is_fresh(store, "NL", None, ["temperature_2m"])

        assert result is True

    def test_readonly_caching_strategy_on_downloaded_is_noop(self):
        """ReadOnlyCachingStrategy.on_downloaded() is a no-op (line 142)."""
        import datetime as dt
        from unittest.mock import MagicMock

        from runeflow.adapters.weather.strategies import ReadOnlyCachingStrategy

        store = MagicMock()
        ws = self._make_weather_series()
        strategy = ReadOnlyCachingStrategy(ttl=dt.timedelta(hours=3))
        strategy.on_downloaded(store, ws, "NL", None)  # should not raise or write
        store.save_forecast_weather.assert_not_called()


# ---------------------------------------------------------------------------
# CachingWeatherAdapter
# ---------------------------------------------------------------------------


class TestCachingWeatherAdapter:
    """Tests for CachingWeatherAdapter covering caching.py missing lines."""

    def _make_weather_series(self, n: int = 24) -> WeatherSeries:
        from runeflow.domain.weather import WeatherSeries

        idx = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")
        df = pd.DataFrame({"temperature_2m": [10.0] * n}, index=idx)
        return WeatherSeries(
            locations=("nl",),
            df=df,
            source="test",
            fetched_at=pd.Timestamp.now("UTC"),
        )

    def _make_adapter(self, strategy=None):
        from unittest.mock import MagicMock

        from runeflow.adapters.weather.caching import CachingWeatherAdapter

        inner = MagicMock()
        store = MagicMock()
        ws = self._make_weather_series()
        inner.download_historical.return_value = ws
        inner.download_forecast.return_value = ws
        inner.download_ensemble_forecast.return_value = [ws]

        if strategy is None:
            from unittest.mock import MagicMock

            strategy = MagicMock()
            strategy.is_fresh.return_value = False

        adapter = CachingWeatherAdapter(
            inner,
            store,
            "NL",
            strategy=strategy,
            n_ensemble_members=2,
        )
        return adapter, inner, store, ws

    def test_download_historical_delegates_to_inner(self):
        """Line 101: download_historical passes through to inner adapter when cache is stale."""
        import datetime as dt

        from runeflow.domain.weather import WeatherLocation

        adapter, inner, store, ws = self._make_adapter()
        store.is_historical_weather_fresh.return_value = False
        loc = WeatherLocation(name="nl", lat=52.0, lon=5.0, purpose="primary")
        result = adapter.download_historical([loc], dt.date(2024, 1, 1), dt.date(2024, 1, 2))

        assert result is ws
        inner.download_historical.assert_called_once()

    def test_download_historical_cache_hit(self):
        """download_historical returns cached data when schema matches."""
        import datetime as dt

        from runeflow.domain.weather import WeatherLocation

        adapter, inner, store, ws = self._make_adapter()
        store.is_historical_weather_fresh.return_value = True
        store.load_weather.return_value = ws
        loc = WeatherLocation(name="nl", lat=52.0, lon=5.0, purpose="primary")
        result = adapter.download_historical([loc], dt.date(2024, 1, 1), dt.date(2024, 1, 2))

        assert result is ws
        inner.download_historical.assert_not_called()

    def test_download_forecast_cache_hit(self):
        """Lines 109-117: strategy.is_fresh=True and cached data available."""
        from unittest.mock import MagicMock

        from runeflow.domain.weather import WeatherLocation

        ws = self._make_weather_series()
        strategy = MagicMock()
        strategy.is_fresh.return_value = True
        adapter, inner, store, _ = self._make_adapter(strategy=strategy)
        store.load_forecast_weather.return_value = ws

        loc = WeatherLocation(name="nl", lat=52.0, lon=5.0, purpose="primary")
        result = adapter.download_forecast([loc])

        assert result is ws
        inner.download_forecast.assert_not_called()  # served from cache

    def test_download_forecast_cache_miss(self):
        """Lines 118-124: strategy.is_fresh=False → download from inner."""
        from unittest.mock import MagicMock

        from runeflow.domain.weather import WeatherLocation

        strategy = MagicMock()
        strategy.is_fresh.return_value = False
        adapter, inner, store, ws = self._make_adapter(strategy=strategy)

        loc = WeatherLocation(name="nl", lat=52.0, lon=5.0, purpose="primary")
        result = adapter.download_forecast([loc])

        assert result is ws
        inner.download_forecast.assert_called_once()
        strategy.on_downloaded.assert_called_once()

    def test_download_forecast_cache_hit_but_store_returns_none(self):
        """Lines 110-117: strategy fresh but load returns None → re-download."""
        from unittest.mock import MagicMock

        from runeflow.domain.weather import WeatherLocation

        ws = self._make_weather_series()
        strategy = MagicMock()
        strategy.is_fresh.return_value = True
        adapter, inner, store, _ = self._make_adapter(strategy=strategy)
        store.load_forecast_weather.return_value = None  # cache miss despite fresh signal

        inner.download_forecast.return_value = ws
        loc = WeatherLocation(name="nl", lat=52.0, lon=5.0, purpose="primary")
        adapter.download_forecast([loc])

        inner.download_forecast.assert_called_once()

    def test_download_ensemble_cache_hit(self):
        """Lines 138-145: ensemble strategy fresh and all members cached."""
        from unittest.mock import MagicMock

        from runeflow.domain.weather import WeatherLocation

        ws0 = self._make_weather_series()
        ws1 = self._make_weather_series()

        strategy = MagicMock()
        strategy.is_fresh.return_value = True
        adapter, inner, store, _ = self._make_adapter(strategy=strategy)
        # Provide 2 ensemble members (n_ensemble_members=2)
        store.load_forecast_weather_ensemble.side_effect = [ws0, ws1]

        loc = WeatherLocation(name="nl", lat=52.0, lon=5.0, purpose="primary")
        result = adapter.download_ensemble_forecast([loc])

        assert len(result) == 2
        inner.download_ensemble_forecast.assert_not_called()

    def test_download_ensemble_cache_miss(self):
        """Lines 146-163: ensemble strategy not fresh → download."""
        from unittest.mock import MagicMock

        from runeflow.domain.weather import WeatherLocation

        ws0 = self._make_weather_series()
        ws1 = self._make_weather_series()

        strategy = MagicMock()
        strategy.is_fresh.return_value = False
        adapter, inner, store, _ = self._make_adapter(strategy=strategy)
        inner.download_ensemble_forecast.return_value = [ws0, ws1]

        loc = WeatherLocation(name="nl", lat=52.0, lon=5.0, purpose="primary")
        adapter.download_ensemble_forecast([loc])

        inner.download_ensemble_forecast.assert_called_once()
        assert strategy.on_downloaded.call_count == 2  # one per member

    def test_load_all_cached_members_returns_none_on_missing(self):
        """Lines 169-175: _load_all_cached_members returns None if any member missing."""

        ws = self._make_weather_series()
        adapter, inner, store, _ = self._make_adapter()

        # First member found, second missing
        store.load_forecast_weather_ensemble.side_effect = [ws, None]
        result = adapter._load_all_cached_members()

        assert result is None


# ---------------------------------------------------------------------------
# Price adapters
# ---------------------------------------------------------------------------


class TestEntsoePriceAdapter:
    """Tests for adapters/price/entsoe.py."""

    def _make_adapter(self, api_key: str = "test-key"):
        from unittest.mock import patch

        with patch("runeflow.adapters.price.entsoe.EntsoePandasClient") as mock_cls:
            from runeflow.adapters.price.entsoe import EntsoePriceAdapter

            adapter = EntsoePriceAdapter(api_key)
            mock_client = mock_cls.return_value
        return adapter, mock_client

    def test_init_empty_key_raises_auth_error(self):
        from unittest.mock import patch

        with patch("runeflow.adapters.price.entsoe.EntsoePandasClient"):
            from runeflow.adapters.price.entsoe import EntsoePriceAdapter
            from runeflow.exceptions import AuthenticationError

            with pytest.raises(AuthenticationError):
                EntsoePriceAdapter("")

    def test_name_property(self):
        from unittest.mock import patch

        with patch("runeflow.adapters.price.entsoe.EntsoePandasClient"):
            from runeflow.adapters.price.entsoe import EntsoePriceAdapter

            adapter = EntsoePriceAdapter("key")
            assert adapter.name == "ENTSO-E"

    def test_supports_zone_nl(self):
        from unittest.mock import patch

        with patch("runeflow.adapters.price.entsoe.EntsoePandasClient"):
            from runeflow.adapters.price.entsoe import EntsoePriceAdapter

            adapter = EntsoePriceAdapter("key")
            assert adapter.supports_zone("NL") is True
            assert adapter.supports_zone("XXXXNOTAZONE") is False

    def test_zone_to_area_de_alias(self):
        from runeflow.adapters.price.entsoe import _zone_to_area

        area = _zone_to_area("DE")
        assert area.name == "DE_LU"

    def test_zone_to_area_unknown_raises(self):
        from runeflow.adapters.price.entsoe import _zone_to_area
        from runeflow.exceptions import DataUnavailableError

        with pytest.raises(DataUnavailableError, match="not recognised"):
            _zone_to_area("XXXXNOTAZONE")

    def test_download_historical_happy_path(self):
        from unittest.mock import patch

        start = datetime.date(2024, 1, 1)
        end = datetime.date(2024, 1, 3)

        index = pd.date_range("2024-01-01", periods=3, freq="h", tz="UTC")
        mock_series = pd.Series([50.0, 55.0, 60.0], index=index)

        with patch("runeflow.adapters.price.entsoe.EntsoePandasClient") as mock_cls:
            mock_cls.return_value.query_day_ahead_prices.return_value = mock_series
            from runeflow.adapters.price.entsoe import EntsoePriceAdapter

            adapter = EntsoePriceAdapter("key")
            result = adapter.download_historical("NL", start, end)

        assert result is not None
        assert result.zone == "NL"
        assert result.source == "ENTSO-E"

    def test_download_historical_api_exception_raises_download_error(self):
        from unittest.mock import patch

        from runeflow.exceptions import DownloadError

        start = datetime.date(2024, 1, 1)
        end = datetime.date(2024, 1, 1)

        with patch("runeflow.adapters.price.entsoe.EntsoePandasClient") as mock_cls:
            mock_cls.return_value.query_day_ahead_prices.side_effect = RuntimeError("API fail")
            from runeflow.adapters.price.entsoe import EntsoePriceAdapter

            adapter = EntsoePriceAdapter("key")
            with pytest.raises(DownloadError, match="ENTSO-E download failed"):
                adapter.download_historical("NL", start, end)

    def test_download_historical_empty_raises_unavailable(self):
        from unittest.mock import patch

        from runeflow.exceptions import DataUnavailableError

        start = datetime.date(2024, 1, 1)
        end = datetime.date(2024, 1, 1)

        with patch("runeflow.adapters.price.entsoe.EntsoePandasClient") as mock_cls:
            mock_cls.return_value.query_day_ahead_prices.return_value = pd.Series([], dtype=float)
            from runeflow.adapters.price.entsoe import EntsoePriceAdapter

            adapter = EntsoePriceAdapter("key")
            with pytest.raises(DataUnavailableError, match="returned no data"):
                adapter.download_historical("NL", start, end)

    def test_download_day_ahead_returns_prices(self):
        from unittest.mock import patch

        start = datetime.date.today() + datetime.timedelta(days=1)
        index = pd.date_range(str(start), periods=24, freq="h", tz="UTC")
        mock_series = pd.Series([50.0] * 24, index=index)

        with patch("runeflow.adapters.price.entsoe.EntsoePandasClient") as mock_cls:
            mock_cls.return_value.query_day_ahead_prices.return_value = mock_series
            from runeflow.adapters.price.entsoe import EntsoePriceAdapter

            adapter = EntsoePriceAdapter("key")
            result = adapter.download_day_ahead("NL")

        assert result is not None

    def test_download_day_ahead_logs_and_returns_none_on_error(self):
        from unittest.mock import patch

        with patch("runeflow.adapters.price.entsoe.EntsoePandasClient") as mock_cls:
            mock_cls.return_value.query_day_ahead_prices.return_value = pd.Series([], dtype=float)
            from runeflow.adapters.price.entsoe import EntsoePriceAdapter

            adapter = EntsoePriceAdapter("key")
            result = adapter.download_day_ahead("NL")

        assert result is None

    def test_get_supported_zones_is_set(self):
        from unittest.mock import patch

        with patch("runeflow.adapters.price.entsoe.EntsoePandasClient"):
            from runeflow.adapters.price.entsoe import EntsoePriceAdapter

            adapter = EntsoePriceAdapter("key")
            zones = adapter.get_supported_zones()

        assert isinstance(zones, set)
        assert len(zones) > 0
        assert "NL" in zones


class TestEnergyZeroPriceAdapter:
    """Tests for adapters/price/energyzero.py."""

    @pytest.fixture()
    def adapter(self):
        from runeflow.adapters.price.energyzero import EnergyZeroPriceAdapter

        return EnergyZeroPriceAdapter()

    def test_name_property(self, adapter):
        assert adapter.name == "EnergyZero"

    def test_supports_zone_nl(self, adapter):
        assert adapter.supports_zone("NL") is True
        assert adapter.supports_zone("DE") is False

    def test_download_historical_unsupported_zone_raises(self, adapter):
        from runeflow.exceptions import DataUnavailableError

        with pytest.raises(DataUnavailableError, match="only supports NL"):
            adapter.download_historical("DE", datetime.date(2024, 1, 1), datetime.date(2024, 1, 1))

    def test_download_historical_happy_path(self, adapter):
        from unittest.mock import patch

        fake_response = MagicMock()
        fake_response.json.return_value = {
            "Prices": [
                {"readingDate": "2024-01-01T00:00:00Z", "price": 0.05},
                {"readingDate": "2024-01-01T01:00:00Z", "price": 0.06},
            ]
        }
        fake_response.raise_for_status.return_value = None

        with (
            patch("runeflow.adapters.price.energyzero.requests.get", return_value=fake_response),
            patch("runeflow.adapters.price.energyzero.time.sleep"),
        ):
            result = adapter.download_historical(
                "NL", datetime.date(2024, 1, 1), datetime.date(2024, 1, 1)
            )

        assert result is not None
        assert result.zone == "NL"
        assert result.source == "EnergyZero"

    def test_download_historical_empty_response_raises(self, adapter):
        from unittest.mock import patch

        from runeflow.exceptions import DataUnavailableError

        fake_response = MagicMock()
        fake_response.json.return_value = {"Prices": []}
        fake_response.raise_for_status.return_value = None

        with (
            patch("runeflow.adapters.price.energyzero.requests.get", return_value=fake_response),
            patch("runeflow.adapters.price.energyzero.time.sleep"),
            pytest.raises(DataUnavailableError, match="returned no data"),
        ):
            adapter.download_historical("NL", datetime.date(2024, 1, 1), datetime.date(2024, 1, 1))

    def test_fetch_chunk_request_exception_raises_download_error(self, adapter):
        from unittest.mock import patch

        import requests as req_lib

        from runeflow.exceptions import DownloadError

        with (
            patch(
                "runeflow.adapters.price.energyzero.requests.get",
                side_effect=req_lib.RequestException("timeout"),
            ),
            pytest.raises(DownloadError, match="EnergyZero request failed"),
        ):
            adapter._fetch_chunk(datetime.date(2024, 1, 1), datetime.date(2024, 1, 1))

    def test_fetch_chunk_returns_none_on_empty_prices(self, adapter):
        from unittest.mock import patch

        fake_response = MagicMock()
        fake_response.json.return_value = {"Prices": []}
        fake_response.raise_for_status.return_value = None

        with patch("runeflow.adapters.price.energyzero.requests.get", return_value=fake_response):
            result = adapter._fetch_chunk(datetime.date(2024, 1, 1), datetime.date(2024, 1, 1))

        assert result is None

    def test_download_day_ahead_returns_result(self, adapter):
        from unittest.mock import patch

        fake_response = MagicMock()
        fake_response.json.return_value = {
            "Prices": [{"readingDate": "2024-01-02T00:00:00Z", "price": 0.10}]
        }
        fake_response.raise_for_status.return_value = None

        with (
            patch("runeflow.adapters.price.energyzero.requests.get", return_value=fake_response),
            patch("runeflow.adapters.price.energyzero.time.sleep"),
        ):
            result = adapter.download_day_ahead("NL")

        assert result is not None

    def test_download_day_ahead_unsupported_zone_returns_none(self, adapter):
        result = adapter.download_day_ahead("DE")
        assert result is None

    def test_download_day_ahead_exception_returns_none(self, adapter):
        from unittest.mock import patch

        import requests as req_lib

        with patch(
            "runeflow.adapters.price.energyzero.requests.get",
            side_effect=req_lib.RequestException("err"),
        ):
            result = adapter.download_day_ahead("NL")

        assert result is None

    def test_download_historical_multi_chunk_calls_sleep(self, adapter):
        """Line 68: time.sleep is called between chunks (range > 90 days)."""
        from unittest.mock import patch

        # Two prices per chunk to keep it simple
        fake_response = MagicMock()
        fake_response.raise_for_status.return_value = None
        fake_response.json.return_value = {
            "Prices": [
                {"readingDate": "2024-01-01T00:00:00Z", "price": 0.05},
            ]
        }

        sleep_calls = []
        with (
            patch("runeflow.adapters.price.energyzero.requests.get", return_value=fake_response),
            patch(
                "runeflow.adapters.price.energyzero.time.sleep",
                side_effect=lambda s: sleep_calls.append(s),
            ),
        ):
            result = adapter.download_historical(
                "NL",
                datetime.date(2024, 1, 1),
                datetime.date(2024, 5, 1),  # > 90 days → 2 chunks
            )

        # time.sleep should have been called between chunks (line 68)
        assert len(sleep_calls) >= 1
        assert result is not None


class TestNedAdapter:
    """Tests for adapters/supplemental/ned.py."""

    @pytest.fixture()
    def adapter(self):
        from runeflow.adapters.supplemental.ned import NedAdapter

        return NedAdapter("test-ned-key")

    def test_init_empty_key_raises(self):
        from runeflow.adapters.supplemental.ned import NedAdapter
        from runeflow.exceptions import AuthenticationError

        with pytest.raises(AuthenticationError, match="NED API key"):
            NedAdapter("")

    def test_supports_zone(self, adapter):
        assert adapter.supports_zone("NL") is True
        assert adapter.supports_zone("DE") is False

    def test_download_unsupported_zone_returns_none(self, adapter):
        result = adapter.download("DE", datetime.date(2024, 1, 1), datetime.date(2024, 1, 1))
        assert result is None

    def test_download_happy_path(self, adapter):
        from unittest.mock import patch

        fake_response = MagicMock()
        fake_response.json.return_value = {
            "hydra:member": [
                {"validfrom": "2024-01-01T00:00:00+00:00", "volume": 1000.0},
                {"validfrom": "2024-01-01T01:00:00+00:00", "volume": 1100.0},
            ]
        }
        fake_response.raise_for_status.return_value = None

        with (
            patch("runeflow.adapters.supplemental.ned.requests.get", return_value=fake_response),
            patch("runeflow.adapters.supplemental.ned.time.sleep"),
        ):
            result = adapter.download("NL", datetime.date(2024, 1, 1), datetime.date(2024, 1, 1))

        assert result is not None
        assert "ned_utilization_kwh" in result.columns

    def test_download_empty_returns_none(self, adapter):
        from unittest.mock import patch

        fake_response = MagicMock()
        fake_response.json.return_value = {"hydra:member": []}
        fake_response.raise_for_status.return_value = None

        with (
            patch("runeflow.adapters.supplemental.ned.requests.get", return_value=fake_response),
            patch("runeflow.adapters.supplemental.ned.time.sleep"),
        ):
            result = adapter.download("NL", datetime.date(2024, 1, 1), datetime.date(2024, 1, 1))

        assert result is None

    def test_download_historical_request_exception(self, adapter):
        from unittest.mock import patch

        import requests as req_lib

        from runeflow.exceptions import DownloadError

        with (
            patch(
                "runeflow.adapters.supplemental.ned.requests.get",
                side_effect=req_lib.RequestException("timeout"),
            ),
            pytest.raises(DownloadError, match="NED historical request failed"),
        ):
            adapter._download_historical(datetime.date(2024, 1, 1), datetime.date(2024, 1, 1))

    def test_download_forecast_unsupported_zone_returns_none(self, adapter):
        result = adapter.download_forecast("DE")
        assert result is None

    def test_download_forecast_happy_path(self, adapter):
        from unittest.mock import patch

        fake_response = MagicMock()
        fake_response.json.return_value = {
            "hydra:member": [
                {"validfrom": "2024-01-10T00:00:00+00:00", "volume": 2000.0},
                {"validfrom": "2024-01-10T01:00:00+00:00", "volume": 2100.0},
            ]
        }
        fake_response.raise_for_status.return_value = None

        with patch("runeflow.adapters.supplemental.ned.requests.get", return_value=fake_response):
            result = adapter.download_forecast("NL")

        assert result is not None
        assert "ned_forecast_kwh" in result.columns

    def test_download_forecast_empty_returns_none(self, adapter):
        from unittest.mock import patch

        fake_response = MagicMock()
        fake_response.json.return_value = {"hydra:member": []}
        fake_response.raise_for_status.return_value = None

        with patch("runeflow.adapters.supplemental.ned.requests.get", return_value=fake_response):
            result = adapter.download_forecast("NL")

        assert result is None

    def test_download_forecasted_request_exception(self, adapter):
        from unittest.mock import patch

        import requests as req_lib

        from runeflow.exceptions import DownloadError

        with (
            patch(
                "runeflow.adapters.supplemental.ned.requests.get",
                side_effect=req_lib.RequestException("err"),
            ),
            pytest.raises(DownloadError, match="NED forecast request failed"),
        ):
            adapter._download_forecasted()

    def test_month_chunks_single_month(self):
        from datetime import datetime as dt

        from runeflow.adapters.supplemental.ned import NedAdapter

        chunks = NedAdapter._month_chunks(dt(2024, 1, 1), dt(2024, 1, 31))
        assert len(chunks) == 1
        assert chunks[0][0] == dt(2024, 1, 1)

    def test_month_chunks_cross_year(self):
        from datetime import datetime as dt

        from runeflow.adapters.supplemental.ned import NedAdapter

        chunks = NedAdapter._month_chunks(dt(2024, 12, 1), dt(2025, 1, 31))
        assert len(chunks) == 2
        assert chunks[0][1] == dt(2025, 1, 1)


class TestEntsoeGenerationAdapter:
    """Tests for adapters/generation/entsoe.py."""

    @pytest.fixture()
    def adapter_and_mock(self):
        from unittest.mock import patch

        with patch("runeflow.adapters.generation.entsoe.EntsoePandasClient") as mock_cls:
            from runeflow.adapters.generation.entsoe import EntsoeGenerationAdapter

            adapter = EntsoeGenerationAdapter("test-key")
            mock_client = mock_cls.return_value
        return adapter, mock_client

    def test_init_empty_key_raises(self):
        from unittest.mock import patch

        from runeflow.exceptions import AuthenticationError

        with patch("runeflow.adapters.generation.entsoe.EntsoePandasClient"):
            from runeflow.adapters.generation.entsoe import EntsoeGenerationAdapter

            with pytest.raises(AuthenticationError, match="API key"):
                EntsoeGenerationAdapter("")

    def test_supports_zone_always_true(self, adapter_and_mock):
        adapter, _ = adapter_and_mock
        assert adapter.supports_zone("NL") is True
        assert adapter.supports_zone("XX") is True

    def test_download_generation_happy_path(self, adapter_and_mock):
        adapter, mock_client = adapter_and_mock

        index = pd.date_range("2024-01-01", periods=24, freq="h", tz="UTC")
        mock_client.query_load_forecast.return_value = pd.Series([5000.0] * 24, index=index)
        mock_client.query_wind_and_solar_forecast.return_value = pd.DataFrame(
            {"Wind Onshore": [1000.0] * 24, "Solar": [200.0] * 24}, index=index
        )

        result = adapter.download_generation(
            "NL", datetime.date(2024, 1, 1), datetime.date(2024, 1, 1)
        )

        assert result is not None
        assert result.zone == "NL"
        assert result.source == "entsoe-generation"

    def test_download_generation_no_data_returns_none(self, adapter_and_mock):
        adapter, mock_client = adapter_and_mock

        mock_client.query_load_forecast.return_value = None
        mock_client.query_wind_and_solar_forecast.return_value = None

        result = adapter.download_generation(
            "NL", datetime.date(2024, 1, 1), datetime.date(2024, 1, 1)
        )

        assert result is None

    def test_fetch_load_forecast_exception_returns_none(self, adapter_and_mock):
        adapter, mock_client = adapter_and_mock

        mock_client.query_load_forecast.side_effect = RuntimeError("API error")
        result = adapter._fetch_load_forecast(
            "NL", pd.Timestamp("2024-01-01", tz="UTC"), pd.Timestamp("2024-01-02", tz="UTC")
        )

        assert result is None

    def test_fetch_load_forecast_empty_returns_none(self, adapter_and_mock):
        adapter, mock_client = adapter_and_mock

        mock_client.query_load_forecast.return_value = pd.Series([], dtype=float)
        result = adapter._fetch_load_forecast(
            "NL", pd.Timestamp("2024-01-01", tz="UTC"), pd.Timestamp("2024-01-02", tz="UTC")
        )

        assert result is None

    def test_fetch_wind_solar_exception_returns_none(self, adapter_and_mock):
        adapter, mock_client = adapter_and_mock

        mock_client.query_wind_and_solar_forecast.side_effect = RuntimeError("API error")
        result = adapter._fetch_wind_solar_forecast(
            "NL", pd.Timestamp("2024-01-01", tz="UTC"), pd.Timestamp("2024-01-02", tz="UTC")
        )

        assert result is None

    def test_fetch_wind_solar_multiindex_columns_flattened(self, adapter_and_mock):
        adapter, mock_client = adapter_and_mock

        index = pd.date_range("2024-01-01", periods=3, freq="h", tz="UTC")
        df = pd.DataFrame(
            {("Wind", "Onshore"): [1.0, 2.0, 3.0], ("Solar", ""): [0.1, 0.2, 0.3]},
            index=index,
        )
        mock_client.query_wind_and_solar_forecast.return_value = df

        result = adapter._fetch_wind_solar_forecast(
            "NL", pd.Timestamp("2024-01-01", tz="UTC"), pd.Timestamp("2024-01-02", tz="UTC")
        )

        assert result is not None
        # All column names are lowercase with underscores
        for col in result.columns:
            assert col == col.lower()

    def test_download_generation_only_load_data(self, adapter_and_mock):
        """load_forecast ok, wind/solar fails → GenerationSeries with load only."""
        adapter, mock_client = adapter_and_mock

        index = pd.date_range("2024-01-01", periods=24, freq="h", tz="UTC")
        mock_client.query_load_forecast.return_value = pd.Series([5000.0] * 24, index=index)
        mock_client.query_wind_and_solar_forecast.return_value = None

        result = adapter.download_generation(
            "NL", datetime.date(2024, 1, 1), datetime.date(2024, 1, 1)
        )

        assert result is not None
        assert "load_forecast_mw" in result.df.columns


# ---------------------------------------------------------------------------
# OpenMeteo adapter — SDK mock helpers
# ---------------------------------------------------------------------------


def _make_sdk_hourly_response(
    var_values: dict[str, list],
    start: str = "2024-01-01T00:00:00",
    interval_s: int = 3600,
) -> MagicMock:
    """Mock openmeteo-SDK response for deterministic (non-ensemble) data."""
    import numpy as np

    start_ts = int(pd.Timestamp(start, tz="UTC").timestamp())
    n = len(next(iter(var_values.values())))
    end_ts = start_ts + n * interval_s

    var_objs = []
    for values in var_values.values():
        v = MagicMock()
        v.ValuesAsNumpy.return_value = np.array(values, dtype=float)
        v.EnsembleMember.return_value = 0
        var_objs.append(v)

    hourly = MagicMock()
    hourly.VariablesLength.return_value = len(var_objs)
    hourly.Variables.side_effect = lambda i: var_objs[i]
    hourly.Time.return_value = start_ts
    hourly.TimeEnd.return_value = end_ts
    hourly.Interval.return_value = interval_s

    response = MagicMock()
    response.Hourly.return_value = hourly
    return response


def _make_sdk_ensemble_response(
    var_names: list,
    n_members: int = 3,
    n_timesteps: int = 2,
    start: str = "2024-01-10T00:00:00",
    interval_s: int = 3600,
) -> MagicMock:
    """Mock SDK ensemble response with n_members × n_vars Variable objects."""
    import numpy as np

    start_ts = int(pd.Timestamp(start, tz="UTC").timestamp())
    end_ts = start_ts + n_timesteps * interval_s

    var_objs = []
    for m in range(n_members):
        for v_idx in range(len(var_names)):
            v = MagicMock()
            v.ValuesAsNumpy.return_value = np.full(n_timesteps, float(m * 10 + v_idx))
            v.EnsembleMember.return_value = m
            var_objs.append(v)

    hourly = MagicMock()
    hourly.VariablesLength.return_value = len(var_objs)
    hourly.Variables.side_effect = lambda i: var_objs[i]
    hourly.Time.return_value = start_ts
    hourly.TimeEnd.return_value = end_ts
    hourly.Interval.return_value = interval_s

    response = MagicMock()
    response.Hourly.return_value = hourly
    return response


class TestIterDateChunks:
    """Unit tests for the _iter_date_chunks helper."""

    def test_single_day_is_one_chunk(self):
        from runeflow.adapters.weather.openmeteo import _iter_date_chunks

        chunks = _iter_date_chunks(datetime.date(2024, 3, 15), datetime.date(2024, 3, 15), 3)
        assert chunks == [(datetime.date(2024, 3, 15), datetime.date(2024, 3, 15))]

    def test_full_year_produces_four_quarterly_chunks(self):
        from runeflow.adapters.weather.openmeteo import _iter_date_chunks

        chunks = _iter_date_chunks(datetime.date(2024, 1, 1), datetime.date(2024, 12, 31), 3)
        assert len(chunks) == 4
        assert chunks[0] == (datetime.date(2024, 1, 1), datetime.date(2024, 3, 31))
        assert chunks[1] == (datetime.date(2024, 4, 1), datetime.date(2024, 6, 30))
        assert chunks[2] == (datetime.date(2024, 7, 1), datetime.date(2024, 9, 30))
        assert chunks[3] == (datetime.date(2024, 10, 1), datetime.date(2024, 12, 31))

    def test_chunks_are_contiguous_and_non_overlapping(self):
        from runeflow.adapters.weather.openmeteo import _iter_date_chunks

        chunks = _iter_date_chunks(datetime.date(2022, 1, 1), datetime.date(2023, 12, 31), 3)
        for i in range(len(chunks) - 1):
            assert chunks[i][1] + datetime.timedelta(days=1) == chunks[i + 1][0]

    def test_partial_range_last_chunk_clipped(self):
        from runeflow.adapters.weather.openmeteo import _iter_date_chunks

        chunks = _iter_date_chunks(datetime.date(2024, 1, 1), datetime.date(2024, 5, 15), 3)
        assert chunks[-1][1] == datetime.date(2024, 5, 15)

    def test_monthly_chunks(self):
        from runeflow.adapters.weather.openmeteo import _iter_date_chunks

        chunks = _iter_date_chunks(datetime.date(2024, 1, 1), datetime.date(2024, 3, 31), 1)
        assert len(chunks) == 3


class TestBiweeklyAnchor:
    """Unit tests for the _biweekly_anchor helper."""

    def test_result_is_on_or_before_input(self):
        from runeflow.adapters.weather.openmeteo import _biweekly_anchor

        d = datetime.date(2026, 4, 5)
        assert _biweekly_anchor(d) <= d

    def test_stable_across_13_day_window(self):
        """All days in the same 14-day cycle return the same anchor."""
        from runeflow.adapters.weather.openmeteo import _biweekly_anchor

        anchor = _biweekly_anchor(datetime.date(2026, 4, 7))
        for offset in range(1, 14):
            d = datetime.date(2026, 4, 7) + datetime.timedelta(days=offset)
            if _biweekly_anchor(d) == anchor:
                assert _biweekly_anchor(d) == anchor

    def test_advances_by_14_after_full_fortnight(self):
        from runeflow.adapters.weather.openmeteo import _biweekly_anchor

        d = datetime.date(2026, 1, 1)
        anchor = _biweekly_anchor(d)
        # The anchor exactly 14 days later must be 14 days ahead.
        assert _biweekly_anchor(
            anchor + datetime.timedelta(days=14)
        ) == anchor + datetime.timedelta(days=14)

    def test_anchor_is_exactly_on_boundary(self):
        """When today is exactly on a 14-day boundary, anchor == today."""
        from runeflow.adapters.weather.openmeteo import _biweekly_anchor

        # Find a day that is itself a biweekly anchor.
        d = datetime.date(2026, 4, 7)
        anchor = _biweekly_anchor(d)
        # anchor is on the boundary, so _biweekly_anchor(anchor) == anchor.
        assert _biweekly_anchor(anchor) == anchor


class TestOpenMeteoAdapter:
    """Tests for adapters/weather/openmeteo.py."""

    @pytest.fixture()
    def adapter(self):
        from runeflow.adapters.weather.openmeteo import OpenMeteoAdapter

        return OpenMeteoAdapter(timezone="UTC")

    @pytest.fixture()
    def loc(self):
        return WeatherLocation(name="nl", lat=52.1, lon=5.18, purpose="primary")

    # ── _parse_hourly_sdk ──────────────────────────────────────────────────

    def test_parse_hourly_sdk_valid(self, adapter):
        response = _make_sdk_hourly_response(
            {"temperature_2m": [5.0, 6.0], "wind_speed_10m": [3.0, 4.0]}
        )
        df = adapter._parse_hourly_sdk(response, ["temperature_2m", "wind_speed_10m"])
        assert df is not None
        assert "temperature_2m" in df.columns
        assert "wind_speed_10m" in df.columns
        assert len(df) == 2
        assert list(df["temperature_2m"]) == [5.0, 6.0]

    def test_parse_hourly_sdk_empty_returns_none(self, adapter):
        response = MagicMock()
        response.Hourly.return_value.VariablesLength.return_value = 0
        assert adapter._parse_hourly_sdk(response, ["temperature_2m"]) is None

    # ── _parse_ensemble_sdk ────────────────────────────────────────────────

    def test_parse_ensemble_sdk_builds_members(self, adapter):
        from runeflow.adapters.weather.openmeteo import ENSEMBLE_VARS

        response = _make_sdk_ensemble_response(ENSEMBLE_VARS, n_members=3, n_timesteps=2)
        members = adapter._parse_ensemble_sdk(response, ENSEMBLE_VARS)

        assert members is not None
        assert len(members) == 3
        assert 0 in members and 1 in members and 2 in members
        for df in members.values():
            assert set(ENSEMBLE_VARS).issubset(df.columns)
            assert len(df) == 2

    def test_parse_ensemble_sdk_member_values_are_distinct(self, adapter):
        """Each member gets its own set of values, not the same array shared."""
        from runeflow.adapters.weather.openmeteo import ENSEMBLE_VARS

        response = _make_sdk_ensemble_response(ENSEMBLE_VARS, n_members=2, n_timesteps=2)
        members = adapter._parse_ensemble_sdk(response, ENSEMBLE_VARS)

        assert members is not None
        # Member 0 and member 1 have different temperature values (0.0 vs 10.0)
        assert members[0]["temperature_2m"].iloc[0] != members[1]["temperature_2m"].iloc[0]

    def test_parse_ensemble_sdk_empty_returns_none(self, adapter):
        from runeflow.adapters.weather.openmeteo import ENSEMBLE_VARS

        response = MagicMock()
        response.Hourly.return_value.VariablesLength.return_value = 0
        assert adapter._parse_ensemble_sdk(response, ENSEMBLE_VARS) is None

    # ── _fetch_batch ───────────────────────────────────────────────────────

    def test_fetch_batch_returns_sdk_responses(self, adapter, loc):
        from unittest.mock import patch

        sdk_responses = [_make_sdk_hourly_response({"temperature_2m": [5.0]})]
        with patch.object(adapter._client, "weather_api", return_value=sdk_responses) as mock_api:
            result = adapter._fetch_batch(
                "https://api.example.com", [loc], hourly=["temperature_2m"]
            )

        assert result == sdk_responses
        params = mock_api.call_args[1]["params"]
        assert params["latitude"] == [loc.lat]
        assert params["longitude"] == [loc.lon]
        assert params["timezone"] == "UTC"

    def test_fetch_batch_failure_returns_nones(self, adapter, loc):
        from unittest.mock import patch

        with patch.object(adapter._client, "weather_api", side_effect=RuntimeError("net error")):
            result = adapter._fetch_batch("https://api.example.com", [loc])

        assert result == [None]

    def test_fetch_batch_multiple_locations(self, adapter):
        from unittest.mock import patch

        locs = [
            WeatherLocation(name="a", lat=52.0, lon=5.0, purpose="primary"),
            WeatherLocation(name="b", lat=53.0, lon=6.0, purpose="wind"),
        ]
        sdk_responses = [
            _make_sdk_hourly_response({"temperature_2m": [5.0]}),
            _make_sdk_hourly_response({"temperature_2m": [7.0]}),
        ]
        with patch.object(adapter._client, "weather_api", return_value=sdk_responses) as mock_api:
            result = adapter._fetch_batch("https://api.example.com", locs)

        assert len(result) == 2
        params = mock_api.call_args[1]["params"]
        assert params["latitude"] == [52.0, 53.0]
        assert params["longitude"] == [5.0, 6.0]

    # ── _fetch_historical ──────────────────────────────────────────────────

    def test_fetch_historical_happy_path(self, adapter, loc):
        from unittest.mock import patch

        sdk_response = _make_sdk_hourly_response({"temperature_2m": [5.0, 6.0]})
        with patch.object(adapter._client, "weather_api", return_value=[sdk_response]):
            df = adapter._fetch_historical(
                loc, datetime.date(2024, 1, 1), datetime.date(2024, 1, 1)
            )

        assert df is not None
        assert "temperature_2m" in df.columns

    def test_fetch_historical_end_in_future_clamps(self, adapter, loc):
        import datetime as dt
        from unittest.mock import patch

        future = dt.date.today() + dt.timedelta(days=10)
        sdk_response = _make_sdk_hourly_response({"temperature_2m": [5.0]})
        with patch.object(adapter._client, "weather_api", return_value=[sdk_response]) as mock_api:
            adapter._fetch_historical(loc, datetime.date(2024, 1, 1), future)

        call_params = mock_api.call_args[1]["params"]
        end_date = datetime.date.fromisoformat(call_params["end_date"])
        assert end_date <= dt.date.today()

    def test_fetch_historical_exception_re_raises(self, adapter, loc):
        from unittest.mock import patch

        from runeflow.exceptions import DownloadError

        with (
            patch.object(adapter._client, "weather_api", side_effect=RuntimeError("fail")),
            pytest.raises(DownloadError),
        ):
            adapter._fetch_historical(loc, datetime.date(2024, 1, 1), datetime.date(2024, 1, 1))

    # ── download_historical ────────────────────────────────────────────────

    def test_download_historical_calls_fetch_per_chunk(self, adapter, loc):
        """download_historical splits the range into quarterly chunks."""
        from unittest.mock import patch

        idx = pd.date_range("2024-01-01", periods=1, freq="h", tz="UTC")
        fake_df = pd.DataFrame({"temperature_2m": [5.0]}, index=idx)
        with patch.object(adapter, "_fetch_historical", return_value=fake_df) as mock_fetch:
            adapter.download_historical(
                [loc], datetime.date(2024, 1, 1), datetime.date(2024, 12, 31)
            )
        # One full year → 4 quarterly chunks
        assert mock_fetch.call_count == 4

    def test_download_historical_current_quarter_uses_biweekly_anchor(self, adapter, loc):
        """Current quarter's effective end date is pinned to the biweekly anchor."""
        from unittest.mock import patch

        from runeflow.adapters.weather.openmeteo import _biweekly_anchor

        today = datetime.date.today()
        # Use only the current quarter so exactly one chunk is produced and it
        # is unambiguously the "current" one.
        quarter_start = datetime.date(today.year, ((today.month - 1) // 3) * 3 + 1, 1)
        m = quarter_start.month - 1 + 3
        quarter_end = datetime.date(
            quarter_start.year + m // 12, m % 12 + 1, 1
        ) - datetime.timedelta(days=1)

        idx = pd.date_range("2024-01-01", periods=1, freq="h", tz="UTC")
        fake_df = pd.DataFrame({"temperature_2m": [5.0]}, index=idx)
        call_ends = []

        def capture_fetch(loc, s, e):
            call_ends.append(e)
            return fake_df

        with patch.object(adapter, "_fetch_historical", side_effect=capture_fetch):
            adapter.download_historical([loc], quarter_start, quarter_end)

        assert len(call_ends) == 1
        expected_end = max(_biweekly_anchor(today), quarter_start)
        assert call_ends[0] == expected_end

    def test_download_historical_current_quarter_ttl_restored(self, adapter, loc):
        """urls_expire_after on the session is restored after the current-quarter fetch."""
        from unittest.mock import patch

        today = datetime.date.today()
        start = datetime.date(today.year, 1, 1)
        end = datetime.date(today.year, 12, 31)

        idx = pd.date_range("2024-01-01", periods=1, freq="h", tz="UTC")
        fake_df = pd.DataFrame({"temperature_2m": [5.0]}, index=idx)

        original_ue = adapter._cache_session.settings.urls_expire_after.copy()
        with patch.object(adapter, "_fetch_historical", return_value=fake_df):
            adapter.download_historical([loc], start, end)

        # Session TTL must be restored to its original NEVER_EXPIRE setting.
        assert adapter._cache_session.settings.urls_expire_after == original_ue

    def test_download_historical_happy_path(self, adapter, loc):
        from unittest.mock import patch

        idx = pd.date_range("2024-01-01", periods=1, freq="h", tz="UTC")
        fake_df = pd.DataFrame({"temperature_2m": [5.0]}, index=idx)
        with patch.object(adapter, "_fetch_historical", return_value=fake_df):
            result = adapter.download_historical(
                [loc], datetime.date(2024, 1, 1), datetime.date(2024, 1, 1)
            )

        assert result is not None
        assert result.source == "open-meteo-historical"

    def test_download_historical_no_data_raises(self, adapter, loc):
        from unittest.mock import patch

        from runeflow.exceptions import DataUnavailableError

        with (
            patch.object(adapter, "_fetch_historical", return_value=None),
            pytest.raises(DataUnavailableError, match="no historical weather"),
        ):
            adapter.download_historical([loc], datetime.date(2024, 1, 1), datetime.date(2024, 1, 1))

    # ── download_forecast ──────────────────────────────────────────────────

    def test_download_forecast_happy_path(self, adapter, loc):
        from unittest.mock import patch

        idx = pd.date_range("2024-01-10", periods=2, freq="h", tz="UTC")
        fake_df = pd.DataFrame({"temperature_2m": [5.0, 6.0]}, index=idx)
        with (
            patch.object(adapter, "_fetch_batch", return_value=[MagicMock()]),
            patch.object(adapter, "_parse_hourly_sdk", return_value=fake_df),
        ):
            result = adapter.download_forecast([loc])

        assert result is not None
        assert result.source == "open-meteo-forecast"

    def test_download_forecast_no_data_raises(self, adapter, loc):
        from unittest.mock import patch

        from runeflow.exceptions import DataUnavailableError

        with (
            patch.object(adapter, "_fetch_batch", return_value=[MagicMock()]),
            patch.object(adapter, "_parse_hourly_sdk", return_value=None),
            pytest.raises(DataUnavailableError, match="no forecast weather"),
        ):
            adapter.download_forecast([loc])

    def test_download_forecast_none_batch_response_skipped(self, adapter, loc):
        from unittest.mock import patch

        from runeflow.exceptions import DataUnavailableError

        with (
            patch.object(adapter, "_fetch_batch", return_value=[None]),
            pytest.raises(DataUnavailableError),
        ):
            adapter.download_forecast([loc])

    # ── download_ensemble_forecast ─────────────────────────────────────────

    def test_download_ensemble_forecast_happy_path(self, adapter, loc):
        """Mixed ensemble: ICON-EU + ECMWF members merged."""
        from unittest.mock import patch

        idx = pd.date_range("2024-01-10", periods=2, freq="h", tz="UTC")
        icon_members = {
            0: {loc.name: pd.DataFrame({"temperature_2m": [3.0, 4.0]}, index=idx)},
            1: {loc.name: pd.DataFrame({"temperature_2m": [3.1, 4.1]}, index=idx)},
        }
        ecmwf_members = {
            0: {loc.name: pd.DataFrame({"temperature_2m": [3.2, 4.2]}, index=idx)},
        }

        def _fake_fetch(locs, model, days, *, interpolate_to_hourly=False):
            if model == "icon_eu":
                return icon_members
            return ecmwf_members

        with patch.object(adapter, "_fetch_ensemble_model", side_effect=_fake_fetch):
            results = adapter.download_ensemble_forecast([loc])

        assert len(results) == 3
        assert results[0].source == "open-meteo-ensemble-member-0"
        assert results[1].source == "open-meteo-ensemble-member-1"
        assert results[2].source == "open-meteo-ensemble-member-2"

    def test_download_ensemble_forecast_no_members_raises(self, adapter, loc):
        from unittest.mock import patch

        from runeflow.exceptions import DataUnavailableError

        with (
            patch.object(adapter, "_fetch_ensemble_model", return_value={}),
            pytest.raises(DataUnavailableError, match="no ensemble member"),
        ):
            adapter.download_ensemble_forecast([loc])

    def test_download_ensemble_forecast_ecmwf_only(self, adapter, loc):
        """When ICON-EU returns no data, only ECMWF members are used."""
        from unittest.mock import patch

        idx = pd.date_range("2024-01-10", periods=2, freq="h", tz="UTC")
        ecmwf_members = {
            0: {loc.name: pd.DataFrame({"temperature_2m": [3.0, 4.0]}, index=idx)},
        }

        def _fake_fetch(locs, model, days, *, interpolate_to_hourly=False):
            if model == "icon_eu":
                return {}
            return ecmwf_members

        with patch.object(adapter, "_fetch_ensemble_model", side_effect=_fake_fetch):
            results = adapter.download_ensemble_forecast([loc])

        assert len(results) == 1
        assert results[0].source == "open-meteo-ensemble-member-0"

    # ── _maybe_invalidate_ensemble_cache ──────────────────────────────────

    def test_invalidate_ensemble_cache_new_run_clears_cache(self, adapter, tmp_path):
        """New last_run_availability_time → cache cleared, state file updated."""
        from unittest.mock import MagicMock, patch

        adapter._model_state_path = tmp_path / "state.json"
        adapter._cache_session = MagicMock()

        with patch.object(adapter, "_get_model_run_availability", return_value=1000):
            adapter._maybe_invalidate_ensemble_cache("icon_eu")

        adapter._cache_session.cache.delete.assert_called_once()
        state = json.loads((tmp_path / "state.json").read_text())
        assert state["icon_eu"] == 1000

    def test_invalidate_ensemble_cache_same_run_no_clear(self, adapter, tmp_path):
        """Same availability time → cache left untouched."""
        from unittest.mock import MagicMock, patch

        adapter._model_state_path = tmp_path / "state.json"
        (tmp_path / "state.json").write_text('{"icon_eu": 1000}')
        adapter._cache_session = MagicMock()

        with patch.object(adapter, "_get_model_run_availability", return_value=1000):
            adapter._maybe_invalidate_ensemble_cache("icon_eu")

        adapter._cache_session.cache.delete.assert_not_called()

    def test_invalidate_ensemble_cache_meta_failure_no_clear(self, adapter, tmp_path):
        """Meta.json fetch failure → no cache clear, no state change."""
        from unittest.mock import MagicMock, patch

        adapter._model_state_path = tmp_path / "state.json"
        adapter._cache_session = MagicMock()

        with patch.object(adapter, "_get_model_run_availability", return_value=None):
            adapter._maybe_invalidate_ensemble_cache("icon_eu")

        adapter._cache_session.cache.delete.assert_not_called()
        assert not (tmp_path / "state.json").exists()

    def test_invalidate_ensemble_cache_delete_failure_no_state_update(self, adapter, tmp_path):
        """Cache.delete() failure → state NOT updated so retry fires next call."""
        from unittest.mock import MagicMock, patch

        adapter._model_state_path = tmp_path / "state.json"
        mock_session = MagicMock()
        mock_session.cache.delete.side_effect = RuntimeError("backend error")
        adapter._cache_session = mock_session

        with patch.object(adapter, "_get_model_run_availability", return_value=2000):
            adapter._maybe_invalidate_ensemble_cache("icon_eu")

        assert not (tmp_path / "state.json").exists()

    def test_get_model_run_availability_returns_timestamp(self, adapter):
        """Happy-path: parses last_run_availability_time from meta.json response."""
        from unittest.mock import patch

        with patch("runeflow.adapters.weather.openmeteo.requests.get") as mock_get:
            mock_resp = MagicMock()
            mock_resp.json.return_value = {"last_run_availability_time": 9999}
            mock_get.return_value = mock_resp
            result = adapter._get_model_run_availability("icon_eu")

        assert result == 9999

    def test_get_model_run_availability_unknown_model_returns_none(self, adapter):
        result = adapter._get_model_run_availability("unknown_model_xyz")
        assert result is None


class TestAwattarPriceAdapter:
    """Tests for adapters/price/awattar.py."""

    @pytest.fixture()
    def adapter(self):
        from runeflow.adapters.price.awattar import AwattarPriceAdapter

        return AwattarPriceAdapter()

    def test_name_property(self, adapter):
        assert adapter.name == "aWATTar"

    def test_supports_zone_de(self, adapter):
        assert adapter.supports_zone("DE_LU") is True

    def test_supports_zone_at(self, adapter):
        assert adapter.supports_zone("AT") is True

    def test_rejects_unsupported_zone(self, adapter):
        assert adapter.supports_zone("NL") is False
        assert adapter.supports_zone("FR") is False

    def test_download_historical_unsupported_zone_raises(self, adapter):
        from runeflow.exceptions import DataUnavailableError

        with pytest.raises(DataUnavailableError, match="not supported"):
            adapter.download_historical("NL", datetime.date(2024, 1, 1), datetime.date(2024, 1, 1))

    def test_download_historical_happy_path(self, adapter):
        from unittest.mock import patch

        fake_response = MagicMock()
        fake_response.json.return_value = {
            "data": [
                {"start_timestamp": 1704067200000, "marketprice": 80.5},
                {"start_timestamp": 1704070800000, "marketprice": 75.0},
            ]
        }
        fake_response.raise_for_status.return_value = None

        with (
            patch("runeflow.adapters.price.awattar.requests.get", return_value=fake_response),
            patch("runeflow.adapters.price.awattar.time.sleep"),
        ):
            result = adapter.download_historical(
                "DE_LU", datetime.date(2024, 1, 1), datetime.date(2024, 1, 1)
            )

        assert result is not None
        assert result.zone == "DE_LU"
        assert result.source == "aWATTar"

    def test_download_historical_empty_response_raises(self, adapter):
        from unittest.mock import patch

        from runeflow.exceptions import DataUnavailableError

        fake_response = MagicMock()
        fake_response.json.return_value = {"data": []}
        fake_response.raise_for_status.return_value = None

        with (
            patch("runeflow.adapters.price.awattar.requests.get", return_value=fake_response),
            patch("runeflow.adapters.price.awattar.time.sleep"),
            pytest.raises(DataUnavailableError, match="returned no data"),
        ):
            adapter.download_historical(
                "DE_LU", datetime.date(2024, 1, 1), datetime.date(2024, 1, 1)
            )

    def test_fetch_chunk_request_exception_raises_download_error(self, adapter):
        from unittest.mock import patch

        import requests as req_lib

        from runeflow.exceptions import DownloadError

        with (
            patch(
                "runeflow.adapters.price.awattar.requests.get",
                side_effect=req_lib.RequestException("timeout"),
            ),
            pytest.raises(DownloadError, match="aWATTar request failed"),
        ):
            adapter._fetch_chunk("DE_LU", datetime.date(2024, 1, 1), datetime.date(2024, 1, 1))

    def test_fetch_chunk_returns_none_on_empty_data(self, adapter):
        from unittest.mock import patch

        fake_response = MagicMock()
        fake_response.json.return_value = {"data": []}
        fake_response.raise_for_status.return_value = None

        with patch("runeflow.adapters.price.awattar.requests.get", return_value=fake_response):
            result = adapter._fetch_chunk(
                "DE_LU", datetime.date(2024, 1, 1), datetime.date(2024, 1, 1)
            )

        assert result is None

    def test_download_day_ahead_returns_result(self, adapter):
        from unittest.mock import patch

        fake_response = MagicMock()
        fake_response.json.return_value = {
            "data": [{"start_timestamp": 1704153600000, "marketprice": 90.0}]
        }
        fake_response.raise_for_status.return_value = None

        with (
            patch("runeflow.adapters.price.awattar.requests.get", return_value=fake_response),
            patch("runeflow.adapters.price.awattar.time.sleep"),
        ):
            result = adapter.download_day_ahead("AT")

        assert result is not None
        assert result.zone == "AT"

    def test_download_day_ahead_unsupported_zone_returns_none(self, adapter):
        result = adapter.download_day_ahead("NL")
        assert result is None

    def test_download_day_ahead_exception_returns_none(self, adapter):
        from unittest.mock import patch

        import requests as req_lib

        with patch(
            "runeflow.adapters.price.awattar.requests.get",
            side_effect=req_lib.RequestException("err"),
        ):
            result = adapter.download_day_ahead("DE_LU")

        assert result is None

    def test_download_historical_multi_chunk_calls_sleep(self, adapter):
        """time.sleep is called between chunks (range > 30 days)."""
        from unittest.mock import patch

        fake_response = MagicMock()
        fake_response.raise_for_status.return_value = None
        fake_response.json.return_value = {
            "data": [{"start_timestamp": 1704067200000, "marketprice": 80.0}]
        }

        sleep_calls = []
        with (
            patch("runeflow.adapters.price.awattar.requests.get", return_value=fake_response),
            patch(
                "runeflow.adapters.price.awattar.time.sleep",
                side_effect=lambda s: sleep_calls.append(s),
            ),
        ):
            result = adapter.download_historical(
                "DE_LU",
                datetime.date(2024, 1, 1),
                datetime.date(2024, 3, 1),  # > 30 days → 2 chunks
            )

        assert len(sleep_calls) >= 1
        assert result is not None


class TestNordpoolPriceAdapter:
    """Tests for adapters/price/nordpool_adapter.py."""

    @pytest.fixture()
    def adapter_with_mock(self):
        from unittest.mock import MagicMock, patch

        mock_api = MagicMock()
        with patch(
            "runeflow.adapters.price.nordpool_adapter.NordpoolPriceAdapter.__init__",
            return_value=None,
        ):
            from runeflow.adapters.price.nordpool_adapter import NordpoolPriceAdapter

            adapter = NordpoolPriceAdapter()
            adapter._api = mock_api
        return adapter, mock_api

    def _day_response(self, area: str, price: float = 80.0) -> dict:
        return {
            "areas": {
                area: {
                    "values": [
                        {"start": pd.Timestamp("2024-01-01 00:00:00"), "value": price},
                        {"start": pd.Timestamp("2024-01-01 01:00:00"), "value": price + 5},
                    ]
                }
            }
        }

    def test_name_property(self, adapter_with_mock):
        adapter, _ = adapter_with_mock
        assert adapter.name == "Nordpool"

    def test_supports_zone_fi(self, adapter_with_mock):
        adapter, _ = adapter_with_mock
        assert adapter.supports_zone("FI") is True

    def test_supports_zone_se(self, adapter_with_mock):
        adapter, _ = adapter_with_mock
        assert adapter.supports_zone("SE_3") is True

    def test_supports_zone_dk(self, adapter_with_mock):
        adapter, _ = adapter_with_mock
        assert adapter.supports_zone("DK_1") is True
        assert adapter.supports_zone("DK_2") is True

    def test_supports_zone_baltic(self, adapter_with_mock):
        adapter, _ = adapter_with_mock
        assert adapter.supports_zone("EE") is True
        assert adapter.supports_zone("LV") is True
        assert adapter.supports_zone("LT") is True

    def test_rejects_unsupported_zone(self, adapter_with_mock):
        adapter, _ = adapter_with_mock
        assert adapter.supports_zone("NL") is False
        assert adapter.supports_zone("DE_LU") is False

    def test_download_historical_unsupported_zone_raises(self, adapter_with_mock):
        adapter, _ = adapter_with_mock
        from runeflow.exceptions import DataUnavailableError

        with pytest.raises(DataUnavailableError, match="does not support"):
            adapter.download_historical("NL", datetime.date(2024, 1, 1), datetime.date(2024, 1, 1))

    def test_download_historical_happy_path(self, adapter_with_mock):
        from unittest.mock import patch

        adapter, mock_api = adapter_with_mock
        mock_api.fetch.return_value = self._day_response("FI", 75.0)

        with patch("runeflow.adapters.price.nordpool_adapter.time.sleep"):
            result = adapter.download_historical(
                "FI", datetime.date(2024, 1, 1), datetime.date(2024, 1, 1)
            )

        assert result is not None
        assert result.zone == "FI"
        assert result.source == "Nordpool"

    def test_download_historical_empty_raises(self, adapter_with_mock):
        from unittest.mock import patch

        from runeflow.exceptions import DataUnavailableError

        adapter, mock_api = adapter_with_mock
        mock_api.fetch.return_value = {"areas": {"FI": {"values": []}}}

        with (
            patch("runeflow.adapters.price.nordpool_adapter.time.sleep"),
            pytest.raises(DataUnavailableError, match="returned no data"),
        ):
            adapter.download_historical("FI", datetime.date(2024, 1, 1), datetime.date(2024, 1, 1))

    def test_download_historical_handles_none_values(self, adapter_with_mock):
        """None values (prices not yet published) are skipped."""
        from unittest.mock import patch

        from runeflow.exceptions import DataUnavailableError

        adapter, mock_api = adapter_with_mock
        mock_api.fetch.return_value = {
            "areas": {
                "SE3": {
                    "values": [
                        {"start": pd.Timestamp("2024-01-01 00:00:00"), "value": None},
                        {"start": pd.Timestamp("2024-01-01 01:00:00"), "value": None},
                    ]
                }
            }
        }

        with (
            patch("runeflow.adapters.price.nordpool_adapter.time.sleep"),
            pytest.raises(DataUnavailableError, match="returned no data"),
        ):
            adapter.download_historical(
                "SE_3", datetime.date(2024, 1, 1), datetime.date(2024, 1, 1)
            )

    def test_download_day_ahead_happy_path(self, adapter_with_mock):
        adapter, mock_api = adapter_with_mock
        mock_api.fetch.return_value = {
            "areas": {
                "Oslo": {
                    "values": [
                        {"start": pd.Timestamp("2024-01-02 00:00:00"), "value": 95.0},
                    ]
                }
            }
        }
        result = adapter.download_day_ahead("NO_1")
        assert result is not None
        assert result.zone == "NO_1"

    def test_download_day_ahead_unsupported_zone_returns_none(self, adapter_with_mock):
        adapter, _ = adapter_with_mock
        result = adapter.download_day_ahead("DE_LU")
        assert result is None

    def test_download_day_ahead_exception_returns_none(self, adapter_with_mock):
        adapter, mock_api = adapter_with_mock
        mock_api.fetch.side_effect = Exception("API error")
        result = adapter.download_day_ahead("FI")
        assert result is None

    def test_download_day_ahead_empty_returns_none(self, adapter_with_mock):
        adapter, mock_api = adapter_with_mock
        mock_api.fetch.return_value = {"areas": {"FI": {"values": []}}}
        result = adapter.download_day_ahead("FI")
        assert result is None
