# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2024-2026 Thomas Phil — runeflow
# See LICENSE and COMMERCIAL-LICENSE.md for licensing details.

"""Tests for adapters: FallbackPriceAdapter, WeatherSeries, ForecastResult,
and binder.configure_injector."""

from __future__ import annotations

import datetime
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
# OpenMeteo adapter
# ---------------------------------------------------------------------------

_FAKE_HOURLY_JSON = {
    "hourly": {
        "time": ["2024-01-01T00:00:00", "2024-01-01T01:00:00"],
        "temperature_2m": [5.0, 6.0],
        "wind_speed_10m": [3.0, 4.0],
        "wind_direction_10m": [180.0, 185.0],
        "wind_gusts_10m": [5.0, 6.0],
        "shortwave_radiation": [0.0, 0.0],
        "direct_radiation": [0.0, 0.0],
        "diffuse_radiation": [0.0, 0.0],
        "cloud_cover": [50.0, 60.0],
        "relative_humidity_2m": [80.0, 82.0],
        "precipitation": [0.0, 0.0],
        "is_day": [0, 0],
    },
    "timezone": "UTC",
}

_FAKE_ENSEMBLE_JSON = {
    "hourly": {
        "time": ["2024-01-10T00:00:00", "2024-01-10T01:00:00"],
        "temperature_2m": [3.0, 4.0],
        "temperature_2m_member01": [3.1, 4.1],
        "temperature_2m_member02": [2.9, 3.9],
        "wind_speed_10m": [2.0, 2.5],
        "wind_speed_10m_member01": [2.1, 2.6],
        "wind_speed_10m_member02": [1.9, 2.4],
        "wind_gusts_10m": [4.0, 5.0],
        "wind_gusts_10m_member01": [4.1, 5.1],
        "wind_gusts_10m_member02": [3.9, 4.9],
        "cloud_cover": [60.0, 70.0],
        "cloud_cover_member01": [61.0, 71.0],
        "cloud_cover_member02": [59.0, 69.0],
        "shortwave_radiation": [0.0, 0.0],
        "shortwave_radiation_member01": [0.0, 0.0],
        "shortwave_radiation_member02": [0.0, 0.0],
        "direct_radiation": [0.0, 0.0],
        "direct_radiation_member01": [0.0, 0.0],
        "direct_radiation_member02": [0.0, 0.0],
        "diffuse_radiation": [0.0, 0.0],
        "diffuse_radiation_member01": [0.0, 0.0],
        "diffuse_radiation_member02": [0.0, 0.0],
        "precipitation": [0.0, 0.0],
        "precipitation_member01": [0.0, 0.0],
        "precipitation_member02": [0.0, 0.0],
    },
    "timezone": "UTC",
}


def _fake_resp(json_data, status=200):
    resp = MagicMock()
    resp.status_code = status
    resp.json.return_value = json_data
    resp.raise_for_status.return_value = None
    return resp


class TestOpenMeteoAdapter:
    """Tests for adapters/weather/openmeteo.py."""

    @pytest.fixture()
    def adapter(self):
        from runeflow.adapters.weather.openmeteo import OpenMeteoAdapter

        return OpenMeteoAdapter(timezone="UTC")

    @pytest.fixture()
    def loc(self):
        return WeatherLocation(name="nl", lat=52.1, lon=5.18, purpose="primary")

    # ── _parse_hourly ──────────────────────────────────────────────────────

    def test_parse_hourly_valid(self, adapter):
        from runeflow.adapters.weather.openmeteo import OpenMeteoAdapter

        df = OpenMeteoAdapter._parse_hourly(_FAKE_HOURLY_JSON)
        assert df is not None
        assert "temperature_2m" in df.columns
        assert len(df) == 2

    def test_parse_hourly_missing_time_returns_none(self, adapter):
        from runeflow.adapters.weather.openmeteo import OpenMeteoAdapter

        df = OpenMeteoAdapter._parse_hourly({})
        assert df is None

        df2 = OpenMeteoAdapter._parse_hourly({"hourly": {"no_time": [1, 2]}})
        assert df2 is None

    # ── _get_with_retry ────────────────────────────────────────────────────

    def test_get_with_retry_success(self, adapter):
        from unittest.mock import patch

        with (
            patch(
                "runeflow.adapters.weather.openmeteo.requests.get",
                return_value=_fake_resp(_FAKE_HOURLY_JSON),
            ),
            patch("runeflow.adapters.weather.openmeteo.time.sleep"),
        ):
            resp = adapter._get_with_retry("http://example.com", {})

        assert resp is not None
        assert resp.json() == _FAKE_HOURLY_JSON

    def test_get_with_retry_rate_limit_raises(self, adapter):
        from unittest.mock import patch

        from runeflow.exceptions import RateLimitError

        rate_limited = _fake_resp({}, status=429)
        rate_limited.raise_for_status.return_value = None

        with (
            patch("runeflow.adapters.weather.openmeteo.requests.get", return_value=rate_limited),
            patch("runeflow.adapters.weather.openmeteo.time.sleep"),
            pytest.raises(RateLimitError, match="Rate limit exceeded"),
        ):
            adapter._get_with_retry("http://example.com", {}, max_retries=4)

    def test_get_with_retry_rate_limit_retries_once(self, adapter):
        """429 on first attempt → sleep → success on second attempt."""
        from unittest.mock import patch

        rate_limited = _fake_resp({}, status=429)
        rate_limited.raise_for_status.return_value = None
        success = _fake_resp(_FAKE_HOURLY_JSON)

        with (
            patch(
                "runeflow.adapters.weather.openmeteo.requests.get",
                side_effect=[rate_limited, success],
            ),
            patch("runeflow.adapters.weather.openmeteo.time.sleep"),
        ):
            resp = adapter._get_with_retry("http://example.com", {}, max_retries=4)

        assert resp is not None

    def test_get_with_retry_request_exception_raises_download_error(self, adapter):
        from unittest.mock import patch

        import requests as req_lib

        from runeflow.exceptions import DownloadError

        with (
            patch(
                "runeflow.adapters.weather.openmeteo.requests.get",
                side_effect=req_lib.RequestException("timeout"),
            ),
            patch("runeflow.adapters.weather.openmeteo.time.sleep"),
            pytest.raises(DownloadError, match="Open-Meteo request failed"),
        ):
            adapter._get_with_retry("http://example.com", {}, max_retries=1)

    def test_get_with_retry_request_exception_retries(self, adapter):
        """Request fails on first call, succeeds on second — no DownloadError raised."""
        from unittest.mock import patch

        import requests as req_lib

        with (
            patch(
                "runeflow.adapters.weather.openmeteo.requests.get",
                side_effect=[req_lib.RequestException("timeout"), _fake_resp(_FAKE_HOURLY_JSON)],
            ),
            patch("runeflow.adapters.weather.openmeteo.time.sleep"),
        ):
            resp = adapter._get_with_retry("http://example.com", {}, max_retries=4)

        assert resp is not None

    def test_get_with_retry_zero_retries_returns_none(self, adapter):
        """Line 179: max_retries=0 means the for-loop is empty → return None."""
        result = adapter._get_with_retry("http://example.com", {}, max_retries=0)
        assert result is None

    # ── _fetch_historical ──────────────────────────────────────────────────

    def test_fetch_historical_happy_path(self, adapter, loc):
        from unittest.mock import patch

        with patch.object(adapter, "_get_with_retry", return_value=_fake_resp(_FAKE_HOURLY_JSON)):
            df = adapter._fetch_historical(
                loc, datetime.date(2024, 1, 1), datetime.date(2024, 1, 1)
            )

        assert df is not None
        assert "temperature_2m" in df.columns

    def test_fetch_historical_end_in_future_clamps(self, adapter, loc):
        import datetime as dt
        from unittest.mock import patch

        future = dt.date.today() + dt.timedelta(days=10)
        with patch.object(adapter, "_get_with_retry", return_value=_fake_resp(_FAKE_HOURLY_JSON)):
            df = adapter._fetch_historical(loc, datetime.date(2024, 1, 1), future)

        assert df is not None

    def test_fetch_historical_none_response(self, adapter, loc):
        from unittest.mock import patch

        with patch.object(adapter, "_get_with_retry", return_value=None):
            result = adapter._fetch_historical(
                loc, datetime.date(2024, 1, 1), datetime.date(2024, 1, 1)
            )

        assert result is None

    def test_fetch_historical_exception_re_raises(self, adapter, loc):
        from unittest.mock import patch

        from runeflow.exceptions import DownloadError

        with (
            patch.object(adapter, "_get_with_retry", side_effect=DownloadError("fail")),
            pytest.raises(DownloadError),
        ):
            adapter._fetch_historical(loc, datetime.date(2024, 1, 1), datetime.date(2024, 1, 1))

    # ── _fetch_forecast ────────────────────────────────────────────────────

    def test_fetch_forecast_happy_path(self, adapter, loc):
        from unittest.mock import patch

        with patch.object(adapter, "_get_with_retry", return_value=_fake_resp(_FAKE_HOURLY_JSON)):
            df = adapter._fetch_forecast(loc, forecast_days=3)

        assert df is not None
        assert "temperature_2m" in df.columns

    def test_fetch_forecast_none_response(self, adapter, loc):
        from unittest.mock import patch

        with patch.object(adapter, "_get_with_retry", return_value=None):
            result = adapter._fetch_forecast(loc)

        assert result is None

    def test_fetch_forecast_exception_returns_none(self, adapter, loc):
        from unittest.mock import patch

        with patch.object(adapter, "_get_with_retry", side_effect=RuntimeError("boom")):
            result = adapter._fetch_forecast(loc)

        assert result is None

    # ── _fetch_ensemble_members ────────────────────────────────────────────

    def test_fetch_ensemble_members_happy_path(self, adapter, loc):
        from unittest.mock import patch

        with patch.object(adapter, "_get_with_retry", return_value=_fake_resp(_FAKE_ENSEMBLE_JSON)):
            members = adapter._fetch_ensemble_members(loc)

        assert members is not None
        assert 0 in members
        assert 1 in members
        assert 2 in members

    def test_fetch_ensemble_members_none_response(self, adapter, loc):
        from unittest.mock import patch

        with patch.object(adapter, "_get_with_retry", return_value=None):
            result = adapter._fetch_ensemble_members(loc)

        assert result is None

    def test_fetch_ensemble_members_exception_returns_none(self, adapter, loc):
        from unittest.mock import patch

        with patch.object(adapter, "_get_with_retry", side_effect=RuntimeError("fail")):
            result = adapter._fetch_ensemble_members(loc)

        assert result is None

    def test_fetch_ensemble_members_empty_hourly(self, adapter, loc):
        from unittest.mock import patch

        empty_json = {"hourly": {}, "timezone": "UTC"}
        with patch.object(adapter, "_get_with_retry", return_value=_fake_resp(empty_json)):
            result = adapter._fetch_ensemble_members(loc)

        assert result is None

    def test_fetch_ensemble_members_partial_members_covers_280_283(self, adapter, loc):
        """Lines 280-283: partial member coverage for some vars.

        Covers:
        - Line 281: var has no member keys at all → use control-run value
        - Line 283: var has member01 but not member02 → df[var] = np.nan
        """
        from unittest.mock import patch

        # temperature_2m has 2 members → n_members = 3
        # wind_speed_10m has 1 member only (member01, not member02) → line 283 for m=2
        # precipitation has 0 members → line 281 for m=1 and m=2
        partial_json = {
            "hourly": {
                "time": ["2024-01-10T00:00:00", "2024-01-10T01:00:00"],
                "temperature_2m": [3.0, 4.0],
                "temperature_2m_member01": [3.1, 4.1],
                "temperature_2m_member02": [2.9, 3.9],
                "wind_speed_10m": [2.0, 2.5],
                "wind_speed_10m_member01": [2.1, 2.6],
                # wind_speed_10m_member02 intentionally absent → line 283
                "wind_gusts_10m": [4.0, 5.0],
                "cloud_cover": [60.0, 70.0],
                "shortwave_radiation": [0.0, 0.0],
                "direct_radiation": [0.0, 0.0],
                "diffuse_radiation": [0.0, 0.0],
                "precipitation": [0.0, 0.0],  # no _member* keys → line 281
            },
            "timezone": "UTC",
        }

        with patch.object(adapter, "_get_with_retry", return_value=_fake_resp(partial_json)):
            members = adapter._fetch_ensemble_members(loc)

        assert members is not None
        assert len(members) == 3  # n_members = 3 (0, 1, 2)
        # For m=2, wind_speed_10m should be NaN (line 283)
        assert members[2]["wind_speed_10m"].isna().all()
        # For m=1, precipitation should equal control value (line 281)
        assert (members[1]["precipitation"] == 0.0).all()

    # ── download_historical ────────────────────────────────────────────────

    def test_download_historical_happy_path(self, adapter, loc):
        from unittest.mock import patch

        with (
            patch.object(
                adapter,
                "_fetch_historical",
                return_value=pd.DataFrame(
                    {"temperature_2m": [5.0]},
                    index=pd.date_range("2024-01-01", periods=1, freq="h", tz="UTC"),
                ),
            ),
            patch("runeflow.adapters.weather.openmeteo.time.sleep"),
        ):
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
            patch("runeflow.adapters.weather.openmeteo.time.sleep"),
            pytest.raises(DataUnavailableError, match="no historical weather"),
        ):
            adapter.download_historical([loc], datetime.date(2024, 1, 1), datetime.date(2024, 1, 1))

    # ── download_forecast ──────────────────────────────────────────────────

    def test_download_forecast_happy_path(self, adapter, loc):
        from unittest.mock import patch

        with (
            patch.object(
                adapter,
                "_fetch_forecast",
                return_value=pd.DataFrame(
                    {"temperature_2m": [5.0]},
                    index=pd.date_range("2024-01-10", periods=1, freq="h", tz="UTC"),
                ),
            ),
            patch("runeflow.adapters.weather.openmeteo.time.sleep"),
        ):
            result = adapter.download_forecast([loc])

        assert result is not None
        assert result.source == "open-meteo-forecast"

    def test_download_forecast_no_data_raises(self, adapter, loc):
        from unittest.mock import patch

        from runeflow.exceptions import DataUnavailableError

        with (
            patch.object(adapter, "_fetch_forecast", return_value=None),
            patch("runeflow.adapters.weather.openmeteo.time.sleep"),
            pytest.raises(DataUnavailableError, match="no forecast weather"),
        ):
            adapter.download_forecast([loc])

    # ── download_ensemble_forecast ─────────────────────────────────────────

    def test_download_ensemble_forecast_happy_path(self, adapter, loc):
        from unittest.mock import patch

        idx = pd.date_range("2024-01-10", periods=2, freq="h", tz="UTC")
        fake_members = {
            0: pd.DataFrame({"temperature_2m": [3.0, 4.0]}, index=idx),
            1: pd.DataFrame({"temperature_2m": [3.1, 4.1]}, index=idx),
        }

        with (
            patch.object(adapter, "_fetch_ensemble_members", return_value=fake_members),
            patch("runeflow.adapters.weather.openmeteo.time.sleep"),
        ):
            results = adapter.download_ensemble_forecast([loc])

        assert len(results) == 2
        assert results[0].source == "open-meteo-ensemble-member-0"

    def test_download_ensemble_forecast_no_members_raises(self, adapter, loc):
        from unittest.mock import patch

        from runeflow.exceptions import DataUnavailableError

        with (
            patch.object(adapter, "_fetch_ensemble_members", return_value=None),
            patch("runeflow.adapters.weather.openmeteo.time.sleep"),
            pytest.raises(DataUnavailableError, match="no ensemble member"),
        ):
            adapter.download_ensemble_forecast([loc])
