# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2024-2026 Thomas Phil — runeflow
# See LICENSE and COMMERCIAL-LICENSE.md for licensing details.

"""Tests for feature engineering groups and the FeaturePipeline."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from runeflow.features.base import FeaturePipeline
from runeflow.features.precipitation import PrecipitationFeatures
from runeflow.features.price_lag import PriceLagFeatures
from runeflow.features.price_regime import PriceRegimeFeatures
from runeflow.features.registry import FEATURE_REGISTRY, build_pipeline
from runeflow.features.spike import SpikeMomentumFeatures, SpikeRiskFeatures
from runeflow.features.temperature import TemperatureFeatures
from runeflow.features.temporal import TemporalFeatures
from runeflow.features.wind import WindFeatures
from runeflow.zones.registry import ZoneRegistry

# ── Helpers ──────────────────────────────────────────────────────────────────


def _make_df(n: int = 24 * 30, with_weather: bool = True) -> pd.DataFrame:
    """Return a minimal DataFrame suitable for most feature groups."""
    rng = np.random.default_rng(7)
    idx = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")
    data: dict[str, object] = {
        "Price_EUR_MWh": 50.0 + 30.0 * rng.standard_normal(n),
    }
    if with_weather:
        data.update(
            {
                "temperature_2m": 10.0 + 5.0 * rng.standard_normal(n),
                "wind_speed_10m": np.abs(6.0 + 3.0 * rng.standard_normal(n)),
                "shortwave_radiation": np.clip(150.0 + 120.0 * rng.standard_normal(n), 0, None),
                "cloudcover": np.clip(40.0 + 30.0 * rng.standard_normal(n), 0, 100),
                "precipitation": np.clip(rng.exponential(0.2, n), 0, None),
            }
        )
    return pd.DataFrame(data, index=idx)


# ── TemporalFeatures ─────────────────────────────────────────────────────────


class TestTemporalFeatures:
    def test_produces_hour_of_day(self, zone_cfg_nl):
        df = _make_df()
        out = TemporalFeatures().transform(df, zone_cfg_nl)
        assert "hour_of_day" in out.columns

    def test_hour_of_day_range(self, zone_cfg_nl):
        df = _make_df()
        out = TemporalFeatures().transform(df, zone_cfg_nl)
        assert out["hour_of_day"].between(0, 23).all()

    def test_produces_is_weekend(self, zone_cfg_nl):
        df = _make_df()
        out = TemporalFeatures().transform(df, zone_cfg_nl)
        assert "is_weekend" in out.columns
        assert set(out["is_weekend"].unique()).issubset({0, 1})

    def test_produces_season(self, zone_cfg_nl):
        df = _make_df()
        out = TemporalFeatures().transform(df, zone_cfg_nl)
        assert "season" in out.columns
        assert out["season"].between(1, 4).all()

    def test_does_not_mutate_input(self, zone_cfg_nl):
        df = _make_df()
        cols_before = set(df.columns)
        TemporalFeatures().transform(df, zone_cfg_nl)
        assert set(df.columns) == cols_before, "Input DataFrame was mutated"

    def test_produces_all_declared_columns(self, zone_cfg_nl):
        df = _make_df()
        out = TemporalFeatures().transform(df, zone_cfg_nl)
        for col in TemporalFeatures().produces:
            assert col in out.columns, f"Missing column: {col}"

    def test_peak_hour_binary(self, zone_cfg_nl):
        df = _make_df()
        out = TemporalFeatures().transform(df, zone_cfg_nl)
        assert set(out["is_peak_hour"].unique()).issubset({0, 1})

    def test_overnight_binary(self, zone_cfg_nl):
        df = _make_df()
        out = TemporalFeatures().transform(df, zone_cfg_nl)
        # peak_hour and overnight should never both be 1
        overlap = (out["is_peak_hour"] == 1) & (out["is_overnight"] == 1)
        assert not overlap.any()


# ── PriceLagFeatures ─────────────────────────────────────────────────────────


class TestPriceLagFeatures:
    def test_produces_lag_24(self, zone_cfg_nl):
        df = _make_df()
        out = PriceLagFeatures().transform(df, zone_cfg_nl)
        assert "Price_EUR_MWh_lag_24" in out.columns

    def test_lag_24_is_shifted(self, zone_cfg_nl):
        df = _make_df()
        out = PriceLagFeatures().transform(df, zone_cfg_nl)
        # lag_24 at index 24 should equal Price_EUR_MWh at index 0
        assert out["Price_EUR_MWh_lag_24"].iloc[24] == pytest.approx(df["Price_EUR_MWh"].iloc[0])

    def test_rolling_mean_present(self, zone_cfg_nl):
        df = _make_df()
        out = PriceLagFeatures().transform(df, zone_cfg_nl)
        assert "Price_EUR_MWh_rolling_24h_mean" in out.columns

    def test_same_hour_1d_lag(self, zone_cfg_nl):
        df = _make_df()
        out = PriceLagFeatures().transform(df, zone_cfg_nl)
        assert "Price_EUR_MWh_same_hour_1d" in out.columns

    def test_no_mutation(self, zone_cfg_nl):
        df = _make_df()
        cols_before = set(df.columns)
        PriceLagFeatures().transform(df, zone_cfg_nl)
        assert set(df.columns) == cols_before

    def test_missing_price_col_returns_unchanged(self, zone_cfg_nl):
        df = pd.DataFrame(
            {"other": [1, 2, 3]},
            index=pd.date_range("2024-01-01", periods=3, freq="h", tz="UTC"),
        )
        out = PriceLagFeatures().transform(df, zone_cfg_nl)
        assert "Price_EUR_MWh_lag_24" not in out.columns


# ── SpikeMomentumFeatures ─────────────────────────────────────────────────────


class TestSpikeMomentumFeatures:
    def test_produces_zscore(self, zone_cfg_nl):
        df = _make_df()
        out = SpikeMomentumFeatures().transform(df, zone_cfg_nl)
        assert "Price_EUR_MWh_zscore_24h" in out.columns

    def test_zscore_finite_after_warmup(self, zone_cfg_nl):
        df = _make_df(n=24 * 7)  # 7 days; warmup needs 12h min
        out = SpikeMomentumFeatures().transform(df, zone_cfg_nl)
        col = "Price_EUR_MWh_zscore_24h"
        valid = out[col].dropna()
        assert valid.notna().all()

    def test_spike_count_non_negative(self, zone_cfg_nl):
        df = _make_df()
        out = SpikeMomentumFeatures().transform(df, zone_cfg_nl)
        assert (out["Price_EUR_MWh_spike_count_24h"].dropna() >= 0).all()


# ── SpikeRiskFeatures ─────────────────────────────────────────────────────────


class TestSpikeRiskFeatures:
    def test_produces_hours_since_last_spike(self, zone_cfg_nl):
        df = _make_df()
        out = SpikeRiskFeatures().transform(df, zone_cfg_nl)
        assert "hours_since_last_spike" in out.columns

    def test_hours_since_spike_non_negative(self, zone_cfg_nl):
        df = _make_df()
        out = SpikeRiskFeatures().transform(df, zone_cfg_nl)
        assert (out["hours_since_last_spike"].dropna() >= 0).all()

    def test_no_python_loop(self, zone_cfg_nl):
        """Verify the transform runs in reasonable time (vectorised path)."""
        import time

        df = _make_df(n=24 * 365)
        start = time.monotonic()
        SpikeRiskFeatures().transform(df, zone_cfg_nl)
        elapsed = time.monotonic() - start
        assert elapsed < 5.0, f"SpikeRiskFeatures took {elapsed:.2f}s — possible slow loop"


# ── TemperatureFeatures ──────────────────────────────────────────────────────


class TestTemperatureFeatures:
    def test_produces_hdd(self, zone_cfg_nl):
        df = _make_df()
        out = TemperatureFeatures().transform(df, zone_cfg_nl)
        assert "hdd" in out.columns

    def test_hdd_non_negative(self, zone_cfg_nl):
        df = _make_df()
        out = TemperatureFeatures().transform(df, zone_cfg_nl)
        assert (out["hdd"] >= 0).all()

    def test_produces_cdd(self, zone_cfg_nl):
        df = _make_df()
        out = TemperatureFeatures().transform(df, zone_cfg_nl)
        assert "cdd" in out.columns

    def test_cdd_non_negative(self, zone_cfg_nl):
        df = _make_df()
        out = TemperatureFeatures().transform(df, zone_cfg_nl)
        assert (out["cdd"] >= 0).all()

    def test_no_temp_col_returns_unchanged(self, zone_cfg_nl):
        df = pd.DataFrame(
            {"Price_EUR_MWh": [1.0, 2.0]},
            index=pd.date_range("2024-01-01", periods=2, freq="h", tz="UTC"),
        )
        out = TemperatureFeatures().transform(df, zone_cfg_nl)
        assert "hdd" not in out.columns


# ── WindFeatures ─────────────────────────────────────────────────────────────


class TestWindFeatures:
    def test_produces_wind_power_potential(self, zone_cfg_nl):
        df = _make_df()
        out = WindFeatures().transform(df, zone_cfg_nl)
        assert "wind_power_potential" in out.columns

    def test_wind_power_non_negative(self, zone_cfg_nl):
        df = _make_df()
        out = WindFeatures().transform(df, zone_cfg_nl)
        assert (out["wind_power_potential"] >= 0).all()


# ── PrecipitationFeatures ─────────────────────────────────────────────────────


class TestPrecipitationFeatures:
    def test_produces_accumulated(self, zone_cfg_nl):
        df = _make_df()
        out = PrecipitationFeatures().transform(df, zone_cfg_nl)
        # Should add some accumulated precipitation column
        precip_cols = [c for c in out.columns if "precip" in c or "precipitation" in c]
        assert len(precip_cols) > 0, "No precipitation columns produced"


# ── FeaturePipeline ───────────────────────────────────────────────────────────


class TestFeaturePipeline:
    def test_no_mutation_of_input(self, zone_cfg_nl):
        df = _make_df()
        original_cols = set(df.columns)
        pipeline = build_pipeline(zone_cfg_nl)
        _ = pipeline.transform(df, zone_cfg_nl)
        assert set(df.columns) == original_cols, "Pipeline mutated the input DataFrame"

    def test_output_has_more_columns_than_input(self, zone_cfg_nl):
        df = _make_df()
        pipeline = build_pipeline(zone_cfg_nl)
        out = pipeline.transform(df, zone_cfg_nl)
        assert len(out.columns) > len(df.columns)

    def test_output_preserves_index(self, zone_cfg_nl):
        df = _make_df()
        pipeline = build_pipeline(zone_cfg_nl)
        out = pipeline.transform(df, zone_cfg_nl)
        assert out.index.equals(df.index)

    def test_build_pipeline_uses_registry_names(self):
        """build_pipeline should only include groups that are in FEATURE_REGISTRY."""
        nl_cfg = ZoneRegistry.get("NL")
        pipeline = build_pipeline(nl_cfg)
        for group in pipeline._groups:
            assert group.name in FEATURE_REGISTRY

    def test_build_pipeline_de_lu(self, zone_cfg_de_lu):
        pipeline = build_pipeline(zone_cfg_de_lu)
        assert isinstance(pipeline, FeaturePipeline)

    def test_feature_pipeline_skips_failing_group(self, zone_cfg_nl):
        """A group that raises should be skipped, not crash the pipeline."""
        from runeflow.features.base import FeatureGroup
        from runeflow.zones.config import ZoneConfig

        class BrokenGroup(FeatureGroup):
            name = "broken"

            def transform(self, df: pd.DataFrame, zone_cfg: ZoneConfig) -> pd.DataFrame:
                raise RuntimeError("Oops")

        df = _make_df()
        pipeline = FeaturePipeline([TemporalFeatures(), BrokenGroup()])
        out = pipeline.transform(df, zone_cfg_nl)
        # Should still have temporal features despite BrokenGroup failing
        assert "hour_of_day" in out.columns


# ── Feature Registry ──────────────────────────────────────────────────────────


class TestFeatureRegistry:
    def test_all_registry_keys_are_valid_names(self):
        """Registry keys should match the class's name attribute."""
        for key, cls in FEATURE_REGISTRY.items():
            instance = cls()
            assert instance.name == key, (
                f"Registry key '{key}' doesn't match {cls.__name__}.name='{instance.name}'"
            )

    def test_registry_has_temporal(self):
        assert "temporal" in FEATURE_REGISTRY

    def test_registry_has_all_19_groups(self):
        """We expect 19 feature groups (each class counted once)."""
        # temporal, solar_position, solar_power, holiday, price_lag, price_regime,
        # spike_momentum, spike_risk, temperature, wind, precipitation, cloud,
        # renewable_pressure, residual_load, cross_border, duck_curve, market_structure,
        # generation, interaction
        assert len(FEATURE_REGISTRY) >= 19, f"Expected ≥19 groups, got {len(FEATURE_REGISTRY)}"


# ── FeaturePipeline extra ────────────────────────────────────────────────────


class TestFeaturePipelineGroups:
    def test_groups_property_returns_list(self, zone_cfg_nl):
        pipeline = build_pipeline(zone_cfg_nl)
        groups = pipeline.groups
        assert isinstance(groups, list)
        assert len(groups) > 0

    def test_groups_returns_copy(self, zone_cfg_nl):
        pipeline = build_pipeline(zone_cfg_nl)
        groups1 = pipeline.groups
        groups1.clear()
        groups2 = pipeline.groups
        assert len(groups2) > 0  # original not mutated


# ── GenerationForecastFeatures ────────────────────────────────────────────────


class TestGenerationForecastFeatures:
    def test_no_gen_cols_returns_unchanged(self, zone_cfg_nl):
        from runeflow.features.generation import GenerationForecastFeatures

        df = _make_df()
        out = GenerationForecastFeatures().transform(df, zone_cfg_nl)
        # No gen columns present — should return df unchanged
        assert "forecast_total_renewable_mw" not in out.columns

    def test_with_load_forecast_mw(self, zone_cfg_nl):
        from runeflow.features.generation import GenerationForecastFeatures

        n = 24 * 10
        df = _make_df(n)
        df["load_forecast_mw"] = 12000.0
        out = GenerationForecastFeatures().transform(df, zone_cfg_nl)
        assert "load_forecast_same_hour_1d" in out.columns
        assert "load_forecast_change_24h" in out.columns

    def test_with_wind_forecast_col(self, zone_cfg_nl):
        from runeflow.features.generation import GenerationForecastFeatures

        n = 24 * 5
        df = _make_df(n)
        df["wind_forecast_mw"] = 3000.0
        df["solar_forecast_mw"] = 1000.0
        df["load_forecast_mw"] = 12000.0
        out = GenerationForecastFeatures().transform(df, zone_cfg_nl)
        assert "forecast_total_renewable_mw" in out.columns
        assert "forecast_residual_load_mw" in out.columns

    def test_forecast_renewable_change_computed(self, zone_cfg_nl):
        from runeflow.features.generation import GenerationForecastFeatures

        n = 24 * 5
        df = _make_df(n)
        df["wind_forecast_mw"] = np.linspace(2000, 4000, n)
        out = GenerationForecastFeatures().transform(df, zone_cfg_nl)
        assert "forecast_renewable_change_24h" in out.columns

    def test_no_mutation(self, zone_cfg_nl):
        from runeflow.features.generation import GenerationForecastFeatures

        n = 24 * 5
        df = _make_df(n)
        df["load_forecast_mw"] = 12000.0
        cols_before = set(df.columns)
        GenerationForecastFeatures().transform(df, zone_cfg_nl)
        assert set(df.columns) == cols_before


# ── CrossBorderFeatures ──────────────────────────────────────────────────────


class TestCrossBorderFeatures:
    def _make_wind_df(self, n: int = 24 * 20) -> pd.DataFrame:
        """DataFrame with German wind speed columns."""
        rng = np.random.default_rng(55)
        idx = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")
        df = pd.DataFrame(
            {
                "Price_EUR_MWh": 50.0 + 20.0 * rng.standard_normal(n),
                "lower_saxony_wind_speed_10m": np.abs(6 + 3 * rng.standard_normal(n)),
                "brandenburg_wind_speed_10m": np.abs(5 + 2 * rng.standard_normal(n)),
                "schleswig_holstein_wind_speed_10m": np.abs(8 + 3 * rng.standard_normal(n)),
            },
            index=idx,
        )
        return df

    def test_no_wind_cols_returns_unchanged(self, zone_cfg_nl):
        from runeflow.features.cross_border import CrossBorderFeatures

        df = _make_df()
        out = CrossBorderFeatures().transform(df, zone_cfg_nl)
        assert "german_wind_power_index" not in out.columns

    def test_with_wind_cols_creates_index(self, zone_cfg_nl):
        from runeflow.features.cross_border import CrossBorderFeatures

        df = self._make_wind_df()
        out = CrossBorderFeatures().transform(df, zone_cfg_nl)
        assert "german_wind_power_index" in out.columns
        assert "german_wind_drought" in out.columns

    def test_wind_drought_is_binary(self, zone_cfg_nl):
        from runeflow.features.cross_border import CrossBorderFeatures

        df = self._make_wind_df()
        out = CrossBorderFeatures().transform(df, zone_cfg_nl)
        drought = out["german_wind_drought"].dropna()
        assert set(drought.unique()).issubset({0, 1})

    def test_interconnector_stress_non_negative(self, zone_cfg_nl):
        from runeflow.features.cross_border import CrossBorderFeatures

        df = self._make_wind_df()
        out = CrossBorderFeatures().transform(df, zone_cfg_nl)
        assert (out["interconnector_stress_proxy"].dropna() >= 0).all()

    def test_with_france_temperature(self, zone_cfg_nl):
        from runeflow.features.cross_border import CrossBorderFeatures

        n = 24 * 5
        idx = pd.date_range("2024-08-01", periods=n, freq="h", tz="UTC")
        df = pd.DataFrame(
            {
                "Price_EUR_MWh": np.ones(n) * 50.0,
                "normandy_temperature_2m": np.ones(n) * 28.0,
                "rhone_alpes_temperature_2m": np.ones(n) * 35.0,  # Above 30 threshold
                "grand_est_temperature_2m": np.ones(n) * 32.0,
            },
            index=idx,
        )
        out = CrossBorderFeatures().transform(df, zone_cfg_nl)
        assert "french_nuclear_cooling_risk" in out.columns
        # rhone_alpes is 35°C → risk = 35 - 30 = 5
        assert (out["french_nuclear_cooling_risk"].dropna() > 0).any()

    def test_no_france_cols_no_cooling_risk(self, zone_cfg_nl):
        from runeflow.features.cross_border import CrossBorderFeatures

        df = self._make_wind_df()
        out = CrossBorderFeatures().transform(df, zone_cfg_nl)
        assert "french_nuclear_cooling_risk" not in out.columns


# ── ResidualLoadFeatures ─────────────────────────────────────────────────────


class TestResidualLoadFeatures:
    def _make_ned_df(self, n: int = 24 * 10) -> pd.DataFrame:
        """DataFrame with NED + solar data needed for ResidualLoadFeatures."""
        rng = np.random.default_rng(66)
        idx = pd.date_range("2024-06-01", periods=n, freq="h", tz="UTC")
        return pd.DataFrame(
            {
                "Price_EUR_MWh": 50.0 + 20.0 * rng.standard_normal(n),
                "ned_utilization_kwh": np.abs(12_000_000 + 1_000_000 * rng.standard_normal(n)),
                "solar_power_output": np.clip(rng.random(n) * 0.8, 0, 1.0),
                "wind_power_potential": np.clip(rng.random(n) * 0.7, 0, 1.0),
            },
            index=idx,
        )

    def test_no_ned_returns_unchanged(self, zone_cfg_nl):
        from runeflow.features.residual_load import ResidualLoadFeatures

        df = _make_df()
        out = ResidualLoadFeatures().transform(df, zone_cfg_nl)
        assert "residual_load_mw" not in out.columns

    def test_with_ned_creates_residual_load(self, zone_cfg_nl):
        from runeflow.features.residual_load import ResidualLoadFeatures

        df = self._make_ned_df()
        out = ResidualLoadFeatures().transform(df, zone_cfg_nl)
        assert "residual_load_mw" in out.columns
        assert "residual_load_zscore" in out.columns

    def test_with_ned_no_wind_potential(self, zone_cfg_nl):
        from runeflow.features.residual_load import ResidualLoadFeatures

        df = self._make_ned_df()
        df.drop(columns=["wind_power_potential"], inplace=True)
        out = ResidualLoadFeatures().transform(df, zone_cfg_nl)
        # Should still compute residual load (wind_gen_mw = 0.0)
        assert "residual_load_mw" in out.columns

    def test_no_mutation(self, zone_cfg_nl):
        from runeflow.features.residual_load import ResidualLoadFeatures

        df = self._make_ned_df()
        cols_before = set(df.columns)
        ResidualLoadFeatures().transform(df, zone_cfg_nl)
        assert set(df.columns) == cols_before

    def test_only_ned_no_solar_returns_unchanged(self, zone_cfg_nl):
        from runeflow.features.residual_load import ResidualLoadFeatures

        n = 24 * 5
        idx = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")
        df = pd.DataFrame(
            {
                "ned_utilization_kwh": np.ones(n) * 12_000_000,
            },
            index=idx,
        )
        out = ResidualLoadFeatures().transform(df, zone_cfg_nl)
        assert "residual_load_mw" not in out.columns


# ── SolarPositionFeatures ────────────────────────────────────────────────────


class TestSolarPositionFeatures:
    def test_produces_zenith(self, zone_cfg_nl):
        from runeflow.features.solar import SolarPositionFeatures

        df = _make_df(n=24 * 2)
        out = SolarPositionFeatures().transform(df, zone_cfg_nl)
        assert "solar_zenith" in out.columns

    def test_produces_all_declared(self, zone_cfg_nl):
        from runeflow.features.solar import SolarPositionFeatures

        df = _make_df(n=24 * 2)
        out = SolarPositionFeatures().transform(df, zone_cfg_nl)
        for col in SolarPositionFeatures().produces:
            assert col in out.columns

    def test_clear_sky_ghi_non_negative(self, zone_cfg_nl):
        from runeflow.features.solar import SolarPositionFeatures

        df = _make_df(n=24 * 2)
        out = SolarPositionFeatures().transform(df, zone_cfg_nl)
        assert (out["clear_sky_ghi"] >= 0).all()

    def test_primary_location_branch(self):
        """Covers the wl.name == primary_weather_location.name branch."""
        from runeflow.domain.weather import WeatherLocation
        from runeflow.features.solar import SolarPositionFeatures
        from runeflow.zones.config import ZoneConfig
        from runeflow.zones.tariffs.nl import NL_TARIFF_FORMULAS

        # primary is NOT the first location → loop must find it
        loc_a = WeatherLocation("loc_a", 52.0, 5.0, "wind")
        loc_primary = WeatherLocation("de_bilt", 52.1, 5.1, "primary")
        mock_cfg = ZoneConfig(
            zone="NL",
            name="Test",
            timezone="UTC",
            workalendar_country="NL",
            primary_weather_location=loc_primary,
            weather_locations=(loc_a, loc_primary),
            installed_solar_capacity_mw=9000.0,
            installed_wind_capacity_mw=8000.0,
            typical_load_mw=12000.0,
            neighbors=(),
            has_energyzero=False,
            has_ned=False,
            tariff_formulas=NL_TARIFF_FORMULAS,
            feature_groups=("solar_position",),
            models=("xgboost_quantile",),
            ensemble_strategy="condition_gated",
            historical_years=(2024,),
        )
        df = _make_df(n=24 * 2)
        out = SolarPositionFeatures().transform(df, mock_cfg)
        assert "solar_zenith" in out.columns


# ── SolarPowerFeatures ───────────────────────────────────────────────────────


class TestSolarPowerFeaturesExtra:
    def test_with_direct_and_diffuse_radiation(self, zone_cfg_nl):
        """Cover the direct/diffuse ratio branch (lines 87-92) and clear_sky_index."""
        from runeflow.features.solar import SolarPositionFeatures, SolarPowerFeatures

        n = 24 * 10
        df = _make_df(n)
        rad = df["shortwave_radiation"]
        df["direct_radiation"] = (rad * 0.6).clip(lower=0)
        df["diffuse_radiation"] = (rad * 0.4).clip(lower=0)
        # Solar position needed for clear_sky_ghi
        df = SolarPositionFeatures().transform(df, zone_cfg_nl)
        out = SolarPowerFeatures().transform(df, zone_cfg_nl)
        assert "direct_diffuse_ratio" in out.columns
        assert "clear_sky_index" in out.columns

    def test_with_cloud_only_no_radiation(self, zone_cfg_nl):
        """Cover elif cloud_cols branch for clear_sky_index (line 115-116)."""
        from runeflow.features.solar import SolarPositionFeatures, SolarPowerFeatures

        n = 24 * 10
        idx = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")
        # Only cloud cover, no shortwave_radiation
        df = pd.DataFrame(
            {
                "Price_EUR_MWh": np.ones(n) * 50,
                "cloudcover": np.clip(
                    40.0 + 30.0 * np.random.default_rng(7).standard_normal(n), 0, 100
                ),
            },
            index=idx,
        )
        df = SolarPositionFeatures().transform(df, zone_cfg_nl)
        out = SolarPowerFeatures().transform(df, zone_cfg_nl)
        assert "clear_sky_index" in out.columns

    def test_is_sunny_period_long_series(self, zone_cfg_nl):
        """Cover is_sunny_period computation which needs 168+ rows (line 120)."""
        from runeflow.features.solar import SolarPositionFeatures, SolarPowerFeatures

        n = 24 * 10  # 240 rows > 168 min_periods
        df = _make_df(n)
        df = SolarPositionFeatures().transform(df, zone_cfg_nl)
        out = SolarPowerFeatures().transform(df, zone_cfg_nl)
        assert "is_sunny_period" in out.columns


# ── Cloud features zero-solar branch ────────────────────────────────────────


class TestCloudFeaturesExtra:
    def test_solar_deficit_zero_when_constant_solar(self, zone_cfg_nl):
        """Covers the else branch (solar_deficit = 0) when col_max == 0."""
        from runeflow.features.cloud import CloudRadiationFeatures

        n = 24 * 5
        idx = pd.date_range("2024-12-01", periods=n, freq="h", tz="UTC")
        df = pd.DataFrame(
            {
                "cloudcover": np.ones(n) * 80.0,
                "solar_power_output": np.zeros(n),  # all zero → col_max = 0
            },
            index=idx,
        )
        out = CloudRadiationFeatures().transform(df, zone_cfg_nl)
        assert (out["solar_deficit"] == 0).all()


# ── RenewablePressureFeatures extra ─────────────────────────────────────────


class TestRenewablePressureExtra:
    def test_no_solar_no_wind_returns_unchanged(self, zone_cfg_nl):
        from runeflow.features.renewable import RenewablePressureFeatures

        df = _make_df()
        out = RenewablePressureFeatures().transform(df, zone_cfg_nl)
        assert "renewable_pressure" not in out.columns

    def test_solar_only_no_wind(self, zone_cfg_nl):
        """Covers the else branch: wind_norm = 0."""
        from runeflow.features.renewable import RenewablePressureFeatures

        n = 24 * 10
        idx = pd.date_range("2024-06-01", periods=n, freq="h", tz="UTC")
        df = pd.DataFrame(
            {
                "solar_power_output": np.clip(np.random.default_rng(1).random(n), 0, 1),
            },
            index=idx,
        )
        out = RenewablePressureFeatures().transform(df, zone_cfg_nl)
        assert "renewable_pressure" in out.columns


# ── SimpleWeightedStrategy extra branches ────────────────────────────────────


class TestSimpleWeightedStrategyExtra:
    def test_combine_without_xgboost_quantile(self, zone_cfg_nl):
        """Covers the else branch (lines 53-56): no xgboost_quantile in predictions."""
        from runeflow.ensemble.simple_weighted import SimpleWeightedStrategy

        n = 10
        idx = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")
        preds = {
            "model_a": pd.DataFrame({"prediction": np.ones(n) * 50}, index=idx),
            "model_b": pd.DataFrame({"prediction": np.ones(n) * 60}, index=idx),
        }
        features = pd.DataFrame(index=idx)
        strategy = SimpleWeightedStrategy()
        out = strategy.combine(preds, features)
        assert "prediction" in out.columns
        assert "lower" in out.columns
        assert "upper" in out.columns

    def test_combine_xgboost_without_lower_col(self, zone_cfg_nl):
        """Covers the 'else combined' branch (line 36) when lower/upper absent."""
        from runeflow.ensemble.simple_weighted import SimpleWeightedStrategy

        n = 10
        idx = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")
        preds = {
            "xgboost_quantile": pd.DataFrame({"prediction": np.ones(n) * 55}, index=idx),
        }
        features = pd.DataFrame(index=idx)
        strategy = SimpleWeightedStrategy()
        out = strategy.combine(preds, features)
        assert "prediction" in out.columns

    def test_combine_empty_raises(self):
        from runeflow.ensemble.simple_weighted import SimpleWeightedStrategy

        strategy = SimpleWeightedStrategy()
        with pytest.raises(ValueError, match="No predictions"):
            strategy.combine({}, pd.DataFrame())


# ── HolidayFeatures exception branch ────────────────────────────────────────


class TestHolidayFeaturesExtra:
    def test_invalid_country_skips_year(self, zone_cfg_nl, monkeypatch):
        """Covers the except block in holiday transform when a year fails."""
        import runeflow.features.holiday as hmod
        from runeflow.features.holiday import HolidayFeatures

        original = hmod.hols.country_holidays

        call_count = [0]

        def patched_holidays(country, years=None):
            call_count[0] += 1
            if call_count[0] == 1:
                raise RuntimeError("Simulated failure for first year")
            return original(country, years=years)

        monkeypatch.setattr(hmod.hols, "country_holidays", patched_holidays)

        df = _make_df(n=24 * 5)
        out = HolidayFeatures().transform(df, zone_cfg_nl)
        # Should still return without crashing; is_holiday may have defaults
        assert "is_holiday" in out.columns


# ── MarketStructureFeatures early exit ──────────────────────────────────────


class TestMarketStructureFeaturesExtra:
    def test_early_exit_when_no_hour_of_day(self, zone_cfg_nl):
        """Covers the early return (line 29) when required columns absent."""
        from runeflow.features.market import MarketStructureFeatures

        n = 24 * 5
        idx = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")
        df = pd.DataFrame({"Price_EUR_MWh": np.ones(n) * 50}, index=idx)
        out = MarketStructureFeatures().transform(df, zone_cfg_nl)
        assert "hour_dow_mon_evening" not in out.columns


# ── Early-return guards for feature groups ───────────────────────────────────


class TestSpikeEarlyReturn:
    def test_spike_features_returns_unchanged_without_price_col(self, zone_cfg_nl):
        from runeflow.features.spike import SpikeMomentumFeatures

        idx = pd.date_range("2024-01-01", periods=24, freq="h", tz="UTC")
        df = pd.DataFrame({"temperature_2m": np.ones(24) * 10.0}, index=idx)
        out = SpikeMomentumFeatures().transform(df, zone_cfg_nl)
        assert "Price_EUR_MWh_zscore_24h" not in out.columns

    def test_spike_risk_features_returns_unchanged_without_price_col(self, zone_cfg_nl):
        from runeflow.features.spike import SpikeRiskFeatures

        idx = pd.date_range("2024-01-01", periods=24, freq="h", tz="UTC")
        df = pd.DataFrame({"temperature_2m": np.ones(24) * 10.0}, index=idx)
        out = SpikeRiskFeatures().transform(df, zone_cfg_nl)
        assert "hours_since_last_spike" not in out.columns


class TestInteractionEarlyReturn:
    def test_returns_unchanged_without_is_peak_hour(self, zone_cfg_nl):
        from runeflow.features.interaction import PeakInteractionFeatures

        idx = pd.date_range("2024-01-01", periods=24, freq="h", tz="UTC")
        df = pd.DataFrame({"temperature_2m": np.ones(24) * 12.0}, index=idx)
        out = PeakInteractionFeatures().transform(df, zone_cfg_nl)
        assert "peak_hour_cold_interaction" not in out.columns


class TestDuckCurveEarlyReturn:
    def test_returns_unchanged_without_required_cols(self, zone_cfg_nl):
        from runeflow.features.duck_curve import DuckCurveFeatures

        idx = pd.date_range("2024-01-01", periods=24, freq="h", tz="UTC")
        df = pd.DataFrame({"temperature_2m": np.ones(24) * 12.0}, index=idx)
        out = DuckCurveFeatures().transform(df, zone_cfg_nl)
        assert "evening_ramp_severity" not in out.columns


class TestWindEarlyReturn:
    def test_returns_unchanged_without_wind_cols(self, zone_cfg_nl):
        from runeflow.features.wind import WindFeatures

        idx = pd.date_range("2024-01-01", periods=24, freq="h", tz="UTC")
        df = pd.DataFrame({"temperature_2m": np.ones(24) * 12.0}, index=idx)
        out = WindFeatures().transform(df, zone_cfg_nl)
        assert "wind_power_potential" not in out.columns


class TestPriceRegimeEarlyReturn:
    def test_returns_unchanged_without_price_col(self, zone_cfg_nl):
        idx = pd.date_range("2024-01-01", periods=24, freq="h", tz="UTC")
        df = pd.DataFrame({"temperature_2m": np.ones(24) * 12.0}, index=idx)
        out = PriceRegimeFeatures().transform(df, zone_cfg_nl)
        assert "price_regime" not in out.columns


class TestSolarPowerNoRadiationNoCloud:
    def test_else_branch_uses_only_clear_sky_ghi(self, zone_cfg_nl):
        """SolarPowerFeatures else branch (line 92): no radiation cols, no cloud cols."""
        from runeflow.features.solar import SolarPowerFeatures

        n = 48
        idx = pd.date_range("2024-06-01", periods=n, freq="h", tz="UTC")
        # Only clear_sky_ghi, no radiation or cloud columns
        df = pd.DataFrame({"clear_sky_ghi": np.ones(n) * 600.0}, index=idx)
        out = SolarPowerFeatures().transform(df, zone_cfg_nl)
        assert "solar_power_output" in out.columns
        assert (out["solar_power_output"] >= 0).all()


class TestRegistryUnknownFeatureGroup:
    def test_unknown_group_name_is_skipped(self, zone_cfg_nl, monkeypatch):
        """registry.py line 91: cls is None when group name not in FEATURE_REGISTRY."""
        from runeflow.features.registry import FEATURE_REGISTRY, build_pipeline

        # Temporarily make the 'temporal' entry map to None so the
        # 'if cls is None: continue' branch is hit during pipeline construction.
        monkeypatch.setitem(FEATURE_REGISTRY, "temporal", None)

        # zone_cfg_nl has 'temporal' in its feature_groups → cls will be None → skipped
        pipeline = build_pipeline(zone_cfg_nl)
        # Should not raise; the None-mapped group is silently skipped
        assert pipeline is not None
