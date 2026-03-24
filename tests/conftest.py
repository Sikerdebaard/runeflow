# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2024-2026 Thomas Phil — runeflow
# See LICENSE and COMMERCIAL-LICENSE.md for licensing details.

"""Shared pytest fixtures for runeflow tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from runeflow.zones.registry import ZoneRegistry

# ---------------------------------------------------------------------------
# Time-series helpers
# ---------------------------------------------------------------------------


def _hourly_utc_index(n: int = 24 * 14) -> pd.DatetimeIndex:
    """Return *n* hours of UTC-aware timestamps starting 2024-01-01."""
    return pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")


@pytest.fixture(scope="session")
def hourly_index() -> pd.DatetimeIndex:
    return _hourly_utc_index(24 * 30)  # 30 days


@pytest.fixture()
def sample_price_df(hourly_index: pd.DatetimeIndex) -> pd.DataFrame:
    """Hourly DataFrame with 'Price_EUR_MWh' column, UTC index."""
    rng = np.random.default_rng(42)
    prices = 50.0 + 30.0 * rng.standard_normal(len(hourly_index))
    return pd.DataFrame({"Price_EUR_MWh": prices}, index=hourly_index)


@pytest.fixture()
def sample_weather_df(hourly_index: pd.DatetimeIndex) -> pd.DataFrame:
    """Hourly weather DataFrame with common OpenMeteo columns."""
    rng = np.random.default_rng(0)
    n = len(hourly_index)
    return pd.DataFrame(
        {
            "temperature_2m": 15.0 + 5.0 * rng.standard_normal(n),
            "wind_speed_10m": 5.0 + 3.0 * np.abs(rng.standard_normal(n)),
            "shortwave_radiation": np.clip(200.0 * rng.random(n), 0, None),
            "cloudcover": 40.0 + 30.0 * rng.standard_normal(n),
            "precipitation": np.clip(0.5 * rng.random(n), 0, None),
        },
        index=hourly_index,
    )


@pytest.fixture()
def full_feature_df(hourly_index: pd.DatetimeIndex) -> pd.DataFrame:
    """Realistic DataFrame with both price and weather columns."""
    rng = np.random.default_rng(123)
    n = len(hourly_index)
    return pd.DataFrame(
        {
            "Price_EUR_MWh": 50.0 + 30.0 * rng.standard_normal(n),
            "temperature_2m": 12.0 + 6.0 * rng.standard_normal(n),
            "wind_speed_10m": np.abs(7.0 + 4.0 * rng.standard_normal(n)),
            "shortwave_radiation": np.clip(150.0 + 120.0 * rng.standard_normal(n), 0, None),
            "cloudcover": np.clip(40.0 + 30.0 * rng.standard_normal(n), 0, 100),
            "precipitation": np.clip(rng.exponential(0.2, n), 0, None),
            "wind_generation_mw": np.abs(2000.0 + 800.0 * rng.standard_normal(n)),
            "solar_generation_mw": np.clip(500.0 + 400.0 * rng.standard_normal(n), 0, None),
            "cross_border_flow_mw": 500.0 * rng.standard_normal(n),
        },
        index=hourly_index,
    )


# ---------------------------------------------------------------------------
# Zone fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def zone_cfg_nl():
    return ZoneRegistry.get("NL")


@pytest.fixture(scope="session")
def zone_cfg_de_lu():
    return ZoneRegistry.get("DE_LU")


# ---------------------------------------------------------------------------
# Temp directory
# ---------------------------------------------------------------------------


@pytest.fixture()
def tmp_cache_dir(tmp_path):
    d = tmp_path / "cache"
    d.mkdir()
    return d
