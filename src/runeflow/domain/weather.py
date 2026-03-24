# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2024-2026 Thomas Phil — runeflow
# See LICENSE and COMMERCIAL-LICENSE.md for licensing details.

"""Weather domain types."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class WeatherLocation:
    """Named geographic location for weather queries."""

    name: str
    lat: float
    lon: float
    purpose: str  # "primary", "wind", "nuclear", "solar"


@dataclass(frozen=True)
class WeatherRecord:
    """Single hourly weather observation/forecast for one location."""

    timestamp: pd.Timestamp
    location: str  # WeatherLocation.name
    temperature_2m: float | None = None
    wind_speed_10m: float | None = None
    wind_direction_10m: float | None = None
    shortwave_radiation: float | None = None
    cloud_cover: float | None = None
    relative_humidity: float | None = None
    precipitation: float | None = None
    wind_gusts_10m: float | None = None
    diffuse_radiation: float | None = None
    direct_radiation: float | None = None
    is_day: float | None = None


@dataclass(frozen=True)
class WeatherSeries:
    """Multi-location weather data (wide format, columns prefixed by location)."""

    locations: tuple[str, ...]
    df: pd.DataFrame  # Wide: DatetimeIndex × location-prefixed columns
    source: str
    fetched_at: pd.Timestamp

    def __post_init__(self) -> None:
        # Allow dict-like access: series["de_bilt_temperature_2m"]
        object.__setattr__(self, "df", self.df.copy())

    def to_dataframe(self) -> pd.DataFrame:
        """Return the wide-format DataFrame (DatetimeIndex)."""
        return self.df.copy()

    @classmethod
    def from_location_frames(
        cls,
        frames: dict[str, pd.DataFrame],
        source: str,
        fetched_at: pd.Timestamp | None = None,
    ) -> WeatherSeries:
        """
        Build a WeatherSeries by merging per-location DataFrames.

        Each frame has a DatetimeIndex and weather columns.
        Columns will be prefixed with ``{location_name}_``.
        """
        if fetched_at is None:
            fetched_at = pd.Timestamp.now("UTC")

        prefixed_frames = []
        for loc_name, df in frames.items():
            renamed = df.rename(columns={c: f"{loc_name}_{c}" for c in df.columns})
            prefixed_frames.append(renamed)

        if not prefixed_frames:
            wide = pd.DataFrame()
        else:
            wide = prefixed_frames[0]
            for rest in prefixed_frames[1:]:
                wide = wide.join(rest, how="outer")

        return cls(
            locations=tuple(frames.keys()),
            df=wide,
            source=source,
            fetched_at=fetched_at,
        )
