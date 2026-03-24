# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2024-2026 Thomas Phil — runeflow
# See LICENSE and COMMERCIAL-LICENSE.md for licensing details.

"""ZoneConfig and NeighborZone dataclasses."""

from __future__ import annotations

from dataclasses import dataclass

from runeflow.domain.tariff import TariffFormula
from runeflow.domain.weather import WeatherLocation


@dataclass(frozen=True)
class NeighborZone:
    """Cross-border weather/generation dependency."""

    zone: str  # ENTSO-E zone code
    purpose: str  # "wind_import", "nuclear_import", etc.
    weather_locations: tuple[WeatherLocation, ...]


@dataclass(frozen=True)
class ZoneConfig:
    """Everything the system needs to know about a single ENTSO-E zone."""

    # Identity
    zone: str  # ENTSO-E bidding zone code ("NL", "DE_LU")
    name: str  # "Netherlands"
    timezone: str  # IANA timezone ("Europe/Amsterdam")
    workalendar_country: str  # ISO country code for holidays ("NL", "DE")

    # Weather
    primary_weather_location: WeatherLocation
    weather_locations: tuple[WeatherLocation, ...]  # All locations incl. cross-border

    # Grid capacities (for residual load estimation)
    installed_solar_capacity_mw: float  # NL: ~9000
    installed_wind_capacity_mw: float  # NL: ~8000
    typical_load_mw: float  # NL: ~12000

    # Neighbors (for cross-border features)
    neighbors: tuple[NeighborZone, ...]

    # Adapters flags
    has_energyzero: bool  # EnergyZero fallback available?
    has_ned: bool  # NED supplemental data available?

    # Tariffs
    tariff_formulas: dict[str, TariffFormula]  # Provider ID → formula

    # ML configuration
    feature_groups: tuple[str, ...]  # Names of feature groups to use
    models: tuple[str, ...]  # Model names from MODEL_REGISTRY
    ensemble_strategy: str  # Strategy name from ENSEMBLE_REGISTRY

    # Training
    historical_years: tuple[int, ...]  # Years to download
    min_training_years: int = 2  # Minimum years required
