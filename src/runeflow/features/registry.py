# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2024-2026 Thomas Phil — runeflow
# See LICENSE and COMMERCIAL-LICENSE.md for licensing details.

"""Feature group registry and default pipeline builders."""

from __future__ import annotations

from runeflow.zones.config import ZoneConfig

from .base import FeatureGroup, FeaturePipeline
from .cloud import CloudRadiationFeatures
from .cross_border import CrossBorderFeatures
from .duck_curve import DuckCurveFeatures
from .generation import GenerationForecastFeatures
from .holiday import HolidayFeatures
from .interaction import PeakInteractionFeatures
from .market import MarketStructureFeatures
from .precipitation import PrecipitationFeatures
from .price_lag import PriceLagFeatures
from .price_regime import PriceRegimeFeatures
from .renewable import RenewablePressureFeatures
from .residual_load import ResidualLoadFeatures
from .solar import SolarPositionFeatures, SolarPowerFeatures
from .spike import SpikeMomentumFeatures, SpikeRiskFeatures
from .temperature import TemperatureFeatures
from .temporal import TemporalFeatures
from .wind import WindFeatures

# Registry maps group name → class (singleton instances created on demand)
FEATURE_REGISTRY: dict[str, type[FeatureGroup]] = {
    "temporal": TemporalFeatures,
    "solar_position": SolarPositionFeatures,
    "solar_power": SolarPowerFeatures,
    "holiday": HolidayFeatures,
    "price_lag": PriceLagFeatures,
    "price_regime": PriceRegimeFeatures,
    "spike_momentum": SpikeMomentumFeatures,
    "spike_risk": SpikeRiskFeatures,
    "temperature": TemperatureFeatures,
    "wind": WindFeatures,
    "precipitation": PrecipitationFeatures,
    "cloud": CloudRadiationFeatures,
    "renewable_pressure": RenewablePressureFeatures,
    "residual_load": ResidualLoadFeatures,
    "cross_border": CrossBorderFeatures,
    "duck_curve": DuckCurveFeatures,
    "market_structure": MarketStructureFeatures,
    "generation": GenerationForecastFeatures,
    "interaction": PeakInteractionFeatures,
}

# Canonical execution order (dependencies respected)
DEFAULT_ORDER = [
    "temporal",
    "solar_position",
    "solar_power",
    "holiday",
    "price_lag",
    "price_regime",
    "spike_momentum",
    "temperature",
    "wind",
    "precipitation",
    "cloud",
    "renewable_pressure",
    "residual_load",
    "cross_border",
    "duck_curve",
    "market_structure",
    "generation",
    "spike_risk",  # needs temperature + wind_scarcity produced by wind group
    "interaction",
]


def build_pipeline(zone_cfg: ZoneConfig) -> FeaturePipeline:
    """
    Construct a FeaturePipeline from the feature groups listed in *zone_cfg*.

    Groups are instantiated in ``DEFAULT_ORDER`` so that dependency order is
    always respected, regardless of the order in ``zone_cfg.feature_groups``.
    """
    enabled: set[str] = set(zone_cfg.feature_groups)

    groups: list[FeatureGroup] = []
    for name in DEFAULT_ORDER:
        if name not in enabled:
            continue
        cls = FEATURE_REGISTRY.get(name)
        if cls is None:
            continue
        groups.append(cls())

    return FeaturePipeline(groups)
