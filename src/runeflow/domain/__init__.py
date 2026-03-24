# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2024-2026 Thomas Phil — runeflow
# See LICENSE and COMMERCIAL-LICENSE.md for licensing details.

"""Domain layer: pure data types, no I/O."""

from runeflow.domain.price import PriceRecord, PriceSeries
from runeflow.domain.weather import WeatherLocation, WeatherRecord, WeatherSeries
from runeflow.domain.generation import GenerationSeries
from runeflow.domain.forecast import ForecastPoint, ForecastResult
from runeflow.domain.training import TrainResult
from runeflow.domain.tariff import TariffFormula, TariffRateSlot

__all__ = [
    "PriceRecord",
    "PriceSeries",
    "WeatherLocation",
    "WeatherRecord",
    "WeatherSeries",
    "GenerationSeries",
    "ForecastPoint",
    "ForecastResult",
    "TrainResult",
    "TariffFormula",
    "TariffRateSlot",
]