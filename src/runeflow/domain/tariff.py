# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2024-2026 Thomas Phil — runeflow
# See LICENSE and COMMERCIAL-LICENSE.md for licensing details.

"""Tariff domain types."""
from __future__ import annotations

import datetime
from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class TariffFormula:
    """End-user tariff formula for a specific provider."""

    provider_id: str          # e.g. "zonneplan", "tibber"
    country: str              # e.g. "NL", "DE"
    label: str                # Human-readable name
    apply: Callable[[float, datetime.date], float]
    # Input: wholesale EUR/kWh, date (for yearly tax rates)
    # Output: all-in EUR/kWh incl. taxes


@dataclass(frozen=True)
class TariffRateSlot:
    """Single price slot for tariff JSON export."""

    start: str    # ISO 8601 timestamp string
    end: str      # ISO 8601 timestamp string
    price: float  # EUR/kWh (all-in)