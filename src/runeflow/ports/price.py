# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2024-2026 Thomas Phil — runeflow
# See LICENSE and COMMERCIAL-LICENSE.md for licensing details.

"""PricePort — abstract interface for electricity price adapters."""

from __future__ import annotations

import datetime
from abc import ABC, abstractmethod

from runeflow.domain.price import PriceSeries


class PricePort(ABC):
    """Download historical and/or day-ahead electricity prices."""

    @abstractmethod
    def download_historical(
        self,
        zone: str,
        start: datetime.date,
        end: datetime.date,
    ) -> PriceSeries:
        """Download spot prices for a date range (inclusive)."""

    @abstractmethod
    def download_day_ahead(self, zone: str) -> PriceSeries | None:
        """Download tomorrow's day-ahead prices if published."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable adapter name for logging."""

    @abstractmethod
    def supports_zone(self, zone: str) -> bool:
        """Return True if this adapter can serve the given zone."""
