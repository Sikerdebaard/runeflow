# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2024-2026 Thomas Phil — runeflow
# See LICENSE and COMMERCIAL-LICENSE.md for licensing details.

"""GenerationPort — abstract interface for grid generation & load data."""

from __future__ import annotations

import datetime
from abc import ABC, abstractmethod

from runeflow.domain.generation import GenerationSeries


class GenerationPort(ABC):
    """Download TSO generation data and load forecasts."""

    @abstractmethod
    def download_generation(
        self,
        zone: str,
        start: datetime.date,
        end: datetime.date,
    ) -> GenerationSeries | None:
        """Download generation mix and load forecast data."""

    @abstractmethod
    def supports_zone(self, zone: str) -> bool:
        """Return True if this adapter can serve the given zone."""
