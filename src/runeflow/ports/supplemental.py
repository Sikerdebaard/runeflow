# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2024-2026 Thomas Phil — runeflow
# See LICENSE and COMMERCIAL-LICENSE.md for licensing details.

"""SupplementalDataPort — zone-specific supplemental data (e.g. NED for NL)."""

from __future__ import annotations

import datetime
from abc import ABC, abstractmethod

import pandas as pd


class SupplementalDataPort(ABC):
    """Optional zone-specific supplemental data source."""

    @abstractmethod
    def download(
        self,
        zone: str,
        start: datetime.date,
        end: datetime.date,
    ) -> pd.DataFrame | None:
        """Download supplemental data; return None if unavailable."""

    @abstractmethod
    def download_forecast(self, zone: str) -> pd.DataFrame | None:
        """Download near-term forecast (e.g. NED 9-day)."""

    @abstractmethod
    def supports_zone(self, zone: str) -> bool:
        """Return True if data is available for the given zone."""
