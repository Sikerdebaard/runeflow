# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2024-2026 Thomas Phil — runeflow
# See LICENSE and COMMERCIAL-LICENSE.md for licensing details.

"""Generation domain type."""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class GenerationSeries:
    """ENTSO-E generation and load forecasts."""

    zone: str
    df: pd.DataFrame         # Columns: load_forecast_mw, forecast_solar, etc.
    source: str
    fetched_at: pd.Timestamp

    def to_dataframe(self) -> pd.DataFrame:
        return self.df.copy()