# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2024-2026 Thomas Phil — runeflow
# See LICENSE and COMMERCIAL-LICENSE.md for licensing details.

"""Training artifact domain types."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class TrainResult:
    """Output of a training run."""

    zone: str
    features: tuple[str, ...]
    metrics: dict[str, float]  # MAE, R², spike_mae, dip_mae, coverage
    quality_assessment: dict[str, Any]
    trained_at: pd.Timestamp
    model_version: str
    data_range: tuple[pd.Timestamp, pd.Timestamp]  # Training data span
