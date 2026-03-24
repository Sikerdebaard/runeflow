# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2024-2026 Thomas Phil — runeflow
# See LICENSE and COMMERCIAL-LICENSE.md for licensing details.

"""EnsembleStrategy — abstract interface for ensemble combination logic."""

from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd


class EnsembleStrategy(ABC):
    """Strategy for combining multiple model predictions."""

    @abstractmethod
    def combine(
        self,
        predictions: dict[str, pd.DataFrame],
        features: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Combine predictions from multiple models.

        Args:
            predictions: mapping of model_name → model output DataFrame
            features: the input feature matrix (for condition gating)

        Returns:
            DataFrame with columns:
            - prediction: central estimate (EUR/MWh)
            - lower: P1 lower bound
            - upper: P99 upper bound
            - uncertainty: interval width (upper - lower)
            - model_agreement: 0-1 score (1 = models fully agree)
        """

    @property
    @abstractmethod
    def name(self) -> str:
        """Strategy name (used as registry key)."""
