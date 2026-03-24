# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2024-2026 Thomas Phil — runeflow
# See LICENSE and COMMERCIAL-LICENSE.md for licensing details.

"""ModelPort — abstract interface for prediction models."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

import pandas as pd

if TYPE_CHECKING:
    from runeflow.ports.store import DataStore


class ModelPort(ABC):
    """Interface for all prediction models."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Model name (used as registry key)."""

    @abstractmethod
    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame | None = None,
        y_val: pd.Series | None = None,
        sample_weight: pd.Series | None = None,
    ) -> dict[str, Any]:
        """Train model; return metrics dict."""

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        """Return DataFrame with model-specific prediction columns."""

    @abstractmethod
    def save(self, store: "DataStore", zone: str) -> None:
        """Persist model artifacts via a DataStore."""

    @abstractmethod
    def load(self, store: "DataStore", zone: str) -> bool:
        """Load model artifacts; return True on success."""

    @property
    @abstractmethod
    def is_trained(self) -> bool:
        """True after a successful train() call."""