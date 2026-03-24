# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2024-2026 Thomas Phil — runeflow
# See LICENSE and COMMERCIAL-LICENSE.md for licensing details.

"""DataValidator — abstract interface for data quality checks."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import pandas as pd


@dataclass
class ValidationResult:
    """Result of a validation run."""

    passed: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def __bool__(self) -> bool:  # noqa: D105
        return self.passed


class DataValidator(ABC):
    """Run data quality checks on a DataFrame."""

    @abstractmethod
    def validate(
        self,
        df: pd.DataFrame,
        context: str = "",
    ) -> ValidationResult:
        """Run all checks; return aggregated result."""
