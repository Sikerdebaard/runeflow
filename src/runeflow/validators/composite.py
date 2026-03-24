# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2024-2026 Thomas Phil — runeflow
# See LICENSE and COMMERCIAL-LICENSE.md for licensing details.

"""CompositeValidator — runs a chain of individual DQ checks."""

from __future__ import annotations

import pandas as pd
from loguru import logger

from runeflow.ports.validator import DataValidator, ValidationResult
from runeflow.validators.checks import (
    ContinuityCheck,
    DuplicatesCheck,
    NaNCheck,
    PriceRangeCheck,
    RowCountCheck,
    StalenessCheck,
    TimezoneCheck,
)

_Check = object  # Any callable(df, context) → ValidationResult


class CompositeValidator(DataValidator):
    """Runs a list of check callables in order; aggregates results."""

    def __init__(self, checks: list[_Check]) -> None:
        self._checks = checks

    def validate(
        self,
        df: pd.DataFrame,
        context: str = "",
    ) -> ValidationResult:
        all_errors: list[str] = []
        all_warnings: list[str] = []

        for check in self._checks:
            result: ValidationResult = check(df, context)  # type: ignore[operator]
            all_errors.extend(result.errors)
            all_warnings.extend(result.warnings)

        if all_warnings:
            for w in all_warnings:
                logger.warning(w)
        if all_errors:
            for e in all_errors:
                logger.error(e)

        return ValidationResult(
            passed=len(all_errors) == 0,
            errors=all_errors,
            warnings=all_warnings,
        )


def default_validator() -> CompositeValidator:
    """Return the standard set of DQ checks used in production."""
    return CompositeValidator(
        checks=[
            RowCountCheck(min_rows=1),
            TimezoneCheck(),
            DuplicatesCheck(),
            ContinuityCheck(),
            NaNCheck(threshold=0.20),
            PriceRangeCheck(),
            StalenessCheck(max_days=3),
        ]
    )


def price_validator() -> CompositeValidator:
    """Strict validator for electricity price DataFrames."""
    return CompositeValidator(
        checks=[
            RowCountCheck(min_rows=100),
            TimezoneCheck(),
            DuplicatesCheck(),
            ContinuityCheck(),
            NaNCheck(threshold=0.05),
            PriceRangeCheck(low=-500.0, high=4000.0),
            StalenessCheck(max_days=2),
        ]
    )
