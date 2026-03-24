# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2024-2026 Thomas Phil — runeflow
# See LICENSE and COMMERCIAL-LICENSE.md for licensing details.

"""Individual DQ check classes."""

from __future__ import annotations

import pandas as pd

from runeflow.ports.validator import ValidationResult


class ContinuityCheck:
    """Verify there are no hourly gaps in the index."""

    def __call__(self, df: pd.DataFrame, context: str) -> ValidationResult:
        idx = self._extract_datetime_index(df)
        if idx is None or len(idx) < 2:
            return ValidationResult(passed=True)

        expected = pd.date_range(idx.min(), idx.max(), freq="h")
        missing = expected.difference(idx)
        if missing.empty:
            return ValidationResult(passed=True)
        return ValidationResult(
            passed=False,
            errors=[
                f"[{context}] ContinuityCheck: {len(missing)} missing hourly slots "
                f"(first: {missing[0]})."
            ],
        )

    @staticmethod
    def _extract_datetime_index(df: pd.DataFrame) -> pd.DatetimeIndex | None:
        if isinstance(df.index, pd.DatetimeIndex):
            return df.index
        if "date" in df.columns:
            return pd.DatetimeIndex(pd.to_datetime(df["date"]))
        return None


class NaNCheck:
    """Warn when a column exceeds a NaN threshold."""

    def __init__(self, threshold: float = 0.10) -> None:
        self._threshold = threshold

    def __call__(self, df: pd.DataFrame, context: str) -> ValidationResult:
        warnings: list[str] = []
        for col in df.columns:
            frac = df[col].isna().mean()
            if frac > self._threshold:
                warnings.append(f"[{context}] NaNCheck: column '{col}' has {frac:.1%} NaN values.")
        return ValidationResult(passed=True, warnings=warnings)


class PriceRangeCheck:
    """Flag prices outside the plausible range."""

    def __init__(
        self,
        low: float = -500.0,
        high: float = 4000.0,
        col: str = "Price_EUR_MWh",
    ) -> None:
        self._low = low
        self._high = high
        self._col = col

    def __call__(self, df: pd.DataFrame, context: str) -> ValidationResult:
        if self._col not in df.columns:
            # No price column — skip
            return ValidationResult(passed=True)
        out_of_range = df[(df[self._col] < self._low) | (df[self._col] > self._high)]
        if out_of_range.empty:
            return ValidationResult(passed=True)
        return ValidationResult(
            passed=False,
            errors=[
                f"[{context}] PriceRangeCheck: {len(out_of_range)} rows outside "
                f"[{self._low}, {self._high}] EUR/MWh."
            ],
        )


class TimezoneCheck:
    """Ensure timestamp index is timezone-aware."""

    def __call__(self, df: pd.DataFrame, context: str) -> ValidationResult:
        if isinstance(df.index, pd.DatetimeIndex) and df.index.tz is None:
            return ValidationResult(
                passed=False,
                errors=[f"[{context}] TimezoneCheck: DataFrame index is timezone-naive."],
            )
        return ValidationResult(passed=True)


class DuplicatesCheck:
    """Ensure no duplicate timestamps exist."""

    def __call__(self, df: pd.DataFrame, context: str) -> ValidationResult:
        idx = df.index
        if isinstance(idx, pd.DatetimeIndex):
            dupes = idx[idx.duplicated()]
            if len(dupes):
                return ValidationResult(
                    passed=False,
                    errors=[
                        f"[{context}] DuplicatesCheck: {len(dupes)} duplicate timestamps "
                        f"(e.g. {dupes[0]})."
                    ],
                )
        if "date" in df.columns:
            n_dupes = df["date"].duplicated().sum()
            if n_dupes:
                return ValidationResult(
                    passed=False,
                    errors=[f"[{context}] DuplicatesCheck: {n_dupes} duplicate 'date' values."],
                )
        return ValidationResult(passed=True)


class StalenessCheck:
    """Warn if the data ends more than *max_days* ago."""

    def __init__(self, max_days: int = 3) -> None:
        self._max_days = max_days

    def __call__(self, df: pd.DataFrame, context: str) -> ValidationResult:
        idx = df.index
        if not isinstance(idx, pd.DatetimeIndex) or idx.empty:
            return ValidationResult(passed=True)

        latest = idx.max()
        if latest.tzinfo is None:
            latest = latest.tz_localize("UTC")
        age = pd.Timestamp.now("UTC") - latest
        if age > pd.Timedelta(days=self._max_days):
            return ValidationResult(
                passed=True,
                warnings=[
                    f"[{context}] StalenessCheck: data ends {latest.date()} "
                    f"({age.days} days ago, threshold={self._max_days}d)."
                ],
            )
        return ValidationResult(passed=True)


class RowCountCheck:
    """Fail if the DataFrame has fewer than *min_rows* rows."""

    def __init__(self, min_rows: int = 1) -> None:
        self._min_rows = min_rows

    def __call__(self, df: pd.DataFrame, context: str) -> ValidationResult:
        if len(df) < self._min_rows:
            return ValidationResult(
                passed=False,
                errors=[
                    f"[{context}] RowCountCheck: only {len(df)} rows (minimum {self._min_rows})."
                ],
            )
        return ValidationResult(passed=True)
