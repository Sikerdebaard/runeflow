# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2024-2026 Thomas Phil — runeflow
# See LICENSE and COMMERCIAL-LICENSE.md for licensing details.

"""Tests for all 7 DQ check classes and CompositeValidator."""

from __future__ import annotations

import pandas as pd

from runeflow.validators.checks import (
    ContinuityCheck,
    DuplicatesCheck,
    NaNCheck,
    PriceRangeCheck,
    RowCountCheck,
    StalenessCheck,
    TimezoneCheck,
)
from runeflow.validators.composite import CompositeValidator

# ── Helpers ────────────────────────────────────────────────────────────────


def _hourly_tz(n: int = 48, tz: str = "UTC") -> pd.DatetimeIndex:
    return pd.date_range("2024-06-01", periods=n, freq="h", tz=tz)


def _price_df(n: int = 48, price: float = 55.0) -> pd.DataFrame:
    return pd.DataFrame(
        {"Price_EUR_MWh": [price] * n},
        index=_hourly_tz(n),
    )


# ── ContinuityCheck ─────────────────────────────────────────────────────────


class TestContinuityCheck:
    def test_passes_on_clean_hourly(self):
        df = _price_df()
        result = ContinuityCheck()(df, "test")
        assert result.passed

    def test_fails_on_gap(self):
        idx = _hourly_tz(48)
        # Drop hour 24 to create a gap
        idx_with_gap = idx.delete(24)
        df = pd.DataFrame({"Price_EUR_MWh": 1.0}, index=idx_with_gap)
        result = ContinuityCheck()(df, "test")
        assert not result.passed
        assert any("missing" in e.lower() or "ContinuityCheck" in e for e in result.errors)

    def test_passes_on_single_row(self):
        df = _price_df(n=1)
        result = ContinuityCheck()(df, "test")
        assert result.passed  # < 2 rows → skip

    def test_reports_first_missing_slot(self):
        idx = _hourly_tz(72)
        idx_with_gap = idx.delete([36, 37, 38])
        df = pd.DataFrame({"Price_EUR_MWh": 1.0}, index=idx_with_gap)
        result = ContinuityCheck()(df, "ctx")
        assert not result.passed
        assert "3" in result.errors[0]  # 3 missing slots mentioned

    def test_uses_date_column_if_no_datetimeindex(self):
        n = 24
        dates = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")
        df = pd.DataFrame({"date": dates, "val": 1.0})
        result = ContinuityCheck()(df, "ctx")
        assert result.passed


# ── NaNCheck ────────────────────────────────────────────────────────────────


class TestNaNCheck:
    def test_passes_on_no_nans(self):
        df = _price_df()
        result = NaNCheck(threshold=0.10)(df, "test")
        assert result.passed
        assert len(result.warnings) == 0

    def test_warns_on_high_nan_fraction(self):
        df = _price_df(48)
        # Inject 50 % NaN → exceeds default 10 % threshold
        df.iloc[:24] = float("nan")
        result = NaNCheck(threshold=0.10)(df, "test")
        assert result.passed  # NaNCheck only warns, never fails
        assert len(result.warnings) > 0

    def test_no_warning_just_below_threshold(self):
        df = _price_df(100)
        df.iloc[:9] = float("nan")  # 9% < 10% threshold
        result = NaNCheck(threshold=0.10)(df, "ctx")
        assert len(result.warnings) == 0

    def test_custom_threshold(self):
        df = _price_df(100)
        df.iloc[:15] = float("nan")  # 15% NaN
        result_strict = NaNCheck(threshold=0.05)(df, "ctx")  # 5% threshold
        result_loose = NaNCheck(threshold=0.20)(df, "ctx")  # 20% threshold
        assert len(result_strict.warnings) > 0
        assert len(result_loose.warnings) == 0


# ── PriceRangeCheck ─────────────────────────────────────────────────────────


class TestPriceRangeCheck:
    def test_passes_normal_prices(self):
        df = _price_df()
        result = PriceRangeCheck()(df, "test")
        assert result.passed

    def test_fails_on_extreme_negative(self):
        df = _price_df()
        df.iloc[0] = -9999.0  # below -500 threshold
        result = PriceRangeCheck()(df, "test")
        assert not result.passed
        assert "PriceRangeCheck" in result.errors[0]

    def test_fails_on_extreme_positive(self):
        df = _price_df()
        df.iloc[0] = 99999.0  # above 4000 threshold
        result = PriceRangeCheck()(df, "test")
        assert not result.passed

    def test_passes_at_exact_boundaries(self):
        df = _price_df()
        df.iloc[0] = -500.0
        df.iloc[1] = 4000.0
        result = PriceRangeCheck(low=-500.0, high=4000.0)(df, "test")
        assert result.passed

    def test_custom_col_missing_skips(self):
        """If the named column doesn't exist, the check should pass (skip)."""
        df = pd.DataFrame({"other_col": [1, 2, 3]}, index=_hourly_tz(3))
        result = PriceRangeCheck(col="Price_EUR_MWh")(df, "test")
        assert result.passed

    def test_custom_bounds(self):
        df = _price_df()
        df.iloc[0] = -200.0
        # With tighter low bound of -100:
        result = PriceRangeCheck(low=-100.0, high=4000.0)(df, "test")
        assert not result.passed


# ── TimezoneCheck ───────────────────────────────────────────────────────────


class TestTimezoneCheck:
    def test_passes_tz_aware(self):
        df = _price_df()
        result = TimezoneCheck()(df, "test")
        assert result.passed

    def test_fails_tz_naive(self):
        idx = pd.date_range("2024-01-01", periods=24, freq="h")  # no tz
        df = pd.DataFrame({"Price_EUR_MWh": 1.0}, index=idx)
        result = TimezoneCheck()(df, "test")
        assert not result.passed
        assert "timezone-naive" in result.errors[0]

    def test_passes_non_datetimeindex(self):
        """Non-DatetimeIndex (e.g. RangeIndex) should not fail."""
        df = pd.DataFrame({"a": [1, 2, 3]})
        result = TimezoneCheck()(df, "ctx")
        assert result.passed

    def test_passes_amsterdam_tz(self):
        idx = pd.date_range("2024-01-01", periods=24, freq="h", tz="Europe/Amsterdam")
        df = pd.DataFrame({"v": 1.0}, index=idx)
        result = TimezoneCheck()(df, "ctx")
        assert result.passed


# ── DuplicatesCheck ─────────────────────────────────────────────────────────


class TestDuplicatesCheck:
    def test_passes_no_duplicates(self):
        df = _price_df()
        result = DuplicatesCheck()(df, "test")
        assert result.passed

    def test_fails_on_duplicate_index(self):
        df = _price_df(48)
        # Manually create a duplicate timestamp
        dup = df.iloc[[0]]
        df_with_dup = pd.concat([df, dup])
        result = DuplicatesCheck()(df_with_dup, "ctx")
        assert not result.passed

    def test_fails_on_duplicate_date_column(self):
        n = 24
        dates = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC").tolist()
        dates.append(dates[0])  # introduce duplicate
        df = pd.DataFrame({"date": dates, "val": 1.0})
        result = DuplicatesCheck()(df, "ctx")
        assert not result.passed


# ── StalenessCheck ──────────────────────────────────────────────────────────


class TestStalenessCheck:
    def test_passes_recent_data(self):
        # Data ending today should pass
        idx = pd.date_range(
            pd.Timestamp.now("UTC").floor("h") - pd.Timedelta(hours=23),
            periods=24,
            freq="h",
            tz="UTC",
        )
        df = pd.DataFrame({"v": 1.0}, index=idx)
        result = StalenessCheck(max_days=3)(df, "ctx")
        assert result.passed

    def test_warns_on_stale_data(self):
        # Data from 30 days ago → should warn
        idx = pd.date_range("2020-01-01", periods=24, freq="h", tz="UTC")
        df = pd.DataFrame({"v": 1.0}, index=idx)
        result = StalenessCheck(max_days=3)(df, "ctx")
        # Staleness check only warns, never fails
        assert result.passed
        assert len(result.warnings) > 0

    def test_passes_empty_df(self):
        df = pd.DataFrame({"v": []}, index=pd.DatetimeIndex([]))
        result = StalenessCheck()(df, "ctx")
        assert result.passed


# ── RowCountCheck ───────────────────────────────────────────────────────────


class TestRowCountCheck:
    def test_passes_enough_rows(self):
        df = _price_df(100)
        result = RowCountCheck(min_rows=24)(df, "ctx")
        assert result.passed

    def test_fails_empty(self):
        df = pd.DataFrame({"Price_EUR_MWh": []})
        result = RowCountCheck(min_rows=1)(df, "ctx")
        assert not result.passed

    def test_fails_below_threshold(self):
        df = _price_df(5)
        result = RowCountCheck(min_rows=24)(df, "ctx")
        assert not result.passed
        assert "5" in result.errors[0] or "minimum" in result.errors[0]

    def test_passes_at_exact_threshold(self):
        df = _price_df(24)
        result = RowCountCheck(min_rows=24)(df, "ctx")
        assert result.passed


# ── CompositeValidator ──────────────────────────────────────────────────────


class TestCompositeValidator:
    def test_passes_all_checks(self):
        df = _price_df()
        validator = CompositeValidator(
            checks=[RowCountCheck(1), TimezoneCheck(), DuplicatesCheck()]
        )
        result = validator.validate(df, "test")
        assert result.passed
        assert len(result.errors) == 0

    def test_aggregates_errors_from_multiple_checks(self):
        # Empty tz-naive DataFrame → fails RowCountCheck + TimezoneCheck
        idx = pd.date_range("2024-01-01", periods=0, freq="h")  # no tz, empty
        df = pd.DataFrame({"Price_EUR_MWh": []}, index=idx)
        validator = CompositeValidator(checks=[RowCountCheck(1), TimezoneCheck()])
        result = validator.validate(df, "ctx")
        assert not result.passed
        assert len(result.errors) >= 1  # at least RowCountCheck fails

    def test_aggregates_warnings(self):
        df = _price_df(100)
        df.iloc[:50] = float("nan")
        validator = CompositeValidator(checks=[NaNCheck(0.10)])
        result = validator.validate(df, "ctx")
        assert result.passed  # NaN only warns
        assert len(result.warnings) > 0

    def test_empty_checks_passes(self):
        df = _price_df()
        result = CompositeValidator(checks=[]).validate(df, "ctx")
        assert result.passed

    def test_passed_is_false_if_any_error(self):
        df = pd.DataFrame({"v": []})  # empty → RowCountCheck fails
        validator = CompositeValidator(checks=[RowCountCheck(10)])
        result = validator.validate(df)
        assert not result.passed


# ── Additional coverage for edge-case branches ──────────────────────────────


class TestContinuityCheckNonIndexed:
    def test_extract_returns_none_for_plain_rangeindex_no_date_col(self):
        """_extract_datetime_index returns None when df has no DatetimeIndex
        and no 'date' column — covering checks.py line 41 (the return None path)."""
        df = pd.DataFrame({"value": [1, 2, 3]})  # RangeIndex, no 'date' column
        result = ContinuityCheck()(df, "ctx")
        # idx is None → check passes (skipped)
        assert result.passed


class TestStalenessCheckNaiveIndex:
    def test_naive_datetime_index_triggers_tz_localize(self):
        """StalenessCheck with tz-naive DatetimeIndex hits line 143 (tz_localize)."""
        idx = pd.date_range("2024-01-01", periods=24, freq="h")  # no tz → tz-naive
        df = pd.DataFrame({"Price_EUR_MWh": [50.0] * 24}, index=idx)
        result = StalenessCheck(max_days=365)(df, "ctx")
        assert result.passed  # data is recent-ish, should not warn


class TestValidationResultBool:
    def test_truthy_when_passed(self):
        """ValidationResult.__bool__ (validator.py line 23)."""
        from runeflow.ports.validator import ValidationResult

        assert bool(ValidationResult(passed=True)) is True

    def test_falsy_when_not_passed(self):
        from runeflow.ports.validator import ValidationResult

        assert bool(ValidationResult(passed=False)) is False


class TestValidatorFactories:
    def test_default_validator_returns_composite(self):
        """default_validator() factory covers composite.py."""
        from runeflow.validators.composite import default_validator

        v = default_validator()
        assert isinstance(v, CompositeValidator)

    def test_default_validator_runs_on_valid_df(self):
        from runeflow.validators.composite import default_validator

        v = default_validator()
        df = _price_df(200)
        result = v.validate(df, "ctx")
        assert result.passed

    def test_price_validator_returns_composite(self):
        """price_validator() factory covers composite.py line 75."""
        from runeflow.validators.composite import price_validator

        v = price_validator()
        assert isinstance(v, CompositeValidator)

    def test_price_validator_runs_on_valid_df(self):
        from runeflow.validators.composite import price_validator

        v = price_validator()
        df = _price_df(200)
        result = v.validate(df, "ctx")
        assert result.passed
