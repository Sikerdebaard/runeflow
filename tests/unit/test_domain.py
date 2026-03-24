# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2024-2026 Thomas Phil — runeflow
# See LICENSE and COMMERCIAL-LICENSE.md for licensing details.

"""Tests for domain types: PriceRecord, PriceSeries."""
from __future__ import annotations

import dataclasses

import pandas as pd
import pytest

from runeflow.domain.price import PriceRecord, PriceSeries


class TestPriceRecord:
    def test_is_frozen(self):
        rec = PriceRecord(
            timestamp=pd.Timestamp("2024-01-01 12:00", tz="UTC"),
            price_eur_mwh=55.0,
        )
        with pytest.raises((dataclasses.FrozenInstanceError, AttributeError)):
            rec.price_eur_mwh = 99.0  # type: ignore[misc]

    def test_equality(self):
        ts = pd.Timestamp("2024-03-15 09:00", tz="UTC")
        a = PriceRecord(timestamp=ts, price_eur_mwh=42.0)
        b = PriceRecord(timestamp=ts, price_eur_mwh=42.0)
        assert a == b

    def test_inequality(self):
        ts = pd.Timestamp("2024-03-15 09:00", tz="UTC")
        a = PriceRecord(timestamp=ts, price_eur_mwh=42.0)
        b = PriceRecord(timestamp=ts, price_eur_mwh=43.0)
        assert a != b

    def test_hash_consistent(self):
        ts = pd.Timestamp("2024-01-01 00:00", tz="UTC")
        r = PriceRecord(timestamp=ts, price_eur_mwh=10.0)
        assert hash(r) == hash(r)


class TestPriceSeries:
    @pytest.fixture()
    def _series(self) -> PriceSeries:
        ts = pd.date_range("2024-01-01", periods=48, freq="h", tz="UTC")
        records = tuple(
            PriceRecord(timestamp=t, price_eur_mwh=float(i)) for i, t in enumerate(ts)
        )
        return PriceSeries(
            zone="NL",
            records=records,
            source="test",
            fetched_at=pd.Timestamp.now("UTC"),
        )

    def test_len(self, _series):
        assert len(_series) == 48

    def test_to_dataframe_shape(self, _series):
        df = _series.to_dataframe()
        assert len(df) == 48
        assert "Price_EUR_MWh" in df.columns

    def test_to_dataframe_index_is_utc(self, _series):
        df = _series.to_dataframe()
        assert isinstance(df.index, pd.DatetimeIndex)
        assert df.index.tz is not None

    def test_from_dataframe_roundtrip(self, _series):
        df = _series.to_dataframe()
        restored = PriceSeries.from_dataframe(df, zone="NL", source="test")
        assert len(restored) == len(_series)
        for orig, res in zip(_series.records, restored.records):
            assert orig.price_eur_mwh == pytest.approx(res.price_eur_mwh, abs=1e-6)

    def test_empty_series_to_dataframe(self):
        ps = PriceSeries(
            zone="NL",
            records=(),
            source="empty",
            fetched_at=pd.Timestamp.now("UTC"),
        )
        df = ps.to_dataframe()
        assert df.empty
        assert "Price_EUR_MWh" in df.columns

    def test_date_range(self, _series):
        dr = _series.date_range()
        assert dr is not None
        start, end = dr
        assert start < end

    def test_date_range_empty(self):
        ps = PriceSeries(
            zone="NL", records=(), source="x", fetched_at=pd.Timestamp.now("UTC")
        )
        assert ps.date_range() is None

    def test_from_dataframe_with_date_column(self):
        """from_dataframe should handle a 'date' column instead of DatetimeIndex."""
        df_reset = pd.DataFrame(
            {
                "date": pd.date_range("2024-06-01", periods=24, freq="h", tz="UTC"),
                "Price_EUR_MWh": range(24),
            }
        )
        ps = PriceSeries.from_dataframe(df_reset, zone="DE_LU", source="x")
        assert len(ps) == 24
        assert ps.zone == "DE_LU"


# ---------------------------------------------------------------------------
# GenerationSeries
# ---------------------------------------------------------------------------

class TestGenerationSeries:
    def test_to_dataframe_returns_copy(self):
        """GenerationSeries.to_dataframe() covers domain/generation.py line 23."""
        from runeflow.domain.generation import GenerationSeries

        idx = pd.date_range("2024-01-01", periods=24, freq="h", tz="UTC")
        df = pd.DataFrame({"load_forecast_mw": range(24)}, index=idx)
        gs = GenerationSeries(
            zone="NL", df=df, source="test", fetched_at=pd.Timestamp.now("UTC")
        )
        out = gs.to_dataframe()
        assert list(out.columns) == ["load_forecast_mw"]
        # It's a copy — mutations don't affect the original
        out["extra"] = 0
        assert "extra" not in gs.df.columns
