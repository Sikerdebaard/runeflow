# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2024-2026 Thomas Phil — runeflow
# See LICENSE and COMMERCIAL-LICENSE.md for licensing details.

"""Parquet-backed DataStore implementation."""

from __future__ import annotations

import datetime
import json
from collections.abc import Callable
from pathlib import Path

import pandas as pd
from loguru import logger

from runeflow.domain.forecast import ForecastPoint, ForecastResult
from runeflow.domain.generation import GenerationSeries
from runeflow.domain.price import PriceSeries
from runeflow.domain.weather import WeatherSeries
from runeflow.ports.store import DataStore


class ParquetStore(DataStore):
    """
    Persistent storage backed by Parquet files (time-series) and Pickle (model artifacts).

    All writes are atomic (write-to-tmp, then rename).
    Sidecar ``.meta.json`` files track provenance and TTL.
    """

    def __init__(self, cache_dir: Path) -> None:
        self._root = Path(cache_dir)
        self._root.mkdir(parents=True, exist_ok=True)

    # ── Price ─────────────────────────────────────────────────────────────────

    def save_prices(self, data: PriceSeries) -> None:
        path = self._prices_path(data.zone)
        df = data.to_dataframe().reset_index()
        # Append / merge with existing
        existing = self._read_parquet(path)
        if existing is not None and not existing.empty:
            combined = pd.concat([existing.reset_index(drop=True), df], ignore_index=True)
            combined.drop_duplicates(subset=["date"], keep="last", inplace=True)
            combined.sort_values("date", inplace=True)
            combined.reset_index(drop=True, inplace=True)
        else:
            combined = df
        self._write_parquet(path, combined, source=data.source, zone=data.zone)
        logger.debug(f"[ParquetStore] Saved {len(combined)} price records for {data.zone}.")

    def load_prices(
        self,
        zone: str,
        start: datetime.date | None = None,
        end: datetime.date | None = None,
    ) -> PriceSeries | None:
        path = self._prices_path(zone)
        df = self._read_parquet(path)
        if df is None or df.empty:
            return None
        if start is not None and end is not None:
            df = self._filter_by_date(df, start, end)
        if df.empty:
            return None
        return PriceSeries.from_dataframe(df, zone=zone, source="parquet")

    # ── Weather ───────────────────────────────────────────────────────────────

    def save_weather(self, data: WeatherSeries, zone: str) -> None:
        path = self._weather_hist_path(zone)
        df = data.to_dataframe().reset_index()
        existing = self._read_parquet(path)
        if existing is not None and not existing.empty:
            combined = pd.concat([existing.reset_index(drop=True), df], ignore_index=True)
            if "date" in combined.columns:
                combined.drop_duplicates(subset=["date"], keep="last", inplace=True)
                combined.sort_values("date", inplace=True)
            combined.reset_index(drop=True, inplace=True)
        else:
            combined = df
        schema = sorted(c for c in combined.columns if c != "date")
        self._write_parquet(path, combined, source=data.source, zone=zone, schema=schema)
        logger.debug(f"[ParquetStore] Saved weather for zone={zone} ({len(combined)} rows).")

    def load_weather(
        self,
        zone: str,
        start: datetime.date | None = None,
        end: datetime.date | None = None,
    ) -> WeatherSeries | None:
        path = self._weather_hist_path(zone)
        df = self._read_parquet(path)
        if df is None or df.empty:
            return None
        if start is not None and end is not None:
            df = self._filter_by_date(df, start, end)
        if df.empty:
            return None
        if "date" in df.columns:
            df = df.set_index("date")
        return WeatherSeries(
            locations=tuple(),
            df=df,
            source="parquet",
            fetched_at=pd.Timestamp.now("UTC"),
        )

    def save_forecast_weather(
        self, data: WeatherSeries, zone: str, member: int | None = None
    ) -> None:
        if member is None:
            path = self._weather_forecast_path(zone)
        else:
            path = self._weather_ensemble_path(zone, member)
        df = data.to_dataframe().reset_index()
        # Store the column schema so staleness checks can detect new features.
        schema = sorted(c for c in df.columns if c != "date")
        self._write_parquet(path, df, source=data.source, zone=zone, schema=schema)

    def load_forecast_weather(self, zone: str) -> WeatherSeries | None:
        path = self._weather_forecast_path(zone)
        return self._load_weather_series(path)

    def load_forecast_weather_ensemble(self, zone: str, member: int) -> WeatherSeries | None:
        path = self._weather_ensemble_path(zone, member)
        return self._load_weather_series(path)

    def is_forecast_weather_fresh(
        self,
        zone: str,
        ttl: datetime.timedelta,
        expected_cols: list[str],
        member: int | None = None,
    ) -> bool:
        """Return True if the cached forecast weather is within *ttl* and
        has at least all *expected_cols* (schema superset check)."""
        path = (
            self._weather_forecast_path(zone)
            if member is None
            else self._weather_ensemble_path(zone, member)
        )
        if self.is_stale(path, ttl):
            return False
        meta = self._read_meta(path)
        if meta is None:
            return False
        cached_schema: list[str] = meta.get("weather_schema", [])
        if not cached_schema:
            # Old file written without schema — treat as stale.
            return False
        required = set(expected_cols)
        return required.issubset(set(cached_schema))

    def is_historical_weather_fresh(
        self,
        zone: str,
        expected_cols: list[str],
    ) -> bool:
        """Eternal cache: fresh if file exists and schema is a superset of *expected_cols*."""
        path = self._weather_hist_path(zone)
        if not path.exists():
            return False
        meta = self._read_meta(path)
        if meta is None:
            return False
        cached_schema: list[str] = meta.get("weather_schema", [])
        if not cached_schema:
            return False
        return set(expected_cols).issubset(set(cached_schema))

    def _load_weather_series(self, path: Path) -> WeatherSeries | None:
        """Shared loader used by both deterministic and ensemble forecast paths."""
        df = self._read_parquet(path)
        if df is None or df.empty:
            return None
        if "date" in df.columns:
            df = df.set_index("date")
        return WeatherSeries(
            locations=tuple(),
            df=df,
            source="parquet",
            fetched_at=pd.Timestamp.now("UTC"),
        )

    # ── Generation ────────────────────────────────────────────────────────────

    def save_generation(self, data: GenerationSeries) -> None:
        path = self._generation_path(data.zone)
        df = data.df.reset_index()
        existing = self._read_parquet(path)
        if existing is not None and not existing.empty:
            combined = pd.concat([existing.reset_index(drop=True), df], ignore_index=True)
            if "date" in combined.columns:
                combined.drop_duplicates(subset=["date"], keep="last", inplace=True)
                combined.sort_values("date", inplace=True)
        else:
            combined = df
        combined.reset_index(drop=True, inplace=True)
        self._write_parquet(path, combined, source=data.source, zone=data.zone)

    def load_generation(
        self,
        zone: str,
        start: datetime.date | None = None,
        end: datetime.date | None = None,
    ) -> GenerationSeries | None:
        path = self._generation_path(zone)
        df = self._read_parquet(path)
        if df is None or df.empty:
            return None
        if start is not None and end is not None:
            df = self._filter_by_date(df, start, end)
        if df.empty:
            return None
        if "date" in df.columns:
            df = df.set_index("date")
        return GenerationSeries(
            zone=zone,
            df=df,
            source="parquet",
            fetched_at=pd.Timestamp.now("UTC"),
        )

    # ── Supplemental ──────────────────────────────────────────────────────────

    def save_supplemental(self, df: pd.DataFrame, zone: str, key: str) -> None:
        path = self._supplemental_path(zone, key)
        existing = self._read_parquet(path)
        save_df = df.reset_index()
        if existing is not None and not existing.empty:
            save_df = pd.concat([existing.reset_index(drop=True), save_df], ignore_index=True)
            idx_col = save_df.columns[0]
            save_df.drop_duplicates(subset=[idx_col], keep="last", inplace=True)
            save_df.sort_values(idx_col, inplace=True)
        save_df.reset_index(drop=True, inplace=True)
        self._write_parquet(path, save_df, source=key, zone=zone)

    def load_supplemental(self, zone: str, key: str) -> pd.DataFrame | None:
        path = self._supplemental_path(zone, key)
        df = self._read_parquet(path)
        if df is None or df.empty:
            return None
        return df

    # ── Model Artifacts ───────────────────────────────────────────────────────

    def save_model(self, model_bytes: bytes, zone: str, model_name: str) -> None:
        path = self._model_path(zone, model_name)
        path.parent.mkdir(parents=True, exist_ok=True)
        _atomic_write(path, lambda tmp: tmp.write_bytes(model_bytes))  # type: ignore[arg-type]
        logger.debug(f"[ParquetStore] Saved model {model_name} for {zone}.")

    def load_model(self, zone: str, model_name: str) -> bytes | None:
        path = self._model_path(zone, model_name)
        if not path.exists():
            return None
        return path.read_bytes()

    # ── Forecast Results ──────────────────────────────────────────────────────

    def save_forecast(self, result: ForecastResult) -> None:
        path = self._forecast_latest_path(result.zone)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._write_forecast_json(result, path)
        self.save_forecast_archive(result)

    def load_latest_forecast(self, zone: str) -> ForecastResult | None:
        path = self._forecast_latest_path(zone)
        return self._load_forecast_json(path)

    def save_forecast_archive(self, result: ForecastResult) -> None:
        """Save a timestamped forecast archive alongside latest.json."""
        ts = result.created_at.strftime("%Y%m%d_%H%M")
        archive_dir = self._root / "forecasts" / result.zone / "archive"
        archive_dir.mkdir(parents=True, exist_ok=True)
        archive_path = archive_dir / f"{ts}.json"
        self._write_forecast_json(result, archive_path)
        self._cleanup_archive(result.zone, max_days=30)

    def load_forecast_archive(self, zone: str, days_back: int = 30) -> list[ForecastResult]:
        """Load archived forecasts from the last *days_back* days."""
        archive_dir = self._root / "forecasts" / zone / "archive"
        if not archive_dir.exists():
            return []
        cutoff = pd.Timestamp.now("UTC") - pd.Timedelta(days=days_back)
        results = []
        for path in sorted(archive_dir.glob("*.json")):
            try:
                ts = pd.Timestamp(path.stem.replace("_", "T") + "00", tz="UTC")
                if ts < cutoff:
                    continue
            except Exception:
                continue
            forecast = self._load_forecast_json(path)
            if forecast is not None:
                results.append(forecast)
        return results

    def _cleanup_archive(self, zone: str, max_days: int = 30) -> None:
        """Delete archived forecasts older than *max_days*."""
        archive_dir = self._root / "forecasts" / zone / "archive"
        if not archive_dir.exists():
            return
        cutoff = pd.Timestamp.now("UTC") - pd.Timedelta(days=max_days)
        for path in archive_dir.glob("*.json"):
            try:
                ts = pd.Timestamp(path.stem.replace("_", "T") + "00", tz="UTC")
                if ts < cutoff:
                    path.unlink()
            except Exception:
                pass

    def _write_forecast_json(self, result: ForecastResult, path: Path) -> None:
        """Serialise *result* to *path* (atomic)."""
        data: dict = {
            "zone": result.zone,
            "created_at": result.created_at.isoformat(),
            "model_version": result.model_version,
            "points": [
                {
                    "timestamp": p.timestamp.isoformat(),
                    "prediction": p.prediction,
                    "lower": p.lower,
                    "upper": p.upper,
                    "uncertainty": p.uncertainty,
                    "model_agreement": p.model_agreement,
                    "lower_static": p.lower_static,
                    "upper_static": p.upper_static,
                    "ensemble_p50": p.ensemble_p50,
                    "ensemble_p25": p.ensemble_p25,
                    "ensemble_p75": p.ensemble_p75,
                }
                for p in result.points
            ],
            "model_predictions": {
                k: {str(ts): float(v) for ts, v in series.items()}
                for k, series in result.model_predictions.items()
            },
        }
        if result.ensemble_members is not None and not result.ensemble_members.empty:
            ens = result.ensemble_members
            data["ensemble_members"] = {
                str(col): {
                    ts.isoformat(): float(v)  # type: ignore[attr-defined]
                    for ts, v in ens[col].items()
                }
                for col in ens.columns
            }
        _atomic_write(path, lambda tmp: tmp.write_text(json.dumps(data), encoding="utf-8"))  # type: ignore[arg-type]

    def _load_forecast_json(self, path: Path) -> ForecastResult | None:
        """Deserialise a forecast JSON file; returns None if missing or corrupt."""
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return None
        points = tuple(
            ForecastPoint(
                timestamp=pd.Timestamp(p["timestamp"]),
                prediction=p["prediction"],
                lower=p["lower"],
                upper=p["upper"],
                uncertainty=p["uncertainty"],
                model_agreement=p["model_agreement"],
                lower_static=p.get("lower_static", p["lower"]),
                upper_static=p.get("upper_static", p["upper"]),
                ensemble_p50=p.get("ensemble_p50", p["prediction"]),
                ensemble_p25=p.get("ensemble_p25", p["lower"]),
                ensemble_p75=p.get("ensemble_p75", p["upper"]),
            )
            for p in data.get("points", [])
        )
        raw_model_preds = data.get("model_predictions", {})
        model_predictions = {
            k: pd.Series({pd.Timestamp(ts): float(v) for ts, v in series.items()})
            for k, series in raw_model_preds.items()
        }
        ens_df = pd.DataFrame()
        raw_ens = data.get("ensemble_members")
        if raw_ens:
            ens_df = pd.DataFrame(
                {
                    col: {pd.Timestamp(ts): float(v) for ts, v in series.items()}
                    for col, series in raw_ens.items()
                }
            )
        return ForecastResult(
            zone=data["zone"],
            points=points,
            ensemble_members=ens_df,
            model_predictions=model_predictions,
            created_at=pd.Timestamp(data["created_at"]),
            model_version=data["model_version"],
        )

    # ── Warmup Cache ──────────────────────────────────────────────────────────

    def save_warmup_cache(self, df: pd.DataFrame, zone: str) -> None:
        path = self._warmup_path(zone)
        path.parent.mkdir(parents=True, exist_ok=True)
        save_df = df.reset_index() if df.index.name else df
        self._write_parquet(path, save_df, source="warmup", zone=zone)
        logger.debug(f"[ParquetStore] Warmup cache saved for {zone} ({len(df)} rows).")

    def load_warmup_cache(self, zone: str) -> pd.DataFrame | None:
        path = self._warmup_path(zone)
        df = self._read_parquet(path)
        if df is None or df.empty:
            return None
        if "date" in df.columns:
            df = df.set_index("date")
        return df

    # ── Staleness check ───────────────────────────────────────────────────────

    def is_stale(self, path: Path, ttl: datetime.timedelta) -> bool:
        if not path.exists():
            return True
        meta = self._read_meta(path)
        if meta is None:
            return True
        try:
            fetched_at = pd.Timestamp(meta["fetched_at"])
            return pd.Timestamp.now("UTC") - fetched_at > ttl
        except Exception:
            return True

    # ── Path helpers ──────────────────────────────────────────────────────────

    def _prices_path(self, zone: str) -> Path:
        return self._root / "prices" / zone / "historical.parquet"

    def _weather_hist_path(self, zone: str) -> Path:
        return self._root / "weather" / zone / "historical.parquet"

    def _weather_forecast_path(self, zone: str) -> Path:
        return self._root / "weather" / zone / "forecast" / "deterministic.parquet"

    def _weather_ensemble_path(self, zone: str, member: int) -> Path:
        return (
            self._root / "weather" / zone / "forecast" / "ensemble" / f"member_{member:02d}.parquet"
        )

    def _generation_path(self, zone: str) -> Path:
        return self._root / "generation" / zone / "historical.parquet"

    def _supplemental_path(self, zone: str, key: str) -> Path:
        return self._root / "supplemental" / zone / f"{key}.parquet"

    def _model_path(self, zone: str, model_name: str) -> Path:
        return self._root / "models" / zone / f"{model_name}.pkl"

    def _forecast_latest_path(self, zone: str) -> Path:
        return self._root / "forecasts" / zone / "latest.json"

    def _warmup_path(self, zone: str) -> Path:
        return self._root / "warmup" / zone / "cache.parquet"

    # ── I/O helpers ───────────────────────────────────────────────────────────

    def _read_parquet(self, path: Path) -> pd.DataFrame | None:
        if not path.exists():
            return None
        try:
            return pd.read_parquet(path)
        except Exception as exc:
            logger.warning(f"[ParquetStore] Could not read {path}: {exc}")
            return None

    def _write_parquet(
        self,
        path: Path,
        df: pd.DataFrame,
        source: str = "",
        zone: str = "",
        schema: list[str] | None = None,
    ) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        meta: dict = {
            "zone": zone,
            "source": source,
            "fetched_at": pd.Timestamp.now("UTC").isoformat(),
            "row_count": len(df),
        }
        if schema is not None:
            meta["weather_schema"] = schema

        def _write(tmp: Path) -> None:
            df.to_parquet(tmp, index=False, compression="snappy")

        _atomic_write(path, _write)
        _atomic_write(
            path.with_suffix(".meta.json"),
            lambda p: p.write_text(json.dumps(meta), encoding="utf-8"),  # type: ignore[arg-type]
        )

    def _read_meta(self, path: Path) -> dict | None:
        meta_path = path.with_suffix(".meta.json")
        if not meta_path.exists():
            return None
        try:
            return json.loads(meta_path.read_text(encoding="utf-8"))  # type: ignore[no-any-return]
        except Exception:
            return None

    @staticmethod
    def _filter_by_date(df: pd.DataFrame, start: datetime.date, end: datetime.date) -> pd.DataFrame:
        """Filter a DataFrame that has a 'date' column or DatetimeIndex."""
        if df.empty:
            return df
        if "date" in df.columns:
            dates = pd.to_datetime(df["date"], utc=True)
            mask = (dates.dt.date >= start) & (dates.dt.date <= end)
            return df[mask].copy()
        if isinstance(df.index, pd.DatetimeIndex):
            idx_dates = df.index.tz_convert("UTC").date
            mask = (idx_dates >= start) & (idx_dates <= end)  # type: ignore[assignment]
            return df[mask].copy()
        return df


def _atomic_write(path: Path, write_fn: Callable[[Path], None]) -> None:
    """Write to a temp file then atomically rename (POSIX-safe)."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    try:
        write_fn(tmp)
        tmp.rename(path)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise
