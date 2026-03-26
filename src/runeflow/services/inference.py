# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2024-2026 Thomas Phil — runeflow
# See LICENSE and COMMERCIAL-LICENSE.md for licensing details.

"""InferenceService — autoregressive 9-day price forecast with 51-member
weather ensemble for uncertainty quantification.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import pickle
from datetime import UTC
from datetime import datetime as dt_cls

import inject
import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer

from runeflow.domain.forecast import ForecastPoint, ForecastResult
from runeflow.ensemble.condition_gated import ConditionGatedStrategy
from runeflow.features import INFERENCE_WARMUP_DAYS, build_pipeline
from runeflow.models.extreme_high import ExtremeHighModel
from runeflow.models.extreme_low import ExtremeLowModel
from runeflow.models.xgboost_quantile import XGBoostQuantileModel
from runeflow.ports.store import DataStore
from runeflow.ports.supplemental import SupplementalDataPort
from runeflow.ports.weather import WeatherPort
from runeflow.zones.config import ZoneConfig

logger = logging.getLogger(__name__)

TARGET_COLUMN = "Price_EUR_MWh"
# Uncertainty grows with forecast distance (1 % per hour, max 3×)
BASE_UNCERTAINTY_GROWTH = 0.01
MAX_UNCERTAINTY_FACTOR = 3.0
N_ENSEMBLE_MEMBERS = 51


# ---------------------------------------------------------------------------
# Module-level worker — must be at module scope to be picklable by
# joblib/loky worker processes.
# ---------------------------------------------------------------------------
def _run_forecast_worker(
    weather_df: pd.DataFrame,
    warmup: pd.DataFrame,
    timestamps: pd.DatetimeIndex,
    xgb: XGBoostQuantileModel,
    ext_high: ExtremeHighModel,
    ext_low: ExtremeLowModel,
    imputer: SimpleImputer,
    features: list[str],
    zone_cfg: ZoneConfig,
    label: str = "det",
) -> dict[pd.Timestamp, dict]:
    """
    Autoregressive inference loop for a single weather scenario.

    Runs inside a loky worker process — each worker is pinned to one thread
    so that joblib's outer parallelism is the only parallelism in play.
    """
    # ── Pin to a single thread ──────────────────────────────────────────────
    # Environment variables must be set before any library uses them;
    # loky workers start fresh so these take effect immediately.
    import os as _os

    for _var in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "BLIS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        _os.environ[_var] = "1"

    # Also tell XGBoost directly via set_params so it respects the limit
    # even if its thread pool was already initialised.
    for _m in (
        getattr(xgb, "_model_lower", None),
        getattr(xgb, "_model_p50", None),
        getattr(xgb, "_model_upper", None),
        getattr(ext_high, "_model", None),
        getattr(ext_low, "_model", None),
    ):
        if _m is not None:
            with contextlib.suppress(Exception):
                _m.set_params(nthread=1)
    # ───────────────────────────────────────────────────────────────────────

    pipeline = build_pipeline(zone_cfg)
    ensemble_strategy = ConditionGatedStrategy()

    warmup_window = warmup.copy()
    weather_df_tz = weather_df.copy()
    if weather_df_tz.index.tz is None:  # type: ignore[attr-defined]
        weather_df_tz.index = weather_df_tz.index.tz_localize("UTC")  # type: ignore[attr-defined]
    else:
        weather_df_tz.index = weather_df_tz.index.tz_convert("UTC")  # type: ignore[attr-defined]

    results: dict[pd.Timestamp, dict] = {}
    n_total = len(timestamps)

    logger.info("[%s] starting (%d timestamps)", label, n_total)
    for h, ts in enumerate(timestamps):
        already_known = ts in warmup_window.index

        if not already_known:
            if ts not in weather_df_tz.index:
                continue
            new_row = weather_df_tz.loc[[ts]]
            warmup_window = pd.concat([warmup_window, new_row])
            warmup_window = warmup_window.tail(INFERENCE_WARMUP_DAYS * 24).sort_index()

        df_features = pipeline.transform(warmup_window, zone_cfg)
        if df_features.empty:
            continue

        # Select current row
        df_features = df_features.sort_index()
        if ts in df_features.index:  # noqa: SIM108
            X_row = df_features.loc[[ts]]
        else:
            X_row = df_features.tail(1)

        X_row = X_row.drop(columns=[TARGET_COLUMN], errors="ignore")

        # Align to training features
        if features:
            missing = {f: np.nan for f in features if f not in X_row.columns}
            if missing:
                X_row = pd.concat([X_row, pd.DataFrame(missing, index=X_row.index)], axis=1)
            X_row = X_row[features]

        # Impute
        if X_row.isnull().any().any():
            X_row = pd.DataFrame(
                imputer.transform(X_row),
                columns=X_row.columns,
                index=X_row.index,
            )

        # Uncertainty grows with horizon
        uf = min(1.0 + h * BASE_UNCERTAINTY_GROWTH, MAX_UNCERTAINTY_FACTOR)

        # Predict from each model
        preds_map: dict[str, pd.DataFrame] = {}
        preds_map["xgboost_quantile"] = xgb.predict(X_row)
        if ext_high.is_trained:
            preds_map["extreme_high"] = ext_high.predict(X_row)
        if ext_low.is_trained:
            preds_map["extreme_low"] = ext_low.predict(X_row)

        combined = ensemble_strategy.combine(preds_map, X_row, weather_uncertainty_factor=uf)

        point = combined.iloc[0]

        # Store all individual model predictions for the plot panel
        _xgb_df = preds_map["xgboost_quantile"]
        _xgb_p50 = float(_xgb_df["prediction"].iloc[0])
        _xgb_p10 = float(_xgb_df["lower"].iloc[0])
        _xgb_p90 = float(_xgb_df["upper"].iloc[0])

        _ext_high = (
            float(preds_map["extreme_high"]["prediction"].iloc[0])
            if "extreme_high" in preds_map
            else None
        )
        _ext_low = (
            float(preds_map["extreme_low"]["prediction"].iloc[0])
            if "extreme_low" in preds_map
            else None
        )

        result = {
            "prediction": float(point["prediction"]),
            "lower": float(point["lower"]),
            "upper": float(point["upper"]),
            "uncertainty": float(point["uncertainty"]),
            "model_agreement": float(point["model_agreement"]),
            "xgboost_p50": _xgb_p50,
            "xgboost_p10": _xgb_p10,
            "xgboost_p90": _xgb_p90,
            "extreme_high": _ext_high,
            "extreme_low": _ext_low,
        }
        results[ts] = result

        # Log progress at every 24-hour boundary (one line per forecast day)
        if (h + 1) % 24 == 0 or (h + 1) == n_total:
            logger.info(
                "[%s] %d/%d hours complete (day %d, latest p50=%.1f)",
                label,
                h + 1,
                n_total,
                (h + 1) // 24 or 1,
                result["prediction"],
            )

        # Autoregressive feed-forward (only for future hours)
        if not already_known:
            warmup_window.at[ts, TARGET_COLUMN] = result["prediction"]

    logger.info("[%s] finished — %d results", label, len(results))
    return results


class InferenceService:
    """
    Runs the full inference pipeline:

    1. Load trained models + feature names from store.
    2. Download deterministic + ensemble weather forecasts.
    3. Execute autoregressive loop over forecast horizon for each weather
       scenario.
    4. Aggregate scenario predictions → ForecastResult.
    """

    @inject.autoparams()
    def __init__(
        self,
        zone_cfg: ZoneConfig = inject.attr("zone_config"),  # type: ignore[assignment]  # noqa: B008
        store: DataStore = inject.attr(DataStore),  # type: ignore[assignment]  # noqa: B008
        weather_port: WeatherPort = inject.attr(WeatherPort),  # type: ignore[assignment]  # noqa: B008, E501
    ) -> None:
        self._zone_cfg = zone_cfg
        self._store = store
        self._weather_port = weather_port
        # Optional supplemental adapter (e.g. NED for NL)
        try:
            self._supplemental_port: SupplementalDataPort | None = inject.instance(
                SupplementalDataPort
            )  # type: ignore[assignment]
        except Exception:
            self._supplemental_port = None

    # ------------------------------------------------------------------
    def run(self) -> ForecastResult:
        zone = self._zone_cfg.zone
        logger.info("InferenceService starting for zone=%s", zone)

        logger.info("Loading models for zone=%s...", zone)
        xgb, ext_high, ext_low, imputer, features, model_version = self._load_models(zone)
        logger.info(
            "Models loaded: xgb=ok, ext_high=%s, ext_low=%s, features=%d",
            "ok" if ext_high.is_trained else "untrained",
            "ok" if ext_low.is_trained else "untrained",
            len(features),
        )

        logger.info("Loading warmup cache for zone=%s...", zone)
        warmup = self._store.load_warmup_cache(zone)
        if warmup is None or warmup.empty:
            raise RuntimeError(f"No warmup cache for zone={zone}. Run 'warmup-cache' first.")
        logger.info("Warmup cache loaded: %d rows", len(warmup))

        logger.info("Fetching forecast weather for zone=%s...", zone)
        locations = list(self._zone_cfg.weather_locations)
        det_weather = self._weather_port.download_forecast(locations, horizon_days=9)
        logger.info(
            "Deterministic weather: %d rows, columns=%s",
            len(det_weather.df),
            list(det_weather.df.columns)[:4],
        )

        # Join supplemental forecast (NED consumption forecast for NL) so
        # that residual-load features are computed from the forecast rather
        # than being imputed with training-set means.
        ned_forecast = self._load_supplemental_forecast(zone)
        if ned_forecast is not None:
            det_weather_df = det_weather.df.copy()
            det_weather_df = det_weather_df.join(ned_forecast, how="left")
            from runeflow.domain.weather import WeatherSeries

            det_weather = WeatherSeries(
                df=det_weather_df,
                source=det_weather.source,
                locations=det_weather.locations,
                fetched_at=det_weather.fetched_at,
            )
            logger.info(
                "Joined supplemental forecast (%d rows) into deterministic weather",
                len(ned_forecast),
            )
        try:
            ens_members = self._weather_port.download_ensemble_forecast(locations, horizon_days=9)
            logger.info("Ensemble weather: %d members", len(ens_members))
        except Exception as exc:
            logger.warning("Ensemble weather forecast unavailable: %s", exc)
            ens_members = []

        # Build forecast timestamps
        now_utc = pd.Timestamp.now("UTC").floor("h")
        end_utc = now_utc + pd.Timedelta(days=9)
        timestamps = pd.date_range(now_utc, end_utc, freq="h", tz="UTC")

        # Deterministic run (sequential — needed before ensemble aggregation)
        logger.info(
            "Running deterministic forecast (%d hours, %s → %s)",
            len(timestamps),
            timestamps[0].strftime("%Y-%m-%d %H:%M"),
            timestamps[-1].strftime("%Y-%m-%d %H:%M"),
        )
        det_results = _run_forecast_worker(
            det_weather.df,
            warmup,
            timestamps,
            xgb,
            ext_high,
            ext_low,
            imputer,
            features,
            self._zone_cfg,
            label="deterministic",
        )
        logger.info("Deterministic forecast done (%d points)", len(det_results))

        # Ensemble runs — one joblib worker per member (loky backend)
        members_to_run = ens_members[:N_ENSEMBLE_MEMBERS]
        member_results: list[dict[pd.Timestamp, dict]] = []

        if members_to_run:
            n_total = len(members_to_run)
            n_workers = min(os.cpu_count() or 1, n_total)
            logger.info(
                "Running %d ensemble members in parallel (workers=%d, backend=loky)",
                n_total,
                n_workers,
            )
            raw_results = joblib.Parallel(n_jobs=n_workers, backend="loky", verbose=5)(
                joblib.delayed(_run_forecast_worker)(
                    member.df,
                    warmup,
                    timestamps,
                    xgb,
                    ext_high,
                    ext_low,
                    imputer,
                    features,
                    self._zone_cfg,
                    f"member_{i:02d}",
                )
                for i, member in enumerate(members_to_run)
            )
            member_results = [r for r in raw_results if r is not None]
            logger.info(
                "Ensemble runs complete: %d/%d succeeded",
                len(member_results),
                n_total,
            )

        return self._build_result(zone, timestamps, det_results, member_results, model_version)

    # ------------------------------------------------------------------
    def _load_supplemental_forecast(self, zone: str) -> pd.DataFrame | None:
        """Load NED-style forecast, renaming to match training column."""
        # Try the live adapter first (downloads fresh data).
        if self._supplemental_port is not None and self._supplemental_port.supports_zone(zone):
            try:
                df = self._supplemental_port.download_forecast(zone)
                if df is not None and not df.empty:
                    # Rename ned_forecast_kwh → ned_utilization_kwh so the
                    # feature pipeline recognises it.
                    if "ned_forecast_kwh" in df.columns:
                        df = df.rename(columns={"ned_forecast_kwh": "ned_utilization_kwh"})
                    # Ensure UTC tz-aware index
                    if df.index.tz is None:  # type: ignore[attr-defined]
                        df.index = df.index.tz_localize("UTC")  # type: ignore[attr-defined]
                    else:
                        df.index = df.index.tz_convert("UTC")  # type: ignore[attr-defined]
                    logger.info("Supplemental forecast downloaded: %d rows", len(df))
                    return df
            except Exception as exc:
                logger.warning("Supplemental forecast download failed: %s", exc)

        # Fall back to the cached copy saved by update-data.
        try:
            cached = self._store.load_supplemental(zone, "forecast")
            if cached is not None and not cached.empty:
                if "ned_forecast_kwh" in cached.columns:
                    cached = cached.rename(columns={"ned_forecast_kwh": "ned_utilization_kwh"})
                if not isinstance(cached.index, pd.DatetimeIndex):
                    logger.warning("Supplemental cache has non-datetime index, skipping")
                    return None
                if cached.index.tz is None:
                    cached.index = cached.index.tz_localize("UTC")
                else:
                    cached.index = cached.index.tz_convert("UTC")
                logger.info("Supplemental forecast loaded from cache: %d rows", len(cached))
                return cached
        except Exception as exc:
            logger.warning("Supplemental cache load failed: %s", exc)

        return None

    # ------------------------------------------------------------------
    def _load_models(
        self, zone: str
    ) -> tuple[
        XGBoostQuantileModel, ExtremeHighModel, ExtremeLowModel, SimpleImputer, list[str], str
    ]:
        logger.info("  loading XGBoostQuantileModel...")
        xgb = XGBoostQuantileModel()
        if not xgb.load(self._store, zone):
            raise RuntimeError(f"XGBoost model not found for zone={zone}. Run 'train' first.")
        logger.info("  XGBoostQuantileModel ok")

        logger.info("  loading ExtremeHighModel...")
        ext_high = ExtremeHighModel()
        ext_high.load(self._store, zone)
        logger.info("  ExtremeHighModel ok (trained=%s)", ext_high.is_trained)

        logger.info("  loading ExtremeLowModel...")
        ext_low = ExtremeLowModel()
        ext_low.load(self._store, zone)
        logger.info("  ExtremeLowModel ok (trained=%s)", ext_low.is_trained)

        logger.info("  loading imputer + feature list...")
        raw_imputer = self._store.load_model(zone, "imputer")
        imputer: SimpleImputer = (
            pickle.loads(raw_imputer) if raw_imputer else SimpleImputer(strategy="mean")
        )

        raw_features = self._store.load_model(zone, "features")
        features: list[str] = json.loads(raw_features.decode()) if raw_features else []
        logger.info("  imputer ok, %d features", len(features))

        raw_version = self._store.load_model(zone, "model_version")
        model_version: str = raw_version.decode() if raw_version else "unknown"

        return xgb, ext_high, ext_low, imputer, features, model_version

    # ------------------------------------------------------------------
    def _build_result(
        self,
        zone: str,
        timestamps: pd.DatetimeIndex,
        det_results: dict[pd.Timestamp, dict],
        member_results: list[dict[pd.Timestamp, dict]],
        model_version: str = "unknown",
    ) -> ForecastResult:
        points: list[ForecastPoint] = []

        # Build ensemble DataFrame (one column per member)
        member_cols: dict[str, list[float]] = {}
        for i, _m in enumerate(member_results):
            col_key = f"member_{i:02d}"
            member_cols[col_key] = []

        ts_list = sorted(det_results.keys())

        for ts in ts_list:
            det = det_results[ts]

            # Collect member predictions for this timestamp
            member_preds = [m[ts]["prediction"] for m in member_results if ts in m]

            if member_preds:
                arr = np.array(member_preds)
                ens_half = float(np.percentile(arr, 97.5) - np.percentile(arr, 2.5)) / 2
                # Centre the ensemble spread on the deterministic prediction
                # so the forecast line sits at the midpoint of the combined band.
                ens_lower = float(det["prediction"] - ens_half)
                ens_upper = float(det["prediction"] + ens_half)
                ens_p25 = float(np.percentile(arr, 25))
                ens_p50 = float(np.percentile(arr, 50))
                ens_p75 = float(np.percentile(arr, 75))
                ens_std = float(np.std(arr))
                ens_agree = float(np.clip(1 - ens_std / (abs(det["prediction"]) + 1), 0, 1))
            else:
                ens_lower = det["lower"]
                ens_upper = det["upper"]
                ens_p25 = det["lower"]
                ens_p50 = det["prediction"]
                ens_p75 = det["upper"]
                ens_agree = det["model_agreement"]

            points.append(
                ForecastPoint(
                    timestamp=ts,
                    prediction=det["prediction"],
                    lower=ens_lower,
                    upper=ens_upper,
                    uncertainty=ens_upper - ens_lower,
                    model_agreement=ens_agree,
                    lower_static=det["lower"],
                    upper_static=det["upper"],
                    ensemble_p50=ens_p50,
                    ensemble_p25=ens_p25,
                    ensemble_p75=ens_p75,
                )
            )

            for i, m in enumerate(member_results):
                col_key = f"member_{i:02d}"
                if (
                    col_key not in member_cols
                ):  # pragma: no cover — defensive; pre-initialised above
                    member_cols[col_key] = [None] * len(ts_list)  # type: ignore[list-item]
                idx = ts_list.index(ts)
                if len(member_cols[col_key]) <= idx:
                    member_cols[col_key].extend([None] * (idx - len(member_cols[col_key]) + 1))  # type: ignore[list-item]
                member_cols[col_key][idx] = m.get(ts, {}).get("prediction", np.nan)

        if ts_list and member_cols:
            ens_df = pd.DataFrame(member_cols, index=ts_list)
        else:
            ens_df = pd.DataFrame(index=ts_list)

        # Individual model series for the plot
        _model_keys = [
            "xgboost_p50",
            "xgboost_p10",
            "xgboost_p90",
            "extreme_high",
            "extreme_low",
        ]
        model_preds: dict[str, pd.Series] = {}
        for mkey in _model_keys:
            vals = [det_results[ts].get(mkey) for ts in ts_list]
            if any(v is not None for v in vals):
                model_preds[mkey] = pd.Series(
                    [v if v is not None else float("nan") for v in vals],
                    index=ts_list,
                )

        return ForecastResult(
            zone=zone,
            points=tuple(points),
            ensemble_members=ens_df,
            model_predictions=model_preds,
            created_at=dt_cls.now(UTC),  # type: ignore[arg-type]
            model_version=model_version,
        )
