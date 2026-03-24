# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2024-2026 Thomas Phil — runeflow
# See LICENSE and COMMERCIAL-LICENSE.md for licensing details.

"""
PlotService — 3-panel matplotlib forecast chart.

Panel 1 – Price Forecast: Ensemble P50 with IQR fill (P25–P75),
           95 % outer envelope, actual prices, and peak / trough
           annotations.  Alternate-day shading + "now" marker.
Panel 2 – Model Predictions: individual model lines (XGB P50, P10–P90
           band, Extreme High / Low) overlaid with Ensemble P50 reference.
Panel 3 – Scorecard: training metrics, live forecast quality, composite grade.

Palette
-------
Dark Violet  #a718c7  — primary accent (ensemble P50, key lines)
Soft Blush   #fae3e3  — light fill (outer envelope)
Desert Sand  #f7d4bc  — warm fill (IQR band)
Pink Orchid  #cfa5b4  — secondary lines, day-shading tint, "now" marker
Evergreen    #042a2b  — actual price line
Black        #000000  — text, axis labels, titles, spines

Supplementary:
Warm Amber   #d4875c  — Extreme Low model line
Teal         #1a6b6d  — XGBoost lines and P10–P90 band
"""

from __future__ import annotations

import datetime
import logging
import pickle
from pathlib import Path

import inject
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import pandas as pd

from runeflow.domain.forecast import ForecastResult
from runeflow.domain.tariff import TariffFormula
from runeflow.ports.price import PricePort
from runeflow.ports.store import DataStore
from runeflow.zones.config import ZoneConfig

logger = logging.getLogger(__name__)

# ── Colour palette ────────────────────────────────────────────────────────────
_DARK_VIOLET = "#a718c7"
_SOFT_BLUSH = "#fae3e3"
_DESERT_SAND = "#f7d4bc"
_PINK_ORCHID = "#cfa5b4"
_EVERGREEN = "#042a2b"
_BLACK = "#000000"
_WARM_AMBER = "#d4875c"
_TEAL = "#1a6b6d"


def _style_axis(ax: plt.Axes, date_fmt: mdates.DateFormatter) -> None:
    """Apply shared visual polish to an axis."""
    ax.xaxis.set_major_formatter(date_fmt)
    ax.grid(True, which="major", color="#e0d8e4", linewidth=0.6, alpha=0.7)
    ax.tick_params(labelsize=9, colors=_BLACK)
    for spine in ax.spines.values():
        spine.set_color(_BLACK)
        spine.set_linewidth(0.6)


def _annotate_extremes(
    ax: plt.Axes,
    df: pd.DataFrame,
    col: str,
    colour: str,
) -> None:
    """Mark the forecast-window peak and trough with small labels."""
    series = df[col].dropna()
    if series.empty or len(series) < 2:
        return
    for idx, fmt, va, y_off in [
        (series.idxmax(), "▲ {:.3f}", "bottom", 8),
        (series.idxmin(), "▼ {:.3f}", "top", -8),
    ]:
        ax.annotate(
            fmt.format(series[idx]),
            xy=(idx, series[idx]),  # type: ignore[arg-type]
            xytext=(0, y_off),
            textcoords="offset points",
            fontsize=7.5,
            color=_BLACK,
            alpha=0.85,
            ha="center",
            va=va,
            fontweight="bold",
        )


class PlotService:
    """Renders a 3-panel price forecast chart to a PNG file."""

    @inject.autoparams()
    def __init__(
        self,
        zone_cfg: ZoneConfig = inject.attr("zone_config"),  # type: ignore[assignment]  # noqa: B008
        store: DataStore = inject.attr(DataStore),  # type: ignore[assignment]  # noqa: B008
        price_port: PricePort = inject.attr(PricePort),  # type: ignore[assignment]  # noqa: B008
    ) -> None:
        self._zone_cfg = zone_cfg
        self._store = store
        self._price_port = price_port

    # ------------------------------------------------------------------
    def run(
        self,
        output_path: Path | None = None,
        provider: str = "wholesale",
    ) -> Path:
        zone = self._zone_cfg.zone
        forecast = self._store.load_latest_forecast(zone)
        if forecast is None:
            raise RuntimeError(f"No forecast found for zone={zone}. Run 'inference' first.")

        # Resolve tariff formula
        tariff: TariffFormula | None = self._zone_cfg.tariff_formulas.get(provider)
        if tariff is None:
            available = list(self._zone_cfg.tariff_formulas.keys())
            raise ValueError(
                f"Provider '{provider}' not found for zone={zone}. Available: {available}"
            )
        tariff_label = tariff.label
        is_wholesale = provider == "wholesale"

        df = forecast.to_dataframe()
        if "timestamp" in df.columns:
            df = df.set_index("timestamp")

        # Filter from start of today
        try:
            tz = self._zone_cfg.timezone
            today = pd.Timestamp.now(tz=tz).normalize().tz_convert("UTC")
            if df.index.tz is None:  # type: ignore[attr-defined]
                today = today.tz_localize(None)
            df = df[df.index >= today]
        except Exception as exc:
            logger.warning("Could not filter df to today: %s", exc)

        model_preds = forecast.model_predictions

        # ── Apply tariff formula: EUR/MWh → EUR/kWh ──────────────────────
        def _apply_tariff(mwh_value: float, ts: pd.Timestamp) -> float:
            kwh_wholesale = mwh_value / 1000.0
            return tariff.apply(kwh_wholesale, ts.date())

        price_cols = [
            "ensemble_p50",
            "ensemble_p25",
            "ensemble_p75",
            "lower",
            "upper",
            "lower_static",
            "upper_static",
        ]
        for col in price_cols:
            if col in df.columns:
                df[col] = [_apply_tariff(v, ts) for ts, v in zip(df.index, df[col], strict=False)]

        # ── Fetch actual prices ───────────────────────────────────────────
        df_actual: pd.Series | None = None
        try:
            today = datetime.date.today()  # type: ignore[assignment]
            start = today - datetime.timedelta(days=7)
            price_series = self._price_port.download_historical(zone, start, today)
            df_actual_raw = price_series.to_dataframe()
            if df.index.tz is None:  # type: ignore[attr-defined]
                df_actual_raw.index = df_actual_raw.index.tz_localize(None)  # type: ignore[attr-defined]
            else:
                df_actual_raw.index = df_actual_raw.index.tz_convert(df.index.tz)  # type: ignore[attr-defined]
            df_actual = df_actual_raw["Price_EUR_MWh"]
            today_ts = pd.Timestamp(today)
            if df_actual.index.tz is not None:  # type: ignore[attr-defined]
                today_ts = today_ts.tz_localize(df_actual.index.tz)  # type: ignore[attr-defined]
            df_actual = df_actual[df_actual.index >= today_ts]
            df_actual = pd.Series(
                [
                    _apply_tariff(v, ts)
                    for ts, v in zip(df_actual.index, df_actual.values, strict=False)
                ],
                index=df_actual.index,
            )
            logger.info(
                "Actual prices fetched: %d hours (%s → %s)",
                len(df_actual),
                df_actual.index.min(),
                df_actual.index.max(),
            )
        except Exception as exc:
            logger.warning("Could not fetch actual prices: %s", exc)

        # ── Load training metrics from model pickles ─────────────────────
        train_metrics = self._load_training_metrics(zone)

        # ── Compute live forecast-quality metrics ─────────────────────────
        live_metrics = self._compute_live_metrics(df, df_actual, forecast)

        # ── Composite grade ───────────────────────────────────────────────
        score, max_score, grade_label = self._composite_grade(
            train_metrics,
            live_metrics,
        )

        # ── Figure layout ─────────────────────────────────────────────────
        has_models = bool(model_preds)
        has_ensemble = forecast.ensemble_members is not None and not forecast.ensemble_members.empty
        n_panels = (
            1 + int(has_ensemble) + int(has_models) + 1
        )  # forecast + ens? + models? + scorecard
        ratios: list[float] = [3]
        if has_ensemble:
            ratios.append(2)
        if has_models:
            ratios.append(2)
        ratios.append(1.5)
        fig_h = 8 + 4 * int(has_ensemble) + 4 * int(has_models)
        fig = plt.figure(
            figsize=(16, fig_h),
            layout="constrained",
        )
        gs = fig.add_gridspec(n_panels, 1, height_ratios=ratios, hspace=0.18)
        idx = 0
        ax1 = fig.add_subplot(gs[idx])
        idx += 1
        ax_ens = fig.add_subplot(gs[idx], sharex=ax1) if has_ensemble else None
        if has_ensemble:
            idx += 1
        ax2 = fig.add_subplot(gs[idx], sharex=ax1) if has_models else None
        if has_models:
            idx += 1
        ax_sc = fig.add_subplot(gs[idx])

        fig.patch.set_facecolor("#fdfbfe")
        date_fmt = mdates.DateFormatter("%a %d %b %H:%M")
        y_label = "Price (EUR/kWh wholesale)" if is_wholesale else "Price (EUR/kWh)"

        # Day shading & "now" marker
        try:
            tz = self._zone_cfg.timezone
            now_ts = pd.Timestamp.now(tz=tz)
            if df.index.tz is None:  # type: ignore[attr-defined]
                now_ts = now_ts.tz_localize(None)
        except Exception:
            now_ts = None

        for ax in (a for a in (ax1, ax_ens, ax2) if a is not None):
            ax.set_facecolor("#fdfbfe")
            day_start = df.index.min().normalize()
            day_end = df.index.max().normalize() + pd.Timedelta(days=1)
            for i, day in enumerate(pd.date_range(day_start, day_end, freq="D")[:-1]):
                if i % 2 == 1:
                    ax.axvspan(
                        day,  # type: ignore[arg-type]
                        day + pd.Timedelta(days=1),  # type: ignore[arg-type]
                        color="#f0ecf3",
                        alpha=0.35,
                        zorder=0,
                        lw=0,
                    )
            if now_ts is not None and day_start <= now_ts <= day_end:
                ax.axvline(
                    now_ts,  # type: ignore[arg-type]
                    color=_PINK_ORCHID,
                    lw=1.0,
                    ls="--",
                    alpha=0.55,
                    zorder=3,
                )

        # ── Panel 1: Price Forecast ───────────────────────────────────────
        ax1.fill_between(
            df.index,
            df["lower"],
            df["upper"],
            color=_SOFT_BLUSH,
            alpha=0.45,
            label="95 % envelope",
        )
        ax1.plot(df.index, df["lower"], color=_PINK_ORCHID, lw=0.6, ls="--", alpha=0.5)
        ax1.plot(df.index, df["upper"], color=_PINK_ORCHID, lw=0.6, ls="--", alpha=0.5)

        ax1.fill_between(
            df.index,
            df["ensemble_p25"],
            df["ensemble_p75"],
            color=_DESERT_SAND,
            alpha=0.6,
            label="IQR (P25–P75)",
        )

        ax1.plot(
            df.index,
            df["ensemble_p50"],
            color=_DARK_VIOLET,
            lw=2.2,
            label="Ensemble P50",
            solid_capstyle="round",
        )

        if df_actual is not None and len(df_actual) > 0:
            ax1.step(
                df_actual.index,
                df_actual.values,  # type: ignore[arg-type]
                where="post",
                color=_EVERGREEN,
                lw=1.8,
                label="Actual (ENTSO-E)",
                zorder=5,
            )

        _annotate_extremes(ax1, df, "ensemble_p50", _DARK_VIOLET)

        ax1.set_title(
            f"{tariff_label} — Zone {zone}",
            fontsize=14,
            fontweight="bold",
            color=_BLACK,
            pad=12,
        )
        ax1.set_ylabel(y_label, fontsize=10, color=_BLACK)
        ax1.legend(
            loc="upper right",
            fontsize=8.5,
            framealpha=0.85,
            edgecolor=_PINK_ORCHID,
            fancybox=True,
        )
        _style_axis(ax1, date_fmt)
        # Hide x-tick labels if there are panels below
        if ax_ens is not None or ax2 is not None:
            ax1.tick_params(labelbottom=False)

        # ── Panel: Ensemble Runs ──────────────────────────────────────────
        if ax_ens is not None and has_ensemble:
            ens = forecast.ensemble_members
            # Apply tariff to each member column
            for col in ens.columns:
                ens_aligned = ens[col].reindex(df.index)
                transformed = pd.Series(
                    [_apply_tariff(v, ts) for ts, v in zip(df.index, ens_aligned, strict=False)],
                    index=df.index,
                )
                ax_ens.plot(
                    df.index,
                    transformed,
                    color=_PINK_ORCHID,
                    lw=0.35,
                    alpha=0.30,
                )
            # Overlay the ensemble P50 for reference
            ax_ens.plot(
                df.index,
                df["ensemble_p50"],
                color=_DARK_VIOLET,
                lw=2.0,
                label="Ensemble P50",
                solid_capstyle="round",
            )
            if df_actual is not None and len(df_actual) > 0:
                ax_ens.step(
                    df_actual.index,
                    df_actual.values,  # type: ignore[arg-type]
                    where="post",
                    color=_EVERGREEN,
                    lw=1.6,
                    label="Actual",
                    zorder=5,
                )
            n_members = len(ens.columns)
            ax_ens.set_title(
                f"Ensemble Runs ({n_members} members)",
                fontsize=12,
                fontweight="bold",
                color=_BLACK,
                pad=8,
            )
            ax_ens.set_ylabel(y_label, fontsize=10, color=_BLACK)
            ax_ens.legend(
                loc="upper right",
                fontsize=8.5,
                framealpha=0.85,
                edgecolor=_PINK_ORCHID,
                fancybox=True,
            )
            _style_axis(ax_ens, date_fmt)
            if ax2 is not None:
                ax_ens.tick_params(labelbottom=False)

        # ── Panel: Model Predictions ──────────────────────────────────────
        if ax2 is not None and has_models:
            _align = lambda s: s.reindex(df.index)  # noqa: E731

            def _transform(raw: pd.Series) -> pd.Series:
                return pd.Series(
                    [_apply_tariff(v, ts) for ts, v in zip(df.index, raw, strict=False)],
                    index=df.index,
                )

            # XGBoost P10–P90 band
            if "xgboost_p10" in model_preds and "xgboost_p90" in model_preds:
                p10 = _transform(_align(model_preds["xgboost_p10"]))
                p90 = _transform(_align(model_preds["xgboost_p90"]))
                ax2.fill_between(
                    df.index,
                    p10,
                    p90,
                    color=_TEAL,
                    alpha=0.10,
                    label="XGB P10–P90",
                )
                ax2.plot(df.index, p10, color=_TEAL, lw=0.5, ls=":", alpha=0.4)
                ax2.plot(df.index, p90, color=_TEAL, lw=0.5, ls=":", alpha=0.4)

            if "xgboost_p50" in model_preds:
                ax2.plot(
                    df.index,
                    _transform(_align(model_preds["xgboost_p50"])),
                    color=_TEAL,
                    lw=1.4,
                    alpha=0.85,
                    label="XGB P50",
                )

            if "extreme_high" in model_preds:
                ax2.plot(
                    df.index,
                    _transform(_align(model_preds["extreme_high"])),
                    color="#c0392b",
                    lw=1.1,
                    ls="--",
                    alpha=0.75,
                    label="Extreme High",
                )

            if "extreme_low" in model_preds:
                ax2.plot(
                    df.index,
                    _transform(_align(model_preds["extreme_low"])),
                    color=_WARM_AMBER,
                    lw=1.1,
                    ls="--",
                    alpha=0.75,
                    label="Extreme Low",
                )

            ax2.plot(
                df.index,
                df["ensemble_p50"],
                color=_DARK_VIOLET,
                lw=2.0,
                label="Ensemble P50",
                solid_capstyle="round",
            )

            if df_actual is not None and len(df_actual) > 0:
                ax2.step(
                    df_actual.index,
                    df_actual.values,  # type: ignore[arg-type]
                    where="post",
                    color=_EVERGREEN,
                    lw=1.6,
                    label="Actual",
                    zorder=5,
                )

            ax2.set_title(
                "Model Predictions",
                fontsize=12,
                fontweight="bold",
                color=_BLACK,
                pad=8,
            )
            ax2.set_xlabel("Time (UTC)", fontsize=10, color=_BLACK)
            ax2.set_ylabel(y_label, fontsize=10, color=_BLACK)
            ax2.legend(
                loc="upper right",
                fontsize=8.5,
                framealpha=0.85,
                edgecolor=_PINK_ORCHID,
                fancybox=True,
                ncol=2,
            )
            _style_axis(ax2, date_fmt)

        # ── Scorecard ──────────────────────────────────────────────────────
        self._render_scorecard(
            ax_sc,
            train_metrics,
            live_metrics,
            score,
            max_score,
            grade_label,
            forecast,
        )

        # Rotate date tick labels on the bottom time-series axis only
        bottom_ax = ax2 or ax_ens or ax1
        for label in bottom_ax.get_xticklabels():
            label.set_rotation(30)
            label.set_ha("right")  # type: ignore[attr-defined]

        if output_path is None:
            output_path = Path(f"forecast_{zone.lower()}.png")
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        fig.savefig(
            output_path,
            dpi=200,
            bbox_inches="tight",
            facecolor=fig.get_facecolor(),
            edgecolor="none",
        )
        plt.close(fig)

        logger.info("Plot saved to %s", output_path)
        return output_path

    # ── Helpers: metrics & scorecard ──────────────────────────────────

    def _load_training_metrics(self, zone: str) -> dict:
        """Load per-model training metrics from cached model pickles."""
        metrics: dict = {}
        # XGBoost quantile
        try:
            raw = self._store.load_model(zone, "xgboost_quantile")
            if raw:
                payload = pickle.loads(raw)
                metrics["xgboost"] = payload.get("metrics", {})
        except Exception:
            pass
        # Extreme High
        try:
            raw = self._store.load_model(zone, "extreme_high")
            if raw:
                payload = pickle.loads(raw)
                metrics["extreme_high"] = payload.get("metrics", {})
        except Exception:
            pass
        # Extreme Low
        try:
            raw = self._store.load_model(zone, "extreme_low")
            if raw:
                payload = pickle.loads(raw)
                metrics["extreme_low"] = payload.get("metrics", {})
        except Exception:
            pass
        return metrics

    @staticmethod
    def _compute_live_metrics(
        df: pd.DataFrame,
        df_actual: pd.Series | None,
        forecast: ForecastResult,
    ) -> dict:
        """Derive forecast-quality metrics from the current run."""
        m: dict = {}

        # Model agreement (mean across forecast horizon)
        if "model_agreement" in df.columns:
            m["model_agreement"] = float(df["model_agreement"].mean())

        # Mean uncertainty band width (already tariff-transformed)
        if "lower" in df.columns and "upper" in df.columns:
            m["mean_band_width"] = float((df["upper"] - df["lower"]).mean())

        # Ensemble spread — std of per-hour member predictions
        if forecast.ensemble_members is not None and not forecast.ensemble_members.empty:
            hourly_std = forecast.ensemble_members.std(axis=1)
            m["ensemble_spread"] = float(hourly_std.mean())

        # Forecast horizon
        m["horizon_hours"] = len(df)

        # Live vs actual (only where timestamps overlap)
        if df_actual is not None and len(df_actual) > 0:
            common = df.index.intersection(df_actual.index)
            if len(common) >= 2:
                pred = df.loc[common, "ensemble_p50"]
                actual = df_actual.reindex(common)
                errors = pred - actual
                m["live_mae"] = float(errors.abs().mean())
                m["live_bias"] = float(errors.mean())
                # Directional accuracy
                pred_diff = pred.diff().iloc[1:]
                actual_diff = actual.diff().iloc[1:]
                if len(pred_diff) > 0:
                    same_dir = (pred_diff * actual_diff > 0).sum()
                    m["directional_accuracy"] = float(same_dir / len(pred_diff) * 100)
                m["n_actual_hours"] = len(common)
        return m

    @staticmethod
    def _composite_grade(
        train_metrics: dict,
        live_metrics: dict,
    ) -> tuple[float, int, str]:
        """Compute a 0–12 composite quality score and label.

        Scoring rubric (inspired by old runeflow judgement_score):
        ─ Training (up to 6 pts):
            MAE ≤ 8   → 2 pts,  ≤ 12 → 1 pt
            R²  ≥ 0.85 → 2 pts, ≥ 0.70 → 1 pt
            Coverage ≥ 93 → 2 pts, ≥ 85 → 1 pt
        ─ Forecast (up to 6 pts):
            Model agreement ≥ 0.7 → 2 pts, ≥ 0.4 → 1 pt
            Ensemble spread ≤ 8 → 2 pts, ≤ 15 → 1 pt
            Live MAE (if available) ≤ 0.02 → 2 pts, ≤ 0.04 → 1 pt
              (otherwise: directional accuracy ≥ 60 % → 2, ≥ 40 % → 1)
        """
        MAX_SCORE = 12
        pts = 0

        xgb = train_metrics.get("xgboost", {})
        mae = xgb.get("mae", float("inf"))
        r2 = xgb.get("r2", 0.0)
        cov = xgb.get("coverage", 0.0)

        pts += 2 if mae <= 8 else (1 if mae <= 12 else 0)
        pts += 2 if r2 >= 0.85 else (1 if r2 >= 0.70 else 0)
        pts += 2 if cov >= 93 else (1 if cov >= 85 else 0)

        agreement = live_metrics.get("model_agreement", 0.0)
        spread = live_metrics.get("ensemble_spread", float("inf"))
        pts += 2 if agreement >= 0.7 else (1 if agreement >= 0.4 else 0)
        pts += 2 if spread <= 8 else (1 if spread <= 15 else 0)

        if "live_mae" in live_metrics:
            lm = live_metrics["live_mae"]
            pts += 2 if lm <= 0.02 else (1 if lm <= 0.04 else 0)
        elif "directional_accuracy" in live_metrics:
            da = live_metrics["directional_accuracy"]
            pts += 2 if da >= 60 else (1 if da >= 40 else 0)

        if pts >= 10:
            label = "Excellent"
        elif pts >= 7:
            label = "Good"
        elif pts >= 4:
            label = "Fair"
        else:
            label = "Poor"

        # Normalise to a 0-10 display scale for readability
        display_score = round(pts / MAX_SCORE * 10, 1)
        return display_score, 10, label

    @staticmethod
    def _render_scorecard(
        ax: plt.Axes,
        train_metrics: dict,
        live_metrics: dict,
        score: float,
        max_score: int,
        grade_label: str,
        forecast: ForecastResult,
    ) -> None:
        """Draw the scorecard panel — a table-like text layout."""
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")

        # ── Separator line at top ────────────────────────────────────
        ax.axhline(y=0.97, xmin=0.01, xmax=0.99, color=_PINK_ORCHID, lw=0.8, alpha=0.5)

        # ── Grade badge (left column) ────────────────────────────────
        if score >= 8.3:
            badge_colour = "#27ae60"  # green  (was 10/12 → 8.3/10)
        elif score >= 5.8:
            badge_colour = _TEAL  # (was 7/12 → 5.8/10)
        elif score >= 3.3:
            badge_colour = _WARM_AMBER  # (was 4/12 → 3.3/10)
        else:
            badge_colour = "#c0392b"  # red

        badge = mpatches.FancyBboxPatch(
            (0.02, 0.40),
            0.20,
            0.50,
            boxstyle="round,pad=0.02",
            facecolor=badge_colour,
            edgecolor="none",
            alpha=0.10,
        )
        ax.add_patch(badge)
        # Format score: show integer if whole, else one decimal
        score_str = f"{score:.0f}" if score == int(score) else f"{score:.1f}"
        ax.text(
            0.12,
            0.74,
            f"{score_str}/{max_score}",
            fontsize=28,
            fontweight="bold",
            color=badge_colour,
            ha="center",
            va="center",
        )
        ax.text(
            0.12,
            0.52,
            grade_label,
            fontsize=9,
            color=badge_colour,
            ha="center",
            va="center",
            style="italic",
        )

        # Row spacing helper
        ROW_H = 0.105
        TITLE_Y = 0.92
        TOP_Y = 0.80  # first data row — larger gap below titles

        # ── Training metrics column (middle) ─────────────────────────
        col1_x = 0.30
        val1_x = 0.48
        ax.text(
            col1_x,
            TITLE_Y,
            "Training Metrics",
            fontsize=10,
            fontweight="bold",
            color=_BLACK,
            va="top",
        )

        xgb = train_metrics.get("xgboost", {})
        rows_train = [
            ("XGB MAE", f"{xgb.get('mae', float('nan')):.2f} EUR/MWh"),
            ("XGB R²", f"{xgb.get('r2', float('nan')):.3f}"),
            ("XGB Coverage", f"{xgb.get('coverage', float('nan')):.1f} %"),
        ]
        eh = train_metrics.get("extreme_high", {})
        el = train_metrics.get("extreme_low", {})
        if eh.get("mae") is not None:
            rows_train.append(("Ext. High MAE", f"{eh['mae']:.2f}"))
        if el.get("mae") is not None:
            rows_train.append(("Ext. Low MAE", f"{el['mae']:.2f}"))

        for i, (lbl, val) in enumerate(rows_train):
            y = TOP_Y - i * ROW_H
            ax.text(col1_x, y, lbl, fontsize=8, color="#555555", va="top")
            ax.text(val1_x, y, val, fontsize=8, fontweight="bold", color=_BLACK, va="top")

        # ── Forecast quality column (right) ──────────────────────────
        col2_x = 0.64
        val2_x = 0.82
        ax.text(
            col2_x,
            TITLE_Y,
            "Forecast Quality",
            fontsize=10,
            fontweight="bold",
            color=_BLACK,
            va="top",
        )

        rows_fc: list[tuple[str, str]] = [
            ("Model Agreement", f"{live_metrics.get('model_agreement', float('nan')):.1%}"),
            ("Ensemble Spread", f"{live_metrics.get('ensemble_spread', float('nan')):.1f} EUR/MWh"),
            ("Band Width", f"{live_metrics.get('mean_band_width', float('nan')):.4f} EUR/kWh"),
            ("Horizon", f"{live_metrics.get('horizon_hours', '?')} h"),
        ]
        if "live_mae" in live_metrics:
            rows_fc.append(
                (
                    f"Live MAE ({live_metrics.get('n_actual_hours', '?')}h)",
                    f"{live_metrics['live_mae']:.4f} EUR/kWh",
                )
            )
        if "live_bias" in live_metrics:
            bias = live_metrics["live_bias"]
            sign = "+" if bias >= 0 else ""
            rows_fc.append(("Live Bias", f"{sign}{bias:.4f}"))
        if "directional_accuracy" in live_metrics:
            rows_fc.append(
                (
                    "Dir. Accuracy",
                    f"{live_metrics['directional_accuracy']:.0f} %",
                )
            )

        for i, (lbl, val) in enumerate(rows_fc):
            y = TOP_Y - i * ROW_H
            ax.text(col2_x, y, lbl, fontsize=8, color="#555555", va="top")
            ax.text(val2_x, y, val, fontsize=8, fontweight="bold", color=_BLACK, va="top")

        # ── Footer: model version + timestamp ────────────────────────
        ts = forecast.created_at
        ver = forecast.model_version
        ax.text(
            0.99,
            0.03,
            f"model {ver}  ·  forecast {ts}",
            fontsize=7,
            color="#aaaaaa",
            ha="right",
            va="bottom",
        )
