"""
Plotting service for electricity price forecasts.

Provides visualisation of inference results with uncertainty bands,
comparing static vs weather-aware uncertainty, and showing the
weather uncertainty factor over the forecast horizon.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

from runeflow.binder import logger, inject, BaseConfig
from runeflow.service.inference import download_actual_electricity_prices
from runeflow.service.tariffs import PRICE_FORMULAS


def plot_uncertainty_bands(
    zone: str,
    output_path: str | Path | None = None,
    price_provider: str | None = None,
) -> Path:
    """
    Generate a 3-panel chart comparing static vs weather-aware uncertainty bands.

    Panel 1 – Ensemble forecast with **combined** uncertainty bands (outer
              envelope of weather-ensemble spread and model quantile uncertainty)
              and the **static** model-quantile-only bands for comparison, plus
              actual prices.
    Panel 2 – Band-width comparison: combined vs model-quantile-only.
    Panel 3 – Individual model predictions (XGBoost P50, KNN, Extreme).

    Parameters
    ----------
    zone : str
        Market zone code (e.g. 'NL').
    output_path : str | Path | None
        Where to save the chart.  Defaults to ``/tmp/runeflow_uncertainty_{zone}.png``.
    price_provider : str | None
        Price provider key (e.g. ``'zonneplan'``, ``'wholesale'``).  When set,
        prices are converted from EUR/MWh to EUR/kWh and the provider formula
        (taxes, fees) is applied so the chart matches the tariff JSON values.
        When ``None`` raw EUR/kWh wholesale prices are shown.

    Returns
    -------
    Path
        The path the chart was saved to.
    """
    config: BaseConfig = inject.instance(BaseConfig)
    orient = config.default_pandas_to_json_orient

    # --- Load raw inference results (EUR/MWh, before kWh conversion) ---------
    results_file = config.data_cache_dir / f"inference_results_{zone}.json"
    if not results_file.exists():
        raise FileNotFoundError(
            f"No inference results found at {results_file}. Run inference first."
        )

    df = pd.read_json(results_file, orient=orient)
    df = df.astype(float)
    logger.info(f"Loaded inference results: {len(df)} rows, columns={list(df.columns)}")

    # Build the price transformation ----------------------------------------
    # When a price_provider is given, apply its formula (taxes + fees) so
    # the chart shows the same consumer prices as the tariff JSON.
    if price_provider and price_provider.lower() in PRICE_FORMULAS:
        _base_formula = PRICE_FORMULAS[price_provider.lower()]
        def _price_transform(p_mwh):
            return _base_formula(p_mwh / 1_000.0)  # MWh → kWh → consumer
        price_label = f"Price (€/kWh, {price_provider})"
    else:
        def _price_transform(p_mwh):
            return p_mwh / 1_000.0  # raw wholesale EUR/kWh
        price_label = "Price (€/kWh)"

    # Apply to every price column (skip n_ensemble_members)
    n_members_raw = df["n_ensemble_members"].copy() if "n_ensemble_members" in df.columns else None
    price_cols = [c for c in df.columns if c != "n_ensemble_members"]
    for col in price_cols:
        df[col] = df[col].apply(_price_transform)

    # --- Load actual prices for overlay -------------------------------------
    try:
        df_actual = download_actual_electricity_prices(zone)
        df_actual.rename(columns={df_actual.columns[0]: "actual"}, inplace=True)
        df_actual["actual"] = df_actual["actual"].apply(_price_transform)
        # Use outer join so that today's actual prices are included even when
        # the inference results only start from tomorrow.
        df = df.join(df_actual, how="outer")
    except Exception as e:
        logger.warning(f"Could not download actual prices: {e}")

    # Always show from start of today so the current day is always visible.
    try:
        today_start = pd.Timestamp.now(tz=config.default_timezone).normalize()
        if df.index.tz is None:
            today_start = today_start.tz_localize(None)
        df = df[df.index >= today_start]
    except Exception as e:
        logger.warning(f"Could not filter to today: {e}")

    # --- Determine output path ----------------------------------------------
    if output_path is None:
        output_path = Path(f"/tmp/runeflow_uncertainty_{zone}.png")
    else:
        output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # --- Check which columns are available -----------------------------------
    has_static = (
        "prediction_lower_static" in df.columns
        and "prediction_upper_static" in df.columns
    )
    has_ensemble_bands = (
        "prediction_lower" in df.columns
        and "prediction_upper" in df.columns
        and has_static  # only show comparison when both exist
    )

    # --- Create figure -------------------------------------------------------
    n_panels = 3 if has_ensemble_bands else 2
    fig, axes = plt.subplots(n_panels, 1, figsize=(16, 4.5 * n_panels), sharex=True)
    if n_panels == 2:
        axes = list(axes) + [None]  # pad so indexing is consistent

    # ---- Panel 1: Ensemble + bands ------------------------------------------
    ax1 = axes[0]
    ax1.plot(
        df.index, df["prediction_ensemble"],
        label="Ensemble", color="purple", linewidth=2,
    )
    # Combined (weather ensemble ∪ model quantiles) bands – light-green, transparent, dashed outlines
    ax1.fill_between(
        df.index,
        df["prediction_lower"].values,
        df["prediction_upper"].values,
        color="lightgreen", alpha=0.20,
        label="95% interval (weather ensemble ∪ model quantiles)",
    )
    ax1.plot(
        df.index, df["prediction_lower"].values,
        color="lightgreen", linewidth=0.9, linestyle="--", alpha=0.8,
    )
    ax1.plot(
        df.index, df["prediction_upper"].values,
        color="lightgreen", linewidth=0.9, linestyle="--", alpha=0.8,
    )
    # Static bands (model quantiles / P50 uncertainty) – purple fill + dashed outlines
    if has_static:
        ax1.fill_between(
            df.index,
            df["prediction_lower_static"].values,
            df["prediction_upper_static"].values,
            color="purple", alpha=0.18,
            label="95% interval (model quantiles)",
        )
        ax1.plot(
            df.index, df["prediction_lower_static"],
            color="purple", linewidth=0.9, linestyle="--", alpha=0.7,
        )
        ax1.plot(
            df.index, df["prediction_upper_static"],
            color="purple", linewidth=0.9, linestyle="--", alpha=0.7,
        )
    # Actual prices
    if "actual" in df.columns:
        ax1.plot(
            df.index, df["actual"],
            label="Actual", color="orange", linewidth=1.5,
        )
    ax1.set_title(f"Ensemble Forecast — Zone {zone}", fontsize=13, fontweight="bold")
    ax1.set_ylabel(price_label)
    ax1.legend(loc="upper right", fontsize=9)
    ax1.grid(alpha=0.25)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%a %d %b %H:%M"))

    # ---- Panel 2: Uncertainty band comparison --------------------------------
    if has_ensemble_bands and axes[1] is not None:
        ax2 = axes[1]

        ensemble_width = df["prediction_upper"] - df["prediction_lower"]
        static_width = df["prediction_upper_static"] - df["prediction_lower_static"]

        ax2.fill_between(df.index, 0, ensemble_width, color="lightgreen", alpha=0.30, label="Combined band width (weather ∪ model quantiles)")
        ax2.fill_between(df.index, 0, static_width, color="purple", alpha=0.25, label="Model quantile band width only")
        ax2.plot(df.index, ensemble_width, color="lightgreen", linewidth=1.2, linestyle="--")
        ax2.plot(df.index, static_width, color="purple", linewidth=1.2, linestyle="--")
        ax2.set_ylabel(price_label)
        ax2.set_title("Uncertainty Band Width: Combined (weather ∪ model quantiles) vs Model-Only", fontsize=12)
        ax2.legend(loc="upper left", fontsize=9)
        ax2.grid(alpha=0.25)
        ax2.xaxis.set_major_formatter(mdates.DateFormatter("%a %d %b %H:%M"))

    # ---- Panel 3: Individual models -----------------------------------------
    ax3 = axes[2] if axes[2] is not None else axes[1]
    ax3.plot(df.index, df["prediction_p50"], label="XGBoost P50", color="blue", alpha=0.7)
    ax3.plot(df.index, df["prediction_knn"], label="KNN", color="green", linestyle="--", alpha=0.7)
    ax3.plot(df.index, df["prediction_extreme"], label="Extreme", color="red", linestyle=":", alpha=0.7)
    ax3.plot(df.index, df["prediction_ensemble"], label="Ensemble", color="purple", linewidth=2)
    ax3.set_title("Individual Model Predictions", fontsize=12)
    ax3.set_xlabel("Time")
    ax3.set_ylabel(price_label)
    ax3.legend(loc="upper right", fontsize=9)
    ax3.grid(alpha=0.25)
    ax3.xaxis.set_major_formatter(mdates.DateFormatter("%a %d %b %H:%M"))

    fig.autofmt_xdate(rotation=30)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    logger.info(f"Uncertainty bands chart saved to {output_path}")
    return output_path
