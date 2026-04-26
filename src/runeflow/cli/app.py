# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2024-2026 Thomas Phil — runeflow
# See LICENSE and COMMERCIAL-LICENSE.md for licensing details.

"""
runeflow CLI — 7 Typer commands wrapping the service layer.

Usage:
    runeflow list-markets
    runeflow update-data --zone NL
    runeflow train --zone NL
    runeflow warmup-cache --zone NL
    runeflow inference --zone NL
    runeflow export-tariffs --zone NL --provider vattenfall
    runeflow plot-uncertainty --zone NL
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import cast

import typer
from loguru import logger

app = typer.Typer(
    name="runeflow",
    help="Day-ahead electricity price forecasting — hexagonal architecture.",
    no_args_is_help=True,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _InterceptHandler(logging.Handler):
    """Forward stdlib logging records to loguru."""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            level: str | int = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno
        frame, depth = logging.currentframe(), 2
        while frame and frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back  # type: ignore[assignment]
            depth += 1
        logger.opt(depth=depth, exception=record.exc_info).log(level, record.getMessage())


def _setup_logging() -> None:
    """Route all stdlib logging through loguru."""
    logging.basicConfig(handlers=[_InterceptHandler()], level=logging.DEBUG, force=True)
    # Silence noisy third-party loggers
    for name in ("httpx", "urllib3", "requests", "hpack", "h2"):
        logging.getLogger(name).setLevel(logging.WARNING)


def _setup(zone: str) -> None:
    """Configure logging and DI injector for *zone* before running any service."""
    _setup_logging()
    from runeflow.binder import configure_injector

    configure_injector(zone)


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


@app.command("list-markets")
def list_markets() -> None:
    """List all registered market zones."""
    from runeflow.zones.registry import ZoneRegistry

    zones = ZoneRegistry.list_zones()
    if not zones:
        typer.echo("No zones registered.")
        raise typer.Exit(1)
    typer.echo("Available market zones:")
    for z in sorted(zones):
        cfg = ZoneRegistry.get(z)
        typer.echo(f"  {z:10s}  {cfg.name}  (tz={cfg.timezone})")


@app.command("update-data")
def update_data(
    zone: str = typer.Option("NL", "--zone", "-z", help="Market zone (e.g. NL, DE_LU)"),
    years: str | None = typer.Option(
        None, "--years", help="Comma-separated list of years, e.g. 2020,2021,2022"
    ),
) -> None:
    """Download and cache prices, weather, generation and supplemental data."""
    _setup(zone)
    from runeflow.services.update_data import UpdateDataService

    parsed_years: tuple[int, ...] | None = None
    if years:
        parsed_years = tuple(int(y.strip()) for y in years.split(","))

    svc = UpdateDataService()
    svc.run(years=parsed_years)
    typer.echo(f"✓ Data updated for zone={zone}")


@app.command("train")
def train(
    zone: str = typer.Option("NL", "--zone", "-z", help="Market zone"),
) -> None:
    """Train the price-prediction ensemble for a zone."""
    _setup(zone)
    from runeflow.services.train import TrainService

    svc = TrainService()
    result = svc.run()
    xgb = result.metrics.get("xgboost_quantile", {})
    typer.echo(
        f"✓ Training complete — MAE={xgb.get('mae', 'n/a'):.4f}  "
        f"R²={xgb.get('r2', 'n/a'):.4f}  "
        f"Coverage={xgb.get('coverage', 'n/a'):.1f}%"
    )


@app.command("warmup-cache")
def warmup_cache(
    zone: str = typer.Option("NL", "--zone", "-z", help="Market zone"),
    force: bool = typer.Option(False, "--force", "-f", help="Regenerate even if cache exists"),
) -> None:
    """Build the feature warmup cache required for inference."""
    _setup(zone)
    from runeflow.services.warmup import WarmupService

    svc = WarmupService()
    df = svc.run(force=force)
    typer.echo(f"✓ Warmup cache ready ({len(df)} rows)")


@app.command("inference")
def inference(
    zone: str = typer.Option("NL", "--zone", "-z", help="Market zone"),
    output: Path | None = typer.Option(  # noqa: B008, E501
        None, "--output", "-o", help="Write forecast JSON"
    ),
) -> None:
    """Generate a 9-day price forecast."""
    _setup(zone)
    import inject

    from runeflow.ports.store import DataStore
    from runeflow.services.inference import InferenceService

    svc = InferenceService()
    result = svc.run()

    store: DataStore = inject.instance(DataStore)  # type: ignore[assignment]
    store.save_forecast(result)
    typer.echo(f"✓ Forecast generated: {len(result.points)} hours")

    if output:
        df = result.to_dataframe()
        output.write_text(df.to_json(orient="records", date_format="iso"), encoding="utf-8")
        typer.echo(f"✓ Forecast written to {output}")


@app.command("export-tariffs")
def export_tariffs(
    zone: str = typer.Option("NL", "--zone", "-z", help="Market zone"),
    provider: str = typer.Option("vattenfall", "--provider", "-p", help="Tariff provider ID"),
    output: Path | None = typer.Option(  # noqa: B008, E501
        None, "--output", "-o", help="Output JSON path"
    ),
) -> None:
    """Export forecast as tariff JSON."""
    _setup(zone)
    from runeflow.services.export_tariffs import ExportTariffsService

    svc = ExportTariffsService()
    slots = svc.run(provider=provider, output_path=output)
    typer.echo(f"✓ Exported {len(slots)} rate slots")


@app.command("plot-uncertainty")
def plot_uncertainty(
    zone: str = typer.Option("NL", "--zone", "-z", help="Market zone"),
    provider: str = typer.Option(
        "wholesale",
        "--provider",
        "-p",
        help="Tariff provider ID (e.g. wholesale, zonneplan, tibber)",
    ),
    output: Path | None = typer.Option(  # noqa: B008, E501
        None, "--output", "-o", help="Output PNG path"
    ),
) -> None:
    """Plot the uncertainty forecast band."""
    _setup(zone)
    from runeflow.services.plot import PlotService

    svc = PlotService()
    path = svc.run(output_path=output, provider=provider)
    typer.echo(f"✓ Plot saved to {path}")


@app.command("export-performance")
def export_performance(
    output: Path = typer.Option(  # noqa: B008
        Path("./site/api/performance.json"),
        "--output",
        "-o",
        help="Output JSON path",
    ),
    zones: str | None = typer.Option(
        None, "--zones", help="Comma-separated zone codes (default: all)"
    ),
) -> None:
    """Export model performance metrics as JSON."""
    from runeflow.zones.registry import ZoneRegistry

    zone_list = [z.strip() for z in zones.split(",")] if zones else ZoneRegistry.list_zones()
    if zone_list:
        _setup(zone_list[0])

    from runeflow.services.export_performance import ExportPerformanceService

    svc = ExportPerformanceService()
    svc.run(output_path=output, zones=zone_list)
    typer.echo(f"✓ Performance exported for {len(zone_list)} zones to {output}")


@app.command("build-site")
def build_site(
    output: Path = typer.Option(  # noqa: B008
        Path("./site"), "--output", "-o", help="Output directory for the static site"
    ),
    zones: str | None = typer.Option(
        None, "--zones", help="Comma-separated zone codes to include (default: all)"
    ),
) -> None:
    """Build a static HTML dashboard under OUTPUT.

    Iterates over every registered zone (or the subset given via --zones),
    exports tariff JSON + CSV, renders charts, and writes static HTML pages.
    No server-side code is needed to serve the result — just point nginx at
    the output directory.

    For local testing:

        runeflow build-site --output ./site
        python -m http.server 8080 --directory ./site
    """
    from runeflow.zones.registry import ZoneRegistry

    zone_list = [z.strip() for z in zones.split(",")] if zones else ZoneRegistry.list_zones()

    output.mkdir(parents=True, exist_ok=True)
    processed: list[str] = []

    for zone_code in zone_list:
        _setup(zone_code)
        import inject

        from runeflow.ports.store import DataStore

        if cast(DataStore, inject.instance(DataStore)).load_latest_forecast(zone_code) is None:
            typer.echo(f"  ⚠ {zone_code} skipped — no forecast data available yet")
            continue
        typer.echo(f"Building zone: {zone_code}")
        from runeflow.dashboard.build import BuildSiteZoneService

        svc = BuildSiteZoneService()
        try:
            svc.run(output_dir=output)
            processed.append(zone_code)
        except Exception as exc:
            typer.echo(f"  ✗ {zone_code} failed: {exc}", err=True)

    # Global files (meta.json, quality.json, index.html) — use last zone's injector
    if processed:
        _setup(processed[-1])
        from runeflow.dashboard.build import BuildSiteGlobalService

        BuildSiteGlobalService().run(output_dir=output, processed_zones=processed)

    typer.echo(f"✓ Site built in {output}  ({len(processed)}/{len(zone_list)} zones)")
    typer.echo(f"  Preview: python -m http.server 8080 --directory {output}")


@app.command("prefetch-archive")
def prefetch_archive(
    zones: str | None = typer.Option(
        None, "--zones", help="Comma-separated zone codes (default: all)"
    ),
    status_file: Path | None = typer.Option(  # noqa: B008
        None, "--status-file", help="Write JSON progress to FILE (updated after each zone)"
    ),
    pass_number: int = typer.Option(1, "--pass", help="Pass counter shown on the status page"),
) -> None:
    """Download and cache the historical weather archive for all configured zones.

    Each zone's full historical date range is fetched in quarterly chunks and
    stored with NEVER_EXPIRE caching in the Open-Meteo HTTP cache.  Already-
    cached quarters are skipped instantly, so this command is safe to interrupt
    and restart — it always picks up where it left off.

    Designed to warm the cache before the export container runs its first full
    pipeline, so that archive downloads do not stall the daily inference cycle.
    No ENTSOE key is required.
    """
    import datetime
    import json
    import os
    import time

    import inject

    from runeflow.exceptions import DailyRateLimitError
    from runeflow.ports.store import DataStore
    from runeflow.ports.weather import WeatherPort
    from runeflow.zones.config import ZoneConfig
    from runeflow.zones.registry import ZoneRegistry

    zone_list = [z.strip() for z in zones.split(",")] if zones else ZoneRegistry.list_zones()

    started_at = datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    run_start = time.monotonic()
    zones_state: dict[str, dict] = {z: {"status": "pending"} for z in zone_list}
    skipped: list[str] = []
    downloaded: list[str] = []
    failed: list[str] = []

    def _write_status(in_progress: str | None = None) -> None:
        if status_file is None:
            return
        status_file.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "started_at": started_at,
            "updated_at": datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "pass": pass_number,
            "elapsed_seconds": round(time.monotonic() - run_start, 1),
            "total": len(zone_list),
            "downloaded": len(downloaded),
            "skipped": len(skipped),
            "failed_count": len(failed),
            "in_progress": in_progress,
            "zones": zones_state,
        }
        tmp = status_file.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        os.replace(tmp, status_file)

    _write_status()  # initial state — all zones pending

    for i, zone_code in enumerate(zone_list):
        if i > 0:
            # Brief pause between zones — courtesy rate-limit buffer.
            time.sleep(5)

        typer.echo(f"[{zone_code}] checking archive…")
        zones_state[zone_code] = {"status": "in_progress"}
        _write_status(in_progress=zone_code)
        _setup(zone_code)

        zone_cfg: ZoneConfig = inject.instance(ZoneConfig)
        store: DataStore = cast(DataStore, inject.instance(DataStore))
        weather_port: WeatherPort = cast(WeatherPort, inject.instance(WeatherPort))

        year_list = list(zone_cfg.historical_years)
        start = datetime.date(year_list[0], 1, 1)
        # end date is previous year's December 31, inclusive — we assume the archive
        # is complete up to the end of last year, but not necessarily for the current year
        current_year = datetime.date.today().year
        end = datetime.date(current_year - 1, 12, 31)
        locations = list(zone_cfg.weather_locations)

        zone_start = time.monotonic()
        existing = store.load_weather(zone_code, start, end)
        full_coverage = False
        if existing is not None:
            try:
                import pandas as pd

                df = existing.to_dataframe()
                if not df.empty and isinstance(df.index, pd.DatetimeIndex):
                    data_start = df.index.min().date()
                    data_end = df.index.max().date()
                    # Require coverage of the full requested range.
                    # data_end only needs to reach end - 1 day because hourly
                    # data for the last calendar date has timestamps on that
                    # date (e.g. 2025-12-31 23:00), so .date() == end - 1 day.
                    full_coverage = data_start <= start and data_end >= end - datetime.timedelta(
                        days=1
                    )
            except Exception:
                pass
        if full_coverage:
            typer.echo("  ✓ already in store, skipping")
            zones_state[zone_code] = {
                "status": "skipped",
                "duration_seconds": round(time.monotonic() - zone_start, 1),
            }
            skipped.append(zone_code)
            _write_status()
            continue

        try:
            series = weather_port.download_historical(locations, start, end)
            store.save_weather(series, zone_code)
            zones_state[zone_code] = {
                "status": "downloaded",
                "duration_seconds": round(time.monotonic() - zone_start, 1),
            }
            downloaded.append(zone_code)
            typer.echo("  ✓ downloaded and cached")
        except DailyRateLimitError as exc:
            typer.echo(f"  ✗ daily API limit reached: {exc}", err=True)
            zones_state[zone_code] = {
                "status": "failed",
                "error": str(exc)[:200],
                "duration_seconds": round(time.monotonic() - zone_start, 1),
            }
            remaining = [z for z in zone_list if zones_state[z]["status"] == "pending"]
            for z in remaining:
                zones_state[z]["status"] = "daily_limit"
            _write_status()
            typer.echo(
                f"  Stopping — {len(remaining)} zone(s) skipped. Retry tomorrow.",
                err=True,
            )
            raise typer.Exit(2) from None
        except Exception as exc:
            typer.echo(f"  ✗ failed: {exc}", err=True)
            zones_state[zone_code] = {
                "status": "failed",
                "error": str(exc)[:200],
                "duration_seconds": round(time.monotonic() - zone_start, 1),
            }
            failed.append(zone_code)

        _write_status()

    _write_status()  # final — in_progress cleared
    typer.echo(
        f"\n✓ Archive prefetch complete — "
        f"downloaded={len(downloaded)}  skipped={len(skipped)}  failed={len(failed)}"
    )
    if failed:
        typer.echo(f"  Failed zones: {', '.join(failed)}", err=True)
        raise typer.Exit(1)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    app()


if __name__ == "__main__":  # pragma: no cover
    main()
