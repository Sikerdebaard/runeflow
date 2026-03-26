# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2024-2026 Thomas Phil — runeflow
# See LICENSE and COMMERCIAL-LICENSE.md for licensing details.
"""
BuildSiteService — orchestrates static site generation.

The service is called once per zone from the CLI command.  The CLI
is responsible for configuring the injector per zone before calling
:meth:`BuildSiteZoneService.run`.

Global files (api/meta.json, api/quality.json, index.html) are
written by :meth:`BuildSiteGlobalService.run` after all zones have
been processed.
"""

from __future__ import annotations

import contextlib
import logging
import shutil
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from jinja2 import Environment as _Jinja2Env

import inject
import pandas as pd

from runeflow.ports.store import DataStore
from runeflow.zones.config import ZoneConfig

logger = logging.getLogger(__name__)


class BuildSiteZoneService:
    """Generate the site subtree for a single zone.

    For each provider in the zone's tariff_formulas this writes:
      ``{output_dir}/{zone}/{provider}/tariff.json``
      ``{output_dir}/{zone}/{provider}/tariff.csv``
      ``{output_dir}/{zone}/{provider}/chart.png``

    It also renders:
      ``{output_dir}/{zone}/index.html``
      ``{output_dir}/{zone}/{provider}/index.html``
    """

    @inject.autoparams()
    def __init__(
        self,
        zone_cfg: ZoneConfig = inject.attr("zone_config"),  # type: ignore[assignment]  # noqa: B008
        store: DataStore = inject.attr(DataStore),  # type: ignore[assignment]  # noqa: B008
    ) -> None:
        self._zone_cfg = zone_cfg
        self._store = store

    # ------------------------------------------------------------------
    def run(self, output_dir: Path) -> None:
        """Build the site subtree for this zone under *output_dir*."""
        zone = self._zone_cfg.zone
        zone_dir = output_dir / zone
        zone_dir.mkdir(parents=True, exist_ok=True)

        from runeflow.services.export_tariffs import ExportTariffsService
        from runeflow.services.plot import PlotService

        export_svc = ExportTariffsService()
        plot_svc = PlotService()

        provider_summaries: list[dict[str, object]] = []
        forecast = self._store.load_latest_forecast(zone)

        for provider_id, formula in sorted(self._zone_cfg.tariff_formulas.items()):
            provider_dir = zone_dir / provider_id
            provider_dir.mkdir(parents=True, exist_ok=True)

            # Export tariff JSON + CSV
            json_path = provider_dir / "tariff.json"
            try:
                export_svc.run(
                    provider=provider_id,
                    output_path=json_path,
                )
                # Load the enriched JSON to get is_actual per slot
                import json

                tariff_data = json.loads(json_path.read_text(encoding="utf-8"))
                # Support both new format (rates/value) and old format (zones/price)
                rich_slots = tariff_data.get("rates") or tariff_data.get("zones", [])
                logger.info("  [%s/%s] wrote %d slots", zone, provider_id, len(rich_slots))
            except Exception as exc:
                logger.error("  [%s/%s] tariff export failed: %s", zone, provider_id, exc)
                rich_slots = []

            # Plot
            try:
                plot_svc.run(
                    output_path=provider_dir / "chart.png",
                    provider=provider_id,
                )
                chart_available = True
            except Exception as exc:
                logger.warning("  [%s/%s] plot failed: %s", zone, provider_id, exc)
                chart_available = False

            # Current price from the first slot
            current_price: float | None = None
            if rich_slots:
                now_utc = pd.Timestamp.now("UTC")
                for s in rich_slots:
                    try:
                        start = pd.Timestamp(s["start"])
                        end = pd.Timestamp(s["end"])
                        if start <= now_utc < end:
                            current_price = float(s.get("value", s.get("price", 0)))
                            break
                    except Exception:
                        pass
                if current_price is None:
                    with contextlib.suppress(Exception):
                        s0 = rich_slots[0]
                        current_price = float(s0.get("value", s0.get("price", 0)))

            provider_summaries.append(
                {
                    "id": provider_id,
                    "label": formula.label if hasattr(formula, "label") else provider_id,
                    "slot_count": len(rich_slots),
                    "current_price": current_price,
                    "chart_available": chart_available,
                }
            )

            # Render provider detail page
            _render_provider_page(
                output_path=provider_dir / "index.html",
                zone_code=zone,
                zone_name=self._zone_cfg.name,
                provider_id=provider_id,
                provider_label=formula.label if hasattr(formula, "label") else provider_id,
                slots=rich_slots,
                current_price=current_price,
                chart_available=chart_available,
                model_version=forecast.model_version if forecast else "unknown",
                generated_at=forecast.created_at.isoformat() if forecast else "",
                timezone=self._zone_cfg.timezone,
            )

        # Render zone overview page
        _render_zone_page(
            output_path=zone_dir / "index.html",
            zone_code=zone,
            zone_name=self._zone_cfg.name,
            providers=provider_summaries,
            generated_at=pd.Timestamp.now("UTC").isoformat(),
            timezone=self._zone_cfg.timezone,
        )

        logger.info("[%s] site built in %s", zone, zone_dir)


# ---------------------------------------------------------------------------
# Global (cross-zone) services
# ---------------------------------------------------------------------------


class BuildSiteGlobalService:
    """Writes the main index.html, api/meta.json and api/quality.json."""

    @inject.autoparams()
    def __init__(
        self,
        store: DataStore = inject.attr(DataStore),  # type: ignore[assignment]  # noqa: B008
    ) -> None:
        self._store = store

    def run(self, output_dir: Path, processed_zones: list[str]) -> None:
        """Write global files under *output_dir*."""
        from runeflow.services.export_meta import ExportMetaService
        from runeflow.services.export_quality import ExportQualityService

        ExportMetaService().run(output_path=output_dir / "api" / "meta.json")
        ExportQualityService().run(
            output_path=output_dir / "api" / "quality.json",
            zones=processed_zones,
        )

        # Copy static assets
        _copy_assets(output_dir / "assets")

        # Render landing page
        _render_index_page(
            output_path=output_dir / "index.html",
            zones=processed_zones,
            generated_at=pd.Timestamp.now("UTC").isoformat(),
        )
        # Render API / docs page
        _render_docs_page(
            output_path=output_dir / "docs" / "index.html",
            generated_at=pd.Timestamp.now("UTC").isoformat(),
        )
        # Write SEO files
        _write_robots_txt(output_dir / "robots.txt")
        _write_sitemap(
            output_dir / "sitemap.xml",
            base_url="https://runeflow.eu",
            zones=processed_zones,
        )
        logger.info("Global site files written under %s", output_dir)


# ---------------------------------------------------------------------------
# Rendering helpers (Jinja2 + bundled templates)
# ---------------------------------------------------------------------------

_TEMPLATES_DIR = Path(__file__).parent / "templates"
_ASSETS_DIR = Path(__file__).parent / "assets"


def _get_env() -> _Jinja2Env:
    """Return (and cache) the Jinja2 Environment."""
    try:
        return _get_env._env  # type: ignore[attr-defined, no-any-return]
    except AttributeError:
        from jinja2 import Environment, FileSystemLoader

        env = Environment(
            loader=FileSystemLoader(str(_TEMPLATES_DIR)),
            autoescape=True,
        )
        _get_env._env = env  # type: ignore[attr-defined]
        return env


def _render(template_name: str, dest: Path, **ctx: object) -> None:
    env = _get_env()
    tmpl = env.get_template(template_name)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(tmpl.render(**ctx), encoding="utf-8")


def _render_index_page(
    *,
    output_path: Path,
    zones: list[str],
    generated_at: str,
) -> None:
    from runeflow.zones.registry import ZoneRegistry

    zone_cards = [
        {
            "code": z,
            "name": ZoneRegistry.get(z).name,
            "provider_count": len(ZoneRegistry.get(z).tariff_formulas),
        }
        for z in zones
    ]
    _render("index.html", output_path, zones=zone_cards, generated_at=generated_at)


def _render_docs_page(
    *,
    output_path: Path,
    generated_at: str,
) -> None:
    _render("docs.html", output_path, generated_at=generated_at)


def _render_zone_page(
    *,
    output_path: Path,
    zone_code: str,
    zone_name: str,
    providers: list[dict[str, object]],
    generated_at: str,
    timezone: str,
) -> None:
    _render(
        "zone.html",
        output_path,
        zone_code=zone_code,
        zone_name=zone_name,
        providers=providers,
        generated_at=generated_at,
        timezone=timezone,
    )


def _render_provider_page(
    *,
    output_path: Path,
    zone_code: str,
    zone_name: str,
    provider_id: str,
    provider_label: str,
    slots: list[object],
    current_price: float | None,
    chart_available: bool,
    model_version: str,
    generated_at: str,
    timezone: str,
) -> None:
    # Price colour bucket: relative to daily range
    price_class = "price-normal"
    if slots and current_price is not None:
        prices = []
        for s in slots:
            s_any: Any = s
            with contextlib.suppress(Exception):
                if isinstance(s_any, dict):
                    prices.append(float(s_any.get("value", s_any.get("price", 0))))
                else:
                    prices.append(float(getattr(s_any, "value", getattr(s_any, "price", 0))))
        if prices:
            day_min, day_max = min(prices), max(prices)
            day_range = day_max - day_min or 1.0
            rel = (current_price - day_min) / day_range
            price_class = (
                "price-cheap"
                if rel < 0.33
                else ("price-expensive" if rel > 0.67 else "price-normal")
            )

    _render(
        "provider.html",
        output_path,
        zone_code=zone_code,
        zone_name=zone_name,
        provider_id=provider_id,
        provider_label=provider_label,
        slots=slots,
        current_price=current_price,
        chart_available=chart_available,
        model_version=model_version,
        generated_at=generated_at,
        price_class=price_class,
        timezone=timezone,
    )


def _copy_assets(dest_dir: Path) -> None:
    """Copy bundled assets (CSS/JS) to *dest_dir*."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    if _ASSETS_DIR.exists():
        for asset in _ASSETS_DIR.iterdir():
            shutil.copy2(asset, dest_dir / asset.name)


def _write_robots_txt(dest: Path) -> None:
    dest.write_text(
        "User-agent: *\nAllow: /\nSitemap: https://runeflow.eu/sitemap.xml\n",
        encoding="utf-8",
    )


def _write_sitemap(dest: Path, base_url: str, zones: list[str]) -> None:
    from runeflow.zones.registry import ZoneRegistry

    today = pd.Timestamp.now("UTC").strftime("%Y-%m-%d")
    urls: list[str] = [base_url + "/", base_url + "/docs/"]
    for zone in zones:
        urls.append(f"{base_url}/{zone}/")
        zone_cfg = ZoneRegistry.get(zone)
        for provider_id in sorted(zone_cfg.tariff_formulas):
            urls.append(f"{base_url}/{zone}/{provider_id}/")

    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">',
    ]
    for url in urls:
        lines += [
            "  <url>",
            f"    <loc>{url}</loc>",
            f"    <lastmod>{today}</lastmod>",
            "  </url>",
        ]
    lines.append("</urlset>")
    dest.write_text("\n".join(lines) + "\n", encoding="utf-8")
