# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2024-2026 Thomas Phil — runeflow
# See LICENSE and COMMERCIAL-LICENSE.md for licensing details.

"""
ExportMetaService — writes site/api/meta.json with the zone/provider registry.

Consumed by the dashboard JavaScript to build the navigation menu without
any server-side logic.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd

from runeflow.zones.registry import ZoneRegistry

logger = logging.getLogger(__name__)

# Provider labels (fallback → provider_id if not found here)
_PROVIDER_LABELS: dict[str, str] = {
    "wholesale": "Wholesale (no markup)",
    "zonneplan": "Zonneplan",
    "tibber": "Tibber",
    "vattenfall": "Vattenfall",
    "eneco": "Eneco",
    "essent": "Essent",
    "greenchoice": "Greenchoice",
    "easy_energy": "EasyEnergy",
    "anwb_energie": "ANWB Energie",
    "leapp": "Leapp",
    "awattar": "aWATTar",
}


class ExportMetaService:
    """Write ``meta.json`` with the full zone/provider registry."""

    def run(self, output_path: Path) -> dict[str, object]:
        """Write meta.json to *output_path*.

        Returns the payload dict for testing / chaining.
        """
        zones_data = []
        for zone_code in ZoneRegistry.list_zones():
            cfg = ZoneRegistry.get(zone_code)
            providers = [
                {
                    "id": pid,
                    "label": _PROVIDER_LABELS.get(pid, pid.replace("_", " ").title()),
                }
                for pid in sorted(cfg.tariff_formulas.keys())
            ]
            zones_data.append(
                {
                    "zone": cfg.zone,
                    "label": cfg.name,
                    "timezone": cfg.timezone,
                    "providers": providers,
                }
            )

        payload: dict[str, object] = {
            "generated_at": pd.Timestamp.now("UTC").isoformat(),
            "zones": zones_data,
        }

        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        logger.info("ExportMeta: wrote %d zones to %s", len(zones_data), output_path)
        return payload
