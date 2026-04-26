# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2024-2026 Thomas Phil — runeflow
# See LICENSE and COMMERCIAL-LICENSE.md for licensing details.

"""ZoneRegistry — central registry of all supported ENTSO-E zones."""

from __future__ import annotations

from runeflow.exceptions import UnsupportedZoneError
from runeflow.zones.config import ZoneConfig


class ZoneRegistry:
    """Central registry of all supported ENTSO-E zones."""

    _zones: dict[str, ZoneConfig] = {}

    @classmethod
    def register(cls, config: ZoneConfig) -> None:
        """Register a zone configuration."""
        cls._zones[config.zone] = config

    @classmethod
    def get(cls, zone: str) -> ZoneConfig:
        """
        Return the ZoneConfig for *zone*.

        Raises UnsupportedZoneError if the zone is not registered.
        """
        # Trigger lazy registration of built-in zones
        _ensure_default_zones_registered()
        if zone not in cls._zones:
            raise UnsupportedZoneError(
                f"Zone '{zone}' not supported. Available: {', '.join(cls.list_zones())}"
            )
        return cls._zones[zone]

    @classmethod
    def list_zones(cls) -> list[str]:
        """Return sorted list of active (non-disabled) zone codes."""
        _ensure_default_zones_registered()
        return sorted(z for z, c in cls._zones.items() if c.disabled_reason is None)

    @classmethod
    def list_disabled_zones(cls) -> list[tuple[str, str]]:
        """Return sorted list of (zone_code, reason) for disabled zones."""
        _ensure_default_zones_registered()
        return sorted(
            (z, c.disabled_reason) for z, c in cls._zones.items() if c.disabled_reason is not None
        )

    @classmethod
    def clear(cls) -> None:
        """Clear all registered zones (for testing only)."""
        cls._zones.clear()


_default_zones_registered = False


def _ensure_default_zones_registered() -> None:
    global _default_zones_registered
    if not _default_zones_registered:
        _default_zones_registered = True
        # Import triggers ZoneRegistry.register() calls
        import runeflow.zones.definitions  # noqa: F401
