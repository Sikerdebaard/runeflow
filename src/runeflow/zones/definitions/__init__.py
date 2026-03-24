# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2024-2026 Thomas Phil — runeflow
# See LICENSE and COMMERCIAL-LICENSE.md for licensing details.

"""
Auto-register all built-in zone definitions.

Importing this module triggers the :func:`ZoneRegistry.register` call
in each zone definition file.
"""

from runeflow.zones.definitions.de_lu import DE_LU  # noqa: F401
from runeflow.zones.definitions.nl import NL  # noqa: F401

__all__ = ["NL", "DE_LU"]