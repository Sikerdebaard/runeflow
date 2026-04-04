# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2024-2026 Thomas Phil — runeflow
# See LICENSE and COMMERCIAL-LICENSE.md for licensing details.

"""
Auto-register all built-in zone definitions.

Importing this module triggers the :func:`ZoneRegistry.register` call
in each zone definition file.
"""

# Core zones (NL and DE_LU — fully enriched with provider tariffs)
from runeflow.zones.definitions.at import AT  # noqa: F401
from runeflow.zones.definitions.balkans import BA, ME, MK, RS, XK  # noqa: F401

# Baltic states
from runeflow.zones.definitions.baltics import EE, LT, LV  # noqa: F401
from runeflow.zones.definitions.be import BE  # noqa: F401
from runeflow.zones.definitions.bg import BG  # noqa: F401
from runeflow.zones.definitions.ch import CH  # noqa: F401
from runeflow.zones.definitions.cz_sk import CZ, SK  # noqa: F401
from runeflow.zones.definitions.de_lu import DE_LU  # noqa: F401

# Nordic
from runeflow.zones.definitions.dk import DK_1, DK_2  # noqa: F401
from runeflow.zones.definitions.fi import FI  # noqa: F401

# Western / Central Europe
from runeflow.zones.definitions.fr import FR  # noqa: F401

# British Isles
from runeflow.zones.definitions.gb_ie import GB, IE  # noqa: F401
from runeflow.zones.definitions.gr import GR  # noqa: F401
from runeflow.zones.definitions.hu import HU  # noqa: F401

# Iberian Peninsula
from runeflow.zones.definitions.iberia import ES, PT  # noqa: F401

# Italy (regional bidding zones)
from runeflow.zones.definitions.it import (  # noqa: F401
    IT_CNOR,
    IT_CSUD,
    IT_NORD,
    IT_SARD,
    IT_SICI,
    IT_SUD,
)

# Island / isolated systems
from runeflow.zones.definitions.mt_cy import CY, MT  # noqa: F401
from runeflow.zones.definitions.nl import NL  # noqa: F401
from runeflow.zones.definitions.no import NO_1, NO_2, NO_3, NO_4, NO_5  # noqa: F401

# Central-Eastern Europe
from runeflow.zones.definitions.pl import PL  # noqa: F401
from runeflow.zones.definitions.ro import RO  # noqa: F401
from runeflow.zones.definitions.se import SE_1, SE_2, SE_3, SE_4  # noqa: F401

# South-Eastern Europe
from runeflow.zones.definitions.si_hr import HR, SI  # noqa: F401

__all__ = [
    # Core
    "NL",
    "DE_LU",
    # Western / Central Europe
    "FR",
    "BE",
    "AT",
    "CH",
    # Iberia
    "ES",
    "PT",
    # Italy
    "IT_NORD",
    "IT_CNOR",
    "IT_CSUD",
    "IT_SUD",
    "IT_SICI",
    "IT_SARD",
    # Central-Eastern Europe
    "PL",
    "CZ",
    "SK",
    "HU",
    "RO",
    "BG",
    "GR",
    # South-Eastern Europe
    "SI",
    "HR",
    "RS",
    "BA",
    "ME",
    "MK",
    "XK",
    # Nordic
    "DK_1",
    "DK_2",
    "FI",
    "NO_1",
    "NO_2",
    "NO_3",
    "NO_4",
    "NO_5",
    "SE_1",
    "SE_2",
    "SE_3",
    "SE_4",
    # Baltic
    "EE",
    "LV",
    "LT",
    # British Isles
    "GB",
    "IE",
    # Islands
    "MT",
    "CY",
]
