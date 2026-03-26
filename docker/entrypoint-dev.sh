#!/bin/bash
# runeflow Development Entrypoint
#
# Runs the full 7-step pipeline once, then starts Python's built-in HTTP
# server to preview the generated site. Designed for local development via
# `make dev` — a single container handles everything.
#
# The pipeline is delegated to /entrypoint-oneshot.sh which handles all
# validation, env defaults, and progress output.
#
# Environment variables (same as entrypoint-oneshot.sh):
#   ENTSOE            — required: ENTSO-E API key
#   NED               — optional: NED API key (NL zone)
#   ZONE              — default: NL
#   TARIFF_PRICE_PROVIDER — default: wholesale
#   OUTPUT_FILE       — default: /outputs/tariffs.json
#   SITE_OUTPUT_DIR   — default: /outputs/site
#   DEV_PORT          — HTTP server port inside the container (default: 8000)
#   LOG_LEVEL         — default: INFO

set -euo pipefail

SITE_OUTPUT_DIR="${SITE_OUTPUT_DIR:-/outputs/site}"
DEV_PORT="${DEV_PORT:-8000}"

# ── 1. Run the full pipeline ───────────────────────────────────────────────
/entrypoint-oneshot.sh

# ── 2. Start local preview server ─────────────────────────────────────────
echo ""
echo "==================================="
echo " Dev server ready"
echo " → http://localhost:${DEV_PORT}"
echo " Ctrl-C to stop"
echo "==================================="

exec python -m http.server "${DEV_PORT}" --directory "${SITE_OUTPUT_DIR}"
