#!/bin/bash
# runeflow One-Shot Pipeline Entrypoint
#
# Runs the full pipeline exactly once (no cron, no daemon) and exits.
# Intended for dev/test runs with --entrypoint, or CI pipelines.
#
# Steps: update-data → train → warmup-cache → inference →
#        export-tariffs → plot-uncertainty → build-site
#
# Required env: ENTSOE
# Optional env: ZONE (NL), PRICE_PROVIDER (wholesale),
#               OUTPUT_FILE (/outputs/tariffs.json),
#               SITE_OUTPUT_DIR (/outputs/site), LOG_LEVEL (INFO)

set -euo pipefail

# ---------------------------------------------------------------------------
# Validate required env
# ---------------------------------------------------------------------------
if [ -z "${ENTSOE:-}" ]; then
    echo "ERROR: ENTSOE environment variable (API key) is required"
    echo "Get your API key from: https://transparency.entsoe.eu/"
    exit 1
fi

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
ZONE="${ZONE:-NL}"
PRICE_PROVIDER="${PRICE_PROVIDER:-wholesale}"
OUTPUT_FILE="${OUTPUT_FILE:-/outputs/tariffs.json}"
SITE_OUTPUT_DIR="${SITE_OUTPUT_DIR:-/outputs/site}"
LOG_LEVEL="${LOG_LEVEL:-INFO}"

export ENTSOE ZONE PRICE_PROVIDER OUTPUT_FILE SITE_OUTPUT_DIR LOG_LEVEL
export XDG_CACHE_HOME=/app/.cache
export HOME=/root

OUTPUT_DIR=$(dirname "$OUTPUT_FILE")
CHART_FILE="${OUTPUT_DIR}/runeflow_uncertainty_${ZONE}.png"

echo "==================================="
echo " runeflow One-Shot Pipeline"
echo "==================================="
echo "  Zone:           $ZONE"
echo "  Price Provider: $PRICE_PROVIDER"
echo "  Output File:    $OUTPUT_FILE"
echo "  Site Dir:       $SITE_OUTPUT_DIR"
echo "==================================="

mkdir -p "$OUTPUT_DIR" "$SITE_OUTPUT_DIR" /app/.cache/runeflow

cd /app

# ---------------------------------------------------------------------------
# 1. Fetch / update historical data
# ---------------------------------------------------------------------------
echo ""
echo "── Step 1/7: Updating data ──────────────────────────────────────────"
runeflow update-data --zone "$ZONE" || {
    echo "WARNING: Data update had issues, continuing..."
}

# ---------------------------------------------------------------------------
# 2. Train model
# ---------------------------------------------------------------------------
echo ""
echo "── Step 2/7: Training model ─────────────────────────────────────────"
runeflow train --zone "$ZONE"

# ---------------------------------------------------------------------------
# 3. Warm up feature cache
# ---------------------------------------------------------------------------
echo ""
echo "── Step 3/7: Warming up cache ───────────────────────────────────────"
runeflow warmup-cache --zone "$ZONE" || {
    echo "WARNING: Warmup had issues, continuing..."
}

# ---------------------------------------------------------------------------
# 4. Run inference (produce forecast)
# ---------------------------------------------------------------------------
echo ""
echo "── Step 4/7: Running inference ──────────────────────────────────────"
runeflow inference --zone "$ZONE"

# ---------------------------------------------------------------------------
# 5. Export tariffs JSON
# ---------------------------------------------------------------------------
echo ""
echo "── Step 5/7: Exporting tariffs ──────────────────────────────────────"
runeflow export-tariffs \
    --zone "$ZONE" \
    --provider "$PRICE_PROVIDER" \
    --output "$OUTPUT_FILE"

# ---------------------------------------------------------------------------
# 6. Generate uncertainty chart
# ---------------------------------------------------------------------------
echo ""
echo "── Step 6/7: Generating chart ───────────────────────────────────────"
runeflow plot-uncertainty \
    --zone "$ZONE" \
    --output "$CHART_FILE" || {
    echo "WARNING: Chart generation failed (non-critical)"
}

# ---------------------------------------------------------------------------
# 7. Build static site
# ---------------------------------------------------------------------------
echo ""
echo "── Step 7/7: Building static site ──────────────────────────────────"
runeflow build-site \
    --zones "$ZONE" \
    --output "$SITE_OUTPUT_DIR"

echo ""
echo "==================================="
echo " Pipeline complete"
echo "  tariff JSON : $OUTPUT_FILE"
echo "  site        : $SITE_OUTPUT_DIR"
echo "==================================="
