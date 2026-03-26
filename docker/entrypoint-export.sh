#!/bin/bash
set -e

# runeflow Tariff Exporter Entrypoint
# Runs inference on a schedule and writes tariff JSON.
#
# Scheduling (Amsterdam / Europe/Amsterdam time):
#   - 08:05  Full run: update-data → train → inference → export → site
#   - 20:05  Full run: update-data → train → inference → export → site
#   - 12:00–16:00 every 15 min: price watcher → if new day-ahead prices
#             detected, run update-data → inference → export → site (no retrain)

# ---------------------------------------------------------------------------
# Required environment variables
# ---------------------------------------------------------------------------
if [ -z "$ENTSOE" ]; then
    echo "ERROR: ENTSOE environment variable (API key) is required"
    echo "Get your API key from: https://transparency.entsoe.eu/"
    exit 1
fi

# ---------------------------------------------------------------------------
# Optional environment variables with defaults
# ---------------------------------------------------------------------------
ZONE="${ZONE:-NL}"
PRICE_PROVIDER="${PRICE_PROVIDER:-wholesale}"
OUTPUT_FILE="${OUTPUT_FILE:-/outputs/tariffs.json}"
SITE_OUTPUT_DIR="${SITE_OUTPUT_DIR:-/outputs/site}"
LOG_LEVEL="${LOG_LEVEL:-INFO}"

echo "==================================="
echo "runeflow Tariff Exporter"
echo "==================================="
echo "Zone:           $ZONE"
echo "Price Provider: $PRICE_PROVIDER"
echo "Output File:    $OUTPUT_FILE"
echo "Site Dir:       $SITE_OUTPUT_DIR"
echo "Timezone:       $(cat /etc/timezone)"
echo "==================================="

# ---------------------------------------------------------------------------
# Shared environment file (used by cron-launched scripts)
# ---------------------------------------------------------------------------
cat > /app/.env.docker << EOF
PATH=/usr/local/bin:/usr/bin:/bin
ENTSOE=$ENTSOE
ZONE=$ZONE
PRICE_PROVIDER=$PRICE_PROVIDER
OUTPUT_FILE=$OUTPUT_FILE
SITE_OUTPUT_DIR=$SITE_OUTPUT_DIR
LOG_LEVEL=$LOG_LEVEL
HOME=/root
XDG_CACHE_HOME=/app/.cache
EOF

export ENTSOE ZONE PRICE_PROVIDER OUTPUT_FILE SITE_OUTPUT_DIR LOG_LEVEL
export XDG_CACHE_HOME=/app/.cache

# ---------------------------------------------------------------------------
# Full run: update-data → train → warmup-cache → inference → export + plot
# ---------------------------------------------------------------------------
cat > /app/run-inference.sh << 'SCRIPT'
#!/bin/bash
set -e

source /app/.env.docker
export PATH ENTSOE XDG_CACHE_HOME HOME

LOGFILE="/var/log/runeflow/inference.log"
TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')
cd /app

echo "[$TIMESTAMP] ── Full run (update + train + inference) ── zone=$ZONE" >> "$LOGFILE"

# update-data MUST succeed — everything downstream depends on fresh data.
echo "[$TIMESTAMP] Updating data..." >> "$LOGFILE"
runeflow update-data --zone "$ZONE" >> "$LOGFILE" 2>&1 || {
    echo "[$TIMESTAMP] ERROR: Data update failed — aborting run." >> "$LOGFILE"
    exit 1
}

echo "[$TIMESTAMP] Training model..." >> "$LOGFILE"
runeflow train --zone "$ZONE" >> "$LOGFILE" 2>&1 || {
    echo "[$TIMESTAMP] ERROR: Training failed" >> "$LOGFILE"
    exit 1
}

echo "[$TIMESTAMP] Warming up cache..." >> "$LOGFILE"
runeflow warmup-cache --zone "$ZONE" >> "$LOGFILE" 2>&1 || \
    echo "[$TIMESTAMP] WARNING: Warmup had issues, continuing..." >> "$LOGFILE"

echo "[$TIMESTAMP] Running inference..." >> "$LOGFILE"
runeflow inference --zone "$ZONE" >> "$LOGFILE" 2>&1

OUTPUT_DIR=$(dirname "$OUTPUT_FILE")
CHART_FILE="${OUTPUT_DIR}/runeflow_uncertainty_${ZONE}.png"

echo "[$TIMESTAMP] Exporting tariffs..." >> "$LOGFILE"
runeflow export-tariffs --zone "$ZONE" \
    --provider "$PRICE_PROVIDER" \
    --output "$OUTPUT_FILE" >> "$LOGFILE" 2>&1

echo "[$TIMESTAMP] Generating uncertainty chart..." >> "$LOGFILE"
runeflow plot-uncertainty --zone "$ZONE" \
    --output "$CHART_FILE" >> "$LOGFILE" 2>&1 || \
    echo "[$TIMESTAMP] WARNING: Chart generation failed (non-critical)" >> "$LOGFILE"

echo "[$TIMESTAMP] Building static dashboard..." >> "$LOGFILE"
runeflow build-site --zones "$ZONE" --output "$SITE_OUTPUT_DIR" >> "$LOGFILE" 2>&1 || \
    echo "[$TIMESTAMP] WARNING: build-site failed (non-critical)" >> "$LOGFILE"

TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')
echo "[$TIMESTAMP] ✓ Full run complete — output: $OUTPUT_FILE  site: $SITE_OUTPUT_DIR" >> "$LOGFILE"
echo "" >> "$LOGFILE"
SCRIPT
chmod +x /app/run-inference.sh

# ---------------------------------------------------------------------------
# Inference-only run: warmup-cache → inference → export + plot (no training)
# Used for the mid-day day-ahead price update (~13:00 CET)
# ---------------------------------------------------------------------------
cat > /app/run-inference-only.sh << 'SCRIPT'
#!/bin/bash
set -e

source /app/.env.docker
export PATH ENTSOE XDG_CACHE_HOME HOME

LOGFILE="/var/log/runeflow/inference.log"
TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')
cd /app

echo "[$TIMESTAMP] ── Price-update inference run (update + inference, no retrain) ── zone=$ZONE" >> "$LOGFILE"

# Fetch latest prices before inference — required for meaningful output.
echo "[$TIMESTAMP] Updating data..." >> "$LOGFILE"
runeflow update-data --zone "$ZONE" >> "$LOGFILE" 2>&1 || {
    echo "[$TIMESTAMP] ERROR: Data update failed — aborting inference-only run." >> "$LOGFILE"
    exit 1
}

echo "[$TIMESTAMP] Warming up cache..." >> "$LOGFILE"
runeflow warmup-cache --zone "$ZONE" >> "$LOGFILE" 2>&1 || \
    echo "[$TIMESTAMP] WARNING: Warmup had issues, continuing..." >> "$LOGFILE"

echo "[$TIMESTAMP] Running inference..." >> "$LOGFILE"
runeflow inference --zone "$ZONE" >> "$LOGFILE" 2>&1

OUTPUT_DIR=$(dirname "$OUTPUT_FILE")
CHART_FILE="${OUTPUT_DIR}/runeflow_uncertainty_${ZONE}.png"

echo "[$TIMESTAMP] Exporting tariffs..." >> "$LOGFILE"
runeflow export-tariffs --zone "$ZONE" \
    --provider "$PRICE_PROVIDER" \
    --output "$OUTPUT_FILE" >> "$LOGFILE" 2>&1

echo "[$TIMESTAMP] Generating uncertainty chart..." >> "$LOGFILE"
runeflow plot-uncertainty --zone "$ZONE" \
    --output "$CHART_FILE" >> "$LOGFILE" 2>&1 || \
    echo "[$TIMESTAMP] WARNING: Chart generation failed (non-critical)" >> "$LOGFILE"

echo "[$TIMESTAMP] Building static dashboard..." >> "$LOGFILE"
runeflow build-site --zones "$ZONE" --output "$SITE_OUTPUT_DIR" >> "$LOGFILE" 2>&1 || \
    echo "[$TIMESTAMP] WARNING: build-site failed (non-critical)" >> "$LOGFILE"

TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')
echo "[$TIMESTAMP] ✓ Inference-only run complete — output: $OUTPUT_FILE  site: $SITE_OUTPUT_DIR" >> "$LOGFILE"
echo "" >> "$LOGFILE"
SCRIPT
chmod +x /app/run-inference-only.sh

# ---------------------------------------------------------------------------
# ENTSO-E day-ahead price watcher
# Checks once whether tomorrow's prices are out; if so, runs inference-only.
# Called every 15 min via cron (12:00–16:00). Uses a state file to avoid
# re-running once tomorrow's batch has already been processed.
# ---------------------------------------------------------------------------
cat > /app/check-prices.sh << 'SCRIPT'
#!/bin/bash
set -e

source /app/.env.docker
export PATH ENTSOE XDG_CACHE_HOME HOME

LOGFILE="/var/log/runeflow/entsoe-check.log"
STATEFILE="/app/.cache/runeflow/entsoe_last_processed.txt"
TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')

# Prevent concurrent runs
exec 200>/tmp/runeflow-check.lock
flock -n 200 || exit 0

TOMORROW=$(date -d "tomorrow" '+%Y-%m-%d')

if [ -f "$STATEFILE" ] && [ "$(cat "$STATEFILE")" = "$TOMORROW" ]; then
    exit 0  # Already processed today
fi

echo "[$TIMESTAMP] Checking for day-ahead prices ($TOMORROW)..." >> "$LOGFILE"

# Use the store to check if tomorrow's prices have been downloaded
PRICES_AVAILABLE=$(python3 - << PYTHON
import os, sys
os.environ.setdefault("XDG_CACHE_HOME", "/app/.cache")
os.environ.setdefault("ENTSOE", os.environ["ENTSOE"])
try:
    from runeflow.binder import configure_injector
    configure_injector(os.environ.get("ZONE", "NL"))
    import inject
    from runeflow.ports.store import DataStore
    from datetime import date, timedelta
    store = inject.instance(DataStore)
    tomorrow = date.today() + timedelta(days=1)
    zone = os.environ.get("ZONE", "NL")
    df = store.load_prices(zone)
    if df is not None and not df.empty:
        tomorrow_data = df[df.index.date == tomorrow]
        print("yes" if len(tomorrow_data) >= 20 else "no")
    else:
        print("no")
except Exception as e:
    print(f"check error: {e}", file=sys.stderr)
    print("no")
PYTHON
)

if [ "$PRICES_AVAILABLE" = "yes" ]; then
    echo "[$TIMESTAMP] Day-ahead prices detected — running inference..." >> "$LOGFILE"
    /app/run-inference-only.sh
    echo "$TOMORROW" > "$STATEFILE"
    echo "[$TIMESTAMP] ✓ Day-ahead update done" >> "$LOGFILE"
else
    echo "[$TIMESTAMP] Prices not yet available" >> "$LOGFILE"
fi
SCRIPT
chmod +x /app/check-prices.sh

# ---------------------------------------------------------------------------
# Cron schedule
# ---------------------------------------------------------------------------
# Full run (data + train + inference) at 08:05 and 14:05 Amsterdam time
cat > /etc/cron.d/runeflow << 'CRON'
SHELL=/bin/bash
PATH=/usr/local/bin:/usr/bin:/bin

# Full run: update-data → train → warmup → inference → export → site
5 8  * * * root /app/run-inference.sh      >> /var/log/runeflow/cron.log 2>&1
5 20 * * * root /app/run-inference.sh      >> /var/log/runeflow/cron.log 2>&1
# Day-ahead price watcher (12:00–16:00 Amsterdam, every 15 min)
# Triggers update-data → inference → export → site when new prices are detected
*/15 12-16 * * * root /app/check-prices.sh >> /var/log/runeflow/cron.log 2>&1
CRON
chmod 0644 /etc/cron.d/runeflow

# ---------------------------------------------------------------------------
# Init log files and output directory
# ---------------------------------------------------------------------------
touch /var/log/runeflow/inference.log \
      /var/log/runeflow/entsoe-check.log \
      /var/log/runeflow/cron.log
mkdir -p "$(dirname "$OUTPUT_FILE")" /app/.cache/runeflow

# ---------------------------------------------------------------------------
# Startup: run once immediately so the output file is available right away
# ---------------------------------------------------------------------------
echo "Running initial full pipeline on startup (update-data → train → inference)..."
echo "On a fresh server with an empty cache this will take several minutes."
/app/run-inference.sh || {
    echo ""
    echo "WARNING: Initial pipeline run failed."
    echo "This is normal on a completely fresh install where the data cache is empty."
    echo "The most common cause is a transient API error during 'update-data'."
    echo "The system will retry automatically at the next scheduled run (08:05 and 20:05 AMS)."
    echo "Check /var/log/runeflow/inference.log for details."
}

echo ""
echo "Cron jobs installed:"
echo "  - Full run (update + train + infer) at 08:05 and 14:05 Amsterdam"
echo "  - Day-ahead price watcher every 15 min (12:00–16:00)"
echo ""
echo "Logs: /var/log/runeflow/"
echo "Output: $OUTPUT_FILE"
echo ""

# ---------------------------------------------------------------------------
# Start cron (foreground) or exec arbitrary command
# ---------------------------------------------------------------------------
if [ "$1" = "cron" ]; then
    tail -F /var/log/runeflow/inference.log \
             /var/log/runeflow/cron.log &
    exec cron -f
else
    exec "$@"
fi
