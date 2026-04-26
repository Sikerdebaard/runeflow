#!/bin/bash
set -euo pipefail

# Open-Meteo Self-Hosted Entrypoint
#
# Downloads ECMWF weather model data directly from ECMWF open-data servers
# (data.ecmwf.int) and starts the Open-Meteo API server. Data is refreshed
# periodically in the background.
#
# Both deterministic (ifs025) and probabilistic ensemble (ifs025_ensemble)
# models are supported — no API key required, ECMWF open data is freely
# available.
#
# Environment variables:
#   OPEN_METEO_SYNC_DOMAINS   Comma-separated list of ECMWF domains to download.
#                             Default: ifs025,ifs025_ensemble
#                               ifs025            — deterministic IFS 0.25° (required for /v1/forecast)
#                               ifs025_ensemble   — probabilistic IFS 0.25° 50-member (required for /v1/ensemble)
#                               ifs04_ensemble    — probabilistic IFS 0.4° (smaller, but runeflow uses ecmwf_ifs025)
#                               aifs025           — ECMWF AI/ML deterministic model
#   OPEN_METEO_SYNC_INTERVAL  Seconds between refresh cycles. Default: 21600 (6 hours)
#                             ECMWF runs at 00, 06, 12, 18 UTC so 6 hour intervals
#                             ensures data stays current.
#   OPEN_METEO_CONCURRENT     Number of concurrent download workers. Default: 4
#
# Disk usage (approximate per domain, per run):
#   ifs025            ~500 MB
#   ifs025_ensemble   ~5–15 GB (50 ensemble members, all surface variables)
#   ifs04_ensemble    ~2–5 GB
#
# Note: The first run will take several minutes (especially for ensemble data).
# The API server starts immediately but will return empty data until at least
# one domain has finished downloading. runeflow retries on failure so it will
# recover automatically once data is available.
#
# Usage (docker compose):
#   docker compose --profile open-meteo up -d

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
OPEN_METEO_SYNC_DOMAINS="${OPEN_METEO_SYNC_DOMAINS:-ifs025,ifs025_ensemble}"
OPEN_METEO_SYNC_INTERVAL="${OPEN_METEO_SYNC_INTERVAL:-21600}"
OPEN_METEO_CONCURRENT="${OPEN_METEO_CONCURRENT:-4}"
DATA_DIR="/app/data"
LOG_DIR="/app/logs"
BIN="/app/openmeteo-api"

mkdir -p "$LOG_DIR"

# ---------------------------------------------------------------------------
# Banner
# ---------------------------------------------------------------------------
echo "==================================="
echo " Open-Meteo Self-Hosted API Server"
echo "==================================="
echo "  Domains:      $OPEN_METEO_SYNC_DOMAINS"
echo "  Sync interval: ${OPEN_METEO_SYNC_INTERVAL}s ($(( OPEN_METEO_SYNC_INTERVAL / 3600 ))h)"
echo "  Concurrent:   $OPEN_METEO_CONCURRENT workers"
echo "  Data dir:     $DATA_DIR"
echo "  Binary:       $BIN"
echo "==================================="

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
log() {
    echo "[$(date -u '+%Y-%m-%d %H:%M:%S UTC')] $*"
}

download_all_domains() {
    local label="${1:-sync}"
    IFS=',' read -ra DOMAINS <<< "$OPEN_METEO_SYNC_DOMAINS"
    local failed=0
    for domain in "${DOMAINS[@]}"; do
        domain="${domain// /}"  # strip any accidental spaces
        [ -z "$domain" ] && continue
        log "[$label] Downloading domain: $domain --concurrent $OPEN_METEO_CONCURRENT"
        local t_start=$SECONDS
        if "$BIN" download-ecmwf --domain "$domain" --concurrent "$OPEN_METEO_CONCURRENT" \
               >> "$LOG_DIR/download-${domain}.log" 2>&1; then
            local elapsed=$(( SECONDS - t_start ))
            log "[$label] ✓ $domain done in ${elapsed}s"
        else
            log "[$label] ✗ $domain FAILED (see $LOG_DIR/download-${domain}.log)"
            failed=1
        fi
    done
    return $failed
}

# ---------------------------------------------------------------------------
# Initial data download (before starting the API server so healthcheck waits)
# ---------------------------------------------------------------------------
log "Starting initial data download — this may take several minutes..."
log "API server will start (with empty data) while download is in progress."

# Start API server immediately so the container appears healthy quickly.
# Data-less responses are expected at startup; runeflow will retry.
log "Starting API server in background..."
"$BIN" serve >> "$LOG_DIR/serve.log" 2>&1 &
API_PID=$!

# Give the server a moment to bind
sleep 2

# Now run the initial download in the background so the container doesn't
# block waiting for potentially multi-GB ensemble data on startup.
(
    if download_all_domains "startup"; then
        log "[startup] All domains downloaded successfully."
    else
        log "[startup] One or more domains failed. Will retry at next interval."
    fi
) &
INIT_DOWNLOAD_PID=$!

# ---------------------------------------------------------------------------
# Background refresh loop
# ---------------------------------------------------------------------------
(
    while true; do
        log "[refresh] Sleeping ${OPEN_METEO_SYNC_INTERVAL}s until next sync..."
        sleep "$OPEN_METEO_SYNC_INTERVAL"
        log "[refresh] Starting periodic refresh..."
        if download_all_domains "refresh"; then
            log "[refresh] All domains refreshed."
        else
            log "[refresh] One or more domains failed. Will retry next interval."
        fi
    done
) &
REFRESH_PID=$!

# ---------------------------------------------------------------------------
# Shutdown handler — clean up child processes on SIGTERM/SIGINT
# ---------------------------------------------------------------------------
cleanup() {
    log "Shutting down..."
    kill "$REFRESH_PID" 2>/dev/null || true
    kill "$INIT_DOWNLOAD_PID" 2>/dev/null || true
    kill "$API_PID" 2>/dev/null || true
    wait "$API_PID" 2>/dev/null || true
    log "Shutdown complete."
    exit 0
}
trap 'cleanup' TERM INT

# ---------------------------------------------------------------------------
# Tail logs and wait for API server (foreground)
# ---------------------------------------------------------------------------
log "API server running (PID $API_PID). Tailing logs..."
tail -F "$LOG_DIR/serve.log" &
TAIL_PID=$!

wait "$API_PID"
kill "$TAIL_PID" 2>/dev/null || true

log "API server exited. Shutting down refresh loop..."
kill "$REFRESH_PID" 2>/dev/null || true
kill "$INIT_DOWNLOAD_PID" 2>/dev/null || true
