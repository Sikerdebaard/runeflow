#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2024-2026 Thomas Phil — runeflow
# See LICENSE and COMMERCIAL-LICENSE.md for licensing details.
#
# deploy-poll.sh — Pull latest Docker image from GHCR and redeploy if updated
#
# Run via the included systemd timer (recommended) or manually via cron.
# The timer fires once per night; if a new image was published to GHCR by the
# publish.yml GitHub Actions workflow, the stack is restarted automatically.
#
# One-time setup:
#   1. Install as a systemd timer (see scripts/systemd/):
#        sudo cp scripts/systemd/runeflow-deploy.{service,timer} /etc/systemd/system/
#        sudo systemctl daemon-reload
#        sudo systemctl enable --now runeflow-deploy.timer
#
#   2. For private repos (GHCR images), log in to GHCR once on bragi:
#        docker login ghcr.io -u YOUR_GITHUB_USERNAME --password-stdin <<< "$GHCR_PAT"
#
#   3. Set RUNEFLOW_IMAGE in .env:
#        RUNEFLOW_IMAGE=ghcr.io/YOUR_ORG/runeflow:latest
#
# Environment (read from .env automatically):
#   RUNEFLOW_IMAGE  — full GHCR image reference  (required)
#   DEPLOY_PATH     — path to the repo checkout   (default: /opt/runeflow)
#   LOG_FILE        — append logs here if set      (default: journald / stdout)

set -euo pipefail

DEPLOY_PATH="${DEPLOY_PATH:-/opt/runeflow}"
LOCKFILE="/tmp/runeflow-deploy.lock"

ts()  { date -u +"%Y-%m-%dT%H:%M:%SZ"; }
log() { echo "[$(ts)] runeflow-deploy: $*"; }
die() { log "ERROR: $*" >&2; exit 1; }

# Optional file logging (journald already captures stdout for systemd services).
if [[ -n "${LOG_FILE:-}" ]]; then
  mkdir -p "$(dirname "$LOG_FILE")"
  exec >> "$LOG_FILE" 2>&1
fi

# ── Lock — prevent overlapping runs ──────────────────────────────────────────
exec 9>"$LOCKFILE"
flock -n 9 || { log "Another deploy is already running — skipping."; exit 0; }

# ── Load .env ─────────────────────────────────────────────────────────────────
[[ -d "$DEPLOY_PATH" ]] || die "$DEPLOY_PATH does not exist."
cd "$DEPLOY_PATH"

if [[ -f .env ]]; then
  # shellcheck source=/dev/null
  set -a; source .env; set +a
fi

IMAGE="${RUNEFLOW_IMAGE:?RUNEFLOW_IMAGE must be set in .env (e.g. ghcr.io/YOUR_ORG/runeflow:latest)}"

# ── Pull image ────────────────────────────────────────────────────────────────
log "Checking for updates: $IMAGE"
PULL_OUTPUT=$(docker pull "$IMAGE" 2>&1)
log "$PULL_OUTPUT"

if echo "$PULL_OUTPUT" | grep -q "Status: Image is up to date"; then
  log "Image is up to date — no deploy needed."
  exit 0
fi

log "New image available. Restarting services…"

# ── Restart stack ─────────────────────────────────────────────────────────────
docker compose --profile tunnel up -d --remove-orphans

log "Pruning old images…"
docker image prune -f

# ── Verify ────────────────────────────────────────────────────────────────────
log "Waiting for services to stabilise…"
sleep 15

log "Service status:"
docker compose --profile tunnel ps

# Health check via docker exec — nginx has no host port exposed.
if docker exec runeflow-server wget -q --spider http://localhost:8080/ 2>/dev/null; then
  log "Health check passed."
else
  log "WARNING: Health check failed after deploy."
  exit 1
fi

log "Deploy complete."
