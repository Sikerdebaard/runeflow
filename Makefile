# ─── runeflow Makefile ─────────────────────────────────────────────────────
# SPDX-License-Identifier: AGPL-3.0-or-later

SHELL       := /bin/bash
PYTHON      := python3
UV          := uv
SRC_DIR     := src/runeflow
TEST_DIR    := tests
IMAGE       := runeflow
TAG         := latest

HEADER_L1   := \# SPDX-License-Identifier: AGPL-3.0-or-later
HEADER_L2   := \# Copyright (C) 2024-2026 Thomas Phil — runeflow
HEADER_L3   := \# See LICENSE and COMMERCIAL-LICENSE.md for licensing details.

# ─── Development ──────────────────────────────────────────────────────────────

.PHONY: install
install: ## Install all deps (incl. dev) via uv
	$(UV) sync --all-extras

.PHONY: sync
sync: install ## Alias for install

# ─── Quality ──────────────────────────────────────────────────────────────────

.PHONY: lint
lint: ## Run ruff linter
	$(UV) run ruff check $(SRC_DIR) $(TEST_DIR)

.PHONY: lint-fix
lint-fix: ## Run ruff linter with auto-fix
	$(UV) run ruff check --fix $(SRC_DIR) $(TEST_DIR)

.PHONY: format
format: ## Run ruff formatter
	$(UV) run ruff format $(SRC_DIR) $(TEST_DIR)

.PHONY: format-check
format-check: ## Check formatting without changing files
	$(UV) run ruff format --check $(SRC_DIR) $(TEST_DIR)

.PHONY: typecheck
typecheck: ## Run pyright / mypy (if installed)
	$(UV) run pyright $(SRC_DIR) || $(UV) run mypy $(SRC_DIR) || echo "No type checker found"

# ─── Testing ──────────────────────────────────────────────────────────────────

.PHONY: test
test: ## Run pytest
	$(UV) run pytest $(TEST_DIR) -v

.PHONY: test-cov
test-cov: ## Run pytest with coverage
	$(UV) run pytest $(TEST_DIR) -v --cov=$(SRC_DIR) --cov-report=term-missing --cov-report=html

.PHONY: test-fast
test-fast: ## Run pytest (no slow/integration markers)
	$(UV) run pytest $(TEST_DIR) -v -m "not slow and not integration"

# ─── Build ────────────────────────────────────────────────────────────────────

.PHONY: build
build: ## Build sdist + wheel
	$(UV) build

.PHONY: clean
clean: ## Remove build artifacts
	rm -rf dist/ build/ *.egg-info src/*.egg-info .pytest_cache .ruff_cache htmlcov
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true

# ─── Docker ───────────────────────────────────────────────────────────────────

.PHONY: docker-build
docker-build: ## Build Docker image
	docker build -t $(IMAGE):$(TAG) .

.PHONY: docker-run
docker-run: docker-build ## Build + run inference (requires .env with ENTSOE, NED)
	docker run --rm --env-file .env \
	  -e XDG_CACHE_HOME=/cache \
	  -v runeflow-cache:/cache \
	  --entrypoint runeflow \
	  $(IMAGE):$(TAG) inference --zone NL

.PHONY: docker-plot
docker-plot: docker-build ## Build + generate plot
	docker run --rm --env-file .env \
	  -e XDG_CACHE_HOME=/cache \
	  -v runeflow-cache:/cache \
	  -v /tmp/runeflow_out:/out \
	  --entrypoint runeflow \
	  $(IMAGE):$(TAG) plot-uncertainty --zone NL --provider zonneplan --output /out/forecast-nl-zonneplan.png

# ─── Dev Container (local development) ───────────────────────────────────────
#
# A single Docker container for iterating on the full pipeline locally.
# The source tree (src/) is bind-mounted into the container so code changes
# take effect immediately — no image rebuild required.
# A Python HTTP server is provided for previewing the generated site.
#
# Quickstart:
#   1. Copy and fill in your API keys (if not already done):
#        cp .env .env.example   # or edit the existing .env
#   2. Build the image once:
#        make dev-build
#   3. Run the full pipeline and open the site:
#        make dev               # pipeline + serve on http://localhost:7071
#
# Individual steps (useful while iterating on model/site changes):
#   make dev-update       step 1/7 — fetch/update historical data
#   make dev-train        step 2/7 — train model
#   make dev-warmup       step 3/7 — warm up feature cache
#   make dev-inference    step 4/7 — run inference (forecast)
#   make dev-export       step 5/7 — export tariff JSON
#   make dev-plot         step 6/7 — generate uncertainty chart
#   make dev-build-site   step 7/7 — build static site
#
# After changing a template or asset, just re-run:
#   make dev-build-site && make dev-serve
#
# Override the zone on any target:
#   make dev-inference ZONE=DE_LU

DEV_IMAGE     := runeflow:dev
DEV_PORT      := 7071
DEV_CACHE_VOL := runeflow-cache

# Overridable per invocation: make dev-update ZONE=DE_LU
ZONE                  ?= NL
TARIFF_PRICE_PROVIDER ?= wholesale

# Shared docker run flags for all individual pipeline step targets.
# Secrets come from .env; zone/provider are Make variables.
_DEV_FLAGS = --rm \
  --env-file .env \
  -e XDG_CACHE_HOME=/app/.cache \
  -v $(CURDIR)/src:/app/src \
  -v $(CURDIR)/outputs:/outputs \
  -v $(DEV_CACHE_VOL):/app/.cache

.PHONY: dev-build
dev-build: ## [dev] Build the development container image
	docker build -f docker/Dockerfile.dev -t $(DEV_IMAGE) .

.PHONY: dev-update
dev-update: ## [dev] Step 1/7 — fetch/update historical data
	docker run $(_DEV_FLAGS) --entrypoint runeflow $(DEV_IMAGE) \
	  update-data --zone $(ZONE)

.PHONY: dev-train
dev-train: ## [dev] Step 2/7 — train model
	docker run $(_DEV_FLAGS) --entrypoint runeflow $(DEV_IMAGE) \
	  train --zone $(ZONE)

.PHONY: dev-warmup
dev-warmup: ## [dev] Step 3/7 — warm up feature cache
	docker run $(_DEV_FLAGS) --entrypoint runeflow $(DEV_IMAGE) \
	  warmup-cache --zone $(ZONE)

.PHONY: dev-inference
dev-inference: ## [dev] Step 4/7 — run inference (produce forecast)
	docker run $(_DEV_FLAGS) --entrypoint runeflow $(DEV_IMAGE) \
	  inference --zone $(ZONE)

.PHONY: dev-export
dev-export: ## [dev] Step 5/7 — export tariff JSON
	docker run $(_DEV_FLAGS) --entrypoint runeflow $(DEV_IMAGE) \
	  export-tariffs \
	  --zone $(ZONE) \
	  --provider $(TARIFF_PRICE_PROVIDER) \
	  --output /outputs/tariffs.json

.PHONY: dev-plot
dev-plot: ## [dev] Step 6/7 — generate uncertainty chart
	docker run $(_DEV_FLAGS) --entrypoint runeflow $(DEV_IMAGE) \
	  plot-uncertainty \
	  --zone $(ZONE) \
	  --output /outputs/runeflow_uncertainty_$(ZONE).png \
	  || echo "WARNING: chart generation failed (non-critical)"

.PHONY: dev-build-site
dev-build-site: ## [dev] Step 7/7 — build static site from cached forecast
	docker run $(_DEV_FLAGS) --entrypoint runeflow $(DEV_IMAGE) \
	  build-site \
	  --zones $(ZONE) \
	  --output /outputs/site

.PHONY: dev-pipeline
dev-pipeline: ## [dev] Run all 7 pipeline steps from scratch (no HTTP server)
	docker run $(_DEV_FLAGS) \
	  --entrypoint /entrypoint-oneshot.sh \
	  $(DEV_IMAGE)

.PHONY: dev-serve
dev-serve: ## [dev] Serve outputs/site on http://localhost:7071 (blocking)
	@echo "→  http://localhost:$(DEV_PORT)  — Ctrl-C to stop"
	docker run --rm \
	  -v $(CURDIR)/outputs/site:/site:ro \
	  -p $(DEV_PORT):8000 \
	  --entrypoint python \
	  $(DEV_IMAGE) -m http.server 8000 --directory /site

.PHONY: dev
dev: ## [dev] Full pipeline + serve — all-in-one (http://localhost:7071)
	docker run $(_DEV_FLAGS) \
	  -e DEV_PORT=8000 \
	  -p $(DEV_PORT):8000 \
	  $(DEV_IMAGE)

.PHONY: dev-shell
dev-shell: ## [dev] Open an interactive shell in the dev container
	docker run $(_DEV_FLAGS) -it \
	  --entrypoint bash \
	  $(DEV_IMAGE)

# ─── License headers ─────────────────────────────────────────────────────────

.PHONY: license-headers
license-headers: ## Add/update SPDX license headers on all .py source files
	@echo "Updating license headers…"
	@find src tests -name '*.py' | while read f; do \
	  if ! head -1 "$$f" | grep -q 'SPDX-License-Identifier'; then \
	    tmp=$$(mktemp); \
	    { echo '$(HEADER_L1)'; echo '$(HEADER_L2)'; echo '$(HEADER_L3)'; echo ''; cat "$$f"; } > "$$tmp" && mv "$$tmp" "$$f"; \
	    echo "  + $$f"; \
	  fi; \
	done
	@echo "Done."

.PHONY: license-check
license-check: ## Verify all .py files have SPDX headers
	@rc=0; \
	for f in $$(find src tests -name '*.py'); do \
	  if ! head -1 "$$f" | grep -q 'SPDX-License-Identifier'; then \
	    echo "MISSING: $$f"; rc=1; \
	  fi; \
	done; \
	if [ "$$rc" = "1" ]; then echo "FAIL: some files lack headers"; exit 1; \
	else echo "OK: all files have SPDX headers"; fi

# ─── CI convenience ──────────────────────────────────────────────────────────

.PHONY: ci
ci: license-check lint format-check test ## Run all CI checks

.PHONY: all
all: install lint format test build ## Full local workflow

# ─── Help ─────────────────────────────────────────────────────────────────────

.PHONY: help
help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
	  awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'

.DEFAULT_GOAL := help
