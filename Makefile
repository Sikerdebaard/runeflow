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
