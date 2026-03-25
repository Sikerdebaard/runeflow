<p align="center">
  <img src="assets/logo_tight_margins.png" alt="Runeflow" width="420">
</p>

<p align="center">
  <a href="https://github.com/Sikerdebaard/runeflow/actions/workflows/test.yml"><img alt="Tests" src="https://github.com/Sikerdebaard/runeflow/actions/workflows/test.yml/badge.svg"></a>
  <a href="https://www.gnu.org/licenses/agpl-3.0"><img alt="License: AGPL v3" src="https://img.shields.io/badge/License-AGPL_v3-blue.svg"></a>
  <a href="https://www.python.org/downloads/"><img alt="Python ≥3.11" src="https://img.shields.io/badge/python-%E2%89%A53.11-blue"></a>
  <a href="https://github.com/astral-sh/ruff"><img alt="Ruff" src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json"></a>
  <a href="https://github.com/Sikerdebaard/runeflow/actions/workflows/test.yml"><img alt="Coverage" src="https://img.shields.io/badge/coverage-checked%20in%20CI-green"></a>
</p>

# Runeflow

Electricity price forecasting for European day-ahead markets (ENTSO-E zones).

Runeflow trains an XGBoost ensemble on historical prices, weather, generation and
supplemental data, then produces a 9-day probabilistic forecast with
uncertainty bands derived from 51-member Open-Meteo weather ensembles.

## Supported zones

| Zone   | Region               | Tariff providers |
|--------|----------------------|------------------|
| `NL`   | Netherlands          | Zonneplan, Tibber, EasyEnergy, Greenchoice, Vattenfall, Eneco, Essent, ANWB, Leapp, Energie van Ons, wholesale |
| `DE_LU`| Germany-Luxembourg   | Tibber, aWATTar, Ostrom, wholesale |

## Quick start

```bash
# Install (requires Python ≥3.11 and uv)
uv sync --all-extras

# Set API keys
cp .env.example .env
# edit .env with your ENTSO-E and (optionally) NED keys

# Run the full pipeline for the Netherlands
runeflow update-data --zone NL
runeflow train       --zone NL
runeflow warmup-cache --zone NL
runeflow inference   --zone NL
```

## CLI commands

| Command | Description |
|---------|-------------|
| `runeflow list-markets` | List all registered market zones |
| `runeflow update-data --zone NL` | Download and cache prices, weather, generation and supplemental data |
| `runeflow train --zone NL` | Train the price-prediction ensemble |
| `runeflow warmup-cache --zone NL` | Build the feature warmup cache required for inference |
| `runeflow inference --zone NL` | Generate a 9-day price forecast |
| `runeflow export-tariffs --zone NL --provider vattenfall` | Export forecast as tariff JSON |
| `runeflow plot-uncertainty --zone NL` | Plot the uncertainty forecast band (3-panel chart) |

## Architecture

Hexagonal (ports-and-adapters) with dependency injection via
[inject](https://github.com/ivankorobkov/python-inject).

```
CLI (Typer)
 └─ Services
     ├─ InferenceService   — autoregressive 9-day forecast, 51-member ensemble
     ├─ TrainService        — train 3-model ensemble
     ├─ UpdateDataService   — fetch & cache external data
     ├─ WarmupService       — pre-compute feature cache
     ├─ PlotService         — 3-panel uncertainty chart
     └─ ExportTariffsService — tariff JSON for smart chargers
 └─ Ports (ABCs)
     ├─ PricePort           → EntsoePriceAdapter / EnergyZeroAdapter / FallbackAdapter
     ├─ WeatherPort         → CachingWeatherAdapter(OpenMeteoAdapter)
     ├─ GenerationPort      → EntsoeGenerationAdapter
     ├─ SupplementalDataPort→ NedAdapter
     └─ DataStore           → ParquetStore (~/.cache/runeflow)
```

### Models

| Model | Quantile | Purpose |
|-------|----------|---------|
| `XGBoostQuantileModel` | P1 / P50 / P99 | Central model with conformal calibration |
| `ExtremeHighModel` | α = 0.90 | Spike detection (top 12 % weighted) |
| `ExtremeLowModel` | α = 0.10 | Dip detection (bottom 12 % weighted) |

The **condition-gated** ensemble blends these based on market conditions
(evening peak, solar cliff, solar midday, night valley) and widens
confidence bands as the forecast horizon grows.

### Feature engineering

19 feature groups executed in dependency order — temporal, solar position,
solar power, holiday, price lag, price regime, spike momentum, temperature,
wind, precipitation, cloud, renewable pressure, residual load, cross-border,
duck curve, market structure, generation, spike risk, and interaction features.

### Data quality

A composite validator runs continuity, NaN, price-range (−500 – 4 000 €/MWh),
timezone, duplicates, row-count and staleness checks before training and
inference.

## Docker

```bash
# Build and run (requires .env with ENTSOE key)
docker compose up runeflow-export

# Intel-optimised variant (scikit-learn-intelex)
docker compose --profile intel up runeflow-export-intel

# Tariff JSON served on port 8080
docker compose up runeflow-server
```

The export container runs a full pipeline at **08:05** and **14:05** (Amsterdam
time) and watches for new day-ahead prices every 15 minutes between 12:00 and
16:00.

## Development

```bash
make install       # Install all dependencies
make lint          # Ruff linter
make format        # Ruff formatter
make test          # pytest
make test-cov      # pytest with coverage report
make clean         # Remove build artifacts
```

## Configuration

All settings are loaded from environment variables or a `.env` file.
See [.env.example](.env.example) for the full list.

| Variable | Default | Description |
|----------|---------|-------------|
| `ENTSOE` | — | ENTSO-E Transparency Platform API key |
| `NED` | — | NED API key (Netherlands supplemental data) |
| `ZONE` | `NL` | Target bidding zone |
| `CACHE_DIR` | `~/.cache/runeflow` | Local data cache (respects `XDG_CACHE_HOME`) |
| `LOG_LEVEL` | `INFO` | Logging level |

## License

Dual-licensed under [AGPL-3.0-or-later](LICENSE) and a commercial license.
See [COMMERCIAL-LICENSE.md](COMMERCIAL-LICENSE.md) for proprietary use.
