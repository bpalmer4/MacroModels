# MacroModels

Australian macroeconomic modelling. Includes both Bayesian state-space estimation (PyMC) and deterministic growth accounting methods.

## Overview

Three approaches to estimating Australia's supply side:

- **NAIRU + Output Gap**: Bayesian state-space model jointly estimating the natural rate of unemployment and potential output — see [`MODEL_NOTES.md`](src/models/nairu/MODEL_NOTES.md)
- **Inflation Expectations**: Bayesian signal extraction model estimating latent expectations from surveys and market data — see [`MODEL_NOTES.md`](src/models/expectations/MODEL_NOTES.md)
- **Cobb-Douglas MFP**: Deterministic growth accounting decomposing output into capital, labour, and productivity
- **DSGE**: Dynamic stochastic general equilibrium model *(in development)*

The NAIRU model uses expectations output as an input. Run the expectations model first if updating from scratch. The NAIRU model derives r\* deterministically from the Cobb-Douglas production function (r\* ≈ potential growth).

## Quickstart

```bash
# Install dependencies
uv sync

# Run the NAIRU + Output Gap model (~3 min)
./run-nairu.sh -v
```

## Running the Models

### NAIRU + Output Gap (Bayesian)

```bash
# Full estimation: data → sample → analyse → charts (~3 min)
./run-nairu.sh -v

# Re-run analysis and scenario forecasts (uses saved trace)
./run-nairu-stage2.sh -v

# Scenario analysis only (deterministic + Monte Carlo forward sampling)
./run-nairu-stage3.sh

# Model variants: simple (core equations) or complex (all features)
./run-nairu.sh -v --variant simple
./run-nairu.sh -v --variant complex
./run-nairu.sh -v --variant both      # Run both and generate comparison chart

# Or via Python directly
uv run python -m src.models.nairu.model -v          # Stage 1: estimation
uv run python -m src.models.nairu.stage2 -v          # Stage 2: analysis
uv run python -m src.models.nairu.stage3             # Stage 3a: deterministic scenarios
uv run python -m src.models.nairu.stage3_forward_sampling  # Stage 3b: Monte Carlo forecasts
```

### Inflation Expectations (Bayesian)

```bash
# Run all four expectation models (~10 min)
./run-expectations.sh

# Or via Python directly
uv run python -m src.models.expectations.model

# Run single model (target, unanchored, short, or market)
uv run python -m src.models.expectations.stage1 --model target

# Generate diagnostics and plots only
uv run python -m src.models.expectations.stage2
```

Four models are estimated:
- **Target Anchored**: Full model with 2.5% anchor post-1998
- **Unanchored**: Same as Target but no anchor — tests de-anchoring
- **Short Run (1yr)**: Market economists 1-year ahead only
- **Long Run (10yr)**: Breakeven inflation only

The NAIRU model uses a **spliced series**: Long Run through 1991Q4 (smooth disinflation decline), then Target Anchored from 1992Q1 onwards.

### Cobb-Douglas Productivity Decomposition

```bash
# Deterministic growth accounting (~30 sec)
./run-cd.sh -v

# Or via Python
uv run python -m src.models.cobb_douglas.model -v
```

### Outputs

| Directory | Contents |
|-----------|----------|
| `model_outputs/` | NAIRU saved traces (`.nc`) and observations (`.pkl`) for multiple model variants |
| `output/expectations/` | Expectations traces (`.nc`), HDI estimates (`.parquet`, `.csv`), metadata (`.pkl`) |
| `charts/nairu_output_gap/` | NAIRU, output gap, Phillips curves, equations, decompositions |
| `charts/expectations/` | Expectations comparison, diagnostics, model fits |
| `charts/cobb_douglas/` | MFP trends, productivity growth, potential output |

### Maintenance

```bash
# Upgrade all dependencies
./uv-upgrade.sh
```

## Project Structure

```
src/
├── data/               # Data fetching (ABS, RBA) and preparation
├── utilities/          # Shared utilities (rate conversion, etc.)
└── models/
    ├── common/         # Shared model utilities (diagnostics, extraction, timeseries)
    ├── expectations/   # Inflation expectations signal extraction
    ├── nairu/          # NAIRU + Output Gap model (stages 1–3)
    │   └── analysis/   # Plotting and diagnostics modules
    ├── cobb_douglas/   # Cobb-Douglas MFP decomposition
    └── dsge/           # DSGE model (in development)
```
