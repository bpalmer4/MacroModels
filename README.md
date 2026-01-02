# MacroModels

Australian macroeconomic modelling. Includes both Bayesian state-space estimation (PyMC) and deterministic growth accounting methods.

## Overview

Three approaches to estimating Australia's supply side:

- **NAIRU + Output Gap**: Bayesian state-space model jointly estimating the natural rate of unemployment and potential output — see [`MODEL_EXPLAINED.md`](src/models/nairu/MODEL_EXPLAINED.md)
- **Cobb-Douglas MFP**: Deterministic growth accounting decomposing output into capital, labour, and productivity
- **DSGE**: Dynamic stochastic general equilibrium model *(in development)*

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

# Re-run analysis only (uses saved trace, ~1 min)
./run-nairu-stage2.sh -v

# Or via Python directly
uv run python -m src.models.nairu.model -v
uv run python -m src.models.nairu.stage2 -v
```

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
| `model_outputs/` | Saved traces (`.nc`) and observations (`.pkl`) |
| `charts/nairu_output_gap/` | NAIRU, output gap, Phillips curves, Taylor rule, decompositions |
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
├── utilities/          # Shared utilities
└── models/
    ├── nairu/          # NAIRU + Output Gap model
    ├── cobb_douglas/   # Cobb-Douglas MFP decomposition
    └── dsge/           # DSGE model (in development)
```
