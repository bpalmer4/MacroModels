# MacroModels

Australian macroeconomic modelling. Includes both Bayesian state-space estimation (PyMC) and deterministic growth accounting methods.

## Overview

### Supply side and structural estimation

- **NAIRU + Output Gap**: Bayesian state-space model jointly estimating the natural rate of unemployment and potential output — see [`MODEL_NOTES.md`](src/models/nairu/MODEL_NOTES.md)
- **Inflation Expectations**: Bayesian signal extraction model estimating latent expectations from surveys and market data — see [`MODEL_NOTES.md`](src/models/expectations/MODEL_NOTES.md)
- **Cobb-Douglas MFP**: Deterministic growth accounting decomposing output into capital, labour, and productivity
- **HLW r\***: Bayesian (PyMC) Holston-Laubach-Williams model estimating the natural rate of interest for Australia — see [`MODEL_NOTES.md`](src/models/rstar_hlw/MODEL_NOTES.md)
- **DSGE**: Dynamic stochastic general equilibrium model *(in development)*

**Run order:** the NAIRU model and the HLW r\* model both read the expectations model's saved output (`output/expectations/`) as an input — **run the expectations model first** whenever updating after new data. The NAIRU model derives r\* deterministically from the Cobb-Douglas production function (r\* ≈ potential growth).

### GDP nowcasting

Four complementary approaches to nowcasting the next unpublished quarterly GDP growth:

- **Bridge equations**: High-frequency monthly indicators completed to quarters via SARIMA, combined by inverse-MSE weights — see [`MODEL_NOTES.md`](src/models/gdp_nowcast_bridge/MODEL_NOTES.md)
- **Dynamic Factor Model**: Common factors extracted from a mixed-frequency panel via Kalman filter, ragged edge handled natively — see [`MODEL_NOTES.md`](src/models/gdp_nowcast_dfm/MODEL_NOTES.md)
- **Bayesian VAR**: Minnesota-prior VAR conditioned on contemporaneous indicators, T-0 only *(comparator, not for operational point forecasts)* — see [`MODEL_NOTES.md`](src/models/gdp_nowcast_bvar/MODEL_NOTES.md)
- **Components (expenditure identity)**: Structural accounting build-up summing component contributions to growth, T-0 only — see [`MODEL_NOTES.md`](src/models/gdp_nowcast_components/MODEL_NOTES.md)

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
# Default run: the policy-relevant NAIRU — simple_excess_regime variant
# (excess-expectations term + regime-switching Phillips slopes), expectations
# folding to the 2.5% target over 1993–1998
./run-nairu.sh -v

# Re-run validate/analyse/forecast from the saved trace (no re-estimation)
./run-nairu.sh -v --skip-estimate

# Estimation only (skip validate/analyse/forecast)
./run-nairu.sh -v --estimate-only

# Other variants: simple (core equations), simple_excess, simple_regime,
# complex (all features); other anchors via --anchor
./run-nairu.sh -v --variant simple_excess
./run-nairu.sh -v --variant complex --anchor unanchored

# Multiple variants in one run also produces comparison charts
./run-nairu.sh -v --variant simple_excess simple_excess_regime

# Or via Python directly
uv run python -m src.models.nairu.run -v
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

### HLW r\* (Bayesian)

Resolution G (blend + hierarchical Beta) is the default and the standard specification to run. It is the end point of a sequence of specifications (A–H) built while diagnosing why canonical HLW fails to identify r\* on Australian data; the earlier resolutions are retained as diagnostic comparators — see the model notes for the full journey.

```bash
# Default: Resolution G (blend + hierarchical Beta)
./run-rstar-hlw.sh -v

# Alternative resolutions
./run-rstar-hlw.sh --resolution C    # blend with fixed Beta(1,1) on alpha
./run-rstar-hlw.sh --resolution A    # canonical HLW: r* = g + z

# Estimate only / re-analyse saved trace
./run-rstar-hlw.sh --estimate-only
./run-rstar-hlw.sh --skip-estimate

# Or via Python
uv run python -m src.models.rstar_hlw.run -v
```

The IS curve does not independently pin r\* in Australian data (the rate channel is too weak); each specification largely returns the structural assumption it imposes. The value is a diagnostic framework and an honest cross-resolution uncertainty band rather than a single point estimate — see the [`MODEL_NOTES.md`](src/models/rstar_hlw/MODEL_NOTES.md).

### GDP Nowcasting

**Timing:** don't run the nowcasts until about one month before the GDP release. Earlier in the cycle almost no indicators for the target quarter are published — the BVAR declines to nowcast at all, and the bridge/DFM intervals are mostly prior.

```bash
# Bridge equations (high-frequency monthly indicators)
./run-gdp-nowcast-bridge.sh

# Dynamic Factor Model
./run-gdp-nowcast-dfm.sh

# Bayesian VAR (T-0 only, comparator)
./run-gdp-nowcast-bvar.sh

# Components / expenditure identity (T-0 only)
./run-gdp-nowcast-components.sh

# Bridge model backtest
uv run python -m src.models.gdp_nowcast_bridge.backtest
```

The bridge and DFM are the workhorses for production nowcasts. The BVAR is a comparator (structurally over-volatile, not recommended for operational point forecasts). The components model is a structural accounting build-up complementary to the statistical nowcasts — at T-0 it measures rather than forecasts the most volatile contributors (inventories, net exports, government).

### Outputs

| Directory | Contents |
|-----------|----------|
| `model_outputs/` | NAIRU saved traces (`.nc`) and observations (`.pkl`) for multiple model variants; GDP nowcast outputs (`gdp_nowcast*/`) |
| `output/expectations/` | Expectations traces (`.nc`), HDI estimates (`.parquet`, `.csv`), metadata (`.pkl`) |
| `charts/nairu_*/` | NAIRU, output gap, Phillips curves, equations, decompositions (per variant) |
| `charts/expectations/` | Expectations comparison, diagnostics, model fits |
| `charts/cobb_douglas/` | MFP trends, productivity growth, potential output |
| `charts/rstar-hlw/` | r\* estimates, cross-resolution comparisons, diagnostics |
| `charts/GDP-Nowcast-*/` | GDP nowcast fan charts and decompositions (Bridge, DFM, BVAR, Components) |

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
    ├── common/                 # Shared model utilities (diagnostics, extraction, timeseries)
    ├── expectations/           # Inflation expectations signal extraction
    ├── nairu/                  # NAIRU + Output Gap model (estimate → validate → analyse → forecast)
    │   └── analysis/           # Plotting and diagnostics modules
    ├── cobb_douglas/           # Cobb-Douglas MFP decomposition
    ├── rstar_hlw/              # HLW Bayesian r* model (AU data)
    ├── gdp_nowcast_bridge/     # GDP nowcast — bridge equations
    ├── gdp_nowcast_dfm/        # GDP nowcast — Dynamic Factor Model
    ├── gdp_nowcast_bvar/       # GDP nowcast — Bayesian VAR (T-0 only)
    ├── gdp_nowcast_components/ # GDP nowcast — expenditure-identity components (T-0 only)
    └── dsge/                   # DSGE model (in development)
```
