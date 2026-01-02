# MacroModels

Australian macroeconomic modelling. Includes both Bayesian state-space estimation (PyMC) and deterministic growth accounting methods.

## Scope

**What this model is for:**
- Estimating latent NAIRU and output gap with uncertainty quantification
- Decomposing inflation into demand (unemployment gap) and supply (import prices, supply chains) components
- Understanding historical macro dynamics under the model's structural assumptions
- Scenario analysis conditional on the estimated relationships

**What this model is not for:**
- Causal policy evaluation (no structural identification of policy shocks)
- Exchange rate or inflation forecasting (UIP and Phillips curve are stylised, not predictive)
- "True" productivity measurement (MFP is derived, not directly observed)
- Detecting structural breaks beyond the pre-specified regimes

**Units and timing:** All growth rates are quarterly percentage changes unless labelled `Δ4` (year-ended/annual). Interest rates and unemployment are levels in percent.

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

## Models

- **NAIRU + Output Gap** (`src/models/nairu/`): Bayesian state-space model jointly estimating NAIRU and potential output using PyMC. Includes 12 equations linking latent states to observables.
- **Cobb-Douglas MFP Decomposition** (`src/models/cobb_douglas/`): Deterministic growth accounting using the Solow residual with HP-filtered trends and periodic re-anchoring.
- **DSGE** (`src/models/dsge/`): *In development.* Dynamic Stochastic General Equilibrium model for Australia.

## Architecture

```
src/
├── data/               # Shared data fetching and preparation
├── utilities/          # Shared utilities (rate conversion, etc.)
└── models/
    ├── nairu/          # NAIRU + Output Gap model
    │   ├── equations/  # PyMC equation blocks
    │   └── analysis/   # Post-sampling analysis and plotting
    ├── cobb_douglas/   # Cobb-Douglas MFP decomposition
    └── dsge/           # DSGE model (in development)
```

### Data Layer (`src/data/`)

Fetches and prepares data from Australian statistical sources. All loaders return `DataSeries` objects with metadata (source, units, description).

**Core loaders:**
- **`abs_loader.py`**: Load ABS time series using `readabs`. Returns `DataSeries` objects with metadata.
- **`rba_loader.py`**: Load RBA data (cash rate, inflation expectations, TWI, inflation anchor).
- **`dataseries.py`**: `DataSeries` dataclass containing series data and metadata.
- **`series_specs.py`**: Pre-defined `ReqsTuple` specifications for common series.

**Specialized loaders** (each returns prepared series):
- **`gdp.py`**: GDP (CVM, current prices), log GDP, GDP growth
- **`labour_force.py`**: Unemployment rate (level, lagged, change, speed-limit), participation rate, hours worked/growth, employment growth (level, lagged), labour force
- **`capital.py`**: Capital stock and capital growth
- **`mfp.py`**: Multi-factor productivity (annual, from ABS 5204)
- **`productivity.py`**: Derived productivity measures (labour productivity, MFP from wage data, real wage gap)
- **`inflation.py`**: CPI trimmed mean (quarterly and annual)
- **`cash_rate.py`**: Cash rate (monthly/quarterly, with historical splicing), r* calculation, real rate gap
- **`ulc.py`**: Unit labour cost growth (level, lagged)
- **`hourly_coe.py`**: Hourly compensation of employees growth (level, lagged, annual)
- **`awe.py`**: Average weekly earnings
- **`dfd_deflator.py`**: Domestic final demand deflator growth
- **`import_prices.py`**: Import price index and growth (level, lagged)
- **`twi.py`**: Trade-Weighted Index (level, changes quarterly/annual, lagged quarterly/annual, real TWI)
- **`energy.py`**: Oil and coal prices (USD and AUD, lagged annual changes)
- **`gov_spending.py`**: Government consumption, fiscal impulse (level, lagged)
- **`household.py`**: Household saving ratio
- **`gscpi.py`**: NY Fed Global Supply Chain Pressure Index (COVID-masked, lagged)
- **`net_exports.py`**: Net exports ratio and changes
- **`foreign_demand.py`**: Major trading partner GDP growth
- **`tot.py`**: Terms of trade changes

**Model data assembly:**
- **`observations.py`**: Builds the observation matrix for model estimation, collating all series from specialized loaders

**Utilities:**
- **`henderson.py`**: Henderson moving average for trend extraction (ABS method)
- **`transforms.py`**: Growth rates, lags, splicing, and rate conversions

```python
from src.data import load_series, get_unemployment_rate_qrtly
from src.data.series_specs import UNEMPLOYMENT_RATE

# Using series specs
ur = load_series(UNEMPLOYMENT_RATE)

# Using convenience functions
ur = get_unemployment_rate_qrtly()
print(ur.data.tail())
print(f"Units: {ur.units}, Source: {ur.source}")
```

### Equations Layer (`src/models/nairu/equations/`)

Reusable PyMC equation functions for the NAIRU model. Each function adds distributions to a PyMC model context and returns key latent variables.

| Equation | Function | Description |
|----------|----------|-------------|
| NAIRU | `nairu_equation()` | Gaussian random walk for natural unemployment rate |
| Potential Output | `potential_output_equation()` | Cobb-Douglas production function with time-varying MFP |
| Price Phillips | `price_inflation_equation()` | Anchor-augmented Phillips curve with regime-switching slopes |
| Wage Phillips (ULC) | `wage_growth_equation()` | Unit labour cost growth with regime-switching slopes |
| Wage Phillips (HCOE) | `hourly_coe_equation()` | Hourly compensation growth with regime-switching + MFP term |
| Okun's Law | `okun_law_equation()` | Links output gap to unemployment changes |
| IS Curve | `is_equation()` | Output gap persistence with real interest rate effects |
| Participation | `participation_equation()` | Discouraged worker effect on labour force participation |
| Employment | `employment_equation()` | Labour demand: output gap and real wage gap effects |
| Exchange Rate | `exchange_rate_equation()` | UIP-style TWI equation with interest rate differential |
| Import Prices | `import_price_equation()` | Exchange rate pass-through to import prices |
| Net Exports | `net_exports_equation()` | Output gap and TWI effects on trade balance |

```python
from src.models.nairu.equations import nairu_equation, potential_output_equation, price_inflation_equation

model = pm.Model()
nairu = nairu_equation(obs, model, constant={"nairu_innovation": 0.25})
potential = potential_output_equation(obs, model)
price_inflation_equation(obs, model, nairu)
```

### Analysis Layer (`src/models/nairu/analysis/`)

Post-sampling utilities for the NAIRU model:

**Diagnostics:**
- **`diagnostics.py`**: MCMC convergence checks (R-hat, ESS, divergences, BFMI, tree depth)
- **`extraction.py`**: Extract posteriors from ArviZ traces (`get_vector_var`, `get_scalar_var`)

**Plotting:**
- **`plot_posteriors_bar.py`**: Horizontal bar chart of coefficient posteriors with HDI
- **`plot_posteriors_kde.py`**: KDE plots of scalar posteriors
- **`plot_posterior_timeseries.py`**: Time series with credible intervals
- **`plot_nairu_unemployment.py`**: NAIRU and unemployment gap plots
- **`plot_nairu_output.py`**: Output gap, GDP vs potential, potential growth
- **`plot_nairu_rates.py`**: Taylor rule and equilibrium rates
- **`observations_plot.py`**: Grid plot of model observables

**Model validation:**
- **`posterior_predictive_checks.py`**: PPC plots comparing predicted vs observed
- **`residual_autocorrelation.py`**: Residual autocorrelation analysis

**Inflation decomposition:**
- **`inflation_decomposition.py`**: Decomposes inflation into demand (unemployment gap) and supply (import prices, GSCPI) components. Includes policy diagnosis.

```python
from src.models.nairu.analysis import (
    check_model_diagnostics,
    decompose_inflation,
    plot_posteriors_bar,
    plot_nairu,
)

check_model_diagnostics(trace)
plot_posteriors_bar(trace, model_name="My Model")
decomp = decompose_inflation(trace, obs, obs_index)
```

### Utilities (`src/utilities/`)

Shared utilities used across models:
- **`rate_conversion.py`**: Compound conversion between quarterly and annual rates

### Models Layer (`src/models/`)

Each model is self-contained with its own equations (if applicable), analysis, and utilities:

- **`nairu/`**: Full Bayesian estimation pipeline
  - `model.py`: Main entry point with `run_model()` and `NAIRUResults`
  - `stage1.py`: Data preparation, model building, sampling
  - `stage2.py`: Loading results, diagnostics, plotting
  - `base.py`: PyMC utilities (`SamplerConfig`, `sample_model()`)
  - `equations/`: PyMC equation blocks
  - `analysis/`: Post-sampling analysis and plotting

- **`cobb_douglas/`**: Deterministic growth accounting
  - `model.py`: `run_decomposition()` and `DecompositionResult`

```bash
# Run from command line
python -m src.models.nairu.model -v
python -m src.models.cobb_douglas.model -v
```

Charts are saved to `charts/<model-name>/`.

## Installation

```bash
uv sync
```

## Data Sources

### ABS (Australian Bureau of Statistics)

| Catalogue | Description | Series Used |
|-----------|-------------|-------------|
| 5206.0 | National Accounts | GDP (CVM), Hours Worked Index, Compensation of Employees |
| 5204.0 | Productivity | Multi-Factor Productivity (Hours Worked basis) |
| 6202.0 | Labour Force | Unemployment Rate, Participation Rate, Monthly Hours Worked |
| 6401.0 | Consumer Price Index | Trimmed Mean (quarterly & annual), All Groups |
| 6457.0 | International Trade Prices | Import Price Index |
| 1364.0.15.003 | Modellers Database | Capital Stock (CVM), Labour Force, Employed, Unemployed |

### RBA (Reserve Bank of Australia)

- **Official Cash Rate (OCR)**: Target cash rate from 1990
- **Inflation Expectations (PIE_RBAQ)**: RBA survey-based expectations series
- **Trade-Weighted Index (TWI)**: F11 exchange rate tables (historical + current)
- **Exchange Rates**: AUD/USD and other bilateral rates from F11

### External

- **NY Fed GSCPI**: Global Supply Chain Pressure Index (COVID supply shock proxy)
- **World Bank**: Oil prices (Brent crude, USD)

### Local Data Files (`input_data/`)

| File | Description |
|------|-------------|
| `Qrtly-CPI-Time-series-spreadsheets-all.zip` | Historical quarterly CPI (until ABS updates in Jan 2026) |
| `PIE_RBAQ.CSV` | RBA inflation expectations (1970Q1-2023Q1) |
| `interbank_overnight_rate_historical.parquet` | Pre-1990 interbank overnight rate |
| `gscpi_data.xls` | NY Fed Global Supply Chain Pressure Index |

## Model Documentation

For detailed documentation of model equations, parameters, priors, design decisions, and forecasting methodology, see **[`src/models/nairu/MODEL_EXPLAINED.md`](src/models/nairu/MODEL_EXPLAINED.md)**.

