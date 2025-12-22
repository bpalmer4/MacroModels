# MacroModels

Australian macroeconomic modelling. Includes both Bayesian state-space estimation (PyMC) and deterministic growth accounting methods.

## Models

- **NAIRU + Output Gap** (`src/models/nairu_output_gap.py`): Bayesian state-space model jointly estimating NAIRU and potential output using PyMC. Includes 9 equations linking latent states to observables.
- **Cobb-Douglas MFP Decomposition** (`src/models/cobb_douglas.py`): Deterministic growth accounting using the Solow residual with HP-filtered trends and periodic re-anchoring.

## Architecture

```
src/
├── data/           # Data fetching and preparation
├── equations/      # PyMC model building blocks (Bayesian models only)
├── models/         # Runnable model scripts
└── analysis/       # Post-sampling analysis and plotting
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
- **`labour_force.py`**: Unemployment rate, participation rate, hours worked, labour force
- **`capital.py`**: Capital stock and capital growth
- **`mfp.py`**: Multi-factor productivity (annual, from ABS 5204)
- **`inflation.py`**: CPI trimmed mean (quarterly and annual)
- **`cash_rate.py`**: Cash rate (monthly/quarterly, with historical splicing)
- **`ulc.py`**: Unit labour cost growth
- **`import_prices.py`**: Import price index and growth
- **`twi.py`**: Trade-Weighted Index (level, changes, real TWI)
- **`energy.py`**: Oil and coal prices (USD and AUD)
- **`gov_spending.py`**: Government consumption and fiscal impulse
- **`household.py`**: Household saving ratio
- **`gscpi.py`**: NY Fed Global Supply Chain Pressure Index
- **`tot.py`**: Terms of trade changes

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

### Equations Layer (`src/equations/`)

Reusable PyMC equation functions for Bayesian models. Each function adds distributions to a PyMC model context and returns key latent variables.

| Equation | Function | Description |
|----------|----------|-------------|
| NAIRU | `nairu_equation()` | Gaussian random walk for natural unemployment rate |
| Potential Output | `potential_output_equation()` | Cobb-Douglas production function with time-varying MFP |
| Phillips Curve | `price_inflation_equation()` | Anchor-augmented Phillips curve (inflation target transition) |
| Wage Phillips | `wage_growth_equation()` | Wage growth response to unemployment gap |
| Okun's Law | `okun_law_equation()` | Links output gap to unemployment changes |
| IS Curve | `is_equation()` | Output gap persistence with real interest rate effects |
| Participation | `participation_equation()` | Discouraged worker effect on labour force participation |
| Exchange Rate | `exchange_rate_equation()` | UIP-style TWI equation with interest rate differential |
| Import Prices | `import_price_equation()` | Exchange rate pass-through to import prices |

```python
from src.equations import nairu_equation, potential_output_equation, price_inflation_equation

model = pm.Model()
nairu = nairu_equation(obs, model, constant={"nairu_innovation": 0.25})
potential = potential_output_equation(obs, model)
price_inflation_equation(obs, model, nairu)
```

### Analysis Layer (`src/analysis/`)

Post-sampling utilities for Bayesian models:

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

**Utilities:**
- **`rate_conversion.py`**: Compound conversion between quarterly and annual rates

```python
from src.analysis import (
    check_model_diagnostics,
    decompose_inflation,
    plot_posteriors_bar,
    plot_nairu,
)

check_model_diagnostics(trace)
plot_posteriors_bar(trace, model_name="My Model")
decomp = decompose_inflation(trace, obs, obs_index)
```

### Models Layer (`src/models/`)

Runnable model scripts that orchestrate data loading, model building, sampling/computation, and output:

- **`nairu_output_gap.py`**: Full Bayesian estimation pipeline
  - `build_observations()`: Load and align all data
  - `build_model()`: Assemble PyMC model from equations
  - `run_model()`: End-to-end estimation
  - `NAIRUResults`: Container with posteriors and computed series

- **`cobb_douglas.py`**: Deterministic growth accounting
  - `run_decomposition()`: Full decomposition pipeline
  - `DecompositionResult`: Container with MFP, potential GDP, output gap

- **`base.py`**: Shared utilities
  - `SamplerConfig`: NUTS sampler configuration
  - `sample_model()`: Run PyMC sampling
  - `set_model_coefficients()`: Create priors from settings dict

```bash
# Run from command line
python -m src.models.nairu_output_gap -v
python -m src.models.cobb_douglas -v
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

### Local Data Files (`data/`)

| File | Description |
|------|-------------|
| `Qrtly-CPI-Time-series-spreadsheets-all.zip` | Historical quarterly CPI (until ABS updates in Jan 2026) |
| `PIE_RBAQ.CSV` | RBA inflation expectations (1970Q1-2023Q1) |
| `interbank_overnight_rate_historical.parquet` | Pre-1990 interbank overnight rate |
| `gscpi_data.xls` | NY Fed Global Supply Chain Pressure Index |

## Model Equations

### NAIRU + Output Gap Model

The joint model estimates NAIRU and potential output using 9 equations:

**State Equations:**
1. **NAIRU**: Random walk without drift
   - `NAIRU_t = NAIRU_{t-1} + ε_t`

2. **Potential Output**: Cobb-Douglas with stochastic drift
   - `g_Y* = α×g_K + (1-α)×g_L + g_MFP + ε_t`

**Observation Equations:**
3. **Phillips Curve**: Anchor-augmented (expectations → target transition 1993-1998)
   - `π_t = quarterly(π_anchor) + γ×u_gap + λ×Δρm + ξ×GSCPI² + ε`

4. **Wage Phillips**: Unit labour cost growth
   - `Δulc = α + γ×u_gap + λ×ΔU/U + ε`

5. **Okun's Law**: Output gap to unemployment change
   - `ΔU = β×output_gap + ε`

6. **IS Curve**: Output gap persistence with interest rate effects
   - `y_gap_t = ρ×y_gap_{t-1} - β×(r_{t-2} - r*) + γ×fiscal_impulse + ε`

7. **Participation**: Discouraged worker effect
   - `Δpr_t = β×(U_{t-1} - NAIRU_{t-1}) + ε`

8. **Exchange Rate**: UIP-style TWI equation
   - `Δe_t = ρ×Δe_{t-1} + β×r_gap_{t-1} + ε`

9. **Import Price Pass-Through**:
   - `Δ4ρm = β_pt×Δ4twi + β_oil×Δ4oil + ρ×Δ4ρm_{t-1} + ε`

### Cobb-Douglas Decomposition

Growth accounting using the Solow residual:
- `g_MFP = g_Y - α×g_K - (1-α)×g_L`

With HP-filtered trends and periodic re-anchoring at business cycle peaks.

## Key Parameters

| Parameter | Description | Prior/Value |
|-----------|-------------|-------------|
| α (alpha) | Capital share of income | N(0.30, 0.05) |
| γ_π (gamma_pi) | Phillips curve slope | N(-0.5, 0.3), expect negative |
| β_okun | Okun's law coefficient | N(-0.2, 0.15), expect negative |
| ρ_is | IS curve persistence | N(0.85, 0.1), expect 0-1 |
| β_is | Interest rate sensitivity | TN(0.2, 0.1, lower=0) |
| β_pr | Discouraged worker effect | TN(-0.05, 0.03, upper=0) |
| β_pt | Exchange rate pass-through | TN(-0.3, 0.15, upper=0) |

