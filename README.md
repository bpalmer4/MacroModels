# MacroModels

Australian macroeconomic modelling. Includes both Bayesian state-space estimation (PyMC) and deterministic growth accounting methods.

## Models

- **NAIRU + Output Gap** (`src/models/nairu_output_gap.py`): Bayesian state-space model jointly estimating NAIRU and potential output using PyMC
- **Cobb-Douglas MFP Decomposition** (`src/models/cobb_douglas.py`): Deterministic growth accounting using the Solow residual

## Architecture

```
src/
├── data/           # Data fetching and preparation
├── equations/      # PyMC model building blocks (Bayesian models only)
├── models/         # Runnable model scripts
└── analysis/       # Post-sampling analysis and plotting
```

### Data Layer (`src/data/`)

Fetches and prepares data from Australian statistical sources:

- **`abs_loader.py`**: Load ABS time series using `readabs`. Returns `DataSeries` objects with metadata.
- **`rba_loader.py`**: Load RBA data (cash rate, inflation expectations).
- **`series_specs.py`**: Pre-defined `ReqsTuple` specifications for common series (GDP, unemployment, CPI, etc.).
- **`henderson.py`**: Henderson moving average for trend extraction.
- **`transforms.py`**: Growth rates, lags, and other transformations.

```python
from src.data import load_series, get_abs_data
from src.data.series_specs import GDP_CVM, UNEMPLOYMENT_RATE

gdp = load_series(GDP_CVM)
print(gdp.data.tail())
```

### Equations Layer (`src/equations/`)

Reusable PyMC equation functions for Bayesian models. Each function adds variables to a PyMC model context:

- **`state_space.py`**: `nairu_equation()` - Gaussian random walk for NAIRU
- **`production.py`**: `potential_output_equation()` - Cobb-Douglas with time-varying drift
- **`phillips.py`**: `price_inflation_equation()`, `wage_growth_equation()`
- **`okun.py`**: `okun_law_equation()` - Links output gap to unemployment
- **`is_curve.py`**: `is_equation()` - Links output gap to interest rates

```python
from src.equations import nairu_equation, potential_output_equation

model = pm.Model()
nairu = nairu_equation(obs, model)
potential = potential_output_equation(obs, model)
```

### Analysis Layer (`src/analysis/`)

Post-sampling utilities for Bayesian models:

- **`extraction.py`**: Extract posteriors from ArviZ traces (`get_vector_var`, `get_scalar_var`)
- **`diagnostics.py`**: MCMC convergence checks (R-hat, ESS, divergences)
- **`plotting.py`**: Scalar/coefficient visualisation (KDE plots, bar charts)
- **`timeseries.py`**: Time series analysis and plotting
  - Compute derived series: `compute_nairu_stats()`, `compute_potential_stats()`, `compute_taylor_rule()`
  - Plotting functions return `Axes` objects for composition

```python
from src.analysis import compute_nairu_stats, plot_nairu

stats = compute_nairu_stats(nairu_posterior, observed_u)
ax = plot_nairu(stats)
```

### Models Layer (`src/models/`)

Runnable model scripts that orchestrate data loading, model building, sampling/computation, and output:

- **`nairu_output_gap.py`**: Bayesian estimation using PyMC
- **`cobb_douglas.py`**: Deterministic growth accounting (no sampling)

```bash
# Run from command line
python -m src.models.nairu_output_gap
python -m src.models.cobb_douglas
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
| 6202.0 | Labour Force | Unemployment Rate, Participation Rate, Monthly Hours Worked |
| 6401.0 | Consumer Price Index | Trimmed Mean (quarterly & annual), All Groups |
| 6457.0 | International Trade Prices | Import Price Index |
| 1364.0.15.003 | Modellers Database | Capital Stock (CVM), Labour Force, Employed, Unemployed |
| 5260.0.55.002 | Industry MFP | Multi-Factor Productivity (Hours Worked basis) |

### RBA (Reserve Bank of Australia)

- **Official Cash Rate (OCR)**: Target cash rate from 1990
- **Inflation Expectations (PIE_RBAQ)**: RBA survey-based expectations series

### External

- **NY Fed GSCPI**: Global Supply Chain Pressure Index (for COVID supply shock)

### Local Data Files (`data/`)

| File | Description |
|------|-------------|
| `Qrtly-CPI-Time-series-spreadsheets-all.zip` | Historical quarterly CPI (until ABS updates in Jan 2026) |
| `PIE_RBAQ.CSV` | RBA inflation expectations (1970Q1-2023Q1) |
| `interbank_overnight_rate_historical.parquet` | Pre-1990 interbank overnight rate |
| `gscpi_data.xls` | NY Fed Global Supply Chain Pressure Index |
