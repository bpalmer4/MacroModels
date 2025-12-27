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

### Local Data Files (`data/`)

| File | Description |
|------|-------------|
| `Qrtly-CPI-Time-series-spreadsheets-all.zip` | Historical quarterly CPI (until ABS updates in Jan 2026) |
| `PIE_RBAQ.CSV` | RBA inflation expectations (1970Q1-2023Q1) |
| `interbank_overnight_rate_historical.parquet` | Pre-1990 interbank overnight rate |
| `gscpi_data.xls` | NY Fed Global Supply Chain Pressure Index |

## Model Equations

### NAIRU + Output Gap Model

The joint model estimates NAIRU and potential output using 12 equations:

**State Equations:**

1. **NAIRU**: Random walk without drift
   - `NAIRU_t = NAIRU_{t-1} + ε_t`

2. **Potential Output**: Cobb-Douglas with stochastic drift
   - `g_Y* = α×g_K + (1-α)×g_L + g_MFP + ε_t`

**Observation Equations:**

3. **Price Phillips**: Anchor-augmented with regime-switching (expectations → target transition 1993-1998)
   - `π_t = quarterly(π_anchor) + γ_regime×u_gap + λ×Δρm + ξ×GSCPI² + ε`
   - Three regimes: pre-GFC (moderate), post-GFC (flat), post-COVID (steep)

4. **Wage Phillips (ULC)**: Unit labour cost growth with regime-switching
   - `Δulc = α + γ_regime×u_gap + λ×ΔU/U + φ×Δdfd + θ×π_anchor + ε`
   - Three regimes parallel to price Phillips

5. **Wage Phillips (HCOE)**: Hourly compensation growth with regime-switching
   - `Δhcoe = α + γ_regime×u_gap + λ×ΔU/U + φ×Δdfd + θ×π_anchor + ψ×g_MFP + ε`
   - Includes MFP term (productivity → wages channel)

6. **Okun's Law**: Output gap to unemployment change
   - `ΔU = β×output_gap + ε`

7. **IS Curve**: Output gap persistence with interest rate effects
   - `y_gap_t = ρ×y_gap_{t-1} - β×(r_{t-2} - r*) + γ×fiscal_impulse + ε`

8. **Participation**: Discouraged worker effect
   - `Δpr_t = β×(U_{t-1} - NAIRU_{t-1}) + ε`

9. **Employment**: Labour demand with wage channel
   - `Δemp_t = α + β_ygap×output_gap + β_wage×real_wage_gap + ε`
   - Real wage gap = ULC growth − MFP growth

10. **Exchange Rate**: UIP-style TWI equation
    - `Δe_t = ρ×Δe_{t-1} + β×r_gap_{t-1} + ε`

11. **Import Price Pass-Through**:
    - `Δ4ρm = β_pt×Δ4twi + β_oil×Δ4oil + ρ×Δ4ρm_{t-1} + ε`

12. **Net Exports**: Trade balance response to demand and competitiveness
    - `Δ(NX/Y) = β_ygap×output_gap + β_twi×Δtwi + ε`
    - Output gap effect: imports rise when economy strong
    - TWI effect: appreciation worsens trade balance
    - No intercept: net exports changes have no drift beyond gap/TWI effects

**Symbol definitions:**
- `log_gdp`, `potential_output` = log GDP × 100 (so output_gap ≈ % deviation)
- `π_anchor` = inflation anchor (expectations → target transition 1993-98)
- `u_gap` = (U − NAIRU) / U (proportional unemployment gap)
- `Δx` = quarterly log difference × 100 (for indices: ulc, hcoe, twi, import prices)
- `Δ4x` = 4-quarter log difference × 100 (year-ended growth)
- `ΔU`, `Δpr`, `Δ(NX/Y)` = simple differences (for rates/ratios)
- `ΔU/U` = speed limit: ΔU_{t-1} / U_t
- `r_gap` = cash rate − π_anchor − r*
- `GSCPI` = NY Fed Global Supply Chain Pressure Index (std devs)

### Cobb-Douglas Decomposition

Growth accounting using the Solow residual:
- `g_MFP = g_Y - α×g_K - (1-α)×g_L`

With HP-filtered trends and periodic re-anchoring at business cycle peaks.

## Key Parameters

| Parameter | Description | Prior/Value |
|-----------|-------------|-------------|
| α (alpha) | Capital share of income | N(0.30, 0.05) |
| γ_π_regime | Price Phillips slope (regime-specific) | Pre-GFC: N(-1.5, 1.0), GFC: N(-0.5, 0.5), COVID: N(-2.5, 1.0) |
| γ_wg_regime | Wage Phillips (ULC) slope (regime-specific) | Pre-GFC: TN(-1.0, 0.75), GFC: TN(-0.5, 0.5), COVID: TN(-1.5, 0.75) |
| γ_hcoe_regime | Wage Phillips (HCOE) slope (regime-specific) | Parallel to ULC regimes |
| ψ_hcoe | MFP → hourly wages | TN(1.0, 0.5, lower=0), positive by theory |
| β_okun | Okun's law coefficient | N(-0.2, 0.15), expect negative |
| ρ_is | IS curve persistence | N(0.85, 0.1), expect 0-1 |
| β_is | Interest rate sensitivity | TN(0.2, 0.1, lower=0) |
| β_pr | Discouraged worker effect | TN(-0.05, 0.03, upper=0) |
| β_emp_ygap | Output gap → employment | TN(0.3, 0.15, lower=0), positive by theory |
| β_emp_wage | Real wage gap → employment | TN(-0.1, 0.1, upper=0), negative by theory |
| β_pt | Exchange rate pass-through | TN(-0.3, 0.15, upper=0) |
| β_nx_ygap | Output gap → net exports | TN(-0.05, 0.05, upper=0), negative (imports rise) |
| β_nx_twi | TWI → net exports | TN(-0.02, 0.02, upper=0), negative (appreciation hurts) |

## Modeling Decisions

Key simplifications and design choices:

### Persistence Terms

- **Employment and net exports equations omit AR(1) persistence terms**. These equations condition on output gap, which is already highly persistent (ρ_is ≈ 0.74). Adding persistence terms led to near-zero or negative estimates, indicating redundancy. The persistence in employment and net exports comes through their dependence on the persistent output gap. *Implication*: employment and net exports respond contemporaneously to output gap movements; if slower adjustment is needed, add partial adjustment terms or lagged output gap.

### Data Simplifications

- **Net exports excludes foreign demand**: The RBA MARTIN model includes major trading partner growth, but reliable quarterly data isn't available in standard ABS/RBA tables. The output gap and TWI capture the main dynamics.

- **MFP derived from wage data**: Rather than using ABS 5204 MFP directly (which has revisions and timing issues), MFP is computed from the identity: `MFP = (Δhcoe - Δulc) - α × (g_K - g_L)`. This ensures internal consistency between wage and production equations. *Caveat*: derived MFP inherits measurement error from HCOE, ULC, capital, and hours inputs. ABS 5204 MFP is not used as a cross-check but could serve as external validation.

- **MFP floored at zero**: The HP-filtered MFP trend is floored at zero. This prevents cyclical capacity underutilization from being misread as technological regress. *Caveat*: flooring biases measured MFP upward in deep downturns—it's a pragmatic fix, not a structural claim.

- **r\* is deterministic**: The neutral real rate is computed from smoothed potential growth rather than estimated as a latent state. This reduces model complexity and avoids identification issues between r* and output gap.

### COVID Adjustments

- **Labour force and hours growth smoothed during COVID (2020Q1-2023Q2)**: Raw data shows extreme volatility from JobKeeper and lockdowns. Henderson MA smoothing is applied during this period to extract underlying trends.

- **GSCPI masked to COVID period only**: The Global Supply Chain Pressure Index is set to zero outside 2020Q1-2023Q2. Supply chain disruptions were transitory and the index has limited relevance outside this period. Lagged 2 quarters to capture delayed pass-through.

### Inflation Anchor Transition

- **Blended anchor (1993-1998)**: Before 1993, the anchor equals inflation expectations. After 1998, it equals the 2.5% target. The transition period uses a linear blend, reflecting the gradual credibility gain after inflation targeting was adopted in 1993.

### Regime Switching

- **Three Phillips curve regimes**: Pre-GFC (to 2008Q3), post-GFC (2008Q4-2020Q4), post-COVID (2021Q1+). This captures the well-documented flattening of the Phillips curve after the GFC and apparent re-steepening post-COVID. *Implementation*: breakpoints are fixed (not estimated); regime slopes have independent priors (not hierarchically pooled). This is simpler than Markov-switching but assumes the structural breaks are known.

