# NAIRU + Output Gap Model Overview

This directory contains a Bayesian state-space model for jointly estimating NAIRU (Non-Accelerating Inflation Rate of Unemployment), potential output, and output gaps for Australia using PyMC.

## Summary

| Component | Method | Key Feature |
|-----------|--------|-------------|
| NAIRU | Gaussian random walk | No drift, σ ≈ 0.25 |
| Potential Output | Cobb-Douglas + innovations | SkewNormal innovations (sticky downwards) |
| Phillips Curves | Regime-switching (3 regimes) | Pre-GFC, GFC-COVID, post-COVID |
| Identification | 10 observation equations | Joint estimation with proper uncertainty |
| Forecasting | Model-consistent projection | 4-quarter horizon with policy scenarios |

The Bayesian approach with multiple observation equations provides good identification. MCMC diagnostics show convergence (R-hat < 1.01, ESS > 400), theoretical sign constraints are satisfied (Phillips slopes negative, Okun coefficient negative), and estimates align with RBA research on transmission magnitudes.

---

## File Structure

### Core Pipeline
```
model.py               # Unified entry point (run_model, main)
stage1.py              # Build model, sample posterior, save results
stage2.py              # Load results, diagnostics, plotting
stage3.py              # Model-consistent forecasting with policy scenarios
base.py                # Sampler config, coefficient utilities
```

### Equations (src/models/nairu/equations/)
```
__init__.py            # Exports + regime boundary constants
state_space.py         # NAIRU random walk (state equation)
production.py          # Potential output via Cobb-Douglas (state equation)
okun.py                # Okun's Law: ΔU ↔ output gap
phillips.py            # Price, wage (ULC), and hourly COE Phillips curves
is_curve.py            # IS curve: output gap ↔ real rate gap
participation.py       # Discouraged worker effect
employment.py          # Labour demand equation
exchange_rate.py       # UIP-style TWI equation
import_price.py        # Import price pass-through
net_exports.py         # Net exports equation
```

### Analysis (src/models/nairu/analysis/)
```
diagnostics.py         # MCMC diagnostics (R-hat, ESS, divergences)
extraction.py          # Extract posterior summaries from trace
posterior_predictive_checks.py  # PPC plots
residual_autocorrelation.py     # Residual analysis
plot_*.py              # Various plotting modules
inflation_decomposition.py      # Demand vs supply decomposition
```

---

## Architecture

### Component Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                         model.py                                 │
│                                                                  │
│   run_model() → Quick estimation, returns NAIRUResults          │
│   main()      → Full pipeline: Stage 1 → Stage 2 → Stage 3      │
└──────────────────────────────┬──────────────────────────────────┘
                               │
         ┌─────────────────────┼─────────────────────┐
         ▼                     ▼                     ▼
┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
│    stage1.py    │  │    stage2.py    │  │    stage3.py    │
│                 │  │                 │  │                 │
│ build_model()   │  │ load_results()  │  │ forecast()      │
│ sample_model()  │  │ NAIRUResults    │  │ run_scenarios() │
│ save_results()  │  │ plot_all()      │  │ ForecastResults │
└─────────────────┘  └─────────────────┘  └─────────────────┘
              │
              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    equations/__init__.py                         │
│                                                                  │
│  State Equations:                                                │
│    nairu_equation()           → NAIRU random walk               │
│    potential_output_equation() → Cobb-Douglas potential         │
│                                                                  │
│  Observation Equations:                                          │
│    okun_law_equation()        → Links output gap to ΔU          │
│    price_inflation_equation() → Price Phillips curve            │
│    wage_growth_equation()     → Wage Phillips (ULC)             │
│    hourly_coe_equation()      → Wage Phillips (HCOE)            │
│    is_equation()              → IS curve (output gap dynamics)  │
│    participation_equation()   → Discouraged worker effect       │
│    employment_equation()      → Labour demand                   │
│    exchange_rate_equation()   → UIP-style TWI                   │
│    import_price_equation()    → Import price pass-through       │
│    net_exports_equation()     → Net exports                     │
└─────────────────────────────────────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────────────────────────────┐
│                         base.py                                  │
│                                                                  │
│  SamplerConfig          → NUTS sampler settings                 │
│  sample_model()         → Run PyMC sampling                     │
│  set_model_coefficients() → Create priors or fix constants     │
│  save_trace/load_trace  → NetCDF persistence                    │
└─────────────────────────────────────────────────────────────────┘
```

### Data Flow

1. **Data Preparation** (`src/data/observations.py`):
   - Fetches ABS data via `readabs` library
   - Computes transformations (growth rates, lags, regime indicators)
   - Returns `obs` dict (numpy arrays) + `obs_index` (PeriodIndex)

2. **Model Building** (`stage1.build_model()`):
   - Creates PyMC model context
   - Adds state equations (NAIRU, potential)
   - Adds observation equations (Phillips curves, Okun, IS, etc.)
   - Returns configured `pm.Model`

3. **Sampling** (`base.sample_model()`):
   - Runs NUTS via NumPyro backend
   - Default: 10k draws, 3.5k tune, 5 chains
   - Returns `az.InferenceData`

4. **Analysis** (`stage2.run_stage2()`):
   - Loads saved trace and observations
   - Runs MCMC diagnostics
   - Generates all plots
   - Tests theoretical expectations

5. **Forecasting** (`stage3.run_stage3()`):
   - Loads saved trace and observations
   - Extracts posterior samples and coefficients
   - Projects IS curve, Okun's Law, Phillips curve forward
   - Runs policy scenarios (+50bp to -50bp)
   - Returns `ForecastResults` with uncertainty

---

## State Equations

### NAIRU (state_space.py)

Gaussian random walk without drift:

```
NAIRU_t = NAIRU_{t-1} + ε_t,  ε_t ~ N(0, σ²)
```

- **σ (nairu_innovation)**: Typically fixed at 0.25 for smoothness
- **Initial distribution**: N(6, 2) — centered on Australian historical NAIRU average (~5-7%) with moderate uncertainty
- **Interpretation**: Natural rate evolves slowly with structural labor market changes

### Potential Output (production.py)

Cobb-Douglas production function with stochastic innovations:

```
g_potential_t = α_t×g_K_t + (1-α_t)×g_L_t + g_MFP_t + ε_t

Y*_t = Y*_{t-1} + g_potential_t
```

Where:
- `α_t` = time-varying capital share from ABS national accounts
- `g_K` = capital stock growth
- `g_L` = labor force growth
- `g_MFP` = multi-factor productivity growth (HP-filtered trend, floored at zero — see [Why Floor MFP at Zero?](#why-floor-mfp-at-zero))

**Capital share**: Rather than fixing α or estimating it, the model uses observed factor income shares from ABS 5206.0:
```
α_t = GOS_t / (GOS_t + COE_t)
```
Where GOS = Gross Operating Surplus and COE = Compensation of Employees. This captures the secular rise in capital's share from ~0.25 in the 1970s to ~0.35 today.

**Key feature**: Zero-mean SkewNormal innovations with positive skew (α=1). The location is shifted negative so E[ε]=0 while ~75% of draws remain positive. This makes potential "sticky downwards" — growth is more likely than decline, reflecting that capital accumulation and productivity gains are hard to reverse.

---

## Observation Equations

### Phillips Curves (phillips.py)

All three Phillips curves use **regime-switching slopes** with three regimes:

| Regime | Period | Characteristic |
|--------|--------|----------------|
| Pre-GFC | Up to 2007Q4 | Moderate slope |
| GFC | 2008Q1 - 2020Q4 | Flat (anchored expectations) |
| Post-COVID | 2021Q1+ | Steep (reawakened sensitivity) |

**Price Phillips Curve:**
```
π = quarterly(π_anchor) + γ_regime × u_gap + controls + ε
```

Where `u_gap = (U - NAIRU) / U` (percentage deviation from NAIRU)

**Wage Phillips Curve (ULC):**
```
Δulc = α + γ_regime × u_gap + λ × (ΔU/U) + φ × Δ4dfd + θ × π_anchor + ε
```

Includes:
- Speed limit effect (λ × ΔU/U)
- Price-to-wage pass-through (φ × demand deflator)
- Trend expectations anchoring (θ)

**Hourly COE Phillips Curve:**
Same as ULC but adds productivity channel:
```
Δhcoe = ... + ψ × mfp_growth + ε
```

The ψ coefficient captures how productivity gains flow to wages.

### Okun's Law (okun.py)

Change form linking output gap to unemployment changes:

```
ΔU = β × output_gap + ε
```

- **β (beta_okun)**: Negative, typically -0.1 to -0.3
- **Interpretation**: 1% positive output gap → ~0.2pp fall in unemployment
- **Note**: Regime-switching was tested but posteriors showed no meaningful difference across regimes

### IS Curve (is_curve.py)

Output gap persistence with interest rate, debt servicing, and housing wealth channels:

```
y_gap_t = ρ × y_gap_{t-1} - β × (r_{t-2} - r*) + γ × fiscal_{t-1} - δ × Δdsr_{t-1} + η × Δhw_{t-1} + ε
```

Where:
- `ρ` = output gap persistence (~0.76)
- `β` = interest rate sensitivity (positive, ~0.06)
- `r - r*` = real rate gap (policy rate minus neutral rate)
- `fiscal_impulse` = government spending growth minus GDP growth
- `Δdsr` = change in debt servicing ratio (interest payments / disposable income)
- `δ` = debt servicing sensitivity (positive, ~0.09)
- `Δhw` = housing wealth growth (quarterly log change × 100)
- `η` = housing wealth sensitivity (positive, ~0.02)

**Debt servicing channel**: The DSR term captures how rate changes feed through to household cash flows:
- Rates ↑ → mortgage payments ↑ → disposable income ↓ → spending ↓
- This provides a direct transmission mechanism from policy rates to aggregate demand
- Data source: ABS 5206.0 Household Income Account (back to 1959Q3)

**Housing wealth channel**: The Δhw term captures how rate changes affect consumption through housing wealth:
- Rates ↑ → house prices ↓ → household wealth ↓ → consumption ↓
- RBA estimates MPC out of housing wealth ~2-3 cents per dollar
- Housing wealth ~$11.5T (land ~80%, dwellings ~20%)
- Data source: ABS 5232.0 National Financial Accounts (1988Q3+, backcast to 1984Q3)

### Participation Rate (participation.py)

Discouraged worker effect:

```
Δpr_t = β_pr × (U_{t-1} - NAIRU_{t-1}) + ε
```

- **β_pr**: Negative — when unemployment exceeds NAIRU, workers exit labor force
- Uses lagged unemployment gap to avoid simultaneity

### Employment (employment.py)

Labour demand based on output and real wages:

```
Δemp = α + β_ygap × output_gap + β_wage × (Δulc - Δmfp) + ε
```

- **β_ygap**: Positive — output expansion raises employment
- **β_wage**: Negative — higher real wages reduce hiring
- Connects wage Phillips curves back to labor demand

### Exchange Rate (exchange_rate.py)

UIP-style equation for Trade-Weighted Index:

```
Δtwi_t = ρ × Δtwi_{t-1} + β_r × r_gap_{t-1} + ε
```

**Important**: The UIP puzzle means interest rate coefficients are empirically weak. Finding β_r ≈ 0.05-0.15 is expected, not a specification error.

### Import Prices (import_price.py)

Pass-through from exchange rate to import prices:

```
Δ4ρm_t = ρ × Δ4ρm_{t-1} + β_pt × Δ4twi_{t-1} + β_oil × Δoil + ε
```

### Net Exports (net_exports.py)

```
Δ(NX/GDP) = α + β_ygap × output_gap + β_twi × Δtwi + ε
```

- **β_ygap**: Negative — domestic demand expansion increases imports
- **β_twi**: Negative — appreciation hurts exports

---

## Monetary Policy Transmission Channels

The model includes four channels through which cash rate changes affect the economy:

### 1. Direct Interest Rate Channel (IS Curve)

```
Rate ↑ → Real rate gap ↑ → Output gap ↓ → Unemployment ↑ → Inflation ↓
```

- **Coefficient**: β_is ≈ 0.06 (output gap reduction per 100bp, lagged 2 quarters)
- **Mechanism**: Higher real rates reduce investment and consumption
- **Timing**: Effect begins at lag 2 (monetary policy works with a lag)

### 2. Debt Servicing Ratio (DSR) Channel

```
Rate ↑ → Mortgage payments ↑ → Disposable income ↓ → Spending ↓ → Output gap ↓
```

- **Coefficient**: δ_dsr ≈ 0.09 (output gap reduction per 1pp DSR increase)
- **Pass-through**: ~1.0pp DSR increase per 100bp rate rise
- **Derivation**:
  - Housing debt: ~$2.2T
  - Gross disposable income: ~$1.6T/year
  - Variable rate share: ~70%
  - Extra interest per 100bp = $2.2T × 70% × 1% = $15.4B
  - DSR change = $15.4B / $1.6T ≈ 1.0pp
- **Data source**: ABS 5206.0 Household Income Account (from 1959Q3)

### 3. Housing Wealth Channel

```
Rate ↑ → House prices ↓ → Household wealth ↓ → Consumption ↓ → Output gap ↓
```

- **Coefficient**: η_hw ≈ 0.02 (output gap change per 1% housing wealth growth change)
- **Pass-through**: ~-1.0% housing wealth growth per 100bp rate rise
- **RBA estimates**:
  - 100bp rate increase → 2-4% house price decline (annualized)
  - MPC out of housing wealth: ~2-3 cents per dollar
  - Housing wealth: ~$11.5T (land ~80%, dwellings ~20%)
- **Data source**: ABS 5232.0 National Financial Accounts (1988Q3+, backcast to 1984Q3)
- **Backcast method**: Pre-1988 uses annual dwelling capital stock (ABS 5204.0) scaled by
  extrapolated land/dwelling ratio

### 4. Exchange Rate (FX) Channel

```
Rate ↑ → AUD appreciates → Import prices ↓ → Inflation ↓
```

- **Pass-through**: ~0.35pp inflation reduction per 100bp (RBA calibration)
- **RBA estimates** (Bulletin April 2025):
  - 100bp rate increase → 5-10% TWI appreciation
  - That appreciation → 0.25-0.5pp lower inflation over 2 years
- **Why RBA calibration?**: The model's estimated coefficients are too small:
  - β_er_r (~0.08% TWI per 100bp) vs RBA's 5-10% — due to UIP puzzle
  - ρ_pi (~0.02) vs literature 0.05-0.15 — weak identification

**Fed counterfactual assumption**: The 0.35pp calibration assumes RBA moves unilaterally
(Fed holds steady). The FX channel depends on *relative* interest rates:
- **OVERSTATES** effect if Fed moves in tandem with RBA (TWI unchanged, no FX transmission)
- **UNDERSTATES** effect if Fed moves opposite to RBA (TWI moves more, stronger transmission)

Example: If Fed cuts 100bp while RBA hikes 50bp, the 150bp relative divergence would
produce ~50% larger FX effect than calibrated.

### Combined Transmission Effect

For a 100bp rate increase, the model projects (by year 2):

| Channel | Output Gap Effect | Inflation Effect |
|---------|------------------|------------------|
| Direct rate (β_is) | -0.06 | (via Phillips) |
| DSR (δ_dsr × 1.0pp) | -0.09 | (via Phillips) |
| Housing wealth (η_hw × -1.0%) | -0.02 | (via Phillips) |
| FX (RBA calibration) | — | -0.35pp |
| **Total** | **-0.17** | **~-0.7pp** |

This is consistent with RBA research showing monetary policy transmission of approximately 0.5-1.0pp inflation reduction per 100bp over 2 years.

---

## Inflation Anchor

The model uses an inflation anchor that transitions from RBA-constructed expectations to the 2.5% target:

| Period | Anchor |
|--------|--------|
| Pre-1993Q1 | RBA expectations (signal extraction from surveys + bond yields) |
| 1993Q1–1998Q1 | Linear blend to target |
| Post-1998Q1 | 2.5% target |

**Data source**: The pre-1998 expectations series (PIE_RBAQ) is constructed by the RBA using the Cusbert (2017) methodology — signal extraction from multiple measures (surveys, bond yields) after controlling for co-movement with recent inflation, bias-adjusted to post-1996 mean. Sourced via [MacroDave/MARTIN](https://github.com/MacroDave/MARTIN).

This means **NAIRU is interpreted as the unemployment rate consistent with achieving the inflation target** (in the post-1998 period). See [Why Phase In the Inflation Target?](#why-phase-in-the-inflation-target) for rationale.

---

## Prior Specification

### set_model_coefficients() Pattern

Each equation uses `set_model_coefficients()` to create priors:

```python
settings = {
    "beta": {"mu": 0.5, "sigma": 0.1},           # Normal prior
    "gamma": {"mu": -0.3, "sigma": 0.2, "upper": 0},  # Truncated Normal
    "epsilon": {"sigma": 0.5},                    # HalfNormal (scale param)
}
mc = set_model_coefficients(model, settings, constant={"beta": 0.5})
```

- If coefficient is in `constant` dict → fixed value (not estimated)
- If `sigma` only (no `mu`) → HalfNormal
- If `lower` or `upper` → TruncatedNormal
- Otherwise → Normal

Fixed constants are stored on `model._fixed_constants` for later retrieval.

### Sign Constraints

Economic theory imposes sign constraints via truncated priors:

| Parameter | Expected Sign | Constraint |
|-----------|--------------|------------|
| beta_okun | Negative | `upper=0` |
| gamma_* (Phillips slopes) | Negative | `upper=0` |
| beta_pr (discouraged worker) | Negative | `upper=0` |
| beta_emp_wage | Negative | `upper=0` |
| beta_is, theta_*, psi_* | Positive | `lower=0` |
| delta_dsr | Positive | `lower=0` |

---

## Sampler Configuration

Default settings (`SamplerConfig`):

```python
SamplerConfig(
    draws=10_000,      # Posterior samples per chain
    tune=3_500,        # Warmup/tuning samples
    chains=5,          # Independent chains
    cores=5,           # Parallel cores
    sampler="numpyro", # JAX-based NUTS
    target_accept=0.90 # Acceptance probability
)
```

Higher `target_accept` (0.90-0.95) reduces divergences in complex posteriors.

---

## Results Container

`NAIRUResults` dataclass provides convenient access:

```python
results = run_model()

# Posterior DataFrames (columns = samples)
results.nairu_posterior()      # Shape: (T, n_samples)
results.potential_posterior()

# Point estimates (posterior median)
results.nairu_median()         # Series
results.potential_median()

# Derived series
results.unemployment_gap()     # U - NAIRU
results.output_gap()           # log(GDP) - log(Y*)
```

---

## Diagnostics

### MCMC Diagnostics

- **R-hat**: Should be < 1.01 for convergence
- **ESS (bulk/tail)**: Should be > 400 per chain
- **Divergences**: Should be 0 or very few

### Theoretical Expectations Tests

`test_theoretical_expectations()` checks:

1. **Sign tests**: P(correct sign) > 99% → PASS
2. **Range tests**: P(α ∈ [0.20, 0.35]) for capital share
3. **Stability tests**: P(0 < ρ < 1) for persistence parameters

### Zero Coefficient Check

Flags parameters whose 90% HDI includes zero — may indicate weak identification.

---

## Stage 3: Forecasting

Stage 3 generates model-consistent forecasts using the estimated coefficients from the posterior. The interpretation is "no new shocks, economy trends toward equilibrium."

### Forecast Horizon

The model produces 4-quarter ahead forecasts. The IS curve uses rate gaps lagged 2 periods:

| Forecast Period | Rate Gap Used | Source |
|-----------------|---------------|--------|
| T+1 | T-1 | Historical |
| T+2 | T | Policy decision |
| T+3 | T+1 | Hold assumption |
| T+4 | T+2 | Hold assumption |

### Forecast Equations

The forecast projects forward using estimated coefficients and RBA-calibrated pass-throughs:

1. **IS Curve** (output gap dynamics with DSR transmission):
   ```
   y_gap_t = ρ_is × y_gap_{t-1} - β_is × rate_gap_{t-2} - δ_dsr × Δdsr_{t-1}
   ```
   Where Δdsr is projected using the rate→DSR pass-through (~1.0pp per 100bp).

2. **Okun's Law** (unemployment from output gap):
   ```
   ΔU_t = β_okun × y_gap_t × 1.6
   ```
   A demand transmission multiplier of 1.6 is applied to match RBA's demand channel estimate.
   See [Calibration Corrections](#calibration-corrections) below.

3. **Phillips Curve** (inflation from unemployment gap + FX channel):
   ```
   π_t = π_anchor + γ_covid × (U_t - NAIRU_t) / U_t + FX_effect_t
   ```
   Where FX_effect uses RBA-calibrated pass-through (~0.35pp per 100bp).

4. **NAIRU**: Random walk — stays at final posterior value

5. **Potential Output**: Grows at Cobb-Douglas drift rate

6. **DSR Transmission**: Rate changes feed through to debt servicing at lag 1:
   ```
   Δdsr_t = 1.0 × Δrate_{t-1}
   ```
   Calibrated from household debt/income ratios and variable rate share.

7. **Housing Wealth Transmission**: Rate changes affect housing wealth growth at lag 1:
   ```
   Δhw_t = Δhw_{t-1} - 1.0 × Δrate_{t-1}
   ```
   RBA estimates 100bp rate rise → 2-4% house price decline (annualized, ~1%/qtr).

8. **FX Transmission**: Rate changes affect inflation via exchange rate channel:
   ```
   FX_effect_t = -0.0875 × Δrate_{t-1}  (quarterly, = 0.35pp/4 per 100bp)
   ```
   RBA calibration used because model's estimated UIP coefficients are too small (UIP puzzle).

### Policy Scenarios

Stage 3 runs nine policy scenarios by default, testing rate changes from the current cash rate:

| Scenario | Rate Change |
|----------|-------------|
| +200bp | +2.00% |
| +100bp | +1.00% |
| +50bp | +0.50% |
| +25bp | +0.25% |
| hold | 0.00% |
| -25bp | -0.25% |
| -50bp | -0.50% |
| -100bp | -1.00% |
| -200bp | -2.00% |

The "move and hold" assumption means the rate stays at the new level for the entire forecast horizon.

### Calibration Corrections

Stage 3 models **four explicit transmission channels**:

1. **Direct rate** → output gap (β_is ≈ 0.06)
2. **DSR** → output gap (δ_dsr ≈ 0.09, calibrated 1pp DSR per 100bp rate)
3. **Housing wealth** → output gap (η_hw ≈ 0.02, calibrated -1% wealth per 100bp)
4. **FX** → inflation (calibrated 0.35pp per 100bp, RBA estimate)

**Channels not explicitly modeled:**

- **Expectations**: Credible policy signals affect wage/price setting directly
- **Credit/lending**: Banks adjust lending standards — though 2024-25 data shows housing credit *accelerating* (4.5% → 6.6% YoY) despite 425bp of hikes, suggesting this channel is not operating in the current cycle
- **Business cash flow**: Higher rates increase firm interest costs, reducing investment
- **Equity/super wealth**: Rate rises reduce asset values beyond housing

**Transmission Gap Analysis:**

| Channel | Model Estimate | RBA Estimate | Gap |
|---------|---------------|--------------|-----|
| Demand (rate→output→U→π) | 0.18pp/100bp | ~0.29pp/100bp | 0.11pp |
| FX (rate→TWI→imports→π) | 0.08pp/100bp* | ~0.35pp/100bp | 0.27pp |
| **Total** | **0.26pp/100bp** | **0.64pp/100bp** | **0.38pp** |

*Model's UIP coefficients understate FX transmission (UIP puzzle)

**Calibration factors applied:**

1. **Demand transmission multiplier**: 1.6× on β_okun
   - Captures missing channels (expectations, credit, business cash flow) in reduced form
   - Scales unemployment response to match RBA's demand channel estimate
   - Brings demand channel from 0.18pp to ~0.29pp per 100bp

2. **FX Channel**: Uses RBA's 0.35pp/100bp directly
   - Replaces model's weak UIP-based estimate
   - See [FX Channel Assumptions](#fx-channel-assumptions) for counterfactual discussion

These corrections are applied only in Stage 3 forecasting, not in Stage 1 estimation. The estimated model remains unchanged and can be used for historical decomposition without calibration adjustments.

### ForecastResults Container

```python
from src.models.nairu.model import run_stage3, ForecastResults

# Run all scenarios
scenario_results = run_stage3()  # dict[str, ForecastResults]

# Access specific scenario
hold = scenario_results["hold"]

# Point forecasts (posterior median)
hold.summary()

# Uncertainty (90% HDI)
hold.output_gap_hdi()
hold.unemployment_hdi()
hold.inflation_annual_hdi()
```

### Example Output

```
================================================================================
POLICY SCENARIO COMPARISON
Current cash rate in model: 3.60%
================================================================================

2026Q3:
--------------------------------------------------------------------------------
Scenario Cash Rate Output Gap     U U Gap π (ann)
   +50bp     4.10%      0.117 4.09% -0.54   3.22%
   +25bp     3.85%      0.205 4.06% -0.57   3.37%
    hold     3.60%      0.294 4.02% -0.60   3.52%
   -25bp     3.35%      0.383 3.99% -0.63   3.67%
   -50bp     3.10%      0.474 3.96% -0.66   3.82%
================================================================================
```

Note: The ~0.30pp inflation difference between +50bp and hold (3.22% vs 3.52%) reflects both the demand channel (Phillips curve via lower output gap/higher unemployment) and the FX channel (RBA-calibrated import price effect).

---

## Running the Model

### Full Pipeline
```bash
python -m src.models.nairu.model -v
```

### Stage 1 Only (Sampling)
```bash
python -m src.models.nairu.stage1 --start 1980Q1 -v
```

### Stage 2 Only (Analysis)
```bash
python -m src.models.nairu.stage2 --show -v
```

### Stage 3 Only (Forecasting)
```bash
python -m src.models.nairu.stage3
```

### From Python
```python
from src.models.nairu.model import run_model, NAIRUResults

# Quick run
results = run_model(start="1980Q1", verbose=True)

# Access results
print(results.nairu_median().tail())
print(results.output_gap().tail())
```

---

## Key Design Decisions

### Why Joint Estimation?

Estimating NAIRU, potential output, and gaps jointly (rather than sequentially) provides:

1. **Proper uncertainty propagation**: Uncertainty in NAIRU flows through to Phillips curve coefficients
2. **Better identification**: Multiple observation equations (10 total) constrain the latent states
3. **Consistent estimates**: All equations share the same latent states

### Why Regime-Switching Phillips Curves?

Post-GFC flattening of the Phillips curve is well-documented. A single time-invariant slope would be misspecified. Three regimes capture:

- **Pre-GFC**: Traditional Phillips curve relationship
- **GFC era**: Anchored expectations, flat curve
- **Post-COVID**: Reawakened inflation sensitivity

### Why SkewNormal for Potential?

Potential output should be "sticky downwards":
- Capital doesn't disappear in recessions
- Productivity gains are hard to reverse
- Labor force grows over time

**Implementation**: Zero-mean SkewNormal with positive skew (α=1).

The SkewNormal distribution with α>0 has ~75% of draws positive, but also has a **positive mean** (not zero). If used naively with μ=0, this adds spurious drift to potential output, biasing the output gap and r* estimates.

**Zero-mean adjustment**: Shift the location parameter to compensate:
```
E[SkewNormal(μ, σ, α)] = μ + σ × δ × sqrt(2/π)
where δ = α / sqrt(1 + α²)

For α=1, σ=0.3: mean_shift ≈ 0.17%/qtr
Set μ = -σ × 0.5642 to get E[ε] = 0
```

This gives asymmetric shocks (growth more likely than decline) without biasing the level of potential output. The Cobb-Douglas drift alone determines trend growth; the SkewNormal innovation only affects the *distribution* of deviations around that trend.

### Why Fix Some Parameters?

Some parameters (like `nairu_innovation`) are poorly identified and can cause sampling issues. Fixing them at reasonable values:
- Improves sampling efficiency
- Prevents unreasonable posterior draws
- Reflects prior knowledge about smoothness

### Why Floor MFP at Zero?

MFP (Multi-Factor Productivity) trend growth is floored at zero before entering the potential output equation (`src/data/productivity.py`):

- **Negative MFP reflects underutilization, not technological regress**: During recessions, measured MFP falls as firms hoard labor and underuse capital. This is cyclical, not structural.
- **True technological progress doesn't reverse**: Knowledge and process improvements persist. A recession doesn't make workers forget how to use computers.
- **Potential output should reflect supply capacity**: The production function estimates what the economy *could* produce at full utilization. Cyclical MFP declines contaminate this with demand-side effects.
- **Implementation**: Raw MFP is HP-filtered (λ=1600) to extract the trend, then floored with `np.maximum(mfp_trend, 0)`.

### Why Smooth Labour Inputs During COVID?

Labour force growth and hours worked are replaced with Henderson-smoothed values during 2020Q1–2023Q2 (`src/data/observations.py`):

- **COVID caused measurement artifacts, not structural breaks**: Lockdowns produced extreme swings in measured labour force participation and hours that don't reflect genuine changes in labour supply capacity.
- **Potential output should be smooth through temporary shocks**: The economy's productive capacity didn't actually collapse and recover in a few quarters — workers were temporarily unable to work, not permanently removed from the labour force.
- **Prevents contaminating NAIRU estimates**: Without smoothing, the model would interpret COVID unemployment spikes as movements in NAIRU rather than cyclical deviations from it.
- **Implementation**: Raw series used normally; Henderson MA (term=13) applied only during 2020Q1–2023Q2.

### Why Use Deterministic r*?

The neutral real interest rate (r*) is computed from Cobb-Douglas potential growth rather than estimated as a latent variable (`src/data/cash_rate.py`):

```
r* ≈ α×g_K + (1-α)×g_L + g_MFP (annualized and Henderson-smoothed)
```

- **Grounded in growth theory**: In the Ramsey model, the equilibrium real rate equals the rate of time preference plus the growth rate of consumption. Using potential growth provides a theory-consistent anchor.
- **Avoids identification problems**: Estimating r* jointly with NAIRU and potential output creates identification challenges — the data can't separately identify all three unobserved states.
- **Reduces model complexity**: One fewer latent variable means faster sampling and fewer divergences.
- **Still time-varying**: r* moves with structural changes in productivity and labour force growth, just not with cyclical noise.

### Why Splice the Cash Rate?

The cash rate series combines modern OCR with historical interbank rates (`src/data/cash_rate.py`):

- **Extends sample back to 1970s**: The RBA's Official Cash Rate only begins in 1990. Splicing with the interbank overnight rate allows estimation over longer samples that include the high-inflation 1970s–80s.
- **Consistent policy rate concept**: Both series represent the overnight interbank rate that the RBA targets/targeted.
- **Favours modern data**: Where both series overlap, the OCR is used.

### Why Smooth Capital Growth?

Capital stock growth is always Henderson-smoothed, not just during COVID (`src/data/observations.py`):

- **Capital stock is inherently slow-moving**: Quarterly capital growth from national accounts contains measurement noise that overstates true variation in the capital stock.
- **Investment is lumpy, capital isn't**: Investment spending is volatile, but the productive capital stock evolves smoothly as new investment is a small fraction of the existing stock.
- **Prevents spurious potential output volatility**: Unsmoothed capital growth would create artificial quarter-to-quarter swings in estimated potential.

### Why Use Percentage Unemployment Gap?

The Phillips curves use `(U - NAIRU) / U` rather than the simple difference `(U - NAIRU)` (`src/models/nairu/equations/phillips.py`):

- **Scale-invariant**: A 1pp gap matters more when unemployment is 4% than when it's 10%. Dividing by U captures this — the same absolute gap represents a larger percentage deviation when unemployment is low.
- **Consistent with wage bargaining theory**: Workers' bargaining power depends on the relative tightness of the labour market, not absolute unemployment levels.
- **Empirically superior**: Models using percentage gaps typically fit Australian wage and price data better than those using level gaps.

### Why Phase In the Inflation Target?

The inflation anchor transitions gradually from expectations to the 2.5% target (`src/data/rba_loader.py`):

| Period | Anchor |
|--------|--------|
| Pre-1993Q1 | RBA-constructed expectations (PIE_RBAQ) |
| 1993Q1–1998Q1 | Linear blend from expectations to target |
| Post-1998Q1 | Fixed at 2.5% |

- **Inflation targeting was announced in 1993**: The RBA adopted the 2–3% target band, but credibility wasn't instant.
- **Credibility takes time to build**: Wage and price setters didn't immediately anchor expectations to the target — surveys show gradual convergence through the mid-1990s.
- **1998 marks full credibility**: By this point, inflation expectations had converged to target and remained anchored through subsequent shocks.
- **Affects NAIRU interpretation**: Post-1998, NAIRU is the unemployment rate consistent with achieving the inflation target. Pre-1993, it's the rate consistent with stable (but potentially high) inflation.
- **Data provenance**: The PIE_RBAQ series is RBA-constructed using Cusbert (2017) signal extraction methodology — combines surveys and bond yields, bias-adjusted to post-1996 inflation mean. Sourced via [MacroDave/MARTIN](https://github.com/MacroDave/MARTIN) GitHub repository.
