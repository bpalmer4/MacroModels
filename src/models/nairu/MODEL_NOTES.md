# NAIRU + Output Gap Model Overview

This directory contains a Bayesian state-space model for jointly estimating NAIRU (Non-Accelerating Inflation Rate of Unemployment), potential output, and output gaps for Australia using PyMC (NumPyro NUTS backend).

## Summary

| Component | Method | Key Feature |
|-----------|--------|-------------|
| NAIRU | Gaussian random walk (default) | No drift, scale≈0.15; Student-t(ν=4) variant available for fat tails |
| Potential Output | Cobb-Douglas + Gaussian innovations (default) | SkewNormal variant available (asymmetric: small positives common, large negatives rare) |
| Phillips Curves | Single slope (default) | Regime-switching variant available (3 regimes: pre-GFC, GFC-COVID, post-COVID) |
| Identification | 2 state + 10 observation equations | Joint estimation with proper uncertainty |
| Scenario Analysis | Model-consistent projection | 4-quarter horizon with policy scenarios |

The Bayesian approach with multiple observation equations provides good identification. MCMC diagnostics show convergence (R-hat < 1.01, ESS > 400), theoretical sign constraints are satisfied (Phillips slopes negative, Okun coefficient negative), and estimates align with RBA research on transmission magnitudes.

---

## File Structure

### Core Pipeline
```
model.py                    # Unified entry point (run_model, main)
stage1.py                   # Build model, sample posterior, save results
stage2.py                   # Load results, diagnostics, plotting
stage3.py                   # Deterministic scenario analysis (clean lines)
stage3_forward_sampling.py  # Monte Carlo scenario analysis (full uncertainty)
base.py                     # Sampler config, coefficient utilities
```

### Equations (src/models/nairu/equations/)
```
__init__.py            # Exports + regime boundary constants
state_space.py         # NAIRU random walk (state equation)
production.py          # Potential output via Cobb-Douglas (state equation)
okun.py                # Okun's Law: error correction form
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
plot_productivity.py   # LP and MFP derived from wage data
plot_capital_deepening.py  # Capital deepening (g_K - g_L)
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
┌─────────────────┐  ┌─────────────────┐  ┌───────────────────────────────────┐
│    stage1.py    │  │    stage2.py    │  │           Stage 3                 │
│                 │  │                 │  │                                   │
│ build_model()   │  │ load_results()  │  │ stage3.py (3a: deterministic)    │
│ sample_model()  │  │ NAIRUResults    │  │ stage3_forward_sampling.py       │
│ save_results()  │  │ plot_all()      │  │   (3b: Monte Carlo)              │
└─────────────────┘  └─────────────────┘  └───────────────────────────────────┘
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
│    okun_equation()            → Error correction Okun's Law     │
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

5. **Scenario Analysis** (Stage 3a & 3b):
   - Loads saved trace and observations
   - Extracts posterior samples and coefficients
   - Projects IS curve, Okun's Law, Phillips curve forward
   - Runs policy scenarios (+200bp to -200bp)
   - Stage 3a (`stage3.py`): Deterministic using posterior medians
   - Stage 3b (`stage3_forward_sampling.py`): Monte Carlo with sampled shocks

### Data Release Timing

The model stages have different data dependencies:

| Stage | Data Required | Earliest Run |
|-------|--------------|--------------|
| Stage 1 & 2 | Modellers database | ~1 day after quarterly National Accounts (GDP) |
| Stage 3 | Financial Accounts (5232.0) | ~1 month after quarterly National Accounts |

**Stage 1 & 2 (Estimation)**: Can run once the modellers database is available, typically the day after the ABS releases the quarterly National Accounts (5206.0). Most observation series are derived from National Accounts data.

**Stage 3 (Scenario Analysis)**: Requires the latest ABS 5232.0 Financial Accounts for housing wealth data (Δhw). This is released approximately one month after the National Accounts. Running Stage 3 before this release means the housing wealth channel will use data from the previous quarter.

---

## State Equations

### NAIRU (state_space.py)

Gaussian random walk without drift (default):

```
NAIRU_t = NAIRU_{t-1} + ε_t,  ε_t ~ N(0, σ)
```

- **σ (nairu_innovation)**: Typically fixed at 0.15 (RBA-consistent, ~0.15pp per quarter)
- **Initial distribution**: N(6, 2) — centered on Australian historical NAIRU average (~5-7%) with moderate uncertainty

**Student-t variant** (`student_t_nairu=True`, used by `complex` model variant):

```
NAIRU_t = NAIRU_{t-1} + ε_t,  ε_t ~ StudentT(ν=4, 0, σ)
```

- **ν (degrees of freedom)**: Fixed at 4 — fat tails with finite variance (many small moves, occasional large moves)
- Most quarter-to-quarter changes are small, but the model allows occasional large shifts (e.g., GFC, COVID) via fat tails without requiring extreme probability draws.

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
- `g_MFP` = multi-factor productivity growth (derived from wage data, HP-filtered, floored at zero — see [Why Floor MFP at Zero?](#why-floor-mfp-at-zero))

**MFP derivation**: Rather than sourcing MFP directly from ABS 5204.0, the model derives it from wage data using the Solow residual identity (`src/data/productivity.py`):
```
Labour Productivity (LP) = Δhcoe - Δulc    (from ULC = HCOE / LP)
MFP = LP - α × capital_deepening
    = LP - α × (g_K - g_L)
```
This provides a wage-consistent productivity measure that aligns with the model's Phillips curve equations.

**Capital share**: Rather than fixing α or estimating it, the model uses observed factor income shares from ABS 5206.0:
```
α_t = GOS_t / (GOS_t + COE_t)
```
Where GOS = Gross Operating Surplus and COE = Compensation of Employees. This captures the secular rise in capital's share from ~0.25 in the 1970s to ~0.35 today.

**Default**: Gaussian innovations (ε_t ~ N(0, σ)).

**SkewNormal variant** (`skewnormal_potential=True`): Zero-mean SkewNormal innovations with positive skew (α=1). The location is shifted negative so E[ε]=0 while ~75% of draws remain positive. This makes downward revisions to potential rarer than upward ones, reflecting that capital accumulation and productivity gains are hard to reverse.

---

## Observation Equations

### Phillips Curves (phillips.py)

By default, the Phillips curves use a **single time-invariant slope**. The `complex` variant (`regime_switching=True`) enables **regime-switching slopes** with three regimes:

| Regime | Period | Characteristic |
|--------|--------|----------------|
| Pre-GFC | Up to 2008Q3 | Moderate slope |
| GFC | 2008Q4 - 2020Q4 | Flat (anchored expectations) |
| Post-COVID | 2021Q1+ | Steep (reawakened sensitivity) |

Regime boundaries are fixed breakpoints chosen to align with well-known macroeconomic shifts (Lehman collapse, COVID reopening) rather than estimated, to avoid overfitting. See `equations/__init__.py` for definitions.

**Price Phillips Curve:**
```
π = quarterly(π_exp) + γ_regime × u_gap + controls + ε
```

Where `u_gap = (U - NAIRU) / U` (percentage deviation from NAIRU)

**Wage Phillips Curve (ULC):**
```
Δulc = α + γ_regime × u_gap + λ × (ΔU/U) + φ × Δ4dfd + θ × π_exp + ε
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

Simple form linking output gap to unemployment changes:

```
ΔU_t = β × OG_t + ε
```

Where:
- **β (beta_okun)**: Output gap effect (negative, ~-0.18)
- A positive output gap (economy above potential) leads to falling unemployment
- A negative output gap leads to rising unemployment

**Stage 3 applies demand transmission multiplier:**
```
ΔU_t = β × OG_t × 1.6
```
The 1.6 multiplier captures transmission channels not explicitly modeled (expectations, credit, business cash flow) to match RBA demand channel estimates.

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

**Units convention**: Output gap is percent (100 × log deviation from potential). Unemployment and NAIRU are percentage points. Interest rates are percent. DSR is percentage points of disposable income. Growth rates are quarterly percent (q/q) unless stated as annualised.

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

`SamplerConfig` dataclass defaults (in `base.py`):

```python
SamplerConfig(
    draws=100_000,     # Posterior samples per chain
    tune=5_000,        # Warmup/tuning samples
    chains=6,          # Independent chains
    cores=6,           # Parallel cores
    sampler="numpyro", # JAX-based NUTS
    target_accept=0.95,# Acceptance probability
    random_seed=42     # For reproducibility
)
```

`stage1.run_stage1()` uses a lighter configuration for faster iteration:

```python
SamplerConfig(
    draws=10_000,
    tune=3_500,
    chains=5,
    cores=5,
    target_accept=0.90,
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

## Stage 3: Scenario Analysis

Stage 3 generates **scenario analysis** (not forecasts) using the estimated coefficients from the Bayesian model. Two approaches are available:

| Approach | File | Use Case |
|----------|------|----------|
| **Stage 3a: Deterministic** | `stage3.py` | Clean scenario lines, policy communication |
| **Stage 3b: Monte Carlo** | `stage3_forward_sampling.py` | Full uncertainty, academic honesty |

**Why "scenario analysis" not "forecasting"?** The model shows what different policy choices would imply, all else equal. It cannot predict:
- Future supply shocks (oil, weather, supply chains)
- Global conditions (Fed policy, China)
- Fiscal policy changes
- Housing/credit surprises

### Stage 3a: Deterministic

Uses posterior **medians** for all coefficients and projects forward with **no sampled shocks**. The interpretation is "if nothing unexpected happens and we use our best-guess parameters."

- Clean scenario lines for policy communication
- 90% HDI captures parameter uncertainty only
- NAIRU and inflation expectations assumed fixed over scenario horizon

### Stage 3b: Monte Carlo Forward Sampling

**Important**: This is **not** a Bayesian model — it is **Monte Carlo simulation** using the posterior samples from the Stage 1 Bayesian estimation. No new inference is performed; instead, we mechanically propagate the estimated relationships forward with sampled shocks.

**What it does:**
1. Takes 5,000 random draws from the Stage 1 posterior
2. For each draw, extracts coefficient values and final state estimates
3. Samples future shocks from the estimated residual distributions
4. Propagates IS curve, Okun's Law, and Phillips curve equations forward mechanically
5. Aggregates results to produce distributions over scenarios

**What it is NOT:**
- Not a PyMC model with priors and likelihoods
- Not re-running MCMC or doing Bayesian inference
- Not proper posterior predictive sampling within PyMC

**Potential output assumption**: Potential growth uses "as-is" values — last quarter's capital growth, labour force growth, and MFP growth held constant over the scenario horizon. This reflects the assumption that supply-side dynamics don't respond to policy within 4 quarters.

**Key differences from deterministic:**
- Shock uncertainty included (Phillips curve, Okun's Law, IS curve residuals)
- Parameter uncertainty from posterior distribution
- Wider confidence bands reflecting honest uncertainty about future shocks

**Additional charts produced:**
- Output gap scenarios (all 9 policy scenarios)
- GDP vs Potential (shows actual vs potential output divergence)

**Result**: Medians align with deterministic version, but 90% HDI is much wider (e.g., inflation ±3-4pp vs ±1pp). This reflects the reality that even with known policy, future outcomes are highly uncertain.

### Scenario Horizon

The model produces 4-quarter ahead scenarios under a **move-and-hold** policy rule: the CB makes a one-off rate change at T, then holds that rate for the remainder of the horizon. The IS curve uses rate gaps lagged 2 periods:

| Scenario Period | Rate Gap Used | Source |
|-----------------|---------------|--------|
| T+1 | T-1 | Historical |
| T+2 | T | Policy decision |
| T+3 | T+1 | Hold assumption |
| T+4 | T+2 | Hold assumption |

### Scenario Equations

The scenario projection uses estimated coefficients and RBA-calibrated pass-throughs:

1. **IS Curve** (output gap dynamics with DSR transmission):
   ```
   y_gap_t = ρ_is × y_gap_{t-1} - β_is × rate_gap_{t-2} - δ_dsr × Δdsr_{t-1}
   ```
   Where Δdsr is projected using the rate→DSR pass-through (~1.0pp per 100bp).

2. **Okun's Law** (simple form):
   ```
   ΔU_t = β_okun × y_gap_t × 1.6
   ```
   - A demand transmission multiplier of 1.6 is applied to β_okun to match RBA's demand channel estimate
   - See [Why Not Error Correction Okun's Law?](#why-not-error-correction-okuns-law) for rationale

3. **Phillips Curve** (inflation from unemployment gap + FX channel):
   ```
   π_t = π_exp + γ_covid × (U_t - NAIRU_t) / U_t + FX_effect_t
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

The "move and hold" assumption means the rate stays at the new level for the entire scenario horizon.

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

These corrections are applied only in Stage 3 scenario analysis, not in Stage 1 estimation. The estimated model remains unchanged and can be used for historical decomposition without calibration adjustments.

### ScenarioResults Container

```python
from src.models.nairu.model import run_stage3, ForecastResults

# Run all scenarios
scenario_results = run_stage3()  # dict[str, ForecastResults]

# Access specific scenario
hold = scenario_results["hold"]

# Point estimates (posterior median)
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

### Model Variants

The `--variant` flag selects between model configurations:

```bash
python -m src.models.nairu.model -v --variant simple    # Core equations only
python -m src.models.nairu.model -v --variant complex   # All features enabled
python -m src.models.nairu.model -v --variant both      # Run both + comparison chart
python -m src.models.nairu.model -v                     # Default configuration
```

| Variant | NAIRU Innovations | Regime Switching | Extra Equations | Import Price Control |
|---------|-------------------|------------------|-----------------|---------------------|
| `default` | Gaussian | No | — | No |
| `simple` | Gaussian | No | — | No |
| `complex` | Student-t(ν=4) | Yes (3 regimes) | Exchange rate, import price, participation, employment, net exports | Yes |

`default` and `simple` have the same model configuration — `simple` only differs in using a separate output prefix (`nairu_simple`) and chart directory. The `complex` variant enables all optional features. Running `both` runs `simple` and `complex` then produces a NAIRU comparison chart.

**Default equations** (used by `default` and `simple`): Phillips curves (price + ULC), hourly COE Phillips curve, Okun's Law, IS curve, with GSCPI as an import price control variable.

**Additional equations** enabled by `complex`: exchange rate (UIP-style TWI), import price pass-through, participation (discouraged worker), employment (labour demand), net exports.

### CLI Arguments

| Flag | Choices | Default | Description |
|------|---------|---------|-------------|
| `-v, --verbose` | — | off | Print detailed output |
| `--variant` | `default`, `simple`, `complex`, `both` | `default` | Model variant |
| `--anchor` | `expectations`, `target`, `rba` | `rba` | Expectations anchor mode |
| `--skip-forecast` | — | off | Skip Stage 3 (scenario analysis) |

### Stage 1 Only (Sampling)
```bash
python -m src.models.nairu.stage1 --start 1980Q1 -v
```

### Stage 2 Only (Analysis)
```bash
python -m src.models.nairu.stage2 --show -v
```

### Stage 3a Only (Deterministic)
```bash
python -m src.models.nairu.stage3
python -m src.models.nairu.stage3 --no-scenarios    # Skip scenario table
python -m src.models.nairu.stage3 --no-plots        # Skip chart generation
```

### Stage 3b Only (Monte Carlo Forward Sampling)
```bash
python -m src.models.nairu.stage3_forward_sampling
python -m src.models.nairu.stage3_forward_sampling --n-samples 10000
python -m src.models.nairu.stage3_forward_sampling --no-plots
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

### Why Regime-Switching Phillips Curves? (Complex Variant)

The `complex` variant enables regime-switching Phillips curves. Post-GFC flattening of the Phillips curve is well-documented. Three regimes capture:

- **Pre-GFC**: Traditional Phillips curve relationship
- **GFC era**: Anchored expectations, flat curve
- **Post-COVID**: Reawakened inflation sensitivity

The default and `simple` variants use a single time-invariant slope.

### Why SkewNormal for Potential? (Optional Variant)

The SkewNormal variant (`skewnormal_potential=True`) makes potential output innovations asymmetric (small positives common, large negatives rare):
- Capital doesn't disappear in recessions
- Productivity gains are hard to reverse
- Labour force grows over time

The default model uses Gaussian innovations. The SkewNormal variant is available for sensitivity analysis.

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
- **Implementation**: MFP is derived from wage data using the Solow residual identity (LP = Δhcoe - Δulc; MFP = LP - α×(g_K - g_L)), then HP-filtered (λ=1600) to extract the trend, and floored with `np.maximum(mfp_trend, 0)`. This replaces direct sourcing from ABS 5204.0 MFP data.

**Implicit assumption**: Structural technological progress is non-negative; measured negative MFP is treated as cyclical or mismeasurement.

**Comparison with external forecasters**: Most economists assume the post-GFC and post-COVID productivity slowdown is more cyclical than structural, and therefore factor in some MFP growth on a return-to-trend basis. Indicative comparison (approximate, varies by vintage):

| Source | MFP Assumption | Potential Growth |
|--------|---------------|------------------|
| This model | 0.0% | ~1.8% |
| RBA | ~0.2% | ~2.0% |
| Treasury / Private banks | ~0.4-0.5% | ~2.25% |

The model takes the HP-filtered trend at face value. If productivity recovers, this assumption will prove too pessimistic; if the productivity drought continues, the model will have been more realistic than consensus.

### Why Smooth Labour Inputs During COVID?

Labour force growth and hours worked are replaced with Henderson-smoothed values during 2020Q1–2023Q2 (`src/data/observations.py`):

- **COVID caused measurement artifacts, not structural breaks**: Lockdowns produced extreme swings in measured labour force participation and hours that don't reflect genuine changes in labour supply capacity.
- **Potential output should be smooth through temporary shocks**: The economy's productive capacity didn't actually collapse and recover in a few quarters — workers were temporarily unable to work, not permanently removed from the labour force.
- **Prevents contaminating NAIRU estimates**: Without smoothing, the model would interpret COVID unemployment spikes as movements in NAIRU rather than cyclical deviations from it.
- **Implementation**: Raw series used normally; Henderson MA (term=13) applied only during 2020Q1–2023Q2.

### Why HMA(13) Smooth Labour Force Growth for Potential?

Labour force growth is Henderson-smoothed (term=13) before entering the potential output calculation (`src/models/nairu/stage1.py`):

- **Distinguishes structural from cyclical**: Raw quarterly labour force growth is noisy. HMA(13) extracts the underlying trend while remaining responsive to genuine structural shifts — unlike HP filter which anchors too heavily to historical averages.
- **Captures the post-2023 immigration slowdown**: Labour force growth collapsed from ~3.0% (2023, post-COVID surge) to ~1.6% (2025) as net overseas migration normalized and policy shifted. This is structural, not cyclical — it reflects deliberate policy choices, not temporary economic weakness.
- **Affects r\* and potential output directly**: Since r\* ≈ α×g_K + (1-α)×g_L + g_MFP, the labour force slowdown materially reduces potential growth. Using raw data would be too noisy; using HP trend would miss the structural break.
- **Implementation**: Data layer provides raw series; model applies `hma(lf_growth, 13)` after loading observations. This keeps data preparation pure and makes the modeling choice explicit.

| Approach | LF Growth (2025) | r\* Implied | Issue |
|----------|-----------------|-------------|-------|
| Raw quarterly | 1.6% (volatile) | ~1.8% | Too noisy for potential |
| HP trend | 2.4% | ~2.2% | Anchored to 2023 surge |
| HMA(13) | ~2.0% | ~2.0% | Captures structural shift |

**Key insight**: The current slowdown in labour force growth is critical to understanding why potential output growth is lower than many forecasters assume. Immigration-driven workforce expansion was exceptional in 2022–23; the new normal is materially slower.

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

### Why Not Error Correction Okun's Law?

An error correction form was considered:

```
ΔU_t = β × OG_t + α × (U_{t-1} - NAIRU_{t-1} - γ × OG_{t-1}) + ε
```

**Why it was rejected:**

1. **Lucas Critique**: The error correction term captures mean reversion toward NAIRU, but we cannot separately identify whether this reflects:
   - Natural labor market adjustment (structural forces)
   - CB policy response (RBA actively steering U toward U*)

   These are observationally equivalent in historical data where the CB was always active. The estimated α captures their *combined* effect.

2. **Pre-inflation targeting period**: Using EC implicitly assumes NAIRU was a policy target throughout the sample, but the RBA only adopted inflation targeting in 1993. Before that, the mean reversion interpretation is unclear.

3. **Simpler is cleaner**: The simple form `ΔU_t = β × OG_t + ε` has one parameter instead of three, reducing estimation uncertainty without sacrificing fit.

**The simple form used:**
```
ΔU_t = β × OG_t + ε
```

Output gap affects unemployment mechanically. The 1.6 demand multiplier applied in Stage 3 scenarios is calibrated to match RBA transmission estimates.

### Why Use a Signal Extraction Model for Expectations?

Rather than imposing a mechanical transition from backward-looking expectations to the 2.5% target, we estimate expectations directly using Bayesian signal extraction models. This approach:

- **Lets the data speak**: The transition from high/volatile expectations (1980s) to anchored expectations (~2.5% post-1998) emerges from the estimation rather than being imposed
- **Captures uncertainty**: The signal extraction model provides credible intervals on expectations, which propagate through to NAIRU uncertainty
- **Uses multiple measures**: Combines surveys (NAB business, market economists), market-based measures (breakeven inflation), and early-period proxies (headline CPI, nominal bonds)
- **Handles regime changes**: Student-t innovations allow the model to capture sharp shifts during the 1988-1992 disinflation while remaining smooth in the targeting era

See `src/models/expectations/MODEL_NOTES.md` for full details on the four expectations models (Target Anchored, Unanchored, Short Run, Long Run).

### Expectations Series

The NAIRU model supports multiple sources for inflation expectations in the Phillips curve. The default uses RBA's PIE_RBAQ series; alternatives use the signal extraction model output.

**Implementation**:
- RBA expectations: `src/data/expectations_rba.py` loads from `input_data/PIE_RBAQ.CSV`
- Model expectations: `src/data/expectations_model.py` loads from `output/expectations/expectations_target_hdi.parquet`

### Anchor Modes for Expectations

The model supports three anchor modes via the `--anchor` CLI argument:

| Mode | Source | Description | Use Case |
|------|--------|-------------|----------|
| `rba` (default) | RBA PIE_RBAQ | Use RBA series to 1992Q4, phase to 2.5% by 1998Q4, then target | Policy analysis |
| `target` | Signal extraction model | Use model expectations to 1992Q4, phase to 2.5% by 1998Q4, then target | Policy analysis (model-based) |
| `expectations` | Signal extraction model | Use full Target Anchored series as-is | Historical analysis |

**`rba` mode (default)**: Uses RBA's official inflation expectations series (PIE_RBAQ), which is derived from surveys and market measures. The expectations path is:

```
1970Q1 – 1992Q4:  RBA PIE_RBAQ (survey/market-based)
1993Q1 – 1998Q4:  Linear phase from PIE_RBAQ → 2.5% target
1999Q1 onwards:   2.5% inflation target
```

This is the default because it uses the RBA's own expectations measure, providing consistency with official analysis.

**`target` mode**: Same phase-in structure as `rba`, but uses the signal extraction model's Target Anchored series instead of PIE_RBAQ. The model series combines multiple survey measures (NAB business, market economists), market-based measures (breakeven inflation), and early-period proxies with bias corrections.

**`expectations` mode**: Uses the full Target Anchored series from the signal extraction model without phasing to the target. NAIRU is defined relative to actual estimated expectations. This is useful for historical analysis — understanding what NAIRU was at the time given prevailing expectations — but less relevant for forward-looking policy.

The phasing period (1993-1998) reflects the RBA's gradual establishment of inflation targeting credibility. By 1998Q4, expectations were effectively anchored at the 2-3% band midpoint.

**NAIRU interpretation**:
- With `rba` or `target` anchor: NAIRU is the unemployment rate where inflation equals the 2.5% target
- With `expectations` anchor: NAIRU is the unemployment rate where inflation equals estimated expectations at that time
