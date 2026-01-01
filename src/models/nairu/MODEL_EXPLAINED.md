# NAIRU + Output Gap Model Overview

This directory contains a Bayesian state-space model for jointly estimating NAIRU (Non-Accelerating Inflation Rate of Unemployment), potential output, and output gaps for Australia using PyMC.

## Summary

| Component | Method | Key Feature |
|-----------|--------|-------------|
| NAIRU | Gaussian random walk | No drift, σ ≈ 0.25 |
| Potential Output | Cobb-Douglas + innovations | SkewNormal innovations (sticky downwards) |
| Phillips Curves | Regime-switching (3 regimes) | Pre-GFC, GFC-COVID, post-COVID |
| Identification | 9 observation equations | Joint estimation with proper uncertainty |

**This model works well.** The Bayesian approach with multiple observation equations provides good identification and plausible estimates with uncertainty quantification.

---

## File Structure

### Core Pipeline
```
model.py               # Unified entry point (run_model, main)
stage1.py              # Build model, sample posterior, save results
stage2.py              # Load results, diagnostics, plotting
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
│   main()      → Full pipeline: Stage 1 → Stage 2                │
└──────────────────────────────┬──────────────────────────────────┘
                               │
              ┌────────────────┴────────────────┐
              ▼                                 ▼
┌─────────────────────────┐       ┌─────────────────────────┐
│       stage1.py         │       │       stage2.py         │
│                         │       │                         │
│  build_observations()   │       │  load_results()         │
│  build_model()          │       │  NAIRUResults container │
│  sample_model()         │       │  test_theoretical_expectations() │
│  save_results()         │       │  plot_all()             │
└─────────────────────────┘       └─────────────────────────┘
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

---

## State Equations

### NAIRU (state_space.py)

Gaussian random walk without drift:

```
NAIRU_t = NAIRU_{t-1} + ε_t,  ε_t ~ N(0, σ²)
```

- **σ (nairu_innovation)**: Typically fixed at 0.25 for smoothness
- **Initial distribution**: N(15, 8) — diffuse prior centered on historical average
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

**Key feature**: SkewNormal innovations with negative skew make potential "sticky downwards" — it's easier to grow than shrink, reflecting that capital accumulation and productivity gains are hard to reverse.

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

Output gap persistence with interest rate channel:

```
y_gap_t = ρ × y_gap_{t-1} - β × (r_{t-2} - r*) + γ × fiscal_impulse_{t-1} + ε
```

Where:
- `ρ` = output gap persistence (~0.85)
- `β` = interest rate sensitivity (positive, ~0.2)
- `r - r*` = real rate gap (policy rate minus neutral rate)
- `fiscal_impulse` = government spending growth minus GDP growth

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

### From Python
```python
from src.models.nairu import run_model, NAIRUResults

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
2. **Better identification**: Multiple observation equations (9 total) constrain the latent states
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

Negative skew in innovations means large downward movements are less likely than upward ones.

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
