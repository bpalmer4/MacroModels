# DSGE Models Overview

This directory contains state-space models for estimating latent macroeconomic variables (NAIRU, r*, output gap) for Australia. **Only the NK model is a true DSGE** (forward-looking, solved via Blanchard-Kahn). The others are backward-looking unobserved components models.

## Summary Table

| Model | Type | Phillips Curve | Outputs | Status |
|-------|------|----------------|---------|--------|
| NK | True DSGE (Blanchard-Kahn) | κ_p × y_gap | Output gap | Many params at bounds |
| HLW | State-space | κ_p × y_gap | Output gap, r* | Reasonable estimates |
| HLW-NAIRU | State-space | γ × (U-NAIRU)/U | Output gap, r*, NAIRU | Over-parameterized |
| NAIRU-Phillips | State-space | γ × (U-NAIRU)/U | NAIRU | NAIRU too high |
| HLW-NAIRU-Phillips | State-space | γ × (U-NAIRU)/U | r*, NAIRU | Not plausible |

**None of these models work particularly well.** This is common for state-space macro models due to identification issues, short samples, and structural breaks.

---

## File Structure

### Core Infrastructure
```
estimation.py          # Generic MLE with ModelSpec, two-stage estimation
kalman.py              # Kalman filter/smoother (time-invariant and time-varying)
solver.py              # Blanchard-Kahn rational expectations solver
data_loader.py         # Common data loading, inflation anchor
shared.py              # Utilities (date filtering, bound checking)
```

### Generic Plotting
```
plot_output_gap.py     # Output gap visualization
plot_rstar.py          # r* visualization
plot_nairu.py          # NAIRU visualization
```

### Models (each contains SPEC for generic estimation)
```
nk_model.py                   # New Keynesian (true DSGE)
hlw_model.py                  # Holston-Laubach-Williams style
hlw_nairu_model.py            # HLW extended with NAIRU
nairu_phillips_model.py       # Pure Phillips curves with NAIRU
hlw_nairu_phillips_model.py   # Combined HLW + NAIRU-Phillips
```

---

## Architecture

### Component Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        Individual Models                         │
│  (nk_model.py, hlw_model.py, nairu_phillips_model.py, etc.)     │
│                                                                  │
│  Each model provides:                                            │
│    - load_*_data()      → Model-specific data preparation        │
│    - *_log_likelihood() → Likelihood function for MLE            │
│    - *_extract_states() → Kalman smoother for state extraction   │
│    - *_SPEC (ModelSpec) → Configuration for generic estimation   │
└──────────────────────────────┬──────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Generic Infrastructure                        │
│                                                                  │
│  estimation.py:                                                  │
│    - ModelSpec dataclass (bounds, params, likelihood, extractor)│
│    - estimate_two_stage() orchestrates full pipeline             │
│    - estimate_mle() runs scipy.optimize.minimize                 │
│                                                                  │
│  kalman.py:                                                      │
│    - kalman_filter() for likelihood computation                  │
│    - kalman_smoother() for time-invariant observation eq         │
│    - kalman_smoother_tv() for time-varying observation eq        │
│                                                                  │
│  solver.py:                                                      │
│    - blanchard_kahn() for forward-looking DSGE (NK only)         │
│                                                                  │
│  data_loader.py:                                                 │
│    - load_common_data() fetches ABS series                       │
│    - compute_inflation_anchor() handles regime transition        │
└──────────────────────────────┬──────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Generic Plotting                            │
│                                                                  │
│  plot_output_gap.py  → plot_output_gap(series, model_name)      │
│  plot_rstar.py       → plot_rstar(series, model_name)           │
│  plot_nairu.py       → plot_nairu(series, unemployment, name)   │
└─────────────────────────────────────────────────────────────────┘
```

### Data Flow

1. **Data Loading**: Each model has `load_*_data()` that calls `data_loader.load_common_data()` then adds model-specific transformations (e.g., constructing observation matrices)

2. **Estimation**: `estimate_two_stage()` orchestrates:
   - Filters data to exclude crisis period (2008Q4-2020Q4)
   - Calls `estimate_mle()` with the model's likelihood function
   - Returns estimated parameters

3. **State Extraction**: Model's `*_extract_states()` function:
   - Takes estimated parameters + full data
   - Runs Kalman smoother (from `kalman.py`)
   - Returns DataFrame with latent states (output_gap, r*, NAIRU, etc.)

4. **Plotting**: Generic plotting functions take extracted state series and produce standardized charts

### Adding a New Model

To add a new model, create a file with:

```python
# 1. Data loading
def load_mymodel_data(start, end, ...) -> dict:
    data = load_common_data(start, end)
    # Add model-specific preparation
    return {"y": observations, "dates": dates, ...}

# 2. Likelihood function
def compute_mymodel_log_likelihood(y, params, ...) -> float:
    # Build state-space matrices from params
    # Run kalman_filter() and return log_likelihood
    ...

# 3. State extractor
def mymodel_extract_states(params, data) -> dict:
    # Run kalman_smoother() with estimated params
    # Return dict of state series
    ...

# 4. ModelSpec configuration
MYMODEL_SPEC = ModelSpec(
    name="MyModel",
    description="...",
    param_class=MyModelParams,
    param_bounds={...},
    estimate_params=[...],
    fixed_params={...},
    likelihood_fn=_mymodel_likelihood,
    state_extractor_fn=mymodel_extract_states,
)

# 5. Main block for standalone execution
if __name__ == "__main__":
    result = estimate_two_stage(MYMODEL_SPEC, load_mymodel_data)
    plot_output_gap(result.states["output_gap"], "MyModel")
```

---

## Estimation Approach

### Two-Stage Estimation

All models use two-stage estimation to handle structural breaks:

1. **Stage 1**: Estimate parameters excluding crisis period (2008Q4-2020Q4)
2. **Stage 2**: Run Kalman smoother on full sample with fixed parameters

This avoids distortions from GFC and COVID while still extracting states for the full period.

```python
from estimation import estimate_two_stage
result = estimate_two_stage(MODEL_SPEC, load_data_fn)
```

### ModelSpec Pattern

Each model defines a `ModelSpec` with:
- Parameter bounds and which to estimate vs fix
- Likelihood function
- State extractor function for Kalman smoothing

---

## 1. NK Model (True DSGE)

**The only forward-looking model, solved via Blanchard-Kahn.**

**Structure:**
- IS curve: ŷ = E[ŷ'] - σ(i - E[π'] - r*) + ε_demand
- Phillips curve: π = β×E[π'] + κ_p×ŷ + ε_supply
- Taylor rule: i = ρ_i×i_{-1} + (1-ρ_i)[φ_π×π + φ_y×ŷ] + ε_monetary

**Issues:**
- 5 of 10 estimated parameters hit bounds
- Taylor rule fights Phillips curve for identification

---

## 2. HLW Model

**Backward-looking state-space model.**

**Structure:**
- IS curve: ŷ = ρ_y×ŷ_{-1} - β_r×(r_{-1} - r*) + ε_demand
- Phillips curve: π = κ_p×ŷ + ρ_m×Δpm + ε_supply
- r* and g (trend growth) as latent random walks

**Why it works better:**
Interest rate exogenous avoids Taylor rule identification issues.

---

## 3. HLW-NAIRU Model

**HLW extended with NAIRU as additional latent state.**

**Structure:**
- Same as HLW plus NAIRU random walk
- Phillips: π = γ_p×(U - NAIRU)/U + ε_s

**Issues:**
Too many latent states for available data.

---

## 4. NAIRU-Phillips Model

**Simplest model: just Phillips curves with NAIRU.**

**Structure:**
- NAIRU: random walk
- Price Phillips: π = γ_p×(U - NAIRU)/U + ρ_m×Δpm + ξ_oil×Δoil + ξ_coal×Δcoal + ε_p
- Wage Phillips: Δulc = γ_w×(U - NAIRU)/U + λ_w×(ΔU/U) + ε_w

**Issues:**
NAIRU estimates too high (~6%) vs RBA's ~4-4.5%.

---

## 5. HLW-NAIRU-Phillips Model

**Combines HLW r* with NAIRU-Phillips.**

**Structure:**
- r* and NAIRU as random walks
- Phillips curves as in NAIRU-Phillips
- U exogenous (not observed)

**Issues:**
Results not economically plausible.

---

## Inflation Anchor

All models use anchor-adjusted inflation: π - π_anchor

- Pre-1993: α = 1.0 (backward-looking)
- 1993-1998: α fades from 1.0 to 0.2
- Post-1998: α = 0.2 (anchored to 2.5% target)

Formula: π_anchor = α × π_{t-1} + (1 - α) × 2.5

---

## Recommendations

For reliable NAIRU/output gap estimation, consider the **Bayesian state-space model** in `src/models/nairu/` which provides:
- Joint estimation with proper uncertainty
- Better identification through multiple equations
- More plausible estimates

---

## References

- Holston, Laubach, Williams (2017): "Measuring the Natural Rate of Interest"
- Blanchard & Kahn (1980): "The Solution of Linear Difference Models under Rational Expectations"
