# DSGE Models Overview

This directory contains several DSGE-style models for estimating Phillips curve parameters and NAIRU for Australia. The models evolved through experimentation to address identification challenges.

## Summary Table

| Model | Key Feature | Phillips Curve | NAIRU | Recommended Use |
|-------|-------------|----------------|-------|-----------------|
| NK DSGE | Taylor rule | κ_p × y_gap | No | Not recommended (Taylor rule fights Phillips) |
| HLW | r* replaces Taylor | κ_p × y_gap | No | Phillips curve estimation by regime |
| HLW-NAIRU | r* + NAIRU latent | γ × (U-NAIRU)/U | Yes (latent) | Not recommended (over-parameterized) |
| NAIRU-Phillips | Pure Phillips | γ × (U-NAIRU)/U | Yes (latent) | Simple NAIRU, but estimates too high |
| HLW-NAIRU-Phillips | r* + NAIRU, U exogenous | γ × (U-NAIRU)/U | Yes (latent) | Results not plausible |

---

## File Structure

### Core Files
```
shared.py                    # Shared utilities (regimes, MLE wrapper, printing)
kalman.py                    # Standard Kalman filter/smoother
solver.py                    # Blanchard-Kahn RE solver
data_loader.py               # Common data loading, inflation anchor
```

### NK DSGE Model
```
nk_model.py                  # 3-equation NK model
estimation.py                # MLE estimation
regime_switching.py          # Regime-switching estimation
```

### HLW Model (Holston-Laubach-Williams style)
```
hlw_model.py                 # HLW with latent r*
hlw_estimation.py            # MLE estimation
```

### HLW-NAIRU Model
```
hlw_nairu_model.py           # HLW extended with NAIRU
hlw_nairu_estimation.py      # MLE estimation
```

### NAIRU-Phillips Model
```
nairu_phillips_model.py      # Pure Phillips curves with NAIRU
nairu_phillips_estimation.py # MLE estimation, regime-switching
plot_nairu_phillips.py       # Visualization
```

### HLW-NAIRU-Phillips Model (Model 5)
```
hlw_nairu_phillips_model.py       # Combined HLW + NAIRU-Phillips
hlw_nairu_phillips_estimation.py  # MLE estimation, regime-switching
plot_hlw_nairu_phillips.py        # Visualization
```

---

## 1. NK DSGE Model

**Structure:**
- Standard 3-equation New Keynesian model
- IS curve: ŷ = E[ŷ'] - σ(i - E[π'] - r*) + ε_demand
- Phillips curve: π = β×E[π'] + κ_p×ŷ + ε_supply
- Taylor rule: i = ρ_i×i_{-1} + (1-ρ_i)[φ_π×π + φ_y×ŷ] + ε_monetary
- Solved via Blanchard-Kahn (1980)

**Identification Issue:**
When the interest rate is observed, the Taylor rule "fights" the Phillips curve for identification. κ_p tends to hit its lower bound while φ_π estimates become implausibly high.

---

## 2. HLW Model

**Structure:**
- Replaces Taylor rule with latent r* (natural rate)
- IS curve: ŷ = ρ_y×ŷ_{-1} - β_r×(r_{-1} - r*) + ε_demand
- Phillips curve: π = κ_p×ŷ + ρ_m×Δpm + ε_supply
- Wage Phillips: π_w = κ_w×ŷ + ε_wage
- Interest rate is exogenous (not modeled)

**Why It Works:**
Treating monetary policy as exogenous avoids Taylor rule identification issues. κ_p is well-identified and shows meaningful variation across regimes.

**Latest Results (Regime-Switching):**
```
              Pre-GFC    GFC-COVID   Post-COVID
κ_p            0.043       0.031       0.238
κ_w            0.500       0.127       0.500
ρ_m            0.006       0.000       0.167
```

---

## 3. HLW-NAIRU Model

**Structure:**
- Extends HLW with NAIRU as additional latent state
- States: [ŷ, r*, NAIRU, ε_s, ε_w]
- Phillips: π = γ_p×(U - NAIRU)/U + ε_s (normalised u_gap)

**Identification Issue:**
Too many latent states (ŷ, r*, NAIRU) for available data. Parameters hit bounds due to competing dynamics between IS curve and Okun's law.

---

## 4. NAIRU-Phillips Model

**Structure:**
- Simplest model: just Phillips curves with NAIRU
- No IS curve, no r*, no output gap dynamics
- States: [NAIRU, ε_p, ε_w]
- NAIRU: random walk
- Price Phillips: π = γ_p×(U - NAIRU)/U + ρ_m×Δpm + ξ_oil×Δoil + ξ_coal×Δcoal + ε_p
- Wage Phillips: Δulc = γ_w×(U - NAIRU)/U + λ_w×(ΔU/U) + ε_w

**Key Features:**
- Normalised u_gap = (U - NAIRU)/U (RBA style)
- ULC growth for wages (better signal than raw compensation)
- Speed limit term (λ_w): ΔU/U captures rapid unemployment changes
- Energy prices: Oil and coal pass-through to prices
- RTS Kalman smoother for improved early-sample estimates

**Observables:**
- [π (anchor-adjusted), Δulc]
- U is exogenous (enters Phillips directly)

**Latest Results (Regime-Switching):**
```
              Pre-GFC    GFC-COVID   Post-COVID
γ_p           -0.303      -1.923      -1.354
γ_w           -2.153      -0.010      -2.150
ρ_m            0.000       0.005       0.165
λ_w           -7.737      -6.136      -2.201
ξ_oil          0.003       0.000       0.014
ξ_coal         0.008       0.000       0.000
```

**Current NAIRU Estimate:** ~6.2% (higher than RBA's ~4-4.5%)

**Known Issues:**
1. NAIRU prior set to mean(U) per regime pulls estimates too high
2. Weak Phillips curve (γ_p ~ -0.3 pre-GFC) provides little NAIRU identification
3. Early sample estimates unreliable even with smoother

---

## 5. HLW-NAIRU-Phillips Model

**Structure:**
- Combines best features of HLW and NAIRU-Phillips
- States: [r*, NAIRU, ε_p, ε_w]
- r* dynamics: r*_t = r*_{t-1} + ε_rstar (random walk)
- NAIRU dynamics: NAIRU_t = NAIRU_{t-1} + ε_nairu (random walk)
- Price Phillips: π = γ_p×(U - NAIRU)/U + ρ_m×Δpm + ξ_oil×Δoil + ε_p
- Wage Phillips: Δulc = γ_w×(U - NAIRU)/U + λ_w×(ΔU/U) + ε_w

**Key Design:**
U is exogenous (not observed), which forces the model to use Phillips curves for
NAIRU identification. This avoids the trap where NAIRU just tracks U directly.
r* is initialized to regime mean real rate and evolves as random walk.

**Observables:**
- [π (anchor-adjusted), Δulc] — only 2 observables

**Exogenous inputs:**
- U: Unemployment rate (enters Phillips curves directly)
- r: Real interest rate (used to initialize r* prior)

**Latest Results (Regime-Switching):**
```
              Pre-GFC    GFC-COVID   Post-COVID
γ_p           -0.303      -1.923      -1.353
γ_w           -2.153      -0.010      -2.149
ρ_m            0.000       0.005       0.165
λ_w           -7.737      -6.136      -2.199
ξ_oil          0.003       0.000       0.014
ξ_coal         0.008       0.000       0.000
```

**Current Estimates:**
- r* = -1.4%
- NAIRU = 6.2%
- r - r* = +2.5pp (restrictive policy)
- U - NAIRU = -1.8pp (tight labor market)

**Why This Design:**
1. U exogenous → NAIRU identified from Phillips curves, not from tracking U
2. Same identification as NAIRU-Phillips (model 4) but with r* added
3. Produces meaningful u-gap variation over time
4. γ_p matches model 4 results (e.g., -0.303 in Pre-GFC)

**Known Issues - Results Not Plausible:**
- Regime-switching causes artificial discontinuities in NAIRU and r* at regime breaks
- NAIRU too flat within regimes, jumps between regimes when prior resets
- r* essentially constant within regimes
- Latent state paths are not economically sensible
- For NAIRU estimation, the Bayesian state-space model in `src/models/nairu/` produces more plausible results

---

## Shared Utilities (`shared.py`)

Common functionality extracted for DRY code:

```python
# Regime definitions
REGIMES = [
    ("Pre-GFC", "1984Q1", "2008Q3"),
    ("GFC-COVID", "2008Q4", "2020Q4"),
    ("Post-COVID", "2021Q1", None),
]

# Helper functions
ensure_period_index(series)      # Ensure PeriodIndex
filter_date_range(df, start, end)  # Date filtering
print_regime_results(...)        # Standardized result printing
estimate_mle(...)                # Generic MLE wrapper
check_bounds(...)                # Parameter bound checking
```

---

## Inflation Anchor

All models use anchor-adjusted inflation: π - π_anchor

The anchor transitions from backward-looking to target-anchored:
- Pre-1993: α = 1.0 (fully backward-looking)
- 1993-1998: α fades linearly from 1.0 to 0.2
- Post-1998: α = 0.2 (mostly anchored to 2.5% target)

Formula: π_anchor = α × π_{t-1} + (1 - α) × 2.5

---

## Recommendations

### For Phillips Curve Estimation
Use **HLW model** with regime-switching:
- κ_p well-identified across regimes
- Shows flattening (GFC-COVID) and steepening (Post-COVID)
- Doesn't require NAIRU estimation

### For NAIRU Estimation
The **Bayesian state-space model** in `src/models/nairu/` is more reliable:
- Joint estimation with output gap
- Proper uncertainty quantification
- Better identified through multiple equations

### NAIRU-Phillips Model Improvements Attempted
1. ✓ Speed limit (ΔU/U) - helps capture rapid U changes
2. ✓ Energy prices (oil, coal) - minimal effect
3. ✓ RTS Kalman smoother - helps but doesn't fix core issue
4. Pending: Lower NAIRU prior, tighter initial covariance

---

## References

- Holston, Laubach, Williams (2017): "Measuring the Natural Rate of Interest"
- Blanchard & Kahn (1980): "The Solution of Linear Difference Models under Rational Expectations"
- RBA (various): Phillips curve and NAIRU research
