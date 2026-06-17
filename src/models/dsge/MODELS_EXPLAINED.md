# DSGE Models Overview

This directory contains state-space models for estimating latent macroeconomic variables (NAIRU, r*, output gap) for Australia, plus forward-looking DSGEs. **Four models are true DSGEs** (forward-looking, solved via Blanchard-Kahn): the original `NK`, the `NK-TwoStar` linear probe, the financial-accelerator `FA-NK`, and `FA-NK-wage` (FA-NK + sticky wages + Galí unemployment). The HLW-family models are backward-looking unobserved-components models.

The DSGE work culminates in `FA-NK` (`fa_nk_model.py`), a financial-accelerator model that carries **two natural rates** - a goods-market safe rate and a return on capital - with an **endogenous external-finance-premium wedge** between them. This is the structural answer to the "great divergence" question: post-GFC the safe rate and the return on capital decoupled, and a one-r* model cannot represent that. See `nk_twostar_model.py` for the reduced-form probe that motivated the proper build.

Two extensions: a **Tier-1 labour block** (structural marginal cost, behind `FANKModel(labour_block=True)`) and **`FA-NK-wage`** (`fa_nk_wage_model.py`), which adds sticky wages and Galí-2011 unemployment. Key honest findings across the family: the financial wedge is essentially an exogenous shock that tracks the **global financial cycle** (not QE narrowly - see `omega_global_test.py`); the labour block **unsticks κ_p**; the **Taylor block is weakly identified** (φ_π pins at its cap unless unemployment is observed); and a **credible NAIRU is out of reach** here (no NAIRU random-walk state - use `src/models/nairu/`).

## Summary Table

| Model | Type | Phillips Curve | Outputs | Status |
|-------|------|----------------|---------|--------|
| **FA-NK** | True DSGE + financial accelerator | κ_p × y_gap (or structural mc if labour_block) | Two r* (safe, capital), endogenous EFP wedge | Determinate, estimable; χ small (wedge ~exogenous), Taylor block weakly identified |
| **FA-NK-wage** | FA-NK + sticky wages + Galí unemployment | κ_p × mc, κ_w × μ_w | + unemployment gap, real wage, U* | Determinate; best-identified (κ_w found, φ_y interior); but NAIRU not credible |
| NK-TwoStar | True DSGE + reduced-form wedge | κ_p × y_gap | Output gap, bolted-on wedge | Probe only; wedge has signal (σ_k≈0.14), φ_π at determinacy floor |
| NK | True DSGE (Blanchard-Kahn) | κ_p × y_gap | Output gap | Many params at bounds |
| HLW | State-space | κ_p × y_gap | Output gap, r* | Reasonable estimates |
| HLW-NAIRU | State-space | γ × (U-NAIRU)/U | Output gap, r*, NAIRU | Over-parameterized |
| NAIRU-Phillips | State-space | γ × (U-NAIRU)/U | NAIRU | NAIRU too high |
| HLW-NAIRU-Phillips | State-space | γ × (U-NAIRU)/U | r*, NAIRU | Not plausible |

**The HLW-family and original NK models do not work particularly well** - common for state-space macro models given identification issues, short samples, and structural breaks. `FA-NK` is the deliberate proper rebuild: it is determinate and estimable and reproduces the divergence, though it has its own open issues (see below).

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
fa_nk_model.py                # Financial-accelerator NK DSGE (two r*, EFP wedge; labour_block flag)
fa_nk_wage_model.py           # FA-NK + sticky wages + Galí (2011) unemployment / U*
omega_global_test.py          # Tests whether the financial shock ω tracks global QE / FCI
nk_twostar_model.py           # NK DSGE + reduced-form wedge (linear probe)
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

## 0a. FA-NK Model (Financial-Accelerator DSGE) — the proper "two r*" model

**Run:** `uv run python -m src.models.dsge.fa_nk_model` (determinacy check → estimate → smoothed series → save + plot). Writes `model_outputs/fa_nk_states.csv`, `model_outputs/fa_nk_params.txt`, and seven charts to `charts/dsge-fa-nk/`: actual safe rate vs cost of capital vs natural rate (re-levelled to per cent), EFP wedge, shock decompositions of the EFP wedge and the output gap, financial-block internals, and IRFs to a financial and a monetary shock. No run script yet.

Note on the rates chart: the safe rate and cost of capital are *actual* rates (`R−π` and `R−π+EFP`), not natural rates - an earlier draft mislabelled them. The model's natural rate of interest (the smoother, ε_d-driven `r^n ≈ ε_d/σ`) is shown as a third, dashed line. The time-varying divergence lives in the actual rates (the EFP), not the natural rate.

A Bernanke-Gertler-Gilchrist-style financial accelerator embedded in the NK DSGE, kept lean (`mc = ξ·y` closure, no separate labour block). It carries two natural rates that emerge from optimisation, not by assumption:

- **safe rate** `R − E[π']` (household consumption Euler)
- **return on capital** `r^k` (firms' capital, via Tobin's Q)
- **wedge** = the external-finance premium `EFP = χ·leverage + ω`, where leverage = `q + k − n`. This is the endogenous "great divergence" object - it widens with leverage or a financial shock and transmits by depressing investment through Q.

**State vector** `[ε_d, ε_s, ω, R₋₁, k, n, q₋₁]` (7 states), **controls** `[c, q, π]` (3 forward). Determinate Blanchard-Kahn solve. A financial shock produces the textbook accelerator: EFP up, q crash, net-worth erosion, investment collapse, output/inflation down, policy eases.

**Estimation:** 4 observables `[output_gap, inflation, cash_rate, credit_spread]` (= 4 shocks, no stochastic singularity). Sample **1993Q1+**; the credit spread (`src/data/bonds.py:get_corporate_spread`, RBA F3 corporate yield − matched F2 CGS yield) only starts 2005Q1, so it is left missing (NaN) before then and the Kalman filter uses 4 observables from 2005 and 3 before — no truncation to the spread's start. **GFC kept** (its spread blowout identifies the financial block), **COVID excluded**.

**Findings (first estimation):**
- **χ ≈ 0.017** - the accelerator is real but *weak*: the wedge is mostly an exogenous financial shock, not endogenous leverage amplification. The EFP-wedge shock decomposition confirms this directly - the financial shock ω accounts for almost the entire wedge, with demand/supply/monetary contributions negligible.
- The smoothed EFP tracks the GFC credit-spread spike (+2.65 dev at 2009Q1); r\*_safe falls through the 2010s to −2.09 by 2019 (the divergence), r\*_capital spikes at the GFC.
- **The Taylor block is weakly identified.** With the original bounds φ_π, φ_y, κ_p pegged at their ceilings; widening the bounds shows them teleporting to different corners (φ_y to 0 or to whatever ceiling) at near-identical likelihood - a flat-ridge non-identification, the textbook reason DSGEs use Bayesian priors. The production run uses economically-reasonable caps (φ_π≤3, φ_y≤1) as effectively-imposed priors. χ (≈0.012-0.024) and the financial block are stable by contrast. (φ_π *does* come off its cap once unemployment is observed - see FA-NK-wage.)
- **Sample is not the driver:** estimated on the same 1993+ sample as the probe (via missing-spread handling), the FA-NK gives essentially the 2005+ values, so the probe-vs-FA-NK difference is structural.

**Open threads:** Bayesian re-estimation with priors on the Taylor block - the one real remaining unlock for the policy-block weak identification.

---

## 0a-2. Labour block (Tier 1) and FA-NK-wage (Tier 2 + Galí unemployment)

**Run:** `FANKModel(labour_block=True)` / `run_fa_nk_labour()` (Tier 1); `uv run python -m src.models.dsge.fa_nk_wage_model`, `run_wage(observe_u=True)`, `produce_ustar()` (Tier 2).

**Tier 1 - structural marginal cost.** Replaces the leaner `mc = ξ·y` with a flexible-wage labour block: `l = (y−α·k)/(1−α)`, `w = c/σ + φ_l·l`, `mc = w + l − y`. Hours and the wage are static functions of `(c, k, q)`, so **no new states**. *Result:* **unsticks κ_p** (settles ~1.93 interior vs the leaner model running to its ceiling - the crude `ξ·y` closure was causing the peg), at slightly worse fit; bound-hitting shifts to the supply shock; does not touch the Taylor block.

**Tier 2 (`FA-NK-wage`) - sticky wages + Galí unemployment.** Adds Calvo (Erceg-Henderson-Levin) wages: the real wage becomes a state, wage inflation `π_w` a forward variable, plus a wage-markup shock - 9 states, 4 forward, 5 shocks, determinate. Unemployment via Galí (2011): `u = μ_w/φ_l` (wage markup over the inverse Frisch). NAIRU recovered post-hoc as `U* = observed U − u_gap`. *Results:* best-identified DSGE in the family (κ_w≈0.45 identified, φ_y interior); IRFs textbook (wage-markup shock → u up & wage inflation down; monetary tightening → u up). **But U\* is not credible** - it swings wildly because there is no smooth NAIRU random-walk state and (when free) unemployment is unobserved. Observing unemployment (`observe_u=True`) centres U\* at a plausible ~5.7% and notably **brings φ_π off its cap (≈1.33)**, but U\* stays too volatile and the wage Phillips curve collapses (κ_w→0). **For a credible NAIRU use `src/models/nairu/`** (it has the NAIRU random-walk state this DSGE lacks).

---

## 0b. NK-TwoStar Model (linear probe)

**Run:** `uv run python -m src.models.dsge.nk_twostar_model`.

A diagnostic stepping-stone, *not* the proper model. The original NK DSGE with the Taylor rule re-anchored to a time-varying r\*_goods (real indexed 10y bond yield) and a **bolted-on, reduced-form** wedge term `−σ_k·ω` in the IS curve, where `ω = trend-growth − bond-yield` (the divergence as observed data). Sample 1993Q1+, COVID excluded.

**Purpose:** test whether the wedge has any signal before committing to the proper (microfounded) build. **It did:** after cleaning (COVID excluded, a GDP-growth bug fixed, wages demeaned), `σ_k ≈ 0.14` and the wedge state traced the divergence (peak ~2019, ≈0 by 2025). Separately, **φ_π pinned at its determinacy floor (1.01)** - profiled and likelihood-ratio-significant, i.e. the data wants a *passive* rule the determinate DSGE cannot represent. The non-collapsing wedge is what justified building `FA-NK`.

---

## 1. NK Model (True DSGE)

**The original forward-looking model, solved via Blanchard-Kahn.**

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
- Bernanke, Gertler, Gilchrist (1999): "The Financial Accelerator in a Quantitative Business Cycle Framework" (the FA-NK external-finance-premium mechanism)
- Lubik & Schorfheide (2004): "Testing for Indeterminacy" (relevant to the φ_π determinacy-floor finding)
