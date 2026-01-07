# Inflation Expectations Signal Extraction Model

This directory contains a Bayesian signal extraction model for estimating latent inflation expectations from multiple survey and market-based measures, following Cusbert (2017).

## Summary

| Component | Method | Key Feature |
|-----------|--------|-------------|
| Sample | 1983Q1 to present | Extended early period using bonds, headline CPI, and HCOE |
| Latent State | Regime-switching Student-t random walk | Innovation variance estimated (Target, Short) or fixed (Unanchored, Long Run) |
| Model Types | Target Anchored, Unanchored, Short Run (1yr), Long Run (10yr) | Different observation equations per type |
| Measures | market_1y, breakeven, business, market_yoy, inflation, HCOE, headline CPI (pre-1993), nominal bonds | Series-specific effects (α) and backward-looking bias (λ) for Target/Unanchored only |

## Reference

Cusbert T (2017), "Estimating the NAIRU and the Unemployment Gap", RBA Bulletin, June, pp 13-22.

---

## Four Model Types

The model estimates four separate latent expectations series:

| Model | Code | Survey Series | Anchor | Use Case |
|-------|------|---------------|--------|----------|
| **Target Anchored** | `target` | market_1y + breakeven + business + market_yoy (with α, λ) | Yes (2.5% post-1998Q4) | Policy analysis - assumes credible anchoring |
| **Unanchored** | `unanchored` | market_1y + breakeven + business + market_yoy (with α, λ) | No | Test de-anchoring - same as target but without 2.5% constraint |
| **Short Run (1 Year)** | `short` | market_1y only (no α, λ) | No | Wage-relevant expectations for Phillips curve |
| **Long Run (10-Year Bond)** | `market` | breakeven only (no α, λ) | No | What markets actually believe |

---

## Output Location

Results are saved to `output/expectations/` with model type suffixes:
- `expectations_{code}_trace.nc` - Full MCMC trace (ArviZ InferenceData)
- `expectations_{code}_metadata.pkl` - Index, measures, inflation data
- `expectations_{code}_hdi.parquet` - Point estimates with HDI bounds
- `expectations_{code}_hdi.csv` - Same in CSV format

Where `{code}` is `target`, `unanchored`, `short`, or `market`.

Charts are saved to `charts/expectations/`.

---

## File Structure

```
stage1.py                   # Model building and estimation
stage2.py                   # Diagnostics and plotting
common.py                   # Constants (ANCHOR_SIGMA, MODEL_TYPES, etc.)
MODEL_NOTES.md              # This documentation
```

---

## Data Sources

### Survey Measures (RBA Table G3)
| Series | Start | Description | Used In |
|--------|-------|-------------|---------|
| market_1y | 1993Q3 | Market economists 1-year ahead | Target Anchored, Unanchored, Short Run |
| business | 1989Q3 | NAB Business Survey inflation expectations | Target Anchored, Unanchored |
| market_yoy | 1994Q3 | Market economists year-on-year | Target Anchored, Unanchored |

### Market Measures (RBA Table F2)
| Series | Start | Description | Used In |
|--------|-------|-------------|---------|
| breakeven | 1986Q3 | 10-year nominal yield minus indexed bond yield | Target Anchored, Unanchored, Long Run |
| nominal_10y | 1969 | Nominal 10-year government bond yield | Target/Unanchored (pre-1993Q3, 7yr overlap), Long Run (pre-1988Q3, 2yr overlap) |

### Inflation Measures (ABS 6401.0)
| Series | Start | Description | Used In |
|--------|-------|-------------|---------|
| Trimmed mean | 1983Q1 | Annual trimmed mean CPI | Target Anchored, Unanchored, Short Run |
| Weighted median | 1983Q1 | Annual weighted median CPI | Target Anchored, Unanchored, Short Run |
| Headline CPI | 1949Q3 | Annual headline CPI | Target Anchored, Unanchored, Short Run (pre-1993 only) |

### Wage Measures (ABS 5206.0, derived)
| Series | Start | Description | Used In |
|--------|-------|-------------|---------|
| HCOE growth | 1978Q4 | Annual hourly compensation of employees growth | Target Anchored, Unanchored |
| MFP trend | 1978Q4 | Derived from wage data: MFP = LP - α×(g_K - g_L), HP-filtered, floored at zero | Target Anchored, Unanchored |

---

## Model Design Decisions

### 1. Regime-Switching Student-t Innovations

The innovation variance switches at 1994Q1 (inflation targeting bedded down). For Target Anchored and Short Run models, the variance is estimated; for Unanchored and Long Run it is fixed.

| Period | Target/Short (estimated) | Unanchored/Long Run (fixed) |
|--------|--------------------------|------------------|
| Pre-1994 (early) | σ_early ~ HalfNormal(0.15), typical ~0.30-0.40 | σ = 0.30 (Unanchored) / 0.12 (Long Run) |
| Post-1994 (late) | σ_late ~ HalfNormal(0.1), typical ~0.07-0.14 | σ = 0.07 (Unanchored) / 0.075 (Long Run) |

Student-t with ν=4 allows occasional larger jumps (e.g., 1988-92 disinflation) while remaining smooth otherwise. The larger early variance reflects that expectations genuinely were more volatile before inflation targeting. The 1994 switchpoint (rather than 1993) gives a year for inflation targeting to bed down.

### 2. Series Effects (α) — Target Anchored and Unanchored Only

Each measure has its own systematic level effect:
- **market_1y**: α ≈ -0.44 (reads below latent)
- **breakeven**: α ≈ -0.46 (includes liquidity/term premium)
- **business**: α ≈ -0.60 (own-price vs CPI)
- **market_yoy**: α ≈ 0.04 (essentially unbiased)

### 3. GST Adjustment

Market economists' expectations in 1999Q3-2000Q3 were distorted by GST anticipation. Both market_1y and market_yoy are interpolated through to avoid spurious expectations jump:
- 1999Q3: 2.6%
- 1999Q4-2000Q3: 2.5%

### 4. Early Period Anchoring

Pre-survey period (1983-1993) is anchored using:
- **Headline CPI** (pre-1993): Informative when expectations were adaptive (Target Anchored, Unanchored, Short Run)
- **Nominal 10y bonds**: Using multiplicative Fisher equation
  - Target Anchored, Unanchored: pre-1993Q3
  - Long Run: pre-1988Q3 (2yr overlap with breakeven which starts 1986Q3)
- **Lagged inflation** observation equation throughout (Target Anchored, Unanchored, Short Run)

### 5. Post-1998 Target Anchoring (Target Anchored model only)

Once RBA credibility was established, expectations should be tightly anchored to target:
- **Inflation target** (post-1998Q4): Expectations observed as 2.5% with σ=0.35
- This tightens the posterior variance in the anchored period
- Only applies to the Target Anchored model

### 6. HCOE Growth Observation (Target Anchored and Unanchored)

Hourly compensation of employees growth provides wage-based information about inflation expectations throughout the sample:
- **HCOE growth** ≈ inflation expectations + MFP growth + adjustment
- MFP is derived from wage data (goes back to 1978Q4, unlike ABS 5204.0 which starts 1995)
- The `hcoe_adjustment` parameter is estimated but typically ≈ 0
- σ_hcoe ≈ 2.0 (noisier than surveys, but informative especially pre-1993)

### 7. Lagged Inflation Observations

Published inflation shapes expectations with a lag. Observation equations use inflation_{t-1} rather than contemporaneous inflation.

### 8. Real Rate (r*) Identification

The nominal bond observation uses the Fisher equation:
```
nominal = π_exp + r* + (π_exp × r* / 100)
```

With limited overlap between nominal bonds and breakeven data, **r* and π_exp are only weakly identified from each other**. In the pre-breakeven period:
- If r* is estimated lower → π_exp adjusts higher to fit nominal yields
- If r* is estimated higher → π_exp adjusts lower

This trade-off means the r* estimate should be interpreted cautiously. The model finds a combination of r* and π_exp that fits the data, but the decomposition depends on the overlap period length. The 2-year overlap provides some anchor for r*, but the pre-breakeven π_exp level can shift to compensate.

---

## Model Specification

### State Equation

Two concatenated random walks with regime-switching innovation variance:

```
# Early period (1983Q1 to 1993Q4)
πᵉ_t = πᵉ_{t-1} + ε_t,    ε_t ~ StudentT(ν=4, μ=0, σ_early)

# Late period (1994Q1 onwards) - continues from early
πᵉ_t = πᵉ_{t-1} + ε_t,    ε_t ~ StudentT(ν=4, μ=0, σ_late)
```

Initial state: `πᵉ_0 ~ N(inflation_1983Q1, 2.0)` ≈ N(10.2, 2.0)

### Observation Equations

**Survey/market measures** (m = market_1y, breakeven as configured):
```
measure_{m,t} = πᵉ_t + α_m + λ_m × π_{t-1} + ε_{m,t}
ε_{m,t} ~ N(0, σ_{obs,m})
```

**Trimmed mean/weighted median inflation** (lagged, Target Anchored, Unanchored, Short Run):
```
inflation_{t-1} ~ N(πᵉ_t, σ_inflation)
```

**Headline CPI** (lagged, pre-1993, Target Anchored, Unanchored, Short Run):
```
headline_{t-1} ~ N(πᵉ_t, σ_headline)
```

**Nominal 10y bonds** (multiplicative Fisher):
```
nominal_t ~ N(πᵉ_t + real_rate + (πᵉ_t × real_rate / 100), σ_nominal)
real_rate ~ N(5.0, 1.5)  # Estimated
```

**HCOE growth** (Target Anchored and Unanchored):
```
hcoe_t ~ N(πᵉ_t + mfp_t + hcoe_adjustment, σ_hcoe)
hcoe_adjustment ~ N(0, 0.5)  # Estimated ≈ 0
```

**Inflation target** (post-1998Q4, Target Anchored only):
```
2.5 ~ N(πᵉ_t, 0.35)
```

### Priors

| Parameter | Prior | Typical Estimate | Models |
|-----------|-------|------------------|--------|
| α (series effects) | N(0, 0.5) | -0.4 to -0.6 | Target, Unanchored |
| λ (backward-looking bias) | N(0.1, 0.15) | -0.2 to 0.2 | Target, Unanchored |
| σ_obs | HalfNormal(1.0) | 0.1 to 0.9 | All |
| σ_inflation | HalfNormal(1.5) | ~0.95 | Target, Unanchored |
| σ_headline | HalfNormal(2.0) | ~1.7 | Target, Unanchored, Short Run |
| σ_nominal | HalfNormal(2.0) | ~0.7 | Target, Unanchored, Long Run |
| σ_hcoe | HalfNormal(2.0) | ~2.0 | Target, Unanchored |
| σ_early (innovation) | HalfNormal(0.15) | ~0.30-0.40 | Target, Short Run |
| σ_late (innovation) | HalfNormal(0.1) | ~0.07-0.14 | Target, Short Run |
| hcoe_adjustment | N(0, 0.5) | ~0 | Target, Unanchored |
| real_rate | N(5.0, 1.5) | 3.8-5.4% (see note) | Target, Unanchored, Long Run |

**Note on real_rate (r*)**: Estimates vary by model due to identification trade-off with π_exp. Target ~5.4%, Unanchored ~5.3%, Long Run ~3.8%. The Long Run estimate is lower because it has less data to anchor r* (only 2yr overlap vs full pre-1993 period for Target/Unanchored).

---

## Model Configuration Summary

| Feature | Target Anchored | Unanchored | Short Run (1yr) | Long Run (10yr) |
|---------|-----------------|------------|-----------------|-----------------|
| Survey series | market_1y, breakeven, business, market_yoy | market_1y, breakeven, business, market_yoy | market_1y | breakeven |
| Survey bias (α, λ) | Yes | Yes | No | No |
| Innovation variance | Estimated | Fixed (0.30/0.07) | Estimated | Fixed (0.12/0.075) |
| Target anchor (post-1998Q4) | Yes (σ=0.35) | No | No | No |
| Headline CPI (pre-1993) | Yes | Yes | Yes | No |
| Nominal bonds | Yes (pre-1993Q3) | Yes (pre-1993Q3) | No | Yes (pre-1988Q3, 2yr overlap) |
| HCOE growth | Yes | Yes | No | No |
| Inflation observation | Yes | Yes | Yes (shared σ) | No |

---

## Per-Model Equations

### Long Run (10-Year Bond) — `market`

The simplest model. Uses only breakeven inflation (no α, λ) and nominal bonds with 2-year overlap to anchor r*.

**State equation:**
```
πᵉ_t = πᵉ_{t-1} + ε_t

where:
  ε_t ~ StudentT(ν=4, μ=0, σ=0.12)   for t < 1994Q1
  ε_t ~ StudentT(ν=4, μ=0, σ=0.075)  for t ≥ 1994Q1

Initial: πᵉ_0 ~ N(inflation_1983Q1, 2.0)
```

**Observation equations:**
```
breakeven_t = πᵉ_t + ε_t                                         [no α, no λ]
  ε_t ~ N(0, σ_obs)

nominal_t = πᵉ_t + real_rate + (πᵉ_t × real_rate / 100) + ε_t   [pre-1988Q3, 2yr overlap]
  ε_t ~ N(0, σ_nominal)
  real_rate ~ N(5.0, 1.5)                                        [estimated ~3.8%]
```

**Note:** The 2-year overlap (1986Q3-1988Q3) provides limited data for separating r* from π_exp. The lower r* estimate (~3.8%) compared to Target/Unanchored (~5.3-5.4%) reflects this identification uncertainty. The π_exp path adjusts to compensate.

### Short Run (1 Year) — `short`

Simplified model with estimated innovation variance. Uses market_1y survey and inflation with shared σ. No survey bias terms (α, λ), no nominal bonds, no HCOE.

**State equation:** Estimated innovation variance (unlike Unanchored/Long Run which use fixed values).
```
πᵉ_t = πᵉ_{t-1} + ε_t

where:
  ε_t ~ StudentT(ν=4, μ=0, σ_early)   for t < 1994Q1
  ε_t ~ StudentT(ν=4, μ=0, σ_late)    for t ≥ 1994Q1

  σ_early ~ HalfNormal(0.15)          [estimated, typical ~0.4]
  σ_late ~ HalfNormal(0.1)            [estimated, typical ~0.14]
```

**Observation equations:**
```
market_1y_t = πᵉ_t + ε_t                                        [no α, no λ]
  ε_t ~ N(0, σ_obs)

π_{t-1} ~ N(πᵉ_t, σ_obs)                                        [shared σ]

headline_{t-1} ~ N(πᵉ_t, σ_headline)                            [pre-1993 only]

σ_obs ~ HalfNormal(1.0)                                         [estimated, typical ~0.46]
```

### Target Anchored — `target`

The full model. Uses market_1y, breakeven, business, market_yoy (all with α, λ), plus inflation, headline CPI, nominal bonds, HCOE, and the 2.5% target anchor. Innovation variance is estimated.

**State equation:** Estimated innovation variance.
```
πᵉ_t = πᵉ_{t-1} + ε_t

where:
  ε_t ~ StudentT(ν=4, μ=0, σ_early)   for t < 1994Q1
  ε_t ~ StudentT(ν=4, μ=0, σ_late)    for t ≥ 1994Q1

  σ_early ~ HalfNormal(0.15)          [estimated, typical ~0.30]
  σ_late ~ HalfNormal(0.1)            [estimated, typical ~0.07]

Initial: πᵉ_0 ~ N(inflation_1983Q1, 2.0)
```

**Observation equations:**
```
market_1y_t = πᵉ_t + α_market_1y + λ_market_1y × π_{t-1} + ε_t
  ε_t ~ N(0, σ_obs_market_1y)

breakeven_t = πᵉ_t + α_breakeven + λ_breakeven × π_{t-1} + ε_t
  ε_t ~ N(0, σ_obs_breakeven)

business_t = πᵉ_t + α_business + λ_business × π_{t-1} + ε_t
  ε_t ~ N(0, σ_obs_business)

market_yoy_t = πᵉ_t + α_market_yoy + λ_market_yoy × π_{t-1} + ε_t
  ε_t ~ N(0, σ_obs_market_yoy)

π_{t-1} ~ N(πᵉ_t, σ_inflation)

headline_{t-1} ~ N(πᵉ_t, σ_headline)                            [pre-1993 only]

nominal_t = πᵉ_t + real_rate + (πᵉ_t × real_rate / 100) + ε_t   [pre-1993Q3 only]
  ε_t ~ N(0, σ_nominal)
  real_rate ~ N(5.0, 1.5)                                        [estimated ~5.4%]

hcoe_t = πᵉ_t + mfp_t + hcoe_adjustment + ε_t
  ε_t ~ N(0, σ_hcoe)

2.5 ~ N(πᵉ_t, 0.35)                                             [post-1998Q4 only]
```

### Unanchored — `unanchored`

Same as Target Anchored but **without the 2.5% target anchor**. Uses fixed innovation variance (matching Target estimates) to avoid funnel geometry in the posterior. This model answers: "What do expectations look like if we don't impose credibility?"

**State equation:** Fixed innovation variance (σ_early=0.30, σ_late=0.07).
```
πᵉ_t = πᵉ_{t-1} + ε_t

where:
  ε_t ~ StudentT(ν=4, μ=0, σ=0.30)   for t < 1994Q1
  ε_t ~ StudentT(ν=4, μ=0, σ=0.07)   for t ≥ 1994Q1

Initial: πᵉ_0 ~ N(inflation_1983Q1, 2.0)
```

**Observation equations:** Same as Target Anchored, minus the target anchor:
```
market_1y_t = πᵉ_t + α_market_1y + λ_market_1y × π_{t-1} + ε_t
breakeven_t = πᵉ_t + α_breakeven + λ_breakeven × π_{t-1} + ε_t
business_t = πᵉ_t + α_business + λ_business × π_{t-1} + ε_t
market_yoy_t = πᵉ_t + α_market_yoy + λ_market_yoy × π_{t-1} + ε_t
π_{t-1} ~ N(πᵉ_t, σ_inflation)
headline_{t-1} ~ N(πᵉ_t, σ_headline)                            [pre-1993 only]
nominal_t = πᵉ_t + real_rate + (πᵉ_t × real_rate / 100) + ε_t   [pre-1993Q3 only]
hcoe_t = πᵉ_t + mfp_t + hcoe_adjustment + ε_t

# NO target anchor observation
```

**Key difference from Target:** Without the anchor, the bias parameters (α, λ) shift to absorb the difference. Typical estimates:
- `λ` for market_1y: ~0 (vs 0.18 in Target) — backward-looking component disappears
- `α` for market_yoy: ~+0.48 (vs +0.04 in Target) — larger positive bias

---

## Validation

### Comparison with RBA PIE_RBAQ

The model is validated against the RBA's inflation expectations series (PIE_RBAQ) from the MacroDave database, which runs from 1983Q1 to 2019Q1.

| Period | Correlation | Mean Diff | RMSE |
|--------|-------------|-----------|------|
| Post-1993 (survey data) | 0.94 | -0.04pp | 0.12pp |
| Post-1998 (anchor established) | 0.92 | -0.03pp | 0.06pp |
| 2009-2019 (last 10 years overlap) | 0.95 | -0.01pp | 0.06pp |

**Post-inflation targeting period (1993-2019)**: Excellent fit. RMSE of 6 basis points post-1998. The model tracks the RBA series very closely once the inflation target anchor is established.

**Pre-inflation targeting period (1983-1993)**: The RBA series shows a sawtooth pattern that doesn't match headline inflation or other indicators. Our model produces a smoother decline informed by HCOE growth, which is more consistent with wage-setting behaviour during the disinflation.

**Post-2019**: The RBA series ends in 2019Q1. Our model continues to present, anchored by surveys, breakevens, HCOE, trimmed mean/weighted median inflation, and the 2.5% target observation.

### Testing for De-anchoring

The four model types provide a natural way to test for de-anchoring:

1. **Target Anchored**: Assumes expectations are anchored at 2.5% post-1998
2. **Unanchored**: Same observations as Target but without the anchor constraint
3. **Short Run** and **Long Run**: Different observation sets, no anchoring assumption

**Key comparison: Target vs Unanchored**

The gap between Target Anchored and Unanchored directly measures how much the 2.5% anchor is pulling expectations down:

| Scenario | Target - Unanchored Gap | Interpretation |
|----------|-------------------------|----------------|
| Gap ≈ 0 | Anchor has no effect | Expectations naturally at target |
| Gap < 0 (Unanchored higher) | Anchor pulling down | Surveys showing elevated expectations |
| Gap > 0 (Target higher) | Anchor pulling up | Would indicate deflationary expectations |

**Current estimates (2025Q3):**
- Target Anchored: 2.58%
- Unanchored: 2.75%
- Gap: -17bp (anchor pulling down modestly)

**Interpretation:**
- If expectations are truly anchored, all four models should produce similar estimates (~2.5%)
- If the Unanchored/Short Run/Long Run models show expectations above Target Anchored, this suggests the anchor is doing work to pull expectations toward target
- The Unanchored model is the cleanest comparison since it uses identical observations to Target

**Signs of de-anchoring:**
1. Growing gap between Unanchored and Target Anchored estimates
2. Short Run/Long Run estimates persistently above Target Anchored
3. Survey alphas shifting upward over time
4. Breakeven inflation rising
5. HCOE growth exceeding π_exp + MFP
6. Wider credible intervals in Target Anchored model as it struggles to reconcile observations with anchor

---

## Usage

### Command Line

```bash
# Stage 1: Run all four models (default: 1983Q1 start, 10000 draws, 4 chains)
uv run python -m src.models.expectations.stage1

# Run single model
uv run python -m src.models.expectations.stage1 --model target

# Quick run for testing
uv run python -m src.models.expectations.stage1 --draws 2000 --tune 1000

# Suppress progress bar
uv run python -m src.models.expectations.stage1 -q

# Stage 2: Diagnostics and plots (loads saved results from stage1)
uv run python -m src.models.expectations.stage2

# Skip plot generation
uv run python -m src.models.expectations.stage2 --no-plots
```

### Python API

```python
from src.models.expectations.stage1 import run_model, run_all_models
from src.models.expectations.stage2 import load_results, load_all_results

# Run single model (use code: "target", "unanchored", "short", or "market")
results = run_model(model_type="target", start="1984Q1", verbose=True)

# Run all four models
run_all_models(start="1984Q1", verbose=True)

# Load saved results (stage2)
results = load_results("target")
all_results = load_all_results()

# Access results
median = results.expectations_median()
hdi = results.expectations_hdi(prob=0.9)
samples = results.expectations_posterior()
```

---

## Sampler Settings

| Setting | Value |
|---------|-------|
| Draws | 10,000 |
| Tune | 4,000 |
| Chains | 4 |
| Sampler | NUMPyro NUTS |
| Total samples | 40,000 |

---

## Integration with NAIRU Model

The expectations series feeds into the NAIRU model's Phillips curves:

1. Run expectations model to extract πᵉ
2. Save results with `results.save()`
3. NAIRU model loads median expectations from `output/expectations/expectations_target_hdi.parquet` via `src/data/expectations_model.py`
4. Phillips curves use `π_exp` as the baseline around which demand/supply effects operate

### Anchor Modes

The NAIRU model supports two modes for anchoring expectations:

| Mode | Description | Use Case |
|------|-------------|----------|
| `target` (default) | Use estimated expectations to 1992Q4, phase linearly to 2.5% by 1998Q4, then 2.5% target thereafter | Policy analysis |
| `expectations` | Use full estimated expectations series as-is | Historical analysis |

**Why `target` is the default**: The key policy question is "what interest rate path gets inflation to target?" Using the 2.5% target post-1998 makes NAIRU directly interpretable as the unemployment rate consistent with hitting the inflation target. This is more policy-relevant than asking what rate achieves whatever expectations happen to be.

The phasing period (1993-1998) reflects the RBA's gradual establishment of inflation targeting credibility. By 1998, expectations were effectively anchored at the 2-3% band midpoint.

**NAIRU interpretation**:
- With `target` anchor: NAIRU is the unemployment rate where inflation equals the 2.5% target
- With `expectations` anchor: NAIRU is the unemployment rate where inflation equals current expectations

See `src/models/nairu/` for NAIRU model implementation.
