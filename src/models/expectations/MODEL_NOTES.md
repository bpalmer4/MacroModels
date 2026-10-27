# Inflation Expectations Signal Extraction Model

This directory contains a Bayesian model of trend inflation: a random-walk trend read by biased surveys, market measures, wages and inflation, whose deviation from the trend is an AR(1) gap. It follows Chan, Clark and Koop (2018), with short-horizon surveys and constant biases where they use one long-run survey with time-varying biases.

## Summary

| Component | Method | Key Feature |
|-----------|--------|-------------|
| Sample | 1983Q1 to present, quarterly | Extended early period using bonds, headline CPI, and HCOE |
| Latent State | Regime-switching Student-t random walk | Innovation variance estimated (Short) or fixed (Unanchored, Long Run) |
| Model Types | Unanchored, Short Run (1yr), Long Run (10yr) | Different observation equations per type |
| Measures | market_1y, breakeven, business, market_yoy, inflation, HCOE, private WPI, headline CPI (pre-1993), nominal bonds | Series-specific effects (α) and backward-looking bias (λ) for Unanchored only |

## Reference

Chan J C C, T E Clark and G Koop (2018), "A New Model of Inflation, Trend Inflation, and Long-Run Inflation Expectations", Journal of Money, Credit and Banking, 50(1), pp 5-53.

---

## Three Model Types

The model estimates three separate latent expectations series. None imposes the inflation target.

| Model | Code | Survey Series | Use Case |
|-------|------|---------------|----------|
| **Unanchored** | `unanchored` | market_1y + breakeven + business + market_yoy (with α, λ) | The inflation expectations series: every input, no target |
| **Short Run (1 Year)** | `short` | market_1y only (no α, λ) | Wage-relevant expectations for Phillips curve |
| **Long Run (10-Year Bond)** | `market` | breakeven only (no α, λ) | What markets actually believe |

---

## Output Location

Results are saved to `output/expectations/` with model type suffixes:
- `expectations_{code}_trace.nc` - Full MCMC trace (xarray DataTree)
- `expectations_{code}_metadata.pkl` - Index, measures, inflation data
- `expectations_{code}_hdi.parquet` - Point estimates with HDI bounds
- `expectations_{code}_hdi.csv` - Same in CSV format
- `expectations_{code}_hdi_quarterly.parquet` - Quarterly HDI, the file downstream models read

Where `{code}` is `unanchored`, `short`, or `market`.

Charts are saved to `charts/expectations/`.

---

## File Structure

```
stage1.py                   # Model building and estimation
stage2.py                   # Diagnostics and plotting
common.py                   # Constants (paths, sampler settings, MODEL_TYPES)
MODEL_NOTES.md              # This documentation
```

---

## Data Sources

### Survey Measures (RBA Table G3)
| Series | Start | Description | Used In |
|--------|-------|-------------|---------|
| market_1y | 1993Q3 | Market economists 1-year ahead | Unanchored, Short Run |
| business | 1989Q3 | NAB Business Survey inflation expectations | Unanchored |
| market_yoy | 1994Q3 | Market economists year-on-year | Unanchored |

**The union series is fetched and not used, and it has stopped.** `src/data/expectations.py`
pulls `union_1y` (GUNIEXPY) and `union_yoy` (GUNIEXPYY) from G3 and its docstring advertises
them, but `stage1.py` selects `["market_1y", "business", "market_yoy"]` and the union measures
never reach any model. Both series end **2023Q3**, the unions having declined to keep
participating, so the column remains in the RBA file and has been empty since. Nothing needs
doing: the model's panel is unchanged and still live at every input. Recorded here because the
end of a G3 series looks alarming in the raw data and the alarm is misplaced.

**What the panel does not cover.** There is no household measure. G3's consumer series
(GCONEXP) holds 18 observations beginning 2022-03, so it cannot serve this model's history, and
a household series for the 1990s would have to come from the Melbourne Institute survey
directly. The panel is therefore financial markets, forecasters and firms' stated expectations,
which is worth remembering whenever the output is read as "inflation expectations" without
qualification.

### Market Measures (RBA Table F2)
| Series | Start | Description | Used In |
|--------|-------|-------------|---------|
| breakeven | 1986Q3 | 10-year nominal yield minus indexed bond yield | Unanchored, Long Run |
| nominal_10y | 1969 | Nominal 10-year government bond yield | Unanchored (pre-1993Q3, 7yr overlap), Long Run (pre-1988Q3, 2yr overlap) |

### Inflation Measures (ABS 6401.0)
| Series | Start | Description | Used In |
|--------|-------|-------------|---------|
| Trimmed mean | 1982Q2 | Quarterly trimmed mean CPI, annualised; year-ended for the survey λ terms | Unanchored, Short Run |
| Weighted median | 1982Q2 | Quarterly weighted median CPI, annualised; year-ended for the survey λ terms | Unanchored, Short Run |
| Headline CPI | 1987Q1 | Quarterly headline CPI, seasonally adjusted, annualised | Unanchored, Short Run (pre-1993 only, so 1987 to 1992) |

### Wage and Labour-Market Measures
| Series | Start | Description | Used In |
|--------|-------|-------------|---------|
| HCOE growth | 1978Q4 | Annual hourly compensation of employees growth | Unanchored |
| MFP trend | 1978Q4 | Derived from wage data: MFP = LP - α×(g_K - g_L), HP-filtered, floored at zero | Unanchored |
| Private WPI growth | 1998Q3 | Year-ended, private sector (ABS 6345.0); public wages are administered | Unanchored |
| GDP per hour trend | — | Change in the annual-average log of GDP per hour worked over a trailing 5 years, per year, with the index flattened 2019Q4-2023Q1 (5206.0) | Unanchored (WPI) |
| Job vacancies | 1979Q2 | ABS 6354.0, seasonally adjusted; surveyed mid-quarter, mapped to calendar quarters; suspended 2008Q3-2009Q3 | Unanchored (WPI) |
| Labour-market tightness | — | log(vacancies / unemployed), centred on its full-history mean, averaged over four quarters | Unanchored (WPI) |

---

## Model Design Decisions

### 1. Regime-Switching Student-t Innovations

The innovation variance switches at 1994Q1 (inflation targeting bedded down). For the Short Run model, the late variance is estimated; for Unanchored and Long Run both are fixed.

| Period | Short (estimated) | Unanchored/Long Run (fixed) |
|--------|-------------------|------------------|
| Pre-1994 (early) | σ = 0.12, fixed | σ = 0.30 (Unanchored) / 0.12 (Long Run) |
| Post-1994 (late) | σ_late ~ HalfNormal(0.075) | σ = 0.07 (Unanchored) / 0.075 (Long Run) |

Student-t with ν=4 allows occasional larger jumps (e.g., 1988-92 disinflation) while remaining smooth otherwise. The larger early variance reflects that expectations genuinely were more volatile before inflation targeting. The 1994 switchpoint (rather than 1993) gives a year for inflation targeting to bed down.

### 2. Series Effects (α): Unanchored Only

Each measure has its own systematic level effect, since none measures the trend directly:
- **market_1y**: a one-year forecast, not a long-run one
- **breakeven**: includes liquidity and term premia
- **business**: firms' own prices, not the CPI
- **market_yoy**: a year-on-year forecast

### 3. GST Adjustment

Market economists' expectations in 1999Q3-2000Q3 were distorted by GST anticipation. Both market_1y and market_yoy are interpolated through to avoid spurious expectations jump:
- 1999Q3: 2.6%
- 1999Q4-2000Q3: 2.5%

### 4. Early Period Anchoring

Pre-survey period (1983-1993) is anchored using:
- **Headline CPI** (1987-1992, the seasonally adjusted series starts 1987): Informative when expectations were adaptive (Unanchored, Short Run)
- **Nominal 10y bonds**: Using multiplicative Fisher equation
  - Unanchored: pre-1993Q3
  - Long Run: pre-1988Q3 (2yr overlap with breakeven which starts 1986Q3)
- **Lagged inflation** observation equation throughout (Unanchored, Short Run)

### 5. No Target Observation

No model observes the 2.5% target. Expectations are free to move away from it, which is what a measure of de-anchoring needs.

### 6. HCOE Growth Observation (Unanchored)

Hourly compensation of employees growth provides wage-based information about inflation expectations throughout the sample:
- **HCOE growth** ≈ inflation expectations + MFP growth + adjustment
- MFP is derived from wage data (goes back to 1978Q4, unlike ABS 5204.0 which starts 1995)
- The `hcoe_adjustment` parameter is estimated
- Noisier than the surveys, but informative especially pre-1993

### 6a. WPI Growth Observation (Unanchored)

Private WPI growth tells the model what wage setters behave as if they expect. Firms can pay
expected inflation plus productivity growth without squeezing margins; what wages grow by
beyond that is either a tight labour market or higher expectations. The tightness term takes
the first, so a tight labour market is not read as expectations.

- **Tightness is log(V/U)**, vacancies over unemployed persons. It needs no trend or u*, so it
  is not circular (u* models read this one) and has no end-point problem. In an off-model
  regression of WPI − πᵉ − productivity over 1998Q3-2019Q4 and 2023Q3 on, log V/U explained
  75% of the variation (slope about 1.6, t about 16); unemployment less its trailing 10-year
  average, the first version tried, explained 5% and left a gap persistent enough (b_wpi 0.97
  in the model) that wages told the model nothing about the level of expectations.

- **Productivity** is GDP per hour worked, not the wage identity (Δhcoe − Δulc): JobKeeper ran
  through compensation of employees and would show up in a productivity backed out of it.
- **The productivity trend is trailing, not HP.** A two-sided filter bends toward the latest
  data at the sample end, which is where the question is, and around COVID. The log index is
  first joined by a straight line from 2019Q4 to 2023Q1, the quarters at which it stood at the
  same level, so the COVID rise and unwind are gone; the trend is then the change over a
  trailing 5 years. Five rather than ten so it responds within a few years to a change in trend
  (an AI lift, if it comes), at the cost of more noise before 2020. The change is between
  annual averages of the level, not single quarters, so quarterly noise at the two ends does
  not swing it; Henderson was rejected for the same reason as HP, since its end weights revise.
  The index has been flat since 2019Q4: the trend reads about zero in 2026.
- **COVID is left out** of this equation, 2020Q1 to 2023Q2: JobKeeper and the fall in
  participation broke the link between measured unemployment and wages, and composition lifted
  productivity and then took it back. GDP per hour peaked 2022Q1, about 5 above its 2019Q4
  level, was back to that level by 2022Q4, and dipped below it until 2023Q2; the window
  runs to the end of the dip. The AR(1) gap restarts after the hole, and after the vacancy
  suspension, which with the four-quarter average leaves 2008Q3-2010Q2 out.
- **Year-ended**, because wage rounds are annual, with an AR(1) gap for the overlap, as for
  inflation.

V/U has no trend correction, so a shift in the Beveridge curve (more vacancies per unemployed
at the same tightness) reads as tightness. If the curve has moved out since COVID, today's
tightness is overstated and less of the wage excess is left for expectations than should be.

**What it found (October 2026).** With log V/U, wages carry information about the level of
expectations: the gap's persistence is near what the year-ended overlap alone produces, and the
tightness slope is well determined. They agree with the surveys. Recent wage growth is close to
what expectations near 2.5%, near-zero productivity growth and a tight labour market explain,
with a small unexplained excess. That conclusion rests on the constant, which is estimated over
the whole sample. Over 2015-2019 wages ran about half a point below the equation; measured
against those years, recent wages are well above it. Whether the 2010s were the normal state or
the unusual one is a judgement the equation cannot make, and it decides whether recent wages
point to higher expectations.

### 7. Inflation Observations: Quarterly Rates with an AR(1) Gap

Published inflation shapes expectations with a lag, so observation equations use inflation_{t-1} rather than contemporaneous inflation.

Inflation is observed as a quarterly annualised rate, and its deviation from expectations (the gap) follows an AR(1) across quarters, after Chan, Clark and Koop (2018). Year-ended rates would not do: consecutive readings share three of their four quarters, so their deviations from expectations are correlated by construction, and treating them as independent counts one persistent surge several times as evidence that expectations moved. The AR(1) lets a persistent inflation surge sit in the gap rather than in expectations. Underlying inflation and pre-1993 headline CPI each have their own persistence b and innovation σ.

The survey bias terms (λ) keep year-ended lagged inflation, since that is the rate forecasters read.

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

Initial state: `πᵉ_0 ~ N(inflation_1983Q1, 2.0)`, year-ended underlying inflation in the first quarter

### Observation Equations

**Survey/market measures** (m = market_1y, breakeven as configured):
```
measure_{m,t} = πᵉ_t + α_m + λ_m × π_{t-1} + ε_{m,t}
ε_{m,t} ~ N(0, σ_{obs,m})
```

**Trimmed mean/weighted median inflation** (lagged, quarterly annualised, Unanchored, Short Run):
```
gap_t = inflation_{t-1} − πᵉ_t
gap_t = b_inflation × gap_{t-1 quarter} + e_t,    e_t ~ N(0, σ_inflation)
b_inflation ~ Beta(2, 2)                          [stationary; first quarter at σ/√(1−b²)]
```

**Headline CPI** (lagged, quarterly annualised, 1987-1992, Unanchored, Short Run):
```
gap_t = headline_{t-1} − πᵉ_t
gap_t = b_headline × gap_{t-1 quarter} + e_t,     e_t ~ N(0, σ_headline)
b_headline ~ Beta(2, 2)
```

**Private WPI growth** (year-ended, 1998Q3 onwards, 2008Q3-2010Q2 and COVID 2020Q1-2023Q2 excluded, Unanchored):
```
gap_t = wpi_t − (πᵉ_t + prod_t + wpi_adjustment + β × tight_t)
gap_t = b_wpi × gap_{t-1 quarter} + e_t,          e_t ~ N(0, σ_wpi)    [restarts after the hole]
prod_t  = 5-year change in annual-average log GDP per hour, per year, index flattened 2019Q4-2023Q1
tight_t = four-quarter average of (log(V/U)_t − its full-history mean)
```

**Nominal 10y bonds** (multiplicative Fisher):
```
nominal_t ~ N(πᵉ_t + real_rate + (πᵉ_t × real_rate / 100), σ_nominal)
real_rate ~ N(5.0, 1.5)  # Estimated
```

**HCOE growth** (Unanchored):
```
hcoe_t ~ N(πᵉ_t + mfp_t + hcoe_adjustment, σ_hcoe)
hcoe_adjustment ~ N(0, 0.5)
```

### Priors

| Parameter | Prior | Models |
|-----------|-------|--------|
| α (series effects) | N(0, 0.5) | Unanchored |
| λ (backward-looking bias) | N(0.1, 0.15) | Unanchored |
| σ_obs | HalfNormal(1.0) | All |
| σ_inflation (gap innovation) | HalfNormal(1.5) | Unanchored (Short Run shares σ_obs instead) |
| b_inflation (gap persistence) | Beta(2, 2) | Unanchored, Short Run |
| σ_headline (gap innovation) | HalfNormal(2.0) | Unanchored, Short Run |
| b_headline (gap persistence) | Beta(2, 2) | Unanchored, Short Run |
| σ_nominal | HalfNormal(2.0) | Unanchored, Long Run |
| σ_hcoe | HalfNormal(2.0) | Unanchored |
| σ_late (innovation) | HalfNormal(0.075), scale from the config's sigma_late | Short Run |
| hcoe_adjustment | N(0, 0.5) | Unanchored |
| wpi_adjustment | N(0, 1.0) | Unanchored |
| β (WPI tightness slope) | N(1.0, 1.0) | Unanchored |
| b_wpi (gap persistence) | Beta(2, 2) | Unanchored |
| σ_wpi (gap innovation) | HalfNormal(1.0) | Unanchored |
| real_rate | N(5.0, 1.5) | Unanchored, Long Run |

**Note on real_rate (r*)**: Estimates differ between Unanchored and Long Run because of the identification trade-off with π_exp. The Long Run estimate is less well pinned, since it has only a 2yr overlap to anchor r* against Unanchored's full pre-1993 period.

---

## Model Configuration Summary

| Feature | Unanchored | Short Run (1yr) | Long Run (10yr) |
|---------|------------|-----------------|-----------------|
| Survey series | market_1y, breakeven, business, market_yoy | market_1y | breakeven |
| Survey bias (α, λ) | Yes | No | No |
| Innovation variance | Fixed (0.30/0.07) | Early fixed (0.12), late estimated | Fixed (0.12/0.075) |
| Headline CPI (1987-1992, AR(1) gap) | Yes | Yes | No |
| Nominal bonds | Yes (pre-1993Q3) | No | Yes (pre-1988Q3, 2yr overlap) |
| HCOE growth | Yes | No | No |
| Private WPI growth (tightness, AR(1) gap) | Yes | No | No |
| Inflation observation (AR(1) gap) | Yes | Yes (σ shared with market_1y) | No |

---

## Per-Model Equations

### Long Run (10-Year Bond): `market`

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
  real_rate ~ N(5.0, 1.5)
```

**Note:** The 2-year overlap (1986Q3-1988Q3) provides limited data for separating r* from π_exp, so its r* differs from Unanchored's and the π_exp path adjusts to compensate.

### Short Run (1 Year): `short`

Simplified model with estimated late innovation variance. Uses market_1y survey and inflation (AR(1) gap). No survey bias terms (α, λ), no nominal bonds, no HCOE.

**State equation:** Late innovation variance estimated (unlike Unanchored/Long Run, where both are fixed).
```
πᵉ_t = πᵉ_{t-1} + ε_t

where:
  ε_t ~ StudentT(ν=4, μ=0, σ=0.12)    for t < 1994Q1    [fixed]
  ε_t ~ StudentT(ν=4, μ=0, σ_late)    for t ≥ 1994Q1

  σ_late ~ HalfNormal(0.075)          [keeps weight off the tail where the walk chases inflation surges]
```

**Observation equations:**
```
market_1y_t = πᵉ_t + ε_t                                        [no α, no λ]
  ε_t ~ N(0, σ_obs)

π_{t-1} − πᵉ_t = b_inflation × (previous quarter's gap) + e_t   [AR(1) gap, e_t ~ N(0, σ_obs): shared]

headline_{t-1} − πᵉ_t = b_headline × (previous gap) + e_t       [AR(1) gap, 1987-1992]

σ_obs ~ HalfNormal(1.0)
```

With one survey and its own free σ_obs, the survey noise collapses and the walk pins to the survey; sharing σ_obs with the inflation gap holds the two apart.

### Unanchored: `unanchored`

The full model, and the series downstream models read. Uses market_1y, breakeven, business, market_yoy (all with α, λ), plus inflation, headline CPI, nominal bonds, HCOE and private WPI. No target observation. Innovation variance is fixed, to avoid funnel geometry in the posterior.

**State equation:** Fixed innovation variance.
```
πᵉ_t = πᵉ_{t-1} + ε_t

where:
  ε_t ~ StudentT(ν=4, μ=0, σ=0.30)   for t < 1994Q1
  ε_t ~ StudentT(ν=4, μ=0, σ=0.07)   for t ≥ 1994Q1

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

π_{t-1} − πᵉ_t = b_inflation × (previous quarter's gap) + e_t   [AR(1) gap, σ_inflation]

headline_{t-1} − πᵉ_t = b_headline × (previous gap) + e_t       [AR(1) gap, 1987-1992]

nominal_t = πᵉ_t + real_rate + (πᵉ_t × real_rate / 100) + ε_t   [pre-1993Q3 only]
  ε_t ~ N(0, σ_nominal)
  real_rate ~ N(5.0, 1.5)

hcoe_t = πᵉ_t + mfp_t + hcoe_adjustment + ε_t
  ε_t ~ N(0, σ_hcoe)

wpi_t − (πᵉ_t + prod_t + wpi_adjustment + β × tight_t) = b_wpi × (previous gap) + e_t
                                [AR(1) gap, 2008Q3-2010Q2 and COVID 2020Q1-2023Q2 excluded]
```

---

## Validation

### Comparison with PIE_RBAQ

PIE_RBAQ is MARTIN's inflation expectations variable, from the MacroDave database (`github.com/MacroDave/MARTIN`), held as a static CSV in `input_data/`. RDP 2019-07 states it is exogenous to MARTIN and built after Cusbert (2017), so both it and this model are long-run anchors rather than one-year-ahead forecasts. It runs from 1970Q1 to 2019Q1; the unanchored series is charted against it.

PIE_RBAQ imposes credibility and this model does not, so they should agree where expectations sat at the target and part where they did not. In the 1983-1993 disinflation, PIE_RBAQ shows a sawtooth that does not match headline inflation or other indicators, where this model produces a smoother decline informed by HCOE growth, more consistent with wage-setting behaviour.

### Testing for De-anchoring

No model observes the target, so de-anchoring is read directly as distance from 2.5%:

1. **Unanchored**: every input, no target. Its distance from 2.5% is the measure.
2. **Short Run** and **Long Run**: different observation sets, a check on whether one horizon is driving it.

**Signs of de-anchoring:**
1. Unanchored persistently away from 2.5%
2. Short Run/Long Run estimates persistently away from 2.5% in the same direction
3. Survey alphas shifting over time
4. Breakeven inflation moving away from target
5. HCOE growth exceeding π_exp + MFP
6. Private WPI growth persistently above π_exp + productivity + tightness (the WPI equation's
   gap staying positive), the sharper wage test since the tightness term was added

---

## Usage

### Command Line

```bash
# Stage 1: Run all three models (default: 1983Q1 start, quarterly, 10000 draws, 4 chains)
uv run python -m src.models.expectations.stage1

# Run single model
uv run python -m src.models.expectations.stage1 --model unanchored

# Monthly grid instead of quarterly
uv run python -m src.models.expectations.stage1 --monthly

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
from src.models.expectations.stage1 import run_model, save_results
from src.models.expectations.stage2 import load_results, load_all_results

# Run and save a single model (code: "unanchored", "short", or "market")
trace, measures, inflation, index = run_model("unanchored", start="1984Q1")
save_results("unanchored", trace, measures, inflation, index)

# Load saved results (stage2)
results = load_results("unanchored")
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

## Downstream Use

The unanchored median is the series other models read, from `expectations_unanchored_hdi_quarterly.parquet` via `src/data/expectations_model.py`.
