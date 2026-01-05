# Inflation Expectations Signal Extraction Model

This directory contains a Bayesian signal extraction model for estimating latent long-run inflation expectations from multiple survey and market-based measures, following Cusbert (2017).

## Summary

| Component | Method | Key Feature |
|-----------|--------|-------------|
| Sample | 1983Q1 to present | Extended early period using bonds and headline CPI |
| Latent State | Student-t random walk | σ=0.075, ν=4 allows occasional large shifts |
| Observation Model | Multiple measures with bias correction | Series effects (α) and backward-looking bias (λ) |
| Measures | Surveys + breakeven + early proxies | Business, market economists, breakevens, headline CPI, nominal bonds |

## Reference

Cusbert T (2017), "Estimating the NAIRU and the Unemployment Gap", RBA Bulletin, June, pp 13-22.

---

## Output Location

Results are saved to `output/expectations/`:
- `expectations_trace.nc` - Full MCMC trace (ArviZ InferenceData)
- `expectations_hdi.parquet` - Point estimates with HDI bounds
- `expectations_hdi.csv` - Same in CSV format

Charts are saved to `charts/expectations/`.

---

## File Structure

```
model.py                    # Model building, estimation, results container, CLI
analysis/
  __init__.py               # Re-exports analysis functions
  validation.py             # Validation utilities
```

---

## Data Sources

### Survey Measures (RBA Table G3)
| Series | Start | Description |
|--------|-------|-------------|
| business | 1989Q3 | NAB Business Survey inflation expectations |
| market_1y | 1993Q3 | Market economists 1-year ahead |
| market_yoy | 1994Q3 | Market economists year-on-year (reference series) |

### Market Measures (RBA Table F2)
| Series | Start | Description |
|--------|-------|-------------|
| breakeven | 1986Q3 | 10-year nominal yield minus indexed bond yield |
| nominal_10y | 1969 | Nominal 10-year government bond yield (pre-1986 only) |

### Inflation Measures (ABS 6401.0)
| Series | Start | Description |
|--------|-------|-------------|
| Trimmed mean | 1983Q1 | Annual trimmed mean CPI |
| Weighted median | 1983Q1 | Annual weighted median CPI |
| Headline CPI | 1949Q3 | Annual headline CPI (pre-1993 only) |

---

## Model Design Decisions

### 1. Student-t Innovations (σ=0.075, ν=4)

Normal innovations struggled to capture the sharp 1988-1992 disinflation while remaining smooth in the targeting era. Student-t with ν=4 allows occasional larger jumps during regime changes while staying smooth otherwise.

### 2. Series Effects (α)

Each measure has a systematic level effect relative to the reference series (market_yoy):
- **business**: α ≈ -0.7 (reads ~0.7pp lower)
- **market_1y**: α ≈ -0.4 (reads ~0.4pp lower)
- **breakeven**: α ≈ -0.6 (includes liquidity/term premium)
- **market_yoy**: α = 0 (reference)

### 3. GST Adjustment

Market economists' year-on-year expectations in 1999Q3-Q4 were distorted by GST anticipation (4.2%, 4.9%). These are interpolated through (2.45%, 2.35%) to avoid spurious expectations jump.

### 4. Early Period Anchoring

Pre-survey period (1983-1989) is anchored using:
- **Headline CPI** (pre-1993): Informative when expectations were adaptive
- **Nominal 10y bonds** (pre-1986): Nominal yield = expectations + real_rate (estimated ~6%)
- **Lagged inflation** observation equation throughout

### 5. Lagged Inflation Observations

Published inflation shapes expectations with a lag. Observation equations use inflation_{t-1} rather than contemporaneous inflation.

---

## Model Specification

### State Equation

```
πᵉ_t = πᵉ_{t-1} + ε_t,    ε_t ~ StudentT(ν=4, μ=0, σ=0.075)
```

Initial state: `πᵉ_0 ~ N(inflation_1983Q1, 2.0)` ≈ N(10.2, 2.0)

### Observation Equations

**Survey/market measures** (m = business, market_1y, breakeven, market_yoy):
```
measure_{m,t} = πᵉ_t + α_m + λ_m × π_{t-1} + ε_{m,t}
ε_{m,t} ~ N(0, σ_{obs,m})
```

**Trimmed mean/weighted median inflation** (lagged, full sample):
```
inflation_{t-1} ~ N(πᵉ_t, σ_inflation)
```

**Headline CPI** (lagged, pre-1993 only):
```
headline_{t-1} ~ N(πᵉ_t, σ_headline)
```

**Nominal 10y bonds** (pre-1986 only):
```
nominal_t ~ N(πᵉ_t + real_rate, σ_nominal)
real_rate ~ N(3.0, 1.0)  # Estimated ~6%
```

### Priors

| Parameter | Prior | Typical Estimate |
|-----------|-------|------------------|
| α (series effects) | N(0, 0.5) | -0.4 to -0.7 |
| λ (backward-looking bias) | N(0.1, 0.15) | -0.1 to 0.2 |
| σ_obs | HalfNormal(1.0) | 0.1 to 0.9 |
| σ_inflation | HalfNormal(1.5) | ~1.0 |
| σ_headline | HalfNormal(2.0) | ~1.9 |
| σ_nominal | HalfNormal(2.0) | ~0.7 |
| real_rate | N(3.0, 1.0) | ~6% |

---

## Usage

### Command Line

```bash
# Run with defaults (1983Q1 start, 10000 draws, 4 chains)
uv run python -m src.models.expectations.model

# Quick run for testing
uv run python -m src.models.expectations.model --draws 2000 --tune 1000

# Suppress progress bar
uv run python -m src.models.expectations.model -q
```

### Python API

```python
from src.models.expectations import run_model

results = run_model(start="1983Q1", verbose=True)

# Access results
median = results.expectations_median()
hdi = results.expectations_hdi(prob=0.9)
samples = results.expectations_posterior()

# Save
results.save()
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

Typical run time: ~30 seconds.

---

## Integration with NAIRU Model

The expectations series feeds into the NAIRU model's Phillips curve as the inflation anchor:

1. Run expectations model to extract πᵉ
2. Save results with `results.save()`
3. NAIRU model loads expectations from `output/expectations/expectations_hdi.parquet`

See `src/models/nairu/` for NAIRU model implementation.
