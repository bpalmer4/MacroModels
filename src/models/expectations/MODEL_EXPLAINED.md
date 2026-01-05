# Inflation Expectations Signal Extraction Model

This directory contains a Bayesian signal extraction model for estimating latent long-run inflation expectations from multiple survey and market-based measures, following Cusbert (2017).

## Summary

| Component | Method | Key Feature |
|-----------|--------|-------------|
| Sample | 1983Q1 to present | Extended early period using bonds and headline CPI |
| Latent State | Regime-switching Student-t random walk | σ_early=0.12 (pre-1993), σ_late=0.075 (post-1993) |
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

### 1. Regime-Switching Student-t Innovations

The innovation variance switches at 1993Q1 (inflation targeting adoption):

| Period | σ | Rationale |
|--------|---|-----------|
| Pre-1993 (early) | 0.12 | Volatile pre-targeting era, expectations unanchored |
| Post-1993 (late) | 0.075 | Anchored post-targeting era, RBA credibility |

Student-t with ν=4 allows occasional larger jumps (e.g., 1988-92 disinflation) while remaining smooth otherwise. The larger early variance reflects that expectations genuinely were more volatile before inflation targeting.

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

Two concatenated random walks with regime-switching innovation variance:

```
# Early period (1983Q1 to 1992Q4)
πᵉ_t = πᵉ_{t-1} + ε_t,    ε_t ~ StudentT(ν=4, μ=0, σ=0.12)

# Late period (1993Q1 onwards) - continues from early
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
real_rate ~ N(5.0, 1.5)  # Estimated ~5-6%
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
| real_rate | N(5.0, 1.5) | ~5-6% |

---

## Usage

### Command Line

```bash
# Run with defaults (1984Q1 start, 10000 draws, 4 chains)
uv run python -m src.models.expectations.model

# Quick run for testing
uv run python -m src.models.expectations.model --draws 2000 --tune 1000

# Suppress progress bar
uv run python -m src.models.expectations.model -q
```

### Python API

```python
from src.models.expectations import run_model

results = run_model(start="1984Q1", verbose=True)

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

The expectations series feeds into the NAIRU model's Phillips curves:

1. Run expectations model to extract πᵉ
2. Save results with `results.save()`
3. NAIRU model loads median expectations from `output/expectations/expectations_hdi.parquet` via `src/data/expectations_model.py`
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
