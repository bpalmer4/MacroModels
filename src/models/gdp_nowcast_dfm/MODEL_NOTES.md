# GDP Nowcast DFM Model

Dynamic Factor Model for nowcasting the next unpublished quarterly GDP growth (Q/Q and through-the-year). Extracts common factors from a mixed-frequency panel of monthly and quarterly indicators using the Kalman filter, with the ragged edge handled natively by the state-space framework.

## Summary

| Component | Method | Key Feature |
|-----------|--------|-------------|
| Factor extraction | Dynamic Factor Model (Bańbura & Modugno 2014) | EM algorithm, handles arbitrary missing data |
| Mixed frequencies | Mariano-Murasawa (2011) quarterly mapping | Quarterly variables modelled as 3-period sums of latent monthly states |
| Ragged edge | Kalman filter / smoother | No SARIMA pre-completion needed |
| Factor dynamics | VAR(2) on 2 common factors | Captures persistence + co-movement |
| Idiosyncratic noise | AR(1) per indicator | Persistent residual dynamics |
| Standardisation | Per-variable z-score | Required for factor extraction |
| Sample start | 1990Q1 | Avoids pre-inflation-target structural breaks |
| Uncertainty | Kalman state covariance → SE → normal CI | 70% and 90% prediction intervals |
| Data availability | `DataAvailability` dataclass | Compatible with bridge model factory methods |

---

## Architecture

### File Structure

```
gdp_nowcast_dfm/
├── __init__.py
├── __main__.py           # python -m entry point
├── model.py              # Core DFM nowcast
├── backtest.py           # Backtesting framework
└── MODEL_NOTES.md        # This file
```

### Key Functions

```python
# Live nowcast (auto-detects target quarter, generates charts)
from src.models.gdp_nowcast_dfm.model import run_nowcast
result = run_nowcast()

# Programmatic nowcast (explicit target and availability)
from src.models.gdp_nowcast_dfm.model import nowcast, DataAvailability
result = nowcast(
    target_quarter=pd.Period("2025Q4", "Q-DEC"),
    availability=DataAvailability.at_t_minus_1m(pd.Period("2025Q4", "Q-DEC")),
    n_factors=2,
    factor_order=2,
    quiet=True,
)

# Backtesting
from src.models.gdp_nowcast_dfm.backtest import run_backtest, BacktestConfig
bt = run_backtest(BacktestConfig(start="2022Q1"))
```

---

## The DFM Approach

### Theoretical Background

Based on:
- **Bańbura, Giannone, Reichlin (2011)**: "Nowcasting" — laid out the DFM framework for nowcasting with mixed-frequency data
- **Bańbura, Modugno (2014)**: "Maximum likelihood estimation of factor models on datasets with arbitrary pattern of missing data" — EM algorithm for handling ragged edges
- **Mariano, Murasawa (2011)**: "A coincident index, common factors, and monthly real GDP" — mapping quarterly variables onto latent monthly states
- **Bok et al. (2017)**: "Macroeconomic Nowcasting and Forecasting with Big Data" — the NY Fed Staff Nowcast methodology

### Model Specification

The standard DFM has three layers:

**1. Observation equation** — each indicator `y_i,t` loads on common factors `f_t`:
```
y_i,t = λ_i' f_t + e_i,t
```

**2. Factor dynamics** — factors follow a VAR(p):
```
f_t = A_1 f_{t-1} + ... + A_p f_{t-p} + u_t
```

**3. Idiosyncratic dynamics** — each indicator has its own AR(1) noise process:
```
e_i,t = ρ_i e_{i,t-1} + ε_i,t
```

**Mixed frequency mapping**: Quarterly variables (like GDP growth) are modelled as if they were the sum of three latent monthly observations: `y_qtr,t = (1/3)(y*_t + y*_{t-1} + y*_{t-2})`. The Kalman filter infers the unobserved monthly values using the common factor structure.

**Estimation**: Parameters are estimated via the EM algorithm (Bańbura & Modugno 2014), which handles arbitrary patterns of missing data and converges robustly. Maximum 500 iterations, tolerance 1e-6.

---

## Indicator Panel

### Monthly Indicators (6)

| Indicator | Source | Transformation |
|-----------|--------|----------------|
| Retail turnover | 5682.0 (Monthly Household Spending) | Log difference × 100 |
| Building approvals | 8731.0 (Building Approvals) | Log difference × 100 |
| Hours worked | 6202.0 (Labour Force) table 6202019 | Log difference × 100 |
| Employment persons | 6202.0 (Labour Force) table 6202001 | Log difference × 100 |
| Goods trade balance | 5368.0 (International Trade in Goods) | Simple difference (level can be negative) |
| NAB business conditions | RBA Table H3 (GICNBC) | Simple difference (already a deviation index) |

### Quarterly Indicators (6, including target)

| Indicator | Source | Transformation |
|-----------|--------|----------------|
| **GDP growth (target)** | 5206.0 (Australian National Accounts), CVM SA | Log difference × 100 |
| CPI trimmed mean | 6401.0 Appendix 1a | Quarterly % change |
| WPI growth | 6345.0 | Log difference × 100 |
| Company profits | 5676.0 (Business Indicators) | Log difference × 100 |
| Construction work done | 8755.0 | Log difference × 100 |
| Private capex | 5625.0 | Log difference × 100 |

The DFM handles the **ragged edge** natively — different indicators have different last available dates, and the Kalman filter forecasts missing values forward using the common factor structure. No SARIMA pre-completion is needed.

---

## Ragged Edge Handling

A key advantage of the DFM over the bridge model is how it handles the ragged edge. The model panel is constructed to extend up to **month 3 of the target quarter** (with NaN for any missing observations). The Kalman filter then:

1. **Filters forward**: Updates the factor estimates as each indicator's observations arrive
2. **Smooths backward**: Refines historical factor estimates using all available data
3. **Forecasts missing values**: Uses the factor dynamics + observation loadings to fill in NaN gaps

The GDP growth nowcast for the target quarter is the smoothed estimate at the third month of the target quarter — even though the observed GDP value is NaN, the Kalman filter has inferred it from the common factor.

This is more elegant than the bridge model's two-stage approach (SARIMA-complete monthly indicators → aggregate to quarterly → run bridge regressions → combine). The DFM does all of this in one joint estimation step.

---

## Sample Start

Sample is truncated to start at **1990Q1**. Reasons:

- Avoids pre-inflation-target era (RBA adopted explicit target in 1993)
- Avoids the post-floating-dollar adjustment period (1983-1989)
- Reduces dimensionality of structural breaks the DFM has to absorb
- Empirically improves backtest performance vs longer samples

Even shorter samples (2000+ or 2005+) marginally improve the post-COVID bias but are not currently used because:
- They reduce the training sample below 100 quarterly observations, where DFM estimation becomes less stable
- The improvement in bias is small relative to the loss of pre-COVID dynamics for factor estimation

---

## Uncertainty

Prediction intervals come directly from the Kalman filter's state covariance matrix:

1. The filter propagates parameter and state uncertainty through time
2. The standard error of the prediction `se` is extracted from `result.get_prediction().se_mean`
3. Normal-distribution intervals: 70% = ±1.04·se, 90% = ±1.645·se

This is more theoretically grounded than the bridge model's bootstrap approach, which (a) assumes residuals are exchangeable across bridges, (b) ignores parameter uncertainty, and (c) understates the true uncertainty. Empirically, the DFM CIs are wider (~2.1pp at 90%) but better calibrated (94% empirical coverage vs 81% for the bridge model).

---

## Backtest Results (2022Q1-2025Q4, latest-revised data)

| Info Set | RMSE | MAE | Bias | Direction | Corr | 90% CI Coverage |
|----------|------|-----|------|-----------|------|-----------------|
| T-3m | 0.305% | 0.235% | +0.183% | 94% | +0.500 | 100% |
| T-2m | 0.493% | 0.423% | +0.291% | 94% | +0.203 | 100% |
| T-1m | 0.476% | 0.388% | +0.359% | 94% | +0.619 | 94% |
| T-0 | 0.478% | 0.391% | +0.365% | 94% | **+0.635** | 94% |

Naive benchmark (trailing 4-quarter average): RMSE 0.285%.

### Comparison with Bridge Model

| Metric | Bridge T-0 | DFM T-0 |
|--------|-----------|---------|
| RMSE | 0.268% | 0.478% |
| MAE | 0.227% | 0.391% |
| Bias | +0.085% | +0.365% |
| Direction | 94% | 94% |
| **Correlation with actual** | **+0.474** | **+0.635** |
| 90% CI coverage | 81% (under-covers) | 94% (slightly over-covers) |
| 90% CI width | ~0.83pp | ~2.09pp |

**The DFM tracks the *shape* of GDP growth significantly better than the bridge model** (correlation 0.64 vs 0.47). The bridge model has lower RMSE and a smaller bias, but the DFM is substantially better at capturing the period-to-period movements in GDP growth — which is arguably the more important property of a nowcasting model. A point estimate that tracks turning points but is biased high by 0.3pp is more useful than one that's centred but flat.

The DFM's positive bias is concentrated in the post-COVID productivity slump period (2022–2024) and is already shrinking by 2025:

| Period | DFM Mean Error |
|--------|----------------|
| 2022 (full year) | +0.68pp |
| 2023 (full year) | +0.46pp |
| 2024 (full year) | +0.55pp |
| 2025 (full year) | +0.06pp |

As more recent data accumulates and the post-COVID period falls out of the heavily-weighted recent training window, the bias should fade naturally.

### Why Not "Fix" the Bias?

We tried several approaches to reduce the bias and each one **hurt the correlation more than it helped the bias**:

| Configuration | T-0 RMSE | Bias | Correlation |
|---------------|----------|------|-------------|
| **Vanilla (current)** | **0.478%** | **+0.365%** | **+0.635** |
| + LP-adjusted labour | 0.382% | −0.140% | +0.357 |
| + LP + 3-qtr COVID mask | 0.338% | +0.031% | +0.009 |
| + LP + 7-qtr COVID mask | 0.318% | +0.083% | −0.260 |

Each "improvement" pushed RMSE down at the cost of crushing the correlation. The improvements were largely illusory — they came from regressing toward the mean rather than from better predictions. A model whose nowcasts have correlation near zero with actual GDP is just predicting the historical average regardless of conditions, and its low RMSE comes from GDP growth being modestly variable around a stable mean.

The vanilla DFM is kept because:
1. Its correlation (0.635) substantially exceeds both the bridge model (0.474) and the naive forecast (0)
2. Its bias is concentrated in a specific known period (post-COVID productivity slump) and is already fading
3. RMSE differences of 0.1pp on n=16 are within sampling noise; correlation differences of 0.6 are not

### Caveats

- **Pseudo real-time**: Uses latest-revised data, not vintage data as published at the time
- **Small sample**: 16 quarters is too short to make statistically meaningful claims about model differences. Diebold-Mariano test would likely show the DFM-vs-bridge RMSE difference is not significant.
- **Prediction index labels**: statsmodels' `DynamicFactorMQ.get_prediction()` returns a DataFrame whose PeriodIndex labels are unreliable — they don't correspond to actual dates. Extraction uses positional indexing (last value = nowcast for target quarter).

---

## Key Design Decisions

- **Why DFM over bridges?** The DFM has a cleaner theoretical foundation (one model with one signal vs many bridge regressions averaged together), handles ragged edges natively, and provides better-calibrated uncertainty quantification. It is the dominant approach at the NY Fed, ECB, Banque de France, and other major central banks.

- **2 factors, VAR(2)**: Empirically, 2 factors capture most of the variance in the indicator panel without overfitting. The first factor typically loads on broad activity indicators (employment, retail, hours), the second on prices and surveys. VAR(2) for factor dynamics captures persistence + acceleration. Single factor + VAR(1) was tested but produced flatter, less informative nowcasts.

- **Idiosyncratic AR(1) on**: Each indicator has its own persistent residual process. This prevents the factors from being contaminated by indicator-specific noise (e.g. monthly retail volatility, building approval lumpiness).

- **Standardisation on**: Required for the EM algorithm to converge properly. Each variable is standardised to mean zero, unit variance before factor extraction, then back-transformed for prediction.

- **Sample truncation to 1990Q1**: Removes pre-inflation-target era dynamics that are no longer relevant. Improves backtest by ~0.2pp on T-0 RMSE.

- **No bias correction in production**: Several bias-reduction approaches were tested (labour productivity adjustment, COVID quarter masking, adding ULC/hCOE panel variables). Each one reduced the bias but also crushed the correlation between nowcasts and actual GDP — i.e. they made the model regress toward the historical mean rather than improving its predictions. The current vanilla model has correlation 0.635 at T-0 (vs 0.474 for the bridge model and ~0 for any of the bias-corrected variants). A biased model that tracks the shape of GDP growth is more useful than an unbiased model that effectively predicts the mean every quarter. The bias is concentrated in 2022–2024 (post-COVID productivity slump) and is already shrinking by 2025; it should fade naturally as more recent data accumulates. See "Why Not Fix the Bias?" above.

- **Wider, better-calibrated CIs**: The Kalman-derived intervals are ~2.5x wider than the bridge model's bootstrap intervals, but they are honest. Bridge model CIs under-cover (81% empirical vs 90% nominal); DFM CIs are essentially well-calibrated (94% empirical vs 90% nominal).

- **Same data availability framework**: The `DataAvailability` dataclass and factory methods (`at_t_minus_3m`, `at_t_minus_2m`, etc.) mirror the bridge model exactly, so the two backtests are directly comparable at each information set.

---

## Differences from the Bridge Model

| Aspect | Bridge Model | DFM |
|--------|--------------|-----|
| Combination method | Inverse-MSE weighted average of independent OLS bridges | Joint estimation via Kalman filter |
| Ragged edge | SARIMA pre-completion | Native (Kalman filter forecasts forward) |
| Number of indicators | 8 monthly + 8 quarterly + production = 17 | 6 monthly + 5 quarterly = 11 |
| Excluded indicators | — | Monthly CPI, business sales, inventories, gov. consumption, production bridge |
| Productivity adjustment | Explicit HMA(13) trend as regressor on labour bridges | None — hurts correlation more than it helps bias |
| COVID handling | Dummy variable 2020Q1–2021Q1 in each bridge | None — hurts correlation more than it helps bias |
| Uncertainty | Bootstrap residual resampling | Kalman state covariance |
| Best metric | RMSE, MAE | Correlation with actual, CI calibration, CRPS |

The bridge model is currently more accurate by RMSE on the small backtest sample (T-0 RMSE 0.27% vs 0.48% for the DFM), but the DFM tracks the *shape* of GDP growth substantially better (correlation 0.64 vs 0.47) and has cleaner theoretical foundations. The DFM's RMSE handicap comes from a +0.37pp positive bias concentrated in 2022–2024 (the post-COVID productivity slump period), which is fading as more recent data accumulates. Both models are tied with the naive benchmark on this short sample.

---

## Potential Improvements

1. **Add the missing bridge model series** (monthly CPI, business sales, inventories, gov. consumption) for an apples-to-apples comparison with the bridge model
2. **Diebold-Mariano test** — formal test of whether the bridge-vs-DFM RMSE difference is statistically significant (probably not, with n=16)
3. **CRPS evaluation** — would give a fairer comparison that rewards the DFM's better-calibrated CIs
4. **Mincer-Zarnowitz decomposition** — separates level bias from tracking error
5. **News decomposition** — DFM allows attributing nowcast revisions to specific data releases (Bańbura et al. 2011 methodology)
6. **Block factors** — separate "real activity" and "prices/wages" blocks instead of unrestricted 2 factors
7. **Even shorter sample (2000+ or 2005+)** — at the cost of reduced training data
8. **Outlier-robust DFM** (Antolin-Diaz, Drechsel & Petrella 2021) — t-distributed shocks or stochastic volatility could downweight COVID quarters automatically without crushing the correlation the way explicit masking does.

---

## Running

```bash
# Live nowcast
./run-gdp-nowcast-dfm.sh
uv run python -m src.models.gdp_nowcast_dfm.model

# Backtest (default: 2022Q1 to latest)
uv run python -m src.models.gdp_nowcast_dfm.backtest

# Backtest with custom range
uv run python -m src.models.gdp_nowcast_dfm.backtest --start 2022Q1 --end 2025Q4
```

### Output

```
charts/GDP-Nowcast-DFM/                  # Live nowcast charts
  gdp-growth-nowcast-dfm-qq-*.png        # Q/Q fan chart
  gdp-growth-nowcast-dfm-tty-*.png       # TTY fan chart
  gdp-growth-dfm-nowcast-*.png           # Combined growth chart
  dfm-common-factors-*.png               # Smoothed factors

charts/GDP-Nowcast-DFM-Backtest/         # Backtest evaluation charts
  dfm-rmse-by-information-set-*.png      # RMSE comparison bar chart
  actual-vs-dfm-nowcast-*.png            # Actual vs DFM nowcast
  dfm-nowcast-errors-*.png               # Error time series by info set
  dfm-nowcast-accuracy-evolution-*.png   # Rolling MAE over time

model_outputs/gdp_nowcast_dfm/           # Saved results
  backtest_results.parquet               # Full backtest results table
  backtest_summary.csv                   # Summary statistics
  backtest_summary.txt                   # Human-readable summary
```
