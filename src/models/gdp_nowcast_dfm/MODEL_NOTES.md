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

### Monthly Indicators (7)

| Indicator | Source | Transformation |
|-----------|--------|----------------|
| Retail turnover | 5682.0 (Monthly Household Spending) | Log difference × 100 |
| Building approvals | 8731.0 (Building Approvals) | Log difference × 100 |
| Hours worked | 6202.0 (Labour Force) table 6202019 | Log difference × 100 |
| Employment persons | 6202.0 (Labour Force) table 6202001 | Log difference × 100 |
| Goods trade balance | 5368.0 (International Trade in Goods) | Simple difference (level can be negative) |
| NAB business conditions | RBA Table H3 (GICNBC) | Simple difference (already a deviation index) |
| Westpac-MI consumer sentiment | RBA Table H3 (GICWMICS) | Log difference × 100 |

**Westpac-MI consumer sentiment was tested and kept.** It improved RMSE, lifted correlation with actual GDP, and reduced nowcast variance over the backtest window. After standardisation the panel treats sentiment as a second soft-data factor — it tends to load onto the prices/surveys factor alongside NAB and adds information during periods where the two surveys diverge (households vs. businesses).

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

### Indicators tested and rejected

- **BoP services-only balance (5302.0)**: Tested 2026-05-28 as a quarterly indicator (SA "Services ;" change from table 530204, T-0 only since 5302.0 publishes ~1 day before GDP). T-0 RMSE was flat at 0.442% with a marginal correlation improvement (+0.682 → +0.688) over the 2022Q1–2025Q4 backtest. Effectively neutral — the factor structure didn't latch onto useful shared variance with the rest of the panel, so the marginal information value was indistinguishable from noise. Not retained, since adding it complicates the panel without measurable benefit. The data loader (`src/data/balance_of_payments.py`) is retained for reference.

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

This is more theoretically grounded than the bridge model's bootstrap approach, which (a) assumes residuals are exchangeable across bridges, (b) ignores parameter uncertainty, and (c) understates the true uncertainty. Empirically, the DFM CIs are noticeably wider than the bridge's bootstrap intervals but cover the actual outturn at close to nominal rates, whereas the bridge intervals tend to under-cover.

---

## Empirical Performance

For current empirical performance, run:
```bash
uv run python -m src.models.gdp_nowcast_dfm.backtest
```

The backtest reports RMSE, MAE, bias, direction accuracy, correlation with actual GDP, and 90% CI coverage at four information sets (T-3m, T-2m, T-1m, T-0).

### Qualitative findings

- **The DFM tracks the *shape* of GDP growth significantly better than the bridge model** — correlation with actual GDP is materially higher. The trade-off is higher headline RMSE driven by a positive bias.
- **Bias is concentrated in the post-COVID productivity slump (2022–2024)** and has been fading. By 2025 it is essentially gone. As the test sample rolls forward, the headline numbers improve naturally.
- **Bias-correction approaches hurt more than they help** — see "Why Not Fix the Bias?" below.
- **CIs are well calibrated** — Kalman-derived intervals come close to nominal coverage, whereas the bridge's bootstrap intervals tend to under-cover.
- **The DFM beats the trailing-4-quarter naive benchmark on correlation** but ties or trails it on RMSE because of the bias.

### Why Not "Fix" the Bias?

Several bias-reduction approaches were tested — labour-productivity-adjusted labour input, COVID-quarter masking (3-quarter and 7-quarter windows), and combinations. **Each one reduced the bias but crushed the correlation between nowcasts and actual GDP.** The improvements were largely illusory: they came from regressing toward the mean rather than from better predictions. A model whose nowcasts have correlation near zero with actual GDP is just predicting the historical average regardless of conditions, and its low RMSE comes from GDP growth being modestly variable around a stable mean.

The vanilla DFM is kept because:
1. Its correlation with actual GDP substantially exceeds the bridge model's and the naive forecast's
2. Its bias is concentrated in a specific known period (post-COVID productivity slump) and is already fading
3. RMSE differences of ~0.1 pp on a small backtest sample are within sampling noise; correlation differences of the magnitudes seen here are not

### Caveats

- **Pseudo real-time**: Uses latest-revised data, not vintage data as published at the time
- **Small sample**: The backtest window is too short to make statistically meaningful claims about model differences. A formal Diebold-Mariano test would likely fail to reject equal RMSE between the DFM and the bridge.
- **Prediction index labels**: statsmodels' `DynamicFactorMQ.get_prediction()` returns a DataFrame whose PeriodIndex labels are unreliable — they don't correspond to actual dates. Extraction uses positional indexing (last value = nowcast for target quarter).

---

## Key Design Decisions

- **Why DFM over bridges?** The DFM has a cleaner theoretical foundation (one model with one signal vs many bridge regressions averaged together), handles ragged edges natively, and provides better-calibrated uncertainty quantification. It is the dominant approach at the NY Fed, ECB, Banque de France, and other major central banks.

- **2 factors, VAR(2)**: Empirically, 2 factors capture most of the variance in the indicator panel without overfitting. The first factor typically loads on broad activity indicators (employment, retail, hours), the second on prices and surveys. VAR(2) for factor dynamics captures persistence + acceleration. Single factor + VAR(1) was tested but produced flatter, less informative nowcasts.

- **Idiosyncratic AR(1) on**: Each indicator has its own persistent residual process. This prevents the factors from being contaminated by indicator-specific noise (e.g. monthly retail volatility, building approval lumpiness).

- **Standardisation on**: Required for the EM algorithm to converge properly. Each variable is standardised to mean zero, unit variance before factor extraction, then back-transformed for prediction.

- **Sample truncation to 1990Q1**: Removes pre-inflation-target era dynamics that are no longer relevant. Empirically improves backtest performance vs longer samples.

- **No bias correction in production**: Several bias-reduction approaches were tested (labour productivity adjustment, COVID quarter masking, adding ULC/hCOE panel variables). Each one reduced the bias but also crushed the correlation between nowcasts and actual GDP — i.e. they made the model regress toward the historical mean rather than improving its predictions. A biased model that tracks the shape of GDP growth is more useful than an unbiased model that effectively predicts the mean every quarter. The bias is concentrated in 2022–2024 (post-COVID productivity slump) and has largely faded by 2025; it should disappear naturally as more recent data accumulates. See "Why Not Fix the Bias?" above.

- **Wider, better-calibrated CIs**: The Kalman-derived intervals are noticeably wider than the bridge model's bootstrap intervals, but they are honest. Bridge intervals under-cover (empirical < nominal); DFM intervals come close to nominal coverage.

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

The bridge model is currently more accurate by RMSE on the small backtest sample, but the DFM tracks the *shape* of GDP growth substantially better and has cleaner theoretical foundations. The DFM's RMSE handicap comes from a positive bias concentrated in 2022–2024 (the post-COVID productivity slump period), which is fading as more recent data accumulates. Both models are roughly tied with the naive benchmark on this short sample.

---

## Potential Improvements

1. **Add the missing bridge model series** (monthly CPI, business sales, inventories, gov. consumption) for an apples-to-apples comparison with the bridge model
2. **Diebold-Mariano test** — formal test of whether the bridge-vs-DFM RMSE difference is statistically significant (probably not given the small backtest sample)
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
