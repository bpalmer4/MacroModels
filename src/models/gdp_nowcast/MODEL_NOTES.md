# GDP Nowcast Model

Bridge equation model for nowcasting the next unpublished quarterly GDP growth (Q/Q and through-the-year). Combines high-frequency monthly indicators with quarterly data, using SARIMA completion for partial quarters and inverse-MSE weighted combination of bridge forecasts.

## Summary

| Component | Method | Key Feature |
|-----------|--------|-------------|
| Monthly indicators | SARIMA-completed partial quarters | Auto-fit from small candidate set by AIC |
| Quarterly indicators | Available or excluded | Switch on as published |
| Bridge equations | OLS with 1 lag + COVID dummy | One per indicator group |
| Production function | Cobb-Douglas with unfloored MFP trend | Independent inputs from Modellers Database + wage data |
| Labour adjustment | HMA(13) labour productivity trend | Corrects for productivity drag on labour bridges |
| Combination | Inverse out-of-sample MSE weights | Expanding-window evaluation |
| Uncertainty | Bootstrap residual resampling (1000 draws) | 70% and 90% prediction intervals |
| Data availability | `DataAvailability` dataclass | Per-indicator cutoffs for backtesting |

---

## Architecture

### File Structure

```
gdp_nowcast/
├── __init__.py
├── model.py              # Core nowcast model
├── backtest.py           # Backtesting framework
└── MODEL_NOTES.md        # This file
```

### Key Functions

```python
# Live nowcast (auto-detects target quarter, generates charts)
from src.models.gdp_nowcast.model import run_nowcast
result = run_nowcast()

# Programmatic nowcast (explicit target and availability)
from src.models.gdp_nowcast.model import nowcast, DataAvailability
result = nowcast(
    target_quarter=pd.Period("2025Q4", "Q-DEC"),
    availability=DataAvailability.at_t_minus_1m(pd.Period("2025Q4", "Q-DEC")),
    quiet=True,
)

# Backtesting
from src.models.gdp_nowcast.backtest import run_backtest, BacktestConfig
bt = run_backtest(BacktestConfig(start="2022Q1"))
```

---

## Bridge Groups

### Monthly Bridges (SARIMA-completed)

These arrive ~2 months before GDP publication. Missing months within the target quarter are forecast using the best-fitting SARIMA model from a small candidate set: (1,1,0), (1,1,1), (2,1,0), (0,1,1), with and without seasonal(12) component. Selected by AIC.

| Bridge | Indicator | Source | Aggregation |
|--------|-----------|--------|-------------|
| Consumption | Retail turnover (nominal) | 5682.0 (Monthly Household Spending) | Quarterly sum |
| Investment | Total dwelling approvals | 8731.0 (Building Approvals) | Quarterly sum |
| Labour: hours | Monthly hours worked | 6202.0 (Labour Force), table 6202019 | Quarterly sum |
| Labour: employment | Employed total persons | 6202.0 (Labour Force), table 6202001 | Quarterly mean |
| Trade | Balance on goods | 5368.0 (International Trade in Goods) | Quarterly sum |
| Prices: monthly CPI | CPI All Groups SA index (spliced) | 6484.0 + 6401.0 table 640106 | Quarterly mean |
| Survey: NAB conditions | NAB business conditions index (SA) | RBA Table H3 (GICNBC) | Quarterly mean |

The two labour bridges include an **HMA(13) labour productivity trend** as an additional regressor. This is derived from wage data (Δhcoe - Δulc) and corrects for periods where employment grows faster than output (negative productivity growth). Without this adjustment, the labour bridges systematically overpredict GDP during productivity downturns.

The **NAB business conditions bridge** uses the index level (deviation from long-run average, in percentage points), not growth — analogous to the goods trade balance bridge. This is the only soft data (survey) indicator in the bridge set. It publishes faster than most hard indicators (~2 weeks lag) and helps at mid-cycle horizons where hard data is still arriving.

The **monthly CPI bridge** uses a spliced index combining the discontinued Monthly CPI Indicator (6484.0, Sep 2017 – Sep 2025) with the current monthly CPI from 6401.0 table 640106 (Apr 2024 onwards). For SARIMA completion, only the genuine monthly observations are used (not interpolated quarterly history), ensuring the SARIMA model learns real monthly dynamics. For bridge equation estimation, the full spliced series (including quarterly-interpolated pre-2017 history) is used to maximise the training sample.

### Quarterly Bridges (available or excluded)

These are published shortly before GDP. The bridge is only active if the indicator is published for the target quarter.

| Bridge | Indicator | Source | Lead over GDP |
|--------|-----------|--------|---------------|
| Prices: CPI | CPI trimmed mean (quarterly) | 6401.0, Appendix 1a | ~4-5 weeks |
| Prices: WPI | WPI growth (log diff) | 6345.0 | ~2-3 weeks |
| Investment: construction | Total construction work done growth (CVM) | 8755.0 | ~2-3 weeks |
| Investment: private capex | Total private capex growth (CVM) | 5625.0 | ~1-2 weeks |
| Government: GFCE | Government consumption growth (spliced 5206.0 + GFS) | 5206.0 + GFS | ~1-2 weeks |
| Business: profits | Gross operating profits growth (nominal) | 5676.0 | ~1-2 days |
| Business: sales | Total sales (summed across industries, CVM) | 5676.0 | ~1-2 days |
| Business: inventories | Inventories growth (CVM) | 5676.0 | ~1-2 days |

### Production Bridge (Cobb-Douglas)

Uses independent inputs from the NAIRU model's production function, not duplicates of the monthly bridges:

```
potential_growth = α × dK + (1-α) × dL + dMFP
```

| Input | Source | Independence from monthly bridges |
|-------|--------|----------------------------------|
| α (capital share) | GOS/(GOS+COE) from 5206.0 income table | Yes: factor income shares |
| dK (capital growth) | Net capital stock from 1364.0 (Modellers Database) | Yes: different from building approvals |
| dL (labour force growth) | Total labour force from 1364.0 (Modellers Database) | Yes: labour force vs employment/hours |
| dMFP (MFP trend) | Solow residual from wage data, HMA(51) smoothed | Yes: derived from ULC and hourly COE |

**Critical: MFP is not floored at zero.** Unlike the NAIRU model (where negative MFP represents cyclical underutilisation, not technological regress), the nowcast model needs to capture actual negative productivity trends to avoid systematic overestimation during productivity downturns.

---

## Bridge Equation Specification

Each bridge equation is estimated by OLS:

```
GDP_growth_t = c + β × indicator_t + γ × indicator_{t-1} + δ × GDP_growth_{t-1} + θ × covid_t + ε_t
```

For labour bridges, the specification includes the productivity adjustment:

```
GDP_growth_t = c + β × indicator_t + γ × indicator_{t-1} + φ × lp_trend_t + ψ × lp_trend_{t-1}
             + δ × GDP_growth_{t-1} + θ × covid_t + ε_t
```

Where `lp_trend` is the HMA(13) labour productivity trend derived from wage data.

The COVID dummy covers 2020Q1-2021Q1.

---

## Combination

Bridge nowcasts are combined using inverse out-of-sample MSE weights:

```
w_i = (1/MSE_i) / Σ(1/MSE_j)
```

MSE is computed from expanding-window one-step-ahead forecasts (minimum training window: max(20, N/3) observations).

Only bridges with available data and finite MSE are included. This means the nowcast automatically improves as more indicators are published.

---

## Uncertainty

Bootstrap prediction intervals from pooled, weighted bridge residuals:

1. Pool residuals from active bridges, scaled by their combination weights
2. Resample with replacement (1000 draws)
3. Add resampled residuals to the combined nowcast point estimate
4. Report 70% (15th-85th percentile) and 90% (5th-95th percentile) intervals

---

## Data Availability

The `DataAvailability` dataclass specifies which data is available for a given nowcast. Monthly indicators have a `Period | None` cutoff (last reference month available). Quarterly indicators have a `bool` (published or not).

### Factory Methods for Backtesting

| Method | Timing | Description |
|--------|--------|-------------|
| `at_t_minus_3m` | Previous GDP publication day | ~0 months of target quarter monthly data |
| `at_t_minus_2m` | 1 month later | ~1 month of fast indicators |
| `at_t_minus_1m` | 2 months later | ~2 months of fast indicators |
| `at_t_minus_0` | Just before GDP publication | All monthly + quarterly data available |
| `from_live_data` | Auto-detected | Inspects actual data endpoints |

Publication lags are approximated per indicator:
- Employment, hours worked: ~1 month lag (fast)
- Retail, building approvals, goods trade: ~2 months lag (slower)

---

## Backtest Results (2022Q1-2025Q4, latest-revised data)

| Info Set | RMSE | MAE | Bias | Direction | 90% CI Coverage | Bridges |
|----------|------|-----|------|-----------|-----------------|---------|
| T-3m | 0.380% | 0.333% | +0.222% | 94% | 94% | 8.0 |
| T-2m | 0.326% | 0.254% | -0.120% | 88% | 94% | 8.0 |
| T-1m | 0.286% | 0.234% | -0.029% | 88% | 100% | 8.0 |
| T-0 | 0.268% | 0.227% | +0.086% | 94% | 81% | 16.0 |

Naive benchmark (trailing 4-quarter average): RMSE 0.285%.

**T-0 beats the naive benchmark (0.268% vs 0.285%).** The NAB business conditions survey bridge improves mid-cycle accuracy: T-2m RMSE dropped from 0.345% to 0.326% and T-1m from 0.303% to 0.286%. T-3m degraded (0.327% → 0.380%) due to SARIMA over-forecasting with minimal data, but this horizon has limited practical value.

### Caveats

- **Pseudo real-time**: Uses latest-revised data, not vintage data as published at the time. Slightly flatters the model.
- **Post-COVID only**: 16 quarters is a short sample. COVID era excluded due to subsidised metrics distorting economic relationships.
- **No monthly CPI**: The ABS monthly CPI indicator (640106) has <2 years of history and does not aggregate cleanly to the quarterly CPI. Excluded.

---

## Key Design Decisions

- **Unfloored MFP in production bridge**: Negative MFP is informative for nowcasting actual GDP, unlike potential output estimation where it represents underutilisation.
- **Labour productivity adjustment**: HMA(13) trend from wage data added to labour bridge equations to correct for productivity drag. Eliminated the +0.3pp positive bias present without it.
- **Independent production bridge inputs**: Uses capital stock and labour force from Modellers Database (1364.0) and MFP from wage data, not the same employment/hours/approvals used by monthly bridges. Earlier version double-counted these inputs.
- **Monthly CPI bridge instead of deflation**: Retail turnover and company profits are nominal (current prices) while GDP is real (chain volume). Rather than deflating these indicators, a monthly CPI bridge is included as a separate bridge equation. Backtesting showed the CPI bridge outperforms deflation: the OLS coefficients on the CPI bridge capture price acceleration as a coincident demand signal (positive net coefficient = rising inflation → stronger demand → higher GDP), which is more informative than the rigid mechanical deflation. With the CPI bridge, T-2m RMSE improved from 0.378% to 0.345% and T-0 from 0.288% to 0.278%. The monthly CPI index is spliced from the discontinued Monthly CPI Indicator (6484.0, Sep 2017 onwards) and the current 6401.0 table 640106 (Apr 2024 onwards), with quarterly Appendix 1a interpolated to monthly for pre-2017 bridge estimation history. SARIMA completion uses only the genuine monthly observations.
- **Business indicators included despite late publication**: 5676.0 arrives ~3 days before GDP, but still adds genuine information at T-0.
- **BoP goods & services balance (5302.0) — excluded**: A quarterly bridge using the change in the SA goods & services balance (table 530204) was tested but excluded after leave-one-out analysis showed it degraded T-0 RMSE from 0.269% to 0.275%. Despite covering services trade (unlike the monthly 5368.0 goods balance), the BoP change series is noisy, arrives only ~1 day before GDP, and adds little over the monthly goods balance bridge which has longer estimation history and SARIMA-refined estimates by T-0. The data loader (`src/data/balance_of_payments.py`) is retained for other uses.
- **Construction work done (8755.0)**: Published ~3 weeks before GDP. Total sectors, all construction types, chain volume measures (SA). Direct read on the investment component of GDP. Complements the monthly building approvals bridge which is a leading indicator of construction activity, not a measure of work actually done.
- **Private capex (5625.0)**: Published ~4 weeks before GDP. Total private capital expenditure, chain volume measures (SA), including education and health. Captures business investment spending directly.
- **Inventories (5676.0)**: Published ~3 days before GDP alongside profits and sales. Inventory swings are a volatile GDP component that is otherwise uncaptured by the monthly bridges. Uses log-differenced chain volume measures.
- **Government GFCE (spliced 5206.0 + GFS)**: Government consumption is ~25% of GDP but was previously uncaptured by any bridge. Uses 5206.0 national accounts GFCE growth for the long bridge estimation history (back to 1959), spliced with the GFS early release for the target quarter. GFS publishes ~2 weeks before GDP; the two sources produce identical growth rates on revised data so no scaling is needed. The GFS workbook is a data cube (not standard ABS time series) and is parsed directly via openpyxl.
- **NAB business conditions survey (RBA H3)**: The first soft data indicator in the bridge set. Published monthly by NAB (~2 weeks after reference month), republished in RBA Statistical Table H3 (series GICNBC). Used as a quarterly mean level (not growth — it's already a deviation from long-run average). The survey adds most value at mid-cycle horizons: T-2m RMSE improved from 0.345% to 0.326%, T-1m from 0.303% to 0.286%. At T-0 the effect is marginal (hard data dominates). Westpac-MI consumer sentiment was also tested but excluded — it added nothing beyond NAB conditions and has shorter history (2010 vs 1997). The key value of survey data is at turning points: hard activity indicators (employment, retail) lagged the 2023 per-capita recession by 2-3 quarters, while business conditions captured the slowdown earlier.
- **No monetary policy variables**: Cash rate, yield curve, credit aggregates, and financial conditions indices were considered but excluded. At the nowcast horizon (0–3 months), monetary policy transmission is already embedded in the real activity indicators (retail trade, building approvals, labour data) by the time they are published. This is consistent with standard practice — bridge equation nowcasts at central banks (RBA, BoE) and in the literature rely on hard activity and survey data, not policy rates. MP variables have predictive power at the 1–8 quarter forecasting horizon but add little once contemporaneous real data is available.

---

## Running

```bash
# Live nowcast
./run-gdp-nowcast.sh
uv run python -m src.models.gdp_nowcast.model

# Backtest (default: 2022Q1 to latest)
uv run python -m src.models.gdp_nowcast.backtest

# Backtest with custom range
uv run python -m src.models.gdp_nowcast.backtest --start 2022Q1 --end 2025Q4
```

### Output

```
charts/GDP-Nowcast/               # Live nowcast charts
  gdp-growth-nowcast-qq-*.png     # Q/Q fan chart
  gdp-growth-nowcast-tty-*.png    # TTY fan chart
  bridge-contributions-*.png      # Horizontal bar chart

charts/GDP-Nowcast-Backtest/      # Backtest evaluation charts
  actual-vs-nowcast-*.png         # Actual vs nowcast time series
  rmse-by-information-set-*.png   # RMSE comparison bar chart
  nowcast-errors-*.png            # Error time series by info set
  nowcast-accuracy-evolution-*.png # Rolling MAE over time

model_outputs/gdp_nowcast/        # Saved results
  backtest_results.parquet        # Full backtest results table
  backtest_summary.csv            # Summary statistics
  backtest_summary.txt            # Human-readable summary
```
