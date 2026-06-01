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
gdp_nowcast_bridge/
├── __init__.py
├── model.py              # Core nowcast model
├── backtest.py           # Backtesting framework
└── MODEL_NOTES.md        # This file
```

### Key Functions

```python
# Live nowcast (auto-detects target quarter, generates charts)
from src.models.gdp_nowcast_bridge.model import run_nowcast
result = run_nowcast()

# Programmatic nowcast (explicit target and availability)
from src.models.gdp_nowcast_bridge.model import nowcast, DataAvailability
result = nowcast(
    target_quarter=pd.Period("2025Q4", "Q-DEC"),
    availability=DataAvailability.at_t_minus_1m(pd.Period("2025Q4", "Q-DEC")),
    quiet=True,
)

# Backtesting
from src.models.gdp_nowcast_bridge.backtest import run_backtest, BacktestConfig
bt = run_backtest(BacktestConfig(start="2022Q1"))
```

---

## Bridge Groups

### Monthly Bridges (SARIMA-completed)

These arrive ~2 months before GDP publication. Missing months within the target quarter are forecast using the best-fitting SARIMA model from a small candidate set: (1,1,0), (1,1,1), (2,1,0), (0,1,1), with and without seasonal(12) component. Selected by AIC.

| Bridge | Indicator | Source | Aggregation |
|--------|-----------|--------|-------------|
| Consumption | Retail turnover, **CPI-deflated** (real) | 5682.0 (Monthly Household Spending) | Quarterly sum |
| Investment | Total dwelling approvals (count) | 8731.0 (Building Approvals) | Quarterly sum |
| Labour: hours | Monthly hours worked | 6202.0 (Labour Force), table 6202019 | Quarterly sum |
| Labour: employment | Employed total persons | 6202.0 (Labour Force), table 6202001 | Quarterly mean |
| Trade | Balance on goods, **CPI-deflated** (real) | 5368.0 (International Trade in Goods) | Quarterly sum |
| Prices: monthly CPI | CPI All Groups SA index (spliced) | 6484.0 + 6401.0 table 640106 | Quarterly mean |
| Survey: NAB conditions | NAB business conditions index (SA) | RBA Table H3 (GICNBC) | Quarterly mean |

The two labour bridges include an **HMA(13) labour productivity trend** as an additional regressor. This is derived from wage data (Δhcoe - Δulc) and corrects for periods where employment grows faster than output (negative productivity growth). Without this adjustment, the labour bridges systematically overpredict GDP during productivity downturns.

The **NAB business conditions bridge** uses the index level (deviation from long-run average, in percentage points), not growth — analogous to the goods trade balance bridge. This is the only soft data (survey) indicator in the bridge set. It publishes faster than most hard indicators (~2 weeks lag) and helps at mid-cycle horizons where hard data is still arriving.

**Westpac-MI consumer sentiment was tested and rejected.** Adding it as a second survey bridge (level, quarterly mean aggregation) left T-0 RMSE essentially unchanged and slightly reduced correlation with actual GDP growth. Households' reported sentiment has historically tracked their income and interest-rate expectations more than real activity, and within the bridge combination the inverse-MSE weighting already discounts noisy indicators — so there was no accuracy gain to justify keeping it. (The DFM, by contrast, *does* benefit from the sentiment series — its factor structure extracts the shared variance with NAB and discards the rest.)

The **monthly CPI bridge** uses a spliced index combining the discontinued Monthly CPI Indicator (6484.0, Sep 2017 – Sep 2025) with the current monthly CPI from 6401.0 table 640106 (Apr 2024 onwards). For SARIMA completion, only the genuine monthly observations are used (not interpolated quarterly history), ensuring the SARIMA model learns real monthly dynamics. For bridge equation estimation, the full spliced series (including quarterly-interpolated pre-2017 history) is used to maximise the training sample.

### Quarterly Bridges (available or excluded)

These are published shortly before GDP. The bridge is only active if the indicator is published for the target quarter.

| Bridge | Indicator | Source | Lead over GDP |
|--------|-----------|--------|---------------|
| Prices: CPI | CPI trimmed mean (quarterly) | 6401.0, Appendix 1a | ~4-5 weeks |
| Prices: WPI | WPI growth (log diff) | 6345.0 | ~2-3 weeks |
| Consumption: Household spending | Total household spending growth (CVM) | 5682.0 table 5682015 | ~5 weeks |
| Investment: construction | Total construction work done growth (CVM) | 8755.0 | ~2-3 weeks |
| Investment: private capex | Total private capex growth (CVM) | 5625.0 | ~1-2 weeks |
| Government: GFCE | Government consumption growth (spliced 5206.0 + GFS) | 5206.0 + GFS | ~1-2 weeks |
| Business: profits | Gross operating profits growth, **CPI-deflated** (real) | 5676.0 | ~1-2 days |
| Business: sales | Total sales (summed across industries, CVM) | 5676.0 | ~1-2 days |
| Business: inventories | Inventories growth (CVM) | 5676.0 | ~1-2 days |

### Deflation of nominal indicators

Three indicators in the panel are published in nominal (current-price) terms while everything else is a volume measure, a physical quantity (counts of dwellings, hours, employed persons), or a survey index. To keep the panel consistently real and remove a nominal-bias channel during inflationary periods, those three indicators are deflated before they enter their bridges:

| Indicator | Deflator | Where |
|-----------|----------|-------|
| Retail turnover (monthly, 5682.0) | Monthly trimmed mean index (spliced) | `get_retail_turnover_real_monthly()` |
| Balance on goods (monthly, 5368.0) | Monthly trimmed mean index (spliced) | `get_goods_balance_real_monthly()` |
| Gross operating profits (quarterly, 5676.0) | Quarterly-mean monthly trimmed mean | `get_company_profits_real_qrtly()` |

Why this matters: the bridge regressions learn a coefficient that maps indicator growth to real GDP growth. Fitted on nominal indicators, that coefficient absorbs whatever average inflation passthrough prevailed in the training sample. During periods of unusually high inflation (e.g. 2022-23), the nominal print mechanically translates to a hot real GDP nowcast — over-cooking it. Deflating upstream removes this bias and lets the bridge coefficient measure the volume relationship directly.

**Deflator choice — trimmed mean over headline CPI.** Trimmed mean (CPI underlying) is the deflator rather than All Groups CPI. Both exist at monthly frequency (spliced from 6401.0 Appendix 1a quarterly + 640106 monthly from Apr 2024), and both produce conceptually-real series. Trimmed mean is preferred because it strips volatile items (energy, food, government rebates) that would otherwise propagate as deflator noise into the deflated indicator. Without that smoothing, the inverse-MSE combination logic downweights the deflated bridges precisely because they have become noisier — defeating the point of the fix.

A national-accounts deflator (GDP deflator or HFCE deflator) would be conceptually purer but is unusable: it is locked behind the same publication barrier as the GDP target variable itself. Monthly trimmed mean is the best available high-frequency price index.

The balance-on-goods deflation is approximate — conceptually you would deflate exports and imports separately by their own price indices, but those are quarterly-only. Trimmed mean strips underlying inflation while leaving terms-of-trade signal intact, which is the available compromise.

### Consumption: Household spending CVM (5682.0)

A second consumption bridge uses the **quarterly Chain Volume Measures** Total Household Spending series from 5682.0 table 5682015. This series is conceptually close to the 5206.0 Household Final Consumption Expenditure (HFCE) component of GDP, but published with the third monthly Household Spending Indicator release of each quarter — roughly five weeks after the reference quarter ends, ahead of the GDP publication.

History begins 2014Q3 (~46 growth obs as of 2026), which is enough for a standard bridge regression. ABS ships the quarterly CVM table only in the quarter-end-month snapshot, so the loader (`get_household_spending_cvm_qrtly(history=...)`) **requires** that month: the request is grounded to the nowcast target quarter (e.g. 2026Q1 → `mar-2026`), derived from the GDP data — never inferred from today's date, which could point at an unpublished future vintage.

This bridge complements the monthly retail bridge:
- **Retail** captures fast-moving signals from the first two months of the quarter via SARIMA completion (~2 months lead, but a narrower consumption basket).
- **Household spending CVM** is broader (full HFCE-style basket) and already real, but only arrives once the third monthly 5682.0 print lands.

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

### Building the nowcast regressor row (and why the zero-fill is safe)

`_bridge_nowcast` assembles the regressor row for the target quarter, then aligns
it to the fitted coefficient index, filling any **missing column** with `0.0`:

```python
X_new = pd.DataFrame([row])
for col in bridge.coefficients.index:
    if col not in X_new.columns:
        X_new[col] = 0.0
X_new = X_new[bridge.coefficients.index]
```

At first glance this looks dangerous: if the contemporaneous indicator value were
missing, the column would be created as `0.0`, and the bridge would "nowcast" off a
zero for its own driver — a spurious strong-negative signal — while still carrying a
full inverse-MSE weight. **That path is unreachable by construction.** The zero-fill
only ever supplies *structural* terms (`const`, and the `_L1` lag when the target
quarter is the first observation), never the contemporaneous indicator.

The guarantee is that **every caller inserts the target-quarter indicator value into
the index before calling `_bridge_nowcast`**:

| Caller | Guard |
|--------|-------|
| Monthly bridges (`_build_monthly_bridges`) | `quarterly_full.loc[target_quarter] = q_value` set immediately before the call; SARIMA-completion failure `continue`s and skips the bridge entirely |
| Production bridge (`_build_production_bridge`) | `indicators_ext.loc[target_quarter] = pg_nowcast` set immediately before the call |
| Quarterly bridges (`_build_quarterly_bridges`) | Called only inside `if available and target_quarter in series.index`; otherwise `available = False` and the bridge is excluded from combination |

So `target_quarter in indicators_nowcast.index` always holds when the contemporaneous
value is read, and the indicator column is always populated. The `available` flag and
the `_combine_bridges` filter (`available and np.isfinite(mse_oos) and mse_oos > 0`)
together ensure that a bridge lacking target-quarter data is *excluded*, not zero-filled.

**Reviewer note on NaN:** the upstream guards (SARIMA-failure skip; `pd.notna` in the
availability detection) mean a NaN indicator does not normally survive to the
combination on the standard path (target = next unpublished quarter). But an available
bridge *can* still produce a NaN point estimate if a regressor is undefined at the
target quarter — concretely, the labour bridges' HMA labour-productivity trend has no
value when nowcasting a quarter beyond its support (e.g. forcing a target two quarters
past the last published GDP). Such a bridge has a finite MSE and `available=True`, so it
would otherwise be combined with full weight and poison the result (NaN). `_combine_bridges`
therefore also requires `np.isfinite(nowcast_qoq)`: bridges with a non-finite nowcast are
excluded (and logged), so the combination degrades gracefully to the remaining bridges.

---

## Combination

Bridge nowcasts are combined using inverse out-of-sample MSE weights:

```
w_i = (1/MSE_i) / Σ(1/MSE_j)
```

MSE is computed from expanding-window one-step-ahead forecasts (minimum training window: max(20, N/3) observations).

Only bridges with available data and finite MSE are included. This means the nowcast automatically improves as more indicators are published.

### Why inverse-MSE weighting is robust

A useful property of the bridge approach (compared with joint covariance models like the BVAR or the DFM) is that **adding noisy indicators rarely hurts performance**. Each bridge is fit independently and weighted by its own out-of-sample MSE: a noisy indicator just gets a low weight and contributes little to the combined nowcast. The good bridges dominate by construction.

Joint-covariance approaches behave differently. In a BVAR conditional update, every variable contributes to the residual covariance matrix Σ, which is then inverted to compute the conditional mean of GDP given the other indicators. A single noisy indicator can pollute the entire update through spurious cross-correlations — its noise gets amplified through Σ⁻¹ and corrupts the conditioning even on the good variables. Empirically, the BVAR comparator in this project tops out around 10 well-chosen variables and *gets worse* when government finance or BoP trade are added to the panel. The bridge model, by contrast, comfortably incorporates 16+ indicators (8 monthly + 8 quarterly + production bridge) because each one is judged on its own out-of-sample track record.

The general principle: **bridge equations degrade gracefully**, joint-covariance models do not. If you're not sure whether an indicator helps, the bridge framework lets you add it cheaply — the worst case is that it earns a low MSE weight and gets ignored. In a BVAR or DFM, the same indicator can quietly degrade the whole model.

---

## Uncertainty

Bootstrap prediction intervals via a **row-wise (block) bootstrap** over bridge residuals:

1. Align active bridges' residuals on the common window where every bridge has a value
2. Each of the 1000 draws picks one quarter `t*` and sums every active bridge's residual **at that same quarter**, weighted: `Σ_i w_i · e_i(t*)`
3. Add that combined residual draw to the combined nowcast point estimate
4. Report 70% (15th-85th percentile) and 90% (5th-95th percentile) intervals

### Why row-wise (preserving cross-bridge correlation)

The combined error is `e_c = Σ_i w_i e_i`, a weighted sum over bridges that all predict
the **same** GDP quarter, so the residuals `e_i` are positively correlated. An earlier
version pooled all weighted residuals and drew each summand *independently*; that
reproduces only the independent-sum variance `Σ_i w_i² Var(e_i)` and drops the positive
covariance `2 Σ_{i<j} w_i w_j Cov(e_i, e_j)`, so the bands came out too tight. Drawing a
whole quarter's residuals together keeps `Cov(e_i, e_j)` intact and widens the intervals
to a properly-calibrated level.

If a short-history bridge collapses the common window below `MIN_BOOTSTRAP_ROWS` (20
quarters), the code falls back to the old independent pooled resample and logs a warning —
in that case the bands again understate uncertainty.

**Known residual gap (not yet addressed):** the bootstrap uses *in-sample* OLS residuals
(`fit.resid`), which are smaller than genuine out-of-sample errors, and it ignores
parameter-estimation uncertainty. Both push intervals narrower. Capturing them would need
the per-quarter out-of-sample errors retained (only the scalar `mse_oos` is kept today).

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

## Empirical Performance

For current empirical performance, run:
```bash
uv run python -m src.models.gdp_nowcast_bridge.backtest
```

The backtest reports RMSE, MAE, bias, direction accuracy, and 90% CI coverage at four information sets (T-3m, T-2m, T-1m, T-0).

### Qualitative findings

- **The bridge model is the best-performing of the three on RMSE** at T-0 — it beats the trailing-4-quarter naive benchmark, and beats both the DFM and the BVAR.
- **Accuracy improves monotonically as the publication cycle progresses** (T-3m → T-2m → T-1m → T-0), which is the expected pattern when more information is genuinely available.
- **The model only beats the naive benchmark once enough in-quarter hard data has landed — at T-0, with a near-tie at T-1m.** At T-3m and T-2m its RMSE is actually *worse* than just carrying the trailing-4-quarter average, because there is too little hard data in the target quarter for the bridges to add signal over persistence. **Treat T-3m/T-2m as indicative, not forecasts**; the nowcast earns its keep from ~1 month out. (The asymmetric early-horizon bias and the >naive RMSE are two faces of the same thing — extra variance, not extra signal.)
- **NAB business conditions adds mid-cycle value**, lifting accuracy at T-2m and T-1m. At T-0 the effect is marginal because hard data dominates.
- **The bridge framework degrades gracefully** when noisy indicators are added — inverse-MSE weighting automatically discounts them. This is in contrast to joint-covariance models (BVAR, DFM) where a single noisy indicator can corrupt the whole posterior.
- **Bootstrap CI coverage is now close to nominal** after the row-wise bootstrap fix (see *Uncertainty*): 90% CI coverage runs ~94–100% across horizons and 70% CI coverage ~56–94% (lightest at T-3m). Earlier, the independent pooled resample dropped the cross-bridge residual correlation and the bands under-covered. Residual narrowing remains from using in-sample residuals and ignoring parameter uncertainty (noted under *Uncertainty*); coverage rates are coarse on the 16-quarter backtest (~6pts per quarter).
- **Correlation with actual GDP is the lowest of the three models** despite the lowest RMSE. The bridge tends toward the historical mean more than the DFM or BVAR — its small RMSE is partly an artefact of GDP growth being moderately variable around a stable mean, not necessarily of accurately tracking turning points.

### Caveats

- **Pseudo real-time**: Uses latest-revised data, not vintage data as published at the time. Slightly flatters the model.
- **Post-COVID only**: The backtest sample is short — too short to make statistically meaningful claims about model differences. COVID era excluded due to subsidised metrics distorting economic relationships.
- **No monthly CPI**: The ABS monthly CPI indicator (640106) has <2 years of history and does not aggregate cleanly to the quarterly CPI. Excluded.

---

## Key Design Decisions

- **Unfloored MFP in production bridge**: Negative MFP is informative for nowcasting actual GDP, unlike potential output estimation where it represents underutilisation.
- **Labour productivity adjustment**: HMA(13) trend from wage data added to labour bridge equations to correct for productivity drag. Eliminates the persistent positive bias that the labour bridges otherwise carry during productivity downturns.
- **Independent production bridge inputs**: Uses capital stock and labour force from Modellers Database (1364.0) and MFP from wage data, not the same employment/hours/approvals used by monthly bridges. Earlier version double-counted these inputs.
- **Monthly CPI bridge instead of deflation**: Retail turnover and company profits are nominal (current prices) while GDP is real (chain volume). Rather than deflating these indicators, a monthly CPI bridge is included as a separate bridge equation. Backtesting showed the CPI bridge outperforms deflation: the OLS coefficients on the CPI bridge capture price acceleration as a coincident demand signal (positive net coefficient = rising inflation → stronger demand → higher GDP), which is more informative than the rigid mechanical deflation. The monthly CPI index is spliced from the discontinued Monthly CPI Indicator (6484.0, Sep 2017 onwards) and the current 6401.0 table 640106 (Apr 2024 onwards), with quarterly Appendix 1a interpolated to monthly for pre-2017 bridge estimation history. SARIMA completion uses only the genuine monthly observations.
- **Business indicators included despite late publication**: 5676.0 arrives 1-2 days before GDP, but still adds genuine information at T-0.
- **BoP goods & services balance (5302.0) — excluded**: A quarterly bridge using the change in the SA goods & services balance (table 530204) was tested but excluded after leave-one-out analysis showed it degraded T-0 RMSE. Despite covering services trade (unlike the monthly 5368.0 goods balance), the BoP change series is noisy, arrives only ~1 day before GDP, and adds little over the monthly goods balance bridge which has longer estimation history and SARIMA-refined estimates by T-0. The data loader (`src/data/balance_of_payments.py`) is retained for other uses.
- **BoP services-only balance (5302.0) — also excluded**: Re-tested 2026-05-28 using the SA *services-only* change (table 530204, did `"Services ;"`), on the theory that the aggregate G+S series was diluted by the goods component already covered by the monthly bridge. The services-only bridge **still degraded** the combination — T-0 RMSE 0.268% → 0.286% and T-0 correlation +0.474 → +0.436 over the 2022Q1–2025Q4 backtest window. The inverse-MSE weighting absorbed the noisy bridge despite assigning it a small weight, because the residual covariance with other quarterly bridges pulled the combined estimate sideways. Same conclusion as the aggregate test: BoP-basis quarterly trade data is too noisy at T-0 to help the bridge model, and the ~1-day lead leaves no early-horizon role either. The `get_bop_services_balance_qrtly` / `get_bop_services_change_qrtly` helpers in `src/data/balance_of_payments.py` are retained for reference.
- **Construction work done (8755.0)**: Published ~3 weeks before GDP. Total sectors, all construction types, chain volume measures (SA). Direct read on the investment component of GDP. Complements the monthly building approvals bridge which is a leading indicator of construction activity, not a measure of work actually done.
- **Private capex (5625.0)**: Published ~4 weeks before GDP. Total private capital expenditure, chain volume measures (SA), including education and health. Captures business investment spending directly.
- **Inventories (5676.0)**: Published 1-2 days before GDP alongside profits and sales. Inventory swings are a volatile GDP component that is otherwise uncaptured by the monthly bridges. Uses log-differenced chain volume measures.
- **Government GFCE (spliced 5206.0 + GFS)**: Government consumption is ~25% of GDP but was previously uncaptured by any bridge. Uses 5206.0 national accounts GFCE growth for the long bridge estimation history (back to 1959), spliced with the GFS early release for the target quarter. GFS publishes 1 day before GDP; the two sources produce identical growth rates on revised data so no scaling is needed. The GFS workbook is a data cube (not standard ABS time series) and is parsed directly via openpyxl.
- **NAB business conditions survey (RBA H3)**: The first soft data indicator in the bridge set. Published monthly by NAB (~2 weeks after reference month), republished in RBA Statistical Table H3 (series GICNBC). Used as a quarterly mean level (not growth — it's already a deviation from long-run average). The survey adds most value at mid-cycle horizons (T-2m and T-1m); at T-0 the effect is marginal because hard data dominates. Westpac-MI consumer sentiment was also tested but excluded — it added nothing beyond NAB conditions and has shorter history (2010 vs 1997). The key value of survey data is at turning points: hard activity indicators (employment, retail) lagged the 2023 per-capita recession by 2–3 quarters, while business conditions captured the slowdown earlier.
- **No monetary policy variables**: Cash rate, yield curve, credit aggregates, and financial conditions indices were considered but excluded. At the nowcast horizon (0–3 months), monetary policy transmission is already embedded in the real activity indicators (retail trade, building approvals, labour data) by the time they are published. This is consistent with standard practice — bridge equation nowcasts at central banks (RBA, BoE) and in the literature rely on hard activity and survey data, not policy rates. MP variables have predictive power at the 1–8 quarter forecasting horizon but add little once contemporaneous real data is available.

---

## Capex-Imports Hotness Diagnostic

After each nowcast, the print summary appends a shared diagnostic from `src/models/common/nowcast_diagnostics.py:capex_imports_hotness()` that compares the latest equipment capex print against the goods imports that should offset it under the GDP identity (I↑ ⇒ M↑).

**How it works:**
- Equipment capex QoQ change (5625.0 CVM SA, all industries) and goods imports QoQ change (5368.0 SA, quarterly sum of monthly) are each expressed as a percentage of contemporaneous GDP.
- Each is compared against its mean (and σ) from 1997Q4 onward — the BVAR's estimation frame, chosen for a stable, identity-relevant baseline.
- Hotness = (capex deviation from mean) − (imports deviation from mean).

**Interpretation:**
- Positive hotness (> +0.10pp): capex is unusually high relative to its history, *and more so than imports are unusually high*. The bridges treat capex as a positive GDP signal but inverse-MSE combination doesn't enforce the matching M offset, so the headline nowcast may be over-stating GDP growth by roughly this amount.
- Negative hotness (< −0.10pp): imports are surging by more than capex — the nowcast may be under-stating GDP growth.
- |hotness| ≤ 0.10pp: negligible, flagged as such.

The diagnostic was added specifically to flag the AI / data-centre buildout starting in mid-2025, which lifts the I-component sharply while the corresponding goods/services imports do not flow through the bridges symmetrically. The diagnostic is purely post-hoc — it does not change the bridge estimates or the combined nowcast — so it is a transparency tool for the user, not a model correction.

## Running

```bash
# Live nowcast
./run-gdp-nowcast-bridge.sh
uv run python -m src.models.gdp_nowcast_bridge.model

# Backtest (default: 2022Q1 to latest)
uv run python -m src.models.gdp_nowcast_bridge.backtest

# Backtest with custom range
uv run python -m src.models.gdp_nowcast_bridge.backtest --start 2022Q1 --end 2025Q4
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
