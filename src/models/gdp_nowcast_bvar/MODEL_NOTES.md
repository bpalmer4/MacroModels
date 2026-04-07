# GDP Nowcast BVAR Model

T-0 only nowcast using a Bayesian VAR with Minnesota prior, conditioned on the contemporaneous values of the other quarterly indicators in the panel. Built as a third comparator alongside the bridge equations and Dynamic Factor Model.

> **Status**: Interesting comparator but **not recommended for operational point forecasts**. The conditional VAR forecast formulation is structurally over-volatile (nowcast std ~2× actual GDP std), so headline RMSE is high. The model still has merits — decent correlation with actual GDP, honest uncertainty intervals, simple closed-form maths — but the bridge and DFM are the workhorses for production use.

## Summary

| Component | Method | Key Feature |
|-----------|--------|-------------|
| Estimation | Closed-form Minnesota-prior posterior | Equation-by-equation, no MCMC |
| Prior | Minnesota: random walk + lag decay + cross-variable shrinkage | Standard textbook formulation |
| Implementation | Direct numpy linear algebra (~150 lines) | No external BVAR library required |
| Information sets | Continuous — uses whichever indicators are published | Conditional update on the available subset; tightens as more data lands |
| Forecast type | Gaussian conditional on other contemporaneous variables | Uses residual covariance to update GDP given indicator surprises |
| Uncertainty | Closed-form conditional variance | 70% / 90% Gaussian intervals |
| Sample start | 1997Q4 (binding constraint: NAB conditions, WPI) | ~112 observations |
| Variables | 10 quarterly indicators including GDP | One per economic block |

---

## Architecture

### File Structure

```
gdp_nowcast_bvar/
├── __init__.py
├── __main__.py            # python -m entry point
├── model.py               # BVAR class + Minnesota prior + nowcast wrapper
├── backtest.py            # T-0 only backtest framework
└── MODEL_NOTES.md         # This file
```

### Key Functions

```python
# Live nowcast (auto-detects target quarter, generates charts)
from src.models.gdp_nowcast_bvar.model import run_nowcast
result = run_nowcast()

# Programmatic nowcast (explicit target)
from src.models.gdp_nowcast_bvar.model import nowcast
result = nowcast(
    target_quarter=pd.Period("2025Q4", "Q-DEC"),
    n_lags=2,
    lambda_tight=0.2,
    quiet=True,
)

# Direct access to the BVAR fit
from src.models.gdp_nowcast_bvar.model import fit_bvar_minnesota, _load_panel
panel = _load_panel()
fit = fit_bvar_minnesota(panel.dropna(), n_lags=2, lambda_tight=0.2)
print(fit.coefficients.shape)  # (1 + n_vars*n_lags) x n_vars

# Backtest
from src.models.gdp_nowcast_bvar.backtest import run_backtest, BacktestConfig
bt = run_backtest(BacktestConfig(start="2022Q1"))
```

---

## The Minnesota-Prior BVAR

### Theoretical background

A Vector Autoregression of order p:
```
y_t = c + A_1 y_{t-1} + A_2 y_{t-2} + ... + A_p y_{t-p} + ε_t,    ε_t ~ N(0, Σ)
```

Standard OLS estimation requires N²p coefficients per equation, where N is the number of variables. With N=10 and p=2, that's 200 parameters per equation × 10 equations = 2000 parameters. With ~112 observations, OLS overfits catastrophically.

The **Minnesota prior** (Litterman 1986) shrinks coefficients toward a sensible default:
1. **Each variable follows AR(1)**: own first-lag coefficient is shrunk toward 1, all other coefficients toward 0
2. **Higher lags shrink toward zero**: coefficient on lag k has variance (λ/k)² — quadratically tighter for distant lags
3. **Cross-variable coefficients shrink harder than own**: variance scaled by an additional factor λ_cross < 1
4. **Constant has diffuse prior**: not shrunk

Hyperparameters:
- `λ_tight` (default 0.2) — overall tightness; smaller = more shrinkage toward random walks
- `λ_cross` (default 0.5) — relative shrinkage of cross-variable vs own-variable coefficients
- `λ_decay` (default 1.0) — exponent on lag decay (1 = harmonic, 2 = quadratic)

### Closed-form posterior

The posterior is computed equation-by-equation since the prior is independent across equations. For equation i:

```
β_post,i = (X'X / σ_i² + Λ_i⁻¹) ⁻¹ (X'Y_i / σ_i² + Λ_i⁻¹ β_prior,i)
```

where Λ_i is the diagonal prior covariance matrix and σ_i² is the equation's residual variance (estimated from a univariate AR(1) on the data). Pure linear algebra; no MCMC required. Fits in milliseconds.

### Conditional forecast at T-0

The standard "VAR forecast" projects y_{t+1} from lagged values only. For nowcasting, we want to use the **contemporaneous** values of the other indicators. The Gaussian conditioning formula:

```
E[gdp_t | other_t = obs, lags] = ŷ_t[gdp] + Σ_yo · Σ_oo⁻¹ · (obs − ŷ_t[other])
Var[gdp_t | other_t = obs, lags] = Σ_yy − Σ_yo · Σ_oo⁻¹ · Σ_oy
```

where ŷ_t is the unconditional VAR forecast (from lags only) and Σ is the residual covariance matrix. The conditional mean adds an "indicator surprise" term to the unconditional forecast: if the other indicators came in higher than the VAR predicted, GDP probably did too.

This is equivalent to a Bayesian update of the GDP nowcast, treating the joint forecast as a multivariate Gaussian.

---

## Indicator Panel (10 variables)

| Variable | Source | Sample start |
|----------|--------|--------------|
| GDP growth (target) | 5206.0, CVM SA | 1959Q3 |
| Building approvals growth | 8731.0 | 1990Q1 |
| Employment growth | 6202.0 | 1990Q1 |
| Hours worked growth | 6202.0 table 6202019 | 1990Q1 |
| Goods trade balance | 5368.0 | 1990Q1 |
| NAB business conditions | RBA H3 (GICNBC) | 1997Q2 |
| CPI trimmed mean | 6401.0 | 1990Q1 |
| WPI growth | 6345.0 | 1997Q4 |
| Construction work done growth | 8755.0 | 1990Q1 |
| Private capex growth | 5625.0 | 1990Q1 |

The binding constraint is **WPI growth (1997Q4)**, giving ~112 quarterly observations of complete data after dropna.

### Indicators tested but excluded

- **Retail growth (5682.0)**: only from 2012Q4 — would shrink the sample to ~52 observations
- **Business profits growth (5676.0)**: only from 2001Q2 — would shrink to ~95 observations
- **Government consumption growth (5206.0 + GFS spliced)**: long history (1959Q4+) so no sample-length issue, but adding it to the 10-variable panel *reduced correlation from 0.594 to 0.569* and increased RMSE slightly (0.597 → 0.611). The bridge model uses this series usefully because each bridge is fit independently; the BVAR's joint conditional update apparently treats it as a noisy signal that pollutes Σ_oo.
- **BoP goods+services balance change**: long history available but adding it cut correlation more sharply (0.594 → 0.521). Same explanation — quarterly noisy series that doesn't fit well into the joint covariance structure.
- **Both gov consumption and BoP added together**: cumulative damage — RMSE 0.626%, correlation 0.493. Worse than either individually.
- **Productivity-adjusted labour input**: tried in DFM, didn't help there, not retried here

### 5-variable panel tested and rejected

A compact 5-variable panel was tested (gdp_growth, hours_growth, construction_growth, cpi_trimmed_mean, nab_conditions) on the theory that fewer collinear variables would dampen the over-volatility. **It was strictly worse**: every NCstd value across the λ sweep exceeded every NCstd in the 10-variable version.

**Why**: with many variables, contemporaneous indicator surprises *partially cancel* (one says "GDP up", another says "GDP down"), which dampens the conditional update. With fewer carefully-chosen variables, all indicators move in the same direction, so the conditional update is *more* confident, not less. Removing collinearity backfired for the conditional forecast formula.

---

## Forward Nowcast Through the Publication Cycle

In live mode, the BVAR refines its nowcast of the next quarter's GDP as each indicator publishes. The Australian publication order for quarter Q is roughly:

| Approx week after Q ends | Indicator | Cumulative count |
|--------------------------|-----------|------------------|
| Week 1-2 | NAB business conditions (M3) | 1/9 |
| Week 4 | Employment, hours worked (M3) | 3/9 |
| Week 4-5 | Building approvals (M3) | 4/9 |
| Week 5 | Goods trade balance (M3) | 5/9 |
| Week 7 | WPI growth (Q) | 6/9 |
| Week 8-9 | Construction work done (Q) | 7/9 |
| Week 9 | Private capex (Q) | 8/9 |
| Week 10 | CPI trimmed mean (Q) | 9/9 |
| Week 10 | **GDP for Q published by ABS** | (everything in) |

Each new release tightens the conditional forecast. As an illustration, simulated 2026Q1 nowcasts at different stages of the cycle:

| Stage | Indicators in | Nowcast Q/Q | 90% CI width |
|-------|--------------|-------------|--------------|
| Just NAB published | 1/9 | +0.58% | 2.73pp |
| + Employment + Hours | 3/9 | +0.62% | 1.91pp |
| + Approvals + Goods + WPI | 6/9 | +0.67% | 1.89pp |
| All indicators | 9/9 | (full conditional) | (tightest) |

The CI tightening is the main practical benefit — the point estimate doesn't necessarily move much, but the confidence in it grows substantially. The BVAR is therefore most useful in the mid-cycle window (weeks 4-9) when you have meaningful information about the current quarter but GDP hasn't published yet. Right after a GDP release (weeks 1-2) the model has nothing to work with and falls back to hindcasting the last published quarter.

---

## Backtest Results (2022Q1–2025Q4, latest-revised data)

| Info Set | RMSE | MAE | Bias | Direction | Corr | NCstd | 90% CI |
|----------|------|-----|------|-----------|------|-------|--------|
| T-0 | 0.597% | 0.458% | +0.281% | 88% | +0.594 | 0.664% | 88% |

Naive benchmark RMSE: 0.285%. Actual GDP std: 0.289%.

### Comparison with Bridge and DFM

| Metric | Bridge | DFM | **BVAR** |
|--------|--------|-----|----------|
| RMSE | 0.268% | 0.478% | 0.597% |
| MAE | 0.227% | 0.391% | 0.458% |
| Bias | +0.085% | +0.365% | +0.281% |
| Direction | 94% | 94% | 88% |
| **Correlation** | +0.474 | **+0.635** | +0.594 |
| **NCstd** (vs actual 0.289) | 0.198 | 0.411 | **0.664** |
| 90% CI coverage | 81% | 94% | 88% |

The BVAR sits between bridge and DFM on correlation, has small-to-moderate bias, well-calibrated CIs — but is **structurally over-volatile** (NCstd ~2× actual). This explains its high RMSE: the model is making bold predictions that get the direction roughly right but overshoot the magnitude.

### Per-year bias decay

Like the DFM, the BVAR's bias is **dominated by 2022** (the post-COVID productivity shock period):

| Year | BVAR Mean Error |
|------|-----------------|
| 2022 | **+0.78pp** |
| 2023 | +0.26pp |
| 2024 | +0.16pp |
| 2025 | **−0.07pp** |

Three quarters in 2022 (Q1: +1.33, Q2: +1.25, Q3: +0.77) account for ~75% of the entire bias. By 2025 the model is essentially unbiased. As the test sample rolls forward, the headline numbers will improve naturally.

### Hyperparameter solution space

Sweep over `lambda_tight` (overall prior tightness) on the 10-variable panel:

```
  lambda     RMSE      MAE      Bias     Corr    NCstd
--------------------------------------------------------
    0.02   0.730%   0.546%   +0.168%   +0.496   0.833%   ← tightest, worst RMSE
    0.05   0.697%   0.519%   +0.195%   +0.520   0.795%
    0.10   0.639%   0.484%   +0.231%   +0.569   0.731%
    0.15   0.611%   0.464%   +0.259%   +0.592   0.693%
    0.20   0.597%   0.458%   +0.281%   +0.594   0.664%   ← current default, best correlation
    0.30   0.577%   0.443%   +0.301%   +0.576   0.617%
    0.50   0.564%   0.422%   +0.306%   +0.538   0.580%   ← best RMSE, best NCstd
    1.00   0.604%   0.435%   +0.310%   +0.492   0.615%
    2.00   0.661%   0.457%   +0.316%   +0.461   0.675%
```

**Key findings:**
- **No setting brings RMSE below 0.564%** — still 2× the bridge model
- **No setting brings NCstd below 0.580%** — still 2× actual GDP std
- The minimum NCstd value across the *entire* hyperparameter sweep is structurally bounded
- Tighter prior (smaller λ) actually *increases* volatility, contrary to intuition — see below

### Why the BVAR is structurally over-volatile

The conditional forecast formula:
```
cond_mean[GDP] = uncond[GDP] + Σ_yo · Σ_oo⁻¹ · (observed_others − uncond[others])
```

is essentially a regression of GDP on the contemporaneous "surprises" in the other 9 indicators, using the residual covariance matrix. In macro data, GDP has strong contemporaneous correlations with every activity indicator (employment, hours, construction, retail, etc.) — so this regression has high explanatory power and produces large updates.

The DFM avoids this because the indicators feed through a 2-factor bottleneck, which dampens overreaction. The bridge model avoids it because each bridge is fit independently, with no joint conditional update. The BVAR has no such damping mechanism — it uses the full residual covariance matrix to compute the conditional update, and that's too aggressive.

**The over-volatility is structural to the conditional VAR forecast formulation, not a hyperparameter or variable-selection problem.** It's the reason DFM-style factor models dominate the central-bank nowcasting literature.

---

## Key Design Decisions

- **Closed-form Minnesota posterior, no MCMC**: Equation-by-equation normal-normal conjugate posterior. Fits in milliseconds. Pure numpy. Easier to debug than PyMC, and exact rather than MCMC-approximate. ~150 lines of model code total.

- **Conditional rather than unconditional forecast**: Uses the Gaussian conditioning formula to incorporate contemporaneous indicator values. This is what makes it a "nowcast" rather than a one-step forecast. The downside is the over-volatility documented above.

- **10-variable panel, not 5**: The 5-variable version was strictly worse — fewer collinear variables removed the natural dampening from conflicting indicator surprises.

- **`lambda_tight` = 0.2**: Default chosen to maximise correlation with actual GDP. The minimum-RMSE setting is `lambda_tight` = 0.5, but the correlation is slightly lower there (0.538 vs 0.594). Since the model is interesting mainly for its tracking ability (high correlation), the higher-correlation default is preferred.

- **VAR(2)**: Two lags is the standard for quarterly macro VARs. Tested VAR(1) — slightly worse. Higher orders are not feasible with 112 observations and 10 variables.

- **No COVID dummy**: The DFM showed that masking COVID quarters destroys the correlation. Same logic applies here. Like the DFM, the BVAR's bias is concentrated in 2022 and is already fading.

- **No productivity adjustment**: The DFM showed that pre-adjusting labour indicators with the productivity trend reduces correlation more than it helps bias. Same logic applies here — the BVAR keeps raw labour variables.

- **T-0 only**: The model has no mechanism for handling earlier information sets. Unlike the DFM's Kalman filter or the bridge model's SARIMA completion, the BVAR's clean closed-form posterior assumes a complete panel. Building a mixed-frequency BVAR (Schorfheide-Song style) would require state-space estimation and lose the closed-form simplicity. Out of scope for this comparator.

- **Partial-data conditioning** (added after initial implementation): The conditional forecast uses *whichever subset* of indicators is observed for the target quarter. As each indicator publishes, the conditioning subset grows and the forecast tightens. The Σ_oo and Σ_yo blocks of the residual covariance matrix are partitioned to only the available indices each call. If zero indicators are available, the model falls back to the unconditional VAR forecast (lagged dynamics only). This means the BVAR is genuinely usable as a continuously-updating forward nowcast through the publication cycle, not just at the moment all indicators land.

- **Hindcast fallback for live mode**: When *no* indicators have been published yet for the next quarter (typical for the first ~1-2 weeks after a GDP release), the live `run_nowcast()` falls back to hindcasting the most recent published GDP quarter. This gives a sanity check showing what the model would have predicted for the last quarter given all the indicators. The auto-detect threshold is `min_indicators=1` — the model switches to forward-mode nowcasting as soon as any single indicator publishes for the next quarter.

- **Indicator availability reporting**: In live mode, the model prints which indicators are available and which are missing for the target quarter. In hindcast mode, it also prints why the next quarter cannot be nowcast yet (which indicators are still missing). This makes it easy to see at a glance how far through the publication cycle you are and which release is the next to update the nowcast.

---

## Differences from the Other Models

| Aspect | Bridge | DFM | BVAR |
|--------|--------|-----|------|
| Combination method | Inverse-MSE weighted bridges | Joint Kalman filter on factors | Joint Bayesian VAR conditional forecast |
| Ragged edge | SARIMA pre-completion | Native (Kalman filter) | None — T-0 only |
| Information sets | T-3m, T-2m, T-1m, T-0 | T-3m, T-2m, T-1m, T-0 | Continuous (any subset of indicators) |
| Number of indicators | 8 monthly + 8 quarterly + production = 17 | 6 monthly + 6 quarterly = 12 | 10 quarterly = 10 |
| Estimation | OLS per bridge | EM algorithm (state space) | Closed-form Minnesota posterior |
| Implementation effort | Moderate | High | Low |
| RMSE @ T-0 | 0.27% | 0.48% | 0.60% |
| Correlation @ T-0 | 0.47 | **0.64** | 0.59 |
| NCstd @ T-0 | 0.20 (under) | 0.41 | 0.66 (over) |
| Best for | **Operational forecasts** | Shape tracking + honest CIs | Conceptual comparator |

---

## Potential Improvements

These are noted but **not currently planned** because the BVAR isn't intended as a primary operational model:

1. **Σ shrinkage**: Add a Wishart prior on the residual covariance matrix to dampen the cross-equation correlations that drive over-volatility. Standard approach, ~30 lines.
2. **Marginal likelihood hyperparameter tuning**: Currently `lambda_tight` is fixed at 0.2. Could be tuned by maximising the marginal likelihood automatically each backtest quarter. Closed form for the dummy-observation prior; ~30 lines.
3. **Mixed-frequency BVAR (Schorfheide-Song)**: Would allow T-3m through T-1m information sets. Requires state-space form and Kalman filtering. Significant implementation effort.
4. **Sum-of-coefficients dummy**: Standard Sims-Zha extension that encourages cointegration / common trends. Useful for level data but less so for growth-rate data like ours.
5. **Optimal Σ structure for nowcasting**: Empirical exploration of which off-diagonal elements of Σ to shrink. Niche.

---

## Running

```bash
# Live nowcast — switches to forward-mode as soon as any indicator publishes
# for the next quarter; falls back to hindcasting the last published quarter
# if no indicators are available yet
./run-gdp-nowcast-bvar.sh
uv run python -m src.models.gdp_nowcast_bvar.model

# Backtest
uv run python -m src.models.gdp_nowcast_bvar.backtest

# Backtest with custom range
uv run python -m src.models.gdp_nowcast_bvar.backtest --start 2022Q1 --end 2025Q4
```

### Output

```
charts/GDP-Nowcast-BVAR/                 # Live nowcast charts
  gdp-growth-bvar-nowcast-qq-*.png       # Q/Q fan chart
  gdp-growth-bvar-nowcast-tty-*.png      # TTY fan chart
  gdp-growth-bvar-nowcast-*.png          # Combined growth chart

charts/GDP-Nowcast-BVAR-Backtest/        # Backtest evaluation charts
  actual-vs-bvar-nowcast-*.png           # Actual vs BVAR nowcast
  bvar-nowcast-errors-*.png              # Error time series

model_outputs/gdp_nowcast_bvar/          # Saved results
  backtest_results.parquet               # Full backtest results table
  backtest_summary.csv                   # Summary statistics
  backtest_summary.txt                   # Human-readable summary
```

---

## References

- **Litterman (1986)** "Forecasting with Bayesian Vector Autoregressions: Five Years of Experience". *Journal of Business & Economic Statistics*. The original Minnesota prior paper.
- **Karlsson (2013)** "Forecasting with Bayesian Vector Autoregression". *Handbook of Economic Forecasting Vol 2B*. The canonical handbook chapter; all the formulas in one place.
- **Bańbura, Giannone, Reichlin (2010)** "Large Bayesian Vector Auto Regressions". *Journal of Applied Econometrics*. Establishes that Minnesota-prior BVARs scale to 20+ variables.
- **Sims & Zha (1998)** "Bayesian Methods for Dynamic Multivariate Models". *International Economic Review*. The dummy-observation construction for Minnesota priors.
- **Schorfheide & Song (2015)** "Real-Time Forecasting with a Mixed-Frequency VAR". *Journal of Business & Economic Statistics*. The mixed-frequency extension we did NOT implement.
