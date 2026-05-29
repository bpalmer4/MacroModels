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
| Goods trade balance, **trimmed-mean-deflated** | 5368.0 | 1990Q1 |
| NAB business conditions | RBA H3 (GICNBC) | 1997Q2 |
| CPI trimmed mean | 6401.0 | 1990Q1 |
| WPI growth | 6345.0 | 1997Q4 |
| Construction work done growth | 8755.0 | 1990Q1 |
| Private capex growth | 5625.0 | 1990Q1 |

**Goods balance deflation**: this is the only nominal indicator in the BVAR panel, so it's divided by the spliced monthly trimmed mean index (and quarterly-summed) before entering the VAR. Every other indicator is already real / a quantity. Keeps the panel consistently real and avoids feeding nominal-inflation drift into the joint covariance matrix.

The binding constraint is **WPI growth (1997Q4)**, giving ~112 quarterly observations of complete data after dropna.

**Labour series aggregation note**: `employment_growth` and `hours_growth` are sourced from the *monthly* Labour Force Survey (cat 6202.0) and aggregated to quarterly — employment by quarterly mean (it's a stock), hours by quarterly sum (it's a flow). Earlier versions of this model sourced these from cat 1364.0.15.003 (Modellers' Database, employment) and cat 5206.0 table 5206001 (National Accounts hours-worked index). Those quarterly publications release alongside GDP itself, making them useless for forward nowcasting — by the time they arrive, the GDP they were meant to predict is already out. The LFS-aggregated versions release ~3 weeks after quarter-end, so all three months of a quarter are typically in hand 5–6 weeks before the GDP print, providing a real conditioning lead. The tradeoff is slightly higher Q/Q variance (the NA hours index is reconciled within the National Accounts framework; LFS hours is the direct survey estimate).

### Indicators tested but excluded

- **Retail growth (5682.0)**: short history (from 2012Q4) shrinks the training sample materially
- **Business profits growth (5676.0)**: short history (from 2001Q2) shrinks the training sample
- **Household spending CVM (5682.0 table 5682015)**: short history (from 2014Q3, ~46 growth obs) would force the panel start from 1997Q4 to 2014Q3, hurting *every* coefficient in the VAR not just consumption. The Bridge and DFM both use this series — they handle short-history series gracefully via per-bridge estimation and ragged-edge EM respectively, but the BVAR cannot
- **Government consumption growth (5206.0 + GFS spliced)**: hurt RMSE and correlation despite long history. The bridge model uses this series usefully because each bridge is fit independently; the BVAR's joint conditional update treats it as noise that pollutes Σ_oo.
- **BoP goods+services balance change**: hurt correlation more sharply than gov consumption — same failure mode through the joint covariance structure.
- **BoP services-only balance change**: re-tested 2026-05-28 as the 11th panel variable (SA "Services ;" change from 5302.0 table 530204), on the theory that stripping out the goods component would isolate a cleaner signal. **Still degraded** the model — T-0 RMSE 0.625% → 0.640%, T-0 correlation +0.680 → +0.580, and nowcast variance climbed (NCstd 0.722% → 0.737%). Same Σ_oo pollution as the aggregate case: even a small noisy variable participates in every conditional update via the inverse covariance matrix.
- **Westpac-MI consumer sentiment (quarterly mean)**: raised nowcast volatility without improving correlation. The DFM absorbs the same series usefully because its factor structure *extracts* the shared variance with NAB and discards the idiosyncratic noise.
- **Productivity-adjusted labour input**: tried in DFM, didn't help there, not retried here

### 5-variable panel tested and rejected

A compact 5-variable panel was tested (gdp_growth, hours_growth, construction_growth, cpi_trimmed_mean, nab_conditions) on the theory that fewer collinear variables would dampen the over-volatility. **It was strictly worse** at every λ tested.

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

Each new release tightens the conditional forecast. The point estimate typically doesn't move dramatically as the conditioning set fills in (since most indicators correlate strongly with GDP and pull in similar directions); the main practical benefit is **CI tightening** — confidence in the nowcast grows substantially as more indicators arrive. Going from 1/9 to 6/9 indicators typically halves the 90% CI width.

The BVAR is therefore most useful in the mid-cycle window (weeks 4–9) when you have meaningful information about the current quarter but GDP hasn't published yet. Right after a GDP release (weeks 1–2) the model has nothing to work with and falls back to hindcasting the last published quarter.

---

## Empirical Performance

For current empirical performance, run:
```bash
uv run python -m src.models.gdp_nowcast_bvar.backtest
```

The backtest reports RMSE, MAE, bias, direction accuracy, correlation with actual GDP, NCstd (nowcast standard deviation vs actual GDP std), 90% CI coverage, per-year bias, and a sweep over `lambda_tight`.

### Qualitative findings

- **The BVAR is structurally over-volatile** — its nowcast standard deviation runs roughly 2× actual GDP standard deviation. This is a property of the conditional VAR formulation (see derivation below), not a hyperparameter or variable-selection issue. It can be reduced but not eliminated.
- **Among the three nowcast models, the BVAR sits in the middle on correlation** with actual GDP — better than bridge, worse than DFM. It overshoots magnitudes, so its headline RMSE is the worst of the three despite reasonable directional accuracy.
- **Bias is concentrated in 2022** (post-COVID productivity shock). Mirrors what the DFM showed. As the test sample rolls forward, the headline numbers improve naturally — by 2025 the model is essentially unbiased.
- **No `lambda_tight` setting brings RMSE or NCstd into a competitive range.** The hyperparameter sweep documents that the over-volatility is structurally bounded — tightening the prior actually *increases* volatility (see below).
- **The 90% CIs are reasonably calibrated** despite the headline volatility — coverage tracks close to nominal.

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

- **`lambda_tight` = 0.2**: Default chosen to maximise correlation with actual GDP. A looser prior (~0.5) gives marginally lower RMSE but at meaningfully reduced correlation. Since the model is interesting mainly for its tracking ability rather than its point accuracy, the higher-correlation default is preferred.

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
| Volatility (NCstd vs actual) | Under | Roughly matched | Over (~2×) |
| Headline RMSE rank | Best | Middle | Worst |
| Correlation rank | Worst | Best | Middle |
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

## Capex-Imports Hotness Diagnostic

After each nowcast, the print summary appends a shared diagnostic from `src/models/common/nowcast_diagnostics.py:capex_imports_hotness()` that compares the latest equipment capex print against the goods imports that should offset it under the GDP identity (I↑ ⇒ M↑).

**How it works:**
- Equipment capex QoQ change (5625.0 CVM SA, all industries) and goods imports QoQ change (5368.0 SA, quarterly sum of monthly) are each expressed as a percentage of contemporaneous GDP.
- Each is compared against its mean (and σ) from 1997Q4 onward — the BVAR's own estimation frame, so the historical baseline is exactly the sample the BVAR coefficients are fit on.
- Hotness = (capex deviation from mean) − (imports deviation from mean).

**Interpretation:**
- Positive hotness (> +0.10pp): capex is unusually high relative to its history, *and more so than imports are unusually high*. The BVAR absorbs the I/M relationship via the joint covariance matrix Σ, but the conditional update at T-0 uses fixed coefficients estimated over the full sample — a fresh regime change (e.g. AI capex surge) is treated as a draw from the historical distribution, not a structural shift. In the meantime, the headline nowcast may be over-stating GDP growth by roughly this amount.
- Negative hotness (< −0.10pp): imports surging by more than capex — possibly an under-stated nowcast.
- |hotness| ≤ 0.10pp: negligible, flagged as such.

The diagnostic was added specifically to flag the AI / data-centre buildout starting in mid-2025, which lifts the I-component sharply while the corresponding goods/services imports do not yet show up symmetrically in the BVAR's conditioning information. The diagnostic is purely post-hoc — it does not change the BVAR estimation or the nowcast — so it is a transparency tool for the user, not a model correction.

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
