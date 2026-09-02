# Potential Output — inflation-defined output gap

A Bayesian unobserved-components model (PyMC + NumPyro NUTS) estimating Australian potential output and the output gap. The gap is **defined** by inflation's deviation from the RBA's target, potential is a random walk, and GDP is fitted around the two with a residual.

There is no Phillips curve, no IS curve, no policy rule and no AR(2) cycle.

```
y*_t      = y*_{t-1} + g_{t-1} + e_y      potential: random walk, drifting
g_t       = g_{t-1} + e_g                 drift: a driftless Gaussian random walk
gap_t     = c · (pi_t - 2.5)              gap: defined by inflation
log_gdp_t = y*_t + gap_t + e_c            output: fitted, residual e_c
```

Three estimated quantities: `c`, `sigma_e`, and the initial level of the drift. Two imposed: `sigma_ystar` and `sigma_g`. One asserted: the 2.5% anchor.

**Independent by construction.** Imports nothing from `nairu`, `rstar_hlw`, `expectations` or `models/common`. Its only dependency outside the package is `src/data`.

---

## The question

Can potential output be identified from inflation alone?

Not via a Phillips curve. The reason is Australian institutional history: the RBA has been targeting inflation since 1993, and a central bank that succeeds in stabilising inflation destroys the very covariance a Phillips curve needs. McLeay & Tenreyro (2019), *Optimal Inflation and the Identification of the Phillips Curve*, is the formal statement. A slope estimated on a target-era sample is biased toward zero by construction.

What survives policy is the **definition**: potential output is the level of output consistent with inflation at target. That is a statement about levels, not a slope to be estimated, and it is what this model operationalises.

---

## Data

| Series | Source | Role |
|--------|--------|------|
| log GDP x 100 | `gdp.get_log_gdp()` | ABS 5206.0 chain volume, SA |
| Annual trimmed mean inflation, % | `inflation.get_trimmed_mean_annual()` | ABS 6401.0 |

Sample **1993Q1 to 2026Q2, 134 quarters**. Two series. Nothing else.

The four-quarter inflation rate is used rather than the quarterly one. In this specification inflation is not a regressor whose residual has to be well behaved, so the overlapping-observations problem does not arise, and "at target" is an annual concept.

---

## Why the gap is defined rather than estimated

`c` converts percentage points of inflation into per cent of output. Inflation supplies the sign, the timing, and the relative magnitude of the gap: 6% inflation is further from target than 3%. What it cannot supply is the conversion factor, because nothing in a price index is denominated in units of GDP.

`c` is estimated here, but it is **not** a Phillips slope and is not obtained by regressing inflation on the gap. It is identified by the third equation: given the trend and the inflation-defined gap, `c` is whatever best reconciles the two with observed GDP.

### The third equation is what makes this a model

An earlier version of this specification set `y* = log_gdp - c·d` outright, with no residual. That version was a definition, not a model:

- GDP was reproduced exactly at every draw, so there was no residual, no fit, and nothing the model could get wrong.
- Every movement in output that inflation did not account for was **forced into potential**. `sigma_ystar` had to rise to 0.965 to accommodate it.
- Potential consequently tracked GDP straight down through the 2020 collapse and back up: the model concluded productive capacity fell 7% in a quarter and recovered within a year.
- Potential growth swung from **−6.0% to +9.6%** across 2020-21.

Adding `e_c` fixes all of it. Trend and gap no longer have to exhaust output, so what they miss is measured rather than absorbed, and `sigma_e` becomes the diagnostic the exercise needs.

---

## The imposed variances

| | value | what it controls |
|---|---|---|
| `sigma_ystar` | 0.078 (`ratio_ystar` 0.13 × `sigma_c` 0.60) | how far potential's *level* wanders off its trend |
| `sigma_g` | 0.015 (`ratio_g` 0.025 × `sigma_c` 0.60) | how fast potential's *growth rate* may bend |

`ratio_g = 0.025` is the HP(1600) analogue: HP corresponds to a trend/cycle innovation sd ratio of 1/40 on the second difference of the trend, and `g` is that second-difference channel.

**`sigma_ystar` and `sigma_e` cannot both be free.** A quarterly wiggle in GDP can be a shift in potential or a residual, and the likelihood cannot fully apportion between them. That is the Stock-Watson pile-up problem in its proper form. `sigma_ystar` is pinned, which is what "potential is smooth" means operationally, and `sigma_e` is estimated so that it can report what is left over. `ModelConfig.free_sigma_ystar` exists to invert that choice; see the iteration log for what happens.

`sigma_g` cannot be freed alongside `sigma_ystar` either, for the same reason: level and drift variances are the other pile-up pair.

---

## Results (2026Q2 vintage)

Converged: all `r_hat` = 1.00, `ess_bulk` 9,574 to 13,945.

| Parameter | mean | 94% HDI |
|---|---|---|
| `c` | **0.468** | [0.264, 0.690] |
| `sigma_e` | 0.981 | [0.856, 1.100] |
| `initial_trend_growth` | 0.991 | [0.895, 1.085] |

| Headline, 2026Q2 | median | 90% band |
|---|---|---|
| Potential growth (year-ended) | **2.14** | [1.77, 2.51] |
| Output gap | **+0.51** | [0.31, 0.72] |

`c` = 0.468 means one percentage point of excess inflation implies about half a per cent of output gap. Its 90% interval is clear of zero.

### Potential growth

| | year-ended % |
|---|---|
| 1997Q4 | 4.10 |
| 2005Q4 | 3.13 |
| 2012Q4 | 2.79 |
| 2019Q4 | 1.88 |
| 2026Q2 | **2.14** |

A decline of roughly two percentage points since the late 1990s, with a dip around 2020 and a partial recovery.

### Output gap

| | % of potential |
|---|---|
| 1997Q4 | −0.28 |
| 2008Q3 | +1.07 |
| 2016Q2 | −0.42 |
| 2020Q2 | −0.56 |
| 2023Q1 | +1.87 |
| 2026Q2 | **+0.51** |

The lockdown quarter reads −0.56 because inflation barely moved. The seven-point collapse in output sits in `e_c`, the residual, which is the right place for it: unexplained, rather than booked as a fall in productive capacity.

### How much of the cycle inflation explains

| | |
|---|---|
| sd of the gap | 0.47 |
| sd of GDP less potential | 1.05 |
| **variance share** | **19.7%** |

**This is the answer to the question the model was built to ask.** The inflation-defined gap accounts for about a fifth of Australian output's deviation from trend. The other four fifths is residual. That is now measured rather than assumed, and measuring it is what the third equation bought.

---

## Limitations

1. **`c` is attenuated and cannot be otherwise.** Conventional Phillips slope estimates of 0.3 to 0.4 imply `c` near 2.5 to 3. This model returns 0.468. The gap between the two is the McLeay-Tenreyro attenuation, and no amount of respecification recovers it from a target-era sample. The implied gap should therefore be read as a **lower bound** on the true one.
2. **Four fifths of the cycle is unexplained.** `sigma_e` = 0.98 against a gap sd of 0.47. The model does not claim to measure the Australian business cycle; it claims to measure the part of it that inflation identifies.
3. **Everything is conditional on the anchor.** 2.5% from 1993. Genuine re-anchoring in either direction would be booked as a gap. This was the single most fragile assumption in the earlier specifications and it has not been removed, only made more visible.
4. **The imposed variances still do most of the smoothing.** `sigma_ystar` = 0.078 is not estimated, and it is what makes potential a smooth line rather than something that follows output.
5. **No lag between the gap and inflation.** The relationship is contemporaneous. Attempts to estimate a lag profile did not converge; see the iteration log.
6. **The two diagnostic modules have not been exercised on this specification.** `realtime.py` raises for any spec other than `core` and has not been rewired. `sigma_sweep.py` should work: its grids already cover `ratio_ystar`, `ratio_g`, `sigma_c` and `anchor`, which are exactly this spec's imposed settings, and a spec-name branch that would have sent it down the labour path has been fixed. It has not been run against this spec, so treat that as untested rather than working.

---

## Other specifications in the package

`config.SPECS` still carries three earlier specifications. They are kept for reference, they are not the model, and their documented numbers predate the corrections in the iteration log.

- **`core`** — an anchored Phillips curve on GDP and inflation, with an AR(2) cycle. Run it with `--pi-basis quarterly`; see iteration log item 2 for why.
- **`labour`** — potential decomposed into trend hours and trend productivity, using ABS 6202.0 hours, population and participation. Five series, six equations. Its trend hours path reproduces an HP(1600) trend of hours at a correlation of 0.9972, which is why it was set aside.
- **`target`** — a sign-only restriction: the gap must share the sign of the inflation deviation, with no magnitude claim. Superseded by the `inflation` spec, which uses the magnitude as well.

---

## File structure

```
potential_uc/
├── config.py            ModelConfig — sample, anchor, imposed variances, spec
├── base.py              SamplerConfig, set_model_coefficients (self-contained)
├── observations.py      GDP + inflation -> aligned numpy arrays
├── estimate.py          build_model / sample / save
├── results.py           PotentialResults + load_results
├── analyse.py           diagnostics + charts
├── run.py               CLI entry point
├── compare.py           overlay saved runs
├── realtime.py          pseudo-real-time revisions (raises for non-core specs)
├── sigma_sweep.py       sweep imposed settings (untested on this spec)
├── MODEL_NOTES.md       this file
└── equations/
    ├── scale.py               sigma_c + the fixed-ratio trend sigmas
    ├── potential.py           g and y* states
    ├── inflation_gap.py       gap = c·(pi - anchor); GDP fitted with residual
    ├── trend_growth.py        g state alone (unused by the live spec)
    ├── target_consistency.py  sign-only restriction (`target` spec)
    ├── output.py              AR(2) cycle (`core`, `labour`)
    ├── phillips.py            anchored Phillips curve (`core`, `labour`)
    ├── trend_hours.py         pr*, hpp*, h* (`labour`)
    ├── trend_productivity.py  g_lp, lp* (`labour`)
    ├── hours.py               hours observation (`labour`)
    └── participation.py       participation observation (`labour`)
```

Charts are written to `charts/PotentialUC/`: `potential-growth`, `output-gap`, and `gdp-and-potential-output` and `actual-growth-versus-potential` on full and recent windows.

## Commands

```bash
./run-potential-uc.sh                  # the model above: estimate + chart
./run-potential-uc.sh --analyse-only   # recharts from the saved trace
./run-potential-uc.sh --anchor 2.25    # move the anchor
./run-potential-uc.sh --ratio-ystar 0.25   # looser potential
```

---

## Iteration log

The package began as a conventional multivariate UC model and was progressively stripped. Each step below removed something that turned out to be doing the work that was supposed to be done by inflation.

**1. Labour decomposition, then set aside.** Potential as trend hours times trend productivity, with observed population and a participation block. Diagnosed and fixed a 7.99 log-point identity residual (the omitted `−log(1−u)` unemployment margin), and removed jitter in trend hours growth by Henderson-smoothing log population with an ARIMA-extended tail. Set aside because trend hours reproduced an HP(1600) trend of hours at 0.9972: an elaborate apparatus returning what a filter gives for free.

**2. The Phillips curve was over-weighted.** The `core` spec's left-hand side was four-quarter trimmed mean inflation observed quarterly, with iid errors. Consecutive observations share three CPI quarters, so the error is MA(3) by construction and the likelihood counted roughly four times as many independent observations as exist. Correcting it (`pi_basis="quarterly"`) moved the 2026Q2 output gap from **1.21 to 0.21**, dropped `beta` from 0.379 to 0.262, and dropped the implied Phillips R² from 31% to 12%.

**3. Which revealed that the corrected model was a filter.** With inflation correctly weighted, the `core` gap correlated **0.966** with an HP(1600) cycle. The thing that made the model more than a filter was the thing that had been mis-specified.

**4. An import-price supply control changes nothing.** `gamma` = 0.030 [0.005, 0.058], statistically real. Demeaned import price growth averaged +5.56 over 2022Q1-2023Q4, so it explains 0.17pp of an overshoot exceeding a percentage point. The 2026Q2 gap was 0.21 with or without it. Default off.

**5. `ratio_ystar = 0` changes nothing either.** HP(1600) is the integrated random walk with no level innovation, so `ratio_ystar > 0` is an addition to the HP analogy. Removing it moves trend growth by 0.001pp.

**6. Pseudo-real-time revisions are large.** Over 19 vintages from 2008Q4, mean absolute revision to the output gap is 1.62, which is **1.5 times the standard deviation of the gap itself** — the Orphanides & van Norden (2002) result reproduced. The mechanism is the fixed anchor: real-time estimates drifted to **−5.45%** in 2019Q4, against a full-sample value near zero, because inflation undershot 2.5 for five years and the model had no other way to express it.

**7. Correlations between persistent series were being over-read.** Raw `corr(HP gap_t, deviation_{t+k})` appeared to peak at +0.28 at one quarter and invert to −0.29 by eight, and that pattern was used to argue about transmission lags. Prewhitened with an AR(4) fitted to the deviation, **no lag from 0 to 8 clears two standard errors** (largest +0.149 against 2se = 0.175). The pattern was the two series' own persistence. The argument built on it was withdrawn. Prewhitening is conservative for a low-frequency relationship, so this is not proof of absence; the honest reading is that the sample cannot settle it.

**8. A sign-only restriction (`target` spec).** Gap constrained to share the sign of the inflation deviation via a split normal, no magnitude claim. An earlier version discarded the sign and used only `|deviation|`, which could say "the gap is zero" loudly or quietly but never "the gap is positive", and duly reproduced an HP cycle at 0.991. With the sign restored and a tight wrong-side tolerance, the model moved off the filter (corr with HP falling to 0.60), but the answer then depended heavily on that tolerance.

**9. Estimating the inflation lag does not work.** Free Dirichlet weights over lags 0 to 4 gave `r_hat` = 1.53, with chains splitting between a contemporaneous mode and a three-to-four quarter mode. A two-parameter beta (MIDAS) lag shape gave `r_hat` = 1.54. The data do not identify the lag, and the answer depends on it: contemporaneous gave potential growth 1.64 and a gap of +2.22; the estimated-lag version gave 2.14 and +1.39.

**10. Defining potential outright, with no residual.** `y* = log_gdp − c·d`. `c` = 0.378, and it was shown to equal, to three decimals, the OLS slope of `(ΔGDP − g)` on `Δ(deviation)`, a regression whose correlation is 0.132. The specification asserted innovations of N(0, 0.078) while producing innovations with sd **0.917**, a posterior predictive check failing by a factor of twelve. Freeing `sigma_ystar` gave 0.965 and widened `c` from ±0.02 to ±0.50, confirming the diagnosis but leaving potential still tracking GDP.

**11. Fitting GDP with a residual: the current model.** Adding `log_gdp = y* + gap + e_c` restored potential to a proper state, returned `sigma_ystar` to its imposed 0.078, and produced `c` = 0.468 [0.26, 0.69] with `sigma_e` = 0.981. Potential growth is a plausible 2.14% with no pandemic collapse, and the variance share gives the amplitude answer directly.

**Next, in priority order:**
1. **Rewire `realtime.py` to the `inflation` spec, then run it.** Endpoint revision is the test that discriminates, and this specification has never been through it. Item 6 of the log is what it did to the earlier one.
2. **Weaken the fixed anchor.** Item 6 identifies it as the binding defect, and this specification inherits it unchanged.
3. **Delete the dead specifications** once nothing further is wanted from them, along with the equation modules only they use.
4. **Import `c` from outside the sample** if a defensible external estimate can be found. Item 1 under Limitations says the internal estimate is a lower bound; the cross-sectional route (state unemployment against capital-city CPIs, where the cash rate is common) is the standard way to get one.
