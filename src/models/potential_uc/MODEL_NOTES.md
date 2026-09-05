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

A second live specification, `--spec production`, keeps the level and the gap and replaces the drift with a Cobb-Douglas production function, so potential growth comes from trend capital, hours and MFP. It agrees at 2.16 against 2.14 and gives trend productivity a credible interval. `inflation` remains the default and the two are deliberately kept apart, since the agreement is only informative while they are separate specifications. See "An alternative source for potential growth".

**Independent by construction.** Imports nothing from `nairu`, `rstar_hlw`, `expectations` or `models/common`. Its only dependency outside the package is `src/data`.

---

## What each output is for, and how far it can be pushed

Two outputs, two uses. The gap says which side of the ledger the economy is on. Trend growth says how limited the world it is operating in is. Both matter, and each is robust in the dimension its use needs while being weaker in the other. Read the rest of this file with that split in mind, because the two carry their uncertainty in quite different places.

**The gap: the statistical sign is solid, its economic interpretation is conditional, and the magnitude is weak.** `gap = c·(pi − 2.5)`, so its sign is the sign of the inflation deviation, which is observed data and carries no estimation uncertainty at all. The only thing standing between the data and the ledger reading is whether `c` is positive, and item 17 puts that at **97.7%** with the residual allowed to be serially correlated, the sign of `c` free to be negative, and nothing excluded from the sample. So "above or below potential" does not depend on the projection.

The sign has one weakness, and it is at turning points. The gap is dated by the inflation deviation, but policy acts with lags and expectations feed inflation, so the pressure and the price move apart in time. Item 12 puts a number on it: under a four-quarter lead the gap crosses zero about two quarters earlier in 2021. So the ledger reading is solid in the middle of an episode and uncertain within a quarter or two of a turn, which is when it is most likely to be consulted. Limitation 3 sets out why the sample cannot resolve the timing.

And it has one substantive exception. The sign is the sign of an inflation deviation, so it reads the ledger correctly only when that deviation is demand-driven. A supply shock that lifts inflation above target while the economy sits at or below capacity produces a positive reading from a negative gap. Trimmed mean does not protect against this, because trimming removes outliers rather than broad correlated shocks; Limitation 6 sets it out and item 18 records why non-tradables inflation cannot be substituted to fix it. So: robust to the statistical objections, not robust to stagflation.

The magnitude does, entirely, and it is the weaker half. `c` = 0.468 [0.28, 0.66], its identification is concentrated in two inflation episodes, and it is sensitive to four pandemic quarters: **0.47 on the continuous sample against 0.27 excluding them** (item 16). That pair is more informative than the posterior interval, because the interval is uncertainty conditional on the specification while the pandemic experiment is uncertainty about the specification. Quote the sign and the direction of travel with confidence; treat a gap of +0.51 against +0.40 as a distinction the model cannot make.

**In short, ranked by how much weight each will bear:**

| | |
|---|---|
| trend growth, about 2.1% | strong: 2.12 to 2.19 across every perturbation tried, and the RBA independently at ~2.0 |
| orientation of the gap | strong statistically: P(`c` > 0) = 97.7% on the hardest specification |
| economic reading of that sign | conditional: holds when the inflation deviation is demand-driven, not under stagflation |
| scale of the gap | weak: `c` wide, and 0.47 against 0.27 on the pandemic quarters |
| timing near turning points | weak: annual inflation is a distributed object, and the sample cannot resolve the lag |
| the *width* of the trend growth band | prior-sensitive, even though the central estimate is not (Limitation 2) |

**Trend growth: the level and the decline are solid, the precision is not.** Item 14 sweeps the smoothing prior over a sixteen-fold range and latest trend growth spans 0.19pp while the decline since the late 1990s runs 1.70 to 1.95. So "about 2.1%, down roughly two points" is in the data. But the 90% band goes from [1.93, 2.37] at the tightest setting to [0.45, 3.53] at the loosest, so the published interval is as narrow as `sigma_g` is tight. Quote the number and the decline; do not lean on the band.

Neither output depends much on the other. `c` moves the *level* of potential, by 0.27% of GDP in standard deviation, but the inflation deviation is mean-reverting, so it contributes almost nothing to the growth rate. Potential growth comes in at 2.12 to 2.19 across every specification tested in items 14 to 17, including ones built to break the identification.

---

## What this model does

Two ideas, and that is the whole model.

**A prior: potential output is a slow-moving trend.** Actual output jumps about from quarter to quarter for all sorts of reasons. The economy's capacity to produce does not. So we assume potential moves slowly relative to GDP.

That gives the trend its shape but not its position. A slow-moving line can be drawn high or low through the same data, and nothing about being slow-moving says which is right.

**A definition: potential output is the level of output consistent with inflation at target.** That is what puts the line in the right place. When inflation runs above 2.5% the economy is running beyond its capacity; when it runs below, it has room to spare.

Between 2015 and 2019 inflation sat under target for five years. This model responds by placing potential output about 0.3% of GDP higher than a filter of the same data puts it, so less of that stretch reads as an economy running beyond its capacity. It does not go so far as to call the whole five years a shortfall, and it should not: an economy that grows slowly for five years is partly an economy whose speed limit has fallen. The definition tilts the trend, it does not override it.

The composition chart shows how much of the distance between GDP and potential inflation accounts for, and how much it does not. Mostly it does not, and that is the honest answer rather than a failure: inflation tells us where potential sits, not what output does from one quarter to the next.

### The advantage over a filter is positioning, not shape

The gap here is not centred on the middle of the data. It is centred on the inflation target, and the target is a fulcrum the data are free to sit off. Mean deviation from the 2.5 anchor, with the implied mean gap at the posterior `c` of 0.468:

| period | mean (pi − 2.5) | implied mean gap |
|---|---|---|
| 1993-1999 | −0.27 | −0.13 |
| 2000-2007 | +0.32 | +0.15 |
| 2008-2014 | +0.40 | +0.19 |
| 2015-2019 | −0.76 | −0.36 |
| 2020-2026 | +0.95 | +0.45 |
| **full sample** | **+0.17** | **+0.08** |

The full-sample +0.08 is the point. It is near zero as an **outcome**, not zero by construction. An HP cycle cannot produce anything else, over the whole sample or over any long stretch of it, because it is a residual from a trend fitted to pass through the middle of the data: a run of years below capacity has to be paid back by a run above. 2015-2019 is where the two methods part company, though by less than an earlier version of this section claimed; the next heading measures the difference on the trend itself rather than on the gap.

**What the definition does to potential, measured directly.** Everything above is about the gap, and the gap is `c·d` by construction, so showing that it tilts with inflation proves nothing. The question that matters is whether the definition moves `y*` itself. Set it beside an HP(1600) trend of the same GDP series:

| period | mean (pi − 2.5) | `y*` − HP trend | `x` = GDP − `y*` | HP cycle |
|---|---|---|---|---|
| 1993-1999 | −0.27 | +0.11 | −0.04 | +0.07 |
| 2000-2007 | +0.32 | −0.18 | +0.15 | −0.03 |
| 2008-2014 | +0.40 | −0.11 | +0.07 | −0.04 |
| 2015-2019 | −0.76 | **+0.28** | +0.12 | +0.40 |
| 2020-2026 | +0.95 | −0.41 | +0.11 | −0.30 |

Every period carries the right sign: potential is placed above the filter's trend where inflation ran below target, and below it where inflation ran above. Full sample, corr(`y*` − HP trend, `d`) = **−0.559** [−0.712, −0.316], P(negative) = 0.996, with the repositioning having a standard deviation of 0.27% of GDP. That is the definition doing its work, on the object it is supposed to move. Note also what happens near the target: the repositioning scales with the size of the miss, so as `d` approaches zero `y*` converges on the filter's trend. That is the intended behaviour rather than a weakness. At target, output is at potential by definition, and a smooth trend through GDP is as good a guess at the level as anything else.

**How much of `c` the filter already had.** The same two regressions decompose `c` exactly:

```
slope of HP(1600) cycle on d        +0.264      a filter that never sees inflation
slope of (y* - HP trend) on d       -0.203      the definition repositioning the trend
                                    -------
slope of x = GDP - y* on d          +0.468      which is c, by the model's own fit
```

A plain HP filter, given nothing but log GDP, already produces a cycle covarying with the inflation deviation at 0.264 (correlation +0.262). So **57% of `c` was available for free**, and the definition contributes the remaining 43% by moving the trend.

Read this as corroboration rather than as deflation, but mild corroboration: both decompositions see the same GDP, and output and inflation ought to covary somewhat at business cycle frequencies, so a sceptic can wave it away more easily than the episode result. HP knows nothing about inflation, yet its cycle leans the right way against the inflation deviation. The identifying idea is therefore not something imposed against the grain of the data: it is a signal already visible in a naive decomposition, which the definition extends by a factor of about 1.8. A restriction that amplifies an existing covariance is in much better shape than one that has to manufacture it. It is also the honest answer to "is this just a filter?": `y*` differs from an HP trend by only 0.27% of GDP in standard deviation, but that difference is systematically signed by inflation, which is not something a filter can produce. Iteration log item 3 asked the same question of the wrong object, comparing the gap with an HP cycle rather than the trend with an HP trend.

**The move is modest, and that is the right answer rather than a shortfall.** In 2015-2019 the definition lifts potential 0.28 above where the filter puts it, and does not lift it far enough to put output below potential: `x` stays at +0.12 against the filter's +0.40. An earlier version of this section claimed the model reads those five years as below capacity throughout while a filter reads them as above. That is not what happens. Both read output as slightly above potential, and what the definition changes is by how much. Nor should the model do more: five years of slow growth is partly five years of a lower speed limit, so splitting the period between a downgraded gap and a downgraded trend is the correct treatment, and booking all of it to the gap would not be.

The shrinkage in `c` works the same way and is not an understatement to be corrected. Sub-period mean deviations are a few tenths of a percentage point, `c` shrinks them further, and 2015-2019 comes out at −0.36 rather than the −1.9 you would get by inverting a Phillips slope. `c·d` is a conditional mean, and shrinking toward zero is the right response to a signal that explains a fifth of the variation. See "Where the identification comes from".

**The rest of the distance is in `e_c`, and it is persistent.** `e_c` averages +0.47 across 2015-2019 and +0.97 across 2018-2019 alone. The residual's lag-1 autocorrelation is 0.51, so a five-year one-sided run is what this residual does rather than a coincidence, and item 15 explains why persistence there is expected rather than troubling. The honest description of the period is three-way: some of it is gap, some is a slower speed limit, and some is a persistent non-inflationary slow patch that the model measures and declines to attribute.

One further qualification, which is not an objection to the claim.

**Iteration log item 6 is this mechanism at full stretch, not a failure of it.** The real-time estimate of −5.45% in 2019Q4 is five years of undershoot against a fixed anchor with no other channel to express it. That is why endpoint revision matters more here than it would for a filter: the positioning is what the model contributes, so the test is how much of it survives to the sample end, and it will show up as a level shift rather than as the tilt an HP trend gives.

---

## The motive

The traditional pieces of the New Keynesian synthesis are either analytically challenging or impossible to estimate on Australian data. The IS curve cannot be identified: the `rstar_hlw` work found the rate channel too weak to pin r\* independently, with each specification largely returning the structural assumption it imposed. The Phillips curve has been corrupted by the RBA acting, which is the whole point of an inflation target and is documented for this package in iteration log items 2 and 3. The policy rule fares no better: in the `dsge` family the Taylor block was only pinned down once informative priors were imposed on it.

So rather than estimate the system, think about the problem differently: **peel off individual elements of the synthesis and answer them with simpler and less compromised models.** This package is one such peeling. It takes potential output and the output gap, and answers them without an IS curve, without a Phillips curve and without a policy rule, using the one thing policy cannot corrupt because policy is what defines it.

That is also why the model looks thin. The pieces are absent by design, not missing.

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

| Parameter | mean | 90% |
|---|---|---|
| `c` | **0.468** | [0.280, 0.657] |
| `sigma_e` | 0.981 | [0.878, 1.094] |
| `initial_trend_growth` | 0.991 | [0.908, 1.074] |

Intervals throughout this file are 90% quantile bands, including this table, which previously quoted ArviZ's default 94% HDI. `analyse.py` still prints the 94% HDI in its own summary.

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

### External comparison

Everything else in this file is internal. This is not: four estimates of Australian potential growth, from four different information sets.

| source | potential growth, 2026Q2 | basis |
|---|---|---|
| **This model** | **2.14** [1.77, 2.51] | log GDP + trimmed mean inflation. No production function. |
| This repo's `cobb_douglas` | **1.86** | α = 0.30, HP(1600) trends, MFP trend −0.02% p.a., re-anchored at 1990Q1 / 2000Q1 / 2008Q1 / 2019Q4. No inflation. |
| RBA, Feb 2026 SMP | ~2.0 | "potential output expected to grow at an annual rate of around 2 per cent over most of the forecast period"; revised down from August 2025, on 0.7% labour productivity. |
| Treasury, Budget 2026-27 | 2.5 | medium-term projection assuming long-run productivity growth returns to 1.2%. |

**`cobb_douglas` is not the independent check it looks like, and an earlier version of this section said it was.** It claimed the two "share no equations and almost no data". They share GDP, and worse, the Cobb-Douglas potential growth path reduces to a filter of it. Its MFP term is the Solow residual `g_Y − α·g_K − (1−α)·g_L`, and HP is linear, so with a common smoothing parameter and a constant α the factor terms cancel exactly:

```
α·HP(g_K) + (1-α)·HP(g_L) + HP(g_Y - α·g_K - (1-α)·g_L)  =  HP(g_Y)
```

Verified numerically, to 0.000 at every quarter: capital, hours, MFP and α contribute nothing to that model's potential growth. On this vintage `HP(g_Y)` gives 1.81 at 2026Q2 and 1.95 at 2025Q2, against the 1.86 and 2.01 in the table, the small gap being sample alignment from separate `dropna()` calls. So the 2.14-against-1.86 comparison is a heavily smoothed trend of GDP set beside a lightly smoothed one, not two routes converging. See iteration log item 20.

**What does provide a cross-check is the `production` specification** in this package, which agrees at 2.16 against 2.14 by a route that does not collapse to a filter. It is a *semi*-validation and only of the growth path: the two share the data, the sample, the gap definition and the level equation, and differ only in where potential's growth comes from. See "An alternative source for potential growth" below. **The RBA's figure is the external one**, and it now carries more weight than the table implied, since it is the only number here from outside the repo. (The Cobb-Douglas figure at α = 0.36 is nearer 1.97; the 1.86 here is the repo's α = 0.30 default.)

The RBA figure agrees as well, and carries extra weight because it is an operational input to a quarterly forecast used to set the cash rate, so it faces a correction loop.

**The comparison is a growth comparison only, and deliberately so.** `cobb_douglas` puts the 2026Q2 output gap at −0.89% against this model's +0.51%, but those two numbers should not be set against each other. Re-anchoring never touches `g_potential`, which is HP-filtered capital and hours plus MFP trend over the whole sample; it resets `log_potential` to actual GDP at 1990Q1, 2000Q1, 2008Q1 and 2019Q4, forcing the gap to zero at each (`cobb_douglas/model.py:316`). So its 2026Q2 gap means "cumulated actual growth less trend growth since 2019Q4", and it rests on output having been exactly at potential in 2019Q4, which is an assumption rather than a finding. The Cobb-Douglas run offers no independent estimate of the level, so there is nothing here to adjudicate. That is its own stated position, not an inference from outside: its module docstring says "the output gap from this model is notional only - it is not disciplined by inflation dynamics".

The re-anchoring is necessary rather than careless: with fixed factor shares and a smooth input trend, cumulating from one base drifts when the underlying speed limit is moving, and the level smears within each block. The printed resets are that drift being discarded, at 0.89%, −0.78%, −0.24% and −0.24%, roughly 0.08pp a year. That is precisely the problem the target fulcrum solves here, and it is why the growth paths can be compared while the levels cannot. Consistent with this, the Cobb-Douglas run reports its own gap correlating **0.160** with the inflation deviation and flags that it "may not capture demand pressure well".

**Treasury is the outlier, and the difference is one assumption.** 2.5 sits at the very top of this model's 90% band. The gap against the RBA is almost entirely labour productivity, 1.2% assumed against 0.7%, and Treasury's is a projection assumption rather than an estimate of current potential: it has been revised down from the 30-year average of about 1.5% to the 20-year average of 1.2%, and the horizon for returning to it pushed out from roughly two years to five. MFP has been flat to negative since the GFC, so 1.2% labour productivity requires an MFP recovery not seen in fifteen years.

Cobb-Douglas figures are from `./run-cd.sh -v` on the 2026Q2 vintage, so they are reproducible here rather than quoted. Other sources: [RBA Statement on Monetary Policy, February 2026](https://www.rba.gov.au/publications/smp/2026/feb/outlook.html); [Budget Paper No. 1, Statement 2, 2026-27](https://budget.gov.au/content/bp1/download/bp1_bs-2.pdf); [Treasury's medium-term economic projection methodology](https://treasury.gov.au/publication/treasurys-medium-term-economic-projection-methodology). The productivity argument against Treasury's 1.2% is set out in [Why 2.5% potential growth is wrong](https://markthegraph.blogspot.com/2026/01/why-25-potential-growth-is-wrong.html) (January 2026).

### How much of the cycle inflation explains, and the implied Phillips slope

`c` runs from inflation to output. A Phillips slope κ runs the other way. Two projections on the same data, so they are not reciprocals; they are related through the shared correlation by

```
c · κ = corr²
```

Computed draw by draw from the saved trace, with `x = log_gdp − y*` per draw and `d = pi − 2.5` (8,000 draws):

| | mean | 90% |
|---|---|---|
| posterior `c` | 0.4681 | [0.280, 0.657] |
| OLS slope of `x` on `d` | 0.4676 | [0.345, 0.590] |
| **implied κ** | **0.4010** | **[0.316, 0.467]** |
| corr² | 0.1906 | [0.109, 0.273] |
| `c_ols · κ` | 0.1906 | [0.109, 0.273] |

Four readings.

**corr² = 0.19 is the answer to the question the model was built to ask.** The inflation-defined gap accounts for about a fifth of Australian output's deviation from trend; the other four fifths is residual. That is now measured rather than assumed, and measuring it is what the third equation bought. `analyse.py` prints the same quantity each run as a variance share of the median paths (0.197 there against 0.191 here, the difference being medians against draw-wise).

**`c` is the OLS projection.** 0.4681 against 0.4676, identical to three decimals. The state-space estimation adds uncertainty to `c`, since `y*` is uncertain, but does not move it. Iteration log item 10 found the same equivalence in the earlier specification; it carries over.

**The identity holds exactly**, which is the check that the two slopes are the same covariance seen from two ends rather than two different facts.

**κ = 0.40 [0.32, 0.47] is the conventional range.** So the inflation-defined gap implies a Phillips slope consistent with the literature, obtained without estimating a Phillips curve on a target-era sample — which was the point of building the model this way. Inverting κ to get a `c` of 2.5 to 3 assumes a correlation of one; the correlation is 0.44.

### Where the identification comes from

`c` is a projection on `d`, so each quarter's weight in it is `d²/Σd²`:

| | share of the weight in `c` |
|---|---|
| top 5 quarters (3.7% of the sample) | **46.7%** |
| top 10 quarters (7.5%) | 61.8% |
| the 23 quarters with abs(d) > 1.0 (17%) | **78%** |
| 2022Q2-2023Q4 | about 53% |
| 2008Q1-2009Q1 | about 13% |

**Read that table as a description of the sample, not as a result.** Least squares weights by the square of the regressor, so identification is bound to sit in the large deviations whatever the data are, whatever the model is, and whether or not any economic story about it is true. The only empirical content in the table is a fact about Australian inflation history: `d` is distributed as two large episodes and a long quiet middle, rather than spread evenly. Worth knowing, and it is why the 2000 GST quarters do not appear, so a one-off tax-driven price level shift is not doing the identifying. But it establishes nothing on its own.

An earlier version of this section used the concentration to argue that the low corr² is not a defect but the expected signature of a central bank hitting its target. That argument does not follow. The weights would look identical if the RBA had done nothing and inflation had simply been volatile twice. The McLeay-Tenreyro reading may well be right, and it is the reason the model is built this way, but the weights are not evidence for it.

**What is not arithmetic, and what licenses a single global `c`.** Three results, none of which is forced by the weighting:

- **The quiet quarters return −0.142** [−0.392, 0.104], OLS standard error 0.110 on the median path. Low weight does not force a coefficient toward zero or the wrong sign: 111 quiet quarters could have returned 0.47 with a wide band, and they did not. This is a real rejection, and it survives the HAC inflation of the standard error that item 15 quantifies.
- **The breakout coefficient exceeds the full-sample one by only +0.043** [−0.019, +0.107] (item 13). So the quarters that carry the weight and the quarters that do not imply nearly the same number when the coefficient is actually fitted.
- **The two identifying episodes agree with each other**, +0.124 [−0.101, +0.348], P = 0.819 (item 16). Two unrelated inflation breakouts twenty years apart imply the same conversion factor. This is the result that earns the model a constant `c`, and it is the one worth quoting.

Applying that constant to the quiet quarters, where by construction the data cannot speak, remains an assumption of linearity and stability. That is an ordinary thing for a linear model to do and the three results above are what make it reasonable. It is an assumption nonetheless, and the low corr² does not stop being a real limit on what the gap can be trusted to say in quiet periods.

**It also explains why κ is the clean number and `c` is the noisy one.** Write the relationship as `d = κ·x + u`, with `u` the non-demand part of inflation. κ has `d` on the left, so `u` goes into the residual and κ is unbiased. `c` has `d` on the right, and

```
c = κ · var(x)/var(d) = corr²/κ
```

so `c` is shrunk by the correlation. That is not a bias to be corrected, and it is the proper reason the "lower bound" language earlier drafts of this file used was wrong: the gap is not understated, it is shrunk, and shrinking toward zero is the right response to a signal that explains a fifth of the variation.

**Say what the shrinkage is optimal for, though.** `c·d` is the minimum-mean-squared-error *linear* predictor of `x` given `d`, which is what a projection always is in sample. That is a real property and it is the one the gap is built on. It is not the same as being the right coefficient for recovering a structural demand gap. If part of `d` is non-demand inflation, then `d` is a noisy measure of the demand signal, it sits on the right-hand side, and `c` is attenuated by classical errors in variables. Relative to a projection the shrinkage is optimal; relative to a demand gap it is a downward bias. The model targets the first, which is why "inflation-consistent" is the honest description of what the gap measures, but the two should not be run together.

**The shrinkage is uniform and the signal-to-noise is not — but it costs almost nothing.** A single constant `c` applies the same discount to the 4.3-point deviation of 2022Q4 as to a 0.2-point deviation in 2015, though in a breakout the demand signal dominates `u` and deserves less shrinkage. Tested by split-sample projection (iteration log item 13, and read its warning about the pandemic quarters before repeating it):

| | n | mean | 90% |
|---|---|---|---|
| full sample | 134 | 0.468 | [0.345, 0.590] |
| breakout, `d` > +1, excl. 2020Q2-2021Q1 | 19 | 0.510 | [0.381, 0.639] |
| quiet, abs(`d`) ≤ 1 | 111 | **−0.142** | [−0.392, 0.104] |

The quiet quarters contain no relationship whatever: the slope straddles zero with the wrong sign, and P(breakout > quiet) = 1.000. That the weight sits in the breakouts is arithmetic, but that the quiet quarters return nothing when a coefficient is actually fitted to them is not, and it is the sharper of the two facts. But the breakout coefficient exceeds the full-sample one by only 0.043, less than a fifth of the width of `c`'s own interval, with a 90% band of [−0.019, +0.107] that crosses zero. The reason is that the signal-free quarters are also the near-weightless ones: 111 quarters carrying 22% of the leverage cannot drag `c` far. **The concern is self-limiting, and no state-dependent `c` is warranted.** At the 2022Q4 peak the uniform shrinkage costs about 0.18% of GDP in gap.

---

## Surviving the pandemic

Most trend/cycle models do not, and many simply say so: a COVID dummy, a dropped stretch of quarters, a separate variance for 2020, or a note that estimates over the period are unreliable. This model has none of that. The sample runs 1993Q1 to 2026Q2 continuously and 2020 is fitted like any other year.

It survives for a reason rather than by luck. Inflation barely moved in 2020, so the defined gap barely moves, and the seven-point collapse in output goes to `e_c`. The 2020Q2 gap of **−0.56** is the model declining to call a lockdown a demand problem. That is the correct answer and it falls out of the definition rather than out of special handling.

Both design choices are doing the protecting, not just the definition. Three counterexamples from this package's own history show what the alternative looks like: the no-residual version of this specification had potential growth swinging from **−6.0% to +9.6%** across 2020-21 as `y*` tracked GDP down and back (item 10); a Henderson-7 filter of log hours books a **−6.2 to +7.3 per cent** pandemic swing as *trend*, which is why the decomposition uses HP; and at the loosest setting in the `ratio_g` sweep the trend bends hard enough around 2020 to drag the 2019Q4 reading down to **0.47%** (item 14). The smoothness prior is what stops the last of those at the chosen setting.

Where the pandemic still bites is in analysis built on top of the model rather than in the model itself. Item 13 had to exclude 2020Q2 to 2021Q1 from the split-sample test: four quarters of deep negative output against mildly negative inflation imply a slope of 2.58, and including them would have manufactured a state-dependence result out of nothing. Any diagnostic that selects quarters on the size of `d` needs the same exclusion.

---

## Endpoint behaviour

Trend/cycle models are usually fragile at the right-hand end of the sample, because nothing has happened yet to say whether recent weakness is a dip below capacity or a fall in capacity. Iteration log item 6 is that failure in this package's own history: on the `core` spec the real-time output gap reached **−5.45%** in 2019Q4 against a full-sample value near zero.

**Two design choices remove most of that problem from this specification, and it is worth being explicit about why.**

**The gap is a data transform, not a filtered state.** `equations/inflation_gap.py` sets `output_gap = c · (pi_t − anchor)`. There is no trend standing between the data and the answer, so the estimate for quarter `t` uses `pi_t`, which is observed, and one global scalar. The whole of the real-time revision is therefore

```
revision(t) = (c_final − c_vintage) · (pi_t − anchor)
```

and the exercise reduces to asking how stable `c` is across vintages. The published band shows the same thing from the other side: the 2026Q2 gap of +0.51 [0.31, 0.72] is `c`'s interval [0.280, 0.657] rescaled by that quarter's deviation of about 1.09, because the deviation is data and carries no uncertainty. Item 6 cannot recur here in any form: reproducing −5.45 would require inflation about eleven percentage points below target.

**Potential is assumed near-static.** `sigma_g` = 0.015 means the drift bends slowly whatever arrives at the sample end, so the trend/cycle question the endpoint problem turns on is answered in advance by the prior rather than by the last few quarters of data.

**What this costs.** Because potential cannot move much and the gap is fixed by inflation, anything else in output has nowhere to go but `e_c`, which is assumed white noise. `sigma_e` = 0.98 is large. That is the price of the smoothness prior and it is paid every quarter. `e_c` is also serially correlated rather than white, with a lag-1 autocorrelation of 0.51, so it runs one-sided for years at a time and 2015-2019 is the clearest instance. Item 15 sets out why that is expected, why it leaves the reported uncertainty on `c` intact, and why the residual is better left unmodelled than fitted.

**Where it is not paid: the 2023 productivity trough.** An earlier version of this section named the migration surge as the episode where `sigma_g` looks like it binds wrongly, and proposed an episode sweep as the sharper test. That was wrong on the face of the specification. Labour is not in the model — the `inflation` spec observes log GDP and trimmed mean inflation only — so `sigma_g` has no labour supply step to refuse, and a sweep over it cannot bear on migration at all. `sigma_g` governs how closely `y*` follows GDP, and GDP grew 2.11% year-ended through 2023 against potential growth of 2.14%, so a looser setting would barely move 2023 and would pull 2024 down toward the 1% GDP was then running. See point 2 of "What it cannot say".

The one thing `sigma_g` genuinely conditions is the trend growth path itself, and the standing check for that is the honesty sweep `sigma_sweep.py` was built for: does the two-point decline in the speed limit survive the grid, or is it the smoothing? That is a general question about the specification, not an episode.

**If the real-time exercise is run, the question to ask is coverage, not revision size.** Estimates are meant to move as data arrive; that is updating, not failure. What would indict the model is a revision that fell outside what it said at the time. `realtime.py` cannot answer that as written: `_paths` keeps only median paths, and the full traces are discarded because dozens of them are tens of megabytes each. Asking the coverage question means retaining a quantile pair per vintage and adding a column to `revisions()` for how often the eventual value fell inside the real-time 90% band.

---

## Growth accounting: hours and productivity (`decompose.py`)

A **post-modelling** split of potential growth. It does not re-estimate anything: `y*` is exactly the path above, and the split is an identity in logs,

```
g_Y*  =  g_POP*  +  g_PR*  +  g_HPP*  +  g_LP*
```

with hours per labour-force participant (`HPP`) absorbing the unemployment margin, so the identity closes in three labour terms without a NAIRU. Trend productivity is the **residual** `lp* = y* − h*`, taken draw by draw, so it inherits the whole of the model's uncertainty about potential. Trend hours is filtered data and carries no band.

### The components do not get the same filter

Population keeps the Henderson-7 treatment `observations.py` already gives it. It is measured and acyclical, and its swings are genuine labour supply: growth ran 0.20% in 2021Q2 to 2.95% in 2023Q3 on the border closure and the migration rebound. HP would smooth that away as cycle, which is exactly backwards.

Participation and hours per participant are cyclical and get HP(1600). A Henderson MA cannot low-pass them at any width: on log hours a 7-term filter books a −6.2 to +7.3 per cent pandemic swing as *trend*, and even 31 terms still books −0.5 to +4.5. Under HP(1600) trend hours growth runs 0.67% in 2021 and 3.67% in 2023 — the border closure and the migration surge, not the lockdown.

### Contributions to potential growth, period averages (year-ended %)

| | Population | Participation | Hours per participant | Productivity | Total |
|---|---|---|---|---|---|
| 1994-1999 | 1.27 | −0.01 | 0.33 | 2.39 | 3.98 |
| 2000-2009 | 1.64 | 0.36 | −0.15 | 1.34 | 3.19 |
| 2010-2019 | 1.62 | −0.01 | −0.34 | 1.25 | 2.52 |
| 2020-2026 | 1.70 | 0.40 | 0.09 | −0.08 | 2.10 |

The two-point fall in the speed limit since the late 1990s is almost entirely the productivity column. The population contribution is flat at 1.3 to 1.7 throughout.

### Two presentations of the same split

`potential-growth-hours-and-productivity` draws all three as lines. `potential-growth-and-labour-input` draws potential growth and trend hours only, with productivity as the shaded wedge between them, which is exact because the three add up.

The wedge version exists because the line version invites a wrong inference. Trend productivity is `potential growth − trend hours`, and potential growth is nearly a straight slow drift, so wherever trend hours moves faster than the speed limit does, the productivity line is the hours line upside down:

| corr(trend hours growth, trend productivity growth) | |
|---|---|
| full sample | −0.711 |
| 1994-2019 | −0.391 |
| 1994-2007 | **−0.864** |
| 2020-2026 | **−0.994** |

sd of the residual is 0.96, against 0.68 for potential growth and 0.59 for trend hours: the residual is the most volatile line on a chart whose subject is the smoothest one.

Note the 1994-2007 figure. This is a property of residuals, not an artefact of the pandemic, so it cannot be fixed by shading an episode or truncating the sample — both would imply the rest of the line can be read as a measurement. The wedge makes the same numbers say the right thing: the 2023 migration surge appears as trend hours crossing above potential growth, so the residual turns negative, rather than as a productivity line plunging to −1.5.

### What it cannot say

1. **Trend productivity is a residual, not an estimate.** Any error in the hours trend lands on it in full. Nothing here is independent evidence about productivity.
2. **The 2023 trough is arithmetic, and it is not an artefact of `sigma_g`.** Trend hours growth reaches 3.7% on the migration surge while potential growth sits near 2.1%, so the residual prints below −1%. An earlier version of this note read that as the smoothness prior refusing a labour supply surge. It cannot be: **labour is not in the model.** The `inflation` spec observes log GDP and trimmed mean inflation only, and hours enter nowhere except this post-modelling identity, so there is nothing for `sigma_g` to refuse. Nor would loosening it help — `sigma_g` controls how closely `y*` follows *GDP*, and year-ended GDP growth averaged 2.11% through 2023 against potential growth of 2.14%, then fell to about 1% through 2024. A looser trend would track that down and make the residual **more** negative, not less.

   What the trough actually says is the arithmetic of the identity: output grew 2.1% while the trend of labour input grew 3.7%, so measured productivity fell. That is a statement about the Australian economy obtained from GDP and hours, and the model's smoothness did not produce it. Point 1 still applies — it is a residual, and any error in the hours trend lands on it in full.
3. **Quarter-to-quarter movements in the residual are not news about productivity.** Trend hours and the residual are near mirror images; see the correlations above. The block averages carry the low-frequency story, and the wedge chart is the safe way to show the quarterly path.
4. **HP has an endpoint problem.** Refitting the hours trend with the last four quarters withheld moves 2025Q2 trend hours growth from 2.75 to 2.34 as those quarters arrive.
5. **Trend hours is not a supply concept.** It is a filter of measured labour input. Nothing here identifies the hours consistent with inflation at target.

Run with `./run-potential-uc.sh`; `--no-decompose` skips it, which is also the only way to run `--analyse-only` without touching ABS sources. Skipped automatically for the `labour` spec, which estimates the split internally.

---

## An alternative source for potential growth (`--spec production`)

A second live specification, not a dead one. The level and the gap are exactly as above, so the inflation fulcrum still positions potential and `gap = c·(pi − 2.5)` is unchanged. What differs is where potential's *growth* comes from: a Cobb-Douglas production function instead of a free drift state.

```
g_K*_t = g_K*_{t-1} + e_K       trend capital growth,  sd = r_K · sigma_obs_gk
g_L*_t = g_L*_{t-1} + e_L       trend hours growth,    sd = r_L · sigma_obs_gl
g_M*_t = g_M*_{t-1} + e_M       trend MFP growth,      sd = r_M · sigma_obs_gm
a*_t   = a*_{t-1} + e_a         trend capital share,   sd = r_a · sigma_obs_a
g_K_t  ~ N(g_K*_t, sigma_obs_gk)     observed capital growth (5204.0)
g_L_t  ~ N(g_L*_t, sigma_obs_gl)     observed hours growth (6202.0)
mfp_t  ~ N(g_M*_t, sigma_obs_gm)     Solow residual, built with the raw share
a_t    ~ N(a*_t, sigma_obs_a)        published capital share (5204.0, 5206.0)
g_Y*_t = a*_t·g_K*_t + (1-a*_t)·g_L*_t + g_M*_t
y*_t   = y*_{t-1} + g_Y*_{t-1}
```

Four trends, four imposed ratios, and no filtering applied to the data before it reaches the model.

**The ratios are not HP lambdas.** An earlier version of this section said they were, and of `r = 1/sqrt(lambda)` that it was an exact equivalence. It is not. HP(lambda) is the local *linear trend* model, where lambda is a variance ratio against the innovation to the **slope** of an I(2) trend. These are local *level* models, I(1) random walks, so the ratio is against the innovation to the **level**, and for the same nominal lambda it smooths very much harder. The interpretable quantity is how far a trend can wander across the sample, `r × sigma_obs × sqrt(T)`.

**The smoothing must differ across the three growth trends, and that is the content of the specification.** With equal ratios and a constant alpha it would reproduce a smoothed GDP growth rate and nothing else, for the reason set out under "External comparison". What breaks that is applying different smoothing to series with different cyclicality, and the data say they differ sharply:

| | total sd | HP(1600) cycle sd | cycle share |
|---|---|---|---|
| capital growth | 0.378 | 0.140 | **0.37** |
| hours growth | 1.102 | 1.069 | **0.97** |

97% of the variation in hours growth is cycle against 37% for capital, so one setting for both under-smooths labour badly. Defaults: `ratio_gk` = 0.05, `ratio_gl` = 0.0125, `ratio_gm` = 0.025.

### alpha is smoothed almost to a constant, and that is the point

`ratio_a` = 0.00625 leaves the latent share at 0.335 to 0.338 across the sample, against a published range of 0.299 to 0.406. It is conditioned on the published series, so the model does the smoothing rather than a filter applied beforehand, but almost all of the published movement is discarded. Two reasons, and the first is empirical.

**Most of the movement is the terms of trade.** alpha is `GOS / (GOS + COE)`, and on this sample:

| correlation of the published share with | level | change |
|---|---|---|
| **terms of trade** | **+0.852** | **+0.459** |
| output gap | +0.452 | +0.236 |
| potential growth | −0.709 | −0.077 |
| HP(1600) GDP cycle | −0.033 | |

The −0.709 against potential growth is spurious, both series trending; in changes it is −0.077, which is nothing. The period means track the terms of trade almost step for step: alpha 0.313 with the ToT index at 102 over 1993-1999, 0.352 at 184 over 2009-2013, 0.333 at 162 over 2014-2019, 0.362 at 201 over 2020-2026. When ore prices rise, mining revenue lands in GOS with no matching rise in COE, because the wage bill does not scale with the price of the ore. The share moves and nothing happens to what the economy can produce.

**And a drifting alpha contradicts the functional form.** Cobb-Douglas assumes an elasticity of substitution of one, which *implies* constant factor shares. If shares genuinely move, the right form is CES with sigma ≠ 1, and a drifting alpha is a patch on that misspecification rather than a feature of the model.

Nothing rests on this numerically. Loosening `ratio_a` to 0.05 admits a 0.036 drift, which is the mining boom, and moves potential growth by 0.01pp; 0.12 and 0.30 admit progressively more of the commodity cycle and move it by 0.01 and 0.02pp. The choice is about what alpha is supposed to represent, not about the answer. `charts/PotentialUC-production/capital-share-used-in-the-production-function.png` carries the explanation on the chart itself, since a flat line against a volatile one otherwise reads as a failure to fit.

### What it gives, on the 2026Q2 vintage

Converged, `r_hat` 1.00, `ess_bulk` 3,489 to 8,901.

| | `inflation` | `production` |
|---|---|---|
| potential growth 2026Q2 | 2.14 [1.77, 2.51] | **2.16 [1.83, 2.50]** |
| output gap 2026Q2 | +0.51 [0.31, 0.72] | +0.55 [0.34, 0.75] |
| `c` | 0.468 | 0.498 |
| variance share | 19.7% | 21.6% |

The paths track to within 0.04pp throughout: 4.08/3.13/2.82/1.91 against 4.10/3.13/2.79/1.88 at 1997Q4, 2005Q4, 2012Q4 and 2019Q4.

**Two things make that agreement worth something.** It is reached by a different route, factor trends and an accounting identity rather than a smoothed drift, and unlike `cobb_douglas` that route does not collapse into a filter of GDP. And the smoothing settings agree without being made to. The three ratios, chosen from the cycle shares above, imply trend innovation sds of 0.0155, 0.0148 and 0.0162 for capital, hours and MFP. The `inflation` spec imposes `sigma_g` = 0.015. Those came from different evidence and landed on the same number, which is why the growth paths coincide.

**But be precise about how much of a validation it is: semi, and narrowly.** The two specifications share log GDP, inflation, the sample, the gap definition `c·(pi − 2.5)` and the whole level equation. The only thing that differs is where potential's growth comes from. So what the agreement tests is whether a smoothness prior on a free drift and a production decomposition put the speed limit in the same place. They do. It is not an independent replication of the model, and it does not corroborate the level, the gap or the identifying idea, all of which the two hold in common.

### Which should be the default: `inflation`

The headline number is the same either way, so the extra machinery buys nothing for it, and four reasons point the other way.

- **The cross-check only exists while they are separate.** Promoting `production` would leave one model rather than two that agree, spending the corroboration to gain nothing.
- **Fewer places to be wrong about smoothing.** `inflation` imposes two variances; `production` imposes four ratios, and the answer is known to move with them — the hours ratio spans roughly 1.8 to 2.2 in the deterministic analogue, and `ratio_a` took two attempts and a terms-of-trade investigation to settle.
- **`inflation` has no known specification defect.** `production` double-counts GDP. Inconsequential, and documented under "What it does not fix", but present.
- **Fewer dependencies.** `inflation` needs GDP and CPI. `production` adds the capital stock, hours worked and the income shares: three more series that can revise or break.

The narrower band on `production`, 1.83-2.50 against 1.77-2.51, is unexplained rather than earned, and is a small warning rather than a point in its favour.

**Where `production` is the better tool is the attribution.** Trend MFP as a state with a credible interval is something no other part of this package can produce, and the two-point fall in the speed limit being almost entirely the productivity column is the substantive economics. So: `inflation` for the headline and the gap, `production` when the question is about productivity, and the agreement quoted as a check on the growth path rather than as validation of the model.

### The real addition is the attribution

Trend MFP is now a state with a credible interval. `decompose.py` can only give productivity as a residual that absorbs every error in the hours trend and carries no uncertainty of its own, which is what "What it cannot say" point 1 concedes. Here:

| trend growth, 2026Q2 | median | 90% |
|---|---|---|
| capital | +2.32 | [1.88, 2.78] |
| hours | +1.67 | [0.99, 2.34] |
| **MFP** | **+0.26** | **[−0.25, +0.80]** |

MFP's band spans zero. That is the honest reading of Australian productivity growth and the deterministic split cannot express it. The two-point fall in the speed limit shows up as MFP collapsing from 1.44 in 1997Q4 to 0.26, while capital's contribution rose through the mining boom and then halved.

`decompose.py` is skipped for this spec, since it would re-derive productivity as a residual from a path that already carries it as a state.

### What it does not fix

The ratios are imposed, not estimated, and the answer moves with them: the hours ratio alone shifts trend growth over roughly 1.8 to 2.2 in a deterministic sweep. Freeing them runs straight into the Stock-Watson pile-up, exactly as `sigma_g` does in the main specification. So this writes the smoothing choice down as four declared numbers with a stated evidential basis, rather than learning it from data. That is an improvement on burying it in a filter where the terms cancel, and it is not the same as escaping it.

**GDP is observed twice. It is a real defect and an inconsequential one, and the MFP observation is kept anyway.** The Solow residual is `g_Y − a·g_K − (1−a)·g_L`, so GDP growth sits inside `mfp`, which is observed against `g_M*`; GDP is *also* observed in the gap equation. On the 2026Q2 vintage the model has five observed vectors of length 134 drawn from four independent data series, so **670 likelihood terms from 536 numbers**, and the 134 redundant ones are exactly the MFP equation.

`ModelConfig.mfp_observed = False` removes it. `sigma_obs_gm` then appears nowhere, so `g_M*` cannot take its innovation sd as a ratio to it and needs an imposed absolute `sigma_gm`, with `g_M*` identified through the level equation by way of the cumulation into `y*`.

| | potential growth 2026Q2 | width | `c` |
|---|---|---|---|
| **MFP observed (default)** | **2.16 [1.83, 2.50]** | 0.68 | 0.498 |
| not observed, `sigma_gm` 0.005 | 2.13 [1.87, 2.39] | 0.52 | 0.485 |
| not observed, `sigma_gm` 0.015 | 2.17 [1.83, 2.50] | 0.66 | 0.500 |
| not observed, `sigma_gm` 0.05 | 2.13 [1.52, 2.74] | 1.23 | 0.539 |

At `sigma_gm` = 0.015, matching the main spec's `sigma_g`, the two are the same model for practical purposes: 2.17 against 2.16, band 0.66 against 0.68, `c` 0.500 against 0.498. Removing 134 redundant likelihood terms changes nothing worth reporting. An earlier version of this section guessed the double-count explained why this spec's band is narrower than the `inflation` spec's, 1.83-2.50 against 1.77-2.51. It does not: the band is unchanged when the redundancy goes, so that narrowing comes from somewhere else and is unexplained.

**Kept observed, and the reason is the attribution.** The risk in dropping it was that `g_M*` would absorb the decomposition, and it does not: trend capital growth is +2.32 or +2.33 in every variant. What `sigma_gm` does control is the *shape* of the MFP path once it loses its own data. Trend MFP in 1997Q4 runs 0.78, 1.35 and 1.99 across the three settings, against 1.44 when observed. So "productivity fell from about 1.4 to 0.3", which is this specification's main contribution, would become a function of an imposed smoothness rather than of the measured Solow residual. That is a worse trade than the redundancy it removes.

Charts go to `charts/PotentialUC-production/`, a separate directory because `run_analysis` clears its chart directory before writing and the two specifications would otherwise delete each other's output.

---

## Limitations

1. **The gap is exactly as precise as `c`, and `c` is wide.** The gap *is* `c` times data, so it inherits `c`'s interval in full and has no other source of uncertainty. `c` is a projection onto a single regressor over a target-era sample, its 90% interval is [0.28, 0.66], and 78% of what identifies it comes from two episodes. Nothing tightens the gap except a better `c`. Two later tests bear on how wide it really is, and they point in opposite directions. Item 15 checks whether the residual's serial correlation means the band is understated, and finds it does not: the HAC standard error on the projection is 0.105 against the published posterior sd of 0.113, so the reported interval is if anything conservative. Item 16 finds the two identifying episodes agree with each other, which is reassuring, but also that a fifth of `c` comes from four pandemic quarters whose implied slope is 2.58 against 0.363 elsewhere. The width is honest; the sensitivity to those four quarters is the real exposure, and it is a declared one rather than a defect: they are kept in the sample deliberately, for the reasons set out under the priority list.
2. **Everything is conditional on the imposed variances.** `sigma_ystar` = 0.078 and `sigma_g` = 0.015 are set, not estimated, and they are what make potential a smooth line rather than something that follows output. This is a declared prior rather than a concealed one, and `ratio_g` = 0.025 is the HP(1600) convention for exactly this belief, so the model is at the field's default rather than at a number chosen to produce an answer. It is still an assumption, and the reported precision on potential growth is largely its precision: the sweep in iteration log item 14 moves the 90% band on latest trend growth from [1.93, 2.37] to [0.45, 3.53] across a sixteen-fold range in `ratio_g`, while barely moving the point estimate. The level and the two-point decline are in the data; the narrowness of the interval around them is not.

   The `nairu` package makes that concrete. It estimates its trend/cycle split rather than imposing it, and reports potential growth at 2026Q1 as +2.91 [1.06, 4.77], a width of 3.71 against this model's 0.74, with an output gap of −0.24 [−1.17, +0.70] against a width of 0.41 here. Four to five times wider. That is not this model knowing more; it is this model assuming more, and the wider bands are the more honest reading of what two series can support.
3. **No lag between the gap and inflation.** The relationship is contemporaneous, and two attempts to relax that failed differently. Estimating a lag profile did not converge (item 9). Fixing a four-quarter lead in advance did converge, and put about 30% weight on it, but left `sigma_e` unchanged at 0.98 on a matched sample while costing the last four quarters of the gap (item 12). The turning-point evidence that motivated it is the kind item 7 shows does not survive prewhitening.
4. **`c` is linear, and the mapping probably is not.** A single constant converts percentage points of inflation into per cent of output at every distance from target, so a 0.2-point deviation and a 4.3-point one are scaled identically. There are standard reasons to expect otherwise: capacity constraints bite convexly, downward nominal rigidity makes overshoots and undershoots asymmetric, price-setting becomes more frequent at high inflation, and small deviations are absorbed by anchored expectations in a way large ones are not. A convex Phillips curve implies a given overshoot maps to a *smaller* gap than the linear reading gives and a given undershoot to a *larger* one, so a single `c` would overstate booms and understate slack.

   Item 13 tested state dependence and found a breakout-versus-quiet difference of +0.043, but the quiet quarters carry no signal, which is that entry's own conclusion, so the comparison had little power to detect anything. Treat non-linearity as undetected on this sample rather than absent. It bears on the magnitude only: any monotone mapping preserves the sign, so the ledger reading in "What each output is for" is unaffected.

5. **The anchor is applied flat from 1993Q1.** 2.5 is a historical fact for the whole sample, so this is not a guess about policy. What is assumed is that the target was equally the operative benchmark in the first five years, while credibility was being established; the `nairu` package takes the other view and transitions its anchor to target only by 1998. The exposure is small: no quarter before 2007Q4 has abs(`d`) > 1.0, so the early sample carries almost no weight in `c`, and a 0.5 error in the effective anchor over 1993-1998 would move the gap there by about 0.24% of GDP.

6. **Inflation misses have causes other than demand, and trimmed mean does not remove them.** Trimming strips the largest price changes each period, so it removes idiosyncratic moves but not a correlated supply shock, which is precisely a shock that pushes many prices the same way at once. `d` is therefore inflation with the tails cut, not domestic demand pressure. Within the model's own terms this is not a misattribution, since the gap is *defined* as the scaled deviation and there is no separate true gap to assign the difference to. But it does two things that matter. It attenuates `c`, because a `d` containing non-demand variation is a noisy regressor (see "Where the identification comes from"), so gaps are understated in magnitude. And it is the one thing that can break the sign: a supply shock lifting inflation above target while the economy is at or below capacity makes the model read "above potential" when it is not, which is the stagflation case, and 2022-2023 carries 53% of `c`'s identification with a large imported component. Item 18 records why substituting non-tradables inflation does not fix it, and item 19 why importing the `nairu` Phillips curve's supply controls does not either. Item 19 also puts a limit on how much this matters for the estimate itself: purging the supply component leaves `c` between 0.456 and 0.478 whatever weight it is given, so the exposure is to the interpretation of the gap rather than to its size.

Two things frequently taken for limitations of this model are not, and are dealt with elsewhere: that `c` looks small beside 1/κ (it is a projection, not a reciprocal; see the implied Phillips slope), and that four fifths of the cycle is unexplained (the expected shape, given that the two identifying episodes agree while the quiet quarters carry no signal; see "Where the identification comes from").

---

## Other specifications in the package

`config.SPECS` carries `production`, documented above as a live alternative, and three earlier specifications below. The three are kept for reference, they are not the model, and their documented numbers predate the corrections in the iteration log.

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
├── decompose.py         post-modelling hours/productivity growth accounting
├── sigma_sweep.py       sweep imposed settings (untested on this spec)
├── MODEL_NOTES.md       this file
└── equations/
    ├── scale.py               sigma_c + the fixed-ratio trend sigmas
    ├── potential.py           g and y* states
    ├── production.py          y* growth from trend K, L and MFP (`production`)
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

Charts are written to `charts/PotentialUC/`: `potential-growth`, `output-gap`, `output-gap-composition`, `potential-growth-hours-and-productivity`, `potential-growth-and-labour-input`, `contributions-to-potential-growth`, and `gdp-and-potential-output` and `actual-growth-versus-potential` on full and recent windows.

`output-gap-composition` splits GDP less potential into the part inflation accounts for and the residual, as stacked bars summing to the line. Its vertical scale excludes 2020Q2 and 2020Q3, which run off the chart and are named in the header; every other quarter fits. The `inflation` spec only, since elsewhere the gap is the identity `log_gdp - y*` and the residual is identically zero.

## Commands

```bash
./run-potential-uc.sh                  # the model above: estimate + chart
./run-potential-uc.sh --analyse-only   # recharts from the saved trace

# The second live specification: potential growth from a production function,
# same level and gap. Charts to charts/PotentialUC-production/.
./run-potential-uc.sh --spec production --prefix potential_uc_production
./run-potential-uc.sh --anchor 2.25    # move the anchor
./run-potential-uc.sh --ratio-ystar 0.25   # looser potential

# Diagnostics, none of them the model. Each is off by default and each has an
# iteration log entry saying what it showed and why it was not adopted.
./run-potential-uc.sh --ar1-residual --prefix potential_uc_ar1        # AR(1) residual (item 15)
./run-potential-uc.sh --two-sided-c --prefix potential_uc_pub2sided   # c's sign estimated (item 17)
./run-potential-uc.sh --zero-deviation 2020Q2 2021Q1 --prefix potential_uc_zerodev  # lockdown carries no d (item 16)
./run-potential-uc.sh --spec production --no-mfp-observation --sigma-gm 0.015 \
    --prefix potential_uc_nomfp   # drop the MFP equation that double-counts GDP
./run-potential-uc.sh --spec production --ratio-a 0.05 --prefix potential_uc_alpha  # let alpha drift
./run-potential-uc.sh --analyse-only --no-decompose  # charts from the trace alone, no ABS access
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

**12. Letting inflation lead the gap: converges, but buys nothing.** Item 9 failed to *estimate* a lag; this asked the narrower question of whether one helps at a lag chosen in advance. The gap became `c·[w·(pi_t − 2.5) + (1−w)·(pi_{t+4} − 2.5)]`, scale and shape separated so that `c` keeps its units and a single `w ~ Beta(1,1)` carries the timing. Note the direction: if inflation *follows* the gap, the extra term is a **lead** of inflation, not a lag. k = 4 rather than 3 because `pi_basis` is "annual", so `d_t` and `d_{t+3}` share a quarter of CPI (corr 0.70) and only k ≥ 4 avoids the overlap (corr 0.55).

Unlike item 9 it sampled cleanly: `r_hat` 1.0, `ess_bulk` ≈ 9,300. The weight came back **0.71 [0.44, 0.95]**, so the model does put about 30% on inflation four quarters ahead, with P(w < 0.9) = 0.89. But on a matched 130-quarter sample (1993Q1-2025Q2, both specifications fitting the same quarters) `sigma_e` went **0.994 → 0.983**, well inside either credible interval. The residual does not shrink, which was the test. `c` rose 0.46 → 0.56, which is arithmetic rather than news: blending two imperfectly correlated regressors shrinks the amplitude and the scale compensates. The gap path crosses zero about two quarters earlier in 2021 and reaches the same peak height, then is indistinguishable from 2023Q3 on.

The motivation was turning-point timing: in both large episodes the residual peaks four quarters before the inflation-defined gap (2007Q3 against 2008Q3; 2021Q4 against 2022Q4). **Item 7 is the warning against that evidence** — it is exactly the kind of pattern between two persistent series that did not survive prewhitening. The honest reading is that the lead is visible at two turning points and invisible in the other 122 quarters, and 130 quarters of mostly quiet inflation cannot separate the two.

Not adopted. The cost is the sample end: the last k quarters have no `pi_{t+k}`, so the gap is undefined for the four quarters of most interest, and no fit gain pays for that. The alternatives were worse — falling back to the contemporaneous term alone in the tail changes the definition of the gap exactly where it matters, and carrying an inflation forecast imports a judgement into an estimate meant to rest on data. Traces retained: `potential_uc_lead4`, and `potential_uc_lead0_short` for the matched-sample comparison.

**13. Is `c` state-dependent? Not detectably, and the quiet quarters are empty.** `c` is a projection, so quarter weights are `d²/Σd²` and they are extremely concentrated: the top 5 quarters carry 46.7% of the information, the 23 quarters with abs(`d`) > 1.0 carry 78%, and it is two episodes — 2022Q2-2023Q4 about 53%, 2008Q1-2009Q1 about 13%. That raised the question of whether a single `c` over-shrinks the breakouts, since it applies the same discount whatever the signal-to-noise.

**The pandemic quarters must be excluded, and this is the trap in the test.** An `abs(d) > 1.0` rule puts 2020Q2-2021Q1 in the breakout set: inflation ran 1.2 to 1.4 *below* target while GDP sat 7.4% below potential. Those four quarters alone imply a slope of **2.58**, and they are what lift a naive breakout estimate to 0.635 (1.049 with an intercept fitted). That is the lockdown, which this specification deliberately books to `e_c`, not evidence about the inflation-output relationship. Every other breakout quarter has `d` above +1, so outside the pandemic "breakout" means the 2007-09 and 2022-24 surges.

With those four excluded: breakout `c` = 0.510 [0.381, 0.639] against a full-sample 0.468, a difference of **+0.043 [−0.019, +0.107]**, P = 0.871. Immaterial. The quiet 111 quarters give **−0.142 [−0.392, 0.104]**: no relationship, wrong sign, P(breakout > quiet) = 1.000. So identification is entirely a breakout phenomenon, and yet dropping the quiet quarters barely moves the coefficient, because 111 quarters carrying 22% of the leverage have no power to distort it. Both facts at once, and they are not in tension.

Not adopted; no state-dependent `c`. But read the verdict as **undetected rather than absent**. The comparison sets a breakout coefficient against a quiet one, and this entry's own finding is that the quiet quarters contain no signal, so the test had little power to detect state dependence in the first place. Limitation 4 records the economic reasons to expect the mapping to be non-linear regardless. Method caveat as well: `y*` is taken from the full-sample fit with a single `c`, so this is a diagnostic on the fitted decomposition rather than a re-estimation with two free coefficients. Item 16 shows that caveat bites, since the projection there understated the pandemic effect by 0.09 against proper re-estimation.

**14. The `ratio_g` sweep: the level and the decline hold, the precision does not.** `sigma_g` is imposed and trend growth is what the model exists to measure, so the standing honesty check is whether the answer survives the grid. Five runs, all converged (max `r_hat` 1.00 to 1.04):

| `ratio_g` | trend g, 1995Q4 | trend g, 2019Q4 | trend g, latest | 90% on latest | decline | potential g, latest | gap |
|---|---|---|---|---|---|---|---|
| 0.0125 | 3.84 | 2.17 | 2.149 | [1.93, 2.37] | 1.70 | 2.139 | 0.462 |
| **0.025** | 3.99 | 2.02 | **2.151** | [1.79, 2.50] | 1.84 | **2.148** | 0.513 |
| 0.05 | 4.06 | 1.77 | 2.157 | [1.59, 2.73] | 1.90 | 2.160 | 0.568 |
| 0.10 | 4.02 | 1.34 | 2.073 | [1.11, 3.00] | 1.95 | 2.095 | 0.584 |
| 0.20 | 3.83 | 0.47 | 1.970 | [0.45, 3.53] | 1.86 | 2.045 | 0.494 |

**The headlines are in the data.** Across a sixteen-fold range in the smoothing prior, latest trend growth spans 0.19pp and potential growth at 2026Q2 spans 0.12pp, from 2.05 to 2.16. The decline since 1995Q4 runs 1.70 to 1.95. So "about 2.1%, down roughly two points since the late 1990s" is not the prior speaking.

**The precision is the prior's.** The 90% band on latest trend growth runs [1.93, 2.37] at the tightest setting and [0.45, 3.53] at the loosest. The published interval is as narrow as it is because `sigma_g` is as tight as it is, which is what Limitation 2 says and this quantifies.

**The pre-COVID reading is not robust, and has two explanations.** Trend growth at 2019Q4 goes 2.17, 2.02, 1.77, 1.34, 0.47 down the grid. Part of that is the smoother: a flexible drift bends down ahead of the 2020 collapse, having seen it. But part is real, since year-ended GDP growth had already fallen from 3.11% in 2018Q2 to 1.66% by 2019Q2, recovering only to 2.20% by 2019Q4. The 0.47% at the loosest setting overshoots what that slowdown alone justifies, so both mechanisms are present and the sweep cannot separate them. Treat any quoted pre-pandemic speed limit as conditional on `sigma_g` in a way the endpoint and the total decline are not.

Traces retained as `potential_uc_sweep_ratio_g_*`.

**15. The residual is serially correlated, which is expected, and modelling it is not the answer.** `e_c` has a lag-1 autocorrelation of 0.51, so the white-noise assertion in the specification is false as a description. That is unsurprising and not a defect: policy acts with lags and expectations feed inflation, so the part of output that contemporaneous inflation does not account for has to persist. `e_c` is where the omitted IS curve and expectations block show through. `ModelConfig.ar1_residual` fits an AR(1) to it, writing the AR term into the mean of the GDP observation rather than adding a state, with the first observation carrying the stationary distribution so the two runs stay comparable at rho = 0. Converged: `r_hat` 1.00, `ess_bulk` 5,158 to 12,396.

| | white noise | AR(1) |
|---|---|---|
| `rho_e` | (0 imposed) | **0.628 [0.486, 0.757]** |
| `c` | 0.468 [0.287, 0.662] | **0.404 [0.097, 0.681]** |
| `sigma_e`, innovation | 0.981 | 0.863 |
| residual sd, stationary | 0.981 | **1.132** |
| potential growth, 2026Q2 | 2.14 [1.77, 2.51] | **2.12 [1.67, 2.56]** |
| output gap, 2026Q2 | +0.51 [0.31, 0.72] | **+0.44 [0.12, 0.77]** |

**Nothing important moves.** Potential growth is 2.12 against 2.14 at the endpoint, and the path holds throughout (1997Q4 3.89 against 4.10, 2005Q4 3.16 against 3.13, 2012Q4 2.68 against 2.79). `c` goes 0.468 to 0.404 and the gap 2026Q2 +0.51 to +0.44, both well inside their own bands. The two-point decline survives. What does change is `c`'s posterior sd, 0.113 to 0.177.

**But that widening is the AR(1) discarding information, not revealing hidden uncertainty.** With persistent errors the likelihood weights `c` like a quasi-differenced GLS regression, and `d` is itself highly persistent, so quasi-differencing strips out much of the low-frequency variation the definition relies on. The check is a HAC standard error, which corrects the uncertainty without re-weighting the estimate. On the median path the OLS slope is 0.468 with an iid standard error of 0.080 and a HAC standard error of **0.105** (stable at 0.105, 0.105, 0.105, 0.106 for lags 2, 4, 6, 8). The published posterior sd of 0.113 already exceeds it, because the state-space estimation carries uncertainty in `y*` that OLS does not. **So the published band on `c` is not understated, and no widening is warranted.**

**Not adopted, and the reason is structural rather than statistical.** Serial correlation in `e_c` is what an omitted IS curve and expectations block look like. AR(1) is a one-parameter reduced form for that structure, and a poor one, since policy transmission runs six to eight quarters rather than decaying geometrically from one. Adopting it would smuggle back in exactly what this package was built to do without, and would do so badly. The option is kept off by default, in the same spirit as `free_sigma_ystar`: to make the check reproducible, not because the alternative is a candidate. Trace retained as `potential_uc_ar1`.

One incidental result worth keeping: trend growth at 2019Q4 goes 1.88 to 2.11, because a persistent residual reads the 2018-19 slowdown as cycle where iid errors read part of it as trend. That is a second reason to treat the pre-pandemic reading as conditional, alongside the `ratio_g` sweep in item 14.

**16. Leave-one-episode-out: the two inflation episodes agree, and the pandemic is doing undeclared work.** Item 13 established that identification is concentrated in two episodes. That is a stability question as well as a precision one, since both are periods in which the assumption that inflation deviations reveal domestic excess demand is least innocuous. The test is whether the episodes agree with each other. Method as in item 13, and with its caveat: `y*` comes from the full-sample fit with a single `c`, so this is a diagnostic on the fitted decomposition rather than a re-estimation. Bands are draw spreads, which carry the uncertainty in `y*` but not the regression standard error; the OLS standard errors on the median path are given separately where they matter.

| subsample | n | weight | `c` | 90% |
|---|---|---|---|---|
| full sample | 134 | 100% | 0.468 | [0.342, 0.593] |
| 2008Q1-2009Q1 only | 5 | 12.8% | 0.406 | [0.199, 0.611] |
| 2022Q2-2023Q4 only | 7 | 53.0% | 0.530 | [0.405, 0.656] |
| drop 2022Q2-2023Q4 | 127 | 47.0% | 0.398 | [0.215, 0.579] |
| drop 2008Q1-2009Q1 | 129 | 87.2% | 0.477 | [0.346, 0.605] |
| drop both | 122 | 34.3% | 0.395 | [0.185, 0.598] |

**The stability test passes, but read it as an absence of contradiction rather than as strong evidence.** The two episodes are twenty years and two very different shocks apart, and they imply the same conversion factor: the difference is **+0.124 [−0.101, +0.348]**, P = 0.819. Drop either, or both, and `c` stays near 0.4 with a band clear of zero.

The caution is power. The OLS standard errors are 0.142 on 2008-09 with five quarters and 0.034 on 2022-23 with seven, so the standard error on the difference is 0.146 and the observed 0.125 is 0.86 of it. The test could only reject equality for a difference above about 0.29, more than half of `c` itself, and the 2008-09 estimate alone spans roughly 0.13 to 0.68. So the episodes agreeing is partly the earlier one being uninformative. Worth having, not worth leaning on.

**The pandemic quarters are the finding.** 2020Q2-2021Q1 carries 4.7% of the weight but implies a slope of 2.58, so it lifts `c` by about 0.105, a fifth of the total. Dropping those four quarters alone moves `c` from 0.468 to **0.363 [0.230, 0.494]**; dropping them together with 2022-2023 gives 0.154 [−0.039, 0.344], and dropping all three episodes leaves the remaining 118 quarters at **0.045 [−0.182, 0.266]** (OLS se 0.097). The direction is unlucky: mean `d` is −1.275 against mean `x` of −3.415, both negative, so the lockdown reads as a large positive-`c` observation and flatters the estimate.

Item 13 excludes those same quarters, but read its scope before concluding this file contradicts itself. That exclusion protects a *diagnostic that selects quarters on the size of `d`*, where a rule built to catch inflation breakouts scoops up a lockdown instead. The estimation sample selects on nothing, so the two treatments are consistent. What is left here is a sensitivity, and a real one: four quarters out of 134 carry about a fifth of `c`. The decision to keep them, and the reasoning, is recorded under the priority list below.

**17. The sign of `c` is imposed by its prior, so the premise was never tested. Tested now, it holds.** `c` is given a `HalfNormal(2.0)`, which forbids a negative value. Every statement in this file of the form "`c`'s interval is clear of zero" therefore rested on a prior that ruled out the alternative, and could not have come out any other way. `ModelConfig.two_sided_c` swaps in `Normal(0, 2)` so the posterior can place mass below zero, which is the only form in which "inflation locates the gap" is a proposition rather than an assumption.

| specification | `c` | P(`c` > 0) | gap 2026Q2 |
|---|---|---|---|
| published, HalfNormal | 0.468 [0.280, 0.657] | 1.000 | +0.51 |
| published, **Normal(0, 2)** | **0.468 [0.283, 0.654]** | 1.000 | +0.51 |
| AR(1), Normal(0, 2) | **0.394 [0.079, 0.705]** | **0.977** | +0.44 |

**The published estimate is not a prior artefact.** Freeing the sign changes `c` in the third decimal, 0.468 either way, because the likelihood puts the coefficient about four posterior standard deviations from zero and the truncation is never binding. The prior was doing no work.

**And it survives the residual correction too.** With AR(1) errors and a two-sided prior together, so neither the serial correlation nor the sign is assumed, `c` = 0.394 with 97.7% of the posterior mass positive. That is the model's central proposition tested on its weakest defensible footing, and it passes.

Where the prior does bind is once the four pandemic quarters are also removed (item 16): `c` then goes to −0.024 [−0.378, +0.320] with P(`c` > 0) = 0.46, against +0.157 under the half-normal. So in that corner the truncation is worth 0.18 and the earlier "still positive" reading of it was the prior talking. Read that as loss of power rather than evidence against: deleting four high-leverage observations from a 134-quarter sample leaves a posterior wide enough to contain 0.394 comfortably, so it cannot see the coefficient rather than contradicting it. Traces retained as `potential_uc_pub2sided`, `potential_uc_ar1_2sided`, `potential_uc_test2sided`.

**18. Non-tradables inflation instead of trimmed mean: checked, and the anchor does not survive it.** Trimmed mean removes the largest price changes each period, which strips idiosyncratic moves but not a correlated supply shock, since a shock that pushes many prices the same way at once is not an outlier once it is broad. Energy pass-through, shipping costs, exchange rate depreciation and administered price rises all survive trimming. So `d` is inflation with the tails cut, not domestic demand pressure, and `c` is attenuated accordingly (see the errors-in-variables note in "Where the identification comes from"). Non-tradables inflation is the obvious closer proxy for domestic pressure, so it was checked as a replacement anchor variable.

Quarterly tradables and non-tradables year-ended growth is ABS 6401.0 table **640108**, and it is no longer in the latest release: the quarterly series ended with the September quarter 2025 CPI, so it has to be fetched from that archived landing page. Both run **1999Q2 to 2025Q3, 106 quarters**, against this model's 134.

**The wedge against trimmed mean is large and unstable, which is disqualifying.** Over the common sample non-tradables averages 3.61 against trimmed mean's 2.75, so the 2.5 fulcrum would have to move to about 3.4. That alone is a recalibration. But the gap between the two does not sit still:

| | non-tradables | trimmed mean | wedge |
|---|---|---|---|
| 2000-2007 | 4.15 | 2.82 | **+1.33** |
| 2008-2014 | 3.73 | 2.90 | +0.83 |
| 2015-2019 | 2.35 | 1.74 | +0.61 |
| 2020-2025 | 4.04 | 3.45 | +0.59 |

The wedge moves about 0.7pp across the sample, while the level corrections this model makes are a few tenths. An anchor error of that size would swamp the signal it is meant to carry, so the anchor would have to become time-varying and estimated. That is the identification problem the package exists to avoid, and it forfeits the model's cleanest property: 2.5 is an institutional fact that can be asserted, whereas 3.4-and-drifting is a parameter that has to be defended.

Two lesser objections. The series is discontinued, so every future vintage needs the 6401.0 plus 6484.0 monthly splice with a methodology break inside the sample. And the sample loses the last three quarters, which matters more here than losing the first twenty-five, since 1993-1999 carries only 4.9% of `c`'s weight while the endpoint is the estimate people use. One point in favour, for the record: non-tradables has sd 1.50 against trimmed mean's 1.08, and identification is `d²`-weighted, so there would genuinely be more to work with. Correlation between the two is 0.722.

Not adopted. The supply component in `d` stays a stated limitation rather than something the data can be re-cut to remove. Note what this also rules out: non-tradables cannot serve even as a cheap sign check on the ledger reading, because defining its deviation requires the anchor that does not work.

**19. Purging the supply component out of `d`: does not work, and fails informatively.** Limitation 6 says trimmed mean carries supply as well as demand inflation. The `nairu` price Phillips curve already controls for exactly that, with import price pass-through `rho_pi` = 0.015 and a sign-preserving quadratic supply-chain term `xi_gscpi` = 0.045 (both from `nairu_simple_excess_target`). The question is whether those augmentations transfer.

They cannot transfer as they stand, because the equations point opposite ways. In `nairu` inflation is on the left and the controls sit beside the gap, soaking up part of inflation. Here inflation is the regressor, so the analogue is to purge the deviation before scaling it: `gap = c·(pi − 2.5 − supply)`. Alignment matters: `d` is a four-quarter rate, so the supply term must be the accumulation of the quarterly contributions over those same four quarters. Annualising the contemporaneous quarterly contribution instead puts the GSCPI peak in 2022Q2 against a CPI peak in 2022Q4 and produces negative deviations in early 2022.

**No scaling of the correction improves anything.** Slopes are OLS projections of `x` on the purged deviation, on the published median `y*` path:

| variant | 2008-09 | 2022-23 | full `c` | se |
|---|---|---|---|---|
| **raw, no purge** | 0.405 | **0.530** | 0.468 | **0.080** |
| import prices only | 0.387 | 0.602 | 0.460 | 0.082 |
| imports + 0.25 GSCPI | 0.386 | 0.689 | 0.472 | 0.087 |
| imports + 0.5 GSCPI | 0.386 | 0.800 | 0.478 | 0.091 |
| imports + 0.75 GSCPI | 0.385 | 0.941 | 0.473 | 0.095 |
| imports + full GSCPI | 0.384 | 1.110 | 0.456 | 0.099 |

`c` never leaves 0.456 to 0.478, so the answer does not change. Precision falls monotonically as more supply is removed, since Σd² drops 28% at full strength. And cross-episode agreement, which item 17 treats as the model's best evidence, worsens monotonically: the two episodes go from differing by a factor of 1.3 to differing by a factor of 2.9. The unpurged data is the best-behaved point on the grid, and even import prices alone, the mild continuous term, makes things worse. This is not a calibration to tune.

**What that establishes, and what it does not.** It establishes that these particular supply controls do not transplant into an inverse specification: on every observable criterion the table reports, precision and cross-episode stability both deteriorate, monotonically. That is enough to reject the correction.

It does not establish that 2022-23 inflation was demand-driven. The mechanism is that output sat 1.71% above trend while inflation ran 3.11pp above target, so removing half the inflation while leaving output where it is forces the episode's mapping to more than double. But that "1.71% above trend" is measured against a `y*` this model produced, so it cannot independently prove what drove the inflation. An earlier version of this entry said strong output with high inflation is a demand signature, which was circular. The concern in Limitation 6 is right in principle and is not addressed by this experiment either way.

**Not adopted.** `beta_pi`, the third `nairu` augmentation, was not considered: importing excess expectations means a time-varying anchor, which is ruled out separately and for stronger reasons.

Method notes. This is a projection on a fixed `y*`, so item 13's caveat applies, and that caveat bit once already (item 16's projection understated the pandemic effect by 0.09 against re-estimation). Settling it properly is one estimation run. `nairu` masks its GSCPI to 2020Q1-2023Q2 and zeroes it elsewhere; the test above used the unmasked series, since the quadratic form already concentrates the effect on extremes and the mask only removes contributions of at most 0.74pp against 2.50pp inside the window. Sign flips from purging are all at deviations of 0.7 or smaller, so they do not touch the ledger reading.

**A data note that outlives the test.** GSCPI is now fetched live rather than read from a stale workbook: `src/data/gscpi_live.py` pulls it from the NY Fed through the same freshness-checked cache the ABS loaders use, and runs to 2026Q2. It is deliberately a separate module from `src/data/gscpi.py`, which reads a checked-in copy that stops at 2024Q1 and which `nairu` depends on; swapping the source underneath a published model was not worth doing for this. On the live data GSCPI has risen from 0.13 in 2025Q4 to **1.63 in 2026Q2**, its highest since 2022. With the two-quarter lag that does not touch the current gap, but it is the Limitation 6 case building in real time: inflation pushed up by something other than domestic demand, which this model will read as a positive gap.

**20. `cobb_douglas` was a filter of GDP, and a production specification that is not.** The external comparison in this file rested on `cobb_douglas` being an independent route to potential growth. It is not. Its MFP term is the Solow residual and HP is linear, so with a common lambda and a constant alpha the factor terms cancel and `g_potential` reduces to `HP(g_Y)` exactly. Confirmed by computing both: identical to **0.000 at every quarter** on the sample, with capital, hours, MFP and alpha contributing nothing.

That is a stronger version of the failure item 3 found in the `core` spec and item 8 found in an early `target` spec. It was invisible for longer because the cancellation is algebraic rather than an empirical near-match, so no correlation diagnostic would have caught it: the terms were there, doing nothing.

**Two candidate escapes, and only one works.** A time-varying alpha breaks the linearity argument, since `HP(alpha_t·g_K) ≠ alpha_t·HP(g_K)`. But a capital share smoothed enough to be usable drifts slowly, and on this sample the whole channel is worth 0.023pp at most, 0.011 at the endpoint. Dead. Differential smoothing is the one that works, and it is justified rather than convenient: the HP(1600) cycle is 97% of the variation in hours growth against 37% for capital, so a common filter is straightforwardly wrong.

A deterministic sweep put numbers on both the gain and the cost. Against an `HP(g_Y)` benchmark of 1.81 at 2026Q2, differential smoothing gives 2.06. But sweeping the hours lambda alone moves the endpoint from 1.50 to 2.24, and the capital lambda does almost nothing, so the answer is a choice about how hard to smooth labour. Two things narrow it: at lambda 400 the path oscillates, dipping to 1.55 in 2019 and spiking to 2.93 in 2022, which is not a potential growth path at all; and the benchmark's steep fall at the endpoint is the filter chasing recent weak GDP, where the smoother variants flatten near the RBA's ~2.0.

**Built as `--spec production`**, documented above. It agrees with the `inflation` spec at 2.15 against 2.14, which restores the cross-check the package lost when `cobb_douglas` turned out to be a filter, and gives trend MFP a credible interval for the first time. Nothing in `cobb_douglas` was changed.

**Next, in priority order:**
1. **Corroborate `c` from outside the sample.** No longer a correction — the implied κ of 0.40 says the internal estimate is not biased down, so this is now a matter of tightening a wide interval [0.28, 0.66] rather than replacing a suspect number. It matters because the gap is `c` times data: `c` is the whole of the gap's uncertainty and the whole content of any real-time revision to it. The cross-sectional route (state unemployment against capital-city CPIs, where the cash rate is common) is the standard way to get an independent read.
2. **Delete the dead specifications** once nothing further is wanted from them, along with the equation modules only they use.
3. **Rewire `realtime.py` to the `inflation` spec, if it is still wanted.** Demoted. "Endpoint behaviour" sets out why this specification has structurally little to fear from the test: the gap is not filtered, so the exercise only measures the stability of `c`. Worth doing for the trend rather than the gap, and only in the coverage form described there, which needs the module to retain quantiles rather than medians.

Settled, and deliberately: **2020Q2-2021Q1 stays in the estimation sample.** Item 16 shows those four quarters carry 4.7% of the weight at an implied slope of 2.58 against 0.363 elsewhere, so they supply roughly a fifth of `c`; removing them takes `c` to 0.273 and the 2026Q2 gap to +0.30. Item 13's exclusion of them does not condemn this: it is scoped to diagnostics that select quarters on the size of `d`, where a rule built to catch inflation breakouts scoops up a lockdown instead, and the estimation sample selects on nothing. Three reasons to keep them. The specification has no dummy machinery and "Surviving the pandemic" is built on a continuous sample, so dropping four quarters is exactly the intervention this package avoids. Zeroing their deviation is not neutral either: it asserts the gap was zero in those quarters, which is a claim rather than an abstention. And the cost is steep, because the variance share is quadratic in `c` and falls from **19.7% to 6.5%**, turning "inflation accounts for about a fifth of the deviation from trend" into about a fifteenth. What remains is a declared sensitivity rather than an inconsistency, and item 16 is where it is recorded.

Not on the list, and deliberately: **weakening the fixed anchor**. Earlier drafts had it as item 2, on the view that the anchor was the model's most fragile assumption. It is not an assumption at all: 2.5 is the RBA's target midpoint for the whole sample, and treating it as data is the identifying idea rather than a weakness in it. Loosening it would remove the level information the anchor supplies and return the model to a filter.
