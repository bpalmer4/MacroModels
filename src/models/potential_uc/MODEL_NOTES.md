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

## What this model does

Two ideas, and that is the whole model.

**A prior: potential output is a slow-moving trend.** Actual output jumps about from quarter to quarter for all sorts of reasons. The economy's capacity to produce does not. So we assume potential moves slowly relative to GDP.

That gives the trend its shape but not its position. A slow-moving line can be drawn high or low through the same data, and nothing about being slow-moving says which is right.

**A definition: potential output is the level of output consistent with inflation at target.** That is what puts the line in the right place. When inflation runs above 2.5% the economy is running beyond its capacity; when it runs below, it has room to spare.

Between 2015 and 2019 inflation sat under target for five years. This model reads that as an economy running below capacity throughout. A statistical filter, knowing nothing about inflation, reads the same five years as running above it. That difference is the definition doing its work, and it is the reason for building the model this way.

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

The full-sample +0.08 is the point. It is near zero as an **outcome**, not zero by construction. An HP cycle cannot produce anything else, over the whole sample or over any long stretch of it, because it is a residual from a trend fitted to pass through the middle of the data: a run of years below capacity has to be paid back by a run above. 2015-2019 is where the two methods part company. This model books five years as a sustained shortfall; the filter books them as a wander that averages out against the neighbouring periods.

Two qualifications, neither of which is an objection to the claim.

**The level correction is modest, and correctly so.** Sub-period mean deviations are a few tenths of a percentage point, and `c` shrinks them further, so 2015-2019 comes out at −0.36 rather than the −1.9 you would get by inverting a Phillips slope. That shrinkage is not an understatement to be corrected: `c·d` is a conditional mean, and shrinking toward zero is the right response to a signal that explains a fifth of the variation. See "Where the identification comes from". The fulcrum fixes the sign and timing of the level firmly, and the magnitude of the correction more weakly.

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

**The first two are the meaningful check.** They share no equations and almost no data: one takes GDP and inflation and defines the gap by the target, the other takes factor inputs and shares and never looks at prices. They differ by 0.28pp, inside this model's 90% band. `cobb_douglas` is also still falling (2.01 in 2025Q2 to 1.86 in 2026Q2), so the two are converging from opposite directions rather than sitting on a shared assumption. Two routes to the same neighbourhood is worth more than either route alone, and it is the only genuinely external evidence the package has. (The Cobb-Douglas figure at α = 0.36 is nearer 1.97; the 1.86 here is the repo's α = 0.30 default.)

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

`c` is a projection on `d`, so each quarter's weight in it is `d²/Σd²`. That weight is extremely concentrated:

| | share of the information in `c` |
|---|---|
| top 5 quarters (3.7% of the sample) | **46.7%** |
| top 10 quarters (7.5%) | 61.8% |
| the 23 quarters with abs(d) > 1.0 (17%) | **78%** |

And it is two episodes rather than a scatter. **2022Q2-2023Q4 carries about 53% on its own; 2008Q1-2009Q1 about 13%.** The remaining 111 quarters supply roughly a fifth of what is known about `c`. The 2000 GST quarters do not appear, so a one-off tax-driven price level shift is not doing the identifying.

**This is why the low corr² is not a defect.** It is not the model failing to explain output. It is the model correctly reporting that in quiet quarters inflation carries almost no information about the gap — which is true, and is precisely what a central bank hitting its target produces. The identification lives in the rare breakouts, and the model uses them.

**It also explains why κ is the clean number and `c` is the noisy one.** Write the relationship as `d = κ·x + u`, with `u` the non-demand part of inflation. κ has `d` on the left, so `u` goes into the residual and κ is unbiased. `c` has `d` on the right, and

```
c = κ · var(x)/var(d) = corr²/κ
```

so `c` is shrunk by the correlation. That is not a bias to be corrected. Shrinking toward zero is the correct response to a noisy signal, and it is what makes `c·d` a conditional mean rather than a structural inversion. It is also the proper reason the "lower bound" language earlier drafts of this file used was wrong: the gap is not understated, it is optimally shrunk.

**The shrinkage is uniform and the signal-to-noise is not — but it costs almost nothing.** A single constant `c` applies the same discount to the 4.3-point deviation of 2022Q4 as to a 0.2-point deviation in 2015, though in a breakout the demand signal dominates `u` and deserves less shrinkage. Tested by split-sample projection (iteration log item 13, and read its warning about the pandemic quarters before repeating it):

| | n | mean | 90% |
|---|---|---|---|
| full sample | 134 | 0.468 | [0.345, 0.590] |
| breakout, `d` > +1, excl. 2020Q2-2021Q1 | 19 | 0.510 | [0.381, 0.639] |
| quiet, abs(`d`) ≤ 1 | 111 | **−0.142** | [−0.392, 0.104] |

The quiet quarters contain no relationship whatever — the slope straddles zero with the wrong sign, and P(breakout > quiet) = 1.000. That is the strongest single confirmation that identification lives in the breakouts. But the breakout coefficient exceeds the full-sample one by only 0.043, less than a fifth of the width of `c`'s own interval, with a 90% band of [−0.019, +0.107] that crosses zero. The reason is that the signal-free quarters are also the near-weightless ones: 111 quarters carrying 22% of the leverage cannot drag `c` far. **The concern is self-limiting, and no state-dependent `c` is warranted.** At the 2022Q4 peak the uniform shrinkage costs about 0.18% of GDP in gap.

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

**What this costs.** Because potential cannot move much and the gap is fixed by inflation, anything else in output has nowhere to go but `e_c`, which is assumed white noise. `sigma_e` = 0.98 is large. That is the price of the smoothness prior and it is paid every quarter.

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

## Limitations

1. **The gap is exactly as precise as `c`, and `c` is wide.** The gap *is* `c` times data, so it inherits `c`'s interval in full and has no other source of uncertainty. `c` is a projection onto a single regressor over a target-era sample, its 90% interval is [0.28, 0.66], and 78% of what identifies it comes from two episodes. Nothing tightens the gap except a better `c`.
2. **Everything is conditional on the imposed variances.** `sigma_ystar` = 0.078 and `sigma_g` = 0.015 are set, not estimated, and they are what make potential a smooth line rather than something that follows output. This is a declared prior rather than a concealed one, and `ratio_g` = 0.025 is the HP(1600) convention for exactly this belief, so the model is at the field's default rather than at a number chosen to produce an answer. It is still an assumption, and the reported precision on potential growth is largely its precision: the sweep in iteration log item 14 moves the 90% band on latest trend growth from [1.93, 2.37] to [0.45, 3.53] across a sixteen-fold range in `ratio_g`, while barely moving the point estimate. The level and the two-point decline are in the data; the narrowness of the interval around them is not.
3. **No lag between the gap and inflation.** The relationship is contemporaneous, and two attempts to relax that failed differently. Estimating a lag profile did not converge (item 9). Fixing a four-quarter lead in advance did converge, and put about 30% weight on it, but left `sigma_e` unchanged at 0.98 on a matched sample while costing the last four quarters of the gap (item 12). The turning-point evidence that motivated it is the kind item 7 shows does not survive prewhitening.
4. **The anchor is applied flat from 1993Q1.** 2.5 is a historical fact for the whole sample, so this is not a guess about policy. What is assumed is that the target was equally the operative benchmark in the first five years, while credibility was being established; the `nairu` package takes the other view and transitions its anchor to target only by 1998. The exposure is small: no quarter before 2007Q4 has abs(`d`) > 1.0, so the early sample carries almost no weight in `c`, and a 0.5 error in the effective anchor over 1993-1998 would move the gap there by about 0.24% of GDP.

Three things frequently taken for limitations of this model are not, and are dealt with elsewhere: that `c` looks small beside 1/κ (it is a projection, not a reciprocal; see the implied Phillips slope), that four fifths of the cycle is unexplained (the expected shape, since identification lives in the breakouts; see "Where the identification comes from"), and that inflation misses have causes other than demand (the gap is defined as the scaled deviation, so there is no separate true gap to misattribute to).

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
├── decompose.py         post-modelling hours/productivity growth accounting
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

Charts are written to `charts/PotentialUC/`: `potential-growth`, `output-gap`, `output-gap-composition`, `potential-growth-hours-and-productivity`, `potential-growth-and-labour-input`, `contributions-to-potential-growth`, and `gdp-and-potential-output` and `actual-growth-versus-potential` on full and recent windows.

`output-gap-composition` splits GDP less potential into the part inflation accounts for and the residual, as stacked bars summing to the line. Its vertical scale excludes 2020Q2 and 2020Q3, which run off the chart and are named in the header; every other quarter fits. The `inflation` spec only, since elsewhere the gap is the identity `log_gdp - y*` and the residual is identically zero.

## Commands

```bash
./run-potential-uc.sh                  # the model above: estimate + chart
./run-potential-uc.sh --analyse-only   # recharts from the saved trace
./run-potential-uc.sh --anchor 2.25    # move the anchor
./run-potential-uc.sh --ratio-ystar 0.25   # looser potential
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

**13. Is `c` state-dependent? No, and the quiet quarters are empty.** `c` is a projection, so quarter weights are `d²/Σd²` and they are extremely concentrated: the top 5 quarters carry 46.7% of the information, the 23 quarters with abs(`d`) > 1.0 carry 78%, and it is two episodes — 2022Q2-2023Q4 about 53%, 2008Q1-2009Q1 about 13%. That raised the question of whether a single `c` over-shrinks the breakouts, since it applies the same discount whatever the signal-to-noise.

**The pandemic quarters must be excluded, and this is the trap in the test.** An `abs(d) > 1.0` rule puts 2020Q2-2021Q1 in the breakout set: inflation ran 1.2 to 1.4 *below* target while GDP sat 7.4% below potential. Those four quarters alone imply a slope of **2.58**, and they are what lift a naive breakout estimate to 0.635 (1.049 with an intercept fitted). That is the lockdown, which this specification deliberately books to `e_c`, not evidence about the inflation-output relationship. Every other breakout quarter has `d` above +1, so outside the pandemic "breakout" means the 2007-09 and 2022-24 surges.

With those four excluded: breakout `c` = 0.510 [0.381, 0.639] against a full-sample 0.468, a difference of **+0.043 [−0.019, +0.107]**, P = 0.871. Immaterial. The quiet 111 quarters give **−0.142 [−0.392, 0.104]**: no relationship, wrong sign, P(breakout > quiet) = 1.000. So identification is entirely a breakout phenomenon, and yet dropping the quiet quarters barely moves the coefficient, because 111 quarters carrying 22% of the leverage have no power to distort it. Both facts at once, and they are not in tension.

Not adopted; no state-dependent `c`. Method caveat: `y*` is taken from the full-sample fit with a single `c`, so this is a diagnostic on the fitted decomposition rather than a re-estimation with two free coefficients. Given a difference of 0.043 that is unlikely to change the verdict, but it is not the same test.

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

**Next, in priority order:**
1. **Corroborate `c` from outside the sample.** No longer a correction — the implied κ of 0.40 says the internal estimate is not biased down, so this is now a matter of tightening a wide interval [0.28, 0.66] rather than replacing a suspect number. It matters because the gap is `c` times data: `c` is the whole of the gap's uncertainty and the whole content of any real-time revision to it. The cross-sectional route (state unemployment against capital-city CPIs, where the cash rate is common) is the standard way to get an independent read.
2. **Delete the dead specifications** once nothing further is wanted from them, along with the equation modules only they use.
3. **Rewire `realtime.py` to the `inflation` spec, if it is still wanted.** Demoted. "Endpoint behaviour" sets out why this specification has structurally little to fear from the test: the gap is not filtered, so the exercise only measures the stability of `c`. Worth doing for the trend rather than the gap, and only in the coverage form described there, which needs the module to retain quantiles rather than medians.

Not on the list, and deliberately: **weakening the fixed anchor**. Earlier drafts had it as item 2, on the view that the anchor was the model's most fragile assumption. It is not an assumption at all: 2.5 is the RBA's target midpoint for the whole sample, and treating it as data is the identifying idea rather than a weakness in it. Loosening it would remove the level information the anchor supplies and return the model to a filter.
