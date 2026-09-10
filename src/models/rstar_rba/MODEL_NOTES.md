# r* from the RBA's reaction to inflation

**This is not an estimate of r\*. It is an estimate of what the RBA's behaviour reveals
about r\*, and it reveals it imperfectly.** Everything below depends on that sentence, so
it is the first one.

A Bayesian unobserved-components model (PyMC + NumPyro NUTS) that decomposes the cash rate
into a slow trend and a response to inflation, and calls the sum neutral.

```
pi_t   = 4 · sum_{j=0}^{4} w_j · q_{t-j},   w_j ∝ rho^j,  sum w_j = 1
g_t    = (pi_t - 2.5) / 0.5                  the inflation gap, in band-widths
b_t    = b_{t-1} + sigma_r · e_t             the BASE: the slow trend, sigma_r imposed
r*_t   = b_t + lambda · g_t                  r*: the complete estimate
r_t    = r*_t + u_t,   u_t ~ Normal(0, sigma_u)
```

`r_t` is the nominal cash rate and `q_t` quarterly trimmed-mean inflation. `r*_t` is the
neutral **nominal** cash rate; real neutral is `r* − 2.5`. Sample 1993Q1-2026Q2, 134
quarters.

**`b_t` is the base, not r\*.** The base is the hidden trend that roughly anchors neutral.
r\* is the base plus the inflation response, and that sum is what the cash rate is compared
against. Reading the base as r\* is the single most common way to misuse this model, and
several charts and one downstream variant were built wrong that way before being fixed. The
distinction is worth holding onto: the base sets the level, the response supplies the turns.

## One of three routes, all flawed

This repo contains three separate attempts at Australian r\*, and the useful thing is that
they fail differently. Read them together rather than picking one.

| package | identified from | what it actually measures | how it fails |
|---|---|---|---|
| [`rstar_hlw`](../rstar_hlw/MODEL_NOTES.md) | trend growth + the IS curve | the textbook definition: the rate at which output sits at potential | the IS curve does not identify anything on AU data, so each specification returns its own prior |
| [`rstar_bonds`](../rstar_bonds/MODEL_NOTES.md) | asset prices | what investors price | the *level* is not identified; it rests on the stationarity prior and an asserted premium |
| **`rstar_rba`** (this one) | the RBA's response to inflation | what the Bank's conduct implies it believed neutral was | a long enough departure from the rule is absorbed into neutral, so it cannot audit the Bank over a decade |

Only the first targets what the theory defines, and it is the one that cannot be estimated.
The other two measure *beliefs* about r\*, held by different people.

Current comparison: `rstar_hlw` Resolution G gives 2.23, `rstar_bonds` 1.08, this **0.99**
real. Nobody has reconciled them.

---

## What "revealed, imperfectly" means

The model assumes the RBA set the cash rate as neutral plus a response to inflation, then
recovers neutral as the rate minus the response. Four consequences, in order of how much
they should change how you read the output.

**1. It sees short departures from the rule, and progressively less of long ones.**

This was written up as "circular: it cannot find the RBA wrong, the residual averages zero
by construction", and that is too strong. What is pinned is the residual's *whole-sample*
mean, because the base's starting level is free and absorbs it. That is a normalisation,
not a finding. Everything else is estimated, and the model plainly does report policy
sitting away from its own rule: the residual has an sd of 0.74 against 1.98 for the cash
rate, so roughly a third of the cash rate's variation is departure from the rule, and the
era means are not zero (2020-21 at −0.69, 2012-15 at −0.15, 2022 on at +0.15).

The real limitation is DURATION, and it is gradual rather than a threshold. The base is a
random walk drifting about `sigma_r` = 0.10 points a quarter, so how far it can wander
grows with the square root of time: roughly 0.20 over a year, 0.35 over three, 0.45 over
five, 0.63 over ten, against a residual sd of 0.74. **The longer the cash rate sits away
from neutral, the more of that gap the model attributes to neutral itself having moved.**
It cannot tell a long stretch of easy policy from a fall in neutral.

Two things follow, and they point in opposite directions, so do not use only one.

*2016-2019 is a finding, not an artefact.* The residual really is −0.02 there and it did not
have to be. Over three or four years the base can wander only about 0.35 to 0.40, so a
sustained full-point stance would not have been quietly absorbed.

*But the absorption is untested.* Those figures are how far the base *could* wander under
its prior, not how much of an imposed stance it *would* take. Nobody has run the experiment:
add a known sustained gap to the cash rate, re-estimate, and see how much lands in the base
and how much in the residual. Until that is done, treat three to four years as the middle of
the range rather than as safely inside it.

For a judgement on the stance over a decade, the neutral rate still has to come from outside
the Bank's own behaviour, which is `rstar_bonds`.

**2. It conflates belief with every other systematic motive.** Anything that moved policy
persistently and was not inflation lands in the base and is reported as neutral. If the Bank
held rates up for financial-stability reasons in 2015-19, this model records a higher
neutral, not a second objective. So the object is really *the neutral rate implied by the
RBA's conduct, given that inflation is the only thing it responded to*.

**3. It inherits the RBA's errors.** If the Bank believed neutral was 1.5% real when it was
0.5%, the model reports 1.5% and reports it as neutral rather than as a mistake.

**4. What makes it a model rather than an identity is `sigma_r`.** Any rate path can be
written as trend plus response if the trend is free enough. The claim has content only
because neutral is *asserted* to move slowly. The revealed r\* is revealed at that
smoothness, and `sigma_r` is imposed.

None of this is a defect to be fixed. It is what the model is, and it is the reason its
comparative advantage is the opposite question from the other two: not whether the RBA was
right, but what its conduct implies it believed.

---

## Read this first

From the saved default run (`model_outputs/rstar_rba_trace.nc`):

| | 2026Q2 |
|---|---|
| `lambda`, per band-width | **0.303** [0.206, 0.398] |
| `lambda`, per percentage point of inflation | **0.61** (Taylor is 1.50) |
| `rho` | 0.869 [0.728, 0.994] |
| `sigma_u` | 0.775 |
| r\* nominal | **3.49** [3.02, 3.94] |
| **r\* real** (r\* − 2.5) | **0.99** |
| the base `b_t` alone | 3.01 |
| cash rate | 4.35 |
| `corr(base, cash rate)` | 0.89 |
| divergences | 0 of 8000, r_hat 1.0 throughout |

**`lambda` UNITS matter and have caused errors.** The gap is scaled by the band half-width,
so `lambda` is per band-width and the response per percentage point of inflation is
`lambda / 0.5`, twice `lambda`. Taylor's 1.5 per point is **0.75** in these units, not 1.5
and not 0.5. Quoting 0.303 next to Taylor's 1.5 makes the RBA look five times less
responsive than the estimate says.

**`lambda` is the durable result, and it is not about r\*.** It is an estimate of the RBA's
systematic response, it survives every respecification tried, and at 0.61 per point it is
well short of the Taylor principle: a one-point rise in inflation raises the *nominal* cash
rate by less than a point, so the *real* rate falls. Two caveats on reading that as a squib,
both established below: `lambda` is carrying the labour-market response as well as the
inflation one, and the residual is autocorrelated so its interval is too tight.

**The base is close to a smoothed cash rate**, `corr` 0.89. Everything below is about how
much of that is unavoidable.

---

## What identifies what

**`lambda`** comes from the covariation of the cash rate with the inflation gap, which over
1993-2026 correlates about 0.6 on the band-scaled measure. Inflation targeting worked, so
most deviations were small and most of what the cash rate did was not a response to them. A
reaction function is hardest to estimate precisely when the central bank is good at its job.
In practice `lambda` is identified largely by the two episodes where the gap was large,
2008 and 2022-24.

**The level of r\*** comes from `u` having mean zero, which asserts that policy averaged
neutral over the sample, conditional on inflation.

**The split between base and response** comes from `sigma_r`, imposed at 0.10. It is best
understood not as a smoothness prior but as a **frequency cutoff**: it decides how fast a
movement has to be before it counts as a response rather than a drift in neutral. At
`sigma_r` → 0 the base is flat and `lambda` absorbs everything, which is the conventional
Taylor setup with a fixed neutral. At a large `sigma_r` the base takes everything and
`lambda` goes to zero.

That reframing matters for comparisons. A conventional Taylor coefficient is fitted against
a **fixed** neutral, so any downward drift in true neutral has nowhere to go but the
inflation coefficient, which comes back inflated. Here the base takes that drift and
`lambda` is left with the cyclical response only. **0.61 and 1.5 are not the same object**,
and the gap between them overstates the difference.

An earlier `sigma_r` sweep, run before the base/r\* rename and before jumps, gave `lambda`
between 0.80 and 1.10 per point across `sigma_r` from 0 to 0.15, while the *base* today ran
1.41 to −0.32 real and the pre-COVID residual swung from −1.81 to +0.08. The levels have
since moved and that table is not reproduced here because it would be quoted as current.
The qualitative conclusion holds and is the one to carry: **quote `lambda`; do not quote the
level or the era residuals without a range across `sigma_r`.**

---

## The residual is autocorrelated, and that is the largest known defect

Lag-1 autocorrelation of the residual is **0.853**, lag-2 0.617. The likelihood assumes
independence, so it does not hold.

It is gradualism, not discontinuity, and the evidence separates the two cleanly. Excluding
the GFC and COVID barely moves it, 0.857 to 0.802. The calmest stretch in the sample,
1994-2007, has the *highest* subperiod reading at 0.831. Kurtosis is **+0.28** against 0 for
a Gaussian, so there are no fat tails and no jump signature. Persistence is everywhere, not
concentrated where the world turned over.

The missing term is interest-rate smoothing. Central banks move in sequences of small steps,
so the standard reaction function is partial-adjustment,
`r_t = phi·r_{t-1} + (1−phi)·(b_t + response) + e_t`, and it is not implemented.

**What it costs.** With residuals this correlated the information in 134 quarters is far
less than 134 independent observations. The standard AR(1) correction factor is
√((1+ρ)/(1−ρ)) ≈ 3.6, so `lambda`'s posterior sd of 0.051 is probably closer to 0.18. The
estimate survives; the precision does not, and every interval in these notes is too tight.

UNVERIFIED: that factor is the textbook approximation applied to a posterior sd, not a
re-estimation.

---

## Five things the exploration established

### 1. Inflation must be averaged, and the memory is a judgement not an estimate

The raw quarterly print is the wrong object: annualised it has an sd of 1.12 and correlates
0.09 with the cash rate. Fitted on it the model collapses completely, `sigma_u` → 0.045,
`lambda` straddling zero, and the base correlating **1.00** with the cash rate.

Averaging fixes that, but the memory length cannot be estimated. Sweeping the truncation:

| `max_lag` | `rho` | mean lag | `lambda` per pp | `sigma_u` |
|---|---|---|---|---|
| 4 | 0.869 | 1.72 | 0.61 | 0.786 |
| 6 | 0.882 | 2.50 | 0.71 | 0.776 |
| 8 | 0.893 | 3.26 | 0.82 | 0.768 |
| 12 | 0.865 | 4.10 | 0.96 | 0.770 |

`rho` does not adjust, the mean lag scales with the truncation at roughly `max_lag`/3, and
`sigma_u` is flat to three decimals, so the fit cannot tell these apart. Meanwhile `lambda`
runs 0.61 to 0.96 per point, the difference between well short of the Taylor principle and
close to it. `rho` = 0.87 implies an untruncated mean lag of 6.6 quarters, so every window
tried cuts the geometric tail and renormalises. **`max_lag` = 4 is a stated judgement**, and
`lambda` should be quoted as a range across it.

Free Dirichlet weights over 13 lags are not identified at all: every weight returns its
uniform prior mean. One shape parameter is recoverable from this data; thirteen are not.

### 2. The response is concave, not convex, and probably neither

Adding `lambda_2 · g·|g|`, `lambda_2` came back **negative in every specification tried**
and positive in none, and `sigma_u` never improved on the linear model:

| spec | `lambda_2` |
|---|---|
| unscaled, `max_lag` 12, walk | 0.053 [−0.330, 0.444] |
| unscaled, `max_lag` 12, constant base | −0.703 [−1.147, −0.317] |
| unscaled, `max_lag` 12, 1993-2019 | −0.029 [−0.602, 0.570] |
| unscaled, `max_lag` 4, walk | −0.210 [−0.345, −0.052] |
| band-scaled, `max_lag` 4, walk | −0.056 [−0.094, −0.018] |
| band-scaled, floor quarters excluded | −0.071 [−0.104, −0.036] |

**Scaling the gap by the band half-width is the right normalisation** because it puts the
pivot of the power function at the edge of the target band: inside the band a convex term
damps, outside it amplifies. Without it a convex response damps most of the sample.

**The floor is not the explanation.** Dropping the nine quarters at the effective lower
bound made `lambda_2` slightly *more* negative, not less. What remains is adjustment speed:
through 2022-23 the cash rate was free to move but was travelling four points from a
standing start, so the rate gap is small in exactly the quarters where the inflation gap is
largest. Partial adjustment is the fix, and it is not implemented.

**The charts mark the bound.** Time-series charts shade 2020Q1-2022Q1, the nine quarters at
or below 0.5, and the scatter circles them. The shading is drawn whether or not the run
excluded them and the footer says which, so a run that kept them cannot be mistaken for one
that did not. On the scatter those nine form a flat line at a rate gap near −1.25 spanning
inflation gaps from −1.3 to +1.1: whatever inflation did, the rate gap could not move. That
is what censoring looks like, and why those quarters cannot inform curvature.

### 3. The RBA's response halved after the GFC, suggestively

Fitting two coefficients split at 2008Q1, everything else pooled:

| | `lambda` per band-width | per pp | P(> 1 per pp) |
|---|---|---|---|
| pre-2008 | 0.638 [0.373, 0.923] | 1.28 | 0.83 |
| 2008 on | 0.273 [0.181, 0.368] | 0.55 | 0.00 |

`lambda_break` = −0.364, 90% [−0.612, −0.116]. Consistent with the international record of
central banks under-responding since the GFC.

**Read it as suggestive, not established.** The nominal P(break < 0) is 0.991, but on
residuals autocorrelated at 0.85 the standard errors are understated by roughly 3.6, which
would take it to something nearer 0.75. The pre-2008 coefficient is also the weak one, sd
0.147 against 0.050, because 1994-2007 had a mean inflation gap of 0.10 and so contributes
little identifying variation.

That last point has a corollary: **the pooled `lambda` is essentially the post-GFC number**,
0.303 against `lambda_late` of 0.273, not a midpoint. The shipped default has been reporting
the post-GFC response all along.

Run with `--lambda-split 2008Q1`; off by default.

### 4. Neutral does not jump in Australia, even when permitted to

The base is a Gaussian random walk, which cannot step. With `--jumps`, on by default, the
innovation is Student-t with `nu` = 3 imposed in quarters the economy moved abruptly, timed
off the absolute quarterly change in real seasonally adjusted GDP at the 95th percentile.

**This is permission, not a step.** At `nu` = 3 most of the mass is still near zero, so the
likelihood has to want the step. The local data says where one *may* be needed; the model
decides whether to take it. That is why the result carries information.

It barely takes it. `lambda` 0.305 → 0.303, `sigma_u` 0.786 → 0.775, r\* 3.48 → 3.49. At the
COVID quarters the base steps 0.08 instead of 0.04: a faster crawl, not a step.

**No detector flags the GFC, and that is correct.** Australia had no bank failures, the banks
stayed funded, and the mining boom held output up. It was mild here next to the US, and
nothing in the Australian data looks discontinuous in 2008-09:

| detector | GFC |
|---|---|
| \|GDP q/q\| | 2008Q4 at the 28th percentile, 2009Q1 at the 73rd |
| demeaned 4q GDP | worst quarter 2009Q3 at the 86th, never clearing 95 |
| world real rate | big move 2007Q4-2008Q1, a *year* before the Australian disinflation |

Reaching the GFC on any of them means dropping to the 75th-85th percentile, which flags a
fifth to a quarter of the sample. That is not a discontinuity rule, it is a licence to
wander.

**Which quarters are flagged matters more than how many.** A flag inside the 2022-24
inflation surge destroys the identification, because that episode is where `lambda` comes
from and a fat-tailed base absorbs the very movement `lambda` needs:

| source | flags | `lambda` | two-gaps corr |
|---|---|---|---|
| `gdp` (default) | stop at 2021Q4 | 0.303 | 0.61 |
| `gdp4` | 2022Q3, 2024Q2-Q3 | 0.162 [0.036, 0.312] | 0.42 |
| `world` | 2022Q2, 2022Q4 | 0.168 | 0.46 |

In both damaging cases `corr(base, cash rate)` rises to 0.94 and `lambda`'s ESS falls by a
factor of five. `gdp` is the default because its flags happen to avoid that window, which is
luck rather than design: any future detector must be checked against it.

### 5. The employment leg cannot be added, and the reason is instructive

The RBA's mandate is inflation **and** full employment, so the inflation-only rule omits a
statutory objective. Adding `lambda_u · (u − u*)`, with u\* from `ystar_ustar`:

| | `lambda_pi` | `lambda_u` | `sigma_u` |
|---|---|---|---|
| inflation only | 0.303 | — | 0.775 |
| two targets | −0.011 [−0.14, +0.15] | −1.228 [−1.52, −0.93] | 0.629 |

`lambda_u` is correctly signed, sharp, and improves the fit more than anything else tried.
But `lambda_pi` goes to zero and `rho` falls apart, 0.869 → 0.473 spanning 0.06 to 0.95, so
the lag weights stop being identified. Read literally it says the RBA ignored inflation,
which is not credible.

**The mechanism is circular, not merely collinear.** The two gaps correlate −0.67 in sample
and the two coefficients correlate +0.71 across draws, but the deeper problem is that the
unemployment gap is partly the *output* of the rate decision. Slack is the instrument by
which inflation is brought down, so a rule with both arguments counts one decision twice,
once as the reason and once as the means. `lambda_u` is therefore not a preference
parameter: it mixes the Bank cutting when the labour market is weak with the labour market
being weak because the Bank tightened. That is simultaneity, and no prior fixes it.

Weighting the two responses by hand would suppress the symptom and is just choosing both
coefficients; a tight prior on `lambda_pi` is the same choice in Bayesian dress.

**Two things survive the rejection.** First, the useful finding is about the base: it barely
moves, 3.01 → 3.06, with r\* 3.49 → 3.52. So the inflation-only rule was **not** parking the
employment response in neutral, which was the worry that prompted the test. Second, what it
*was* doing is loading both responses into `lambda`. **Quote `lambda` as the response to
inflation and the labour market together, never to inflation alone**, and discount the
"squibbed on Taylor" reading accordingly.

Run with `--employment`; off by default.

---

## This r\* cannot be used to test an IS curve

In this model the rate gap against the base *is* the inflation response by construction, so
an IS regression on it is a Phillips curve in disguise. The `rule` variant in `is_curve`
uses this model's **r\***, the complete estimate:

| variant | slope | t | R² |
|---|---|---|---|
| none (raw real cash) | +0.041 | +1.86 | 0.027 |
| bond-market r\* | +0.123 | +2.82 | 0.060 |
| **reaction-function r\*** | **−0.009** | **−0.19** | **0.000** |
| flat r\* | +0.041 | +1.86 | 0.027 |

Correctly signed and indistinguishable from zero: the least informative of the four, which
is what its construction predicts, since its rate gap is close to the rule's own residual
and carries only high-frequency timing.

**An earlier version of this table reported +0.201 with t = 4.53**, "the best fit and the
most wrongly-signed slope". That was built on the **base** rather than r\*, and it is
withdrawn. It is the clearest example of why the base/r\* distinction at the top of these
notes matters.

Note the two variants that do reach significance are both *positively* sloped, which is the
wrong sign for an IS curve. Nothing here supports one. The likely reason is the same
simultaneity as in point 5: the RBA raises rates when the gap is positive, so the reaction
function sits inside the regression and biases the slope upward.

---

## Limitations

1. **It measures revealed belief**, conflated with every other systematic motive, and the
   longer a departure from the rule lasts the more of it is absorbed into neutral, so it
   cannot audit the Bank over a decade. It can over a year or two. See "What revealed,
   imperfectly means"; "circular by construction" was the earlier wording and overstated it.
2. **The residual is autocorrelated at 0.85**, so every interval quoted is too tight, by
   roughly a factor of 3.6 on standard errors. The largest known defect.
3. **No partial adjustment**, which is the cause of 2 and the most likely single fix.
4. **`sigma_r` is imposed** and decides the split between base and response, hence the level
   and the era residuals.
5. **The memory length is a judgement**, not an estimate, and `lambda` moves 0.61 to 0.96
   per point across defensible truncations.
6. **`lambda` carries the labour-market response too**, per point 5 above. It is not a pure
   inflation coefficient.
7. **Realised inflation stands in for forecast inflation.** The RBA responds to forecasts;
   realised inflation puts measurement error in the regressor and attenuates `lambda`.
8. **The floor quarters are in by default.** `--floor 0.5` excludes them; it did not change
   any conclusion.
9. **The default depends on ABS GDP**, because jumps are on. `--no-jumps` returns the model
   to two series if that fetch ever breaks.
10. **The `stance` variable is misnamed.** It is the residual, not a stance; renaming it
    needs a re-sample and has been deferred to the next estimation run.

## Refinements

1. **Partial adjustment**, per Limitations 2 and 3. The most likely to change a conclusion,
   and it would probably cut `lambda` further, since part of what `lambda` explains is just
   last quarter's cash rate.
2. **Forecast inflation** in place of realised. Needs an RBA forecast series or a nowcast.
3. **Model the floor as censoring** rather than dropping it: the prescription is latent and
   what is observed is bounded below. The only approach that could still identify curvature
   from the 2020-21 episode.
4. **Compare the base against the RBA's own published neutral estimates.** The direct test
   of the "revealed belief" reading: close agreement means the model recovers a belief,
   systematic divergence means it is picking up other motives, and the sign says which.
   UNVERIFIED: whether those estimates exist as a downloadable series or only in chart packs
   and speeches.
5. **Reconcile the three routes**, still the most useful thing left to write in this corner
   of the repo.

---

## Files and usage

```
src/models/rstar_rba/
├── config.py      # ModelConfig: window, weights, band, floor, walk, jumps, employment, priors
├── estimate.py    # observations, lag matrix, weights, jump mask, model, sampling
├── analyse.py     # charts, prior-posterior plots, printed diagnostics
└── run.py         # CLI
```

```bash
./run-rstar-rba.sh -v                     # the default: walk, jumps on, floor kept
./run-rstar-rba.sh --no-jumps             # Gaussian base throughout, two series only
./run-rstar-rba.sh --jump-source world    # world real rate timing (breaks lambda)
./run-rstar-rba.sh --lambda-split 2008Q1  # two inflation coefficients
./run-rstar-rba.sh --employment           # add lambda_u (u - u*); rejected, see above
./run-rstar-rba.sh --no-walk              # constant base: the regression version
./run-rstar-rba.sh --nonlinear            # add lambda_2 g|g|
./run-rstar-rba.sh --max-lag 8            # the memory is a judgement: sweep it
./run-rstar-rba.sh --floor 0.5            # drop quarters at the effective lower bound
./run-rstar-rba.sh --sigma-r 0.05         # the setting that decides the split
```

Charts land in `charts/RStarRBA/`, a non-default prefix in its own directory:

- real and nominal r\* with 90% credible bands
- the cash rate, r\*, and the base
- what r\* is made of: the base and the inflation response on one axis
- how the cash rate responded to inflation (the two-gaps scatter)
- a Taylor rule on this model's r\*
- one prior-against-posterior chart per estimated parameter

Every chart carries the fitted equation in its header, read off the run rather than
hardcoded.

**Two charts were built and deliberately removed**, with the reasoning kept in comments in
`analyse.py` where the functions were. The residual chart, titled "How far the cash rate sat
from the RBA's estimated reaction function", read as a stance measure and is not one: the
base absorbs anything persistent, so the line cannot stay away from zero and the era means
are near zero by construction. And an earlier Taylor chart was built on the base, which is
the error these notes open with.

The surviving Taylor chart is on r\* and carries a caveat in its header: r\* already contains
the RBA's response to inflation and Taylor's rule adds a second, so the inflation gap is
counted twice and the prescription overstates whenever inflation is away from target. That
is a known bias in a correctly built chart, not a reason to substitute the base.
