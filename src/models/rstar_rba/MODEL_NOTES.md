# Neutral from the RBA's reaction to inflation

**What the model assumes, in one sentence: there is a neutral cash rate that moves slowly
over time, and on top of it the RBA reacts fast to inflation sitting above or below the 2.5%
target. The model splits the observed cash rate into those two pieces.**

The slow piece is neutral. The fast piece is the reaction function. Everything else follows,
including the model's limits, because **both speeds are asserted rather than estimated**:
`sigma_r` says how fast neutral is allowed to move, and having no smoothing term in the
observation equation says the reaction is immediate. It is a frequency decomposition of the
cash rate with an economic label on each band.

**It is not an estimate of the neutral rate. It is an estimate of what the RBA's behaviour
reveals about it, and it reveals it imperfectly.** That sentence governs everything below.

```
pi_t   = 4 · sum_{j=0}^{4} w_j · q_{t-j},   w_j ∝ rho^j,  sum w_j = 1
g_t    = (pi_t - 2.5) / 0.5                  the inflation gap, in band-widths
b_t    = b_{t-1} + sigma_r · e_t             NEUTRAL: the slow piece, sigma_r imposed
d_t    = b_t + lambda · g_t                  the rule's PRESCRIBED rate, not neutral
r_t    = d_t + u_t,   u_t ~ Normal(0, sigma_u)
```

**The response is to the deviation from 2.5, not to being outside the band.** Inflation of
2.7 is inside the 2-3 band and still produces a response, because `g_t` is linear in
`pi_t − 2.5`. The band half-width is only the *unit*: dividing by 0.5 makes `|g| = 1` at the
band edge, so `lambda` is per band-width and the response per percentage point is twice it.
There is no threshold at the band anywhere in the rule, and two were looked for and not
found: curvature pivoting at the band edge, and a no-response deadband around target. Both
tested and rejected under point 2.

`r_t` is the nominal cash rate and `q_t` quarterly trimmed-mean inflation. Sample
1993Q1-2026Q2, 134 quarters. The trace records `neutral`, `neutral_real`, `prescribed`,
`prescribed_real`, `stance` (cash less neutral) and `rule_residual` (cash less prescribed).

**Neutral is `b_t`, and the inflation response is a departure from it**, which is the
conventional reading of a policy rule: `i_t = i*_t + lambda(pi_t − pi*)` has the intercept as
neutral.

**One caveat on the real deflator.** Real neutral is reported as `b_t − 2.5`, which assumes
expected inflation is anchored at target. The equation has no `pi_t` term, so `lambda` is the
Fisher pass-through and the real response summed and the model cannot separate them. See
"The Taylor comparison does not work in nominal terms".

Putting `pi_t` on the right-hand side with a fixed unit coefficient would make `b_t` an
unambiguous *real* neutral and settle the deflator. That has not been done.

## One of three routes, all flawed

This repo contains three separate attempts at Australian r\*, and the useful thing is that
they fail differently. Read them together rather than picking one.

| package | identified from | what it actually measures | how it fails |
|---|---|---|---|
| [`rstar_hlw`](../rstar_hlw/MODEL_NOTES.md) | trend growth + the IS curve | the textbook definition: the rate at which output sits at potential | the IS curve does not identify anything on AU data, so each specification returns its own prior |
| [`rstar_bonds`](../rstar_bonds/MODEL_NOTES.md) | asset prices | what investors price | the *level* is not identified; it rests on the stationarity prior and an asserted premium |
| **`rstar_rba`** (this one) | the RBA's response to inflation | what the Bank's conduct implies it believed neutral was | a departure from the rule is absorbed into neutral as it lasts: measured, 0.78 of it survives at one year and 0.44 at four |

Only the first targets what the theory defines, and it is the one that cannot be estimated.
The other two measure *beliefs* about r\*, held by different people.

Current comparison, real: `rstar_hlw` Resolution G gives 2.23, `rstar_bonds` 1.08, this
**0.49** (−0.05 to 1.05 across `sigma_r`). Nobody has reconciled them, and the spread is now
wider than it looked. Until recently this package published 0.98, which sat next to
`rstar_bonds` and read as corroboration. That was an artefact of comparing `b + lambda·g`,
which contains a policy response, against two measures that estimate an intercept. On
like-for-like terms the three span two full points.

---

## What this model can and cannot say

It assumes the RBA set the cash rate as neutral plus a response to inflation, and recovers
neutral as the rate minus the response. Everything below follows from that.

### It assumes the Bank was competent

The level of neutral comes from the residual having mean zero, which says the Bank was right
**on average over the sample**. The injection test below says neutral absorbs anything
persistent, so the Bank is also assumed right **over any multi-year window**. Grant both and
the low-frequency component of the cash rate simply *is* neutral, by construction.

Which is what the data shows: `b_t` correlates **0.95** with a centred 13-quarter moving
average of the cash rate. A smoothed cash rate is not a rival estimate, it is what the
assumption entails.

**What the model adds to smoothing is an adjustment for inflation away from target**, and it
is systematic: the departure from the moving average correlates **−0.73** with the inflation
gap. It is largest in the episodes where inflation was furthest from 2.5, which happen also
to be the episodes outside the band, but the adjustment is proportional throughout and has no
threshold.

| | `b_t` less the 13q moving average | mean inflation |
|---|---|---|
| 1994-99 | −0.03 | 2.25 |
| 2000-07 | −0.24 | 2.78 |
| 2008-15 | −0.07 | 2.75 |
| **2016-19** | **+0.56** | 1.68 |
| **2020-21** | **+0.77** | 1.60 |
| **2022-26** | **−1.02** | 4.16 |

Near zero through the quiet decades; above the naive smooth when inflation ran below band,
because a low cash rate then reads as partly a response rather than a fall in neutral; below
it when inflation ran above band. **In quiet periods this model tells you the RBA did what the
RBA did. Its content is `lambda`, and the excursions.**

**The competence assumption is indirectly testable and broadly holds.** If the Bank had been
systematically too loose, inflation would have averaged above target. Over 1993-2026 it
averaged **2.64**, a gap of +0.14, so the assumption misses mildly loose and not by much.
Flipping today's +1.36 stance would need a uniform bias of 1.36 points sustained for three
decades, and an RBA that loose would not have delivered 2.64.

That qualifies one conclusion. The sign of the current stance is robust to `sigma_r`, positive
at every ensemble member, but **not** to this assumption, since a uniform bias shifts every
member together. It survives on the 2.64 figure, not on the ensemble.

### It sees short departures from the rule and progressively less of long ones

"Circular, the residual averages zero by construction" is too strong. Only the residual's
*whole-sample* mean is pinned, by the free starting level, and that is a normalisation. The
residual has an sd of 0.75 against 1.98 for the cash rate, and era means are not zero:
2020-21 at −0.73, 2012-15 at −0.15, 2022-on at +0.14.

The real limit is DURATION, measured rather than assumed. A known +1.00pp was added to the
cash rate over windows of several lengths, all ending 2019Q4, and the model re-estimated:

| stance lasts | reported as a departure | absorbed into neutral |
|---|---|---|
| 1 year | **0.78** | 0.22 |
| 2 years | **0.64** | 0.36 |
| 4 years | **0.44** | 0.54 |
| 6 years | **0.36** | 0.66 |
| 10 years | **0.25** | 0.78 |

No threshold and no safe zone, only a slope. **By four years it already reports less than
half.** The trap this invites is reading "how far neutral *could* wander under its prior" as
"how much of a stance it *would* absorb"; the likelihood decides the second.

A corollary. The residual through 2016-2019 is −0.01, which is not evidence policy sat at its
own rule: across `sigma_r` it runs −0.46, −0.01, +0.07, and over that four-year span a genuine
full point of easy policy would show up as only about −0.44 anyway. **The model cannot
separate "policy was neutral" from "policy was easy" over four years.**

Run with `--injection-test`, on by default. Three things about the measure. The reported
figure is the CONTRAST, within-window less outside-window, because `base_0` moves to absorb a
level shift whatever the duration; at four years the raw number is 0.39 and the contrast 0.44.
Recovered and absorbed sum to about one at every duration, as the identity requires, and
`lambda` barely moves across the set, so the stance is being fought over by neutral and the
residual rather than quietly rescaling the response. And inflation is held at what it actually
did, so this is not "what if the RBA had been tighter": real tightening would have pulled
inflation down and given the model an easier signal, which makes these shares a lower bound.
UNVERIFIED, and one placement only: every window ends 2019Q4, so the slope across durations is
robust but the level at each duration is that placement's.

### It conflates belief with everything else systematic

Anything that moved policy persistently and was not inflation lands in neutral. If the Bank
held rates up for financial-stability reasons in 2015-19, this records a higher neutral, not
a second objective. It also conflates belief with the reaction function itself changing, and
it inherits the Bank's errors: a Bank that believed neutral was 1.5% real when it was 0.5%
is reported at 1.5%, as neutral rather than as a mistake.

So the object is *the neutral rate implied by the RBA's conduct, given that inflation is the
only thing it responded to*. None of this is a defect to be fixed. It is what the model is,
and it is why its comparative advantage is the opposite question from the other two routes:
not whether the RBA was right, but what its conduct implies it believed. For a judgement on
the stance beyond a year or two, neutral has to come from outside the Bank's own behaviour,
which is `rstar_bonds`.

---

## Read this first

From the saved default run (`model_outputs/rstar_rba_trace.nc`), jumps off. The comparison
tables further down were sampled with jumps on and are not re-run; the difference is in the
third decimal, and point 4 gives it.

| | 2026Q2 |
|---|---|
| `lambda`, per band-width | **0.305** [0.213, 0.399] |
| `lambda`, per percentage point of inflation | **0.61**, 0.52-0.72 across defensible `sigma_r` |
| `rho` | 0.869 [0.727, 0.994] |
| `sigma_u` | 0.786 |
| **neutral `b_t`, nominal** | **2.99** [2.50, 3.46] |
| **neutral `b_t`, real** (less 2.5) | **0.49**, and **−0.05 to 1.05 across defensible `sigma_r`**: quote the range |
| the rule's prescribed rate, `b_t + lambda·g_t` | 3.48 |
| cash rate | 4.35, so the stance is **+1.36** |
| `corr(base, cash rate)` | 0.88 |
| divergences | 0 of 8000, r_hat 1.0 throughout |

**`lambda` UNITS matter and have caused errors.** The gap is scaled by the band half-width,
so `lambda` is per band-width and the response per percentage point of inflation is
`lambda / 0.5`, twice `lambda`. Taylor's 1.5 per point would be **0.75** in these units, not
1.5 and not 0.5, so quoting 0.303 beside 1.5 makes the RBA look five times less responsive
than the estimate says. That units trap is now the second reason not to put the two numbers
side by side; the first is that they are not comparable at all in nominal terms, per "The
Taylor comparison does not work in nominal terms".

**`lambda` is the durable result, and it is not about r\*.** It is an estimate of the RBA's
systematic response and it survives every respecification tried.

It does **not** follow that the RBA fell short of the Taylor principle. A sub-unit *nominal*
response lowers the real rate only if expected inflation moves one for one with actual, which
is an assumption the model neither makes nor tests. `lambda` is a nominal response; whether
the real one is positive is not established here either way. See "The Taylor comparison does
not work in nominal terms".

**Neutral is a smoothed cash rate, adjusted for bouts of above- and below-band inflation.**
`corr(b_t, cash)` is 0.88, and 0.95 against a 13-quarter moving average. The smoothing is the
identifying assumption showing through rather than a defect; the adjustment is what the model
adds. See "The model assumes the RBA was competent".

---

## What identifies what

**`lambda`** comes from the covariation of the cash rate with the inflation gap, which over
1993-2026 correlates about 0.6 on the band-scaled measure. Inflation targeting worked, so
most deviations were small and most of what the cash rate did was not a response to them. A
reaction function is hardest to estimate precisely when the central bank is good at its job.
In practice `lambda` is identified largely by the two episodes where the gap was large,
2008 and 2022-24.

**The level of neutral** comes from `u` having mean zero, which asserts that policy averaged
neutral over the sample, conditional on inflation.

**The split between base and response** comes from `sigma_r`, and **0.10 is an arbitrary
choice**. Not a calibration, not an estimate, and not the value the data prefers, because the
data does not prefer one: it is a round number inside a band of defensible values, picked so
the model has a default. Everything the package reports about the LEVEL of neutral is
conditional on it, which is why the ensemble runs by default and why the range is the
headline rather than the point. It is best
understood not as a smoothness prior but as a **frequency cutoff**: it decides how fast a
movement has to be before it counts as a response rather than a drift in neutral. Both limits
were checked, and `lambda` is stable across the whole span: 0.53 to 0.72 per point from a flat
base out to 0.20. It neither absorbs everything as `sigma_r` → 0 nor goes to zero at a large
one. What moves is the fit and the level, not the response.

### The fixed-neutral alternative is rejected by its own residual

The walk is usually defended as a prior, that neutral moves slowly. It does not have to be:
the fixed-neutral specification, `--no-walk`, fails a test it sets itself.

| | `lambda`/pp | `sigma_u` | resid sd | lag-1 ac | trend | 1994-99 | 2016-26 |
|---|---|---|---|---|---|---|---|
| default walk | 0.61 | 0.786 | 0.75 | 0.857 | −0.04/decade | +0.16 | −0.08 |
| **flat base** | 0.53 | 1.930 | 1.91 | **0.974** | **−1.54/decade** | **+2.05** | **−1.99** |

The likelihood asserts the residual is iid Normal(0, `sigma_u`). Under a flat base it is a
trending, near-unit-root series: era means march from +2.05 to −1.99 and the residual sd of
1.91 is essentially the cash rate's own 1.98, so the model explains almost nothing. **That is
a direct contradiction of the model's own assumption, not a fit-versus-flexibility argument**,
and it holds however many parameters you are willing to spend.

The economics is plain. The cash rate fell about three points over the sample while inflation
spent most of it inside the band, so no coefficient on the inflation gap can track that drift.
Something trend-like is needed and **the data says so rather than the modeller assuming it.**

Two things this does not establish. It does not vindicate `sigma_r` = 0.10 or any other
value: the walk's residual is still autocorrelated at 0.857, the same violation in milder
form, and the missing partial adjustment is why. And it does not discriminate within the
ensemble, since all four members walk. What it closes off is the "just run a Taylor
regression" alternative, which is the specification most readers would reach for.

UNVERIFIED: no LOO or WAIC comparison has been run. The rejection above rests on residual
diagnostics, which is the stronger ground here anyway, since a flexible state will always
win a raw fit comparison.

---

That reframing matters for comparisons. A conventional Taylor coefficient is fitted against
a **fixed** neutral, so any downward drift in true neutral has nowhere to go but the
inflation coefficient, which comes back inflated. Here the base takes that drift and
`lambda` is left with the cyclical response only. **0.61 and 1.5 are not the same object**,
and the gap between them overstates the difference.

### The Taylor comparison does not work in nominal terms

The rule has no `pi_t` term. Inflation enters exactly once, through `g_t`, so the estimated
coefficient has to contain both the Fisher pass-through and the real response. Write `theta`
for how much of an inflation deviation passes into *expected* inflation:

```
r  =  b + pi^e + lambda_real·(pi − 2.5)
   =  [b + 2.5]  +  (theta + lambda_real)·(pi − 2.5)
```

so **`lambda` = 0.61 per point is `theta + lambda_real`**, one number doing two jobs, and the
model cannot separate them because `pi` appears only once on the right-hand side. At the
anchored corner (`theta` = 0) the real response is +0.61; at the unanchored corner
(`theta` = 1) it is −0.39. Same estimate, opposite sign on the thing that matters.

**Taylor's 1.5 is the threshold only at `theta` = 1**, since that is the case where a nominal
response of one merely holds the real rate still. Quoting 0.61 against 1.5 therefore asserts
full pass-through without saying so.

**So the direction of the real response is unknown here, and neither sign is asserted.**
Estimating `theta` needs an expectations series, which is a different model's output and is
not imported: the rule is deliberately answerable from two published series. What follows is
that **`lambda` is a nominal response and should be quoted as one**, with no claim about the
Taylor principle in either direction. Anyone wanting the real reading has to supply a
pass-through assumption and state it.

**The `sigma_r` ensemble.** Re-estimated at each value, everything else the default. Runs by
default.

| `sigma_r` | `lambda` per band-width | per pp | neutral nominal | **neutral real** | 2016-19 residual | `corr(base, cash)` |
|---|---|---|---|---|---|---|
| 0.05 | 0.358 | 0.72 | 2.45 | **−0.05** | −0.46 | 0.80 |
| **0.10** (default) | 0.305 | 0.61 | 2.99 | **0.49** | −0.01 | 0.88 |
| 0.15 | 0.262 | 0.52 | 3.55 | **1.05** | +0.07 | 0.93 |
| 0.20 | 0.243 | 0.49 | 3.75 | **1.25** | +0.04 | **0.96** |

**0.20 is the boundary, not a fourth defensible value.** At `corr(base, cash rate)` = 0.96 the
base is very nearly the cash rate smoothed, falling to 0.3 in 2021 against a cash rate of
0.10. Two symptoms confirm it: neutral gains only 0.20 on that step against 0.56 on the one
before, and the 2016-19 residual stops moving monotonically because little residual is left
anywhere. **Quote the range over 0.05 to 0.15**; 0.20 is carried to show where the method
fails.

Read the rest in three parts.

*`lambda` moves, but not enough to change what it says.* 0.72 to 0.52 per point over the
defensible range, 0.49 out at the boundary, straddling the headline 0.61 and staying well
short of Taylor's 1.50 everywhere including at the degenerate end. It moves in the direction
the frequency-cutoff reading predicts: a smoother base leaves more for `lambda`. **This is
the result that survives the conditioning**, and the range to quote is 0.52 to 0.72 rather
than 0.61 alone.

*The level does not survive it.* Real neutral runs **−0.05 to 1.05** over the defensible
range, a spread of 1.10 points against a 90% credible interval of 0.96 at the default, and
1.25 at the boundary. **The structural uncertainty is larger than the statistical
uncertainty**, so reporting the band alone understates the uncertainty on the level by more
than half. Quote the range. Note it spans zero: whether Australian real neutral is currently
positive is not settled by this model.

*The era residuals do not survive it either*, which is what withdrew the 2016-19 result
above. Nothing in the era table should be quoted without this column beside it.

*Neutral is more sensitive to `sigma_r` than the prescribed rate is*, 1.10 points against
0.95. What `sigma_r` takes from the base it hands to the response, `lambda` rising from 0.52
to 0.72 per point as the base is tightened, so the two partly cancel in the sum. Neutral,
carrying none of that offset, takes the full swing. A practical consequence: the level this
package now publishes is the more assumption-sensitive of the two lines, not the less.

`corr(base, cash rate)` rising 0.80, 0.88, 0.93, 0.96 is the failure mode arriving: at a
large `sigma_r` the base is just the cash rate smoothed and `lambda` becomes decoration. At
0.15 it is not there yet; at 0.20 it effectively is, which is what makes that member a
boundary rather than a candidate. It is the reason not to read the top of the range as
equally defensible with the bottom. The base chart shows it plainly: at 0.15 the line follows
the cash rate through 2007-08 and down into the 2020-21 trough, at 0.20 it very nearly
reproduces it, while at 0.05 it is a smooth trend through both.

**Quote `lambda` as a range; quote the level as a range; do not quote the era residuals
without one.**


## The residual is autocorrelated, and that is the largest known defect

Lag-1 autocorrelation of the residual is **0.857**, lag-2 0.627. The likelihood assumes
independence, so it does not hold.

It is gradualism, not discontinuity, and the evidence separates the two cleanly. Excluding
the GFC and COVID barely moves it, 0.857 to 0.818. The calmest stretch in the sample,
1994-2007, has the *highest* subperiod reading at 0.831. Kurtosis is **+0.18** against 0 for
a Gaussian, so there are no fat tails and no jump signature. Persistence is everywhere, not
concentrated where the world turned over.

The missing term is interest-rate smoothing. Central banks move in sequences of small steps,
so the standard reaction function is partial-adjustment,
`r_t = phi·r_{t-1} + (1−phi)·(b_t + response) + e_t`, and it is not implemented.

**What it costs.** With residuals this correlated the information in 134 quarters is far
less than 134 independent observations. The standard AR(1) correction factor is
√((1+ρ)/(1−ρ)) ≈ 3.6, so `lambda`'s posterior sd of 0.050 is probably closer to 0.18. The
estimate survives; the precision does not, and every interval quoted here is too tight.

UNVERIFIED: that factor is the textbook approximation applied to a posterior sd, not a
re-estimation.

---

## What the exploration established

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
runs 0.61 to 0.96 per point, a range wide enough to matter for anything built on it. `rho` = 0.87 implies an untruncated mean lag of 6.6 quarters, so every window
tried cuts the geometric tail and renormalises. **`max_lag` = 4 is a stated judgement**, and
`lambda` should be quoted as a range across it.

Free Dirichlet weights over 13 lags are not identified at all: every weight returns its
uniform prior mean. One shape parameter is recoverable from this data; thirteen are not.

**Whether the memory is constant cannot be tested here.** Splitting `rho` at 2008Q1 moves
`sigma_u` from 0.786 to 0.787 and returns `rho_early` at 0.575 [0.24, 0.87], an interval
spanning nearly the whole admissible range and overlapping the late one. The early sample
cannot adjudicate, which is the same weakness that gives `lambda_early` an sd three times
`lambda_late`'s. Not exposed as a flag, since the point estimates would be read as a result.

### 2. The response is linear: no curvature and no deadband

Adding `lambda_2 · g·|g|`, `lambda_2` came back **negative in every specification tried** and
positive in none, across six variants of scaling, `max_lag`, sample and floor treatment, and
`sigma_u` never improved on the linear model. The band-scaled default gives −0.056
[−0.094, −0.018]. Run with `--nonlinear`.

**There is no deadband either.** The rule responds to any deviation from 2.5, so inflation of
2.7 draws a response even though it is inside the band. Tested by replacing the response with
`lambda · sgn(g) · max(|g| − d, 0)`:

| deadband each side of 2.5 | `lambda`/pp | `sigma_u` |
|---|---|---|
| **0.00pp** (shipped) | 0.61 | **0.786** |
| 0.125pp | 0.63 | 0.791 |
| 0.25pp | 0.63 | 0.800 |
| 0.50pp | 0.63 | 0.816 |

Imposing one makes the fit monotonically worse, and estimated freely `d` comes back at
**0.072pp [0.004, 0.199]**, excluding 0.25. The test has power: 41 of 134 quarters sit within
0.25pp of target and 73 within 0.5pp, so a third of the sample is in the region where a
deadband would bite. A deadband is also a form of convexity, which is the opposite sign to
`lambda_2` above, so the two shape tests agree.

**The floor is not the explanation.** Dropping the nine quarters at the effective lower bound
makes `lambda_2` slightly *more* negative, not less. What remains is adjustment speed:
through 2022-23 the cash rate was free to move but was travelling four points from a standing
start, so the rate gap is small in exactly the quarters where the inflation gap is largest.

**The charts mark the bound.** Time-series charts shade the nine quarters at or below 0.5 and
the scatter circles them, whether or not the run excluded them, with the footer saying which.
On the scatter those nine form a flat line at a rate gap near −1.25 spanning inflation gaps
from −1.3 to +1.1: whatever inflation did, the rate gap could not move. That is censoring, and
why those quarters cannot inform curvature.

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

**A second reason to read it as suggestive, independent of the standard errors.** The
equation is contemporaneous, so `lambda` is a same-quarter response, and a central bank that
moves fast shows a larger same-quarter coefficient than one that moves in a sequence of small
steps even when the ultimate response is identical. The adjustment speed did change: inertia
rises from about 0.91 before 2016 to 0.975 after, per point 4a. **`lambda_early` against
`lambda_late` may therefore be measuring speed rather than strength**, and this model
separates the two nowhere. Partial adjustment would, which raises a difficulty for that
refinement in turn: if `phi` itself moved, a constant `phi` is misspecified too.

**One thing the split does survive.** The worry that the two coefficients are fitted against
different inflation objects, because the memory might have changed too, was tested by
splitting `rho` at the same quarter (point 1). `lambda_early` falls from 0.638 to 0.576 once
`rho` is free as well, 1.28 to 1.15 per point, while `lambda_late` is unchanged at 0.273. The
ratio goes from 2.3 to 2.1. **The halving is not an artefact of the two eras being fed
different inflation measures.** It remains exposed to the standard-error and speed
objections above.

That last point has a corollary: **the pooled `lambda` is essentially the post-GFC number**,
0.303 against `lambda_late` of 0.273, not a midpoint. The shipped default has been reporting
the post-GFC response all along.

Run with `--lambda-split 2008Q1`; off by default.

### 4. Neutral does not jump in Australia, even when permitted to

The base is a Gaussian random walk, which cannot step. With `--jumps`, a sensitivity test
and **off by default**, the innovation is Student-t with `nu` = 3 imposed in quarters the
economy moved abruptly, timed off the absolute quarterly change in real seasonally adjusted
GDP at the 95th percentile.

**This is permission, not a step.** At `nu` = 3 most of the mass is still near zero, so the
likelihood has to want the step, which is why the result carries information. It barely takes
it: `lambda` 0.305 → 0.303, `sigma_u` 0.786 → 0.775. At the COVID quarters neutral steps 0.08
instead of 0.04, a faster crawl rather than a step.

**That is why jumps are off by default.** A permission the model declines changes no number
worth quoting, and switching it on costs the ABS GDP dependency. The comparison tables further
down were run with jumps on; the difference is in the third decimal.

**No detector flags the GFC, and that is correct.** Australia had no bank failures and the
mining boom held output up, and nothing in the Australian data looks discontinuous in 2008-09.
Reaching it on any detector means dropping to the 75th-85th percentile, which flags a fifth of
the sample: not a discontinuity rule but a licence to wander.

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

**The detector does not flag the episode that would have tested it.** The `gdp` flags are
1995Q3, 1997Q2 and the five COVID quarters. The 1990s episode that matters is **1994Q4**,
when the cash rate went 4.75 to 7.50 in two quarters, and it is not flagged. So "neutral
does not jump in Australia" was established by offering the permission in quarters where the
case for a step was weak and declining it there. It is evidence, but less than it reads: the
test was never run where a step was most arguable. See below.

### 4a. The 1994 tightening is where `sigma_r` does its heaviest work

| | cash rate | trimmed-mean inflation | gap |
|---|---|---|---|
| 1994Q2 | 4.75 | 2.2 | −0.3 |
| 1994Q4 | **7.50** | 2.2 | −0.3 |
| 1995Q4 | 7.50 | 3.1 | +0.6 |

The RBA tightened 275bp while realised inflation sat slightly **below** target. It was aimed
at a forecast, and the inflation did arrive, reaching +0.6 through 1995. The rule as written
cannot see any of that: it has the largest pre-GFC rate rise in the sample beside a negative
inflation gap, so the whole move must go to the base or the residual, and `sigma_r` alone
decides which. Across the ensemble the base rises by **0.00, 0.25 and 0.79 points** from
1994Q2 to 1995Q4. At 0.05 the model says policy was tight; at 0.15 it says neutral rose most
of a point. That is the widest disagreement anywhere in the sample and it is visible at the
left edge of the base chart.

**The adjustment speed is not constant either**, which compounds it. AR(1) inertia in the
cash rate runs 0.904, 0.912, 0.916 and 0.975 across 1993-99, 2000-07, 2008-15 and 2016-26,
and persistence of the quarterly change runs 0.34, 0.48, 0.32 and 0.69. The first three eras
are indistinguishable and the recent decade is markedly more gradual, and it is not the lower
bound doing it: excluding 2020-21 the recent figure is still 0.96. The model assumes one
speed throughout.

### 5. The employment leg cannot be added as a level, and the reason is instructive

The RBA's mandate is inflation **and** full employment, so the inflation-only rule omits a
statutory objective. Adding `lambda_u · (u − u*)`, with u\* from `ystar_ustar`:

| | `lambda_pi` | `lambda_u` | `sigma_u` |
|---|---|---|---|
| inflation only | 0.303 | — | 0.775 |
| two targets | −0.011 [−0.14, +0.15] | −1.228 [−1.52, −0.93] | 0.629 |

`lambda_u` is correctly signed, sharp, and improves the fit more than anything else tried.
But `lambda_pi` goes to zero and `rho` falls apart, spanning 0.06 to 0.95, so the lag weights
stop being identified. Read literally it says the RBA ignored inflation, which is not
credible.

**The mechanism is circular, not merely collinear.** The unemployment gap is partly the
*output* of the rate decision: slack is the instrument by which inflation is brought down, so
a rule with both arguments counts one decision twice, once as the reason and once as the
means. `lambda_u` mixes the Bank cutting when the labour market is weak with the labour market
being weak because the Bank tightened. No prior fixes that, and weighting the two responses by
hand is just choosing both coefficients.

**Two things survive the rejection.** Neutral barely moves, 3.01 → 3.06, so the inflation-only
rule was **not** parking the employment response in neutral, which was the worry that prompted
the test. What it *was* doing is loading both responses into `lambda`. **Quote `lambda` as the
response to inflation and the labour market together, never to inflation alone.**

**What was rejected is the LEVEL specification, not the labour market.** `u − u*` is a level,
and on this sample the level gets the 1990s backwards: unemployment in 1994 was 10.3% against
a long-run `u*` nearer 5.5, so a level term prescribes easing in the quarters the RBA
tightened 275bp. The *change* says the opposite, and it is the change that carries the
information: in the forecasting check under Refinement 1, the level of `u` adds essentially
nothing to an AR (rmse 1.17 against 1.20) while the four-quarter change is the strongest term
tried (0.86). The RBA in 1994 was responding to the speed of improvement, not the amount of
slack.

So the honest statement is that a level-based second objective is rejected, and a
change-based labour term has not been tried in the rule. It would still face the
simultaneity problem above, which is why it is not simply the obvious next step. The repo
already carries the concept as `get_unemployment_speed_limit_qrtly()`, used in wage equations.

Run with `--employment`; off by default.

### 6. Partial adjustment works, and costs more than it buys

The fix for the autocorrelation: `r_t = phi·r_{t-1} + (1 − phi)·(b_t + lambda·g_t) + eps_t`,
so the rule's rate becomes what the Bank moves *toward* and it closes `(1 − phi)` of the gap
each quarter. Built and tested. `--partial-adjustment`, off by default.

| `phi` | `lambda`/pp | `sigma_u` | residual ac1 | neutral | prescribed 2022-26 |
|---|---|---|---|---|---|
| 0.00 (default) | **0.61** | 0.790 | 0.857 | 2.99 | 3.39 |
| 0.50 | 0.72 | 0.518 | 0.703 | 2.77 | 3.55 |
| 0.70 | 0.87 | 0.448 | 0.566 | 2.57 | 3.79 |
| 0.85 | 1.19 | 0.422 | 0.458 | 2.62 | 4.56 |
| free → 0.953 | **2.57** [1.14, 4.44] | 0.424 | 0.414 | 3.20 | **7.30** |

It halves the autocorrelation, 0.857 to 0.414, and cuts `sigma_u` nearly in half. It does not
eliminate either, so the persistence is not only adjustment lag.

**The cost is that it destroys the one durable result.** `lambda` is 0.52 to 0.72 per point
across the whole `sigma_r` ensemble; across `phi` it runs 0.61 to 2.57. It stops being a range
and becomes whatever `phi` says. `phi` would have to be imposed exactly as `sigma_r` is,
leaving two arbitrary smoothness parameters instead of one.

**And the free estimate is not usable.** `phi` = 0.953 implies a 3.6-year half-life, which is
not how a bank moving in 25bp steps behaves; the fit does not support it either, since
`sigma_u` saturates around `phi` ≈ 0.85 and is fractionally worse at 0.953. It produces a
prescribed rate averaging 7.30 through 2022-26 with the Bank persistently 3.8 points below it,
and `phi·r_{t-1}` displaces neutral, `corr` with the cash rate falling to 0.759 and neutral
going nearly flat after 2012.

`lambda` also changes meaning: with `phi` free it is the LONG-RUN response, in the default the
same-quarter one. The two are not comparable.

---

## 2016-2019: the rule itself was too tight

A Taylor rule on this model's own neutral, less the actual cash rate. Negative means policy
tighter than Taylor wants:

| | mean | range | core inflation | output gap |
|---|---|---|---|---|
| 2012-15 | −0.17 | −0.85 to +0.60 | 2.32 | −0.37 |
| **2016-19** | **−1.03** | **−1.32 to −0.80** | 1.58 | −0.33 |
| 2022-26 | +0.90 | −0.47 to +2.38 | 3.66 | +0.62 |

The 2016-19 gap is negative in **every quarter of the four years** and survives the smoothness
choice: −0.49, −1.03, −1.18, −1.18 across the ensemble. Almost nothing else in this package is
robust to `sigma_r`; this is.

**The contrast with the model's own residual is the finding.** Over the same window that
residual is **−0.01**, so policy was entirely typical for this RBA. Taylor on the same neutral
says it was a full point too tight. The model is not reporting a deviation from the rule, it
is reporting that **the rule was too tight**. The arithmetic is clean: core inflation averaged
0.92 below target, Taylor responds 1.5 per point and wants 1.38 lower, the RBA's revealed 0.61
per point wants only 0.56 lower, a difference of 0.82, plus Taylor's output-gap term at 0.5 ×
−0.33 ≈ −0.17. Together ≈ −0.99 of the −1.03.

**It is symmetric**, which is what a response coefficient below Taylor's implies: too little
movement in both directions. −1.03 in 2016-19 with inflation at 1.58, +0.90 since 2022 with
inflation at 3.66.

**The outcome data makes the point without any rule.** Trimmed-mean inflation was below the
2-3 band in **16 quarters out of 16** across 2016-2019, averaging 1.67 and never exceeding
1.81, with a negative output gap alongside. The four years before averaged 2.29 with one
quarter below 2. Four consecutive years wholly outside the target on the low side is the
mandate not being met, and the model corroborates rather than carries that.

**−1.03 is a floor, not a central estimate.** The two obvious objections both bias it the same
way. If policy really was persistently tight, the base absorbed part of it and sits too high,
raising the Taylor prescription and understating the gap. And if the Bank was holding rates up
for financial-stability reasons, this model books that as a higher neutral rather than a second
objective (see "It conflates belief with everything else systematic"), which again raises it. The
main contemporary defence of the stance, if true, makes the measured undercooking larger.

**One limit.** "Should have been lower" needs the counterfactual that lower rates would have
lifted inflation, and this repo's central negative finding is that the rate-to-activity link is
not identifiable on Australian data (see [`is_curve`](../is_curve/MODEL_NOTES.md)). The model
can say policy was inconsistent with the target; it cannot say how much inflation a cut would
have bought.

This comparison is **nominal against nominal**, so it is unaffected by the Fisher confounding
that withdrew the "squibbed the Taylor principle" reading. `pi_core` and `ygap` come from
`ystar_ustar` and the 0.5/0.5 weights are imposed.

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
most wrongly-signed slope". That was built on the **base** while the row was labelled r\*,
so the table did not report what it said it did, and it is withdrawn on that ground. The two
quantities give materially different answers here, +0.201 against −0.009, which is the
sharpest illustration of why a number from this model has to say which line it came from.

Note the two variants that do reach significance are both *positively* sloped, which is the
wrong sign for an IS curve. Nothing here supports one. The likely reason is the same
simultaneity as in point 5: the RBA raises rates when the gap is positive, so the reaction
function sits inside the regression and biases the slope upward.

---

## Observations

What follows changes how a number from this model should be read. Some of it is
conditioning the model was built with and some of it is a defect waiting to be fixed, and
the entries say which.

1. **It measures revealed belief**, conflated with every other systematic motive, and the
   longer a departure from the rule lasts the more of it is absorbed into neutral. Measured
   by the injection test: 0.78 of a known stance survives at one year, 0.64 at two, 0.44 at
   four, 0.25 at ten. **It can audit the Bank over one to two years and not beyond.** See
   "What this model can and cannot say".
1a. **The reaction function is assumed constant over 33 years, and there is reason to doubt
   it.** What is held fixed: `lambda`, `rho`, `sigma_u`, the objective set, and the
   adjustment speed, which is not a parameter at all because the equation is
   contemporaneous. The model's own split test says `lambda` halved after 2008; inertia in
   the cash rate rises from about 0.91 before 2016 to 0.975 after; and 1994 shows the Bank
   acting on a forecast in a way the rule cannot represent.

   **The consequence depends on whether the changing piece has a parameter.** Where it does,
   the model uses it and leaves neutral alone: allowing `lambda` to break at 2008 moves the
   base by at most 0.18 in any quarter, era means by at most 0.13, and the endpoint from
   2.99 to 3.08. Where it does not, there is exactly one time-varying object left, `b_t`, so
   **a change in the rule is reported as a change in neutral**. Adjustment speed and
   forward-looking-ness both fall in the second group. The 1994 episode is the worked
   example, and the base takes 0.00 to 0.79 points of it depending on `sigma_r`.

   This is a sharper form of point 1. It is not only that belief is conflated with other
   motives; belief is conflated with the rule itself changing. And per point 1 of "What the
   exploration established", the early sample cannot test the constancy either way.
2. **The residual is autocorrelated at 0.85**, so every interval quoted is too tight, by
   roughly a factor of 3.6 on standard errors. The largest known defect.
3. **No partial adjustment**, which is the cause of 2. Built and tested, and rejected as a
   default: it halves the autocorrelation but rescales `lambda` from 0.61 to 2.57 with a
   freely estimated `phi`. See point 6.
4. **`sigma_r` is imposed and arbitrary**, and decides the split between neutral and the
   response, hence the level. Quantified rather than asserted: across the defensible 0.05 to
   0.15 real neutral runs −0.05 to 1.05, wider than the credible interval, and the 2016-19
   residual changes sign. The ensemble runs by default.
5. **The memory length is a judgement**, not an estimate, and `lambda` moves 0.61 to 0.96
   per point across defensible truncations.
6. **`lambda` carries the labour-market response too**, per "What the exploration
   established", point 5. It is not a pure inflation coefficient.
7. **Realised inflation stands in for forecast inflation, and in the 1990s that is not a
   small thing.** The RBA responds to forecasts; realised inflation puts measurement error
   in the regressor and attenuates `lambda`. The scale of it: the 1994 tightening of 275bp
   happened with the inflation gap at −0.3, so the single largest pre-GFC policy move in the
   sample is **entirely invisible to the rule** and is absorbed by `sigma_r`'s choice
   instead. A first-order problem for the first half of the sample, not a mild attenuation.
   See point 4a.
8. **The floor quarters are in by default.** `--floor 0.5` excludes them; it did not change
   any conclusion.
9. ~~**The default depends on ABS GDP**, because jumps are on.~~ Resolved: jumps are off by
   default, so the headline needs the cash rate and trimmed-mean inflation and nothing else.
   `--jumps` reintroduces the GDP dependency for the sensitivity test.
10. ~~**The `stance` variable is misnamed.**~~ Resolved. `stance` is now the cash rate less
    neutral, which is what the word means, and `rule_residual` is the cash rate less the
    rule's prescribed rate, which is what the old `stance` actually held.

## Refinements

1. **Forecast inflation** in place of realised, per Observation 7 and point 4a. The largest
   open item, and now buildable with a design rather than blocked on data.

   **No RBA forecast series is held**, and none of the loaders provides one, so the forecast
   has to be constructed, and a check settles what it needs. Refitting a regression each
   quarter on an expanding window, so the coefficients never see the outcomes they forecast,
   predicting annualised trimmed-mean inflation four quarters ahead, and asking what it said
   through 1994:

   | activity term | 1994 forecasts, Q1-Q4 | in-sample rmse |
   |---|---|---|
   | none, AR only | 2.55 2.67 2.72 2.44 | 1.20 |
   | `u` level | 2.40 2.66 2.81 2.66 | 1.17 |
   | `du`, 1q | 3.43 3.55 3.22 2.94 | 0.99 |
   | **`du`, 4q** | 3.34 3.55 **3.95 3.85** | **0.86** |

   Actual outturns were 2.20, 2.50, 2.90, 3.10. **An AR alone is useless**: it forecasts
   target throughout 1994, so the episode stays as invisible as it is with realised
   inflation. Extrapolating inflation from inflation cannot see a turning point. **The
   four-quarter change in unemployment transforms it**, forecasting 3.3 to 4.0 in exactly the
   quarters the RBA tightened.

   **It does not fully close the puzzle.** Averaged over 1994 the forecast gap is about
   +0.79pp, prescribing roughly +0.48 of cash rate at `lambda` = 0.61, against −0.18 on
   realised inflation. A swing of 0.66 where the RBA moved 275bp: it flips the sign of the
   problem and explains perhaps a quarter of the move.

   **Two constraints on the build.** The activity term must not be inflation-derived, which
   rules out the `ystar` and `ystar_ustar` output gaps, since `ystar` defines the gap as
   `c·(pi − anchor)` and subtracting `y*` from GDP returns the same thing plus a residual.
   The change in unemployment needs no other model. And it reintroduces the endogeneity that
   sank the employment leg, weaker here because the term enters as a forecast input rather
   than a second objective, but the same mechanism.
2. **A joint `phi`/`sigma_r` surface**, per point 6. Partial adjustment on its own is built
   and rejected because a free `phi` rescales `lambda` fourfold. Exploring the two imposed
   parameters together is the honest form of the same idea, and `ensemble.py` is the
   machinery for it.
3. **Replicate the injection test at a second placement.** Every window currently ends
   2019Q4, and neutral is least constrained near the sample ends, so the slope across
   durations is established but the level at each duration is not.
4. **Model the floor as censoring** rather than dropping it: the prescription is latent and
   what is observed is bounded below. The only approach that could still identify curvature
   from the 2020-21 episode.
5. **Compare the base against the RBA's own published neutral estimates.** The direct test
   of the "revealed belief" reading: close agreement means the model recovers a belief,
   systematic divergence means it is picking up other motives, and the sign says which.
   UNVERIFIED: whether those estimates exist as a downloadable series or only in chart packs
   and speeches.
6. **Reconcile the three routes**, still the most useful thing left to write in this corner
   of the repo.

---

## Files and usage

```
src/models/rstar_rba/
├── config.py      # ModelConfig: window, weights, band, floor, walk, jumps, employment, priors
├── estimate.py    # observations, lag matrix, weights, jump mask, model, sampling
├── ensemble.py    # re-estimation across sigma_r: the imposed smoothness as a range
├── injection.py   # add a known stance, re-estimate, see how much comes back
├── analyse.py     # charts, prior-posterior plots, printed diagnostics
└── run.py         # CLI
```

```bash
./run-rstar-rba.sh -v                     # the default: estimate + ensemble + injection test,
                                          #   walk, Gaussian base, floor kept. ~38 seconds
./run-rstar-rba.sh --no-sigma-r-ensemble --no-injection-test   # the estimate alone, ~8 seconds
./run-rstar-rba.sh --jumps                # permit base steps at abrupt GDP quarters
./run-rstar-rba.sh --jump-source world    # world real rate timing (breaks lambda)
./run-rstar-rba.sh --lambda-split 2008Q1  # two inflation coefficients
./run-rstar-rba.sh --employment           # add lambda_u (u - u*); rejected, see above
./run-rstar-rba.sh --no-walk              # constant neutral: the regression version, rejected
./run-rstar-rba.sh --partial-adjustment   # interest-rate smoothing; rescales lambda, see above
./run-rstar-rba.sh --nonlinear            # add lambda_2 g|g|
./run-rstar-rba.sh --max-lag 8            # the memory is a judgement: sweep it
./run-rstar-rba.sh --floor 0.5            # drop quarters at the effective lower bound
./run-rstar-rba.sh --sigma-r 0.05         # the setting that decides the split
./run-rstar-rba.sh --sigma-r-ensemble 0.05,0.075,0.10,0.125,0.15   # override the values
./run-rstar-rba.sh --injection-test 3,5    # override the durations, in years
```

Charts land in `charts/RStarRBA/`, a non-default prefix in its own directory:

- real and nominal neutral `b_t` with 90% credible bands
- how much of r\* is the imposed smoothness: the `sigma_r` envelope against that band, drawn
  only when `--sigma-r-ensemble` has been run. **The chart to quote for the level**, because
  the one above it shows only the conditional uncertainty
- the base across the same `sigma_r` values, against the cash rate. Where the r\* envelope
  shows the spread on the published number, this shows where it comes from, and it makes the
  failure mode visible: the 0.15 base tracks the cash rate, the 0.05 base does not. Its
  header names the line as `b_t` and gives the other one beside it, since the two are easy
  to confuse and the naming is unsettled
- how much of a known policy stance the model still sees, by how long the stance lasted.
  Drawn only when `--injection-test` has been run. **The chart to show anyone who wants to
  read the era residuals as an audit of the Bank**
- the cash rate, neutral `b_t`, and the rule's prescribed rate
- what the rule prescribes: neutral and the inflation response on one axis
- how the cash rate responded to inflation (the two-gaps scatter)
- a Taylor rule on this model's neutral `b_t`, with no double-count now that it is built on
  the intercept. The source of the 2016-19 result above
- one prior-against-posterior chart per estimated parameter

Every chart carries the fitted equation in its header, read off the run rather than
hardcoded.

The Taylor chart is on `neutral`, the intercept a Taylor rule wants. Built on the
prescribed rate instead it would double-count, since that line already carries the RBA's own
inflation response and Taylor's adds a second.
