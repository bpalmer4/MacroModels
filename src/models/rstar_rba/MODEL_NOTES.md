# Neutral from the RBA's reaction to inflation

**What the model assumes, in one sentence: there is a neutral cash rate that moves slowly
over time, and on top of it the RBA responds to inflation sitting above or below the 2.5%
target. The model splits the observed cash rate into those two pieces.**

The slow piece is neutral. The other piece is the reaction function. Everything else follows,
including the model's limits, because the split rests on an **asserted** speed: `sigma_r`
says how fast neutral is allowed to move, and it is not estimated. The default also gives the
response no lag of its own, which is a second assertion and a weaker one, since the residual
autocorrelation says the cash rate plainly does adjust gradually (point 6).

**It is not an estimate of the neutral rate. It is an estimate of what the RBA's behaviour
reveals about it, and it reveals it imperfectly.** That sentence governs everything below.

```
pi_t   = 4 · sum_{j=0}^{4} w_j · q_{t-j},   w_j ∝ rho^j,  sum w_j = 1
g_t    = (pi_t - 2.5) / 0.5                  the inflation gap, in band-widths
b_t    = b_{t-1} + sigma_r · e_t             NEUTRAL: the slow piece, sigma_r imposed
d_t    = b_t + lambda · g_t                  the rule's PRESCRIBED rate, not neutral
r_t    = d_t + eps_t                         window one: the cash rate
f_t    = b_t + bias + u_t                    window two: the market's 5y5y forward
```

## THE BIGGEST THING TO KNOW: the level is pinned by a market price

**Added 2026-09-16, and it changes what this model is.** Read this before any number.

With the cash rate as the ONLY observable, the level of neutral was not a choice anyone
made. Take the sample average of `r = b + lambda·g + eps`: the residual averages to zero by
construction, so

    mean(b) = mean(r) − lambda · mean(g)

**Neutral's level WAS the historical average cash rate, adjusted for whether inflation
averaged on target.** Measured: mean neutral 3.952 against mean cash 4.038, a gap of −0.085,
and `lambda × mean gap` is +0.085, exact to three decimals. Nothing else pinned it, and no
reparameterisation could, because one observable cannot identify two levels.

That had a consequence the notes did not previously state: **this model was structurally
incapable of reporting that the whole level of neutral had shifted.** It could say neutral
was 1.6 points off its own floor; it could not say neutral is 3.9 in level terms. When CBA
put nominal neutral at 3.85 in September 2026 and this model said 2.99, that was not an
empirical disagreement. It was two different objects.

**The second window is the AOFM 5y5y risk-neutral forward rate**, `2·RNY10 − RNY5` from the
term-premium decomposition in [`src/data/aofm_loader.py`](../../data/aofm_loader.py). It is a
market price for where the cash rate settles over years five to ten, with AOFM's model
stripping the term premium. It is the only series available that speaks to the LEVEL of
neutral without being the cash rate's own history.

**It leads policy and does not echo it**, which is the check that matters given the RBA also
watches it. On quarterly changes, `corr(Δ5y5y_t, Δcash_{t+k})` peaks at **+0.340 at k = +2**
and is NEGATIVE at k = −1 and k = −2. So it moves two to three quarters ahead of the cash
rate and does not chase past decisions. Its sd is 0.99 against the cash rate's 1.97.

**THE ASSERTION HAS MOVED, NOT VANISHED.** `bias` is what the forward carries that neutral
does not: the market's view of the cycle over years five to ten, plus whatever premium AOFM
left in. It is free but tightly priored at N(0, 0.5), and **that prior is now what holds the
level**. Widen it and the level is unidentified again. The posterior comes back
**−0.109 [−0.289, +0.068]**, straddling zero, so the model is reading the forward as neutral
very nearly one-for-one.

**And the series revises.** AOFM re-estimates the whole decomposition monthly, so every
historical value moves when the file updates. This is not a real-time series.

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

## What it tells you

Two statements. Both are conditional, and both are worth quoting whole rather than as a
number.

**On the level.** Decompose the RBA's cash-rate decisions into a persistent neutral component
and a systematic response to inflation, and its conduct implies a real neutral rate of roughly
**zero to one per cent** today. Over horizons beyond a year or two the decomposition cannot
distinguish a persistent policy stance, a change in the reaction function, and a change in
perceived neutral.

**On the response.** Conditional on a contemporaneous reaction function, the Bank moved the
nominal cash rate by roughly **0.5 to 0.7 points** for each point that persistent underlying
inflation sat away from target. Allow realistic policy smoothing and the distinction between
the immediate and the long-run response is not identified.

Neither of those is "Australian r\* is 0.49%".

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
**0.49** (−0.05 to 1.05 across `sigma_r`). On like-for-like terms the three span two full
points, and note that this one is substantially the policy rate smoothed while the other two
are not, so agreement between them would not be corroboration.

**Each fails at a different point, and that is the finding rather than an inconvenience.**
`rstar_hlw` needs an IS curve the data do not contain. `rstar_bonds` observes a market price
but cannot identify its equilibrium level. This one identifies a policy-implied level only by
deciding which frequencies count as neutral rather than as policy. Three different
identification failures, not three bad models: the concept is coherent, and mapping Australian
data onto a number for it is not. That is a stronger statement about "neutral is 3.5%" than
any of the three makes alone.

---

## What this model can and cannot say

It assumes the RBA set the cash rate as neutral plus a response to inflation, and recovers
neutral as the rate minus the response. Everything below follows from that.

### The level of neutral comes from an assumption, and here is what it costs

The level comes from the rule residual `eps_t` having mean zero, which asserts that policy
averaged neutral over the sample, conditional on inflation. Neutral also absorbs anything persistent, per the
injection test below.

*Note.* That is the identifying assumption and the data cannot check it, which is normal for
a latent variable. Two consequences worth holding. It makes `b_t` close to a smoothed cash
rate, `corr` 0.95 against a 13-quarter moving average, which is the assumption showing
through rather than a defect. And it means `b_t` is not independent of policy, so agreement
with `rstar_bonds` is not corroboration: one is substantially the policy rate smoothed, the
other is not.

The model is not only smoothing, though. Against a naive filter the stance sits within 0.13
of zero in every era; against `b_t` it is +1.36 today and −0.51 in 2016-19. That difference
is the inflation adjustment, and it is where the model's content is.

A common level bias in the Bank's belief is exactly what the normalisation assumes away, so
**the positive current stance is conditional on it** and the `sigma_r` ensemble says nothing
about that, since a uniform bias shifts every member together.

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

**Two windows since 2026-09-16, and `sigma_r` loosened to 0.125. Every number below is from
that specification and is NOT comparable with earlier vintages**, which had one window and
`sigma_r` at 0.10. The previous headline was nominal 2.99 / real 0.49.

| | 2026Q2 |
|---|---|
| `lambda`, per band-width | **0.228** |
| `lambda`, per percentage point of inflation | **0.46**, 0.45-0.55 across defensible `sigma_r` |
| **neutral `b_t`, nominal** | **3.89** |
| **neutral `b_t`, real** (less long-run expectations) | **1.35**, and **1.10 to 1.41 across defensible `sigma_r`** |
| `forward_bias` | **−0.109** [−0.287, +0.069], straddles zero |
| `sigma_f` | 0.146 |
| the rule's prescribed rate, `b_t + lambda·g_t` | 4.25 |
| cash rate | 4.35, so the stance is **+0.46** and the rule residual **+0.10** |
| sd of the quarterly change in neutral | **0.118** |
| sampling | 0 divergences, r_hat 1.0, min ESS 2,229, MCSE/sd 0.026 |

**`sigma_r` = 0.125 rather than 0.15, on sampling.** 0.15 was tried and fails three checks:
1 divergence in 8,000 (0.0125% against a 0.0100% rule), MCSE/sd **0.056** against 0.05, and
min ESS 1,296. The culprit is `sigma_f` alone, everything else samples cleanly at both
settings with `lambda` at ESS 14,141, and the reason is structural: `sigma_f` and `sigma_r`
compete to explain the same gap between the forward and the fitted neutral, so loosening the
walk makes them harder to separate.

The substance is unchanged between the two: neutral 3.89 against 3.91, stance +0.46 against
+0.44, `forward_bias` −0.109 at both to three decimals. The only real cost is neutral's
quarterly volatility, **0.118 against 0.151**, which loosens the three-way match with
`rstar_bonds` (0.155) and `rstar_tvpvar` (0.153). That match was a nice-to-have, not a result
the model rests on, and it was being bought with a sampler that could not cleanly resolve the
new parameter.

**Three things in that table are new and worth pausing on.**

**The level range collapsed.** Real neutral ran **−0.05 to 1.05** across the `sigma_r` sweep
with one window, a spread of 1.10, wider than the credible interval at any single value. It
now runs **1.10 to 1.41**, a spread of 0.31, and is nearly flat from `sigma_r` 0.10 upward
(3.85, 3.91, 3.91 nominal at 0.10, 0.15, 0.20). That is what pinning the level externally
buys, and it is why `sigma_r` could be loosened.

**Neutral's speed now matches the rest of the package.** sd of its quarterly change is
**0.151**, against **0.155** for `rstar_bonds` and **0.153** for `rstar_tvpvar`. At the old
`sigma_r` = 0.10 with one window it was 0.063, by some way the slowest-moving neutral in the
package. At `sigma_r` = 0.15 it reaches 0.151 and matches the other two almost exactly, but
that setting fails three sampling checks, so 0.125 is shipped and the match is looser.

**The rule residual all but vanished**, +0.87 to **+0.08**. Under one window the model said
the RBA was sitting nearly a point above its own reaction function. With a higher neutral it
says the Bank is on its rule and the stance is +0.44: policy just tight.

**`lambda` UNITS matter and have caused errors.** The gap is scaled by the band half-width,
so `lambda` is per band-width and the response per percentage point of inflation is
`lambda / 0.5`, twice `lambda`. Taylor's 1.5 per point would be **0.75** in these units, not
1.5 and not 0.5, so quoting 0.303 beside 1.5 makes the RBA look five times less responsive
than the estimate says. That units trap is now the second reason not to put the two numbers
side by side; the first is that they are not comparable at all in nominal terms, per "The
Taylor comparison does not work in nominal terms".

**`lambda` is the most durable result, conditional on zero policy smoothing.** It is stable
across the whole `sigma_r` ensemble, but not across `phi`: allow interest-rate smoothing and
it runs 0.61 to 2.57, because the contemporaneous form makes one coefficient carry both the
immediate and the ultimate response (point 6). It is an estimate of the RBA's
systematic response and it survives every respecification tried.

It does **not** follow that the RBA fell short of the Taylor principle. A sub-unit *nominal*
response lowers the real rate only if expected inflation moves one for one with actual, which
is an assumption the model neither makes nor tests. `lambda` is a nominal response; whether
the real one is positive is not established here either way. See "The Taylor comparison does
not work in nominal terms".

**Neutral is close to a smoothed cash rate, adjusted for inflation away from target.**
`corr(b_t, cash)` is 0.88, and 0.95 against a 13-quarter moving average. The smoothing is the
identifying assumption showing through rather than a defect; the adjustment is what the model
adds. See "The level of neutral comes from an assumption".

---

## What identifies what

**`lambda`** comes from the covariation of the cash rate with the inflation gap, which over
1993-2026 correlates about 0.6 on the band-scaled measure. Inflation targeting worked, so
most deviations were small and most of what the cash rate did was not a response to them. A
reaction function is hardest to estimate precisely when the central bank is good at its job.
In practice `lambda` is identified largely by the two episodes where the gap was large,
2008 and 2022-24.

**The level of neutral** comes from the rule residual `eps_t` having mean zero, which asserts
that policy averaged neutral over the sample, conditional on inflation.

**The split between neutral and the response** comes from `sigma_r`, and **0.10 is an arbitrary
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

The likelihood asserts the residual is iid Normal(0, `sigma_eps`). With neutral held constant
it is nothing of the kind: **autocorrelation 0.974 against 0.857 for the walk, a trend of
−1.54 a decade, and era means marching from +2.05 in 1994-99 to −1.99 in 2016-26**, with a
residual sd of 1.91 against the cash rate's own 1.98, so the model explains almost nothing.
**That is a direct contradiction of the model's own assumption, not a fit-versus-flexibility
argument**, and it holds however many parameters you are willing to spend.

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
default. **WITH the 5y5y second window**, which changes what this table says about the model.

| `sigma_r` | `lambda` per band-width | per pp | neutral nominal | **neutral real** | 2016-19 residual | `corr(base, cash)` |
|---|---|---|---|---|---|---|
| 0.05 | 0.276 | 0.55 | 3.60 | **1.10** | −0.90 | 0.84 |
| 0.10 | 0.235 | 0.47 | 3.85 | **1.35** | −0.96 | 0.85 |
| **0.125** (default) | 0.228 | 0.46 | 3.89 | **1.35** | −0.98 | 0.84 |
| 0.15 | 0.225 | 0.45 | 3.91 | **1.41** | −1.00 | 0.83 |
| 0.20 | 0.223 | 0.45 | 3.91 | **1.41** | −1.01 | 0.82 |

**The level has stopped riding on `sigma_r`, which is the whole point of the second window.**
Real neutral spans **0.31** here against **1.10** on the one-window version, whose table was:

| `sigma_r` | neutral real, ONE window |
|---|---|
| 0.05 | −0.05 |
| 0.10 | 0.49 |
| 0.15 | 1.05 |
| 0.20 | 1.25 |

It is also nearly FLAT from 0.10 up, at 3.85, 3.91, 3.91, where before it climbed
monotonically with no interior answer. That is the difference between a level pinned by an external price
and a level that was the cash rate's own average being rationed by a smoothness knob.

**`corr(base, cash rate)` no longer runs away either.** It sat at 0.80, 0.88, 0.93, **0.96**
under one window, so 0.20 had to be carried as the boundary where neutral became the cash rate
smoothed. It now reads 0.84, 0.85, 0.83, 0.82: flat, and slightly FALLING as `sigma_r` rises.
The old boundary is gone, which is why the default could move from 0.10 to 0.15.

**What did NOT improve: the 2016-19 residual.** It is around −0.96 to −1.01 at every setting,
where the one-window model had it near zero at the default. With a higher neutral the model now
says the RBA ran persistently BELOW its own rule through that period, by about a point. Whether
that is a finding or a symptom is open; it is at least consistent with the two-indicator
evidence (demand gap −0.35, inflation gap −0.52) that policy was too tight, since a rate below
the rule can still sit above neutral.

Read the rest in three parts.

*`lambda` moves, but not enough to change what it says.* 0.72 to 0.52 per point over the
defensible range, 0.49 out at the boundary, straddling the headline 0.61. It moves in the
direction the frequency-cutoff reading predicts: a smoother neutral leaves more for `lambda`.
The range to quote is 0.52 to 0.72 rather than 0.61 alone.

**State that narrowly.** What this shows is that *conditional on zero policy smoothing*, the
inflation response is insensitive to how fast neutral is permitted to move. It is not
insensitive to the smoothing assumption itself: free `phi` and it runs to 2.57 (point 6).
`sigma_r` and `phi` decide the same thing and only one of them is being varied here.

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
less than 134 independent observations. **The nominal posterior intervals should not be
read literally.** The estimates survive; the precision does not.

On the scale of the problem: the textbook effective-sample-size calculation for a mean under
AR(1) dependence, √((1+ρ)/(1−ρ)), is about 3.6 at ρ = 0.857, which would put `lambda`'s
posterior sd nearer 0.18 than 0.050. Treat that as an indication of magnitude only. It is a
heuristic for a sample mean, not a corrected posterior for a parameter in a non-linear
state-space model, and no corrected intervals have been computed.

---

## What the exploration established

### 1. Inflation must be averaged, and the memory is a judgement not an estimate

The raw quarterly print is the wrong object: annualised it has an sd of 1.12 and correlates
0.09 with the cash rate. Fitted on it the model collapses completely, `sigma_eps` → 0.045,
`lambda` straddling zero, and the base correlating **1.00** with the cash rate.

Averaging fixes that, but the memory length cannot be estimated. Sweeping the truncation:

| `max_lag` | `rho` | mean lag | `lambda` per pp | `sigma_eps` |
|---|---|---|---|---|
| 4 | 0.869 | 1.72 | 0.61 | 0.786 |
| 6 | 0.882 | 2.50 | 0.71 | 0.776 |
| 8 | 0.893 | 3.26 | 0.82 | 0.768 |
| 12 | 0.865 | 4.10 | 0.96 | 0.770 |

`rho` does not adjust, the mean lag scales with the truncation at roughly `max_lag`/3, and
`sigma_eps` is flat to three decimals, so the fit cannot tell these apart. Meanwhile `lambda`
runs 0.61 to 0.96 per point, a range wide enough to matter for anything built on it. `rho` = 0.87 implies an untruncated mean lag of 6.6 quarters, so every window
tried cuts the geometric tail and renormalises. **`max_lag` = 4 is a stated judgement**, and
`lambda` should be quoted as a range across it.

Free Dirichlet weights over 13 lags are not identified at all: every weight returns its
uniform prior mean. One shape parameter is recoverable from this data; thirteen are not.

**Whether the memory is constant cannot be tested here.** Splitting `rho` at 2008Q1 moves
`sigma_eps` from 0.786 to 0.787 and returns `rho_early` at 0.575 [0.24, 0.87], an interval
spanning nearly the whole admissible range and overlapping the late one. The early sample
cannot adjudicate, which is the same weakness that gives `lambda_early` an sd three times
`lambda_late`'s. Not exposed as a flag, since the point estimates would be read as a result.

### 2. The response is linear: no curvature and no deadband

Adding `lambda_2 · g·|g|`, `lambda_2` came back **negative in every specification tried** and
positive in none, across six variants of scaling, `max_lag`, sample and floor treatment, and
`sigma_eps` never improved on the linear model. The band-scaled default gives −0.056
[−0.094, −0.018]. Run with `--nonlinear`.

**There is no deadband either**, so inflation of 2.7 draws a response despite sitting inside
the band. Tested with `lambda · sgn(g) · max(|g| − d, 0)`: imposing a deadband makes the fit
monotonically worse, and estimated freely `d` returns **0.072pp [0.004, 0.199]**, excluding a
quarter-point. The test has power, since a third of the sample sits within 0.25pp of target.
A deadband is a form of convexity, the opposite sign to `lambda_2`, so the two shape tests
agree.

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

**Read it as suggestive, not established.** The nominal P(break < 0) is 0.991, but the
residuals are autocorrelated at 0.85 so that interval cannot be read literally; on the
heuristic scaling above it would be something nearer 0.75, which is an indication rather
than a corrected figure. The pre-2008 coefficient is also the weak one, sd
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
it: `lambda` 0.305 → 0.303, `sigma_eps` 0.786 → 0.775. At the COVID quarters neutral steps 0.08
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
inflation gap, so the whole move must go to neutral or the residual, and `sigma_r` alone
decides which. Across the ensemble neutral rises by **0.00, 0.25 and 0.79 points** from
1994Q2 to 1995Q4. At 0.05 the model says policy was tight; at 0.15 it says neutral rose most
of a point. That is the widest disagreement anywhere in the sample and it is visible at the
left edge of the neutral chart.

**Substituting a forecast does not fix it**, which was tested and is the reason Refinement 1
is struck. A constructed forecast turns the 1994 gap from −0.30 into +1.17 and still buys
only about 0.47 of cash rate. The obstacle is the size of the move, not the inflation
measure: 275bp is beyond what any plausible response coefficient times any plausible
inflation signal can produce.

**The adjustment speed is not constant either**, which compounds it. Inertia in the cash rate
sits around 0.91 through 1993-2015 and rises to 0.975 after 2016, and it is not the lower
bound doing it: excluding 2020-21 it is still 0.96. The Bank became markedly more gradual,
and the model assumes one speed throughout.

### 5. The employment leg cannot be added, and the reason is instructive

The RBA's mandate is inflation **and** full employment, so the inflation-only rule omits a
statutory objective. Adding `lambda_u · (u − u*)`, with u\* from `ystar_ustar`:

| | `lambda_pi` | `lambda_u` | `sigma_eps` |
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

**One thing survives the rejection.** Neutral barely moves, 3.01 → 3.06, so the inflation-only
rule was **not** parking the employment response in neutral, which was the worry that prompted
the test.

What does *not* follow is that `lambda` therefore carries the labour-market response, which
these notes used to claim. The augmented regression shows the two terms compete for the same
variation; it does not say how much of the employment mandate sits in each. The defensible
statement is weaker and applies more broadly: **`lambda` is not a structural response to
inflation alone.** Any omitted systematic motive that covaries with inflation can affect it,
and labour-market conditions are one candidate among several. Persistent omitted motives
affect neutral instead, per "It conflates belief with everything else systematic".

**What was tested is the LEVEL, `u − u*`.** A change-based labour term in the rule has never
been tried, and the simultaneity objection above applies to it too, since unemployment
responds to policy over exactly the horizon such a term would measure. Whether the change
would behave differently is unknown here. The first thing to look at would be whether
`lambda_pi` survives its addition, since its collapse is what made the level version
uninterpretable.

Run with `--employment`; off by default.

### 6. Partial adjustment exposes a second non-identification

`r_t = phi·r_{t-1} + (1 − phi)·(b_t + lambda·g_t) + eps_t`, so the rule's rate becomes what the
Bank moves *toward* and it closes `(1 − phi)` of the gap each quarter. Built and tested.
`--partial-adjustment`, off by default.

**This makes it a three-way split rather than a two-way one.** The contemporaneous model
divides cash-rate movement between neutral and response. Adding `phi` divides it between
neutral, adjustment speed and response, and that separates two things the default cannot tell
apart: moving *little* and moving *slowly*.

Swept over `phi` from 0 to 0.95, the long-run `lambda` rises from 0.61 to 2.57 while the
immediate response `(1 − phi)·lambda` falls from 0.61 to 0.12; `sigma_eps` improves from 0.790
to about 0.42 and stops improving around `phi` = 0.85; and the residual autocorrelation falls
from 0.857 to 0.414. Neither the autocorrelation nor the error is eliminated, so the
persistence is not only adjustment lag.

**The headline `lambda` answers a question with two answers.** At `phi` = 0.85 the immediate
response is 0.18 and the ultimate one 1.19; the contemporaneous estimate of 0.61 is neither,
sitting between them because a persistent inflation gap gives the rate time to travel. One
coefficient is carrying both concepts.

**So `lambda` = 0.61 was only ever stable conditional on imposing no smoothing.** Across the
`sigma_r` ensemble it runs 0.52 to 0.72; across `phi` it runs 0.61 to 2.57. That does not
vindicate `phi` = 0, it exposes a **second** non-identification beside `sigma_r`. Both
parameters decide the same thing, whether a movement in the cash rate is called neutral,
adjustment dynamics, or systematic response, and two published series cannot pin both. The
claim these notes make elsewhere should be read narrowly: *conditional on zero policy
smoothing, the inflation response is insensitive to how fast neutral is permitted to move.*

**The free estimate is separately incoherent**, which is a different objection from weak
identification and survives it. `phi` = 0.953 implies a 3.6-year half-life, which is not how a
bank moving in 25bp steps behaves; the fit does not support it either, since `sigma_eps`
saturates around `phi` ≈ 0.85 and is fractionally worse at 0.953. It produces a prescribed rate
averaging 7.30 through 2022-26 with the Bank persistently 3.8 points below it for four years,
and `phi·r_{t-1}` displaces neutral, `corr` with the cash rate falling to 0.759 and neutral
going nearly flat after 2012.

**Why the default keeps `phi` = 0.** For parsimony and interpretability, **not because the data
reject interest-rate smoothing**: they plainly do not, given the autocorrelation. A
partial-adjustment model improves the residual dynamics substantially but is itself weakly
identified against the drifting neutral rate, and imposing `phi` by hand would give the package
two arbitrary smoothness parameters where it currently has one. The honest next step is a joint
`phi`/`sigma_r` surface, per Refinement 2, not a choice between them.

---

## 2016-2019: the rule prescribed less easing than a Taylor benchmark

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
prescribes a full point lower. So this is not a deviation from the rule, it is **the rule
itself sitting a point above the benchmark**. Note what that does and does not establish: the
Taylor rule is a benchmark, not an authority, so the comparison shows the RBA's revealed
reaction function prescribing substantially less easing than a conventional rule would, and
does not by itself adjudicate what policy should have been. The outcome evidence below is what
carries the normative weight. The arithmetic is clean: core inflation averaged
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

**−1.03 is arguably a floor rather than a central estimate**, and this is the most contestable
claim in these notes. The two obvious objections bias it the same way: if policy really was
persistently tight, neutral absorbed part of it and sits too high, raising the Taylor
prescription and understating the gap; and if the Bank was holding rates up for
financial-stability reasons, this model books that as a higher neutral rather than a second
objective (see "It conflates belief with everything else systematic"), which again raises it.

**Treat that as an interpretive argument, not a result.** Enough endogenous quantities interact
here that giving the bias an unambiguous direction is a judgement. Nothing important rests on
it: the gap is negative throughout 2016-19 at every `sigma_r` regardless.

**One limit.** "Should have been lower" needs the counterfactual that lower rates would have
lifted inflation, and this repo's central negative finding is that the rate-to-activity link is
not identifiable on Australian data (see [`is_curve`](../is_curve/MODEL_NOTES.md)). The model
can say policy was inconsistent with the target; it cannot say how much inflation a cut would
have bought.

This comparison is **nominal against nominal**, so it is unaffected by the Fisher confounding
that withdrew the "squibbed the Taylor principle" reading. `pi_core` and `ygap` come from
`ystar_ustar` and the 0.5/0.5 weights are imposed.

---

## This neutral cannot be used to test an IS curve

[`is_curve`](../is_curve/MODEL_NOTES.md)'s `rule` variant puts the output gap against the
real cash rate less this model's **neutral**, `neutral_real`. At lag 2:

| variant | slope | t | R² |
|---|---|---|---|
| none (raw real cash) | +0.041 | +1.86 | 0.027 |
| bond-market r\* | +0.123 | +2.82 | 0.060 |
| **reaction-function neutral** | **+0.201** | **+4.54** | **0.143** |
| flat r\* | +0.041 | +1.86 | 0.027 |

**The largest of the four and the most wrongly signed.** An IS curve needs a negative slope.
This says the output gap is *higher* when the real rate sits above neutral, which is the
reaction function reversed: the RBA sets a positive stance when the economy runs hot.

**But none of these t-statistics means anything.** The `rule` residuals are autocorrelated at
0.892, so the classical standard error is understated roughly fourfold; corrected, t falls
from +4.54 to about +1.1 and the slope's 90% interval spans [−0.10, +0.51]. The scatter is a
blob with a tilt. The right conclusion is not "wrongly signed and significant" but **wrongly
signed and not significantly anything**, which buries the IS curve rather than rescuing it.

The lag structure is the more informative part. The `rule` slope decays monotonically from
+0.325 at lag 0 to zero by lag five or six, and every other variant does the same.
Transmission would be the mirror image, weak on impact and strengthening over three to six
quarters with a negative sign.

**This variant used to read the prescribed rate and returned −0.009 with an R² of 0.000.**
That looked like a clean null and was an artefact: a rate gap measured against
`b_t + lambda·g_t` is close to this model's own rule residual, which is high-frequency
timing noise. Switching to neutral on 2026-09-11 restored the +0.201 an earlier note had
recorded and then withdrawn. **The withdrawal was the mistake**, not the number: the base is
neutral, and neutral is the input the comparison wants.

So the result stands as evidence *for* the repo's central negative finding rather than
against it. Nothing here supports an IS curve, and the strongest-looking fit is the
simultaneity showing through.

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
   it.** What is held fixed: `lambda`, `rho`, `sigma_eps`, the objective set, and the
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
6. **`lambda` is not a structural response to inflation alone.** Any omitted systematic
   motive that covaries with inflation can affect it, labour-market conditions among them.
   How much of any of them it carries is not identified. See "What the exploration
   established", point 5.
7. **The 1994 tightening is invisible to the rule, and swapping the inflation measure does
   not help.** The RBA raised 275bp with the inflation gap at −0.3, so the single largest
   pre-GFC policy move is absorbed by `sigma_r`'s choice instead. Realised inflation standing
   in for forecast inflation looked like the cause; it is not. A constructed forecast flips
   the 1994 gap to +1.17 and still prescribes only about 0.47 of cash rate, because the
   binding constraint is **magnitude**, not the measure. See point 4a and Refinement 1.
8. **The floor quarters are in by default.** `--floor 0.5` excludes them; it did not change
   any conclusion.
9. ~~**The default depends on ABS GDP**, because jumps are on.~~ Resolved: jumps are off by
   default, so the headline needs the cash rate and trimmed-mean inflation and nothing else.
   `--jumps` reintroduces the GDP dependency for the sensitivity test.
10. ~~**The `stance` variable is misnamed.**~~ Resolved. `stance` is now the cash rate less
    neutral, which is what the word means, and `rule_residual` is the cash rate less the
    rule's prescribed rate, which is what the old `stance` actually held.

## Refinements

1. ~~**Forecast inflation** in place of realised.~~ **Tried and it does not deliver.** The
   case for it was 1994: the RBA tightened 275bp with realised inflation at 2.2, a gap of
   −0.3, so the rule cannot see the episode at all. No RBA forecast series is held, so a
   forecast was constructed, an expanding-window regression of annualised trimmed-mean
   inflation four quarters ahead on its own lags plus the four-quarter change in
   unemployment, and fed to the model in place of realised inflation.

   It fixes the sign and not the size. The 1994 gap goes from −0.30 to **+1.17**, but at the
   fitted response that prescribes **0.47** of cash rate against −0.18 before, a swing of
   about 0.65 where 275bp needs accounting for. Everything else is slightly worse:
   `sigma_eps` 0.833 against 0.786, `lambda` 0.40 against 0.61, residual autocorrelation
   unchanged, `corr(neutral, cash)` up to 0.902.

   **The diagnosis was wrong, and that is the finding.** 1994 is not a realised-versus-
   forecast problem. It is a **magnitude** problem: explaining 275bp needs either an
   implausible response coefficient or an inflation signal several times larger than any on
   offer, so no inflation-based regressor reaches it. Observation 7 and point 4a should be
   read that way.

   ONE-OFF, 2026-09-11: run from a throwaway script, not reproducible from this package, and
   its charts have been deleted. The forecast also carried four of its own coefficients
   outside the posterior, so its intervals would have been too tight even had it worked.
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
