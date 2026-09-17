# r*, the natural rate from the bond market

A Bayesian unobserved-components model (PyMC + NumPyro NUTS) estimating the Australian
natural rate of interest from asset prices. One latent state, an Australia-specific wedge
over a published world real rate, read off three windows on the same curve.

```
wedge_t = wedge_{t-1} + sigma_walk · e_t,  e_t ~ StudentT(nu)   the only state; nu imposed at 9
r*_t    = w_t + wedge_t                    b_world imposed at 1; w is data
g_t     = r_t - r*_t                       identity: the policy stance
tp_t    = y_t - r*_t - k·g_t               identity: the term premium
spread_t = tp_t - au_tp_t                  the AOFM premium is DATA
f_t     = r*_t + bias + e_t                the 5y5y forward: the one window on the LEVEL
g, spread ~ stationary AR(1)               the identifying priors
```

`y` is the AU indexed real 10-year yield, `r` the real overnight cash rate, `w` the
Cleveland Fed's 10-year expected real rate LESS the published US term premium, `au_tp`
the AOFM's published Australian 10-year term premium, and `f` the AOFM's 5y5y risk-neutral
forward deflated by long-run expectations. Sample 1993Q1-2026Q3, 135 quarters.
The equation-by-equation section takes every line one at a time.

**What is asserted, in one place.** The premium is no longer stationary about a free `mu_tp`;
it tracks a published Australian series and only the spread over it is estimated. That spread
is a real-minus-nominal difference on the same country's curve, so it is an inflation risk
premium and nothing else. `mu_spread` comes back **0.241 [−0.394, 0.813]** against an
N(0.25, 0.5) prior: a posterior sd of 0.320 against the prior's 0.500, so the data now move
it about a third of the way rather than the 5% they moved it before the third window went in.
The level is better identified than it has ever been here and is now worth quoting; the 90%
band on r\* is 1.28 points and no longer contains zero.

Two things are imposed rather than estimated and both are load-bearing: `b_world` at 1, and
`nu_walk` at 9.0. The second exists only because the third window will not sample without it;
9.0 is what this model's own free posterior chooses when the forward is off.

## One of three routes, all flawed

This repo now contains three separate attempts at Australian r\*, and the useful thing is
that they fail differently. Read them together rather than picking one.

| package | identified from | what it actually measures | how it fails |
|---|---|---|---|
| [`rstar_hlw`](../rstar_hlw/MODEL_NOTES.md) | trend growth + the IS curve | the textbook definition: the rate at which output sits at potential | the IS curve does not identify anything on AU data, so each specification returns its own prior |
| **`rstar`** (this one) | asset prices | what investors price | the *level* is not identified; it rests on the stationarity prior and an asserted premium |
| [`rstar_rba`](../rstar_rba/MODEL_NOTES.md) | the RBA's response to inflation | not r\* but what the Bank's conduct *reveals* about it, conflated with every other systematic motive | a long enough departure from the rule is absorbed into neutral, so it cannot audit the Bank over a decade |

Only the first targets what the theory defines, and it is the one that cannot be estimated.
The other two measure *beliefs* about r\*, held by different people. That is a sharper
statement of the package's thesis than "r\* is imported", and it was arrived at by
exhausting the alternatives rather than by assertion.

**There is no IS curve here, deliberately.** The repo's central negative finding is that the
interest rate does not visibly move real activity at these frequencies. `rstar_hlw` measures
it at a_r ≈ −0.04 against σ_IS ≈ 0.70; `nairu`'s IS curve gives β_is ≈ 0.084;
[`is_curve`](../is_curve/MODEL_NOTES.md) shows the raw scatter does not even recover the
*sign*. This package reads r\* off an asset price instead, sidestepping the broken link
rather than trying to strengthen it.

**The tension that creates, stated up front.** This model carries a Taylor rule, and a rule
is only worth prescribing if moving the rate moves the economy. Read the prescription as
what a standard reaction function *would* say, not as a forecast of what would follow.

---

## Read this first

| | 2026Q2 |
|---|---|
| real r\* | **0.80** [0.15, 1.42] |
| world real rate, for comparison | 0.99 |
| the Australian wedge | **−0.20** |
| term premium | 1.56 |
| nominal r\* (r\* + long-run expectations) | **3.33** |
| r\* for firms (r\* + credit spread) | 1.56 |
| pre-GFC r\* (1994-2007) | 1.70, so today is **47%** of it |

**QUOTE THE LAST COMPLETE QUARTER.** The bond block is daily, so the model also estimates the
quarter in progress, off a part-month average with inflation and both gaps missing and the
mortgage rate carried forward. That quarter reads 0.25pp higher (1.05 real, 3.57 nominal, wedge
−0.29) on a 90% band no wider than a finished quarter's, which is the problem with it. Charts
now stop at the last finished quarter; the trace still carries the extra one if you want it.

Zero divergences, all `r_hat` 1.00, minimum `ess_bulk` 2,534, BFMI 0.75.

**The specification changed TWICE on 2026-09-16 and these numbers are not comparable with
earlier vintages.** In the morning three defaults moved together: the term premium is now
pinned to the AOFM's published Australian series, the world anchor is premium-stripped, and
`b_world` is imposed at 1. In the evening a THIRD OBSERVATION WINDOW was added, the AOFM's
5y5y risk-neutral forward deflated by long-run expectations, loading directly on r\* as
`f_t = r*_t + bias + e_t`, with `nu_walk` fixed at 9.0 so that it samples. The next two
sections are why the premium is pinned; the third window has its own section below.

To reproduce the morning default: `--no-forward`. To reproduce the old default exactly:
`--no-forward --no-au-premium --no-impose-world-loading --world-source cleveland`.

**Australia sits BELOW the world rate, and this reverses an earlier headline.** The wedge is
−0.29. The pre-morning default put it at +0.05 and the notes said "Australia currently sits
on the world rate... nothing Australia-specific is depressing neutral today". That sentence
does not survive the respecification and should not be quoted.

Be clear about why it moved, because it is not a discovery. Imposing `b_world` at 1 on a
premium-stripped anchor *defines* the wedge as Australian r\* less the world rate, with no
loading free to absorb part of the difference. The old +0.05 was a wedge measured against
0.481 of a premium-bearing US yield, which is a different quantity wearing the same name. The
era pattern is the robust part: **+0.38** through 1994-2007, **−0.13** in 2020-21, and
**−0.87** across 2022 to now. Australia's spread over the world opened negative in the 2010s
and has not closed.

**THE LEVEL IS NOW WORTH QUOTING, AND THAT IS NEW.** Until the third window these notes said
the opposite, and the reason was `mu_spread`: it carried the level and came back barely moved
off its N(0.25, 0.5) prior, a posterior sd of 0.477 against the prior's 0.500, which is data
moving a parameter 5%. The 90% interval on r\* was 2.6 points wide and straddled zero at every
date.

The 5y5y forward is the only observable in this model that speaks to the level directly: the
long yield and the cash rate between them pin r\* + tp but not the split. With it, `mu_spread`
comes back 0.241 [−0.394, 0.813], a posterior sd of **0.320**, so the data now move it 36% off
prior. The interval on r\* is **1.28 points wide and no longer contains zero**. That is the
single biggest improvement in this model's history and it is why the window is on by default.

**Two things it costs, and both are real.** The wedge is about four times jumpier quarter to
quarter, sd(Δwedge) **0.196** against 0.052 on two windows, while sd(wedge) barely moves,
0.694 against 0.698. The forward injects high-frequency movement rather than a new trend, and
a 5y5y forward moves with market sentiment in a way r\* should not. Some unknown share of that
0.196 is bond-market noise booked as r\*, and nothing here apportions it.

And `forward_bias` comes back **+0.141 [−0.470, 0.764]**, an interval barely narrower than its
0.5 prior and straddling zero. `rstar_rba` gets −0.109 [−0.289, +0.068] from the same series.
That model watches only the cash rate; this one already reads the indexed 10-year yield, so
the forward competes with `tp` and the wedge for the same variation instead of adding a clean
new fact. Read the level improvement as real and the bias as uninterpreted.

**The amplitude problem came partly back, and this is a regression.** The pre-morning default
swung **4.39** points of nominal r\* across the sample, the morning's two-window version
**3.66**, and this swings **4.12**, against CBA's 3.15. `var share, r*` is 0.273 with the
premium taking 0.229. Pinning the premium bought amplitude discipline; the third window spent
some of it back. If you want the tighter amplitude and will pay for it with an unidentified
level, `--no-forward` is that model and it is still supported.

---

## Why the premium is pinned: the assertion that failed

Added 2026-09-16, and it is the most consequential check this package has had. **The
specification below is the one that lost, and it is no longer the default.** It is recorded
because the reason for the change is the evidence, and because `--no-au-premium` restores it.

Until 2026-09-16, `tp` was a stationary AR(1) about a constant **because the model said so**.
That assertion was what split the yield, and the notes always said the level rested on it.
What was missing was any Australian estimate of the premium to test it against. The AOFM publishes one: a
daily ACM decomposition of the nominal Treasury Bond curve back to 1992-07, with a term
premium and a risk-neutral yield at every tenor 1 to 10 years. See
[`src/data/aofm_loader.py`](../../data/aofm_loader.py). It is carried on **every** run now
and charted against the model's own premium, so the check runs whether or not it is used.

### What the audit said

On the OLD default (`--no-au-premium`), comparing four-year averages at each end of the
sample:

| | fall across the sample | sd |
|---|---|---|
| the model's fitted `tp` | **0.29** | 0.43 |
| the AOFM premium (bias-corrected) | **1.45** | 0.76 |
| **the wedge** | **1.79** | |

`corr(model tp, AOFM tp)` is **0.30** in levels and 0.27 in changes: the model's premium and
the published one are barely the same object. `corr(wedge, AOFM tp)` is **0.92**.

The mechanism is not mysterious and it is not a bug. The Australian long yield fell a long
way. The model cannot put that decline in the premium, because the premium was pinned
stationary, so it goes into r\* and hence into the wedge. **What the wedge fell by is close
to what the published premium fell by.**

### Three honest qualifications

1. **Real against nominal.** The model's `tp` is on the indexed curve; the AOFM's is on the
   nominal one, and the difference is an inflation risk premium. Anchoring plausibly removed
   a lot of that through the 1990s, so part of the early gap is legitimate. It explains the
   1990s far better than it explains 2012-2021, well after anchoring.
2. **Level correlations here are weak evidence.** Within the AOFM decomposition,
   `corr(TP10, RNY10)` is 0.998 over this sample: both components inherit the yield's
   downtrend, so almost anything trending correlates with either. The changes correlations
   are the ones to weigh.
3. **The AOFM's "ols" sheet is a plain ACM**, the estimator this package tested and rejected
   as a *US* premium on Bauer-Rudebusch-Wu persistence-bias grounds. The same scepticism
   applies here, which is why `bc` is the default. Note the direction: `bc` falls 1.45 where
   `ols` falls more, so the more defensible sheet is the *conservative* one and the finding
   survives it.

### The AOFM decomposition passes the checks that killed US ACM

Qualification 3 is the one that had to be answered before the premium could be taken as data,
because pinning makes AOFM's specification error invisible: it enters as data, with no
residual. Three tests, all run 2026-09-16.

**The era pattern is not inverted.** US ACM was rejected because the implied US neutral rate
*rose* 1.7 points from 2008-2015 into 2016-2019, putting the trough in the mining-boom years.
The AU expectations component does the opposite, on both sheets:

| implied expectations component | 2008-2015 | 2016-2019 | change |
|---|---|---|---|
| bias-corrected | 3.58 | 2.39 | **−1.20** |
| OLS | 3.74 | 2.91 | **−0.83** |

Monotone decline into ZIRP, which is the right sign.

**It predicts the short rate that followed.** `RNY10` is the average expected nominal short
rate over ten years, so it can be checked ex post against what the cash rate actually
averaged over the next forty quarters. Correlation **+0.77** on both methods, RMSE 1.1, with
a small upward bias (+0.37 to +0.48). The premium is not absorbing variation that belonged to
expectations, which is exactly what US ACM did.

**The two methods disagree less than the two US providers do.** The OLS−BC gap has an sd of
**0.42** against 0.66 for Kim-Wright vs ACM, era means running +0.26 to −0.74 against +0.82
to −0.54, and they converge to −0.05 at the latest quarter. The same *shape* of disagreement,
OLS putting more into the premium, but smaller, and it never flips the era pattern's sign.

The premium sd ordering does mirror the US one, `ols` 1.19 against `bc` 0.78 as ACM 1.07 ran
against Kim-Wright 0.63. That is why `bc` is the default, and the choice is now evidenced
rather than argued by analogy.

---

## What happens when the premium is not asserted

Every spec below was run on 2026-09-16 with the nominal conversion on long-run expectations.
"old default" is `--no-au-premium --no-impose-world-loading --world-source cleveland`.
The "two-window" column was the shipped default for part of that day and is now `--no-forward`.
**"NEW DEFAULT" is the shipped specification, with the 5y5y forward as a third window.**

| | old default | `--au-premium` only | two-window | **NEW DEFAULT** | AU-only, no world | `--nominal-window` |
|---|---|---|---|---|---|---|
| r\*, 2026Q3 real | 1.09 | 0.82 | 0.84 | **1.05** | 0.72 | 1.18 |
| nominal | 3.61 | 3.35 | 3.36 | **3.57** | 3.24 | 3.68 |
| wedge, 2026Q3 | +0.05 | +0.34 | −0.50 | **−0.29** | +0.72 | −0.13 |
| r\*, 1994-2007 | 2.43 | 1.86 | 2.06 | **1.70** | 1.72 | 2.05 |
| r\*, 2016-2019 | −0.49 | −0.09 | −0.11 | **+0.42** | −0.06 | +0.36 |
| r\*, 2020-2021 | −1.10 | −0.43 | −0.70 | **+0.14** | −0.24 | +0.01 |

Every 2026Q3 row above is the quarter-in-progress reading, kept as-is because the five
comparison columns would each need re-running to restate them and the comparison is between
columns rather than against a level. For the shipped column the complete-quarter figures are
0.80 real, 3.33 nominal, wedge −0.20.
| var share, r\* | 0.792 | 0.347 | 0.486 | **0.273** | - | 0.911 |
| nominal amplitude | 4.39 | 2.95 | 3.66 | **4.12** | 2.48 | - |
| real 90% band, now | 3.43 | 2.43 | 2.59 | **1.28** | 2.38 | - |
| `mu_spread` sd | - | - | 0.477 | **0.320** | - | - |
| `b_world` | 0.481 | 0.226 | imposed 1 | **imposed 1** | none | 0.606 |
| sampling | 1 divergence | clean | clean | **clean** | clean | **broken** |

The two rows that matter most are the band and `mu_spread`. The band halves, 2.59 to **1.28**,
and `mu_spread` finally moves off its prior, 0.477 to **0.320** against a prior sd of 0.500.
Those are the third window doing the one job it was added for. The amplitude row is the price:
3.66 back up to 4.12, worse than the two-window model though still short of the old default's
4.39.

**The forward abolishes negative r\*, and that is the largest substantive change it makes.**
Every other column here puts r\* below zero in 2016-2019 and 2020-2021; the new default puts
it at **+0.42** and **+0.14**. The reason is mechanical rather than a discovery: the deflated
5y5y forward never went negative, so a window loading directly on r\* will not let r\* go
there either. Whether that is the market disciplining a model that had drifted, or the market
refusing to price something the model was right about, this model cannot say. Sections below
written when the trough was negative ("Results by era" and the pre-COVID stance discussion in
in particular) were written against the two-window reading and should be read with this row in
front of them.

For scale on the amplitude row: CBA's published nominal neutral swings **3.15** points over
the same span, and the old default's 4.39 was the outlier among everything here.

**The headline is the flattening, not the endpoint.** Taking the premium as data pulls the
pre-GFC level down and lifts the trough hard: the peak-to-trough swing in r\* goes from 3.53
points to 2.29. Under the pin, r\* accounts for **35%** of the yield's variance rather than
79%. The three paths still correlate at 0.90 to 0.995, so the *shape* is robust. Its
amplitude is not, and neither is anything that depends on the amplitude.

**The 2016-2019 stance reading is the casualty.** It runs −0.49, −0.09, +0.36 across the
three. The notes already said that reading was not robust; this says the sign is decided by
the premium assumption.

**The level did NOT move toward the consensus, which was the expectation going in.** CBA's
September 2026 nominal neutral is 3.85 and the RBA's August 2026 range is 2.8 to 4.3. The
prediction before running was that stripping a falling premium would raise today's r\*.
It did not: 3.59 goes to 3.32 under the pin and 3.68 under the nominal window. The
flattening is real and the endpoint lift is not.

**`mu_spread` is still nearly its prior**: 0.333 [−0.560, 1.204] against N(0.25, 0.5). Same
outcome as `--us-premium`. The level remains asserted; what has improved is *what* is
asserted. Under the US pin the spread carried a liquidity difference against TIPS, a currency
risk premium and an inflation risk premium together. Under the AU pin it is an inflation risk
premium and nothing else, because both sides are the same country's curve.

### Why `b_world` is imposed rather than estimated

The standing diagnosis was that 0.481 is partly a premium-stripping coefficient rather than a
pass-through, so removing the Australian premium should push it toward 1. **It does the
opposite, and that is why the default now imposes it.**

| | `b_world` |
|---|---|
| old default (Cleveland anchor, latent premium) | 0.481 |
| `--au-premium`, Cleveland anchor | 0.226 |
| `--au-premium`, premium-stripped anchor | **0.015** |

Read 0.015 as a collapse, not a finding. "Almost no pass-through from world real rates" is
not credible for an open economy with a floating currency, and the mechanism is plain: a free
loading competes with a free random-walk wedge to explain the same variation, and once the
Australian premium is data the curve pins r\* directly and the loading has nothing left to
do. The notes already recorded the symptom before the pin existed, that a free `b_world` on
the market anchor "collapses to 0.233".

**Dropping the world anchor entirely confirms it.** `--au-premium --no-world` gives an r\*
path correlated **0.9999** with the free-loading market-anchor run, with a maximum difference
of **0.021** across all 135 quarters, and zero divergences. The world series was contributing
nothing.

That result is the reason imposing is preferred to estimating *and* to dropping. Imposing
makes `wedge` mean exactly what this package says it means, Australian r\* less the world
rate, with nothing free to absorb part of the difference. Dropping the anchor would make
`wedge` and `r_star` the same object and cost the package its organising idea for 0.021 of
r\*. Estimating publishes a number that means nothing.

**It is now an assertion, and should be read as one.** The "r\* is imported" premise used to
be put at risk by a free loading; it no longer is. What the data still say about it is in
`corr(r*, world r*)` = 0.76 and the era pattern of the wedge, not in a coefficient.

### The older, incomplete version of that test

Before the market anchor was tried, only this much was known: under `--nominal-window`
`b_world` rises to 0.606, the right direction and nowhere near enough. The test was
incomplete because the world anchor was still the raw Cleveland yield, which contains a US
term premium. The equivalent test on the *pinned* spec was run instead and is above; the
`--nominal-window` version of it has not been, and is not worth running while that spec does
not sample.

### `--nominal-window` does not sample, and the reason is instructive

**Do not use its numbers.** R-hat 1.020, `ess_bulk` **42** on the worst parameter, MCSE/sd
0.166, 21 divergences, 52.4% of transitions at maximum tree depth, BFMI 0.13.

The cause is visible in two parameters. `sigma_rn` collapses to **0.063** with an ESS of 373,
and `nu_walk` falls to **2.42** [1.56, 3.38], below the ν = 3.64 the notes already called
"the model straining", and with its interval reaching into the ν < 2 region where the
Student-t has no variance at all and `sigma_walk` stops being a standard deviation.

That is precisely the pathology this package documented and removed once before. The
abandoned `y = r* + tp + e_y` version collapsed `sigma_y` toward zero because r\* and a free
premium both explained one series and the noise had nothing to do. Here there is no free
premium, but there is still a free wedge, and a risk-neutral yield is very nearly an exact
function of `r* + k·g`. **The wedge can fit the series outright, so the residual again has
nothing to do, and the walk buys the room with fat tails.** Removing `mu_tp` removed the
level trade-off and replaced it with a `sigma_rn`-against-wedge one.

So the structural idea is sound and this parameterisation of it is not. What it would need is
the imposed variance moved off the wedge and onto the residual, or `sigma_rn` pinned from
AOFM's own estimation error rather than estimated. Neither is done.

---

## The model, equation by equation

Every number is from the current default run (`model_outputs/rstar_trace.nc`); intervals
are 94% HDIs unless stated.

### What is actually fitted

Five observed series, 1993Q1 to 2026Q3, no gaps. Nothing else enters the likelihood: the
corporate spread, the mortgage rate, inflation and the output gap arrive afterwards for the
derived series and are not allowed to shorten the sample.

| | series | source |
|---|---|---|
| `y` | indexed real 10y yield | RBA F2 |
| `w` | 10-year expected real rate, premium-stripped | Cleveland Fed via FRED `REAINTRATREARAT10Y`, less Kim-Wright |
| `r` | real cash rate | RBA F1 less inflation expectations |
| `au_tp` | Australian 10y term premium | AOFM decomposition, bias-corrected sheet |
| `f` | 5y5y risk-neutral forward, deflated | AOFM, `2 x RNY10 - RNY5`, less long-run expectations |

The last two arrived on 2026-09-16. `au_tp` pins the premium rather than leaving it latent
(next section); `f` is the third window, and is the only one of the five that speaks to the
level of r\* on its own.

**Why not Holston-Laubach-Williams for `w`.** HLW was the anchor until this vintage and is
still available via `--world-source mean`. It was dropped because it is not a price: it is
`r* = g + z` identified through the IS and Phillips curves, so a global bond selloff cannot
move it, and it did not. Between the COVID-QE era and the tightening era the HLW
three-country mean moved **−0.03** while the US 10-year TIPS yield rose 2.31 and the real
fed funds rate 2.62. Anchoring a market-read model on a macro-read estimate, produced by
the identification strategy this package rejects, was incoherent.

**What the Cleveland series is, and is not.** A ten-year expected real rate: the nominal
Treasury less their modelled expected inflation. It responds to monetary conditions, which
is the point. But it is a *yield*, so it contains a US term premium, and it is US rather
than global. Both matter, and `--world-source market` addresses the first (see Alternatives).

### 1. The state: the Australian wedge

```
wedge_t = wedge_0 + sigma_walk · sum_{s<=t} e_s,   e_s ~ StudentT(nu, 0, 1)
```

Carries everything about Australian r\* that is not imported. It is the only latent state;
everything else is data or an identity built on it.

| | prior | posterior |
|---|---|---|
| `wedge_0` | Normal(0, 2) | 1.504 [−0.435, 3.454] |
| `nu_walk` | Gamma(2, 2/6), mean 6 | **8.212** [2.459, 15.860] |
| `sigma_walk` | **imposed at 0.12** | not estimated |

**`nu` is the model's tell, and it is now healthy.** At `sigma_walk` = 0.08 it came back at
3.64, inside the infinite-kurtosis regime where single jumps are nearly free; under the
three-window variant it fell to 1.87, close to the ν = 2 line below which the Student-t has
no variance at all and `sigma_walk` stops being a standard deviation. Both were the model
buying with fat tails the room the variance denied it. At 0.12 with a market anchor, `nu`
returns to 8.2, above its prior mean of 6, and the walk stops straining.

**Do not read fat tails as a finding about Australia.** Earlier write-ups did. The wedge's
largest quarterly move is now +0.28 at 2022Q2, then −0.24 at 2012Q2, −0.21 at 1993Q3,
+0.18 at 2023Q2. There is no leap: the wedge is a smooth secular path, +1.21 through
1994-2007, crossing zero around 2012, bottoming at −1.07 in 2020-21, and back to +0.05.
That is a far more plausible object to call "Australia's structural spread over the world"
than the +0.82 jump the flat HLW anchor required.

### 2. How much of the world rate passes through

```
r*_t = b_world · w_t + wedge_t
```

| | prior | posterior |
|---|---|---|
| `b_world` | Normal(1, 1) | **0.481** [0.297, 0.662] |

**IMPOSED AT 1 SINCE 2026-09-16, so that posterior is no longer produced by the default.** It
is kept here because the argument below is why it had to be imposed, and because
`--no-impose-world-loading` still estimates it. Note what happens when it is free under the
current specification: it collapses to 0.015, which is not the finding "almost nothing is
imported" but a loading that is no longer identified against a free wedge once the premium is
pinned. The discussion that follows was written when 0.481 was the shipped number.

**This is the model's most uncomfortable number.** Taken at face value it says only half a
move in the world real rate reaches Australian r\*, which is hard to defend for an open
economy with a floating currency: it implies a differential that widens without arbitrage.

It is probably not a structural pass-through. The anchor is a ten-year real *yield*
containing a US term premium, and the AU observable is a ten-year real yield containing an
Australian one. Faced with two co-moving premia the model can either load fully on the
anchor and shrink its own premium, or damp the loading and let its premium absorb the
co-moving part. It chose the second, so `b_world` is partly a premium-stripping coefficient
rather than a pass-through. `--world-source market` removes the premium explicitly and is
the cleaner treatment (see Alternatives).

### 3. Window one: the long yield

```
tp_t = y_t - r*_t - k·g_t,      k = (1 - rho_g^H)/((1 - rho_g)·H),   H = 40
tp ~ stationary AR(1) about mu_tp
```

Posteriors below are from the SHIPPED specification, refreshed 2026-09-16 after the third
window went in. Under it `tp` tracks the AOFM series and only `tp_spread` is estimated, so
`mu_tp` is carried as a reported quantity rather than as the thing setting the level.

| | prior | posterior |
|---|---|---|
| `mu_tp` | Normal(0.75, 1) | 0.745 [−0.871, 2.356] |
| `mu_spread` | Normal(0.25, 0.5) | **0.241** [−0.394, 0.813] |
| `rho_tp` | TruncatedNormal(0.8, 0.2) on [0, 0.98] | 0.816 [0.724, 0.909] |
| `sigma_tp` | HalfNormal(1) | 0.284 [0.253, 0.317] |
| `sigma_f` | HalfNormal(1) | 0.156 [0.124, 0.192] |
| `forward_bias` | Normal(0, 0.5) | 0.141 [−0.470, 0.764] |
| `k` | computed from `rho_g`, not sampled | 0.179 [0.142, 0.222] |

`mu_tp` sits exactly on its prior mean with almost its prior width, which is the honest sign
that nothing in the likelihood speaks to it any more: the premium is data now. `mu_spread` is
the parameter that carries the level, and the third window is what moves it.

**Why it is an identity with no residual.** An earlier version gave the yield equation an
observation error. With r\* and `tp` both free to explain one series the noise had nothing
to do: `sigma_y` collapsed toward zero (mean 0.035, ESS 11, `r_hat` 1.30) and `mu_tp` rode a
ridge against the level of r\*, giving 76 divergences. Removing the redundant parameter
removed the ridge. The data enter through the AR(1) *prior* on the resulting series, which
is why the model carries `pm.Potential` terms and not likelihood terms.

**Stationarity is the identifying assumption.** r\* is a random walk and `tp` must revert to
a constant mean. Permanent against transitory is the entire mechanism splitting the yield.
Nothing else does this work, which is why the level rests on it.

**`k` is not a free parameter.** A long real yield is roughly the average expected real
short rate over its term plus a premium, so if the gap reverts with persistence `rho_g`, the
average puts weight `k` on today's gap. `k` is computed from `rho_g`, which the second
window estimates. Adding a maturity adds an equation without adding a knob.

### 4. Window two: the short rate

```
g_t = r_t - r*_t
g ~ stationary AR(1) about mu_g
```

| | prior | posterior |
|---|---|---|
| `mu_g` | Normal(0, 3) | 0.213 [−1.611, 2.079] |
| `rho_g` | TruncatedNormal(0.85, 0.1) on [0, 0.98] | 0.823 [0.765, 0.884] |
| `sigma_g` | HalfNormal(1) | 0.418 [0.365, 0.474] |

`g` is the policy stance, positive restrictive. It is an output, not an input: nothing here
asks the stance to move output.

**Why the second window exists.** From one series the starting point of a random walk and
the mean of a stationary process are not separable. The short rate speaks to r\* directly
rather than through a premium. It does not make the level free, since `wedge_0`, `mu_tp` and
`mu_g` are three levels against two observables, but it makes the assertion checkable, and
it makes `rho_g` estimable, which is what makes `k` computable. It also fixed the sampling:
the one-window model produces 168 divergences, the two-window model zero.

**The assumption to notice.** `g` is stationary about a single mean, which claims policy
cannot sit away from neutral indefinitely. That assumption does the work in every statement
this model makes about the stance.

**And `g` is the risk-free stance, not the one households faced.** See the pass-through
section below; the two diverge by up to 1.75 points between 2009 and 2022.

### 5. The policy rule, outside the likelihood

```
i* = r* + pi_core + 0.5·(pi_core - 2.5) + 0.5·ygap,   pi_core = pi - supply
```

Read from a completed `ystar_ustar` run. No prior, no posterior, cannot move r\*, the wedge
or the premium. It is also the only part of the package that depends on rates moving output.
Its inputs end a quarter before the market anchor does, so the rule's last quarter is
2026Q2 while the state runs to 2026Q3.

---

## The third window: the market's 5y5y forward

    f_t = r*_t + bias + e_t,    e ~ Normal(0, sigma_f)

`f` is the AOFM's 5y5y risk-neutral forward, `2 x RNY10 - RNY5`, deflated by long-run
expectations. Risk-neutral means AOFM has already removed the term premium, so unlike window
one there is no premium to split off, and unlike window two there is no policy gap in the way.
Five to ten years ahead the cycle should be over. That is the whole case for it: it is the
only thing this model observes that speaks to the LEVEL of r\* rather than to r\* + something.

It is on by default because it works, and the numbers are in "Read this first". What follows
is what it took to make it sample, because four attempts failed first and the record is worth
more than the conclusion.

### Why `nu_walk` must be fixed

The forward and `sigma_walk` disagree, and the size of the disagreement is the point:

| | quarterly wedge change, sd |
|---|---|
| two windows | **0.052** |
| three windows | **0.254** |

`sigma_walk` is imposed at 0.12 in both. Two windows want wedge movement well inside it; three
want about twice it. Left free, `nu_walk` collapses from 9.20 to **1.88**, below 2, where the
Student-t has no variance at all, because fat tails make the large jumps the forward demands
cheap. That collapse is the pathology: 22 divergences, BFMI 0.16, a quarter of transitions at
maximum tree depth.

Fixing `nu` high makes the jumps expensive, so the forward's high-frequency variation goes
into `sigma_f` instead, which is what an observation error is for. The sweep is monotonic:

| `nu` | `sigma_f` | sd(Δwedge) | divergences | BFMI |
|---|---|---|---|---|
| 2.4 | 0.102 | 0.247 | 5 | 0.32 |
| 6 | 0.140 | 0.213 | 0 | 0.65 |
| **9 (shipped)** | **0.156** | **0.196** | **0** | **0.74** |

9.0 is not a tuning choice. It is what this model's own free posterior picks when the forward
is OFF (9.20), so fixing it there estimates the tail behaviour from the data that can identify
it and then stops the third window distorting it. Note that `mu_spread`'s sd is 0.318-0.322
across all three rows: the level improvement does not depend on which `nu` you pick.

### Four reparameterisations that did not work

Recorded because each looked reasonable and the failures locate the problem.

1. **Non-centred wedge** (`noncentred_wedge`). Not retried: already on record at 511
   divergences against 12. Non-centring pays when the data are weakly informative about the
   latent and costs when they are not, and a third window makes them more informative, not
   less.
2. **Impose `forward_bias` at zero.** The argument was a parameter count: a free bias is a
   fourth level parameter (`wedge_0`, `mu_spread`, `mu_g`, `bias`) against three level-bearing
   observables, so the forward adds an observable and a parameter together and closes nothing.
   Result: **35** divergences against 22, `r_hat` 1.020, BFMI 0.16. The bias is not a spare
   wheel, it is a release valve: the forward genuinely disagrees with the other two windows
   about the level, and removing the valve pushes that disagreement onto the wedge walk.
3. **Raise `max_tree_depth` to 12.** Null. At a cap of 12 the deepest trajectory is still 10
   and mean depth moves 9.08 to 9.10, so nothing had been truncated: these trajectories U-turn
   at 10 of their own accord. The "8.4% at max" that prompted it was the diagnostic comparing
   against the deepest depth OBSERVED rather than the cap configured, which is the same number
   whether or not anything was cut off. That check now reads the configured cap from the
   trace; see `SamplerConfig.max_tree_depth`.
4. **Zero-avoiding prior on `sigma_f`.** Divergent draws sat at `sigma_f` 0.059 against a mean
   of 0.102, a −2.06 sd shift and much the largest of any parameter, so a prior with no mass at
   the boundary looked right. InverseGamma(3, 0.2), mean 0.10, gave **15** divergences against
   5 and BFMI 0.21 against 0.32: zero density at the boundary is bought with a sharp barrier
   beside it, and that curvature is worse than the smooth approach it replaced.

Attempt 4 settled something worth keeping, though. The posterior did NOT move under it:
`sigma_f` 0.102 to 0.093, `mu_spread` 0.232 to 0.237, r\* 1.005 to 1.001, so the likelihood
pins that residual and the low-`sigma_f` region is a real feature of the geometry rather than
something a diffuse prior invited. The divergences were the model reporting a genuine conflict
between the forward and the other two windows, not a prior artefact, which is why the fix that
worked was the one that changed the model's incentives rather than its priors.

---

## Results by era

| era | r\* | wedge | world | stance `g` | Taylor less actual | borrower stance |
|---|---|---|---|---|---|---|
| 1994-2007 | 2.06 | +0.74 | 1.32 | +0.80 | −0.95 | +2.50 |
| 2008-2011 | 0.76 | +0.54 | 0.22 | +1.02 | −0.24 | +3.12 |
| 2012-2015 | 0.02 | −0.16 | 0.18 | +0.03 | −0.63 | +2.64 |
| 2016-2019 | −0.11 | −0.76 | 0.65 | −0.60 | −0.61 | +2.46 |
| 2020-2021 | −0.70 | −0.97 | 0.26 | −1.03 | −0.32 | +2.44 |
| 2022- | 0.47 | −0.73 | 1.21 | +0.12 | +1.47 | +2.92 |

The decomposition attributes **48.6%** of the yield's variance to r\* and **20.6%** to the
premium, with `corr(r*, world r*)` = 0.76. Under the old default those were 79.2% and 8.2%,
and the reallocation is the point of the respecification rather than a side effect.
`sd(dr*)` is 0.15 against the world's 0.18. The wedge's range over the sample is 1.99,
against 2.81 before.

**The 2016-2019 stance reading is the one to treat as unsettled.** It runs −0.60 here,
against −0.21 on the old default, and across the specifications built on 2026-09-16 the
implied r\* for that era moved between −0.49 and +0.36, which is a change of sign. The notes
already said the pre-COVID stance was not robust; it is now known *what* it is not robust to.

---

## What the model identifies, and what it does not

**Identified: the wedge and the changes.** The wedge path is the model's real output, and
its current reading of zero is robust: it came back −0.02, +0.05 and −0.09 under three
quite different anchors.

**Not identified: the level.** `wedge_0` and `mu_tp` correlate at **−0.87**. The data pin
their sum, the level of the yield, not the split between "Australia's r\* sits above the
world's" and "the average term premium is large". The 90% interval on r\* straddles zero
throughout.

### The wedge is not an exchange rate story

Under real interest parity an AU-US real rate gap is expected real depreciation plus a
currency risk premium, and this model has nowhere to put either except the wedge. So the
wedge and the real exchange rate path are, in principle, observationally equivalent. No
exchange rate appears anywhere in the model, which makes this worth checking rather than
assuming.

There is a real relationship, and it is only a cyclical one. Regressing the wedge's
quarterly changes on log changes in the exchange rate gives a positive coefficient with the
sign parity implies, and it holds across both the real TWI and AUD/USD and both
specifications: `gamma` 0.377 to 0.584, t between 3.25 and 3.94. But R² is only 0.07 to 0.11,
and once cumulated the FX-attributed component spans just 0.28 to 0.34 points against a
wedge spanning 1.80 to 2.80, so 10 to 18 per cent of its range.

**The era pattern survives removing it.** Default spec, raw against FX-removed:

| | real TWI | AUD/USD |
|---|---|---|
| 1993-2007 | +1.20 → +1.16 | +1.20 → +1.21 |
| 2008-2015 | +0.24 → +0.02 | +0.24 → +0.14 |
| 2016-2019 | −0.86 → −1.03 | −0.86 → −0.88 |
| 2020-2022 | −1.00 → −1.17 | −1.00 → −1.01 |
| 2023-2026 | −0.27 → −0.47 | −0.25 → −0.23 |

Same sign sequence, same ordering, largest single move 0.22 and under AUD/USD never more
than 0.11. The headline decline does not weaken: first era to fourth is **−2.20 raw against
−2.33 FX-removed** on the real TWI and −2.22 on AUD/USD, so stripping the exchange rate
makes the fall marginally larger. The pin behaves the same way.

So the exchange rate moves the wedge around without taking it anywhere, and incorporating it
was rejected on that basis: a parity equation would import the PPP puzzle's unidentified
mean reversion, and the terms of trade plausibly drive the real exchange rate and the return
on capital together, so putting the rate on the right-hand side would attribute common
variation to FX by construction. Note also that the *level* correlation between the wedge
and the exchange rate is not usable in either direction: it runs −0.627 on the real TWI and
−0.111 on AUD/USD for the default, and −0.483 against **+0.084** for the pin, flipping sign
with the choice of series. The wedge is a random walk by construction, so a level
correlation against any persistent series is whatever the two trends happen to do.

### How much of the bond selloff is neutral rate?

The real yield rose 3.24 points from 2021Q4 to 2026Q3, and the model splits it: r\* +2.13,
premium +0.85, carried policy stance +0.24. So about two thirds is neutral.

That share is not identified either. Across a sweep of `sigma_walk` on the previous vintage
the r\* share ran from 28% at 0.03 to 70% at 0.20. What survives is the direction, and one
thing that does not depend on the split: the +0.24 of carried policy stance is a genuine
gain from the second window, which the one-window model set to zero by construction and
misallocated to the premium.

### The pre-COVID stance, and the standing critique

The objection to this package is Matt Cowgill's: through 2015-2019 inflation sat at or below
the bottom of the band, unemployment ran above most NAIRU estimates, and the cash rate was
between 0.75 and 1.5. If r\* says policy was well below neutral for five years, where was
the demand?

This vintage reads the 2015-19 stance at **−0.22**, against −0.73 on the published
one-window vintage. But that number moved between −0.89 and +0.37 across the specifications
built in one sitting, and it moved with settings nobody estimates, so it should be quoted as
a range and not a point.

**The stronger answer is not in this model at all.** See the pass-through section: measured
at the rate households actually paid, the stance barely moved across twenty years. Policy as
experienced was not meaningfully looser in 2015-19 than in the mid-2000s, so the absence of
a demand response needs no appeal to a broken IS curve.

### The QE finding did not survive

Earlier write-ups called the negative term premium through 2020-22 the strongest internal
validation the decomposition had produced: bond purchases showing up in the price of
duration, not in the natural rate, with nothing in the model marking those quarters.

It does not survive. The premium over 2020Q1-2022Q4 now averages **+1.02** and its sample
minimum is +0.04, in 2006Q3, not under QE at all. Across specifications it ran −0.05 with
one window and an imposed loading, +0.22 with a free loading, +0.58 with two windows, +1.02
with three. Only the original combination showed it. Note also that the mechanical `k·g`
correction the second window applies is small, so the shift is the re-identification of r\*
rather than the correction. **Do not repeat the published claim without this caveat.**

### The risk-free stance is not the stance households faced

`g` compares the real cash rate with a neutral rate read off a government bond. That is the
stance the economy faces only if the cash rate summarises the price of credit, and after the
GFC it stopped doing so.

| era | risk-free stance `g` | borrower stance | mortgage spread to cash |
|---|---|---|---|
| 2004-2007 | +1.22 | 2.46 | 1.24 |
| 2012-2014 | −0.20 | 2.39 | 2.59 |
| 2015-2019 | −0.89 | 2.10 | 2.99 |
| 2020-2021 | −1.39 | 2.07 | 3.47 |
| 2022- | −0.33 | 2.51 | 2.80 |

The risk-free stance swings 2.6 points across those eras. The borrower stance moves 0.4.
From 2014 to 2019 the cash rate fell 1.38 and the discounted owner-occupier mortgage rate
fell 0.70, so **half the easing never reached borrowers**.

And it was not margin. Banks' term deposit rates moved from 1.68 *below* the cash rate in
2004-07 to 0.39 above in 2015-19, a larger shift than the mortgage spread's 1.74, while the
90-day bill spread barely moved. The marginal funding dollar repriced when liquidity rules
pushed banks from cheap offshore wholesale funding toward competing for retail deposits.
`bank_costs` charts this in full.

`borrower_stance()` and `plot_borrower_stance` report it here. Levels are not comparable
between the two measures, since the borrower stance carries a credit spread; read the
changes.

---

## Alternatives, and why they are not the default

Every one of these is a switch, and each was tried as the default at some point.

**`--world-source market`: strip the US term premium from the anchor.** Cleveland less
Kim-Wright (`THREEFYTP10`) gives a world *neutral* rate rather than a yield: 1.30 over
1993-2007, 0.16 through the QE years, 1.31 now, so on this measure the world neutral rate
has fully recovered to its pre-GFC average and the rest of today's high yield is premium.
Conceptually the right object. It is not the default only because a free `b_world` on it
collapses to 0.233, so it must be run with `--impose-world-loading`.

**`--us-premium`: pin the term premium to the published US one.** The most defensible
treatment of the level, though it does not solve it. Instead of `tp` floating
about an unanchored `mu_tp`, it must track the Kim-Wright US premium and only the spread is
estimated, `mu_spread` = 0.329 [−0.588, 1.186] against a `N(0.25, 0.5)` prior. Run together
with `market` and `--impose-world-loading` (saved as `rstar_pin`) it gives r\* 1.22, a wedge
of −0.09 and an Australian premium 0.58 above the US, against the free-`mu_tp` version's
implausible +0.84 premium *and* −0.40 neutral. That combination is better argued than the free-`mu_tp`
version, but note what it does and does not achieve: `mu_spread` comes back 0.329
[-0.588, 1.186], an interval nearly as wide as its `N(0.25, 0.5)` prior. The level is still
asserted; the assertion has just moved to a quantity with a published benchmark behind it,
which is arguable on liquidity grounds rather than pulled from nowhere. Liquidity is not all
it carries, though: `mu_spread` is the average Australian premium over the US one, so it
also absorbs the **currency risk premium** a global investor demands for AUD exposure, and
the inflation risk premium left behind by subtracting a nominal US premium from a real
Australian one. The estimated spread does not track the currency (changes correlation
+0.009 against AUD/USD, +0.018 against the TWI), so this is a gap in the justification
rather than a demonstrated contamination, but the liquidity story alone understates what is
being asserted. That is an
improvement in accountability, not in identification, and it is why the two-window model on
a market anchor remains the default.

**`--no-au-premium`: infer the premium again, as the old default did.** `tp` stationary about
a free `mu_tp`. Kept as the comparator, and because the contrast against the pinned spec is
the evidence for the change. `--au-premium-source ols` switches the pinned series to the
plain ACM sheet.

**`--no-impose-world-loading`: estimate `b_world` rather than imposing it.** Retained as the
demonstration that it is not identified once the premium is data: it returns 0.015.

**`--no-world`: drop the world anchor entirely.** On the pinned spec this moves r\* by at
most 0.021 across 135 quarters and samples cleanly, which is the sharpest statement of how
little the anchor now does. Not the default because it collapses `wedge` into `r_star` and
the package's organising idea goes with it.

**`--nominal-window`: read the long end off the risk-neutral curve instead.** Removes the
term premium from the model entirely, and with it `mu_tp` and the −0.87 trade-off that is the
level problem. **It does not sample** (ESS 42, BFMI 0.13, `nu_walk` 2.42): the wedge can fit
a risk-neutral yield exactly, so `sigma_rn` collapses. Kept because the failure is
diagnostic, not because the numbers are usable.

**`--premium-source acm`: the other published US term premium. Tried and rejected.**
Adrian, Crump and Moench (2013) is the obvious second opinion on the premium that both
`--world-source market` and `--us-premium` subtract, and the flag governs both places at
once, since mixing providers across the two is uninterpretable. It is read from the NY Fed
directly because FRED does not carry it.

It fails on the anchor, before Australia enters. Cleveland less ACM puts the world neutral
rate at **−0.62 over 2008-2015 and +1.12 over 2016-2019**: a 1.7 point rise into the ZIRP
and QE era, with the sample trough in the mining-boom years. The derived series has an sd of
**1.08** against Kim-Wright's 0.61, and a neutral rate does not move like that. The
mechanism is visible in the inputs: ACM reads the post-GFC US premium at 1.27 against
Kim-Wright's 0.45, then turns it negative through 2016-2022, so the whole decline in the
10-year real yield lands in the expectations component instead of the premium. That is the
Bauer-Rudebusch-Wu persistence bias arriving as a sign error on the era pattern rather than
as noise.

Downstream it drags r\* with it, era means 1.56 / −0.16 / 1.02 / 0.78 / 1.53 against
2.13 / 0.70 / 0.21 / −0.17 / 1.06, and moves the stance by up to a point. The endpoint
barely notices (r\* 1.42 against 1.23, wedge +0.05 against −0.11). **The wedge is untouched
in shape**, 0.53 / 0.46 / −0.11 / −0.19 / −0.05, and that is the diagnostic: the model
behaved and the input did not.

**The triangulation is what settles it.** The default specification and the Kim-Wright pin
are identified quite differently, a free `b_world` on the raw Cleveland yield with `mu_tp`
floating against an imposed loading on a premium-stripped anchor with `tp` pinned to a
published series, and they agree: their r\* paths correlate at **0.966**, and the pin's
median sits inside the default's 90% band in **every quarter** of the sample. ACM correlates
at **0.442** with the published default, worse than its 0.618 against the pin, and falls
outside that band about a quarter of the time. It is the outlier against two independent
readings, not one of two defensible ones. Era means, default / pin-KW / pin-ACM:
2.42 / 2.13 / 1.56, then 0.56 / 0.70 / −0.16, then −0.52 / 0.21 / 1.02, then
−0.86 / −0.17 / 0.78, then 0.59 / 1.06 / 1.53.

Two honest qualifications. Kim-Wright *also* has the 2008-2015 trough and a rise after it,
so what fails here is ACM's level and amplitude, not the direction on its own. And nothing
in this says Kim-Wright is correct, only that ACM fails a check it passes. Kept as a switch
because it is the cleanest demonstration the package has that the **anchor**, not the
Australian data, sets this model's era pattern. ACM also produced 1 divergence against
Kim-Wright's 0.

**`--curve`: a third window on the belly.** Identifies the premium curve's slope
(`tp_slope` = 0.700 [0.026, 1.323]) and costs r\*: it drives r\* over 2016-19 from −0.29 to
−0.74 and pushes `nu` to 1.87. Forcing three points of one curve onto one state and one
stationary `g` leaves a persistent stretch of low real rates only one affordable home, and
that home is r\*. Kept as the cleanest demonstration of the absorption failure.

**`--short-rate bill`: the 90-day bank bill as the short rate.** Tried and reverted. BBSW is
a bank *credit* benchmark: decomposed against OIS, bill-minus-cash over 2015-19 was +0.26,
of which the expected policy path was −0.05 and bank funding +0.31, and through 2018 the
market expected nothing at all while the bill sat +0.49 over cash. It also breaks this
package's own rule that credit premia stay out of the state. The 30-day bill is worse: it
keeps the contamination and loses the anticipation.

**`--steps`, `--no-world`, `--no-short`, `--assert-stance`, `--impose-world-loading`,
`--deflator trimmed`** are retained comparators. `--no-world` drops pre-GFC r\* to 1.99 and
produces divergences: the Australian bond market alone does not locate r\*.

---

## Limitations

1. **`sigma_walk` is imposed at 0.12.** Chosen because `nu` is an internal check on whether
   the imposed variance is defensible and at 0.08 it was failing. Still a choice.
2. **The level is now partly identified, and this limitation has been downgraded.** It used
   to read "quoting the level is still wrong", on the grounds that `mu_spread` came back
   0.303 [−0.600, 1.200] against an N(0.25, 0.5) prior and the 90% interval on r\* straddled
   zero throughout. The third window changed that: `mu_spread` is 0.241 [−0.394, 0.813], a
   posterior sd of 0.320 against the prior's 0.500, and the band is 1.28 points and clear of
   zero. Pinning the premium bought accountability without identification; the 5y5y forward
   bought some of the identification. What is left of the limitation is that the level still
   leans on ONE observable, and that observable is not independent of window one, which is
   visible in `forward_bias` coming back +0.141 [−0.470, 0.764] with an interval barely
   narrower than its prior. Quote the level, quote the band with it, and do not pretend the
   forward has been interpreted.
3. **`b_world` is imposed at 1 and is no longer at risk.** Free, it collapses to 0.015 once
   the premium is data, so the "r\* is imported" premise is now an assumption rather than
   something this model tests. See Alternatives.
4. **The anchor is US, not world.** "World r\*" in the code and charts means a US series.
   A GDP-weighted market measure across the US, Euro Area and Canada is the honest version
   and does not exist here. This matters more now that the loading is imposed at 1: the wedge
   is defined as Australia's spread over *that* series.
5. **Indexed AGS are thin**, so the yield carries a liquidity premium a nominal bond does
   not, and the pinned spread absorbs it inseparably from the inflation risk premium.
5a. **The pinned premium is nominal and `tp` is real.** AOFM publishes no indexed
   decomposition at any vintage, so the inflation risk premium is what the estimated spread
   contains. That is one interpretable object rather than the three `--us-premium` bundled
   together, but it is not zero and it is not separately identified.
5b. **The premium is now someone else's model, entering as data with no residual.** AOFM's
   specification error lands in r\* invisibly. It passes the three checks that rejected US
   ACM (see above), which is why this is acceptable, but it is a dependency on an outside
   estimate that the old default did not have.
5c. **The AOFM series is re-estimated in full every month.** Every historical value moves
   when the file updates, so the published path will drift for reasons unconnected to any
   Australian data arriving. See Refinement 1: this is incompatible with a real-time
   exercise, which would otherwise put future data into the premium.
6. **The stance rests on `g` being stationary about one mean**, which is an assumption about
   policy, not a finding.
7. **The QE-era premium is specification-dependent**, as above.
8. **The headline is an endpoint** and nothing tests its stability. See Refinement 1.

---

## Refinements

### 1. Endpoint fragility, the one that matters most, and it now has a conflict

Nothing here tests it and the headline *is* the endpoint. `ystar` has `realtime.py`, which
re-estimates on progressively truncated samples. r\* is a random walk, so its last value is
the least constrained point in the sample. Until this exists, treat every current-quarter
number as provisional.

**The premium pin makes this harder, not easier.** AOFM re-estimates its decomposition over
the full sample every month, so the premium series available today embeds information from
after any truncation date. A naive `realtime.py` would hand the model a premium that knew the
future and report a stability it had not earned, which by this repo's standard is a test that
passes for the wrong reason. Doing it properly needs either archived AOFM vintages, which are
not published, or the truncated runs falling back to `--no-au-premium`, which tests a
different specification from the one shipped. Neither is satisfactory and the conflict is
unresolved.

**The third window makes it worse again, for the same reason twice over.** `f` is built from
the same AOFM decomposition, so it carries the same full-sample re-estimation. And because it
loads directly on r\* it pins the endpoint harder than either of the other two windows, so a
truncated run would inherit more look-ahead at exactly the point the exercise is meant to
test. Any real-time work here has to start by deciding what a 2015-vintage 5y5y forward would
have been, and AOFM does not publish that.

### 2. Promote the pinned specification: DONE 2026-09-16

Shipped as the default, together with the premium-stripped anchor and an imposed `b_world`.
What remains of the original item: the charts still say "world r\*" for a US series
(Limitation 4), and the term-premium language throughout could be tightened now that the
estimated quantity is a spread rather than a level.

### 3. A GDP-weighted market anchor

The current anchor is one country. Building the Euro Area and Canadian equivalents and
weighting them would make "world" true rather than approximate.

### 4. Reconcile the three routes

`rstar_hlw` Resolution G gives 2.23, this gives 1.08, `rstar_rba` gives 0.99 real on
its own saved run. All three are in this repo and none has been reconciled with the others.
That comparison is the most useful thing left to write.

### 5. Out of sample

The 2016-19 and present-day Taylor readings matched a judgement stated in advance, which is
one observation. A second, stating the expected reading before running it, would be worth
more than any in-sample diagnostic.

### 6. Smaller things

- The corporate spread starts 2005Q1, so `r*` for firms is blank for the first twelve years.
- `end_break_check()` uses a fixed eight-quarter window and a one-sd threshold, both picked
  by eye. Currently +0.63 sds.
- The Taylor coefficients are Taylor's originals and unswept.
- The spec printout in `estimate.py` states the gap without the band scaling used elsewhere.

---

## Files and usage

```
src/models/rstar_bonds/
├── config.py         # ModelConfig: sample, anchor, windows, sigma_walk, nu_walk, the rule
├── observations.py   # the yield, the short rate, the anchor, the AOFM premium and
│                     #   5y5y forward, plus ragged extras
├── estimate.py       # builds and samples the PyMC model
├── results.py        # RStarResults: posteriors, derived series, diagnostics
├── analyse.py        # charts and printed diagnostics
└── run.py            # CLI
```

Run order: `expectations` → `ystar_ustar` → `rstar`. r\* itself needs no upstream model;
only the Taylor rule does, and the affected charts are skipped with a note if inputs are
missing. The anchor needs a FRED API key in `fred.api` (gitignored); see the README.

```bash
./run-rstar-bonds.sh -v
./run-rstar-bonds.sh --analyse-only                 # recharts from the saved trace
./run-rstar-bonds.sh --no-forward                   # the two-window model: tighter amplitude, unidentified level
./run-rstar-bonds.sh --nu-walk 2.4                  # the low-nu variant: 5 divergences, BFMI 0.32
./run-rstar-bonds.sh --nu-walk 6                    # the middle of the sweep: 0 divergences, BFMI 0.65
./run-rstar-bonds.sh --world-source mean            # the old HLW anchor
./run-rstar-bonds.sh --sigma-walk 0.08              # watch nu fall into the strain regime
./run-rstar-bonds.sh --curve                        # a DIFFERENT third window: watch r* absorb the stance
./run-rstar-bonds.sh --short-rate bill              # the rejected bank bill
./run-rstar-bonds.sh --no-short                     # one window: 168 divergences, for the record
./run-rstar-bonds.sh --steps                        # the asserted-break comparator
./run-rstar-bonds.sh --no-world                     # does the anchor do the work? (yes)
```

Charts land in `charts/RStarBonds/`: r\* against the real yield and the anchor, real and nominal
r\* with bands, the Australian wedge, the policy stance on both neutral definitions, the
policy gap, the risk-free stance against the borrower stance, the term premium with and
without the `k·g` correction, r\* for firms, and the Taylor rule. A non-default prefix gets
its own directory.
