# u\*: a NAIRU from one Phillips curve

## The choices, up front

```
sample      1993Q1 onwards
u* path     natural cubic spline, one interior knot at 2013Q1
equations   price Phillips curve only. The gap-form Okun equation is OFF
anchor      2.5% target, flat, with expectations entering as a deviation from it
```

Four things are asserted rather than estimated: the start date, the absence of
Okun, the spline, and the knot date. Everything else is estimated. Each is
argued below, but the spline deserves its reason here, because it is the
choice that decides what the model is allowed to say about now.

**The spline is preferred because it lets the endpoint rise if the data
support it.** The alternative, a decay toward one equilibrium, cannot: the
sign of `phi x (eq - u*)` is fixed by which side of the equilibrium the state
opened on, and from 10.75 in 1993Q1 with an equilibrium near 4.9 it approaches
from above, never crosses, and can only ever report a fall. Every decay run
returns -0.32 to -0.34 over 2015-2026 whatever the data say. A spline has no
such constraint and does turn up when pushed, +0.36 in one setting tried.

So the default's post-2015 slope of -0.06, which is flat, is flat because the
data put it there. Under the decay law that reading was unavailable, and the
continuing decline it reported was a property of the curve.

Latest reading **4.70**, unemployment 4.35, gap **-0.35**, mean 90% posterior
band 0.58. Do not quote anything before 2000.

### What u\* is here

One observation equation, and nothing else. Invert it:

```
u_t - u*_t  =  - u_t x (pi_t - pi^e_t) / gamma
```

**u\* is whatever makes the unemployment gap account for the inflation gap.**
That is the definition operating here, not a by-product of it. No
labour-market observable enters except `u`, which sits on both sides, so the
path is inflation's deviation from expected inflation, rescaled by `u/gamma`,
smoothed and given a shape.

Two things follow and are worth holding onto while reading the rest. Every
argument below about the start date, Okun, the spline and the knots is an
argument about the smoothing and the scaling, never about adding information.
And where inflation sits at expectations, the equation has nothing to say and
u\* is whatever the smoothing puts there, which is what makes the 1990s the
problem they are.

---

## The start date is forced, and it is the model's central problem

The sample cannot begin before 1993Q1, because the output gap this package
supplies is defined against a 2.5% target that did not exist earlier. That is
a hard constraint, not a preference.

**The disinflation is outside the sample.** Year-ended trimmed mean inflation
fell 1.82 points through 1991 and 2.30 through 1992, and was flat by 1993.
So the model opens on a quiet nominal picture sitting beside an unemployment
rate of 10.9, and concludes, correctly given what it can see, that this is
normal. By 1993 the level signal is -0.37 and the change signal -0.45; at any
plausible slope both imply u\* between 9.9 and 10.6.

**It is not the initial prior.** `ustar_init` is centred on the first
quarter's unemployment rate, which looks like the culprit and is not. Sweeping
that centre from 6.0 to 10.85, three prior standard deviations, moves the
posterior from 10.75 to 10.73 and changes `phi`, `ustar_eq`, the 1993-98 mean
and the latest reading by nothing at two decimals. The likelihood pulls it
back every time.

**So 1993-1999 is shaded on every chart.** Across 1993-98 the Okun
specifications put u\* at 8.69 against an unemployment rate of 8.90, a
reported gap of -0.22 through the deepest slack in the sample. The deviation
between the fitted u\* and what inflation alone implies averages -0.62 to
-0.76 across that window for every specification that keeps Okun, and -0.15
without it. Two independent diagnostics, the level and the bias, point at the
same seven years.

---

## The gap-form Okun problem: there is no independent output gap to anchor on

The equation is `u = u* - beta x ygap + e`, which rearranges to
`u - u* = -beta x ygap`: the unemployment gap against the output gap. That is
the **gap form** of Okun's law, not Okun's original, which relates the change
in unemployment to output growth. Two things follow from the distinction.
`beta` is not comparable with a textbook Okun coefficient of 0.3 to 0.5, which
`config.py` already warns about. And it is the gap-on-gap structure
specifically that lets the equation collapse into a Phillips curve.

On paper it is what sets u\*'s level. In practice it cannot, because the gap
it is given is not an output measurement.

**The supplied gap is inflation.** Regressing `ystar`'s defined gap on
year-ended trimmed mean inflation less 2.5, through the origin, gives a
coefficient of 0.1882 with **R² = 1.0000 and a maximum residual of 0.0000**.
The gap *is* `0.1882 x (pi - 2.5)`, identically. Substituting:

```
u_t  =  u*_t  -  beta_okun x 0.1882 x (pi_t - 2.5)  +  e_o
```

which is a Phillips curve in levels. GDP enters only through `c`, and `c` was
itself fitted to inflation. So the model had two observation equations and one
signal, and the Okun equation carried no output information at all.

**Three measured consequences**, each from running it both ways:

| | with Okun | without |
|---|---|---|
| mean 90% band | 0.35 | 0.58 |
| 90% band over 1993-98 | 0.36 | 1.15 |
| bias against the implied series | -0.13 to -0.23 | -0.00 |
| u\* over 1993-98 | 8.69 | 7.14 |

The band nearly doubles because one signal was being counted twice. The bias
disappears because Okun was holding u\* above what inflation implied. And the
1990s fall a point and a half, which is the difference between claiming
equilibrium at 10.9% unemployment and not.

**Why Okun won when it was in.** The defined gap enters as exact data, so
`u* = u + beta x ygap` is effectively an identity, while the Phillips residual
is worth about 1.5 points of u\*. One equation was nearly dogmatic about the
level and the other vague, so the vague one lost.

**Swapping the gap does not help.** `--gap-source actual`, `log_gdp - y*`,
carries genuine uncertainty and a different path, and changes u\* over 1993-98
by 0.01. Australia's measured output gap in 1993 is -0.25 per cent on that
basis and -0.07 on the defined one. Reaching u\* of 7 needs a 1993 gap of
**-1.80**. No gap series here says the early 1990s were a period of deficient
demand, because potential is estimated from inflation and inflation was at
target.

**Okun solves one problem and brings another.** It is the only way to make
u\* come down through the 1990s, which was a regime change from high
inflation to low taking years to work through the labour market: unemployment
fell 4.53 points over 1993-1999 and with Okun u\* falls 3.43 to 3.95 with it,
against 0.50 to 0.81 without. Excluding it leaves the model asserting that
almost the whole descent was cyclical and the equilibrium barely moved, which
is a strong claim in its own right.

What it brings in exchange is everything in the table above: a band halved by
counting one signal twice, a systematic bias against what inflation alone
implies, and a level in 1993-98 that calls the deepest slack in the sample
equilibrium. The default takes that trade, and the cost of taking it is the
flat 1990s. `--compare` charts one Okun setting alongside the default so the
choice stays visible (see "Comparing specifications").

`--okun` restores the equation.

---

## The spline, and why not the decay law

The alternative, `--state converge`, lets u* decay toward one equilibrium:
`u*_t = u*_{t-1} + phi (eq - u*_{t-1}) + e`. It has two defects.

**It can draw only one shape.** A monotone approach to a single equilibrium,
so it cannot decline and then stop, and it reports a decline that never ends:
-0.32 over 2015-2026 while the Phillips signal over the same years oscillates
between 4.3 and 5.3 with no trend. Roughly 80% of its total fall happens
before 2005 and what continues is `phi x (eq - u*)` still running.

**And it cannot turn up at all.** The sign of `phi x (eq - u*)` is fixed by
which side of the equilibrium the state is on. Opening at 10.75 with `eq` at
4.86, u* approaches from above and never crosses, so the decay law can only
ever report a fall, whatever the data say. Every decay run here returns -0.32
to -0.34 over 2015-2026.

A spline has no such constraint and will turn up when the likelihood pushes
it: the one-knot spline with Okun returns **+0.36** over the same years. That
is the property that matters most for the endpoint, because "has u* stopped
falling, or started rising?" is a question only a specification capable of
answering yes can be asked. The current default answers -0.06, which is flat,
and it is flat because the data put it there rather than because the
functional form insisted.

**Both ends are the same defect.** Differencing the decay against the
one-knot spline: +1.52 at 1993Q1, -0.02 across 2005-2012, -0.10 at the
endpoint. Above at the start, indistinguishable through the middle, below at
the end. One exponential cannot be steep in the 1990s and flat in the 2010s,
so it splits the difference and the residual appears at both ends. The high
opening and the endless decline are not two problems: the curve has to start
high to have room to fall, and having chosen a rate it cannot stop. The
spline is free at each end independently, which is why it gives better start
and end points rather than just a better endpoint.

**Its stochastic part is decorative.** The fitted path's entire 4.20-point
fall is 4.28 points of deterministic decay from three numbers. The 134
innovations contribute a maximum deviation of 0.07 and an sd of 0.02. So
despite appearances it is a three-parameter exponential, and `sigma_ustar`,
imposed at 0.020 and measured by nothing, moves the path by 0.07. Sweeping it
from 0.020 to 0.150 moves the 1993-98 mean by 0.03.

The spline is also three coefficients, so this is not a trade of stiffness for
flexibility. It is a trade of shape families: exponential-to-an-asymptote
against cubic, and only the second can change slope. It also removes
`sigma_ustar` entirely, since u\* is deterministic given its coefficients.

`sigma_ustar`, `phi_ustar`, `ustar_eq` and `ustar_init` exist only under
`--state converge`.

---

## The knot dates

**2013Q1**, one interior knot. It marks the start of the low-inflation era and
it is where the decline stops: the post-2015 slope turns from -0.32 under the
decay law to -0.06.

**2008Q1 was tried and is 4 quarters worse** on the band test, 85.9% against
92.2%, though the two give near-identical endpoints and coefficients (8.42,
3.62, 5.06 against 8.13, 3.50, 5.06). With one interior knot and the natural
boundary conditions there are only three coefficients, so the curve is nearly
determined and the knot only nudges where its single bend sits. The knot
*count* matters far more than the date.

**A second knot at 1996Q1 is the interesting case, and it depends on Okun.**

With Okun in, it is a clear loss: 79.7% on the band test, 1993-98 back up to
8.74, post-2015 slope -0.59. The extra freedom near the start lets the curve
begin at 10.15 and glide, and it reproduces the decay law almost exactly,
correlation **0.9957** and mean absolute difference 0.112. The band even
narrows to 0.24, which is the familiar warning sign.

**Without Okun the second knot is a robustness check, and the model passes
it.** The extra knot gives the curve a fourth coefficient and licence to bend
anywhere around 1996, and the data decline to use it: 0.05pp difference over
the whole sample, 0.03pp at the endpoint, band test 84% either way, post-2015
-0.06 against -0.13. Given the freedom to bend in the mid-1990s, nothing in
the inflation record asks for one.

That is the stronger reading of the agreement. It is not that the two
specifications happen to coincide; it is that the extra flexibility was
available and went unused, which is what robustness to a modelling choice
looks like. The one cost is a wider posterior, 0.826 against 0.578, and
somewhat lower sampling efficiency, min ESS 2034 against 4508, both of which
are the price of a coefficient that earns nothing.

It also sharpens what Okun is doing. With the equation in, the same second
knot **is** used, and heavily: the curve opens at 10.15 instead of 7.57. So
the bend at 1996 is not something the data ask for. It is something the Okun
equation asks for.

**What the second knot buys, with Okun, is 1995.** Inflation ran at 3.65 and
4.06, the clearest above-band episode of the decade, and the one-knot curve
calls slack and misses both quarters while the two-knot with Okun calls tight
with gaps of -0.76 and -0.74. It pays for that with five misses across
1999-2002, where it still has u\* above the unemployment rate while inflation
ran over the band.

One knot is the default on parsimony: the same answers as two without Okun,
and a band of 0.58 against 0.83.

---

## Strengths and weaknesses of each specification

`./run-ustar.sh --compare` charts the three of these kept for comparison; see
"Comparing specifications".

| | band test | 1993-98 | post-2015 | latest | band | bias |
|---|---|---|---|---|---|---|
| Decay, with Okun | 82.8% | 8.68 | -0.32 | 4.83 | 0.25 | -0.141 |
| Decay | 81.2% | 7.77 | -0.34 | 4.60 | 0.55 | -0.004 |
| Spline 1 knot, with Okun | **92.2%** | 7.93 | +0.36 | 5.06 | 0.35 | -0.229 |
| **Spline 1 knot** (default) | 84.4% | 7.14 | -0.06 | 4.70 | 0.58 | -0.000 |
| Spline 2 knots, with Okun | 79.7% | 8.74 | -0.59 | 4.67 | 0.24 | -0.128 |
| Spline 2 knots | 84.4% | 7.30 | -0.13 | 4.67 | 0.83 | -0.003 |

**Decay with Okun**: the narrowest band and the best correlation with the
implied series, and both are artefacts. Claims equilibrium at 10.9%
unemployment in 1993 and declines forever.

**Decay without Okun**: unbiased, sensible 1990s, but still cannot stop
declining. The best of the three on closeness to the implied series, 0.911
against 1.133 and 1.086.

**Spline 1 knot with Okun**: the best band test of any specification, and it
is the only one that turns u\* *up* after 2015. Biased by -0.229, the worst of
the six.

**Spline 1 knot**: middle on everything except closeness, where it is worst,
for the reason that it moves least. Unbiased, flat after 2015, narrower band
than the two-knot version.

**Spline 2 knots, with Okun**: reproduces the decay law. No reason to prefer
it over the decay law itself.

**Spline 2 knots**: equal to one knot on every headline, better on 1995,
worse on 1999-2002, and the widest band of the six.

### How to read the table, and how not to

`sd(implied - u*)` and the correlation with the implied series are **not**
selection criteria, though they are reported. Both are maximised by *being*
the implied series, which swings between -20 and +33 under some settings, and
a stiffer curve deviates more by construction. The band width is an output,
not a virtue: narrower is better only at equal information, and these differ
in information.

The band test is the only column that separates them, and it is not
independent evidence, since every specification is fitted to the inflation
series the test scores against. It also weights a 0.24 breach of the band as
heavily as a 4-point one. Three of the default's five misses are 0.24
breaches; only 2013Q4 is a genuine disagreement.

A real ranking would need something outside the fitted sample, which means a
recursive real-time exercise of the kind `ystar/realtime.py` runs. That has
not been done here.

---

## The band, and why the charts draw it twice as wide

**Every band figure in these notes is the raw posterior**, including the 0.58
above. **Every chart draws it doubled**, so the same run shows 1.16 on screen
and 1.25 at the endpoint. That is deliberate, and the footer says so, but the
two numbers have to be reconciled by anyone comparing them.

The widening stands in for uncertainty about the **imposed structure**, which
the posterior cannot express. u\* is a spline with a knot placed by hand, so
the interval answers "where is u\* given this shape" and says nothing about
whether the shape is right. Under `--state converge` the imposed thing is
`sigma_ustar` instead, and the same argument applies to the drift rate.

**The factor of 2 is inherited, not derived for the spline.** It was
calibrated against the decay law's `sigma_ustar` sweep: the conditional band
was 0.48pp at the endpoint, u\* moved 0.39pp across `sigma_ustar` from 0.024
to 0.05, and the union of the conditional bands over that range was about
0.89pp, which doubling reproduced.

No equivalent calibration exists here, and the two obvious candidates
disagree. Varying the knot count moves u\* by 0.05pp over the sample, which
would argue for well under 2. At 1993Q1 the spread across specifications is
2.87pp, which would argue for far more. So the convention errs wide in the
settled part of the sample and nowhere near wide enough in the early part,
where the shaded window carries the warning instead.

It has not been re-derived, and probably should not be. The recipe would
transfer mechanically, sweeping the knots and taking the union, but it would
capture only uncertainty about where the knots go, not about the choice of a
cubic spline at all. The shape family is the larger imposition and cannot be
swept, so a re-derived number would look more rigorous than it is.

---

## What every specification shares, and therefore cannot test

The identity above holds in all of them, so none is a second opinion on any
other. What differs is only how the same signal is smoothed.

The deviation between the fitted u\* and what inflation alone implies is
autocorrelated at +0.23 to +0.34 in every specification, so a persistent
component is unmodelled throughout.

On the size of the uncertainty, the order matters and is easy to get
backwards. Across the three specifications that exclude Okun the spread is
**0.21pp** over the sample and 0.10pp today, against a mean 90% band within
any one of them of **0.65**: the estimation uncertainty is the larger term and
the choice of path barely matters. Pooling in the Okun settings raises the
spread to 0.55pp, but that gap is the distance between two readings of the
1990s rather than error, so it should be quoted as a range with each end
named. See "Comparing specifications".

The exception is the early sample, where the spread reaches 3.47pp at 1993Q1.
There the specification is the whole of the answer.

---

## Comparing specifications (`--compare`)

`--compare` runs this model three ways and charts the results together. It is
not a different model: each specification is a set of this model's own flags,
re-estimated only if its saved run is not from today. One is the default run
itself. Every run then writes its own charts, the default to `charts/UStar/` as
a plain run does and the others beside it (`charts/UStar-k2/`,
`charts/UStar-k2_okun/`), and the combined charts follow.

| specification | knots | Okun |
|---|---|---|
| **Default run** (spline, 1 knot) | 2013Q1 | out |
| Spline, 2 knots | 1996Q1, 2013Q1 | out |
| Spline, 2 knots, with gap-form Okun | 1996Q1, 2013Q1 | in |

**Why these three.** The knot count asks how much the flexibility allowed to u\*
matters. The Okun setting is kept because it is the only one in which u\* comes
down through the 1990s. That decade was a regime change, inflation moving from
high to low and taking years to work through the labour market, and the default
reads almost all of the fall in unemployment over those years as cyclical. The
Okun run reads much of it as a fall in u\* itself. Neither reading is settled,
and the comparison exists so the choice is visible rather than buried in a
default. The same Okun run also claims the steepest recent decline in u\* of
anything tried; its 1990s reading lends that claim no weight.

**Why no decay settings.** Under the decay law the sign of u\*'s movement is
fixed by which side of its equilibrium it starts on, so from the high
unemployment of 1993 it can only ever report a fall. On a chart about how much
the specification matters, that shape would be read as evidence. Every spline
here can turn u\* up at the end if the data warrant it.

**How to read it.** The three share the sample, the Phillips curve, the
expectations series and the inflation measure, so their agreement is close to
arithmetic and only their disagreement informs.

- The knot count barely matters once Okun is out.
- The spread is widest in the early 1990s, where u\* is least identified, and
  has all but closed today. The choices argued above bear on the 1990s
  narrative, not on the number to quote now.
- The spread is not an error band. The specifications differ in a structured
  way, whether Okun is in, so the range is the distance between two readings of
  the 1990s. Quote the range and name what sits at each end.
- The mean line on the range chart is a mean, not a median (with three series
  the median switches identity wherever lines cross), and it describes where
  the specifications sit; it is not an estimate.

**What it prints and charts.** Each run's own full set of charts, as above, then a table per specification (the inflation band
test, the 1993-98 level, the post-2015 slope, the latest value, the average 90%
band, bias against the inflation-implied series, and the volatility of u\*),
the spread with and without Okun, and four charts in `charts/UStar-compare/`:
the three u\* paths against unemployment (colour for the knot count, dashes for
Okun), the unemployment gap each implies, the range with its mean, and the
range's width over time. The poorly identified 1993-99 window is shaded.

## Files and usage

```bash
./run-ustar.sh                                   # the default above
./run-ustar.sh --okun                            # restore the Okun equation
./run-ustar.sh --ustar-structure decay           # the decay law
./run-ustar.sh --knots 1996Q1 2013Q1             # a second knot
./run-ustar.sh --gap-source actual               # log_gdp - y* instead of the defined gap
./run-ustar.sh --analyse-only                    # re-chart a saved run
./run-ustar.sh --prefix name                     # write somewhere other than `ustar`
./run-ustar.sh --compare                         # three specifications, each charted, then combined
./run-ustar.sh --compare --analyse-only          # the same, from the saved runs as they stand
```

The comparison lives in `compare.py` (the specifications) and `compare_charts.py`
(the table and charts). `cli.py` holds the flags, shared by the default run and
the comparison.

`config.py` holds every imposed quantity and records it in `constants`, saved
beside the trace. Charts and this run's diagnostics go to `charts/UStar/`.
Prior-posterior charts come from `common/prior_posterior.py`, shared with
every Bayesian model in the package.
