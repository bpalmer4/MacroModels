# u\*: a NAIRU from one Phillips curve

## The choices, up front

```
sample      1993Q1 onwards
u* path     random walk, step size 0.3 tapering to 0.10 by 2002Q1, flat after;
            start prior N(8, 5)
            (alternatives: the stepped and flat walks; a natural cubic spline,
            one interior knot at 2013Q1)
equations   price Phillips curve only. The gap-form Okun equation is OFF
phillips    hybrid: last quarter's inflation (backward-looking expectations)
            beside trend expectations (--no-inflation-lag drops it)
anchor      2.5% target, flat, with expectations entering as a deviation from it
```

Asserted rather than estimated: the start date, the absence of Okun, and the
random walk with its step-size schedule. Everything else is estimated.

**The random walk is preferred because it imposes the least structure.** It
fixes only how far u\* may move each quarter and leaves the shape to the data.
The one-knot spline decides where u\* may bend. This is a judgement, not a
score: the spline does slightly better on the band test, and fit cannot
arbitrate because a looser walk always fits better.

**Either is preferred to the decay law, because both let the endpoint rise if
the data support it.** A decay toward one equilibrium approaches from the side
it starts on and never crosses, so from the high unemployment of 1993 it can
only ever report a fall, whatever the data say.

Do not quote anything before 2004: see the start-date section.

### What u\* is here

One observation equation, and nothing else. Invert it:

```
u_t - u*_t  =  - u_t x (pi_t - pi^e_t) / gamma
```

**u\* is whatever makes the unemployment gap account for the inflation gap.**
That is the definition operating here, not a by-product of it. No
labour-market observable enters except `u`, which sits on both sides, so the
path is inflation's deviation from expected inflation, rescaled by `u/gamma`,
smoothed and given a shape. Here `pi^e` stands for the whole baseline the curve
expects: the target, last quarter's inflation and trend expectations, weighted
as in the equation below, with supply shocks netted off.

Two things follow and are worth holding onto while reading the rest. Every
argument below about the start date, Okun, the spline and the knots is an
argument about the smoothing and the scaling, never about adding information.
And where inflation sits at expectations, the equation has nothing to say and
u\* is whatever the smoothing puts there, which is what makes the 1990s the
problem they are.

### The Phillips curve

A hybrid Phillips curve, on quarterly rates:

```
pi_t = q(2.5) + lambda x [pi_t-1 - q(2.5)] + (1 - lambda) x beta x [q(pi^e_t) - q(2.5)]
       + gamma x (u_t - u*_t) / u_t + rho x d4pm_t + xi x GSCPI_t^2 + e_t
```

In words: inflation this quarter is the target, plus a share `lambda` of how
far last quarter's inflation was from target, plus the rest of the weight on
how far trend expectations sit above target, plus slack, plus import prices
and supply-chain pressure.

"Backward-looking expectations" is the name the charts and notes give the
`lambda` term; the code calls it the inflation lag (`inflation_lag`,
`lambda_pi`). Trend expectations are the expectations model's unanchored
series. The lag and the target share one unit of weight, so with no slack, no
excess expectations and no supply shock inflation settles at target. What the
lag adds is persistence: inflation above target carries into the next quarter,
and only slack works it off. `--no-inflation-lag` drops the term and gives the
expectations term its full weight.

The inflation decomposition chart stacks these terms against observed
inflation, each bar coloured as labelled in the equation beneath it:
backward-looking expectations in light purple, trend expectations in purple.

---

## The start date is forced, and it is the model's central problem

The sample cannot begin before 1993Q1, because the output gap this package
supplies is defined against a 2.5% target that did not exist earlier. That is
a hard constraint, not a preference.

**The disinflation is outside the sample.** Inflation fell sharply through
1991-92 and was flat by 1993, so the model opens on a quiet nominal picture
beside unemployment near 11 per cent, and concludes, correctly given what it
can see, that this is close to normal. The start prior is not the culprit:
moving its centre across a wide range leaves the posterior where it was.

**So 1993-2003 is shaded on every chart.** Through the 1990s the model's u\*
is set by the smoothing, not the data. The problem recurs over 2002-04, when
unemployment fell fast while inflation eased, and in the 2001 slowdown, when
the falling dollar lifted inflation and reads as tightness. Both ask u\* to
move further and faster than the walk allows, so its level through 2003
depends on where the sample starts: runs from 1993, 2000, 2002 and 2004
disagree materially in 2004 and agree closely only from about 2010.

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

**The supplied gap is inflation.** The defined output gap is an exact multiple
of inflation's deviation from target, so substituting it into Okun gives

```
u_t  =  u*_t  -  beta_okun x c x (pi_t - 2.5)  +  e_o
```

a Phillips curve in levels. The model would have two observation equations and
one signal, and the Okun equation no output information at all.

**With Okun in, three things go wrong.** The band narrows because one signal is
counted twice; u\* sits systematically above what inflation alone implies; and
because the gap enters as exact data while the Phillips residual is vague, Okun
dictates the level and calls the deepest slack of the 1990s equilibrium.

**Swapping the gap does not help.** `--gap-source actual`, `log_gdp - y*`,
barely moves u\*. No gap series here says the early 1990s were a period of
deficient demand, because potential is estimated from inflation and inflation
was at target.

**The trade.** Okun is the only way to make u\* come down with unemployment
through the 1990s; without it the model says most of that descent was cyclical,
a strong claim in its own right. The default accepts that cost rather than
Okun's. No Okun setting is kept in `--compare`: with it, the path swept through
the sample in wide bends and was not credible.

`--okun` restores the equation.

---

## The spline, and why not the decay law

`--ustar-structure decay` lets u\* decay toward one equilibrium:
`u*_t = u*_{t-1} + phi (eq - u*_{t-1}) + e`. It has two defects, and they are
one defect.

**It can draw only one shape.** A monotone approach to a single equilibrium: it
cannot decline and then stop, so it reports a decline that never ends, and it
cannot turn up at all, since the sign of `phi x (eq - u*)` is fixed by which
side of the equilibrium the state starts on. One exponential cannot be steep in
the 1990s and flat in the 2010s, so it starts too high and ends too low.

**Its stochastic part is decorative.** Almost all of its path is the
deterministic decay from three numbers; the innovations barely move it, so the
imposed `sigma_ustar` is measured by nothing and matters little.

A spline is also three coefficients, so the choice is not stiffness against
flexibility but shape families, and only the cubic can change slope. That is
what matters for the endpoint: "has u\* stopped falling, or started rising?" can
only be asked of a specification able to answer yes.

`sigma_ustar`, `phi_ustar`, `ustar_eq` and `ustar_init` exist only under
`--ustar-structure decay`.

---

## The knot dates

**2013Q1**, one interior knot: the start of the low-inflation era, and where the
decline stops. A knot at 2008Q1 scores a little worse on the band test with
near-identical endpoints. With one knot there are only three coefficients, so
the knot's date only nudges the single bend; the count matters far more.

**A second knot at 1996Q1 is a robustness check, and the model passes it.**
Without Okun, the data decline to use the extra freedom to bend in the
mid-1990s: the path barely changes and only the band widens. With Okun the same
knot is used heavily, starting the curve high and reproducing the decay law, so
that bend is something Okun asks for, not the data. One knot is kept on
parsimony.

---

## The random walk: a step size instead of a shape

`--ustar-structure taper`, the default. The splines are its alternatives in
`--compare`.

```
u*_t      = u*_{t-1} + sigma_t x z_t          z_t ~ N(0, 1)
sigma_t   = late + (early - late) x w_t
w_t       = 1 at the sample start, falling linearly to 0 at taper_end, 0 after
u*_1993Q1 ~ N(taper_init_mu, taper_init_sd)
```

**What it trades.** The spline fixes where u\* may bend; the walk fixes only
how far it may move each quarter, and leaves the shape to the data. It can
fall, level off, turn up and turn back, anywhere. Where it agrees with the
spline, the spline's shape is not what drives the answer.

**Why the step size tapers.** The sample covers two labour markets: a slow
transition out of high unemployment through the 1990s, then a long stretch of
low, fairly stable unemployment. One step size cannot serve both. Tight enough
to keep u\* from echoing the cycle later, it cannot afford the 1990s decline;
loose enough for the decline, it lets u\* rise with unemployment in every
slowdown, booking the cycle as structural. So the step is loose at the start
and falls to a tight value by `taper_end`. The late value allows a small rise in
the 2001 slowdown, inside the shaded window, in return for letting the path
settle after the fast fall in unemployment over 2002-04, without following the
cycle around 2008 or after 2020.

**The first quarter** carries a wide prior centred below the 1993Q1 rate, which
was deep slack, so the likelihood places the start.

**Everything here is asserted.** The step sizes, the taper end and the start
prior are settings in `config.py`, and `--free-sigma-ustar` is refused under
this structure: the fit improves every time the walk is loosened, so they
cannot be estimated. The choice is a judgement about how slow u\* should be.

**Reading the diagnostic.** "sd of du\* from taper_end" is measured on the
median path, which averages away each draw's wander, so it reads well below the
imposed step even when the draws use all of it. A low reading is not evidence
that the data are holding u\* still.

**The flat walk** (`--flat-walk`): the sample from 2004Q1, one flat step size
(the late one), and a tighter start prior centred on a judgement of u\* at the
time, set by the `FLAT_WALK_*` settings. The data, not the prior, settle the
start. It gives up the 1990s decline and gains a chart scaled to the present
regime.

**The stepped walk** (`--stepped-walk`): no taper, the early step until
`--stepped-walk-end` (default `STEPPED_WALK_END`, the end of the shaded window),
then the late one. It keeps u\* free through the 2002-04 fall and so settles
sooner, at the cost of a rise in the 2001 slowdown; both happen inside the
shaded window. The two step sizes are the taper's own.

---

## How the specifications compare, and how not to compare them

- **Decay, with or without Okun**: cannot stop declining; with Okun it also
  calls 1993 equilibrium. The narrow band with Okun is an artefact of counting
  one signal twice.
- **Spline, one knot, with Okun**: the best band test of any setting and the
  only one to turn u\* up after 2015, but the most biased against what inflation
  alone implies.
- **Spline, one or two knots, without Okun**: unbiased and flat after 2015.
- **Spline, two knots, with Okun**: reproduces the decay law.

**Closeness to the inflation-implied series is not a selection criterion.** It
is maximised by *being* that series, which swings wildly, and a stiffer path
deviates more by construction. **The band width is an output, not a virtue**:
narrower is better only at equal information.

**The band test is the only score that separates them, and it is weak.** Every
specification is fitted to the inflation series it scores against, and a small
breach of the band counts as much as a large one. A real ranking needs
something outside the fitted sample, a recursive real-time exercise, which has
not been done here.

---

## The band, and why the charts draw it twice as wide

**The printed diagnostics report the raw posterior band; every chart draws it
doubled**, and the footer says so. The widening stands in for uncertainty about
the **imposed structure** (the step schedule, or a spline's knots), which the
posterior cannot express: the interval answers "where is u\* given this
structure", not whether the structure is right.

**The factor of 2 is a convention, not derived for the walk.** It was
calibrated on the decay law, where doubling reproduced the union of bands across
a sweep of its imposed variance. No equivalent exists here, and the candidates
disagree: across structures u\* barely moves in the settled part of the sample,
arguing for less, while at the start the spread is several points, arguing for
far more. So it errs wide where u\* is settled and nowhere near wide enough in
the early sample, where the shaded window carries the warning instead. A
re-derived number would look more rigorous than it is, since the structure
itself cannot be swept.

---

## What every specification shares, and therefore cannot test

The identity above holds in all of them, so none is a second opinion on any
other. What differs is only how the same signal is smoothed.

The deviation between the fitted u\* and what inflation alone implies is
positively autocorrelated in every specification, so a persistent component is
unmodelled throughout.

**Estimation uncertainty is the larger term.** Outside the early sample, the
spread across specifications is well inside the band of any one of them, so
the choice of structure barely matters for the number to quote now. In the
early sample the reverse holds: the specification is the whole of the answer.

---

## Comparing specifications (`--compare`)

`--compare` runs this model four ways and charts the results together. It is
not a different model: each specification is a set of this model's own flags,
re-estimated only if its saved run is not from today. One is the default run
itself. Every run then writes its own charts, the default to `charts/UStar/` as
a plain run does and the others beside it (`charts/UStar-stepped/`,
`charts/UStar-k1/`, `charts/UStar-k2/`), and the combined charts follow.

| specification | u\* structure |
|---|---|
| **Default run** (tapered walk) | step size tapering from loose to tight by 2002Q1 |
| Stepped walk (`--stepped-walk`) | one large step size to the end of 2003, the tight one after |
| Spline, 1 knot | knot 2013Q1 |
| Spline, 2 knots | knots 1996Q1, 2013Q1 |

All four leave Okun out.

**Why these four.** The spline settings ask whether the answer depends on the
structure imposed on u\*: the walks impose a step size and no shape, the
splines a shape and no step size, so where they agree neither assumption is
driving the answer. The knot count asks how much the spline's flexibility
matters. The stepped walk asks how much the taper's gradual tightening matters:
it keeps u\* free through the whole shaded early window, including the fast
fall in unemployment over 2002-04, where the taper has already tightened. No
Okun setting is included, because its path is not credible (see the Okun
section).

**Why no decay settings.** Under the decay law the sign of u\*'s movement is
fixed by which side of its equilibrium it starts on, so from the high
unemployment of 1993 it can only ever report a fall. On a chart about how much
the specification matters, that shape would be read as evidence. Every
specification here can turn u\* up at the end if the data warrant it.

**How to read it.** The four share the sample, the Phillips curve, the
expectations series and the inflation measure, so their agreement is close to
arithmetic and only their disagreement informs.

- The knot count barely matters.
- The spread is widest in the early 1990s, where u\* is least identified, and
  has all but closed today. The choices argued above bear on the 1990s
  narrative, not on the number to quote now.
- The spread is not an error band. The specifications differ in a structured
  way, the structure imposed on u\*, so the range is the distance between
  readings under different assumptions. Quote the range and name what sits at
  each end.
- The mean line on the range chart describes where the specifications sit; it
  is not an estimate.

**What it prints and charts.** Each run's own charts, a table per specification
(the band test, the 1993-98 level, the post-2015 slope, the latest value, the
average band, bias against the inflation-implied series, and u\*'s volatility),
and four charts in `charts/UStar-compare/`: the u\* paths against unemployment,
the unemployment gap each implies, the range with its mean, and the range's
width over time. The 1993-2003 window is shaded.

## Files and usage

```bash
./run-ustar.sh                                   # the default above
./run-ustar.sh --okun                            # restore the Okun equation
./run-ustar.sh --no-inflation-lag                # drop last quarter's inflation from the Phillips curve
./run-ustar.sh --ustar-structure decay           # the decay law
./run-ustar.sh --taper-end 2004Q1                # --taper-* flags change the walk's schedule
./run-ustar.sh --flat-walk                       # the flat walk, written to ustar_flat_walk
./run-ustar.sh --stepped-walk                    # the stepped walk, written to ustar_stepped_walk
./run-ustar.sh --stepped-walk --stepped-walk-end 2004Q4 --prefix ustar_sw2005   # step down a year later
./run-ustar.sh --ustar-structure spline          # the one-knot spline instead of the walk
./run-ustar.sh --ustar-structure spline --knots 1996Q1 2013Q1   # a second knot
./run-ustar.sh --gap-source actual               # log_gdp - y* instead of the defined gap
./run-ustar.sh --analyse-only                    # re-chart a saved run
./run-ustar.sh --prefix name                     # write somewhere other than `ustar`
./run-ustar.sh --compare                         # four specifications, each charted, then combined
./run-ustar.sh --compare --analyse-only          # the same, from the saved runs as they stand
```

The comparison lives in `compare.py` (the specifications) and `compare_charts.py`
(the table and charts). `cli.py` holds the flags, shared by the default run and
the comparison.

`config.py` holds every imposed quantity and records it in `constants`, saved
beside the trace. Charts and this run's diagnostics go to `charts/UStar/`.
Prior-posterior charts come from `common/prior_posterior.py`, shared with
every Bayesian model in the package.
