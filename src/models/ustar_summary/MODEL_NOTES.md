# u\* summary: three specifications of one model

## Not a model, and not three models

This estimates nothing. It runs `ustar` three times, loads the results and
charts them together.

The three differ in the number of spline knots and in whether the Okun
equation is included. Everything else is shared, including the sample, the
Phillips curve, the expectations series and the inflation measure. Where
`rstar_summary` gathers estimates built on different data and different
identifying assumptions, **agreement between these lines is close to
arithmetic. Only their disagreement is informative.**

| | knots | Okun |
|---|---|---|
| **Spline 1 knot** | 2013Q1 | out |
| Spline 2 knots | 1996Q1, 2013Q1 | out |
| Spline 2 knots, with Okun | 1996Q1, 2013Q1 | in |

**Every line can turn up at the endpoint**, which is why the decay settings
are absent. Under `--state converge` the sign of `phi x (eq - u*)` is fixed by
which side of the equilibrium the state opened on, so from 10.75 it can only
ever report a fall; its -0.32 and -0.34 over 2015-2026 are properties of the
shape rather than readings of the data. On a chart about how much the
specification matters, that invites being read as evidence. If u\* starts
rising, these three will show it.

The bolded row is `ustar`'s default. See `ustar/MODEL_NOTES.md` for why.

---

## What the Okun specifications are for

`ustar` excludes the Okun equation, and the argument for excluding it is
strong: the gap it reads is `0.1882 x (pi - 2.5)` exactly, so it is a second
copy of the Phillips curve, it halves the reported band by counting one signal
twice, and it biases u\* by -0.13 to -0.23 against what inflation alone
implies.

One setting keeps it anyway, because **it is the only one in which u\*
comes down through the 1990s**, and the 1990s were a regime change. Inflation
moved from high to low and that took years to work through the labour market.
Over 1993-1999:

| | u\* 1993Q1 | u\* 1999Q4 | fall |
|---|---|---|---|
| Spline 2 knots, with Okun | 10.15 | 6.72 | **3.43** |
| Spline 2 knots | 7.57 | 6.75 | 0.81 |
| Spline 1 knot (default) | 7.28 | 6.77 | 0.50 |
| *unemployment rate* | *10.93* | *6.40* | *4.53* |

Unemployment fell 4.53 points. The Okun run has u\* falling with it. The
default has it falling 0.50, which asserts that almost the whole descent was
cyclical and that the equilibrium barely adjusted to the new regime. That is a
strong claim, and it is not obviously the right one.

**This is not the 1995 episode.** Two quarters of inflation just above the
band, at 3.65 and 4.06, is a short and shallow breach and a line should not be
dragged by it. It is worth recording only as a symptom: the specifications
that still have u\* elevated and falling in the mid-1990s are the ones that
read 1995 as a tight labour market, and none of the flat ones comes close,
their gaps running +0.43 to +1.20.

Neither reading is settled. The chart exists so that the choice is visible
rather than buried in a default.

### The Okun setting that is not here

`--state spline --knots 2013Q1 --okun` carries the equation's costs, including
the worst bias against the implied series of anything tried at -0.229, without
delivering the 1990s reading that is the reason for keeping any Okun run: its
1993-1999 fall is 0.76, which puts it with the flat specifications rather than
with the other two. It is also the only setting that turns u\* up after 2015,
at +0.36. Its one virtue is the best band test of the six tried, 92.2%.

---

## What the spread is, and what it is not

```
all three:            mean 0.51pp, worst 2.87pp at 1993Q1, latest 0.03pp
the two without Okun: mean 0.05pp, worst 0.29pp at 1993Q1, latest 0.03pp
mean 90% band within a single no-Okun run: 0.70
latest u* ranges 4.67 to 4.70, against unemployment of 4.35
```

**The knot count does almost nothing** once Okun is out: 0.05pp over the whole
sample. And the endpoint is robust to everything here, 4.67 to 4.70.

**The profile is the point.** 2.87pp at 1993Q1 falling to 0.03pp today is a
fair reflection of a period where pinning u\* is genuinely harder, and it
agrees with the two other diagnostics that say so: the shaded window, and a
deviation from the implied series that is biased through 1993-99 and unbiased
after. Three routes, same seven years.

**It is not a symmetric error band.** The three differ in a structured way,
whether Okun is in, rather than randomly, so the pooled range is the distance
between two readings of the 1990s rather than a distribution around a centre.
An average across them describes neither. Quote the range and say what sits at
each end.

Among the two that share a reading of the 1990s the spread is 0.05pp, against
a within-run band of about 0.70. Conditional on that reading, estimation
uncertainty is overwhelmingly the larger term.

**From 2000 everything converges.** The three sit within 0.03pp today, so the
choices argued at length in `ustar/MODEL_NOTES.md` bear on the 1990s narrative
rather than on the number to quote now.

**And the agreement is mostly structural.** A straight line fitted through the
unemployment rate sits about half a point from the mean across
specifications, correlation above 0.9. All three are curves of three or four
coefficients fitted to the same inflation signal, so a smooth path a little
below unemployment was never in doubt.

---

## The mean line is a description

The centre is a mean, not a median: with three series the median is whichever
specification sits in the middle that quarter, so it switches identity
wherever the lines cross and picks up kinks that say nothing about u\*.

It is **not an estimate**, and it is weaker than the same line on
`rstar_summary`, which at least averages across models built differently.

---

## Charts

| file | what it shows |
|---|---|
| `u-one-model-three-specifications` | all three against the unemployment rate. Colour is the knot count, dashing is Okun |
| `how-much-the-specification-matters-for-u` | the range shaded, with the mean and the unemployment rate. The band only: the individual lines are on the levels chart |
| `how-much-the-specification-matters` | the range as a single width series |
| `the-unemployment-gap-by-specification` | u less u\* for each |

The 1993Q1-1999Q4 window is shaded on all but the range chart, where it is the
same orange as the band and the two overlaid exactly where the range is
widest.

The window is read from `ustar.analyse.UNIDENTIFIED_WINDOW` rather than
restated, so the two cannot drift apart.

---

## Files and usage

```bash
./run-ustar-summary.sh                # refresh anything stale, then chart
./run-ustar-summary.sh --no-refresh   # chart the saved runs as they stand
```

Each specification writes to its own `ustar_sum_*` prefix, so a refresh never
touches `ustar`'s own outputs or `charts/UStar`. A saved trace counts as
current if the file was written TODAY, the same rule `rstar_summary` uses: a
proxy for the data being current rather than a check of ABS and RBA vintages,
conservative in the right direction since a stale file is always re-run.

A full refresh is three estimations at about 20 seconds each.
