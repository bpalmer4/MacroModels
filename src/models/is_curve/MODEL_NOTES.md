# The IS curve, plotted rather than estimated

**A test bench, not a model.** Nothing here is estimated in the Bayesian sense and nothing
here produces a number another package consumes. It exists to put the r\* estimates from
the other packages on a common scatter and see whether any of them can recover the
relationship an IS curve asserts. The finding is that none of them can, and that is a
result about the other models rather than about this one.

**THE IS-CURVE PROBLEM IN AUSTRALIAN DATA REMAINS UNRESOLVED.** Nothing here settles
whether the interest rate moves Australian activity. It establishes that a reduced-form
scatter cannot show it, that no r\* in this repo rescues it, and that a sample cut can
manufacture a convincing answer with the wrong provenance. Whether the transmission is
absent, too slow and small to see at this frequency, or real but split across a dozen
differently timed channels that wash out in aggregate, is open. The refinements at the
bottom are the ways to close it and none has been run. Read every number here as a
description of the failure, not as a measurement of the economy.

Concretely: it puts the two sides of the IS curve on one scatter, fits a line by OLS, and
reports where that line crosses zero output gap. It exists because every IS-curve result in this repo has come out of a
state-space model in which the output gap was *also* latent, and a weak channel cannot
be told apart from an unidentified latent in that setting. Here both sides are given:

```
gap_t = a + b . x_{t-lag} + e      x = one of three real-rate variants
```

`gap` is the median output gap from a completed `ystar_ustar` run. `x` is the real cash
rate, `cash - pi_exp`, under one of three treatments of r\*:

| variant | x | what its intercept means |
|---|---|---|
| `none` | `cash - pi_exp` | the zero-gap crossing *is* an estimate of r\* |
| `rstar` | `cash - pi_exp - r*_t` | how far [`rstar_bonds`](../rstar_bonds/MODEL_NOTES.md)'s r\* sits from what the scatter wants |
| `rule` | `cash - pi_exp - b_t` | the same, against [`rstar_rba`](../rstar_rba/MODEL_NOTES.md)'s neutral `b_t` |
| `constant` | `cash - pi_exp - 1.37` | the same, against a flat r\* (the sample mean) |

The `rule` variant reads `neutral_real`, that model's slow intercept `b_t` less the target,
not `prescribed_real`, which carries the RBA's own inflation response on top. Against neutral
the slope at lag 2 is **+0.201**, R\u00b2 0.143: the largest of the four and the most wrongly
signed. The rate gap against a slow-moving neutral is the policy stance, and the RBA sets a
positive stance when the economy runs hot, so the regression recovers the reaction function
with the sign reversed.

Its t of +4.54 is not significance. Residuals are autocorrelated at **0.892**, inflating the
classical standard error by about 4.2, which puts corrected t near **+1.1** and the 90%
interval at roughly **[\u22120.10, +0.51]**. The tilt survives detrending, at +0.196, but is not
distinguishable from zero. The same correction applies to every row below, per Limitation 1.

Slopes by lag, all four variants:

| variant | 0 | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|---|
| `none` | +0.081 | +0.062 | +0.041 | +0.022 | +0.005 | −0.006 | −0.015 |
| `rstar` | +0.238 | +0.185 | +0.123 | | | | |
| `rule` | **+0.325** | +0.270 | +0.201 | +0.129 | +0.068 | +0.033 | +0.014 |
| `constant` | +0.081 | +0.062 | +0.041 | +0.022 | +0.005 | −0.006 | −0.015 |

Every one is strongest **contemporaneously** and decays to zero within four to six quarters.
Transmission would be the mirror image: weak on impact, strengthening over three to six
quarters, and **negative**. The scatter contains the policy reaction, not the policy effect,
and the choice of r\* only changes how loudly it says so.

---

## The default lag is now 6, not 2

**Changed 2026-09-11.** `DEFAULT_LAG` was 2, inherited from `nairu`. It is now **6**, and
`rstar_hlw` and `rstar_invert` were moved to match so the three are comparable on timing
(`rstar_invert` uses a weighted pair at 4 and 8 whose effective mean lag is 6.3).

The reason is this package's own sweep rather than convention. The slope strengthens
monotonically with the lag and only turns negative around 4 to 5 on the full sample; on the
sample that drops 2008Q4-2021Q3 it reaches −0.139 at five quarters and peaks at lag 6.
`rstar_invert`'s single-lag sweep finds the same shape inside a state-space model, −0.007 at
lag 1 rising to −0.034 at lag 5, with the posterior only coming off its sign bound at 5.

That is simultaneity, not fit-chasing: the RBA reacts to conditions within a quarter or two
while output responds over one to two years, so a short lag mostly measures the reaction
function and returns the wrong sign (+0.081 at lag 0). Reaching back is a partial fix only,
since the real cash rate is persistent and `r_{t-6}` stays correlated with recent rates that
are reacting.

**Every "lag 2" number below is still correct as a lag-2 number**, and they have not been
restated. What changed is which lag the headline chart uses. At lag 6 on the default sample:

| variant | slope | t | R² | n |
|---|---|---|---|---|
| `none` | −0.015 | −0.70 | 0.004 | 122 |
| `rstar` | −0.028 | −0.63 | 0.003 | 122 |
| `rule` | +0.014 | +0.28 | 0.001 | 122 |
| `constant` | −0.015 | −0.70 | 0.004 | 122 |

All four are indistinguishable from zero. The longer lag removes the wrongly-signed reaction
function without putting an IS curve in its place.

**And the block split at lag 6 is the sharpest version of this package's central finding:**

| block | n | slope | t | R² |
|---|---|---|---|---|
| 1993Q1-2020Q1 | 103 | **+0.099** | +4.58 | 0.172 |
| 2021Q4-2026Q2 | 19 | **−0.136** | −4.76 | 0.571 |

Twenty-seven years give a significantly *positive* slope. The last nineteen quarters give a
significantly negative one. The whole negative reading of Australian data sits in the
post-pandemic tightening, which is also the period in which a reaction function and
transmission are hardest to tell apart.

---

## Read this first

**The slope's sign is a function of which quarters you include, and the honest answer is
that this scatter does not identify an IS curve at all.**

| sample | n | slope at lag 2 | t | R² |
|---|---|---|---|---|
| 1993Q1-2026Q2, lockdowns excluded (default) | 126 | **+0.041** | +1.86 | 0.027 |
| the same, all quarters kept | 132 | +0.048 | +2.34 | 0.040 |
| GFC to end of pandemic excluded | 80 | **−0.108** | −4.05 | 0.174 |

An IS curve needs a negative slope. The default sample gives a positive one. Cut out
2008Q4 to 2021Q3 and the pooled line turns negative, strengthens with the lag to −0.139
(t −6.5) at five quarters with an R² of 0.36, peaking at 0.374 by lag 6, and implies
r\* of about 3.8 to 4.3. It
looks exactly like monetary transmission. It is not.

---

## Why the negative slope is not transmission

**The within-block fits contradict it.** Excluding the middle of the sample leaves two
disconnected blocks. Fitted separately, at lag 2:

| block | n | `none` | `rstar` |
|---|---|---|---|
| 1993Q1-2008Q3 | 61 | **+0.105** (t +1.76) | **+0.203** (t +4.89) |
| 2021Q4-2026Q2 | 19 | −0.114 (t −3.37) | −0.182 (t −5.02) |
| pooled | 80 | −0.108 (t −4.05) | −0.008 (t −0.21) |

Three quarters of the surviving data sit in a block that still slopes the *wrong* way,
and on the bond-market r\* it does so emphatically. The pooled negative slope is a line
drawn between two clusters: the recent quarters sit at low real rates with gaps of +0.3
to +1.1, the older ones at real rates of 2 to 4% with gaps scattered about zero.

**This is Simpson's paradox in the sense that matters.** The aggregate says the opposite
of what the subgroups say, because the subgroups sit at different average levels of both
variables and the pooled line runs between them rather than through either. The pooled
slope is therefore not evidence about the within-group relationship, and the sign of the
published answer depends on which groups are in the sample.

The one departure from the textbook version is worth recording, because it makes the
result stronger rather than weaker: the classic case has the same direction in every
subgroup, reversed in aggregate, whereas here the two blocks disagree with *each other*
as well, +0.105 against −0.114. So the pooled line is not masking a common effect that
the groups agree on. There is no agreed within-group effect to mask.

**A second, internal confirmation.** Pooled at lag 2, the `rstar` variant gives −0.008
against −0.108 for `none` and `constant`. Subtracting a time-varying r\*, which itself
falls from about 2.4 to 1.2 across the sample, absorbs precisely the between-era level
difference, and the negative slope vanishes with it. Real transmission would be sharpened
by measuring the rate against neutral, not erased.

---

## What the default sample shows instead

The relationship that *is* in the data has the wrong sign and the wrong timing for an IS
curve, and the right ones for a policy reaction function. Lockdowns excluded, `none`:

| lag | 0 | 1 | 2 | 3 | 4 | 6 | 8 |
|---|---|---|---|---|---|---|---|
| slope | +0.081 | +0.062 | +0.041 | +0.022 | +0.005 | −0.015 | −0.028 |
| t | +3.77 | +2.86 | +1.86 | +0.99 | +0.24 | −0.71 | −1.30 |

Strongest **contemporaneously** and decaying to nothing. On the `rstar` variant it is
stronger still at lag 0, +0.157 with t = 4.50. Transmission would show a negative slope
building over roughly two to six quarters, which is exactly where this goes flat. What
this shape says is that the RBA raises the rate when the gap is positive, within the
quarter, and that this dominates everything else in the scatter.

Note also where the wrong-signed relationship is *strongest*: the pre-GFC block, the
cleanest stretch of conventional policy in the sample, at t +4.89. The reaction function
shows through most clearly where policy was most conventional.

---

## The lockdown exclusion barely matters

Dropping 2020Q2-2021Q3, the window `ystar` and `ystar_ustar` drop from their likelihoods,
was expected to matter and does not. At lag 2 the slope moves +0.048 to +0.041, and at
lag 0, where the relationship lives, +0.082 to +0.081 (t 4.18 to 3.77). Removing the six
most extreme quarters in the sample costs the wrong-signed relationship almost nothing,
which makes it a feature of the whole targeting era rather than a pandemic artefact.

The exclusion is applied **after** the lag, never before. Dropping quarters from the
series first would make `shift` step across the hole and pair a gap with the wrong
quarter's rate. It is applied on the *gap's* quarter, not the rate's: the cash rate is
observed through the lockdowns as accurately as ever, and it is potential output that is
not well defined when much of the economy is closed by order.

---

## Where this sits against the rest of the repo

It agrees with the repo's central negative finding and sharpens it. `rstar_hlw` reports
a_r ≈ −0.04 ± 0.01 flat across eight resolutions, `nairu` β_is ≈ 0.084, the `dsge` family
the same. Those say the channel is too weak to identify. This adds two things:

1. **A raw scatter cannot even recover the sign**, because the policy reaction function
   dominates it. The weak coefficient in the structural models is what remains after
   they impose enough structure to net that out.
2. **A sample cut can manufacture a convincing IS curve.** Anyone tempted to rescue the
   IS curve by dropping the QE era, on the perfectly good ground that the cash rate is
   not the stance at the lower bound, will get a beautifully behaved answer with the
   right sign, the right lag profile, an R² of 0.37, and a plausible r\*. The block check
   is what stops that being written up.

---

## Why this matters: it is HLW's identification strategy that fails

The stakes are larger than one equation, and this is the argument that makes the package
worth keeping rather than a curiosity. The argument below is the author's, set out in a
blog exchange in September 2026, in reply to a challenge that `src/models/rstar_bonds`'s r\*
must be too high. The regression results it rests on are the ones above; the reading of
what they imply is his. (Link to be added.)

**HLW-style models do not observe neutral. They identify it through the IS curve.** The
strategy is to find the real rate around which output moves from above potential to below
it, and to infer r\* from that crossing. That is precisely the intercept this package
plots. If there is no stable relationship within the data, the intercept does not locate
neutral either, and the failure documented above is not a weak coefficient in an
otherwise sound model: it is the identifying assumption of an entire literature failing
on Australian data. `rstar_hlw`'s eight resolutions each returning the structural
assumption they imposed is what that failure looks like from the inside.

**It also disarms a specific inference that is often made against `src/models/rstar_bonds`.**
The challenge runs: before COVID the cash rate sat below your bond-market r\*, and if it
had really been that far below neutral we should have seen more demand and more
inflation; we saw below-target inflation and unemployment above the NAIRU; therefore the
r\* estimate is too high.

The observation is real and should not be explained away. The inference does not follow.
It requires exactly the domestic IS relationship these regressions cannot find. What the
episode establishes is narrower: a cash rate below that estimate did not produce the
positive output gap the standard IS mechanism predicts. Reading it as evidence about the
level of r\* assumes the mechanism whose absence is the finding.

**The concept is not what is in doubt, the identification strategy is.** r\* here is used
in its older sense, the rate at which desired saving and investment are in equilibrium.
The question this package raises is whether that equilibrium can be recovered from a
*domestic* IS curve when saving and investment meet in a global capital market. That is
not necessarily a claim that there are two different r\*s, a global one and a
policy-neutral domestic one. It may be a distinction between the concept and one
particular way of measuring it. `src/models/rstar_bonds` estimates the global capital market's
price plus an Australian wedge; the conventional approach tries to recover the same price
from Australian output and interest rates. It is the second strategy the data will not
support.

### A hypothesis for why, clearly labelled as one

ASSUMPTION, and not established by anything in this package: in a highly interconnected
capital market the cash rate is only one price among several affecting demand. The
exchange rate, bank funding costs, asset prices and household cash flow act alongside the
textbook cost-of-investment channel, and each moves output on a different timeline.

UNVERIFIED here, and worth checking before quoting: the RBA's own structural models are
understood to find these channels individually, with dwelling investment consistently the
most interest-sensitive component of GDP, and business investment responding once
firm-level rather than aggregate data is used. That has not been checked against the
source papers in this repo.

If that is right, it is the point rather than a counterexample. Those effects become
visible when each channel is modelled separately with its own structure imposed. A single
reduced-form regression of the aggregate output gap on one rate series asks one number to
summarise several differently timed channels at once, and if they do not move together
the aggregate can wash out even while the individual channels are doing something real.

That is a hypothesis about *why* the simple IS relationship is hard to find. What these
regressions establish is only the narrower claim: there is no stable relationship between
the real-rate gap and the output gap of the kind needed to identify r\* from Australian
macro data. Refinement 3 below is the test that would tell the two apart.

---

## Limitations

1. **OLS on autocorrelated series.** Standard errors are classical, not HAC. Both sides are
   highly persistent: on the `rule` variant at lag 2 the residuals are autocorrelated at
   **0.892**, inflating the standard error by about **4.2** on the textbook AR(1) correction,
   which takes t from +4.69 to +1.1. Divide every t here by something of that order, worst on
   the 19-quarter block where t = −3.4 is quoted. **No slope reported here survives as
   significant.** Nothing in the package corrects for it.
2. **Nineteen observations is not a sample.** The post-pandemic block is the only
   IS-shaped piece in the exercise, and over those quarters rates rose steadily while the
   gap narrowed. Two trending series with no causal link between them would produce the
   same picture.
3. **Neither axis is data.** The gap is a posterior median from `ystar_ustar`, plotted as
   a point with no uncertainty; expectations come from the project's own model; the
   `rstar` variant carries a third model's median. None of that uncertainty propagates.
4. **The rate is endogenous, and nothing here addresses it.** This is the whole problem,
   and the package documents it rather than solving it. See the refinements below.
5. **Single equation, no dynamics.** No lagged dependent variable, no fiscal term, no
   open-economy block. The structural models in this repo have all of those, which is
   why their coefficients are the ones to quote; this is a diagnostic picture, not an
   estimate.
6. **The zero-gap crossing is fragile arithmetic.** `-a/b` explodes as b approaches zero.
   It is suppressed unless the slope is negative and |t| ≥ 2, which stops the worst
   readings but does not make the surviving ones precise.

---

## Refinements to explore

### 1. An instrument for policy, which is the only thing that would settle it

Every result here is contaminated by the reaction function, and no sample cut fixes that.
What would is a series of monetary policy *shocks*: the component of the cash rate not
predicted by the RBA's own information set, whether Romer-Romer style from forecasts or
from cash-rate futures surprises around announcements. Local projections of the gap on
such a shock at horizons 0 to 12 would answer the question this scatter cannot.

This is the single largest gap in the repo's IS-curve work. Resolutions A through H of
`rstar_hlw` all put the policy rate into an IS curve without instrumenting it.

### 2. Split the older block

1993Q1-2008Q3 is treated here as one era, and it contains the 1990s disinflation, the
Asian crisis and the mining boom. If the positive slope survives inside each of those it
is a reaction-function reading; if it does not, the two-block comparison above is itself
too coarse.

### 3. The components that actually respond, which tests the channels hypothesis

This is the direct test of the argument above. If the aggregate null is really several
differently timed channels washing each other out, then the channels should be visible
one at a time even though their sum is not. Dwelling approvals, dwelling investment,
capex and household cash flow are all in `src/data`. Australia's transmission is thought
to run through housing and household cash flow in particular, given the variable-rate
mortgage share.

The result cuts both ways, which is what makes it worth running. Find the channels
separately and the aggregation explanation stands, and the reduced-form null becomes a
statement about measurement rather than about transmission. Fail to find them there too,
on data where each series has its own structure and its own timing, and the aggregation
defence is gone and the null is about the economics.

### 4. HAC standard errors

Cheap, and it would stop the t-statistics in these notes overstating the case. Every
significance claim above should be read as provisional until this is done.

---

## Files and usage

```
src/models/is_curve/
├── observations.py   # the gap, cash - expectations, the three r* variants, the windows
├── fit.py            # OLS, the zero-gap crossing, the lag sweep
├── analyse.py        # one scatter per variant, points coloured by date
└── run.py            # CLI
```

Needs a completed `ystar_ustar` run for the gap, a completed `rstar_bonds` run for the
`rstar` variant and a completed `rstar_rba` run for the `rule` variant. Either r\* variant
is skipped with a note if its run is missing. Run order: `expectations` → `ystar_ustar` →
`rstar_bonds` → `rstar_rba` → `is_curve`.

```bash
uv run python -m src.models.is_curve.run                     # lockdowns excluded (default)
uv run python -m src.models.is_curve.run --keep-all          # every quarter
uv run python -m src.models.is_curve.run --drop-gfc-pandemic # 2008Q4-2021Q3 out
uv run python -m src.models.is_curve.run --lag 5             # where the cut sample peaks
uv run python -m src.models.is_curve.run --exclude-window 1997Q1 1999Q4 --exclude-window 2020Q2 2021Q3
```

Charts land in `charts/ISCurve/`, one per variant. Points are coloured by date, which is
not decoration: the two-cluster structure that produces the sign reversal is visible in
the chart before it is visible in any statistic. Whenever the exclusions leave more than
one block of consecutive quarters, the per-block fits print automatically, because the
pooled line on its own is not interpretable.
