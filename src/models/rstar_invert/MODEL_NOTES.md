# r\* by conditional inversion of an asserted IS curve

**This package does not estimate r\*.** It asserts an IS curve, asserts that a neutral rate
exists and moves slowly, and reports what r\* path those two assertions force. Every number
below is conditional on them, and the notes exist mostly to say how conditional.

## Read this first

**The answer is decided by one number nobody can measure**, `sigma_rstar`, the speed of
r\*. Not "influenced by": decided. Below a threshold the model explains nothing and r\* is
flat; above it the model fits well and r\* swings by five points. The 2016-19 policy stance
flips sign across that threshold, from −3.19 to +0.52. Nothing in the data picks a side.

**A defensible slope and a usable r\* are mutually exclusive here.** The parameterisation
that gives a credible slope (−0.09, matching the `is_curve` bench's −0.108) produces an r\*
running −3.9 to +7.0. The one that gives a well-behaved r\* needs a slope of −0.38, which
survives only because that parameterisation rewards inflating it.

**What the model actually measures well is not r\*.** It is the output gap's own
slow-moving component, which it then divides by a small number and relabels as a rate.

## The model

Two axes, both gaps, and a line through the origin: at a zero rate gap the economy sits at
potential, which is what neutral means. There is no constant term.

```
(1)  rstar_t = rstar_{t-1} + eta_t                  eta_t ~ N(0, sigma_rstar)
(2)  x_t     = is_slope . [ w.(r - r*)_{t-4}
                          + (1-w).(r - r*)_{t-8} ]  + e_t
```

| | |
|---|---|
| `x_t` | output gap, **given**: the posterior median from a completed `ystar_ustar` run |
| `r_t` | real cash rate, **given**: RBA cash rate less unanchored model expectations |
| `is_slope` | TruncatedNormal(−0.30, 0.10, upper = 0) |
| `w` | Beta(2, 2) |
| `sigma_rstar` | **asserted** at 0.10 pp per quarter |
| `sigma_e` | HalfNormal(2.0) |
| `rstar_0` | Normal(1.50, 2.00) |

Sample 1993Q1 to 2026Q2, 134 quarters. Eight lost to the longest lag, six lockdown quarters
(2020Q2 to 2021Q3) carried in the state but out of the likelihood, leaving **120**.

### Why the sign is truncated and the magnitude is not

A positive slope is not a weak IS curve, it is no IS curve: New Keynesian transmission
requires a restrictive stance to contract output. So the sign is theory. The magnitude is a
belief the data may argue with.

**A posterior pressed against that bound is a rejection, not a measurement.** It means the
likelihood wanted the wrong sign and the constraint stopped it. Never quote such a run as
"the slope is −0.01".

### Why the prior centre is deliberately too strong

−0.30 is far past anything measured in this repo. It is the **favourable** case: r\* absorbs
the gap divided by the slope, so a big slope keeps the implied neutral rate sane and a small
one blows it up. If the answer is absurd at −0.30 it is worse everywhere else.

## The mechanism, which is the whole story

Rearrange (2):

```
rstar_bar_t = rbar_t - (x_t - e_t) / is_slope
```

Define `q_t = rbar_t − x_t/is_slope`, which is fully observable. The equation says `q_t` is
made of two unobserved pieces, `rstar` and `e_t/is_slope`, and the model's only job is to
split them. Its **only criterion is persistence**: r\* is constrained to move slowly, `e_t`
is assumed independent. So

> **r\* is a smoothed version of `rbar_t − x_t/is_slope`.**

`q_t` contains the real cash rate undiluted, and smoothing cannot remove what the first term
put there. Hence `corr(r*, real cash) = +0.826` in the headline run.

**On levels it says almost nothing.** Average (2) across the sample: if the gap averages
zero then r\* must average the real cash rate. The level is forced by that requirement, not
discovered. It is the same answer as a flat r\* at the sample mean real rate, which the
`is_curve` bench reports as 1.37.

### The last four quarters of r\* are not estimated

`rstar` is a random walk over the whole index, but the likelihood only ever touches it through
the lags: the stance at row `t` uses `rstar[t−4]` and `rstar[t−8]` (`estimate.py:188-193`). So
the last quarter any observation constrains is **T−4**. At a sample ending 2026Q2 that is
2025Q2, and 2025Q3 through 2026Q2 are the prior random walk carried forward, nothing more.

Unlike `rstar_hlw`, there is no second component to pick up the slack. HLW's r\* is `g + z`, so
its recent end still gets current trend growth through the potential equation. Here r\* is a
bare random walk conditioned only through the inversion. When the lags run out, nothing is left.

The posterior says so plainly, fanning out over exactly four quarters:

| | median | 90% width |
|---|---|---|
| 2025Q2, last informed | 1.53 | 0.96 |
| 2025Q3 | 1.53 | 1.02 |
| 2025Q4 | 1.53 | 1.07 |
| 2026Q1 | 1.52 | 1.13 |
| 2026Q2 | 1.53 | **1.16** |

Mid-sample width is 0.67. The median is flat at 1.53 because a random walk's best guess is its
last informed value, and the band widens 0.20pp across the four uninformed quarters.

The middle of the sample is affected too, at half strength: quarters between T−8 and T−4 are
informed only by the lag-4 term, which carries weight `w` = 0.435. The falling-off is gradual,
not a cliff at four quarters.

This is structural, not a defect to fix. Any model identifying r\* off a lagged rate channel is
silent about the most recent quarters by construction; the `is_curve` bench's default lag of 6
and `rstar_hlw`'s t−6 have the same property. What makes it sharper here is the absence of any
anchor, so the silence is total rather than partial. **No chart marks the uninformed tail**, so
the widening band is the only visible sign of it.

## Headline run

`./run-rstar-invert.sh`, defaults as above.

| | |
|---|---|
| `is_slope` | −0.383, 90% [−0.439, −0.326] |
| `w` (on lag 4) | 0.435, 90% [0.328, 0.527] |
| r\* latest | +1.53% real |
| r\* range | −1.19 to +3.99, sd 1.62 |
| `corr(r*, real cash)` | +0.826 |
| `sigma_e` | 0.239, against a gap sd of 0.420 |

Read that as: given a strong asserted channel and r\* allowed to move 0.10pp a quarter, the
IS curve requires a neutral rate that swings five points and tracks the policy rate at 0.83.

## The two sweeps

### How slow is r\* (`--ensemble`)

```
sigma_rstar  is_slope   r* sd   corr cash   sigma_e   stance 16-19
      0.02     -0.032    0.01       0.476     0.432         -3.19
      0.05     -0.034    0.04       0.247     0.431         -3.13
      0.10     -0.383    1.62       0.826     0.239         +0.52
      0.15     -0.389    1.81       0.814     0.167         +0.80
      0.30     -0.368    1.98       0.801     0.105         +0.90
      0.50     -0.283    2.25       0.731     0.072         +1.16
```

**A switch, not a dial.** Everything changes at once between 0.05 and 0.10: the slope jumps
elevenfold, the residual halves, r\* goes from frozen to swinging. There is no intermediate
state, which is the signature of two separated modes.

**`sigma_e` never stops falling**, 0.432 down to 0.072, with no interior optimum. The fit
improves monotonically the more r\* is allowed to move, because a faster r\* slides each
point onto the line. **So the data cannot choose `sigma_rstar`.**

**The stance flips sign across the switch.** That is what the choice costs.

### Where the rate enters (`--lag-sweep`, single lags)

```
lag   is_slope      q95    boundary   corr cash   sigma_e
  1     -0.007   -0.0005   TOUCHING      -0.548     0.430
  2     -0.010   -0.0007   TOUCHING      -0.445     0.431
  3     -0.015   -0.0015   TOUCHING      -0.278     0.430
  4     -0.024   -0.0031   TOUCHING      +0.020     0.427
  5     -0.034   -0.0063   clear         +0.360     0.425
```

The slope strengthens **monotonically** with the lag and only comes off its sign bound at 5.
That is the shape transmission should have and the opposite of what a pure reaction function
gives, and it is the reason for the long lags: the RBA reacts to conditions within a quarter
or two while output responds over one to two years, so a regressor further back carries less
of the reaction function. A weak fix, not a clean one, since the real cash rate is persistent.

**Caveat**: the five estimates are not independent. Lags 1 to 5 of a persistent series
overlap heavily, so a monotone pattern is weaker evidence than it looks.

**The default is the weighted pair (4, 8)**, whose effective mean lag is 0.435×4 + 0.565×8 =
**6.3 quarters**, matching the lag 6 that `is_curve` and `rstar_hlw` now use. The weight
itself buys little: 0.5 sits inside its interval, and when the slope is not inflated that
interval widens to [0.099, 0.608], close to the Beta prior. `--fix-lag-weight` costs almost
nothing.

### The ridge between the slope and r\*'s amplitude (2026-09-14)

Two wider lag structures were tried against the (4, 8) default. Neither beat it, and the
failure of one of them is the useful result.

**Lags (1, 2), the reaction-function end.** Samples cleanly (R-hat 1.0, 0/8000 divergences,
min ESS 4162) and lands on the null:

| | (4, 8) | (1, 2) |
|---|---|---|
| `is_slope` | −0.383 [−0.440, −0.326] | **−0.008** [−0.029, −0.001] |
| P(slope > −0.01) | 0.00 | **0.57** |
| `sigma_e` | 0.239 | 0.431 (gap's own sd 0.420) |
| R² of the fitted gap | +0.714 | **−0.021** |
| r\* median path | −1.19 to +3.99 | +2.37 to +2.69, sd 0.08 |

The posterior travelled **2.9 prior sd** to get there, into a region holding 0.18% of the
prior's mass, so it is the likelihood putting it on the bound and not the prior. At −0.008
the inversion is meaningless anyway: r\* movement scales as 1/|is_slope|, so one point of
output gap needs **119pp** of r\*. The flat r\* is therefore not a stable neutral rate, it is
r\* with no leverage falling back on its walk prior.

**Lags (1, 4, 7), a Dirichlet across three sticks (`lag_weight_conc`, 2.0 each).** This one
**failed to converge**: R-hat 1.110, ESS 27, 258/8000 divergences. The cause is not the
Dirichlet and not collinearity. The three regressors correlate 0.73 to 0.87 but the design's
condition number is only **5.4** and the pairwise differences have sd 0.93 to 1.32pp, and the
four chains agreed on the weights while disagreeing about everything else:

| chain | w(t−1) | w(t−4) | w(t−7) | `is_slope` | r\* sd | divergences |
|---|---|---|---|---|---|---|
| 0 | 0.123 | 0.308 | 0.547 | −0.056 | 0.79 | 0 |
| 1 | 0.194 | 0.289 | 0.463 | −0.030 | 0.16 | 31 |
| 2 | 0.113 | 0.304 | 0.570 | −0.235 | 1.30 | 0 |
| 3 | 0.096 | 0.310 | 0.582 | −0.319 | 1.47 | 227 |

**What failed is a ridge between the slope and r\*'s amplitude**, and it is the same
`|is_slope| × sigma_rstar` trade-off the next section documents, seen from the sampler's side
rather than the prior's. The likelihood is nearly flat along it, each chain parks somewhere
and does not traverse, and the divergences concentrate at the steep-slope end (227 of 258 in
chain 3, none in the two flattest). A lag set that spans the reaction-function end and the
transmission end puts both ends of the ridge in one posterior; a lag set at one end does not.

**Pinning `sigma_rstar` cuts off the wandering end and both runs then converge**, at R-hat
1.0 and 0/8000 divergences, but they converge onto the null:

| lags (1, 4, 7) | sigma_rstar 0.10 | 0.05 | 0.02 |
|---|---|---|---|
| `is_slope` | −0.058 (did not converge) | **−0.021** [−0.055, −0.002] | **−0.020** [−0.053, −0.002] |
| r\* sd | 0.16 to 1.47 by chain | 0.03 | 0.01 |
| `sigma_e` | 0.404 | 0.433 | 0.433 |
| w on t−1 / t−4 / t−7 | 0.12 / 0.30 / 0.55 | 0.22 / 0.32 / 0.41 | 0.22 / 0.32 / 0.40 |

The weights tilt toward the long lag, which is the right direction, but the marginals span
most of the simplex ([0.025, 0.467] on t−1) and flatten toward the 1/3 prior once r\* is
pinned. **Weakly identified; do not quote them.**

**(4, 8) is not on the ridge, and that was checked rather than assumed.** Its four chains
agree to a between-chain sd of 0.0011 against a pooled posterior sd of 0.0350, r\*'s amplitude
agrees to three decimals (1.615 to 1.617), and an independent seed reproduces the slope at
−0.3836 against −0.3827. Across 16,000 draws from the two seeds **nothing gets closer to zero
than −0.20**, so the flat-slope basin carries no posterior mass at this lag structure. The
mode is real, not a stuck chain.

**The reading.** This settles the LAG STRUCTURE and not the answer. Changing the lags moved
the slope by a few hundredths; changing `sigma_rstar` moves it twentyfold, at (4, 8) as much
as at (1, 4, 7). The conditioning in "How slow is r\*" above is untouched by any of this.

## The parameterisation trap, found and documented

The model can be written with the prior on r\* (rate units, as above) or on the line's
height (gap units). These are the same algebra and **different models**, because a prior
carried across coordinates without its Jacobian is a different prior.

With the prior on r\*, `is_slope` appears **twice**: as the steepness, and multiplying r\* to
position the line. The line's vertical freedom is `|is_slope| × sigma_rstar`, so a bigger
slope buys a more movable line, and the likelihood takes the deal.

| | prior on r\* | prior on the line's height |
|---|---|---|
| `is_slope` | −0.383 | **−0.092** |
| `sigma_e` | 0.240 | 0.225 |
| r\* range | −1.19 to 3.99 | −3.85 to 6.96 |
| r\* latest | +1.52 | +4.89 |

Same fit, a fourfold difference in the slope. The −0.092 version sits essentially on the
`is_curve` bench's independent −0.108, and is the more credible slope. Its r\* is unusable.

**Neither is neutral.** Prior on r\* is the quantity you can judge the plausibility of, and
it rewards inflating the slope. Prior on the height kills that incentive, and puts the prior
on a quantity nobody has an intuition about. The default is the first, with this warning.

## What the model measures well, and it is not r\*

Under the height parameterisation the intercept is **precisely identified**: posterior band
±0.1 against a prior band of ±1.65. Its path runs 0.25 through the 1990s, peaks at 0.64 in
2007, troughs at −0.35 in 2016, ends at 0.453.

Decomposing the gap's variance:

```
gap variance                 0.1772
explained by the intercept   0.0784    44%
explained by the rate term   0.0247    14%
```

The intercept does more than three times the work of the interest rate. **So what the model
calls r\* is, to a first approximation, the output gap's own slow-moving component divided by
a small number.** The shape every r\* chart here shows, flat 1990s, 2007 peak, 2016 trough,
recovery, is the gap's cycle, not information about rates.

That explains the apparent contradiction: the intercept is precisely identified because it
tracks something real; r\* is wild because getting from one to the other divides by a slope
the data can barely see.

## The disciplined version, and why it fails

`--rstar-form constant` gives one line for the whole sample: 120 observations, one free
number, points that cannot slide.

| | |
|---|---|
| `is_slope` | −0.034, 90% [−0.070, −0.005] |
| `sigma_e` | **0.432**, against a gap sd of 0.420 |
| r\* | 3.25%, 90% [1.37, 13.33] |

`sigma_e` equals the gap's own standard deviation: **the IS curve accounts for none of the
output gap.** The r\* interval spans an order of magnitude because the division reaches a
slope near zero.

```
intercepts    slope     sigma_e    r*
   1         -0.034      0.432     3.25  [1.4, 13.3]
 134         -0.092      0.225     4.89  a path
```

The middle column is what the extra freedom buys and the first is what it costs. With 134
intercepts there are more free latent values (135) than observations (120): every point gets
its own line and the fit cannot fail.

## The unfitted picture

`the-is-curve-before-the-model-touches-it.png` plots the gap against the raw weighted lagged
real rate, nothing on either axis fitted, free intercept.

- OLS slope **−0.015**, zero-gap crossing at **+5.53%**, which is off the right edge of the
  data. That is what dividing a small intercept by a near-zero slope produces.
- Three era clusters that contradict each other. 1995-2008 at rates of 2 to 4% with gaps
  scattered ±1 and no slope within it. **2013-2018 at rates below zero with NEGATIVE gaps**,
  the wrong sign, which is the RBA cutting because the economy was weak. 2020-2022 at −2%
  with gaps of +0.6 to +1.1, the right sign.

The companion `the-is-curve-drawn.png` uses the model's own stance on the x-axis and shows a
clean −0.383 line. **It is not evidence**: the horizontal coordinate was constructed from the
answer. The difference between the two pictures is entirely what r\* was allowed to do.

## What this package inherits

**The gap is a model output, not data.** It comes from `ystar_ustar`, where it is *defined*
as `c × (inflation − 2.5) + v` with c = 0.275, and potential is then the residual. In the
current run the inflation-defined part has sd 0.31 and the free part 0.28, so **roughly half
the left-hand side is a rescaling of inflation**. A relationship recovered between the rate
and that is part reaction function.

It also inherits `ystar_ustar`'s imposed `sigma_okun` of 0.20.

And the gap is only part of the cycle. `ystar_ustar`'s GDP equation is
`log_gdp = y* + gap + e_c` with `e_c` sd 0.462 against a gap sd of 0.420, so the gap is about
**45% of the variance** of GDP's deviation from potential. If rates work through channels
that do not show in inflation, that effect lands in `e_c` and this IS curve never sees it.

## The slope on a common footing with `rstar_hlw`

`is_slope` here has **no gap persistence behind it**: the line is `gap = is_slope × stance`,
so the coefficient is already the response to a SUSTAINED stance. `rstar_hlw`'s `a_r` is an
IMPACT coefficient and is not the same object. The comparable quantity there is
`a_r/(1 − a_y1 − a_y2)`, which its own units warning says in as many words.

Put on that common footing, the two families still disagree, and by a lot:

| | level slope |
|---|---|
| this model, prior on the line's height | −0.092 |
| this model, prior on r\* (the default) | −0.383 |
| `is_curve` bench, independent | −0.108 |
| `rstar_hlw`, long-run, across specifications | −0.42 to −2.11 |

So converting HLW's impact coefficient to a level response does not reconcile the two. It
moves HLW from looking implausibly weak to looking three to ten times STEEPER than the two
models that measure a level slope directly. Both readings cannot be right, and nothing here
establishes which is.

**Do not read the gap between them as a clean contradiction.** These are slopes against
different gaps. This model takes `ystar_ustar`'s gap as data; `rstar_hlw` estimates its own,
which has sd 2.04 against the 0.42 inherited here. A slope against a five-times-wider gap is
not the same coefficient, and no exercise in this package has put them on one gap.

**What does survive the comparison is the pattern.** In both models the IS-side coefficient
is settled by a modelling choice rather than by the data, and each reaches that outcome by a
different route:

- Here it is the **parameterisation**. −0.383 under the prior on r\*, −0.092 under the prior
  on the line's height, at essentially the same fit (`sigma_e` 0.240 against 0.225). See
  "The parameterisation trap".
- There it is the **lag**. −0.73 at t−6 against −1.29 at t−1, while persistence stays flat
  at 0.92 to 0.94 across the whole grid, so the AR(2) is not fitting anything about the
  rate either.

That is the more durable statement than any single slope: across both specifications the
data does not choose the IS-side coefficient, and the analyst's parameterisation or lag
choice does.

## Diagnosis: why the slope comes out weak

Three candidates, ranked, and only the first is well supported.

1. **Simultaneity.** The RBA raises rates when the gap is positive, which induces positive
   covariation whatever the structural slope. The monotone strengthening from lag 1 to 5 is
   the main evidence, and it is why the lags are long.
2. **Measurement.** The gap is 45% of the cycle and half of that is inflation, so the target
   is narrow and oddly constructed.
3. **Many and varied lags.** Plausible but **untested here**. These weights sum to one, so
   `rbar` measures the response to a *sustained* stance, the same object a single lag
   measures. Testing accumulation needs weights that do not sum to one, and nothing here
   does that.

For the neutral-rate question specifically the sustained-stance object is the right one, so
(3) matters much less than it would for measuring transmission.

## What would change the answer

**Give r\* a second job.** It is anchored to nothing here, which is deliberate (every anchor
imports the answer it then reports) and is why it can absorb anything. HLW ties r\* to trend
growth, `rstar_bonds` to a world real rate, `rstar_rba` to a policy rule.

**Or cut its degrees of freedom.** `--rstar-form linear` gives r\* two numbers: it still
moves over time, but the points cannot slide to meet the line. This is the only untried
option that both answers the original question and can be wrong. **Not yet run.**

**Or break the simultaneity.** Monetary policy surprises, measured from market pricing around
RBA announcements, would raise the identifying signal. That is a data acquisition problem and
would help every r\* model here.

## Not done

- `--rstar-form linear`, the one specification that could be falsified.
- An injection test: add a known +1pp stance over windows of several lengths, re-estimate,
  and measure how much comes back versus how much is absorbed into r\*. `rstar_rba` has one
  and its notes call it the model's honesty curve.
- Residual autocorrelation. `e_t` almost certainly inherits the gap's, which would mean the
  quoted intervals are several times too narrow. The `is_curve` notes measure 0.892 on the
  same data, inflating classical t-statistics about 4.2 times.
- Gap persistence terms. Deliberately absent. Adding them would NOT rescue the magnitude:
  with both series persistent, a no-persistence regression already recovers the long-run
  relationship, so persistence would split −0.03 into a smaller impact plus dynamics summing
  back to about −0.03.

## Quoting this model

Quote the **conditioning**, never the number. If a single figure is needed, r\* today is
around 1.5 to 2.7% real across every setting tried, which is the least sensitive thing the
model produces. "Today" is loose: see below.

**The latest value is not a current reading.** r\* at the final quarter is 2025Q2's estimate
carried forward four quarters by the random walk, because the likelihood reaches r\* only at
T−4. Quote it as "r\* as at T−4, extrapolated", or quote T−4 directly and say so. The path is not robust: the 2007 peak ranges 4.0 to 7.0 and the 2016 trough
−1.2 to −2.4 depending on `sigma_rstar` alone.

The **timing** of the peaks and troughs is stable across parameterisations. Note that r\*
correlates 0.73 to 0.83 with the real cash rate across the sweep, so much of that timing is
policy's own cycle traced back rather than an independent read on neutral.
