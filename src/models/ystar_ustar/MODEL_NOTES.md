# Joint y\* / u\*, one likelihood, and a gap that is not entirely inflation

`ystar` and `ustar` estimated together, plus one addition. Three states, three observation
equations, nine estimated parameters and two initial conditions, plus the three spline
coefficients that replace u\*'s state law. Four imposed: `sigma_c`, `sigma_ystar`, `sigma_g`
and `sigma_okun`.

```
STATES
  g_t   = g_{t-1} + e_g                        sigma_g imposed (0.015)
  y*_t  = y*_{t-1} + g_{t-1} + e_y             sigma_ystar imposed (0.078)
  u*_t  = sum_j c_j B_j(t)                     natural cubic spline, one knot
                                               at 2013Q1; 3 coefficients, and
                                               NO innovation variance at all

GAP
  gap_t = c·(4·pi_q,t - 2.5) + v_t             v ~ N(0, sigma_v)
                                               (--gap-pi-basis annual uses pi_ann instead)

OBSERVED
  log_gdp_t = y*_t + gap_t + e_c               sigma_e estimated
  u_t       = u*_t - beta·gap_t + e_o          sigma_okun imposed (0.20)
  pi_q,t    = q(2.5) + beta_pi·[q(pi^e_t) - q(2.5)] + gamma·(u_t - u*_t)/u_t
              + rho·d4pm_t + xi·GSCPI_t²·sign(GSCPI_t) + e_p
```

**u\* IS A SPLINE, NOT A DECAY.** The structure it replaced,
`u*_t = u*_{t-1} + phi·(u*_eq - u*_{t-1}) + e_u`, can only draw a monotone
approach: the sign of `phi·(eq - u*)` is fixed by which side of the
equilibrium the state opened on, so from an opening level of 10.77 it could
only ever report a fall. On the previous vintage 5.96 of its 6.03 point
decline was the zero-innovation curve, the 134 innovations moved it by at most
0.16 anywhere, and by 0.002 over the last two years, u\* having asymptoted
onto `ustar_eq`. Its endpoint was a fitted scalar rather than a reading of
recent quarters. The spline can turn, and does: +0.38 over 2015-2026 against
-0.38. It also removes `sigma_ustar`, one of the imposed variances. See "The
structure imposed on u\*" for the settings and what each one showed.

Every prior is taken unchanged from the two parent models, and so are `sigma_ystar`, `sigma_g`
and the spline coefficient prior, so a difference in the posterior is attributable to joint estimation and to
`v` rather than to re-tuning. The one departure is `sigma_okun`, which `ustar` estimates and
this model imposes: see "Why `sigma_okun` is imposed" for the sweep, which shows the answer does
depend on it beyond 0.4, and for why the data do not go there.

---

## Read this first: the model exists to estimate one number

`sigma_v`, and nothing else here is new.

**Why it cannot be estimated in `ystar`.** Write the free component into that model and the
GDP equation reads `log_gdp = y* + c·d + v + e_c`. Two mean-zero terms, one equation. Only
`Var(v) + Var(e_c)` is visible, the split is a flat ridge, and whatever comes back is the
prior. Worse than useless: the reported gap becomes `c·d + λ·(residual)` with λ set by the
prior variances alone, which is a continuous dial between `ystar` as it stands (λ = 0) and
detrended GDP, a filter (λ = 1). Both endpoints already exist as switches, since
`ustar --gap-source actual` is the λ = 1 case.

**Why it can be estimated here.** The gap enters the Okun equation too, as `-beta·gap`. So

```
  cov(GDP residual, unemployment residual) = -beta·Var(v)
```

and the split is identified. That single covariance is the entire informational gain from
joining the two models, and `--no-okun` is the control that demonstrates it: without the
second equation `sigma_v` should return its prior.

**What the answer decides.** If `sigma_v` is near zero, `ystar`'s identity is vindicated,
its 0.188 gap *is* the output gap, and `ustar`'s `beta_okun` = 2.14 is a real and
uncomfortable finding rather than an artefact. If `sigma_v` is large, `ystar` has been
reporting a slice of the cycle, `ustar` has been fed that slice, and 2.14 is the slice
showing up as an inflated slope.

**Expect it to be weakly identified.** The moment is a covariance between two large
residuals (`ystar` reports `sigma_e` = 0.508, `ustar` `sigma_okun` = 0.486). The
prior-versus-posterior check in `analyse.py` is there because "the posterior equals the
prior" is a live outcome, and it would mean the joint model failed at the one thing it was
built for.

**And the model is small on purpose.** Three states, three observation equations, five imposed
variances. It is not attempting a comprehensive business cycle decomposition, and it does not
claim to have measured the output gap independently. `sigma_v` is a **diagnostic on `ystar`**: it
asks how much cyclical movement the inflation identity misses, given that unemployment can see
some of it. Answering "materially more than a little" does not require knowing what the missing
part is made of, and the model makes no claim about that.

That boundary is worth stating because the model resembles a class of larger models it is not a
member of, and a reader who assumes otherwise will find defects that are declared choices.
Separating a latent cycle from correlated equation disturbances, adding cyclical measurements
(hours, vacancies, capacity utilisation, surveys) until the trend/cycle split is free, replacing
the defined gap with an estimated one: each is a coherent research programme and each is a
different model. `ystar`'s own notes make the same point about the machinery it dropped.

The imposed variances are the price of being small. The honest response to them is to sweep and
report, which is what the sweeps in this file do. It is not to add structure until they become
estimable, because the free-cycle experiment below shows where that ends on this observable set.

---

## What this does not fix

**The circularity is attenuated, not removed.** `ustar`'s Phillips curve regresses inflation
on `(u - u*)/u`, and by the Okun identity `u - u* = -beta·gap + e_o`, where `gap` still
contains `c·(pi - 2.5)`. Part of the regressor remains a rescaled copy of the dependent
variable, which manufactures a negative `gamma_pi` whether or not a Phillips relationship
exists. Measured in `ustar`, that mechanical term is **21.4%** of the variance of `u - u*`.

What changes is that `gap` now also contains `v`, which is not inflation, so the manufactured
share falls in proportion to how much of the gap is `v`. For the first time that proportion is
a number the posterior reports rather than an unknown. It does not reach zero, and no version
of this model makes it reach zero while the gap is defined off inflation at all.

**The default makes the circularity worse, deliberately.** The gap and the Phillips curve are
both on the quarterly trimmed mean, so the gap is `c·(4·pi_q - 2.5)`, an exact multiple of the
Phillips curve's own dependent variable, and the correlation between them is 1.0 by
construction. On the annual basis it is 0.828, measured on this sample, so `--gap-pi-basis
annual` is the lower-contamination version and it remains available.

Quarterly is the default anyway, on two grounds. It samples better everywhere except
`sigma_okun`: 45 divergences against 84, `c` ESS 3,176 against 2,170, `sigma_v` 1,689 against
1,154. And it matches the basis the Phillips curve is forced onto, and the one `ystar` is
moving to, so the package stops carrying two horizons of the same series.

The cost is real and unquantified. A correlation is not the contamination share, and nothing
has established what the 21.4% becomes under either basis. There used to be a measurable cost
as well, in that `sigma_okun` sampled worse on quarterly (ESS 77 against 257, with 25% of its
posterior mass below 0.10 against the prior's 8%, where the annual run had 6%). That is moot
now that `sigma_okun` is imposed, and it was part of why imposing it was worth doing.

**Two state variances are imposed, and neither sets anything.** `sigma_ystar` and `sigma_g`,
for the reason all the parents impose them: a free state beside a free residual is the
Stock-Watson pile-up pair, and `ustar` documented all three routes to estimating its own drift
failing. `sigma_ustar` used to be a third and is gone, because the spline has no innovation
variance to impose.

`sigma_ystar` shifts `sigma_v` by 0.031 across a grid from zero to three times the default (see
its sweep). `sigma_g` has not been swept here. The imposed variance that does matter is not a
state variance at all: it is `sigma_okun`, which has no external anchor and beyond 0.40
collapses the model into `ystar`. That leaves **one** imposed number carrying the answer, where
there used to be two.

**No IS curve, no r\*.** The repo's central negative finding is that the rate-to-activity link
is not identified in Australian data: `rstar_hlw` measures `a_r` ≈ −0.04 against `σ_IS` ≈ 0.70,
`nairu` gets `beta_is` = 0.084 [0.024, 0.139]. A three-star joint model was considered and
rejected on that basis, since `r*` would attach to the system only through the leg that is
missing. Note that 0.084 excludes zero, so "small and swamped" is more accurate than "absent".

---

## The pandemic window

`ystar` drops 2020Q2-2021Q3 from its likelihood, on the ground that potential output is not
well defined in a lockdown rather than merely hard to estimate. Here the same window is
dropped from **all three** equations by default, which is the consistent extension of that
argument rather than a new one. If inflation in those quarters was moved by free childcare and
administered fuel prices, the gap built from it is not a gap, and the Okun equation would be
fitting unemployment to a number that means nothing. Unemployment was distorted in its own way,
by JobKeeper holding the measured rate far below any reasonable reading of labour market slack.

`--exclude-scope gdp` restricts the exclusion to the GDP equation, which is literally what
`ystar` does when run alone, and is the setting to use if you want `c` directly comparable
with `ystar`'s. It is not the default because it leaves Okun fitting a meaningless regressor in
six quarters.

`ystar`'s notes record that the window boundaries have never been tested. That is inherited
here unchanged.

---

## Design decisions

**`v` is iid, not AR(1).** A persistent `v` is more plausible economically, since business
cycles persist. It is also how a joint model turns into a shock-allocation machine: a free
AR(1) cycle beside two free trends is three places for unexplained persistence to hide. iid
understates the cycle `v` can find, which makes the test conservative in the right direction.
A large `sigma_v` under iid is strong evidence; a small one is not proof of absence.

**`v` is non-centred**, `v = z·sigma_v` with `z ~ N(0, 1)`. This is the case non-centring was
designed for and the opposite of `rstar`'s: there the yield data pin the innovations well and
the transform built a funnel rather than removing one, costing 511 divergences. Here the data
are weakly informative about `v` by construction, which is exactly when non-centring pays.

**`y*` should not get uglier, and that is checkable.** `nairu`'s potential output is a
Cobb-Douglas production function built from HP-filtered factor inputs, so it inherits every
wobble in those filters. This model keeps `ystar`'s random walk with `sigma_ystar` imposed at
the same 0.078, and nothing in the new structure touches it. Adding `v` moves variation from
`e_c` into the gap, which if anything protects `y*`.

---

## Results (2026Q2 vintage)

**`sigma_v` is identified, and the output gap is about twice as wide as `ystar` reports.**

That is the statement to quote. The defined/free split below is an accounting convention that
moves with the inflation horizon and with `sigma_ustar`; sd(gap) does not.

Defaults: quarterly gap basis, u\* converging, `sigma_ustar` = 0.020, `sigma_okun` imposed at
0.20, 10,000 draws. **0 divergences, `r_hat` 1.000 throughout, minimum ESS 4,528.**

| Parameter | mean | 90% HDI | `ess_bulk` |
|---|---|---|---|
| `c` | **0.278** | [0.196, 0.361] | 4,952 |
| `sigma_v` | **0.334** | [0.238, 0.422] | 4,528 |
| `sigma_e` | 0.464 | [0.403, 0.528] | 10,556 |
| `phi_ustar` | 0.039 | [0.034, 0.044] | — |
| `ustar_eq` | **4.78** | [4.62, 4.93] | — |
| `beta_okun` | **1.265** | [0.948, 1.580] | 6,480 |
| `gamma_pi` | −1.152 | [−1.395, −0.937] | 12,891 |
| `beta_pi` | 0.361 | [0.115, 0.602] | 17,155 |
| `epsilon_pi` | 0.165 | [0.147, 0.182] | 18,235 |

| The gap | |
|---|---|
| defined by inflation, `c·(pi − 2.5)` | 51.9% of variance |
| free component `v` | 47.5% |
| sd(gap) | **0.420**, against `ystar`'s 0.188 |
| corr(e_c, e_o) | **−0.584** |

**`corr(e_c, e_o)` is a leftover, not the identifying moment.** The moment that identifies
`sigma_v` is the covariance of the residuals *before* `v` is removed, as derived above. What this
row reports is what is left after `v` has been extracted, and the likelihood assumes it is zero.
Across the 2x2 below it runs −0.578 / −0.165 / −0.623 / −0.584: only in the cell where `sigma_v`
is 0.873 does it approach zero, because there `v` is large enough to absorb the common movement.
So −0.58 is not corroboration. It is the part `v` failed to absorb, and it is the one piece of
evidence that the model's `cov(e_c, e_o) = 0` restriction is straining. Testing that means freeing
the correlation, which is only estimable while `sigma_okun` stays imposed: `sigma_v`, `sigma_e`,
`sigma_okun` and `rho` are four unknowns against the three moments of a 2x2 residual covariance.

| Headline | joint | separate |
|---|---|---|
| potential growth, y/y | 1.99 | 1.94 |
| output gap | +0.30 | +0.21 |
| u\* | **4.74** | 4.83 (`ustar`) |
| u − u\* | −0.39 | −0.48 |

### The u\* state law, run as a 2x2

u\* used to be a driftless random walk here as in `ustar`, and that prior is about 8 standard
deviations from its own fitted path. Backporting `ustar`'s convergence specification made **two**
changes at once: it added a pull toward an estimated equilibrium, and it halved `sigma_ustar` from
0.040 to 0.020. Those changes push `sigma_v` in opposite directions, and this section used to
report them as one move. Run as a 2x2, `sigma_v` (mean, 94% HDI):

| | `sigma_ustar` = 0.040 | 0.020 |
|---|---|---|
| driftless | 0.495 [0.344, 0.664] | 0.873 [0.621, 1.120] |
| converging | 0.277 [0.182, 0.376] | **0.334** [0.231, 0.442] |

sd(gap) 0.556 / 0.935 / 0.352 / **0.421**; free share 66.7 / 80.2 / 39.6 / **47.5**%;
`beta_okun` 1.343 / 1.452 / 1.130 / **1.265**; `gamma_pi` −1.022 / −0.564 / −1.481 / **−1.152**.
All four sample cleanly: 0 divergences, max `r_hat` 1.010, minimum ESS 686.

**The decay is what mattered.** At either variance it roughly halves `sigma_v`. A substantial part
of what this model was attributing to "cycle inflation cannot see" was u\* being unable to fall as
fast as the 1990s required: the sample starts 1993Q1 with unemployment near its post-recession
peak, and a driftless walk cannot make that descent, so the Okun equation needed a large free gap
component to reconcile unemployment with output. Give u\* a pull toward equilibrium and the demand
for `v` halves.

**The halved variance was working against it.** Tightening `sigma_ustar` stiffens u\* everywhere, so
unemployment variation u\* can no longer absorb must be explained by `beta·gap` instead. That is why
the driftless row rises from 0.495 to 0.873. The two changes netted to 0.334, so the decay's own
effect is larger than the bundled move suggested.

**The decay also removes the model's dependence on `sigma_ustar`, and that is the more useful
result.** Halving `sigma_ustar` moves `sigma_v` by 0.378 under the driftless walk and by 0.057
under the decay. The mechanism is that with the pull term u\* makes its big move through
`phi·(u*_eq − u*)` whatever its innovations are allowed to be, so it stops caring about
`sigma_ustar`; without it, innovations are the only way u\* can move at all. `sigma_ustar` has an
external anchor in `ustar`'s 2012Q4-2015Q4 inflation-band test, but a weak one, and under the decay
the answer no longer turns on it. This is the `sigma_ustar` sweep that "Still to run" used to list
as mattering most, done on the axis that mattered.

**`phi` is inert now, which is why the decay row is nearly flat.** With `u*_eq` = 4.78 and u\* at
4.74, the pull term is about 0.002 percentage points a quarter. The decay does its work in the
1990s descent and then gets out of the way, so what governs the recent u\* path is `sigma_ustar`
alone. The 0.277-to-0.334 spread is what remains of that, and it is small.

**What survives.** `sigma_v` is 0.28 to 0.33 across the two decay cells, the free share 40 to 48%,
sd(gap) 0.35 to 0.42 against `ystar`'s 0.188. The finding has now shrunk twice. Quote sd(gap) as a
range across `sigma_ustar`, not as a point, and do not quote the free share at all.

### The wage check, and why it does not bear on the decay

External validation, outside every likelihood: private-sector WPI growth (y/y, 1998Q3-2026Q2,
n = 112) regressed on `u − u*` and inflation expectations, HAC(4). Wages are in no model here, and
WPI is not in the gap's construction, so this is genuinely out of sample in the way that matters.

| | corr | slope | t | R² |
|---|---|---|---|---|
| joint `u − u*` | −0.748 | −0.476 | −3.84 | 0.672 |
| `ustar` `u − u*` | −0.737 | −0.449 | −3.48 | 0.663 |
| raw unemployment | −0.430 | +0.000 | +0.00 | 0.575 |

**The gap beats the level, and not marginally.** With expectations controlled, raw unemployment's
coefficient is exactly zero. u\* is adding information rather than relabelling `u`.

**It is the one labour-market slope here that is not circular.** In the `(u − u*)/u` form the
Phillips curve uses, the wage slope is **−2.00** against `gamma_pi`'s −1.15. `gamma_pi` is
uninterpretable because its regressor is partly a rescaling of its own dependent variable; the
wage slope has no such problem. Expectations pass through at 1.019.

**NAIRU-consistent private wage growth is 2.92%**, at `u = u*` with expectations at target. It
comes back at 2.92 across the entire `sigma_okun` sweep and both decay cells (2.82 and 2.86 in the
driftless cells), because it rests on the constant and the pass-through rather than on the gap.
That makes it the least specification-sensitive number this model produces. Implied trend labour
productivity growth, on the identity and a constant labour share, is about 0.4%.

**It cannot rank `sigma_okun`**, which was the hope: correlation runs −0.747 to −0.760 across 0.10,
0.20, 0.40, 0.70 and the free run. There is no external anchor for that parameter here. The
gradient that exists mildly favours 0.70, the cell where the model degenerates into `ystar`, and is
far too small to carry weight.

**It ranks the u\* specifications and prefers the driftless walk at 0.040** (corr −0.785, t −5.15,
R² 0.707, against the headline's −0.748, −3.84, 0.672), and `ustar` records the same direction from
its own wage check. **Neither bears on the decay**, for two reasons. WPI private annual begins
1998Q3, so the sample starts after the 1990s descent and never sees the failure the decay was
adopted to fix. And post-1998 the decay term is inert, since u\* is already at its equilibrium and
the pull is about 0.002pp a quarter, so what separates the cells over that sample is mostly
`sigma_ustar`, meaning how much u\* wiggles recently. If a wigglier u\* tracked wages better,
`conv040` should beat `conv020`; it does not, −0.727 against −0.748. There is no consistent story
in the ranking, which is the second reason not to lean on it.

The check is a regression on saved output, not a model run.

### The early sample: the model over-predicts inflation in the 1990s

Phillips residuals, observed less fitted, quarterly percentage points:

| period | n | mean residual | t |
|---|---|---|---|
| 1993-1995 | 12 | −0.071 | −1.81 |
| **1996-1998** | 12 | **−0.119** | **−2.73** |
| 1999-2019 | 84 | +0.009 | +0.54 |
| 2020-2026 | 26 | −0.019 | −0.63 |

Negative means fitted above observed, so over 1996-98 the model **over**-predicts inflation by
about half a point annualised, systematically.

**Which points at u\* being too high early, not too low.** `gamma_pi` is negative, so
over-prediction means the demand term is contributing too much inflation, which means the model
sees too little slack, which means u\* sits above where inflation wants it. The decay was adopted
because a driftless walk could not descend fast enough. These residuals say the decay does not
descend fast enough either over 1996-98. Smaller in degree, same in kind, and it is the second
piece of evidence about the early sample after the implied-u\* chart.

**An expectations explanation was tested and does not work.** The natural story is that
pass-through was stronger before the target was credible, so a fixed `beta_pi` = 0.361
under-weights expectations early and the shortfall lands on the demand term. That predicts
**under**-prediction. The model over-predicts, so raising `beta_pi` would add inflation to
quarters that are already too high and make the miss worse. The residual's correlation with the
expectations term is also −0.08 pre-1996, so it is not tracking that variable. The raw excess is
certainly present, with expectations above anchor by +0.64 in 1993-95 and +0.40 in 1996-98, but
the model is not missing on account of it. If anything the data could support *less* early
pass-through, which would want a reason before it was tried.

Twelve quarters and half a point annualised, so this is a lead rather than a finding.

### The early sample: u\* is not well identified, and the charts now say so

The lead above was followed. It ends in a negative result worth recording, and in the shaded
window on the u\* charts, `analyse.UNIDENTIFIED_WINDOW` = 1993Q1-1998Q4.

**Why u\* sits next to u in the early years.** The sample opens in 1993Q1 with u = 10.93 and the
posterior u\* at 10.77, a gap of 0.16 at the trough of the deepest recession since the 1930s. That
is not a judgement about 1993, it is arithmetic. Trimmed mean inflation averaged the 2.5 anchor
over 1993-95 (mean deviation −0.05), so `c·(pi − anchor)` has almost nothing to work with, and
Okun then places u\* within half a point of u. Across the whole sample u\* tracks the identity
`u + beta_okun·c·(pi − anchor)` with correlation 0.949 and mean absolute error 0.38 points, and
the fit is *tightest* in 1993-96 at 0.236.

**Okun outweighs the Phillips curve about 8:1 in placing the level.** Move u\* by 1pp at 1993Q1:
fitted u moves 1.00pp against `sigma_okun` = 0.20, which is 5.0 sd; the Phillips gap `(u − u*)/u`
moves 0.092, so with `gamma_pi` = −1.150 fitted quarterly inflation moves 0.105 against
`epsilon_pi` = 0.165, which is 0.64 sd. In log-likelihood terms that is roughly 60:1 per quarter.
Nothing on the nominal side can outvote it.

**Tried and failed: a phased anchor** (`anchor_phase`, kept and defaulting to `"none"`). The
expectations series does not reach the target until 1998: 3.50 in 1995Q1, 3.04 in 1997Q1, 2.70 in
1998Q1, 2.48 by 1998Q3, so holding the anchor at 2.5 from 1993 asserts an anchoring that had not
happened. Under `"step"` the anchor is expectations until 1998Q1, blended across 1998, the target
after. The prediction, registered before running, was that the 1996-98 residual bias would shrink
toward zero and 1999-2019 would not move. Result:

| window | n | none | t | step | t |
|---|---|---|---|---|---|
| 1993-1995 | 12 | −0.071 | −1.81 | **−0.145** | **−3.08** |
| 1996-1998 | 12 | −0.119 | −2.73 | **−0.168** | **−4.09** |
| 1999-2019 | 84 | +0.009 | 0.54 | +0.007 | 0.43 |
| 2020-2026 | 26 | −0.019 | −0.63 | −0.018 | −0.60 |

The middle held, as predicted. The early bias roughly doubled. u\* fell only 0.29 at 1993Q1 and
nothing after 2010, and the implied-u\* diagnostic got worse too: correlation 0.839 to 0.785, R²
0.705 to 0.617, amplitude 17.1x to 19.6x. **`beta_pi` is what ate it**, going 0.361 to 0.697. The
baseline is `(1 − beta_pi)·a_t + beta_pi·pi_exp`, so as `beta_pi` approaches 1 the anchor stops
mattering; and since `a_t` = expectations early, the higher weight pushed fitted inflation up
toward 3.4 where observed was 2.0. Everything else was stable: `c` 0.275 to 0.276, `beta_okun`
1.244 to 1.231, `phi` 0.039 both, `sigma_v` 0.332 to 0.342.

**What that settles.** The early level is not an anchor problem and cannot be fixed from the
nominal side. Okun is the binding constraint. The remaining untried candidate is a window-specific
`sigma_okun`, which would need to reach 0.70 before Okun's edge falls to 2.2:1. See the sweep in
*Why `sigma_okun` is imposed* for why loosening it across the whole sample degenerates the model
into `ystar`.

**Why the window ends at 1995Q4.** The shaded window is the band criterion's, not the widest one
available:

| year | 90% band | x mid-sample (0.210) | u\* change/qtr |
|---|---|---|---|
| 1993 | 0.550 | **2.61** | −0.225 |
| 1994 | 0.389 | 1.85 | −0.199 |
| 1995 | 0.304 | 1.45 | −0.167 |
| 1996 | 0.272 | 1.29 | −0.140 |
| 1998 | 0.263 | 1.25 | −0.102 |
| 2002 | 0.236 | 1.12 | −0.059 |

The concern is concentrated in 1993-94, at 2.6x and 1.9x the mid-sample width. By 1995 it is
1.45x and by 1996 1.29x, against the 1.1-1.2x the band holds through to 2002. So the estimate is
most of the way to its normal precision by 1995 and **largely settled by 1996**, which is what
the window marks.

What runs past 1996 is not the band but the other two diagnostics: the Phillips residuals stay
systematically negative until 1999 (table above), and expectations do not reach the target until
1998, which is also the phase end date here and in `nairu`. **That wider window is deliberately
not shaded.** A flat block to 1998Q4 asserts that 1997 is as doubtful as 1993, and the band says
plainly that it is not. The shading marks where the concern is; the residual bias running on to
1998 is this section's job, because prose can say it in degrees and a shaded rectangle cannot.

The shading claims only that u\* is not well identified there, which is what the band on the same
chart shows. The mechanism is this section's business, not the chart's.

### Why `sigma_okun` is imposed

Free, it was the model's one bad parameter and the only thing standing between this model and
clean diagnostics: `ess_bulk` 50, `r_hat` 1.08, four chains peaking in four different places,
45 divergences, and 24,000 draws still not enough. It trades off against `sigma_e` at a
correlation of −0.55, and that ridge was what the sampler crawled along.

Pinning one end dissolves it, and the gain is not marginal:

| | free (20,000 draws) | imposed at 0.20 (10,000 draws) |
|---|---|---|
| divergences | 45 | **0** |
| max `r_hat` | 1.080 | **1.000** |
| min scalar ESS | 50 | **1,610** |
| `sigma_e` ESS | 280 | **10,524** |
| `beta_okun` ESS | 1,200 | 6,551 |

Better absolute sampling from half the draws.

**The sweep: `sigma_v` does depend on it.** Four imposed values, everything else at the defaults:

| `sigma_okun` | `sigma_v` | sd(gap) | free share | `c` | `beta_okun` |
|---|---|---|---|---|---|
| 0.10 | 0.340 [0.24, 0.44] | 0.431 | 54.5% | 0.264 | 1.343 |
| **0.20** | **0.334** [0.23, 0.44] | **0.421** | **47.5%** | 0.278 | 1.265 |
| 0.40 | 0.261 [0.13, 0.40] | 0.332 | 24.7% | 0.264 | 1.226 |
| 0.70 | 0.085 [0.00, 0.22] | 0.224 | 0.1% | 0.206 | 1.395 |

All four sample cleanly: `r_hat` 1.000, minimum ESS 1,708, 0 divergences except 5 at 0.70.

At 0.70 the free component is gone, its interval includes zero, `c` = 0.206 is within noise of
`ystar`'s 0.188, and **the model degenerates into `ystar`**. So `sigma_okun` is not a nuisance
parameter, it is the finding restated: *if* unemployment tracks the gap tightly enough for its
equation residual to sit between 0.1 and 0.4, a large part of the cycle is invisible to inflation;
at 0.7 it is not.

This section used to report that doubling to 0.40 moves `sigma_v` by −0.009, and conclude that
nothing depends on it. That was a driftless-era measurement, taken when `v` was large enough to
swamp the change. Under the decay the move is **−0.073**, about 0.8 of a posterior sd, with
sd(gap) down a fifth and the free share halved. The decay did not remove the model's dependence on
an imposed variance. It removed the `sigma_ustar` one and left this one exposed.

**The free run says the data will not go near the cliff.** `--free-sigma-okun` under the decay,
10,000 draws:

| | free | imposed at 0.20 |
|---|---|---|
| `sigma_okun` | 0.141, ESS **16**, `r_hat` **1.19** | imposed |
| `sigma_v` | 0.335 | 0.334 |
| `c` | 0.270 | 0.278 |
| `beta_okun` | 1.304 | 1.265 |
| sd(gap) | 0.426 | 0.421 |
| divergences | 1 | 0 |

The ridge survives the decay, so imposing stays necessary: `sigma_okun` itself is no better
identified than it was. But the rest of the geometry improved a lot, from 45 divergences to 1, and
every parameter that matters sampled acceptably and did not move. And across 10,000 draws the
`sigma_okun` posterior put **all** its mass below 0.40 and **none** above 0.60, median 0.142, which
is below the imposed value. Read that softly. ESS 16 is not a posterior anyone should quote and
0.141 is not an estimate; what it supports is only the weaker claim that four badly-mixed chains
all declined to wander toward the collapse region.

**So the defence changes, and improves.** It used to be "`sigma_v` does not depend on
`sigma_okun`", which the sweep falsifies. The defensible version is that **the data confine
`sigma_okun` to the region where `sigma_v` is flat**: across 0.10 to 0.20 `sigma_v` moves 0.340 to
0.334, and the free run puts no mass beyond 0.40. The cliff is real and it is outside where the
likelihood goes.

**The objection, stated rather than buried.** 0.20 is this model's own posterior mean, so imposing
it is circular in a way the package's other imposed variances are not. `ystar`'s `sigma_ystar`
rests on an 8%-of-observed-variation rule; `ustar`'s `sigma_ustar` on the 2012Q4-2015Q4
inflation-band test. This one has no external anchor and the sweep shows it matters, so what
defends it is where the free run locates it, not the insensitivity claim that used to sit here.
`ustar`'s own `sigma_okun`, currently 0.486, sits between the 0.40 and 0.70 sweep points, implying
`sigma_v` near 0.21: accepting it would shrink the finding by about a third rather than destroy
it. This model's free posterior still puts no mass there, and 0.486 was already argued down as an
upper bound conditional on a frozen gap this model rejects, which is now an empirical statement
rather than a plausibility argument. `--free-sigma-okun`
restores the original specification; expect `sigma_okun` ESS around 16 and everything else fine.

**On the annual gap basis.** **PRE-BACKPORT: driftless u\*, `sigma_ustar` = 0.040. These figures
are not comparable with the headline table above and have not been remade under the decay.**
`c` = 0.376, `sigma_v` = 0.461, `beta_okun` = 1.244, free share 52.3%. The gap itself is unchanged:
sd 0.545 against the driftless quarterly run's 0.549, and the same 0.354 correlation with NAB
business conditions. **The defined/free split is an accounting convention, not a result:** it
moves with the inflation horizon while the object it decomposes does not. Do not quote it as a
finding.

That warning no longer depends on these stale figures. The `sigma_okun` sweep above moves the free
share from 54.5% to 0.1% while sd(gap) falls only from 0.431 to 0.224, and the u\* 2x2 above moves
it from 39.6% to 80.2%. Three separate settings the data do not determine swing the split across
almost its whole range, and two of those measurements are current. sd(gap) is the object with a
stable meaning; the share is bookkeeping.

### The `sigma_ystar` sweep: it does not matter, and the reason is instructive

`ystar`'s own grid for `ratio_ystar`, where `sigma_ystar = ratio_ystar x sigma_c` and the default
0.13 gives 0.078. `ratio_ystar` = 0 is the informative end rather than a corner case: it strips
the level innovation and leaves the integrated random walk that HP(1600) actually is.

| `ratio_ystar` | `sigma_v` | sd(gap) | free share | `sigma_e` | `beta_okun` | potential growth |
|---|---|---|---|---|---|---|
| 0.00 | 0.339 | 0.427 | 47.5% | 0.512 | 1.244 | 1.92 |
| 0.05 | 0.340 | 0.428 | 47.4% | 0.504 | 1.245 | 1.94 |
| 0.10 | 0.337 | 0.424 | 47.6% | 0.482 | 1.256 | 1.97 |
| **0.13** | **0.334** | **0.421** | **47.5%** | 0.464 | 1.265 | **1.99** |
| 0.25 | 0.321 | 0.403 | 47.8% | 0.386 | 1.321 | 2.14 |
| 0.40 | 0.309 | 0.386 | 48.1% | 0.301 | 1.384 | 2.27 |

Zero divergences in every cell, minimum ESS 1,194 at the loosest.

**The gap result is not conditional on it.** Across a grid spanning zero to three times the
default, `sigma_v` moves 0.031, about half a posterior sd, and sd(gap) stays at roughly twice
`ystar`'s 0.188 throughout. The free share is 47.4 to 48.1% in every cell.

**The `sigma_e` column says why.** A wobblier potential competes with the **GDP residual**, not
with the gap: `sigma_e` falls 41% across the grid while `sigma_v` falls 9%. That is the design
note in "Design decisions" confirmed from the other direction. The gap is tied to an observed
series, so loosening potential cannot eat it; only the unexplained part of GDP is available.

**One number does move, and it is not the gap.** Potential growth runs 1.92 to 2.27. That is
`ystar`'s headline product, quoted against the RBA's ~2.0, and a threefold change in an imposed
smoothness parameter shifts it by a third of a point, with the looser settings heading toward
Treasury's 2.5. `ystar`'s notes carry that sensitivity; it is repeated here because this model
reports potential growth too.

**So `sigma_okun` is the only imposed variance the headline still turns on.** `sigma_ustar` was
settled by the 2x2 above, `sigma_ystar` by this. `sigma_g` remains untested.

### `beta_okun` is prior-sensitive in level

`beta_okun`'s `Normal(0.5, 0.5)` prior is inherited from `ustar` and it binds. Doubling its sd:

| | prior sd 0.5 | prior sd 1.0 |
|---|---|---|
| `beta_okun` | 1.265 [0.91, 1.64] | **1.425 [0.94, 1.98]** |
| `sigma_v` | 0.334 | 0.302 |
| `c` | 0.278 | 0.251 |

A move of 0.16, about 0.8 of a posterior sd, and still rising as the prior is released: a prior
mean of 0.5 that this sample rejects at three standard deviations is doing real work.

**The comparison with `ustar` survives**, because 2.144 was estimated under the same prior, so
"joint estimation drops `beta_okun` from 2.14 to 1.27" is like-for-like. **The level does
not.** `beta_okun` is somewhere between 1.3 and 1.5, is still well above the textbook 0.4, and
its exact value is partly the prior's. Do not read "close to textbook" into it. 0.5 is kept as
the default because inheriting both parents' priors unchanged is what makes the comparison mean
anything; `--beta-prior-sd` runs the alternative.

### The three things that make this believable

**1. The `--no-okun` control fails exactly as predicted.** Drop the Okun equation and `sigma_v`
loses its identifying moment. It does not merely widen: `ess_bulk` collapses to **14**, `r_hat`
to 1.21, with 398 divergences. The sampler cannot explore a ridge that has become flat. And
`c` returns to **0.190**, which is `ystar`'s 0.188 to within noise. The covariance is doing the
work, demonstrated rather than asserted.

**2. The `--no-phillips` control changes nothing.** `sigma_v` = 0.334, `c` = 0.277,
`beta_okun` = 1.265, sd(gap) 0.418. Not merely inside the main run's intervals: identical to it
to three decimals. **The result does not depend on the Phillips curve being in the likelihood at
all**, so it cannot be an artefact of the circularity that motivated this whole exercise. This is
the strongest single piece of evidence here.

That `sigma_v` survives without the Phillips curve is a statement about `sigma_v`, not a case for
dropping the equation. See "Why the Phillips curve stays".

**3. The prior does not drive it.** Quadruple the prior mean and the posterior moves 1%:

| prior sd | prior mean | `sigma_v` | posterior sd | `c` | `beta_okun` | sd shrinkage |
|---|---|---|---|---|---|---|
| 0.5 | 0.399 | 0.331 | 0.056 | 0.275 | 1.275 | 81% |
| **1.0** | 0.798 | **0.334** | 0.057 | 0.278 | 1.265 | 91% |
| 2.0 | 1.596 | 0.335 | 0.057 | 0.278 | 1.263 | 95% |

The posterior mean moves 0.004 across a fourfold change in prior scale, and its sd does not move
at all. The posterior is the likelihood's, not the prior's.

### What it says about the two parent models

`ystar` has been reporting a slice of the cycle rather than the output gap. Its own residual
`e_c` was carrying roughly as much cyclical variation again as its published gap, and
unemployment can see it. The gap is about twice as wide as `ystar` reports, sd 0.421 against
0.188.

`ustar`'s `beta_okun` = 2.14 was substantially the consequence of being fed that slice. Given
the whole gap it falls to **1.27**, a 41% reduction toward the textbook 0.4. It does not reach
it: the puzzle is reduced, not resolved, and an Okun coefficient three times textbook still
wants an explanation.

`gamma_pi` barely moves, −1.150 in `ustar` against −1.152 here, which is consistent with the
contamination diagnosis: the manufactured component was a fifth of the regressor's variance, and
diluting it should shift the coefficient a little rather than overturn it. Note how little
reassurance that carries on its own, given the same coefficient runs −0.564 to −1.481 across the
u\* 2x2.

### Why the Phillips curve stays

`--no-phillips` is a control, not a candidate default, and the reason is worth stating because the
control's own result invites the opposite conclusion.

**It belongs here.** This is the model where output, unemployment and inflation are reconciled
against one set of stars. Prices are one of the three legs, not an optional extra bolted to a
GDP-and-Okun core: drop the equation and u\* keeps only an indirect tie to inflation through the
gap, and the model stops being able to say anything about what the labour market is doing to
prices. `sigma_v` surviving `--no-phillips` shows the *gap* result is not an artefact of the
circularity. It does not show the equation is surplus.

**And it is the only place the supply decomposition comes from.** `results.inflation_decomposition`
splits observed inflation into anchor, expectations, demand and supply, where supply is
`rho_pi·d4pm + xi_gscpi·GSCPI²`. `rstar` reads that supply column through its `observations.py`,
puts it on a four-quarter basis, and its Taylor rule looks through it, asymmetrically under
`config.supply_positive_only`: decline to tighten into a supply-driven overshoot, but still ease
when supply is holding inflation down. Running this model with `--no-phillips` as the default would
remove the input to the only other model that consumes this one.

**Two caveats, because the circularity reaches the supply term too.** First, `gamma_pi` should not
be quoted as an estimate of the Phillips slope: its regressor `(u − u*)/u` is partly a rescaled
copy of its own dependent variable. The 2x2 above makes that concrete, with `gamma_pi` running
−0.564 to −1.481 across four cells that differ only in the u\* state law.

Second, and this one lands on the supply term rather than on `gamma_pi`. Through the Okun identity
the demand term contains roughly `−gamma·beta·c·4·pi_q / u`, which at the headline values
(−1.152, 1.265, 0.278, u ≈ 4.7) is about **+0.35·pi_q on the right-hand side**. The equation is
then effectively fitting 0.65 of quarterly inflation with everything else, so `rho_pi` and
`xi_gscpi` are scaled up by something like 1/(1 − 0.35), and the supply contribution `rstar` looks
through would be too large by that factor. That is arithmetic on the reported coefficients, not a
measured bias.

The check is cheap and has not been done: compare `rho_pi` and `xi_gscpi` across the quarterly
basis, the annual basis (`--gap-pi-basis annual`, where the gap-to-inflation correlation is 0.828
rather than 1.0) and `ustar`'s own Phillips curve. Those two coefficients sit on external
regressors that are not functions of any state, so the scaling above is the only route by which
the circularity can reach them. If they are stable across bases the concern is empirically small;
if they move by something like half, `rstar` has been looking through too much and should read the
annual-basis run.

### Sampling

The posterior has a ridge, because `sigma_v` and `sigma_e` split one variance. That is the
geometry of the question rather than a defect, and it shows up as low ESS rather than as bias.
`target_accept` 0.99 cut divergences from 22 to 3 while cutting `sigma_e`'s `ess_bulk` from 398
to 140, and moved no estimate by more than 0.005, so the default stays at 0.95 and the answer
is bought with draws instead.

**Traces now carry the pointwise log likelihood**, so variants can be ranked with LOO or WAIC
instead of by comparing coefficients across tables, which is what every sweep in this file
currently does. `SamplerConfig.log_likelihood` defaults on in `ystar/base.py`, which `ystar`,
`ustar`, `rstar` and this model all share. Two things to know before using it.

**Only compare runs that observe the same data.** `--no-okun` and `--no-phillips` drop an observed
variable, so their criteria are not comparable with the full model's. Every sweep here is.

**And treat a ranking that prefers flexibility with suspicion.** Pointwise LOO assumes
observations are exchangeable given the parameters, which is false for a state-space model on
time series: neighbouring quarters share latent states, so leaving one out leaves most of its
information in the model through its neighbours. LOO therefore under-penalises a flexible latent
path. Given that the wage check independently favoured the more mobile driftless u\*, a LOO
ranking that also favours flexibility is partly the method talking.

`rstar` is the exception and stores nothing, by design. Its likelihood is written entirely with
`pm.Potential`, so it has no observed RVs, there are no pointwise contributions to store, and
LOO would be meaningless on it. `sample_model` detects that and skips the request; without the
guard PyMC's JAX path raises, since it returns `None` where it expects a list.

### Reproducing

```bash
./run-ystar-ustar.sh                                 # the headline run, 10,000 draws
./run-ystar-ustar.sh --no-okun     --prefix ystar_ustar_nookun   # the control that must fail
./run-ystar-ustar.sh --no-phillips --prefix ystar_ustar_nophil   # inflation off the LHS
./run-ystar-ustar.sh --sigma-v-prior 0.5 --prefix ystar_ustar_sv05
./run-ystar-ustar.sh --sigma-v-prior 2.0 --prefix ystar_ustar_sv20
./run-ystar-ustar.sh --free-sigma-okun --draws 6000 --prefix ystar_ustar_freeso
./run-ystar-ustar.sh --sigma-okun 0.40 --prefix yus_so40   # is the imposed value binding?
./run-ystar-ustar.sh --beta-prior-sd 1.0 --prefix ystar_ustar_bwide

# The sigma_ystar sweep, on ystar's own grid (0.13 is the default, i.e. the headline run).
for r in 0.0 0.05 0.10 0.25 0.40; do
  ./run-ystar-ustar.sh --ratio-ystar $r --prefix "yus_ry$(echo $r | tr -d '.')" --no-analyse
done

# The phased anchor, which does not work. See "The early sample".
./run-ystar-ustar.sh --anchor-phase step  --prefix yus_anchor_step
./run-ystar-ustar.sh --anchor-phase glide --prefix yus_anchor_glide

# The u* state law 2x2. The fourth cell is the headline run itself.
./run-ystar-ustar.sh --sigma-ustar 0.040 --prefix yus_conv040   --no-analyse
./run-ystar-ustar.sh --no-ustar-converge --sigma-ustar 0.020 --prefix yus_drift020 --no-analyse
./run-ystar-ustar.sh --no-ustar-converge --sigma-ustar 0.040 --prefix yus_drift040 --no-analyse
```

## The structure imposed on u\*

u\* is in no dataset, so something must say what shapes it may take. That is
`--ustar-structure`, and it is the choice the fork was built to test.

| | u\* 1993Q1 | now | post-2015 | band | gap 1993Q1 | now | ESS |
|---|---|---|---|---|---|---|---|
| `decay` | 10.77 | 4.74 | -0.38 | 0.25 | -0.10 | +0.30 | 4257 |
| **`spline`, 1 knot 2013Q1** | **8.17** | **5.06** | **+0.38** | 0.37 | -1.80 | **+0.49** | 1333 |
| `spline`, 2 knots | 10.20 | 4.56 | -0.66 | 0.23 | -0.42 | +0.16 | 3500 |
| `spline`, 3 knots | 9.95 | 4.31 | -1.15 | 0.24 | -0.56 | +0.01 | 4393 |

`walk` also exists, a driftless random walk at the imposed `sigma_ustar`, and
`--ustar-drift` applies only to it.

**One knot, at 2013Q1.** Three coefficients after the natural boundary
reduction: enough for u\* to decline and then level off or turn, not enough to
trace a cycle. It is the only setting whose post-2015 slope is positive, which
is the property the spline was adopted for: the decay structure cannot report
a rise at all, so its -0.38 is the shape rather than a reading.

**More knots do not help.** Two and three knots both sample better but return
the decay structure's answer: they reopen a high 1993 level and a falling
endpoint, and the third knot takes u\* to 4.31 with a post-2015 slope of
-1.15. The extra freedom is spent on the early sample, where nothing can
arbitrate it.

**The sampling cost is real.** One knot has the weakest ESS of the four at
1333, against 4257 for decay. R-hat is 1.00 and divergences zero in every
case, so it passes, but it is the least comfortable of them.

---

## The gap definition, and what the identity version showed

`--gap-spec identity` replaces `gap = c·(pi - 2.5) + v` with `gap = y - y*`,
deleting `c`, `v`, `sigma_v` and `sigma_e` and estimating `sigma_okun` instead
of imposing it. It is not the default, and the comparison is close enough to
be worth recording rather than dismissing.

| | beta | sigma_okun | u\* now | gap 1993Q1 | min | now | sd | resid 93-99 | ESS |
|---|---|---|---|---|---|---|---|---|---|
| **defined (default)** | 1.434 | 0.200 imposed | 5.06 | -1.80 | -1.88 | +0.49 | 0.67 | **-0.80** | 1333 |
| identity | 0.468 | 0.346 estimated | 4.67 | -6.44 | -8.73 | +1.28 | 2.32 | -0.24 | 6145 |

**What the identity gap wins.** Far better sampling, no imposed `sigma_okun`,
and a much smaller bias against what the Phillips curve alone implies for u\*
over 1993-99, -0.24 against -0.80.

**What it loses.** A 1992 output gap of -8.7 and a current gap of +1.28 with
inflation at 3.6. Its gap has an sd of 2.32, which is not a recognisable
business cycle, and u\* no longer turns up after 2015.

**The known trap.** Under the identity gap the Okun equation sees only
`beta·gap`, so `(beta, y*)` and `(-beta, y* reflected through y)` fit it
identically and nothing else breaks the tie: the Phillips curve runs on the
unemployment gap and never touches the output gap. The mirror is reachable
whenever the trend prior is loose enough. At `ratio_g` 0.05 one chain in four
finds it, R-hat goes to 1.53 and ESS to 7. `--one-sided-beta` bounds the slope
at zero and closes it, and that bound is an assertion that Okun's law has the
expected sign rather than a finding.

**The circularity charge is weaker here than in `ustar`.** There the gap
arrived as data and was exactly `0.1882·(pi - 2.5)`, so the Okun equation was
a second copy of the Phillips curve. Here `sigma_v` is 0.603 against `c` of
0.334, so most of the gap is `v` rather than inflation.

**Quote the -0.80 wherever the default's 1990s numbers are used.** It is the
largest residual bias of any specification tried, and it is the price of the
defined gap.

---

## Explored and did not work: an error-correction Okun

`--okun-form ec` replaces the level relation with

```
  du_t = -kappa·(u - u*)_{t-1} - theta·gap_{t-1} - gamma·d(gap)_t + e_o
```

and reports the long-run slope as `beta = theta/kappa`. It is Okun's original
statement, changes against growth relative to potential, with an
error-correction term so the level of u\* is still identified. **It does not
identify on this data.**

Two attempts, and the second failure is the informative one.

The first corrected toward `u = u*`, dropping the gap from the long-run
relation. Under the identity gap that leaves `y*` appearing only inside
`d(gap)`, a difference, so the LEVEL of `y*` has nothing pinning it: the
current output gap came back at **+26.10** with an sd of 8.05.

The second corrected toward the right relation, `u = u* - beta·gap`, which
fixed the economics and not the sampling: R-hat 1.35, ESS 9, and 3296
divergences in 10,000. Estimating the product `theta = kappa·beta` directly,
which removes the ridge where `kappa` up and `beta` down leave the likelihood
unchanged, improved it to 1516 divergences and no further.

**The cause is in the data, not the parameterisation.** The three regressors
the form introduces are collinear because Okun's law is what makes them so:

| | u - u\* (lag) | gap (lag) | d(gap) |
|---|---|---|---|
| u - u\* (lag) | 1.000 | **-0.936** | 0.176 |
| gap (lag) | -0.936 | 1.000 | -0.261 |
| d(gap) | 0.176 | -0.261 | 1.000 |

The level form *asserts* `u - u* = -beta·gap` and estimates one coefficient.
The error-correction form puts both series in with separate coefficients and
asks the data to say how much of unemployment's movement is the level gap and
how much is its own disequilibrium, when the two move together almost one for
one. There is a long flat ridge in `(kappa, theta)` and the chains sit at
different points on it.

**So the level form is not a fallback.** It is the same long-run relation with
the adjustment speed set to one instead of estimated, which is the only
version this sample can support.

---

## Explored and did not work: a free cycle instead of the inflation anchor

`--gap-spec cycle` replaces `gap = c·(pi - anchor) + v` with a free AR(1) latent that GDP, Okun
and the Phillips curve all observe. The motivation was real: it would remove the residual
circularity outright, since inflation would appear once as a dependent variable rather than
also constructing the Phillips curve's regressor, and it would retire the two-horizon problem
with it. It was also suggested by the data, since `v` comes back with a lag-1 autocorrelation
of 0.913 despite an iid prior.

**It collapses, twice, and the second failure is the informative one.**

The first attempt used `sigma_c` = 0.60 as the AR(1) *innovation* sd, which implies an
unconditional amplitude of `sigma_c/sqrt(1 - rho^2)`. With `rho` free to 0.99 that is an
amplitude prior of up to 4.3, and the gap duly became a second trend at sd 2.15.

The second attempt fixed that, writing the state in terms of its stationary sd so persistence
and amplitude are orthogonal. The amplitude behaved (sd 2.15 to 0.74). Nothing else did.

| | first attempt | after the amplitude fix |
|---|---|---|
| `ess_bulk`, typical | 6 to 12 | 7 to 11 |
| `r_hat` | up to 1.71 | up to 1.58 |
| `rho_gap` | 0.976 | 0.978 |
| `beta_okun` | 0.289 [−0.721, 0.675] | 0.770 [−2.405, 2.120] |
| `kappa_gap` | 0.020 [−0.070, 0.064] | 0.064 [−0.188, 0.189] |
| `sigma_okun` | 0.024 | 0.023 |
| corr(gap, u − u\*) | −0.999 | −0.999 |

`rho` = 0.978 on 134 quarters is indistinguishable from a unit root, so the gap is a second
trend competing with `y*` for the level of GDP. It absorbs the unemployment cycle entirely and
leaves inflation to expectations (`beta_pi` → 1.0). Both slopes straddle zero in both runs: the
model reports no relationship between the output gap and either unemployment or inflation,
which is the signature of a state that fits everything.

**What it establishes about the specification that works.** Tying the gap to an observed series is
what stops it drifting into being a trend. Loosen that to `c·d + v` and the model estimates
cleanly; remove it and no parameterisation rescues the decomposition. The only thing that would is
bounding `rho` well below 1, which imposes the cycle's persistence by hand and buys nothing over
`defined`.

State the conclusion no wider than the experiment supports: this shows that **some** observed
anchor is needed, on the observable set this model has. It does not show that inflation is the
only series that could serve, and nothing here tests any other candidate. Inflation is the anchor
this package uses because it is the one `ystar` was built around, not because it was selected
against alternatives.

The switch is kept so the test is repeatable, in the same spirit as `rstar`'s
`noncentred_wedge`. It is not a candidate specification.

---

### Still to run

Inherited obligations rather than new ideas.

```bash
./run-ystar-ustar.sh --gap-pi-basis quarterly   # the higher-contamination basis
./run-ystar-ustar.sh --exclude-scope gdp        # makes c comparable with ystar's
./run-ystar-ustar.sh --two-sided-c              # is the sign of c a finding or a prior?
./run-ystar-ustar.sh --sigma-ustar 0.024        # and the rest of ustar's sweep
```

**`sigma_ustar` and `sigma_ystar` are both now answered**, in "The u\* state law, run as a 2x2"
and "The `sigma_ystar` sweep". Neither moves the headline: 0.057 and 0.031 on `sigma_v`
respectively. **`sigma_g` is the one state variance never swept on this model**, and it is the
remaining item of that kind, though `sigma_ystar`'s result makes a large effect unlikely since
`sigma_g` reaches the gap only through the same potential-output block.

Two other obligations. The supply-term check in "Why the Phillips curve stays": compare `rho_pi`
and `xi_gscpi` across the two gap bases and against `ustar`, because `rstar`'s Taylor rule looks
through a supply contribution this equation may be scaling up by about half. And the annual-basis
run, which is the one pre-backport figure still quoted anywhere in this file; it is marked as such
in "Why `sigma_okun` is imposed".

The four robustness runs (`--no-phillips`, both `sigma-v-prior` runs, `--beta-prior-sd`) **have**
been remade under the current defaults, and all four now agree with the headline. They previously
reported `sigma_v` in the 0.44 to 0.50 driftless-era band, outside the current interval, which is
what made this section necessary.

---

## Comparing specifications (`--compare`)

`--compare` runs this model nine ways and charts the results together. It is not a different
model: each specification is a set of this model's own flags, re-estimated only if its saved run is
not from today. One is the default run itself; the others save to their own `yus_sum_*` prefix.
Every run then writes its own charts, the default to `charts/YStarUStar/` as a plain run does and
the others beside it, named for what sets them apart (`charts/YStarUStar-k2/`,
`charts/YStarUStar-decay_id/` and so on), and the combined charts follow. Eight cross the two
choices the model actually has to make, and the ninth adds the tapered random walk:

| | inflation-defined gap | gap = y - y\* |
|---|---|---|
| u\* decays to a level | x | x |
| **u\* is a spline, 1 knot** | **x** (the default) | x |
| u\* is a spline, 2 knots | x | x |
| u\* is a spline, 3 knots | x | x |
| u\* is a tapered random walk | x | |

The tapered walk is a driftless random walk whose step size falls linearly from
`taper_sigma_early` at the sample start to `taper_sigma_late` at `taper_end`. It imposes no shape,
only how far u\* may move each quarter. It needs a much tighter schedule here than a model without
Okun would: the Okun equation pulls u\* towards unemployment, and a looser walk follows it through
every cycle, collapsing `sigma_v` and erasing the labour-market tightness of 2007-08 and 2022-23.
It is run on the inflation-defined gap only.

Down a column is how much the structure imposed on u\* matters; across a row, how much the
definition of the gap matters. **Crossed rather than laddered** so the two cannot be confounded:
with a cell missing, a difference between columns could always be the structure that was only
tried on one side. The second knot sits at 1996Q1, giving the early sample a shape of its own,
which is where the specifications disagree most. The default's cell is the default run itself.

All nine share a sample, an expectations series and an inflation measure, so agreement within a
column is close to arithmetic and only disagreement informs. **Nothing here is a
recommendation**: settings argued against elsewhere in these notes, decay above all, are in it
because it is the set tested before settling.

**The identity gap needs a bound on the Okun slope.** With the gap defined as y - y\*, the Okun
equation sees only `beta x gap`, so `(beta, y*)` and `(-beta, y*` reflected through `y)` fit it
identically, and the Phillips curve, which runs on the unemployment gap, cannot break the tie.
Every identity-gap specification therefore runs `--one-sided-beta`, an assertion that Okun's law
has the expected sign. The cost: with no GDP residual, national accounts noise lands in the gap
and from there in the Okun residual, so `beta_okun` is attenuated and the gap is several times
wider than under the inflation-defined definition. The identity gap with a two-sided beta is
deliberately absent: at the default trend prior it matches the one-sided run, and its only
distinctive behaviour is the mirror mode.

**How the fit column is scored.** Under the identity gap the GDP equation is a definition and
carries no likelihood, so a model-wide criterion would compare models fitted to different data.
The score is leave-one-out over the two equations all eight observe, unemployment and inflation,
concatenated so a point is one quarter of one equation. It ranks predictive accuracy on those
targets and settles nothing about which model is true.

**What it found.**

- **The fit column does not separate them.** The spread of elpd is within its standard errors;
  no specification should be chosen on it.
- **Pareto k is the real signal.** Every inflation-defined run has a large share of unreliable
  observations; the identity runs have almost none. That is the circularity showing as a
  diagnostic: when the gap is `c x (pi - 2.5)`, inflation sits on both sides of the Phillips
  curve, so dropping one quarter moves the fit a lot. The inflation-defined elpd figures should
  not be read at face value.
- **For the endpoint, the gap definition matters and the u\* structure barely does.** Potential
  growth is untouched by either choice.
- **Every specification reads u\* above what the Phillips curve alone implies in 1993-99.** The
  bias is shared, so it discriminates nothing and points at the sample start; the identity runs
  are the less biased half.
- **None of them fixes 1993Q1.** On the same data the eight place u\* and the gap there over a
  wide range: the inflation-defined gap puts output barely below potential with unemployment near
  11 per cent, which cannot be right, and the identity gap gets the order right but swings by a
  factor of two across settings the fit cannot tell apart. The 1990-91 recession and the
  disinflation are outside the sample, so the model opens mid-recovery with no information about
  what it is recovering from. Nothing before 2000 should be treated as an estimate.

**What it prints and charts.** Each run's own full set of charts, as above, then a table per specification (the shared-target elpd and its standard
error, bad Pareto k, R-hat, ESS, divergences, the Phillips-implied residual overall and for
1993-99, u\* at the start and now, its post-2015 change and band width, the gap at the start and
now and its volatility, and potential growth), and six charts in `charts/YStarUStar-compare/`:
u\* against the unemployment rate, the output gap, potential growth, potential output, and the
range across specifications for u\* and for the gap. Colour is the u\* structure and dashing the
gap definition. The unidentified window is shaded only when the sample opens on it. The gap
chart's axis is set by the identity runs, which squashes the inflation-defined gaps near zero.

A saved run counts as current if its trace was written today: a proxy for current data rather
than a check of ABS and RBA vintages, erring the right way since a stale run is always re-run. A
full refresh is nine estimations at a few minutes each.

```bash
./run-ystar-ustar.sh --compare                 # re-estimate anything not from today, chart every run, then combined
./run-ystar-ustar.sh --compare --analyse-only  # the same from the saved runs as they stand
```

---

## Files

```
src/models/ystar_ustar/
├── config.py         # ModelConfig: the imposed variances, sigma_v, the switches
├── observations.py   # assembles GDP, both inflation horizons, u, expectations, shocks
├── estimate.py       # builds and samples the PyMC model, saves the trace
├── results.py        # JointResults: the gap decomposition and the residual covariance
├── analyse.py        # charts and the prior-versus-posterior check
├── cli.py            # the command-line parser, and a run from its arguments
├── compare.py        # the --compare specifications, refreshing and loading them
├── compare_charts.py # the --compare table and charts
└── run.py            # entry point: a run, or --compare
```

Reuses `ystar`'s `scale_equation` and `potential_output_equation` unchanged rather than
copying them, so the potential block cannot drift away from its parent.

Charts land in `charts/YStarUStar/`. Most are drawn by calling `ystar`'s and `ustar`'s own
plotting functions through the adapters in `analyse.py`, rather than reimplementing them: those
modules carry quarterly-axis handling, the excluded-window shading, the off-scale annotation on
the gap composition and the band-widening convention on u*, none of which is worth maintaining
twice.

Four are specific to this model.

- **The gap decomposition** and **the residual pair**, which neither parent can draw: `ystar` has
  no free component to separate out, and `ustar` receives the gap as data.
- **"What inflation alone says u\* is, quarter by quarter"** inverts the Phillips curve each
  quarter, residual set to zero, and plots that against the fitted u\*. The implied series moves
  17 times as much quarter to quarter (sd 0.995 against 0.058) while correlating 0.84 with the
  fitted path, so the state law is filtering rather than overriding. It also makes visible what
  the notes can only assert in words: the 90% band is roughly ±0.15 while the implied series
  scatters ±1, because the band reports uncertainty conditional on `sigma_ustar` = 0.020 rather
  than uncertainty about where u\* is.
- **"The Phillips curve as specified"** is a partial-regression plot: the equation's own
  regressor `(u − u*)/u` against inflation stripped of anchor, expectations and supply terms, so
  the fitted line is `gamma_pi` through the origin. It shows the slope is identified almost
  entirely by the tight side. Splitting the fitted sample: gap < −0.10 gives slope −1.82 with
  r = −0.66 on 30 quarters, while gap > 0 gives −0.45 with r = −0.13 on 66. The slack side is a
  formless cloud. Those 30 tight quarters are two episodes, 14 in the 2000s and 16 since 2020,
  which is consistent with the relationship being observable only when policy is not offsetting
  it. The competing reading is that the post-2020 points are the supply-shock quarters and the
  model's supply terms under-remove the shock; the 2000s cluster is the counterweight, since it
  is a tight labour market without a global supply shock and sits on the same line.
