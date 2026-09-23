# HLW Bayesian r-star Model: Model Notes

A Bayesian (PyMC + NumPyro NUTS) implementation of the Holston-Laubach-Williams 2017 model,
applied to Australian quarterly data. It sets out to estimate the natural rate of interest r*
jointly with potential output, trend growth and the output gap.

## THE VERDICT: IT DOES NOT WORK AS AN r* MODEL, AND CANNOT

**It does not produce a usable r\*, and you can establish that by counting, before running
anything.** r* here is trend growth plus an unanchored latent that no observation equation
touches. What comes out is the assumption, at whatever smoothing you chose.

This was not fixed on 2026-09-12 and is not fixable inside HLW. **`rstar_hlw` is not a source
of r\* for Australia**, it is deliberately excluded from `rstar_summary`, and its r* series
should not be quoted, charted as an answer, or fed downstream. Use
[`rstar_bonds`](../rstar_bonds/MODEL_NOTES.md) or [`rstar_rba`](../rstar_rba/MODEL_NOTES.md),
which have what this lacks: an observable that loads on the target roughly one-for-one.

**What the 2026-09-12 work DID fix is the trend/cycle decomposition**, which is now credible
and passes an external Okun check at three separate points. That is worth using, and it is a
separate claim from the one above. Do not let the repaired gap be read as a rehabilitated r*.

---

# Why r* cannot be identified here

This section is first because it is prior to the evidence. The empirical work further down
confirms it; none of it was needed to know it.

## Three latent states, two observation equations

Resolution A, canonical HLW, has three latents and two equations connecting anything to data:

| latent | what pins it |
|---|---|
| potential output, y* | the IS curve, with a loading of **1** |
| trend growth, g | nothing directly; only through y*'s drift |
| **z** | **nothing at all** |

There are only four observation equations in the whole package (`grep observed=`): the IS
curve, the Phillips curve, the indexed-bond equation used by B alone, and the soft
trend-growth anchor used by C through H. Resolution A has the first two. **The system is
under-determined by construction, and z is exactly the state carrying the r\* information the
model claims to produce.**

## The one piece of structure is what produces the answer

Itemise what HLW actually asserts:

- two random walks (y*, g), which are smoothness assumptions, not economics;
- an IS curve that is a reduced-form AR(2) with a rate term;
- a Phillips curve that is a reduced-form regression;
- **one economic statement: `r* = g + z`**, which has no free coefficient left in the
  reference implementation: LW2001 wrote `r* = c x g + z` and estimated c at 0.97, and HLW2017
  imposes c = 1 because the relationship is "not well identified in the data".

That statement is the whole structural content, and with z unanchored the model has no way to
fill it in. On the shipped settings it fills it in with nothing: z = **+0.02**,
`corr(r*, g)` = **0.998**, so the reported r* of 2.25% real and 5.03% nominal is Australian
trend growth of 2.23% wearing a different label.

**But `r* = g` is a property of the smoothing, not of the model, and it is important to state
this correctly.** Loosen the σ_z prior and z does move: at a prior scale of 0.30 it spans 1.49,
`corr(r*, g)` falls to 0.745, and r* latest rises to 3.10. So HLW does not *say* r* equals
trend growth. It says r* equals trend growth *at this σ_z*, and something else at another.

The question that settles it is whether the smoothing is suppressing a signal. It is not:

| σ_z prior scale | 0.025 | 0.05 | 0.10 | 0.30 |
|---|---|---|---|---|
| z span | 0.075 | 0.074 | 0.269 | **1.491** |
| `sigma_IS` | 0.5664 | 0.5627 | 0.5593 | **0.5559** |
| `sigma_pi` | 0.5644 | 0.5620 | 0.5565 | **0.5649** |

**Letting z wander twenty times further improves the IS fit by 1.9% and the Phillips fit not
at all.** The likelihood is flat along the entire range. (On the old 1986Q3 sample: 50x the
wander for 2.6% and 1.1%. Same verdict.)

So the correct formulation is not "r* = g". It is:

> **At tight smoothing r\* is trend growth; at loose smoothing it is an unconstrained
> wanderer; and the data barely distinguishes between them, because letting z move 50x
> further buys a 2.6% improvement in fit. Choosing σ_z is not choosing how fast r\* moves. It
> is choosing which answer to report from a set the likelihood is indifferent between.**

That is what an unanchored latent looks like from the inside, and the counting argument
guarantees the flat ridge in advance. Every number in the table above is a consequence of it
rather than a discovery.

One degenerate story worth ruling out, because it is the obvious one and it is wrong: loose z
does NOT chase the real cash rate to kill the rate gap. `corr(r*, real cash)` FALLS from 0.848
to 0.586 as the prior loosens, and the rate gap's sd is 2.09 either way. z is not explaining
the rate gap away; it simply wanders, and the likelihood does not object.

## The cross-resolution diagnostics say the same thing

Reread through the count, the sampling statistics stop looking like sampler trivia:

| resolution | z with a free variance? | `r_star` ESS |
|---|---|---|
| A, B, D | **yes** | 243, 366, 548 |
| E, F (σ_z pinned) | no | 12,420, 16,565 |
| C, G, H (no z at all) | no | 6,133, 6,169, 6,246 |

Every resolution carrying an unanchored latent whose variance is also free samples two orders
of magnitude worse. Pin the variance or delete the state and the model behaves beautifully,
and hands back whatever was assumed. B is the single case where a third observation equation
is added, the indexed yield, and z does come alive: by becoming the bond yield, which is what
adding an observable that loads on r* one-for-one will always do.

## The two honest routes, and HLW takes neither

**A latent is identified when something you observe loads on it near one-for-one, or when
theory ties it to things that are.** There are therefore two ways to get r*:

- **Real structure**: a DSGE, where r* falls out of an Euler equation tied to preferences and
  technology, carrying cross-equation restrictions you can test. See `src/models/dsge`.
- **A direct observable**: [`rstar_bonds`](../rstar_bonds/MODEL_NOTES.md), which has no more
  structure than HLW but gives r* a near-unit loading on the indexed real yield.

HLW has an asserted identity and an elasticity. That elasticity is the quantitative version of
the same point: **y\* enters the IS curve with a loading of 1 and r\* with a loading of
0.044.** y* is a level in the same units as the observable, so GDP measures it directly; r*
reaches the observable only through a behavioural elasticity, and quarterly elasticities of
output to interest rates are small.

**HLW is a filtering device wearing structural clothes.** It is genuinely good at trend/cycle
decomposition and constitutionally incapable of identifying r*, because it contains no
statement linking r* to anything observable. That is not a defect of this implementation. It
is what the model is.

## The original papers do not estimate the variances either

The non-identification above is not a finding about Australian data or about this
implementation. LW2001 and HLW2017 reach it, patch it, and say so in the text. Quotations are
from LW2001 (FEDS 2001-56) and HLW2017 (FRBSF WP 2016-11); published versions may word things
differently.

**Maximum likelihood returns zero.** LW2001 on the innovation standard deviations of trend
growth and z: "likely to be biased towards zero owing to the so-called 'pile-up problem'
discussed by Stock (1994). Indeed, the maximum likelihood estimates of both of these
parameters were zero." HLW2017 repeats the reasoning.

**So the ratios are imposed rather than estimated.** Both papers use Stock-Watson (1998)
median-unbiased estimation to obtain `lambda_g` = sigma_g / sigma_ystar and
`lambda_z` = a_r x sigma_z / sigma_ygap, then "impose these ratios when estimating the
remaining model parameters". Three sequential steps:

1. Kalman filter a cut-down model with the real rate gap deleted from the IS equation and
   trend growth held constant, giving a preliminary potential output. Andrews-Ploberger
   exponential Wald test for a break at unknown date on its first difference, mapped to
   `lambda_g` through Stock-Watson's tables.
2. Impose `lambda_g`. Include the rate gap but hold z constant. Exponential Wald test for an
   intercept shift in the IS equation, mapped to `lambda_z`.
3. Impose both and estimate the remaining parameters by maximum likelihood.

Each stage takes its input from a model that assumes away what the next stage measures:
`lambda_g` from a specification in which r* has no effect on output, `lambda_z` from one in
which z does not move.

**The likelihood cannot choose between the resulting answers.** LW2001's Table 1 reports
`lambda_g` 0.039 with a 90% interval of [0.000, 0.103], which contains zero, and `lambda_z`
0.071 with [0.015, 0.117]. Re-estimating with `lambda_z` imposed at each end of its interval:
"based on the log likelihood values, the data are not able to discern between the alternative
specifications." Where z is non-stationary they also decline to report t-statistics for the
coefficient on trend growth, because "inference based on the standard errors is invalid".

**Two further quantities are imposed.** HLW2017 fixes the coefficient on trend growth at one:
"Because this relationship is not well identified in the data, we chose to impose a
coefficient of unity." LW2001 estimated it and got 0.97. HLW2017 also constrains the signs,
"we impose the constraints that the slope a_r of the IS equation is negative and the slope b_y
of the Phillips curve is positive", calling them "minimal priors" that "facilitate the
convergence of the numerical optimization". The sign of the IS slope is therefore not a result
in the reference implementation.

**Nor is it a result here, and the two sign constraints do opposite amounts of work.** Both
are hard truncations (`set_model_coefficients` routes any prior carrying a bound to a
`TruncatedNormal`), so the inequality matches HLW. What differs is that ours also carry a
location.

| | prior | prior median | posterior median |
|---|---|---|---|
| `a_r` | TruncNormal(−0.15, 0.08, ≤ 0) | −0.153 | −0.042 [−0.098, −0.008] |
| `b_y` | TruncNormal(0.10, 0.05, ≥ 0.02) | 0.103 | 0.277 [0.221, 0.336] |

`b_y` is inert: it sits nearly three times its prior median and more than ten times its floor,
with zero draws below 0.025. Drop the constraint and nothing moves. `a_r` binds: 7.4% of draws
lie within 0.01 of the zero bound, and the posterior median sits in the top 6.8% of its own
prior's mass. The likelihood is pulling `a_r` toward zero and the truncation is what stops it
crossing. Since `a_r` is the loading of r* on the only observable that sees it, the one
constraint that carries the answer is the one that binds, and it was imposed rather than
found. The prior is also informative in a direction the data reject: centring at −0.15 with an
sd of 0.08 asserts a magnitude roughly four times what comes back, so part of what remains of
the negative slope is prior. `--canonical` replaces both with sign-only forms.

With the coefficient on trend growth fixed at one and `lambda_z` small, r* is trend growth
plus a driftless random walk of imposed amplitude. That is the specification behaving as
written, and it is what `corr(r*, g)` = 0.998 records.

**And the imprecision survives good identification.** HLW2017's US estimates are the
favourable case: "The slope coefficients a_r and b_y for the U.S. are reasonably large and
precisely estimated, suggesting that both the output gap and the real rate gap are well
identified." Even there, "even with hindsight, the natural rate of interest is estimated
imprecisely, with a sample average standard error of 1.1 percentage points", and the one-sided
estimates that correspond to real-time use are worse again. A wide band on r* is therefore not
a symptom of Australian data, of this sample, or of estimating the model by MCMC rather than
maximum likelihood. It is what the model returns where its own identifying equation works.

**What this implementation does instead.** There is no median-unbiased step here. `sigma_z` is
free under a HalfNormal prior and `sigma_ystar` is imposed at 0.078, which is HLW's device
applied to one variance. `sigma_z_sweep.py` and `sigma_z_prior_sweep.py` are the Bayesian
counterpart of the median-unbiased step: rather than taking a ratio from a break test and
conditioning on it, they vary the prior and report how much of the answer follows it. Most of
it does. LW2001's own sensitivity check reports the same fact by a different route.

### The full inventory, against Resolution A

Which of the papers' restrictions Resolution A carries. The exact form of HLW's inflation lag
polynomial and import-price term is **ASSUMPTION, carried from memory and not checked against
the paper**; everything in the first column is verified against this repo's own code.

| Restriction | Resolution A |
|---|---|
| c = 1 on trend growth, r\* = g + z | **In** (`equations/z_star.py`) |
| a_r < 0 | **In**, as a truncation. HLW use a strict −0.0025 bound (UNVERIFIED) |
| b_y > 0 | **In**, floor at 0.02 |
| Output-gap stationarity, a_y1 + a_y2 < 1 | **In**, but implied by the priors rather than enforced: a_y1 is truncated to (0, 1) and a_y2 to ≤ 0 |
| Vertical long-run Phillips curve | **In, and harder than HLW's.** The coefficient on `pi_exp` is literally 1, not a sum-to-one restriction on estimated lag weights |
| y\*\_t = y\*\_{t−1} + g\_{t−1} + ε | **In** (annualised, so g/4) |
| `lambda_z` = a_r·σ_z/σ_ygap | **Out entirely.** `sigma_z` is free under HalfNormal(0.10) |
| `lambda_g` = σ_g/σ_ystar | **Off by default,** and HLW's form of it is untested: the sweep that dismissed it held σ_y\* imposed, so it constrained both variances rather than their ratio |
| Stock-Watson median-unbiased estimation | **Out entirely.** No exponential Wald test, no lookup tables |
| σ_y\* estimated in stage 3 | **Out.** Imposed at 0.078, which is a substitute for λ_g rather than an HLW device |
| IS rate gap averaged over t−1, t−2 | **Out by default.** A single t−6 lag; `--rate-lag 0` restores HLW's shape |
| Phillips lag polynomial on past inflation | **Out.** Annual trimmed mean on a model expectations series |
| Import-price and oil relative-price terms | **Out** |
| π^e as a 4-quarter MA of past inflation | **Out.** The same model expectations series serves the Phillips anchor and the IS real rate, where HLW use different objects in those two places |
| Lockdown exclusion | **Added here.** Not an HLW device |
| Pre-sample OLS state initialisation | **Out.** Priors instead, and with no anchor on g the `N(3.5, 1.5)` init carries real weight at the sample start |

`--canonical` moves the four rows it can reach without restructuring: HLW's averaged rate gap,
sign-only priors, `sigma_ystar` estimated, and the exclusion off. The next subsection is what
happened when it was run.

### The canonical bundle does not sample, and that is the finding

`--canonical` on Resolution A returns R-hat 1.290, ESS 13, MCSE/sd 0.296, 2,939 divergences in
50,000 draws, tree depth at maximum in 93.5% of transitions and BFMI 0.01. **No number from
that run is quotable.** What is readable is the direction of travel before the sampler gave
up: `sigma_ystar` reached 0.845 against a prior median of 0.081, and potential's quarterly sd
reached 0.992 against GDP's own 0.972. That is the unconstrained-trend pile-up described in
`equations/potential.py`, arrived at from the other side.

The reason matters more than the failure. **Canonical HLW never has `sigma_ystar` free and
`sigma_g` free at the same time: `lambda_g` is what ties them.** Freeing the first without
imposing the ratio builds a specification neither the papers nor this repo estimates, and the
funnel between the two variances is the Bayesian face of exactly what LW2001 report as a
corner solution at zero. They get a pile-up, this gets a funnel, and in both cases the data
cannot pin both variances.

So `lambda_g` is not a refinement of canonical HLW that can be added later. It is what makes
canonical HLW estimable, and a canonical run has to carry it. Two routes remain: hoist the
scalar priors out of the equation functions so `sigma_g = 4·lambda_g·sigma_ystar` can be
formed with `sigma_ystar` free, which is the genuinely canonical object; or accept
`--canonical --sigma-ystar 0.078 --lambda-g 0.039`, which ties the two by LW2001's ratio but
fixes rather than estimates σ_y\*, and which would at least isolate what the rest of the
bundle does. `lambda_z` needs the same hoist and more, because it needs `sigma_IS` and `a_r`,
which are built in an observation equation that runs after z.

## Estimated their way, by Kalman filter: the same conclusion, harder

Everything above is MCMC. The model was also estimated the way the papers estimate it, with
the states integrated out by a Kalman filter and the parameters by constrained maximum
likelihood. That package is separate and disposable; what follows is the part that is about
CANONICAL HLW rather than about either implementation, and so belongs here.

**The state space is validated by simulation recovery**, so none of this is an implementation
fault. And the validation carries its own finding: on 4,000 quarters generated from the model
itself, the slopes recover to a hundredth while `sigma_ystar` comes back 0.24 against a true
0.30 and `sigma_z` 0.063 against 0.10, both biased DOWN. **LW2001's pile-up reproduces on
synthetic data thirty times the length of the real sample.** It is not a fact about Australia.

**`lambda_g` = 0.0447** from HLW's own stage 1 break test, against LW2001's published 0.039
and 0.0497 from a Bayesian analogue of the same idea. Three routes agree.

**The fitted model is degenerate, and it is the global optimum.** Four starting points,
including one seeded at the MCMC posterior, reach the same answer: `a_r` on its lower bound at
−0.0025, `sigma_ystar` = 0.963 giving potential a quarterly sd of 0.975 against GDP's 0.972,
`a_y1` = 1.49 and `b_y` = 1.45. **The MCMC priors were not regularising this model, they were
the only thing keeping it out of that corner.**

### Imposing sigma_ystar = 0.078 costs 139.7 log-likelihood points

A profile over `sigma_ystar` with everything else re-optimised:

| `sigma_ystar` | 0.078 | 0.25 | 0.40 | 0.60 | 0.80 | 0.963 |
|---|---|---|---|---|---|---|
| log-likelihood | −343.7 | −318.9 | −299.7 | −244.7 | −209.1 | −204.0 |
| cost vs optimum | **139.7** | 115.0 | 95.7 | 40.7 | 5.1 | 0 |

A likelihood ratio statistic near 279 on one restriction. The prior-versus-posterior charts
could only show the two were disjoint. **This says what the restriction costs, and the data
reject it overwhelmingly.** `a_r` sits on its bound at every point of the profile, so the dead
rate channel and the pile-up are independent failures.

### lambda_z divides by a_r, and a_r is on its bound

HLW set `sigma_z = lambda_z x sigma_IS / |a_r|`, which presumes an identified IS slope. Theirs
is "reasonably large and precisely estimated"; ours is 0.0025, so the expression returns
`sigma_z` = 3.90 and an r\* running from −7.3 to +22.0. **It also fits worse**: −207.17 against
−203.98 for a small imposed `sigma_z`. Their parameterisation is not merely unusable here, it
is beaten on their own criterion.

### z's shape is identified and its scale is not

Imposing `sigma_z` directly, a twenty-fold change in z's amplitude costs **0.06** log-likelihood
points while the path keeps a correlation above 0.996. The curve is the right shape and
arbitrarily stretched on the y axis. `a_r` stays pinned throughout, so this is not a trade-off
that conserves `a_r·z`: it is that `a_r` is small enough that z barely enters the likelihood.

### Whatever you observe z with, r\* becomes it

The AOFM 5y5y risk-neutral forward was added as a third observable loading on r\*, then
removed. With the bias free, pinned at zero, and pinned at −0.109, the log-likelihood is
identical to three decimals and **corr(r\*, the market forward) is 1.000 in all three**. At a
pinned bias of zero, r\* reproduces the market series to two decimals. The forward's
observation sd came back at 0.063, estimated rather than imposed: the likelihood CHOSE to
track it almost exactly.

This is stronger than "z is unidentified". **With `a_r` near zero the model has no view about
r\* that it can defend against any observable, so whatever equation is attached to z, r\*
becomes that equation's right-hand side.** It is the `rstar_tvpvar` failure in another costume,
where r\* came back as the real cash rate at corr 0.95.

### What survives

**g agrees across estimators**: 3.75 to 2.21 by Kalman and maximum likelihood, 3.79 to 2.20 by
MCMC. The trend and cycle decomposition is robust to how the model is estimated. It is
specifically r\* that fails, and it fails the same way whichever estimator is pointed at it.

## One of three routes, all flawed

This repo contains three separate attempts at Australian r\*, and the useful thing is that
they fail differently. Read them together rather than picking one.

| package | identified from | what it actually measures | how it fails |
|---|---|---|---|
| **`rstar_hlw`** (this one) | trend growth + the IS curve | the textbook definition: the rate at which output sits at potential | the IS curve does not identify anything on AU data, so each specification returns its own prior |
| [`rstar_bonds`](../rstar_bonds/MODEL_NOTES.md) | asset prices | what investors price | the *level* is not identified; it rests on the stationarity prior and an asserted premium |
| [`rstar_rba`](../rstar_rba/MODEL_NOTES.md) | the RBA's response to inflation | not r\* but what the Bank's conduct *reveals* about it, conflated with every other systematic motive | a long enough departure from the rule is absorbed into neutral, so it cannot audit the Bank over a decade |

**This is the only one of the three that targets what the theory defines**, and it is the
one that cannot be estimated. The other two measure *beliefs* about r\*, held by different
people: bond investors in one case, the RBA in the other. Neither is the saving-investment
equilibrium the concept names.

That reframes what follows. The eight resolutions below are not a failed search for a
number. They are the reason the other two packages exist, and the evidence that the honest
object here is a belief rather than the thing itself.

---

# What the model does produce

Not r*: see the section above. What follows is the trend/cycle decomposition, which is the
part worth using and the part that was repaired on 2026-09-12.

Current as at **2026-09-12**, Resolution A (canonical HLW), **1993Q1 to 2026Q2, 134
quarters**, single rate lag t−6.

**The sample start changed from 1986Q3 to 1993Q1 on 2026-09-12, and it matters more than any
other setting here.** See "The sample start is the biggest single choice" below.

**Imposed settings**, both reported in the run log and on the charts:

1. **`sigma_ystar` = 0.078**, `ystar`'s value, imposed rather than given a prior. This is
   HLW's device applied to one variance. A tightened HalfNormal(0.12) was tried first and
   the posterior sat at 0.862, seven standard deviations into its own tail: a prior cannot
   win an argument the likelihood insists on.
2. **2020Q2 to 2021Q3 dropped from the IS and Phillips likelihoods** (`equations/exclusion.py`),
   matching `ystar`. The states still run through the window under their priors; what goes is
   the claim that potential plus a cyclical gap should account for output and inflation while
   the economy was shut. The Phillips curve is masked too, deliberately: the lockdown gap is
   a shuttered economy rather than deficient demand, so pricing it into inflation would drag
   potential down by a second route.

### The window does not cover the IS curve's own lags

`exclusion.py` drops a row whose **own date** is in the window. It does not mask the lagged
regressors, which are still built from the full series. The Phillips curve is safe, because it
passes `first=0` and so masks on the output gap's date rather than the inflation date a quarter
later. The IS curve is not: at `t` it uses `output_gap[t−1]`, `output_gap[t−2]` and
`r_gap[t−rate_lag]`, so with the window ending 2021Q3, 2021Q4 and 2022Q1 still take lockdown
output gaps, and at the default rate lag of 6, every quarter from 2021Q4 to 2023Q1 takes a
lockdown-window rate gap. With `a_y1+a_y2` ≈ 0.94 (half-life ~12 quarters) those rows do not
wash out quickly either.

**Tested 2026-09-14 by widening the window to 2022Q1**, which covers the AR(2) gap lags but
deliberately not the rate lag: the rate lags carry ELB-era policy rates, which are real data,
not shutdown-distorted GDP. Same 134-quarter sample, 8 quarters dropped instead of 6.

| | to 2021Q3 | to 2022Q1 |
|---|---|---|
| `a_r` | −0.044 | **−0.021** |
| `sigma_IS` | 0.559 | 0.481 |
| `a_r`/`sigma_IS` | 0.079 | **0.044** |
| long-run slope | −0.84 | **−0.42** |
| persistence `a_y1+a_y2` | 0.947 | 0.950 |
| `b_y` / `sigma_pi` | 0.273 / 0.556 | 0.272 / 0.553 |
| r* range / latest | [1.29, 3.64]% / 2.25% | [0.97, 3.63]% / **2.08%** |
| g latest | 2.23% | 2.24% |
| output gap sd | 2.04 | 2.05 |
| gap, 2022Q4 → 2026Q2 | +4.43 → +2.56 | +4.46 → +2.66 |
| divergences | 2 | **0** |
| overall R-hat / min ESS | 1.200 / 8 | **1.040 / 72** |
| `r_star` R-hat / min ESS | 1.010 / 402 | 1.010 / 395 |
| BFMI | 0.02 | 0.03 |

**It buys sampling, and it costs the rate channel.** Divergences go to zero, `r_star` mixes
properly (ESS 39 → 395), and overall R-hat falls from 1.20 to 1.04. Five of six diagnostics
still fail, `sigma_z` is still the worst parameter on every measure, and **BFMI is unchanged at
0.03**: the posterior got smoother, not identified. Meanwhile `a_r` halves, the per-quarter
signal falls from 0.079 to 0.044, and the long-run slope goes from −0.84 to −0.42 with its 90%
band reaching −0.039, close enough to zero to be doing no work. The Phillips curve, the gap and
potential growth are all untouched.

Read plainly: the 2021Q4–2022Q1 quarters were carrying much of what identified the rate channel,
so protecting potential from them also removes the identification. That is the trade, and it is
the same shape as the sample-start trade below.

**The code default is still 2020Q2–2021Q3.** The wider window is a tested alternative, not the
shipped setting: `./run-rstar-hlw.sh --exclude-window 2020Q2:2022Q1`.

`lambda_g`, the σ_g/σ_ystar ratio, is implemented (`--lambda-g`) and **off by default**.

| | 1993Q1 (current) | 1986Q3 (previous) |
|---|---|---|
| r* range / latest | [1.29, 3.64]% / **2.25%** | [1.29, 3.94]% / 1.73% |
| g range / latest | [1.23, 3.82]% / 2.23% | [1.51, 4.17]% / 1.88% |
| output gap sd | **2.04** | 2.61 |
| gap, 2022Q4 → 2026Q2 | **+4.43 → +2.56** | +4.95 → +4.39 |
| `a_r` | −0.044 | −0.112 |
| `sigma_IS` | 0.559 | 0.577 |
| `b_y` / `sigma_pi` | **0.273 / 0.556** | 0.169 / 0.706 |
| `a_r`/`sigma_IS` | **0.079** | 0.193 |
| long-run slope | **−0.84** | −2.11 |
| persistence `a_y1+a_y2` | 0.947 | 0.953 |
| potential, quarterly change sd | 0.189 (GDP 0.972) | 0.195 (GDP 0.950) |
| divergences | 2 | 5 |
| `r_star` R-hat / min ESS | **1.010 / 402** | 1.010 / 1,203 |

**This is a trade, not a free win.** The gap, the Phillips curve and the long-run IS slope all
improve. The IS curve's per-quarter identification and r*'s sampling both get materially
worse. Both columns are shown because neither dominates.

## 1. The trend/cycle decomposition is repaired

Potential is no longer more volatile than output. 2020Q2 reads as a −6.0 output gap instead
of a fall in potential. The gap by era is defensible almost everywhere:

| 1988-89 | 1991-93 | 2005-19 | 2022-26 |
|---|---|---|---|
| +2.3 | −4.9 | +0.3 | **+4.5** |

The first three are about what Okun implies at the unemployment rates of the day. **The
fourth is not, and it is the outstanding defect.** The gap is near-unit-root (`a_y1` ≈ 0.97,
half-life 14 quarters) and the Phillips curve is too weak to force mean reversion, so the gap
barely moved through the 2022-26 disinflation even as unemployment rose 0.85pp. See "The
recent gap is wrong" below: it is a persistence failure, not a level bias, and no setting of
λ_g touches it.

### How big should the gap be

HLW's gap is far bigger than this repo's other two: on the common 1993Q1 sample, sd **2.04**
against `ystar`'s deviation-from-potential of 1.08 and `ystar_ustar`'s published 0.42. They
correlate 0.50 and 0.73 with it, so the three agree on timing and disagree violently on scale.
Two separate things cause that, and only the second one matters.

**Part of it is comparing different objects.** The ystar family DEFINES the gap as
`c x (pi - 2.5)`, so what it publishes is only the part of output's deviation from potential
that inflation explains. `ystar` also exposes the deviation itself
(`actual_output_gap_posterior`), and that has **sd 1.08, not 0.188**. Compare like with like.

**The rest is not definitional, and this is the important half.** All three potentials are
equally smooth (quarterly change sd 0.179, 0.180, 0.195). What differs is **where potential
drifts**: `ystar`'s 4-quarter potential growth runs 3.96, 4.06, 2.43, 2.05 across 1990-94,
1995-99, 2015-19 and 2020-24 against HLW's 3.49, 3.33, 1.96, 1.57. Sustained differences of 0.3
to 0.7pp a year compound, and **the two potential LEVELS drift 9.1 log points apart over the
sample**, about twice the amplitude of any Australian business cycle. That is the whole
explanation for the different gap sizes.

**The drift is the trilemma seen from the other side.** In `ystar` the gap is nailed to
inflation, so nothing persistent can live in it and every multi-year deviation is absorbed by
potential, whose actual-gap era means are essentially zero in every decade. In HLW the gap is a
near-unit-root AR(2), so persistence lives there and potential keeps its own path. Mirror
images, both assumptions rather than findings.

**A Phillips curve relocates the decision rather than settling it.** `ystar_ustar` has one, and
a strong one: `gamma_pi` = −1.152, 90% [−1.395, −0.937]. But it is attached to the UNEMPLOYMENT
gap, and **77% of the variance in Australian unemployment has already been assigned to u\***
(sd 1.48 of unemployment's 1.68), which runs from 10.43 in 1991-93 to 4.74 now. The persistent
part is declared structural before inflation gets a vote, and the imposed `sigma_ustar` governs
that, not the Phillips curve.

So the same choice appears three times, and each time one imposed variance with no external
anchor makes it:

| model | the question | decided by |
|---|---|---|
| `rstar_hlw` | potential or gap? | `sigma_ystar`, imposed |
| `ystar` | same, and the gap is nailed to inflation, so potential takes it | `ratio_ystar`, imposed |
| `ystar_ustar` | u* or unemployment gap? | `sigma_ustar`, imposed |

Adding equations moves the decision; it does not remove it. This is also why `ystar_ustar`
reads u* near 10.4 for the early 1990s, the same number `long_run_ustar` gets under a loose
rule: not a defect of the joint model, but the NAIRU concept dissolving in a re-anchoring.

**Checked against unemployment, HLW's amplitude is the defensible one.** In LEVELS, with an
external u* and textbook Okun (β = 2 on the output side):

| | u | stated u* | Okun-implied gap | HLW | ystar | ystar_ustar |
|---|---|---|---|---|---|---|
| 1993-95 | 9.6% | 8.0% | −3.3 | **−2.91** | −0.03 | −0.02 |
| 2008-09 | 4.9% | 5.2% | +0.7 | **+1.16** | +0.28 | +0.32 |
| 2026Q2 | 4.4% | 5.5% | +2.3 | **+2.56** | +0.17 | +0.30 |

HLW is close at all three, spanning a deep recession, a boom peak and the present; the other
two are out by a factor of 8 to 100. It falls out of `ystar_ustar`'s own coefficient too:
regressing HLW's gap on `ystar_ustar`'s unemployment gap gives **−2.07**, textbook Okun, while
its own gap on its own unemployment gap gives **−0.60**. That is why its `beta_okun` is 1.27
against a textbook 0.5, and `ustar`'s 2.14: those coefficients absorb a gap that is too small.

In quarterly CHANGES it goes the other way. Regressing Δgap on Δu, lockdowns excluded, HLW's
correlation is **−0.16** against `ystar_ustar`'s **−0.90** (partly circular, since it fits an
Okun equation, where HLW has no labour market at all).

**So HLW has roughly the right amplitude and poor high-frequency behaviour; the ystar family
has excellent timing at a fraction of the right amplitude.** Neither is the output gap. Do not
treat the larger sd as the error, and do not use `ystar_ustar`'s 0.42 as a benchmark.

### The sample start is the biggest single choice

**This was found by a direct test that the old default failed.** Over 2022Q4 to 2026Q2
trimmed mean inflation fell from 6.8% to 3.6% and unemployment rose from 3.50% to 4.35%. The
economy cooled substantially. On the 1986Q3 sample HLW's gap went from +4.95 to +4.39, a move
of 0.56 where Okun implies 1.71. **On the 1993Q1 sample it goes +4.43 to +2.56, a move of
1.87.** Practically the whole defect was the pre-1993 quarters.

| | 1986Q3 start | 1993Q1 start | external benchmark |
|---|---|---|---|
| gap, 2022Q4 → 2026Q2 | −0.56 | **−1.87** | Okun: −1.71 |
| gap at 2026Q2 | +4.39 | **+2.56** | Okun: +0.8 to +2.3 |
| gap mean, 2005-19 | +0.34 | −0.04 | ~0 |
| `b_y` | 0.169 | **0.273** | |
| `sigma_pi` | 0.706 | **0.556** | |

The Phillips slope nearly doubling and its residual falling is the clue to the mechanism:
pre-1993 Australia had no inflation target, and asking one `b_y` to span that regime change
flattens it, which in turn lets the gap drift. The regime argument for 1993Q1 was already in
these notes; what is new is that the data now insists on it.

On the current sample the gap matches Okun at three separate points, which is external
evidence the model never sees:

| | unemployment | stated u* | Okun-implied | HLW |
|---|---|---|---|---|
| 1993-95 | 9.6% | 8.0% | −3.3 | **−2.91** |
| 2008-09 | 4.9% | 5.2% | +0.7 | **+1.16** |
| 2026Q2 | 4.4% | 5.5% | +2.3 | **+2.56** |

**What it costs.** `a_r` falls from −0.112 to −0.044 and the per-quarter signal
`a_r`/`sigma_IS` from 0.193 back to **0.079**, near the 0.064 of the original broken model.
r*'s sampling degrades with it (R-hat 1.090, min ESS 39). Against that, the long-run slope
becomes −0.84 rather than the frankly incredible −2.11, so the IS curve is better behaved
where it can be checked and worse identified where it cannot.

**Why `a_r` weakens is itself interesting, and the obvious explanation is probably not the
right one.** The mechanical story is that the 1980s cash rate reached 18.16% against a 0.10
to 7.50 range after 1993, so the early swings were carrying the identification.

HYPOTHESIS, NOT YET TESTED HERE: the larger effect is **endogeneity from successful policy**.
The post-1993 sample is precisely the period in which the RBA is actively setting the cash
rate to lean against the output gap. The rate gap is then not an exogenous regressor: it moves
because the gap moves, which biases `a_r` toward zero. On that reading a small `a_r` in the
inflation-targeting era is evidence of effective stabilisation rather than of weak
transmission, and the two are genuinely hard to tell apart in a single equation.

This is not idle: it is what `is_curve` already found in reduced form, where the strongest
relationship is contemporaneous and POSITIVE, which is the reaction function rather than
transmission, and it is why the rate lag here is t−6 at all, a regressor further from t
carrying less of the RBA's response. **The test is a lag sweep on the current sample:** if
endogeneity is the binding problem, `|a_r|` should strengthen as the lag lengthens. If it is
just lost rate variation, the lag should not matter much.

**Persistence is unchanged**: `a_y1+a_y2` = 0.947, a half-life of 12.8 quarters. The gap still
decays slowly; it simply no longer starts from a level that slow decay cannot clear. That
remains the outstanding structural issue, and an Okun equation is the way to pin the gap's
scale rather than leave it to the sample.

### Tested and dead ends

Two hypotheses about the gap were tested and neither survived. Recorded in one line each
because a negative result is worth knowing and not worth a section.

**r\* was not holding the gap open.** `given_rstar_test.py` hands the model `rstar_bonds`' real
r* as data, turning the rate gap at 2026Q2 from −0.68 to +0.70 and making it positive in all 14
quarters since 2022, where the model's own r* never reaches neutral. It moved the output gap by
**0.18pp** and the rate term's sd not at all (0.057 to 0.058). `a_r` is too small for r*'s level
to matter: the loading problem again, from the other end. Do not expect a better r* to fix the
gap, or a better gap to fix r*.

**`a_r` is not obviously endogenous to the RBA's response.** If short lags were contaminated by
the Bank leaning against the gap, `|a_r|` should strengthen with the lag. Across a sweep of
avg(1,2), 1, 2, 4, 6, 8, 10 it is **largest at t−1** (−0.077) and smallest at the t−6 default
(−0.044), which is the wrong shape. Weak evidence either way, since every interval nearly
touches zero and the differences sit inside them, but no lag rescues `a_r`. Persistence is
also flat at 0.92 to 0.94 across the whole grid, so the AR(2) is not fitting anything about the
rate. What the sweep does establish is that **the long-run slope is a lag choice**, −0.73 at
t−6 against −1.29 at t−1, so quote it as a range of about −0.7 to −1.3 rather than a number.
Default left at t−6: picking the lag with the largest coefficient would be selecting on the
outcome, and t−6 keeps timing comparable with `is_curve` and `rstar_invert`.

On the 1986Q3 sample the repair also tripled the per-quarter IS signal, 0.064 to 0.193.
**That gain does not survive the move to 1993Q1**, where it is 0.079: the 1980s rate cycle was
carrying it. What the repair did buy unconditionally is a decomposition the coefficients can
be measured against at all, which the old one was not.

## 2. The variance has to go somewhere

The sharpest finding here, and it does not involve r* at all.

| what is imposed | where the cycle ends up |
|---|---|
| nothing | **potential** absorbs it (σ_ystar posterior 1.112, the original bug) |
| σ_ystar only | **g** absorbs it (σ_g posterior 0.105 against a 0.04 prior scale) |
| σ_ystar and λ_g | **the gap** absorbs it, and the Phillips curve flattens rather than price it |

Australian GDP's low-frequency variance goes into whichever state is left free. **In this
model** the only equation that could adjudicate is the Phillips curve, and it is not strong
enough to: `b_y` × gap against `sigma_pi` is a signal-to-noise of about 0.6. Do not generalise
that to "a Phillips curve cannot adjudicate": `ystar_ustar` has a strong one and still does not
settle it, for a different reason. See "How big should the gap be" below.

**Why the pile-up problem does not bite the same way here.** LW and HLW impose both ratios
because maximum likelihood drives the two variances to zero (see "The original papers do not
estimate the variances either"). This build leaves `lambda_g` free and still returns a sensible
`sigma_g` of 0.105, because `sigma_ystar` is imposed: fixing one member of the pair gives the
other a scale to be measured against. `sigma_z` has no such partner and does not escape. It is
the worst-sampled parameter in the model on every measure, which is the pile-up appearing as
posterior geometry rather than as a point estimate of zero.

**λ_g at HLW's own US value is rejected by Australian data** (`lambda_g_sweep.py`). That value
is not an estimate in HLW either: it comes from the break test in the median-unbiased procedure
above, and HLW's own 90% interval for it runs 0.004 to 0.132. Re-run on the 1993Q1 default,
with the original 1986Q3 run in brackets:

| λ_g | 0.053 (HLW US) | 0.15 | 0.34 | free |
|---|---|---|---|---|
| gap mean, 2005-19 | **+2.41** (+4.35) | +0.25 (+0.78) | +0.00 (+0.31) | −0.04 (+0.34) |
| `b_y` | **0.110** (0.054) | 0.256 (0.156) | 0.274 (0.169) | 0.273 (0.169) |
| `sigma_pi` | **0.764** | 0.642 | 0.576 | 0.556 |
| g, first → last | 2.96 → 2.39 | 3.49 → 1.89 | 3.75 → 2.09 | 3.82 → 2.24 |
| r* latest | **3.25** (2.75) | 2.02 (1.70) | 2.19 (1.71) | 2.25 (1.73) |

**The conclusion survives the sample change unaltered.** 0.053 is still the outlier on every
line: the only setting holding g nearly flat, the only one reading 2005-19 as well above
capacity, the only one where the Phillips slope collapses. Australian trend growth genuinely
fell, and forcing g constant pushes that decline into the cycle, which the Phillips curve then
flattens rather than price. 0.34 still matches the free run closely enough to buy nothing.
Binding values are wrong and non-binding ones inert, hence the default: off.

Note that **λ_g moves r\* by a full percentage point** (3.25 against 2.25). A variance ratio
nobody can measure decides the headline number, before any of the r* priors Resolutions C
through H argue over.

**THE SWEEP DOES NOT TEST HLW'S DEVICE, AND THE CONCLUSION ABOVE IS NARROWER THAN IT READS.**
`lambda_g_sweep.py` calls `build_model` with `lambda_g` and nothing else, so `sigma_ystar`
stays at its imposed default of 0.078 in every column. Each run therefore has **both**
variances nailed down, σ_y\* at 0.078 and σ_g at 4·λ_g·0.078. HLW impose the ratio and
estimate the level. Those are different specifications, and the doubly-constrained one is the
harder of the two on the model: it asks the data to accept a trend whose level and whose
drift are both dictated.

So "λ_g at HLW's own US value is rejected by Australian data" is established for a
specification HLW do not estimate. It may well hold for the one they do. **It has not been
tested here**, and the reason is mechanical rather than considered: `_sigma_g_from_lambda`
raised whenever `sigma_ystar` was free, because the state equations built trend growth before
potential and a ratio to a parameter that does not yet exist cannot be formed. Hoisting the
scalar priors ahead of the state equations removes that obstacle, and re-running this sweep
with σ_y\* free is the outstanding test.

That matters beyond canonicity, because **the ratio is the only device that disciplines both
variances without nailing either**, and the default still needs one. Imposing σ_y\* alone
repaired potential and moved the variance into g: `sigma_g` comes back at 0.124 against a
prior scale of 0.04, and g carries a pandemic-shaped trough, 1.26 at 2020Q1 recovering 0.94pp
to 2.20. Since r\* = g + z at corr 0.998, r\* inherits that trough whole. Freeing σ_y\*
instead does not sample at all. The ratio with a free level is the one configuration nobody
has run.

## 3. The empirical confirmation that r* is not identified

The counting argument at the top settles this before any sampling. What follows is the
evidence that the data behaves exactly as the count predicts. It is worth keeping because it
was found the hard way, and because it puts numbers on how badly the failure bites.

`sigma_z_prior_sweep.py` varies the HalfNormal scale on σ_z. On the 1993Q1 default, with the
1986Q3 run in brackets:

| prior scale | 0.025 | 0.05 | 0.10 | 0.30 |
|---|---|---|---|---|
| σ_z posterior / prior mean | 1.08 (1.04) | 1.13 (1.12) | 1.17 (1.10) | 1.26 (1.58) |
| z span | 0.075 (0.057) | 0.074 (0.105) | 0.269 (0.300) | **1.491** (2.841) |
| corr(r*, g) | 1.000 | 1.000 | 0.998 | 0.745 (0.418) |
| r* latest | 2.21 (1.61) | 2.23 (1.62) | 2.25 (1.73) | **3.10** (3.14) |

**Same verdict on both samples.** The posterior on σ_z is its prior, and r* latest moves 0.9pp
across a prior nothing measures.

σ_z's posterior is its prior, z's wander is proportional to the rope it is given, and the
fit barely moves across the whole range (the `sigma_IS` table in the opening section). The
flat ridge is what the count predicts.

Treating the IS curve as a noisy ruler for r* puts a number on it: the effective measurement
sd is `sigma_IS/|a_r|` = **12.7pp per quarter**, against an r* whose entire plausible range is
about 3pp. To pin r* to ±0.25pp would need `a_r` roughly six times larger, or σ_z six times
smaller, which asserts the answer rather than measuring it.

This is **Buncic-Pagan-Robinson 2023** made concrete on Australian data. It survived a retest
at three times the per-quarter signal, on the 1986Q3 sample where `a_r`/`sigma_IS` reached
0.193.

## What is NOT the finding

**The rate channel is not negligible.** An earlier version of these notes said it was, on the
basis of `a_r` ≈ −0.04, which compared an **impact** coefficient against **level** slopes from
models without gap persistence. The long-run slope `a_r/(1−a_y1−a_y2)` excludes zero. The gap
is persistent, so a sustained rate gap accumulates into a substantial output response.

What fails is r*, and for a different reason: identification runs through the per-quarter
impact coefficient, not the long-run multiplier.

What also survives, unrelated to r*: a reduced-form scatter of the output gap on the lagged
real rate is flat or wrongly signed at every lag (`is_curve`).

---

# The model

## Sample, data, and the indexed-yield fill

**Sample**: 1993Q1 → 2026Q2 (134 quarters as at the 2026-09-12 run; the end advances with each quarterly re-run). `DEFAULT_START` in `observations.py` sets it, and the reasons are in "The sample start is the biggest single choice" above: pre-1993 Australia had no inflation target, and including those quarters leaves the output gap unable to close through the 2022-26 disinflation.

`--start 1980Q1` restores the previous default. It is a no-op before 1986Q3: the indexed bond yield series starts there, and `observations.py` joins all series and drops incomplete quarters, so the effective floor is 1986Q3 whatever earlier date is passed.

| Series | Source | Notes |
|--------|--------|-------|
| log GDP × 100 | `gdp.get_log_gdp()` | ABS 5206.0 chain volume, SA |
| Cash rate | `cash_rate.get_cash_rate_qrtly()` | RBA OCR + historical interbank, end of quarter |
| π_exp (annualised %) | `expectations_model.get_model_expectations_unanchored()` | Project's own signal-extraction model, unanchored variant |
| π_4 (annual trimmed mean) | `inflation.get_trimmed_mean_annual()` | ABS 6401.0 |
| Fiscal impulse (lag 1) | `gov_spending.get_fiscal_impulse_lagged_qrtly()` | Same series the NAIRU model uses |
| Indexed 10y bond yield | `bonds.get_indexed_yield_filled()` | RBA F2 (1986+); see fill below |
| Linear trend (g anchor) | derived in `observations.py` | Linear regression of YoY GDP growth over the sample |

**Indexed yield gap fill**. RBA's F2 series for the 10y indexed bond yield has a 5-quarter gap (2013Q3–2014Q3) when Treasury was transitioning between the maturing 2020 indexed benchmark and the new 2025 benchmark. `get_indexed_yield_filled()` patches it with `nominal_10y − interpolated breakeven`. Breakeven inflation is anchored and moves slowly; the nominal 10y is observed throughout the gap and contributes the actual real-rate dynamics, including the 2013 taper-tantrum spike that a naïve interpolation of indexed_10y would miss. Filling the gap reduced divergences across most resolutions (A: 4,050→111; B: 11,458→3,349; C: 428→193; E: 819→173).

## Model structure (shared across resolutions)

All resolutions share the same potential output, Phillips curve, and (where wired in) soft anchor on g.

**Latent states**

| State | Symbol | Equation | Units |
|-------|--------|----------|-------|
| Trend growth | g_t | g_t = g_{t-1} + ε_g | annualised % |
| Potential output | y*_t | y*_t = y*_{t-1} + g_{t-1}/4 + ε_{y*} | log × 100 |

**Observation equations**

1. **IS curve** (D and F additionally include open-economy regressors: ToT growth, real TWI change, RBA ICP A$ growth):
   `log_gdp_t = y*_t + a_y1·y_gap_{t-1} + a_y2·y_gap_{t-2} + a_r·r_gap_{t-6} + γ_fi·fiscal_{t-1} + ε_IS`
2. **Phillips curve** (annual trimmed mean, anchor-augmented):
   `π_4_t = π_exp_t + b_y · y_gap_{t-1} + ε_π`
3. **Soft anchor on g** (C, D, E, F, G, H only):
   `anchor_t = g_t + ε_trend`,  `ε_trend ~ N(0, 2.0)` (σ fixed; a free σ collapses to ~0.02 and turns the soft anchor into a hard constraint).
   `--g-anchor` picks the series. `linear` (default) is a regression of year-on-year growth on time across the sample: unmoved by a single shock, but monotone by construction and fitted on quarters later than each date it describes. `cagr40` is the 40-quarter trailing compound annual growth rate, computed on the whole GDP series so the window is already full at the sample start: one-sided, and free to flatten out. **The choice changes nothing** (see Blind alleys), because at σ = 2.0 the anchor is close to inert. A non-default anchor writes to its own prefix and chart directory.

The differences between resolutions are entirely in the **r\* identity** and in whether the open-economy IS-curve regressors are wired in.

**The rate lag is a single t−6** (changed 2026-09-11 from HLW's averaged t−1, t−2; `--rate-lag 0` restores it). It matches the `is_curve` bench and `rstar_invert`, whose weighted 4/8 pair has an effective mean lag of 6.3, so the three are comparable on timing. The evidence is those two packages' lag sweeps: the slope strengthens monotonically with the lag because a regressor further from t carries less of the RBA's reaction to the economy.

**UNITS WARNING on `a_r`.** It is an IMPACT coefficient and is not comparable with a level slope from a model without gap persistence. The comparable quantity is `a_r/(1−a_y1−a_y2)`.

**Reparameterisation choices that matter:** trend_growth is centred (non-centring breaks the doubly-cumulated y* equation); σ_g uses a tight HalfNormal(0.04); r_innovation is non-centred (this fixed the σ_r funnel).

---

# The eight resolutions

**Only A has been re-run since the 2026-09-12 repair.** B through H were estimated with free
σ_ystar, the lockdown quarters fitted, and the averaged (t−1, t−2) rate gap. Their numbers
are the record of what those specifications did, not current output, and are not comparable
with A's.

### Resolution A: canonical HLW (closed economy)

`r*_t = g_t + z_t`, z a random walk. No anchor on g. **The live resolution**; current numbers
in "What the model does produce" above.

z carries a **level** rather than a path: span 0.269 and corr(r*, g) = 0.998 on the current
sample, 0.300 and 0.994 on the previous one. r* is g plus an offset in either.

**What we learnt**: Buncic's finding for Euro Area / UK / Canada also holds for Australia, and
it survived being retested at three times the per-quarter signal on the longer sample.
McCririck-Rees (RBA 2017) and Ellis (RBA 2022) report similar identification failure. On the
current sample r*'s own sampling is poor (R-hat 1.090, min ESS 39); on the longer one it was
clean and σ_z alone sampled badly. The failure moves around; it does not go away.

### B: canonical + indexed-bond observation

Adds `indexed_10y_t = r*_t + tp + ε_tp`, constant tp. z absorbs all r* dynamics; r* becomes the
bond yield less a constant ~0.93pp premium. r* latest 1.48%, trough −1.64% (2020Q4), 3,349
divergences.

**What we learnt**: with a strong external identifier the data does not estimate r*, it
relabels the bond yield as r*, and the HLW machinery becomes decorative. The negative trough
is post-GFC term-premium compression pushed into r* by the constant-tp assumption. B is
nonetheless closest to the RBA's stated working view, and its mechanism is the implicit
framework most central banks use.

### C: deterministic blend

`r*_t = α·g_t + (1−α)·(indexed_10y_t − k) + ε_t`, α ~ Uniform. Replaces the unidentified third
latent with a scalar. α median 0.56, 90% HDI [0.07, 0.96], tracking the prior almost 1:1.
r* latest 2.19%.

**What we learnt**: collapsing the unidentified state to one scalar produces a clean, in-range
r* path whose level is mostly the α prior, dressed in Bayesian language.

### D: canonical r* + open-economy IS curve

A's identity plus ToT, TWI and ICP regressors. All three coefficients land near zero; σ_IS and
`a_r` unchanged; z still dead. r* latest 1.77%.

**What we learnt**: refutes the hypothesis that A fails through SOE mis-specification. Upgrades
the finding from "weak rate channel in a closed-economy model" to "weak rate channel even with
the SOE block".

### E: blend + AR(1) z

C plus an AR(1) z with σ_z fixed at 0.15, so the IS curve may disagree with the blend. It
doesn't: z mean absolute 0.041pp over 40 years. r* latest 2.20%, near-identical to C.

**What we learnt**: C's deterministic identity is not over-constraining the answer. The blend
is what r* is.

### F: E + open-economy IS curve

Both extensions at once. z mean absolute 0.051pp; SOE coefficients dead as in D; r* latest 2.15%.

**What we learnt**: closes the empirical loop. Across A–F, σ_IS sits at 0.70 ± 0.02 and impact
`a_r` at −0.04 ± 0.01. No choice of r* identification or IS-curve regressor moves either. That
stability is the finding.

### G: blend with hierarchical Beta(a, b) on α

Lets the data choose the prior shape: a, b ~ Uniform(0.25, 2). Both hyperparameters straddle 1
with very wide HDIs; α 0.58 [0.03, 0.99]. r* latest 2.20%. **The implementation default among
the blends**, not because it is the answer but because it makes the one analyst-chosen quantity
as data-driven as the framework allows, and documents that the data declines even that.

**The G story.** α's posterior is nearly flat from 0 to 1, so the model splits into two stories:
draws track either trend growth or the bond yield, with little between. The blended median is
the average of two stories and almost no single draw sits at it, so quoting it as "the estimate"
overstates what is identified. The charts show the draw cloud and each mode's level rather than
the median.

Which anchor? The data will not say. Two outside pieces of evidence lean to the market reading:
the post-GFC decade fits a low yield-anchored r* better than a high growth-anchored one, and RBA
commentary describing the stance as having eased materially with no change in the cash rate, which
only a yield-anchored r* can do. A corollary worth carrying: a market-anchored r* moves, so the
gap between the cash rate and neutral can narrow without the Bank doing anything, and can widen
again the same way.

An earlier version used HalfNormal(1) as the hyperprior, which puts ~58% of its mass below 1 and
produced a strongly bimodal α. **Endpoint stacking appears only when the hyperprior allows Beta
shape parameters below 1**, putting the Beta in U-shape territory: a constraint-structure effect,
not a data preference for extreme α.

### H: blend with time-varying α_t (logit-RW)

α_t free to drift ~1pp/quarter. It lands flat: sd 0.002 across 158 quarters, total drift 0.01pp.
r* latest 2.20%.

**What we learnt**: the chain actively pulls α_t toward a constant despite ample room. There is no
era-specific signal about which anchor matters when. A framing in which r* "has shifted upward"
recently, which would need α_t drifting toward 0, is not supported inside this model.

### S: staged estimation (not an r\* identity)

`--resolution S`, or `uv run python -m src.models.rstar_hlw.stepwise`.

**S is not one model.** It is three, in serial, each holding still whatever the next one
measures and handing its answer forward as an imposed setting. The r\* identity is A's,
`r* = g + z`, unchanged. What differs is the estimator, and the letter is out of sequence to
say so.

| stage | held still | estimated | locked and carried |
|---|---|---|---|
| 1 | g, z, and the rate gap (`a_r` = 0) | `sigma_ystar` | σ_y\* |
| 2 | z | `sigma_g`, against a fixed σ_y\* | λ_g = σ_g / 4σ_y\* |
| 3 | nothing | everything else | reports the result |

"Held still" means the state does not MOVE; its level is still estimated. A zero-scale random
walk is not a distribution PyMC will build, so `equations/states.py` returns a constant path
at a free level instead.

**Why stage it.** No single estimation can pin `sigma_ystar` and `sigma_g` together. Free both
and the posterior funnels (`--canonical`: ESS 13, 2,939 divergences). Impose σ_y\* alone and
the variance reappears in g. Staging never asks the data to pin two at once, which is the
whole content of the papers' median-unbiased procedure.

**AN ANALOGUE, NOT A REPLICATION.** The papers stage because maximum likelihood returns
exactly zero for these variances, so they recover each ratio from an Andrews-Ploberger break
test through Stock-Watson's tables. MCMC does not pile up at zero, so each stage here reads
its quantity off a posterior instead. The sequencing is HLW's; the estimator is not.

**Result.** σ_y\* = 0.6243 against the 0.078 the default imposes, and λ_g = 0.0247, below
LW2001's 0.039. Stage 3 fixes what the default gets wrong about g and breaks what it gets
right about potential:

| | S (stage 3) | A (default) |
|---|---|---|
| g range over the sample | 1.11 | 2.53 |
| g at 2019Q4 | 2.39 | **1.28** |
| potential, sd of quarterly change | **0.615** | 0.187 |
| gap, level sd | 1.92 | 2.01 |
| gap 2026Q2 | 1.47 | 2.63 |
| `b_y` | 0.453 | 0.277 |
| `a_r` | −0.042 | −0.042 |

g loses the pandemic trough that the default carries and that r\* inherits whole. Potential
then moves at 63% of GDP's quarterly volatility, which is the pile-up. **The staging does not
rescue the decomposition; it relocates the defect and shows that only the imposed 0.078 was
holding it together.** The cycle survives better than that suggests: the gap's level sd barely
changes and corr(A gap, S gap) is 0.84, though only 0.70 over 1998-2019. `a_r` does not move
at all, so none of this touches the rate channel.

**TWO REASONS NOT TO QUOTE IT YET.**

Sampling fails on every stage: 1,950, 113 and 51 divergences against a limit of 15, worst in
the stage that sets the σ_y\* lock. Stage 1's funnel is between `sigma_ystar` and
`potential_innovations` and is a centred-parameterisation problem, so it is probably fixable.

And **S's bands are not comparable with A through H's.** Stage 3's intervals are conditional
on σ_y\* = 0.6243 and λ_g = 0.0247 being exactly right, when both were estimated with error
and estimated badly. Serial conditioning understates uncertainty by an amount nothing here
measures. HLW carry the same defect and say so.

## Cross-resolution summary

| | A | B | C | D | E | F | G | H |
|---|---|---|---|---|---|---|---|---|
| Divergences | 111 | 3,349 | 193 | 85 | 173 | 249 | 181 | 206 |
| r* span (median) | 0.42 | 6.51 | 3.31 | 0.91 | 3.26 | 3.50 | 3.22 | 3.19 |
| r* trough | 2.41% | −1.64% | 0.82% | 1.77% | 0.83% | 0.63% | 0.89% | 0.92% |
| r* latest | 2.43% | 1.48% | 2.19% | 1.77% | 2.20% | 2.15% | 2.20% | 2.20% |
| z status | dead | wild | n/a | dead | dead | dead | n/a | n/a |
| `a_r` (IMPACT) | −0.033 | −0.031 | −0.035 | −0.032 | −0.035 | −0.036 | −0.034 | −0.034 |
| `σ_IS` | 0.68 | 0.68 | 0.68 | 0.67 | 0.67 | 0.67 | 0.67 | 0.67 |
| `α` posterior | n/a | n/a | 0.56 [0.07, 0.96] | n/a | 0.57 [0.08, 0.96] | 0.53 [0.06, 0.95] | 0.58 [0.03, 0.99] | flat at ~0.59 |

**All pre-repair, at the averaged rate gap.** A's current figures are in the section above and
are not comparable with this row. The picture the table paints is unchanged: σ_IS and impact
`a_r` are flat across all eight, and r* tracks whichever observable the structural identity
admits: g (A, D), the bond yield (B), or the blend (C, E, F, G, H). The IS curve does not
adjudicate.

**S is deliberately absent from this table.** It is a current-settings run, so it would not
belong in a pre-repair comparison whatever else were true, and its bands are conditional on
two locks treated as known, so its uncertainty is not on the same footing as these columns
even once the settings match. Its own section above carries its numbers against the default.

---

# Reading r* for Australia

## The bond-versus-growth axis

Every credible Australian r* estimate sits on one axis, which the α-blend formalises:

- **α near 0** (B, RBA-style, market-leaning frameworks): the bond yield carries the structural signal; r* latest ~1.5%.
- **α ≈ 0.5** (C, E, F, G, H): equal-weight blend; r* latest ~2.2%.
- **α near 1** (A, D, canonical HLW for SOEs): trend growth carries it; r* latest 1.8–2.4%.

The data does not distinguish between these positions. **Disagreements about r\* are
disagreements about α**, and the choice is a prior commitment. That reading frame is the most
durable thing this package produced.

## Bullock cross-validation (May 2026)

Governor Bullock characterised cash 4.35% as "a bit restrictive, but less restrictive than 16
months ago, due to shifts in r*", which is informative on both level and dynamics: real cash ≈
1.65%, so "a bit restrictive" implies r* ≈ 1.0–1.4%, and r* has *risen* ~0.3–0.5pp over 16
months at an unchanged cash rate. On the pre-repair numbers, B (1.48%, +0.17pp gap) is closest
and the blends are incompatible; the indexed yield's 0.2pp rise over 2025 gives B ~0.5pp of r*
rise, consistent with the dynamics, while A and D imply zero transmission.

**The deflationary caveat.** This is a model preference, not external evidence. The RBA's
methodology, like most central banks', almost certainly weights the bond market heavily, so B's
proximity reflects two methods reading the same signal rather than independent confirmation.
The published Australian estimates (McCririck-Rees, Ellis 2022, IMF Article IV) mostly lean on
bond-market information too. Bullock's view sits at α ≈ 0 on the same axis as everything else:
one defensible position, not the truth.

## What will resolve the disagreement

Forward inflation outcomes, and the question is testable. Disinflation to target within ~12
months vindicates the low, yield-anchored reading; inflation still elevated 18 months on at an
unchanged cash rate vindicates the ~2.2% blends; disinflation only after further tightening
points to the middle. Re-estimate at Q4 2026 for a first read, Q2 2027 for a meaningful test,
and Q4 2027 or 2028 for a definitive one absent a major exogenous shock.

## Implications for downstream use

1. **A single number with credible bands overstates precision.** The within-resolution CI is mostly the prior projected through the model. The honest band is the cross-resolution spread, ~1.0pp.
2. **For NAIRU integration**, the median r* series from a blend resolution (C, G or H, all near-identical at the median) is a defensible input, but it is one possible r* path chosen by the analyst's α weighting, not a separately-identified structural quantity.
3. **The leverage is in richer external anchors, not in richer IS-curve identification.** That is now a measured statement rather than an impression: see the loading argument above.

Treat the model as a framework for thinking about r*, not a measurement of it.

## NAIRU integration

The original motivation. The realistic path is sequential coupling: run the pipeline from 1993Q1,
save the posterior median r* to disk, and expose an option in `src/models/nairu/observations.py`
to load it into `obs["det_r_star"]` in place of the Cobb-Douglas r*. Pre-1993 values fall back to
Cobb-Douglas or hold the 1993Q1 estimate. Propagating r* uncertainty through to NAIRU would need
a joint model or a Monte Carlo loop over r* draws; not implemented.

---

# Running it

```bash
./run-rstar-hlw.sh -v                          # Resolution A (default)
./run-rstar-hlw.sh -v --resolution G           # blend + hierarchical Beta on alpha
./run-rstar-hlw.sh -v --resolution B           # canonical + indexed-bond observation
./run-rstar-hlw.sh -v --estimate-only          # or --skip-estimate to re-chart

# undoing the 2026-09-12 repair
./run-rstar-hlw.sh -v --sigma-ystar free --exclude-window none   # pre-repair behaviour
./run-rstar-hlw.sh -v --lambda-g 0.053                           # HLW's US ratio (rejected, but only with sigma_ystar imposed too)
./run-rstar-hlw.sh -v --rate-lag 0                               # HLW's averaged t-1, t-2 gap

# the two sweeps that decided the defaults
uv run python -m src.models.rstar_hlw.lambda_g_sweep
uv run python -m src.models.rstar_hlw.sigma_z_prior_sweep
```

`analyse.py` always emits an overlay of the model's median r* against the NY Fed HLW estimates
for the US, Euro Area and Canada, pulled fresh each run. The chart is purely descriptive: none
of the foreign series are observations in the model.

---

# The record

## Blind alleys

Each of these was built and abandoned. One line is the right amount of space.

| Attempt | What happened |
|---|---|
| Tightened prior on σ_ystar (0.55 → 0.12) | Posterior 0.862, seven sd into the tail. The variance had to be imposed, not priored |
| λ_g imposed at HLW's US 0.053 | g held flat, `b_y` collapses (0.110 on the current sample, 0.054 on the old), 2005-19 gap well above zero. AU trend growth genuinely fell; forcing g constant pushes that into the cycle. Rejected on both samples |
| λ_g imposed at 0.34 (AU-implied) | Matches the free run closely. Not binding, so it buys nothing |
| Non-centring `trend_growth` | 5,976 divergences. y* already cumulates g, so the doubly-cumulated structure breaks NUTS gradients |
| Loose σ_g without an anchor on g | Divergences blow up, ESS collapses: σ_g and σ_y* funnel |
| HMA(13) of YoY growth as the g anchor, free σ | σ collapsed to 0.022; the COVID dip bled into g. Free measurement σ over-fits when it is the only constraint |
| 40q trailing CAGR as the g anchor, fixed σ = 2.0 | **The anchor series is not a lever.** The two anchors differ by up to 1.05pp and average 0.34pp apart at the endpoint; posterior g differs by at most 0.15pp, a pass-through of 0.124. Resolution C returns the same answer either way (persistence 0.946 vs 0.944, `a_r` −0.040 vs −0.041, `b_y` 0.281 vs 0.276), and the COVID-shaped dip in g survives both. The σ is the lever, not the series, and 2.0 is the opposite extreme to the HMA's collapsed 0.022 |
| Prescribing g as data (`given_g_test.py`) | **Nothing survives.** `ystar`'s g collapses persistence 0.945 → 0.655 and nearly closes the gap (2.63 → 0.47), which would overturn the claim that slow gap decay is structural. It does not replicate: on the 40q CAGR, a g nothing smoothed deliberately, persistence goes the other way to 0.966 and the 2026Q2 gap to −2.98 with unemployment at 4.35%. A trailing window dates a trend change half its length late, so cumulating it overshoots when growth slows. A flat 2.0 fails outright, +23.5: y\*'s level is the cumulated sum of g, so a wrong g destroys the decomposition rather than reparameterising it. r\* is untouched in all four, zero quarters of positive rate gap since 2022 |
| Regime-switching α (GFC split; then 2011Q3–2021Q4) | α posteriors overlap wherever the break is placed. Too little identifying power to support time-variation |
| Time-varying k, slope-based on the term spread | 10,655 divergences, ESS 17, r* path barely moved. Flexibility without identifying information |
| Intercept c replacing, then alongside, k | Level shift without releasing g; then c and (1−α)·k near-collinear |
| Constant tp in B's bond equation | r* trough −1.64%: post-GFC term-premium compression pushed into the r* level |
| Time-varying tp in B | Rejected conceptually: re-introduces the canonical identification failure |
| `target_accept` = 0.97 | Hides the geometry, does not fix it |
| 1980Q1 sample start | Pre-1993 regime contaminates the Phillips curve |
| Tighter σ_g and σ_z together | Fewer divergences but z dies: the model becomes univariate trend extraction |

## Not tried

- **Long-run survey expectations** (Del Negro et al 2017): Consensus 6–10y cash rate forecasts as an extra observation, pinning r*'s long-run mean. **The cheapest remaining unlock**, and the one the loading argument points at.
- **Term-structure block** (Bauer-Rudebusch 2020): arbitrage-free curve across maturities, making the term premium a proper latent. Significant engineering.
- **Convenience-yield observation** (Szoke et al 2024): needs long-history AU AA corporate yields; RBA F3 starts ~2005.
- **AR(1) trend growth**: would mean-revert g and might stabilise σ_g, at the cost of changing the long-run interpretation.
- **Dropping the IS curve**: tried, and it worked, but as a separate package. See `rstar_bonds`. It could not be a ninth resolution because removing the IS curve removes the thing that makes this HLW.
- **Reconciling with `rstar_bonds`**: it gives 1.24 against G's 2.23, a gap wider than this model's own cross-resolution spread. Outstanding in both sets of notes.

## Corrections to earlier versions of these notes

Three claims here have been wrong, all in the direction of over-reading the model. Kept as a
list because the pattern matters: twice the conclusion outlived the reason given for it.

| Claim | Status |
|---|---|
| "The rate channel is too weak, `a_r` ≈ −0.04" | **Wrong.** Compared an impact coefficient with level slopes. The long-run slope excludes zero. r* still fails, for a different reason |
| "`a_r`/`σ_IS` = 0.064, so r* would have to be wrong by 15pp" | **Superseded.** Measured against the broken decomposition. On the repaired model it is 0.193 and 5.2pp, and r* is still not identified. The per-quarter framing is also incomplete: see the 70-quarter filter argument |
| "The r* CI scales linearly with the prior σ_z" | **Mechanism wrong.** The CI scales sublinearly because g's uncertainty dominates it, while z itself scales superlinearly. Read z's own span against the prior. Conclusion unchanged |
| "Potential output is broken" (2026-09-11) | **Fixed** for Resolution A. Potential's quarterly change sd is 0.189 against GDP's 0.972 |
| "The repair tripled the per-quarter IS signal to 0.193" | **Sample-specific.** True on 1986Q3, but 0.079 on the 1993Q1 default. The 1980s rate cycle was carrying it |
| "The 2022-26 gap is too big by about a point" | **Wrong twice over.** It was too big by 2 to 3.6pp, and the cause was the sample start, not a level bias. On 1993Q1 the gap closes at about the Okun-implied rate |
| "`ystar_ustar`'s gap sd of 0.42 is the benchmark" | **Wrong.** That is the inflation-explained sliver, not the deviation from potential, and on the Okun test HLW's amplitude is the defensible one |

## File structure

```
src/models/rstar_hlw/
├── observations.py           # Data loading (incl. indexed_10y, the g anchors, SOE regressors)
├── equations/
│   ├── trend_growth.py       # g state equation (centred RW) + soft observation on g; sigma_g imposable via lambda_g
│   ├── potential.py          # y* state equation; sigma_ystar IMPOSED at 0.078 by default
│   ├── exclusion.py          # drops a window of quarters from a likelihood (the lockdowns)
│   ├── r_star.py             # r* = α·g + (1−α)·(indexed_10y − k) + ε   (C and G)
│   ├── r_star_blended_z.py   # r* = α·g + (1−α)·(indexed_10y − k) + z   (E and F)
│   ├── r_star_tv_alpha.py    # time-varying alpha_t via logit-RW (H)
│   ├── z_star.py             # r* = g + z (A, B, D)
│   ├── is_curve.py           # IS curve; fiscal + opt-in SOE block (ToT, TWI, ICP)
│   ├── phillips.py           # Phillips curve on annual π
│   └── indexed_bond.py       # separate indexed_10y observation (B only)
├── estimate.py               # Model assembly, NUTS sampling, save/load; resolution dispatch
├── results.py                # RStarResults dataclass
├── analyse.py                # Fan charts: r*, output gap, g, decomposition, alpha posterior
├── run.py                    # CLI: --resolution (A-H, S), --rate-lag, --sigma-ystar,
│                             #      --lambda-g, --exclude-window, --g-anchor,
│                             #      --canonical, --seed
├── stepwise.py               # Resolution S: three models in serial, each locking a variance
├── equations/states.py       # a random walk, or a constant at a free level when sigma is 0
├── given_rstar_test.py       # r* supplied as data: does the gap close? (moves it 0.18pp)
├── given_g_test.py           # g supplied as data, against a like-for-like baseline
├── lambda_g_sweep.py         # sigma_g/sigma_ystar ratio sweep; why lambda_g defaults off
├── sigma_z_prior_sweep.py    # sigma_z prior-scale sweep; the r* non-identification test
├── sigma_z_sweep.py          # fixed-sigma_z sweep on Resolution E (pre-repair)
├── alpha_prior_sweep.py      # alpha prior-shape sweep on Resolution C
├── refresh_all.py            # re-estimate every resolution
├── refresh_canonical.py      # re-estimate C, E, F + the hierarchical-Beta variant
└── MODEL_NOTES.md            # This file

run-rstar-hlw.sh              # Shell wrapper
```

External data dependencies for the SOE block and the comparison chart: `src/data/tot.py`,
`src/data/twi.py`, `src/data/commodity_prices.py` (RBA ICP, table I2), and
`src/data/world_rstar.py` (NY Fed HLW for US/Euro/Canada, chart only).

## References

- Laubach, Williams (2001): "Measuring the Natural Rate of Interest", FEDS 2001-56 (published REStat 2003): the origin of the three-step median-unbiased procedure, and of the finding that maximum likelihood puts both innovation variances at zero
- Holston, Laubach, Williams (2017): "Measuring the Natural Rate of Interest", FRBSF WP 2016-11 (published JIE 2017); (2023 update): NY Fed Staff Report 1063
- Buncic (2021): "On a standard method for measuring the natural rate of interest": MUE critique. Code: https://github.com/4db83/Issues-with-HLWs-natural-rate-Code
- **Buncic, Pagan, Robinson (2023)**: "On Constructing a Country-Specific Time Series for the Natural Rate of Interest", the formal identification critique these notes confirm on AU data
- Lewis, Vazquez-Grande (2019): "Measuring the Natural Rate of Interest": λ_z reparameterisation, AR(1) z. Code: https://github.com/kflewis/rStarLVGPublic
- Del Negro, Giannone, Giannoni, Tambalotti (2017): "Safety, Liquidity, and the Natural Rate of Interest". Code: https://github.com/FRBNY-DSGE/rstarBrookings2017
- Bauer, Rudebusch (2020): "Interest Rates Under Falling Stars"
- Szoke, Vazquez-Grande, Xavier (2024 FEDS Note): "Convenience Yield as a Driver of r*"
- McCririck, Rees (RBA Bulletin Sep 2017): "The Neutral Interest Rate"
- Ellis (RBA speech 2022): "The Neutral Rate: The Pole-star Casts Faint Light"
