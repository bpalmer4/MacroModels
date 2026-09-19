# Regime u\*: a spline over imposed regimes

## Verdict: this model does not work

It fails in two independent ways, and the fixes for each destroy the other.

**It does not work within regimes.** The residual is serially correlated at
+0.60 to +0.88 inside every one of the six, so the equation is treating a
persistent component as independent noise. Residual scale ranges from 0.20
(2015-19) to 4.06 (1974-83) against a single fitted `sigma` of 0.595, so the
1970s are absorbed by the Student-t tail rather than explained. Fitted values
have a standard deviation of 3.47 against a surprise standard deviation of
2.99, which only reconciles because fitted and residual correlate at about
-0.51: the equation manufactures swings larger than the thing it explains and
cancels half of them.

**It does not work across regimes.** u\* is forced continuous at every
boundary by the spline, while the parameters are allowed to jump. With the
pre-Accord decade given its own slope, `beta` steps 2.41 to 6.57 at 1974Q1 and
back at 1983Q3: the same unemployment gap implies 2.7 times the inflation
effect on one side of a date as the other, instantaneously. The model asserts
that a regime change is "a change in direction rather than in position" for
u\* and the opposite for the structure that reads it.

Do not quote a level from this model.

---

## What it is

u\* as a natural cubic spline with knots at imposed regime dates, read off
**one** observation equation:

```
pi_t - pi^e_t  =  alpha_k  -  beta_k x (u_t - u*_t)/u_t
                  +  rho x d4pm_t  +  xi x GSCPI_t^2 x sign(GSCPI_t)  +  e_t
```

Inverting it,

```
u_t - u*_t  =  - u_t x (pi_t - pi^e_t) / beta
```

**The unemployment gap is the inflation gap rescaled by `u/beta`.** No
labour-market observable enters except `u`, which is on both sides. Nothing
about the regime dates, the spline or the knots changes that. Two second
equations exist to break it, on unit labour costs and on output (`--wage`,
`--okun`), and neither does. Both are off by default.

**The Okun equation is well determined and useless here.** As error
correction, `du_t = a + b x dy_t + lambda x (u_{t-1} - u*_{t-1}) + e`, it
returns `b` = -0.14 [-0.172, -0.108] and `lambda` = -0.022 [-0.035, -0.009],
both entirely the right side of zero under two-sided priors. But a `lambda`
that small means a 1-point error in u\* shifts predicted `du` by 0.022 against
a residual scale of 0.308, so u\* would have to be wrong by fourteen points
before this equation objected. Turning it on moves u\* by 0.088 on average and
leaves the credible band at 1.28 against 1.29. A second observable that cannot
disagree is not a second opinion.

Breaks: 1974Q1, 1983Q3, 1993Q1, 2015Q1, 2020Q1. Institutional, so they can be
argued on history rather than fit. See `config.DEFAULT_BREAKS` for why 2008Q4
was tried and moved to 2015Q1.

---

## What the sweeps establish

**The sample start does not matter after 1983.** 1983Q1, 1970Q1 and 1959Q3
starts agree to 0.08-0.10pp over their common span. Ten years of run-up is
enough to stop 1993 being set by the unemployment rate beside it: this model
reads about 7.0 at 1993Q1 where `ustar` reads 10.75 and `ystar_ustar` 10.77,
both opening there with a diffuse prior.

**The expectation before 1983 decides the 1970s and nothing else.** PIE_RBAQ
is MARTIN's expectations variable, exogenous to that model and built after
Cusbert (2017) as a random walk in trend inflation. Fitted as adaptive
learning it has a half-life of 23 quarters, so it lags a fast climb: 6.52
below year-ended headline across 1974-79, then 0.97 above it across 1983-92.
Used as `pi^e` it puts u\* near 10 in the late 1970s against unemployment
near 5. The salience rule gives about 5. Both constructions are asserted and
nothing in the data chooses. Post-1983 u\* moves by less than 0.1pp either way.

**The inflation measure matters and the trimmed mean is cleaner.** Splicing
headline to the trimmed mean at 1983Q1 cuts residual scale from 1.043 to
0.694. Around the GST, headline lifts 4.29 points over its 1999 base and the
trimmed mean 0.90, so four fifths of a pure tax event is removed. The two
correlate 0.905 over 174 overlapping quarters with a mean difference of -0.04,
so the splice needs no offset.

**Three things that change nothing.** Knot multiplicity at 1974Q1 (3, 2 or 1)
moves the 1970s peak between 10.12 and 10.26. Doubling `beta_prior_sd` from
1.5 to 3 moves `beta` by 7 per cent, so the prior was never binding. Dropping
the 1960s from the sample moves the 1970s by 0.15pp.

**One beta across the sample is refuted.** Holding u\* at its fitted path,
four regimes want 3.36 to 4.38 against a pooled 3.60 with residual means
inside 0.18. 1974Q1-1983Q2 is left with a residual averaging **+2.57** across
ten years, which is a specification failure. But allow that regime a constant
and its slope falls to 3.86, near the pooled value, with a +2.34 intercept: it
wants a **level**, not a steeper curve. Fitting it through the origin forces
the steepness.

**The Accord decade needs less than it appears to.** Against a pooled 3.60 it
wants a slope of 0.06 and looks like a decade with no Phillips curve. But that
pooled slope was inflated by the 1970s. Once the pre-Accord decade is
separated and the pooled slope falls to 2.43, the Accord fits it with a
constant of +0.31, 90% [-0.073, +0.681], which may be zero. Giving it its own
slope instead returns 0.55 with the interval touching zero, improves `sigma`
by 0.007, and puts a near-zero denominator into the inversion, which sends the
`implied_ustar` diagnostic to -20 and +33 in that window.

---

## The two fixes, and why neither is available

**Per-regime intercepts and slopes together are not identified.** Within a
regime the equation is `(alpha_k - beta_k) + beta_k x u* x (1/u)`, so only the
variation in `1/u` separates u\*'s level from the constant. That variation is
thin: the coefficient of variation of `1/u` is 0.060 in 2015-19. Fitted
saturated, `alpha` and u\*'s level correlate at -0.72 to -0.94 within regimes,
the mean credible band more than doubles to 3.27 points, spline coefficients
hit both the 0.5 floor and the 12.0 ceiling, and u\* correlates **-0.273**
with a smoothed unemployment rate. It samples cleanly and returns the best
`sigma` of any variant, 0.577, which is the warning: fit cannot arbitrate here.

**An AR(1) error removes the identification entirely.** `phi_e` comes back at
**0.965** [0.945, 0.984], a near unit root, so the error is itself a slow
stochastic trend, which is what u\* is. The two compete for the same
persistent movement and the error wins, being unconstrained where u\* must
pass through a spline. Both slopes collapse toward zero (1.52 and 0.53, the
second touching zero), the mean credible band widens fourfold to 5.09 points
on a series that has ranged 1.6 to 11.1, and u\* averages 4.12 through the
Accord decade against unemployment of 8.23.

The second result is the important one. **The persistent component of the
inflation surprise is what identifies u\*.** The static form's apparent
precision, a band of about 1.3 points, comes from treating a serially
correlated error as 226 independent observations. An AR(1) at 0.965 and a
slow-moving u\* cannot be told apart from one series.

---

## Other things wrong

- `eq_prior`'s upper bound of 12.0 binds in every spliced-expectations
  variant: the late-1970s coefficient sits at 11.75 to 11.85 with its interval
  reaching the bound. The level is capped, not estimated.
- `config.beta_prior_sd` is justified against `ustar`'s `gamma_pi` of about
  1.15, but that coefficient is on a quarterly inflation basis and this model
  is year-ended, so the comparable figure is roughly 4.6.
- `regime_sigma` splits the residual scale at the expectations handoff, which
  is inert under the default since every quarter's expectation is measured.
  The heteroskedasticity that matters is by regime and is not modelled.
- `implied_ustar` omits the supply controls the fitted equation includes,
  worth +0.54 on average over 2021-23 and 1.80 at 2022Q2.
- No trimmed mean exists before 1983Q1, so the 1970s carry every policy-driven
  price movement, including the Medibank changes of 1975-76 and 1978. Not
  measured here.

---

## Files and usage

```bash
./run-regime-ustar.sh                       # defaults: 1970Q1, spliced expectations,
                                            #   spliced CPI, one beta, no intercept
./run-regime-ustar.sh --beta-groups 0 1 0 0 0 0 --intercept-regimes 2
                                            # pre-Accord its own slope, Accord a constant
./run-regime-ustar.sh --expectations model  # salience rule before 1983 instead of PIE_RBAQ
./run-regime-ustar.sh --inflation headline  # one measure end to end
./run-regime-ustar.sh --okun                # output error correction as a second equation
./run-regime-ustar.sh --ar1                 # AR(1) error; see above
./run-regime-ustar.sh --analyse-only        # re-chart a saved trace
```

`config.py` holds every imposed quantity and records it in `constants`, which
is saved beside the trace. Charts and this run's diagnostics go to
`charts/RegimeUStar/`.
