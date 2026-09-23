# Canonical HLW by Kalman filter and maximum likelihood

Canonical HLW, estimated the way the papers estimate it: states integrated out by a Kalman
filter, parameters by constrained maximum likelihood, and the signal-to-noise ratios read off
structural break tests rather than off the likelihood.

**THE VERDICT: IT CONVERGES, AND THE ANSWER IS DEGENERATE.** `a_r` goes to its own lower
bound, potential output becomes indistinguishable from actual output, and r\* spans thirty
percentage points. Four different starting points reach the same optimum, and the state space
is validated by simulation recovery, so this is the model meeting Australian data rather than
an implementation fault.

## Why it exists

Sampling this model's states is what creates its funnels: `sigma_ystar` against 133 potential
innovations, `sigma_z` against 134 z states, `sigma_g` against 134 trend-growth states. The
MCMC implementation spent a long time reparameterising around them. Conditional on the
parameters the model is linear and Gaussian, so the states can be integrated out exactly,
leaving **nine scalars** with no funnel available to them.

The second reason is speed. A filter evaluation is microseconds, so a profile likelihood or a
lambda sweep costs seconds instead of the minutes per cell that MCMC charges. Two of the
findings below exist only because that became affordable.

## The state space

    s_t = [ y*_t, y*_{t-1}, y*_{t-2}, g_t, g_{t-1}, g_{t-2}, z_t, z_{t-1}, z_{t-2} ]

Nine states, three shocks, two observables. **HLW's averaged t-1, t-2 rate gap is what keeps
it that small**: a single t-6 lag would need seven lags each of g and z and run past twenty
states, which is presumably why the papers wrote the IS curve the way they did.

Both observation equations are rearranged so every known term sits on the left, because the
repo's `kalman_filter` has no observation intercept. That leaves `Z` constant given the
parameters, so no time-varying observation matrix is needed. `sigma_IS` and `sigma_pi` map to
`H` and the three state innovations to `Q`, which is exact rather than conventional: both
enter the MCMC model as the sd of a Normal on an observed quantity.

Initialisation is diffuse and explicit. All three states are random walks, so `T` has three
unit eigenvalues and the unconditional covariance does not exist; `P0` is always passed in
rather than left to `solve_discrete_lyapunov`. HLW initialise from a pre-sample regression
instead, so this is a departure.

## The form is validated, and the validation is itself a finding

`validate.py` simulates 4,000 quarters from the equations, then recovers the parameters by
maximum likelihood. Everything comes back:

| | truth | recovered | | | truth | recovered |
|---|---|---|---|---|---|---|
| a_y1 | 0.95 | 0.960 | | `sigma_IS` | 0.55 | 0.562 |
| a_y2 | −0.05 | −0.053 | | `sigma_pi` | 0.55 | 0.563 |
| a_r | −0.06 | −0.081 | | `sigma_ystar` | 0.30 | **0.241** |
| b_y | 0.28 | 0.265 | | `sigma_g` | 0.10 | 0.116 |
| | | | | `sigma_z` | 0.10 | **0.063** |

The log-likelihood at the optimum exceeds the log-likelihood at truth, as it must.

**Note which parameters recover worst: the variances, and both biased DOWN, on 4,000 quarters
of data generated from the model itself.** That is LW2001's pile-up reproduced in this repo's
own code, on a sample thirty times the real one. It is not a fact about Australia.

## HLW's process, as far as it goes

**Stage 1** deletes the rate gap and holds g and z constant, then reads `lambda_g` off an
Andrews-Ploberger exponential Wald test for a break in the mean of the preliminary `Dy*`. The
misspecification is the measurement: g is forbidden to move, so the growth slowdown is forced
into potential's drift, and the size of that break says how far g should have been allowed to
travel.

**`lambda_g` = 0.0447**, and three routes now agree:

| route | lambda_g |
|---|---|
| this package, HLW's own stage 1 break test | **0.0447** |
| `rstar_hlw`'s Bayesian analogue of the same idea | 0.0497 |
| LW2001, published | 0.039 |

The smoothed `Dy*` runs 4.24 to 1.76, against 4.24 to 2.46 from the MCMC implementation's
equivalent stage. Two separate implementations of the same cut-down model agreeing on the
preliminary potential is the strongest cross-check available here.

**Stage 2** restores the rate gap with `lambda_g` imposed and z still constant.

**Stage 3** estimates the rest with both ratios imposed.

### Their constraints

`c = 1` on trend growth is an identity here rather than a bound: the IS curve is built on
`real - (g + z)`, so r\* is g + z by construction. `a_r < 0` and `b_y > 0` are bounds, which
means **the sign of the IS slope is an assumption in this package exactly as it is in theirs**.
`a_y1 + a_y2 < 1` is a linear inequality, so the optimiser is SLSQP.

### What is not implemented

**Stage 2's `lambda_z` extraction.** HLW read it off a break test for an intercept shift in
the stage 2 IS equation. The machinery is in `mue.py` and works, but the exact regression they
run has not been verified against their code, and guessing it would put an unchecked
construction at the centre of the result. `lambda_z` is an input, not a measurement.

Buncic (arXiv 2002.11583) argues that step is unsound and inflates `lambda_z` anyway, so the
gap is worth filling carefully rather than quickly.

### The Stock-Watson table is UNVERIFIED

`mue.EW_CRITICAL_VALUES` is transcribed from the `rStar` R package's source, cross-read
between its stage 1 and stage 2 files, which agree. **Stock and Watson (1998) Table 3 itself
has not been consulted**, nor has the New York Fed's replication code. Everything that matters
lives in that one constant so it can be checked in one place.

## The result

All three stages converge. Four starting points, including one seeded at the MCMC posterior,
reach the same optimum (log-likelihood −204.23), so this is global rather than a local trap.

| | stage 1 | stage 2 | stage 3 |
|---|---|---|---|
| log-likelihood | −201.13 | −203.98 | −204.23 |
| `a_r` | 0 (deleted) | **−0.0025** | **−0.0025** |
| `a_y1` | 1.504 | 1.501 | 1.492 |
| `b_y` | 1.782 | 1.504 | 1.451 |
| `sigma_IS` | 0.117 | 0.139 | 0.137 |
| `sigma_ystar` | 0.974 | 0.964 | **0.963** |
| `sigma_z` | 0 | 0 | **3.903** |

**`a_r` sits exactly on its lower bound.** HLW's own constraint is the only thing keeping the
rate channel off zero.

**Potential output is output.** At `sigma_ystar` = 0.963 the quarterly sd of potential is
0.975 against GDP's 0.972. There is no trend and cycle, only trend.

**`sigma_z` = 3.90 because `lambda_z` divides by `a_r`.** HLW set
`sigma_z = lambda_z x sigma_IS / |a_r|`, which presumes an identified IS slope. Theirs is
"reasonably large and precisely estimated". Ours is 0.0025, so the expression divides by
almost nothing and r\* runs from −7.3 to +22.0.

`a_y1` = 1.49, `b_y` = 1.45 and `sigma_IS` = 0.137 are all far outside the MCMC posterior.
**The MCMC priors were not regularising, they were the only thing keeping the model out of
this corner.**

## Three findings the Kalman filter made affordable

### 1. Imposing `sigma_ystar` = 0.078 costs 139.7 log-likelihood points

A profile over `sigma_ystar`, everything else re-optimised at each point, with `lambda_g` at
0.0447:

| `sigma_ystar` | 0.078 | 0.25 | 0.40 | 0.60 | 0.80 | 0.963 |
|---|---|---|---|---|---|---|
| log-likelihood | −343.7 | −318.9 | −299.7 | −244.7 | −209.1 | **−204.0** |
| cost vs optimum | **139.7** | 115.0 | 95.7 | 40.7 | 5.1 | 0 |
| `b_y` | 0.036 | 0.356 | 0.385 | 1.764 | 1.665 | 1.509 |

A likelihood ratio statistic near 279 on one restriction. **The data do not mildly prefer the
pile-up; they reject 0.078 overwhelmingly.** The MCMC implementation could only show that the
prior and posterior on `sigma_ystar` were disjoint. This says what the restriction costs.

`a_r` is on its bound at **every** point of that profile, so the dead rate channel and the
pile-up are independent failures, neither causing the other.

### 2. z's shape is identified and its scale is not

Imposing `sigma_z` directly rather than through `lambda_z`:

| `sigma_z` | 0.05 | 0.10 | 0.25 | 0.50 | 1.00 | 3.90 (`lambda_z`) |
|---|---|---|---|---|---|---|
| log-likelihood | −203.983 | −203.983 | −203.988 | −203.999 | −204.041 | **−207.17** |
| z range | 0.02 | 0.08 | 0.48 | 1.75 | 5.75 | 43.9 |
| corr of `a_r·z` with the 0.50 case | 0.9996 | 0.9996 | 0.9999 | 1.0000 | 0.9967 | 0.437 |

**A twenty-fold change in z's amplitude costs 0.06 log-likelihood points**, and the path keeps
a correlation above 0.996 throughout. The curve is the right shape and arbitrarily stretched
on the y axis.

The mechanism is not a trade-off between `a_r` and z, which would conserve their product:
`a_r` stays pinned at its bound throughout. It is simpler and worse. **`a_r` is so small that
`a_r·z` contributes almost nothing to the IS equation, so the likelihood barely notices z at
all.**

And **HLW's own `lambda_z` value fits worse**, −207.17 against −203.98. Imposing `sigma_z`
directly is not a compromise with their method, it beats it on their own criterion. That is
what `sigma_z_imposed` in `stages.py` is for, and it is the package's one deliberate departure
from the papers.

### 3. Whatever you observe z with, r\* becomes it

The 5y5y risk-neutral forward was added as a third observable loading on r\*, then removed.
With `forward_bias` free, pinned at zero, and pinned at `rstar_rba`'s −0.109:

| bias | log-likelihood | r\* 1993 | r\* 2026Q2 | corr(r\*, market 5y5y) |
|---|---|---|---|---|
| free (−1.598) | −213.197 | 5.23 | 2.63 | **1.000** |
| pinned 0.0 | −213.197 | 3.63 | 1.03 | **1.000** |
| pinned −0.109 | −213.197 | 3.74 | 1.14 | **1.000** |

Market 5y5y real: 3.63 to 1.03. At a pinned bias of zero, **r\* reproduces the market series
to two decimals**. The log-likelihood is identical across all three, so the level is no more
identified than before: shifting the bias slides r\* bodily and z absorbs it. `sigma_fwd` came
back at 0.063, estimated not imposed, meaning the likelihood CHOSE to track the forward almost
exactly.

**With `a_r` near zero, the model has no view about r\* that it can defend against any
observable you give it.** This is not "z is unidentified". It is stronger: whatever equation
you attach to z, r\* becomes that equation's right-hand side. The code for this was removed
after the test; the finding is why.

## What does work

**g agrees across estimators.** Kalman/ML gives 3.75 to 2.21; the MCMC implementation gives
3.79 to 2.20. The trend and cycle decomposition is robust to how it is estimated, and it is
specifically r\* that fails.

## Known defects

- **The smoother is ill-conditioned when `sigma_z` is at `HELD_CONSTANT`.** It inverts a
  near-singular prediction covariance and returns smoothed states of order 1e90. The filter's
  log-likelihood is unaffected, so profile results stand, but no smoothed state from a
  held-constant configuration should be read.
- The Stock-Watson table's provenance, above.
- `lambda_z` is an input rather than a measurement, above.

## File structure

```
src/models/rstar_hlw_kalman/
├── state_space.py   # T, R, Z, Q, H and the marginal log-likelihood
├── mue.py           # Stock-Watson tables, exp-Wald / mean-Wald / QLR, the mapping
├── stages.py        # the three stages, their constraints, the stage 1 extraction
├── validate.py      # simulate at known parameters, recover them
├── run.py           # estimate and chart against the MCMC run
└── MODEL_NOTES.md   # this file
```

It imports `build_observations` from `rstar_hlw` and `kalman_filter` from `dsge`, both one
directional, so deleting this directory breaks nothing.
