# r\* from a TVP-VAR with stochastic volatility

**RETIRED.** A Lubik-Matthes time-varying-parameter VAR: three variables, coefficients that
drift as random walks, stochastic volatility, and r\* defined as the H-quarter-ahead projection
of the real policy rate. It runs, it samples cleanly, and it is not useful.

It was removed from `rstar_summary` and then retired, on the same day it was stripped back to
the canonical specification. The strip-back is what made the problem legible.

## Why it was retired

**The approach needs the economy to settle, and Australia's never does in this sample.**

r\* here is read off the VAR's own estimated dynamics: project the system forward 20 quarters
and report where the real rate goes. That is only meaningful if the system reverts to something
over 20 quarters. The fitted VAR has a median spectral radius of **0.983**, so 0.983^20 = 0.71
of today's state is still in the answer, and r\* comes back correlated **0.953 with the real
cash rate**. It is reporting the current real rate with extra steps.

That is not a fixable defect, because there is no calm stretch to estimate against:

```
1993-96    emerging from the early-90s recession, disinflation
1997-2012  a decade-long build-up and its collapse      61 quarters
2013-2021  negative real rates, ELB-adjacent            36 quarters
2022-26    the inflation shock                          18 quarters
```

Thirty-three years with no stationary period. A VAR cannot distinguish a very long swing from a
near unit root; they are the same object to it. So the persistence is probably not even an
error. It is the model correctly describing an economy that was not mean-reverting on any
horizon it could see.

**The finding generalises, and it is the useful thing this package produced.** It sinks
approaches that *measure* equilibrium from realised behaviour. It does not sink those that
*assert* a structure defining equilibrium: `rstar_rba` identifies neutral from a policy rule and
needs only that the rule is stable, `rstar_bonds` reads a market price that is quoted
continuously through turbulence. Both convert the problem into a conditioning assumption, which
their notes state. This model had no such assumption to fall back on, which was the point of
building it and is the reason it fails.

## Why not just fix it

**Not by choosing a different estimand.** All three available definitions fail on the same
number. The projection is part nowcast; the steady state solves `(I - F)^-1 d`, dividing by one
minus a radius of 0.983, so its median is set by near-unit-root draws; the forward window sits
between them. The estimand was changed three times in one day, each time argued from agreement
with the models this one exists to be independent of, and none of the changes helped.

**Not by choosing a different sample.** Four variants were estimated.

| spec | quarters | radius | explosive | r\* latest | corr cash |
|---|---|---|---|---|---|
| 1993 start, COVID kept | 132 | 0.983 | 32.3% | 1.38 | 0.957 |
| 1993 start, COVID excluded | 132 | 0.974 | 24.5% | 1.66 | 0.931 |
| 1999 start, COVID kept | 108 | 0.996 | 47.3% | 1.61 | 0.955 |
| 1999 start, COVID excluded | 108 | 0.985 | 38.1% | 1.76 | 0.930 |

The r\* paths correlate 0.993 to 0.998 with each other and spread 0.38pp at the last common
quarter. Four specifications that disagree sharply on diagnostics agree almost exactly on the
answer, because the answer is the policy rate.

COVID is not the cause: excluding it moves the radius 0.983 to 0.974 and the radius is *worst in
2000-07*, a calm period. A 1999 start makes it worse, to 0.996, because it keeps the worst era
and discards a fifth of the sample. 1994-95 is not the cause either. And the elevated radius is
one contiguous run of **61 quarters, 1997Q1 to 2012Q1**, so there is no window to excise: that is
46% of the data.

**Not by adding shrinkage.** `theta_0` has a near-flat `sigma = 1.0` prior and Minnesota
shrinkage is the textbook remedy for a trending VAR. But shrinking toward stationarity imposes
the mean reversion the data denies, and the projection would then return to a resting point that
exists because the prior put it there. That is the same move as the inflation conditioning that
was stripped out, under another name. It is available, but only as a declared belief about the
Australian economy, swept like `sigma_q`.

**Not by sampling harder.** `--target-accept 0.99` cuts divergences 5 to 1 and raises min ESS
368 to 522, while the radius and the explosive share do not move at all. Better exploration of
the same badly-shaped posterior.

## What the model says, for the record

Headline run, canonical spec at `target_accept` 0.99:

| | |
|---|---|
| r\* latest (2026Q2) | 1.38% real |
| r\* sample mean | 1.58 |
| median spectral radius | 0.983 |
| draw-quarters explosive | 32.3% |
| corr(r\*, real cash rate) | 0.953 |
| corr(r\*, constant-coefficient baseline) | 0.953 |
| `sigma_q` posterior median | 0.0022, against a prior median of 0.0135 |
| divergences / min ESS | 1 in 4,000 / 522 |

**Do not quote the latest level.** Across `sigma_q` it runs 1.08 to 3.17, non-monotonically.

Two readings do survive the `sigma_q` sweep, and are the only quotable outputs: the **sample-mean
real rate of 1.34 to 1.66**, and a **negative 2016-19 reading of -0.65 to -1.07**, in an era where
`rstar_bonds` and `rstar_invert` disagree about the sign.

```
sigma_q  r*_latest  r*_mean  r*_range  corr_cash  r*_2016_19  pct_explosive
  0.000      2.227    1.658     7.532      0.964      -0.848          0.595
  0.002      1.082    1.597     6.275      0.956      -0.653          0.284
  0.005      1.591    1.575     6.992      0.915      -0.651          0.219
  0.010      1.969    1.473     7.876      0.890      -0.824          0.126
  0.020      1.752    1.359    11.025      0.895      -1.067          0.093
  0.050      3.170    1.338    23.247      0.660      -1.000          0.095
```

Sampled at `target_accept` 0.95. The cash-rate correlation only falls below 0.89 at
`sigma_q = 0.05`, whose path swings 23pp: the only setting that breaks r\*'s dependence on the
policy rate is one that makes it meaningless.

## Three results worth keeping

**The drift is identified and does almost nothing.** `sigma_q`'s posterior median is 0.0022
against a prior median of 0.0135, so the data has a clear opinion and it is "much less than you
assumed". This is *not* the `rstar_hlw` failure of a posterior sitting on its prior. But r\*
correlates 0.953 with a constant-coefficient VAR run through the identical projection. Several
thousand drifting states, no visible effect.

**Stochastic volatility works.** Growth volatility of 11.4 in 2020Q2 against a normal 3 to 5,
absorbed by the volatility states rather than the coefficients, which is exactly its job. The
argument for keeping COVID in the sample held up.

**2008Q4 is the sharpest single illustration.** The real cash rate fell 2.60pp as the RBA cut in
the crisis; r\* fell **3.24pp**, more than the policy rate did, and 2.2 times the next largest
move in the sample. A neutral rate is what policy moves around. The level just before was 5.20%
real, roughly 9% nominal.

## The specification, for anyone reading the code

```
(1)  y_t      = c_t + B1_t y_{t-1} + B2_t y_{t-2} + e_t     y = [pi, growth, r]
(2)  Theta_t  = Theta_{t-1} + sigma_q · eta_t
(3)  A e_t    = diag(exp(h_t/2)) u_t
(4)  h_{i,t}  = h_{i,t-1} + sigma_h,i · xi_{i,t}
```

Inflation is the trimmed mean and growth is GDP chain volume, both one-quarter changes
annualised; the real policy rate is the cash rate less unanchored model expectations, built as
`rstar_bonds` builds its `r`. Two lags, 1993Q1 to 2026Q2, 132 usable quarters, horizon 20.
`A` is lower triangular with a unit diagonal so the likelihood needs no Jacobian, and its three
free elements are constant.

Every random walk is non-centred, which is what makes NUTS cope with ~2,800 drifting states.
r\* is computed in `results.py` from the saved posterior, not in the graph, so the horizon can
be changed without re-sampling.

Two conventions that were never resolved and would need to be for any successor:
`rstar_posterior` keeps explosive draws while `steady_state_posterior` drops them, and
restricting to fully stable draws moves r\* from 1.41 to 1.09. Bands are quantiles rather than
HDIs, because a highest-density interval on a heavy-tailed projection is unstable and misleading
about the centre.

## If anyone picks this up again

The question to answer first is not about this model. It is whether a five-year-ahead neutral
rate is a meaningful object for Australia over a sample containing no stationary period. If the
answer is no, no VAR of this family will help. If the answer is yes, it will be because a
structure was asserted that defines equilibrium without needing to observe it, and at that point
`rstar_rba` and `rstar_bonds` already exist.

A successor that avoided the trap would need an estimand that survives non-stationarity: a
shorter horizon, or a statement about direction rather than level.

Reproduce the comparison charts with
`uv run python -m src.models.rstar_tvpvar.compare_specs`, which reads the saved traces and
estimates nothing.
