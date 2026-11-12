# Joint y\* / u\*, one likelihood, and a gap that is not entirely inflation

`ystar` and `ustar` estimated together, with one addition to the output gap and, by default, a
second labour-market series: underemployment, observed beside unemployment (the "slack split").

```
STATES
  g_t          = g_{t-1} + e_g                      sigma_g imposed (0.015)
  y*_t         = y*_{t-1} + g_{t-1} + e_y           sigma_ystar imposed (0.078)
  s*_t         = s*_{t-1} + sigma_t·z_t             total structural slack; sigma_t
                                                    falls linearly from 0.35 at the
                                                    start to 0.06 at 2002Q1, flat after
  logit(θ_t)   = logit(θ_{t-1}) + tau_t·w_t         θ = unemployment's share of s*;
                                                    tau_t falls from 0.10 to 0.02 by 2002Q1
  u*_t  = θ_t·s*_t                                  the unemployment NAIRU
  ue*_t = (1 − θ_t)·s*_t                            its underemployment counterpart

GAP
  gap_t = c·(ann(pi_q,t) - 2.5) + v_t               v ~ N(0, sigma_v); ann() compounds

OBSERVED
  log_gdp_t = y*_t + gap_t + e_c                    sigma_e estimated
  u_t       = u*_t  - beta·gap_t    + e_o           sigma_okun imposed (0.20)
  ue_t      = ue*_t - beta_ue·gap_t + e_ue          same noise and prior as unemployment's
  pi_q,t    = q(2.5) + lambda·[pi_q,t-1 - q(2.5)] + (1 - lambda)·beta_pi·[q(pi^e_t) - q(2.5)]
              + gamma·(u_t - u*_t)/u_t + gamma_ue·(ue_t - ue*_t)/ue_t
              + rho·d4pm_t + xi·GSCPI_t²·sign(GSCPI_t) + e_p
```

**The Phillips curve is a hybrid one by default**: backward-looking expectations (last quarter's
inflation) share one unit of weight with the target; `--no-inflation-lag` removes them. See "The
hybrid Phillips curve".

**Why two slack series.** Unemployment misses the underemployed, so slack that moves from
unemployment into underemployment looks, to an unemployment-only model, like a fall in u\*. With
one series the model cannot tell the two apart, and with Okun pulling u\* onto unemployment any
fall lands wherever the walk is left free. The slack split ties the two series by a common trend:
u\* can fall only if total slack s\* falls, which the output gap and inflation must support, or if
underemployment rises to take up the share. The composition question comes from Trent Saunders
(QTC, 2022), who finds the fall in the Australian NAIRU largely disappears for underutilisation.

**u\* stays a standard unemployment NAIRU**, on the same scale as the single-series model, so it
is reported and charted like any other. s\* walks on the underutilisation scale, so its prior and
steps are the unemployment walk's doubled; the share walk is tight after 2002, a setting swept and
found not to move the split. All of the step sizes are judgements, not estimates. `--slack
unemployment` runs the single-series model, with its own walk scale; see "The slack measure".

Every prior is taken unchanged from the two parent models, as are `sigma_ystar` and `sigma_g`,
so a difference in the posterior comes from joint estimation and from `v`, not from re-tuning.
The one departure is `sigma_okun`, which `ustar` estimates and this model imposes (see "Why
`sigma_okun` is imposed").

---

## Read this first: the model exists to estimate one number

`sigma_v`, and nothing else here is new.

**Why it cannot be estimated in `ystar`.** Write the free component into that model and the
GDP equation reads `log_gdp = y* + c·d + v + e_c`. Two mean-zero terms, one equation. Only
`Var(v) + Var(e_c)` is visible, the split is a flat ridge, and whatever comes back is the
prior. Worse than useless: the reported gap becomes `c·d + λ·(residual)` with λ set by the
prior variances alone, which is a continuous dial between `ystar` as it stands (λ = 0) and
detrended GDP, a filter (λ = 1).

**Why it can be estimated here.** The gap enters the Okun equation too, as `-beta·gap`. So

```
  cov(GDP residual, unemployment residual) = -beta·Var(v)
```

and the split is identified. That single covariance is the entire informational gain from
joining the two models, and `--no-okun` is the control that demonstrates it: without the
second equation `sigma_v` should return its prior.

**What the answer decides.** If `sigma_v` is near zero, the inflation-defined gap is the whole
output gap. If it is large, inflation sees only a slice of the cycle, and the rest is cyclical
movement that unemployment can see and inflation cannot.

**Expect weak identification.** The moment is a covariance between two large residuals, so "the
posterior equals the prior" is a live outcome. The prior-versus-posterior check in `analyse.py`
is there to catch it.

**The model is small on purpose.** `sigma_v` is a diagnostic on `ystar`: how much cyclical
movement the inflation identity misses, given that unemployment can see some of it. It is not a
comprehensive business cycle decomposition and makes no claim about what the missing part is made
of. Freeing correlated disturbances, adding cyclical measurements until the trend/cycle split is
free, or estimating the gap outright are each a different model. The imposed variances are the
price of being small, and the response to them is to sweep and report, not to add structure until
they become estimable.

---

## What this does not fix

**The circularity is attenuated, not removed.** The Phillips curve regresses inflation on
`(u - u*)/u`, and through Okun `u - u*` still contains `c·(pi - 2.5)`. Part of the regressor is a
rescaled copy of the dependent variable, which manufactures a negative `gamma_pi` whether or not a
Phillips relationship exists. Because the gap now also contains `v`, which is not inflation, the
manufactured share shrinks with the free share of the gap, but it never reaches zero while the gap
is defined off inflation.

**The quarterly default makes it worse, deliberately.** On the quarterly basis the gap is an exact
multiple of the Phillips curve's own dependent variable; on the annual basis (`--gap-pi-basis
annual`) the correlation is well below one. Quarterly is kept because it samples better and
matches the basis the Phillips curve uses. The cost is real and unquantified.

**Imposed variances.** `sigma_ystar` and `sigma_g` are imposed for the parents' reason, the
pile-up between a free state and a free residual. `sigma_ystar` barely moves the gap; `sigma_g`
has not been swept. The imposed number that carries the answer is `sigma_okun`, which has no
external anchor.

**No IS curve, no r\*.** The rate-to-activity link is not identified on this data, so a third
star would attach to the system only through the missing leg.

---

## The pandemic window

2020Q2-2021Q3 is dropped from **all three** equations by default: potential output is not well
defined in a lockdown, inflation was moved by free childcare and administered prices, and
JobKeeper held measured unemployment far below any reading of slack. `--exclude-scope gdp`
drops it from the GDP equation only. The window boundaries are a judgement.

---

## Design decisions

**`v` is iid, not AR(1).** A free persistent cycle beside two free trends is three places for
unexplained persistence to hide. iid understates the cycle `v` can find, so a large `sigma_v` is
strong evidence and a small one is not proof of absence.

**`v` is non-centred**, because the data are weakly informative about it by construction, which
is when non-centring pays.

**`y*` is `ystar`'s random walk, untouched.** Adding `v` moves variation from `e_c` into the gap,
which if anything protects `y*`.

---

## How to read it

**Quote the gap's width, never its split.** How the gap divides between the inflation-defined part
and `v` swings with settings the data do not determine (the inflation horizon, `sigma_okun`, the u\*
law); the gap's width barely moves.

**`sigma_okun` is imposed because, free, it does not mix.** It sits on a ridge with `sigma_e`, and
imposing it dissolves the ridge. The answer depends on it: loosen it far enough and `v` vanishes and
the model reduces to the inflation-defined gap alone. `--free-sigma-okun` runs it free.

**The early sample is not well identified.** The u\* charts shade 1993Q1-2003Q4
(`analyse.UNIDENTIFIED_WINDOW`). Inflation sat near target through the mid-1990s, so Okun rather
than inflation places u\* there, and the fast falls in unemployment around 1995 and over 2002-04
ask u\* to move further and faster than the walk allows.

**`gamma_pi` is not a clean Phillips slope.** Its regressor is partly its own dependent variable,
through Okun. For the same reason the supply coefficients `rho_pi` and `xi_gscpi` are likely
scaled up.

**Ranking runs.** Traces carry the pointwise log likelihood, so variants can be ranked with LOO.
Compare only runs that observe the same data (`--no-okun` and `--no-phillips` do not), and distrust
a ranking that prefers flexibility: neighbouring quarters share latent states, so LOO
under-penalises a flexible latent path. `--no-okun` and `--no-phillips` are controls: without Okun
`sigma_v` returns its prior; without the Phillips curve the gap result stands.

## The structure imposed on u\*

u\* is in no dataset, so something must say what shapes it may take: `--ustar-structure`.

- **`taper`, the default structure**, for u\* on one series and for s\* under the slack split. A
  driftless random walk whose step size falls linearly from
  `taper_sigma_early` at the start to `taper_sigma_late` at `taper_end`, flat after. No shape,
  only how far u\* may move each quarter.
- **The stepped walk** (`--stepped-walk`). The same two step sizes without the taper: the early
  one to `--stepped-walk-end` (default `STEPPED_WALK_END`, the end of the shaded window), the late
  one after.
- **`spline`.** A natural cubic spline with knots at `--knots`. It fixes where u\* may bend, so its
  path through the 1990s and 2000s is set by the knot, not the data. More knots spend their freedom
  on the early sample, where nothing can arbitrate it.
- **`decay`.** A walk pulled toward one estimated equilibrium. It can only draw a monotone
  approach, so it cannot report a rise.
- **`walk`.** A driftless walk at one imposed `sigma_ustar`; `--ustar-drift` applies only to it.

**Okun makes the step size decisive.** Wherever the walk is free, Okun pulls u\* onto
unemployment: the free part of the gap empties into u\*, and `sigma_v` shrinks. A stepped walk
with the early step of the tapered default tracks unemployment through its whole free period.
A much smaller early step stops that through the 1990s but not through 2002-04, where
unemployment was falling fast. Hence the tight late step, and a taper that tightens by 2002.

**The walk cannot date a break.** With Okun pulling u\* onto unemployment, a step-down in the
walk puts the sharpest fall in u\* wherever the step is placed, so the model cannot tell a
structural fall in u\* from unemployment falling.

**The recent end depends on the late step.** Loosening it lets u\* dip with the tight labour
market of 2022-23, pulling the latest reading down; tightening it holds u\* higher.

---

## The slack measure (`--slack`)

`--slack` sets what the labour-market equations see. The walk's start prior and step sizes follow
it (`WALK_SCALES` in `config.py`), since the star lives on that measure's scale; any `--taper-*`
flag overrides.

- **`both`**, the slack split, the default: described at the top.
- **`unemployment`**, unemployment alone: the single-series model, u\* a walk from N(8, 5) with
  steps 0.175 to 0.03. Every other `--compare` specification runs on it.
- **`underutilisation`**, unemployment plus underemployment, replacing unemployment. u\* becomes
  the underutilisation rate consistent with stable inflation, about twice as high; the walk scales
  automatically, but `sigma_okun` and the Okun prior (`--beta-prior-mu`) should be scaled by hand.

The share walk is kept tight because a looser one only adds quarter-to-quarter wiggle in u\* that
lines up with unemployment's own moves; the split itself does not depend on it.

---

## The hybrid Phillips curve (on by default; `--no-inflation-lag` removes it)

```
pi_q,t = q(a) + lambda·[pi_q,t-1 - q(a)] + (1 - lambda)·beta·excess_t + demand + supply + e_p
```

A hybrid Phillips curve: inflation's baseline puts weight on the target, on trend expectations
(the expectations model's series, entering as its excess over target) and on last quarter's
inflation. The charts call the last of these **backward-looking expectations**; the code calls it
persistence or the inflation lag (`inflation_lag`, `lambda_pi`).

Part of last quarter's deviation from target carries into this one, so inflation above target
stays above it unless slack works it off, and the model can say unemployment must go above u\* to
bring inflation down; whether it does depends on the run. Backward-looking expectations and the
target share one unit of weight, so inflation still settles at target with no slack, no trend
expectations above target and no supply shock. Prior on `lambda_pi`: TruncatedNormal(0.5, 0.3) on
[0, 0.99] (`config.LAMBDA_PI_PRIOR`). `--no-inflation-lag` restores the curve without
backward-looking expectations, where once u reaches u\* inflation is back at target the next
quarter. `--inflation-lag` is still accepted and does nothing extra.

**What the lag stands for.** It is read here as the backward-looking part of expectations, as the
RBA describes the same term in its own Phillips curve models. The data cannot tell that apart from
pricing mechanics, such as indexation and price changes staggered across quarters. In the 12
November tests a four-quarter window fitted worse than a single quarter, and the short-run survey
expectations series did not remove the residual's autocorrelation. Both are more consistent with
mechanics than with beliefs learned slowly from past inflation.

The run reports the unemployment rate that returns **year-ended** inflation to target in 4 and 8
quarters (`results.disinflation_unemployment`). From each quarter's observed inflation, it holds
unemployment at one rate for the horizon, holds the expectations excess where it stands and sets
supply to zero (shocks fade). Underemployment moves with unemployment through the shared gap. It
also prints a sacrifice ratio (point-years above u\* per point of year-ended inflation removed),
the effective weights, the long-run slope `gamma / (1 - lambda)` per draw, and the residual's
autocorrelation, and writes the same block into the run's diagnostics file.

The single lag is the default because it takes up the Phillips residual's autocorrelation. Three
alternatives did no better and remain as options: the mean of the last four quarters
(`--inflation-lag-window`), inflation computed unrounded from the trimmed mean index
(`--pi-source index`), and the short-run expectations series (`--expectations-series short`).

### The decomposition charts

- **"Price inflation decomposition, year-ended"** is the headline. Every quarterly term of the
  fitted equation is summed over four quarters, against the four-quarter sum of the quarterly
  rates the equation fits: inflation target, backward-looking expectations (last quarter's
  inflation, light purple), trend expectations above target (purple), demand, supply and noise.
  The equation and its colour key sit below the axis.
- **"Price inflation decomposition, quarterly annualised"** is the same terms quarter by quarter.
  Without the lag both charts draw without the backward-looking expectations bar.

---

## Comparing specifications (`--compare`)

`--compare` runs this model seven ways and charts the results together. It is not a different
model: each specification is a set of this model's own flags, re-estimated only if its saved run is
not from today. One is the default run itself; the others save to their own `yus_sum_*` prefix.
Every run then writes its own charts, the default to `charts/YStarUStar/` as a plain run does and
the others beside it, named for what sets them apart (`charts/YStarUStar-k2/` and so on), and the
combined charts follow. All seven use the inflation-defined gap and vary the structure imposed on
u\*:

- u\* decays to a level
- u\* is a spline with 1, 2 or 3 knots
- u\* is a tapered random walk
- u\* is a stepped random walk
- **u\* is the slack split** (unemployment and underemployment): the default

Every specification but the default observes unemployment alone (`--slack unemployment`).

All seven share the data, so agreement is close to arithmetic and only disagreement informs.
**Nothing here is a recommendation**: settings argued against elsewhere, decay above all, are in it
because they are the set tested before settling.

**How the fit column is scored.** Leave-one-out over the two equations every run observes,
unemployment and inflation. It ranks predictive accuracy on those and settles nothing about which
model is true.

**What it charts.** Each run's own charts, a table per specification, and six combined charts in
`charts/YStarUStar-compare/`: u\* against unemployment, the output gap, potential growth, potential
output, and the range across specifications for u\* and the gap. Colour is the u\* structure. A
saved run counts as current if its trace was written today.

```bash
./run-ystar-ustar.sh --compare                 # re-estimate anything not from today, chart every run, then combined
./run-ystar-ustar.sh --compare --analyse-only  # the same from the saved runs as they stand
```

---

## Files and usage

```bash
./run-ystar-ustar.sh                                  # the default: the slack split
./run-ystar-ustar.sh --slack unemployment --prefix yus_u             # unemployment alone, tapered walk
./run-ystar-ustar.sh --slack underutilisation --sigma-okun 0.40 \
    --beta-prior-mu 1.0 --beta-prior-sd 1.0 --prefix yus_underutil             # underutilisation
./run-ystar-ustar.sh --no-inflation-lag --prefix yus_nolag            # without backward-looking expectations
./run-ystar-ustar.sh --no-okun --prefix yus_nookun    # control: sigma_v should return its prior
./run-ystar-ustar.sh --no-phillips --prefix yus_nophil                # control: the gap result stands
```

Other switches. All are off by default; the first four are alternatives that were tried and are
not used.

- `--gap-spec identity`: the gap is `y - y*`, deleting `c`, `v`, `sigma_v` and `sigma_e`. Needs
  `--one-sided-beta`, or the chains can find a mirror-image solution.
- `--gap-spec cycle`: the gap is a free AR(1) latent observed by all three equations. On this
  data its persistence goes to a unit root and it becomes a second trend.
- `--okun-form ec`: Okun's law in error-correction form. The two lagged gaps move almost one for
  one, so it does not identify.
- `--anchor-phase step|glide`: the inflation anchor follows expectations through the 1990s
  before settling on the target, instead of being constant.
- `--two-sided-c`: a Normal prior on `c` rather than a half-normal, so its sign is tested rather
  than asserted.

The default prefix charts to `charts/YStarUStar/`; any other prefix charts beside it, to
`charts/YStarUStar-<prefix>/`, so a variant never clears the default's charts. `--chart-dir`
overrides both.

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

Most charts are drawn by `ystar`'s and `ustar`'s own plotting functions through adapters in
`analyse.py`, so axis handling, window shading and band conventions are maintained once. Four are
specific to this model:

- **The gap decomposition** and **the residual pair**, which neither parent can draw.
- **"What inflation alone says u\* is, quarter by quarter"** inverts the Phillips curve each
  quarter and plots it against the fitted u\*, which shows how far the walk is filtering the data.
  The band is uncertainty conditional on the imposed step size, not uncertainty about where u\* is.
- **"The Phillips curve as specified"**, a partial-regression plot of the equation's own regressor
  against inflation net of the other terms, with the short-run and long-run slopes.
