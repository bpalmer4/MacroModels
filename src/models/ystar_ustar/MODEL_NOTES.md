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
  pi_q,t    = q(2.5) + beta_pi·[q(pi^e_t) - q(2.5)]
              + gamma·(u_t - u*_t)/u_t + gamma_ue·(ue_t - ue*_t)/ue_t
              + rho·d4pm_t + xi·GSCPI_t²·sign(GSCPI_t) + e_p
```

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
detrended GDP, a filter (λ = 1). Both endpoints already exist as switches, since
`ustar --gap-source actual` is the λ = 1 case.

**Why it can be estimated here.** The gap enters the Okun equation too, as `-beta·gap`. So

```
  cov(GDP residual, unemployment residual) = -beta·Var(v)
```

and the split is identified. That single covariance is the entire informational gain from
joining the two models, and `--no-okun` is the control that demonstrates it: without the
second equation `sigma_v` should return its prior.

**What the answer decides.** If `sigma_v` is near zero, `ystar`'s inflation-defined gap is the
output gap, and `ustar`'s large Okun slope is a real finding. If `sigma_v` is large, `ystar` has
been reporting a slice of the cycle, `ustar` has been fed that slice, and its large Okun slope is
the slice showing up as an inflated coefficient.

**Expect weak identification.** The moment is a covariance between two large residuals, so "the
posterior equals the prior" is a live outcome. The prior-versus-posterior check in `analyse.py`
is there to catch it.

**The model is small on purpose.** `sigma_v` is a diagnostic on `ystar`: how much cyclical
movement the inflation identity misses, given that unemployment can see some of it. It is not a
comprehensive business cycle decomposition and makes no claim about what the missing part is made
of. Freeing correlated disturbances, adding cyclical measurements until the trend/cycle split is
free, or estimating the gap outright are each a different model. The imposed variances are the
price of being small, and the response to them is to sweep and report, not to add structure until
they become estimable: the free-cycle experiment below shows where that ends on this data.

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
drops it from the GDP equation only, as `ystar` does, which makes `c` comparable with `ystar`'s.
The window boundaries have never been tested.

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

## Findings

**`sigma_v` is identified, and the output gap is about twice as wide as `ystar`'s.** That is the
statement to quote, and it holds under both the tapered walk and the slack split. How the gap splits between the
inflation-defined part and `v` is bookkeeping: the split swings across most of its range with
settings the data do not determine (the inflation horizon, `sigma_okun`, the u\* law) while the
gap's width barely moves. Quote the width, never the share.

**The leftover residual correlation is a warning, not corroboration.** `corr(e_c, e_o)` is what
remains after `v` is extracted, and the likelihood assumes it is zero. It stays clearly negative,
the one sign that the restriction is straining. Freeing it is estimable only while `sigma_okun`
stays imposed.

The sweeps below were run when u\* followed earlier laws, a decay toward an equilibrium and then
a one-knot spline, and have not been repeated under the walk. They are conclusions about the
model's structure, not current numbers.

### How the u\* law feeds `sigma_v`

A u\* that cannot fall fast enough through the 1990s forces the Okun equation to explain the gap
between falling unemployment and output with a large free gap, and `sigma_v` inflates. Giving u\*
enough early freedom roughly halved `sigma_v`; after that, what governs the recent path is u\*'s
step size alone. The opposite failure is a u\* free enough to follow unemployment, which empties
`v` into u\*: see "The structure imposed on u\*".

### The wage check

An out-of-sample check: private-sector wage growth regressed on `u − u*` and inflation
expectations. Wages are in no likelihood here and not in the gap's construction.

- **The unemployment gap beats the unemployment rate.** With expectations controlled, the raw
  rate adds nothing; u\* is adding information, not relabelling `u`.
- **It gives the one labour-market slope here that is not circular**, since `gamma_pi`'s regressor
  is partly its own dependent variable and the wage regression's is not.
- **NAIRU-consistent wage growth is the least specification-sensitive number the model makes**,
  because it rests on the constant and the pass-through, not on the gap.
- **It cannot rank `sigma_okun` or the u\* laws.** Its sample starts after the 1990s descent, and
  the differences it finds between specifications have no consistent story.

### The early sample: u\* is not well identified

The u\* charts shade 1993Q1-2003Q4 (`analyse.UNIDENTIFIED_WINDOW`). Through those years u\* is
placed by the structure imposed on it and by where the sample starts more than by the data.

**Okun, not inflation, places the early level.** Inflation sat near target through the mid-1990s,
so the inflation-defined gap has little to work with and Okun holds u\* close to unemployment. A
move in u\* costs the Okun equation several times what it costs the Phillips curve, so nothing on
the nominal side can outvote it. The model also over-predicts inflation over 1996-98, which says
u\* sits too high there.

**A phased inflation anchor was tried and failed** (`--anchor-phase`, default `none`). Tracking
expectations until they reached the target should have removed that bias. It made it worse,
because the expectations weight rose and pushed fitted inflation up. The early level is not an
anchor problem.

**Why the window runs to 2003.** Unemployment fell fast again over 2002-04 while inflation eased,
and fell fast around 1995 while inflation rose. Both ask u\* to move further and faster than a
tight walk allows, and a loose one lets Okun pull u\* onto unemployment instead: see "The
structure imposed on u\*".

### Why `sigma_okun` is imposed

Free, `sigma_okun` sits on a ridge with `sigma_e` that the sampler cannot mix along: a handful of
effective draws, chains peaking in different places, many divergences. Imposing it dissolves the
ridge and every other parameter samples cleanly.

**The answer depends on it.** At small values `sigma_v` is flat. Loosen it and the free component
shrinks, until at large values it vanishes, `c` returns to `ystar`'s value and **the model
degenerates into `ystar`**. So `sigma_okun` is the finding restated: if unemployment tracks the gap
tightly, a large part of the cycle is invisible to inflation; if it does not, there is nothing for
`v` to find.

**The data stay off the cliff.** Run free (`--free-sigma-okun`), the badly mixed chains all keep
`sigma_okun` in the region where `sigma_v` is flat and never wander toward the collapse. That is
the defence of the imposed value, and it is weaker than an external anchor would be: the value is
close to this model's own posterior, and `ustar`'s larger estimate would shrink the finding by
about a third without destroying it.

**The annual gap basis** was last run under a driftless u\*, and those figures are stale. What it
showed still stands: the gap's width is unchanged, and only the defined/free split moves.

### `sigma_ystar` does not move the gap

Across `ystar`'s own grid, from zero to three times the default, `sigma_v` and the gap's width
barely move. A wobblier potential competes with the GDP residual, not the gap, because the gap is
tied to an observed series. What does move is **potential growth**, by about a third of a point
across the grid, a sensitivity `ystar`'s notes carry too. `sigma_g` has not been swept.

### `beta_okun` is prior-sensitive in level

The inherited `Normal(0.5, 0.5)` prior binds: doubling its sd raises `beta_okun` by most of a
posterior sd, and it is still rising. The comparison with `ustar` survives, because both use the
same prior, and joint estimation cuts the Okun slope substantially. The level does not: it is well
above the textbook value and partly the prior's. `--beta-prior-sd` runs the alternative.

### The three things that make this believable

**1. Drop Okun and it fails, as predicted.** `--no-okun` removes the identifying moment: `sigma_v`
becomes a flat ridge the sampler cannot explore, and `c` returns to `ystar`'s value. The covariance
is doing the work.

**2. Drop the Phillips curve and nothing changes.** `--no-phillips` reproduces the gap result, so
it cannot be an artefact of the circularity. This is the strongest single piece of evidence here.

**3. The prior does not drive it.** A fourfold change in the `sigma_v` prior barely moves the
posterior.

### What it says about the two parent models

`ystar` reports a slice of the cycle: its GDP residual carries roughly as much cyclical variation
again as its published gap, and unemployment can see it. `ustar`'s large Okun slope is largely the
result of being fed that slice; given the whole gap it falls substantially, though it stays well
above textbook. `gamma_pi` barely moves between the two models, which says little on its own,
since it moves a lot across u\* laws.

### Why the Phillips curve stays

`--no-phillips` is a control, not a candidate default. Prices are one of the three legs the model
reconciles, and the Phillips curve is where the inflation decomposition (anchor, expectations,
demand, supply) comes from. That `sigma_v` survives without it shows the gap result is not an
artefact; it does not show the equation is surplus.

**Two caveats.** `gamma_pi` is not a Phillips slope: its regressor is partly its own dependent
variable, and it moves a lot across u\* laws. And through Okun the demand term carries part of
inflation on the right-hand side, so the supply coefficients `rho_pi` and `xi_gscpi` are likely
scaled up, and the supply contribution overstated. That is arithmetic on the coefficients, not a
measured bias. Comparing the supply coefficients across the quarterly and annual gap bases would
measure it; that has not been done.

### Sampling

The posterior has a ridge, because `sigma_v` and `sigma_e` split one variance. That is the geometry
of the question, and it shows up as lower ESS rather than bias; the answer is bought with draws,
not a higher `target_accept`.

Traces carry the pointwise log likelihood, so variants can be ranked with LOO. Compare only runs
that observe the same data (`--no-okun` and `--no-phillips` do not), and distrust a ranking that
prefers flexibility: neighbouring quarters share latent states, so LOO under-penalises a flexible
latent path.

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

**The walk cannot date a break.** Sweeping the stepped walk's step-down date from 2002 to 2006
moves the sharpest fall in u\* with it, each time to the quarter before the step. The level it
settles at is roughly where unemployment was at the step. So the joint model cannot tell a
structural fall in u\* around 2004 from unemployment falling. A single Phillips curve, with no
Okun equation pulling u\* onto unemployment, is the better place to test for one.

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

**What the slack split shows.** The fall in u\* over 2002-06 is roughly half a fall in total slack
and half slack moving into underemployment, which rose over 2002-03. After 2008 total slack is
roughly flat while unemployment's share falls: the post-GFC decline in the unemployment NAIRU is
largely rising underemployment. The split does not depend on the share walk's step size: swept
over a fivefold range, total slack and the share move the same. A looser share only adds
quarter-to-quarter wiggle in u\* that lines up with unemployment's own moves, so the tightest
setting is used. At that setting underemployment slack probably moves inflation too, though the
evidence is modest.

---

## The gap definition, and what the identity version showed

`--gap-spec identity` replaces `gap = c·(pi - 2.5) + v` with `gap = y - y*`, deleting `c`, `v`,
`sigma_v` and `sigma_e` and estimating `sigma_okun` instead of imposing it.

**What it wins.** Far better sampling, no imposed `sigma_okun`, and a smaller early bias against
what the Phillips curve alone implies for u\*.

**What it loses.** A deep early-1990s output gap and a positive current one, and a gap too
volatile to be a recognisable business cycle.

**The mirror trap.** Okun then sees only `beta·gap`, so `(beta, y*)` and `(-beta, y*` reflected
through `y)` fit it identically, and a loose trend prior lets chains find the mirror.
`--one-sided-beta` closes it by asserting Okun's law has the expected sign.

The defined gap carries the largest early residual bias of the options tried; that is its price.

---

## Explored and did not work: an error-correction Okun

`--okun-form ec` replaces the level relation with

```
  du_t = -kappa·(u - u*)_{t-1} - theta·gap_{t-1} - gamma·d(gap)_t + e_o
```

and reports the long-run slope as `beta = theta/kappa`: Okun's original statement, with an
error-correction term so u\*'s level is still identified. **It does not identify on this data.**
The lagged unemployment gap and the lagged output gap move together almost one for one, which is
Okun's law itself, so the form asks the data to split one movement two ways and the chains sit at
different points on a flat ridge. No reparameterisation fixed it. The level form is the same
long-run relation with the adjustment speed set to one, the only version this sample supports.

---

## Explored and did not work: a free cycle instead of the inflation anchor

`--gap-spec cycle` replaces the inflation-anchored gap with a free AR(1) latent that all three
equations observe. It would remove the circularity outright, and `v` does come back persistent
despite an iid prior. **It collapses.** The cycle's persistence goes to a unit root, so the gap
becomes a second trend competing with `y*`, absorbs the unemployment cycle, and leaves both slopes
straddling zero: a state that fits everything. Fixing the amplitude prior did not help.

So **some** observed anchor is needed on this observable set. That does not show inflation is the
only series that could serve; nothing here tests another. The switch is kept so the test is
repeatable, not as a candidate.

---

### Still to run

- **Re-run the sweeps under the current default**: `sigma_okun`, `sigma_ystar`, the `sigma_v` prior,
  `--no-okun` and `--no-phillips`. Their conclusions were established under earlier u\* laws.
- **`sigma_g`**, the one state variance never swept here.
- **The supply-term check**: `rho_pi` and `xi_gscpi` on the quarterly and annual gap bases.
- **`--exclude-scope gdp`** and **`--two-sided-c`**: is `c` comparable with `ystar`'s, and is its
  sign a finding or a prior?

---

## Inflation persistence (`--inflation-lag`)

```
pi_q,t = q(a) + lambda·[pi_q,t-1 - q(a)] + (1 - lambda)·beta·excess_t + demand + supply + e_p
```

Off by default. Without it the Phillips curve has no persistence: once u reaches u\*, inflation
is back at target the next quarter, so the model can never say unemployment must go *above* u\*
to bring inflation down. The lag and the anchor share one unit of weight, so inflation still
settles at target with no slack, excess or supply. Prior on `lambda_pi`: TruncatedNormal(0.5,
0.3) on [0, 0.99] (`config.LAMBDA_PI_PRIOR`).

With it on, the run also charts the unemployment rate that returns **year-ended** inflation to
target in 8 quarters (`results.disinflation_unemployment`). From each quarter's observed
inflation, it holds unemployment at one rate for the horizon, holds the expectations excess where
it stands and sets supply to zero (shocks fade). Underemployment moves with unemployment through
the shared gap. The headline also prints 4 quarters and a sacrifice ratio (point-years above u\*
per point of year-ended inflation removed).

**First run** (`--inflation-lag --prefix ystar_ustar_lag`, 2026-10-27, sample to 2026Q2, slack
split, defined gap). Clean: r_hat 1.00 throughout.

- `lambda_pi` 0.30 (90% 0.17 to 0.43), moved well off its 0.5 prior. The Phillips residual's
  lag-1 autocorrelation falls from 0.23 to -0.08, so the lag takes up what was left.
- `gamma_pi` weakens to -0.64 (90% -1.22 to +0.02) from -1.0 without the lag.
- Year-ended inflation 3.55, +1.05 on target. Over the last four quarters: persistence 0.24,
  demand 0.53, residual 0.24, supply 0.02, excess 0.00 (pp, quarterly terms summed).
- U for target in 4 quarters 4.79 (+0.06 over u\* 4.73); in 8 quarters 4.73 (+0.00).
  Unemployment is 4.35.

So persistence is real but short: a quarterly lambda of 0.3 halves a deviation in about 0.6 of a
quarter, and leaves under 1% of it after four. The model puts the overshoot mostly on the labour market being tight (u 0.38 below u\*),
and its answer is to close that gap, not overshoot it.

**Read that with the circularity in mind.** Under the defined gap, inflation builds the gap
(`c·(pi - anchor)`) and the gap sets u - u\* through Okun, so inflation that stays high is
mechanically booked as a tight labour market, and the demand term can take persistence that the
lag would otherwise carry. The test is the same lag on a gap inflation does not define:
`--gap-spec identity --one-sided-beta --inflation-lag`. Not yet run.

---

## Comparing specifications (`--compare`)

`--compare` runs this model eleven ways and charts the results together. It is not a different
model: each specification is a set of this model's own flags, re-estimated only if its saved run is
not from today. One is the default run itself; the others save to their own `yus_sum_*` prefix.
Every run then writes its own charts, the default to `charts/YStarUStar/` as a plain run does and
the others beside it, named for what sets them apart (`charts/YStarUStar-k2/`,
`charts/YStarUStar-decay_id/` and so on), and the combined charts follow. Eight cross the two
choices the model actually has to make, and two random walks are added:

| | inflation-defined gap | gap = y - y\* |
|---|---|---|
| u\* decays to a level | x | x |
| u\* is a spline, 1 knot | x | x |
| u\* is a spline, 2 knots | x | x |
| u\* is a spline, 3 knots | x | x |
| u\* is a tapered random walk | x | |
| u\* is a stepped random walk | x | |
| **u\* is the slack split** (unemployment and underemployment) | **x** (the default) | |

Every row but the default observes unemployment alone (`--slack unemployment`).

Down a column is how much the u\* structure matters; across a row, how much the gap definition
matters, on unemployment alone. **Crossed rather than laddered**, so the two cannot be confounded. The walks run on the
inflation-defined gap only. All eleven share the data, so agreement is close to arithmetic and only
disagreement informs. **Nothing here is a recommendation**: settings argued against elsewhere,
decay above all, are in it because they are the set tested before settling.

Every identity-gap specification runs `--one-sided-beta` to close the mirror trap. With no GDP
residual, national accounts noise then lands in the gap, attenuating `beta_okun` and widening the
gap several times over.

**How the fit column is scored.** The identity gap gives the GDP equation no likelihood, so the
score is leave-one-out over the two equations every run observes, unemployment and inflation. It
ranks predictive accuracy on those and settles nothing about which model is true.

**What it found.**

- **The fit column does not separate them**: the spread is within its standard errors.
- **Pareto k is the real signal.** The inflation-defined runs have many unreliable observations
  and the identity runs almost none: the circularity showing as a diagnostic, since inflation sits
  on both sides of the Phillips curve.
- **For the endpoint the gap definition matters and the u\* structure barely does.** Potential
  growth is untouched by either.
- **Every specification reads u\* above what the Phillips curve alone implies in the 1990s**, and
  none can place 1993Q1: the recession and disinflation before it are outside the sample.

**What it charts.** Each run's own charts, a table per specification, and six combined charts in
`charts/YStarUStar-compare/`: u\* against unemployment, the output gap, potential growth, potential
output, and the range across specifications for u\* and the gap. Colour is the u\* structure,
dashing the gap definition. A saved run counts as current if its trace was written today.

```bash
./run-ystar-ustar.sh --compare                 # re-estimate anything not from today, chart every run, then combined
./run-ystar-ustar.sh --compare --analyse-only  # the same from the saved runs as they stand
```

---

## Files and usage

```bash
./run-ystar-ustar.sh                                  # the default: the slack split
./run-ystar-ustar.sh --share-sigma-late 0.05 --prefix yus_share005   # a looser share walk
./run-ystar-ustar.sh --slack unemployment --prefix yus_u             # unemployment alone, tapered walk
./run-ystar-ustar.sh --slack unemployment --stepped-walk --prefix yus_sw   # the stepped walk
./run-ystar-ustar.sh --slack unemployment --stepped-walk --stepped-walk-end 2004Q4 --prefix yus_sw2005
./run-ystar-ustar.sh --slack underutilisation --sigma-okun 0.40 \
    --beta-prior-mu 1.0 --beta-prior-sd 1.0 --prefix yus_underutil             # underutilisation
./run-ystar-ustar.sh --no-okun --prefix yus_nookun    # the control that must fail
./run-ystar-ustar.sh --no-phillips --prefix yus_nophil
./run-ystar-ustar.sh --free-sigma-okun --prefix yus_freeso
./run-ystar-ustar.sh --sigma-okun 0.40 --prefix yus_so40
./run-ystar-ustar.sh --beta-prior-sd 1.0 --prefix yus_bwide
./run-ystar-ustar.sh --ratio-ystar 0.25 --prefix yus_ry025
./run-ystar-ustar.sh --anchor-phase step --prefix yus_anchor_step   # the phased anchor that failed
```

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
  quarter and plots it against the fitted u\*. The implied series is far noisier but tracks the
  fitted path, so the walk is filtering, not overriding. It also shows that the band is uncertainty
  conditional on the imposed step size, not uncertainty about where u\* is.
- **"The Phillips curve as specified"**, a partial-regression plot of the equation's own regressor
  against inflation net of the other terms. The slope is identified almost entirely by the tight
  side, in two episodes, the 2000s and since 2020; the slack side is a formless cloud.
