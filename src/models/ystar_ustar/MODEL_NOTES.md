# Joint y\* / u\* — one likelihood, and a gap that is not entirely inflation

`ystar` and `ustar` estimated together, plus one addition. Three states, three observation
equations, nine estimated parameters and two initial conditions, five imposed.

```
STATES
  g_t   = g_{t-1} + e_g                        sigma_g imposed (0.015)
  y*_t  = y*_{t-1} + g_{t-1} + e_y             sigma_ystar imposed (0.078)
  u*_t  = u*_{t-1} + phi·(u*_eq - u*_{t-1})    sigma_ustar imposed (0.020)
          + e_u                                phi, u*_eq estimated

GAP
  gap_t = c·(4·pi_q,t - 2.5) + v_t             v ~ N(0, sigma_v)
                                               (--gap-pi-basis annual uses pi_ann instead)

OBSERVED
  log_gdp_t = y*_t + gap_t + e_c               sigma_e estimated
  u_t       = u*_t - beta·gap_t + e_o          sigma_okun imposed (0.20)
  pi_q,t    = q(2.5) + beta_pi·[q(pi^e_t) - q(2.5)] + gamma·(u_t - u*_t)/u_t
              + rho·d4pm_t + xi·GSCPI_t²·sign(GSCPI_t) + e_p
```

Every prior is taken unchanged from the two parent models, and so are `sigma_ystar`, `sigma_g`
and `sigma_ustar`, so a difference in the posterior is attributable to joint estimation and to
`v` rather than to re-tuning. The one departure is `sigma_okun`, which `ustar` estimates and
this model imposes: see "Why `sigma_okun` is imposed" for the evidence that nothing reported
turns on it.

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
its 0.188 gap *is* the output gap, and `ustar`'s `beta_okun` = 2.03 is a real and
uncomfortable finding rather than an artefact. If `sigma_v` is large, `ystar` has been
reporting a slice of the cycle, `ustar` has been fed that slice, and 2.03 is the slice
showing up as an inflated slope.

**Expect it to be weakly identified.** The moment is a covariance between two large
residuals (`ystar` reports `sigma_e` = 0.508, `ustar` `sigma_okun` = 0.685). The
prior-versus-posterior check in `analyse.py` is there because "the posterior equals the
prior" is a live outcome, and it would mean the joint model failed at the one thing it was
built for.

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

**Three imposed variances still set the answers.** `sigma_ystar`, `sigma_g` and `sigma_ustar`
are all imposed, for the reason all three parents impose them: a free state beside a free
residual is the Stock-Watson pile-up pair, and `ustar` documented all three routes to
estimating its own drift failing. Joining the models does not relieve that and arguably makes
it harder to see, since the three now interact. `ystar`'s `sigma_sweep.py` and `ustar`'s
`sigma_ustar` sweep both remain the honest way to report it, and neither has been run on this
model yet.

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

**`sigma_v` is identified, and about half the output gap is cycle that inflation cannot see.**

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

| Headline | joint | separate |
|---|---|---|
| potential growth, y/y | 1.99 | 1.94 |
| output gap | +0.30 | +0.21 |
| u\* | **4.74** | 4.83 (`ustar`) |
| u − u\* | −0.39 | −0.48 |

### u\* converges, and it changed the answer

u\* used to be a driftless random walk here as in `ustar`, and that prior is about 8 standard
deviations from its own fitted path. Backporting `ustar`'s convergence specification, and with
it `sigma_ustar` = 0.020 in place of 0.040, moved the model materially:

| | driftless, `sigma_ustar` 0.040 | converging, 0.020 |
|---|---|---|
| `sigma_v` | 0.495 | **0.334** |
| sd(gap) | 0.549 | **0.420** |
| free share of the gap | 66.7% | **47.5%** |
| `beta_okun` | 1.343 | 1.265 |
| `gamma_pi` | −1.022 | −1.152 |
| u\* 2026Q2 | 4.69 | **4.74** |

**`sigma_v` fell by a third and the free share from two thirds to under a half.** The reading
is uncomfortable and worth stating plainly: a substantial part of what this model was
attributing to "cycle inflation cannot see" was a **mis-specified u\* trend**. Forced to be a
driftless walk, u\* could not fall as fast as the 1990s required, so the Okun equation needed a
large free gap component to reconcile unemployment with output. Fix u\* and the demand for `v`
drops sharply.

The finding survives, smaller. `sigma_v` = 0.334 still shows 91% prior shrinkage, the
identifying covariance is still −0.58, and the gap is still more than twice `ystar`'s. But the
headline this model was built on was overstated, and the cause was a defect elsewhere in it.

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

**Nothing reported depends on it.** Free against imposed: `c` 0.284 → 0.285, `sigma_v`
0.492 → 0.495, `beta_okun` 1.333 → 1.343, `gamma_pi` −1.021 → −1.022. And doubling the
imposed value to 0.40 moves `c` by +0.005 and `sigma_v` by −0.009; `beta_okun` shifts −0.113,
about half a posterior sd, with `sigma_e` absorbing the rest at 0.404, exactly as the −0.55
correlation predicts.

**The objection, stated rather than buried.** 0.20 is this model's own posterior mean, so
imposing it is circular in a way the package's other imposed variances are not. `ystar`'s
`sigma_ystar` rests on an 8%-of-observed-variation rule; `ustar`'s `sigma_ustar` on the
2012Q4-2015Q4 inflation-band test; both are swept and reported as conditional. This one has no
external anchor. What defends it is the insensitivity, not the value: the honest reading is
that `sigma_okun` is a nuisance parameter the data do not determine and the answers do not
need. `ustar`'s 0.685 is not a candidate, being conditional on a frozen gap that this model
rejects. `--free-sigma-okun` restores the original specification; expect `r_hat` 1.08 and
budget 24,000 draws.

**On the annual gap basis**, for comparison, `c` = 0.376, `sigma_v` = 0.461, `beta_okun` =
1.244 and the free share is 52.3%. The gap itself is unchanged: sd 0.545 against 0.549, and
the same 0.354 correlation with NAB business conditions. **The defined/free split is an
accounting convention, not a result** — it moves with the inflation horizon while the object
it decomposes does not. Do not quote it as a finding.

### `beta_okun` is prior-sensitive in level

`beta_okun`'s `Normal(0.5, 0.5)` prior is inherited from `ustar` and it binds. Doubling its sd:

| | prior sd 0.5 | prior sd 1.0 |
|---|---|---|
| `beta_okun` | 1.331 [1.00, 1.67] | **1.482 [1.00, 1.94]** |
| `sigma_v` | 0.491 | 0.443 |
| `c` | 0.283 | 0.257 |

A move of 0.15, about 0.7 of a posterior sd, and still rising as the prior is released: a prior
mean of 0.5 that this sample rejects at three standard deviations is doing real work.

**The comparison with `ustar` survives**, because 2.033 was estimated under the same prior, so
"joint estimation drops `beta_okun` from 2.03 to 1.34" is like-for-like. **The level does
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

**2. The `--no-phillips` control changes nothing.** `sigma_v` = 0.467, `c` = 0.389,
`beta_okun` = 1.226, all inside the main run's intervals. **The result does not depend on the
Phillips curve being in the likelihood at all**, so it cannot be an artefact of the circularity
that motivated this whole exercise. This is the strongest single piece of evidence here.

**3. The prior does not drive it.** Quadruple the prior mean and the posterior moves 3%:

| prior sd | prior mean | `sigma_v` | `c` | `beta_okun` | sd shrinkage |
|---|---|---|---|---|---|
| 0.5 | 0.399 | 0.447 | 0.375 | 1.248 | 71% |
| **1.0** | 0.798 | **0.458** | 0.379 | 1.233 | 85% |
| 2.0 | 1.596 | 0.461 | 0.381 | 1.230 | 93% |

### What it says about the two parent models

`ystar` has been reporting a slice of the cycle rather than the output gap. Its own residual
`e_c` was carrying roughly as much cyclical variation again as its published gap, and
unemployment can see it. The gap is about three times wider than `ystar` reports.

`ustar`'s `beta_okun` = 2.03 was substantially the consequence of being fed that slice. Given
the whole gap it falls to **1.23**, a 40% reduction toward the textbook 0.4. It does not reach
it: the puzzle is reduced, not resolved, and an Okun coefficient three times textbook still
wants an explanation.

`gamma_pi` barely moves, −1.055 to −0.997, which is consistent with the contamination
diagnosis: the manufactured component was a fifth of the regressor's variance, and diluting it
should shift the coefficient a little rather than overturn it.

### Sampling

The posterior has a ridge, because `sigma_v` and `sigma_e` split one variance. That is the
geometry of the question rather than a defect, and it shows up as low ESS rather than as bias.
`target_accept` 0.99 cut divergences from 22 to 3 while cutting `sigma_e`'s `ess_bulk` from 398
to 140, and moved no estimate by more than 0.005, so the default stays at 0.95 and the answer
is bought with draws instead.

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
```

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

**What it establishes about the specification that works.** `ystar`'s inflation anchor is not
merely a signal about the cycle, it is what makes the trend/cycle split identified. Tying the
gap to an observed series stops it drifting into being a trend. Loosen that to `c·d + v` and
the model estimates cleanly; remove it and no parameterisation rescues the decomposition. The
only thing that would is bounding `rho` well below 1, which imposes the cycle's persistence by
hand and buys nothing over `defined`.

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

The `sigma_ustar` and `sigma_ystar` sweeps matter most. Both parents report their headline as
conditional on an imposed variance, and this model has three of them interacting. Nothing here
has yet established how much of the 0.458 is `sigma_ystar`'s doing.

---

## Files

```
src/models/ystar_ustar/
├── config.py         # ModelConfig: the imposed variances, sigma_v, the switches
├── observations.py   # assembles GDP, both inflation horizons, u, expectations, shocks
├── estimate.py       # builds and samples the PyMC model, saves the trace
├── results.py        # JointResults: the gap decomposition and the residual covariance
├── analyse.py        # charts and the prior-versus-posterior check
└── run.py            # CLI
```

Reuses `ystar`'s `scale_equation` and `potential_output_equation` unchanged rather than
copying them, so the potential block cannot drift away from its parent.

Charts land in `charts/YStarUStar/`, eighteen of them. Sixteen are drawn by calling `ystar`'s
and `ustar`'s own plotting functions through the adapters in `analyse.py`, rather than
reimplementing them: those modules carry quarterly-axis handling, the excluded-window shading,
the off-scale annotation on the gap composition and the band-widening convention on u*, none
of which is worth maintaining twice. The two that are new here are the gap decomposition and
the residual pair, and neither parent can draw either: `ystar` has no free component to
separate out, and `ustar` receives the gap as data.
