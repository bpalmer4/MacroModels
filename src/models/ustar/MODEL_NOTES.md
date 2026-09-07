# u* — a NAIRU from a given output gap

A Bayesian unobserved-components model (PyMC + NumPyro NUTS) estimating the Australian
unemployment rate consistent with output at potential and inflation at target. One latent
state, two observation equations, nine estimated parameters.

```
u*_t = u*_{t-1} + phi·(u*_eq - u*_t-1) + e_u   u*: converges, sigma imposed
u_t  = u*_t - beta·ygap_t + e_o                 Okun:     unemployment fitted
pi_t = q(2.5) + beta_pi·[q(pi^e_t) - q(2.5)]    Phillips: inflation fitted
       + gamma·(u_t - u*_t)/u_t
       + rho·d4pm_t + xi·GSCPI_t²·sign(GSCPI_t) + e_p
```

`q(·)` converts an annual rate to a quarterly one. The output gap `ygap` is **not estimated
here**: it is read from a completed `ystar` run. Inflation expectations come from the
`expectations` model. Run order is `expectations` → `ystar` → `ustar`, all from saved
output, nothing re-estimated.

Estimated: `phi_ustar`, `ustar_eq`, `beta_okun`, `sigma_okun`, `gamma_pi`, `beta_pi`,
`rho_pi`, `xi_gscpi`, `epsilon_pi`. Imposed: `sigma_ustar` = 0.020. Asserted: the 2.5% target,
flat.

**u\* converges rather than wandering, and that is new.** It used to be a driftless random
walk, which turned out to be about 8 standard deviations from its own fitted path and 17 over
1993-1999. See "The driftless random walk is the wrong prior"; `--no-ustar-converge` restores
the old specification.

---

## Read this first: what the model determines, and what you determine

**A vintage warning before anything else.** `ystar` now drops 2020Q2-2021Q3 from its
likelihood, and its `c` fell from 0.468 to 0.188 as a result. The gap this model reads is
therefore a different series: sd 0.188 against 0.467, and +0.21 at 2026Q2 against +0.51. The
default run has been re-estimated on it. **One thing has not been
re-run and is one vintage old**: the free- and bounded-drift experiments in "Three ways of
trying to estimate the drift". Their conclusion is not in doubt — the pile-up is a property of
a free state sitting beside a free residual, not of the gap — but their levels are
pre-exclusion. The `--no-output-gap` control needs no re-run at all, for the reason given in
that section.

**The level of u\* is set by numbers you choose, not by the data.** That was always true of
`sigma_ustar`, which is imposed and cannot be estimated (see "Three ways of trying to estimate
the drift"). It is more true now: u\* converges to an estimated equilibrium, and the chart
`what-moves-u-the-specification-or-the-data` shows that of u\*'s 6.1pp fall across the sample,
about 97% is the convergence mechanism and 3% is the data. The 1990s are essentially the
specification drawing a curve.

**`sigma_ustar` remains the second choice, and it is not swept here.** It decides how far u\*
may wander from its equilibrium path, and at 0.020 the answer is: not far, deviations of +0.16
and −0.09 at the extremes. A sweep used to sit here, showing u\* running 5.04 to 4.58 across
0.024 to 0.065 with `gamma_pi` doubling against it. It has been removed rather than annotated,
because every row was estimated on the driftless random walk and none of it survives the
change of specification: the mechanism now carries the 1990s, so the sweep would be measuring a
different and much narrower thing. Re-running it is outstanding work.

The trade-off it demonstrated is structural and has not gone away. `sigma_ustar` decides how
much of unemployment's movement is trend and how much is cycle, and `gamma_pi` takes whatever
is left: tightening from 0.040 to 0.020 flattened `gamma_pi` from −1.48 to −1.15 for exactly
that reason. What the removed table cannot now tell you is the *size* of that trade under
convergence.

So: **this model does not have a robust headline.** Quote "u\* near 4.8, and the labour market
tight by roughly half a point". Do not quote decimals.

**What the model does, stated plainly:** it splits the unemployment rate into a slow trend and
a cycle, where how much is trend you fix in advance, and the Phillips curve sets the amplitude
of the cycle by asking how large a gap is needed to explain inflation's distance from target.

---

## Results (2026Q2 vintage, `sigma_ustar` = 0.020)

Converged: all `r_hat` = 1.00, `ess_bulk` 3,545 to 9,600, no divergences. Estimated on the
current `ystar` gap, the one with 2020Q2-2021Q3 out of the likelihood.

Converged: `r_hat` = 1.00 throughout, `ess_bulk` 8,022 to 13,049, no divergences.

| Parameter | mean | 90% HDI | |
|---|---|---|---|
| `phi_ustar` | 0.040 | [0.03, 0.04] | ~4% of the distance to equilibrium closed per quarter |
| `ustar_eq` | 4.86 | [4.71, 5.01] | the equilibrium being converged on |
| `beta_okun` | 2.144 | [1.74, 2.55] | P(> 0) = 100% |
| `sigma_okun` | 0.486 | [0.43, 0.55] | |
| `gamma_pi` | −1.150 | [−1.39, −0.91] | P(< 0) = 100% |
| `beta_pi` | 0.391 | [0.12, 0.65] | de-anchoring pass-through |
| `rho_pi` | 0.006 | [0.001, 0.011] | import prices |
| `xi_gscpi` | 0.040 | [0.028, 0.052] | supply chains |
| `epsilon_pi` | 0.164 | [0.145, 0.183] | |

| Headline, 2026Q2 | |
|---|---|
| u\* | **4.83** |
| u\*'s own equilibrium | 4.86 |
| u − u\* | **−0.48** |

**Two specification changes separate this from the 2.033 / 0.685 / −1.055 / u\* 4.71 that this
section used to report.** u\* now converges rather than wandering, which is what moved the
1990s (see "The driftless random walk is the wrong prior"). And `sigma_ustar` fell from 0.040
to 0.020, which is what moved the endpoint: u\* no longer chases post-2022 unemployment down,
so it finishes at 4.83 just below its estimated equilibrium of 4.86 rather than at 4.54 well
below it. The two are related — the old 0.040 was calibrated for a driftless walk in which the
innovation had to carry the whole decline — and the arithmetic is in `ModelConfig.sigma_ustar`.

`beta_okun` = 2.03 is far above a textbook Okun coefficient and should not be read as one. The
`ystar` gap is a shrunk regressor — `c` is a conditional mean on a signal explaining about an
eighth of output's variation — so the slope compensates. It is a scaling onto this particular
gap series, and the surest sign of that is what happened when the series changed: `c` fell
0.468 to 0.188 and `beta_okun` went 1.065 to 2.033 to sit on the same unemployment data.

**But it is not a clean rescale, and the residual is the informative part.** The gap shrank by
a factor of 2.48 while the slope rose by only 1.91, so the fitted cyclical amplitude
`beta_okun × sd(ygap)` fell from 0.497 to 0.382. The new gap explains less of the swing in
unemployment, not the same amount in smaller units. `sigma_okun` nonetheless barely moved,
0.684 to 0.685, because u\* took up the slack. That is the trade this model always makes: the
same imposed `sigma_ustar` buys a more mobile u\* when the gap explains less.

Both of those numbers are from the driftless specification and are kept because the *argument*
is what matters here, not the levels. On the current specification the same quantities are
`beta_okun` = 2.144 and `sigma_okun` = 0.486.

Two things to keep in view when reading `beta_okun` = 2.03. Its prior is `Normal(0.5, 0.5)`,
so the posterior now sits three prior standard deviations out, where before it was one; the
data are pulling hard and the prior is not restraining the number. And whatever the gap's
units, `beta_okun` remains an artefact of them, so it is not comparable with `nairu`'s Okun
coefficient or with anyone else's.

---

## Three ways of trying to estimate the drift, and why all three fail

The likelihood has a **monotone preference for more state variance**: a u\* that tracks
unemployment fits better quarter by quarter than one that doesn't. That single fact defeats
every route.

Routes 1 and 2 below are pre-exclusion runs and the numbers in them are one vintage old. The
result is not at risk from that: the preference is a property of a free state sitting beside a
free residual, and the gap enters the Okun equation as a regressor rather than as anything
that constrains `sigma_ustar`. A smaller gap makes u\* *more* attractive as an explanation of
ΔU, not less, so if anything the current gap sharpens the pile-up.

**1. Free prior.** `TruncatedNormal(0.03, 0.012, lower=0.005)`, which spans the whole
defensible range. Posterior came back at **0.131 [0.115, 0.147]** — 8.4 prior standard
deviations above the mean, in a region where the prior density is effectively zero. u\*
collapsed to 4.63 with a gap of −0.28, `sd(du*)` at 98% of what the prior allowed, and
correlation with unemployment of 0.981. The model stops estimating a structural rate and
returns a filter of its input.

**2. Bounded prior.** Same, truncated above at 0.05. Posterior **0.049 [0.049, 0.050]**, sd
0.001, with 100% of draws above 0.045 and 62% within 1% of the bound. It reproduces the
fixed-at-the-bound model exactly, and — contrary to what one might hope — does *not* widen the
u\* band, because the posterior collapses onto the boundary with nothing left to integrate
over. The bound becomes the specification.

**3. Fixed and swept.** The table above.

`ystar` can free `sigma_ystar` because in its `inflation` spec potential is a residual,
so once `c` and `g` are known the innovation is directly observable and its standard deviation
is an ordinary estimation problem. That does not hold here: u\* is a free state sitting next to
a free residual `e_o`, which is exactly the Stock-Watson pile-up pair.

**The conclusion is not a defect, it is the operational meaning of the prior.** "u\* is slow
moving" cannot be expressed as a belief the data are permitted to revise, because the data will
revise it away every time. It has to be imposed — which is what `ystar` means when it
says `sigma_ystar` is pinned.

Reproduce with `--free-sigma-ustar`. The switch is kept for that reason, in the same spirit as
`ystar`'s `free_sigma_ystar`: to make the check repeatable, not because it is a
candidate.

### Why 0.040

Three readings, and it sits between them.

- `ystar`'s rule — a trend innovation sd around 8% of the observed variation in the
  series it trends — gives 8% of `sd(du)` = 0.300, so **0.024**.
- The `nairu` model's *realised* `sd(dNAIRU)` is 0.032, which is 11% of `sd(du)`. It imposes
  0.15 but only uses a fifth of it, because its other six equations bind.
- The ceiling comes from the inflation-band chart. Across 2012Q4-2015Q4 — when inflation sat
  **below** the RBA band, signalling genuine slack — u\* should not be rising. It changes by
  −0.14, −0.13, −0.07, **+0.01** and **+0.11** across the swept settings, so the sign flips
  between 0.040 and 0.050 and everything at 0.050 or looser books part of the
  post-mining-boom rise in unemployment as structural. `beta_pi` also falls monotonically,
  0.80 to 0.35, as a freer u\* crowds out the de-anchoring term.

  **The re-run tightened this.** On the pre-exclusion sweep the flip fell between 0.050 and
  0.065, with 0.050 reading as flat at −0.01; it is now on the wrong side of zero. 0.040 is
  therefore closer to the ceiling than it was, and is the loosest setting that still passes
  the test rather than one of two. That is an argument for 0.040 over 0.050, not an argument
  for moving the choice down: 0.030 and 0.024 pass the same test and fail the others.

None of these is external to this repo. **The outstanding improvement is to calibrate the drift
against a published NAIRU series** — RBA, Treasury, OECD — whose realised drift is an
observable rather than a modelling choice. That is the one change that would resolve the
assumption rather than relocate it.

---

## The driftless random walk is the wrong prior, and two ways to fix it

**Fix 2 is now the default.** Fix 1 is a switch. The evidence for the diagnosis is stronger
than the evidence for either cure, and the external check at the end of this section does not
favour either, so read this as a defensible choice rather than a settled one.

### The diagnosis

u\* is specified as a **driftless** random walk. The fitted path is not. It runs 8.43 to 4.71,
a fall of 3.72pp over 134 quarters, where a driftless walk at `sigma_ustar` = 0.040 puts the sd
of the total change at 0.46pp. The fitted path is about **8 standard deviations** out. The
per-quarter increments use 92% of the allowed movement and three quarters of that is directed
rather than random. The prior is not stretched, it is overwhelmed.

Split by era it is worse at the front:

| | u\* change | vs a driftless walk |
|---|---|---|
| 1993-1999 | −3.66pp over 28 quarters | **17.3 sd** |
| 2000-2026 | −1.98pp over 106 quarters | **4.8 sd** |

**The visible cost is Limitation 4.** To start where unemployment actually was in 1993Q1, at
10.93, and still reach 4.71 needs 6.2pp, which the prior cannot afford. So the posterior
compromises by starting u\* at 8.43 and booking the remaining +2.5pp in the Okun residual.

**And that reading contradicts the inflation data.** With no drift, u\* sits below unemployment
in all 16 quarters of 1994-1997, so the model says slack throughout — while the annual trimmed
mean broke above 3% in late 1995. A Phillips curve cannot explain an inflation breakout out of
slack. The summary statistic over 1993-1999:

| corr(u − u\*, inflation four quarters ahead) | |
|---|---|
| driftless | **−0.105** |
| with either fix | about **−0.81** |

A Phillips curve wants that negative. Without a fix the early sample carries essentially no
Phillips information at all.

### Fix 1: `--ustar-drift`, a drift on excess expectations

    u*_t = u*_{t-1} - lambda·max(0, pi^e_{t-1} - 2.5)·1{t < 2000Q1} + e_u

The story is that the target was not yet believed, so wage-setting had not adapted to it.
Expectations ran +0.64 above target across 1994-1996 against +0.07 across 2000-2019, even
though realised trimmed mean inflation was already 2.1% in 1993Q1: credibility and realised
inflation are different things and only the first should move u\*.

`lambda` = 0.250 [0.22, 0.28], and it is data-determined rather than prior-driven: widening its
prior tenfold moves it to 0.260 and stops.

Two objections. The **2000 cutoff is asserted**, and it is needed because the level of excess
expectations cannot tell a credibility transition from a supply shock — it reads +0.87 in
2022-23 too, and ungated the drift pushed u\* to 4.25 and flipped the current gap from −0.36 to
+0.10. And **the gap becomes a partial proxy for expectations**: corr(u − u\*, excess
expectations) goes +0.048 to −0.405 over 1993-99, while `beta_pi` falls 0.578 to 0.378, so the
gap takes over part of the Phillips curve's own expectations term.

### Fix 2: `--ustar-converge`, convergence to a new equilibrium

    u*_t = u*_{t-1} + phi·(u*_eq - u*_{t-1}) + e_u

The story is that u\* was moving to a new equilibrium as the economy left high inflation, and
decelerated as it arrived. `phi` = 0.039 [0.03, 0.04], so about 4% of the remaining distance
closes each quarter, a half-life near four and a half years. `u*_eq` = 4.79 [4.56, 5.02],
estimated rather than asserted.

**This is the default.** It needs no cutoff date, because the process stops by arriving. u\*
depends on nothing but its own past, so the gap cannot inherit an expectations signal. And it
fits best.

| | driftless | drift | converge |
|---|---|---|---|
| `sigma_okun` | 0.685 | 0.466 | **0.426** |
| `beta_okun` | 2.033 | 2.088 | 1.996 |
| `gamma_pi` | −1.055 | −1.284 | −1.349 |
| u\* 1993Q1 | 8.43 | 10.34 | **10.76** |
| u\* 2026Q2 | 4.71 | 4.66 | 4.68 |
| divergences | 0 | 0 | 0 |
| min `ess_bulk` | 3,545 | 5,094 | **5,127** |

The Okun residual variance falls 61% against the driftless version, and the endpoint is
untouched, so the fix is confined to the era it is about.

### What to be wary of in both

**The early credible band is narrow, and that is the mechanism, not the data.** Through
1993-96 the band is tighter than in the middle of the sample, which is backwards for a state
estimated from a short run of observations. It is `phi` and `u*_eq` (or `lambda`) pinning the
level. Do not read that band as confidence in a NAIRU near 10.8 in 1993.

**`u*_eq` = 4.79 sits just above the current u\* of 4.68**, so the model says the secular
decline has essentially finished. That is a claim about today, arrived at from a specification
fitted to explain the 1990s, and nothing here tests it.

**The pandemic is not excluded and not marked.** `ustar` fits 2020-21 like any other quarters,
so u\* glides through a period when JobKeeper held measured unemployment far below any
reasonable reading of slack. The joint `ystar_ustar` model drops those quarters from all three
of its equations; this one does not.

### The one external check

Wages are not in this model's likelihood, so they are a genuine out-of-sample test of whether
the gap identifies labour-market tightness at all. A tight market should mean faster wages, so
every correlation should be negative.

| | same quarter | wages 4q ahead |
|---|---|---|
| WPI, 1997Q4-2026Q2, driftless | **−0.523** | −0.426 |
| WPI, converge | −0.485 | −0.380 |
| WPI, raw unemployment (no u\*) | −0.243 | |
| Hourly COE, 1993Q1-2026Q2, driftless | −0.205 | −0.338 |
| Hourly COE, converge | −0.059 | −0.310 |

**The gap passes**: against WPI it roughly doubles what raw unemployment achieves, so
subtracting u\* adds real information. **But the fix does not improve it, and slightly worsens
it on all four measurements**, most visibly on hourly compensation over the full sample where
the same-quarter correlation falls from −0.205 to −0.059.

That is the counterweight to everything above, and it is not resolved. The driftless prior is
demonstrably wrong about the 1990s: 17 sd, slack throughout an inflation breakout, a +2.5pp gap
after a recession. Convergence fixes all of that and fits and samples better. It has not been
shown to make the gap a better measure of labour-market tightness, and on the longer wage
series it is worse. Both statements are true and they are in tension. The specification was
adopted on the first; anyone relying on the gap for tightness should know about the second.

---

## Does the output gap actually matter?

The package was built on the premise that a credible output gap from `ystar` is what
makes a two-equation u\* possible. That is testable. Zeroing the gap while keeping the Okun
equation's structure — so `u = u* + e_o` still fits a trend through unemployment — isolates
the gap's contribution (`--no-output-gap`).

| | with gap (pre-exclusion) | with gap (current) | gap zeroed |
|---|---|---|---|
| u\* 2026Q2 | 4.78 | **4.71** | 4.60 |
| gap 2026Q2 | −0.43 | **−0.36** | −0.25 |
| `sigma_okun` | 0.684 | **0.685** | 0.908 |
| `beta_okun` | 1.065 | **2.033** | 0.499 |
| `gamma_pi` | −1.020 | **−1.055** | −1.032 |
| `epsilon_pi` | 0.160 | **0.160** | 0.159 |

**The gap-zeroed column needs no vintage caveat.** `use_output_gap = False` substitutes an
array of zeros for the gap, so that run reads nothing from `ystar` and the respecification
cannot have moved it. Only the with-gap side changed, which is why the comparison can be
re-struck here without re-running the control. The path statistics below the table are the
pre-exclusion pairing and have not been recomputed.

u\* path correlation **0.9973**, mean absolute difference **0.10pp**, max 0.38pp — and the max
is in 1993, the least identified end of the sample.

**The gap does real work in the Okun equation**: `sigma_okun` rises 0.685 → 0.908 without it, a
43% reduction in residual variance, and the current gap buys the same reduction the old one
did. `beta_okun` collapsing to 0.499, its prior mean, confirms the test removed what it was
meant to.

**But it barely moves u\*, and now less than before.** 0.11pp at the endpoint on the current
gap, against 0.18pp pre-exclusion and 0.48pp from choosing `sigma_ustar` within its defensible
range. And the Phillips side is untouched — `gamma_pi` −1.055 against −1.032, `epsilon_pi`
0.160 against 0.159 — so the two channels are not sharing identification.

**Verdict, and the respecification pushes it further.** The premise is partly vindicated and
partly not. The gap explains the cyclical component of unemployment well, and it moves the
headline gap from −0.25 to −0.36, which is not decorative. But it is not what makes the
two-equation u\* possible: the Phillips curve and the imposed drift do that, and they would do
it nearly as well with the gap set to zero. The gap is a tilt, not a foundation — and the tilt
is now two-thirds of what it was, because a smaller `c` means a smaller gap to tilt with.

---

## Specification decisions

**The Phillips curve is anchored on the target, not on expectations.** `q(2.5)` is the
baseline and expectations enter only as `beta_pi × [q(pi^e) − q(2.5)]`. The pairing matters:
with a target baseline the second term is the pass-through of de-anchoring, `beta_pi` = 0
meaning the target holds and 1 meaning expectations are what bind. An earlier version used the
*Target Anchored* expectations series as the baseline together with an excess term built as
unanchored-minus-anchored, which put two estimates of one quantity in one equation and made
`beta_pi` a blend weight between two measurements rather than an economic parameter. It also
left `beta_pi` straddling zero. Fixing the baseline moved it to 0.568 [0.311, 0.831], and it
sits at 0.578 [0.347, 0.813] on the current gap — the Phillips side of the model is almost
untouched by what happened upstream.

**Expectations are the unanchored series.** The Target Anchored series is constructed with a
2.5% anchor post-1998, so its distance from 2.5 is near zero by construction and the excess
term would test nothing.

**No phase-in.** `nairu` phases expectations to the target over 1993-1998 because its sample
starts in 1984. This sample starts 1993Q1, inside the inflation-targeting era, so there is
nothing to phase from — the same choice `ystar` makes from the same start date.

**No supply-shock masking, and the live GSCPI.** `nairu` keeps GSCPI only over 2020Q1-2023Q2,
leaving 14 non-zero quarters. Here it is unmasked, so the coefficient is identified on the
whole history. That requires `gscpi_live`: the checked-in workbook stops at 2024Q1 while the
published series runs past 2026, and unmasked those quarters matter. Masking turns out to be
nearly redundant given the squared form — 74% of total `GSCPI²` still falls in the pandemic
quarters with every quarter included — and the coefficient survives unmasking at 0.042 against
`nairu`'s masked 0.047.

**The relationship is contemporaneous, deliberately.** Unemployment responds to demand with a
lag and so does inflation, so aligning them at *t* assumes those lags are equal. Modelling long
and variable lags is complicated, and `ystar` already found this sample cannot resolve
the timing: a free lag profile did not converge, and a fixed four-quarter lead converged but
left `sigma_e` unchanged while costing the last four quarters of the gap. Lagging the gap here
would also break the only channel tying u\*'s level to contemporaneous inflation.

**The drawn band is the posterior band doubled, and that factor now rests on nothing.** It was
chosen because two quantities happened to match on the driftless model: the conditional 90%
band was 0.47pp wide at the endpoint, and the spread of u\* across the swept `sigma_ustar`
range was 0.46pp, so doubling reproduced band-plus-sweep almost exactly. Both halves of that
coincidence have gone. The sweep has been removed as stale, and the band itself is narrower now
that u\* converges. **The factor of two is therefore inherited rather than justified**, and
re-deriving it depends on re-running the sweep. It remains an approximation to a sweep rather
than a posterior, and the left footer of every chart says so.

---

## Comparison with the `nairu` model

| | `ustar` | `nairu` (`simple_excess_rstar_blend`) |
|---|---|---|
| `gamma_pi` | −1.055 [−1.26, −0.86] | −0.709 [−0.91, −0.51] |
| `xi_gscpi` | 0.042 [0.031, 0.052] | 0.047 |
| `rho_pi` | 0.007 [0.003, 0.012] | 0.014 |
| u\* / NAIRU 2026Q2 | 4.71 | 4.88 |
| gap | −0.36 | −0.53 |
| realised `sd(d·)` | 0.023 (of 0.040 allowed) | 0.032 (of 0.150 allowed) |
| equations | 2 | 7 |

The `nairu` column is unaffected by the `ystar` respecification: it builds its own potential
from Cobb-Douglas and reads nothing from `ystar`. So the two models have moved apart slightly,
0.10pp on the level against 0.17pp now, and the whole of that movement is on this side.

Close on the level and on the supply coefficient, with `gamma_pi` steeper here. At
`sigma_ustar` = 0.024 the slope comes to −0.73 against `nairu`'s −0.709, which is another way
of seeing that the slope and the drift trade off: the two models agree on the Phillips slope
once this one is told to hold u\* as still as `nairu` effectively holds its NAIRU. The re-run
confirms it — the pre-exclusion figure was −0.71.

**Do not read this as independent corroboration.** Both models read the same expectations model
output, the same trimmed mean series, the same GSCPI, and now the same equation form. Agreement
under those conditions is substantially mechanical. It is reassuring about arithmetic, not
confirmation of the economics.

**What this model adds is legibility, not identification.** `nairu` is not strongly identified
on the real side either — its `beta_is` is 0.084 [0.024, 0.139], `gamma_fi` touches zero at
0.043 [0.000, 0.094], and its Okun residual of 0.248 against `sd(dU)` = 0.300 leaves most of
ΔU unexplained. Both models rest on an inflation relationship plus an imposed smoothness. Here
you can see that in two lines; there it is distributed across seven equations, where a realised
drift one-fifth of what is allowed makes it look as though the data are binding when much of
the work is being done by other imposed structure.

---

## Limitations

1. **The headline is conditional on chosen structure, and more so than it used to be.** u\*
   converges to an estimated equilibrium, and about 97% of its total fall is that mechanism
   rather than the data. `sigma_ustar` = 0.020 then decides how far it may wander from that
   curve. Neither is estimable. This is the limitation; everything else is detail.
2. **The uncertainty band understates, and the correction is a hand-applied factor of two.**
3. **u\* does not respond to COVID.** It glides through 2020-22 while unemployment goes 5.2 →
   7.0 → 3.5. Defensible — booking a pandemic as structural is the error smoothness priors
   exist to prevent — but it means the model has nothing to say about post-COVID structural
   change, which is the question people most want a NAIRU for.
4. **1993 is weak.** u\* = 8.4 against unemployment of 10.9 gives a +2.5pp gap at the start of
   the sample, and it is where the with-gap and without-gap estimates differ most.
5. **The inflation-band chart is illustration, not validation.** The Phillips curve fits
   inflation with `gamma × u_gap` and gamma is negative, so the estimation is not neutral about
   whether a negative gap coincides with above-band inflation. The disagreements are
   informative; the agreements largely are not.
6. **The headline moves with `ystar`'s decisions, not only with its data.** Excluding six
   pandemic quarters upstream took `c` from 0.468 to 0.188 and this model's unemployment gap
   from −0.43 to −0.36, with no new observation involved. That is a second conditioning choice
   sitting behind the first: the level of u\* is set by `sigma_ustar`, and the size of the gap
   partly by a judgement made in another package. Both are defensible and neither is data.
   When quoting the gap, quote the vintage.
7. **No wage equation, no IS curve, no r\*, no regime switching, no forecast scenarios.** Use
   `nairu` for those.

---

## Files

```
src/models/ustar/
├── config.py         # ModelConfig: sample, gap source, the imposed drift, the target
├── observations.py   # assembles u, the given gap, inflation, expectations, the shocks
├── estimate.py       # builds and samples the PyMC model, saves the trace
├── results.py        # UStarResults: posterior accessors and the failure diagnostics
├── analyse.py        # charts and printed diagnostics
└── run.py            # CLI
```

Charts land in `charts/UStar/`: u\* against unemployment, the same with the RBA band shaded,
the unemployment gap, and the price inflation decomposition.

```bash
./run-ustar.sh --verbose
./run-ustar.sh --analyse-only            # recharts from the saved trace
./run-ustar.sh --sigma-ustar 0.030       # the setting the answer hinges on
./run-ustar.sh --free-sigma-ustar        # the pile-up demonstration
./run-ustar.sh --no-output-gap           # does the gap matter?
./run-ustar.sh --gap-source actual       # log_gdp - y* instead of c·(pi - 2.5)
./run-ustar.sh --no-phillips             # Okun only
```

### The sweep, when it is re-run

The `sigma_ustar` sweep was removed from "Read this first" because every row was estimated on
the driftless random walk. Re-running it under convergence is outstanding, and two things in
this file depend on it: how conditional u\* = 4.83 really is, and the factor of two on the
drawn band. Variant prefixes route their charts to `charts/UStar-<prefix>` automatically, so
no `--no-analyse` is needed to protect the default directory:

```bash
for s in 0.010 0.015 0.030 0.040; do
  ./run-ustar.sh --sigma-ustar $s --prefix "ustar_su$s"
done
```

0.020 is the default run itself.

**Still outstanding**, and the last thing in this file on the pre-exclusion gap:

```bash
./run-ustar.sh --free-sigma-ustar        # the pile-up, expected to sharpen
```
