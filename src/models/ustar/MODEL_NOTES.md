# u* — a NAIRU from a given output gap

A Bayesian unobserved-components model (PyMC + NumPyro NUTS) estimating the Australian
unemployment rate consistent with output at potential and inflation at target. One latent
state, two observation equations, seven estimated parameters.

```
u*_t = u*_{t-1} + e_u                          u*: random walk, sigma imposed
u_t  = u*_t - beta·ygap_t + e_o                 Okun:     unemployment fitted
pi_t = q(2.5) + beta_pi·[q(pi^e_t) - q(2.5)]    Phillips: inflation fitted
       + gamma·(u_t - u*_t)/u_t
       + rho·d4pm_t + xi·GSCPI_t²·sign(GSCPI_t) + e_p
```

`q(·)` converts an annual rate to a quarterly one. The output gap `ygap` is **not estimated
here**: it is read from a completed `ystar` run. Inflation expectations come from the
`expectations` model. Run order is `expectations` → `ystar` → `ustar`, all from saved
output, nothing re-estimated.

Estimated: `beta_okun`, `sigma_okun`, `gamma_pi`, `beta_pi`, `rho_pi`, `xi_gscpi`,
`epsilon_pi`. Imposed: `sigma_ustar` = 0.040. Asserted: the 2.5% target, flat.

---

## Read this first: what the model determines, and what you determine

**The level of u\* is set by a number you choose, not by the data.** `sigma_ustar` — how fast
u\* is allowed to drift — is imposed, and it cannot be estimated (see "Three ways of trying to
estimate the drift"). Across the defensible range the answer moves as much as the answer
itself is interesting:

| `sigma_ustar` | u\* 2026Q2 | gap | u\* range | `gamma_pi` | `beta_pi` | `epsilon_pi` |
|---|---|---|---|---|---|---|
| 0.024 | 5.10 | −0.75 | 1.91 | −0.71 | 0.79 | 0.167 |
| 0.030 | 4.92 | −0.57 | 2.65 | −0.83 | 0.71 | 0.163 |
| **0.040** | **4.78** | **−0.43** | 3.65 | −1.02 | 0.57 | 0.160 |
| 0.050 | 4.68 | −0.33 | 4.43 | −1.21 | 0.45 | 0.158 |
| 0.065 | 4.62 | −0.27 | 5.15 | −1.45 | 0.34 | 0.155 |

Every one of those runs samples cleanly and reports a tight credible interval that excludes
the others. By the standard `ystar` sets for itself — trend growth 2.12 to 2.19 across
a sixteen-fold sweep — **this model does not have a robust headline.** Quote "u\* is somewhere
near 4.8 and the labour market is tight by roughly half a point". Do not quote decimals.

Note also what the sweep does *not* move: `xi_gscpi` sits at 0.037-0.042 and `epsilon_pi` at
0.155-0.167 throughout. Those are identified by the data. `gamma_pi` and the gap trade off
almost exactly — their product is what the inflation data pin, and `sigma_ustar` chooses the
split between them.

**What the model does, stated plainly:** it splits the unemployment rate into a slow trend and
a cycle, where how much is trend you fix in advance, and the Phillips curve sets the amplitude
of the cycle by asking how large a gap is needed to explain inflation's distance from target.

---

## Results (2026Q2 vintage, `sigma_ustar` = 0.040)

Converged: all `r_hat` = 1.00, `ess_bulk` 3,654 to 9,811.

| Parameter | mean | 90% HDI | |
|---|---|---|---|
| `beta_okun` | 1.065 | [0.810, 1.319] | P(> 0) = 100% |
| `sigma_okun` | 0.684 | [0.567, 0.798] | |
| `gamma_pi` | −1.020 | [−1.251, −0.796] | P(< 0) = 100% |
| `beta_pi` | 0.568 | [0.311, 0.831] | de-anchoring pass-through |
| `rho_pi` | 0.007 | [0.002, 0.012] | import prices |
| `xi_gscpi` | 0.042 | [0.029, 0.053] | supply chains |
| `epsilon_pi` | 0.160 | [0.141, 0.179] | |

| Headline, 2026Q2 | |
|---|---|
| u\* | **4.78** |
| u − u\* | **−0.43** |

`beta_okun` = 1.07 is far above a textbook Okun coefficient and should not be read as one. The
`ystar` gap is a shrunk regressor — `c` is a conditional mean on a signal explaining
about a fifth of output's variation — so the slope compensates. It is a scaling onto this
particular gap series.

---

## Three ways of trying to estimate the drift, and why all three fail

The likelihood has a **monotone preference for more state variance**: a u\* that tracks
unemployment fits better quarter by quarter than one that doesn't. That single fact defeats
every route.

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
  **below** the RBA band, signalling genuine slack — u\* declines at 0.040 and tighter (−0.09),
  is flat at 0.050 (−0.01), and **rises** at 0.065 (+0.09), booking part of the
  post-mining-boom rise in unemployment as structural. `beta_pi` also falls monotonically,
  0.79 to 0.34, as a freer u\* crowds out the de-anchoring term.

None of these is external to this repo. **The outstanding improvement is to calibrate the drift
against a published NAIRU series** — RBA, Treasury, OECD — whose realised drift is an
observable rather than a modelling choice. That is the one change that would resolve the
assumption rather than relocate it.

---

## Does the output gap actually matter?

The package was built on the premise that a credible output gap from `ystar` is what
makes a two-equation u\* possible. That is testable. Zeroing the gap while keeping the Okun
equation's structure — so `u = u* + e_o` still fits a trend through unemployment — isolates
the gap's contribution (`--no-output-gap`).

| | with gap | gap zeroed |
|---|---|---|
| u\* 2026Q2 | 4.78 | 4.60 |
| gap 2026Q2 | −0.43 | −0.25 |
| `sigma_okun` | 0.684 | 0.908 |
| `beta_okun` | 1.065 | 0.499 |
| `gamma_pi` | −1.020 | −1.032 |
| `epsilon_pi` | 0.160 | 0.159 |

u\* path correlation **0.9973**, mean absolute difference **0.10pp**, max 0.38pp — and the max
is in 1993, the least identified end of the sample.

**The gap does real work in the Okun equation**: `sigma_okun` rises 0.684 → 0.908 without it, a
43% reduction in residual variance. `beta_okun` collapsing to 0.499, its prior mean, confirms
the test removed what it was meant to.

**But it barely moves u\*.** 0.18pp at the endpoint, against 0.48pp from choosing `sigma_ustar`
within its defensible range. And the Phillips side is untouched — `gamma_pi` −1.020 against
−1.032, `epsilon_pi` 0.160 against 0.159 — so the two channels are not sharing identification.

**Verdict.** The premise is partly vindicated and partly not. The gap explains the cyclical
component of unemployment well, and it moves the headline gap from −0.25 to −0.43, which is
not decorative. But it is not what makes the two-equation u\* possible: the Phillips curve and
the imposed drift do that, and they would do it nearly as well with the gap set to zero. The
gap is a tilt, not a foundation.

---

## Specification decisions

**The Phillips curve is anchored on the target, not on expectations.** `q(2.5)` is the
baseline and expectations enter only as `beta_pi × [q(pi^e) − q(2.5)]`. The pairing matters:
with a target baseline the second term is the pass-through of de-anchoring, `beta_pi` = 0
meaning the target holds and 1 meaning expectations are what bind. An earlier version used the
*Target Anchored* expectations series as the baseline together with an excess term built as
unanchored-minus-anchored, which put two estimates of one quantity in one equation and made
`beta_pi` a blend weight between two measurements rather than an economic parameter. It also
left `beta_pi` straddling zero. Fixing the baseline moved it to 0.568 [0.311, 0.831].

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

**The drawn band is the posterior band doubled.** The conditional 90% band is 0.49pp wide at
the endpoint and is near-invariant to the setting; the spread of u\* across the swept range is
0.48pp. Doubling therefore reproduces band-plus-sweep almost exactly. It is an approximation to
a sweep, not a posterior, and the left footer of every chart says so.

---

## Comparison with the `nairu` model

| | `ustar` | `nairu` (`simple_excess_rstar_blend`) |
|---|---|---|
| `gamma_pi` | −1.020 [−1.25, −0.80] | −0.709 [−0.91, −0.51] |
| `xi_gscpi` | 0.042 [0.029, 0.053] | 0.047 |
| `rho_pi` | 0.007 [0.002, 0.012] | 0.014 |
| u\* / NAIRU 2026Q2 | 4.78 | 4.88 |
| gap | −0.43 | −0.53 |
| realised `sd(d·)` | 0.023 (of 0.040 allowed) | 0.032 (of 0.150 allowed) |
| equations | 2 | 7 |

Close on the level and on the supply coefficient, with `gamma_pi` steeper here. At
`sigma_ustar` = 0.024 the slope comes to −0.71, matching `nairu` almost exactly, which is
another way of seeing that the slope and the drift trade off.

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

1. **The headline is conditional on a chosen number.** See the sweep. This is the limitation;
   everything else is detail.
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
6. **No wage equation, no IS curve, no r\*, no regime switching, no forecast scenarios.** Use
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
