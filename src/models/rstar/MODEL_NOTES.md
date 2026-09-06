# r* — the natural rate from the bond market

A Bayesian unobserved-components model (PyMC + NumPyro NUTS) estimating the Australian
natural rate of interest. One latent state — an Australia-specific wedge over published
world r\* — read off the indexed (real) 10-year bond yield.

```
wedge_t = wedge_{t-1} + sigma_walk · e_t,  e_t ~ StudentT(nu)   the only state
r*_t    = w_t + wedge_t                    w is data, not an observation
tp_t    = y_t - r*_t                       identity, no residual
tp ~ AR(1) about mu_tp, stationary          the identifying prior
```

`y` is the AU indexed real 10-year yield, `w` is the mean of the NY Fed's
Holston-Laubach-Williams r\* for the US, Euro Area and Canada. Estimated: `wedge_0`,
`nu_walk`, `mu_tp`, `rho_tp`, `sigma_tp`. Imposed: `sigma_walk`. Sample 1986Q3-2026Q2,
160 quarters.

**There is no IS curve, deliberately.** The repo's central negative finding is that **the
IS curve is fragile in Australian data** — the interest rate does not visibly move real
activity at the frequencies these models use. `rstar_hlw` measures it at a_r ≈ −0.04
against σ_IS ≈ 0.70, a signal-to-noise ratio of about 0.11; `nairu`'s IS curve gives β_is
≈ 0.084 with fiscal touching zero; the `dsge` family found the same. r\* non-identification
is a symptom of that, not a separate problem. This package reads r\* off an asset price
instead, which sidesteps the broken link rather than trying to strengthen it.

**The tension that creates, stated up front.** This model carries a Taylor rule, and a
policy rule is only worth prescribing if moving the rate moves the economy. Read the
prescription as what a central bank following a standard reaction function *would* do —
not as a forecast of what would follow if it did. The transmission the rule relies on is
exactly what `rstar_hlw` could not find. See
[`rstar_hlw/MODEL_NOTES.md`](../rstar_hlw/MODEL_NOTES.md).

---

## Read this first

**The headline is r\* around 1.2 to 1.6, and the cash rate about a point below where a
Taylor rule would put it.** Both survive a sweep of the one imposed setting. What does
*not* survive is the recent path — how far r\* fell in 2021 and therefore how much of the
bond selloff is neutral rate rather than term premium.

| `sigma_walk` | r\* now | r\* 2021 | rise since | % of yield rise | pre-GFC | ν | Taylor | gap |
|---|---|---|---|---|---|---|---|---|
| 0.03 | 1.62 | 0.54 | 1.08 | 36% | 2.37 | 1.8 | 5.98 | +1.63 |
| 0.05 | 1.39 | 0.19 | 1.20 | 39% | 2.30 | 2.1 | 5.75 | +1.40 |
| **0.08** | **1.24** | −0.26 | 1.50 | 50% | 2.18 | 2.4 | **5.61** | **+1.26** |
| 0.12 | 1.24 | −0.68 | 1.92 | 63% | 2.09 | 2.9 | 5.61 | +1.26 |
| 0.20 | 1.25 | −1.40 | 2.66 | 88% | 1.89 | 4.5 | 5.62 | +1.27 |

r\* today spans 1.24-1.62 across a near seven-fold range of the setting, and is pinned at
1.24-1.25 from 0.08 upward. The Taylor gap is +1.26 to +1.63 throughout — always positive,
always more than a point. Pre-GFC r\* is 1.89-2.37 on every setting, so **today is 55-68%
of the pre-GFC level** regardless.

The 2021 trough runs +0.54 to −1.40, so "r\* has risen 1.5 points since 2021" is a
statement about `sigma_walk`, not about Australia. Quote the level and the comparison to
pre-GFC; do not quote the rise.

Note also that `nu` and `sigma_walk` substitute for each other: 1.8 at the tightest
setting, 4.5 at the loosest. A larger permitted step needs less extreme tails to fit the
same jumps.

---

## Results (2026Q2 vintage, `sigma_walk` = 0.08)

All `r_hat` = 1.00, 1 divergence. `ess_bulk` 820 to 9,236 — `nu_walk` at 820 is the
weakest and the one to watch.

| Parameter | mean | 90% HDI | |
|---|---|---|---|
| `wedge_0` | 0.951 | [−1.02, 2.88] | wide: see identification |
| `mu_tp` | 0.976 | [−0.67, 2.75] | wide, and correlated with the above |
| `rho_tp` | 0.919 | [0.83, 0.98] | persistent but stationary |
| `sigma_tp` | 0.251 | [0.20, 0.30] | |
| `nu_walk` | 2.360 | [1.11, 3.85] | nearly Cauchy |

| Headline, 2026Q2 | |
|---|---|
| real r\* | **1.24** |
| world r\*, for comparison | 0.95 |
| term premium | 1.26 |
| nominal r\* (r\* + 2.5% target) | **3.74** |
| cash rate | 4.35 → **0.6 restrictive** |
| Taylor prescription | **5.61** → 1.26 below what the rule wants |
| r\* for firms (r\* + credit spread) | 2.01 |

`nu_walk` = 2.36 is the substantive finding of the specification: the data want the wedge
to sit still and then leap, not to drift. That was asserted before it was estimated, and
the estimate is emphatic.

---

## What the model identifies, and what it does not

**Identified: the changes.** The free walk found its ten largest wedge moves at 1993Q3,
1994Q1, 1995Q2, 1996Q4, 2008Q4, 2009Q2, 2012Q2, 2014Q4, 2019Q2 and 2022Q2 — the last of
these, +1.26, the largest in the sample. Four of those coincide with dates picked
independently from the era means of the AU-world spread: 1994Q1 (targeting established,
the global bond rout), 2008Q4 (the GFC), 2019Q2 (the RBA easing cycle) and 2022Q2 (liftoff
and the QE exit).

**Not identified: the level, directly.** In the step-break variant `wedge_0` and `mu_tp`
correlate at **−0.76**: the data pin their sum — the total level of the yield — but not
the split between "Australia's r\* sits above the world's" and "the average term premium
is large". Shift a point from one to the other and the fit is unchanged. The free walk
inherits the same weakness, which is why both parameters have 90% intervals nearly two
points wide while the wedge *moves* are estimated to ±0.28.

What rescues the level in practice is the stationarity prior on `tp` plus the world series
as a base. It is not an assertion in the way `ystar`'s 2.5% anchor is, and it is weaker
for that.

### COVID was not a break

The step variant estimated the 2020Q2 jump at −0.285 [−0.738, 0.175] — straddling zero,
the only one of five that did. The free walk does not place a large move there either.
Two structurally different specifications agree that the pandemic did not shift Australia's
wedge over world r\* beyond where the 2019 easing cycle had already taken it.

Note the dating: COVID is 2020**Q2**, not Q1. In 2020Q1 the AU-world spread *rose* 0.50,
because the March dash-for-cash pushed the real yield up. The suppression from bond
purchases and yield curve control shows from Q2.

### The term premium finds QE without being told

`tp` goes **negative** through 2020-2022, to about −0.4 — precisely the period of RBA bond
purchases and yield curve control, which is exactly what those policies do to a term
premium. Nothing in the model marks those quarters. It is the strongest internal
validation the decomposition has produced. Currently 1.26, the highest since 2009 and the
73rd percentile of its own history.

---

## Three specifications tried, and why the third is the default

**1. r\* as its own smooth random walk, with world r\* as a noisy observation.** Failed
twice over. Mechanically, `sigma_y` was redundant — with r\* and `tp` both free to explain
one yield series the observation noise had nothing to do, so it collapsed toward zero
(mean 0.035, ESS 11, `r_hat` 1.30) and `mu_tp` rode a ridge against the level of r\*,
producing 76 divergences. Substantively, even after fixing the geometry, `corr(r*, world
r*)` was **0.99** and r\* explained 21% of the yield's variance against the premium's 41%.
The model returned the NY Fed's number and called it Australian.

Dropping the world equation (`--no-world`) made it worse, not better: r\* went flat, moving
0.14pp across forty years, explaining 0.1% of the variance, with `r_star_0` spanning [0.14,
4.54]. Sweeping `sigma_r` changed nothing — it saturated by 0.10. **The Australian bond
market contributes no information about r\* once you insist r\* be smooth.**

**2. A step function with asserted break dates.** Fixed the sampling and produced clean
jumps, four of five clear of zero. But it left r\* explaining only 6.8% of the yield's
variance against the premium's 73%, put r\* at 2.55, and gave a Taylor rule that prescribed
tightening through 2012-2019 — a decade of below-target inflation and a negative output
gap. Retained as `--steps`, because it is the transparent comparator and the source of the
COVID finding above.

**3. A free random walk with Student-t innovations.** The default. r\* explains 62.0% of
the yield's variance, the premium 9.4%; `corr(r*, world r*)` falls to 0.91; `sd(dr*)` is
0.15 against world r\*'s 0.09, so r\* moves faster than the series it is anchored on. And
the Taylor rule built on it behaves across the whole sample.

| | step wedge | free walk |
|---|---|---|
| variance share, r\* | 6.8% | **62.0%** |
| variance share, term premium | 73.0% | 9.4% |
| r\* now | 2.55 | **1.24** |
| Taylor gap, 2016-19 | +1.59 | **−0.56** |
| Taylor gap, now | +2.57 | **+1.26** |

---

## The policy rule

**A level Taylor rule is the headline**, on Taylor's original 0.5/0.5, with the supply
contribution removed from inflation before the rule is applied:

```
i* = r* + pi_core + 0.5·(pi_core - 2.5) + 0.5·ygap,   pi_core = pi - supply
```

Inputs from elsewhere: `ygap` from `ystar`, and the supply term from `ustar`'s Phillips
decomposition (`rho·d4pm + xi·GSCPI²·sign`) on a four-quarter rolling sum — not
`annualize()`, which is a compounding transform and not additive across components.

Prescribed less actual, by era:

| era | gap |
|---|---|
| 1994-2007 | −0.56 |
| 2008-2011 | +0.39 |
| 2012-2015 | −0.16 |
| 2016-2019 | −0.56 |
| 2019-2021 | −0.17 |
| post-COVID | +2.19 |
| **now** | **+1.26** |

That pattern — at or below the actual rate through 2012-2019, above it now — was the
user's stated judgement *before* the free-walk specification was run, and the model
reproduces it without being calibrated to it. It is the best external check the package
has.

**Looking through supply matters in 2022-23 and nowhere else at present.** At 2022Q4 the
headline rule said 11.08 against the look-through 7.86, a 3.2-point difference; today the
two are 5.65 and 5.61. The removal is `(1 + a_pi)` times the supply contribution, because
it comes out of the Fisher term as well as the response term — the rule runs end to end on
supply-adjusted inflation.

Looking through supply does not blind the rule to a shock becoming embedded. Across
2022-2023 the supply contribution fell 1.72 → 1.11 while demand rose 0.63 → 1.54 and excess
expectations 0.04 → 0.46; the inflation the rule responds to *rose* over that window, 3.18
→ 4.49. What migrates into demand and expectations stays in.

**A first-difference rule is retained as a diagnostic** (`policy_change`, printed but not
charted). `d_i = 0.125·(pi - 2.5) + 0.125·ygap`, needing no r\* at all, which made it the
right fallback while the level was unidentified. Coefficients are Taylor's 0.5 divided by
four because they apply to a quarterly change; undivided they over-move fourfold. It
cumulates to −4.85 across 2012-2021 against 4.15 delivered, and +4.46 post-COVID against
+4.25 — but it fails at the GFC, wanting +2.2 while the RBA cut 2.5, because a
contemporaneous rule cannot cut pre-emptively into a collapse.

Do not cumulate it into a path. Cumulating from 1994 drifts to +4.35 against the actual
rate by 2012, because the gaps in the data were generated under the policy actually run.
That counterfactual needs the feedback from rates back to the gap — the IS curve this
package omits because Australian data will not identify it.

---

## Two rates, and the wedge is observed

The bond market prices the globally-arbitraged risk-free rate. What governs investment is
the cost of capital to firms, above it by the external finance premium. That premium is
**data** here — the corporate-to-CGS spread — not a latent state, which is the substantive
difference from the `dsge` FA-NK models whose latent wedge produced an r\* those notes call
not credible.

The spread begins 2005Q1 against the yield's 1986Q3, so it is applied *after* estimation:
the state is estimated on the long sample and the business rate derived where the spread
exists. Nothing about the identification depends on the short series. Currently 0.79
against a 2005-2026 mean of 1.34, so the risk-free real rate is up while the business
premium is unusually compressed.

---

## Limitations

1. **`sigma_walk` is imposed.** The sweep above is the honest report. The level survives
   it; the recent path does not.
2. **The level is weakly identified even so.** `wedge_0` and `mu_tp` trade off, and both
   carry intervals nearly two points wide.
3. **World r\* enters as data with no error term.** If the NY Fed's series is wrong we
   inherit the error wholesale, with no residual to absorb it. That is the price of taking
   "r\* is largely imported" as the maintained hypothesis rather than testing it.
4. **`nu_walk` at 2.36 is close to Cauchy**, and `ess_bulk` of 820 is the weakest sampling
   in the package. Worth checking the tails are not letting the wedge absorb things they
   should not.
5. **Indexed AGS are thin**, so the yield carries a liquidity premium a nominal bond does
   not. That premium is part of what `mu_tp` absorbs, and it is not separable from the term
   premium proper.
6. **The break-date list in `ModelConfig` is only used by `--steps`.** The default does not
   need it. `end_break_check()` reports whether the term premium has drifted from `mu_tp`
   at the end of the sample, which would signal a shift the step variant could not see;
   currently +0.06 stationary sds, well under the 1.0 threshold.

---

## Refinements to explore

This package was built in a single session and has had one vintage. The list below is
roughly in order of how much each would change what the model can claim, not how hard it is.

### 1. Endpoint fragility — the one that matters most

**Nothing here tests it, and the headline *is* the endpoint.** `ystar` has `realtime.py`,
which re-estimates on progressively truncated samples to show what a real-time user would
have seen. `rstar` needs the same, and has more reason to: r\* is a random walk, so its last
value is the least constrained point in the sample, and the wedge's fat tails mean a single
new quarter can move it discontinuously by design.

The test: re-estimate ending at each of the last twenty quarters and plot the final r\*
from each vintage against the full-sample path. If the endpoint swings by more than the
`sigma_walk` sweep does, the sweep understates the real uncertainty and the notes above
should say so.

### 2. An external term premium, to pin the level rather than relocate it

The level is weakly identified because `wedge_0` and `mu_tp` trade off (−0.76 in the step
variant). Every fix tried so far moves the assumption somewhere else. The only thing that
would genuinely resolve it is an **external estimate of the average real term premium on AU
indexed bonds** — an ACM- or Kim-Wright-style decomposition, or an RBA published estimate —
imposed as `mu_tp` rather than estimated.

That would be this model's equivalent of `ystar`'s 2.5% anchor: one asserted number, from
outside the model, that pins a level the data cannot. It would also let the term premium
chart be checked against something rather than only against its own plausibility.

Caveat worth stating in advance: indexed AGS are thin, so their yield carries a liquidity
premium a nominal bond does not, and a term premium estimate built on nominal bonds is not
measuring the same object. The gap between them would need its own assumption.

### 3. `nu_walk` sampling

`ess_bulk` of 820 is the weakest in the package, and the posterior mean of 2.36 is close
enough to Cauchy that the wedge's tails are doing a great deal of work. Two things to check:
whether a longer run or a reparameterisation tightens it, and whether the fat tails are
absorbing movement that belongs in the term premium. A prior-predictive check — simulating
wedges at `nu` = 2 and asking whether they look like a natural rate — would be cheap and
informative.

### 4. Does the world anchor have to be HLW?

World r\* currently enters as data with no error term, so an error in the NY Fed's series
passes through undamped. Worth trying: the US series alone rather than the three-country
mean (the marginal pricer rather than the average, which is arguably the truer mechanism);
a global real yield instead of a model output, which would remove the dependence on
somebody else's model entirely; and giving the anchor a measurement error after all, now
that the wedge is free enough for the level not to collapse onto it.

### 5. The `--steps` variant deserves the free walk's break dates

The step variant currently uses dates chosen by reading the era means. The free walk found
its own, and four of five agreed. The obvious next version asserts *the free walk's* dates
and compares — if the two converge, the step variant becomes a genuinely independent
confirmation rather than a differently-parameterised version of the same reading.

### 6. Cross-checks the model has not been put through

- **Against `rstar_hlw`**: Resolution G gives 2.23 where this gives 1.24. Both are in this
  repo and neither has been reconciled with the other. The comparison belongs in one of the
  two sets of notes.
- **Against the RBA's own statements.** Bullock's "the neutral rate shifted in about sixteen
  months" is a claim about speed that this model can test directly, since the wedge's jumps
  are dated and sized.
- **Out of sample.** The 2016-19 and present-day Taylor readings matched a judgement stated
  in advance, which is one observation. A second — picking a period, stating the expected
  reading, then running it — would be worth more than any in-sample diagnostic here.

### 7. Smaller things

- The corporate spread starts 2005Q1, so `r*` for firms is blank for the first nineteen
  years. A longer or spliced credit spread would extend it.
- `end_break_check()` uses a fixed eight-quarter window and a threshold of one stationary sd,
  both picked by eye. Neither has been calibrated against how often it would have fired
  historically.
- The Taylor coefficients are Taylor's originals and unswept. Given how much else here moves
  with its settings, they should be swept too before the +1.26 gap is quoted anywhere.

---

## Files and usage

```
src/models/rstar/
├── config.py         # ModelConfig: sample, world anchor, sigma_walk, the rule
├── observations.py   # the yield and world r*, plus ragged extras for the charts
├── estimate.py       # builds and samples the PyMC model
├── results.py        # RStarResults: posteriors, derived series, diagnostics
├── analyse.py        # charts and printed diagnostics
└── run.py            # CLI
```

Run order: `expectations` → `ystar` → `ustar` → `rstar`. r\* itself needs neither `ystar`
nor `ustar`; the Taylor rule needs both, and the affected charts are skipped with a note if
they are missing.

```bash
./run-rstar.sh -v
./run-rstar.sh --analyse-only         # recharts from the saved trace
./run-rstar.sh --sigma-walk 0.03      # the setting the answer leans on
./run-rstar.sh --steps                # the asserted-break comparator
./run-rstar.sh --no-world             # does the global anchor do the work? (yes)
./run-rstar.sh --world-source US      # the marginal pricer, not the average
./run-rstar.sh --no-look-through      # respond to headline inflation instead
```

Charts land in `charts/RStar/`: the Taylor rule with real and nominal r\* against the cash
rate, r\* against the real yield and the world anchor, the term premium, and r\* for firms.
