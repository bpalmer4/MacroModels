# r*, the natural rate from the bond market

A Bayesian unobserved-components model (PyMC + NumPyro NUTS) estimating the Australian
natural rate of interest from asset prices. One latent state, an Australia-specific wedge
over a published world real rate, read off two windows on the same curve.

```
wedge_t = wedge_{t-1} + sigma_walk · e_t,  e_t ~ StudentT(nu)   the only state
r*_t    = b_world · w_t + wedge_t          w is data, not an observation
g_t     = r_t - r*_t                       identity: the policy stance
tp_t    = y_t - r*_t - k·g_t               identity: the term premium
g, tp ~ stationary AR(1)                   the identifying priors
```

`y` is the AU indexed real 10-year yield, `r` the real overnight cash rate, `w` the
Cleveland Fed's 10-year expected real rate. Sample 1993Q1-2026Q3, 135 quarters. The
equation-by-equation section takes every line one at a time.

## One of three routes, all flawed

This repo now contains three separate attempts at Australian r\*, and the useful thing is
that they fail differently. Read them together rather than picking one.

| package | identified from | what it actually measures | how it fails |
|---|---|---|---|
| [`rstar_hlw`](../rstar_hlw/MODEL_NOTES.md) | trend growth + the IS curve | the textbook definition: the rate at which output sits at potential | the IS curve does not identify anything on AU data, so each specification returns its own prior |
| **`rstar`** (this one) | asset prices | what investors price | the *level* is not identified; it rests on the stationarity prior and an asserted premium |
| [`rstar_rba`](../rstar_rba/MODEL_NOTES.md) | the RBA's response to inflation | not r\* but what the Bank's conduct *reveals* about it, conflated with every other systematic motive | a long enough departure from the rule is absorbed into neutral, so it cannot audit the Bank over a decade |

Only the first targets what the theory defines, and it is the one that cannot be estimated.
The other two measure *beliefs* about r\*, held by different people. That is a sharper
statement of the package's thesis than "r\* is imported", and it was arrived at by
exhausting the alternatives rather than by assertion.

**There is no IS curve here, deliberately.** The repo's central negative finding is that the
interest rate does not visibly move real activity at these frequencies. `rstar_hlw` measures
it at a_r ≈ −0.04 against σ_IS ≈ 0.70; `nairu`'s IS curve gives β_is ≈ 0.084;
[`is_curve`](../is_curve/MODEL_NOTES.md) shows the raw scatter does not even recover the
*sign*. This package reads r\* off an asset price instead, sidestepping the broken link
rather than trying to strengthen it.

**The tension that creates, stated up front.** This model carries a Taylor rule, and a rule
is only worth prescribing if moving the rate moves the economy. Read the prescription as
what a standard reaction function *would* say, not as a forecast of what would follow.

---

## Read this first

| | 2026Q3 |
|---|---|
| real r\* | **1.08** [−0.67, 2.72] |
| world real rate, for comparison | 2.13 |
| the Australian wedge | **+0.05** |
| term premium | 1.54 |
| nominal r\* (r\* + 2.5% target) | **3.58** |
| cash rate | 4.35, so **0.51 restrictive** on the model's own stance `g` |
| Taylor prescription (2026Q2, the last quarter its inputs cover) | 5.14 against a 4.35 cash rate |
| r\* for firms (r\* + credit spread) | 1.86 |
| pre-GFC r\* (1994-2007) | 2.43, so today is **44%** of it |

Zero divergences, all `r_hat` 1.00, `ess_bulk` 3,056 to 8,223.

**Australia currently sits on the world rate.** The wedge is +0.05, which is the cleanest
result the model produces: nothing Australia-specific is depressing neutral today. The
divergence opened over 2012-2021, bottoming at −1.07, and has closed.

**Quote the wedge and the era pattern. Do not quote the level.** The 90% interval on r\* is
three points wide and straddles zero at every date, `wedge_0` and `mu_tp` correlate at
**−0.87**, and the level moved between 0.83 and 1.22 across four defensible specifications
built in one sitting. What survives respecification is the *relative* reading.

---

## The model, equation by equation

Every number is from the current default run (`model_outputs/rstar_trace.nc`); intervals
are 94% HDIs unless stated.

### What is actually fitted

Three observed series, 1993Q1 to 2026Q3, no gaps. Nothing else enters the likelihood: the
corporate spread, the mortgage rate, inflation and the output gap arrive afterwards for the
derived series and are not allowed to shorten the sample.

| | series | source |
|---|---|---|
| `y` | indexed real 10y yield | RBA F2 |
| `w` | 10-year expected real rate | Cleveland Fed, via FRED `REAINTRATREARAT10Y` |
| `r` | real cash rate | RBA F1 less inflation expectations |

**Why not Holston-Laubach-Williams for `w`.** HLW was the anchor until this vintage and is
still available via `--world-source mean`. It was dropped because it is not a price: it is
`r* = g + z` identified through the IS and Phillips curves, so a global bond selloff cannot
move it, and it did not. Between the COVID-QE era and the tightening era the HLW
three-country mean moved **−0.03** while the US 10-year TIPS yield rose 2.31 and the real
fed funds rate 2.62. Anchoring a market-read model on a macro-read estimate, produced by
the identification strategy this package rejects, was incoherent.

**What the Cleveland series is, and is not.** A ten-year expected real rate: the nominal
Treasury less their modelled expected inflation. It responds to monetary conditions, which
is the point. But it is a *yield*, so it contains a US term premium, and it is US rather
than global. Both matter, and `--world-source market` addresses the first (see Alternatives).

### 1. The state: the Australian wedge

```
wedge_t = wedge_0 + sigma_walk · sum_{s<=t} e_s,   e_s ~ StudentT(nu, 0, 1)
```

Carries everything about Australian r\* that is not imported. It is the only latent state;
everything else is data or an identity built on it.

| | prior | posterior |
|---|---|---|
| `wedge_0` | Normal(0, 2) | 1.504 [−0.435, 3.454] |
| `nu_walk` | Gamma(2, 2/6), mean 6 | **8.212** [2.459, 15.860] |
| `sigma_walk` | **imposed at 0.12** | not estimated |

**`nu` is the model's tell, and it is now healthy.** At `sigma_walk` = 0.08 it came back at
3.64, inside the infinite-kurtosis regime where single jumps are nearly free; under the
three-window variant it fell to 1.87, close to the ν = 2 line below which the Student-t has
no variance at all and `sigma_walk` stops being a standard deviation. Both were the model
buying with fat tails the room the variance denied it. At 0.12 with a market anchor, `nu`
returns to 8.2, above its prior mean of 6, and the walk stops straining.

**Do not read fat tails as a finding about Australia.** Earlier write-ups did. The wedge's
largest quarterly move is now +0.28 at 2022Q2, then −0.24 at 2012Q2, −0.21 at 1993Q3,
+0.18 at 2023Q2. There is no leap: the wedge is a smooth secular path, +1.21 through
1994-2007, crossing zero around 2012, bottoming at −1.07 in 2020-21, and back to +0.05.
That is a far more plausible object to call "Australia's structural spread over the world"
than the +0.82 jump the flat HLW anchor required.

### 2. How much of the world rate passes through

```
r*_t = b_world · w_t + wedge_t
```

| | prior | posterior |
|---|---|---|
| `b_world` | Normal(1, 1) | **0.481** [0.297, 0.662] |

**This is the model's most uncomfortable number.** Taken at face value it says only half a
move in the world real rate reaches Australian r\*, which is hard to defend for an open
economy with a floating currency: it implies a differential that widens without arbitrage.

It is probably not a structural pass-through. The anchor is a ten-year real *yield*
containing a US term premium, and the AU observable is a ten-year real yield containing an
Australian one. Faced with two co-moving premia the model can either load fully on the
anchor and shrink its own premium, or damp the loading and let its premium absorb the
co-moving part. It chose the second, so `b_world` is partly a premium-stripping coefficient
rather than a pass-through. `--world-source market` removes the premium explicitly and is
the cleaner treatment (see Alternatives).

### 3. Window one: the long yield

```
tp_t = y_t - r*_t - k·g_t,      k = (1 - rho_g^H)/((1 - rho_g)·H),   H = 40
tp ~ stationary AR(1) about mu_tp
```

| | prior | posterior |
|---|---|---|
| `mu_tp` | Normal(0.75, 1) | 1.138 [−0.337, 2.740] |
| `rho_tp` | TruncatedNormal(0.8, 0.2) on [0, 0.98] | 0.841 [0.726, 0.958] |
| `sigma_tp` | HalfNormal(1) | 0.274 [0.235, 0.315] |
| `k` | computed from `rho_g`, not sampled | 0.146 [0.098, 0.195] |

**Why it is an identity with no residual.** An earlier version gave the yield equation an
observation error. With r\* and `tp` both free to explain one series the noise had nothing
to do: `sigma_y` collapsed toward zero (mean 0.035, ESS 11, `r_hat` 1.30) and `mu_tp` rode a
ridge against the level of r\*, giving 76 divergences. Removing the redundant parameter
removed the ridge. The data enter through the AR(1) *prior* on the resulting series, which
is why the model carries `pm.Potential` terms and not likelihood terms.

**Stationarity is the identifying assumption.** r\* is a random walk and `tp` must revert to
a constant mean. Permanent against transitory is the entire mechanism splitting the yield.
Nothing else does this work, which is why the level rests on it.

**`k` is not a free parameter.** A long real yield is roughly the average expected real
short rate over its term plus a premium, so if the gap reverts with persistence `rho_g`, the
average puts weight `k` on today's gap. `k` is computed from `rho_g`, which the second
window estimates. Adding a maturity adds an equation without adding a knob.

### 4. Window two: the short rate

```
g_t = r_t - r*_t
g ~ stationary AR(1) about mu_g
```

| | prior | posterior |
|---|---|---|
| `mu_g` | Normal(0, 3) | 0.213 [−1.611, 2.079] |
| `rho_g` | TruncatedNormal(0.85, 0.1) on [0, 0.98] | 0.823 [0.765, 0.884] |
| `sigma_g` | HalfNormal(1) | 0.418 [0.365, 0.474] |

`g` is the policy stance, positive restrictive. It is an output, not an input: nothing here
asks the stance to move output.

**Why the second window exists.** From one series the starting point of a random walk and
the mean of a stationary process are not separable. The short rate speaks to r\* directly
rather than through a premium. It does not make the level free, since `wedge_0`, `mu_tp` and
`mu_g` are three levels against two observables, but it makes the assertion checkable, and
it makes `rho_g` estimable, which is what makes `k` computable. It also fixed the sampling:
the one-window model produces 168 divergences, the two-window model zero.

**The assumption to notice.** `g` is stationary about a single mean, which claims policy
cannot sit away from neutral indefinitely. That assumption does the work in every statement
this model makes about the stance.

**And `g` is the risk-free stance, not the one households faced.** See the pass-through
section below; the two diverge by up to 1.75 points between 2009 and 2022.

### 5. The policy rule, outside the likelihood

```
i* = r* + pi_core + 0.5·(pi_core - 2.5) + 0.5·ygap,   pi_core = pi - supply
```

Read from a completed `ystar_ustar` run. No prior, no posterior, cannot move r\*, the wedge
or the premium. It is also the only part of the package that depends on rates moving output.
Its inputs end a quarter before the market anchor does, so the rule's last quarter is
2026Q2 while the state runs to 2026Q3.

---

## Results by era

| era | r\* | wedge | world | stance `g` | Taylor less actual | borrower stance |
|---|---|---|---|---|---|---|
| 1994-2007 | 2.43 | +1.21 | 2.53 | +0.43 | −0.59 | 2.48 |
| 2008-2011 | 1.14 | +0.70 | 0.94 | +0.64 | +0.14 | 2.74 |
| 2012-2015 | 0.00 | −0.17 | 0.37 | +0.05 | −0.66 | 2.67 |
| 2016-2019 | −0.50 | −0.84 | 0.71 | −0.21 | −0.99 | 2.85 |
| 2020-2021 | −1.11 | −1.07 | −0.08 | −0.63 | −0.73 | 2.84 |
| 2022- | 0.43 | −0.34 | 1.60 | +0.16 | +1.41 | 2.96 |

The decomposition attributes 79.2% of the yield's variance to r\* and 8.2% to the premium,
with `corr(r*, world r*)` = 0.91. `sd(dr*)` is 0.15 against the world's 0.27, so r\* is
smoother than the series it is anchored on. The wedge's range over the sample is 2.81.

---

## What the model identifies, and what it does not

**Identified: the wedge and the changes.** The wedge path is the model's real output, and
its current reading of zero is robust: it came back −0.02, +0.05 and −0.09 under three
quite different anchors.

**Not identified: the level.** `wedge_0` and `mu_tp` correlate at **−0.87**. The data pin
their sum, the level of the yield, not the split between "Australia's r\* sits above the
world's" and "the average term premium is large". The 90% interval on r\* straddles zero
throughout.

### The wedge is not an exchange rate story

Under real interest parity an AU-US real rate gap is expected real depreciation plus a
currency risk premium, and this model has nowhere to put either except the wedge. So the
wedge and the real exchange rate path are, in principle, observationally equivalent. No
exchange rate appears anywhere in the model, which makes this worth checking rather than
assuming.

There is a real relationship, and it is only a cyclical one. Regressing the wedge's
quarterly changes on log changes in the exchange rate gives a positive coefficient with the
sign parity implies, and it holds across both the real TWI and AUD/USD and both
specifications: `gamma` 0.377 to 0.584, t between 3.25 and 3.94. But R² is only 0.07 to 0.11,
and once cumulated the FX-attributed component spans just 0.28 to 0.34 points against a
wedge spanning 1.80 to 2.80, so 10 to 18 per cent of its range.

**The era pattern survives removing it.** Default spec, raw against FX-removed:

| | real TWI | AUD/USD |
|---|---|---|
| 1993-2007 | +1.20 → +1.16 | +1.20 → +1.21 |
| 2008-2015 | +0.24 → +0.02 | +0.24 → +0.14 |
| 2016-2019 | −0.86 → −1.03 | −0.86 → −0.88 |
| 2020-2022 | −1.00 → −1.17 | −1.00 → −1.01 |
| 2023-2026 | −0.27 → −0.47 | −0.25 → −0.23 |

Same sign sequence, same ordering, largest single move 0.22 and under AUD/USD never more
than 0.11. The headline decline does not weaken: first era to fourth is **−2.20 raw against
−2.33 FX-removed** on the real TWI and −2.22 on AUD/USD, so stripping the exchange rate
makes the fall marginally larger. The pin behaves the same way.

So the exchange rate moves the wedge around without taking it anywhere, and incorporating it
was rejected on that basis: a parity equation would import the PPP puzzle's unidentified
mean reversion, and the terms of trade plausibly drive the real exchange rate and the return
on capital together, so putting the rate on the right-hand side would attribute common
variation to FX by construction. Note also that the *level* correlation between the wedge
and the exchange rate is not usable in either direction: it runs −0.627 on the real TWI and
−0.111 on AUD/USD for the default, and −0.483 against **+0.084** for the pin, flipping sign
with the choice of series. The wedge is a random walk by construction, so a level
correlation against any persistent series is whatever the two trends happen to do.

### How much of the bond selloff is neutral rate?

The real yield rose 3.24 points from 2021Q4 to 2026Q3, and the model splits it: r\* +2.13,
premium +0.85, carried policy stance +0.24. So about two thirds is neutral.

That share is not identified either. Across a sweep of `sigma_walk` on the previous vintage
the r\* share ran from 28% at 0.03 to 70% at 0.20. What survives is the direction, and one
thing that does not depend on the split: the +0.24 of carried policy stance is a genuine
gain from the second window, which the one-window model set to zero by construction and
misallocated to the premium.

### The pre-COVID stance, and the standing critique

The objection to this package is Matt Cowgill's: through 2015-2019 inflation sat at or below
the bottom of the band, unemployment ran above most NAIRU estimates, and the cash rate was
between 0.75 and 1.5. If r\* says policy was well below neutral for five years, where was
the demand?

This vintage reads the 2015-19 stance at **−0.22**, against −0.73 on the published
one-window vintage. But that number moved between −0.89 and +0.37 across the specifications
built in one sitting, and it moved with settings nobody estimates, so it should be quoted as
a range and not a point.

**The stronger answer is not in this model at all.** See the pass-through section: measured
at the rate households actually paid, the stance barely moved across twenty years. Policy as
experienced was not meaningfully looser in 2015-19 than in the mid-2000s, so the absence of
a demand response needs no appeal to a broken IS curve.

### The QE finding did not survive

Earlier write-ups called the negative term premium through 2020-22 the strongest internal
validation the decomposition had produced: bond purchases showing up in the price of
duration, not in the natural rate, with nothing in the model marking those quarters.

It does not survive. The premium over 2020Q1-2022Q4 now averages **+1.02** and its sample
minimum is +0.04, in 2006Q3, not under QE at all. Across specifications it ran −0.05 with
one window and an imposed loading, +0.22 with a free loading, +0.58 with two windows, +1.02
with three. Only the original combination showed it. Note also that the mechanical `k·g`
correction the second window applies is small, so the shift is the re-identification of r\*
rather than the correction. **Do not repeat the published claim without this caveat.**

### The risk-free stance is not the stance households faced

`g` compares the real cash rate with a neutral rate read off a government bond. That is the
stance the economy faces only if the cash rate summarises the price of credit, and after the
GFC it stopped doing so.

| era | risk-free stance `g` | borrower stance | mortgage spread to cash |
|---|---|---|---|
| 2004-2007 | +1.22 | 2.46 | 1.24 |
| 2012-2014 | −0.20 | 2.39 | 2.59 |
| 2015-2019 | −0.89 | 2.10 | 2.99 |
| 2020-2021 | −1.39 | 2.07 | 3.47 |
| 2022- | −0.33 | 2.51 | 2.80 |

The risk-free stance swings 2.6 points across those eras. The borrower stance moves 0.4.
From 2014 to 2019 the cash rate fell 1.38 and the discounted owner-occupier mortgage rate
fell 0.70, so **half the easing never reached borrowers**.

And it was not margin. Banks' term deposit rates moved from 1.68 *below* the cash rate in
2004-07 to 0.39 above in 2015-19, a larger shift than the mortgage spread's 1.74, while the
90-day bill spread barely moved. The marginal funding dollar repriced when liquidity rules
pushed banks from cheap offshore wholesale funding toward competing for retail deposits.
`bank_costs` charts this in full.

`borrower_stance()` and `plot_borrower_stance` report it here. Levels are not comparable
between the two measures, since the borrower stance carries a credit spread; read the
changes.

---

## Alternatives, and why they are not the default

Every one of these is a switch, and each was tried as the default at some point.

**`--world-source market`: strip the US term premium from the anchor.** Cleveland less
Kim-Wright (`THREEFYTP10`) gives a world *neutral* rate rather than a yield: 1.30 over
1993-2007, 0.16 through the QE years, 1.31 now, so on this measure the world neutral rate
has fully recovered to its pre-GFC average and the rest of today's high yield is premium.
Conceptually the right object. It is not the default only because a free `b_world` on it
collapses to 0.233, so it must be run with `--impose-world-loading`.

**`--us-premium`: pin the term premium to the published US one.** The most defensible
treatment of the level, though it does not solve it. Instead of `tp` floating
about an unanchored `mu_tp`, it must track the Kim-Wright US premium and only the spread is
estimated, `mu_spread` = 0.329 [−0.588, 1.186] against a `N(0.25, 0.5)` prior. Run together
with `market` and `--impose-world-loading` (saved as `rstar_pin`) it gives r\* 1.22, a wedge
of −0.09 and an Australian premium 0.58 above the US, against the free-`mu_tp` version's
implausible +0.84 premium *and* −0.40 neutral. That combination is better argued than the free-`mu_tp`
version, but note what it does and does not achieve: `mu_spread` comes back 0.329
[-0.588, 1.186], an interval nearly as wide as its `N(0.25, 0.5)` prior. The level is still
asserted; the assertion has just moved to a quantity with a published benchmark behind it,
which is arguable on liquidity grounds rather than pulled from nowhere. Liquidity is not all
it carries, though: `mu_spread` is the average Australian premium over the US one, so it
also absorbs the **currency risk premium** a global investor demands for AUD exposure, and
the inflation risk premium left behind by subtracting a nominal US premium from a real
Australian one. The estimated spread does not track the currency (changes correlation
+0.009 against AUD/USD, +0.018 against the TWI), so this is a gap in the justification
rather than a demonstrated contamination, but the liquidity story alone understates what is
being asserted. That is an
improvement in accountability, not in identification, and it is why the two-window model on
a market anchor remains the default.

**`--premium-source acm`: the other published US term premium. Tried and rejected.**
Adrian, Crump and Moench (2013) is the obvious second opinion on the premium that both
`--world-source market` and `--us-premium` subtract, and the flag governs both places at
once, since mixing providers across the two is uninterpretable. It is read from the NY Fed
directly because FRED does not carry it.

It fails on the anchor, before Australia enters. Cleveland less ACM puts the world neutral
rate at **−0.62 over 2008-2015 and +1.12 over 2016-2019**: a 1.7 point rise into the ZIRP
and QE era, with the sample trough in the mining-boom years. The derived series has an sd of
**1.08** against Kim-Wright's 0.61, and a neutral rate does not move like that. The
mechanism is visible in the inputs: ACM reads the post-GFC US premium at 1.27 against
Kim-Wright's 0.45, then turns it negative through 2016-2022, so the whole decline in the
10-year real yield lands in the expectations component instead of the premium. That is the
Bauer-Rudebusch-Wu persistence bias arriving as a sign error on the era pattern rather than
as noise.

Downstream it drags r\* with it, era means 1.56 / −0.16 / 1.02 / 0.78 / 1.53 against
2.13 / 0.70 / 0.21 / −0.17 / 1.06, and moves the stance by up to a point. The endpoint
barely notices (r\* 1.42 against 1.23, wedge +0.05 against −0.11). **The wedge is untouched
in shape**, 0.53 / 0.46 / −0.11 / −0.19 / −0.05, and that is the diagnostic: the model
behaved and the input did not.

**The triangulation is what settles it.** The default specification and the Kim-Wright pin
are identified quite differently, a free `b_world` on the raw Cleveland yield with `mu_tp`
floating against an imposed loading on a premium-stripped anchor with `tp` pinned to a
published series, and they agree: their r\* paths correlate at **0.966**, and the pin's
median sits inside the default's 90% band in **every quarter** of the sample. ACM correlates
at **0.442** with the published default, worse than its 0.618 against the pin, and falls
outside that band about a quarter of the time. It is the outlier against two independent
readings, not one of two defensible ones. Era means, default / pin-KW / pin-ACM:
2.42 / 2.13 / 1.56, then 0.56 / 0.70 / −0.16, then −0.52 / 0.21 / 1.02, then
−0.86 / −0.17 / 0.78, then 0.59 / 1.06 / 1.53.

Two honest qualifications. Kim-Wright *also* has the 2008-2015 trough and a rise after it,
so what fails here is ACM's level and amplitude, not the direction on its own. And nothing
in this says Kim-Wright is correct, only that ACM fails a check it passes. Kept as a switch
because it is the cleanest demonstration the package has that the **anchor**, not the
Australian data, sets this model's era pattern. ACM also produced 1 divergence against
Kim-Wright's 0.

**`--curve`: a third window on the belly.** Identifies the premium curve's slope
(`tp_slope` = 0.700 [0.026, 1.323]) and costs r\*: it drives r\* over 2016-19 from −0.29 to
−0.74 and pushes `nu` to 1.87. Forcing three points of one curve onto one state and one
stationary `g` leaves a persistent stretch of low real rates only one affordable home, and
that home is r\*. Kept as the cleanest demonstration of the absorption failure.

**`--short-rate bill`: the 90-day bank bill as the short rate.** Tried and reverted. BBSW is
a bank *credit* benchmark: decomposed against OIS, bill-minus-cash over 2015-19 was +0.26,
of which the expected policy path was −0.05 and bank funding +0.31, and through 2018 the
market expected nothing at all while the bill sat +0.49 over cash. It also breaks this
package's own rule that credit premia stay out of the state. The 30-day bill is worse: it
keeps the contamination and loses the anticipation.

**`--steps`, `--no-world`, `--no-short`, `--assert-stance`, `--impose-world-loading`,
`--deflator trimmed`** are retained comparators. `--no-world` drops pre-GFC r\* to 1.99 and
produces divergences: the Australian bond market alone does not locate r\*.

---

## Limitations

1. **`sigma_walk` is imposed at 0.12.** Chosen because `nu` is an internal check on whether
   the imposed variance is defensible and at 0.08 it was failing. Still a choice.
2. **The level is weakly identified.** `wedge_0` and `mu_tp` at −0.87; the 90% interval on
   r\* straddles zero throughout.
3. **`b_world` = 0.481 is not credible as a pass-through** and is partly absorbing a US term
   premium. See Alternatives.
4. **The anchor is US, not world.** "World r\*" in the code and charts means a US series.
   A GDP-weighted market measure across the US, Euro Area and Canada is the honest version
   and does not exist here.
5. **Indexed AGS are thin**, so the yield carries a liquidity premium a nominal bond does
   not, and `mu_tp` absorbs it inseparably from the term premium proper.
6. **The stance rests on `g` being stationary about one mean**, which is an assumption about
   policy, not a finding.
7. **The QE-era premium is specification-dependent**, as above.
8. **The headline is an endpoint** and nothing tests its stability. See Refinement 1.

---

## Refinements

### 1. Endpoint fragility, the one that matters most

Nothing here tests it and the headline *is* the endpoint. `ystar` has `realtime.py`, which
re-estimates on progressively truncated samples. r\* is a random walk, so its last value is
the least constrained point in the sample. Until this exists, treat every current-quarter
number as provisional.

### 2. Promote the pinned specification

`rstar_pin` is better argued than the default on every axis discussed above. Promoting it
means renaming what the charts call "world r\*", reworking the term-premium language, since
under a stripped anchor the premium is relative rather than absolute, and re-running
everything.

### 3. A GDP-weighted market anchor

The current anchor is one country. Building the Euro Area and Canadian equivalents and
weighting them would make "world" true rather than approximate.

### 4. Reconcile the three routes

`rstar_hlw` Resolution G gives 2.23, this gives 1.08, `rstar_rba` gives 0.99 real on
its own saved run. All three are in this repo and none has been reconciled with the others.
That comparison is the most useful thing left to write.

### 5. Out of sample

The 2016-19 and present-day Taylor readings matched a judgement stated in advance, which is
one observation. A second, stating the expected reading before running it, would be worth
more than any in-sample diagnostic.

### 6. Smaller things

- The corporate spread starts 2005Q1, so `r*` for firms is blank for the first twelve years.
- `end_break_check()` uses a fixed eight-quarter window and a one-sd threshold, both picked
  by eye. Currently +0.63 sds.
- The Taylor coefficients are Taylor's originals and unswept.
- The spec printout in `estimate.py` states the gap without the band scaling used elsewhere.

---

## Files and usage

```
src/models/rstar_bonds/
├── config.py         # ModelConfig: sample, anchor, windows, sigma_walk, the rule
├── observations.py   # the yield, the short rate, the anchor, plus ragged extras
├── estimate.py       # builds and samples the PyMC model
├── results.py        # RStarResults: posteriors, derived series, diagnostics
├── analyse.py        # charts and printed diagnostics
└── run.py            # CLI
```

Run order: `expectations` → `ystar_ustar` → `rstar`. r\* itself needs no upstream model;
only the Taylor rule does, and the affected charts are skipped with a note if inputs are
missing. The anchor needs a FRED API key in `fred.api` (gitignored); see the README.

```bash
./run-rstar-bonds.sh -v
./run-rstar-bonds.sh --analyse-only                 # recharts from the saved trace
./run-rstar-bonds.sh --world-source market --impose-world-loading --us-premium   # the pinned spec
./run-rstar-bonds.sh --world-source mean            # the old HLW anchor
./run-rstar-bonds.sh --sigma-walk 0.08              # watch nu fall into the strain regime
./run-rstar-bonds.sh --curve                        # the third window: watch r* absorb the stance
./run-rstar-bonds.sh --short-rate bill              # the rejected bank bill
./run-rstar-bonds.sh --no-short                     # one window: 168 divergences, for the record
./run-rstar-bonds.sh --steps                        # the asserted-break comparator
./run-rstar-bonds.sh --no-world                     # does the anchor do the work? (yes)
```

Charts land in `charts/RStarBonds/`: r\* against the real yield and the anchor, real and nominal
r\* with bands, the Australian wedge, the policy stance on both neutral definitions, the
policy gap, the risk-free stance against the borrower stance, the term premium with and
without the `k·g` correction, r\* for firms, and the Taylor rule. A non-default prefix gets
its own directory.
