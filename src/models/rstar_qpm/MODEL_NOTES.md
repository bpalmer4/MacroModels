# Trend r\* and short-run neutral from a semi-structural open-economy model

## Overview

A small model of the Australian economy as a small open economy, in the style of the IMF's
Quarterly Projection Model (QPM), hence the package name, `rstar_qpm`. It describes five things
at once: output against its
potential, the exchange rate against its equilibrium, inflation, how the RBA sets the cash
rate, and what the bond market says about where rates settle. Because they are estimated
together, each disciplines the others.

It answers three questions:

1. **Trend r\*:** the real cash rate that is neutral once today's shocks have faded.
2. **Short-run neutral:** the real cash rate that would close the output gap within three
   years, given the shocks hitting the economy now.
3. **Transmission:** how much, and through which channel, a change in the cash rate moves
   output.

It is estimated on 1993Q1 to 2026Q2 (134 quarters) as a linear Gaussian state space: a Kalman
filter infers the unobserved quantities, and NUTS samples the 24 parameters.

**Its central finding is that the IS curve is weak**: interest rates move Australian output
only a little, and the data, not the prior, say so. See "What it finds" below.

## The equations

```
Trends (slow states)
  y*_t   = y*_{t-1} + g_{t-1}/4 + e             potential, 100 x log GDP
  g_t    = g_{t-1} + e                          trend growth, annualised
  r*_t   = rw_t + w_t,   w_t = w_{t-1} + e      world real rate (data) + AU wedge
  q*_t   = q*_{t-1} + e                         equilibrium real TWI, 100 x log

Gaps
  ygap_t = b1 ygap_{t-1} - b2 (r_{t-1} - r*_{t-1}) - b3 qgap_{t-1} - b4 dDSR_{t-1} + e   IS curve
  qgap_t = rho_q qgap_{t-1} + kappa (r_t - r*_t) + e                                    UIP, in gaps
  dDSR_t = delta E_t (i_t - i_{t-1}) + e                                                cash flow

Observed
  y_t  = y*_t + ygap_t
  q_t  = q*_t + qgap_t
  pi_t = a1 pi_{t-1} + (1 - a1) pie_t + a2 ygap_{t-1} + a3 (m4_{t-1} - pie_{t-1}) + e
  i_t  = rho_i i_{t-1} + (1 - rho_i)(r*_t + pie_t + phi_pi (pi4_t - 2.5) + phi_y ygap_t) + e
  f_t  = r*_t + bias + e
```

`r` is the real cash rate on measured expectations, `pie` long-run inflation expectations,
`rw` the Cleveland Fed 10-year expected real rate less the Kim-Wright US term premium, `q` the
RBA real TWI (up is appreciation), `pi` quarterly trimmed mean annualised, `pi4` year-ended,
`m4` year-ended consumption import price inflation, `f` the AOFM 5y5y risk-neutral forward
less `pie`, `DSR` household interest payments over disposable income, and `E` last quarter's
DSR over the standard variable mortgage rate, which is household debt relative to income. Sample 1993Q1 to 2026Q2, 134 quarters.

## How it works, in plain English

**Potential and the gap.** GDP is split into potential output, which drifts upward at a trend
growth rate that itself changes slowly, and an output gap, the cyclical part. Neither is
observed; only their sum, GDP, is.

**Interest rates reach output through an IS curve.** When the real cash rate sits above trend
r\*, policy is tight, and next quarter's output gap is pushed down (`b2`). The gap is also
persistent: part of this quarter's gap carries into the next (`b1`).

**And through the exchange rate.** A real cash rate above trend r\* makes Australian assets
more attractive, so the real exchange rate appreciates (`kappa`), and a high dollar drags on
output the following quarter (`b3`). This is the open-economy channel a single-equation IS
curve leaves out.

**And through repayments.** When the cash rate rises, interest payments on household debt
rise with it (`delta`), by more the more debt households carry relative to income (`E`), and
the extra repayments squeeze spending the following quarter (`b4`). This is the cash-flow
channel. It works on the nominal cash rate and the stock of debt, which a real rate gap
cannot capture.

**Inflation follows the gap.** Quarterly inflation is a mix of its own past and of expected
inflation, pushed up when output is above potential (`a2`) and when import prices outpace
expectations (`a3`). This is what lets inflation tell the model where the gap is.

**The RBA follows a rule.** The cash rate moves gradually (`rho_i`) toward a target: neutral
nominal (trend r\* plus expected inflation), raised when inflation is above 2.5 (`phi_pi`) or
output above potential (`phi_y`). Modelling the rule matters: rates respond to the gap, so
without it the model would read the Bank's reaction as the economy's.

**The market pins trend r\*.** Trend r\* is the world real rate plus an Australian wedge that
drifts slowly. The AOFM's 5y5y forward, a market price for where the cash rate settles five to
ten years out, is observed as trend r\* plus a small bias and noise. How fast the wedge may
move is imposed, and slow: the forward then sets the level on average, while its
quarter-to-quarter wiggles go to its noise and the rest of the system shapes the path.

**How the numbers come out.** For any set of parameters, the Kalman filter works out the most
likely path of every unobserved quantity (potential, the gap, the wedge, the exchange-rate
gap) and how well the whole system then fits the data. NUTS explores which parameters fit
well, weighed against their priors. After sampling, 1,000 parameter draws are each pushed back
through a simulation smoother to draw the unobserved paths, so the bands carry uncertainty
about both.

**Short-run neutral is computed, not estimated.** At each quarter the model is run forward
with every shock switched off and the real cash rate held constant; short-run neutral is the
constant rate at which the gap is zero twelve quarters on.

## What it finds

**The IS curve is weak.** Rates move output, but not by much: a real cash rate held above
trend r\* takes only a small bite out of the output gap even after three years, and the rate
term explains almost none of the gap's movements, which come from the gap's own persistence
and its shocks. The data pulled the IS slope and the exchange-rate response well below where
the priors put them, and the recovery test shows the model would have found strong
transmission had it been there, so the weakness is the data's, not the prior's. What the
model cannot say is that rates have NO effect or the wrong sign: the sign is imposed.

**The total is identified; how it divides between channels is not.** The recovery test gets the
overall effect of a rate rise on output right, in a weak world and a strong one, but not the
exchange-rate block's two parts: it consistently trades a smaller response of the exchange rate
to rates for more persistence in the exchange-rate gap, and the pair is recovered only in
combination. So the split of transmission between the direct rate channel and the exchange-rate
channel, and the exchange rate's response to rates on its own, should not be quoted. Most
movement in the TWI is booked to its equilibrium rather than to its gap.

**The cash-flow channel is real but brief, and does not rescue the IS curve.** Debt servicing
follows the cash rate closely and that link is well pinned down. The drag it puts on output
is small, well below where its prior put it, and it fades within a couple of years: in the
first quarters after a rate rise it matters as much as the direct rate channel, but it adds
almost nothing to the effect after three years. So the weak IS curve is not an artefact of
lumping the cash-flow channel in with the real rate. Part of the fade is a modelling choice:
the IS curve reads the CHANGE in debt servicing, so a permanently higher repayment burden
squeezes spending once rather than for as long as it lasts.

**The level of trend r\* comes from the market; the path is the model's.** Only the forward
can pin the level. Remove it and the rest of the system cannot find one: trend r\* drifts toward
the cash rate, which is the policy rule reading policy back. But left free to move quickly, the
wedge follows the forward quarter by quarter, the forward is fitted almost exactly, and trend
r\* simply is the forward, with none of the structure in it. That was so with the IS curve on
or off, so the IS curve was not what let the forward win; the wedge's speed was. With the
wedge slowed, the forward sets the level on average and the structure and the world rate shape
the path, most visibly in 2022-23, when the world real rate rose and the Australian forward
did not follow.

**The weak IS curve depends on that anchor.** Without the forward, the IS slope and the
multiplier roughly double, because a trend r\* that tracks the cash rate leaves smaller rate
gaps to explain the same swings in output. Transmission is weak on either anchor, but how weak
is conditional on where r\* is pinned.

**Short-run neutral stays close to trend r\*.** The output gap is small and short-lived, so the
momentum short-run neutral has to offset is small, and it rarely departs far from trend r\*.
With trend r\* slowed, the recovery test recovers those departures well, in a world like the
one estimated as well as a strong one. Their size is small either way, and it rests on an IS
curve that is not load-bearing.

**How smooth potential is comes from a prior.** The data cannot separate potential from the
gap on their own, so the potential-growth path reflects that choice as much as the data.

## Technical notes

**Expectations are measured, not model-consistent.** That keeps the model a plain linear state
space whose likelihood NUTS can differentiate. The cost is that nothing here is forward-looking
in the rational-expectations sense: the exchange rate responds to today's rate gap, not to the
expected path.

**The cash rate is both an input and an observation.** It drives the IS curve and the exchange
rate as data and is scored by the rule. The map from the rule and UIP shocks to the cash rate
and the TWI is triangular with a unit diagonal, so the product the filter computes is the
joint density.

## What is imposed

- **Signs.** `b2`, `b3`, `b4`, `kappa` and `delta` have priors truncated at zero, so the
  posterior cannot report a wrong-signed IS curve, exchange-rate or cash-flow channel.
- **How fast trend r\* may move.** The wedge's innovation sd is imposed at 0.10 a quarter. Left
  free, it runs fast enough for trend r\* to copy the forward, so the structure has no say; slowed,
  the forward sets the level on average and its short-run wiggles go to its noise. The value
  decides how much room the structure gets, so it is a choice, not an estimate.
- **The pass-through is one quarter.** Debt servicing responds to the cash rate in the quarter it
  moves. Spreading it over three quarters was tried and changed nothing.
- **Debt exposure.** `E` uses the standard variable mortgage rate because it runs from 1959;
  the discounted rate borrowers actually pay starts only in 2004. The standard rate overstates
  what is paid once discounts widened, so exposure is understated in later years. `delta`
  absorbs the level of that error but not its drift. Exposure is held at its latest value
  when projecting short-run neutral and the transmission charts.
- **How smooth potential is.** `sigma_ystar` is not identified: the recovery test cannot pull
  it back from its prior, and left loose the likelihood favours potential absorbing the cycle.
  Its prior, InverseGamma with mean 0.17 and sd 0.05, therefore sets it. The InverseGamma has
  no mass at zero, which removes the rigid-potential end of the ridge between potential and
  the gap that the sampler otherwise wanders.
- **Exclusions.** GDP leaves the likelihood over 2020Q2 to 2021Q3, so potential and the gap
  are interpolated through the lockdowns. The rule leaves it wherever the quarterly cash rate
  averaged below 0.5 (2020Q1 to 2022Q1): at the lower bound the rule's prescription was not
  deliverable.
- **The horizon.** Short-run neutral closes the gap in 12 quarters. ASSUMPTION: that the
  FRB/US-based measures use a three-year horizon has not been checked against a Fed source.

## Short-run neutral

At each quarter, hold the real cash rate at a constant level from the next quarter, hold the
world rate and the wedge where they are, switch the shocks off, and project the IS curve and
exchange rate forward. Short-run neutral is the constant rate at which the output gap is zero
12 quarters on. The projection is linear, so it is solved:

```
ygap_{t+12}(x) = A + B x,     x = rate held - r*_t,     short-run neutral = r*_t - A / B
```

`A` is the momentum the current gaps leave behind; `B` is the multiplier. **B divides**, so a
small multiplier turns modest momentum into large departures from trend r\*. `B` is saved per
draw and charted against its prior, and should be read beside short-run neutral.

## The recovery test

`--recovery` simulates an economy from the model at known parameters, with the cash rate
generated by the rule each quarter and fed back into the IS curve and the exchange rate, so
the thermostat problem is present exactly as the model says it is in the data. Exogenous
series are the real ones. It then re-estimates with the same priors.

It runs twice: once with the real-data posterior means as the truth, and once with strong
transmission (`b2` = 0.25, `kappa` = 3), to see whether the model can find transmission that is
really there. For each it prints every parameter's truth against its estimate, the multiplier,
and the correlation with the truth of the output gap and of short-run neutral less trend r\*.

Read the last of those, not the correlation of short-run neutral itself: the level passes on
trend r\* alone, which the forward pins in truth and estimate alike.

What it establishes: the model separates weak transmission from strong, so a weak real-data
reading is informative, and short-run neutral's departures from trend r\* are recovered well.
Even in the strong case the IS slope is shrunk toward its prior, so the real-data slope may be
pulled up toward its prior too. And the exchange rate's response to rates and its persistence
are not separately recovered: the estimate trades one for the other, getting only their
combined effect right.

## Diagnostics

Each run writes `run-diagnostics-rstar_qpm.txt` into its chart directory: R-hat, ESS, MCSE,
divergences, tree depth and BFMI against the repo's thresholds, and the run's headline
numbers. Charts carry any sampling failure in their header.

## Running it

```bash
./run-rstar-qpm.sh                 # estimate, charts to charts/RStarQPM/
./run-rstar-qpm.sh --analyse-only  # charts from the saved run
./run-rstar-qpm.sh --horizon 8     # short-run neutral at another horizon
./run-rstar-qpm.sh --recovery      # also the recovery test, charts to charts/RStarQPM_recovery/

# Comparisons, each under its own prefix so the default run is untouched
./run-rstar-qpm.sh --sigma-w free --prefix rstar_qpm_free     # wedge estimated, not clipped
./run-rstar-qpm.sh --sigma-w 0.05 --prefix rstar_qpm_w005     # another clip
./run-rstar-qpm.sh --no-forward --prefix rstar_qpm_nofwd      # no forward: no usable level
./run-rstar-qpm.sh --no-is --prefix rstar_qpm_nois            # IS curve off
```

Charts for a prefixed run go to `charts/RStarQPM_<prefix>/`.

Outputs: `model_outputs/rstar_qpm_trace.nc` (parameters) and `rstar_qpm_states.nc` (the
observations, 1,000 simulation-smoother state paths, the multipliers and the run's settings).
