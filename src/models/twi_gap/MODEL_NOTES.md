# The real TWI gap from commodity prices

## Overview

Exploratory, not a structural model. The question: is the exchange rate adding
to or offsetting what the cash rate is doing? The answer is drawn as a
backplane behind the cash rate: each quarter shaded by how far the real TWI
sits from the level commodity prices would explain.

It measures the exchange rate's part in monetary conditions, not the policy
stance. A dollar that is high *because* commodity prices are high counts as
neutral, so the peak of 2007-08 (cash rate 7.25%) is barely shaded.

Quarterly, 1993Q1 on. Real TWI from RBA table F15 (quarterly, March 1995 =
100); the RBA index of commodity prices in US$ from table I2 (monthly,
averaged to quarters); the cash rate target from A2, charted monthly as steps.
A dotted line marks the last quarter of the gap, since the cash rate runs on
past the real TWI and commodity price data.

## Equation

$$
\ln \text{RTWI}_t = \alpha + \beta \ln \text{ICP}^{US\$}_t + \varepsilon_t
$$

by OLS. The gap is $100\,\varepsilon_t$: per cent above (+) or below (-) the
commodity-implied level. Estimates over 1993Q1-2026Q2 (134 quarters):
$\beta = 0.28$ (a 10% rise in US\$ commodity prices goes with a 2.8% real
appreciation), $R^2 = 0.82$, gap sd 7.1%.

**Cointegration.** Engle-Granger p = 0.010. Both series wander; the test says
they wander together, so the gap returns towards zero. Without that, the
residual would be a random walk and "above" or "below" would mean nothing.
The OLS standard error on $\beta$ is printed but is not valid for inference
on cointegrated levels (the estimate itself is consistent).

## Specification choices

- **Real, not nominal, TWI.** Over three decades the nominal TWI drifts with
  inflation differences against trading partners.
- **US\$, not A\$, commodity prices.** The A\$ index is the US\$ prices
  converted at the exchange rate, so the dollar would sit on both sides.
- **Commodity prices, not the terms of trade.** The terms of trade carry
  import prices too.

Checked on 1993Q1-2026Q2 and rejected (Engle-Granger p): nominal TWI on the
terms of trade (0.171), nominal TWI on commodity prices (0.130), real TWI on
the terms of trade (0.450). Only real TWI on US\$ commodity prices passes.

The bulk-commodity spot price index (I2, GRCPAISUSD) was also tried. It starts
in 2009Q1, and on that short sample the relationship does not pass
(p = 0.247).

## Robustness

`run.py` refits from start years 1983 to 2002. On the data to 2026Q2 the
Engle-Granger p-value stays below 0.05 for every start (0.003 to 0.041),
$\beta$ stays between 0.22 and 0.28, and the latest gap between +4.8% and
+6.0%.

## The backplane

Each quarter's gap in standard deviations, $z_t = \varepsilon_t / \text{sd}$:

- within $\pm 0.75$ sd (about $\pm 5.3\%$): clear. About half the quarters.
- 0.75 to 1.6 sd: light red, "Tighter FX" (dear dollar), or light blue,
  "Looser FX" (cheap dollar).
- beyond 1.6 sd (about $\pm 11.4\%$): dark, "Much tighter FX" or "Much looser
  FX". About the most extreme tenth.

The legend speaks of FX conditions, not policy: a looser reading can come from
risk aversion (2008Q4) as readily as from interest rates.

Red tightens conditions (a dear dollar squeezes exporters and import-competing
firms); blue eases them. The chart quotes the band shares from the data,
rounded to the nearest 10%.

## What the gap contains

Everything that moves the dollar other than commodity prices:

- interest-rate differentials: the monetary channel, left in deliberately. The
  RBA's 1990s real exchange rate equations (UNVERIFIED reference, from memory:
  Blundell-Wignall et al. 1993; Gruen and Wilkinson 1994) put a real interest
  differential on the right-hand side; here it is kept out so its effect stays
  in the gap.
- risk sentiment and capital flows: the 2008Q4 low (-22%) is a flight from
  risk, not a policy setting.
- possibly measurement (UNVERIFIED): iron ore and coal were priced on annual
  benchmark contracts until about 2010, so the index may have lagged spot
  prices, flattering the gap in 2004 and 2009.

## Reading it

- Dear dollar against falling rates: 2012-16, the dollar 6-12% above its
  commodity level while the cash rate fell from 4.25% to 1.5%.
- Cheap dollar through the end of emergency settings and the first hikes:
  2021Q3-2023Q1, beyond 1.6 sd for seven quarters running.
- Near neutral now: +0.68 sd in 2026Q2, so the cash rate is doing the
  tightening.

## Limits

1. Neutral is "average for the sample": the constant forces the gap to average
   zero over 1993-2026.
2. Full-sample fit: each new quarter re-estimates $\alpha$, $\beta$ and the sd,
   so the history moves a little.
3. One $\beta$ across the mining boom and its unwinding.
4. Static: no lags or adjustment dynamics.
5. F15 and I2 lag the cash rate, so the shading stops before the line does.
