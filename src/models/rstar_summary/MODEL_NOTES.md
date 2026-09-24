# The r\* summary: every model on one nominal scale

**Not a model.** It estimates nothing and adds nothing to the evidence. It loads the other
packages' saved runs, re-runs any that are not from today, converts them all to a nominal
scale, and charts them together.

Its purpose is the **disagreement**. Each of those models anchors r\* to a different thing
and returns roughly what its anchor implies, so the spread between the lines is three
structural assumptions, not sampling error.

## What the package concludes

**In Australia, r\* is a market-and-behaviour concept, not an IS-equilibrium one.** The models
here say where markets and the RBA's conduct put neutral. None of them can say, with any
weight, where the rate sits that closes the output gap, because the link that concept needs,
an IS curve from interest rates to output, cannot carry that weight on Australian data.

**The IS curve is not load-bearing.** The repo has found this several independent ways. No
single-equation IS curve recovers even the sign of the rate effect. A model whose r\* rests on
the IS curve cannot identify r\* at all. And in the semi-structural model the rate term explains
almost none of the output gap's movements, while switching the IS curve off leaves the level of
r\* where it was.

**That is a statement about measurement, not a finding that monetary policy does not work.** The
semi-structural model's recovery test shows it would detect strong transmission if the data held
it, so the weakness is real within its assumptions. But those assumptions include a linear curve
with its sign imposed, measured rather than model-consistent expectations, and a central bank
that offsets demand shocks and so removes the very correlation the IS curve is estimated from.
How weak transmission looks also depends on where r\* is anchored. The defensible claim is that
Australian data do not let an IS curve pin neutral down.

**So the three lines are three ways of shaping a market-anchored level, not three independent
estimates.** Every model on the chart leans on the AOFM 5y5y forward for its level, two of them
almost entirely. They differ in what shapes the path around that level: the bond model balances
three market windows against a world real rate; the reaction-function model reads the Bank's
response to inflation; the semi-structural model lets its structure and the world rate move a
slow-moving trend. Their agreement is partly one series counted more than once. Their
disagreements are the informative part: where the world rate and the Australian forward part
company, as in 2022-23, the lines separate.

**Short-run neutral inherits the weakness.** It is the rate that closes the output gap within a
horizon, so it is IS-based by construction, and it should not be leaned on.

## What is on the chart

| line | anchored to | what to distrust |
|---|---|---|
| Bond market (`rstar_bonds`) | a premium-stripped world real rate, plus an AU wedge, with the AOFM 5y5y forward pinning the level | the wedge is four times jumpier since the forward went in, so some of the path is bond-market noise booked as r\*; `forward_bias` is uninterpreted |
| RBA reaction function (`rstar_rba`) | the Bank's response to inflation away from target, with the same 5y5y forward as a second window | the level is conditional on `sigma_r`, though the forward cut that dependence from a 1.10 spread to 0.31 |
| Semi-structural open economy (`rstar_qpm`) | trend r\*: a world real rate plus an AU wedge, inside an IS / exchange-rate / Phillips / policy-rule system, with the same 5y5y forward as an observation | how fast the wedge may move is imposed, and that choice decides how much the path departs from the forward; the level is still the forward's on average |
| ~~TVP-VAR (`rstar_tvpvar`)~~ | REMOVED 2026-09-17 | see below |

Both are conditional. That is not a reason to prefer one: it is the state of the
literature on Australian data.

**`rstar_tvpvar` was REMOVED on 2026-09-17**, the day it was stripped back to the canonical
Lubik-Matthes specification. Not discredited: not ready to carry a level. Its fitted VAR has a
median spectral radius of 0.983, and that one number spoils every estimand open to it. At the
20-quarter horizon 0.983^20 = 0.71 of today's state survives, so the projection is mostly a
nowcast and correlates 0.953 with the real cash rate; push the horizon out and 32.3% of
draw-quarters are explosive; take the infinite-horizon limit and `(I - F)^-1` divides by
almost nothing. Across its own `sigma_q` the sample-mean level holds (1.34 to 1.66) and the
2016-19 sign holds (-0.65 to -1.07), but r\* **latest** runs 1.08 to 3.17, and the latest value
is the only thing this chart plots. Sampling was ruled out as the cause: at `target_accept`
0.99 divergences fall 5 to 1 and min ESS rises 368 to 522, while r\* moves 0.03pp and the
spectral radius does not move at all. See `sources.py` for what would bring it back.

**`rstar_invert` was REMOVED from this summary on 2026-09-16.** It inverted an asserted IS
curve, and the repo's own evidence is that no such curve is identifiable on Australian data:
five methods here fail to recover even its sign. They rest on three distinct output gaps
rather than five, since `is_curve` loads the same `ystar_ustar` gap that `rstar_invert`
inverts, so "five independent" overstated it (corrected 2026-09-17). Charting a line whose every
value follows from a relationship nobody in this package believes in gave the summary a
fourth "answer" that was really a restatement of its own assumption. The package remains,
with its notes, as the record of that attempt. Sections below written when it was on the
chart still refer to it.

**Both remaining real-rate models now read the same AOFM 5y5y forward**, which is worth
saying plainly because it weakens the independence the chart trades on. `rstar_bonds` and
`rstar_rba` agreeing about the level is now partly the same observable speaking twice.

### The nominal conversion

`rstar_rba` records `neutral` in nominal terms already. The other two are real and have
**unanchored long-run inflation expectations** added.

**This changed on 2026-09-16.** It used to add the flat 2.5% target, on the ground that a
neutral rate is defined at target inflation. The reason for changing is comparability: the
RBA and CBA both convert using long-run expectations, so a chart built on the target was
never quite like for like against a published neutral rate.

The change is small and lands where it should. Over 1993Q1 onward the anchored series runs
2.13 to 3.50 with an sd of 0.28:

| 1993-99 | 2000-07 | 2008-15 | 2016-21 | 2022- |
|---|---|---|---|---|
| 2.98 | 2.50 | 2.60 | 2.32 | 2.57 |

So it is worth about +0.5pp through the 1990s re-anchoring, where expectations genuinely sat
above target and the old convention understated every nominal path on this chart, and close
to nothing after 2000.

**It is the ANCHORED series, not the unanchored one.** The unanchored median moves with the
cycle (1.72 at its trough, 3.32 in 2023), and converting a neutral rate with it would drag
the inflation cycle into r\*: nominal r\* would have fallen to 0.51 in 2020Q4 purely because
expectations dipped. That is the same objection `rstar_rba`'s notes raise against deflating
by realised inflation, and the anchored series does not attract it, as its 2022- mean of 2.57
shows.

**It creates a dependency.** This package now needs a completed `./run-expectations.sh`, and
it fails loudly rather than falling back, because a silent fallback would publish one
convention under the label of another. `--nominal-on target` restores the old behaviour, so
every previously published number stays reproducible.

The convention lives in [`src/models/common/inflation_scale.py`](../common/inflation_scale.py)
and is used in both directions: `rstar_bonds` and this package convert real to nominal,
`rstar_rba` converts nominal to real.

Note it reads `rstar_rba`'s `neutral` (the slow base `b_t`), **not** `prescribed`, which adds
the Bank's inflation response on top and is not a neutral rate.

## Why `rstar_hlw` is not here

It was, and it was removed on 2026-09-11. **No resolution of that model produces an
identified r\* path**, so any line it contributed would be a picture of a prior.

- **A (canonical)**: `sigma_z` is free with a HalfNormal(0.10) prior and posteriors at 0.0657
  against a prior median of 0.0674. The data says nothing about how far z moves. The flat
  path is the median of many draws each wandering differently, not a finding that r\* held
  still. ESS on `r_star` is 243.
- **B**: 3,349 divergences, and its own notes call z "wild".
- **C, E, F, G, H**: all rest on `alpha`, which posteriors at 0.58 with a 90% interval of
  [0.03, 0.99] and is bimodal. Those notes are explicit that the blended median is the
  average of two stories almost no single draw sits at.

**The distinction that justifies keeping the other three**: they give conditional answers
whose conditioning can be stated. HLW gives no answer at all, since its posterior is its
prior. That is a difference in kind.

What HLW does establish belongs in prose, not on a chart of levels: the per-quarter signal is
`a_r/sigma_IS` = 0.044/0.685 = 0.064, so r\* would have to be wrong by about 15pp to move the
likelihood by one standard deviation. See `src/models/rstar_hlw/MODEL_NOTES.md`.

## The vintage check

A saved trace counts as **current if the file was written today**. That is a proxy for the
data being current, not a check of the underlying ABS and RBA vintages: a model re-run today
picks up whatever those series then hold. It is conservative in the right direction, since a
stale file is always re-run and a fresh one never is.

**Refreshing runs the other model's full pipeline.** `refresh()` calls that package's own
`run-*.sh` with no arguments, which is estimate *and* analyse, so it regenerates that model's
entire chart directory too. For `rstar_rba` it also runs the injection test and the `sigma_r`
ensemble. This takes minutes and overwrites files outside this package, so it is announced
before it happens and `--no-refresh` turns it off.

## Mean, not median

The spread chart draws the range across models and a central line. That line is the **mean**.

With three series the median is whichever model happens to sit in the middle that quarter, so
it switches identity wherever the lines cross (around 2001, 2010 and 2019) and picks up kinks
that say nothing about r\*. It also discards two thirds of the information at every point.

**Neither is an estimate.** An average across structural assumptions is a value no model
produces, which is exactly the objection `rstar_hlw`'s notes make to its own blended median.
It describes where the models sit.

## Reading the charts

As at 2026Q2: bond market **3.33**, RBA reaction function **3.89**, midpoint **3.61** nominal.
The two span 0.56pp.

**Quarters in progress are dropped**, in `sources._drop_incomplete`, using the shared
`last_complete_quarter()` rule. `rstar_bonds` reads bond yields daily and so produces an
estimate for the unfinished quarter, where the reaction function waits on GDP and the output
gap. Charting it put a fortnight's average beside 30 years of whole quarters and left the two
endpoint labels on different dates, inviting a subtraction that corresponded to nothing. Both
lines now end on the same finished quarter.

That does NOT align the models in general. Once the quarter closes, `rstar_bonds` will have it
and the reaction function will not, so the labels will sit on different dates again. **Do not
average across quarters** when they do.

For the record, the removed TVP-VAR line read 3.92 nominal at 2026Q2 on the canonical spec,
which would have widened the span to 0.59pp. It is not in the numbers above.

For scale against a published number: CBA's September 2026 nominal neutral is **3.85**, inside
this range and nearest `rstar_rba`.

All three tell the same broad story: a fall from around 5.5 to 6% in the mid-1990s to a trough
in 2015-2021, and a recovery since.

**The bonds line moved on 2026-09-16 and its shape changed, not just its level.** Adding the
5y5y forward window lifted it from 3.06 to 3.33 at 2026Q2 and, more importantly, abolished its
negative stretch: 2016-2019 and 2020-2021 go from −0.11 and −0.70 real to +0.42 and +0.14. The
deflated forward never went negative, and a window loading directly on r\* will not let r\* go
where the observable does not.

### The band understates

It is the range across three particular modelling choices, not a sample from anything, so a
fourth reasonable model could sit outside it. And each line is itself conditional on a number
nobody measures: `rstar_rba`'s `sigma_r` and `rstar_bonds`' imposed `nu_walk` are both choices
rather than estimates. `rstar_tvpvar` failed that test outright, which is why it is no longer
here.

The band is also narrower than it looks, because **both remaining lines read the same AOFM
5y5y forward**. Some of the agreement between `rstar_bonds` and `rstar_rba` is one observable
counted twice rather than two methods converging. That was a caveat when there were three
lines; with `rstar_tvpvar` removed on 2026-09-17 it applies to the whole chart, which is why
the spread panel now says so in its header. The mean line is, at n = 2, the arithmetic
midpoint of the band and carries nothing the band does not already show.

So the chart maps the landscape of *published* possibilities. The landscape of defensible
ones is larger.

## The proxies chart

`plot_proxies` draws **Macroeconomic proxies for nominal r\***: the AOFM 5y5y risk-neutral
forward, trend real GDP per capita growth plus long-run inflation expectations (the same
conversion the model lines use), and the cash rate behind
both. It was called `plot_forward_against_cash` and titled "The 5y5y forward and the cash
rate" until the growth line went on.

**No model output, and that is the point.** Everything here was produced by the world rather
than estimated in this repo. The two proxies are the standing reference points a neutral rate
gets judged against: one a market price, one the golden-rule statement that nominal neutral
is real growth plus expected inflation. Neither is r\*. The cash rate is
behind them because it is what both are a comparison for.

As at 2026Q2: forward **3.80**, growth proxy **3.27**, cash rate **4.35**. Against the models
on the summary chart, 3.33 to 3.89, the growth proxy sits just under the bond-market line.

**The two proxies disagree as readily as the models do.** Forward less growth proxy is +0.53pp
latest, −0.42pp on average, and has been **1.58pp** at its widest (1993Q1), nearly three times
the 0.56pp span across the models. Agreement between them now is not a general fact about
them.

### Per capita, not aggregate

This decides the answer rather than decorating it. The consumption-Euler link that makes r\*
track g is about growth per head; the version quoted in passing is aggregate. Australian
population growth sits between them and is worth roughly 1.2pp, **larger than the entire
spread across the r\* models on these charts**. Aggregate potential growth runs near 1.9 (see
[`gstar_summary`](../gstar_summary/MODEL_NOTES.md)), so aggregate-plus-target would put this
line about 4.4 and above every model on the chart; per capita puts it at 3.24, among them.
Switching the series would not shift the line, it would change what the chart says.

### The flat target, not anchored expectations

The one place this package departs from [its own convention](#the-nominal-conversion), and
deliberately. `to_nominal` converts the SCALE of an *estimated* real neutral rate, and matches
what the RBA and CBA publish. This is not an estimate being converted: the golden rule says
nominal neutral is real growth plus the inflation being targeted, so the target IS the second
term. Substituting what people expect would make the benchmark drift with sentiment. The two
conventions differ by up to about 0.5pp through the 1990s re-anchoring and little since.

**So one chart carries two nominal conventions.** The models on the summary chart use anchored
expectations; the growth proxy here uses the flat 2.5. They are not interchangeable and the
difference is not zero.

### The window

A 40-quarter rolling mean of year-ended growth, `_TREND_WINDOW`. Ten years, so it spans a
cycle and neither the mining boom nor the pandemic can own it: at 40 quarters the trend runs
0.63 to 2.84 over its life, where 20 quarters runs 0.11 to 3.18 and reads as a cycle rather
than a trend. It is still a backward-looking mean, so the mid-2000s hump is the boom sitting
inside the window rather than a contemporaneous belief about neutral.

The series is ABS 5206.0 key aggregates, "GDP per capita: Chain volume measures", seasonally
adjusted, via `src.data.gdp.get_gdp_per_capita`. Quarterly back to 1959Q3, so the trend starts
1984Q2 and covers the whole 1993Q1-on chart window.

## Usage

```bash
./run-rstar-summary.sh                  # refresh anything stale, then chart
./run-rstar-summary.sh --no-refresh     # chart the saved runs as they stand
./run-rstar-summary.sh --start 2000Q1   # shorter window
```

Adding a model means one entry in `SOURCES` in `sources.py`: a label, the saved prefix, the
run script, a loader, whether it is already nominal, and a one-line note saying what its r\*
is anchored to. That note is printed on every run and is not optional.
