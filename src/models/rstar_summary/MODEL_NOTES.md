# The r\* summary: every model on one nominal scale

**Not a model.** It estimates nothing and adds nothing to the evidence. It loads the other
packages' saved runs, re-runs any that are not from today, converts them all to a nominal
scale, and charts them together.

Its purpose is the **disagreement**. Each of those models anchors r\* to a different thing
and returns roughly what its anchor implies, so the spread between the lines is three
structural assumptions, not sampling error.

## What is on the chart

| line | anchored to | what to distrust |
|---|---|---|
| Bond market (`rstar_bonds`) | a premium-stripped world real rate, plus an AU wedge, with the AOFM 5y5y forward pinning the level | the wedge is four times jumpier since the forward went in, so some of the path is bond-market noise booked as r\*; `forward_bias` is uninterpreted |
| RBA reaction function (`rstar_rba`) | the Bank's response to inflation away from target, with the same 5y5y forward as a second window | the level is conditional on `sigma_r`, though the forward cut that dependence from a 1.10 spread to 0.31 |
| TVP-VAR steady state (`rstar_tvpvar`) | the VAR's own long-run mean | the steady state is undefined for a quarter of draws: 26.6% of draw-quarters are explosive, median spectral radius 0.967 |

All three are conditional. That is not a reason to prefer one: it is the state of the
literature on Australian data.

**`rstar_invert` was REMOVED from this summary on 2026-09-16.** It inverted an asserted IS
curve, and the repo's own evidence is that no such curve is identifiable on Australian data:
five independent methods here fail to recover even its sign. Charting a line whose every
value follows from a relationship nobody in this package believes in gave the summary a
fourth "answer" that was really a restatement of its own assumption. The package remains,
with its notes, as the record of that attempt. Sections below written when it was on the
chart still refer to it.

**Both remaining real-rate models now read the same AOFM 5y5y forward**, which is worth
saying plainly because it weakens the independence the chart trades on. `rstar_bonds` and
`rstar_rba` agreeing about the level is now partly the same observable speaking twice.

### The nominal conversion

`rstar_rba` records `neutral` in nominal terms already. The other two are real and have
**target-anchored long-run inflation expectations** added.

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

As at 2026Q2: bond market **3.33**, RBA reaction function **3.89**, TVP-VAR steady state
**3.96**, mean **3.73** nominal. The three span 0.63pp. `rstar_bonds` alone runs a quarter
further, to 2026Q3, because it reads bond yields while the other two need GDP and the output
gap; there it reads 3.57. **Do not average across quarters.** Mixing the bonds 2026Q3 value
with the other two at 2026Q2 gives 3.81 rather than 3.73, and that difference is calendar
arithmetic, not a finding.

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
rather than estimates, and `rstar_tvpvar`'s steady state does not exist at all for 26.6% of
its draw-quarters.

The band is also narrower than it looks, because two of the three lines now read the same AOFM
5y5y forward. Some of the agreement between `rstar_bonds` and `rstar_rba` is one observable
counted twice rather than two methods converging.

So the chart maps the landscape of *published* possibilities. The landscape of defensible
ones is larger.

## Usage

```bash
./run-rstar-summary.sh                  # refresh anything stale, then chart
./run-rstar-summary.sh --no-refresh     # chart the saved runs as they stand
./run-rstar-summary.sh --start 2000Q1   # shorter window
```

Adding a model means one entry in `SOURCES` in `sources.py`: a label, the saved prefix, the
run script, a loader, whether it is already nominal, and a one-line note saying what its r\*
is anchored to. That note is printed on every run and is not optional.
