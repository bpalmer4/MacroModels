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
| Bond market (`rstar_bonds`) | Cleveland Fed 10y expected real rate, plus an AU wedge | the LEVEL is not identified: `wedge_0` against `mu_tp` at −0.87, and `b_world` of 0.481 is not credible as a pass-through |
| RBA reaction function (`rstar_rba`) | the Bank's response to inflation away from target | the level is conditional on an arbitrary `sigma_r`, running −0.05 to 1.05 across defensible values |
| IS inversion (`rstar_invert`) | **nothing** | the path is decided by the asserted speed of r\*; its 2007 peak spans 4.0 to 7.0 real across `sigma_rstar` alone |

All three are conditional. That is not a reason to prefer one: it is the state of the
literature on Australian data.

### The nominal conversion

`rstar_rba` records `neutral` in nominal terms already. The other two are real and have the
**2.5% target** added, because a neutral rate is defined at target inflation rather than at
whatever inflation happened to be. Same convention `rstar_rba` settled on internally.

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

As at 2026Q2: bond market 3.58, RBA reaction function 2.99, IS inversion 4.03, mean **3.47**
nominal, about 0.97 real. The three span 1.03pp; the widest they have ever been is 2.12pp in
2005Q4.

All three tell the same broad story: a fall from around 5.5 to 6% in the mid-1990s to roughly
1.3 to 1.5% by 2015-2021, and a recovery since. They differ mainly in **timing**: the IS
inversion peaks in 2006 and troughs in 2015, while the two anchored models trough around
2021. That lead is plausible rather than suspicious, since the IS inversion's r\* is driven by
the output gap's cycle, which turned before the policy rate did.

### The band understates

It is the range across three particular modelling choices, not a sample from anything, so a
fourth reasonable model could sit outside it. And each line is itself conditional on a number
nobody measures: `rstar_invert`'s own `sigma_rstar` sweep moves its 2007 peak between 4.0 and
7.0 real, **wider than the entire cross-model band at that date**.

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
