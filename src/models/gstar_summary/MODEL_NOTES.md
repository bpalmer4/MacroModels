# g* Summary: Model Notes

**NOT A MODEL.** This loads potential output growth from every model in the repo that
estimates one, puts them on a single chart, and charts the spread between them. Nothing here
is estimated.

```bash
./run-gstar-summary.sh                 # re-run stale models first, then chart
./run-gstar-summary.sh --no-refresh    # read saved runs as they stand
./run-gstar-summary.sh --start 2000Q1
```

## The headline: they agree

Latest quarter, 2026Q2:

| source | g* | what it is built from |
|---|---|---|
| Semi-structural open economy (`rstar_qpm`) | **2.12** | y* a drifting random walk inside an IS / exchange-rate / Phillips / rule system; its trend-growth state g |
| Joint y*/u* | **1.95** | y* and u* estimated together, with Okun and a Phillips curve |
| y* (inflation spec) | **1.94** | potential is a slow random walk; the gap is defined by inflation |
| y* (production spec) | **1.90** | growth from capital, hours and MFP trends; level and gap still inflation-defined |

**A spread of 0.22pp, against about 1.0pp for r\*.** That contrast is the reason this package
exists as a sibling to [`rstar_summary`](../rstar_summary/MODEL_NOTES.md) rather than a section
inside it. The two answer the same kind of question, "what does this repo actually know", and
get opposite answers. For r* no model identifies a level and the honest band is the
cross-model spread. For g* four estimates land within about a sixth of a percentage point,
though see the caveat below about how much of that is independence.

The mean across models ends at **1.98**, which is close to the RBA's ~2.0 and well below
Treasury's 2.5. It is a description of where the models sit, not an estimate: an average
across structural assumptions is a value no model produces, the same objection `rstar_hlw`'s
notes make to its own blended median.

## Four lines, and not four independent votes

**Three share the y\* state-space core, and the fourth shares its key assumption**, so the
agreement is weaker evidence than it looks.

- `ystar`'s **inflation** and **production** specs are one package run two ways. They share the
  data, the sample, the level equation and the gap definition, differing only in where
  potential's growth comes from. `ystar`'s notes call their agreement a semi-validation and
  warn it only holds while the two are kept separate. **`production` is not a supply-side-only
  estimate**: its growth comes from factor trends, but inflation still positions its level and
  defines its gap, which is why its chart directory contains an inflation-defined output gap.
- The **joint y*/u\*** model is built on the same y* core and adds Okun and a Phillips curve.
- **`rstar_qpm`** is a separate package, potential drifting with a trend-growth state inside an
  open-economy system. Its line is that trend-growth state, potential growth without the
  level shocks to potential, which would otherwise read as swings in growth. But it too
  treats potential as a slowly drifting random walk, and how smooth that walk is comes from
  a prior, because `sigma_ystar` is not identified there. A partial outside check, not an
  independent one. It sits a little above the rest in recent years.

`cobb_douglas`, the one line built a different way, is excluded for COVID artefacts, below.
**If a smoothing assumption common to all four were wrong, nothing on this chart would catch
it**, and that is the honest limitation of the agreement above.

## Why `cobb_douglas` is not here: COVID artefacts

It was the one line built without a random-walk potential, which is exactly the independent
check the others lack, so it was worth several attempts. All failed.

Its three HP filters run through the pandemic. Filtered straight through, potential growth
humps to 2.48 in 2022Q2 against about 1.85 for the state-space models, and is still falling
0.16pp a year at the endpoint while every other model has flattened and turned up. Excluding a
window does not remove the artefact, it only changes its sign and size:

| excluded window | deviation from the state-space lines |
|---|---|
| none | +0.42 |
| 2020Q2-2021Q3, as the others use | +0.96 |
| 2020Q1-2022Q4, through the rebound | −0.39 |

**And the windows that minimise it, around 8 to 10 quarters, were found by comparing against
these very models.** Choosing one on that basis turns an independent check into a line
calibrated to agree, which is worth nothing. That is the reason it stays out, rather than any
single failed configuration.

Two things were learnt worth keeping. The distortion is the **rebound** rather than the
lockdown: hours grew +10.3 log points over 2021Q4-2023Q4, which is why excluding only the
lockdown made matters worse than doing nothing. And **imputation is the wrong device for a
filter**: replacing a window forces the filter to fit invented data, so the imputation drives
the answer. `cobb_douglas` is otherwise untouched and remains good for the growth accounting
it exists for.

## A vintage trap, found the hard way

Until 2026-09-12 the saved `ystar_production` trace was from 5 September and had
`exclude_window = NONE`: it had been fitted straight through the lockdowns. On that vintage it
read **2.16** against the inflation spec's 1.94, and the gap looked like a specification
disagreement. Re-run on the current default it reads **1.90**, so 0.26pp of the apparent
disagreement was vintage rather than specification.

**The lesson for this package: a summary chart silently compares whatever was last saved.**
`is_current()` reports each source's vintage in the run log for that reason, and it is worth
reading before quoting the spread.

**Refresh re-runs each stale source with the arguments that reproduce it.** `ystar`'s
production spec is not what `run-ystar.sh` produces by default, so its source carries
`--spec production --prefix ystar_production`; without them a refresh would overwrite the
inflation spec's run and merge the two y* lines into one.

## Why `rstar_hlw` is excluded

At the user's direction, and the reasons hold up. Its trend growth g is a state in a model
whose r* is not identified; g is the drift of a single random walk with no labour-market or
factor information behind it; and its level moved 1.88 to 2.23 purely because the sample start
changed from 1986Q3 to 1993Q1. See [`rstar_hlw/MODEL_NOTES.md`](../rstar_hlw/MODEL_NOTES.md).

`nairu` is absent too: its potential is not estimated, and its
posterior median reproduces the Cobb-Douglas input, so it would contribute a line this package
has already decided not to carry.

## File structure

```
src/models/gstar_summary/
├── sources.py       # the registry: what to load, how, and why HLW is not here
├── analyse.py       # the two charts
├── run.py           # CLI: --no-refresh, --start
└── MODEL_NOTES.md   # this file

run-gstar-summary.sh
```

Cobb-Douglas is the one source **computed rather than loaded**: it writes no trace, so it
re-derives potential from ABS data on every call, which makes it the slowest part of a run.
