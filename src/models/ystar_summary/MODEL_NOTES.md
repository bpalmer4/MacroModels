# y* Summary: Model Notes

**NOT A MODEL.** This loads the output gap from every live model that estimates potential
output (y\*) and puts them on one chart, with a chart of the range across them and a combined
estimate that pools their posterior draws. Nothing here is estimated.

```bash
./run-ystar-summary.sh                 # re-run stale models first, then chart
./run-ystar-summary.sh --no-refresh    # read saved runs as they stand
./run-ystar-summary.sh --start 2000Q1
```

Chart: `charts/ystar-summary/`.

## Three lines, three definitions

Each line is the gap its model reports. **They are not the same object**, so the lines differ
partly because the definitions differ.

| line | package | the gap is | log GDP − y\*? |
|---|---|---|---|
| y\* (inflation spec) | `ystar`, default run | c·(π − anchor) | No: GDP is fitted around it with a residual |
| Joint y\*/u\* (slack split) | `ystar_ustar`, default run | c·(π − anchor) + v | No: GDP still carries a residual on top |
| Semi-structural open economy | `rstar_qpm` | `ygap` in y = y\* + ygap | Yes, up to a small measurement error |

In plain English:

- **`ystar`** defines the gap as whatever inflation's deviation from its anchor implies. The
  part of GDP's deviation from potential that inflation does not account for goes to a
  residual, not the gap. It is the narrowest of the three by construction.
- **The joint model** starts from the same inflation-defined gap and adds a free component
  v, disciplined by Okun's law and a Phillips curve on unemployment and underemployment. Part
  of its gap is therefore not tied to inflation, so it moves more than `ystar`'s.
- **`rstar_qpm`** treats the gap as GDP's deviation from a drifting potential, inside an
  IS / exchange-rate / Phillips / policy-rule system. Nothing forces it to agree with
  inflation quarter by quarter, so it is the most volatile.

**What the chart can and cannot say.** Agreement on the sign and timing of the gap is
informative. Disagreement on amplitude is mostly definitional: comparing the `ystar` line's
size with the others is comparing a component with a whole. The range chart shows that
disagreement quarter by quarter; it draws no mean, since an average of a component and a whole
describes nothing.

**The combined estimate pools different objects.** It weights each model's draws equally and
reports the pooled median and 50% and 90% bands. Because the three gaps are defined differently,
the bands hold the difference in definition as well as each model's uncertainty and the models'
disagreement. Read it as where the three definitions jointly sit, not as an estimate of one gap.

## The lockdown window

All three runs drop GDP over the lockdown quarters, and their gaps carry on through those
quarters under the states' dynamics. The window is read from each run's recorded settings,
not typed here. It is shaded only when every loaded run recorded the same window; when they
differ, nothing is shaded and the run log names each run's window.

## Why these three

- `rstar_bonds` and `rstar_rba` use an output gap or unemployment gap, but read it from a
  completed `ystar` or `ystar_ustar` run rather than estimating one.
- `nairu`, `rstar_hlw` and the other retired or superseded packages are left out as they are
  from the other summaries.
- `cobb_douglas` produces potential by HP filters and is not COVID-robust.

## File structure

```
src/models/ystar_summary/
├── sources.py       # the registry: what to load, how, and what each gap is
├── analyse.py       # the chart
├── run.py           # CLI: --no-refresh, --start
└── MODEL_NOTES.md   # this file

run-ystar-summary.sh
```
