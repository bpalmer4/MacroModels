# u* Summary: Model Notes

**NOT A MODEL.** This loads u* from four specifications of the two models in the repo that
estimate one, puts them on a single chart against the unemployment rate, charts the spread
between them, and pools their posterior draws into a combined estimate. Nothing here is
estimated.

```bash
./run-ustar-summary.sh                 # re-run stale specifications first, then chart
./run-ustar-summary.sh --no-refresh    # read saved runs as they stand
./run-ustar-summary.sh --start 2000Q1
```

## The four lines

| model | specification | what it is built from |
|---|---|---|
| `ustar` | tapered walk (default) | one Phillips curve, u* a random walk whose step size tapers, loose early and tight later |
| `ustar` | spline, 1 knot | the same curve, u* a spline |
| `ystar_ustar` | slack split u*, inflation-defined gap (default) | y* and u* in one likelihood, observing underemployment beside unemployment; u* is unemployment's share of total structural slack |
| `ystar_ustar` | tapered walk u*, inflation-defined gap | the same on unemployment alone, u* a random walk whose step size tapers |

Each is a specification from its model's `--compare`, found there by prefix, so the flags that
reproduce it are defined once. The rest of each model's `--compare` (the two-knot spline, the
joint model's stepped walk, splines, decay settings and `y - y*` gap settings) stays there.

## Four lines, and not four independent votes

**All four are fitted to the same unemployment rate, the same trimmed mean inflation and the
same expectations series**, and every one reads u* off a Phillips curve; the slack split also
sees underemployment. They differ in the
shape imposed on u* and in whether an output gap enters. So
agreement between them is weak evidence: it says the shape assumptions do not matter much, not
that u* is well measured. Their disagreement is the informative part, and it is concentrated
where the data place u* least well, the shaded 1993-2003 window both models flag.

## The combined estimate

The posterior draws of all four are pooled, each specification carrying equal weight however
many draws it holds, and the median and the 50% and 90% central bands are read off the pool
quarter by quarter.

- **The bands hold two things**: the uncertainty within each specification and the disagreement
  between them. They are wider than any one specification's band where the specifications
  disagree, and they are not the credible interval of any single model.
- **Pooling adds no evidence.** All four see the same data, so more specifications do not narrow
  the bands, which is the right behaviour given they are not independent votes.
- **The equal weights are a choice**, not an estimate. No fit statistic picks them.
- **The median can sit where no specification does.** Where the specifications split into two
  clusters, as they do early in the shaded window, the pool has two humps and its median falls
  between them. Read the bands there, not the line.

## The three charts

- **Every line**: the four posterior medians, with the unemployment rate in black behind them.
  mgplot chooses the colours and line styles, except that its gold is swapped for dark cyan,
  which reads better as a line.
- **The spread**: the range across the four on their common quarters, with the mean across them.
  The mean is a value no specification produces; it describes where they sit and is not an
  estimate. The range is not an error band either, since the four differ in a structured way
  rather than at random.
- **The combined estimate**: the pooled median with its 50% and 90% bands, and the unemployment
  rate in black.

## Vintage and refreshing

A saved trace counts as current if it was written today. That is a proxy for the data being
current, not a check of ABS vintages, and it errs the right way: a stale file is always re-run
and a fresh one never is. Refreshing re-estimates through the specification's own `--compare`
machinery, so it writes to that specification's prefix and redraws its own charts. The
`ystar_ustar` runs take a few minutes each.
