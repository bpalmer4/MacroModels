# u* Summary: Model Notes

**NOT A MODEL.** This loads u* from four specifications of the two models in the repo that
estimate one, puts them on a single chart against the unemployment rate, and charts the
spread between them. Nothing here is estimated.

```bash
./run-ustar-summary.sh                 # re-run stale specifications first, then chart
./run-ustar-summary.sh --no-refresh    # read saved runs as they stand
./run-ustar-summary.sh --start 2000Q1
```

## The four lines

| model | specification | what it is built from |
|---|---|---|
| `ustar` | random walk (default) | one Phillips curve, u* a random walk whose step size tapers, loose early and tight later |
| `ustar` | spline, 1 knot | the same curve, u* a spline |
| `ustar` | spline, 2 knots | the same, with a second knot |
| `ystar_ustar` | random walk u*, inflation-defined gap | y* and u* in one likelihood, u* a random walk whose step size tapers |

Each is a specification from its model's `--compare`, found there by prefix, so the flags that
reproduce it are defined once. Only `ystar_ustar`'s random walk is charted, beside the `ustar`
walk; its splines, decay setting and `y - y*` gap settings are left out.

## Four lines, and not four independent votes

**All four are fitted to the same unemployment rate, the same trimmed mean inflation and the
same expectations series**, and every one reads u* off a Phillips curve. They differ in the
shape imposed on u*, in whether an output gap enters, and in how that gap is defined. So
agreement between them is weak evidence: it says the shape assumptions do not matter much, not
that u* is well measured. Their disagreement is the informative part, and it is concentrated
where the data place u* least well, the shaded 1993-1999 window both models flag.

## The two charts

- **Every line**: the four posterior medians, with the unemployment rate in black behind them.
  mgplot chooses the colours and line styles, except that its gold is swapped for dark cyan,
  which reads better as a line.
- **The spread**: the range across the four on their common quarters, with the mean across them.
  The mean is a value no specification produces; it describes where they sit and is not an
  estimate. The range is not an error band either, since the four differ in a structured way
  rather than at random.

## Vintage and refreshing

A saved trace counts as current if it was written today. That is a proxy for the data being
current, not a check of ABS vintages, and it errs the right way: a stale file is always re-run
and a fresh one never is. Refreshing re-estimates through the specification's own `--compare`
machinery, so it writes to that specification's prefix and redraws its own charts. The
`ystar_ustar` runs take a few minutes each.
