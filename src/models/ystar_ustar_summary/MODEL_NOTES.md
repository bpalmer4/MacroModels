# y\*/u\* summary: eight settings of one model

## Not eight models

This estimates nothing. It runs `ystar_ustar` eight times and charts the
results together. The eight cross the two choices the joint model actually
has to make:

| | inflation-defined gap | gap = y - y\* |
|---|---|---|
| u\* decays to a level | x | x |
| **u\* is a spline, 1 knot** | **x** (the model's default) | x |
| u\* is a spline, 2 knots | x | x |
| u\* is a spline, 3 knots | x | x |

Down a column is how much the structure imposed on u\* matters. Across a row
is how much the definition of the gap matters. **Crossed rather than
laddered** so the two cannot be confounded: with a cell missing, a difference
between columns could always be the structure that was only tried on one
side.

All eight share a sample, an expectations series and an inflation measure, so
agreement within a column is close to arithmetic and only disagreement
informs. The two columns are different models of the same data.

**Nothing here is a recommendation.** It is the set tested before settling, so
settings argued against elsewhere are in it. Decay with the identity gap is
the clearest case: a structure that can only report a fall, paired with a gap
definition under test.

---

## Results

```
                                    elpd  se    bad k  ESS   resid  93-99   u* 93Q1  now   p2015  gap 93Q1   now    sd    g*
Decay,   gap = y - y*               0.00  13.0    1    3924  -0.04  -0.38     9.43   4.67  -0.32    -3.05   +1.52  1.80  1.98
Decay,   inflation-defined         -7.32  10.2   92    4257  -0.11  -0.65    10.77   4.74  -0.38    -0.10   +0.30  0.42  1.99
Spline 1 knot, gap = y - y*        -9.20  13.6    0    6145  -0.07  -0.24     7.37   4.67  -0.15    -6.44   +1.28  2.32  1.97
Spline 2 knots, gap = y - y*       -9.52  13.3    1    3223  -0.04  -0.30     8.49   4.58  -0.41    -4.35   +1.20  1.93  1.99
Spline 2 knots, inflation-defined -11.32  10.0   86    3500  -0.10  -0.63    10.20   4.56  -0.66    -0.42   +0.16  0.37  2.00
Spline 3 knots, gap = y - y*      -11.49  13.2    1    5131  -0.04  -0.30     8.47   4.42  -0.77    -4.61   +0.87  1.82  2.06
Spline 3 knots, inflation-defined -12.89  10.4   81    4393  -0.09  -0.67     9.95   4.31  -1.15    -0.56   +0.01  0.31  2.02
Spline 1 knot, inflation-defined  -15.87  10.7  114    1333  -0.23  -0.80     8.17   5.06  +0.38    -1.80   +0.49  0.67  1.95
```

R-hat is 1.00 and divergences are zero in all eight, so the sampling gate
eliminates nobody.

**The fit column does not separate them.** The whole spread is 15.9 elpd
against standard errors of 10 to 14. Nothing here is a significant win over
anything else and no specification should be chosen on this column.

**The Pareto k counts are the real signal.** Every inflation-defined run has
81 to 114 unreliable observations out of 268; every identity run has 0 or 1.
That is the circularity showing up as a diagnostic: when the gap is
`c x (pi - 2.5)`, inflation sits on both sides of the Phillips curve, so
dropping one quarter moves the fitted values a lot and importance sampling
breaks down. **Those four elpd figures should not be read at face value.**

**The gap definition matters and the u\* structure does not, for the
endpoint.** Down any column u\* today moves 4.31 to 5.06; across a row it
barely moves. Potential growth is untouched throughout at 1.95 to 2.06.

**Every specification is biased in 1993-99**, reading u\* above what the
Phillips curve alone implies, by -0.24 to -0.80. That is shared, so it
discriminates nothing and points at the sample start rather than at any
structure. The identity runs are the less biased half.

---

## How the fit column is computed, and why it is narrow

The eight do not observe the same data. Under the identity gap the GDP
equation is a definition and carries no likelihood, so a model-wide
information criterion would compare models fitted to different data.

Both columns do carry pointwise log-likelihood for `observed_u` and
`observed_pi`. The score concatenates those two into one observation axis and
computes leave-one-out over both, so a point is one quarter of one equation.
That answers "how well does this setting predict unemployment and inflation",
which is comparable across all eight.

The caveat to state with it: the settings condition on different information
even though they are scored on the same targets, so this ranks predictive
accuracy and settles nothing about which model is true.

---

## The 1993Q1 problem, which none of these fixes

u\* at 1993Q1 runs 7.37 to 10.77 and the output gap -0.10 to -6.44, on the
same data. The inflation-defined gap says output was 0.1 to 1.8 per cent
below potential in a quarter with unemployment at 10.9 per cent, which cannot
be right; the identity gap says -3.0 to -6.4, which is the right order but
swings by a factor of two across settings that are statistically
indistinguishable.

The cause is structural. The 1990-91 recession and the disinflation are both
outside a sample that opens in 1993Q1, so the model begins mid-recovery with
no information about what it is recovering from. The shaded window on the
charts marks 1993Q1-1999Q4; these numbers say that is not conservative
enough, and nothing before 2000 should be treated as an estimate.

Starting earlier was tested and is not a repair: it needs a phased anchor,
and under a random-walk trend that collapses `c`. See `ystar`'s notes.

---

## Charts

| file | what it shows |
|---|---|
| `u-by-specification` | all eight u\* paths against the unemployment rate |
| `the-output-gap-by-specification` | the eight gaps |
| `potential-growth-by-specification` | the eight g\* paths |
| `potential-output-by-specification` | the levels, which are nearly identical and the least useful of the set |
| `how-much-the-specification-matters-for-u` | the range across settings, quarter by quarter |
| `how-much-the-specification-matters-for-the-output-gap` | the same for the gap |

Colour is the u\* structure and dashing is the gap definition, so the two
choices read separately. The shaded window is drawn only when the sample
actually opens on it, the same guard `ustar` applies: it was measured on a
1993Q1 start and marks nothing on a longer sample.

The output gap chart's y-axis is set by the identity runs, so the
inflation-defined gaps are squashed near zero and their own movement is hard
to see there.

---

## Files and usage

```bash
./run-ystar-ustar-summary.sh              # refresh anything stale, then chart
./run-ystar-ustar-summary.sh --no-refresh # chart the saved runs as they stand
```

Each setting writes to its own `yus_sum_*` prefix, so a refresh never touches
the model's own default outputs or `charts/YStarUStar`. A saved trace counts
as current if the file was written TODAY, the same rule `rstar_summary` uses:
a proxy for the data being current rather than a check of ABS and RBA
vintages, conservative in the right direction since a stale file is always
re-run.

A full refresh is eight estimations at a few minutes each.

Note that `yus_sum_k1`, spline with one knot and the inflation-defined gap, is
the model's own default configuration. Its row here is the same specification
the headline run uses, estimated into a separate prefix.
