# y\* summary: five specifications, and what comparing them showed

## Read this first: the result is against the package it summarises

This estimates nothing. It runs `ystar` five times from 1984Q1 and charts the
results together. What it established is not which specification is best but
that **three of the five reproduce an HP filter**, and that inflation does
very little work in any of them.

| | corr(y\*, HP) | sd(y\* - HP) | corr(g\*, g_HP) | inflation's share |
|---|---|---|---|---|
| inflation, random walk trend | **1.0000** | 0.337 | 0.974 | 7% |
| production function | **1.0000** | 0.329 | 0.979 | 5% |
| production, no MFP observation | **1.0000** | 0.334 | 0.976 | 7% |
| core, Phillips curve on a cycle | 0.9997 | 1.057 | 0.822 | |
| inflation, polynomial trend | 0.9997 | 0.880 | 0.715 | 21% |

HP is HP(1600) on log GDP. "Inflation's share" is the sd of
`c x (pi - anchor)` against the sd of `y - y*`.

The state-space apparatus, the capital and labour data and the production
function all return, to four decimal places in levels, what a filter gives
for free. `ystar`'s own notes already recorded the `labour` spec reproducing
an HP(1600) trend of hours at 0.9972 and called it "an elaborate apparatus
returning what a filter gives for free". The same charge now applies to the
default inflation spec and to the production function.

The two that depart are the ones with a real restriction on the trend: the
AR(2) cycle in `core`, and a global polynomial. They depart because the trend
is stopped from chasing output, and when it is, inflation starts doing work:
`c` triples to 0.325.

---

## The five

| spec | potential's growth from | gap identified by |
|---|---|---|
| `inflation` | a free drift state | defined as `c x (pi - anchor)` |
| `production` | capital, labour and MFP | defined as `c x (pi - anchor)` |
| `core` | a free drift state | a Phillips curve on an AR(2) cycle |
| `labour` | trend hours x productivity | a Phillips curve on an AR(2) cycle |
| `target` | a free drift state | inflation's later sign only, no slope |

`inflation` and `production` are a pair differing only in potential;
`core` and `labour` are a pair differing only in potential; `target` is
`core` with the Phillips curve replaced by a sign restriction.

"core" is the base specification and has nothing to do with core inflation.

---

## Results

```
                                          elpd  bad k  R-hat   ESS   g* now  g* 1990  g* 15-19  gap 92Q4  gap now  gap sd
Inflation's sign only, no Phillips slope   0.00    1   1.00   3895    2.20    2.25      2.30     -0.88    -0.04    1.12
Phillips curve on a cycle, free trend    -15.62    2   1.00   4760    1.93    3.14      2.26     -2.53    +1.97    1.68
Phillips curve on a cycle, hours x prod  -16.45    3   1.00   8036    1.72    2.88      2.43     -1.70    +1.77    1.77
Gap = c x (pi - anchor), production      -77.21    0   1.01   1380    1.87    2.39      2.43     -0.57    +0.03    1.21
Gap = c x (pi - anchor), free trend      -83.09    0   1.00   3977    1.94    2.48      2.43     -0.74    -0.06    1.22
```

**Read the fit column lightly.** `target` and `inflation` both observe GDP
alone yet sit at opposite ends, 0.00 and -83.09, so the column is not
measuring what it appears to. It scores GDP only, over the 162 quarters every
specification fitted, because GDP is the one series all five observe. That
favours a specification spending everything on GDP: `inflation` and `target`
observe nothing else, while `labour` also answers for hours and
participation with the same trends and `production` for four factor series.
One input, not a ranking.

Even GDP is not observed over the same quarters: the inflation family drops
the lockdown window from its likelihood and the AR(2) specs lose leading
lags, so the score is restricted to the intersection. `sources.gdp_quarters`
derives that alignment from what each run recorded rather than by counting,
because lining up the wrong quarters under one column would be invisible.

---

## What should not be quoted from here

**The three walk-based runs carry a collapsed `c`.** Running from 1984Q1
requires a phased anchor, and under a random-walk trend that takes `c` from
0.188 to 0.082 and nearly doubles `sigma_e`. The inflation-defined gap is
then switched off, reading -0.07 to +0.04 straight through the 1990-92
recession, and the recession goes into potential instead. That is the visible
dip to 2.2-2.5 in 1990 for `inflation`, `production` and `target`.

The mechanism is in `ystar`'s notes and is a statement about the
specification, not about the glide: within every sub-period the inflation
deviation comoves **negatively** with an HP cycle, and `c > 0` over a long
sample is supported by a between-era level difference that the glide removes
by construction.

**So the 1990-92 potential growth figures here are not an outside check on
anything.** In particular `production`'s 2.39 is not independent evidence: it
has the same 1991 dip and the same 1998 spike as the free random walk, and
the factor charts show the whole dip sitting in trend MFP, which is the Solow
residual and therefore a smoothed residual of GDP.

**The endpoint spread is 1.72 to 2.20** and does not narrow over the sample.
The specification question is not resolved by the extra data.

---

## The deeper problem, which this package cannot fix

`ystar` never observes unemployment. No spec has an Okun equation or the
unemployment rate in its data. So when output falls for two years the model
has only smoothness to tell it whether capacity fell too, and an 11 per cent
unemployment rate is invisible to it.

Every structural fix tried moved the problem rather than removing it:
constrain the trend and the cycle appears in the GDP residual; constrain MFP
and it appears in trend hours. The trend/cycle split has to come from
somewhere, and with inflation contributing 5 to 21 per cent, smoothness
assumptions supply the rest wherever they are loosest.

There is also an internal contradiction worth knowing. `c` and the Phillips
slope are reciprocals. `core` and `labour` estimate the slope at 0.242 and
0.273, implying `c` near 4; the inflation-defined specs estimate `c` at 0.082
to 0.325, implying a slope of 3 to 12. A slope of 0.25 is defensible, 12 is
not, and both ends of the reciprocal give an implausible gap, so no single
`c` repairs it.

---

## Charts

| file | what it shows |
|---|---|
| `potential-growth-by-specification` | the five g\* paths |
| `the-output-gap-by-specification` | log GDP less potential in every row, not the inflation-defined series, so the column compares like with like |
| `potential-output-by-specification` | the levels, nearly identical and the least useful of the set |
| `how-much-the-specification-matters-for-potential-growth` | the range across specifications |
| `how-much-the-specification-matters-for-the-output-gap` | the same for the gap |

---

## Files and usage

```bash
./run-ystar-summary.sh              # refresh anything stale, then chart
./run-ystar-summary.sh --no-refresh # chart the saved runs as they stand
```

Every run uses `--start 1984Q1 --anchor-phase glide` and writes to its own
`yss84_*` prefix, so a refresh never touches `ystar`'s own outputs or
`charts/YStar`. A saved trace counts as current if the file was written
TODAY, the same rule the other summaries use.

`run_analysis` takes a `chart_dir`, so a set of specifications that is not
these five can be charted without clearing this one: the charting clears its
directory first.
