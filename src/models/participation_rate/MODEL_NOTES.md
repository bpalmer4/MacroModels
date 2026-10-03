# Participation rate after RBA rate moves

## Overview

Exploratory regressions, not a structural model. The question: does the
participation rate rise in the months after the RBA raises the cash rate?

The theory of the case: a rate rise puts budgetary pressure on households, and
some respond by taking on work. Mortgagors are the obvious channel, and the
response should come from people at the margin of the labour force, more
likely second earners than primary ones.

Monthly data from August 1990, when the cash rate starts moving in discrete
target steps. Participation (persons, males, females), unemployment and
employment are from the Labour Force Survey (6202.0, seasonally adjusted); the
cash rate is the RBA target.

The age breakdown uses the Labour Force Survey datacube LMS2 (6202.0), which
gives labour force status by sex and age. It comes in Original terms only, so
it is compared with the same calendar window in other years rather than
seasonally adjusted.

## Definitions

- **Decision month**: the cash rate differs from the previous month. A hike or a
  cut.
- **First move of a cycle**: a hike (cut) with no hike (cut) in the previous 12
  months. These are the cleanest events: the start of a tightening is less
  contaminated by the moves before it.
- **Outcome**: $\Delta_h PR_t = PR_{t+h} - PR_{t-1}$, the change from the month
  before the decision to $h$ months after, $h = 0, \dots, 18$.
- Windows that touch March 2020 to December 2021 are dropped.

## Equations

**Event study.** The mean of $\Delta_h PR_t$ over hike months, first hikes,
cut months and first cuts, against its mean over all months. The all-months
mean is the trend: participation drifts up over the sample, so any window
shows a rise.

**Local projections.** For each horizon $h$:

$$
\Delta_h PR_t = \alpha_h + \beta_h\,\text{Hike}_t + \gamma_h\,\text{Cut}_t
  + \delta_h' X_{t-1} + \varepsilon_{t,h}
$$

with $X_{t-1}$ the state of the labour market before the decision: the 3- and
12-month changes in participation and in unemployment, 12-month employment
growth, the unemployment rate, and the 12-month change in the cash rate.
Newey-West errors with $h+1$ lags, because overlapping windows make
$\varepsilon_{t,h}$ serially correlated. $\beta_h$ is the change in
participation after a hike relative to a month with the same pre-decision
labour market and no decision.

A variant adds the unemployment change over the same window,
$U_{t+h} - U_{t-1}$. Then $\beta_h$ is the rise in participation beyond what
the unemployment cycle would explain. This conditions on something the
decision itself affects, so it is a decomposition, not a cleaner estimate.

**Discriminating tests.** Each asks something the budget channel predicts and
the alternative (below) does not.

- *Dose*: replace the dummies with the size of rate rises and falls over the
  next 12 months. A budget channel should scale with how much rates rise.
- *Era*: split the hike effect before and after 2005. Household debt relative
  to income is far higher in the later period, so a budget channel should be
  stronger there.
- *Pre-trend*: regress the participation change over the 12 months *before* a
  first move on the event and the non-participation controls. A rise already
  under way would mean the hike marks a phase of the cycle rather than
  starting anything.

**By sex.** The local projections are repeated for female and male
participation, every decision. For first hikes, each sex's path in the latest
cycle is set against the average path of the earlier cycles. That average is
raw: it includes each sex's own trend.

**Latest cycle by age.** For each sex and age band (15-24, 25-54, 55-64, 65+),
the participation change from the month before the latest first hike to the
latest month, minus the average change over the same calendar window in
normal years (2010 on, leaving out the years distorted by the lockdowns and
the reopening):

$$
D_{s,a} = \Delta PR_{s,a}^{\text{latest}} - \overline{\Delta PR}_{s,a}^{\text{normal}}
$$

The contribution of each band to its sex's change is $D_{s,a}$ times the
band's share of that sex's population in the latest month. The contributions
sum to an approximate decomposition of the sex's change; it ignores shifts in
the age mix.

## Plain English

Participation rises after the RBA starts hiking and falls after it starts
cutting, and the pattern survives controls for where the labour market was
when the decision was made. It is not already under way beforehand. It also
survives holding the path of unemployment fixed.

What the tests cannot do is separate the budget channel from the obvious
alternative. The RBA raises rates when the labour market is strong, and
participation follows job availability with a lag: people are drawn in when
work is easy to find (the encouraged worker effect). That predicts the same
sign. Unemployment typically keeps falling for a year after a first hike, so
the labour market is still tightening while participation rises.

The discriminating tests lean against the budget channel rather than for it.
Cuts mirror hikes, which both stories allow. The effect is not stronger in the
high-debt era, which the budget channel needs. The dose response is
suggestive but the size of a tightening is itself set by how strong the labour
market stays.

By sex, women's participation has typically risen much more than men's after
a first hike, and men's has barely moved. Part of that is trend: women's
participation has been rising for decades and men's falling. The trend is
not steady, though. It differs a lot from decade to decade, and it is bounded,
so a sample-average trend cannot be subtracted at the end of the sample. The
comparison is therefore left raw.

In the latest cycle, women are on their usual path. Men are not: their
participation has risen as fast as in the fastest earlier cycle at the same
stage, far above the usual pace, and in a few months has gone beyond what it usually does over a year
and a half. That departure from the usual pattern is the notable feature of
this cycle.

By age, against normal years, men's rise is spread across every age band, with
the prime-age group making the largest contribution. Women's rise comes from
young and older women, while prime-age women are below normal.

Separating the channels would need a source of rate variation unrelated to the
labour market (an estimated monetary policy shock series), or household data
that splits mortgagors from renters and outright owners.

## Running

```bash
./run-participation-rate.sh
```

Prints the tables and writes eleven charts to `charts/participation-after-hikes/`.
For participation and again for unemployment: the path around each first hike
and each first cut (each cycle and the average), the two averages against the
all-months trend, and the local projections for first moves with 90% bands.
Then participation by sex around first hikes (the latest cycle against the
earlier cycles' average), and the latest cycle by age against normal years,
as changes and as contributions.
