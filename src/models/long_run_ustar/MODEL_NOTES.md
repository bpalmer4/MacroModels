# Long-run u\* — read off the quarters where inflation was flat

## Read this first: there is no estimate here

Every other u\* in this package is inferred: a state equation, a likelihood, an imposed variance,
a posterior. This one infers nothing. The NAIRU is *defined* as the unemployment rate at which
inflation stops changing, so this model finds the stretches where inflation in fact stopped
changing, reads the unemployment rate across them, and reports what it saw.

There is no band, because nothing is estimated. What stands in for one is the sweep: the answer
is worth as much as it is stable across the rule that produced it, and no more.

**What it buys is reach.** `ystar`, `ustar` and `ystar_ustar` all start in 1993Q1 because their
gap is defined against an inflation target that did not exist earlier. A rule about flat
inflation needs no target, so it runs back to **1959Q3**, where the quarterly unemployment rate
begins.

**What it costs is everything those models do carefully**: no control for supply shocks, no
expectations, no allowance for the lag between slack and prices, and a reading only as good as
the flatness rule. Limitations, below, is the long list.

---

## The rule

1. Year-ended headline CPI inflation, smoothed with a centred 4-quarter moving average.
2. A quarter qualifies if, over the centred 8-quarter window around it, the smoothed series
   moves by no more than **1.0 percentage point** high to low.
3. Contiguous runs of at least 2 qualifying quarters are episodes.
4. The reading is mean unemployment over the episode, reported at lags 0, 2 and 4.

**Range rather than slope**, because the distinction that matters is wide-based against
V-shaped. A turning point has a small slope at its centre and a large range around it, and only
the range test rejects it. `--flatness-rule slope` is kept as the comparator and finds more,
since it accepts a steady gentle climb.

**Smoothed first, or the rule finds nothing before 2002.** Unsmoothed, year-ended headline
inflation is volatile enough that no window in the 1960s or 1970s passes any tolerance that
means anything later: 16 qualifying quarters in the whole sample, every one of them after
2002Q3. That is a fact about the measure, not about the economy.

**The lag is reported, not chosen.** Inflation follows slack, so the rate that produced a flat
stretch may be the one before it. `ystar` items 7, 9 and 12 establish that the lag is not
identifiable on Australian data, so the table carries lags 0, 2 and 4 and lets the reader see
the spread.

---

## Results (2026Q2 vintage, sample 1959Q3-2026Q2, 268 quarters)

| episode | quarters | inflation | u lag 0 | lag 2 | lag 4 |
|---|---|---|---|---|---|
| 1967Q2-1969Q3 | 10 | 2.96 | **1.82** | 1.82 | 1.82 |
| 1988Q4-1989Q4 | 5 | 7.54 | **6.20** | 6.75 | 7.23 |
| 2002Q4-2005Q2 | 11 | 2.65 | **5.57** | 5.79 | 6.04 |
| 2013Q1-2014Q1 | 5 | 2.47 | **5.68** | 5.43 | 5.22 |
| 2015Q4-2019Q4 | 17 | 1.68 | **5.45** | 5.54 | 5.67 |

By decade, length-weighted: **1960s 1.82, 1980s 6.20, 2000s 5.57, 2010s 5.50.**

No single number is printed across decades, and the code refuses to compute one by default.
Averaging 1.8 and 5.5 would assert the constancy the exercise exists to test.

### The second category: U-shaped troughs

The rule above finds **plateaus**, stretches where inflation held level wherever that level was.
It does not find the base of a wide U, because a broad trough's sides rise enough to break the
range test. Six such troughs exist and the plateau rule catches none of them, so they are
detected separately (`find_troughs`) and reported in their own table:

| U base | quarters | inflation | u lag 0 |
|---|---|---|---|
| **1962Q3-1963Q2** | 4 | −0.04 | **2.21** |
| 1978Q2-1979Q2 | 5 | 8.19 | 6.23 |
| 1984Q3-1985Q1 | 3 | 3.92 | 8.55 |
| 1992Q3-1993Q2 | 4 | 1.07 | 10.83 |
| 1997Q3-1998Q2 | 4 | 0.22 | 7.95 |
| 2009Q3-2010Q1 | 3 | 1.98 | 5.50 |

**They are never merged into the decade readings**, because an inflation trough usually arrives
at the end of a disinflation, when unemployment is at its worst. Four of the six sit at 6.2, 8.0,
8.6 and 10.8.

### Plateau against trough, checked on the `nairu` model

The two categories can be compared against an estimate built with entirely different machinery.
`nairu` starts in 1984Q3 and carries Okun, a Cobb-Douglas potential and a phased expectations
anchor, none of which exists here:

| date | `nairu` model | rule reading | kind |
|---|---|---|---|
| 1988Q4-1989Q4 | 6.22 | **6.20** | **plateau** |
| 1984Q3-1985Q1 | 6.25 | 8.55 | trough |
| 1992Q3-1993Q2 | 6.26 | 10.83 | trough |
| 1997Q3-1998Q2 | 6.13 | 7.95 | trough |

**On the one plateau available, the two agree to two hundredths of a point.** That is the
strongest corroboration this rule receives anywhere: a method that fits nothing and a Bayesian
state-space model land on 6.20 and 6.22 for the same five quarters.

**Every reading above 8 is a trough**, and at those dates `nairu` says the labour market was slack
by 2.3, 4.6 and 1.8 points. So the high readings are not measuring a high u\*, they are measuring
actual unemployment during three disinflations. Around 7 is defensible for the early 1990s; above
8 is not supported by anything here, and the 10.83 is the same artefact that gives `ystar_ustar`
10.77 for 1993.

**Two caveats against the model that happens to agree.** `nairu`'s estimate is suspiciously flat,
6.13 to 6.26 across fourteen years spanning a boom, a deep recession and a disinflation, which is
a smoothness prior at work and may be under-moving. And both methods read the same unemployment
series, so they are independent only in what they do with inflation.

**What would change the conclusion**: a *plateau* reading above 8, meaning inflation genuinely
level rather than at a minimum, for two years, with unemployment above 8. No such episode exists
in the sample.

### Two early-1960s readings, from different tests

The 1962Q3-1963Q2 trough reads **2.21** and the 1967Q2-1969Q3 plateau reads **1.82**. Different
quarters, different tests, four tenths of a point apart, in a decade no model in this package can
reach. That agreement is the best evidence here that the early-1960s reading is real rather than
an artefact of one rule.

A note on how nearly it was lost: an earlier version removed plateau-claimed quarters before
grouping troughs, which split 1962Q3-1963Q2 at its 1963Q1 (a plateau quarter) into runs of two
and one, both below the three-quarter minimum, and dropped the whole episode. Troughs are now
found whole and discarded only when a plateau covers all of them.

### What the sweep does and does not show

Moving each of the rule's numbers one at a time gives readings of 1.73-1.88 in the 1960s and
5.41-5.54 in the 2010s, and an early draft of these notes called that robustness. **It is not.**
Within a selected window unemployment is itself flat, so the reading cannot move much whatever
the rule does. The stability is arithmetic.

The informative column is which episodes appear at all, and there only one thing moves: under
loose settings (`smooth` 6, `tolerance` 1.5 or 2.0, `window` 6) the **early 1990s appear and read
9.0 to 10.9**. See below.

### The level readings are close to a local average

Comparing each episode with the ten years around it:

| episode | u in episode | u over surrounding 10 years | difference |
|---|---|---|---|
| 1967Q2-1969Q3 | 1.82 | 1.95 | -0.13 |
| 1988Q4-1989Q4 | 6.20 | 8.55 | -2.35 |
| 2002Q4-2005Q2 | 5.57 | 5.77 | -0.21 |
| 2013Q1-2014Q1 | 5.68 | 5.39 | +0.29 |
| 2015Q4-2019Q4 | 5.45 | 5.17 | +0.28 |

Outside 1988-89, which is the weakest row anyway, the flat-inflation criterion lands within about
a quarter of a point of averaging unemployment over the surrounding decade.

Two readings of that are available and the notes decline to pick one. It may mean the inflation
information adds nothing over a smoother. Or it may be what the theory predicts: if u\* is where
unemployment settles when inflation is stable, then flat-inflation windows *should* sit near the
local average, and the coincidence is corroboration rather than vacuity. What separates them is
the contrast below, not the levels.

### The direction contrast, which is the model's actual test

Mean unemployment by what inflation was doing, over the same centred window:

| decade | rising | flat | falling | ordering |
|---|---|---|---|---|
| 1960s | 1.72 | 1.86 | 2.30 | **rise < flat < fall** |
| 1970s | 2.80 | 6.19 | 5.45 | breaks |
| 1980s | 7.12 | 6.22 | 8.59 | rise < fall |
| 1990s | 8.69 | 8.69 | 8.77 | no signal |
| 2000s | 5.27 | 4.92 | 5.89 | rise < fall |
| 2010s | 5.39 | 5.43 | 5.58 | **rise < flat < fall** |
| 2020s | 4.86 | 4.88 | 4.01 | **breaks** |
| whole | 5.09 | 5.06 | 6.04 | rise < fall |

If unemployment carries information about where inflation is going, the ordering runs
rising < flat < falling. It does, cleanly, in the **1960s and the 2010s**. It **breaks in the
1970s and the 2020s**, which are the two supply-shock decades: inflation rose through 1973-74
with unemployment at 2.8%, and fell after 2022 while unemployment fell. The 1990s show nothing
at all, which is the re-anchoring again.

### The 1990s, and why the two series came apart

Inflation was above 5% in **89% of quarters from 1973 to 1991**, including an unbroken 11.2-year
run from 1973Q1 to 1984Q1. The natural reading of the 1990s is that unwinding twenty years of
that was slow and painful. The second half is right and the first is not, and the distinction is
what makes the decade's bars flat.

| | 1990Q1 | 1991Q1 | 1992Q1 | 1993 | 1995Q1 | 1999Q1 | 2003Q1 |
|---|---|---|---|---|---|---|---|
| inflation | 8.7 | 4.9 | 1.7 | 1.2 | 3.9 | 1.2 | |
| unemployment | 6.3 | 8.5 | 10.3 | **11.1 peak** | 8.8 | 7.1 | 5.9 |

**The disinflation took two years. The unemployment cost took thirteen to fifteen.** Inflation
fell 8.7 to 1.7 between 1990Q1 and 1992Q1 and then stopped. Unemployment peaked at 11.1 in 1992Q4
and took this long to come back, depending on where the finish line is set:

| unemployment back to | from the 1992Q4 peak | from 1990Q1 |
|---|---|---|
| 8.0 | 5.0 yrs | 7.8 yrs |
| 7.0 | 6.5 yrs | 9.2 yrs |
| 6.3, its 1990Q1 level | 7.8 yrs | 10.5 yrs |
| **5.7, its best of 1985-89** | **11.0 yrs** | **13.8 yrs** |
| 5.0 | 12.8 yrs | 15.5 yrs |

Unemployment takes the elevator up and the stairs down; inflation does not.

**Which means u\* was carrying the recession long after the nominal problem was solved.** By
1998 inflation targeting had done its work: the expectations series had reached 2.5 and stayed.
Yet the joint model's u\* was still 7.18 at 1998Q4, 6.60 in 2000Q3, and did not reach 6.0 until
2003Q1 or 5.5 until 2005Q3. Nothing nominal was unresolved across those seven years. What u\* was
tracking was the labour market still climbing the stairs, because in these models u\* is close to
a smoothed unemployment path (see the joint model's notes, where it follows
`u + beta_okun·c·(pi − anchor)` at correlation 0.949).

That is worth stating plainly because it cuts against the natural reading of a falling u\*. The
descent from 7 to 5 across 1998-2005 looks like a structural improvement being discovered. It is
better described as an estimate still burdened by a recession that the inflation data had
finished with a decade earlier.

That asymmetry is the mechanism behind the empty 1990s row. From 1993 inflation had arrived at
its new level and only wiggled, 1.2 up to 3.9 and back to 1.2, while unemployment was on a long
independent descent from 11 to 7. So rising-inflation and falling-inflation quarters occurred at
high unemployment *and* at low unemployment, and both average to 8.7. The decade is not a Phillips
curve with a weak signal. It is one series that had finished moving sitting beside one that had
not started.

It is also why every method in this package returns roughly 10.7 for 1993. They all read a
stationary inflation rate next to an 11% unemployment rate and conclude the two are compatible,
which they were, temporarily, for reasons that have nothing to do with equilibrium.

### The scarring was earlier and deeper than the 1990s

"Hysteresis" is not a candidate answer to why the recovery was slow: it names the condition, that
where unemployment has been affects where it settles, without saying through what. The mechanisms
it labels are separate claims. Three of them can be dated with series already in this package, and
when they are, **the damage turns out to belong to 1974-1983 and not to the decade that followed
the 1990-91 recession at all.**

**Profit share** (capital share of factor income, GOS/(GOS+COE), ABS 5206.0):

| period | mean | min |
|---|---|---|
| 1959-1969 | 27.6 | 24.4 |
| 1970-1974 | 26.4 | 20.9 |
| **1975-1983** | **24.1** | **20.8** |
| 1984-1989 | 29.7 | 27.2 |
| 1990-1993 | 30.1 | 28.0 |
| 1994-1999 | 31.1 | 29.9 |
| 2000-2005 | 32.4 | 30.7 |

The trough is 1975Q1 at **20.8%**, nearly seven points below the pre-1973 mean of 27.6, and it
took **nine years, to 1984Q1**, to recover. Automatic wage indexation converted the oil shock into
wages and the margin absorbed it. Recovery arrives with the Accord and the end of the
pass-through, not with anything monetary.

**Private investment** (GFCF, per cent of GDP, chain volume):

| 1975-1983 | 1984-1989 | 1990-1993 | 1994-1999 | 2000-2005 |
|---|---|---|---|---|
| 12.5 | 14.5 | 13.6 | **16.0** | **18.1** |

**Capital deepening** (annualised growth, capital minus hours):

| period | capital | hours | employment | deepening |
|---|---|---|---|---|
| **1975-1983** | 5.01 | 0.71 | **0.97** | **4.29** |
| 1984-1989 | 4.25 | 3.54 | 3.47 | 0.70 |
| 1990-1993 | 2.25 | -0.46 | -0.39 | 2.71 |
| 1994-1999 | 3.40 | 2.03 | **2.04** | 1.37 |
| 2000-2005 | 3.70 | 1.43 | 2.11 | 2.27 |

**1975-1983 is the scarring, and all three series agree on it.** Capital grew at 5% a year while
hours grew 0.7%, deepening at 4.3 points a year, with employment growth of barely 1% across nine
years and the profit share at its floor throughout. Labour had been made expensive and risky by
indexation, so firms substituted capital for it and stopped hiring. That is where the level of
unemployment ratcheted up, and it is a different event from the 1990-91 recession.

**It stops with the Accord.** Over 1984-89 deepening falls to 0.70 and employment grows 3.5% a
year. The substitution ends when the pass-through mechanism does.

**And the 1990s look normal rather than scarred.** Through 1994-99 employment grew 2.0% a year,
margins sat above their pre-1973 norm, and investment was rising. That is not a cautious economy
and not a constrained one. Unemployment still took thirteen years to normalise because it started
from 11.1%, and four points of recovery at ordinary hiring speed takes about that long.

So u\* was burdened by history, but the history that bound it is 1974-1983, and the channel is the
ratcheted level of unemployment those years left behind, not managers still flinching in 1997. The
scarring was earlier and deeper than the decade in which it was still being paid off.

Two cautions. The investment ratio is a chain-volume one, and the capital-goods deflator fell
relative to GDP over this period, so part of the rise after 1994 is a price artefact; a nominal
ratio would be flatter. And a rising profit share alongside high unemployment is also what weak
labour bargaining power looks like, so these series rule the profitability channel out as a
*constraint* in the 1990s without establishing what it was doing to hiring.

### What the firm-side story would need, and why the data cannot supply it

A residue of the inflation era in how managers set hurdle rates, wages and prices is not refuted
by the tables above: those measure balance sheets and behaviour in aggregate, not beliefs. The
nearest thing to evidence in this package is the expectations model's own series, which does not
behave like a switch being thrown: 7.46 in 1990Q1, 2.34 by 1993Q1, then back up to 3.50 in 1995Q1
and 3.04 in 1997Q1 before settling at 2.4-2.5 from 1999. Five years of sitting a point above the
target after the disinflation was complete.

**But it is weak evidence, and mostly for a different population.** The expectations model's panel
is bond breakevens from 1986Q3, a business survey from 1989Q3, and market economists from 1993Q3.
G3's union series is fetched by the loader but never reaches the model (`stage1.py` selects
market_1y, business and market_yoy), and has in any case ended at 2023Q3, the unions having
declined to keep participating. So across the decade in question the panel is the bond market
plus one survey, with the financial-market measure the only one present throughout. What that
panel
measures is what traders and forecasters expected the CPI to print. The mechanism here is about
something else: the hurdle rates, pricing conventions and wage offers of managers trying to
restore margins, which nobody surveys and which need not move with a breakeven. Slow-settling
market expectations are consistent with the story and are not a measurement of it.

**Households are the third population, and there is no series for them.** A generation that had
lived through two decades of inflation is the group a political line about the recovery lasting
five minutes has to land on, and the package cannot measure it. RBA table G3 does carry a
consumer series, `GCONEXP`, but it holds **18 observations beginning 2022-03**, so
`src/data/expectations.py` is not omitting a long one: it does not exist there. A household
measure for the 1990s would have to come from the Melbourne Institute survey directly, which is
a data-collection job nobody in this package has done. Anyone who goes looking for household
expectations in the existing loaders should stop here rather than repeat the search.

So the expectations story has three populations, and this package observes one and a half of
them: markets throughout, firms' stated expectations from 1989Q3, unions from 1996Q3, managers'
decision rules never, and households not before 2022.
The asymmetry is measured; the mechanism behind it is not, and calling it hysteresis would only
restate the measurement in Greek.

That is worth more than the level readings, because it measures the headline measure's known
weakness rather than asserting it: the Phillips mechanism is visible in these data when supply is
quiet and invisible when it is not.

---

## Tried and deleted: a state-space NAIRU over the same period

The obvious next step was a state-space model over 1960-2026: a driftless u\* observed through an
accelerationist Phillips curve, `Δpi` on the unemployment gap with an import-price control. The
accelerationist form needs no target and no expectations series, which is what would have let it
run back to 1960 where `ystar`, `ustar` and `ystar_ustar` cannot.

It was built, it failed three ways, and it has been deleted along with its runs and the
import-price loader written for it. Nothing from the specification is worth keeping; what follows
is the only part that was.

**Three specifications, three failures:**

| | outcome |
|---|---|
| Phillips alone, Student-t walk | residual sd 0.94 against a dependent variable of sd 0.97, so ~6% of variance explained; 90% band 5.8 points wide, widening to 10.4 when `sigma_ustar` went 0.10 to 0.30 |
| plus a reversion equation, `Δu = −kappa·(u − u*)`, Student-t | u\* became a copy of u: 11.9 in 1983, **10.9 in 2020Q1** against actual 7.0. `nu` collapsed to 1.185, at its floor, so near-Cauchy tails made the jumps free. `gamma_pi`'s P(<0) fell from 0.985 to 0.791 |
| the same, Gaussian walk | `kappa` estimated at **0.013 [0.000, 0.028]**, i.e. the reversion switched itself off, and u\* flattened to a near-constant 5.1 |

**Why no calibration was ever going to work.** The three rows above are not three unlucky
settings, they are the two ends of one dial with nothing in between. It was hard to get enough
genuine movement in u\* without it hugging the unemployment rate, and that difficulty is the
diagnosis rather than a tuning problem: when a variance is identified the data choose it and a
middle ground exists; when it is not, the dial runs between two failure modes and passes through
nothing useful. Turn it down and u\* is a smooth prior line ignoring a series that ranges from 1.4
to 11.1. Turn it up, or buy the movement with fat tails and an equation keyed to unemployment, and
u\* copies the series it is supposed to be a benchmark for. The 10.9 reading for 2020Q1 is the dial
at that end.

What would separate "genuine movement" from "u" has to be an observation equation carrying content
from outside the labour market. That is what Okun plus an inflation-defined output gap supplies in
`ustar`, and it is why that model can impose `sigma_ustar` = 0.020 and defend it with a sweep,
while nothing here could defend any value.

The same phenomenon is visible in the models that *do* have the extra equation, which is the
uncomfortable part: `ystar_ustar`'s u\* follows `u + beta_okun·c·(pi − anchor)` at correlation
0.949 with a mean absolute error of 0.38 points. Even there the level is largely the unemployment
rate plus a small correction, and it is the smallness of the correction, not its absence, that
makes the result look like an estimate.

**The conclusion, and why it justifies the package's design.** Inflation and unemployment alone do
not contain enough information to locate a NAIRU over the long period. Every route either finds
nothing or finds the unemployment rate wearing a different label. `ustar`'s own notes say Okun
"sets the level from a given output gap" while the Phillips curve "supplies the nominal content";
this was that claim tested by removing Okun, and the nominal content alone yields a prior with a
slight tilt. It sits beside the package's other central negative, that interest rates do not
visibly move real activity on Australian data.

**Which is the argument for the rule in this module.** It reaches the same decades and does not
pretend to estimate: five plateaus, six troughs, and silence in between. The silence is the
honest part, and it is what the state-space version could not manage.

Deleted rather than kept because a model that reports a prior is worse than no model: someone
would eventually quote its line.

## What it says about 1993, and about the other models

The `ystar_ustar` and `ustar` models both open at u\* ≈ 10.75 in 1993Q1 against unemployment of
10.93, which is not credible as a structural statement and is shaded on their charts as not
well identified. The obvious suspicion is that this is an artefact of their specification: an
inflation-defined gap, an imposed `sigma_okun`, a decay state law, a left endpoint at a regime
break.

**It is not.** This model shares none of that machinery, and under a loose rule it reads
**9.0 to 10.9 for the early 1990s** — the same answer. Inflation was flat in 1993-94 because
expectations had collapsed and were settling at a new level, not because the labour market was
in balance, and *any* non-accelerating-inflation rule reads that as equilibrium.

So the early-1990s problem belongs to the concept, not to the model. That is worth more than a
correction to `ystar_ustar` would have been, and it is why the two attempts to fix the level
from the nominal side failed: there was nothing there to fix.

---

## Limitations

1. **Headline CPI carries supply shocks.** Flat headline inflation can be offsetting demand and
   supply pressure rather than balance. The trimmed mean starts in 1983 and the whole point is
   the decades before that, so headline is what there is. Not yet done: comparing headline and
   trimmed episodes over the 1983-onward overlap, which would tell you how much to trust
   headline where only headline exists.
2. **No expectations control**, and it cannot have one before 1983Q1, where the expectations
   model's series begins. The concept wants inflation stable *relative to what was expected*,
   and re-anchoring is exactly when that comes apart. This is the 1990s failure above, stated
   as a missing variable.
3. **`--require-flat-u` is the one correction that works over the whole sample**, and it is off
   by default. Requiring unemployment to be flat too is the difference between "inflation had
   settled" and "both had settled". At `u_tolerance` 0.5 it leaves 2 episodes, at 1.5 it leaves
   5; the sweep carries the grid rather than the code carrying a choice.
4. **The sample starts 1959Q3**, where 1364.0.15.003 begins, so the 1950s are absent. Extending
   needs the RBA OP8 CES-registered backcast, which the ABS notebooks implement
   (`abs_spliced_series.get_unemployment_rate`) and this package does not.
5. **Episodes are not a sample.** They are chosen by a rule, they cluster, and there are five of
   them over 67 years. Nothing here supports a standard error, and none is offered.
6. **The 1980s reading of 6.20 sits on 5 quarters** at inflation of 7.5%, which is flat only in
   the sense that it had stopped rising. Treat it as the weakest row in the table, while noting
   it is also the row `nairu` independently confirms at 6.22.
7. **Trough readings are cyclically contaminated by construction** and must not be quoted as u\*
   without the qualification. They are collected because seeing the bias beats not measuring it,
   and because the 1962-63 one is genuine. Everything above 8 in this package's output comes from
   them.

---

## Files and usage

```
src/models/long_run_ustar/
├── config.py         # the rule's three numbers, and the two optional filters
├── observations.py   # long CPI + the quarterly unemployment rate
├── model.py          # flatness, plateaus, U-troughs, stationary-u windows, the contrast
├── sweep.py          # move one number at a time
├── analyse.py        # tables and charts
└── run.py            # CLI

src/data/long_cpi.py  # headline CPI back to 1948, rebuilt from the quarterly change
```

```bash
./run-long-run-ustar.sh
./run-long-run-ustar.sh --require-flat-u      # both series settled, not just inflation
./run-long-run-ustar.sh --window 12           # a longer stretch must be flat
./run-long-run-ustar.sh --tolerance 1.5       # picks up the 1990s false positive
./run-long-run-ustar.sh --smooth 1            # finds nothing before 2002
./run-long-run-ustar.sh --require-target      # post-1993, flat AND near 2.5
```

Four charts land in `charts/LongRunUStar/`:

- **where-inflation-stopped-moving** — inflation raw and smoothed, with plateaus shaded green and
  U-bases purple.
- **what-inflation-says-u-was-two-ways** — the unemployment rate with each reading drawn across
  its own span, orange for plateaus and purple for troughs, and nothing in between. The gaps are
  the point: the rule is silent where inflation was moving.
- **where-unemployment-held-still-and-whether-it-counted** — every stretch where unemployment was
  flat, green where inflation was flat too and red where it was not. Shows which levels were
  excluded and why, and that the excluded ones skew low.
- **unemployment-by-what-inflation-was-doing** — the direction contrast by decade, which is the
  model's test rather than its reading.
