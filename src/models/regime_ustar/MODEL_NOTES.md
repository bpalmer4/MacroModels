# Regime u\*: a spline over imposed regimes, back to 1959

## Read this first: the gap is the inflation gap rescaled

The model has **one observation equation**. Inverting it,

```
    u_t - ustar_t  =  - u_t * (pi_t - pi_e_t) / beta
```

So the unemployment gap is the inflation gap multiplied by `u/beta`. **No
labour-market observable enters anywhere except `u` itself, which appears on
both sides.** Nothing about the regime dates, the spline, the knot
multiplicities or the functional form changes that.

Read the path as a statement about what inflation surprises imply, smoothed and
given a shape. Do not read it as independent evidence about the labour market,
and do not quote a gap from it as a measure of slack.

Two attempts were made to break this. A wage equation on unit labour costs was
built and turned off (below). Okun was specified and not built.

---

## What it is for

`ystar`, `ustar` and `ystar_ustar` all start at 1993Q1 because their gap is
defined against an inflation target that did not exist earlier. This model
reaches **1959Q3** by making the expectation regime-dependent: asserted before
1983Q1, measured after.

That reach is the point, and it bears on a specific defect elsewhere. `ustar`
opens at 1993Q1 with a diffuse prior, so u\* there is placed by the unemployment
rate sitting beside it, and it reports **10.75** against unemployment of 10.93.
No repair from inside that sample can work: inflation in 1993 was at target, so
the Phillips curve sees no disequilibrium to attribute. Running from 1959 means
u\* ARRIVES at 1993 carrying a level inherited from the 1980s, and this model
reads **6.92** there.

`long_run_ustar` reaches the same decades by a different route, reading
unemployment off stretches where inflation was flat. It estimates nothing and
is silent between episodes. This one estimates and is never silent, which is
both what it adds and what to distrust about it.

---

## The specification

```
OBSERVATION EQUATION  (one equation, this is the whole likelihood)

    pi_t - pi_e_t  =  -beta * (u_t - ustar_t) / u_t  +  supply_t  +  e_t

    e_t ~ StudentT(nu, 0, sigma_m)          nu = 2 + Exponential(1/5)
    supply_t = rho * d4pm_t + xi * GSCPI_t^2 * sign(GSCPI_t)


EXPECTATIONS  (regime-dependent; switch at T0 = 1983Q1)

    pi_e_t  =  pi_{t-1}                     for t <  T0    asserted, adaptive
            =  E_t from expectations model  for t >= T0    measured

    m = 0 before T0, 1 after; the residual scale sigma_m is separate either side


STATE EQUATION  (deterministic given the coefficients)

    ustar_t  =  sum_j  c_j * B_j(t)

    B = natural cubic B-spline basis, a partition of unity
    knots at 1974Q1 (multiplicity 3), 1983Q3, 1993Q1, 2015Q1, 2020Q1


PRIORS                                                  posterior
    c_j    ~ TruncNormal(5, 3) on [0.5, 12]             see below
    beta   ~ HalfNormal(1.5)                            2.628 [1.814, 3.407]
    sigma_0, sigma_1 ~ HalfNormal(1.0)                  0.728, 0.968
    rho    ~ Normal(0, 0.1)                             0.022 [-0.006, 0.050]
    xi     ~ Normal(0, 0.1)                             0.144 [ 0.067, 0.225]


DATA
    pi   year-ended headline CPI          src/data/long_cpi.py
    u    quarterly unemployment rate      ABS 1364.0.15.003
    E    expectations model output        from 1983Q1
    d4pm import price growth, lagged      from 1984Q3, zero-filled before
    GSCPI supply-chain pressure, lagged 2 from 1998Q1, zero-filled before

    Sample 1959Q3-2026Q2, 268 quarters.
```

### Three choices that carry weight

**The proportional gap, `(u - ustar)/u`**, following `ustar/estimate.py:212` and
`ystar_ustar/estimate.py:411`. Inverting gives `ustar = u * (1 + surprise/beta)`,
so the factor turning an inflation surprise into an unemployment statement is
`u/beta` and therefore SCALES WITH THE UNEMPLOYMENT RATE: small when the labour
market is tight, large when slack. Under a level gap the 1960s came back at
2.49 against mean unemployment of 1.92, because `+0.28` of average surprise was
multiplied by `1/beta` regardless of unemployment being at 1.9. There is also
direct evidence for convexity: `ystar_ustar`'s notes split its own sample and
find a slope of -1.82 on the tight side against -0.45 on the slack side.

**The triple knot at 1974Q1.** A cubic knot of multiplicity 3 gives C0 there,
matching the level and freeing the slope to turn a corner. The ratchet was a
break, not a transition. Under C2 the model could not turn a corner and read
1967-69 at 1.90; with the corner it reads **1.82**, against `long_run_ustar`'s
1.82 from a method with no Phillips curve. It did NOT fix the 1970-73 rise,
which turned out to be in the data (below).

**The 1983Q1 expectations switch is a measurement change as well as a regime
change**, because the expectations model's series begins exactly there. A level
shift in u\* at 1983 could be either and nothing in the sample separates them.
This is the model's weakest join.

---

## Results (2026Q2 vintage)

| regime | qtrs | mean u | mean surprise | u\* start | u\* end | u\* min | u\* max |
|---|---|---|---|---|---|---|---|
| 1959Q3-1973Q4 | 58 | 1.98 | +0.20 | 1.99 | 3.12 | 1.74 | 3.12 |
| 1974Q1-1983Q2 | 38 | 5.69 | +0.11 | 3.24 | 7.29 | 3.24 | 7.29 |
| 1983Q3-1992Q4 | 38 | 8.23 | -0.35 | 7.34 | 6.95 | 6.95 | **7.60** |
| 1993Q1-2014Q4 | 88 | 6.45 | -0.03 | 6.92 | 4.74 | 4.74 | 6.92 |
| 2015Q1-2019Q4 | 20 | 5.54 | -0.75 | 4.69 | 4.22 | 4.15 | 4.69 |
| 2020Q1-2026Q2 | 26 | 4.48 | **+1.20** | 4.27 | **6.20** | 4.27 | 6.20 |

Peak **7.60 at 1986Q2**. Latest **6.20 [5.25, 7.50] at 2026Q2**.

Two segments do something a constant could not: 1983-92 peaks partway through
and comes down, and 2015-19 falls and flattens. That is what the cubics buy
over one level per regime.

### Against methods that share none of this machinery

| episode | this model | source | reading | diff |
|---|---|---|---|---|
| 1967Q2-1969Q3 | **1.81** | `long_run_ustar` plateau | 1.82 | **-0.01** |
| 1988Q4-1989Q4 | 7.41 | `long_run_ustar` / `nairu` | 6.20 / 6.22 | **+1.21** |
| 2002Q4-2005Q2 | 5.99 | `long_run_ustar` plateau | 5.57 | +0.42 |
| 2013Q1-2014Q1 | 4.97 | `long_run_ustar` plateau | 5.68 | -0.71 |
| 2015Q4-2019Q4 | 4.27 | `long_run_ustar` plateau | 5.45 | **-1.18** |

The 1960s agreement is the strongest corroboration here: two methods with
nothing in common landing on 1.81 and 1.82 in a decade no other model in the
package can reach.

`corr(u*, 5-year moving average of u)` is **0.906**, which is the number to
watch. It was 0.781 with the wage equation on, and that was the only
specification that moved it much.

---

## Known defects, all four the same defect

Every persistent inflation surprise is booked to the labour market, because
there is one equation and nothing else can absorb one.

**1970-73, u\* rising 2.0 to 3.1 while unemployment was 2.1.** Inflation went
2.9% to 13.1% across those years, and under an adaptive expectation almost all
of it arrives as positive surprise: mean **+0.64** over 1970-73 against **-0.03**
over 1965-69, with +1.99 in 1971Q3 and +2.94 in 1973Q4. The equation can only
read that as an overheated labour market. Much of it was the first oil shock and
the wool and wheat boom, which import prices would control for and cannot,
because they start 1984Q3.

**2020-26, u\* rising to 6.20 against unemployment of 4.35.** The same thing at
the other end. Mean surprise in that regime is **+1.20**, with 2022Q4 at +5.17.
The GSCPI term absorbs some of it and nowhere near enough: `--no-supply` puts
the end point at **6.48** against 6.20 with it, so the supply block is worth
0.28 points at the place it should matter most.

**2015-19 reading 4.27 against 5.45**, the same defect with the sign reversed:
six years of surprises from -0.42 to -1.15 read as slack. Moving the knot to
2015Q1 to isolate the period changed it by 0.01, which establishes that this is
not a knot-placement problem.

**1988-89 reading 7.41 against 6.20**, where two independent methods agree with
each other at 6.20 and 6.22.

**Ill-conditioning.** The coefficients run 1.99, 0.87, 3.24, 4.94, **8.56**,
5.90, 6.19, 3.42, 6.20 on a partition-of-unity basis, where they should roughly
track a curve that peaks at 7.60 and declines. One sits above the curve's
maximum and the neighbours below it, so adjacent basis functions are partly
cancelling. Adjacent basis functions are partly cancelling. More
knots or a higher degree would make this worse.

**The 1993-2014 segment is 88 quarters under one cubic**, spanning the
disinflation, the mining boom and the GFC. It is the longest segment by far and
the least likely to be adequately described by four parameters.

---

## Tried and rejected

Each of these ran, sampled cleanly and was turned off or replaced. Kept because
the negative results are most of what the exercise established.

**A step function, one free level per regime.** The first attempt. `beta` came
back at **0.039**, so `1/beta` was 26 and every level was its prior: the 1960s
read 4.96 against mean unemployment of 1.98, the 1980s 4.25 against 8.23, and
posterior bands were 6.4 to 7.5 points wide against a prior band of 8.40. The
level solves as `mean u + mean surprise / beta`, which for the 1960s is
`1.98 + 0.20/0.039 = 7.1` before the prior truncation drags it back. Discarded
for the spline, which cannot be a step and has no innovation variance.

**An attractor state law**, `ustar_t = ustar_{t-1} + phi(eq_k - ustar_{t-1}) + e`.
Better than the step function and still limited in two ways: it needs
`sigma_ustar` imposed, which nothing measures, and an exponential approach to a
level can only do one shape. Tested with the 1983-97 boundary, it came back flat
at 7.3 across a period containing both a peak and a descent. Retained as
`--state attractor` for comparison.

**Terms of trade as the supply control.** `gamma_tot` = **0.010
[-0.022, +0.043]**, straddling zero, with every path value unchanged to two
decimals. Tested twice, on the level-gap and the proportional-gap
specifications, in case the first result was an artefact of the older spec. It
was not.

**Import price growth**, kept but doing nothing: `rho_pi` = 0.022
[-0.006, 0.050]. Only the GSCPI term earns its place, `xi` = 0.144
[0.067, 0.225]. The supply pressure that registers is supply chains, not the
import price index.

**Adaptive innovations**, `sigma_t = sigma_u * (m_t / mean(m))^kappa` with
`m_t = |MA4(u)_t - MA4(u)_{t-8}|`, letting u\* move more where unemployment had
moved. `kappa` = **1.418 [0.242, 2.604]**, excluding zero, and the mechanism
worked: `sigma_t` ran 0.41 in 1983Q4 and 0.39 in 1992Q2 against 0.007 through
1966-68, a 55-fold range. It changed the answer by almost nothing, because with
`sigma_u` at 0.05 the path was driven by the attractors rather than the
innovations. Moot under the spline, which has no innovations at all. Retained as
`--adaptive-sigma` on the attractor state.

**A wage equation on unit labour costs**, `ulc_yoy - pi_e = alpha + gamma*(u -
ustar)/u + lambda*dU/U + v`, after `nairu/equations/phillips_wage.py`. This was
the attempt on the circularity, and it is the only change that brought a second
series. It is **not weak**: `gamma_wage` = **-4.667 [-5.789, -3.626]**, about 2.8
times the price slope in the same units, `alpha_wage` = -0.005 (trend real ULC
growth of zero over 66 years, which nothing forced), and
`corr(u*, smoothed u)` fell **0.881 to 0.781**, the largest such fall anything
achieved.

It is off because the PATH is worse everywhere it can be checked. Those
comparisons were made at the 2008Q4 knot configuration and have not been rerun
since it moved to 2015Q1. The hump
flattens from a peak near 7.7 to near 6.5; the early-1970s rise worsens from
3.14 to 4.37 by 1973Q4; the end point rises 6.14 to 6.65; and 1993-2001 acquires
a fall-then-rise no account of the period supports. The 1988-89 benchmark does
improve sharply, 7.27 to 6.32 (measured at the 2008Q4 knot, before it moved),
and that was initially read as the equation working. The likelier reading is that the whole hump fell about a point and that
benchmark sat under it. One benchmark improving while the shape deteriorates is
not evidence. Retained as `--wage`.

**A 2008Q4 knot.** Moved to 2015Q1 because 2008Q4 marks the GFC, a demand shock,
not a change in the inflation regime. Over 2009-2013 inflation averaged 2.42 and
the surprise -0.20, with 2011 at 3.31: an ordinary stretch inside the band. The
break is at 2015, where inflation goes 2.47 to 1.51 to 1.25 and the surprise
goes -0.11 to -1.05 to -1.15 and does not recover before the pandemic. The move
improved 2013-14 by 0.26 and 2002-05 by 0.15, left 2015-19 unchanged, and raised
`corr(u*, smoothed u)` from 0.881 to 0.906. A judgement rather than a clear win.

**A Henderson trend through the raw inversion** (`inversion.py`, still present
as a separate exercise). It rings. Henderson weights pass a cubic exactly at the
cost of negative outer weights, so on an input as noisy as an inflation surprise
divided by a slope the side lobes manufacture oscillations at about the filter's
own length. A least-squares fit of a low-order polynomial has no side lobes and
cannot invent a cycle, which is the cleanest argument for the spline. The
inversion module is kept because the raw per-quarter series is the right
diagnostic for how much of the answer the smoothing supplied.

---

## Not done

**Okun.** Specified and not built. `log GDP = ystar + ygap` with `ystar` a
second spline on the same knots, and `u - ustar = -beta_okun * ygap`. GDP covers
1959Q3-2026Q2, so it is possible. It is the strongest remaining candidate
because it is the only untried change that brings a new series, and it would
address both large errors directly: if output was strong in 2022-23, Okun says
the gap was genuinely negative and u\* need not rise.

Three costs, stated so the next attempt does not rediscover them. Two latent
splines roughly doubles the parameters on a basis already ill-conditioned.
`ystar` and `ustar` become partly interchangeable, since both are smooth latents
linked by Okun, and `ystar_ustar` handles that by imposing `sigma_okun` at 0.20,
which its own notes call the one imposed variance in the package with no
external anchor. And the lockdown quarters would bend `ystar` badly right before
the segment whose end point is already the weakest part; `ystar` excludes
2020Q2-2021Q3 for this reason.

**A degree-4 spline.** Not worth it. Cubic to quartic with five knots takes the
coefficient count from 9 to 10, so it buys one parameter, while increasing
oscillation and worsening boundary extrapolation at the one place the model is
weakest. Flexibility is controlled by knots, not degree.

**More knots.** The raw inversion is the limit the spline approaches as knots
are added, and it reads 4.14 over 2015-19 against the spline's 4.27 and the
target's 5.45. So loosening moves that period further from the benchmark. The
current errors are level errors, not stiffness errors.

---

## Files and usage

```
src/models/regime_ustar/
├── config.py          # regimes, knots, priors, and what each switch costs
├── observations.py    # the series, and the regime-dependent expectation
├── spline.py          # the natural cubic basis, knot multiplicity
├── estimate.py        # both state laws, the price equation, the wage equation
├── analyse.py         # tables and charts
├── inversion.py       # the raw per-quarter inversion, and Henderson on it
├── inversion_run.py   # CLI for the inversion exercise
└── run.py             # CLI
```

```bash
./run-regime-ustar.sh                        # the default spec above
./run-regime-ustar.sh --wage                 # add the ULC equation
./run-regime-ustar.sh --no-supply            # drop import prices and GSCPI
./run-regime-ustar.sh --state attractor      # the pre-spline state law
./run-regime-ustar.sh --breaks 1974Q1 1983Q3 1998Q1 2015Q1 2020Q1
./run-regime-ustar-inversion.sh --window 21  # the inversion exercise
```

Charts land in `charts/RegimeUStar/`:

- **u-as-a-spline-with-knots-at-the-regime-dates**: u\* against the unemployment
  rate, with the knots marked.
- **unemployment-gap-implied-by-the-regimes**: `u - u*`, which is the inflation
  gap rescaled and should be read as such.
- **what-inflation-alone-says-u-is-quarter-by-quarter**: the acceptance test.
  The grey line owes nothing to the knots or the spline; if the orange path is
  not a plausible reading of it, the structure is inventing the answer.
- **each-regimes-attractor-and-where-u-actually-got-to**: only meaningful on
  `--state attractor`.
