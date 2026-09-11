# MacroModels

Australian macroeconomic modelling. Includes both Bayesian state-space estimation (PyMC) and deterministic growth accounting methods.

## Overview

### Supply side and structural estimation

- **NAIRU + Output Gap**: Bayesian state-space model nominally estimating the natural rate of unemployment and potential output jointly. **Superseded for potential output, the output gap, and the NAIRU itself.** This model does not estimate potential output: its posterior median traces the Cobb-Douglas production-function input turning point for turning point across forty years, ending at 1.66 against the input's 1.74, so the credible bands are bands around a number the data never moved. Worse, that input is not a potential path: potential growth falls from 4.8% to 0.15% between 1990 and 1992, is back at 4.8% by 1996, and goes negative in 2020. Potential output does not contract during a lockdown, measured output does. The series is tracking the cycle, which means the output gap is close to actual minus a filtered version of actual, the cyclical signal is absorbed into "potential" before the Phillips curve ever sees it, and Okun carries the damage straight into the NAIRU. **Use `ystar` for potential growth, and the joint y\*/u\* model for the output gap and u\*.** What it keeps: the wage equation, the expectations-to-target anchor transition, the regime split, and the LOO/WAIC comparison across variants that nothing else in the package has. Those are about inflation dynamics and the re-anchoring narrative, and they are the reason to run it. The NAIRU it prints is not. See [`MODEL_NOTES.md`](src/models/nairu/MODEL_NOTES.md)
- **Inflation Expectations**: Bayesian signal extraction model estimating latent expectations from surveys and market data — see [`MODEL_NOTES.md`](src/models/expectations/MODEL_NOTES.md)
- **Cobb-Douglas MFP**: Deterministic growth accounting decomposing output into capital, labour, and productivity
- **y\* (Potential Output)**: Bayesian unobserved-components model where potential is a slow-moving random walk and the output gap is *defined* by inflation's deviation from the 2.5% target — no Phillips curve, no IS curve, no policy rule. Self-contained (imports only `src/data`). **The preferred source for potential growth** (1.94% y/y), where the joint model agrees and this is the simpler statement of the same answer. **Superseded for the output gap** by the joint y\*/u\* model, which finds a gap about twice as wide. See [`MODEL_NOTES.md`](src/models/ystar/MODEL_NOTES.md)
- **u\* (NAIRU from a given output gap)**: A deliberately small Bayesian model — one state, two observation equations — taking `ystar`'s output gap as an input rather than estimating it. Okun fits unemployment, an expectations-augmented Phillips curve fits inflation. u\* **converges to an estimated equilibrium** rather than wandering: the driftless random walk it replaced sat about 8 standard deviations from its own fitted path and could not accommodate the 1990s decline. Currently u\* = 4.83 against an equilibrium of 4.86. **Its headline is conditional** on two imposed choices, the convergence shape and `sigma_ustar` = 0.020, and about 97% of u\*'s total fall is the mechanism rather than the data. **Superseded for u\* by the joint y\*/u\* model**: taking the gap as data, this model cannot notice when that gap is too narrow, and pays for the mismatch with `beta_okun` = 2.15, which falls to 1.27 once the gap is estimated alongside. Kept as the component model and for its own diagnostics. Its value is legibility about what an Australian NAIRU rests on, not a better number. See [`MODEL_NOTES.md`](src/models/ustar/MODEL_NOTES.md)

- **Joint y\*/u\***: `ystar` and `ustar` in one likelihood, plus a free component `v` on the output gap. `v` cannot be estimated in `ystar` alone (it and the GDP residual are one additive term); adding the Okun equation makes the covariance between the two residuals identify it. The result is an output gap about twice `ystar`'s, sd 0.421 against 0.188. **This is the preferred source for the output gap and for u\***, because it is the only model in the package where the two can disagree and be reconciled: `ustar`'s `beta_okun` of 2.15 falls to 1.27 once the gap is estimated rather than imported. Feeds `rstar`'s Taylor rule. See [`MODEL_NOTES.md`](src/models/ystar_ustar/MODEL_NOTES.md)
- **Long-run u\***: Not an estimate. Every other u\* here starts in 1993, because its gap is defined against an inflation target that did not exist earlier. This one finds the stretches where inflation *actually* stopped changing, reads the unemployment rate off them, and reports what it saw, so it reaches back to **1959Q3**. Two categories, kept apart: five **plateaus** where inflation held level, reading 1.82 in the late 1960s and 5.45-5.68 since 2002; and six **U-bases** where it sat at the bottom of a wide trough, which skew high because a trough usually arrives at the end of a disinflation and must not be quoted as u\* without that qualification. On the one plateau where it overlaps the `nairu` model, the two agree to two hundredths: 6.20 against 6.22. A state-space version over the same period was built and deleted; the notes record why. See [`MODEL_NOTES.md`](src/models/long_run_ustar/MODEL_NOTES.md)
- **r\* (natural rate from the bond market)**: One latent state — an Australia-specific wedge over a market world real rate, moving as a Student-t random walk — read off two windows on the same curve, the indexed real 10-year yield and the real cash rate. The anchor is the Cleveland Fed's 10-year expected real rate, **not** Holston-Laubach-Williams, which is identified through the IS curve and moved −0.03 between the COVID-QE era and the tightening era while every market measure moved two to three points. No IS curve here either, because three separate efforts in this repo found the rate-to-output-gap link too weak to identify anything on Australian data. Carries a level Taylor rule and observed credit wedges for firms and for households. **r\* is 1.08 with the Australian wedge at +0.05, so Australia currently sits on the world rate. The level is not identified and moved 0.83-1.22 across four defensible specifications: quote the wedge and the era pattern, not the level** — see [`MODEL_NOTES.md`](src/models/rstar_bonds/MODEL_NOTES.md)
- **HLW r\***: Bayesian (PyMC) Holston-Laubach-Williams model estimating the natural rate of interest for Australia — see [`MODEL_NOTES.md`](src/models/rstar_hlw/MODEL_NOTES.md)
- **Neutral (from the RBA's reaction to inflation)** (`rstar_rba`): assumes a neutral cash rate that moves **slowly**, with the RBA reacting **fast** on top of it to inflation above or below the 2.5% target, and splits the observed cash rate into those two pieces. The response is to the deviation from 2.5, not to being outside the band: the band half-width only sets `lambda`'s units. **Neutral is the slow piece `b_t`**; `b_t + lambda x g_t` is the rule's prescribed rate, not neutral. Two published series and nothing else: the cash rate and trimmed-mean inflation. No IS curve, no world anchor, no bond market. **This is not an estimate of neutral: it is an estimate of what the RBA's behaviour reveals about it, and it reveals it imperfectly** — it conflates belief with every other systematic motive, and a departure from the rule is absorbed into neutral the longer it lasts. That absorption is now measured rather than asserted: 0.78 of a known stance survives at one year, 0.44 at four, 0.25 at ten, so it can audit the Bank over **one to two years and no further**. Neutral is 2.99 nominal, 0.49 real, but the level is conditional on an **arbitrary** smoothness `sigma_r` and spans −0.05 to 1.05 across defensible values, which is wider than the credible interval: quote the range. The durable result is `lambda`, 0.61 per percentage point, which is a **nominal** response and not comparable with Taylor's 1.5; it also carries the labour-market response, and the residual is autocorrelated at 0.85 so every interval is too tight. Its sharpest finding is that a Taylor rule on its own neutral says 2016-2019 was about a point too tight, in every quarter and at every `sigma_r` — see [`MODEL_NOTES.md`](src/models/rstar_rba/MODEL_NOTES.md)
- **The IS curve, plotted rather than estimated** (`is_curve`): a test bench, not a model. Puts the output gap against the real rate under four r\* treatments and fits a line. Nothing is estimated and nothing downstream consumes it. The slope's sign depends on which quarters are included, the strongest relationship is contemporaneous and positive, which is the policy reaction function rather than transmission, and a sample cut can manufacture a convincing IS curve out of two clusters. **The IS-curve problem in Australian data remains unresolved** — see [`MODEL_NOTES.md`](src/models/is_curve/MODEL_NOTES.md)
- **Bank funding and lending costs** (`bank_costs`): exploratory, charts only, no model and no notes yet. Pulls RBA bank bill, deposit and lending rates and plots them against the cash rate to see where pass-through goes
- **DSGE** — **experimental, work in progress; none usable yet.** A family of forward-looking DSGE models (New Keynesian; financial-accelerator `FA-NK` with two natural rates and an endogenous external-finance-premium wedge; sticky-wage `FA-NK-wage` with Galí unemployment; and a reduced-form `NK-TwoStar` probe) built to explore the post-GFC "great divergence". They are research and diagnostic builds, not production tools. A Bayesian re-estimation (`fa_nk_bayes.py`, PyMC/DEMetropolis-Z) now **identifies the policy block** — the cash-rate rule responds aggressively to inflation (φ_π≈2.6), and both FA-NK models converge cleanly — but the r\* and NAIRU/U\* these models produce remain **not credible**, and the φ_π result is not yet robustness-tested. For credible r\* and NAIRU use the HLW r\* and NAIRU models above. See [`MODELS_EXPLAINED.md`](src/models/dsge/MODELS_EXPLAINED.md).

**Run order:** the NAIRU model and the HLW r\* model both read the expectations model's saved output (`output/expectations/`) as an input — **run the expectations model first** whenever updating after new data. The u\* model sits one step further down: it reads both the expectations output and a completed `ystar` run. The r\* model sits below that again, and now reads all three Taylor-rule inputs from a completed **joint y\*/u\*** run so they share one potential output, one u\* and one `c` (`--input-source separate` restores the older wiring). The full chain is `expectations` → `ystar` → `ustar`, with `expectations` → `ystar_ustar` → `rstar_bonds` alongside it. `rstar_rba` is independent of all of it and needs only the cash rate and trimmed-mean inflation, though its Taylor-rule chart reads core inflation and the output gap from a completed `ystar_ustar` run. `is_curve` runs last, reading completed `ystar_ustar`, `rstar_bonds` and `rstar_rba` runs. The NAIRU model's operational r\* is deterministic: a fixed 35/65 convex blend of the Cobb-Douglas growth anchor (r\* ≈ potential growth) and the real bond-yield anchor. The 35/65 weight is imposed, not estimated — the rate channel cannot identify it, and the yield lean fixes the perverse Cobb-Douglas r\* profile (high in the 2010s, low now). NAIRU and output-gap estimates are near-insulated from the choice (<0.08pp); it mainly shapes the r\* level and the monetary-stance narrative.

### GDP nowcasting

Four complementary approaches to nowcasting the next unpublished quarterly GDP growth:

- **Bridge equations**: High-frequency monthly indicators completed to quarters via SARIMA, combined by inverse-MSE weights — see [`MODEL_NOTES.md`](src/models/gdp_nowcast_bridge/MODEL_NOTES.md)
- **Dynamic Factor Model**: Common factors extracted from a mixed-frequency panel via Kalman filter, ragged edge handled natively — see [`MODEL_NOTES.md`](src/models/gdp_nowcast_dfm/MODEL_NOTES.md)
- **Bayesian VAR**: Minnesota-prior VAR conditioned on contemporaneous indicators, T-0 only *(comparator, not for operational point forecasts)* — see [`MODEL_NOTES.md`](src/models/gdp_nowcast_bvar/MODEL_NOTES.md)
- **Components (expenditure identity)**: Structural accounting build-up summing component contributions to growth, T-0 only — see [`MODEL_NOTES.md`](src/models/gdp_nowcast_components/MODEL_NOTES.md)

## Quickstart

```bash
# Install dependencies
uv sync

# Run the NAIRU + Output Gap model (~3 min)
./run-nairu.sh -v
```

ABS and RBA data are fetched and cached automatically, no credentials needed. The only
exception is `src/data/fred_loader.py`, used for the world real-rate comparators in the r\*
notes: it reads a FRED API key from `fred.api` in the project root, one line, gitignored.
A free key comes from [FRED](https://fred.stlouisfed.org/docs/api/api_key.html). Nothing in
the default model runs depends on it.

## Running the Models

### NAIRU + Output Gap (Bayesian)

```bash
# Default run: the policy-relevant NAIRU — simple_excess_rstar_blend variant
# (excess-expectations term, with the fixed 35/65 growth/yield r* blend),
# expectations folding to the 2.5% target over 1993–1998
./run-nairu.sh -v

# Re-run validate/analyse/forecast from the saved trace (no re-estimation)
./run-nairu.sh -v --skip-estimate

# Estimation only (skip validate/analyse/forecast)
./run-nairu.sh -v --estimate-only

# Other variants: simple (core equations), simple_excess, simple_regime,
# simple_excess_regime, the Student-t and trimmed-price robustness variants
# (simple_excess_studentt, simple_excess_regime_studentt, simple_excess_tprice,
# simple_excess_regime_tprice), simple_excess_nohcoe, the free-alpha r* probe
# (simple_excess_rstar_est), and complex / complex_excess (all features);
# other anchors via --anchor
./run-nairu.sh -v --variant simple_excess
./run-nairu.sh -v --variant complex --anchor unanchored

# Multiple variants in one run also produces comparison charts
./run-nairu.sh -v --variant simple_excess simple_excess_regime

# Or via Python directly
uv run python -m src.models.nairu.run -v
```

### Inflation Expectations (Bayesian)

```bash
# Run all four expectation models (~10 min)
./run-expectations.sh

# Or via Python directly
uv run python -m src.models.expectations.model

# Run single model (target, unanchored, short, or market)
uv run python -m src.models.expectations.stage1 --model target

# Generate diagnostics and plots only
uv run python -m src.models.expectations.stage2
```

Four models are estimated:
- **Target Anchored**: Full model with 2.5% anchor post-1998
- **Unanchored**: Same as Target but no anchor — tests de-anchoring
- **Short Run (1yr)**: Market economists 1-year ahead only
- **Long Run (10yr)**: Breakeven inflation only

The NAIRU model uses a **spliced series**: Long Run through 1991Q4 (smooth disinflation decline), then Target Anchored from 1992Q1 onwards.

### Cobb-Douglas Productivity Decomposition

```bash
# Deterministic growth accounting (~30 sec)
./run-cd.sh -v

# Or via Python
uv run python -m src.models.cobb_douglas.model -v
```

**Superseded by `ystar` and the joint y\*/u\* model for potential output and the output gap.** The potential path this model produces is notional: its level is set by re-anchoring to actual GDP at four dates and is not disciplined by inflation, so it cannot say whether the economy is running hot or cold. What it still uniquely provides is the growth accounting itself, the decomposition of growth into capital, labour and MFP, which neither Bayesian model attempts. Use it for that and nothing else. See the [`MODEL_NOTES.md`](src/models/cobb_douglas/MODEL_NOTES.md).

### y\* — Potential Output (Bayesian)

```bash
# Default: spec inflation, sample from 1993Q1, 2.5% anchor
./run-ystar.sh -v

# Recharts from the saved trace / estimate without charting
./run-ystar.sh --analyse-only
./run-ystar.sh --no-analyse

# Alternative specification: potential growth from a Cobb-Douglas production
# function (trend capital, hours and MFP) rather than a random-walk drift
./run-ystar.sh --spec production

# The variance settings are imposed, not estimated — sweep them, and the anchor
uv run python -m src.models.ystar.sigma_sweep --param ratio_g
uv run python -m src.models.ystar.sigma_sweep --param ratio_ystar
uv run python -m src.models.ystar.sigma_sweep --param anchor

# Endpoint fragility: the estimate a real-time user would have had
uv run python -m src.models.ystar.realtime
```

Read the two outputs differently. **Trend growth (~2.1%, down roughly two points since the
late 1990s)** is robust across every perturbation tried, though the *width* of its band is
only as narrow as the smoothing prior is tight. The **sign of the gap** is the sign of the
inflation deviation and is solid statistically, but it reads the ledger correctly only when
that deviation is demand-driven — a supply shock inverts it. The **magnitude** of the gap is
the weak half. See the [`MODEL_NOTES.md`](src/models/ystar/MODEL_NOTES.md).

### u\* — NAIRU from a given output gap (Bayesian)

Reads saved output from the expectations and `ystar` models, so run those first.

```bash
# Default: u* converges, sigma_ustar 0.020, sample from 1993Q1
./run-ustar.sh --verbose

# Recharts from the saved trace
./run-ustar.sh --analyse-only

# The setting the answer hinges on — sweep it, don't trust it
./run-ustar.sh --sigma-ustar 0.030

# Specification alternatives, kept so the comparison is reproducible
./run-ustar.sh --no-ustar-converge   # the driftless random walk it replaced
./run-ustar.sh --ustar-drift         # drift on excess inflation expectations instead

# Diagnostics
./run-ustar.sh --free-sigma-ustar    # why the drift cannot be estimated
./run-ustar.sh --no-output-gap       # does the given gap actually matter?
./run-ustar.sh --no-phillips         # Okun only
```

Read the result as conditional, and more so than the number suggests. u\* = **4.83** against an
estimated equilibrium of 4.86, with the gap at −0.48. But the chart
`what-moves-u-the-specification-or-the-data` shows that of u\*'s 6.1pp fall across the sample,
about **97% is the convergence mechanism and 3% is the data** — the 1990s are essentially the
specification drawing a curve. `sigma_ustar` = 0.020 governs how far u\* may wander from that
curve, and it too is imposed: a free prior escapes to 0.131 and a bounded one pins to its
bound. Quote "u\* near 4.8, labour market tight by roughly half a point" and no decimals. What
*is* identified by the data and stable across specifications: the supply-chain coefficient
(0.037 to 0.043, against the NAIRU model's 0.047) and the Phillips residual. See the
[`MODEL_NOTES.md`](src/models/ustar/MODEL_NOTES.md).

### Joint y\* / u\* (Bayesian)

Reads saved output from the expectations model only; it re-estimates both `ystar` and `ustar`
rather than reading them.

```bash
# Default: quarterly gap basis, u* converging, sigma_ustar 0.020, 10,000 draws
./run-ystar-ustar.sh

# Recharts from the saved trace
./run-ystar-ustar.sh --analyse-only

# The controls the result rests on
./run-ystar-ustar.sh --no-okun       # sigma_v should return its prior — it does
./run-ystar-ustar.sh --no-phillips   # sigma_v with inflation off the left-hand side
./run-ystar-ustar.sh --sigma-v-prior 0.5   # is sigma_v prior-driven? no
```

The point of joining is one parameter, `sigma_v`, the scale of a free component on the output
gap. Inside `ystar` it cannot be estimated at all — it and the GDP residual are two mean-zero
terms in one equation — but the gap also enters the Okun equation, so the covariance between
the GDP and unemployment residuals identifies it. Currently `sigma_v` = 0.334 with 91% prior
shrinkage, giving an output gap with sd 0.421 against `ystar`'s 0.188.

Two caveats worth carrying. The **share** of that gap attributable to the free component is not a
quotable number: it runs from 0.1% to 54.5% across the `sigma_okun` sweep and from 39.6% to 80.2%
across the u\* state-law 2x2, while sd(gap) stays well above `ystar`'s throughout. And `sigma_v`
was substantially larger before u\* was given a convergence mechanism, which means part of what
the model attributed to hidden cycle was a mis-specified u\* trend. See the
[`MODEL_NOTES.md`](src/models/ystar_ustar/MODEL_NOTES.md).

### Long-run u\* — read off flat inflation, back to 1959

No likelihood and no priors: a rule, and the sensitivity of the answer to it.

```bash
./run-long-run-ustar.sh
./run-long-run-ustar.sh --require-flat-u      # both series settled, not just inflation
./run-long-run-ustar.sh --tolerance 1.5       # looser: picks up the 1990s false positive
./run-long-run-ustar.sh --window 12           # a longer stretch must be flat
./run-long-run-ustar.sh --smooth 1            # no smoothing: finds nothing before 2002
```

Readings by decade: **1.82** in the 1960s, 6.20 in the 1980s, 5.57 in the 2000s, 5.50 in the
2010s. No single number is printed across decades, and the code declines to compute one:
averaging 1.8 and 5.5 would assert the constancy the exercise exists to test. The **direction
contrast** is the model's actual test, and it shows the Phillips ordering holding cleanly in the
1960s and 2010s and breaking in the 1970s and 2020s, the two supply-shock decades. Read the
trough table knowing it is cyclically contaminated by construction. See the
[`MODEL_NOTES.md`](src/models/long_run_ustar/MODEL_NOTES.md).

### r\* — the natural rate from the bond market (Bayesian)

Reads the Taylor rule's inputs (output gap, unemployment gap, supply decomposition) from a
completed **joint y\*/u\*** run by default, so they share one potential output, one u\* and one
`c`; `--input-source separate` restores the older wiring off `ystar` and `ustar`. r\* itself needs
none of them, and the affected charts are skipped with a note if they are missing.

```bash
./run-rstar-bonds.sh -v

# The setting the answer leans on — sweep it
./run-rstar-bonds.sh --sigma-walk 0.03

# Specifications tried as the default and rejected, kept as comparators
./run-rstar-bonds.sh --curve           # a third window: r* starts absorbing the policy stance
./run-rstar-bonds.sh --short-rate bill # the 90-day bill: mixes bank credit into the stance

# Diagnostics, kept so the checks are reproducible
./run-rstar-bonds.sh --no-short        # one window: 168 divergences, for the record
./run-rstar-bonds.sh --assert-stance   # assert the stance, report the premium (a tighter mu_tp)
./run-rstar-bonds.sh --steps           # the asserted-break comparator
./run-rstar-bonds.sh --no-world        # does the global anchor do the work? (it does)
./run-rstar-bonds.sh --no-look-through # respond to headline inflation instead
```

**r\* is 0.83 now, against world r\* of 0.95**, with a nominal neutral cash rate of 3.33
against an actual 4.35 — so policy is around 1.0 restrictive, and 0.74 below what a Taylor
rule wants given inflation at 3.6 and a positive output gap. Across a seven-fold sweep of
`sigma_walk` the level holds at 0.81-1.13 and the Taylor prescription stays above the actual
rate throughout, so both are results rather than settings. **Two things are not.** How far
r\* fell in 2021 moves from −0.29 to −1.22 across the same sweep, so the rise since is a
statement about the setting. And the pre-COVID policy stance runs −0.67 to −0.16, which
matters because that is the quantity the standing critique of this model turns on. Quote the
level and that today is about a third of the pre-GFC rate; not the rise, and not the stance
without its range.

Two specifications were tried as the default and rejected on the evidence, both documented
in the notes: a third window on the belly of the curve, which identifies the premium curve's
slope but drives r\* over 2016-19 from −0.29 to −0.74 as the natural rate starts absorbing
the policy stance; and the 90-day bank bill as the short rate, which reads better until you
decompose it against OIS and find that its entire 2015-19 advantage is a bank funding spread.
Note also that the negative QE-era term premium reported in earlier write-ups **does not
survive the second window** — see the [`MODEL_NOTES.md`](src/models/rstar_bonds/MODEL_NOTES.md).

### HLW r\* (Bayesian)

Resolution G (blend + hierarchical Beta) is the default and the standard specification to run. It is the end point of a sequence of specifications (A–H) built while diagnosing why canonical HLW fails to identify r\* on Australian data; the earlier resolutions are retained as diagnostic comparators — see the model notes for the full journey.

```bash
# Default: Resolution G (blend + hierarchical Beta)
./run-rstar-hlw.sh -v

# Alternative resolutions
./run-rstar-hlw.sh --resolution C    # blend with fixed Beta(1,1) on alpha
./run-rstar-hlw.sh --resolution A    # canonical HLW: r* = g + z

# Estimate only / re-analyse saved trace
./run-rstar-hlw.sh --estimate-only
./run-rstar-hlw.sh --skip-estimate

# Or via Python
uv run python -m src.models.rstar_hlw.run -v
```

The IS curve does not independently pin r\* in Australian data (the rate channel is too weak); each specification largely returns the structural assumption it imposes. The value is a diagnostic framework and an honest cross-resolution uncertainty band rather than a single point estimate — see the [`MODEL_NOTES.md`](src/models/rstar_hlw/MODEL_NOTES.md).

### r* from the RBA's reaction to inflation

```bash
./run-rstar-rba.sh -v                     # the default: walking base, Gaussian, floor kept
./run-rstar-rba.sh --jumps                # permit base steps at abrupt GDP quarters
#   the default also runs the sigma_r ensemble and the injection test (~38s):
./run-rstar-rba.sh --no-sigma-r-ensemble --no-injection-test   # the estimate alone (~8s)
./run-rstar-rba.sh --lambda-split 2008Q1  # did the response halve after the GFC?
./run-rstar-rba.sh --employment           # add lambda_u (u - u*); rejected, see the notes
./run-rstar-rba.sh --sigma-r 0.05         # the imposed setting that decides the split
```

Charts land in `charts/RStarRBA/`. **Read `lambda` in the right units**: it is per band-width
of the inflation gap, so the response per percentage point is twice it. Do not put it beside
Taylor's 1.5: in nominal terms the two are not the same object, because `lambda` is the
inflation pass-through and the real response summed — see the
[`MODEL_NOTES.md`](src/models/rstar_rba/MODEL_NOTES.md).

### The IS curve, plotted rather than estimated

```bash
uv run python -m src.models.is_curve.run                     # lockdowns excluded (default)
uv run python -m src.models.is_curve.run --drop-gfc-pandemic # manufactures a convincing IS curve
uv run python -m src.models.is_curve.run --lag 5             # where the cut sample peaks
```

Charts land in `charts/ISCurve/`, one per r\* variant, points coloured by date because the
two-cluster structure behind the sign reversal is visible before any statistic shows it —
see the [`MODEL_NOTES.md`](src/models/is_curve/MODEL_NOTES.md).

### Bank funding and lending costs

Exploratory. `./run-bank-costs.sh` writes charts to `charts/BankCosts/`. No model, no notes.

### GDP Nowcasting

**Timing:** don't run the nowcasts until about one month before the GDP release. Earlier in the cycle almost no indicators for the target quarter are published — the BVAR declines to nowcast at all, and the bridge/DFM intervals are mostly prior.

```bash
# Bridge equations (high-frequency monthly indicators)
./run-gdp-nowcast-bridge.sh

# Dynamic Factor Model
./run-gdp-nowcast-dfm.sh

# Bayesian VAR (T-0 only, comparator)
./run-gdp-nowcast-bvar.sh

# Components / expenditure identity (T-0 only)
./run-gdp-nowcast-components.sh

# Bridge model backtest
uv run python -m src.models.gdp_nowcast_bridge.backtest
```

The bridge and DFM are the workhorses for production nowcasts. The BVAR is a comparator (structurally over-volatile, not recommended for operational point forecasts). The components model is a structural accounting build-up complementary to the statistical nowcasts — at T-0 it measures rather than forecasts the most volatile contributors (inventories, net exports, government).

### Outputs

| Directory | Contents |
|-----------|----------|
| `model_outputs/` | NAIRU saved traces (`.nc`) and observations (`.pkl`) for multiple model variants; `ystar_*` traces, observations and the real-time run (`.pkl`); `ustar_*` and `rstar_*` traces and observations; `rstar_hlw_*` traces (resolutions A–H); GDP nowcast outputs (`gdp_nowcast*/`) |
| `output/expectations/` | Expectations traces (`.nc`), HDI estimates (`.parquet`, `.csv`), metadata (`.pkl`) |
| `charts/nairu_*/` | NAIRU, output gap, Phillips curves, equations, decompositions (per variant) |
| `charts/expectations/` | Expectations comparison, diagnostics, model fits |
| `charts/cobb_douglas/` | MFP trends, productivity growth, potential output |
| `charts/YStar*/` | y\* potential output, output gap, trend growth, sweeps (one dir per spec) |
| `charts/UStar/` | u\* against unemployment (plain and RBA-band shaded), the unemployment gap, price inflation decomposition |
| `charts/RStarBonds/` | Taylor rule with real and nominal r\*, r\* against the real yield, the Australian wedge, the policy stance and the policy gap, the term premium with and without the `k·g` correction, r\* for firms |
| `charts/rstar-hlw-*/` | r\* estimates, cross-resolution comparisons, diagnostics (one dir per resolution) |
| `charts/GDP-Nowcast-*/` | GDP nowcast fan charts and decompositions (Bridge, DFM, BVAR, Components) |

### Maintenance

```bash
# Upgrade all dependencies
./uv-upgrade.sh
```

## Project Structure

```
src/
├── data/               # Data fetching (ABS, RBA) and preparation
├── utilities/          # Shared utilities (rate conversion, etc.)
└── models/
    ├── common/                 # Shared model utilities (diagnostics, extraction, timeseries, sources)
    ├── expectations/           # Inflation expectations signal extraction
    ├── nairu/                  # NAIRU + Output Gap model (estimate → validate → analyse → forecast)
    │   └── analysis/           # Plotting and diagnostics modules
    ├── cobb_douglas/           # Cobb-Douglas MFP decomposition
    ├── ystar/                  # y* potential output — inflation-defined output gap
    │                           #   (self-contained: imports only src/data)
    ├── ustar/                  # u* from a given output gap — Okun + Phillips, one state
    │                           #   (reads expectations and ystar output)
    ├── ystar_ustar/            # y* and u* estimated jointly, gap partly free
    │                           #   (preferred for the output gap and u*)
    ├── long_run_ustar/         # u* read off flat-inflation stretches, back to 1959Q3
    │                           #   (a rule, not an estimate: no likelihood, no priors)
    ├── rstar_bonds/            # r* from the bond market — AU wedge over world r*, two windows, no IS curve
    ├── rstar_hlw/              # HLW Bayesian r* model (AU data)
    ├── rstar_rba/              # neutral revealed by the RBA's reaction to inflation — a slow
    │                           #   neutral b_t plus a fast response on top (the Bank's implied
    │                           #   belief; a departure from the rule is absorbed into neutral
    │                           #   the longer it lasts, 0.44 of it surviving at four years)
    ├── is_curve/               # the IS curve plotted, not estimated — a test bench for the r* models
    ├── bank_costs/             # bank funding and lending costs vs the cash rate (exploratory, charts only)
    ├── gdp_nowcast_bridge/     # GDP nowcast — bridge equations
    ├── gdp_nowcast_dfm/        # GDP nowcast — Dynamic Factor Model
    ├── gdp_nowcast_bvar/       # GDP nowcast — Bayesian VAR (T-0 only)
    ├── gdp_nowcast_components/ # GDP nowcast — expenditure-identity components (T-0 only)
    └── dsge/                   # DSGE models — experimental, work in progress, NOT usable yet (see MODELS_EXPLAINED.md)
```
