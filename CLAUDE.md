# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## PRIORITY: Flag assumptions vs. verified facts

Whenever a statement rests on an **assumption, inference, guess, or memory** rather than something just **fact-checked against the code, data, a file, or a source the user provided**, say so explicitly and mark it. The user relies on this distinction; presenting an assumption as fact causes false trust and is a serious failure.

- **Never invent specifics**: dates, numbers, file paths, line numbers, release schedules, API behaviour, etc. If a value isn't verified, do not fill it in. State that it is unknown.
- **Label clearly.** Prefix unverified claims with `ASSUMPTION:` (or `GUESS:` / `UNVERIFIED:`), e.g. "ASSUMPTION: GDP releases ~4 June: I have not verified this date." Keep verified facts unmarked.
- **Prefer verifying.** If something can be checked (read the file, run the query, grep the code), check it rather than assume. Only fall back to a labelled assumption when verification isn't possible, and say what would confirm it.
- **When in doubt, surface it.** It is always better to flag an uncertainty than to let the user discover it was a guess.

## Project Overview

MacroModels is an Australian macroeconomic modeling project that estimates NAIRU (Non-Accelerating Inflation Rate of Unemployment), potential output, and output gaps using Bayesian state-space models. The models use PyMC for estimation and draw on Australian Bureau of Statistics (ABS) data.

## Commands

```bash
# Environment management (uses uv)
./uv-upgrade.sh                    # Upgrade dependencies
uv sync                            # Install dependencies

# Run models
./run-nairu.sh                     # Run NAIRU + output gap model
./run-expectations.sh              # Run expectations model
./run-cd.sh                        # Run Cobb-Douglas model
./run-gdp-nowcast-bridge.sh        # Run GDP nowcast (bridge equations)
./run-gdp-nowcast-dfm.sh           # Run GDP nowcast (Dynamic Factor Model)
./run-gdp-nowcast-bvar.sh          # Run GDP nowcast (Bayesian VAR, T-0 only)
./run-gdp-nowcast-components.sh    # Run GDP nowcast (expenditure-identity components, T-0 only)
./run-rstar-hlw.sh                 # Run HLW: trend/cycle decomposition. NOT a source of r*
./run-ystar.sh                     # Run y* potential output model (inflation-defined output gap)
./run-ustar.sh                     # Run u* model (ONE Phillips curve, u* a spline; Okun is OFF)
./run-rstar-bonds.sh               # Run r* from the bond market (needs ystar_ustar for the Taylor rule)
./run-rstar-rba.sh                 # Run neutral revealed by the RBA's reaction to inflation (two series;
                                   #   also runs the sigma_r ensemble and the injection test, ~38s)
./run-rstar-invert.sh              # r* by conditional inversion of an ASSERTED IS curve
                                   #   (--ensemble sweeps how slow r* is; --lag-sweep the rate lag)
./run-rstar-tvpvar.sh              # RETIRED. TVP-VAR (Lubik-Matthes); still runs, but r* comes
                                   #   back as the real cash rate (see MODEL_NOTES)
./run-rstar-summary.sh             # every r* model on one nominal scale; re-runs any whose
                                   #   saved trace is not from today, which regenerates THEIR charts
./run-ustar-summary.sh             # three specifications of the u* model on one chart
./run-gstar-summary.sh             # every g* (potential growth) estimate on one chart; refresh
                                   #   is OFF by default (--refresh would overwrite ystar's
                                   #   production spec with the inflation spec)
./run-bank-costs.sh                # Bank funding and lending costs vs the cash rate (charts only)
uv run python -m src.models.is_curve.run   # IS-curve scatter: a test bench for the r* models
uv run python -m src.models.common.diagnostics_report  # MCMC diagnostics for EVERY saved trace,
                                   #   PRINTED, not written: nothing is re-sampled and no file is
                                   #   produced. --only <str> or --dir narrows it.
                                   # The only diagnostics FILE is per run: each model's analysis
                                   #   writes run-diagnostics-<prefix>.txt into that run's chart
                                   #   directory, beside the charts it describes. Any other run's
                                   #   file there is cleared, so one directory = one run's charts
                                   #   plus its diagnostics.
./run-ystar-ustar.sh               # Run joint y*/u* model (gap partly free, u* a spline;
                                   #   needs expectations)
./run-ystar-summary.sh             # five specifications of the y* model on one chart
./run-ystar-ustar-summary.sh       # the u* structure crossed with the gap definition, eight
uv run python -m src.models.dsge.fa_nk_model         # Run financial-accelerator DSGE (two r* + EFP wedge)
uv run python -m src.models.dsge.fa_nk_wage_model    # Run FA-NK + sticky wages + Galí unemployment
uv run python -m src.models.dsge.nk_twostar_model    # Run NK two-star linear probe
uv run python -m src.models.dsge.fa_nk_bayes         # Bayesian re-estimation (Taylor-block priors); --smoke for quick check, --extract-only for posterior r*/EFP bands
uv run python -m src.models.gdp_nowcast_bridge.backtest  # Run nowcast backtest
```

## Project Structure

```
src/
├── data/                          # Data loading and transformation modules
│   ├── abs_loader.py              # ABS data retrieval via readabs
│   ├── rba_loader.py              # RBA data retrieval
│   ├── henderson.py               # Henderson moving average
│   ├── transforms.py              # Data transformations
│   ├── series_specs.py            # Series specification definitions
│   ├── dataseries.py              # Data series utilities
│   ├── long_cpi.py                 # Headline CPI back to 1948, rebuilt from the quarterly change
│   ├── retail_trade.py             # Monthly household spending (5682.0)
│   ├── building_approvals.py       # Monthly dwelling approvals (8731.0)
│   ├── goods_trade.py              # Monthly goods trade balance (5368.0)
│   ├── business_indicators.py      # Quarterly profits, inventories, wages, sales (5676.0)
│   └── ...                        # Individual data series modules (inflation, gdp, etc.)
│
├── models/
│   ├── nairu/                     # NAIRU + output gap model. SUPERSEDED THROUGHOUT: potential,
│   │                              #   gap and NAIRU. Potential is not estimated, the posterior
│   │                              #   median reproduces the Cobb-Douglas input (1.66 vs 1.74),
│   │                              #   and that input tracks the cycle rather than trend, going
│   │                              #   negative in 2020 and swinging 4.8 -> 0.15 across 1990-92.
│   │                              #   The gap is therefore actual minus filtered actual, and
│   │                              #   Okun carries it into the NAIRU. Use ystar for potential
│   │                              #   growth and ystar_ustar for the gap and u*. Still the only
│   │                              #   model with a wage equation, the anchor transition, the
│   │                              #   regime split and LOO/WAIC variant comparison, which is
│   │                              #   what it is for (see MODEL_NOTES.md).
│   ├── gdp_nowcast_bridge/        # GDP nowcasting via bridge equations (see MODEL_NOTES.md)
│   ├── gdp_nowcast_dfm/            # GDP nowcasting via Dynamic Factor Model (see MODEL_NOTES.md)
│   ├── gdp_nowcast_bvar/           # GDP nowcasting via Bayesian VAR, T-0 only (see MODEL_NOTES.md)
│   ├── gdp_nowcast_components/     # GDP nowcasting via expenditure-identity components, T-0 only (see MODEL_NOTES.md)
│   ├── rstar_hlw/                 # NOT A SOURCE OF r*, and cannot be: z has no observation
│   │                              #   equation, so r* is trend growth (corr 0.998) and sigma_z
│   │                              #   only picks which answer to report. Excluded from
│   │                              #   rstar_summary; use rstar_bonds or rstar_rba instead.
│   │                              #   THE DECOMPOSITION IS A SEPARATE CLAIM AND IT WORKS:
│   │                              #   repaired 2026-09-12 (sigma_ystar imposed at 0.078,
│   │                              #   lockdowns excluded) and the output gap now matches Okun
│   │                              #   at 1993-95, 2008-09 and 2026Q2. Do not read that as a
│   │                              #   rehabilitated r*. Default: Resolution A, start 1993Q1,
│   │                              #   rate lag t-6, lambda_g off (see MODEL_NOTES.md).
│   ├── ystar_ustar/               # ** PREFERRED for the output gap and u*. ** y* and u*
│   │                              #   estimated JOINTLY, gap = c x (pi - 2.5) + v, so the gap is
│   │                              #   not frozen and the GDP and Okun equations negotiate over
│   │                              #   it. Estimates sigma_v, which neither parent can identify.
│   │                              #   Ranks above both because it resolves their inconsistency:
│   │                              #   the gap is ~2x ystar's and ustar's beta_okun falls once
│   │                              #   fed the whole gap.
│   │                              #   u* IS A SPLINE, one knot at 2013Q1, not the decay it used
│   │                              #   to be. The decay could only draw a monotone approach, so
│   │                              #   from 10.77 it could only ever report a fall and its
│   │                              #   endpoint was a fitted scalar; 5.96 of its 6.03 point
│   │                              #   decline was the zero-innovation curve. The spline turns:
│   │                              #   +0.38 over 2015-2026 against -0.38, and sigma_ustar is
│   │                              #   gone. Two and three knots return the decay's answer.
│   │                              #   CONDITIONAL on sigma_okun (imposed 0.20, now the ONLY
│   │                              #   imposed variance carrying the answer): sigma_v is flat
│   │                              #   over 0.10-0.20 and the model degenerates into ystar by
│   │                              #   0.70, but the free posterior puts no mass above 0.40.
│   │                              #   --gap-spec identity makes the gap y - y*, which samples
│   │                              #   far better (ESS 6145 vs 1333) and halves the 1993-99
│   │                              #   residual bias, but gives a 1992 gap of -8.7; needs
│   │                              #   --one-sided-beta or a mirror mode opens.
│   │                              #   --okun-form ec DOES NOT IDENTIFY: u - u* and the gap are
│   │                              #   94% collinear, so 1516 divergences even reparameterised.
│   │                              #   Feeds rstar's Taylor rule (see MODEL_NOTES.md).
│   ├── ystar_summary/            # NOT A MODEL. The five ystar specifications on one chart.
│   │                              #   ITS MAIN FINDING IS AGAINST ITS OWN PACKAGE: three of the
│   │                              #   five reproduce an HP(1600) filter of GDP at corr 1.0000,
│   │                              #   and inflation contributes 5-21% of the deviation from
│   │                              #   potential. Runs from 1984Q1 with a phased anchor, and the
│   │                              #   three walk-based runs behind it carry a c collapse, so
│   │                              #   their numbers should not be quoted (see MODEL_NOTES.md).
│   ├── ystar_ustar_summary/      # NOT A MODEL. Eight settings of the joint model: the u*
│   │                              #   structure (decay, 1/2/3 knots) crossed with the gap
│   │                              #   definition. Scores leave-one-out on the two equations all
│   │                              #   eight observe, since the identity gap's GDP equation
│   │                              #   carries no likelihood (see MODEL_NOTES.md).
│   ├── ystar/                     # y* potential output: potential is a slow-moving random walk,
│   │                              #   the gap is DEFINED as c x (pi - 2.5). No Phillips curve, no
│   │                              #   IS curve, no policy rule (see MODEL_NOTES.md).
│   │                              #   Self-contained: imports only src/data, no other model.
│   │                              #   Still the preferred source for POTENTIAL GROWTH: the joint
│   │                              #   model agrees (1.99 vs 1.94) and this is the simpler
│   │                              #   statement of the same answer. Superseded for the GAP.
│   ├── ustar/                     # u* from ONE expectations-augmented Phillips curve, with u*
│   │                              #   a natural cubic spline, one knot at 2013Q1. Sample 1993Q1.
│   │                              #   THE OKUN EQUATION IS OFF. ystar's defined gap IS
│   │                              #   0.1882 x (pi - 2.5) exactly (R2 = 1.0000), so
│   │                              #   u = u* - beta x ygap is a Phillips curve in levels and the
│   │                              #   two equations read ONE signal. Dropping it widens the mean
│   │                              #   90% band 0.35 -> 0.58 (0.36 -> 1.15 over 1993-98), removes
│   │                              #   a -0.13/-0.23 bias against what inflation alone implies,
│   │                              #   and takes 1993-98 from 8.69 to 7.14. --okun restores it.
│   │                              #   THE SPLINE replaces a decay law that could only draw a
│   │                              #   monotone approach and so declined forever; --state
│   │                              #   converge restores it, and only there do sigma_ustar,
│   │                              #   phi_ustar and ustar_eq exist.
│   │                              #   DO NOT QUOTE ANYTHING BEFORE 2000: the disinflation is in
│   │                              #   1991-92, outside the sample, so the model opens on a calm
│   │                              #   nominal picture beside 10.9% unemployment. Charts shade
│   │                              #   1993Q1-1999Q4 (see MODEL_NOTES.md).
│   ├── rstar_bonds/               # r* from the bond market: one state, an AU wedge over
│   │                              #   a market world real rate, moving as a StudentT random
│   │                              #   walk, read off THREE windows on one curve: the indexed
│   │                              #   real 10y yield, the real cash rate, and the AOFM 5y5y
│   │                              #   risk-neutral forward (deflated). Anchor is the
│   │                              #   Cleveland Fed 10y expected real rate LESS the published
│   │                              #   US term premium, NOT HLW, which is inert across the
│   │                              #   whole monetary cycle. The term premium is PINNED to the
│   │                              #   AOFM's published Australian series; only the real-nominal
│   │                              #   spread is estimated. b_world is IMPOSED at 1 (free, it
│   │                              #   collapses to 0.015 once the premium is data, which is
│   │                              #   non-identification not a finding). nu_walk IMPOSED at 9;
│   │                              #   the third window will not sample without it.
│   │                              #   NO IS CURVE, three efforts here found the
│   │                              #   rate/output-gap link unidentifiable on AU data. Level
│   │                              #   Taylor rule on top. r* 0.80 real / 3.33 nominal with
│   │                              #   the wedge at -0.20, so Australia sits BELOW the world
│   │                              #   rate. QUOTE THE LAST COMPLETE QUARTER: the bond block is
│   │                              #   daily, so the model also estimates the quarter in
│   │                              #   progress off a part-month average with inflation and the
│   │                              #   gaps missing, and that ran 0.25pp higher (1.05 / 3.57,
│   │                              #   wedge -0.29) on no extra uncertainty. Charts stop at the
│   │                              #   last finished quarter; the trace does not.
│   │                              #   THE LEVEL IS NOW PARTLY IDENTIFIED and is worth
│   │                              #   quoting: the 5y5y window took the 90% band from 2.59 to
│   │                              #   1.28 and it no longer contains zero. Costs: the wedge is
│   │                              #   4x jumpier quarter to quarter (some of that is market
│   │                              #   noise booked as r*), amplitude worsens 3.66 -> 4.12, and
│   │                              #   forward_bias is uninterpreted. The forward also ABOLISHES
│   │                              #   negative r*: 2016-19 and 2020-21 go from -0.11/-0.70 to
│   │                              #   +0.42/+0.14. --no-forward restores the two-window model.
│   │                              #   A DIFFERENT third window (--curve) and the 90-day bank
│   │                              #   bill (--short-rate bill) were both tried as defaults and
│   │                              #   rejected; the QE term-premium finding does not survive
│   │                              #   the second window (see MODEL_NOTES.md).
│   ├── cobb_douglas/              # Cobb-Douglas MFP decomposition. NOT COVID-ROBUST: its
│   │                              #   three HP filters run through the pandemic, leaving a
│   │                              #   COVID-shaped wobble of a few tenths in g* from 2020 on.
│   │                              #   Excluding a window was tried several ways and abandoned
│   │                              #   (it changes the wobble's sign, not its existence), so
│   │                              #   gstar_summary excludes this model and its post-2019
│   │                              #   potential growth should not be quoted. The growth
│   │                              #   ACCOUNTING is unaffected. Also SUPERSEDED by ystar and
│   │                              #   ystar_ustar for potential output and the output gap: its
│   │                              #   potential path is re-anchored to actual GDP at four dates
│   │                              #   and is not disciplined by inflation. Use it only for the
│   │                              #   growth accounting (capital / labour / MFP), which neither
│   │                              #   Bayesian model attempts.
│   ├── dsge/                      # DSGE + HLW-style models (see MODELS_EXPLAINED.md)
│   │                              #   fa_nk_model.py: financial-accelerator DSGE, two r* + endogenous EFP wedge (labour_block flag)
│   │                              #   fa_nk_wage_model.py: FA-NK + sticky wages + Galí unemployment / U*
│   │                              #   nk_twostar_model.py: NK + reduced-form wedge (linear probe)
│   │                              #   fa_nk_bayes.py: Bayesian re-estimation (black-box Op + priors, DEMetropolis-Z); identifies the Taylor block (φ_π≈2.6)
│   ├── rstar_rba/                 # Neutral revealed by the RBA's reaction function. Assumes a
│   │                              #   neutral cash rate that moves SLOWLY, with the RBA reacting
│   │                              #   responding on top of it to inflation away from the 2.5 TARGET
│   │                              #   (not to being outside the band: g_t is linear in
│   │                              #   pi - 2.5, and the band half-width only sets lambda's
│   │                              #   units), and splits the cash rate into those two pieces.
│   │                              #   NEUTRAL IS b_t, stored as `neutral`. b_t + lambda.g_t is
│   │                              #   the rule's PRESCRIBED rate, stored as `prescribed`, and is
│   │                              #   not neutral. `stance` = cash less neutral,
│   │                              #   `rule_residual` = cash less prescribed. Say which one a
│   │                              #   number is: 3.89 vs 4.25 nominal at 2026Q2.
│   │                              #   THE LEVEL IS PINNED BY A MARKET PRICE since 2026-09-16:
│   │                              #   the AOFM 5y5y forward (deflated) is a SECOND observation
│   │                              #   window, f_t = b_t + bias + e_t. Before it, the level
│   │                              #   rested on the sample-average cash rate and ran -0.05 to
│   │                              #   1.05 real across defensible sigma_r, wider than the
│   │                              #   credible interval; now the spread across sigma_r is 0.31
│   │                              #   and the rule residual falls +0.87 -> +0.10. Real neutral
│   │                              #   1.35. forward_bias -0.109 [-0.289, +0.068], i.e. the
│   │                              #   market's 5y5y IS the model's neutral, which is a result
│   │                              #   rather than an assumption. sigma_r = 0.125, chosen on
│   │                              #   SAMPLING grounds (0.15 failed diagnostics).
│   │                              #   lambda = 0.46 per pp is a NOMINAL response; not comparable
│   │                              #   with Taylor's 1.5. UNITS: stored per BAND-WIDTH (0.228),
│   │                              #   and the band half-width is 0.5, so per pp is twice the
│   │                              #   stored value. It is stable across sigma_r only
│   │                              #   CONDITIONAL ON ZERO POLICY SMOOTHING: allow partial
│   │                              #   adjustment and it runs to 2.57, because one coefficient
│   │                              #   carries both the immediate and the ultimate response.
│   │                              #   sigma_r and phi decide the same thing and two series
│   │                              #   cannot pin both. Two published series only
│   │                              #   (see MODEL_NOTES.md for everything else).
│   ├── rstar_invert/              # r* by CONDITIONAL INVERSION of an asserted IS curve.
│   │                              #   Asserts the line (negative slope, through the origin on
│   │                              #   gap-vs-gap axes) and a slow r*, takes the ystar_ustar gap
│   │                              #   and the real cash rate as GIVEN, and reports the r* path
│   │                              #   those assertions force. NOT AN ESTIMATE.
│   │                              #   THE ANSWER IS DECIDED BY sigma_rstar, which nothing
│   │                              #   measures: below 0.05-0.10 the model explains nothing and
│   │                              #   r* is flat, above it r* swings 5pp and the 2016-19 stance
│   │                              #   flips sign (-3.19 to +0.52). sigma_e falls monotonically
│   │                              #   as r* is loosened, so the data cannot choose.
│   │                              #   A DEFENSIBLE SLOPE AND A USABLE r* ARE INCOMPATIBLE:
│   │                              #   -0.09 (matching is_curve's -0.108) gives r* of -3.9 to
│   │                              #   +7.0; a well-behaved r* needs -0.38, which survives only
│   │                              #   because the r*-prior parameterisation rewards inflating it.
│   │                              #   What it measures well is the GAP's own slow component
│   │                              #   (44% of gap variance vs the rate term's 14%), divided by a
│   │                              #   small number. Quote the conditioning (see MODEL_NOTES.md).
│   │                              #   REMOVED FROM rstar_summary 2026-09-16: its line restated
│   │                              #   the asserted IS curve rather than adding a third view.
│   ├── rstar_tvpvar/              # RETIRED. TVP-VAR after Lubik-Matthes: three variables,
│   │                              #   drifting coefficients, r* = the 20-quarter projection.
│   │                              #   It reads neutral off the economy's own dynamics, which
│   │                              #   needs the economy to SETTLE. Australia's does not: no
│   │                              #   stationary stretch exists in 1993-2026, so the fitted VAR
│   │                              #   sits at a spectral radius of 0.983 and r* comes back as
│   │                              #   the real cash rate (corr 0.95). Not fixable by estimand,
│   │                              #   sample, shrinkage or sampling; all four were tried.
│   │                              #   THE FINDING GENERALISES and is why the notes are kept:
│   │                              #   it sinks any model that MEASURES equilibrium from
│   │                              #   behaviour, not those that ASSERT a structure defining it
│   │                              #   (rstar_rba's rule, rstar_bonds' market price).
│   │                              #   Do not quote a level (see MODEL_NOTES.md).
│   ├── rstar_summary/             # NOT A MODEL. Loads every r* the repo produces, re-runs any
│   │                              #   whose trace is not from TODAY (which regenerates that
│   │                              #   model's own charts), converts all to NOMINAL on
│   │                              #   long-run expectations, and charts them. TWO lines since
│   │                              #   2026-09-17: rstar_bonds and rstar_rba.
│   │                              #   rstar_hlw is EXCLUDED: its z state has no
│   │                              #   observation equation, so its r* is trend growth.
│   │                              #   rstar_invert was REMOVED 2026-09-16: it asserts an IS
│   │                              #   curve no method here can recover the sign of, so its
│   │                              #   line restated an assumption rather than adding a view.
│   │                              #   rstar_tvpvar was REMOVED 2026-09-17, not discredited but
│   │                              #   not ready to carry a level: at a median spectral radius
│   │                              #   of 0.983 the 20q projection is 71% nowcast, a third of
│   │                              #   draw-quarters are explosive, and the steady state
│   │                              #   divides by almost nothing. Its sample-MEAN level and
│   │                              #   2016-19 sign survive its sigma_q sweep; its LATEST value
│   │                              #   (the only thing the chart plots) runs 1.08 to 3.17.
│   │                              #   NOTE both remaining models share an observable
│   │                              #   (the AOFM 5y5y forward), so some of their agreement is
│   │                              #   one series counted twice, and that now applies to the
│   │                              #   WHOLE chart. Models end on DIFFERENT
│   │                              #   quarters (bonds runs a quarter longer); never average
│   │                              #   across them. The central line is a MEAN, not a median,
│   │                              #   and at n=2 it is just the band's midpoint; it is not an
│   │                              #   estimate (see MODEL_NOTES.md).
│   ├── ustar_summary/            # NOT A MODEL. Three SPECIFICATIONS of ustar on one chart, not
│   │                              #   three models: knot count (1 or 2) crossed with whether
│   │                              #   Okun is in. They share a sample, a Phillips curve and an
│   │                              #   expectations series, so agreement is close to arithmetic
│   │                              #   and only disagreement informs.
│   │                              #   ALL THREE ARE SPLINES, so any of them can turn u* UP at
│   │                              #   the endpoint if the data warrant it. The decay settings
│   │                              #   are absent for that reason: the sign of phi x (eq - u*)
│   │                              #   is fixed by which side of eq the state opened on, so
│   │                              #   from 10.75 they can only ever report a fall, and their
│   │                              #   -0.32/-0.34 post-2015 is the shape rather than the data.
│   │                              #   ONE CARRIES OKUN, which ustar's default excludes. It is
│   │                              #   the only one in which u* comes DOWN through the 1990s, a
│   │                              #   regime change from high to low inflation working slowly
│   │                              #   through the labour market: unemployment fell 4.53pp over
│   │                              #   1993-99 and it falls 3.43, against 0.50 and 0.81. Not
│   │                              #   about the 1995 episode, two shallow quarters that should
│   │                              #   drag nothing. It also carries the steepest post-2015
│   │                              #   decline of anything tried, -0.59; its 1990s credentials
│   │                              #   lend that no weight.
│   │                              #   SPREAD 2.87pp at 1993Q1 to 0.03pp now, latest u*
│   │                              #   4.67-4.70. Not an error band: the gap is the two
│   │                              #   readings of the 1990s. Knot count alone is worth 0.05pp.
│   │                              #   The mean is a description. Each spec writes to its own
│   │                              #   ustar_sum_* prefix (see MODEL_NOTES.md).
│   ├── gstar_summary/            # NOT A MODEL. Potential growth on one chart: ystar
│   │                              #   (inflation + production specs) and the joint model.
│   │                              #   They agree to 0.09pp (1.90-1.99 at 2026Q2) against
│   │                              #   ~1.0pp for r*, which is the point of the package. BUT
│   │                              #   all three share the y* core, so nothing here would catch
│   │                              #   a smoothing assumption common to them. cobb_douglas was
│   │                              #   the outside check and is EXCLUDED for COVID artefacts.
│   │                              #   Refresh is OFF by default (--refresh would overwrite
│   │                              #   ystar's production spec). rstar_hlw also excluded
│   │                              #   (see MODEL_NOTES.md).
│   ├── is_curve/                  # THE IS CURVE PLOTTED, NOT ESTIMATED. A test bench, not a
│   │                              #   model: nothing estimated, nothing downstream consumes it.
│   │                              #   Output gap against the real rate under four r* treatments.
│   │                              #   The slope's sign depends on the sample; the strongest
│   │                              #   relationship is contemporaneous and POSITIVE, which is the
│   │                              #   policy reaction function rather than transmission; and
│   │                              #   dropping 2008Q4-2021Q3 manufactures a convincing IS curve
│   │                              #   out of two clusters that individually disagree.
│   │                              #   Default lag 6, matching rstar_hlw and rstar_invert. At
│   │                              #   that lag all four variants are indistinguishable from
│   │                              #   zero, and the block split is the sharpest form of the
│   │                              #   finding: 1993-2020 +0.099, 2021Q4-2026Q2 -0.136, so the
│   │                              #   whole negative reading sits in 19 quarters.
│   │                              #   THE IS-CURVE PROBLEM IN AU DATA REMAINS UNRESOLVED
│   │                              #   (see MODEL_NOTES.md).
│   ├── bank_costs/                # Bank funding and lending costs against the cash rate.
│   │                              #   EXPLORATORY: charts only, no model, no MODEL_NOTES.
│   ├── expectations/              # Inflation expectations model
│   └── common/                    # Shared model utilities (diagnostics, extraction, timeseries, sources)
│
└── utilities/                     # General utilities (rate_conversion)

charts/                            # Generated chart output
input_data/                        # Input data files
model_outputs/                     # Model output files
```

## Key Dependencies

- **readabs**: Custom library for fetching ABS data (also `mgplot` for plotting, `sdmxabs` for SDMX)
- **PyMC/ArviZ**: Bayesian modeling and diagnostics
- **JAX/NumPyro**: Backend for PyMC sampling
- **pandas/numpy/statsmodels**: Data manipulation and econometrics

## Architecture

### readabs Library (~/readabs)

The `readabs` library provides ABS and RBA data access. Source at `~/readabs/`.

**Key functions:**
- `read_abs_cat(cat, single_excel_only=table, verbose=False)`: Main loader. Returns `(dict[str, DataFrame], DataFrame)` where dict keys are table names, metadata DataFrame has `metacol` columns. Always specify `single_excel_only` to avoid downloading every table in the catalogue.
- `read_abs_by_desc(wanted, cat=, table=, stype=, single_excel_only=)`: Search by data item description. Returns `(dict[str, Series], DataFrame)`. Preferred over hardcoded series IDs which break when ABS changes identifiers.
- `find_abs_id(meta, search_terms, validate_unique=True)`: Find series ID from metadata search. Returns `(table, series_id, units)`. Used by `abs_loader.py:load_series()`.
- `search_abs_meta(meta, search_terms)`: Search metadata DataFrame, returns matching rows.

**Metadata columns (`metacol` frozen dataclass):**
- `mc.did`: Data Item Description (search key for finding series)
- `mc.stype`: Series Type ("Original", "Seasonally Adjusted", "Trend")
- `mc.id`: Series ID (e.g. "A84423050A")
- `mc.table`: Table name (e.g. "6202001")
- `mc.unit`: Unit of measure
- `mc.cat`: Catalogue number

**Best practices for data loaders:**
- Always specify `single_excel_only=table` to target a specific table
- Search by description (`mc.did`) not hardcoded series IDs: ABS changes IDs
- Use `abs_loader.py:load_series(ReqsTuple)` or `ra.read_abs_by_desc()` patterns
- Results are cached via `@cache` decorator

### Data Pipeline
1. ABS data fetched via `readabs` library with local caching (`.readabs_cache/`)
2. Data modules in `src/data/` provide standardized retrieval and transformation
3. `henderson.py` implements Henderson moving average for trend smoothing

### Model Structure (NAIRU+Output Gap)
The main model jointly estimates:
- **NAIRU**: Random walk state-space model
- **Potential Output**: Cobb-Douglas production function (capital + labor + MFP)
- **Phillips Curve**: Links unemployment gap to inflation
- **Okun's Law**: Links output gap to unemployment changes
- **Wage Equation**: Unit labor cost growth
- **IS Equation**: Output gap persistence with interest rate effects

Key parameters:
- α (alpha): Capital share of income (~0.25-0.30)
- Inflation anchor transitions from expectations (pre-1993) to target (2.5%, post-1998)
- Deterministic r*: a fixed 35/65 blend of the Cobb-Douglas growth anchor and the real
  bond-yield anchor (`config.DEFAULT_RSTAR_ALPHA`), applied globally in `observations.py`

### Code Style
- Ruff configured with aggressive linting (`line-length=119`, most rules enabled)
- Specific ignores for Jupyter patterns (useless expressions, module-level imports)
- Uses `.loc[]` over `.at[]` per mypy preferences

### Git
- Never use git commands - no commits, no status, nothing
- User manages all version control manually

### Interaction
- Never provide clickable suggested next steps (user hits them accidentally)
- Text suggestions in responses are fine

### Plotting
- Only create multi-panel plots when specifically asked for them
- Default to separate charts for each series

## mgplot Package Reference

The `mgplot` package (source in `~/mgplot`) wraps matplotlib for economic data charting.
**Prefer `*_finalise` functions** for simple single-layer charts.
For composite charts (e.g. fan charts, overlaid fills + lines), layer mgplot functions
with `ax=` chaining, then call `finalise_plot()` to close out. Avoid raw matplotlib
(`ax.plot()`, `ax.fill_between()`, etc.) when an mgplot function exists.

### Architecture
```
# Simple charts: use *_finalise (one-step convenience)
line_plot_finalise(data, **kwargs)
  └─ plot_then_finalise()
       ├─ line_plot(data, **plot_kwargs)    → returns Axes
       └─ finalise_plot(axes, **fp_kwargs)  → styles, saves, closes

# Composite charts: layer mgplot functions, then finalise
ax = fill_between_plot(band_data, color="red", alpha=0.1, label="90% CI")
line_plot(history, ax=ax, color=["navy"], width=2)
finalise_plot(ax, title="...", ylabel="...", show=False)

# finalise_plot() does NOT support plot-level kwargs like annotate, width, color.
```

### Chart Directory Management
```python
import mgplot as mg
mg.set_chart_dir("./charts/MyCharts/")
mg.clear_chart_dir()
```

### All *_finalise Functions
Each plots data AND saves to file. Pass combined plot + finalise kwargs in one call.

```python
mg.line_plot_finalise(df, ...)           # Line charts
mg.bar_plot_finalise(df, ...)            # Bar charts (grouped or stacked)
mg.growth_plot_finalise(growth_df, ...)  # QoQ bars + TTY line
mg.series_growth_plot_finalise(s, ...)   # Calculates growth from index, then plots
mg.fill_between_plot_finalise(df, ...)   # Shaded area between two columns
mg.postcovid_plot_finalise(s, ...)       # Line with post-COVID projection
mg.revision_plot_finalise(df, ...)       # ABS data revisions
mg.run_plot_finalise(s, ...)             # Highlights runs in a series
mg.seastrend_plot_finalise(df, ...)      # Seasonal + trend overlay
mg.summary_plot_finalise(df, ...)        # Z-score summary (creates 2 plots)
```

### Line Plot Parameters (LineKwargs)
```python
mg.line_plot_finalise(
    data,                # Series or DataFrame
    width=2,             # Line width (float, int, or list per series). NOT lw.
    color=["blue"],      # Colors (str or list per series)
    style="-",           # Line style (str or list)
    alpha=1.0,           # Opacity (float or list)
    marker=None,         # Marker style
    markersize=None,     # Marker size
    drawstyle=None,      # e.g. "steps-post"
    annotate=True,       # Add endpoint value labels
    rounding=1,          # Decimal places for annotations
    fontsize="small",    # Annotation font size
    annotate_color=None, # Annotation color (str, bool, or list)
    plot_from=None,      # Start index (int offset or Period)
    label_series=None,   # Label lines directly instead of legend
    dropna=True,         # Drop NaN values
    # ... plus all Finalise kwargs below
)
```

### Finalise Parameters (FinaliseKwargs)
These work on ALL `*_finalise` functions:
```python
# Titles and labels
title="Chart Title",       # Also used for filename
suptitle="Super Title",    # Above the title
ylabel="Per cent",
xlabel="Year",

# Footers and headers (annotations outside plot area)
rfooter="Source: ABS",     # Right footer
lfooter="Australia. ",     # Left footer
rheader="",                # Right header
lheader="",                # Left header

# Axis limits and ticks
xlim=(0, 100),
ylim=(0, 100),
xticks=[...],
yticks=[...],

# Legend: True, False, None, or dict with any matplotlib legend kwargs
legend=True,
legend={"loc": "upper left", "fontsize": "small", "title": "Quantiles", "ncol": 2},

# Reference lines and bands (single dict or list of dicts)
axhline={"y": 2.5, "color": "red", "linestyle": "--"},
axvline={"x": pd.Period("2020-03"), "color": "grey"},
axhspan={"ymin": 2, "ymax": 3, "color": "lightgreen"},
axvspan={"xmin": ..., "xmax": ...},

# Display and save
y0=True,           # Horizontal line at y=0 if data crosses zero
show=False,        # Display in notebook
tag="mytag",       # Filename becomes: title-mytag.png
pre_tag="prefix",  # Filename becomes: prefix-title.png
file_type="png",   # Output format
dpi=300,           # Resolution
figsize=(8, 6),    # Figure size
dont_save=False,   # Skip saving
dont_close=False,  # Keep figure open
```

### Bar Plot Specific (BarKwargs)
```python
mg.bar_plot_finalise(
    df,
    stacked=False,         # True = stacked, False = grouped side by side
    annotate=True,         # Value labels on bars
    width=0.8,             # Bar width (0-1)
    above=True,            # Annotations above bars
    label_rotation=0,      # X-axis label rotation
    color=["blue", "red"],
)
```

### Multi-Plot Functions
```python
# Same chart at multiple starting points
mg.multi_start(df, function=mg.line_plot_finalise, starts=[0, -20], title="Chart")

# One chart per column
mg.multi_column(df, function=mg.line_plot_finalise, title="Chart")

# Chain any plot function + finalise (used internally by *_finalise)
mg.plot_then_finalise(data, function=mg.line_plot, title="Chart")
```

### Utility Functions
```python
mg.calc_growth(series)           # Returns DataFrame with QoQ and TTY columns
mg.get_color("NSW")              # State color
mg.abbreviate_state("Victoria")  # → "Vic."
mg.contrast("blue")              # Contrasting color for text
```
