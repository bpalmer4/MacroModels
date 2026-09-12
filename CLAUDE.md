# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## PRIORITY: Flag assumptions vs. verified facts

Whenever a statement rests on an **assumption, inference, guess, or memory** rather than something just **fact-checked against the code, data, a file, or a source the user provided**, say so explicitly and mark it. The user relies on this distinction; presenting an assumption as fact causes false trust and is a serious failure.

- **Never invent specifics** — dates, numbers, file paths, line numbers, release schedules, API behaviour, etc. If a value isn't verified, do not fill it in. State that it is unknown.
- **Label clearly.** Prefix unverified claims with `ASSUMPTION:` (or `GUESS:` / `UNVERIFIED:`), e.g. "ASSUMPTION: GDP releases ~4 June — I have not verified this date." Keep verified facts unmarked.
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
./run-ustar.sh                     # Run u* model (Okun + Phillips; needs expectations + ystar)
./run-rstar-bonds.sh               # Run r* from the bond market (needs ystar_ustar for the Taylor rule)
./run-rstar-rba.sh                 # Run neutral revealed by the RBA's reaction to inflation (two series;
                                   #   also runs the sigma_r ensemble and the injection test, ~38s)
./run-rstar-invert.sh              # r* by conditional inversion of an ASSERTED IS curve
                                   #   (--ensemble sweeps how slow r* is; --lag-sweep the rate lag)
./run-rstar-summary.sh             # every r* model on one nominal scale; re-runs any whose
                                   #   saved trace is not from today, which regenerates THEIR charts
./run-gstar-summary.sh             # every g* (potential growth) estimate on one chart; refresh
                                   #   is OFF by default (--refresh would overwrite ystar's
                                   #   production spec with the inflation spec)
./run-bank-costs.sh                # Bank funding and lending costs vs the cash rate (charts only)
uv run python -m src.models.is_curve.run   # IS-curve scatter: a test bench for the r* models
./run-ystar-ustar.sh               # Run joint y*/u* model (gap partly free; needs expectations)
./run-long-run-ustar.sh            # Read u* off flat-inflation stretches, back to 1959 (no estimation)
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
│   │                              #   the gap is ~2x ystar's (sd 0.421 vs 0.188) and ustar's
│   │                              #   beta_okun falls 2.14 -> 1.27 once fed the whole gap.
│   │                              #   CONDITIONAL on sigma_okun (imposed 0.20, the one imposed
│   │                              #   variance in the package with no external anchor): sigma_v
│   │                              #   is flat over 0.10-0.20, and the model degenerates into
│   │                              #   ystar by 0.70, but the free posterior puts no mass above
│   │                              #   0.40. Feeds rstar's Taylor rule (see MODEL_NOTES.md).
│   ├── ystar/                     # y* potential output: potential is a slow-moving random walk,
│   │                              #   the gap is DEFINED as c x (pi - 2.5). No Phillips curve, no
│   │                              #   IS curve, no policy rule (see MODEL_NOTES.md).
│   │                              #   Self-contained: imports only src/data, no other model.
│   │                              #   Still the preferred source for POTENTIAL GROWTH: the joint
│   │                              #   model agrees (1.99 vs 1.94) and this is the simpler
│   │                              #   statement of the same answer. Superseded for the GAP.
│   ├── ustar/                     # u* from a GIVEN output gap: one state (u*), two observation
│   │                              #   equations (Okun, expectations-augmented Phillips). Reads
│   │                              #   expectations + ystar output; estimates neither.
│   │                              #   SUPERSEDED by ystar_ustar for u*: taking the gap as data,
│   │                              #   it cannot notice that the gap is too narrow and pays for
│   │                              #   the mismatch with beta_okun = 2.14. Kept as the component
│   │                              #   model and for its own diagnostics (the wage check, the
│   │                              #   sigma_ustar sweep). HEADLINE IS CONDITIONAL: u*'s level is
│   │                              #   set by the imposed sigma_ustar (see MODEL_NOTES.md).
│   ├── rstar_bonds/               # r* from the bond market: one state, an AU wedge over
│   │                              #   a market world real rate, moving as a StudentT random
│   │                              #   walk, read off TWO windows on one curve: the indexed
│   │                              #   real 10y yield and the real cash rate. Anchor is the
│   │                              #   Cleveland Fed 10y expected real rate (FRED), NOT HLW,
│   │                              #   which is inert across the whole monetary cycle. The
│   │                              #   loading is estimated: b_world 0.481, which is NOT
│   │                              #   credible as a pass-through and is partly stripping a US
│   │                              #   term premium. NO IS CURVE — three efforts here found the
│   │                              #   rate/output-gap link unidentifiable on AU data. Level
│   │                              #   Taylor rule on top. r* 1.08 with the wedge at +0.05,
│   │                              #   so Australia currently sits ON the world rate. The
│   │                              #   LEVEL is not identified (wedge_0 vs mu_tp at -0.87) and
│   │                              #   moved 0.83-1.22 across four defensible specs; the PATH
│   │                              #   and the pre-COVID STANCE are not robust either. Quote
│   │                              #   the wedge and the era pattern, not the level.
│   │                              #   A third window (--curve) and the 90-day bank bill
│   │                              #   (--short-rate bill) were both tried as defaults and
│   │                              #   rejected; the QE term-premium finding does not survive
│   │                              #   the second window (see MODEL_NOTES.md).
│   ├── long_run_ustar/            # u* WITHOUT estimation, back to 1959Q3. Finds the stretches
│   │                              #   where inflation actually stopped changing and reads
│   │                              #   unemployment off them. Reaches where the state-space
│   │                              #   models cannot: 1.82 in the late 1960s, 5.45-5.68 since
│   │                              #   2002, both robust across the rule. Its main finding is
│   │                              #   about the others: under a loose rule it reads 9-10.9 for
│   │                              #   the early 1990s, the same as ystar_ustar, so that number
│   │                              #   is the NAIRU concept failing in a re-anchoring rather
│   │                              #   than a defect in the joint model (see MODEL_NOTES.md).
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
│   │                              #   number is: 2.99 vs 3.48 nominal at 2026Q2.
│   │                              #   The LEVEL is conditional on an arbitrary sigma_r: real
│   │                              #   neutral 0.49, but -0.05 to 1.05 across defensible values,
│   │                              #   wider than the credible interval. Quote the range.
│   │                              #   lambda = 0.61 per pp is a NOMINAL response; not comparable
│   │                              #   with Taylor's 1.5. UNITS: stored per BAND-WIDTH (0.305),
│   │                              #   so per pp is twice it. It is stable across sigma_r only
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
│   ├── rstar_summary/             # NOT A MODEL. Loads every r* the repo produces, re-runs any
│   │                              #   whose trace is not from TODAY (which regenerates that
│   │                              #   model's own charts), converts all to NOMINAL and charts
│   │                              #   them. rstar_hlw is EXCLUDED: its z state has no
│   │                              #   observation equation, so its r* is trend growth. The
│   │                              #   central line is a MEAN, not a median (n=3), and is not an
│   │                              #   estimate (see MODEL_NOTES.md).
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
- `read_abs_cat(cat, single_excel_only=table, verbose=False)` — Main loader. Returns `(dict[str, DataFrame], DataFrame)` where dict keys are table names, metadata DataFrame has `metacol` columns. Always specify `single_excel_only` to avoid downloading every table in the catalogue.
- `read_abs_by_desc(wanted, cat=, table=, stype=, single_excel_only=)` — Search by data item description. Returns `(dict[str, Series], DataFrame)`. Preferred over hardcoded series IDs which break when ABS changes identifiers.
- `find_abs_id(meta, search_terms, validate_unique=True)` — Find series ID from metadata search. Returns `(table, series_id, units)`. Used by `abs_loader.py:load_series()`.
- `search_abs_meta(meta, search_terms)` — Search metadata DataFrame, returns matching rows.

**Metadata columns (`metacol` frozen dataclass):**
- `mc.did` — Data Item Description (search key for finding series)
- `mc.stype` — Series Type ("Original", "Seasonally Adjusted", "Trend")
- `mc.id` — Series ID (e.g. "A84423050A")
- `mc.table` — Table name (e.g. "6202001")
- `mc.unit` — Unit of measure
- `mc.cat` — Catalogue number

**Best practices for data loaders:**
- Always specify `single_excel_only=table` to target a specific table
- Search by description (`mc.did`) not hardcoded series IDs — ABS changes IDs
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
