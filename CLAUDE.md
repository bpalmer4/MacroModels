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
./run-rstar-bonds.sh               # Run r* from the bond market (needs expectations; ystar_ustar for Taylor charts)
./run-rstar-rba.sh                 # Run neutral revealed by the RBA's reaction to inflation (two series;
                                   #   also runs the sigma_r ensemble and the injection test, ~38s)
./run-rstar-qpm.sh                 # Semi-structural open-economy model: trend r* and short-run
                                   #   neutral (--recovery re-estimates simulated economies)
uv run python -m src.models.rstar_hlw_kalman.run   # canonical HLW by Kalman filter + ML:
                                   #   a FAILED attempt at the original specification
./run-rstar-invert.sh              # r* by conditional inversion of an ASSERTED IS curve
                                   #   (--ensemble sweeps how slow r* is; --lag-sweep the rate lag)
./run-rstar-tvpvar.sh              # RETIRED. TVP-VAR (Lubik-Matthes); still runs, but r* comes
                                   #   back as the real cash rate (see MODEL_NOTES)
./run-rstar-summary.sh             # every r* model on one nominal scale; re-runs any whose
                                   #   saved trace is not from today, which regenerates THEIR charts
./run-ustar.sh --compare           # three specifications of the u* model on one chart
./run-gstar-summary.sh             # every g* (potential growth) estimate on one chart; refresh
                                   #   is OFF by default (--refresh would overwrite ystar's
                                   #   production spec with the inflation spec)
./run-bank-costs.sh                # Bank funding and lending costs vs the cash rate (charts only)
uv run python -m src.models.is_curve.run   # IS-curve scatter (retired: the search found no IS curve)
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
./run-ystar.sh --compare           # five specifications of the y* model on one chart
./run-ystar-ustar.sh --compare     # eight specifications (u* structure x gap definition)
uv run python -m src.models.dsge.fa_nk_model         # Run financial-accelerator DSGE (two r* + EFP wedge)
uv run python -m src.models.dsge.fa_nk_wage_model    # Run FA-NK + sticky wages + Galí unemployment
uv run python -m src.models.dsge.nk_twostar_model    # Run NK two-star linear probe
uv run python -m src.models.dsge.fa_nk_bayes         # Bayesian re-estimation (Taylor-block priors); --smoke for quick check, --extract-only for posterior r*/EFP bands
uv run python -m src.models.gdp_nowcast_bridge.backtest  # Run nowcast backtest
```

Don't run the GDP nowcasts until about a month before the release: earlier, almost no
indicators for the target quarter are out, the BVAR declines to nowcast and the bridge/DFM
intervals are mostly prior.

## Project Structure

```
src/
├── paths.py                       # ROOT, CHARTS, MODEL_OUTPUTS, INPUT_DATA, OUTPUT, CACHE.
│                                  #   Import these rather than counting `.parent` back
│                                  #   from __file__: the depth differs by module and a
│                                  #   file moved one level then reads the wrong place.
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
├── models/                        # Every model has a MODEL_NOTES.md: read it before quoting.
│   │   # LIVE
│   ├── expectations/              # Inflation expectations (target-anchored, unanchored, short, market)
│   ├── ystar/                     # Potential output; preferred source for POTENTIAL GROWTH; --compare
│   ├── ustar/                     # u* from one Phillips curve, spline; don't quote before 2000; --compare
│   ├── ystar_ustar/               # Joint y*/u*: PREFERRED for the output gap and u*; --compare
│   ├── cobb_douglas/              # Growth accounting only; not COVID-robust, don't quote post-2019 g*
│   ├── gstar_summary/             # NOT A MODEL: potential growth estimates on one chart
│   ├── rstar_bonds/               # r* from the bond market; quote the last complete quarter
│   ├── rstar_rba/                 # Neutral from the RBA's reaction function; neutral != prescribed
│   ├── rstar_qpm/                 # QPM-style semi-structural r*; wedge clipped by default; IS weak
│   ├── rstar_summary/             # NOT A MODEL: r* lines on one nominal scale; all share the 5y5y
│   ├── gdp_nowcast_bridge/        # GDP nowcast, bridge equations
│   ├── gdp_nowcast_dfm/           # GDP nowcast, dynamic factor model
│   ├── gdp_nowcast_bvar/          # GDP nowcast, Bayesian VAR (T-0 only)
│   ├── gdp_nowcast_components/    # GDP nowcast, expenditure components (T-0 only)
│   ├── bank_costs/                # Exploratory charts only
│   │   # RETIRED, SUPERSEDED OR NOT WORKING (kept for their notes; don't quote)
│   ├── nairu/                     # SUPERSEDED for potential, gap and NAIRU; still runs, kept for wages, LOO/WAIC
│   ├── rstar_hlw/                 # NOT a source of r*; still runs, its trend/cycle split does work
│   ├── rstar_hlw_kalman/          # Failed attempt at canonical HLW (Kalman + ML); degenerate
│   ├── rstar_tvpvar/              # RETIRED: r* comes back as the real cash rate
│   ├── rstar_invert/              # r* from an ASSERTED IS curve; not an estimate
│   ├── is_curve/                  # The IS curve plotted, not estimated; the search found none
│   ├── dsge/                      # Experimental DSGE family; not usable (see MODELS_EXPLAINED.md)
│   └── common/                    # Shared machinery, no economics (results, cli, diagnostics, charts)
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
- **Do not add `# noqa`.** Either agree a rule-level ignore in `pyproject.toml`,
  with a comment saying why, or fix the code. A per-line suppression hides a real
  diagnostic where nobody rereads it, and its justification goes stale unchecked:
  three in this repo asserted a circular import that did not exist. Directives
  already in the tree are legacy, to be removed as their files are worked on.
  Never run `ruff check --select <RULE> --fix`: narrowing `--select` deselects
  every other rule, so `RUF100` then judges nearly every existing directive
  unused and strips them all.
- **Economics notation is exempt, by rule not by suppression.** `N803` and `N806`
  are ignored so `T`, `R`, `Z`, `Q`, `H`, `P0` can keep the names of the
  state-space algebra they implement, and `U` can be unemployment.
- **All imports at the top of the file.** Not inside functions, not inside
  `if __name__` blocks. Where two modules need each other, move the shared
  names into a third module rather than deferring an import: `nairu/forecast`
  and `nairu/forecast_plots` both take `ForecastResults` from
  `nairu/forecast_types`, and `gdp_nowcast_components/model` and its
  `diagnostics` both take `CHART_DIR` from the package `__init__`.
- Magic numbers get a named constant with a comment, and any label quoting the
  value is built from the constant so the two cannot drift.

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
