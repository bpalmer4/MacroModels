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
./run-rstar-hlw.sh                 # Run HLW Bayesian r-star model
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
│   ├── retail_trade.py             # Monthly household spending (5682.0)
│   ├── building_approvals.py       # Monthly dwelling approvals (8731.0)
│   ├── goods_trade.py              # Monthly goods trade balance (5368.0)
│   ├── business_indicators.py      # Quarterly profits, inventories, wages, sales (5676.0)
│   └── ...                        # Individual data series modules (inflation, gdp, etc.)
│
├── models/
│   ├── nairu/                     # NAIRU + output gap model (see MODEL_NOTES.md)
│   ├── gdp_nowcast_bridge/        # GDP nowcasting via bridge equations (see MODEL_NOTES.md)
│   ├── gdp_nowcast_dfm/            # GDP nowcasting via Dynamic Factor Model (see MODEL_NOTES.md)
│   ├── gdp_nowcast_bvar/           # GDP nowcasting via Bayesian VAR, T-0 only (see MODEL_NOTES.md)
│   ├── gdp_nowcast_components/     # GDP nowcasting via expenditure-identity components, T-0 only (see MODEL_NOTES.md)
│   ├── rstar_hlw/                 # HLW Bayesian r-star model, AU data (see MODEL_NOTES.md)
│   ├── cobb_douglas/              # Cobb-Douglas MFP decomposition
│   ├── dsge/                      # DSGE / HLW-style models (in development)
│   ├── expectations/              # Inflation expectations model
│   └── common/                    # Shared model utilities (diagnostics, extraction, timeseries)
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
- Deterministic r* derived from Cobb-Douglas potential growth

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
