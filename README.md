# MacroModels

Australian macroeconomic modelling. Includes both Bayesian state-space estimation (PyMC) and deterministic growth accounting methods.

## Overview

### Supply side and structural estimation

- **NAIRU + Output Gap**: Bayesian state-space model jointly estimating the natural rate of unemployment and potential output — see [`MODEL_NOTES.md`](src/models/nairu/MODEL_NOTES.md)
- **Inflation Expectations**: Bayesian signal extraction model estimating latent expectations from surveys and market data — see [`MODEL_NOTES.md`](src/models/expectations/MODEL_NOTES.md)
- **Cobb-Douglas MFP**: Deterministic growth accounting decomposing output into capital, labour, and productivity
- **y\* (Potential Output)**: Bayesian unobserved-components model where potential is a slow-moving random walk and the output gap is *defined* by inflation's deviation from the 2.5% target — no Phillips curve, no IS curve, no policy rule. Self-contained (imports only `src/data`). The preferred source for the output gap and for trend growth (~2.1%) — see [`MODEL_NOTES.md`](src/models/ystar/MODEL_NOTES.md)
- **u\* (NAIRU from a given output gap)**: A deliberately small Bayesian model — one state, two observation equations — taking `ystar`'s output gap as an input rather than estimating it. Okun fits unemployment, an expectations-augmented Phillips curve fits inflation. **Its headline is conditional**: u\*'s level is set by an imposed drift rate that cannot be estimated, and moves from 5.10 to 4.62 across the defensible range. Its value is legibility about what an Australian NAIRU rests on, not a better number — use `nairu` for the operational estimate. See [`MODEL_NOTES.md`](src/models/ustar/MODEL_NOTES.md)
- **HLW r\***: Bayesian (PyMC) Holston-Laubach-Williams model estimating the natural rate of interest for Australia — see [`MODEL_NOTES.md`](src/models/rstar_hlw/MODEL_NOTES.md)
- **DSGE** — **experimental, work in progress; none usable yet.** A family of forward-looking DSGE models (New Keynesian; financial-accelerator `FA-NK` with two natural rates and an endogenous external-finance-premium wedge; sticky-wage `FA-NK-wage` with Galí unemployment; and a reduced-form `NK-TwoStar` probe) built to explore the post-GFC "great divergence". They are research and diagnostic builds, not production tools. A Bayesian re-estimation (`fa_nk_bayes.py`, PyMC/DEMetropolis-Z) now **identifies the policy block** — the cash-rate rule responds aggressively to inflation (φ_π≈2.6), and both FA-NK models converge cleanly — but the r\* and NAIRU/U\* these models produce remain **not credible**, and the φ_π result is not yet robustness-tested. For credible r\* and NAIRU use the HLW r\* and NAIRU models above. See [`MODELS_EXPLAINED.md`](src/models/dsge/MODELS_EXPLAINED.md).

**Run order:** the NAIRU model and the HLW r\* model both read the expectations model's saved output (`output/expectations/`) as an input — **run the expectations model first** whenever updating after new data. The u\* model sits one step further down: it reads both the expectations output and a completed `ystar` run, so the order there is `expectations` → `ystar` → `ustar`. The NAIRU model's operational r\* is deterministic: a fixed 35/65 convex blend of the Cobb-Douglas growth anchor (r\* ≈ potential growth) and the real bond-yield anchor. The 35/65 weight is imposed, not estimated — the rate channel cannot identify it, and the yield lean fixes the perverse Cobb-Douglas r\* profile (high in the 2010s, low now). NAIRU and output-gap estimates are near-insulated from the choice (<0.08pp); it mainly shapes the r\* level and the monetary-stance narrative.

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

Growth accounting is the solid product here. The potential output path it also produces is notional: its level is set by re-anchoring to actual GDP at four dates and is not disciplined by inflation, so prefer `ystar` for the output gap. See the [`MODEL_NOTES.md`](src/models/cobb_douglas/MODEL_NOTES.md).

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
# Default: sigma_ustar 0.040, sample from 1993Q1
./run-ustar.sh --verbose

# Recharts from the saved trace
./run-ustar.sh --analyse-only

# The setting the answer hinges on — sweep it, don't trust it
./run-ustar.sh --sigma-ustar 0.030

# Diagnostics, kept so the checks are reproducible
./run-ustar.sh --free-sigma-ustar    # why the drift cannot be estimated
./run-ustar.sh --no-output-gap       # does the given gap actually matter?
./run-ustar.sh --no-phillips         # Okun only
```

Read the result as conditional. **The level of u\* is set by `sigma_ustar`, the imposed drift
rate, which cannot be estimated** — a free prior escapes to 0.131, a bounded one pins to its
bound, and across the defensible range u\* runs 5.10 to 4.62 with the gap moving −0.75 to
−0.27. Quote "u\* near 4.8, labour market tight by roughly half a point" and no decimals. What
*is* identified by the data and stable across the sweep: the supply-chain coefficient (0.037 to
0.042, against the NAIRU model's 0.047) and the Phillips residual. See the
[`MODEL_NOTES.md`](src/models/ustar/MODEL_NOTES.md).

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
| `model_outputs/` | NAIRU saved traces (`.nc`) and observations (`.pkl`) for multiple model variants; `ystar_*` traces, observations and the real-time run (`.pkl`); `ustar_*` traces and observations; `rstar_hlw_*` traces (resolutions A–H); GDP nowcast outputs (`gdp_nowcast*/`) |
| `output/expectations/` | Expectations traces (`.nc`), HDI estimates (`.parquet`, `.csv`), metadata (`.pkl`) |
| `charts/nairu_*/` | NAIRU, output gap, Phillips curves, equations, decompositions (per variant) |
| `charts/expectations/` | Expectations comparison, diagnostics, model fits |
| `charts/cobb_douglas/` | MFP trends, productivity growth, potential output |
| `charts/YStar*/` | y\* potential output, output gap, trend growth, sweeps (one dir per spec) |
| `charts/UStar/` | u\* against unemployment (plain and RBA-band shaded), the unemployment gap, price inflation decomposition |
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
    ├── common/                 # Shared model utilities (diagnostics, extraction, timeseries)
    ├── expectations/           # Inflation expectations signal extraction
    ├── nairu/                  # NAIRU + Output Gap model (estimate → validate → analyse → forecast)
    │   └── analysis/           # Plotting and diagnostics modules
    ├── cobb_douglas/           # Cobb-Douglas MFP decomposition
    ├── ystar/                  # y* potential output — inflation-defined output gap
    │                           #   (self-contained: imports only src/data)
    ├── ustar/                  # u* from a given output gap — Okun + Phillips, one state
    │                           #   (reads expectations and ystar output)
    ├── rstar_hlw/              # HLW Bayesian r* model (AU data)
    ├── gdp_nowcast_bridge/     # GDP nowcast — bridge equations
    ├── gdp_nowcast_dfm/        # GDP nowcast — Dynamic Factor Model
    ├── gdp_nowcast_bvar/       # GDP nowcast — Bayesian VAR (T-0 only)
    ├── gdp_nowcast_components/ # GDP nowcast — expenditure-identity components (T-0 only)
    └── dsge/                   # DSGE models — experimental, work in progress, NOT usable yet (see MODELS_EXPLAINED.md)
```
