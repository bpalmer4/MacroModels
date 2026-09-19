# MacroModels

Developing economic models of the Australian macroeconomy: the NAIRU and u\*, potential output
and the output gap, the neutral rate r\*, and GDP nowcasts. Mostly Bayesian state-space
estimation in PyMC, with some deterministic growth accounting, on ABS and RBA data.

Each model has its own `MODEL_NOTES.md` recording what it assumes, what it is conditional on,
and what should not be quoted from it. The links below go there; this page is only a map.

## Overview

### Supply side and structural estimation

- **Inflation Expectations** (`expectations`): latent inflation expectations from surveys and market data. Run it first; much of the rest reads its output. See [notes](src/models/expectations/MODEL_NOTES.md)
- **y\* potential output** (`ystar`): potential as a slow-moving random walk, with the output gap defined by inflation's deviation from target. *Preferred for potential growth*. See [notes](src/models/ystar/MODEL_NOTES.md)
- **u\*** (`ustar`): a NAIRU from one expectations-augmented Phillips curve, with u\* a spline. See [notes](src/models/ustar/MODEL_NOTES.md)
- **u\* summary** (`ustar_summary`): the three specifications of `ustar` on one chart. See [notes](src/models/ustar_summary/MODEL_NOTES.md)
- **Joint y\*/u\*** (`ystar_ustar`): both of the above in one likelihood, with the gap partly free. *Preferred for the output gap and u\**. See [notes](src/models/ystar_ustar/MODEL_NOTES.md)
- **NAIRU + Output Gap** (`nairu`): the original joint NAIRU and potential-output model. *Superseded for potential, the gap and the NAIRU*; kept for its wage equation, regime split and variant comparison. See [notes](src/models/nairu/MODEL_NOTES.md)
- **Cobb-Douglas MFP** (`cobb_douglas`): deterministic growth accounting into capital, labour and MFP. See [notes](src/models/cobb_douglas/MODEL_NOTES.md)
- **g\* summary** (`gstar_summary`): every potential-growth estimate on one chart. See [notes](src/models/gstar_summary/MODEL_NOTES.md)

### The neutral rate

Four routes to r\*, kept separate because they disagree, plus the tooling that compares them.
Each set of notes says what its number is conditional on, and none should be quoted without that.

- **From the bond market** (`rstar_bonds`): an Australian wedge over a market world real rate, read off the indexed 10-year yield, the real cash rate and the AOFM 5y5y forward. See [notes](src/models/rstar_bonds/MODEL_NOTES.md)
- **From the RBA's reaction to inflation** (`rstar_rba`): splits the cash rate into a slowly moving neutral and a response to inflation away from target. Two published series, nothing else. See [notes](src/models/rstar_rba/MODEL_NOTES.md)
- **By conditional inversion** (`rstar_invert`): asserts an IS curve and reports the r\* path that assertion forces on the observed gap and cash rate. Not an estimate, and no longer on the summary chart. See [notes](src/models/rstar_invert/MODEL_NOTES.md)
- **From a TVP-VAR** (`rstar_tvpvar`): *retired.* A Lubik-Matthes time-varying-parameter VAR. It reads neutral off the economy's own dynamics, which needs the economy to settle; Australia's does not, anywhere in the sample. The notes are kept for that finding. See [notes](src/models/rstar_tvpvar/MODEL_NOTES.md)
- **HLW** (`rstar_hlw`): a Bayesian Holston-Laubach-Williams build. *Not a source of r\**, and the notes explain why; its trend/cycle decomposition is a separate and working claim. See [notes](src/models/rstar_hlw/MODEL_NOTES.md)
- **r\* summary** (`rstar_summary`): all of the above that qualify, on one nominal scale. See [notes](src/models/rstar_summary/MODEL_NOTES.md)
- **The IS curve, plotted rather than estimated** (`is_curve`): a test bench for the r\* models. The rate-to-output-gap link in Australian data remains unresolved. See [notes](src/models/is_curve/MODEL_NOTES.md)

### Exploratory

- **Bank funding and lending costs** (`bank_costs`): RBA bill, deposit and lending rates against the cash rate. Charts only, no model
- **DSGE** (`dsge`): a family of forward-looking models (financial-accelerator, sticky-wage, and a Bayesian re-estimation). **Experimental; none usable yet**, and their r\* and NAIRU are not credible. See [notes](src/models/dsge/MODELS_EXPLAINED.md)

**Run order.** Several models read another's saved output, so order matters when updating
after new data:

```
expectations → ystar → ustar
expectations → ystar_ustar → rstar_bonds, rstar_invert
is_curve, rstar_summary        (last: they read completed runs)
rstar_rba                      (independent, bar its Taylor-rule chart)
```

### GDP nowcasting

Four complementary approaches to nowcasting the next unpublished quarterly GDP growth:

- **Bridge equations** (`gdp_nowcast_bridge`): monthly indicators completed to quarters via SARIMA, combined by inverse-MSE weights. See [notes](src/models/gdp_nowcast_bridge/MODEL_NOTES.md)
- **Dynamic Factor Model** (`gdp_nowcast_dfm`): common factors from a mixed-frequency panel via Kalman filter, ragged edge handled natively. See [notes](src/models/gdp_nowcast_dfm/MODEL_NOTES.md)
- **Bayesian VAR** (`gdp_nowcast_bvar`): Minnesota-prior VAR conditioned on contemporaneous indicators. T-0 only, a comparator. See [notes](src/models/gdp_nowcast_bvar/MODEL_NOTES.md)
- **Components** (`gdp_nowcast_components`): expenditure-identity build-up of component contributions. T-0 only. See [notes](src/models/gdp_nowcast_components/MODEL_NOTES.md)

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
# Default run: the policy-relevant NAIRU, simple_excess_rstar_blend variant
# (excess-expectations term, with the fixed 35/65 growth/yield r* blend),
# expectations folding to the 2.5% target over 1993–1998
./run-nairu.sh -v

# Re-run validate/analyse/forecast from the saved trace (no re-estimation)
./run-nairu.sh -v --skip-estimate

# Estimation only (skip validate/analyse/forecast)
./run-nairu.sh -v --estimate-only

# 14 variants and 5 anchors; ./run-nairu.sh --help lists them
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

Four models: target-anchored, unanchored, short-run (1yr surveys) and long-run (10yr
breakevens). The NAIRU model reads a spliced series, long-run through 1991Q4 then
target-anchored.

### Cobb-Douglas Productivity Decomposition

```bash
# Deterministic growth accounting (~30 sec)
./run-cd.sh -v

# Or via Python
uv run python -m src.models.cobb_douglas.model -v
```

Use it for the growth accounting, not for potential output or the gap.

### y\*: Potential Output (Bayesian)

```bash
# Default: spec inflation, sample from 1993Q1, 2.5% anchor
./run-ystar.sh -v

# Recharts from the saved trace / estimate without charting
./run-ystar.sh --analyse-only
./run-ystar.sh --no-analyse

# Alternative specification: potential growth from a Cobb-Douglas production
# function (trend capital, hours and MFP) rather than a random-walk drift
./run-ystar.sh --spec production

# The variance settings are imposed, not estimated: sweep them, and the anchor
uv run python -m src.models.ystar.sigma_sweep --param ratio_g
uv run python -m src.models.ystar.sigma_sweep --param ratio_ystar
uv run python -m src.models.ystar.sigma_sweep --param anchor

# Endpoint fragility: the estimate a real-time user would have had
uv run python -m src.models.ystar.realtime
```

### u\*: a NAIRU from one Phillips curve (Bayesian)

Reads saved output from the expectations model. With `--okun` it also reads `ystar`'s gap.

```bash
# Default: u* is a spline with a knot at 2013Q1, no Okun equation, from 1993Q1
./run-ustar.sh --verbose

# Recharts from the saved trace
./run-ustar.sh --analyse-only

# Specification alternatives, kept so the comparison is reproducible
./run-ustar.sh --okun                # restore the Okun equation
./run-ustar.sh --state converge      # the decay law the spline replaced
./run-ustar.sh --knots 1996Q1 2013Q1 # a second knot

# Diagnostics (the last two need --okun, which supplies the gap)
./run-ustar.sh --free-sigma-ustar         # why the drift cannot be estimated
./run-ustar.sh --okun --no-output-gap     # does the given gap actually matter?
./run-ustar.sh --okun --no-phillips       # Okun only
```

### Joint y\* / u\* (Bayesian)

Reads saved output from the expectations model only; it re-estimates both `ystar` and `ustar`
rather than reading them.

```bash
# Default: quarterly gap basis, u* converging, sigma_ustar 0.020, 10,000 draws
./run-ystar-ustar.sh

# Recharts from the saved trace
./run-ystar-ustar.sh --analyse-only

# The controls the result rests on
./run-ystar-ustar.sh --no-okun       # sigma_v should return its prior (it does)
./run-ystar-ustar.sh --no-phillips   # sigma_v with inflation off the left-hand side
./run-ystar-ustar.sh --sigma-v-prior 0.5   # is sigma_v prior-driven? no
```

### Long-run u\*: read off flat inflation, back to 1959

No likelihood and no priors: a rule, and the sensitivity of the answer to it.

```bash
./run-long-run-ustar.sh
./run-long-run-ustar.sh --require-flat-u      # both series settled, not just inflation
./run-long-run-ustar.sh --tolerance 1.5       # looser: picks up the 1990s false positive
./run-long-run-ustar.sh --window 12           # a longer stretch must be flat
./run-long-run-ustar.sh --smooth 1            # no smoothing: finds nothing before 2002
```

### r\*: the natural rate from the bond market (Bayesian)

Reads the Taylor rule's inputs from a completed joint y\*/u\* run (`--input-source separate`
restores the older wiring). r\* itself needs none of them.

```bash
./run-rstar-bonds.sh -v

# The setting the answer leans on: sweep it
./run-rstar-bonds.sh --sigma-walk 0.03

# Specifications tried as the default and rejected, kept as comparators
./run-rstar-bonds.sh --no-forward      # drop the 5y5y window: tighter amplitude, unidentified level
./run-rstar-bonds.sh --curve           # a different third window: r* starts absorbing the policy stance
./run-rstar-bonds.sh --short-rate bill # the 90-day bill: mixes bank credit into the stance

# Diagnostics, kept so the checks are reproducible
./run-rstar-bonds.sh --no-short        # one window: 168 divergences, for the record
./run-rstar-bonds.sh --assert-stance   # assert the stance, report the premium (a tighter mu_tp)
./run-rstar-bonds.sh --steps           # the asserted-break comparator
./run-rstar-bonds.sh --no-world        # does the global anchor do the work? (it does)
./run-rstar-bonds.sh --no-look-through # respond to headline inflation instead
```

### HLW r\* (Bayesian)

Resolutions A to H are specifications built while diagnosing why canonical HLW does not
identify r\* on Australian data. The earlier ones are kept as diagnostic comparators.

```bash
# Default: Resolution A (canonical HLW, r* = g + z), from 1993Q1
./run-rstar-hlw.sh -v

# Alternative resolutions
./run-rstar-hlw.sh --resolution C    # blend with fixed Beta(1,1) on alpha
./run-rstar-hlw.sh --resolution G    # blend + hierarchical Beta

# Estimate only / re-analyse saved trace
./run-rstar-hlw.sh --estimate-only
./run-rstar-hlw.sh --skip-estimate

# Or via Python
uv run python -m src.models.rstar_hlw.run -v
```

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

Charts land in `charts/RStarRBA/`.

### r\* by conditional inversion of an asserted IS curve

Needs a completed joint y\*/u\* run, which supplies the output gap as data.

```bash
./run-rstar-invert.sh                      # the deliverable: one conditional r* path
./run-rstar-invert.sh --ensemble           # sweep sigma_rstar, the one open question
./run-rstar-invert.sh --lag-sweep          # where the rate enters
./run-rstar-invert.sh --rate-lags 1,4,7    # three lags, sharing a Dirichlet
./run-rstar-invert.sh --skip-estimate      # re-chart a saved run
```

### Comparing the estimates

```bash
./run-rstar-summary.sh    # every r* on one nominal scale; re-runs any stale trace
./run-gstar-summary.sh    # every potential-growth estimate on one chart
./run-ustar-summary.sh    # three specifications of the u* model on one chart
```

### The IS curve, plotted rather than estimated

```bash
uv run python -m src.models.is_curve.run                     # lockdowns excluded (default)
uv run python -m src.models.is_curve.run --drop-gfc-pandemic # manufactures a convincing IS curve
uv run python -m src.models.is_curve.run --lag 5             # where the cut sample peaks
```

Charts land in `charts/ISCurve/`, one per r\* variant.

### Bank funding and lending costs

Exploratory. `./run-bank-costs.sh` writes charts to `charts/BankCosts/`. No model, no notes.

### GDP Nowcasting

**Timing:** don't run the nowcasts until about one month before the GDP release. Earlier in the cycle almost no indicators for the target quarter are published: the BVAR declines to nowcast at all, and the bridge/DFM intervals are mostly prior.

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

Bridge and DFM are the workhorses. The BVAR and components models are comparators, T-0 only.

### Outputs

| Directory | Contents |
|-----------|----------|
| `model_outputs/` | Saved traces (`.nc`) and observations (`.pkl`), one pair per model run |
| `output/expectations/` | Expectations traces, HDI estimates (`.parquet`, `.csv`), metadata |
| `charts/` | One directory per model run, named for the model and its variant |

Each chart directory also holds a `run-diagnostics-<prefix>.txt` for the run that produced
it. `uv run python -m src.models.common.diagnostics_report` prints MCMC diagnostics for every
saved trace without re-sampling.

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
    ├── ystar/                  # y* potential output, inflation-defined output gap
    │                           #   (self-contained: imports only src/data)
    ├── ustar/                  # u* from a given output gap: Okun + Phillips, one state
    │                           #   (reads expectations and ystar output)
    ├── ystar_ustar/            # y* and u* estimated jointly, gap partly free
    │                           #   (preferred for the output gap and u*)
    │                           #   (a rule, not an estimate: no likelihood, no priors)
    ├── rstar_bonds/            # r* from the bond market: AU wedge over world r*, two windows
    ├── rstar_hlw/              # HLW Bayesian r* model (AU data)
    ├── rstar_rba/              # neutral revealed by the RBA's reaction to inflation
    ├── rstar_invert/           # r* by conditional inversion of an asserted IS curve
    ├── rstar_summary/          # every r* on one nominal scale (not a model)
    ├── gstar_summary/          # every potential-growth estimate on one chart (not a model)
    ├── ustar_summary/          # three specifications of the u* model on one chart (not a model)
    ├── is_curve/               # the IS curve plotted, not estimated: a test bench for the r* models
    ├── bank_costs/             # bank funding and lending costs vs the cash rate (exploratory, charts only)
    ├── gdp_nowcast_bridge/     # GDP nowcast: bridge equations
    ├── gdp_nowcast_dfm/        # GDP nowcast: Dynamic Factor Model
    ├── gdp_nowcast_bvar/       # GDP nowcast: Bayesian VAR (T-0 only)
    ├── gdp_nowcast_components/ # GDP nowcast: expenditure-identity components (T-0 only)
    └── dsge/                   # DSGE models: experimental, NOT usable yet
```
