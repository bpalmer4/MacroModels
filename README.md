# MacroModels

Developing economic models of the Australian macroeconomy. Mostly Bayesian state-space
estimation in PyMC, with some deterministic growth accounting, on ABS and RBA data.

Each model has its own `MODEL_NOTES.md` recording what it assumes, what it is conditional on,
and what should not be quoted from it. The links below go there; this page is only a map.

## Live models

### Supply side and structural estimation

- **Inflation Expectations** (`expectations`): latent inflation expectations from surveys and market data. See [notes](src/models/expectations/MODEL_NOTES.md)
- **y\* potential output** (`ystar`): potential as a slow-moving random walk, with the output gap defined by inflation's deviation from target. See [notes](src/models/ystar/MODEL_NOTES.md)
- **u\*** (`ustar`): a NAIRU from one expectations-augmented Phillips curve, with u\* a spline. See [notes](src/models/ustar/MODEL_NOTES.md)
- **Joint y\*/u\*** (`ystar_ustar`): both of the above in one likelihood, with the gap partly free and u\* a spline. See [notes](src/models/ystar_ustar/MODEL_NOTES.md)
- **Cobb-Douglas MFP** (`cobb_douglas`): deterministic growth accounting into capital, labour and MFP. See [notes](src/models/cobb_douglas/MODEL_NOTES.md)
- **g\* summary** (`gstar_summary`): every potential-growth estimate on one chart. See [notes](src/models/gstar_summary/MODEL_NOTES.md)
- **u\* summary** (`ustar_summary`): six u\* specifications from `ustar` and `ystar_ustar` on one chart. See [notes](src/models/ustar_summary/MODEL_NOTES.md)

### The neutral rate

- **From the bond market** (`rstar_bonds`): an Australian wedge over a market world real rate, read off the indexed 10-year yield, the real cash rate and the AOFM 5y5y forward. See [notes](src/models/rstar_bonds/MODEL_NOTES.md)
- **From the RBA's reaction to inflation** (`rstar_rba`): splits the cash rate into a slowly moving neutral and a response to inflation away from target. See [notes](src/models/rstar_rba/MODEL_NOTES.md)
- **Semi-structural open economy** (`rstar_qpm`): a small QPM-style model (IS curve, exchange rate, Phillips curve, policy rule) with the 5y5y forward setting the level and the structure shaping the path. See [notes](src/models/rstar_qpm/MODEL_NOTES.md)
- **r\* summary** (`rstar_summary`): the three above on one nominal scale. See [notes](src/models/rstar_summary/MODEL_NOTES.md)

### GDP nowcasting

Four complementary approaches to nowcasting the next unpublished quarterly GDP growth:

- **Bridge equations** (`gdp_nowcast_bridge`): monthly indicators completed to quarters via SARIMA, combined by inverse-MSE weights. See [notes](src/models/gdp_nowcast_bridge/MODEL_NOTES.md)
- **Dynamic Factor Model** (`gdp_nowcast_dfm`): common factors from a mixed-frequency panel via Kalman filter, ragged edge handled natively. See [notes](src/models/gdp_nowcast_dfm/MODEL_NOTES.md)
- **Bayesian VAR** (`gdp_nowcast_bvar`): Minnesota-prior VAR conditioned on contemporaneous indicators. T-0 only, a comparator. See [notes](src/models/gdp_nowcast_bvar/MODEL_NOTES.md)
- **Components** (`gdp_nowcast_components`): expenditure-identity build-up of component contributions. T-0 only. See [notes](src/models/gdp_nowcast_components/MODEL_NOTES.md)

### Exploratory

- **Bank funding and lending costs** (`bank_costs`): RBA bill, deposit and lending rates against the cash rate. Charts only, no model

## Retired, superseded or not working

Kept for their notes, which record what was tried and why it did not hold.

- **NAIRU + Output Gap** (`nairu`): the original joint NAIRU and potential-output model, with a wage equation. See [notes](src/models/nairu/MODEL_NOTES.md)
- **HLW** (`rstar_hlw`): a Bayesian Holston-Laubach-Williams build. See [notes](src/models/rstar_hlw/MODEL_NOTES.md)
- **HLW by Kalman filter** (`rstar_hlw_kalman`): the original HLW specification, estimated by Kalman filter and maximum likelihood. See [notes](src/models/rstar_hlw_kalman/MODEL_NOTES.md)
- **From a TVP-VAR** (`rstar_tvpvar`): a Lubik-Matthes time-varying-parameter VAR. See [notes](src/models/rstar_tvpvar/MODEL_NOTES.md)
- **By conditional inversion** (`rstar_invert`): r\* backed out of an asserted IS curve. See [notes](src/models/rstar_invert/MODEL_NOTES.md)
- **The IS curve, plotted rather than estimated** (`is_curve`): the search for an IS curve in Australian data. See [notes](src/models/is_curve/MODEL_NOTES.md)
- **DSGE** (`dsge`): a family of forward-looking models (financial-accelerator, sticky-wage, and a Bayesian re-estimation). See [notes](src/models/dsge/MODELS_EXPLAINED.md)

## Quickstart

```bash
# Install dependencies
uv sync

# Expectations first (much of the rest reads it), then the joint y*/u* model
./run-expectations.sh
./run-ystar-ustar.sh
```

Each `run-*.sh` says in its header which other models must be run first, and lists its options
at the bottom.

ABS and RBA data are fetched and cached automatically, no credentials needed. The only
exception is `src/data/fred_loader.py`, used for the world real-rate comparators in the r\*
notes: it reads a FRED API key from `fred.api` in the project root, one line, gitignored.
A free key comes from [FRED](https://fred.stlouisfed.org/docs/api/api_key.html). Nothing in
the default model runs depends on it.

## Outputs

| Directory | Contents |
|-----------|----------|
| `model_outputs/` | Saved traces (`.nc`) and observations (`.pkl`), one pair per model run |
| `output/expectations/` | Expectations traces, HDI estimates (`.parquet`, `.csv`), metadata |
| `charts/` | One directory per model run, named for the model and its variant |

Each chart directory also holds a `run-diagnostics-<prefix>.txt` for the run that produced
it. `uv run python -m src.models.common.diagnostics_report` prints MCMC diagnostics for every
saved trace without re-sampling.

## Project Structure

```
src/
├── paths.py            # Where the project's directories are, resolved once
├── data/               # Data fetching (ABS, RBA) and preparation
├── utilities/          # Shared utilities (rate conversion, etc.)
└── models/
    ├── common/                 # Shared model machinery (results, cli, diagnostics,
    │                           #   extraction, timeseries, sources, charts)
    ├── expectations/           # Inflation expectations signal extraction
    ├── nairu/                  # NAIRU + Output Gap model (estimate → validate → analyse → forecast)
    │   └── analysis/           # Plotting and diagnostics modules
    ├── cobb_douglas/           # Cobb-Douglas MFP decomposition
    ├── ystar/                  # y* potential output, inflation-defined output gap
    │                           #   (self-contained: imports only src/data)
    ├── ustar/                  # u* from a given output gap: Okun + Phillips, one state
    │                           #   (reads expectations and ystar output)
    ├── ystar_ustar/            # y* and u* estimated jointly, gap partly free, u* a spline
    │                           #   (preferred for the output gap and u*)
    │                           #   (a rule, not an estimate: no likelihood, no priors)
    ├── rstar_bonds/            # r* from the bond market: AU wedge over world r*, two windows
    ├── rstar_hlw/              # HLW Bayesian r* model (AU data)
    ├── rstar_rba/              # neutral revealed by the RBA's reaction to inflation
    ├── rstar_invert/           # r* by conditional inversion of an asserted IS curve
    ├── rstar_summary/          # every r* on one nominal scale (not a model)
    ├── gstar_summary/          # every potential-growth estimate on one chart (not a model)
    ├── ustar_summary/          # six u* specifications from two models on one chart (not a model)
    ├── is_curve/               # the IS curve plotted, not estimated: a test bench for the r* models
    ├── bank_costs/             # bank funding and lending costs vs the cash rate (exploratory, charts only)
    ├── gdp_nowcast_bridge/     # GDP nowcast: bridge equations
    ├── gdp_nowcast_dfm/        # GDP nowcast: Dynamic Factor Model
    ├── gdp_nowcast_bvar/       # GDP nowcast: Bayesian VAR (T-0 only)
    ├── gdp_nowcast_components/ # GDP nowcast: expenditure-identity components (T-0 only)
    └── dsge/                   # DSGE models: experimental, NOT usable yet
```
