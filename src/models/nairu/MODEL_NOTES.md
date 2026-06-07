# NAIRU + Output Gap Model

Bayesian state-space model for jointly estimating NAIRU, potential output, and output gaps for Australia using PyMC (NumPyro NUTS backend).

## Summary

| Component | Method | Key Feature |
|-----------|--------|-------------|
| NAIRU | Gaussian random walk (default) | Student-t(nu=4) variant for fat tails |
| Potential Output | Cobb-Douglas + Gaussian innovations (default) | SkewNormal variant available |
| Phillips Curves | Regime-switching, 3 regimes (default preset) | Single-slope variants; excess-expectations term (default preset) |
| Identification | 2 state + N observation equations | Joint estimation with proper uncertainty |
| Scenario Analysis | Model-consistent projection | 4-quarter horizon with policy scenarios |

---

## Architecture

### File Structure

```
nairu/
├── config.py                 # ModelConfig dataclass — single source of truth
├── base.py                   # SamplerConfig, coefficient utilities
├── observations.py           # Observation matrix assembly (data → numpy arrays)
├── estimate.py               # Build model, sample posterior, save results
├── results.py                # Result loading and management
├── run.py                    # Model run entry point
├── validate.py               # Model validation checks
├── analyse.py                # Load results, diagnostics, plotting
├── forecast.py               # Deterministic scenario analysis
├── forecast_bayesian.py      # Monte Carlo scenario analysis
├── forecast_plots.py         # Forecast visualisation
├── MODEL_NOTES.md            # This file
│
├── equations/                # One file per equation, standard API
│   ├── __init__.py           # API contract documentation
│   ├── nairu.py              # NAIRU state equation (Gaussian / Student-t)
│   ├── potential.py          # Potential output state equation (Normal / SkewNormal)
│   ├── okun.py               # Okun's Law (simple / gap-to-gap)
│   ├── phillips_price.py     # Price Phillips curve (single / regime-switching)
│   ├── phillips_wage.py      # Wage Phillips curve ULC (single / regime-switching)
│   ├── phillips_hcoe.py      # Hourly COE Phillips curve (single / regime-switching)
│   ├── is_curve.py           # IS curve (output gap dynamics)
│   ├── participation.py      # Participation rate (discouraged worker)
│   ├── employment.py         # Employment (labour demand)
│   ├── exchange_rate.py      # Exchange rate (UIP-style TWI)
│   ├── import_price.py       # Import price pass-through
│   └── net_exports.py        # Net exports
│
└── analysis/                 # Plotting and diagnostics modules
    ├── __init__.py
    ├── decomposition_types.py          # Decomposition data structures
    ├── _decomposition_helpers.py       # Shared decomposition utilities
    ├── decompose_inflation.py          # Price inflation decomposition
    ├── decompose_wage_inflation.py     # Wage inflation decomposition
    ├── decompose_hcoe_inflation.py     # Hourly COE inflation decomposition
    ├── plot_decomposition.py           # Decomposition charts
    ├── plot_equations.py               # Equation fit charts
    ├── plot_equilibrium_rates.py       # r* and equilibrium rate charts
    ├── plot_gdp_vs_potential.py        # GDP vs potential output
    ├── plot_nairu.py                   # NAIRU estimate charts
    ├── plot_nairu_comparison.py        # Cross-variant NAIRU comparison
    ├── plot_obs_grid.py                # Observation data grid
    ├── plot_output_gap.py              # Output gap charts
    ├── plot_output_gap_comparison.py   # Cross-variant output gap comparison
    ├── plot_phillips_curves.py         # Phillips curve scatter plots
    ├── plot_phillips_slope.py          # Time-varying Phillips slope
    ├── plot_posteriors_bar.py          # Posterior bar charts
    ├── plot_posteriors_kde.py          # Posterior KDE plots
    ├── plot_potential_growth.py        # Potential growth charts
    ├── plot_potential_growth_comparison.py  # Cross-variant growth comparison
    ├── plot_potential_growth_smoothing.py   # Growth smoothing charts
    ├── plot_taylor_rule.py             # Taylor rule charts
    ├── plot_unemployment_gap.py        # Unemployment gap charts
    ├── posterior_predictive.py         # Posterior predictive checks
    └── residual_autocorrelation.py     # Residual autocorrelation diagnostics
```

### Pipeline Flow

```
┌─────────────────────────────────────────────────────────────┐
│                         run.py                              │
│                                                             │
│   Orchestrates: estimate → validate → analyse → forecast    │
│   CLI: python -m src.models.nairu.run                       │
└──────────────────────────┬──────────────────────────────────┘
                           │
         ┌─────────────────┼─────────────────────┐
         ▼                 ▼                     ▼
┌─────────────────┐ ┌─────────────────┐ ┌────────────────────────┐
│  estimate.py    │ │  analyse.py     │ │  forecast.py           │
│  results.py     │ │                 │ │  forecast_bayesian.py  │
│                 │ │                 │ │  forecast_plots.py     │
│ build_model()   │ │ load_results()  │ │                        │
│ sample_model()  │ │ diagnostics     │ │ deterministic +        │
│ save_results()  │ │ all plots       │ │ Monte Carlo scenarios  │
└─────────────────┘ └─────────────────┘ └────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────┐
│                    equations/*                               │
│                                                             │
│  Standard API:                                              │
│    def equation(obs, model, latents, constant) -> str       │
│                                                             │
│  State equations populate latents dict:                     │
│    latents["nairu"] = ...                                   │
│    latents["potential_output"] = ...                        │
│                                                             │
│  Observation equations read from latents dict               │
│  All equations return a self-describing model string        │
└─────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────┐
│                       config.py                             │
│                                                             │
│  ModelConfig     — all model options in one place           │
│  SamplerConfig   — NUTS sampler settings (in base.py)      │
│                                                             │
│  Presets: SIMPLE, COMPLEX, PRESETS dict                     │
│  Serializable: saved with results, loaded by analyse/fcst  │
│  config.rfooter  — chart label derived from config.label   │
└─────────────────────────────────────────────────────────────┘
```

---

## ModelConfig

`ModelConfig` is the single source of truth for all model options. It is:

- **Defined in one place** (`config.py`)
- **Passed to `build_model()`** instead of 20+ keyword arguments
- **Serialized with saved results** so analyse and forecast know the variant
- **Provides chart labels** via `config.rfooter` and `config.chart_dir_name`

### Preset Variants

| Preset | NAIRU | Regimes | Excess Expectations | Extra Equations | Import Control |
|--------|-------|---------|---------------------|-----------------|----------------|
| `default` | Gaussian | No | No | — | No |
| `simple` | Gaussian | No | No | — | Yes |
| `simple_excess` | Gaussian | No | Yes | — | Yes |
| `simple_regime` | Gaussian | Yes | No | — | Yes |
| `simple_excess_regime` *(run.py default)* | Gaussian | Yes | Yes | — | Yes |
| `complex` | Student-t | Yes | No | Exchange rate, import price, participation, employment, net exports | Yes |

### Creating Custom Variants

```python
from src.models.nairu.config import ModelConfig

custom = ModelConfig(
    label="my_variant",
    student_t_nairu=True,
    regime_switching=False,
    include_participation=True,
    nairu_const={"nairu_innovation": 0.20},
)
```

---

## Equation API

Every equation follows the same interface:

```python
def equation_name(
    obs: dict[str, np.ndarray],    # Observed data
    model: pm.Model,                # PyMC model context
    latents: dict[str, Any],        # Shared latent variables
    constant: dict[str, Any] | None = None,  # Fixed values
) -> str:                           # Self-describing model string
```

**State equations** (nairu, potential) add their latent to `latents`:
```python
latents["nairu"] = nairu
return "NAIRU_t = NAIRU_{t-1} + e_t,  e ~ N(0, sigma)"
```

**Observation equations** read from `latents`:
```python
nairu = latents["nairu"]
potential = latents["potential_output"]
```

This decouples equations from each other — they communicate only through the `latents` dict.

### Coefficient Creation

Each equation uses `set_model_coefficients()` to create priors:

```python
settings = {
    "beta": {"mu": 0.5, "sigma": 0.1},              # Normal
    "gamma": {"mu": -0.3, "sigma": 0.2, "upper": 0}, # TruncatedNormal
    "epsilon": {"sigma": 0.5},                        # HalfNormal
}
mc = set_model_coefficients(model, settings, constant={"beta": 0.5})
```

If a coefficient is in `constant`, it is fixed (not estimated).

---

## State Equations

### NAIRU (`equations/nairu.py`)

Gaussian random walk (default):
```
NAIRU_t = NAIRU_{t-1} + e_t,  e ~ N(0, sigma)
```
- sigma (nairu_innovation): typically fixed at 0.15

Student-t variant (`student_t_nairu=True`):
```
NAIRU_t = NAIRU_{t-1} + e_t,  e ~ StudentT(nu=4, 0, sigma)
```
- Fat tails: many small moves, occasional large shifts (GFC, COVID)

### Potential Output (`equations/potential.py`)

Cobb-Douglas production function:
```
g_potential_t = alpha_t x g_K_t + (1-alpha_t) x g_L_t + g_MFP_t + e_t
Y*_t = Y*_{t-1} + g_potential_t
```

- alpha_t = time-varying capital share from ABS national accounts
- MFP derived from wage data using Solow residual identity
- Default: Gaussian innovations
- SkewNormal variant: asymmetric (growth more likely than decline)

---

## Inflation Expectations Anchor

The expectations series `π_exp` feeds the price and wage Phillips curves. The choice
of anchor determines what question the model is answering — it is not a pure fit
decision.

### Two research questions

| Question | Appropriate anchor | Rationale |
|----------|-------------------|-----------|
| **Policy counterfactual**: *"What setting (output gap, unemployment gap, cash rate) is consistent with inflation returning to the 2.5% target?"* | `rba`, `target` | Pins the long-run anchor to 2.5%, so the Phillips curve residual measures distance from the target-consistent equilibrium. |
| **Inflation explanation**: *"Given where expectations actually were, what is driving observed inflation?"* | `unanchored`, `unanchored_raw` | Lets the anchor move with survey/market evidence so demand, import price and GSCPI channels are not contaminated by a forced 2.5% lock. |

The policy-counterfactual anchors tend to *widen* the residual during 2022–2024 (when
actual expectations drifted above 2.5%), which is correct behaviour — the residual is
meant to signal distance from target. The inflation-explanation anchors will always
fit tighter on those same windows because they absorb that drift into the anchor
instead of the residual.

### Anchor modes

| `--anchor` | Pre-1993 | 1993→1998 | Post-1998 |
|------------|----------|-----------|-----------|
| `rba` | RBA PIE_RBAQ | phased | **2.5% target** |
| `target` *(default)* | model expectations (target-anchored series) | phased | **2.5% target** |
| `expectations` | model expectations (target-anchored series) — full series, no phasing | | |
| `unanchored` | RBA PIE_RBAQ | phased | **unanchored model median** |
| `unanchored_raw` | unanchored model median — full series, no phasing | | |

- "phased" = linear interpolation between pre-1993 source and post-1998 destination across 1992Q4–1998Q4.
- The unanchored model is `expectations_unanchored_hdi.parquet` (signal-extraction run with no 2.5% prior).
- `unanchored` was designed to be the "similar treatment" counterpart to `rba`: same phase window, same pre-1993 history, only the post-1998 destination changes.
- `unanchored_raw` is the cleanest inflation-explanation anchor but loses the RBA early-1980s history.

### Interpreting the NAIRU estimate under each anchor

The anchor choice changes what the estimated NAIRU means:

- **Target-locked anchors (`rba`, `target`)** — the NAIRU is the unemployment rate consistent with inflation returning to the **2.5% target**. This is the policy-relevant NAIRU: the gap `U − NAIRU` measures distance from a target-consistent equilibrium.
- **Non-target anchors (`unanchored`, `unanchored_raw`, `expectations`)** — the NAIRU is the unemployment rate consistent with **prevailing expectations**, whatever they happen to be. If expectations sit above (or below) 2.5%, closing the unemployment gap to this NAIRU will stabilise inflation at that expectations level — not at target. Reaching the target then requires either (a) waiting for expectations to re-anchor at 2.5% or (b) running unemployment *above* this NAIRU until they do.

In short: use the target-locked NAIRU to ask *"what rate gets us back to 2.5%?"* and the non-target NAIRU to ask *"what rate would stabilise inflation where expectations currently sit?"* The two coincide when expectations are already at target.

### Empirical comparison (2025Q4 endpoint)

Running the `default` variant with each anchor:

| Anchor | Endpoint π_exp | Endpoint NAIRU (median) | 90% HDI |
|---|---|---|---|
| `rba` (RBA → Target)        | 2.50% | 4.84% | [4.32, 5.39] |
| `unanchored` (RBA → Unanchored) | 2.70% | 4.61% | [4.05, 5.19] |
| **Δ**                        | **+0.20** | **−0.23** | |

The difference is consistent with the price Phillips slope γ_π ≈ −0.82: a +0.20 pp shift in π_exp implies Δ(U − U*) ≈ −0.20 / −0.82 ≈ −0.24 pp to fit the same observed inflation, so holding U fixed, U* drops by ~0.24 pp — matching the 0.23 pp NAIRU gap. Higher expectations "absorb" inflation pressure, so the model infers a lower NAIRU.

### Implementation

`src/models/nairu/observations.py` dispatches on `anchor_mode`:

- `rba` → `get_rba_expectations()` — RBA PIE_RBAQ, phased to 2.5%
- `target` / `expectations` → `get_model_expectations()` + optional `apply_anchor_mode()`
- `unanchored` → `_build_rba_to_unanchored()` — RBA pre-1993, phased to unanchored post-1998
- `unanchored_raw` → `get_model_expectations_unanchored()` — unanchored series unmodified

### Excess expectations (`excess_expectations` config option)

Bridges the two research questions above: keeps the target-locked anchor (policy
relevance) but adds an estimated pass-through from de-anchored expectations
(explanatory power). Designed for `--anchor target`; presets `simple_excess` and
`simple_excess_regime` (the run.py default).

```
pi = quarterly(pi_exp) + beta x excess_exp + gamma x u_gap + controls + e

excess_exp_t = unanchored model median_t − π_exp_t   (zero through 1992Q4)
```

- `excess_exp` is built in `observations.py:_build_excess_expectations()` as the
  unanchored model median less the anchor actually fed to the Phillips curves.
  It is zeroed through `PHASE_START` (1992Q4): pre-target the anchor is itself an
  expectations model, so any difference is estimator noise, not excess over target.
- The beta term is computed as a quarterly-space difference
  (`quarterly(π_exp + excess) − quarterly(π_exp)`), so the effective anchor is
  exactly `(1−beta) x anchor + beta x actual` in quarterly units.
- **Nesting**: beta = 0 recovers the pure `target` model; beta = 1 reproduces the
  unanchored expectations model post-1998. beta is the estimated *degree of
  de-anchoring* (priors: Normal(0.5, 0.3) on `beta_pi`, `beta_wg`, `beta_hcoe`).
- **Interpretation**: the NAIRU remains target-consistent (conditional on
  expectations at target). At U = NAIRU, inflation settles at
  `2.5 + beta x excess` until expectations re-anchor — the beta term quantifies
  how much de-anchoring is costing relative to target.
- **Identification**: the excess series is two-sided — ≈ +0.4 around 2008,
  −0.4 to −0.6 through 2016–2019, +0.84 at 2022Q4, +0.2 at the 2025 endpoint.
  The negative 2016–19 episode identifies beta in a window where GSCPI and
  import-price controls are quiet.
- **Known wrinkle**: at 1993Q1–Q2 the series reads −0.29/−0.56 then flips sign —
  residual estimator disagreement leaking through the switch-on. Two observations
  out of ~130; left as-is.
- Wage equations get the same term (own betas) when `wage_expectations` is on.
- The `r_gap` fed to the IS curve still uses the locked anchor (`cash_rate −
  π_exp − r*`), overstating the ex-ante real rate when expectations drift above
  target. Deliberately unchanged in the first pass; candidate refinement.

---

## Observation Equations

### Price Phillips Curve (`equations/phillips_price.py`)

```
pi = quarterly(pi_exp) + gamma x u_gap [+ rho x d4pm] [+ xi x GSCPI^2] + e
```
- u_gap = (U - NAIRU) / U (percentage deviation)
- Regime-switching variant: gamma_pre_gfc, gamma_gfc, gamma_covid

### Wage Phillips Curve — ULC (`equations/phillips_wage.py`)

```
dulc = alpha + pi_exp + gamma x u_gap + lambda x dU/U [+ phi x d4dfd] + e
```
- Speed limit effect (lambda x dU/U)
- Optional demand deflator pass-through

### Wage Phillips Curve — Hourly COE (`equations/phillips_hcoe.py`)

```
dhcoe = alpha + pi_exp + gamma x u_gap + lambda x dU/U [+ phi x d4dfd] + psi x MFP + e
```
- Adds productivity channel (psi x MFP)

### Okun's Law (`equations/okun.py`)

Simple form:
```
dU = beta x output_gap + e
```

Gap-to-gap form:
```
U_gap = tau2 x U_gap_{-1} + tau1 x Y_gap + e
```

### IS Curve (`equations/is_curve.py`)

```
y_gap = rho x y_gap_{-1} - beta x r_gap_{-2} + gamma x fiscal_{-1} + e
```

### Participation Rate (`equations/participation.py`)

```
dpr = beta_pr x (U_{-1} - NAIRU_{-1}) + e
```
- Discouraged worker effect (beta_pr < 0)

### Employment (`equations/employment.py`)

```
demp = alpha + beta_ygap x output_gap + beta_wage x (dulc - dmfp) + e
```

### Exchange Rate (`equations/exchange_rate.py`)

```
de = rho x de_{-1} + beta_r x r_gap_{-1} + e
```
- UIP-style; weak coefficients expected (UIP puzzle)

### Import Price Pass-Through (`equations/import_price.py`)

```
d4pm = beta_pt x d4twi_{-1} + beta_oil x d4oil_{-1} + rho x d4pm_{-1} + e
```

### Net Exports (`equations/net_exports.py`)

```
d(NX/Y) = beta_ygap x output_gap + beta_twi x dtwi + e
```

---

## Data Flow

1. **Data Preparation** (`src/data/observations.py`): ABS data via `readabs`, transformations, regime indicators
2. **Model Building** (`estimate.build_model(obs, config)`): config-driven equation assembly
3. **Sampling** (`base.sample_model()`): NUTS via NumPyro backend
4. **Saving** (`estimate.save_results()`): trace (.nc) + obs/config (.pkl)
5. **Analysis** (`analyse.py`): loads saved results, diagnostics, charts
6. **Scenarios** (`forecast.py`, `forecast_bayesian.py`): policy scenario projections

### Saved Results Format

Each run produces two files:
- `{prefix}_trace.nc` — ArviZ InferenceData (posterior samples)
- `{prefix}_obs.pkl` — dict containing:
  - `obs`: observation arrays
  - `obs_index`: PeriodIndex
  - `constants`: fixed parameter values
  - `anchor_label`: expectations anchor description
  - `chart_obs`: extended observations for charting
  - `config`: serialized ModelConfig dict

The config is saved so that `analyse.py` and `forecast.py` can reconstruct the exact variant without needing the original ModelConfig object.

---

## Chart Labelling

All charts include `config.rfooter` which shows the variant label:
```
NAIRU + Output Gap Model [simple]
NAIRU + Output Gap Model [complex]
```

Charts are saved to `charts/nairu_{label}_{anchor}/`:
```
charts/nairu_simple_excess_regime_target/
charts/nairu_simple_target/
charts/nairu_complex_unanchored/
```

---

## Running the Model

### Full Pipeline
```bash
python -m src.models.nairu.run -v                     # default: simple_excess_regime, target anchor
python -m src.models.nairu.run -v --variant simple
python -m src.models.nairu.run -v --variant simple complex --anchor unanchored
```

### Estimation Only
```bash
python -m src.models.nairu.run -v --estimate-only
```

### From Python
```python
from src.models.nairu.config import ModelConfig, SIMPLE
from src.models.nairu.estimate import run_estimate

# Default model
trace, obs, obs_index, anchor_label = run_estimate()

# Custom variant
config = ModelConfig(label="tight_nairu", nairu_const={"nairu_innovation": 0.10})
trace, obs, obs_index, anchor_label = run_estimate(config=config)

# Preset
trace, obs, obs_index, anchor_label = run_estimate(config=SIMPLE)
```

---

## Key Design Decisions

- **Joint estimation**: NAIRU + potential + gaps estimated together for proper uncertainty propagation
- **Deterministic r***: Computed from Cobb-Douglas growth, not estimated as latent
- **Percentage unemployment gap**: `(U - NAIRU) / U` for scale invariance
- **MFP floored at zero**: Negative MFP reflects cyclical underutilization, not technological regress
- **COVID smoothing**: Labour inputs Henderson-smoothed during 2020Q1-2023Q2
- **Fixed nairu_innovation**: Poorly identified; fixing improves sampling (0.15 Gaussian, 0.10 Student-t)

---

## Equation Ordering and NUTS Sensitivity

**The order in which equations are added to the PyMC model matters for sampling.**

NUTS builds a mass matrix from the model's random variables. The order variables are
registered affects the mass matrix structure and leapfrog trajectory, which can cause
divergent transitions or uneven chain progress even when the model is mathematically
identical.

The known-good ordering (zero divergences over 30+ runs) is:

```
1. NAIRU              (state)
2. Potential output   (state)
3. Okun's Law         (observation)
4. Price Phillips     (observation)
5. Wage Phillips ULC  (observation)
6. IS curve           (observation)
7. Hourly COE Phillips (observation)
8. Participation      (observation, optional)
9. Employment         (observation, optional)
10. Exchange rate     (observation, optional)
11. Import price      (observation, optional)
12. Net exports       (observation, optional)
```

**Do not reorder equations without re-testing sampling.** Moving IS before the Phillips
curves or swapping IS and HCOE both produced divergences and uneven chain speeds in
testing. The ordering above was established empirically.
