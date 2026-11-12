"""Robustness tests for the inflation expectations model.

Asks whether the recent level of the unanchored series, and its distance from
2.5, depends on a modelling choice that departs from published practice: the
fixed step size of the trend, wages as measurements, and the prior on each
survey's level effect. Also compares the series with simple external
benchmarks and reruns the wage check that reads it.

A diagnostic harness. It never calls `save_results()`, so production output in
`output/expectations/` is untouched: traces go to `output/expectations_tests/`,
charts and REPORT.md to `charts/expectations_tests/`. Each trace is cached and
not re-estimated while its file exists.

Run with: uv run python -m src.models.expectations.robustness
"""

import argparse
import json
import pickle
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from statistics import NormalDist

import arviz as az
import matplotlib.pyplot as plt
import mgplot as mg
import numpy as np
import pandas as pd
import pymc as pm
import statsmodels.api as sm
import xarray as xr
from mgplot.finalise_plot import sanitize_filename

from src.data.expectations import G3_SERIES, G3_URL
from src.data.expectations_model import get_model_expectations_unanchored
from src.data.inflation import get_trimmed_mean_qrtly, get_weighted_median_qrtly
from src.data.labour_force import get_log_vu_ratio_qrtly
from src.data.productivity import get_gdp_per_hour_trend_growth_ex_covid
from src.data.rba_loader import get_inflation_expectations
from src.data.wpi import get_wpi_private_growth_annual
from src.models.common.results import posterior_summary, trace_group, vector_draws
from src.models.expectations.common import DEFAULT_DRAWS, DEFAULT_TUNE, OUTPUT_DIR
from src.models.expectations.stage1 import (
    ALPHA_PRIOR_SIGMA,
    MODEL_CONFIGS,
    PRODUCTIVITY_TREND_YEARS,
    QUARTERS_PER_YEAR,
    WPI_EXCLUDE_END,
    WPI_EXCLUDE_START,
    ModelConfigDict,
    _build_pymc_model,
    load_data,
)
from src.models.expectations.stage2 import PLOT_KWARGS
from src.models.ystar_ustar.results import load_results as load_joint_results
from src.paths import CHARTS, OUTPUT, ROOT
from src.utilities.rate_conversion import annualize

# --- Paths ---

TEST_OUTPUT = OUTPUT / "expectations_tests"
TEST_CHARTS = CHARTS / "expectations_tests"
REPORT_PATH = TEST_CHARTS / "REPORT.md"
DATA_SNAPSHOT = TEST_OUTPUT / "data_snapshot.pkl"
NOWAGES_PATH = TEST_OUTPUT / "expectations_unanchored_nowages_quarterly.parquet"
PRODUCTION_PATH = OUTPUT_DIR / "expectations_unanchored_hdi_quarterly.parquet"

# --- Sampler ---

START = "1983Q1"
SEED = 42
TEST_DRAWS = 4000
TEST_TUNE = 2000
TEST_CHAINS = 4
TARGET_ACCEPT = 0.8
MARKET_TARGET_ACCEPT = 0.9
RETRY_TARGET_ACCEPT = 0.95

# Convergence gate. Stricter than the repo's general rule on divergences.
GATE_MAX_R_HAT = 1.01
GATE_MIN_ESS_BULK = 400
GATE_MAX_DIVERGENCES = 0

# --- Definitions ---

BAND_LOWER = 0.05
BAND_UPPER = 0.95
INTERVAL_PROB = BAND_UPPER - BAND_LOWER  # every "90% interval" in the report
LOW_WINDOW = ("2016Q1", "2019Q4")
PEAK_WINDOW = ("2021Q1", "2024Q4")
RECENT_QUARTERS = 4
COMPARE_FROM = "1998Q1"
PRE_TARGETING = ("1983Q1", "1992Q4")

# T0: the most the baseline may differ from the production file and still count as reproduced.
REPRODUCE_TOLERANCE = 0.05

ROBUST_BELOW = 0.15
SENSITIVE_UPTO = 0.30

INFLATION_TARGET = 2.5
TARGET_BAND = (2.0, 3.0)

# T1: two estimated-step posteriors are called materially different when their
# medians differ by more than this share of the smaller median. The harness's
# own threshold, not the spec's.
MATERIAL_SIGMA_SHIFT = 0.25

# T3
BENCH_FROM = "1996Q1"
BENCH_MEASURES = ["market_1y", "market_yoy", "business", "breakeven"]
PIE_WINDOW = ("1998Q1", "2019Q1")

# T4
LEVEL_WINDOWS = [("1996Q1", None), ("1996Q1", "2007Q4"), ("2008Q1", "2019Q4"), ("2021Q1", None)]

# T5: the regimes the market model's inflation risk premium uses.
BREAKEVEN_REGIMES = [(None, "2008Q3"), ("2008Q4", "2020Q4"), ("2021Q1", None)]
PREMIUM_SPREAD_LIMIT = 0.25
# Rows above the data in G3's "Data" sheet, ending with the Series ID row.
G3_HEADER_ROWS = 11

# T2: WPI gap windows; the second starts the quarter after the excluded window.
WPI_GAP_PRE = ("2015Q1", "2019Q4")
WPI_GAP_POST = (str(pd.Period(WPI_EXCLUDE_END, freq="Q") + 1), None)

# T6: HAC lag length used by the documented wage check.
WAGE_HAC_LAGS = 4

RFOOTER = "Expectations robustness tests"
LFOOTER = "Australia. Posterior medians, quarterly. "


# --- Run registry ---


@dataclass(frozen=True)
class RunSpec:
    """One estimation: a production config with some keys overridden."""

    name: str
    test: str
    model_type: str
    label: str
    overrides: ModelConfigDict = field(default_factory=ModelConfigDict)
    verdict: bool = True
    # Sample at the production budget instead of the test budget. Only for the
    # reproduction diagnostic, which separates sampler budget from data revisions.
    production_budget: bool = False


RUNS: list[RunSpec] = [
    RunSpec("T0_unanchored", "T0", "unanchored", "Baseline unanchored", verdict=False),
    RunSpec("T0_short", "T0", "short", "Baseline short", verdict=False),
    RunSpec("T0_market", "T0", "market", "Baseline market", verdict=False),
    RunSpec("T0_prodbudget", "T0", "unanchored", "Baseline at production budget", verdict=False,
            production_budget=True),
    RunSpec("T1a", "T1", "unanchored", "sigma_late 0.035 fixed", ModelConfigDict(sigma_late=0.035)),
    RunSpec("T1b", "T1", "unanchored", "sigma_late 0.14 fixed", ModelConfigDict(sigma_late=0.14)),
    RunSpec("T1c", "T1", "unanchored", "sigma_late 0.21 fixed", ModelConfigDict(sigma_late=0.21)),
    RunSpec("T1d", "T1", "unanchored", "sigma_late est., prior 0.07",
            ModelConfigDict(sigma_late=0.07, estimate_innovation=True), verdict=False),
    RunSpec("T1e", "T1", "unanchored", "sigma_late est., prior 0.21",
            ModelConfigDict(sigma_late=0.21, estimate_innovation=True), verdict=False),
    RunSpec("T2a", "T2", "unanchored", "No WPI (HCOE only)", ModelConfigDict(use_wpi=False)),
    RunSpec("T2b", "T2", "unanchored", "No HCOE (WPI only)", ModelConfigDict(use_hcoe=False)),
    RunSpec("T2c", "T2", "unanchored", "No wages", ModelConfigDict(use_wpi=False, use_hcoe=False)),
    RunSpec("T4a", "T4", "unanchored", "alpha prior sd 0.25", ModelConfigDict(alpha_prior_sigma=0.25)),
    RunSpec("T4b", "T4", "unanchored", "alpha prior sd 1.0", ModelConfigDict(alpha_prior_sigma=1.0)),
]
RUNS_BY_NAME = {spec.name: spec for spec in RUNS}
BASELINE = "T0_unanchored"
ALL_TESTS = ["T0", "T1", "T2", "T3", "T4", "T5", "T6"]


# --- Data snapshot ---


@dataclass
class Snapshot:
    """Every external series the tests read, frozen at first estimation.

    Saved so --analyse-only rebuilds the report from the data the traces saw,
    not from whatever the loaders return on a later day. The WPI productivity
    and tightness series repeat `stage1._add_wpi_obs`, which builds them inline.
    """

    measures: pd.DataFrame
    inflation: pd.Series
    index: pd.PeriodIndex
    underlying_qa_lag: pd.Series
    pie_rbaq: pd.Series
    wpi: pd.Series
    productivity: pd.Series
    tightness: pd.Series


def _snapshot() -> Snapshot:
    """Load the saved snapshot, or build and save it."""
    if DATA_SNAPSHOT.exists():
        with DATA_SNAPSHOT.open("rb") as f:
            loaded = pickle.load(f)
        if not isinstance(loaded, dict):
            raise TypeError(f"{DATA_SNAPSHOT} does not hold a dict")
        return Snapshot(**loaded)

    measures, inflation, index = load_data(START)
    underlying_qa = annualize((get_trimmed_mean_qrtly().data + get_weighted_median_qrtly().data) / 2)
    if not isinstance(underlying_qa, pd.Series):
        raise TypeError("annualize did not return a Series")
    log_vu = get_log_vu_ratio_qrtly().data
    snapshot = Snapshot(
        measures=measures,
        inflation=inflation,
        index=index,
        underlying_qa_lag=underlying_qa.reindex(index).shift(1),
        pie_rbaq=get_inflation_expectations().data,
        wpi=get_wpi_private_growth_annual().data,
        productivity=get_gdp_per_hour_trend_growth_ex_covid(PRODUCTIVITY_TREND_YEARS).data,
        tightness=(log_vu - log_vu.mean()).rolling(QUARTERS_PER_YEAR).mean(),
    )
    TEST_OUTPUT.mkdir(parents=True, exist_ok=True)
    with DATA_SNAPSHOT.open("wb") as f:
        pickle.dump(snapshot.__dict__, f)  # a plain dict, so the file does not depend on this class
    return snapshot


# --- Estimation ---


@dataclass
class Gate:
    """The convergence gate's inputs and outcome for one run."""

    r_hat: float
    ess_bulk: float
    divergences: int
    target_accept: float
    retried: bool

    @property
    def passed(self) -> bool:
        """Whether the run may be used in a verdict."""
        return (self.r_hat <= GATE_MAX_R_HAT and self.ess_bulk >= GATE_MIN_ESS_BULK
                and self.divergences <= GATE_MAX_DIVERGENCES)


def _gate(trace: xr.DataTree, target_accept: float, *, retried: bool) -> Gate:
    summary = posterior_summary(trace, decimals=None)
    diverging = trace_group(trace, "sample_stats")["diverging"]
    return Gate(
        r_hat=float(np.nanmax(summary["r_hat"].to_numpy())),
        ess_bulk=float(np.nanmin(summary["ess_bulk"].to_numpy())),
        divergences=int(diverging.sum()),
        target_accept=target_accept,
        retried=retried,
    )


def _gate_path(name: str) -> Path:
    return TEST_OUTPUT / f"{name}_gate.json"


def estimate(spec: RunSpec, snapshot: Snapshot, draws: int, tune: int, chains: int) -> None:
    """Sample one run unless its trace is cached; retry once at a higher target_accept."""
    trace_file = TEST_OUTPUT / f"{spec.name}.nc"
    if trace_file.exists():
        print(f"{spec.name}: cached, skipping")
        return

    config = MODEL_CONFIGS[spec.model_type].copy()
    config.update(spec.overrides)
    first_accept = MARKET_TARGET_ACCEPT if spec.model_type == "market" else TARGET_ACCEPT
    if spec.production_budget:
        draws, tune = DEFAULT_DRAWS, DEFAULT_TUNE

    gate: Gate | None = None
    trace: xr.DataTree | None = None
    for attempt, target_accept in enumerate((first_accept, RETRY_TARGET_ACCEPT)):
        print(f"{spec.name}: sampling ({spec.label}), target_accept={target_accept}")
        model = _build_pymc_model(snapshot.measures, snapshot.inflation, snapshot.index, config)
        with model:
            sampled = pm.sample(draws=draws, tune=tune, chains=chains, cores=chains, nuts_sampler="numpyro",
                                target_accept=target_accept, random_seed=SEED, progressbar=False)
        if not isinstance(sampled, xr.DataTree):
            raise TypeError(f"pm.sample returned {type(sampled).__name__}")
        trace = sampled
        gate = _gate(trace, target_accept, retried=attempt > 0)
        print(f"  r_hat {gate.r_hat:.3f}, ess_bulk {gate.ess_bulk:.0f}, divergences {gate.divergences}"
              f" -> {'PASS' if gate.passed else 'FAIL'}")
        if gate.passed:
            break

    if trace is None or gate is None:
        raise RuntimeError(f"{spec.name}: no trace produced")
    trace.to_netcdf(trace_file)
    with _gate_path(spec.name).open("w", encoding="utf-8") as f:
        json.dump({**gate.__dict__, "draws": draws, "tune": tune, "chains": chains}, f, indent=2)


# --- Loaded runs ---


@dataclass
class Run:
    """A cached run with its pi_exp draws and gate."""

    spec: RunSpec
    trace: xr.DataTree
    gate: Gate
    draws: pd.DataFrame

    @property
    def path(self) -> pd.Series:
        """Posterior median of pi_exp."""
        return self.draws.median(axis=1)

    @property
    def lower(self) -> pd.Series:
        """5th percentile of pi_exp."""
        return self.draws.quantile(BAND_LOWER, axis=1)

    @property
    def upper(self) -> pd.Series:
        """95th percentile of pi_exp."""
        return self.draws.quantile(BAND_UPPER, axis=1)

    def param(self, name: str) -> np.ndarray:
        """Draws of a parameter: (samples,) for a scalar, (samples, k) for a vector."""
        values = trace_group(self.trace, "posterior")[name]
        stacked = values.stack(sample=("chain", "draw"))
        array = np.asarray(stacked.values)
        return array if array.ndim == 1 else array.T

    def has(self, name: str) -> bool:
        """Whether the posterior holds this variable."""
        return name in trace_group(self.trace, "posterior").data_vars


def load_runs(index: pd.PeriodIndex) -> dict[str, Run]:
    """Load every cached run."""
    runs: dict[str, Run] = {}
    for spec in RUNS:
        trace_file = TEST_OUTPUT / f"{spec.name}.nc"
        if not trace_file.exists():
            continue
        trace = az.from_netcdf(str(trace_file))
        if not isinstance(trace, xr.DataTree):
            raise TypeError(f"{trace_file} did not load as a DataTree")
        with _gate_path(spec.name).open(encoding="utf-8") as f:
            saved = json.load(f)
        gate = Gate(r_hat=saved["r_hat"], ess_bulk=saved["ess_bulk"], divergences=saved["divergences"],
                    target_accept=saved["target_accept"], retried=saved["retried"])
        draws = vector_draws(trace_group(trace, "posterior")["pi_exp"], index)
        runs[spec.name] = Run(spec=spec, trace=trace, gate=gate, draws=draws)
    return runs


# --- Formatting ---


def _interval(draws: np.ndarray) -> str:
    lo, med, hi = np.quantile(draws, [BAND_LOWER, 0.5, BAND_UPPER])
    return f"{med:.2f} [{lo:.2f}, {hi:.2f}]"


def _table(headers: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join(lines)


def _window(series: pd.Series, lo: str | None, hi: str | None) -> pd.Series:
    start = pd.Period(lo, freq="Q") if lo else series.index[0]
    end = pd.Period(hi, freq="Q") if hi else series.index[-1]
    return series[(series.index >= start) & (series.index <= end)]


def verdict(delta: float) -> str:
    """Place a change in the endpoint median on the spec's scale."""
    size = abs(delta)
    if size < ROBUST_BELOW:
        return "robust"
    if size <= SENSITIVE_UPTO:
        return "sensitive"
    return "assumption-dependent"


def _run_verdict(run: Run, baseline: Run) -> str:
    if not run.gate.passed:
        return "excluded (failed gate)"
    return verdict(run.path.iloc[-1] - baseline.path.iloc[-1])


# --- Summary rows ---


SUMMARY_HEADERS = [
    "Run", "Label", "Endpoint [band]", "Recent mean", "Low mean", "Peak (date)",
    f"Max abs diff from {COMPARE_FROM} (date)", "Mean band width", "Max r_hat", "Min ESS", "Div.", "Gate",
]


def summary_row(run: Run, baseline: Run | None) -> list[str]:
    """Return the spec's summary row for one run."""
    path = run.path
    end = path.index[-1]
    peak = _window(path, *PEAK_WINDOW)
    width = _window(run.upper - run.lower, COMPARE_FROM, None).mean()
    if baseline is None or run.spec.test == "T0":
        diff_text = "n/a"
    else:
        diff = _window(path - baseline.path, COMPARE_FROM, None).abs()
        diff_text = f"{diff.max():.2f} ({diff.idxmax()})"
    gate = run.gate
    gate_text = ("pass" if gate.passed else "FAIL") + (f" (retry {gate.target_accept})" if gate.retried else "")
    return [
        run.spec.name, run.spec.label,
        f"{path.iloc[-1]:.2f} [{run.lower.iloc[-1]:.2f}, {run.upper.iloc[-1]:.2f}] ({end})",
        f"{path.iloc[-RECENT_QUARTERS:].mean():.2f}",
        f"{_window(path, *LOW_WINDOW).mean():.2f}",
        f"{peak.max():.2f} ({peak.idxmax()})",
        diff_text, f"{width:.2f}", f"{gate.r_hat:.3f}", f"{gate.ess_bulk:.0f}", str(gate.divergences), gate_text,
    ]


# --- T0 ---


def t0(runs: dict[str, Run], out: list[str], headline: list[list[str]]) -> None:
    """Baseline and reproduction against the production file."""
    out.append("## T0. Baseline and reproduction\n")
    baseline = runs.get(BASELINE)
    if baseline is None:
        out.append("T0 unanchored was not run.\n")
        return
    if not PRODUCTION_PATH.exists():
        out.append(f"Production file `{PRODUCTION_PATH.relative_to(ROOT)}` is missing; reproduction skipped.\n")
        headline.append(["T0", "Production file missing", "n/a"])
    else:
        production = pd.read_parquet(PRODUCTION_PATH)["median"]
        common = production.index.intersection(baseline.path.index)
        diff = (baseline.path.loc[common] - production.loc[common]).abs()
        end_diff = baseline.path.loc[common[-1]] - production.loc[common[-1]]
        out.append(
            f"Production file sample: {production.index[0]} to {production.index[-1]}; T0 sample: "
            f"{baseline.path.index[0]} to {baseline.path.index[-1]}."
            + (" The production file ends earlier than T0." if production.index[-1] < baseline.path.index[-1]
               else "") + "\n",
        )
        out.append(_table(["Statistic", "Value"], [
            ["Max abs difference over common sample", f"{diff.max():.2f} ({diff.idxmax()})"],
            [f"Difference at common endpoint {common[-1]} (T0 less production)", f"{end_diff:.2f}"],
            [f"Under {REPRODUCE_TOLERANCE} pp?", "yes" if diff.max() < REPRODUCE_TOLERANCE else "NO"],
        ]) + "\n")
        check = runs.get("T0_prodbudget")
        if check is not None:
            budget = (baseline.path - check.path).abs()
            data = (check.path.loc[common] - production.loc[common]).abs()
            out.append(
                f"Attribution, from `T0_prodbudget` (today's data, production budget {DEFAULT_DRAWS} draws / "
                f"{DEFAULT_TUNE} tune): sampler budget, T0 against it, max {budget.max():.2f} "
                f"({budget.idxmax()}); data, it against the production file, max {data.max():.2f} "
                f"({data.idxmax()}), mean {(check.path.loc[common] - production.loc[common]).mean():+.2f}. "
                "The data side is revisions to inputs since the production run plus the extra quarter, which "
                "the model smooths back through the sample.\n",
            )
        headline.append(["T0", f"Max abs(T0 - production) {diff.max():.2f}",
                         "reproduces" if diff.max() < REPRODUCE_TOLERANCE
                         else f"does not reproduce within {REPRODUCE_TOLERANCE}"])
    rows = [summary_row(runs[name], None) for name in ("T0_unanchored", "T0_short", "T0_market", "T0_prodbudget")
            if name in runs]
    out.append(_table(SUMMARY_HEADERS, rows) + "\n")


# --- T1 ---


def _halfnormal_interval(scale: float) -> tuple[float, float, float]:
    """Quantiles of HalfNormal(scale): the q-quantile is scale * Phi^-1((1 + q) / 2)."""
    lo, med, hi = (scale * NormalDist().inv_cdf((1 + q) / 2) for q in (BAND_LOWER, 0.5, BAND_UPPER))
    return lo, med, hi


def _contraction(posterior: np.ndarray, prior_var: float) -> float:
    """1 - posterior variance / prior variance."""
    return 1 - float(np.var(posterior)) / prior_var


def t1(runs: dict[str, Run], out: list[str], headline: list[list[str]]) -> None:
    """Step size of the trend."""
    out.append("## T1. Step size of the trend (`sigma_late`)\n")
    baseline = runs.get(BASELINE)
    names = [s.name for s in RUNS if s.test == "T1" and s.name in runs]
    if baseline is None or not names:
        out.append("Not run.\n")
        return
    rows = [[*summary_row(runs[n], baseline), _run_verdict(runs[n], baseline) if runs[n].spec.verdict else "-"]
            for n in names]
    out.append(_table([*SUMMARY_HEADERS, "Verdict"], rows) + "\n")
    for n in names:
        if runs[n].spec.verdict:
            delta = runs[n].path.iloc[-1] - baseline.path.iloc[-1]
            headline.append([f"T1 {n} ({runs[n].spec.label})", f"endpoint change {delta:+.2f}",
                             _run_verdict(runs[n], baseline)])

    # Estimated step size
    est_rows = []
    medians = {}
    for n in ("T1d", "T1e"):
        if n not in runs or not runs[n].has("sigma_late"):
            continue
        scale = runs[n].spec.overrides.get("sigma_late", 0.0)
        post = runs[n].param("sigma_late")
        p_lo, p_med, p_hi = _halfnormal_interval(scale)
        prior_var = scale**2 * (1 - 2 / np.pi)
        medians[n] = float(np.median(post))
        post_text = (f"{np.median(post):.3f} [{np.quantile(post, BAND_LOWER):.3f}, "
                     f"{np.quantile(post, BAND_UPPER):.3f}]")
        est_rows.append([n, f"HalfNormal({scale})", f"{p_med:.3f} [{p_lo:.3f}, {p_hi:.3f}]", post_text,
                         f"{_contraction(post, prior_var):.2f}"])
    if est_rows:
        out.append("Estimated `sigma_late` (median [90% interval]); contraction is 1 - posterior var / prior var.\n")
        out.append(_table(["Run", "Prior", "Prior", "Posterior", "Contraction"], est_rows) + "\n")
    if {"T1d", "T1e"} <= medians.keys():
        shift = abs(medians["T1e"] - medians["T1d"]) / min(medians.values())
        material = shift > MATERIAL_SIGMA_SHIFT
        out.append(
            f"Posterior medians differ by {shift:.0%} of the smaller (harness threshold for 'material': "
            f"{MATERIAL_SIGMA_SHIFT:.0%}). "
            + ("**The two priors give materially different posteriors: the data do not identify the step size.**"
               if material else "The two priors give similar posteriors by that threshold.") + "\n",
        )
        headline.append(["T1 d/e (estimated step)",
                         f"sigma_late medians {medians['T1d']:.3f} vs {medians['T1e']:.3f}",
                         "not identified" if material else "prior-robust"])

    # Gap persistence
    gap_rows = [[n, _interval(runs[n].param("b_inflation")), _interval(runs[n].param("sigma_inflation"))]
                for n in [BASELINE, *names] if runs[n].has("b_inflation")]
    out.append("Inflation-gap parameters (median [90% interval]).\n")
    out.append(_table(["Run", "b_inflation", "sigma_inflation"], gap_rows) + "\n")

    title = "Expectations robustness T1 step size"
    lines = pd.DataFrame({f"{n} {runs[n].spec.label}": runs[n].path for n in [BASELINE, *names]})
    band = pd.DataFrame({"lower": baseline.lower, "upper": baseline.upper})
    start = pd.Period("2010Q1", freq="Q")
    ax = mg.fill_between_plot(band[band.index >= start], color="lightgrey", alpha=0.6, label="Baseline 90% band")
    ax = mg.line_plot(lines[lines.index >= start], ax=ax, width=1.5, annotate=False)
    mg.finalise_plot(ax, title=title, lfooter=LFOOTER, rfooter=RFOOTER, show=False, **PLOT_KWARGS)
    out.append(f"![T1]({sanitize_filename(title)}.png)\n")


# --- T2 ---


def _wpi_gap(run: Run, snapshot: Snapshot) -> pd.Series:
    """Posterior median of observed private WPI less its fitted level, on the fitted quarters."""
    index = run.draws.index
    wpi = snapshot.wpi.reindex(index)
    prod = snapshot.productivity.reindex(index)
    tight = snapshot.tightness.reindex(index)
    covid = (index >= pd.Period(WPI_EXCLUDE_START, freq="Q")) & (index <= pd.Period(WPI_EXCLUDE_END, freq="Q"))
    mask = wpi.notna() & prod.notna() & tight.notna() & ~covid
    adj = run.param("wpi_adjustment")
    beta = run.param("beta_wpi_tightness")
    pi = run.draws.loc[mask].to_numpy()
    level = pi + prod[mask].to_numpy()[:, None] + adj[None, :] + tight[mask].to_numpy()[:, None] * beta[None, :]
    gap = wpi[mask].to_numpy()[:, None] - level
    return pd.Series(np.median(gap, axis=1), index=index[mask.to_numpy()], name="WPI gap")


def t2(runs: dict[str, Run], snapshot: Snapshot, out: list[str], headline: list[list[str]]) -> None:
    """Wages as measurements."""
    out.append("## T2. Wages as measurements of expectations\n")
    baseline = runs.get(BASELINE)
    names = [s.name for s in RUNS if s.test == "T2" and s.name in runs]
    if baseline is None or not names:
        out.append("Not run.\n")
        return
    rows = []
    for n in names:
        run = runs[n]
        pre = _window(run.path - baseline.path, *PRE_TARGETING).abs()
        rows.append([*summary_row(run, baseline), f"{pre.max():.2f} ({pre.idxmax()})", _run_verdict(run, baseline)])
        delta = run.path.iloc[-1] - baseline.path.iloc[-1]
        headline.append([f"T2 {n} ({run.spec.label})", f"endpoint change {delta:+.2f}",
                         _run_verdict(run, baseline)])
    out.append(_table([*SUMMARY_HEADERS, "Max abs diff 1983-92 (date)", "Verdict"], rows) + "\n")

    surveys = MODEL_CONFIGS["unanchored"].get("survey_series", [])
    shift_rows = []
    for param in ("alpha", "lambda_bias"):
        base_med = np.median(baseline.param(param), axis=0)
        for i, survey in enumerate(surveys):
            shifts = [f"{np.median(runs[n].param(param), axis=0)[i] - base_med[i]:+.2f}" for n in names]
            shift_rows.append([param, survey, f"{base_med[i]:.2f}", *shifts])
    out.append("Survey parameters: baseline posterior median, and each variant's shift from it.\n")
    out.append(_table(["Param", "Survey", "Baseline", *[f"{n} shift" for n in names]], shift_rows) + "\n")

    wage_rows = [[p, _interval(baseline.param(p))]
                 for p in ("wpi_adjustment", "beta_wpi_tightness", "b_wpi", "sigma_wpi") if baseline.has(p)]
    out.append("Baseline WPI parameters (median [90% interval]).\n")
    out.append(_table(["Parameter", "Posterior"], wage_rows) + "\n")
    gap = _wpi_gap(baseline, snapshot)
    gap.to_frame().to_csv(TEST_OUTPUT / "baseline_wpi_gap.csv")
    pre_mean = _window(gap, *WPI_GAP_PRE).mean()
    post_mean = _window(gap, *WPI_GAP_POST).mean()
    out.append(_table(["Fitted WPI gap (observed less level), posterior median", "Mean"], [
        [f"{WPI_GAP_PRE[0]} to {WPI_GAP_PRE[1]}", f"{pre_mean:.2f}"],
        [f"{WPI_GAP_POST[0]} onward", f"{post_mean:.2f}"],
    ]) + "\n")
    out.append("Full series: `output/expectations_tests/baseline_wpi_gap.csv`.\n")
    gap_title = "Expectations robustness T2 baseline WPI gap"
    ax = mg.line_plot(gap.reindex(pd.period_range(gap.index[0], gap.index[-1], freq="Q")), width=1.5,
                      annotate=False, dropna=False)
    mg.finalise_plot(ax, title=gap_title, ylabel="Percentage points", y0=True, lfooter=LFOOTER
                     + f"Fitted quarters only; {WPI_EXCLUDE_START}-{WPI_EXCLUDE_END} excluded. ",
                     rfooter=RFOOTER, show=False)
    out.append(f"![WPI gap]({sanitize_filename(gap_title)}.png)\n")

    title = "Expectations robustness T2 wages"
    lines = pd.DataFrame({f"{n} {runs[n].spec.label}": runs[n].path for n in [BASELINE, *names]})
    ax = mg.line_plot(lines[lines.index >= pd.Period(COMPARE_FROM, freq="Q")], width=1.5, annotate=False)
    mg.finalise_plot(ax, title=title, lfooter=LFOOTER, rfooter=RFOOTER, show=False, **PLOT_KWARGS)
    out.append(f"![T2]({sanitize_filename(title)}.png)\n")

    if "T2c" in runs:
        nowages = runs["T2c"]
        pd.DataFrame({"lower": nowages.lower, "median": nowages.path, "upper": nowages.upper}).to_parquet(
            NOWAGES_PATH)
        out.append(f"T2c path saved to `{NOWAGES_PATH.relative_to(ROOT)}`.\n")


# --- T3 ---


def _compare(bench: pd.Series, path: pd.Series) -> list[str]:
    common = bench.dropna().index.intersection(path.dropna().index)
    b, p = bench.loc[common], path.loc[common]
    diff = p - b
    return [
        f"{common[0]} to {common[-1]}",
        f"{np.corrcoef(b, p)[0, 1]:.2f}",
        f"{b.diff(QUARTERS_PER_YEAR).corr(p.diff(QUARTERS_PER_YEAR)):.2f}",
        f"{np.sqrt((diff**2).mean()):.2f}",
        f"{diff.mean():.2f}",
        f"{diff.iloc[-1]:.2f} ({common[-1]})",
    ]


def _composite(measures: pd.DataFrame, inflation: pd.Series) -> pd.Series:
    """T3a: each measure net of lagged year-ended underlying, averaged, re-levelled."""
    lag = inflation.shift(1)
    start = pd.Period(BENCH_FROM, freq="Q")
    adjusted = {}
    for col in BENCH_MEASURES:
        y = measures[col]
        frame = pd.DataFrame({"y": y, "x": lag})
        frame = frame[(frame.index >= start)].dropna()
        fit = sm.OLS(frame["y"], sm.add_constant(frame["x"])).fit()
        adjusted[col] = (y - fit.params["x"] * lag)[y.index >= start]
    composite = pd.DataFrame(adjusted).mean(axis=1, skipna=True)
    target_mean = inflation[inflation.index >= start].mean()
    return composite + (target_mean - composite.mean())


def _pc1(measures: pd.DataFrame, path: pd.Series) -> tuple[pd.Series, pd.Series, float]:
    """T3b: first principal component of the four measures, on the T0 path's scale."""
    data = measures[BENCH_MEASURES].dropna()
    z = (data - data.mean()) / data.std()
    _, s, vt = np.linalg.svd(z.to_numpy(), full_matrices=False)
    loadings = vt[0] * np.sign(vt[0].sum())
    share = float(s[0] ** 2 / (s**2).sum())
    pc = pd.Series(z.to_numpy() @ loadings, index=z.index)
    target = path.loc[z.index]
    rescaled = (pc - pc.mean()) / pc.std() * target.std() + target.mean()
    return rescaled, pd.Series(loadings, index=BENCH_MEASURES), share


def t3(runs: dict[str, Run], snapshot: Snapshot, out: list[str], headline: list[list[str]]) -> None:
    """External benchmarks."""
    out.append("## T3. External benchmarks\n")
    baseline = runs.get(BASELINE)
    if baseline is None:
        out.append("Not run.\n")
        return
    measures = snapshot.measures
    inflation = snapshot.inflation
    path = baseline.path

    composite = _composite(measures, inflation)
    pc, loadings, share = _pc1(measures, path)
    pie = _window(snapshot.pie_rbaq, *PIE_WINDOW)
    rows = [["T3a composite", *_compare(composite, path)], ["T3b PC1", *_compare(pc, path)],
            ["T3c PIE_RBAQ", *_compare(pie, path)]]
    out.append("Differences are T0 less the benchmark. Changes are four-quarter changes.\n")
    out.append(_table(["Benchmark", "Sample", "Corr levels", "Corr 4q changes", "RMSE", "Mean diff",
                       "Diff at end (date)"], rows) + "\n")
    out.append(f"T3b loadings: {', '.join(f'{k} {v:.2f}' for k, v in loadings.items())}; "
               f"share of variance explained {share:.2f}.\n")
    headline.extend([row[0], f"corr {row[2]}, RMSE {row[4]}, end diff {row[6]}", "-"] for row in rows)

    since = _window(path, BENCH_FROM, None)
    outside = since[(since < TARGET_BAND[0]) | (since > TARGET_BAND[1])]
    out.append(f"T3d: {len(outside)} of {len(since)} quarters from {BENCH_FROM} sit outside "
               f"{TARGET_BAND[0]:.0f} to {TARGET_BAND[1]:.0f} per cent.\n")
    if len(outside):
        out.append(_table(["Quarter", "T0 median"], [[str(q), f"{v:.2f}"] for q, v in outside.items()]) + "\n")
    headline.append(["T3d target band", f"{len(outside)} of {len(since)} quarters outside 2-3", "-"])

    title = "Expectations robustness T3 benchmarks"
    start = pd.Period(BENCH_FROM, freq="Q")
    lines = pd.DataFrame({"T0 median": path, "T3a composite": composite, "T3b PC1 (rescaled)": pc,
                          "PIE_RBAQ": pie.reindex(path.index)})
    band = pd.DataFrame({"lower": baseline.lower, "upper": baseline.upper})
    ax = mg.fill_between_plot(band[band.index >= start], color="lightgrey", alpha=0.6, label="T0 90% band")
    ax = mg.line_plot(lines[lines.index >= start], ax=ax, width=1.5, annotate=False, dropna=True)
    mg.finalise_plot(ax, title=title, lfooter=LFOOTER, rfooter=RFOOTER, show=False, **PLOT_KWARGS)
    out.append(f"![T3]({sanitize_filename(title)}.png)\n")


# --- T4 ---


def t4(runs: dict[str, Run], snapshot: Snapshot, out: list[str], headline: list[list[str]]) -> None:
    """Where the level comes from."""
    out.append("## T4. Where the level comes from\n")
    baseline = runs.get(BASELINE)
    if baseline is None:
        out.append("Not run.\n")
        return
    gap = snapshot.underlying_qa_lag - baseline.path
    rows = []
    for lo, hi in LEVEL_WINDOWS:
        piece = _window(gap.dropna(), lo, hi)
        rows.append([f"{piece.index[0]} to {piece.index[-1]}", f"{piece.mean():.2f}"])
    out.append("Lagged quarterly annualised underlying inflation less the T0 path.\n")
    out.append(_table(["Window", "Mean"], rows) + "\n")
    headline.append(["T4 level normalisation", f"mean (inflation - path) from {BENCH_FROM}: {rows[0][1]}", "-"])

    surveys = MODEL_CONFIGS["unanchored"].get("survey_series", [])
    alpha_rows = []
    for name in [BASELINE, "T4a", "T4b"]:
        if name not in runs:
            continue
        sd = runs[name].spec.overrides.get("alpha_prior_sigma", ALPHA_PRIOR_SIGMA)
        z = NormalDist().inv_cdf(BAND_UPPER)
        draws = runs[name].param("alpha")
        for i, survey in enumerate(surveys):
            alpha_rows.append([name, survey, f"0.00 [{-z * sd:.2f}, {z * sd:.2f}]", _interval(draws[:, i]),
                               f"{_contraction(draws[:, i], sd**2):.2f}"])
    out.append("Survey `alpha`: prior and posterior (median [90% interval]); contraction is 1 - var ratio.\n")
    out.append(_table(["Run", "Survey", "Prior", "Posterior", "Contraction"], alpha_rows) + "\n")

    names = [n for n in ("T4a", "T4b") if n in runs]
    if names:
        rows = [[*summary_row(runs[n], baseline), _run_verdict(runs[n], baseline)] for n in names]
        out.append(_table([*SUMMARY_HEADERS, "Verdict"], rows) + "\n")
        for n in names:
            delta = runs[n].path.iloc[-1] - baseline.path.iloc[-1]
            headline.append([f"T4 {n} ({runs[n].spec.label})", f"endpoint change {delta:+.2f}",
                             _run_verdict(runs[n], baseline)])


# --- T5 ---


def _label_check(out: list[str]) -> None:
    header = pd.read_excel(G3_URL, sheet_name="Data", header=None, nrows=G3_HEADER_ROWS, index_col=0)
    ids, titles, descriptions = (header.loc[row] for row in ("Series ID", "Title", "Description"))
    if not (isinstance(ids, pd.Series) and isinstance(titles, pd.Series) and isinstance(descriptions, pd.Series)):
        raise TypeError("G3 header rows are not unique")
    rows = []
    for name, code in G3_SERIES.items():
        hit = ids[ids == code]
        if hit.empty:
            rows.append([name, code, "not found in G3", ""])
            continue
        col = hit.index[0]
        rows.append([name, code, str(titles[col]).strip(), str(descriptions[col]).strip()])
    out.append("G3 series pulled by `src/data/expectations.py`, with the RBA's own title and description.\n")
    out.append(_table(["Name in code", "Series ID", "RBA title", "RBA description"], rows) + "\n")

    files = [ROOT / "src/data/expectations.py", ROOT / "src/models/expectations/MODEL_NOTES.md"]
    hits = [
        [f"`{path.relative_to(ROOT)}:{number}`", line.strip().replace("|", "\\|")]
        for path in files
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1)
        if "market_yoy" in line.lower() or "gmarexpyy" in line.lower()
    ]
    out.append("Every line that mentions `market_yoy` or GMAREXPYY:\n")
    out.append(_table(["Location", "Text"], hits) + "\n" if hits else "None found.\n")


def t5(runs: dict[str, Run], snapshot: Snapshot, out: list[str], headline: list[list[str]]) -> None:
    """Horizons and the bond measure."""
    out.append("## T5. Horizons and the bond measure\n")
    names = ["T0_unanchored", "T0_short", "T0_market"]
    if not all(n in runs for n in names):
        out.append("Needs all three T0 runs.\n")
        return
    base = runs[BASELINE]
    out.append(_table(["Run", "Endpoint [band]"], [
        [n, f"{runs[n].path.iloc[-1]:.2f} [{runs[n].lower.iloc[-1]:.2f}, {runs[n].upper.iloc[-1]:.2f}]"]
        for n in names]) + "\n")
    spreads = pd.DataFrame({"short less unanchored": runs["T0_short"].path - base.path,
                            "market less unanchored": runs["T0_market"].path - base.path})
    title = "Expectations robustness T5 horizon spreads"
    ax = mg.line_plot(spreads[spreads.index >= pd.Period(COMPARE_FROM, freq="Q")], width=1.5, annotate=True,
                      rounding=2)
    mg.finalise_plot(ax, title=title, ylabel="Percentage points", y0=True, lfooter=LFOOTER, rfooter=RFOOTER,
                     legend={"loc": "best", "fontsize": "x-small"}, show=False)
    out.append(f"![T5]({sanitize_filename(title)}.png)\n")

    surveys = MODEL_CONFIGS["unanchored"].get("survey_series", [])
    i = surveys.index("breakeven")
    out.append(_table(["Parameter", "Posterior (median [90% interval])"], [
        ["alpha breakeven", _interval(base.param("alpha")[:, i])],
        ["sigma_obs breakeven", _interval(base.param("sigma_obs")[:, i])],
    ]) + "\n")
    gap = (snapshot.measures["breakeven"] - base.path).dropna()
    rows, means = [], []
    for lo, hi in BREAKEVEN_REGIMES:
        piece = _window(gap, lo, hi)
        means.append(piece.mean())
        rows.append([f"{piece.index[0]} to {piece.index[-1]}", f"{piece.mean():.2f}"])
    spread = max(means) - min(means)
    out.append("Observed breakeven less the T0 unanchored path.\n")
    out.append(_table(["Regime", "Mean"], rows) + "\n")
    out.append(f"Range across regimes {spread:.2f} pp; "
               + ("**above** " if spread > PREMIUM_SPREAD_LIMIT else "within ")
               + f"the {PREMIUM_SPREAD_LIMIT:.2f} pp limit for a constant premium.\n")
    headline.append(["T5 breakeven premium", f"regime range {spread:.2f} pp",
                     "constant premium not tenable" if spread > PREMIUM_SPREAD_LIMIT else "constant premium ok"])
    _label_check(out)


# --- T6 ---


def _wage_regression(expectations: pd.Series, ugap: pd.Series, wpi: pd.Series) -> list[str]:
    frame = pd.DataFrame({"wpi": wpi, "ugap": ugap, "exp": expectations}).dropna()
    fit = sm.OLS(frame["wpi"], sm.add_constant(frame[["ugap", "exp"]])).fit(
        cov_type="HAC", cov_kwds={"maxlags": WAGE_HAC_LAGS})
    params, se = fit.params, fit.bse
    nairu_wage = params["const"] + params["exp"] * INFLATION_TARGET
    return [f"{frame.index[0]} to {frame.index[-1]} (n={len(frame)})",
            f"{params['ugap']:.2f} ({se['ugap']:.2f})", f"{params['exp']:.2f} ({se['exp']:.2f})",
            f"{params['const']:.2f} ({se['const']:.2f})", f"{fit.rsquared:.2f}", f"{nairu_wage:.2f}"]


def t6(snapshot: Snapshot, out: list[str], headline: list[list[str]]) -> None:
    """Rerun the wage check in ystar_ustar with two expectations series."""
    out.append("## T6. The wage check in `ystar_ustar`\n")
    out.append(
        "**No wage check code exists in `src/models/ystar_ustar/`, nor anywhere in `src/`.** It is described "
        "in `src/models/ystar_ustar/MODEL_NOTES.md` (section 'The wage check'), and an earlier version of "
        "those notes gave its specification: private WPI growth, year-ended, regressed on `u - u*` and "
        f"inflation expectations, HAC({WAGE_HAC_LAGS}), n = 112, with NAIRU-consistent wage growth taken "
        f"at `u = u*` and expectations at target ({INFLATION_TARGET}). The regression below is a "
        "**reconstruction from that description**, not a re-run of the original code. u - u* is the "
        "posterior median from the saved `model_outputs/ystar_ustar_trace.nc`, which is not re-estimated.\n",
    )
    try:
        joint = load_joint_results()
    except FileNotFoundError:
        out.append("No saved `ystar_ustar` results; skipped.\n")
        return
    if not NOWAGES_PATH.exists():
        out.append("T2c path not saved yet; skipped.\n")
        return
    ugap = joint.ugap_median()
    wpi = snapshot.wpi
    production = get_model_expectations_unanchored().data
    nowages = pd.read_parquet(NOWAGES_PATH)["median"]
    headers = ["Expectations series", "Sample", "Gap slope (se)", "Pass-through (se)", "Constant (se)", "R2",
               "NAIRU-consistent wage growth"]
    rows = [["Production unanchored", *_wage_regression(production, ugap, wpi)],
            ["T2c no wages", *_wage_regression(nowages, ugap, wpi)]]
    out.append(_table(headers, rows) + "\n")
    out.append("Standard errors are HAC. NAIRU-consistent wage growth = constant + pass-through x "
               f"{INFLATION_TARGET}.\n")
    pass_through, nairu_wage = headers.index("Pass-through (se)"), headers.index("NAIRU-consistent wage growth")
    summary = (f"pass-through {rows[0][pass_through]} prod vs {rows[1][pass_through]} no-wages; NAIRU wage "
               f"{rows[0][nairu_wage]} vs {rows[1][nairu_wage]}")
    headline.append(["T6 wage check (reconstructed)", summary, "-"])


# --- Report ---

EDITED_FILES = """## 4. Source files edited

- `src/models/expectations/stage1.py`: one hook for T4. The survey `alpha` prior scale was a literal
  0.5 in `_add_survey_obs`; it is now the constant `ALPHA_PRIOR_SIGMA = 0.5`, overridable by a new
  optional `ModelConfigDict` key `alpha_prior_sigma`. Production configs do not set the key, so
  production behaviour is unchanged.
- `src/models/expectations/robustness.py`: new, this harness.

No other source file, default or MODEL_NOTES was changed.
"""

OUT_OF_SCOPE = """## 5. Looked wrong, out of scope (not fixed)

- **`market_yoy` is mislabelled.** The RBA's G3 header gives GMAREXPYY as market economists'
  expectations two years ahead, year-ended. `src/data/expectations.py` (the `G3_SERIES` comment and
  the docstring) and `MODEL_NOTES.md` (the data-sources table and the "Series Effects" list) call it
  "year-on-year". The same applies to `union_yoy` (GUNIEXPYY, also two years ahead), which no model
  uses. See T5 for the lines.
- **The business series is a three-month-ahead measure** of firms' own final prices, annualised, per
  the RBA's description. `MODEL_NOTES.md` lists it only as "NAB Business Survey inflation
  expectations". Its "firms' own prices" caveat is right; the horizon is not stated.
- **The wage check in `ystar_ustar` has no code.** `MODEL_NOTES.md` describes it as a result, but
  nothing in `src/` computes it and its numbers cannot be regenerated. T6 is a reconstruction from an
  earlier version of the notes, so its agreement with the original is unverified.
- **The wage check is not out of sample with respect to wages** while it reads the production series,
  which is fitted to private WPI. T6 shows how much that matters for this regression.
- **The sample ends in a quarter still in progress.** On this run the last quarter is observed only
  through the November market-economist surveys, the breakeven, and inflation lagged to the previous
  quarter; the business survey is missing for it. The production run of 29 September did the same for
  its last quarter. Every "endpoint" in this report is that quarter.
- **`_build_pymc_model` fetches inflation, headline CPI, bonds, HCOE and WPI live** inside its
  observation functions, so the data snapshot here freezes only what the analysis reads. A re-estimate
  on a later day can see revised data even with the same snapshot.
- **The harness repeats `stage1._add_wpi_obs`'s construction** of productivity, tightness and the
  excluded window to compute the fitted WPI gap, because stage1 builds them inline. If stage1 changes
  that construction, T2's gap series will drift from the model's.
"""


def build_report(tests: list[str]) -> None:
    """Rebuild REPORT.md and the charts from cached traces."""
    snapshot = _snapshot()
    runs = load_runs(snapshot.index)
    TEST_CHARTS.mkdir(parents=True, exist_ok=True)
    mg.set_chart_dir(str(TEST_CHARTS))

    headline: list[list[str]] = []
    sections: list[str] = []
    steps: dict[str, Callable[[], None]] = {
        "T0": lambda: t0(runs, sections, headline),
        "T1": lambda: t1(runs, sections, headline),
        "T2": lambda: t2(runs, snapshot, sections, headline),
        "T3": lambda: t3(runs, snapshot, sections, headline),
        "T4": lambda: t4(runs, snapshot, sections, headline),
        "T5": lambda: t5(runs, snapshot, sections, headline),
        "T6": lambda: t6(snapshot, sections, headline),
    }
    for test in ALL_TESTS:
        if test in tests:
            steps[test]()
            plt.close("all")

    baseline = runs.get(BASELINE)
    all_rows = [summary_row(run, baseline) for run in runs.values()]
    report = [
        "# Inflation expectations model: robustness tests\n",
        (f"Sampler: {TEST_CHAINS} chains x {TEST_DRAWS} draws, {TEST_TUNE} tune unless a run's gate file says "
         f"otherwise; seed {SEED}; quarterly from {START}. Gate: max r_hat <= {GATE_MAX_R_HAT}, min bulk ESS >= "
         f"{GATE_MIN_ESS_BULK}, divergences <= {GATE_MAX_DIVERGENCES}. Verdict on abs(change in endpoint "
         f"median): < {ROBUST_BELOW} robust, {ROBUST_BELOW} to {SENSITIVE_UPTO} sensitive, > {SENSITIVE_UPTO} "
         "assumption-dependent.\n"),
        "## 1. Summary\n",
        _table(["Test", "Headline", "Verdict"], headline) + "\n",
        "## 2. Summary rows, every run\n",
        (f"Band is the 5th to 95th percentile. Low = {LOW_WINDOW[0]}-{LOW_WINDOW[1]}; peak within "
         f"{PEAK_WINDOW[0]}-{PEAK_WINDOW[1]}; recent = last {RECENT_QUARTERS} quarters.\n"),
        _table(SUMMARY_HEADERS, all_rows) + "\n",
        "## 3. Tests\n",
        *sections,
        EDITED_FILES,
        OUT_OF_SCOPE,
    ]
    REPORT_PATH.write_text("\n".join(report), encoding="utf-8")
    print(f"Report: {REPORT_PATH}")


# --- CLI ---


def main() -> None:
    """Estimate the selected tests' runs, then write the report."""
    parser = argparse.ArgumentParser(description="Robustness tests for the expectations model")
    parser.add_argument("--tests", nargs="+", choices=ALL_TESTS, default=ALL_TESTS,
                        help="Tests to run (T0 is always estimated: every test compares against it)")
    parser.add_argument("--draws", type=int, default=TEST_DRAWS)
    parser.add_argument("--tune", type=int, default=TEST_TUNE)
    parser.add_argument("--chains", type=int, default=TEST_CHAINS)
    parser.add_argument("--analyse-only", action="store_true", help="Rebuild the report from saved traces")
    args = parser.parse_args()

    tests = sorted({"T0", *args.tests}, key=ALL_TESTS.index)
    if "T6" in tests and "T2" not in tests:
        tests.insert(tests.index("T6"), "T2")  # T6 reads the T2c path

    if not args.analyse_only:
        snapshot = _snapshot()
        for spec in RUNS:
            if spec.test in tests:
                estimate(spec, snapshot, args.draws, args.tune, args.chains)
    build_report(tests)


if __name__ == "__main__":
    main()
