"""The comparison specifications behind `--compare`, and how to refresh them.

Five specifications of this model, differing in where potential's growth comes
from and in what identifies the gap. All run from 1984Q1 with the phased
anchor. MODEL_NOTES, "Comparing specifications", explains why, what they
observe, and how to read the comparison.

Each specification is the command-line flags it would be run with, parsed by
the same parser as the default run. Each writes to its own `yss84_*` prefix,
so a refresh never touches the default run's outputs or charts. A saved run
counts as current if its trace was written today.

THEY DO NOT OBSERVE THE SAME DATA. Only GDP is common, so the fit column scores
GDP alone, over the quarters every specification actually fitted:
`gdp_quarters` derives each run's quarters from what the run recorded, because
lining up the wrong quarters under one column would be invisible.
"""

import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import arviz as az
import numpy as np
import pandas as pd

from src.models.ystar.cli import build_parser, run_from_args
from src.models.ystar.results import load_results
from src.paths import MODEL_OUTPUTS

OUTPUT_DIR = MODEL_OUTPUTS

START = "1984Q1"

# Above this the importance-sampling estimate for an observation is unreliable.
# ArviZ's own threshold.
_PARETO_LIMIT = 0.7


@dataclass(frozen=True)
class Specification:
    """One comparison specification, and how to reproduce it."""

    label: str
    spec: str
    prefix: str
    colour: str
    style: str

    @property
    def trace_path(self) -> Path:
        """Where this specification's saved trace lives."""
        return OUTPUT_DIR / f"{self.prefix}_trace.nc"

    @property
    def flags(self) -> list[str]:
        """The command line this specification is run with."""
        return ["--spec", self.spec, "--start", START, "--anchor-phase", "glide",
                "--prefix", self.prefix, "--no-analyse"]

    def is_current(self) -> bool:
        """Report whether the saved trace was written today."""
        if not self.trace_path.exists():
            return False
        written = datetime.fromtimestamp(self.trace_path.stat().st_mtime).astimezone()
        return written.date() == datetime.now().astimezone().date()

    def refresh(self) -> None:
        """Re-estimate this specification into its own prefix."""
        print(f"  re-running {self.label}", flush=True)
        run_from_args(build_parser().parse_args(self.flags))


# Labelled by what each one asserts, not by its internal name: "core" is the
# base specification and has nothing to do with core inflation. Colour pairs
# the specifications that share a gap: orange and green define it by inflation,
# blue and purple read it off a Phillips curve. Grey does neither.
SPECIFICATIONS: list[Specification] = [
    Specification("Gap = c x (pi - anchor), free trend", "inflation",
                  "yss84_inflation", "darkorange", "-"),
    Specification("Gap = c x (pi - anchor), production function", "production",
                  "yss84_production", "seagreen", "-"),
    Specification("Phillips curve on a cycle, free trend", "core",
                  "yss84_core", "tab:blue", "-"),
    Specification("Phillips curve on a cycle, hours x productivity", "labour",
                  "yss84_labour", "rebeccapurple", "-"),
    Specification("Inflation's sign only, no Phillips slope", "target",
                  "yss84_target", "dimgrey", "--"),
]


def gdp_quarters(results: object, n_rows: int) -> pd.PeriodIndex:
    """Return the quarters the GDP likelihood's rows correspond to.

    The inflation family drops an excluded window from the middle, so its rows
    are the sample under a mask. The AR(2) specifications lose leading lags, so
    theirs are the tail of the sample. Read from what the run recorded rather
    than by counting.
    """
    index = pd.PeriodIndex(getattr(results, "obs_index", None))
    constants = getattr(results, "constants", {})
    window = constants.get("exclude_window") if isinstance(constants, dict) else None
    if window is not None:
        lo, hi = window
        keep = ~((index >= pd.Period(lo, freq="Q")) & (index <= pd.Period(hi, freq="Q")))
        quarters = index[keep]
    else:
        quarters = index[len(index) - n_rows:]
    if len(quarters) != n_rows:
        raise ValueError(
            f"cannot align {n_rows} GDP likelihood rows to a sample of {len(index)} "
            f"quarters (excluded window: {window})",
        )
    return quarters


@dataclass
class Loaded:
    """One specification's loaded series and diagnostics."""

    spec: Specification
    potential: pd.Series
    potential_growth: pd.Series
    output_gap: pd.Series
    elpd_by_quarter: pd.Series
    pareto_bad: int
    max_rhat: float
    min_ess: int
    divergences: int


def load(spec: Specification) -> Loaded:
    """Read one specification's saved run."""
    results = load_results(prefix=spec.prefix)
    trace = results.trace

    loo = az.loo(trace, var_name="observed_gdp", pointwise=True)
    pointwise = np.asarray(loo.loo_i).ravel()
    quarters = gdp_quarters(results, len(pointwise))

    summary = az.summary(trace)
    sample_stats = getattr(trace, "sample_stats", None)
    divergences = int(sample_stats["diverging"].to_numpy().sum()) if sample_stats is not None else 0
    # `actual_output_gap` is log GDP less potential in every specification,
    # unlike `output_gap`, which is the inflation-defined series in two of them
    # and a latent cycle in the other three. Only the former compares.
    return Loaded(
        spec=spec,
        potential=results.potential_posterior().median(axis=1),
        # Derived from the potential level, not the `trend_growth` state, which
        # the labour and production specifications do not have.
        potential_growth=results.potential_growth_posterior().median(axis=1),
        output_gap=results.actual_output_gap_posterior().median(axis=1),
        elpd_by_quarter=pd.Series(pointwise, index=quarters),
        pareto_bad=int((np.asarray(loo.pareto_k) > _PARETO_LIMIT).sum()),
        max_rhat=float(summary["r_hat"].max()),
        min_ess=int(summary["ess_bulk"].min()),
        divergences=divergences,
    )


def common_quarters(loaded: list[Loaded]) -> pd.PeriodIndex:
    """Return the quarters every specification's GDP likelihood covers."""
    shared = pd.PeriodIndex(loaded[0].elpd_by_quarter.index)
    for item in loaded[1:]:
        shared = shared.intersection(pd.PeriodIndex(item.elpd_by_quarter.index))
    return shared


def load_all(*, refresh: bool = True) -> list[Loaded]:
    """Re-estimate anything not from today (if `refresh`), then load every specification."""
    stale = [s for s in SPECIFICATIONS if not s.is_current()]
    if refresh and stale:
        print(f"Re-running {len(stale)} of {len(SPECIFICATIONS)} specifications:")
        for spec in stale:
            spec.refresh()
    elif stale:
        names = ", ".join(s.label for s in stale)
        print(f"Charting saved runs as they stand. Not from today: {names}")

    loaded = []
    for spec in SPECIFICATIONS:
        if not spec.trace_path.exists():
            print(f"  no saved run for {spec.label}; run --compare without --analyse-only",
                  file=sys.stderr)
            continue
        loaded.append(load(spec))
    return loaded
