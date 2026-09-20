"""The five y* specifications this summary compares, and how to refresh them.

NOT FIVE MODELS. Five specifications of one model, differing in where
potential's growth comes from and in what identifies the gap:

    inflation   gap = c x (pi - anchor); potential is a free drift state
    production  gap = c x (pi - anchor); potential from the factor trends
    core        gap is an AR(2) cycle, and a Phillips curve reads it
    labour      as core, plus hours and participation observed separately
    target      gap is an AR(2) cycle, sign-restricted by later inflation

All five run from 1984Q1 with a PHASED ANCHOR: measured expectations before
1993Q1, gliding to the target across 1993Q1-1998Q4. A flat 2.5 over a sample
that opens in 1984 would judge nine years of 8 per cent inflation against a
target that did not exist, and the first two specifications put the anchor
directly into the gap, so it would not be a nuisance there but the answer.

THEY DO NOT OBSERVE THE SAME DATA, and that is the central difficulty in
comparing them:

    inflation   observed_gdp
    production  observed_gdp, observed_gk, observed_gl, observed_gm, observed_a
    core        observed_gdp, observed_pi
    labour      observed_gdp, observed_pi, observed_hours, observed_participation
    target      observed_gdp

Only `observed_gdp` is common, so that is what the fit column scores. Even
there the quarters differ: the inflation family drops the lockdown window from
the GDP likelihood and the AR(2) specifications lose their leading lags, so
the score is restricted to the quarters every specification actually fitted.

WHAT THE FIT COLUMN IS NOT. Scoring GDP alone favours a specification that
spends everything on fitting GDP. `inflation` and `target` observe nothing
else, while `labour` is also answering for hours and participation with the
same trends. Read it as one input, against the sampling gate and the
descriptive columns, not as a ranking.
"""

import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import arviz as az
import numpy as np
import pandas as pd

from src.models.ystar.results import load_results
from src.paths import MODEL_OUTPUTS, ROOT

OUTPUT_DIR = MODEL_OUTPUTS
RUN_SCRIPT = ROOT / "run-ystar.sh"

START = "1984Q1"

# Above this the importance-sampling estimate for an observation is unreliable.
# ArviZ's own threshold.
_PARETO_LIMIT = 0.7


@dataclass(frozen=True)
class SpecSource:
    """One specification, and how to reproduce it."""

    label: str
    spec: str
    prefix: str
    colour: str
    style: str

    @property
    def trace_path(self) -> Path:
        """Where this specification's saved trace lives."""
        return OUTPUT_DIR / f"{self.prefix}_trace.nc"

    def is_current(self) -> bool:
        """Report whether the saved trace was written today."""
        if not self.trace_path.exists():
            return False
        written = datetime.fromtimestamp(self.trace_path.stat().st_mtime).astimezone()
        return written.date() == datetime.now().astimezone().date()

    def refresh(self) -> None:
        """Re-estimate this specification into its own prefix."""
        print(f"  re-running {self.label}", flush=True)
        subprocess.run(  # noqa: S603 — our own script, arguments from this module
            [
                str(RUN_SCRIPT), "--spec", self.spec, "--start", START,
                "--anchor-phase", "glide", "--prefix", self.prefix, "--no-analyse",
            ],
            check=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT,
        )


# Labelled by what each one asserts, not by its internal name. "core" in
# particular reads as core inflation and means nothing of the kind: it is the
# base specification, a drifting random walk with an AR(2) cycle.
#
# Colour pairs the specifications that share a gap: orange and green define it
# by inflation, blue and purple read it off a Phillips curve. Grey is the one
# that does neither.
SOURCES: list[SpecSource] = [
    SpecSource("Gap = c x (pi - anchor), free trend", "inflation",
               "yss84_inflation", "darkorange", "-"),
    SpecSource("Gap = c x (pi - anchor), production function", "production",
               "yss84_production", "seagreen", "-"),
    SpecSource("Phillips curve on a cycle, free trend", "core",
               "yss84_core", "tab:blue", "-"),
    SpecSource("Phillips curve on a cycle, hours x productivity", "labour",
               "yss84_labour", "rebeccapurple", "-"),
    SpecSource("Inflation's sign only, no Phillips slope", "target",
               "yss84_target", "dimgrey", "--"),
]


def gdp_quarters(results: object, n_rows: int) -> pd.PeriodIndex:
    """Return the quarters the GDP likelihood's rows correspond to.

    Two shapes, distinguished by what the run recorded rather than by counting.
    The inflation family drops an excluded window from the middle, so its rows
    are the sample under a mask. The AR(2) specifications lose leading lags, so
    theirs are the tail of the sample. Getting this wrong would line up
    different quarters under one column and the error would be invisible.
    """
    index = pd.PeriodIndex(results.obs_index)
    window = results.constants.get("exclude_window")
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

    source: SpecSource
    potential: pd.Series
    potential_growth: pd.Series
    output_gap: pd.Series
    elpd_by_quarter: pd.Series
    pareto_bad: int
    max_rhat: float
    min_ess: int
    divergences: int


def load(source: SpecSource) -> Loaded:
    """Read one specification's saved run."""
    results = load_results(prefix=source.prefix)
    trace = results.trace

    loo = az.loo(trace, var_name="observed_gdp", pointwise=True)
    pointwise = np.asarray(loo.loo_i).ravel()
    quarters = gdp_quarters(results, len(pointwise))

    summary = az.summary(trace)
    # `actual_output_gap` is log GDP less potential in every specification,
    # unlike `output_gap`, which is the inflation-defined series in two of them
    # and a latent cycle in the other three. Only the former compares.
    return Loaded(
        source=source,
        potential=results.potential_posterior().median(axis=1),
        # Derived from the potential level, not the `trend_growth` state,
        # which the labour and production specifications do not have: they
        # build potential from component trends instead.
        potential_growth=results.potential_growth_posterior().median(axis=1),
        output_gap=results.actual_output_gap_posterior().median(axis=1),
        elpd_by_quarter=pd.Series(pointwise, index=quarters),
        pareto_bad=int((np.asarray(loo.pareto_k) > _PARETO_LIMIT).sum()),
        max_rhat=float(summary["r_hat"].max()),
        min_ess=int(summary["ess_bulk"].min()),
        divergences=int(trace.sample_stats["diverging"].to_numpy().sum()),
    )


def common_quarters(loaded: list[Loaded]) -> pd.PeriodIndex:
    """Return the quarters every specification's GDP likelihood covers."""
    shared = loaded[0].elpd_by_quarter.index
    for item in loaded[1:]:
        shared = shared.intersection(item.elpd_by_quarter.index)
    return shared


def load_all(*, refresh: bool = True) -> list[Loaded]:
    """Refresh anything stale, then load every specification."""
    stale = [s for s in SOURCES if not s.is_current()]
    if refresh and stale:
        print(f"Re-running {len(stale)} of {len(SOURCES)} specifications:")
        for source in stale:
            source.refresh()
    elif stale:
        names = ", ".join(s.label for s in stale)
        print(f"Charting saved runs as they stand. Not from today: {names}")

    loaded = []
    for source in SOURCES:
        if not source.trace_path.exists():
            print(f"  no saved run for {source.label}", file=sys.stderr)
            continue
        loaded.append(load(source))
    return loaded
