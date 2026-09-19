"""The eight joint specifications this summary tests, and how to refresh them.

NOT EIGHT MODELS. Eight settings of one model, a full crossing of the
structure imposed on u* with the definition of the output gap:

                                  inflation-defined gap    gap = y - y*
    u* decays to a level                    x                   x
    u* is a spline, 1 knot                  x                   x
    u* is a spline, 2 knots                 x                   x
    u* is a spline, 3 knots                 x                   x

Down a column is how much the structure matters. Across a row is how much the
gap definition matters. Crossed rather than laddered so the two cannot be
confounded: with one cell missing, a difference between columns could always
be the structure that was only tried on one side.

All eight share a sample, an expectations series and an inflation measure, so
agreement within a column is close to arithmetic. The two columns are
different models of the same data.

NOTHING HERE IS A RECOMMENDATION. It is the wider set to test before settling,
so specifications argued against elsewhere are in it. Decay with the identity
gap is the clearest case: a structure that can only report a fall, paired with
a gap definition under test.

THE SECOND KNOT SITS AT 1996Q1. One knot gives three coefficients after the
natural boundary reduction, enough to decline and then level off; a second
buys the early sample a shape of its own, which is where these specifications
disagree most and where the stars are least identified.

WHY C BOUNDS THE OKUN SLOPE. With the gap defined as y - y*, the Okun
equation sees only `beta x gap`, so `(beta, y*)` and `(-beta, y* reflected
through y)` fit it identically and nothing else in the model breaks the tie:
the Phillips curve runs on the unemployment gap and never touches the output
gap. The mirror is reachable whenever the trend prior is loose enough to get
there. At `ratio_g` 0.05, one chain in four finds it, R-hat goes to 1.53 and
ESS to 7. Bounding beta at zero closes it and the run is clean. That bound is
an assertion that Okun's law has the expected sign, and it is stated here
rather than buried because without it C is not identified.

WHAT C COSTS. The gap is then actual less potential with no GDP residual, so
every bit of quarterly national accounts noise lands in the gap and from
there in the Okun residual, whose sd is estimated rather than imposed. Two
things follow: `beta_okun` is attenuated by the noise in its own regressor,
and the reported gap is several times wider than A's or B's.

WHAT IS DELIBERATELY ABSENT. The identity gap with a two-sided beta. At the
default trend prior it is nearly identical to C, so it would spend a line on
a prior bound rather than on a modelling choice, and its only distinctive
behaviour is the mirror mode, which is a pathology rather than a view.

WHAT "VINTAGE" MEANS HERE, as in `rstar_summary`: a saved trace counts as
current if the file was written TODAY. That is a proxy for the data being
current rather than a check of ABS and RBA vintages, and it errs the right
way since a stale file is always re-run and a fresh one never is.

REFRESHING RE-ESTIMATES. Each run takes a few minutes and writes to its own
prefix, so refreshing never touches the model's own default outputs.
"""

import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import arviz as az
import numpy as np
import pandas as pd
import xarray as xr

from src.models.ystar_ustar.results import load_results

ROOT = Path(__file__).parent.parent.parent.parent
OUTPUT_DIR = ROOT / "model_outputs"
RUN_SCRIPT = ROOT / "run-ystar-ustar.sh"


@dataclass(frozen=True)
class JointSource:
    """One specification, and how to reproduce it.

    `prefix` is owned by this summary, so a refresh cannot overwrite the
    model's own default run. `flags` is the exact command line, which doubles
    as the record of what the specification is.
    """

    label: str
    prefix: str
    flags: list[str]
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
        print(f"  re-running {self.label} ({' '.join(self.flags)})", flush=True)
        subprocess.run(  # noqa: S603 — our own script, arguments from this module
            [str(RUN_SCRIPT), *self.flags, "--prefix", self.prefix, "--no-analyse"],
            check=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT,
        )


# Colour carries the structure imposed on u*, dashing carries the gap
# definition, so the two choices read separately.
_DEFINED = ["--gap-spec", "defined"]
_IDENTITY = ["--gap-spec", "identity", "--one-sided-beta"]
_K1 = ["--ustar-structure", "spline", "--knots", "2013Q1"]
_K2 = ["--ustar-structure", "spline", "--knots", "1996Q1", "2013Q1"]
_K3 = ["--ustar-structure", "spline", "--knots", "1996Q1", "2008Q1", "2013Q1"]
_DECAY = ["--ustar-structure", "decay"]

SOURCES: list[JointSource] = [
    JointSource("Decay u*, inflation-defined gap", "yus_sum_decay",
                [*_DECAY, *_DEFINED], "tab:blue", "-"),
    JointSource("Spline u*, 1 knot, inflation-defined gap", "yus_sum_k1",
                [*_K1, *_DEFINED], "darkorange", "-"),
    JointSource("Spline u*, 2 knots, inflation-defined gap", "yus_sum_k2",
                [*_K2, *_DEFINED], "seagreen", "-"),
    JointSource("Spline u*, 3 knots, inflation-defined gap", "yus_sum_k3",
                [*_K3, *_DEFINED], "rebeccapurple", "-"),
    JointSource("Decay u*, gap = y - y*", "yus_sum_decay_id",
                [*_DECAY, *_IDENTITY], "tab:blue", "--"),
    JointSource("Spline u*, 1 knot, gap = y - y*", "yus_sum_k1_id",
                [*_K1, *_IDENTITY], "darkorange", "--"),
    JointSource("Spline u*, 2 knots, gap = y - y*", "yus_sum_k2_id",
                [*_K2, *_IDENTITY], "seagreen", "--"),
    JointSource("Spline u*, 3 knots, gap = y - y*", "yus_sum_k3_id",
                [*_K3, *_IDENTITY], "rebeccapurple", "--"),
]


@dataclass
class Loaded:
    """One specification's loaded series and the diagnostics the table reports."""

    source: JointSource
    ustar: pd.Series
    implied_ustar: pd.Series
    potential: pd.Series
    potential_growth: pd.Series
    output_gap: pd.Series
    unemployment: pd.Series
    gap_band: float
    ustar_band: float
    elpd: float
    elpd_se: float
    pareto_bad: int
    max_rhat: float
    min_ess: int
    divergences: int


# The two equations every specification observes. The GDP equation is left out
# on purpose: under the identity gap it is a definition and carries no
# likelihood at all, so scoring it would compare models fitted to different
# data. Scored on the shared targets, the comparison is a fair one about
# predicting unemployment and inflation, and it stays silent about which
# specification is true.
SHARED_TARGETS = ("observed_u", "observed_pi")


def shared_target_loo(trace: az.InferenceData) -> tuple[float, float, int]:
    """Return (elpd_loo, its standard error, count of bad Pareto k) on the shared targets.

    The two equations' pointwise log-likelihoods are concatenated into one
    observation axis, so a point is one quarter of one equation and the score
    is leave-one-observation-out over both.
    """
    stacked = xr.concat(
        [trace.log_likelihood[name].rename({f"{name}_dim_0": "shared"}) for name in SHARED_TARGETS],
        dim="shared",
    )
    # Added to the trace's own log_likelihood group rather than wrapped in a
    # bare InferenceData, which arviz does not accept as a scoring object.
    trace.log_likelihood["shared"] = stacked
    loo = az.loo(trace, var_name="shared", pointwise=True)
    bad = int((np.asarray(loo.pareto_k) > _PARETO_LIMIT).sum())
    return float(loo.elpd_loo), float(loo.se), bad


# Above this, the importance-sampling estimate for that observation is not
# reliable and the LOO figure should be read with suspicion. ArviZ's own
# threshold.
_PARETO_LIMIT = 0.7


def sampling_diagnostics(trace: az.InferenceData) -> tuple[float, int, int]:
    """Return (max R-hat, minimum bulk ESS, divergences) for the whole trace.

    The gate, not a score: a specification that did not sample is out however
    well it fits, and a mirror mode shows up here long before it shows up in
    the fit.
    """
    summary = az.summary(trace)
    return (
        float(summary["r_hat"].max()),
        int(summary["ess_bulk"].min()),
        int(trace.sample_stats["diverging"].to_numpy().sum()),
    )


def load(source: JointSource) -> Loaded:
    """Read one specification's saved run."""
    results = load_results(prefix=source.prefix)
    ustar = results.ustar_posterior()
    gap = results.output_gap_posterior()
    mean_width = lambda f: float(  # noqa: E731 — one expression, used twice
        (f.quantile(0.95, axis=1) - f.quantile(0.05, axis=1)).mean(),
    )
    elpd, elpd_se, pareto_bad = shared_target_loo(results.trace)
    max_rhat, min_ess, divergences = sampling_diagnostics(results.trace)
    return Loaded(
        source=source,
        ustar=ustar.median(axis=1),
        implied_ustar=results.implied_ustar(),
        potential=results.potential_posterior().median(axis=1),
        potential_growth=results.potential_growth_posterior().median(axis=1),
        output_gap=gap.median(axis=1),
        unemployment=pd.Series(results.obs["u"], index=results.obs_index),
        gap_band=mean_width(gap),
        ustar_band=mean_width(ustar),
        elpd=elpd,
        elpd_se=elpd_se,
        pareto_bad=pareto_bad,
        max_rhat=max_rhat,
        min_ess=min_ess,
        divergences=divergences,
    )


def load_all(*, refresh: bool = True) -> list[Loaded]:
    """Refresh anything stale, then load every specification."""
    stale = [s for s in SOURCES if not s.is_current()]
    if refresh and stale:
        print(f"Re-running {len(stale)} of {len(SOURCES)} specifications, a few minutes each:")
        for source in stale:
            source.refresh()
    elif stale:
        names = ", ".join(s.label for s in stale)
        print(f"Charting saved runs as they stand. Not from today: {names}")

    loaded = []
    for source in SOURCES:
        if not source.trace_path.exists():
            print(f"  no saved run for {source.label}; use a refresh", file=sys.stderr)
            continue
        loaded.append(load(source))
    return loaded
