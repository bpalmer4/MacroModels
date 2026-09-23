"""The comparison specifications behind `--compare`, and how to refresh them.

Eight settings of this model, a full crossing of the structure imposed on u*
(decay, or a spline with 1, 2 or 3 knots) with the definition of the output
gap (inflation-defined, or y - y*). MODEL_NOTES, "Comparing specifications",
explains the crossing, the Okun bound the identity gap needs, and how to read
the comparison.

Each specification is the command-line flags it would be run with, parsed by
the same parser as the default run. One is the default run itself, saving and
charting exactly where a plain run does; the others save to their own
`yus_sum_*` prefix and chart to their own directory beside the default run's.
A saved run counts as current if its trace was written today.
"""

import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import arviz as az
import numpy as np
import pandas as pd
import xarray as xr

from src.models.ystar_ustar.analyse import run_analysis
from src.models.ystar_ustar.cli import build_parser, run_from_args
from src.models.ystar_ustar.config import CHART_DIR, ModelConfig
from src.models.ystar_ustar.results import load_results
from src.paths import MODEL_OUTPUTS

OUTPUT_DIR = MODEL_OUTPUTS

# The comparison runs' prefixes share this stem; a run's own chart directory is
# the default one with the rest of its prefix appended, e.g. YStarUStar-k2.
_PREFIX_STEM = "yus_sum_"

# Above this the importance-sampling estimate for an observation is unreliable.
# ArviZ's own threshold.
_PARETO_LIMIT = 0.7


@dataclass(frozen=True)
class Specification:
    """One comparison specification, and how to reproduce it.

    `flags` is the exact command line, which doubles as the record of what the
    specification is.
    """

    label: str
    prefix: str
    flags: list[str]
    colour: str
    style: str
    # The default run: no flags but its prefix, so it is the same run a plain
    # `./run-ystar-ustar.sh` makes, and it charts where a plain run does.
    default: bool = False

    @property
    def trace_path(self) -> Path:
        """Where this specification's saved trace lives."""
        return OUTPUT_DIR / f"{self.prefix}_trace.nc"

    @property
    def chart_dir(self) -> Path:
        """Where this specification's own charts go: beside the default run's."""
        if self.default:
            return CHART_DIR
        return CHART_DIR.with_name(f"{CHART_DIR.name}-{self.prefix.removeprefix(_PREFIX_STEM)}")

    def is_current(self) -> bool:
        """Report whether the saved trace was written today."""
        if not self.trace_path.exists():
            return False
        written = datetime.fromtimestamp(self.trace_path.stat().st_mtime).astimezone()
        return written.date() == datetime.now().astimezone().date()

    def refresh(self) -> None:
        """Re-estimate this specification into its own prefix."""
        print(f"  re-running {self.label} ({' '.join(self.flags)})", flush=True)
        run_from_args(build_parser().parse_args([*self.flags, "--prefix", self.prefix, "--no-analyse"]))

    def chart(self) -> None:
        """Write this specification's own charts from its saved run."""
        run_analysis(prefix=self.prefix, sigma_v_prior=ModelConfig.sigma_v_prior, chart_dir=self.chart_dir)


# Colour carries the structure imposed on u*, dashing carries the gap
# definition, so the two choices read separately. The identity gap needs the
# one-sided Okun slope: without it a mirror mode opens (MODEL_NOTES).
_DEFINED = ["--gap-spec", "defined"]
_IDENTITY = ["--gap-spec", "identity", "--one-sided-beta"]
_K1 = ["--ustar-structure", "spline", "--knots", "2013Q1"]
_K2 = ["--ustar-structure", "spline", "--knots", "1996Q1", "2013Q1"]
_K3 = ["--ustar-structure", "spline", "--knots", "1996Q1", "2008Q1", "2013Q1"]
_DECAY = ["--ustar-structure", "decay"]

SPECIFICATIONS: list[Specification] = [
    Specification("Decay u*, inflation-defined gap", "yus_sum_decay",
                  [*_DECAY, *_DEFINED], "tab:blue", "-"),
    Specification("Default run", "ystar_ustar", [], "darkorange", "-", default=True),
    Specification("Spline u*, 2 knots, inflation-defined gap", "yus_sum_k2",
                  [*_K2, *_DEFINED], "seagreen", "-"),
    Specification("Spline u*, 3 knots, inflation-defined gap", "yus_sum_k3",
                  [*_K3, *_DEFINED], "rebeccapurple", "-"),
    Specification("Decay u*, gap = y - y*", "yus_sum_decay_id",
                  [*_DECAY, *_IDENTITY], "tab:blue", "--"),
    Specification("Spline u*, 1 knot, gap = y - y*", "yus_sum_k1_id",
                  [*_K1, *_IDENTITY], "darkorange", "--"),
    Specification("Spline u*, 2 knots, gap = y - y*", "yus_sum_k2_id",
                  [*_K2, *_IDENTITY], "seagreen", "--"),
    Specification("Spline u*, 3 knots, gap = y - y*", "yus_sum_k3_id",
                  [*_K3, *_IDENTITY], "rebeccapurple", "--"),
]


@dataclass
class Loaded:
    """One specification's loaded series and the diagnostics the table reports."""

    spec: Specification
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
# because under the identity gap it is a definition carrying no likelihood, so
# scoring it would compare models fitted to different data.
SHARED_TARGETS = ("observed_u", "observed_pi")


def _group(trace: az.InferenceData, name: str) -> xr.Dataset:
    """Return one group of the trace, failing loudly if the run did not record it."""
    group = getattr(trace, name, None)
    if not isinstance(group, xr.Dataset):
        raise TypeError(f"trace has no '{name}' group")
    return group


def shared_target_loo(trace: az.InferenceData) -> tuple[float, float, int]:
    """Return (elpd_loo, its standard error, count of bad Pareto k) on the shared targets.

    The two equations' pointwise log-likelihoods are concatenated into one
    observation axis, so a point is one quarter of one equation and the score
    is leave-one-observation-out over both.
    """
    log_likelihood = _group(trace, "log_likelihood")
    stacked = xr.concat(
        [log_likelihood[name].rename({f"{name}_dim_0": "shared"}) for name in SHARED_TARGETS],
        dim="shared",
    )
    # Added to the trace's own log_likelihood group rather than wrapped in a
    # bare InferenceData, which arviz does not accept as a scoring object.
    log_likelihood["shared"] = stacked
    loo = az.loo(trace, var_name="shared", pointwise=True)
    bad = int((np.asarray(loo.pareto_k) > _PARETO_LIMIT).sum())
    return float(loo.elpd_loo), float(loo.se), bad


def sampling_diagnostics(trace: az.InferenceData) -> tuple[float, int, int]:
    """Return (max R-hat, minimum bulk ESS, divergences) for the whole trace."""
    summary = az.summary(trace)
    return (
        float(summary["r_hat"].max()),
        int(summary["ess_bulk"].min()),
        int(_group(trace, "sample_stats")["diverging"].to_numpy().sum()),
    )


def _mean_band_width(draws: pd.DataFrame) -> float:
    """Return the mean width of the 90% interval across quarters."""
    return float((draws.quantile(0.95, axis=1) - draws.quantile(0.05, axis=1)).mean())


def load(spec: Specification) -> Loaded:
    """Read one specification's saved run."""
    results = load_results(prefix=spec.prefix)
    ustar = results.ustar_posterior()
    gap = results.output_gap_posterior()
    elpd, elpd_se, pareto_bad = shared_target_loo(results.trace)
    max_rhat, min_ess, divergences = sampling_diagnostics(results.trace)
    return Loaded(
        spec=spec,
        ustar=ustar.median(axis=1),
        implied_ustar=results.implied_ustar(),
        potential=results.potential_posterior().median(axis=1),
        potential_growth=results.potential_growth_posterior().median(axis=1),
        output_gap=gap.median(axis=1),
        unemployment=pd.Series(results.obs["u"], index=results.obs_index),
        gap_band=_mean_band_width(gap),
        ustar_band=_mean_band_width(ustar),
        elpd=elpd,
        elpd_se=elpd_se,
        pareto_bad=pareto_bad,
        max_rhat=max_rhat,
        min_ess=min_ess,
        divergences=divergences,
    )


def load_all(*, refresh: bool = True) -> list[Loaded]:
    """Re-estimate anything not from today (if `refresh`), then load every specification."""
    stale = [s for s in SPECIFICATIONS if not s.is_current()]
    if refresh and stale:
        print(f"Re-running {len(stale)} of {len(SPECIFICATIONS)} specifications, a few minutes each:")
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
