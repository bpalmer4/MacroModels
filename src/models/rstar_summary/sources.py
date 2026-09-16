"""The r* estimates this repo produces, gathered on one comparable scale.

Each r* model here answers a different question and anchors r* to a different
thing, which is why they disagree. This module does not adjudicate between
them. It loads them, puts them all on a NOMINAL scale, and records what each
one actually is so the chart can say so.

WHAT "VINTAGE" MEANS HERE. A saved trace is treated as current if the file was
written TODAY. That is a proxy for the data being current, not a check of the
underlying ABS and RBA vintages: a model re-run today picks up whatever those
series then hold. It is the practical test, and it is conservative in the right
direction, since a stale file is always re-run and a fresh one never is.

REFRESHING RUNS THE OTHER MODEL. That is a real side effect: it takes minutes
and it overwrites that model's own outputs and charts. Nothing is refreshed
without saying so first, and `--no-refresh` skips it entirely.
"""

import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

from src.models.common import inflation_scale

if TYPE_CHECKING:
    from collections.abc import Callable

ROOT = Path(__file__).parent.parent.parent.parent
OUTPUT_DIR = ROOT / "model_outputs"

# Real converts to nominal by adding LONG-RUN INFLATION EXPECTATIONS, which is
# what the RBA and CBA both do, so these lines are comparable with a published
# neutral rate. It used to add the flat 2.5% target. The two agree closely after
# 2000 and differ by up to a point through the 1990s re-anchoring, where
# expectations genuinely sat above target and the old convention understated
# every nominal path on the chart.
#
# `src/models/common/inflation_scale.py` holds the convention and the argument
# for the anchored series over the unanchored one. `--nominal-on target`
# restores the old behaviour, so previously published numbers stay reproducible.
TARGET = inflation_scale.TARGET
DEFAULT_SCALE = "expectations"


@dataclass(frozen=True)
class RstarSource:
    """One model's r*, and how to get it.

    Attributes:
        label: how the chart names it
        prefix: the saved run this reads
        script: the run script that refreshes it
        loader: returns the series, on whatever scale `nominal` says
        nominal: True if the loader already returns a nominal rate
        note: what this r* is anchored to, for the run log

    """

    label: str
    prefix: str
    script: str
    loader: Callable[[str], pd.Series]
    nominal: bool
    note: str

    @property
    def trace_path(self) -> Path:
        """Where the saved trace lives."""
        return OUTPUT_DIR / f"{self.prefix}_trace.nc"

    def is_current(self, *, today: datetime | None = None) -> bool:
        """Return True if the saved trace was written today."""
        if not self.trace_path.exists():
            return False
        stamp = datetime.fromtimestamp(self.trace_path.stat().st_mtime)  # noqa: DTZ006 — local, matches the file system
        now = today or datetime.now()  # noqa: DTZ005 — local, as above
        return stamp.date() == now.date()


def _load_bonds(prefix: str) -> pd.Series:
    """Bond-market r*, real. Anchored to a world real rate plus an AU wedge."""
    from src.models.rstar_bonds.results import load_results  # noqa: PLC0415

    return load_results(prefix=prefix).rstar_median()


def _load_rba(prefix: str) -> pd.Series:
    """RBA reaction-function neutral, ALREADY NOMINAL.

    `neutral` is the slow base `b_t`, NOT `prescribed`, which adds the Bank's
    inflation response on top and is not a neutral rate. That distinction is
    the one `rstar_rba`'s notes insist on, so it is made explicitly here.
    """
    from src.models.rstar_rba.estimate import load_results, posterior_median  # noqa: PLC0415

    trace, frame, _ = load_results(prefix=prefix)
    index = frame.index
    if not isinstance(index, pd.PeriodIndex):
        index = pd.PeriodIndex(index, freq="Q")
    return posterior_median(trace, "neutral", index)


def _load_invert(prefix: str) -> pd.Series:
    """IS-inversion r*, real. Anchored to NOTHING, which is its whole problem."""
    from src.models.rstar_invert.estimate import load_results, posterior_median  # noqa: PLC0415

    trace, frame, _ = load_results(prefix=prefix)
    index = frame.index
    if not isinstance(index, pd.PeriodIndex):
        index = pd.PeriodIndex(index, freq="Q")
    return posterior_median(trace, "rstar", index)


def _load_tvpvar(prefix: str) -> pd.Series:
    """TVP-VAR r*, real: a model-implied 5y5y.

    The average of the conditioned projection over quarters 20 to 40.

    CHANGED 2026-09-16 from the steady state, and the reason is comparability.
    `rstar_bonds` and `rstar_rba` are both pinned to the AOFM's 5y5y forward,
    and CBA's published figure sits within 0.01 of that same forward. A 5y5y is
    a FINITE-HORIZON object, the average expected short rate over years five to
    ten, so the comparable quantity here is a window average over those years
    rather than an infinite-horizon limit. The chart was putting a different
    estimand beside the other two and calling it a third opinion.

    The steady state also failed on its own terms. Unconditioned it is the real
    rate at the VAR's own long-run inflation of 3.15%, not at target, so it is
    not r*; 29.6% of draws had no steady state at all; and its median was set by
    near-unit-root draws, since `(I - F)^-1` has a vanishing denominator as the
    spectral radius approaches one. Conditioning fixes the concept but leaves
    the level swinging from -0.58 to +1.44 across defensible inflation anchors.
    Over a finite window none of that has time to act: no draw is undefined and
    the 90% band falls from 9.77 to 3.59.

    `--rstar-definition steady` and `projection` both remain. Note that neither
    this nor `projection` is the published Lubik-Matthes estimator: theirs is
    the five-year POINT, and the inflation conditioning is not in their method
    at all.
    """
    from src.models.rstar_tvpvar.results import load_results  # noqa: PLC0415

    return load_results(prefix=prefix).rstar_median()


# WHY `rstar_invert` IS NOT HERE. Removed 2026-09-16. It asserts an IS curve,
# and five independent methods in this repo now say there is not one to assert:
# `rstar_hlw` measures the link at -0.04, `nairu` at +0.084, `is_curve` cannot
# recover the SIGN, `rstar_tvpvar` returns +0.04 with the wrong sign in 87% of
# draws even after commodity prices and the exchange rate are added, and
# `rstar_invert`'s own measurements are -0.015 to -0.034 before its prior
# overrides them.
#
# THE DECIDING TEST was the sweep. Every other model here has something that
# SURVIVES varying its imposed number: `rstar_bonds` keeps its wedge reading
# across three anchors, `rstar_rba` keeps `lambda` across `sigma_r`. Across
# `sigma_rstar` this model's slope runs -0.032 to -0.389, r* goes from a flat
# line to a 9.7-point swing, and the 2016-19 stance flips from -3.19 to +1.16.
# Nothing holds. The fit improves monotonically to 0.50, so the data cannot
# choose either.
#
# AND THE SLOPE PRIOR IS DECORATIVE. Moving `is_slope_mu` from -0.30 to -0.10
# leaves the posterior at -0.371 against -0.389, because the parameterisation
# rewards a large slope: it buys a more movable line. So a reader who set a
# defensible prior would still be shown -0.37.
#
# The package is kept and still runs. It is the best-sampled model here (R-hat
# 1.00, ESS 4,987, zero divergences) and it measures the output gap's own slow
# component well. It is not a neutral rate.


# WHY `rstar_hlw` IS NOT HERE, and please do not add it back without reading this.
#
# No resolution of that model produces an IDENTIFIED r* path, so any line it
# contributes would be a picture of a prior.
#
#   A (canonical)  sigma_z is free with a HalfNormal(0.10) prior and its
#                  posterior median comes back at 0.0657 against a prior median
#                  of 0.0674: the data says nothing about how far z moves. The
#                  flat path is the median of many draws each wandering
#                  differently, not a finding that r* held still. ESS on
#                  `r_star` is 243.
#   B              3,349 divergences, and its own notes call z "wild".
#   C, E, F, G, H  all rest on alpha, which posteriors at 0.58 with a 90%
#                  interval of [0.03, 0.99] and is bimodal. The notes are
#                  explicit that the blended median is the average of two
#                  stories that almost no single draw sits at.
#
# What HLW DOES establish is why: the per-quarter signal is a_r/sigma_IS =
# 0.044/0.685 = 0.064, so r* would have to be wrong by ~15pp to move the
# likelihood by one sd. That belongs in the notes, not on a chart of levels.
# See src/models/rstar_hlw/MODEL_NOTES.md.
SOURCES: tuple[RstarSource, ...] = (
    RstarSource(
        label="Bond market (world rate + AU wedge)",
        prefix="rstar_bonds",
        script="run-rstar-bonds.sh",
        loader=_load_bonds,
        nominal=False,
        note=("anchored to a premium-stripped US real rate with b_world imposed at 1; "
              "AU term premium taken from the AOFM; LEVEL not identified"),
    ),
    RstarSource(
        label="RBA reaction function (neutral b_t)",
        prefix="rstar_rba",
        script="run-rstar-rba.sh",
        loader=_load_rba,
        nominal=True,
        note=("neutral b_t, NOT prescribed; the level is pinned by the AOFM 5y5y "
              "forward as a second window, with a free but tightly priored bias"),
    ),
    RstarSource(
        label="TVP-VAR (5y5y-equivalent)",
        prefix="rstar_tvpvar",
        script="run-rstar-tvpvar.sh",
        loader=_load_tvpvar,
        nominal=False,
        note=("no IS curve and no term premium; r* is the VAR's RESTING POINT, whose "
              "level is the sample-average real cash rate and whose posterior is a "
              "ratio with fat tails (50% band, not 90%); ESS 245 and 7 divergences"),
    ),
)


def refresh(source: RstarSource, *, timeout: int = 3600) -> None:
    """Re-run one model's script, so its saved output carries today's data."""
    script = ROOT / source.script
    if not script.exists():
        raise FileNotFoundError(f"no run script for {source.label}: {script}")
    print(f"  refreshing {source.label} via ./{source.script} ...")
    # Our prints are buffered, the child writes straight to the terminal, so
    # without this the child's output appears BEFORE the header explaining why
    # it is running, and the log reads out of order.
    sys.stdout.flush()
    subprocess.run(  # noqa: S603 — our own script, path built from the registry
        [str(script)], cwd=str(ROOT), check=True, timeout=timeout,
    )
    sys.stdout.flush()


def gather(
    *,
    allow_refresh: bool = True,
    verbose: bool = True,
    scale: str = DEFAULT_SCALE,
) -> tuple[pd.DataFrame, dict[str, str]]:
    """Return every model's r* on a NOMINAL scale, refreshing stale runs first.

    Args:
        allow_refresh: re-run any model whose saved trace is not from today
        verbose: print what was current, what was refreshed and what was read
        scale: how real converts to nominal. "expectations" (default) adds
            long-run inflation expectations, matching the RBA and CBA;
            "target" adds 2.5%, which is what this package did before
            2026-09-16 and is kept so those numbers stay reproducible.

    Returns:
        (frame, notes). `frame` holds one nominal column per model on a shared
        quarterly index; `notes` maps each label to what its r* is anchored to.

    """
    if verbose:
        print("Checking vintages (a trace counts as current if written today):")

    stale = [source for source in SOURCES if not source.is_current()]
    if verbose:
        for source in SOURCES:
            state = "current" if source.is_current() else "STALE"
            print(f"  {state:>8}  {source.label}")

    if stale and allow_refresh:
        print(f"\nRefreshing {len(stale)} stale model(s). This re-runs them and "
              "overwrites their own outputs and charts.")
        for source in stale:
            refresh(source)
    elif stale and verbose:
        print(f"\n  --no-refresh: reading {len(stale)} stale run(s) as they stand")

    columns, notes = _load_all(verbose=verbose, scale=scale)
    if not columns:
        raise RuntimeError("no r* models could be loaded")
    return pd.DataFrame(columns).sort_index(), notes


def _as_quarterly(series: pd.Series) -> pd.Series:
    """Return `series` on a quarterly PeriodIndex."""
    index = series.index
    if isinstance(index, pd.PeriodIndex):
        series.index = index.asfreq("Q")
    elif isinstance(index, pd.DatetimeIndex):
        series.index = index.to_period("Q")
    return series.astype(float)


def _load_all(*, verbose: bool, scale: str = DEFAULT_SCALE) -> tuple[dict[str, pd.Series], dict[str, str]]:
    """Load each source, converting every column to a NOMINAL rate.

    A model that cannot be loaded is skipped and named rather than fatal: one
    missing run should not stop the comparison of the others. The scale
    conversion is NOT treated that way: if the expectations run is missing, the
    whole comparison is on the wrong footing, so it raises.
    """
    columns: dict[str, pd.Series] = {}
    notes: dict[str, str] = {}
    label = inflation_scale.scale_label(scale)
    if verbose:
        print(f"\nLoading (real converts to nominal by adding {label}):")
    for source in SOURCES:
        try:
            series = _as_quarterly(source.loader(source.prefix))
        except (FileNotFoundError, KeyError, ValueError) as exc:
            print(f"  SKIPPED {source.label}: {type(exc).__name__}")
            continue
        # Every column leaves here NOMINAL, so the chart never has to ask.
        columns[source.label] = (
            series if source.nominal else inflation_scale.to_nominal(series, scale=scale)
        )
        notes[source.label] = source.note
        if verbose:
            how = "nominal already" if source.nominal else f"real + {label}"
            print(f"  {source.label:<40} {source.prefix} ({how})")
    return columns, notes
