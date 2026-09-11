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

if TYPE_CHECKING:
    from collections.abc import Callable

ROOT = Path(__file__).parent.parent.parent.parent
OUTPUT_DIR = ROOT / "model_outputs"

# A neutral rate is defined at TARGET inflation, not at whatever inflation
# happened to be, so real converts to nominal by adding the target. Same
# convention as `rstar_rba`, which records both scales for this reason.
TARGET = 2.5


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
        note="anchored to the Cleveland Fed 10y expected real rate; LEVEL not identified",
    ),
    RstarSource(
        label="RBA reaction function (neutral b_t)",
        prefix="rstar_rba",
        script="run-rstar-rba.sh",
        loader=_load_rba,
        nominal=True,
        note="neutral, NOT prescribed; level conditional on an arbitrary sigma_r",
    ),
    RstarSource(
        label="IS inversion (slope asserted)",
        prefix="rstar_invert",
        script="run-rstar-invert.sh",
        loader=_load_invert,
        nominal=False,
        note="anchored to nothing; path decided by the asserted speed of r*",
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
) -> tuple[pd.DataFrame, dict[str, str]]:
    """Return every model's r* on a NOMINAL scale, refreshing stale runs first.

    Args:
        allow_refresh: re-run any model whose saved trace is not from today
        verbose: print what was current, what was refreshed and what was read

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

    columns, notes = _load_all(verbose=verbose)
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


def _load_all(*, verbose: bool) -> tuple[dict[str, pd.Series], dict[str, str]]:
    """Load each source, converting every column to a NOMINAL rate.

    A model that cannot be loaded is skipped and named rather than fatal: one
    missing run should not stop the comparison of the others.
    """
    columns: dict[str, pd.Series] = {}
    notes: dict[str, str] = {}
    if verbose:
        print("\nLoading:")
    for source in SOURCES:
        try:
            series = _as_quarterly(source.loader(source.prefix))
        except (FileNotFoundError, KeyError, ValueError) as exc:
            print(f"  SKIPPED {source.label}: {type(exc).__name__}")
            continue
        # Every column leaves here NOMINAL, so the chart never has to ask.
        columns[source.label] = series + (0.0 if source.nominal else TARGET)
        notes[source.label] = source.note
        if verbose:
            scale = "nominal" if source.nominal else f"real + {TARGET:g}"
            print(f"  {source.label:<40} {source.prefix} ({scale})")
    return columns, notes
