"""Where each estimate of potential growth comes from, and how to load it.

NOT A MODEL. This gathers g* from every model in the repo that produces one and
puts them on a single chart. Nothing here is estimated.

Deliberately narrower than `rstar_summary`, and the contrast is the point. r*
spans about 1.0pp across defensible models and every one of them fails on a
level. g* spans about 0.2pp, and the models that disagree about it disagree for
reasons you can name: whether potential is a slow random walk disciplined by
inflation, or built up from factor trends, or fitted jointly with u*.

FOUR LINES, AND NOT FOUR INDEPENDENT VOTES. Three share the y* core, and the
fourth shares its key assumption.

- `ystar`'s inflation and production specs are the same package run two ways,
  sharing the data, the sample, the level equation and the gap definition, and
  differing only in where potential's GROWTH comes from. `ystar`'s own notes
  call their agreement a semi-validation and warn that it only exists while the
  two are kept separate. Note in particular that **`production` is not a
  supply-side-only estimate**: its growth comes from factor trends, but
  inflation still positions its level and still defines its gap, which is why
  its charts include an inflation-defined output gap.
- The joint y*/u* model builds on the same y* core and adds Okun and a Phillips
  curve.
- `rstar_qpm` is a separate package: potential drifts with a trend-growth
  state inside an open-economy system (IS curve, exchange rate, Phillips
  curve, policy rule). But it too treats potential as a slowly drifting random
  walk, and how smooth that walk is comes from a prior (`sigma_ystar` is not
  identified there). So it is a partial outside check, not an independent one.

**So the agreement here is weaker evidence than it looks.** `cobb_douglas`, the
one line built a different way, is excluded for COVID artefacts, below. If a
smoothing assumption common to all four were wrong, nothing here would catch it.
"""

import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

from src.models.rstar_qpm.estimate import load_results as load_qpm_results
from src.models.ystar.results import load_results as load_ystar_results
from src.models.ystar_ustar.results import load_results as load_joint_results
from src.paths import MODEL_OUTPUTS, ROOT

if TYPE_CHECKING:
    from collections.abc import Callable

OUTPUT_DIR = MODEL_OUTPUTS


@dataclass(frozen=True)
class GstarSource:
    """One model's potential growth, and how to get it.

    Attributes:
        label: how the chart names it
        prefix: the saved run this reads, or "" when the source is computed
        script: the run script that refreshes it
        loader: returns year-ended potential growth, per cent
        note: what this g* is built from, for the run log
        script_args: arguments the script needs to reproduce this run, when it
            is not the script's default run

    """

    label: str
    prefix: str
    script: str
    loader: Callable[[str], pd.Series]
    note: str
    script_args: tuple[str, ...] = ()

    @property
    def trace_path(self) -> Path:
        """Where the saved trace lives, or a path that never exists."""
        if not self.prefix:
            return OUTPUT_DIR / "__computed__"
        return OUTPUT_DIR / f"{self.prefix}_trace.nc"

    def is_current(self, *, today: datetime | None = None) -> bool:
        """Return True if the saved trace was written today.

        A computed source has no trace, so it is never current and never
        refreshed: it reads live data every time it is loaded.
        """
        if not self.prefix or not self.trace_path.exists():
            return False
        stamp = datetime.fromtimestamp(self.trace_path.stat().st_mtime)  # noqa: DTZ006 — local, matches the file system
        now = today or datetime.now()  # noqa: DTZ005 — local, as above
        return stamp.date() == now.date()


def _load_ystar(prefix: str) -> pd.Series:
    """Potential growth from `ystar`, on whichever spec `prefix` was run with."""
    return load_ystar_results(prefix=prefix).potential_growth_posterior().median(axis=1)


def _load_joint(prefix: str) -> pd.Series:
    """Potential growth from the joint y*/u* model."""
    return load_joint_results(prefix=prefix).potential_growth_posterior().median(axis=1)


def _load_qpm(prefix: str) -> pd.Series:
    """Potential growth from `rstar_qpm`: its trend-growth state `g`.

    Potential without shocks. The year-ended change in the model's y* path
    also carries the level shocks to potential, which read as swings in growth
    (0.25pp above `g` in 2022Q4) that are not changes in the trend.
    """
    _, _, _, states = load_qpm_results(prefix=prefix)
    return states["paths"]["g"].median(axis=1)


# WHY `cobb_douglas` IS NOT HERE: COVID artefacts.
#
# It was the one line from outside the y* state-space family, which is exactly
# the independent check the three below lack, so it was worth several attempts.
# Its three HP filters run through the pandemic and leave a COVID-shaped wobble
# of a few tenths from 2020 on. Excluding a window does not remove it, only
# changes its sign and size (+0.42 with no exclusion, +0.96 excluding the
# lockdown alone, -0.39 excluding through the rebound), and the windows that
# minimise it were picked by comparing against these very models, which is
# tuning rather than fixing. See src/models/cobb_douglas/model.py.

# WHY `rstar_hlw` IS NOT HERE.
#
# Its trend growth g is a state in a model whose r* is not identified, and its
# potential output was repaired only on 2026-09-12. The repaired version is
# defensible, and on the Okun checkpoints its OUTPUT GAP is the best validated
# in the repo. But g there is pinned by GDP alone with no labour-market or
# factor information, it is the drift of a single random walk, and its level
# moved 1.88 -> 2.23 purely because the sample start changed from 1986Q3 to
# 1993Q1. Excluded at the user's direction; see src/models/rstar_hlw/MODEL_NOTES.md.
SOURCES: tuple[GstarSource, ...] = (
    GstarSource(
        label="y* (inflation spec)",
        prefix="ystar",
        script="run-ystar.sh",
        loader=_load_ystar,
        note="potential is a slow random walk; the gap is defined by inflation",
    ),
    GstarSource(
        label="y* (production spec)",
        prefix="ystar_production",
        script="run-ystar.sh",
        loader=_load_ystar,
        note="growth from capital, hours and MFP trends; level and gap still inflation-defined",
        script_args=("--spec", "production", "--prefix", "ystar_production"),
    ),
    GstarSource(
        label="Joint y*/u*",
        prefix="ystar_ustar",
        script="run-ystar-ustar.sh",
        loader=_load_joint,
        note="y* and u* estimated together, with Okun and a Phillips curve",
    ),
    GstarSource(
        label="Semi-structural open economy",
        prefix="rstar_qpm",
        script="run-rstar-qpm.sh",
        loader=_load_qpm,
        note=("y* a drifting random walk inside an IS / exchange-rate / Phillips / rule "
              "system; its smoothness is set by the sigma_ystar prior"),
    ),
)


def refresh(source: GstarSource, *, timeout: int = 3600) -> None:
    """Re-run one model's script, so its saved output carries today's data."""
    script = ROOT / source.script
    if not script.exists():
        raise FileNotFoundError(f"no run script for {source.label}: {script}")
    print(f"  refreshing {source.label} via ./{source.script} ...")
    sys.stdout.flush()
    subprocess.run(  # noqa: S603 — our own script, path built from the registry
        [str(script), *source.script_args], cwd=str(ROOT), check=True, timeout=timeout,
    )
    sys.stdout.flush()


def gather(
    *,
    allow_refresh: bool = True,
    verbose: bool = True,
) -> tuple[pd.DataFrame, dict[str, str]]:
    """Return every model's potential growth on a shared quarterly index.

    Args:
        allow_refresh: re-run any model whose saved trace is not from today,
            with that source's `script_args`, which also redraws its charts.
        verbose: print what was current, what was refreshed and what was read

    Returns:
        (frame, notes). `frame` holds one column per source; `notes` maps each
        label to what its g* is built from.

    """
    if verbose:
        print("Checking vintages (a trace counts as current if written today):")
        for source in SOURCES:
            state = ("current" if source.is_current() else "STALE") if source.prefix else "computed"
            print(f"  {state:>8}  {source.label}")

    if allow_refresh:
        for source in SOURCES:
            if source.prefix and not source.is_current():
                refresh(source)

    columns, notes = _load_all(verbose=verbose)
    if not columns:
        raise RuntimeError("no g* models could be loaded")
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
    """Load each source. One that cannot be read is named and skipped."""
    columns: dict[str, pd.Series] = {}
    notes: dict[str, str] = {}
    if verbose:
        print("\nLoading:")
    for source in SOURCES:
        try:
            series = _as_quarterly(source.loader(source.prefix))
        except (FileNotFoundError, KeyError, ValueError) as exc:
            print(f"  SKIPPED {source.label}: {type(exc).__name__}: {exc}")
            continue
        columns[source.label] = series
        notes[source.label] = source.note
        if verbose:
            where = source.prefix or "computed from ABS"
            print(f"  {source.label:<36} {where}")
    return columns, notes
