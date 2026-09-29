"""Where each output gap comes from, and how to load it.

NOT A MODEL. This gathers the output gap from every live model that estimates
y* and puts them on one chart. Nothing here is estimated.

THREE LINES, THREE DEFINITIONS. Each line is the gap its model reports, which is
not the same object in each:

- `ystar` (inflation spec): the inflation-defined gap, c·(pi - anchor). GDP is
  fitted around it with a residual, so it is NOT log GDP - y*; it is the part
  of GDP's deviation from potential that inflation accounts for.
- The joint y*/u* model (slack split, its default): c·(pi - anchor) plus a free
  component v, so part of the gap is not tied to inflation. GDP still carries a
  residual on top, so this is not log GDP - y* either.
- `rstar_qpm`: the `ygap` state in y = y* + ygap, which is GDP's deviation from
  potential up to a small measurement error on that identity.

So the lines differ partly because the definitions differ, and the chart has
to be read that way: the first is the narrowest by construction.
"""

import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

from src.models.common.staleness import is_current
from src.models.rstar_qpm.estimate import load_results as load_qpm_results
from src.models.ystar.results import load_results as load_ystar_results
from src.models.ystar_ustar.results import load_results as load_joint_results
from src.paths import MODEL_OUTPUTS, ROOT

if TYPE_CHECKING:
    from collections.abc import Callable

OUTPUT_DIR = MODEL_OUTPUTS

# A run's excluded window, (first quarter, last quarter), or None.
type Window = tuple[str, str] | None

# A window as a run records it: two quarter strings.
_WINDOW_LEN = 2


@dataclass(frozen=True)
class GapReading:
    """One run's output gap and the window its GDP observation dropped."""

    gap: pd.Series
    window: Window


@dataclass(frozen=True)
class GapSource:
    """One model's output gap, and how to get it.

    Attributes:
        label: how the chart names it
        prefix: the saved run this reads
        script: the run script that refreshes it
        loader: returns the posterior median gap (per cent of potential) and
            the run's excluded window
        note: what this gap is, for the run log
        script_args: arguments the script needs to reproduce this run, when it
            is not the script's default run

    """

    label: str
    prefix: str
    script: str
    loader: Callable[[str], GapReading]
    note: str
    script_args: tuple[str, ...] = ()

    @property
    def trace_path(self) -> Path:
        """Where the saved trace lives."""
        return OUTPUT_DIR / f"{self.prefix}_trace.nc"

    def is_current(self) -> bool:
        """Return True if the saved trace was written today."""
        return is_current(self.trace_path)


def _as_window(value: object) -> Window:
    """Return a recorded window as a tuple of two quarter strings, or None."""
    if isinstance(value, (tuple, list)) and len(value) == _WINDOW_LEN and all(
        isinstance(q, str) and q for q in value
    ):
        return (value[0], value[1])
    return None


def _load_ystar(prefix: str) -> GapReading:
    """Return the inflation-defined gap from `ystar`, c·(pi - anchor)."""
    results = load_ystar_results(prefix=prefix)
    return GapReading(
        gap=results.output_gap_median(),
        window=_as_window(results.constants.get("exclude_window")),
    )


def _load_joint(prefix: str) -> GapReading:
    """Return the joint model's whole gap, c·(pi - anchor) + v."""
    results = load_joint_results(prefix=prefix)
    return GapReading(
        gap=results.output_gap_posterior().median(axis=1),
        window=results.excluded_window,
    )


def _load_qpm(prefix: str) -> GapReading:
    """Return the `ygap` state from `rstar_qpm`, in y = y* + ygap."""
    _, _, constants, states = load_qpm_results(prefix=prefix)
    return GapReading(
        gap=states["paths"]["ygap"].median(axis=1),
        window=_as_window((constants.get("exclude_start"), constants.get("exclude_end"))),
    )


SOURCES: tuple[GapSource, ...] = (
    GapSource(
        label="y* (inflation spec)",
        prefix="ystar",
        script="run-ystar.sh",
        loader=_load_ystar,
        note="c x (pi - anchor): the part of GDP's deviation that inflation accounts for; "
             "not log GDP - y*",
    ),
    GapSource(
        label="Joint y*/u* (slack split)",
        prefix="ystar_ustar",
        script="run-ystar-ustar.sh",
        loader=_load_joint,
        note="c x (pi - anchor) + a free component v; GDP still has a residual on top; the default run",
    ),
    GapSource(
        label="Semi-structural open economy",
        prefix="rstar_qpm",
        script="run-rstar-qpm.sh",
        loader=_load_qpm,
        note="ygap in y = y* + ygap: GDP's deviation from potential, up to a small measurement error",
    ),
)


def refresh(source: GapSource, *, timeout: int = 3600) -> None:
    """Re-run one model's script, so its saved output carries today's data."""
    script = ROOT / source.script
    if not script.exists():
        raise FileNotFoundError(f"no run script for {source.label}: {script}")
    print(f"  refreshing {source.label} via ./{source.script} ...")
    sys.stdout.flush()
    subprocess.run(
        [str(script), *source.script_args], cwd=str(ROOT), check=True, timeout=timeout,
    )
    sys.stdout.flush()


def gather(
    *,
    allow_refresh: bool = True,
    verbose: bool = True,
) -> tuple[pd.DataFrame, dict[str, str], dict[str, Window]]:
    """Return every model's output gap on a shared quarterly index.

    Args:
        allow_refresh: re-run any model whose saved trace is not from today,
            with that source's `script_args`, which also redraws its charts.
        verbose: print what was current, what was refreshed and what was read

    Returns:
        (frame, notes, windows). `frame` holds one column per source; `notes`
        maps each label to what its gap is; `windows` maps each label to the
        quarters its run dropped from the GDP observation.

    """
    if verbose:
        print("Checking vintages (a trace counts as current if written today):")
        for source in SOURCES:
            state = "current" if source.is_current() else "STALE"
            print(f"  {state:>8}  {source.label}")

    if allow_refresh:
        for source in SOURCES:
            if not source.is_current():
                refresh(source)

    columns, notes, windows = _load_all(verbose=verbose)
    if not columns:
        raise RuntimeError("no output gaps could be loaded")
    return pd.DataFrame(columns).sort_index(), notes, windows


def _as_quarterly(series: pd.Series) -> pd.Series:
    """Return `series` on a quarterly PeriodIndex."""
    index = series.index
    if isinstance(index, pd.PeriodIndex):
        series.index = index.asfreq("Q")
    elif isinstance(index, pd.DatetimeIndex):
        series.index = index.to_period("Q")
    return series.astype(float)


def _load_all(*, verbose: bool) -> tuple[dict[str, pd.Series], dict[str, str], dict[str, Window]]:
    """Load each source. One that cannot be read is named and skipped."""
    columns: dict[str, pd.Series] = {}
    notes: dict[str, str] = {}
    windows: dict[str, Window] = {}
    if verbose:
        print("\nLoading:")
    for source in SOURCES:
        try:
            reading = source.loader(source.prefix)
        except (FileNotFoundError, KeyError, ValueError) as exc:
            print(f"  SKIPPED {source.label}: {type(exc).__name__}: {exc}")
            continue
        columns[source.label] = _as_quarterly(reading.gap)
        notes[source.label] = source.note
        windows[source.label] = reading.window
        if verbose:
            print(f"  {source.label:<32} {source.prefix}")
    return columns, notes, windows
