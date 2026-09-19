"""The three u* specifications this summary charts, and how to refresh them.

NOT THREE MODELS. Three settings of one model, `ustar`, differing in the
number of spline knots and in whether the gap-form Okun equation is included.
Gap form means `u - u* = -beta x ygap`, the unemployment gap against the
output gap, not Okun's original relating the CHANGE in unemployment to output
growth. The coefficient is therefore not comparable with a textbook 0.3 to
0.5, and it is the gap-on-gap structure specifically that collapses into a
Phillips curve once `ygap` is itself a multiple of `pi - 2.5`.
Everything else is shared, including the sample, the Phillips curve and the
expectations series. Where `rstar_summary` gathers estimates built on
different data and different identifying assumptions, agreement between these
lines is close to arithmetic; only their disagreement is informative.

EVERY LINE CAN TURN UP AT THE ENDPOINT, which is why the decay settings are
not here. Under `--state converge` the sign of `phi x (eq - u*)` is fixed by
which side of the equilibrium the state opened on: from 10.75 with `eq` at
4.86, u* approaches from above, never crosses, and can only ever report a
fall. Its -0.32 and -0.34 over 2015-2026 are properties of the shape rather
than readings of the data, and on a chart about how much the specification
matters they invite being read as evidence. A spline has no such constraint,
so if u* starts rising these lines will show it.

ONE OF THE THREE KEEPS THE OKUN EQUATION, which `ustar`'s default excludes,
because it is the only remaining setting in which u* comes down through the
1990s. That decade was a regime change, inflation moving from high to low and
taking years to work through the labour market. Unemployment fell 4.53 points
over 1993-1999; two-knots-with-Okun opens at 10.15 and falls 3.43, while the
two without open near 7.3 and fall 0.50 and 0.81, asserting the descent was
almost entirely cyclical.

IT CARRIES TWO SEPARATE CLAIMS AND ONLY THE FIRST IS WHY IT IS HERE. Its
post-2015 slope is -0.59, the steepest of anything tried, so it is also the
strongest claim that u* is still falling now. Its 1990s credentials lend that
no weight.

THE SPREAD IS 0.03pp at the endpoint and reaches 2.87pp at 1993Q1. The profile
is the point: it fairly reflects a period where pinning u* is harder, and it
agrees with the shaded window and with the bias against the implied series.
It is not a symmetric error band, since the three differ in a structured way
rather than randomly.

WHAT "VINTAGE" MEANS HERE, as in `rstar_summary`: a saved trace counts as
current if the file was written TODAY. That is a proxy for the data being
current rather than a check of ABS and RBA vintages, and it errs the right way
since a stale file is always re-run and a fresh one never is.

REFRESHING RE-ESTIMATES. Each run takes about 20 seconds and writes to its own
prefix, so refreshing never touches `ustar`'s own default outputs or charts.
"""

import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import pandas as pd

from src.models.ustar.results import load_results

ROOT = Path(__file__).parent.parent.parent.parent
OUTPUT_DIR = ROOT / "model_outputs"
RUN_SCRIPT = ROOT / "run-ustar.sh"


@dataclass(frozen=True)
class UstarSource:
    """One specification, and how to reproduce it.

    `prefix` is owned by this summary, so a refresh cannot overwrite the
    default `ustar` run. `flags` is the exact command line, which doubles as
    the record of what the specification is.
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
            [str(RUN_SCRIPT), *self.flags, "--prefix", self.prefix],
            check=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT,
        )


# Colour carries the knot count, dashing carries Okun, so the two dimensions
# read separately.
SOURCES: list[UstarSource] = [
    UstarSource("Spline 1 knot", "ustar_sum_k1",
                ["--state", "spline", "--knots", "2013Q1"], "darkorange", "-"),
    UstarSource("Spline 2 knots", "ustar_sum_k2",
                ["--state", "spline", "--knots", "1996Q1", "2013Q1"], "tab:blue", "-"),
    UstarSource("Spline 2 knots, with gap-form Okun", "ustar_sum_k2_okun",
                ["--state", "spline", "--knots", "1996Q1", "2013Q1", "--okun"], "tab:blue", "--"),
]


@dataclass
class Loaded:
    """One specification's loaded series and the diagnostics the table reports."""

    source: UstarSource
    ustar: pd.Series
    implied: pd.Series
    unemployment: pd.Series
    band: float
    extras: dict[str, float] = field(default_factory=dict)


def without_okun(loaded: list[Loaded]) -> list[Loaded]:
    """Return only the specifications that exclude the Okun equation."""
    return [item for item in loaded if not item.source.prefix.endswith("_okun")]


def load(source: UstarSource) -> Loaded:
    """Read one specification's saved run."""
    results = load_results(prefix=source.prefix)
    posterior = results.ustar_posterior()
    return Loaded(
        source=source,
        ustar=posterior.median(axis=1),
        implied=results.implied_ustar(),
        unemployment=pd.Series(results.obs["u"], index=results.obs_index),
        band=float((posterior.quantile(0.95, axis=1) - posterior.quantile(0.05, axis=1)).mean()),
    )


def load_all(*, refresh: bool = True) -> list[Loaded]:
    """Refresh anything stale, then load every specification."""
    stale = [s for s in SOURCES if not s.is_current()]
    if refresh and stale:
        print(f"Re-running {len(stale)} of {len(SOURCES)} specifications, about 20s each:")
        for source in stale:
            source.refresh()
    elif stale:
        names = ", ".join(s.label for s in stale)
        print(f"Charting saved runs as they stand. Not from today: {names}")

    loaded = []
    for source in SOURCES:
        if not source.trace_path.exists():
            print(f"  no saved run for {source.label}; use --refresh", file=sys.stderr)
            continue
        loaded.append(load(source))
    return loaded
