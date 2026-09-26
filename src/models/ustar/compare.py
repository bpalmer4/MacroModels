"""The comparison specifications behind `--compare`, and how to refresh them.

Settings of this model differing in the structure imposed on u*: a random
walk whose step size tapers, or a spline with one or two knots. Everything else is shared, so
agreement between them is close to arithmetic and only their disagreement is
informative. MODEL_NOTES, "Comparing specifications", explains the choice of
the three and how to read them.

Each specification is the command-line flags it would be run with, parsed by
the same parser as the default run, so the list doubles as the record of what
each one is. One is the default run itself, saving and charting exactly where
a plain run does; the others save to their own prefix and chart to their own
directory beside the default run's. The combined charts go to
`COMPARE_CHART_DIR`. A saved run counts as current if its trace was written
today.
"""

import sys
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from src.models.common.staleness import is_current
from src.models.ustar.analyse import CHART_DIR, run_analysis
from src.models.ustar.cli import build_parser, run_from_args
from src.models.ustar.results import load_results
from src.paths import CHARTS, MODEL_OUTPUTS

OUTPUT_DIR = MODEL_OUTPUTS

# The combined charts only. Each run's own charts go beside the default run's
# directory; see `Specification.chart_dir`.
COMPARE_CHART_DIR = CHARTS / "UStar-compare"

# The comparison runs' prefixes share this stem; a run's own chart directory is
# the default one with the rest of its prefix appended, e.g. UStar-k2.
_PREFIX_STEM = "ustar_sum_"


@dataclass(frozen=True)
class Specification:
    """One comparison specification, and how to reproduce it."""

    label: str
    prefix: str
    flags: list[str]
    colour: str
    style: str
    # The default run: no flags but its prefix, so it is the same run a plain
    # `./run-ustar.sh` makes, and it charts where a plain run does.
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
        return is_current(self.trace_path)

    def refresh(self) -> None:
        """Re-estimate this specification into its own prefix."""
        print(f"  re-running {self.label} ({' '.join(self.flags)})", flush=True)
        run_from_args(build_parser().parse_args([*self.flags, "--prefix", self.prefix, "--no-analyse"]))

    def chart(self) -> None:
        """Write this specification's own charts from its saved run."""
        run_analysis(prefix=self.prefix, chart_dir=self.chart_dir)


# Colour carries the u* structure and knot count.
SPECIFICATIONS: list[Specification] = [
    Specification("Random walk", "ustar", [], "brown", "-", default=True),
    Specification("Spline 1 knot", "ustar_sum_k1",
                  ["--ustar-structure", "spline", "--knots", "2013Q1"], "darkorange", "-"),
    Specification("Spline 2 knots", "ustar_sum_k2",
                  ["--ustar-structure", "spline", "--knots", "1996Q1", "2013Q1"], "tab:blue", "-"),
]


@dataclass
class Loaded:
    """One specification's loaded series and the band the table reports."""

    spec: Specification
    ustar: pd.Series
    implied: pd.Series
    unemployment: pd.Series
    band: float
    extras: dict[str, float] = field(default_factory=dict)


def without_okun(loaded: list[Loaded]) -> list[Loaded]:
    """Return only the specifications that exclude the Okun equation."""
    return [item for item in loaded if not item.spec.prefix.endswith("_okun")]


def load(spec: Specification) -> Loaded:
    """Read one specification's saved run."""
    results = load_results(prefix=spec.prefix)
    posterior = results.ustar_posterior()
    return Loaded(
        spec=spec,
        ustar=posterior.median(axis=1),
        implied=results.implied_ustar(),
        unemployment=pd.Series(results.obs["u"], index=results.obs_index),
        band=float((posterior.quantile(0.95, axis=1) - posterior.quantile(0.05, axis=1)).mean()),
    )


def load_all(*, refresh: bool = True) -> list[Loaded]:
    """Re-estimate anything not from today (if `refresh`), then load every specification."""
    stale = [s for s in SPECIFICATIONS if not s.is_current()]
    if refresh and stale:
        print(f"Re-running {len(stale)} of {len(SPECIFICATIONS)} specifications, about 20s each:")
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
