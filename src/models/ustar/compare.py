"""The comparison specifications behind `--compare`, and how to refresh them.

Three settings of this model, differing in the number of spline knots and in
whether the gap-form Okun equation is included. Everything else is shared, so
agreement between them is close to arithmetic and only their disagreement is
informative. MODEL_NOTES, "Comparing specifications", explains the choice of
the three and how to read them.

Each specification is the command-line flags it would be run with, parsed by
the same parser as the default run, so the list doubles as the record of what
each one is. Each writes to its own prefix, so a refresh never touches the
default run's outputs or charts. A saved run counts as current if its trace
was written today.
"""

import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import pandas as pd

from src.models.ustar.cli import build_parser, run_from_args
from src.models.ustar.results import load_results
from src.paths import MODEL_OUTPUTS

OUTPUT_DIR = MODEL_OUTPUTS


@dataclass(frozen=True)
class Specification:
    """One comparison specification, and how to reproduce it."""

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
        run_from_args(build_parser().parse_args([*self.flags, "--prefix", self.prefix]))


# Colour carries the knot count, dashing carries Okun, so the two dimensions
# read separately.
SPECIFICATIONS: list[Specification] = [
    Specification("Spline 1 knot", "ustar_sum_k1",
                  ["--ustar-structure", "spline", "--knots", "2013Q1"], "darkorange", "-"),
    Specification("Spline 2 knots", "ustar_sum_k2",
                  ["--ustar-structure", "spline", "--knots", "1996Q1", "2013Q1"], "tab:blue", "-"),
    Specification("Spline 2 knots, with gap-form Okun", "ustar_sum_k2_okun",
                  ["--ustar-structure", "spline", "--knots", "1996Q1", "2013Q1", "--okun"],
                  "tab:blue", "--"),
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
