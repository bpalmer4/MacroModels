"""Configuration for the long-run u* reading.

The model has no likelihood and no priors. What it has instead is a rule, and
the rule has three numbers in it. They are collected here so a run states them,
and so the sweep can move them one at a time.
"""

from dataclasses import dataclass
from pathlib import Path

DEFAULT_OUTPUT_DIR = Path("model_outputs")
DEFAULT_CHART_BASE = Path("charts")

# How the flatness of inflation is judged over a window.
#   "range"  — max minus min of the smoothed series, in percentage points
#   "slope"  — absolute OLS slope per quarter, in percentage points
FLATNESS_RULES = ("range", "slope")


@dataclass
class ModelConfig:
    """The flat-stretch rule, and the sample it is applied to."""

    # --- Sample ---
    # 1959Q3 is where the quarterly unemployment rate starts (1364.0.15.003).
    # The CPI reaches 1949Q4, so the binding constraint is unemployment, and
    # extending earlier needs the RBA OP8 CES backcast. See MODEL_NOTES.
    start: str | None = None
    end: str | None = None

    # --- The rule ---
    # Quarters of centred moving average applied to year-ended inflation before
    # flatness is judged. Without it the rule finds nothing before 2002: headline
    # year-ended inflation carries enough quarter-to-quarter noise that no window
    # in the 1960s or 1970s is flat by any tolerance that means anything later.
    # 4 is one year, which is also the horizon the inflation measure already
    # spans, so it smooths the noise without reaching across a cycle.
    smooth: int = 4

    # Length of the centred window over which inflation must be flat. Two years
    # is long enough that a V-shaped turning point fails the test, which is the
    # distinction the whole exercise rests on.
    window: int = 8

    # How flat is flat, in percentage points of the smoothed series.
    tolerance: float = 1.0

    flatness_rule: str = "range"

    # --- U-shaped troughs, the second category ---
    #
    # The flatness rule above finds plateaus: stretches where inflation was
    # level, wherever that level was. It does not find the base of a wide U,
    # because a broad trough's sides rise enough to break the range test over
    # `window` quarters. Six such troughs exist in the sample and the plateau
    # rule catches none of them.
    #
    # A quarter is a trough base when the smoothed series sits within
    # `trough_tolerance` of its minimum over `trough_window`, and both edges of
    # that window are higher, which is what makes it a U rather than a step or
    # the flat part of a descent.
    #
    # **They are reported separately, and should be.** An inflation trough
    # usually arrives at the end of a disinflation, when unemployment is at its
    # worst: four of the six sit at 6.2, 8.0, 8.6 and 10.8 per cent. Merging
    # them into the plateau readings would average a cyclical artefact into an
    # estimate. Kept because the sixth, 1962Q3-1963Q2 at 2.21, is a genuine
    # early-1960s reading, and because seeing the bias is better than not
    # collecting it.
    find_troughs: bool = True
    trough_window: int = 12
    trough_tolerance: float = 0.75
    trough_min_base: int = 3

    # Episodes shorter than this are dropped. A single qualifying quarter is a
    # coincidence of the window's placement rather than a period of stable
    # inflation, and it would carry the same weight in the table as a five-year
    # stretch.
    min_quarters: int = 2

    # Quarters by which unemployment is read ahead of the flat window. Inflation
    # responds to slack with a lag, so the rate that produced a flat stretch may
    # be the one before it. The lag is not identified on Australian data (see
    # `ystar` items 7, 9 and 12), so the model reports a spread rather than
    # choosing: 0 is contemporaneous, 4 reads unemployment a year earlier.
    lags: tuple[int, ...] = (0, 2, 4)

    # Require unemployment to be flat across the window as well as inflation.
    #
    # This is the one correction that works over the whole sample, and it is the
    # difference between "inflation had settled" and "both had settled". A window
    # where inflation is flat while unemployment moves half a point a year is not
    # an equilibrium reading, it is a moving rate that happened to cross one. The
    # 1993Q1-1994Q1 episode is the case in point: inflation flat at 1.8% while
    # unemployment fell 10.93 to 10.33.
    #
    # Judged by the same `flatness_rule` on the raw unemployment rate, which is
    # far less noisy than year-ended inflation and so needs no smoothing. Off by
    # default: the unfiltered table is what shows the rule's own behaviour.
    require_flat_u: bool = False
    u_tolerance: float = 0.5

    # Post-1993 the target binds, so flat inflation away from it is not an
    # equilibrium reading but a failure to hit the target. With this set, flat
    # stretches from `target_from` are additionally required to sit within
    # `target_tolerance` of `target`. Off by default: the unfiltered table is
    # what shows the rule's own behaviour, and the filter is a judgement about
    # regimes rather than a property of the data.
    require_target: bool = False
    target: float = 2.5
    target_from: str = "1993Q1"
    target_tolerance: float = 1.0

    # --- Output ---
    prefix: str = "long_run_ustar"
    output_dir: Path = DEFAULT_OUTPUT_DIR

    def __post_init__(self) -> None:
        """Validate the rule's numbers."""
        if self.flatness_rule not in FLATNESS_RULES:
            raise ValueError(
                f"flatness_rule must be one of {FLATNESS_RULES}, got {self.flatness_rule!r}",
            )
        if self.window < 2:  # noqa: PLR2004 — a window of one quarter is not a window
            raise ValueError(f"window must be at least 2 quarters, got {self.window}")
        if self.smooth < 1:
            raise ValueError(f"smooth must be at least 1 quarter, got {self.smooth}")
        if self.tolerance <= 0:
            raise ValueError(f"tolerance must be positive, got {self.tolerance}")
        if any(lag < 0 for lag in self.lags):
            raise ValueError(f"lags must be non-negative, got {self.lags}")

    @property
    def constants(self) -> dict[str, object]:
        """The rule, for recording alongside a run."""
        recorded: dict[str, object] = {
            "smooth": self.smooth,
            "window": self.window,
            "tolerance": self.tolerance,
            "flatness_rule": self.flatness_rule,
            "min_quarters": self.min_quarters,
            "lags": list(self.lags),
            "require_flat_u": self.require_flat_u,
            "require_target": self.require_target,
        }
        if self.require_flat_u:
            recorded["u_tolerance"] = self.u_tolerance
        if self.require_target:
            recorded["target"] = self.target
            recorded["target_from"] = self.target_from
            recorded["target_tolerance"] = self.target_tolerance
        return recorded
