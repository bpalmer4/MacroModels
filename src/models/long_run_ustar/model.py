"""Find the quarters where inflation was flat, and read unemployment off them.

The NAIRU is defined as the unemployment rate at which inflation stops changing.
Every model in this package infers that with a state equation and a likelihood.
This one does not infer it at all: it finds the stretches where inflation was in
fact not changing, reads the unemployment rate over them, and reports what it
saw. There is no estimate here, and no uncertainty band, because nothing is
being estimated.

What that buys is reach. The state-space models start in 1993 because their gap
is defined against an inflation target that did not exist earlier. A rule about
flat inflation needs no target, so it runs back as far as the data, which is
1959Q3 on the unemployment side.

What it costs is everything the models do carefully: no control for supply
shocks, no expectations, no allowance for the lag between slack and prices, and
a reading that is only as good as the flatness rule. The rule is stated in
`ModelConfig` and swept in `sweep.py` for that reason.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.models.long_run_ustar.config import ModelConfig


@dataclass
class Episode:
    """One stretch of flat inflation, and the unemployment rates over it."""

    start: pd.Period
    end: pd.Period
    quarters: int
    inflation: float
    #: Mean unemployment over the window, keyed by the lag it was read at.
    unemployment: dict[int, float]
    #: "plateau" where inflation was level, "trough" where it was at the base
    #: of a wide U. Reported separately: see `ModelConfig.find_troughs`.
    kind: str = "plateau"

    @property
    def decade(self) -> int:
        """The decade the episode starts in, for grouping the table."""
        return self.start.year // 10 * 10


def smooth_inflation(pi: pd.Series, quarters: int) -> pd.Series:
    """Return `pi` under a centred moving average of `quarters`.

    Centred rather than trailing: the question is whether inflation was flat
    *around* a quarter, and a trailing filter would shift every episode later by
    half its window.
    """
    if quarters <= 1:
        return pi
    return pi.rolling(quarters, center=True).mean()


def _slope(values: np.ndarray) -> float:
    """Return the OLS slope per quarter through `values`."""
    x = np.arange(len(values), dtype=float)
    return float(np.polyfit(x, values, 1)[0])


def flat_mask(pi_smooth: pd.Series, config: ModelConfig) -> pd.Series:
    """Return a boolean Series marking quarters at the centre of a flat window.

    Under "range" the window's high-to-low spread must be within tolerance,
    which is the direct reading of "wide-based rather than V-shaped": a turning
    point has a small slope at its centre but a large range around it. "slope"
    is offered as the comparator, and finds more, because it accepts a steady
    climb as long as it is gentle.
    """
    rolling = pi_smooth.rolling(config.window, center=True)
    if config.flatness_rule == "slope":
        measured = rolling.apply(_slope, raw=True).abs()
    else:
        measured = rolling.max() - rolling.min()
    return (measured <= config.tolerance).fillna(value=False)


def _flat_u_mask(frame: pd.DataFrame, config: ModelConfig) -> pd.Series:
    """Return a mask requiring unemployment to be flat too, or all-True if off.

    Judged on the raw rate: unemployment does not carry the quarter-to-quarter
    noise that forces year-ended inflation to be smoothed first, so smoothing it
    would only widen the windows that pass.
    """
    if not config.require_flat_u:
        return pd.Series(data=True, index=frame.index)
    rolling = frame["u"].rolling(config.window, center=True)
    measured = (
        rolling.apply(_slope, raw=True).abs()
        if config.flatness_rule == "slope"
        else rolling.max() - rolling.min()
    )
    return (measured <= config.u_tolerance).fillna(value=False)


def _target_mask(frame: pd.DataFrame, pi_smooth: pd.Series, config: ModelConfig) -> pd.Series:
    """Return a mask enforcing the post-target requirement, or all-True if off."""
    if not config.require_target:
        return pd.Series(data=True, index=frame.index)
    near = (pi_smooth - config.target).abs() <= config.target_tolerance
    before = frame.index < pd.Period(config.target_from, freq="Q")
    return pd.Series(before, index=frame.index) | near.fillna(value=False)


def find_episodes(frame: pd.DataFrame, config: ModelConfig) -> list[Episode]:
    """Return the flat-inflation episodes in `frame`, longest-run first in time.

    `frame` carries `pi` and `u` on a quarterly PeriodIndex.
    """
    pi_smooth = smooth_inflation(frame["pi"], config.smooth)
    mask = (
        flat_mask(pi_smooth, config)
        & _flat_u_mask(frame, config)
        & _target_mask(frame, pi_smooth, config)
    )

    runs = (mask != mask.shift()).cumsum()
    episodes: list[Episode] = []
    for index in frame[mask].groupby(runs[mask]).groups.values():
        quarters = pd.PeriodIndex(index)
        if len(quarters) < config.min_quarters:
            continue
        unemployment = {
            lag: float(frame["u"].reindex(quarters - lag).mean()) for lag in config.lags
        }
        episodes.append(Episode(
            start=quarters.min(),
            end=quarters.max(),
            quarters=len(quarters),
            inflation=float(pi_smooth.reindex(quarters).mean()),
            unemployment=unemployment,
        ))
    return episodes


def episode_frame(episodes: list[Episode], config: ModelConfig) -> pd.DataFrame:
    """Return the episodes as a DataFrame, one row each, for printing and saving."""
    rows = [
        {
            "start": str(e.start),
            "end": str(e.end),
            "quarters": e.quarters,
            "inflation": e.inflation,
            **{f"u_lag{lag}": e.unemployment[lag] for lag in config.lags},
        }
        for e in episodes
    ]
    return pd.DataFrame(rows)


def trough_mask(pi_smooth: pd.Series, config: ModelConfig) -> pd.Series:
    """Return a mask marking quarters at the base of a wide U in inflation.

    Three conditions, and the third is what separates a U from a step or the
    flat part of a long descent: the quarter sits within tolerance of its
    window's minimum, and the window's left and right edges are both above it.
    """
    half = config.trough_window // 2
    rolling = pi_smooth.rolling(config.trough_window, center=True)
    at_base = (pi_smooth - rolling.min()) <= config.trough_tolerance
    rises_left = pi_smooth.shift(half) > pi_smooth + config.trough_tolerance
    rises_right = pi_smooth.shift(-half) > pi_smooth + config.trough_tolerance
    return (at_base & rises_left & rises_right).fillna(value=False)


def find_troughs(frame: pd.DataFrame, config: ModelConfig) -> list[Episode]:
    """Return the U-shaped troughs, as episodes tagged "trough".

    Excludes any quarter the plateau rule already claimed, so the two lists do
    not double-count the stretches that are both flat and at a minimum.
    """
    if not config.find_troughs:
        return []

    pi_smooth = smooth_inflation(frame["pi"], config.smooth)
    mask = trough_mask(pi_smooth, config)
    flat = flat_mask(pi_smooth, config)

    runs = (mask != mask.shift()).cumsum()
    troughs: list[Episode] = []
    for index in frame[mask].groupby(runs[mask]).groups.values():
        quarters = pd.PeriodIndex(index)
        if len(quarters) < config.trough_min_base:
            continue
        # Dropped only when the plateau rule already covers the whole trough.
        # Removing the overlapping quarters first instead would split a U in
        # two and lose both halves to the minimum-length test: that is what
        # happened to 1962Q3-1963Q2, whose 1963Q1 is also a plateau quarter.
        if bool(flat.reindex(quarters).all()):
            continue
        troughs.append(Episode(
            start=quarters.min(),
            end=quarters.max(),
            quarters=len(quarters),
            inflation=float(pi_smooth.reindex(quarters).mean()),
            unemployment={
                lag: float(frame["u"].reindex(quarters - lag).mean()) for lag in config.lags
            },
            kind="trough",
        ))
    return troughs


@dataclass
class StationaryWindow:
    """A stretch where unemployment held still, whatever inflation was doing."""

    start: pd.Period
    end: pd.Period
    quarters: int
    unemployment: float
    inflation: float
    #: Share of the window's quarters that also passed the inflation-flatness test.
    flat_share: float
    #: What inflation was mostly doing across it.
    direction: str

    @property
    def counted(self) -> bool:
        """Whether the window contributed to a reading, i.e. inflation was flat too."""
        return self.flat_share >= 0.5  # noqa: PLR2004 — a majority of the window


def stationary_u_windows(frame: pd.DataFrame, config: ModelConfig) -> list[StationaryWindow]:
    """Return the stretches where unemployment was flat, passed or not.

    The complement of `find_episodes`: that one asks where inflation held still,
    this one asks where unemployment did, and the interesting rows are the ones
    that appear here and not there. A stationary low unemployment rate alongside
    inflation that was still moving is not a u* reading, and this is how to see
    which levels were excluded and why.
    """
    pi_smooth = smooth_inflation(frame["pi"], config.smooth)
    rolling = frame["u"].rolling(config.window, center=True)
    measured = (
        rolling.apply(_slope, raw=True).abs()
        if config.flatness_rule == "slope"
        else rolling.max() - rolling.min()
    )
    mask = (measured <= config.u_tolerance).fillna(value=False)

    flat_pi = flat_mask(pi_smooth, config)
    direction = inflation_direction(pi_smooth, config)

    runs = (mask != mask.shift()).cumsum()
    windows: list[StationaryWindow] = []
    for index in frame[mask].groupby(runs[mask]).groups.values():
        quarters = pd.PeriodIndex(index)
        if len(quarters) < config.min_quarters:
            continue
        labels = direction.reindex(quarters).dropna()
        windows.append(StationaryWindow(
            start=quarters.min(),
            end=quarters.max(),
            quarters=len(quarters),
            unemployment=float(frame["u"].reindex(quarters).mean()),
            inflation=float(pi_smooth.reindex(quarters).mean()),
            flat_share=float(flat_pi.reindex(quarters).mean()),
            direction=str(labels.mode().iloc[0]) if not labels.empty else "-",
        ))
    return windows


def inflation_direction(pi_smooth: pd.Series, config: ModelConfig) -> pd.Series:
    """Label each quarter by what inflation was doing around it.

    "rising", "flat" or "falling", from the change across the centred window,
    with the same tolerance the flatness rule uses so the three states partition
    the sample rather than overlapping it.
    """
    change = pi_smooth.rolling(config.window, center=True).apply(
        lambda values: values[-1] - values[0], raw=True,
    )
    half = config.tolerance / 2.0
    labels = np.where(change > half, "rising", np.where(change < -half, "falling", "flat"))
    state = pd.Series(labels, index=pi_smooth.index, dtype=object)
    return state.where(change.notna())


def direction_contrast(frame: pd.DataFrame, config: ModelConfig) -> pd.DataFrame:
    """Return mean unemployment by inflation direction, per decade and overall.

    This is the model's test rather than its reading. If unemployment carries any
    information about inflation's direction, the ordering should run
    rising < flat < falling: a labour market tighter than u* pushes inflation up,
    one looser pulls it down. Where the ordering breaks, something other than
    demand was moving prices, which is the headline measure's known weakness
    made visible rather than asserted.
    """
    state = inflation_direction(smooth_inflation(frame["pi"], config.smooth), config)
    table = pd.DataFrame({
        "u": frame["u"],
        "state": state,
        "decade": [period.year // 10 * 10 for period in frame.index],
    }).dropna()

    rows = []
    for decade, group in table.groupby("decade"):
        means = group.groupby("state")["u"].mean()
        counts = group.groupby("state").size()
        rows.append({
            "period": f"{decade}s",
            **{f"u_{s}": float(means.get(s, float("nan"))) for s in ("rising", "flat", "falling")},
            **{f"n_{s}": int(counts.get(s, 0)) for s in ("rising", "flat", "falling")},
        })
    means = table.groupby("state")["u"].mean()
    counts = table.groupby("state").size()
    rows.append({
        "period": "whole",
        **{f"u_{s}": float(means.get(s, float("nan"))) for s in ("rising", "flat", "falling")},
        **{f"n_{s}": int(counts.get(s, 0)) for s in ("rising", "flat", "falling")},
    })

    result = pd.DataFrame(rows).set_index("period")
    # Read out as floats rather than off `itertuples`, whose attributes are typed
    # as the union of every column's dtype and so compare against anything.
    result["ordering"] = [
        _ordering(float(row["u_rising"]), float(row["u_flat"]), float(row["u_falling"]))
        for _period, row in result.iterrows()
    ]
    return result


def _ordering(rising: float, flat: float, falling: float) -> str:
    """Name the ordering of the three means, for the contrast table's last column."""
    if rising < flat < falling:
        return "rise<flat<fall"
    return "rise<fall" if rising < falling else "BREAKS"


def reading(episodes: list[Episode], lag: int = 0, *, weighted: bool = True) -> float:
    """Return the single u* reading implied by `episodes`, in per cent.

    Weighted by episode length by default, so a five-year stretch of stable
    inflation counts for more than a two-quarter one. This is a summary of a
    period, not an estimate: episodes decades apart are averaged only when the
    caller has a reason to believe the answer did not move between them.
    """
    if not episodes:
        return float("nan")
    values = np.array([e.unemployment[lag] for e in episodes], dtype=float)
    weights = np.array([e.quarters for e in episodes], dtype=float) if weighted else None
    return float(np.average(values, weights=weights))
