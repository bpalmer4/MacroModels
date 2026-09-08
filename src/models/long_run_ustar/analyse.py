"""Charts and printed tables for the long-run u* reading."""

from pathlib import Path  # noqa: TC003 — used at runtime in function signatures
from typing import Any

import mgplot as mg
import pandas as pd

from src.models.common.sources import SourceSet
from src.models.long_run_ustar.config import DEFAULT_CHART_BASE, ModelConfig
from src.models.long_run_ustar.model import (
    Episode,
    direction_contrast,
    episode_frame,
    find_troughs,
    reading,
    smooth_inflation,
    stationary_u_windows,
)
from src.models.long_run_ustar.sweep import sweep_all

CHART_DIR = DEFAULT_CHART_BASE / "LongRunUStar"

_LFOOTER = "Australia. Long-run u* reading. "

# Shading for the flat-inflation windows. Deliberately not the orange the other
# models use for their weakly-identified early sample: here the shaded quarters
# are the ones being believed, not the ones being warned about.
_EPISODE_SPAN: dict[str, Any] = {"color": "seagreen", "alpha": 0.13}

# Troughs get their own colour, because they are a different claim: the base of
# a wide U rather than a plateau, and skewed toward high unemployment.
_TROUGH_SPAN: dict[str, Any] = {"color": "mediumpurple", "alpha": 0.16}


def _spans(
    episodes: list[Episode],
    style: dict[str, Any] | None = None,
    label: str = "Inflation flat",
) -> list[dict[str, Any]]:
    """Return axvspan dicts for the episodes, the first carrying the legend label."""
    style = style if style is not None else _EPISODE_SPAN
    spans = []
    for i, episode in enumerate(episodes):
        span = {"xmin": episode.start, "xmax": episode.end, **style}
        if i == 0:
            span["label"] = label
        spans.append(span)
    return spans


def _step(frame: pd.DataFrame, episodes: list[Episode], lag: int) -> pd.Series:
    """Return a series holding each episode's mean unemployment over its own span.

    NaN everywhere else, so plotting it draws one horizontal segment per episode
    and nothing in between. That absence is the point: the rule says nothing
    about the quarters where inflation was moving.
    """
    step = pd.Series(float("nan"), index=frame.index)
    for episode in episodes:
        step.loc[episode.start:episode.end] = episode.unemployment[lag]
    return step


def print_tables(
    frame: pd.DataFrame,
    episodes: list[Episode],
    config: ModelConfig,
) -> None:
    """Print the episodes, the decade readings, and the sweeps."""
    print(f"\nSample: {frame.index.min()} to {frame.index.max()}  ({len(frame)} quarters)")
    print(
        f"Rule: {config.flatness_rule} <= {config.tolerance:g}pp over {config.window} quarters, "
        f"inflation smoothed {config.smooth}q"
        + (f", unemployment flat within {config.u_tolerance:g}pp" if config.require_flat_u else ""),
    )

    print("\nFlat-inflation episodes")
    print("-" * 70)
    table = episode_frame(episodes, config)
    if table.empty:
        print("  none: no window met the rule")
        return
    print(table.to_string(index=False, float_format=lambda v: f"{v:.2f}"))

    troughs = find_troughs(frame, config)
    if troughs:
        print("\nU-shaped troughs, reported separately")
        print("-" * 70)
        print(episode_frame(troughs, config).to_string(
            index=False, float_format=lambda v: f"{v:.2f}"))
        print(
            "\n  An inflation trough usually arrives at the end of a disinflation, when\n"
            "  unemployment is at its worst, so these skew high and are NOT merged above.",
        )

    print("\nReading by decade (length-weighted, unemployment at lag 0)")
    print("-" * 70)
    for decade in sorted({e.decade for e in episodes}):
        chosen = [e for e in episodes if e.decade == decade]
        quarters = sum(e.quarters for e in chosen)
        print(f"  {decade}s   u* ~ {reading(chosen):.2f}%   from {len(chosen)} episode(s), "
              f"{quarters} quarters")
    print(
        "\n  No single number is printed across decades. The readings are 1.8 in the 1960s and "
        "5.4 in the 2010s;\n  averaging them would assert the constancy this model exists to "
        "test.",
    )

    print("\nUnemployment by what inflation was doing (the model's test, not its reading)")
    print("-" * 70)
    print(direction_contrast(frame, config).to_string(float_format=lambda v: f"{v:.2f}"))
    print(
        "\n  rising < flat < falling is the Phillips ordering. Where it breaks, something other\n"
        "  than demand was moving prices.",
    )

    print("\nSensitivity to the rule")
    print("-" * 70)
    print(
        "  Read the episode counts and the earliest date, not the readings. Within a selected\n"
        "  window unemployment is itself flat, so the reading cannot move much whatever the\n"
        "  rule does; the informative column is which episodes appear at all.",
    )
    for param, result in sweep_all(frame, config).items():
        print(f"\n  moving {param}:")
        print(result.to_string(float_format=lambda v: f"{v:.2f}"))


def plot_episodes_on_inflation(
    frame: pd.DataFrame,
    episodes: list[Episode],
    config: ModelConfig,
    footer: str,
) -> None:
    """Inflation, raw and smoothed, with the flat windows shaded."""
    data = pd.DataFrame({
        "Headline CPI, year-ended": frame["pi"],
        f"Smoothed {config.smooth}q": smooth_inflation(frame["pi"], config.smooth),
    })
    mg.line_plot_finalise(
        data,
        color=["darkgrey", "darkblue"],
        width=[1.0, 2.0],
        annotate=False,
        title="Where inflation stopped moving",
        ylabel="Per cent per year",
        legend={"loc": "best", "fontsize": "small"},
        axvspan=_spans(episodes) + _spans(
            find_troughs(frame, config), _TROUGH_SPAN, "Base of a wide U",
        ),
        axhline={"y": 0, "color": "black", "linewidth": 0.5},
        lfooter=_LFOOTER + f"Flat: {config.flatness_rule} <= {config.tolerance:g}pp "
                           f"over {config.window}q. ",
        rfooter=footer,
        show=False,
    )


def plot_readings(
    frame: pd.DataFrame,
    episodes: list[Episode],
    config: ModelConfig,
    footer: str,
) -> None:
    """Draw the unemployment rate, with each flat episode's mean across its own span."""
    lag = config.lags[0]
    troughs = find_troughs(frame, config)
    data = pd.DataFrame({
        "Unemployment rate": frame["u"],
        "Read off flat inflation": _step(frame, episodes, lag),
        "Read off the base of a U": _step(frame, troughs, lag),
    })
    mg.line_plot_finalise(
        data,
        color=["black", "darkorange", "mediumpurple"],
        width=[1.2, 3.0, 3.0],
        style=["-", "-", "-"],
        annotate=False,
        dropna=False,
        title="What inflation says u* was, two ways",
        ylabel="Per cent",
        legend={"loc": "best", "fontsize": "small"},
        axvspan=_spans(episodes) + _spans(troughs, _TROUGH_SPAN, "Base of a wide U"),
        lheader="Purple are U-bases: informative, but they skew to high unemployment",
        lfooter=_LFOOTER + f"Unemployment read at lag {lag}. ",
        rfooter=footer,
        show=False,
    )


def plot_stationary_u(
    frame: pd.DataFrame,
    config: ModelConfig,
    footer: str,
) -> None:
    """Every stretch where unemployment held still, marked by whether it counted.

    Green where inflation was flat too, so the window produced a reading. Red
    where it did not, which is the chart's point: those are levels unemployment
    actually held, excluded because inflation was still moving at the time.
    """
    windows = stationary_u_windows(frame, config)
    counted = pd.Series(float("nan"), index=frame.index)
    excluded = pd.Series(float("nan"), index=frame.index)
    for window in windows:
        target = counted if window.counted else excluded
        target.loc[window.start:window.end] = window.unemployment

    mg.line_plot_finalise(
        pd.DataFrame({
            "Unemployment rate": frame["u"],
            "Held still, inflation flat too": counted,
            "Held still, inflation still moving": excluded,
        }),
        color=["black", "seagreen", "indianred"],
        width=[1.0, 4.0, 4.0],
        style=["-", "-", "-"],
        annotate=False,
        dropna=False,
        title="Where unemployment held still, and whether it counted",
        ylabel="Per cent",
        legend={"loc": "best", "fontsize": "small"},
        lheader="Red bars are levels unemployment actually held while inflation was still moving",
        lfooter=_LFOOTER + f"Unemployment flat within {config.u_tolerance:g}pp "
                           f"over {config.window}q. ",
        rfooter=footer,
        show=False,
    )


def plot_direction_contrast(
    frame: pd.DataFrame,
    config: ModelConfig,
    footer: str,
) -> None:
    """Mean unemployment by inflation direction, per decade.

    The chart the model is actually for. Where the bars run low-to-high left to
    right, unemployment carried information about where inflation was going.
    """
    contrast = direction_contrast(frame, config).drop(index="whole")
    data = contrast[["u_rising", "u_flat", "u_falling"]].rename(columns={
        "u_rising": "Inflation rising",
        "u_flat": "Inflation flat",
        "u_falling": "Inflation falling",
    })
    mg.bar_plot_finalise(
        data,
        stacked=False,
        color=["indianred", "darkorange", "steelblue"],
        annotate=True,
        rounding=1,
        title="Unemployment by what inflation was doing",
        ylabel="Per cent",
        xlabel="",
        legend={"loc": "best", "fontsize": "small"},
        lheader="Low to high, left to right, is the Phillips ordering",
        lfooter=_LFOOTER + f"Direction over {config.window}q, +/-{config.tolerance / 2:g}pp. ",
        rfooter=footer,
        show=False,
    )


def run_analysis(
    frame: pd.DataFrame,
    episodes: list[Episode],
    config: ModelConfig,
    sources: SourceSet,
    chart_dir: Path | str | None = None,
) -> None:
    """Print the tables and write the charts."""
    print_tables(frame, episodes, config)

    if chart_dir is None:
        chart_dir = CHART_DIR
    mg.set_chart_dir(str(chart_dir))
    mg.clear_chart_dir()

    footer = sources.footer()
    plot_episodes_on_inflation(frame, episodes, config, footer)
    plot_readings(frame, episodes, config, footer)
    plot_stationary_u(frame, config, footer)
    plot_direction_contrast(frame, config, footer)

    print(f"\nCharts written to: {chart_dir}")
