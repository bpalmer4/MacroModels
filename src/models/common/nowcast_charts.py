"""Shared headline charts for the GDP nowcast suite.

Every GDP nowcast model (bridge, DFM, BVAR, components) produces the same three
charts from a Q/Q nowcast and its annual (through-the-year) re-expression:

  1. Q/Q fan      — q/q GDP growth history + nowcast + 70/90% CI fan
  2. TTY fan      — through-the-year growth history + nowcast + 70/90% CI fan
  3. Combined     — annual growth line + quarterly growth bars

This module is the single source of that chart code (previously copy-pasted into
each model). A model fills a :class:`NowcastChartSpec` and calls
:func:`plot_nowcast_charts`; chart-directory management stays with the caller's
``run_nowcast``. The Q/Q→annual conversion used to fill the spec lives alongside
the other shared helpers in ``nowcast_core`` (:func:`~nowcast_core.compute_tty`).

Per-model differences are parameters: ``model_label`` (titles/footer) and
``accent`` (CI-fan colour — BVAR green, bridge/DFM red, components goldenrod).
"""

from dataclasses import dataclass, field

import mgplot as mg
import numpy as np
import pandas as pd


@dataclass
class NowcastChartSpec:
    """Inputs for the three standard nowcast charts.

    ``gdp`` is the CVM SA GDP **level** series (ending at the last published
    quarter); both fan histories and the combined chart are derived from it. The
    CI fields are ``(lower, upper)`` tuples in the same units as ``gdp_qoq`` /
    ``gdp_tty`` (per cent).
    """

    model_label: str          # e.g. "BVAR", "DFM", "Bridge Model", "Components"
    target_quarter: pd.Period
    gdp: pd.Series
    gdp_qoq: float
    gdp_tty: float
    gdp_qoq_70: tuple[float, float]
    gdp_qoq_90: tuple[float, float]
    gdp_tty_70: tuple[float, float]
    gdp_tty_90: tuple[float, float]
    accent: str = "green"     # CI-fan colour
    source: str = "Source: ABS 5206.0"
    show: bool = False
    n_history: int = 20
    date: str = field(default_factory=lambda: pd.Timestamp.now().strftime("%d %b %Y"))


def _fan(
    hist: pd.Series,
    nowcast_value: float,
    ci_70: tuple[float, float],
    ci_90: tuple[float, float],
    target_quarter: pd.Period,
    *,
    title: str,
    lfooter: str,
    accent: str,
    source: str,
    show: bool,
    n_history: int,
) -> None:
    """Fan chart: history line + dashed nowcast + 70/90% CI fan (mgplot layering)."""
    # Drop the target from history (hindcast case) so the nowcast line doesn't
    # create a duplicate index point.
    hist_excluding_target = hist.loc[hist.index < target_quarter]
    recent = hist_excluding_target.iloc[-n_history:]

    nowcast_idx = pd.PeriodIndex([recent.index[-1], target_quarter], freq="Q-DEC")
    nowcast_line = pd.Series([recent.iloc[-1], nowcast_value], index=nowcast_idx)
    band_90 = pd.DataFrame({"lower": [recent.iloc[-1], ci_90[0]],
                            "upper": [recent.iloc[-1], ci_90[1]]}, index=nowcast_idx)
    band_70 = pd.DataFrame({"lower": [recent.iloc[-1], ci_70[0]],
                            "upper": [recent.iloc[-1], ci_70[1]]}, index=nowcast_idx)

    recent = recent.rename("GDP growth")
    nowcast_line = nowcast_line.rename("Nowcast")

    ax = mg.fill_between_plot(band_90, color=accent, alpha=0.1, label="90% CI")
    mg.fill_between_plot(band_70, ax=ax, color=accent, alpha=0.2, label="70% CI")
    mg.line_plot(recent, ax=ax, color=["navy"], width=2)
    mg.line_plot(nowcast_line, ax=ax, color=[accent], width=2, style="--", annotate=True, rounding=2)

    mg.finalise_plot(
        ax,
        title=title,
        ylabel="Per cent",
        rfooter=source,
        lfooter=lfooter,
        y0=True,
        legend={"loc": "best", "fontsize": "x-small"},
        show=show,
    )


def plot_nowcast_charts(spec: NowcastChartSpec) -> None:
    """Draw the Q/Q fan, TTY fan, and combined charts into the current chart dir."""
    gdp = spec.gdp.dropna()
    label = spec.model_label
    t = spec.target_quarter

    qoq_hist = (np.log(gdp).diff() * 100).dropna()
    tty_hist = ((gdp / gdp.shift(4)) - 1) * 100

    _fan(
        qoq_hist, spec.gdp_qoq, spec.gdp_qoq_70, spec.gdp_qoq_90, t,
        title=f"GDP Growth {label} Nowcast (Q/Q): {t}",
        lfooter=f"Australia. {label} nowcast: {spec.gdp_qoq:+.2f}%. {spec.date}. ",
        accent=spec.accent, source=spec.source, show=spec.show, n_history=spec.n_history,
    )
    _fan(
        tty_hist.dropna(), spec.gdp_tty, spec.gdp_tty_70, spec.gdp_tty_90, t,
        title=f"GDP Growth {label} Nowcast (TTY): {t}",
        lfooter=f"Australia. {label} nowcast: {spec.gdp_tty:+.2f}%. {spec.date}. ",
        accent=spec.accent, source=spec.source, show=spec.show, n_history=spec.n_history,
    )

    # Combined: annual line + quarterly bars, nowcast appended at the target.
    qoq = qoq_hist.loc[qoq_hist.index < t].copy()
    tty = tty_hist.dropna().loc[tty_hist.dropna().index < t].copy()
    qoq.loc[t] = spec.gdp_qoq
    tty.loc[t] = spec.gdp_tty
    growth_df = pd.DataFrame({
        "Annual Growth": tty,        # first column → line
        "Quarterly Growth": qoq,     # second column → bars
    }).iloc[-spec.n_history:]
    # Highlight the nowcast quarter behind the final bar (mgplot positions bars by
    # Period ordinal), matching the components stacked-contributions chart.
    forecast_pos = t.ordinal
    mg.growth_plot_finalise(
        growth_df,
        title=f"GDP Growth {label} Nowcast: {t}",
        ylabel="Per cent",
        rfooter=spec.source,
        lfooter=f"Australia. {label} Q/Q: {spec.gdp_qoq:+.2f}%, TTY: {spec.gdp_tty:+.2f}%. "
                f"{spec.date}. ",
        legend={"loc": "best", "fontsize": "x-small"},
        axvspan={"xmin": forecast_pos - 0.5, "xmax": forecast_pos + 0.5,
                 "color": "goldenrod", "alpha": 0.25, "zorder": 0},
        show=spec.show,
    )
