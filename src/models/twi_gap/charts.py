"""Chart: the cash rate with the real TWI gap as a backplane."""

import mgplot as mg
import pandas as pd
from matplotlib.axes import Axes

from src.models.twi_gap import DARK_SD, NEUTRAL_SD, START
from src.models.twi_gap.analysis import GapFit

DEARER = "tab:red"
CHEAPER = "tab:blue"
LIGHT_ALPHA = 0.10  # NEUTRAL_SD to DARK_SD
DARK_ALPHA = 0.28  # beyond DARK_SD
SHARE_ROUNDING = -1  # quote band shares to the nearest 10 per cent
PERCENT = 100
RFOOTER = "RBA: A2, F15, I2"


def _months(quarter: pd.Period) -> tuple[int, int]:
    """Return a quarter's span on a monthly axis: its first month's ordinal, and the next quarter's."""
    return quarter.asfreq("M", how="start").ordinal, quarter.asfreq("M", how="end").ordinal + 1


def _shade(ax: Axes, fit: GapFit) -> None:
    """Shade each quarter beyond NEUTRAL_SD on a monthly axis: red dearer, blue cheaper, darker beyond DARK_SD.

    Zero-width key spans lead, so the legend shows all four shades. A quarter
    spans its three months, at monthly period ordinals, the x-coordinates
    mgplot uses for a monthly period axis.
    """
    first = _months(fit.gap.index[0])[0]
    for color, side in ((DEARER, "tighter"), (CHEAPER, "looser")):
        ax.axvspan(first, first, color=color, alpha=LIGHT_ALPHA, lw=0, label=f"{side.capitalize()} FX")
        ax.axvspan(first, first, color=color, alpha=DARK_ALPHA, lw=0, label=f"Much {side} FX")
    for period, value in fit.gap.items():
        if not isinstance(period, pd.Period):
            raise TypeError(f"gap index {period!r} is not a Period")
        z = value / fit.sd
        if abs(z) < NEUTRAL_SD:
            continue
        start, end = _months(period)
        ax.axvspan(
            start,
            end,
            color=DEARER if z > 0 else CHEAPER,
            alpha=DARK_ALPHA if abs(z) >= DARK_SD else LIGHT_ALPHA,
            lw=0,
            zorder=0,
        )


def _share(share: float) -> str:
    """Return a band share as a rounded percentage, e.g. 0.507 -> '50%'."""
    return f"{round(share * PERCENT, SHARE_ROUNDING):.0f}%"


def backplane_chart(cash: pd.Series, fit: GapFit, shares: dict[float, float]) -> None:
    """Draw the monthly cash rate from START as steps, with each quarter shaded by the real TWI gap.

    A dotted line marks where the gap ends: the cash rate runs on past the last
    quarter of real TWI and commodity price data.
    """
    cash = cash[cash.index >= START.asfreq("M", how="start")].rename("Cash rate")
    ax = mg.line_plot(cash, color="black", width=1.5, drawstyle="steps-post", annotate=False)
    _shade(ax, fit)
    model_end = {
        "x": _months(fit.gap.index[-1])[1],
        "color": "grey",
        "linestyle": ":",
        "linewidth": 1.5,
        "label": f"Model end ({fit.gap.index[-1]})",
    }
    mg.finalise_plot(
        ax,
        title="Cash rate and the real TWI gap",
        ylabel="Per cent",
        lheader=(
            f"Real TWI gap from commodity prices. Shaded beyond ±{NEUTRAL_SD:g}σ "
            f"(≈{_share(shares[NEUTRAL_SD])} of observations inside); "
            f"darker beyond ±{DARK_SD:g}σ (≈{_share(shares[DARK_SD])} inside)"
        ),
        lfooter=f"Australia. Cash rate target, monthly; real TWI gap model, quarterly. Gap σ {fit.sd:.1f}%. ",
        rfooter=RFOOTER,
        axvline=model_end,
        legend={"loc": "upper right", "fontsize": "xx-small", "ncol": 2},
        show=False,
    )
