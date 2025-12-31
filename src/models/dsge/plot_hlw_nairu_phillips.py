"""Plot latent states from HLW-NAIRU-Phillips model."""

from pathlib import Path

import mgplot as mg
import numpy as np
import pandas as pd
from scipy import stats

from src.models.dsge.hlw_nairu_phillips_estimation import estimate_and_extract_latents
from src.data.abs_loader import load_series
from src.data.series_specs import CPI_TRIMMED_MEAN_QUARTERLY

# Chart directory
CHART_DIR = Path(__file__).parent.parent.parent.parent / "charts" / "dsge-hlw-nairu-phillips"

# Plotting constants
START = pd.Period("1985Q1", freq="Q")
RFOOTER = "HLW-NAIRU-Phillips MLE Model"

# Pre-target warning region
PI_TARGET_FULL = pd.Period("1998Q1", freq="Q")

NAIRU_WARN = {
    "axvspan": {
        "xmin": START.ordinal,
        "xmax": PI_TARGET_FULL.ordinal,
        "label": r"Before inflation target fully anchored",
        "color": "goldenrod",
        "alpha": 0.2,
        "zorder": -2,
    }
}

ANNUAL_RANGE = {
    "axhspan": {
        "ymin": 2,
        "ymax": 3,
        "color": "#dddddd",
        "label": "2-3% annual inflation target range",
        "zorder": -1,
    }
}


def setup_charts() -> None:
    """Set up chart directory."""
    mg.set_chart_dir(str(CHART_DIR))
    mg.clear_chart_dir()
    print(f"Charts will be saved to: {CHART_DIR}")


def plot_nairu_and_unemployment(show: bool = True) -> None:
    """Plot NAIRU estimate with unemployment overlay."""
    print("Estimating model...")
    results, df = estimate_and_extract_latents(start="1984Q1", anchor_inflation=True)

    # Filter to start
    df = df[df.index >= START]

    # Build credible interval bands for NAIRU
    cuts = [0.005, 0.025, 0.16]
    alphas = [0.1, 0.2, 0.3]

    ax = None
    for cut, alpha in zip(cuts, alphas):
        z = stats.norm.ppf(1 - cut)
        lower = pd.Series(df["nairu"].values - z * df["nairu_std"].values, index=df.index)
        upper = pd.Series(df["nairu"].values + z * df["nairu_std"].values, index=df.index)
        band = pd.DataFrame({"lower": lower, "upper": upper}, index=df.index)
        ax = mg.fill_between_plot(
            band, ax=ax, color="blue", alpha=alpha,
            label=f"NAIRU {int((1 - 2 * cut) * 100)}% CI", zorder=3,
        )

    # NAIRU median
    nairu_series = pd.Series(df["nairu"].values, index=df.index, name="NAIRU")
    ax = mg.line_plot(nairu_series, ax=ax, color="blue", width=1.5, annotate=True, zorder=4)

    # Unemployment overlay
    U_series = pd.Series(df["U"].values, index=df.index)
    for color, width, label in zip(["white", "brown"], [3, 1.5], ["_", "Unemployment Rate"]):
        U_series.name = label
        mg.line_plot(U_series, ax=ax, color=color, width=width, zorder=5)

    # Load and plot inflation
    inf_series = load_series(CPI_TRIMMED_MEAN_QUARTERLY)
    inflation = inf_series.data
    if not isinstance(inflation.index, pd.PeriodIndex):
        inflation.index = pd.PeriodIndex(inflation.index, freq="Q")
    inflation_annual = ((1 + inflation / 100) ** 4 - 1) * 100
    inflation_f = inflation_annual[inflation_annual.index >= START]

    for color, width, label in zip(["white", "darkorange"], [3, 1.5], ["_", "Inflation rate"]):
        inflation_f.name = label
        mg.line_plot(inflation_f, ax=ax, color=color, width=width, zorder=5)

    mg.finalise_plot(
        ax,
        title="NAIRU Estimate for Australia (HLW-NAIRU-Phillips Model)",
        ylabel="Per cent",
        legend={"loc": "best", "fontsize": "x-small", "ncol": 2},
        lfooter=r"Australia. Combined HLW + NAIRU-Phillips model with Okun-IS hybrid.",
        rfooter=RFOOTER,
        axisbelow=True,
        **ANNUAL_RANGE,
        **NAIRU_WARN,
        show=show,
    )


def plot_rstar(show: bool = True) -> None:
    """Plot r* estimate with actual real rate overlay."""
    print("Estimating model...")
    results, df = estimate_and_extract_latents(start="1984Q1", anchor_inflation=True)

    # Filter to start
    df = df[df.index >= START]

    # Build credible interval bands for r*
    cuts = [0.005, 0.025, 0.16]
    alphas = [0.1, 0.2, 0.3]

    ax = None
    for cut, alpha in zip(cuts, alphas):
        z = stats.norm.ppf(1 - cut)
        lower = pd.Series(df["rstar"].values - z * df["rstar_std"].values, index=df.index)
        upper = pd.Series(df["rstar"].values + z * df["rstar_std"].values, index=df.index)
        band = pd.DataFrame({"lower": lower, "upper": upper}, index=df.index)
        ax = mg.fill_between_plot(
            band, ax=ax, color="darkgreen", alpha=alpha,
            label=f"r* {int((1 - 2 * cut) * 100)}% CI", zorder=3,
        )

    # r* median
    rstar_series = pd.Series(df["rstar"].values, index=df.index, name="r* (Natural Rate)")
    ax = mg.line_plot(rstar_series, ax=ax, color="darkgreen", width=1.5, annotate=True, zorder=4)

    # Actual real rate overlay
    r_series = pd.Series(df["r"].values, index=df.index)
    for color, width, label in zip(["white", "purple"], [3, 1.5], ["_", "Real Interest Rate"]):
        r_series.name = label
        mg.line_plot(r_series, ax=ax, color=color, width=width, zorder=5)

    mg.finalise_plot(
        ax,
        title="Natural Rate of Interest (r*) for Australia",
        ylabel="Per cent",
        legend={"loc": "best", "fontsize": "x-small"},
        lfooter=r"Australia. r* from HLW-NAIRU-Phillips model. Real rate = Cash rate - Inflation.",
        rfooter=RFOOTER,
        axisbelow=True,
        y0=True,
        **NAIRU_WARN,
        show=show,
    )


def plot_gaps(show: bool = True) -> None:
    """Plot unemployment gap and real rate gap."""
    print("Estimating model...")
    results, df = estimate_and_extract_latents(start="1984Q1", anchor_inflation=True)

    # Filter to start
    df = df[df.index >= START]

    # Compute gaps
    u_gap = df["U"] - df["nairu"]
    r_gap = df["r"] - df["rstar"]

    # Plot unemployment gap
    ax = None
    u_gap_series = pd.Series(u_gap.values, index=df.index, name="U-gap (U - NAIRU)")
    ax = mg.line_plot(u_gap_series, ax=ax, color="blue", width=1.5, annotate=True, zorder=4)

    # Plot real rate gap
    r_gap_series = pd.Series(r_gap.values, index=df.index, name="r-gap (r - r*)")
    ax = mg.line_plot(r_gap_series, ax=ax, color="darkgreen", width=1.5, annotate=True, zorder=4)

    mg.finalise_plot(
        ax,
        title="Unemployment Gap and Real Rate Gap for Australia",
        ylabel="Percentage points",
        legend={"loc": "best", "fontsize": "small"},
        lfooter=r"Australia. Positive U-gap = slack. Positive r-gap = restrictive policy.",
        rfooter=RFOOTER,
        axisbelow=True,
        y0=True,
        **NAIRU_WARN,
        show=show,
    )


if __name__ == "__main__":
    setup_charts()
    plot_nairu_and_unemployment(show=False)
    plot_rstar(show=False)
    plot_gaps(show=False)
    print(f"\nCharts saved to: {CHART_DIR}")
