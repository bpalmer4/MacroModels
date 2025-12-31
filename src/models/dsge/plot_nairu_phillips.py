"""Plot NAIRU estimates from the NAIRU-Phillips model."""

from pathlib import Path

import mgplot as mg
import numpy as np
import pandas as pd
from scipy import stats

from src.models.dsge.nairu_phillips_model import (
    NairuPhillipsParameters,
    extract_nairu_estimates,
)
from src.models.dsge.nairu_phillips_estimation import (
    load_nairu_phillips_data,
    estimate_nairu_phillips,
    estimate_and_extract_nairu_regimes,
)
from src.data.abs_loader import load_series
from src.data.series_specs import CPI_TRIMMED_MEAN_QUARTERLY

# Chart directory
CHART_DIR = Path(__file__).parent.parent.parent.parent / "charts" / "dsge-nairu-phillips"

# Plotting constants
START = pd.Period("1985Q1", freq="Q")
RFOOTER = "NAIRU-Phillips MLE Model"

# Pre-target warning region
PI_TARGET_START = pd.Period("1993Q1", freq="Q")
PI_TARGET_FULL = pd.Period("1998Q1", freq="Q")

NAIRU_WARN = {
    "axvspan": {
        "xmin": START.ordinal,
        "xmax": PI_TARGET_FULL.ordinal,
        "label": r"NAIRU ($U^*$) WRT $\pi^e$ (before inflation target fully anchored)",
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

ANNUAL_TARGET = {
    "axhline": {
        "y": 2.5,
        "linestyle": "dashed",
        "linewidth": 0.75,
        "color": "darkred",
        "label": "2.5% annual inflation target",
    }
}


def setup_charts() -> None:
    """Set up chart directory."""
    mg.set_chart_dir(str(CHART_DIR))
    mg.clear_chart_dir()
    print(f"Charts will be saved to: {CHART_DIR}")


def plot_nairu_and_unemployment(show: bool = True, regime_switching: bool = True) -> None:
    """Plot NAIRU estimate with unemployment and inflation overlay."""

    if regime_switching:
        print("Estimating by regime...")
        results, nairu_df = estimate_and_extract_nairu_regimes(
            start="1984Q1", anchor_inflation=True
        )

        # Filter to start
        nairu_df = nairu_df[nairu_df.index >= START]

        # Build credible interval bands
        cuts = [0.005, 0.025, 0.16]
        alphas = [0.1, 0.2, 0.3]

        ax = None
        for cut, alpha in zip(cuts, alphas):
            z = stats.norm.ppf(1 - cut)
            lower = pd.Series(
                nairu_df["nairu"].values - z * nairu_df["nairu_std"].values,
                index=nairu_df.index
            )
            upper = pd.Series(
                nairu_df["nairu"].values + z * nairu_df["nairu_std"].values,
                index=nairu_df.index
            )
            band = pd.DataFrame({"lower": lower, "upper": upper}, index=nairu_df.index)
            ax = mg.fill_between_plot(
                band,
                ax=ax,
                color="blue",
                alpha=alpha,
                label=f"NAIRU {int((1 - 2 * cut) * 100)}% Credible Interval",
                zorder=3,
            )

        # NAIRU median
        nairu_series = pd.Series(nairu_df["nairu"].values, index=nairu_df.index, name="NAIRU Median")
        ax = mg.line_plot(nairu_series, ax=ax, color="blue", width=1, annotate=True, zorder=4)

        # Unemployment overlay
        U_series = pd.Series(nairu_df["U"].values, index=nairu_df.index)
        for color, width, label in zip(["white", "brown"], [3, 1.5], ["_", "Unemployment Rate"]):
            U_series.name = label
            mg.line_plot(U_series, ax=ax, color=color, width=width, zorder=5)

    else:
        # Single estimation (original code path)
        print("Loading data...")
        y, U_obs, import_prices, delta_U_over_U, oil_change, coal_change, dates = load_nairu_phillips_data(
            start="1984Q1", end=None, anchor_inflation=True
        )

        nairu_prior = np.mean(U_obs)
        print(f"Estimating single model...")
        result = estimate_nairu_phillips(
            y, U_obs, import_prices, delta_U_over_U, oil_change, coal_change, nairu_prior=nairu_prior
        )

        nairu, nairu_std = extract_nairu_estimates(
            y, U_obs, result.params, import_prices, delta_U_over_U, oil_change, coal_change, nairu_prior
        )

        mask = dates >= START
        dates_f = dates[mask]
        nairu_f = nairu[mask]
        nairu_std_f = nairu_std[mask]
        U_f = U_obs[mask]

        cuts = [0.005, 0.025, 0.16]
        alphas = [0.1, 0.2, 0.3]

        ax = None
        for cut, alpha in zip(cuts, alphas):
            z = stats.norm.ppf(1 - cut)
            lower = pd.Series(nairu_f - z * nairu_std_f, index=dates_f)
            upper = pd.Series(nairu_f + z * nairu_std_f, index=dates_f)
            band = pd.DataFrame({"lower": lower, "upper": upper}, index=dates_f)
            ax = mg.fill_between_plot(
                band,
                ax=ax,
                color="blue",
                alpha=alpha,
                label=f"NAIRU {int((1 - 2 * cut) * 100)}% Credible Interval",
                zorder=3,
            )

        nairu_series = pd.Series(nairu_f, index=dates_f, name="NAIRU Median")
        ax = mg.line_plot(nairu_series, ax=ax, color="blue", width=1, annotate=True, zorder=4)

        U_series = pd.Series(U_f, index=dates_f)
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

    title_suffix = " (Regime-Switching)" if regime_switching else ""
    mg.finalise_plot(
        ax,
        title=f"NAIRU Estimate for Australia{title_suffix}",
        ylabel="Per cent",
        legend={"loc": "best", "fontsize": "x-small", "ncol": 2},
        lfooter=r"Australia. NAIRU-Phillips model with normalised unemployment gap and ULC wage measure.",
        rheader=f"From {PI_TARGET_FULL} onwards, NAIRU is the unemployment rate consistent with target inflation.",
        rfooter=RFOOTER,
        axisbelow=True,
        **ANNUAL_RANGE,
        **ANNUAL_TARGET,
        **NAIRU_WARN,
        show=show,
    )


def plot_unemployment_gap(show: bool = True) -> None:
    """Plot unemployment gap (U - NAIRU)."""

    y, U_obs, import_prices, delta_U_over_U, oil_change, dates = load_nairu_phillips_data(
        start="1984Q1", end=None, anchor_inflation=True
    )

    nairu_prior = np.mean(U_obs)

    result = estimate_nairu_phillips(
        y, U_obs, import_prices, delta_U_over_U, oil_change, nairu_prior=nairu_prior
    )

    nairu, nairu_std = extract_nairu_estimates(
        y, U_obs, result.params, import_prices, delta_U_over_U, oil_change, nairu_prior
    )

    # Filter to start
    mask = dates >= START
    dates_f = dates[mask]
    nairu_f = nairu[mask]
    nairu_std_f = nairu_std[mask]
    U_f = U_obs[mask]

    # U-gap
    u_gap = U_f - nairu_f

    # Build credible interval bands
    cuts = [0.005, 0.025, 0.16]
    alphas = [0.1, 0.2, 0.3]

    ax = None
    for cut, alpha in zip(cuts, alphas):
        z = stats.norm.ppf(1 - cut)
        lower = pd.Series(u_gap - z * nairu_std_f, index=dates_f)
        upper = pd.Series(u_gap + z * nairu_std_f, index=dates_f)
        band = pd.DataFrame({"lower": lower, "upper": upper}, index=dates_f)
        ax = mg.fill_between_plot(
            band,
            ax=ax,
            color="darkred",
            alpha=alpha,
            label=f"U-gap {int((1 - 2 * cut) * 100)}% CI",
            zorder=3,
        )

    u_gap_series = pd.Series(u_gap, index=dates_f, name="U-gap Median")
    ax = mg.line_plot(u_gap_series, ax=ax, color="darkred", width=1, annotate=True, zorder=4)

    mg.finalise_plot(
        ax,
        title="Unemployment Gap Estimate for Australia",
        ylabel="Percentage points (U - U*)",
        lfooter=r"Australia. $U\text{-}gap = U - U^*$. Positive = slack/disinflationary, Negative = tight/inflationary.",
        rfooter=RFOOTER,
        legend={"loc": "best", "fontsize": "x-small"},
        axisbelow=True,
        y0=True,
        **NAIRU_WARN,
        show=show,
    )


if __name__ == "__main__":
    setup_charts()
    plot_nairu_and_unemployment(show=False)
    plot_unemployment_gap(show=False)
    print(f"\nCharts saved to: {CHART_DIR}")
