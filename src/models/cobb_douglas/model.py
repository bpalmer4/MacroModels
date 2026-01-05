"""Cobb-Douglas MFP Decomposition model.

Growth accounting using the Cobb-Douglas production function to decompose
GDP growth into contributions from capital, labor, and multi-factor productivity.

Production function: Y = A × K^α × L^(1-α)
In log growth: g_Y = g_MFP + α×g_K + (1-α)×g_L

Where:
    - Y = GDP (chain volume measures)
    - K = Net capital stock
    - L = Hours worked
    - A = Multi-factor productivity (Solow residual)
    - α = Capital share of income (~0.3)

Important: The output gap from this model is notional only - it is not
disciplined by inflation dynamics. For policy-relevant estimates, prefer
models that incorporate Phillips curve information.
"""

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import mgplot as mg
import numpy as np
import pandas as pd
from statsmodels.tsa.filters.hp_filter import hpfilter

from src.data import (
    get_capital_share,
    get_capital_stock_qrtly,
    get_hourly_coe_growth_qrtly,
    get_hours_worked_qrtly,
    get_trimmed_mean_annual,
    get_labour_force_growth_qrtly,
    get_mfp_growth,
    get_ulc_growth_qrtly,
)
from src.data.gdp import get_gdp
from src.utilities.rate_conversion import annualize

# --- Constants ---

ALPHA = 0.3  # Capital share of income (ABS estimate)
HP_LAMBDA = 1600  # HP filter smoothing parameter for quarterly data

# Anchor points for potential GDP re-anchoring (business cycle peaks)
ANCHOR_POINTS = ["1990Q1", "2000Q1", "2008Q1", "2019Q4"]


# --- Data Loading ---


@dataclass
class CobbDouglasData:
    """Container for Cobb-Douglas model data."""

    gdp: pd.Series
    capital: pd.Series
    hours: pd.Series
    index: pd.PeriodIndex

    @property
    def aligned(self) -> pd.DataFrame:
        """Return aligned DataFrame of levels."""
        return pd.DataFrame({
            "GDP": self.gdp,
            "Capital": self.capital,
            "Hours": self.hours,
        }, index=self.index)


def load_data(
    start: str | None = None,
    end: str | None = None,
) -> CobbDouglasData:
    """Load and align data for Cobb-Douglas decomposition.

    Args:
        start: Start period (e.g., "1985Q1")
        end: End period (e.g., "2024Q3")

    Returns:
        CobbDouglasData with aligned series

    """
    # Fetch data using dedicated loaders (all quarterly)
    gdp = get_gdp(gdp_type="CVM", seasonal="SA").data
    capital = get_capital_stock_qrtly().data
    hours = get_hours_worked_qrtly().data

    # Combine into DataFrame
    df = pd.DataFrame({
        "GDP": gdp,
        "Capital": capital,
        "Hours": hours,
    })

    # Apply sample period
    if start:
        df = df[df.index >= pd.Period(start, freq="Q")]
    if end:
        df = df[df.index <= pd.Period(end, freq="Q")]

    # Drop missing values
    df = df.dropna()

    return CobbDouglasData(
        gdp=df["GDP"],
        capital=df["Capital"],
        hours=df["Hours"],
        index=df.index,
    )


def load_inflation_data() -> pd.Series:
    """Load trimmed mean annual inflation for Phillips curve cross-check.

    Returns:
        Annual trimmed mean inflation (year-over-year growth)

    """
    return get_trimmed_mean_annual().data


# --- Growth Accounting ---


def calculate_growth_rates(data: CobbDouglasData) -> pd.DataFrame:
    """Calculate quarterly log growth rates.

    Using log differences ensures growth rates are additive in the
    Cobb-Douglas decomposition.

    Returns:
        DataFrame with log levels and growth rates

    """
    df = data.aligned
    growth = pd.DataFrame(index=df.index)

    # Log levels (×100 for percentage scale)
    growth["log_GDP"] = np.log(df["GDP"]) * 100
    growth["log_Capital"] = np.log(df["Capital"]) * 100
    growth["log_Hours"] = np.log(df["Hours"]) * 100

    # Quarterly growth rates
    growth["g_GDP"] = growth["log_GDP"].diff(1)
    growth["g_Capital"] = growth["log_Capital"].diff(1)
    growth["g_Hours"] = growth["log_Hours"].diff(1)

    # Annual growth rates (4-quarter difference)
    growth["g_GDP_annual"] = growth["log_GDP"].diff(4)
    growth["g_Capital_annual"] = growth["log_Capital"].diff(4)
    growth["g_Hours_annual"] = growth["log_Hours"].diff(4)

    return growth


def calculate_solow_residual(
    growth: pd.DataFrame,
    alpha: float = ALPHA,
) -> pd.DataFrame:
    """Calculate MFP as the Solow residual.

    g_MFP = g_Y - α×g_K - (1-α)×g_L

    This is "raw" MFP - includes cyclical fluctuations and measurement error.

    Args:
        growth: DataFrame from calculate_growth_rates()
        alpha: Capital share parameter

    Returns:
        DataFrame with factor contributions and MFP

    """
    result = growth.copy()

    # Factor contributions
    result["contrib_Capital"] = alpha * result["g_Capital"]
    result["contrib_Hours"] = (1 - alpha) * result["g_Hours"]

    # MFP as residual (quarterly)
    result["g_MFP"] = (
        result["g_GDP"]
        - result["contrib_Capital"]
        - result["contrib_Hours"]
    )

    # Annual versions
    result["contrib_Capital_annual"] = alpha * result["g_Capital_annual"]
    result["contrib_Hours_annual"] = (1 - alpha) * result["g_Hours_annual"]
    result["g_MFP_annual"] = (
        result["g_GDP_annual"]
        - result["contrib_Capital_annual"]
        - result["contrib_Hours_annual"]
    )

    return result


def apply_hp_filter(
    series: pd.Series,
    hp_lambda: float = HP_LAMBDA,
) -> tuple[pd.Series, pd.Series]:
    """Apply Hodrick-Prescott filter to extract trend and cycle.

    Args:
        series: Input series
        hp_lambda: Smoothing parameter (1600 for quarterly)

    Returns:
        Tuple of (trend, cycle)

    """
    clean = series.dropna()
    cycle, trend = hpfilter(clean, lamb=hp_lambda)
    trend = pd.Series(trend, index=clean.index)
    cycle = pd.Series(cycle, index=clean.index)
    return trend, cycle


def extract_mfp_trend(
    growth: pd.DataFrame,
    hp_lambda: float = HP_LAMBDA,
) -> pd.DataFrame:
    """Extract trend MFP from raw Solow residual using HP filter.

    Args:
        growth: DataFrame with g_MFP column
        hp_lambda: Smoothing parameter for HP filter

    Returns:
        DataFrame with trend and cycle components

    """
    mfp = growth["g_MFP"].dropna()
    trend, cycle = apply_hp_filter(mfp, hp_lambda)

    return pd.DataFrame({
        "MFP Raw": mfp,
        "MFP Trend": trend,
        "MFP Cycle": cycle,
    }, index=mfp.index)


def calculate_potential_gdp(
    growth: pd.DataFrame,
    mfp_trend: pd.Series,
    alpha: float = ALPHA,
    anchor_points: list[str] | None = None,
    hp_lambda: float = HP_LAMBDA,
) -> pd.DataFrame:
    """Calculate potential GDP using trend MFP with periodic re-anchoring.

    Potential growth = α×g_K_trend + (1-α)×g_L_trend + g_MFP_trend

    Re-anchoring at specified points prevents cumulative drift from compounding
    small biases in trend growth into large level gaps over long samples.

    Args:
        growth: DataFrame from calculate_solow_residual()
        mfp_trend: Trend MFP series
        alpha: Capital share parameter
        anchor_points: Periods where potential is re-anchored to actual
        hp_lambda: Smoothing parameter for factor growth trends

    Returns:
        DataFrame with potential GDP and output gap

    """
    if anchor_points is None:
        anchor_points = []

    result = pd.DataFrame(index=growth.index)

    # HP-filter capital and hours growth for potential
    g_K_trend, _ = apply_hp_filter(growth["g_Capital"].dropna(), hp_lambda)
    g_L_trend, _ = apply_hp_filter(growth["g_Hours"].dropna(), hp_lambda)

    # Potential GDP growth (quarterly)
    result["g_potential"] = (
        alpha * g_K_trend.reindex(result.index)
        + (1 - alpha) * g_L_trend.reindex(result.index)
        + mfp_trend.reindex(result.index)
    )

    # Actual GDP growth
    result["g_actual"] = growth["g_GDP"]

    # Log GDP levels
    result["log_GDP"] = growth["log_GDP"]

    # Convert anchor points to periods
    anchor_periods = [
        pd.Period(ap, freq="Q") for ap in anchor_points
        if pd.Period(ap, freq="Q") in result.index
    ]

    # Reconstruct potential GDP level with periodic re-anchoring
    start_idx = result["g_potential"].first_valid_index()
    result["log_potential"] = np.nan

    # Initialize at actual GDP level
    result.loc[start_idx, "log_potential"] = result.loc[start_idx, "log_GDP"]

    # Cumulate potential growth with re-anchoring
    reanchor_log = []
    for idx in result.index:
        if idx > start_idx and pd.notna(result.loc[idx, "g_potential"]):
            prev_idx = result.index[result.index.get_loc(idx) - 1]

            if idx in anchor_periods:
                # Re-anchor to actual GDP
                result.loc[idx, "log_potential"] = result.loc[idx, "log_GDP"]
                gap_before = (
                    result.loc[prev_idx, "log_GDP"] -
                    result.loc[prev_idx, "log_potential"]
                ) if pd.notna(result.loc[prev_idx, "log_potential"]) else 0
                reanchor_log.append((str(idx), gap_before))
            elif pd.notna(result.loc[prev_idx, "log_potential"]):
                # Normal cumulation
                result.loc[idx, "log_potential"] = (
                    result.loc[prev_idx, "log_potential"]
                    + result.loc[idx, "g_potential"]
                )

    # Store re-anchoring info
    result.attrs["reanchor_log"] = reanchor_log
    result.attrs["anchor_points"] = anchor_points

    # Convert to levels
    result["GDP_actual"] = np.exp(result["log_GDP"] / 100)
    result["GDP_potential"] = np.exp(result["log_potential"] / 100)

    # Output gap (% deviation from potential)
    result["output_gap"] = (
        (result["GDP_actual"] - result["GDP_potential"])
        / result["GDP_potential"] * 100
    )

    return result


# --- Sensitivity Analysis ---


def sensitivity_analysis_alpha(
    growth: pd.DataFrame,
    alphas: list[float] | None = None,
    hp_lambda: float = HP_LAMBDA,
) -> pd.DataFrame:
    """Test sensitivity of MFP to different capital share assumptions.

    Args:
        growth: DataFrame from calculate_growth_rates()
        alphas: List of alpha values to test
        hp_lambda: HP filter smoothing parameter

    Returns:
        DataFrame with MFP statistics for each alpha

    """
    if alphas is None:
        alphas = [0.20, 0.25, 0.30, 0.35, 0.40]

    results = []
    for alpha in alphas:
        # Calculate MFP with this alpha
        mfp = (
            growth["g_GDP"]
            - alpha * growth["g_Capital"]
            - (1 - alpha) * growth["g_Hours"]
        )

        # HP filter
        mfp_trend, _ = apply_hp_filter(mfp.dropna(), hp_lambda)

        # Recent period
        recent = mfp_trend.loc["2015Q1":] if "2015Q1" in mfp_trend.index else mfp_trend.tail(40)

        results.append({
            "alpha": alpha,
            "mfp_raw_mean": mfp.dropna().mean(),
            "mfp_trend_mean": mfp_trend.mean(),
            "mfp_recent": recent.mean() if len(recent) > 0 else np.nan,
        })

    return pd.DataFrame(results)


# --- Phillips Curve Cross-Check ---


def phillips_curve_crosscheck(
    output_gap: pd.Series,
    inflation_annual: pd.Series,
) -> pd.DataFrame:
    """Compute Phillips curve cross-check statistics.

    Args:
        output_gap: Output gap series
        inflation_annual: Annual trimmed mean inflation

    Returns:
        DataFrame with aligned data and statistics

    """
    common = pd.DataFrame({
        "Output Gap": output_gap,
        "Inflation": inflation_annual,
    }).dropna()

    common["Inflation Deviation"] = common["Inflation"] - 2.5

    # Correlation
    common.attrs["corr_level"] = common["Output Gap"].corr(common["Inflation"])
    common.attrs["corr_deviation"] = common["Output Gap"].corr(common["Inflation Deviation"])

    # Slope
    if len(common) > 2:
        z = np.polyfit(common["Output Gap"], common["Inflation Deviation"], 1)
        common.attrs["slope"] = z[0]
    else:
        common.attrs["slope"] = np.nan

    return common


# --- Decomposition Results ---


@dataclass
class DecompositionResult:
    """Results from Cobb-Douglas decomposition."""

    data: CobbDouglasData
    growth: pd.DataFrame
    mfp: pd.DataFrame
    potential: pd.DataFrame
    alpha: float
    sensitivity: pd.DataFrame | None = None
    phillips: pd.DataFrame | None = None

    def summary_by_decade(self) -> pd.DataFrame:
        """Average growth contributions by decade."""
        # Annual data
        annual = pd.DataFrame({
            "Capital": self.growth["contrib_Capital"],
            "Labour": self.growth["contrib_Hours"],
            "MFP": self.mfp["MFP Trend"].reindex(self.growth.index),
        })

        # Sum to calendar years
        annual_sum = annual.groupby(annual.index.year).sum()

        # Also get actual GDP growth
        gdp_annual = self.growth["g_GDP"].groupby(self.growth["g_GDP"].index.year).sum()

        results = []
        for decade_start in range(1990, 2030, 10):
            decade_end = decade_start + 9
            mask = (annual_sum.index >= decade_start) & (annual_sum.index <= decade_end)
            if mask.sum() > 0:
                decade_data = annual_sum.loc[mask].mean()
                total = gdp_annual.loc[mask].mean()
                results.append({
                    "Decade": f"{decade_start}s",
                    "Capital": decade_data["Capital"],
                    "Labour": decade_data["Labour"],
                    "MFP": decade_data["MFP"],
                    "Total": total,
                })

        return pd.DataFrame(results).set_index("Decade")

    def summary(self) -> pd.DataFrame:
        """Average growth contributions over sample period."""
        cols = ["g_GDP", "contrib_Capital", "contrib_Hours", "g_MFP"]
        annual = annualize(self.growth[cols].dropna())
        return annual.describe().loc[["mean", "std"]]


def run_decomposition(
    start: str | None = None,
    end: str | None = None,
    alpha: float = ALPHA,
    anchor_points: list[str] | None = None,
    include_sensitivity: bool = True,
    include_phillips: bool = True,
) -> DecompositionResult:
    """Run full Cobb-Douglas MFP decomposition.

    Args:
        start: Start period
        end: End period
        alpha: Capital share parameter
        anchor_points: Re-anchoring points for potential GDP
        include_sensitivity: Run sensitivity analysis
        include_phillips: Run Phillips curve cross-check

    Returns:
        DecompositionResult with all computed series

    """
    if anchor_points is None:
        anchor_points = ANCHOR_POINTS

    # Load data
    data = load_data(start=start, end=end)

    # Calculate growth rates
    growth = calculate_growth_rates(data)

    # Solow residual
    growth = calculate_solow_residual(growth, alpha=alpha)

    # Extract MFP trend
    mfp = extract_mfp_trend(growth)

    # Calculate potential GDP with re-anchoring
    potential = calculate_potential_gdp(
        growth, mfp["MFP Trend"],
        alpha=alpha,
        anchor_points=anchor_points,
    )

    # Sensitivity analysis
    sensitivity = None
    if include_sensitivity:
        sensitivity = sensitivity_analysis_alpha(growth)

    # Phillips curve cross-check
    phillips = None
    if include_phillips:
        try:
            inflation_annual = load_inflation_data()
            phillips = phillips_curve_crosscheck(
                potential["output_gap"],
                inflation_annual,
            )
        except Exception as e:
            print(f"Warning: Could not load inflation data for Phillips check: {e}")

    return DecompositionResult(
        data=data,
        growth=growth,
        mfp=mfp,
        potential=potential,
        alpha=alpha,
        sensitivity=sensitivity,
        phillips=phillips,
    )


# --- Plotting ---


def plot_raw_data(result: DecompositionResult, show: bool = True) -> None:
    """Plot the raw input data series."""
    data = result.data.aligned

    fig, axes = plt.subplots(3, 1, figsize=(9.0, 9.0))

    # GDP
    gdp_data = pd.DataFrame({"GDP": data["GDP"].dropna()})
    mg.line_plot(gdp_data, ax=axes[0], color="blue", width=1.5)
    mg.finalise_plot(
        axes[0],
        title="GDP (Chain Volume Measures)",
        ylabel="$ Millions",
        dont_save=True,
        dont_close=True,
    )

    # Capital
    capital_data = pd.DataFrame({"Capital Stock": data["Capital"]})
    mg.line_plot(capital_data, ax=axes[1], color="green", width=1.5)
    mg.finalise_plot(
        axes[1],
        title="Net Capital Stock (CVM)",
        ylabel="$ Millions",
        dont_save=True,
        dont_close=True,
    )

    # Hours
    hours_data = pd.DataFrame({"Hours Worked": data["Hours"]})
    mg.line_plot(hours_data, ax=axes[2], color="orange", width=1.5)
    mg.finalise_plot(
        axes[2],
        title="Hours Actually Worked",
        ylabel="Millions of Hours",
        figsize=(9.0, 9.0),
        rfooter="ABS 5206, 1364, 6202",
        lfooter="Australia. ",
        show=show,
    )


def plot_mfp_trend(result: DecompositionResult, show: bool = True) -> None:
    """Plot MFP: raw vs trend (2 panels)."""
    mfp = result.mfp[["MFP Raw", "MFP Trend"]].dropna()

    fig, axes = plt.subplots(2, 1, figsize=(9.0, 9.0))

    # Panel 1: Full sample - quarterly
    panel1_data = pd.DataFrame({
        "Raw MFP": mfp["MFP Raw"],
        f"HP Filter (λ={HP_LAMBDA})": mfp["MFP Trend"],
    })

    mg.line_plot(
        panel1_data,
        ax=axes[0],
        color=["gray", "blue"],
        alpha=[0.3, 1.0],
        width=[0.8, 2],
    )

    mg.finalise_plot(
        axes[0],
        title=f"Multi-Factor Productivity Growth (α={result.alpha})",
        ylabel="Quarterly Growth (%)",
        y0=True,
        legend={"loc": "best", "fontsize": "x-small"},
        dont_save=True,
        dont_close=True,
    )

    # Panel 2: Annualized trend
    panel2_data = pd.DataFrame({
        f"HP Filter (λ={HP_LAMBDA})": annualize(mfp["MFP Trend"]),
    })

    mg.line_plot(
        panel2_data,
        ax=axes[1],
        color="blue",
        width=2,
    )

    # Add reference line at 1% growth
    axes[1].axhline(y=1.0, color="gray", linewidth=0.5, linestyle="--", alpha=0.5)

    mg.finalise_plot(
        axes[1],
        title="Trend MFP Growth (Annualised)",
        ylabel="Annual Growth (% p.a.)",
        y0=True,
        legend={"loc": "best", "fontsize": "x-small"},
        figsize=(9.0, 9.0),
        lfooter="Australia. MFP (Solow residual) = GDP growth − α×Capital growth − (1−α)×Hours growth",
        rfooter="ABS 5206, 1364, 6202",
        show=show,
    )


def plot_potential_gdp(result: DecompositionResult, show: bool = True) -> None:
    """Plot actual vs potential GDP and output gap (3 panels)."""
    pot = result.potential
    anchor_points = pot.attrs.get("anchor_points", [])

    fig, axes = plt.subplots(3, 1, figsize=(9.0, 9.0))

    # Panel 1: GDP levels
    gdp_levels = pd.DataFrame({
        "Actual GDP": pot["GDP_actual"],
        "Potential GDP": pot["GDP_potential"],
    }).dropna(axis=0, how="any")
    mg.line_plot(
        gdp_levels,
        ax=axes[0],
        color=["blue", "red"],
        width=[1.5, 2],
        style=["solid", "dashed"],
    )
    # Mark anchor points
    label = "Anchor points"
    for ap in anchor_points:
        try:
            ap_period = pd.Period(ap, freq="Q")
            if ap_period in gdp_levels.index:
                axes[0].axvline(x=ap_period.ordinal, color="darkred",
                               linewidth=1, linestyle=":", label=label)
                label = "_"
        except (ValueError, KeyError):
            pass
    axes[0].set_yscale("log")
    mg.finalise_plot(
        axes[0],
        title="Actual vs Potential GDP (HP Filter MFP, with re-anchoring)",
        ylabel="GDP (log scale)",
        legend={"loc": "best", "fontsize": "x-small"},
        dont_save=True,
        dont_close=True,
    )

    # Panel 2: GDP growth rates
    growth_rates = pd.DataFrame({
        "Actual Growth": annualize(pot["g_actual"]),
        "Potential Growth": annualize(pot["g_potential"]),
    })
    mg.line_plot(
        growth_rates,
        ax=axes[1],
        color=["blue", "red"],
        alpha=[0.5, 1.0],
        width=[1, 2],
        style=["solid", "dashed"],
    )
    mg.finalise_plot(
        axes[1],
        title="GDP Growth: Actual vs Potential (Annualized)",
        ylabel="Growth (% p.a.)",
        y0=True,
        legend={"loc": "best", "fontsize": "x-small"},
        dont_save=True,
        dont_close=True,
    )

    # Panel 3: Output gap
    ax = axes[2]
    gap_df = pd.DataFrame({"Output Gap": pot["output_gap"]}).dropna()

    # Use mgplot fill_between_plot
    positive_fill = pd.DataFrame({
        "Zero": 0.0,
        "Positive": gap_df["Output Gap"].clip(lower=0),
    }, index=gap_df.index)
    negative_fill = pd.DataFrame({
        "Negative": gap_df["Output Gap"].clip(upper=0),
        "Zero": 0.0,
    }, index=gap_df.index)
    mg.fill_between_plot(positive_fill, ax=ax, color="green", alpha=0.3, label="Positive gap")
    mg.fill_between_plot(negative_fill, ax=ax, color="red", alpha=0.3, label="Negative gap")

    # Use mgplot line_plot
    mg.line_plot(gap_df, ax=ax, color="black", width=1.5)

    # Mark anchor points
    label = "Anchor points"
    for ap in anchor_points:
        try:
            ap_period = pd.Period(ap)
            if ap_period in gap_df.index:
                ax.axvline(x=ap_period.ordinal, color="darkred",
                          linewidth=1, linestyle=":", label=label)
                label = "_"
        except (ValueError, KeyError):
            pass
    ax.axhline(y=0, color="black", linewidth=1)
    mg.finalise_plot(
        ax,
        title="Output Gap (Actual - Potential) / Potential",
        ylabel="Gap (%)",
        legend={"loc": "best", "fontsize": "x-small"},
        figsize=(9.0, 9.0),
        rfooter="ABS 5206, 1364, 6202",
        lfooter="Australia. Note: Output gap is a mechanical estimate, not informed by inflation dynamics.",
        show=show,
    )


def plot_growth_decomposition(result: DecompositionResult, show: bool = True) -> None:
    """Plot annual stacked bar chart of growth decomposition."""
    # Prepare annual data (sum of quarterly)
    annual = pd.DataFrame({
        "Capital": result.growth["contrib_Capital"],
        "Labour (Hours)": result.growth["contrib_Hours"],
        "MFP (Trend)": result.mfp["MFP Trend"].reindex(result.growth.index),
    })

    # Group by year and sum
    annual_sum = annual.groupby(annual.index.year).sum()
    annual_sum = annual_sum.loc[annual_sum.index >= 1986]  # Full years only
    annual_sum = annual_sum.loc[annual_sum.index <= annual_sum.index[-1] - 1]  # Drop incomplete

    # Convert to annual PeriodIndex for mgplot
    annual_sum.index = pd.PeriodIndex([pd.Period(y, freq="Y") for y in annual_sum.index])

    # Get actual GDP growth for line
    gdp_annual = result.growth["g_GDP"].groupby(result.growth["g_GDP"].index.year).sum()
    gdp_annual = gdp_annual.loc[gdp_annual.index.isin(annual_sum.index.year)]
    gdp_annual.index = annual_sum.index

    # Use mgplot for stacked bar
    ax = mg.bar_plot(
        annual_sum,
        stacked=True,
        color=["#1f77b4", "#ff7f0e", "#2ca02c"],
    )

    # Add GDP growth line on top
    gdp_line = pd.Series(gdp_annual.values, index=annual_sum.index, name="Actual GDP Growth")
    mg.line_plot(gdp_line, ax=ax, color="black", width=1.5, marker="o", markersize=4)

    mg.finalise_plot(
        ax,
        title=f"GDP Growth Decomposition:\nContributions from Capital, Labour, and MFP (α={result.alpha})",
        ylabel="Growth Contribution (% p.a.)",
        legend={"loc": "best", "fontsize": "x-small"},
        lfooter="Australia. Stacked bars use HP-filtered trend MFP. Bars may not sum exactly to actual GDP (dots) "
                "because the HP filter smooths cyclical MFP fluctuations.",
        rheader="ABS 5206, 1364, 6202",
        y0=True,
        show=show,
    )


def plot_sensitivity(result: DecompositionResult, show: bool = True) -> None:
    """Plot MFP trends for different alpha values."""
    if result.sensitivity is None:
        return

    growth = result.growth
    alphas = [0.25, 0.30, 0.35]

    trends = pd.DataFrame()
    for alpha in alphas:
        mfp = (
            growth["g_GDP"]
            - alpha * growth["g_Capital"]
            - (1 - alpha) * growth["g_Hours"]
        )
        mfp_trend, _ = apply_hp_filter(mfp.dropna())
        trends[f"α = {alpha}"] = annualize(mfp_trend)

    mg.line_plot_finalise(
        trends,
        title="MFP Trend Sensitivity to Capital Share (α)",
        ylabel="Annual MFP Growth (%)",
        y0=True,
        width=2,
        rfooter="ABS 5206, 1364, 6202",
        lfooter="Australia.",
        show=show,
    )


def plot_phillips_crosscheck(result: DecompositionResult, show: bool = True) -> None:
    """Plot Phillips curve cross-check (4 panels)."""
    if result.phillips is None:
        return

    common = result.phillips

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Time series: Output gap and inflation (twin axes)
    ax = axes[0, 0]
    ax2 = ax.twinx()

    gap_df = pd.DataFrame({"Output Gap": common["Output Gap"]})
    inflation_df = pd.DataFrame({"Trimmed Mean Inflation": common["Inflation"]})

    # Use mgplot for output gap (fill_between and line)
    positive_fill = pd.DataFrame({
        "Zero": 0.0,
        "Positive": gap_df["Output Gap"].clip(lower=0),
    }, index=gap_df.index)
    negative_fill = pd.DataFrame({
        "Negative": gap_df["Output Gap"].clip(upper=0),
        "Zero": 0.0,
    }, index=gap_df.index)
    max_ticks = 8
    mg.fill_between_plot(positive_fill, ax=ax, color="green", alpha=0.3, max_ticks=max_ticks)
    mg.fill_between_plot(negative_fill, ax=ax, color="red", alpha=0.3, max_ticks=max_ticks)
    mg.line_plot(gap_df, ax=ax, color="black", width=1.5, max_ticks=max_ticks)

    ax.axhline(y=0, color="black", linewidth=0.5)
    ax.set_ylabel("Output Gap (%)", color="black")
    ax.set_ylim(-8, 4)

    # Use mgplot for inflation on twin axis
    mg.line_plot(inflation_df, ax=ax2, color="darkorange", width=2, max_ticks=max_ticks)
    ax2.axhline(y=2.5, color="darkorange", linewidth=1, linestyle="--", alpha=0.7)
    ax2.axhspan(2.0, 3.0, color="orange", alpha=0.1)
    ax2.set_ylabel("Annual Inflation (%)", color="darkorange")
    ax2.set_ylim(0, 8)

    ax.set_title("Output Gap vs Core Inflation\n(Time Series)")
    ax.legend(loc="upper left", fontsize="x-small")
    ax2.legend(loc="upper right", fontsize="x-small")

    # 2. Scatter plot: Output gap vs inflation level (keep matplotlib)
    ax = axes[0, 1]
    ax.scatter(common["Output Gap"], common["Inflation"],
               alpha=0.5, s=30, c="blue")

    # Fit line
    z = np.polyfit(common["Output Gap"], common["Inflation"], 1)
    p = np.poly1d(z)
    x_line = np.linspace(common["Output Gap"].min(), common["Output Gap"].max(), 100)
    ax.plot(x_line, p(x_line), "r-", linewidth=2, label=f"Fitted: slope={z[0]:.2f}")

    ax.axhline(y=2.5, color="gray", linewidth=1, linestyle="--", alpha=0.7)
    ax.axvline(x=0, color="gray", linewidth=1, linestyle="--", alpha=0.7)
    ax.set_xlabel("Output Gap (%)")
    ax.set_ylabel("Annual Trimmed Mean Inflation (%)")
    ax.set_title("Phillips Curve:\nOutput Gap vs Inflation Level")
    ax.legend(fontsize="x-small")

    # 3. Scatter: Output gap vs inflation deviation from target (keep matplotlib)
    ax = axes[1, 0]
    ax.scatter(common["Output Gap"], common["Inflation Deviation"],
               alpha=0.5, s=30, c="darkgreen")

    # Fit line for deviation
    z2 = np.polyfit(common["Output Gap"], common["Inflation Deviation"], 1)
    p2 = np.poly1d(z2)
    ax.plot(x_line, p2(x_line), "r-", linewidth=2, label=f"Fitted: slope={z2[0]:.2f}")

    ax.axhline(y=0, color="gray", linewidth=1, linestyle="--", alpha=0.7)
    ax.axvline(x=0, color="gray", linewidth=1, linestyle="--", alpha=0.7)
    ax.set_xlabel("Output Gap (%)")
    ax.set_ylabel("Inflation Deviation from 2.5% Target")
    ax.set_title("Phillips Curve:\nOutput Gap vs Inflation Deviation")
    ax.legend(fontsize="x-small")

    # 4. Rolling correlation - use mgplot
    rolling_corr = common["Output Gap"].rolling(window=20).corr(common["Inflation Deviation"])
    corr_data = pd.DataFrame({"Rolling Correlation": rolling_corr})

    mg.line_plot(corr_data, ax=axes[1, 1], color="purple", width=1.5, max_ticks=8)
    axes[1, 1].axhline(y=0, color="black", linewidth=0.5)
    axes[1, 1].axhline(y=rolling_corr.mean(), color="gray", linewidth=1, linestyle="--")
    axes[1, 1].set_ylim(-1, 1)

    mg.finalise_plot(
        axes[1, 1],
        title="Rolling 5-Year Correlation:\nOutput Gap vs Inflation Deviation",
        ylabel="Correlation",
        figsize=(9.0, 9.0),
        rfooter="ABS 5206, 1364, 6401",
        lfooter="Australia. ",
        show=show,
    )


def plot_capital_growth(result: DecompositionResult, show: bool = True) -> None:
    """Plot capital stock growth (raw and HP-filtered)."""
    g_capital = result.growth["g_Capital"].dropna()
    g_capital_hp, _ = apply_hp_filter(g_capital)

    # Annualize for interpretation
    raw_annual = annualize(g_capital)
    hp_annual = annualize(g_capital_hp)

    plot_data = pd.DataFrame({
        "Capital Growth": raw_annual,
        "HP(1600)": hp_annual,
    })

    mg.line_plot_finalise(
        plot_data,
        title="Capital Stock Growth",
        ylabel="Annual Growth (% p.a.)",
        y0=True,
        width=[1.5, 2],
        alpha=[0.6, 1.0],
        color=["gray", "blue"],
        legend={"loc": "best", "fontsize": "small"},
        lfooter="Australia. Net capital stock (chain volume measures). NAIRU model uses raw growth.",
        rfooter="ABS 1364",
        show=show,
    )


def plot_labour_growth(result: DecompositionResult, show: bool = True) -> None:
    """Plot hours worked growth (raw and HP-filtered trend)."""
    g_hours = result.growth["g_Hours"].dropna()
    g_hours_trend, _ = apply_hp_filter(g_hours)

    # Annualize for interpretation
    raw_annual = annualize(g_hours)
    trend_annual = annualize(g_hours_trend)

    # COVID volatility for rheader
    covid_data = raw_annual.loc["2020Q1":"2021Q4"]
    covid_min = covid_data.min()
    covid_max = covid_data.max()

    plot_data = pd.DataFrame({
        "Hours Growth (Raw)": raw_annual,
        "Hours Growth (HP Trend)": trend_annual,
    })

    mg.line_plot_finalise(
        plot_data,
        title="Labour Input Growth (Hours Worked)",
        ylabel="Annual Growth (% p.a.)",
        ylim=(-5.0, 10.0),
        width=[0.8, 2],
        alpha=[0.3, 1.0],
        color=["gray", "orange"],
        legend={"loc": "best", "fontsize": "small"},
        lfooter="Australia. Hours actually worked in all jobs.",
        rfooter="ABS 5206, 6202",
        rheader=f"Plot constrained to -5% to +10%; COVID volatility ranged from {covid_min:.0f}% to +{covid_max:.0f}%.",
        show=show,
    )


def plot_labour_force_growth(show: bool = True) -> None:
    """Plot labour force growth (through-the-year with HP and HMA trends)."""
    from src.data import get_labour_force_qrtly
    from src.data.henderson import hma

    # Get labour force levels and compute through-the-year growth
    lf = get_labour_force_qrtly().data.dropna()
    lf_tty = (np.log(lf) - np.log(lf.shift(4))) * 100  # 4-quarter log change

    # HP filter on TTY growth
    lf_tty_clean = lf_tty.dropna()
    _, lf_hp = hpfilter(lf_tty_clean, lamb=HP_LAMBDA)
    lf_hp = pd.Series(lf_hp, index=lf_tty_clean.index)

    # HMA(13) on TTY growth
    lf_hma = hma(lf_tty_clean, 13)

    plot_data = pd.DataFrame({
        "Labour Force Growth": lf_tty,
        "HP Trend": lf_hp,
        "HMA(13)": lf_hma,
    })

    mg.line_plot_finalise(
        plot_data,
        title="Labour Force Growth (Through-the-Year)",
        ylabel="% change from year ago",
        y0=True,
        width=[0.8, 2, 2],
        alpha=[0.3, 1.0, 1.0],
        color=["gray", "blue", "red"],
        legend={"loc": "best", "fontsize": "small"},
        lfooter="Australia. Total labour force (employed + unemployed).",
        rfooter="ABS 1364",
        annotate=True,
        show=show,
    )


def plot_labour_force_growth_quarterly(show: bool = True) -> None:
    """Plot labour force Q/Q growth as used in NAIRU model."""
    from src.data.henderson import hma

    # Q/Q growth (what the NAIRU model uses)
    lf_qq = get_labour_force_growth_qrtly().data.dropna()

    # HMA(13) - what the NAIRU model applies
    lf_hma = hma(lf_qq, 13)

    # HP(1600) for comparison
    _, lf_hp = hpfilter(lf_qq, lamb=HP_LAMBDA)
    lf_hp = pd.Series(lf_hp, index=lf_qq.index)

    plot_data = pd.DataFrame({
        "LF Q/Q": lf_qq,
        "HMA(13)": lf_hma,
        "HP(1600)": lf_hp,
    })

    mg.line_plot_finalise(
        plot_data,
        title="Labour Force Growth (Q/Q, as used in NAIRU model)",
        ylabel="% per quarter",
        ylim=(-1.0, 2.0),
        width=[0.8, 2, 2],
        alpha=[0.3, 1.0, 1.0],
        color=["gray", "red", "blue"],
        legend={"loc": "best", "fontsize": "small"},
        lfooter="Australia. Quarterly log difference. NAIRU model uses HMA(13). Outliers excluded.",
        rfooter="ABS 1364",
        annotate=True,
        show=show,
    )


def plot_labour_productivity(result: DecompositionResult, show: bool = True) -> None:
    """Plot labour productivity growth (GDP per hour worked)."""
    # Labour productivity = g_GDP - g_Hours
    g_lp = (result.growth["g_GDP"] - result.growth["g_Hours"]).dropna()
    g_lp_trend, _ = apply_hp_filter(g_lp)

    # Annualize
    raw_annual = annualize(g_lp)
    trend_annual = annualize(g_lp_trend)

    plot_data = pd.DataFrame({
        "Labour Productivity (Raw)": raw_annual,
        "Labour Productivity (HP Trend)": trend_annual,
    })

    mg.line_plot_finalise(
        plot_data,
        title="Labour Productivity Growth (GDP per Hour Worked)",
        ylabel="Annual Growth (% p.a.)",
        y0=True,
        width=[0.8, 2],
        alpha=[0.3, 1.0],
        color=["gray", "purple"],
        legend={"loc": "best", "fontsize": "small"},
        lfooter="Australia. Labour productivity = GDP / Hours worked.",
        rfooter="ABS 5206, 6202",
        show=show,
    )


def plot_labour_productivity_decomposition(result: DecompositionResult, show: bool = True) -> None:
    """Plot labour productivity decomposed into capital deepening and MFP."""
    alpha = result.alpha

    # Labour productivity = g_GDP - g_Hours
    g_lp = (result.growth["g_GDP"] - result.growth["g_Hours"]).dropna()
    g_lp_trend, _ = apply_hp_filter(g_lp)

    # Capital deepening = α × (g_K - g_L)
    capital_deepening = alpha * (result.growth["g_Capital"] - result.growth["g_Hours"]).dropna()
    cd_trend, _ = apply_hp_filter(capital_deepening)

    # MFP trend (already computed)
    mfp_trend = result.mfp["MFP Trend"]

    # Annualize and align
    plot_data = pd.DataFrame({
        "Labour Productivity": annualize(g_lp_trend),
        "Capital Deepening": annualize(cd_trend),
        "MFP": annualize(mfp_trend),
    }).dropna()

    mg.line_plot_finalise(
        plot_data,
        title=f"Productivity Decomposition (α={alpha})",
        ylabel="Annual Growth (% p.a.)",
        y0=True,
        width=2,
        color=["purple", "green", "blue"],
        legend={"loc": "best", "fontsize": "small"},
        lfooter="Australia. LP = Capital deepening + MFP = α×(g_K − g_L) + g_MFP. HP-filtered trends.",
        rfooter="ABS 5206, 6202, 1364",
        show=show,
    )


def plot_hours_vs_labour_force(result: DecompositionResult, show: bool = True) -> None:
    """Compare trend hours worked growth with employment × avg hours decomposition."""
    # Hours worked growth (already in result)
    g_hours = result.growth["g_Hours"].dropna()
    g_hours_trend, _ = apply_hp_filter(g_hours)

    # Employment growth (employed persons, not labour force)
    from src.data import get_employment_growth_qrtly
    emp_growth = get_employment_growth_qrtly().data
    emp_growth_trend, _ = apply_hp_filter(emp_growth.dropna())

    # Labour force growth (for comparison - includes unemployed)
    lf_growth = get_labour_force_growth_qrtly().data
    lf_growth_trend, _ = apply_hp_filter(lf_growth.dropna())

    # Avg hours per employed person: g_avg_hours = g_hours - g_employment
    common_idx = g_hours_trend.index.intersection(emp_growth_trend.index).intersection(lf_growth_trend.index)
    g_avg_hours_trend = g_hours_trend.loc[common_idx] - emp_growth_trend.loc[common_idx]

    # Annualize and align
    hours_annual = annualize(g_hours_trend.loc[common_idx])
    emp_annual = annualize(emp_growth_trend.loc[common_idx])
    lf_annual = annualize(lf_growth_trend.loc[common_idx])
    avg_hours_annual = annualize(g_avg_hours_trend)

    # Potential labour input: Labour Force × Avg Hours per Worker
    potential_labour_annual = lf_annual + avg_hours_annual

    plot_data = pd.DataFrame({
        "Hours Worked (actual)": hours_annual,
        "Labour Force": lf_annual,
        "Avg Hours per Worker": avg_hours_annual,
        "Labour Force × Avg Hours (potential)": potential_labour_annual,
    }).dropna()

    ax = mg.line_plot(
        plot_data,
        width=[2.5, 2, 2, 2],
        color=["orange", "blue", "red", "green"],
        style=["solid", "solid", "solid", "dashed"],
        annotate=True,
    )
    ax.axhline(y=0, color="gray", linewidth=0.5)

    mg.finalise_plot(
        ax,
        title="Labour Input Growth: Labour Force × Avg Hours (Potential)",
        ylabel="Annual Growth (% p.a.)",
        legend={"loc": "best", "fontsize": "x-small"},
        lfooter="Australia. HP-filtered trends. Potential = Labour Force × Avg Hours (if full employment).",
        rfooter="ABS 5206, 6202, 1364",
        show=show,
    )


def plot_mfp_lambda_comparison(result: DecompositionResult, show: bool = True) -> None:
    """Compare MFP trends with different HP filter smoothing parameters.

    Smaller lambda = more responsive to recent data (but noisier).
    Larger lambda = smoother trend (but slower to detect turning points).
    """
    mfp_raw = result.mfp["MFP Raw"].dropna()
    lambdas = [1600, 800, 400, 200]

    trends = pd.DataFrame({"Original": mfp_raw})
    for lam in lambdas:
        _, trend = hpfilter(mfp_raw, lamb=lam)
        trend = pd.Series(trend, index=mfp_raw.index)
        trends[f"λ={lam}"] = annualize(trend)

    # Drop the raw column for plotting
    plot_data = trends.drop(columns=["Original"])

    ax = mg.line_plot(
        plot_data,
        width=[2.5, 2, 1.5, 1],
        color=["#1f77b4", "#2ca02c", "#ff7f0e", "#d62728"],
    )
    ax.axhline(y=0, color="black", linewidth=0.8)

    # Add regime lines
    ax.axvline(x=pd.Period("2008Q4").ordinal, color="darkred", linewidth=1, linestyle="--", alpha=0.5)
    ax.axvline(x=pd.Period("2021Q1").ordinal, color="darkgreen", linewidth=1, linestyle="--", alpha=0.5)

    mg.finalise_plot(
        ax,
        title=f"MFP Trend Sensitivity to HP Filter Smoothing (α={result.alpha})",
        ylabel="Annual Growth (% p.a.)",
        y0=True,
        legend={"loc": "best", "fontsize": "small"},
        lheader="Smaller λ = more responsive, larger λ = smoother",
        lfooter="Australia. Vertical lines: GFC (2008Q4), COVID (2021Q1).",
        rfooter="ABS 5206, 1364, 6202",
        show=show,
    )


def plot_mfp_comparison(result: DecompositionResult, show: bool = True) -> None:
    """Compare MFP from Cobb-Douglas (GDP-based) vs wage-derived approach.

    Cobb-Douglas: MFP = g_Y - α×g_K - (1-α)×g_L (traditional Solow residual)
    Wage-derived: MFP = (Δhcoe - Δulc) - α×(g_K - g_L) (from wage identities)

    The wage-derived approach exploits the identity ULC = HCOE / LP,
    so labour productivity LP = Δhcoe - Δulc.
    """
    # Get wage data
    ulc_growth = get_ulc_growth_qrtly().data
    hcoe_growth = get_hourly_coe_growth_qrtly().data

    # Capital and hours growth from the Cobb-Douglas result
    capital_growth = result.growth["g_Capital"]
    hours_growth = result.growth["g_Hours"]

    # Wage-derived MFP
    wage_mfp = get_mfp_growth(
        ulc_growth, hcoe_growth, capital_growth, hours_growth, alpha=result.alpha
    ).data

    # Align to common index
    common_idx = result.mfp["MFP Raw"].dropna().index.intersection(wage_mfp.dropna().index)
    cd_mfp = result.mfp["MFP Raw"].loc[common_idx]
    wage_mfp = wage_mfp.loc[common_idx]

    # HP filter both for trend comparison
    cd_trend, _ = apply_hp_filter(cd_mfp)
    wage_trend, _ = apply_hp_filter(wage_mfp)

    # Annualize for plotting
    cd_annual = annualize(cd_trend)
    wage_annual = annualize(wage_trend)

    # Stats for footer
    corr = cd_annual.corr(wage_annual)

    # Plot comparison
    comparison_data = pd.DataFrame({
        "Cobb-Douglas (GDP-based)": cd_annual,
        "Wage-derived (HCOE-ULC)": wage_annual,
    })
    mg.line_plot_finalise(
        comparison_data,
        title=f"MFP Growth Comparison: Cobb-Douglas vs Wage-Derived (α={result.alpha})",
        ylabel="Annual Growth (% p.a.)",
        y0=True,
        width=2,
        color=["blue", "darkorange"],
        legend={"loc": "upper right", "fontsize": "small"},
        lfooter=f"CD: g_Y − α×g_K − (1−α)×g_L. Wage: (Δhcoe − Δulc) − α×(g_K − g_L). Correlation: {corr:.3f}",
        rfooter="ABS 5206, 6202",
        show=show,
    )


def plot_capital_share(show: bool = True) -> None:
    """Plot time-varying capital share from national accounts.

    α = GOS / (GOS + COE) where:
    - GOS = Gross Operating Surplus
    - COE = Compensation of Employees
    """
    alpha = get_capital_share().data
    alpha_trend, _ = apply_hp_filter(alpha.dropna())

    plot_data = pd.DataFrame({
        "Capital Share (Raw)": alpha * 100,
        "Capital Share (HP Trend)": alpha_trend * 100,
    })

    ax = mg.line_plot(
        plot_data,
        width=[0.8, 2],
        alpha=[0.3, 1.0],
        color=["gray", "darkblue"],
    )

    # Add reference lines for common assumptions
    ax.axhline(y=30, color="red", linewidth=1, linestyle="--", alpha=0.7, label="α = 0.30 (typical assumption)")
    ax.axhline(y=25, color="orange", linewidth=1, linestyle="--", alpha=0.5, label="α = 0.25")
    ax.axhline(y=35, color="orange", linewidth=1, linestyle="--", alpha=0.5, label="α = 0.35")

    mg.finalise_plot(
        ax,
        title="Capital Share of Income (α)",
        ylabel="Per cent",
        legend={"loc": "best", "fontsize": "x-small", "ncol": 2},
        lfooter="Australia. α = GOS / (GOS + COE). Factor income from National Accounts.",
        rfooter="ABS 5206",
        show=show,
    )


def plot_all(result: DecompositionResult, show: bool = True) -> None:
    """Generate all standard plots."""
    plot_raw_data(result, show=show)
    plot_capital_share(show=show)
    plot_capital_growth(result, show=show)
    plot_labour_growth(result, show=show)
    plot_labour_force_growth(show=show)
    plot_labour_force_growth_quarterly(show=show)
    plot_hours_vs_labour_force(result, show=show)
    plot_labour_productivity(result, show=show)
    plot_labour_productivity_decomposition(result, show=show)
    plot_mfp_trend(result, show=show)
    plot_mfp_lambda_comparison(result, show=show)
    plot_potential_gdp(result, show=show)
    plot_growth_decomposition(result, show=show)
    plot_sensitivity(result, show=show)
    plot_phillips_crosscheck(result, show=show)
    plot_mfp_comparison(result, show=show)


# --- Summary Output ---


def print_summary(result: DecompositionResult, verbose: bool = False) -> None:
    """Print summary statistics."""
    print("SUMMARY: Cobb-Douglas MFP Decomposition")

    if verbose:
        print("\nModel Parameters:")
        print(f"  Capital share (α):     {result.alpha}")
        print(f"  Labour share (1-α):    {1 - result.alpha}")
        print(f"  HP filter lambda:      {HP_LAMBDA}")

        idx = result.growth["g_MFP"].dropna().index
        print(f"\nSample Period: {idx[0]} to {idx[-1]} ({len(idx)} quarters)")

        # Re-anchoring info
        reanchor_log = result.potential.attrs.get("reanchor_log", [])
        anchor_points = result.potential.attrs.get("anchor_points", [])
        if anchor_points:
            print(f"\nPotential GDP Anchor Points: {['Start'] + anchor_points}")
            if reanchor_log:
                print("  Re-anchoring adjustments (gap before reset):")
                for period, gap in reanchor_log:
                    print(f"    {period}: {gap:.2f}%")

        # Growth by decade
        print("\nMean Annual Growth by Decade:")
        decade_summary = result.summary_by_decade()
        print(decade_summary.round(2).to_string())

        # Latest values
        print("\nLatest Values:")
        print(f"  Output gap:        {result.potential['output_gap'].dropna().iloc[-1]:.2f}%")
        print(f"  MFP trend growth:  {annualize(result.mfp['MFP Trend'].iloc[-1]):.2f}% p.a.")

        # Sensitivity summary
        if result.sensitivity is not None:
            print("\nSensitivity Analysis (α):")
            print("         Raw MFP      Trend MFP     Recent Trend")
            print("  α     (Q mean)      (Q mean)       (2015+)")
            print("-" * 55)
            for _, row in result.sensitivity.iterrows():
                print(f" {row['alpha']:.2f}    {row['mfp_raw_mean']:.3f}%        {row['mfp_trend_mean']:.3f}%        {row['mfp_recent']:.3f}%")

        # Phillips curve
        if result.phillips is not None:
            corr = result.phillips.attrs.get("corr_deviation", np.nan)
            slope = result.phillips.attrs.get("slope", np.nan)
            print("\nPhillips Curve Cross-Check:")
            print(f"  Correlation (gap vs inflation deviation): {corr:.3f}")
            print(f"  Slope (gap → inflation deviation):        {slope:.3f}")
            if corr > 0.3:
                print("  ✓ Positive relationship suggests some validity")
            elif corr < -0.1:
                print("  ✗ Negative relationship - gap may be poorly identified")
            else:
                print("  ~ Weak relationship - gap may not capture demand pressure well")


# --- CLI Entry Point ---


def main(verbose: bool = False) -> None:
    """Run Cobb-Douglas MFP decomposition from command line."""
    chart_dir = Path(__file__).parent.parent.parent.parent / "charts" / "cobb_douglas"
    mg.set_chart_dir(str(chart_dir))
    mg.clear_chart_dir()

    print("Running Cobb-Douglas MFP Decomposition...")

    result = run_decomposition(start="1985Q1")

    print_summary(result, verbose=verbose)

    plot_all(result, show=False)

    print(f"Charts saved to: {chart_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Cobb-Douglas MFP Decomposition")
    parser.add_argument("-v", "--verbose", action="store_true", help="Print detailed output")
    args = parser.parse_args()

    main(verbose=args.verbose)
