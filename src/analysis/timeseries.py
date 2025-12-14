"""Time series analysis for latent state variables.

Provides functions to:
- Compute derived series from model posteriors (gaps, Taylor rule, etc.)
- Plot time series with uncertainty bands

All plotting functions return Axes objects for composition.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from matplotlib.axes import Axes


# --- Derived Series Computation ---


def compute_nairu_stats(
    nairu_samples: pd.DataFrame,
    observed_u: pd.Series,
) -> pd.DataFrame:
    """Compute NAIRU statistics from posterior samples.

    Args:
        nairu_samples: DataFrame of posterior samples (rows=time, cols=draws)
        observed_u: Observed unemployment rate series

    Returns:
        DataFrame with columns: NAIRU_median, NAIRU_lower, NAIRU_upper,
        NAIRU_std, U, U_gap
    """
    stats = pd.DataFrame({
        "NAIRU_median": nairu_samples.median(axis=1),
        "NAIRU_lower": nairu_samples.quantile(0.1, axis=1),
        "NAIRU_upper": nairu_samples.quantile(0.9, axis=1),
        "NAIRU_std": nairu_samples.std(axis=1),
        "U": observed_u,
    })
    stats["U_gap"] = stats["U"] - stats["NAIRU_median"]

    return stats


def compute_potential_stats(
    potential_samples: pd.DataFrame,
    log_gdp: pd.Series,
) -> pd.DataFrame:
    """Compute potential output statistics from posterior samples.

    Args:
        potential_samples: DataFrame of posterior samples (rows=time, cols=draws)
        log_gdp: Log of actual GDP (×100 scale)

    Returns:
        DataFrame with columns: potential_median, potential_lower, potential_upper,
        potential_std, log_gdp, gdp_level, potential_level, output_gap, potential_growth
    """
    stats = pd.DataFrame({
        "potential_median": potential_samples.median(axis=1),
        "potential_lower": potential_samples.quantile(0.1, axis=1),
        "potential_upper": potential_samples.quantile(0.9, axis=1),
        "potential_std": potential_samples.std(axis=1),
        "log_gdp": log_gdp,
    })
    stats["gdp_level"] = np.exp(stats["log_gdp"] / 100)
    stats["potential_level"] = np.exp(stats["potential_median"] / 100)
    stats["output_gap"] = stats["log_gdp"] - stats["potential_median"]
    stats["potential_growth"] = stats["potential_median"].diff(1) * 4  # Annualized

    return stats


def compute_taylor_rule(
    nairu_stats: pd.DataFrame,
    potential_stats: pd.DataFrame,
    pi_anchor: pd.Series,
    r_star: pd.Series,
    alpha_pi: float = 1.5,
    alpha_u: float = 0.5,
) -> pd.Series:
    """Compute Taylor rule implied interest rate.

    Taylor rule: r = r* + α_π × (π - π*) + α_u × (U* - U)

    Args:
        nairu_stats: DataFrame from compute_nairu_stats()
        potential_stats: DataFrame from compute_potential_stats()
        pi_anchor: Inflation anchor (annual %)
        r_star: Equilibrium real rate
        alpha_pi: Weight on inflation gap (default 1.5)
        alpha_u: Weight on unemployment gap (default 0.5)

    Returns:
        Taylor rule implied rate
    """
    # Use output gap as proxy for activity (inverted U gap)
    output_gap = potential_stats["output_gap"]

    # Taylor rule (simplified)
    taylor = r_star + alpha_pi * (pi_anchor - 2.5) / 4 + alpha_u * output_gap

    return taylor


# --- Time Series Plotting ---


def plot_nairu(
    stats: pd.DataFrame,
    ax: Axes | None = None,
) -> Axes:
    """Plot NAIRU estimate with uncertainty bands.

    Args:
        stats: DataFrame from compute_nairu_stats()
        ax: Matplotlib axes (created if None)

    Returns:
        Matplotlib axes
    """
    import mgplot as mg

    # NAIRU credible interval band
    band = pd.DataFrame({
        "lower": stats["NAIRU_lower"],
        "upper": stats["NAIRU_upper"],
    }, index=stats.index)
    ax = mg.fill_between_plot(band, ax=ax, color="red", alpha=0.3, label="80% credible interval")

    # NAIRU median
    nairu_median = stats["NAIRU_median"].copy()
    nairu_median.name = "NAIRU (median)"
    ax = mg.line_plot(nairu_median, ax=ax, color="red", width=2)

    # Unemployment rate
    u_rate = stats["U"].copy()
    u_rate.name = "Unemployment Rate"
    ax = mg.line_plot(u_rate, ax=ax, color="blue", width=1)

    return ax


def plot_unemployment_gap(
    stats: pd.DataFrame,
    ax: Axes | None = None,
) -> Axes:
    """Plot unemployment gap (U - NAIRU).

    Args:
        stats: DataFrame from compute_nairu_stats()
        ax: Matplotlib axes (created if None)

    Returns:
        Matplotlib axes
    """
    import mgplot as mg

    gap = stats["U_gap"]

    # Create fill bands for positive/negative
    positive_fill = pd.DataFrame({
        "lower": 0.0,
        "upper": gap.clip(lower=0),
    }, index=gap.index)
    negative_fill = pd.DataFrame({
        "lower": gap.clip(upper=0),
        "upper": 0.0,
    }, index=gap.index)

    ax = mg.fill_between_plot(positive_fill, ax=ax, color="blue", alpha=0.3)
    ax = mg.fill_between_plot(negative_fill, ax=ax, color="blue", alpha=0.3)

    # Gap line
    gap_series = gap.copy()
    gap_series.name = "Unemployment Gap"
    ax = mg.line_plot(gap_series, ax=ax, color="blue", width=1.5)

    return ax


def plot_output_gap(
    stats: pd.DataFrame,
    ax: Axes | None = None,
) -> Axes:
    """Plot output gap (log GDP - log potential).

    Args:
        stats: DataFrame from compute_potential_stats()
        ax: Matplotlib axes (created if None)

    Returns:
        Matplotlib axes
    """
    import mgplot as mg

    gap = stats["output_gap"]

    # Create fill bands for positive/negative
    positive_fill = pd.DataFrame({
        "lower": 0.0,
        "upper": gap.clip(lower=0),
    }, index=gap.index)
    negative_fill = pd.DataFrame({
        "lower": gap.clip(upper=0),
        "upper": 0.0,
    }, index=gap.index)

    ax = mg.fill_between_plot(positive_fill, ax=ax, color="blue", alpha=0.3)
    ax = mg.fill_between_plot(negative_fill, ax=ax, color="blue", alpha=0.3)

    # Gap line
    gap_series = gap.copy()
    gap_series.name = "Output Gap"
    ax = mg.line_plot(gap_series, ax=ax, color="blue", width=1.5)

    return ax


def plot_gdp_vs_potential(
    stats: pd.DataFrame,
    ax: Axes | None = None,
) -> Axes:
    """Plot actual GDP vs potential GDP.

    Args:
        stats: DataFrame from compute_potential_stats()
        ax: Matplotlib axes (created if None)

    Returns:
        Matplotlib axes
    """
    import mgplot as mg

    # Actual GDP
    actual = stats["gdp_level"].copy()
    actual.name = "Actual GDP"
    ax = mg.line_plot(actual, ax=ax, color="blue", width=1.5)

    # Potential GDP
    potential = stats["potential_level"].copy()
    potential.name = "Potential GDP"
    ax = mg.line_plot(potential, ax=ax, color="red", width=1.5, style="--")

    return ax


def plot_potential_growth(
    stats: pd.DataFrame,
    ax: Axes | None = None,
) -> Axes:
    """Plot potential output growth rate.

    Args:
        stats: DataFrame from compute_potential_stats()
        ax: Matplotlib axes (created if None)

    Returns:
        Matplotlib axes
    """
    import mgplot as mg

    growth = stats["potential_growth"].dropna()
    growth_series = growth.copy()
    growth_series.name = "Potential Growth"
    ax = mg.line_plot(growth_series, ax=ax, color="blue", width=1.5)

    return ax


def plot_gaps_comparison(
    nairu_stats: pd.DataFrame,
    potential_stats: pd.DataFrame,
    ax: Axes | None = None,
) -> Axes:
    """Plot output gap and unemployment gap together.

    Unemployment gap is inverted for visual comparison (Okun's Law).

    Args:
        nairu_stats: DataFrame from compute_nairu_stats()
        potential_stats: DataFrame from compute_potential_stats()
        ax: Matplotlib axes (created if None)

    Returns:
        Matplotlib axes
    """
    import mgplot as mg

    # Output gap
    output_gap = potential_stats["output_gap"].copy()
    output_gap.name = "Output Gap"
    ax = mg.line_plot(output_gap, ax=ax, color="blue", width=1.5)

    # Unemployment gap (inverted)
    u_gap_inv = (-nairu_stats["U_gap"]).copy()
    u_gap_inv.name = "Unemployment Gap (inverted)"
    ax = mg.line_plot(u_gap_inv, ax=ax, color="red", width=1.5)

    return ax
