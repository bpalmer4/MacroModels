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
    import matplotlib.pyplot as plt


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
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Plot NAIRU estimate with uncertainty bands.

    Args:
        stats: DataFrame from compute_nairu_stats()
        ax: Matplotlib axes (created if None)

    Returns:
        Matplotlib axes
    """
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(12, 6))

    # Unemployment rate
    ax.plot(
        stats.index.to_timestamp(),
        stats["U"].values,
        "b-",
        linewidth=1,
        label="Unemployment Rate",
        alpha=0.7,
    )

    # NAIRU with bands
    ax.plot(
        stats.index.to_timestamp(),
        stats["NAIRU_median"].values,
        "r-",
        linewidth=2,
        label="NAIRU (median)",
    )
    ax.fill_between(
        stats.index.to_timestamp(),
        stats["NAIRU_lower"].values,
        stats["NAIRU_upper"].values,
        alpha=0.3,
        color="red",
        label="80% credible interval",
    )

    ax.set_ylabel("Percent")
    ax.legend(loc="upper right")

    return ax


def plot_unemployment_gap(
    stats: pd.DataFrame,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Plot unemployment gap (U - NAIRU).

    Args:
        stats: DataFrame from compute_nairu_stats()
        ax: Matplotlib axes (created if None)

    Returns:
        Matplotlib axes
    """
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(12, 6))

    gap = stats["U_gap"]

    ax.plot(gap.index.to_timestamp(), gap.values, "b-", linewidth=1.5)
    ax.axhline(0, color="black", linestyle="--", linewidth=0.8)
    ax.fill_between(gap.index.to_timestamp(), 0, gap.values, alpha=0.3)

    ax.set_ylabel("Percentage points (U - NAIRU)")

    return ax


def plot_output_gap(
    stats: pd.DataFrame,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Plot output gap (log GDP - log potential).

    Args:
        stats: DataFrame from compute_potential_stats()
        ax: Matplotlib axes (created if None)

    Returns:
        Matplotlib axes
    """
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(12, 6))

    gap = stats["output_gap"]

    ax.plot(gap.index.to_timestamp(), gap.values, "b-", linewidth=1.5)
    ax.axhline(0, color="black", linestyle="--", linewidth=0.8)
    ax.fill_between(gap.index.to_timestamp(), 0, gap.values, alpha=0.3)

    ax.set_ylabel("% of potential GDP")

    return ax


def plot_gdp_vs_potential(
    stats: pd.DataFrame,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Plot actual GDP vs potential GDP.

    Args:
        stats: DataFrame from compute_potential_stats()
        ax: Matplotlib axes (created if None)

    Returns:
        Matplotlib axes
    """
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(12, 6))

    ax.plot(
        stats.index.to_timestamp(),
        stats["gdp_level"].values,
        "b-",
        linewidth=1.5,
        label="Actual GDP",
    )
    ax.plot(
        stats.index.to_timestamp(),
        stats["potential_level"].values,
        "r--",
        linewidth=1.5,
        label="Potential GDP",
    )

    ax.set_ylabel("$ Millions (CVM)")
    ax.legend(loc="upper left")

    return ax


def plot_potential_growth(
    stats: pd.DataFrame,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Plot potential output growth rate.

    Args:
        stats: DataFrame from compute_potential_stats()
        ax: Matplotlib axes (created if None)

    Returns:
        Matplotlib axes
    """
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(12, 6))

    growth = stats["potential_growth"].dropna()

    ax.plot(growth.index.to_timestamp(), growth.values, "b-", linewidth=1.5)
    ax.axhline(0, color="black", linestyle="--", linewidth=0.8)

    ax.set_ylabel("Annualized quarterly growth (%)")

    return ax


def plot_gaps_comparison(
    nairu_stats: pd.DataFrame,
    potential_stats: pd.DataFrame,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Plot output gap and unemployment gap together.

    Unemployment gap is inverted for visual comparison (Okun's Law).

    Args:
        nairu_stats: DataFrame from compute_nairu_stats()
        potential_stats: DataFrame from compute_potential_stats()
        ax: Matplotlib axes (created if None)

    Returns:
        Matplotlib axes
    """
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(12, 6))

    ax.plot(
        potential_stats.index.to_timestamp(),
        potential_stats["output_gap"].values,
        "b-",
        linewidth=1.5,
        label="Output Gap",
    )
    ax.plot(
        nairu_stats.index.to_timestamp(),
        -nairu_stats["U_gap"].values,
        "r-",
        linewidth=1.5,
        label="Unemployment Gap (inverted)",
    )
    ax.axhline(0, color="black", linestyle="--", linewidth=0.8)

    ax.set_ylabel("Percent")
    ax.legend(loc="upper right")

    return ax
