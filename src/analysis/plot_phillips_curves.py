"""Phillips curve visualizations showing non-linear relationship.

Plots the convex Phillips curve specification used by the RBA:
    pi - pi_anchor = gamma * (u - u*) / u

This specification implies a slope of gamma * u* / u^2, meaning the curve
is steeper when:
1. Unemployment is low (convex shape)
2. NAIRU is high (shifting steepness over time)

Reference: RBA RDP 2021-09, Debelle & Vickery (1997)
"""

from typing import Any

import arviz as az  # noqa: TC002
import matplotlib.pyplot as plt
import mgplot as mg
import numpy as np
import pandas as pd

from src.analysis.extraction import get_scalar_var, get_vector_var
from src.analysis.rate_conversion import annualize
from src.equations import REGIME_COVID_START, REGIME_GFC_START


def _compute_phillips_curve(
    u_range: np.ndarray,
    nairu: float,
    gamma: float,
) -> np.ndarray:
    """Compute inflation deviation for given unemployment rates.

    Uses convex specification: gamma * (u - u*) / u
    """
    return gamma * (u_range - nairu) / u_range


def _compute_slope(u: float, nairu: float, gamma: float) -> float:
    """Compute Phillips curve slope at unemployment rate u.

    Slope = gamma * u* / u^2
    """
    return gamma * nairu / (u**2)


def plot_phillips_curves(
    trace: az.InferenceData,
    obs: dict[str, np.ndarray],
    obs_index: pd.PeriodIndex,
    model_name: str = "Model",
    show: bool = False,
    **kwargs: Any,  # noqa: ANN401
) -> None:
    """Plot price and wage Phillips curves on the same panel.

    Shows all three curves (price, wage-ULC, wage-HCOE) with posterior median
    and 90% credible intervals, following the RBA's presentation style.
    Uses current (post-COVID) regime gammas.
    """
    # Extract posteriors - use current regime (covid) gammas
    gamma_pi = get_scalar_var("gamma_pi_covid", trace).to_numpy()
    gamma_wg = get_scalar_var("gamma_wg_covid", trace).to_numpy()
    gamma_hcoe = get_scalar_var("gamma_hcoe_covid", trace).to_numpy()
    nairu_samples = get_vector_var("nairu", trace).to_numpy()  # shape: (time, samples)

    # Use recent NAIRU (last quarter median across all samples)
    nairu_recent = float(np.median(nairu_samples[-1, :]))

    # Unemployment range for curve
    u_min, u_max = 3.0, 8.0
    u_range = np.linspace(u_min, u_max, 100)

    # Current unemployment
    u_current = float(obs["U"][-1])

    # Subsample posteriors for speed
    n_curves = min(500, len(gamma_pi))
    rng = np.random.default_rng(42)
    idx = rng.choice(len(gamma_pi), n_curves, replace=False)

    # Compute curves for each posterior sample (annualized)
    curves_pi = np.array([
        annualize(_compute_phillips_curve(u_range, nairu_samples[-1, i], gamma_pi[i]))
        for i in idx
    ])
    curves_wg = np.array([
        annualize(_compute_phillips_curve(u_range, nairu_samples[-1, i], gamma_wg[i]))
        for i in idx
    ])
    curves_hcoe = np.array([
        annualize(_compute_phillips_curve(u_range, nairu_samples[-1, i], gamma_hcoe[i]))
        for i in idx
    ])

    # Create figure
    _, ax = plt.subplots(figsize=(9, 6))

    # Price Phillips curve
    lower_pi = np.percentile(curves_pi, 5, axis=0)
    upper_pi = np.percentile(curves_pi, 95, axis=0)
    median_pi = np.percentile(curves_pi, 50, axis=0)

    ax.fill_between(u_range, lower_pi, upper_pi, alpha=0.2, color="steelblue")
    ax.plot(u_range, median_pi, color="steelblue", linewidth=2, label="Price Phillips Curve")

    # Wage-ULC Phillips curve
    lower_wg = np.percentile(curves_wg, 5, axis=0)
    upper_wg = np.percentile(curves_wg, 95, axis=0)
    median_wg = np.percentile(curves_wg, 50, axis=0)

    ax.fill_between(u_range, lower_wg, upper_wg, alpha=0.2, color="darkorange")
    ax.plot(u_range, median_wg, color="darkorange", linewidth=2, label="Wage-ULC Phillips Curve")

    # Wage-HCOE Phillips curve
    lower_hcoe = np.percentile(curves_hcoe, 5, axis=0)
    upper_hcoe = np.percentile(curves_hcoe, 95, axis=0)
    median_hcoe = np.percentile(curves_hcoe, 50, axis=0)

    ax.fill_between(u_range, lower_hcoe, upper_hcoe, alpha=0.2, color="green")
    ax.plot(u_range, median_hcoe, color="green", linewidth=2, label="Wage-HCOE Phillips Curve")

    # Reference lines
    ax.axhline(0, color="black", linewidth=0.5)
    ax.axvline(nairu_recent, color="purple", linestyle="-.", alpha=0.7,
               linewidth=1.5, label=f"NAIRU = {nairu_recent:.1f}%")
    ax.axvline(u_current, color="darkred", linestyle="--", alpha=0.7,
               linewidth=1.5, label=f"Current U = {u_current:.1f}%")

    # Annotate slopes at current unemployment (from annualized curves)
    # Find index closest to current unemployment
    u_idx = np.abs(u_range - u_current).argmin()
    # Numerical slope: rise/run using adjacent points
    if u_idx > 0:
        slope_pi = (median_pi[u_idx] - median_pi[u_idx - 1]) / (u_range[u_idx] - u_range[u_idx - 1])
        slope_wg = (median_wg[u_idx] - median_wg[u_idx - 1]) / (u_range[u_idx] - u_range[u_idx - 1])
        slope_hcoe = (median_hcoe[u_idx] - median_hcoe[u_idx - 1]) / (u_range[u_idx] - u_range[u_idx - 1])
    else:
        slope_pi = (median_pi[1] - median_pi[0]) / (u_range[1] - u_range[0])
        slope_wg = (median_wg[1] - median_wg[0]) / (u_range[1] - u_range[0])
        slope_hcoe = (median_hcoe[1] - median_hcoe[0]) / (u_range[1] - u_range[0])

    # Get deviations at current unemployment
    dev_pi = median_pi[u_idx]
    dev_wg = median_wg[u_idx]
    dev_hcoe = median_hcoe[u_idx]

    ax.annotate(
        f"At U={u_current:.1f}%:\n"
        f"Demand pressures:\n"
        f"  Price: {dev_pi:+.2f}pp\n"
        f"  ULC: {dev_wg:+.2f}pp\n"
        f"  HCOE: {dev_hcoe:+.2f}pp\n"
        f"Slopes:\n"
        f"  Price: {slope_pi:.2f}\n"
        f"  ULC: {slope_wg:.2f}\n"
        f"  HCOE: {slope_hcoe:.2f}",
        xy=(0.97, 0.97), xycoords="axes fraction",
        ha="right", va="top", fontsize=9,
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.8}
    )

    defaults = {
        "title": "Phillips Curve Demand Pressures - Post-COVID Regime",
        "xlabel": "Unemployment Rate (%)",
        "ylabel": "Demand pressures (pp, annualised)",
        "legend": {"loc": "best", "fontsize": "x-small"},
        "lheader": "ULC = Unit Labour Costs; HCOE = Hourly Compensation of Employees.",
        "lfooter": f"Chart for {obs_index[-1]}. Convex form: gamma*(U-NAIRU)/U. Slope = gamma*NAIRU/U^2. 90% CI shown.",
        "rfooter": model_name,
    }
    for key in list(defaults.keys()):
        if key in kwargs:
            defaults.pop(key)

    mg.finalise_plot(ax, **defaults, **kwargs, show=show)


def plot_phillips_curve_slope(
    trace: az.InferenceData,
    obs_index: pd.PeriodIndex,
    curve_type: str = "price",
    model_name: str = "Model",
    show: bool = False,
    **kwargs: Any,  # noqa: ANN401
) -> None:
    """Plot how Phillips curve steepness has evolved over time.

    Shows the implied slope (gamma / u*) over the sample period,
    with regime-switching gammas (pre-GFC, post-GFC, post-COVID).

    Args:
        trace: ArviZ InferenceData with posterior samples
        obs_index: PeriodIndex for the observation period
        curve_type: "price", "wage" (ULC), or "hcoe" for respective Phillips curves
        model_name: Name for the chart footer
        show: Whether to display the plot
        **kwargs: Additional arguments passed to finalise_plot
    """
    # Configuration based on curve type
    if curve_type == "wage":
        gamma_prefix = "gamma_wg"
        color = "darkorange"
        title = "Wage-ULC Phillips Curve Slope Over Time"
        lheader = "ULC = Unit Labour Costs."
    elif curve_type == "hcoe":
        gamma_prefix = "gamma_hcoe"
        color = "green"
        title = "Wage-HCOE Phillips Curve Slope Over Time"
        lheader = "HCOE = Hourly Compensation of Employees."
    else:
        gamma_prefix = "gamma_pi"
        color = "steelblue"
        title = "Price Phillips Curve Slope Over Time"
        lheader = ""

    # Extract regime-specific gamma posteriors
    gamma_pre_gfc = float(np.median(get_scalar_var(f"{gamma_prefix}_pre_gfc", trace).to_numpy()))
    gamma_gfc = float(np.median(get_scalar_var(f"{gamma_prefix}_gfc", trace).to_numpy()))
    gamma_covid = float(np.median(get_scalar_var(f"{gamma_prefix}_covid", trace).to_numpy()))

    nairu_samples = get_vector_var("nairu", trace).to_numpy()  # shape: (time, samples)
    nairu_median = np.median(nairu_samples, axis=1)  # median across samples for each time

    # Compute regime-specific gamma for each period
    gamma_by_period = np.where(
        obs_index < REGIME_GFC_START,
        gamma_pre_gfc,
        np.where(obs_index < REGIME_COVID_START, gamma_gfc, gamma_covid),
    )

    # Compute slope at NAIRU over time: slope = gamma / u*
    slope = pd.Series(gamma_by_period / nairu_median, index=obs_index, name="Phillips Curve Slope")

    ax = mg.line_plot(slope, color=color, width=1.5)
    ax.axhline(0, color="black", linewidth=0.5)

    # Add regime boundary lines with labels
    ax.axvline(x=REGIME_GFC_START.ordinal, color="darkred", linestyle="--", linewidth=1, alpha=0.7,
               label=f"Regime change: GFC ({REGIME_GFC_START})")
    ax.axvline(x=REGIME_COVID_START.ordinal, color="darkgreen", linestyle="-.", linewidth=1, alpha=0.7,
               label=f"Regime change: COVID ({REGIME_COVID_START})")

    defaults = {
        "title": title,
        "ylabel": "Slope at NAIRU (gamma/u*)",
        "legend": {"loc": "best", "fontsize": "x-small"},
        "lheader": lheader,
        "lfooter": f"γ: pre-GFC={gamma_pre_gfc:.2f}, post-GFC={gamma_gfc:.2f}, post-COVID={gamma_covid:.2f}",
        "rfooter": model_name,
    }
    for key in list(defaults.keys()):
        if key in kwargs:
            defaults.pop(key)

    mg.finalise_plot(ax, **defaults, **kwargs, show=show)
