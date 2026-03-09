"""Phillips curve demand pressure visualization."""

import matplotlib.pyplot as plt
import mgplot as mg
import numpy as np

from src.models.common.extraction import get_scalar_var, get_vector_var
from src.models.nairu.results import NAIRUResults
from src.utilities.rate_conversion import annualize

# (gamma_var_suffix, color, label) for each curve type
_CURVE_SPECS = [
    ("pi", "steelblue", "Price"),
    ("wg", "darkorange", "Wage-ULC"),
    ("hcoe", "green", "Wage-HCOE"),
]


def _curve_bands(
    u_range: np.ndarray, gamma: np.ndarray, nairu_last: np.ndarray, idx: np.ndarray,
) -> np.ndarray:
    """Compute (lower, median, upper) annualised Phillips curves."""
    curves = np.array([
        annualize(gamma[i] * (u_range - nairu_last[i]) / u_range) for i in idx
    ])
    return np.percentile(curves, [5, 50, 95], axis=0)


def plot_phillips_curves(
    results: NAIRUResults,
    *,
    rfooter: str = "",
    show: bool = False,
) -> None:
    """Plot convex Phillips curves with posterior uncertainty bands."""
    trace = results.trace
    nairu_last = get_vector_var("nairu", trace).to_numpy()[-1, :]
    u_current = float(results.obs["U"][-1])
    u_range = np.linspace(3.0, 8.0, 100)

    n_draw = min(500, nairu_last.shape[0])
    rng = np.random.default_rng(42)
    idx = rng.choice(nairu_last.shape[0], n_draw, replace=False)

    _, ax = plt.subplots(figsize=(9, 6))
    annotations = []
    lheader_parts = []

    for suffix, color, label in _CURVE_SPECS:
        covid_var = f"gamma_{suffix}_covid"
        single_var = f"gamma_{suffix}"
        var = covid_var if covid_var in trace.posterior else single_var
        if var not in trace.posterior:
            continue

        gamma = get_scalar_var(var, trace).to_numpy()
        lower, median, upper = _curve_bands(u_range, gamma, nairu_last, idx)
        ax.fill_between(u_range, lower, upper, alpha=0.2, color=color)
        ax.plot(u_range, median, color=color, linewidth=2, label=f"{label} Phillips Curve")

        u_idx = np.abs(u_range - u_current).argmin()
        dev = median[u_idx]
        slope = np.gradient(median, u_range)[u_idx]
        annotations.append(f"  {label}: {dev:+.2f}pp (slope {slope:.2f})")

        if suffix == "wg":
            lheader_parts.append("ULC = Unit Labour Costs.")
        elif suffix == "hcoe":
            lheader_parts.append("HCOE = Hourly Compensation of Employees.")

    # Reference lines
    nairu_recent = float(np.median(nairu_last))
    ax.axhline(0, color="black", linewidth=0.5)
    ax.axvline(nairu_recent, color="purple", linestyle="-.", alpha=0.7,
               linewidth=1.5, label=f"NAIRU = {nairu_recent:.1f}%")
    ax.axvline(u_current, color="darkred", linestyle="--", alpha=0.7,
               linewidth=1.5, label=f"Current U = {u_current:.1f}%")

    if annotations:
        ax.annotate(
            f"At U = {u_current:.1f}%:\n" + "\n".join(annotations),
            xy=(0.97, 0.97), xycoords="axes fraction",
            ha="right", va="top", fontsize=9,
            bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.8},
        )

    mg.finalise_plot(
        ax,
        title="Phillips Curve Demand Pressures - Post-COVID Regime",
        xlabel="Unemployment Rate (%)",
        ylabel="Demand pressures (pp, annualised)",
        legend={"loc": "best", "fontsize": "x-small"},
        lheader=" ".join(lheader_parts),
        lfooter=f"Chart for {results.obs_index[-1]}. Convex form: γ(U-U*)/U. 90% CI.",
        rfooter=rfooter,
        show=show,
    )
