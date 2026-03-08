"""KDE plots for scalar posterior distributions."""

import arviz as az
import matplotlib.pyplot as plt
import mgplot as mg
import numpy as np
from scipy import stats

from src.models.common.extraction import get_scalar_var, get_scalar_var_names


def plot_posteriors_kde(
    trace: az.InferenceData,
    *,
    rfooter: str = "",
    show: bool = False,
) -> None:
    """Plot separate Kernel Density Estimates for each coefficient posterior."""
    scalar_vars = get_scalar_var_names(trace)

    for var_name in sorted(scalar_vars):
        samples = get_scalar_var(var_name, trace)

        kde = stats.gaussian_kde(samples)
        x_range = np.linspace(samples.min(), samples.max(), 200)
        kde_values = kde(x_range)

        _, ax = plt.subplots()
        ax.plot(x_range, kde_values, color="steelblue", linewidth=2, label=var_name)
        ax.fill_between(x_range, 0, kde_values, color="steelblue", alpha=0.3)

        median_val = float(samples.quantile(0.5))
        ax.axvline(x=median_val, color="black", linestyle="--", linewidth=1, alpha=0.7)
        ax.text(median_val, kde_values.max(), f"{median_val:.3f}",
                ha="center", va="bottom", fontsize=9, color="black")

        mg.finalise_plot(
            ax,
            title=f"Posterior: {var_name}",
            xlabel="Coefficient value",
            lfooter="Red dashed line marks zero. Black dashed line marks median.",
            rfooter=rfooter,
            axvline={"x": 0, "color": "darkred", "linestyle": "--", "linewidth": 1.5},
            show=show,
        )
