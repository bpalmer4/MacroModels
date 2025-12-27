"""KDE plots for scalar posterior distributions."""

import arviz as az
import matplotlib.pyplot as plt
import mgplot as mg
import numpy as np
from scipy import stats

from src.models.nairu.analysis.extraction import get_scalar_var, get_scalar_var_names


def _place_model_name(model_name: str, kwargs: dict) -> dict:
    """Place model_name in first available footer/header slot."""
    for slot in ["rfooter", "rheader", "lheader"]:
        if slot not in kwargs:
            return {slot: model_name}
    return {}


def plot_posteriors_kde(
    trace: az.InferenceData,
    model_name: str = "Model",
    **kwargs,
) -> None:
    """Plot separate Kernel Density Estimates for each coefficient posterior."""
    scalar_vars = get_scalar_var_names(trace)

    for var_name in sorted(scalar_vars):
        samples = get_scalar_var(var_name, trace)

        _, ax = plt.subplots()

        samples.plot.kde(ax=ax, color="steelblue", linewidth=2)

        kde = stats.gaussian_kde(samples)
        x_range = np.linspace(samples.min(), samples.max(), 200)
        kde_values = kde(x_range)
        ax.fill_between(x_range, kde_values, alpha=0.3, color="steelblue")

        ax.axvline(x=0, color="darkred", linestyle="--", linewidth=1.5)

        median_val = samples.quantile(0.5)
        ax.axvline(x=median_val, color="black", linestyle="--", linewidth=1, alpha=0.7)

        max_y = kde_values.max()
        ax.text(
            median_val,
            max_y,
            f"{median_val:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
            color="black",
        )

        defaults = {
            "title": f"{var_name} Posterior",
            "xlabel": "Coefficient value",
            "lfooter": "Red dashed line marks zero. Black dashed line marks median.",
            **_place_model_name(model_name, kwargs),
        }
        for key in list(defaults.keys()):
            if key in kwargs:
                defaults.pop(key)

        mg.finalise_plot(ax, **defaults, **kwargs)
