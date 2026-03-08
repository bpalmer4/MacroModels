"""Observation model plotting utilities."""

import math

import matplotlib.pyplot as plt
import mgplot as mg
import numpy as np
import pandas as pd


def plot_obs_grid(
    obs: dict[str, np.ndarray],
    obs_index: pd.PeriodIndex,
    *,
    title: str = "Model Input Variables",
    show: bool = False,
) -> None:
    """Plot all observation variables in a grid for quick visual inspection."""
    n_vars = len(obs)
    n_cols = 4
    n_rows = math.ceil(n_vars / n_cols)

    _fig, axes = plt.subplots(n_rows, n_cols, figsize=(14.0, 2.5 * n_rows))
    axes = axes.flatten()

    last_used = 0
    for i, (name, values) in enumerate(obs.items()):
        ax = axes[i]
        series = pd.Series(values, index=obs_index, name=name)
        mg.line_plot(series, ax=ax, width=1, max_ticks=5)
        ax.set_title(name, fontsize=10)
        ax.axhline(y=0, color="grey", linewidth=0.5, linestyle="-")
        last_used = i

    for j in range(last_used + 1, len(axes)):
        axes[j].set_visible(False)

    mg.finalise_plot(
        axes[last_used],
        suptitle=title,
        figsize=(14.0, 2.5 * n_rows),
        show=show,
    )
