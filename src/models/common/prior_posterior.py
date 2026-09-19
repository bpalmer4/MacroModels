"""One chart per estimated scalar: where the prior put it, where the data moved it.

Worth a chart per parameter rather than a summary table, because the table
hides what matters. A parameter can report a tidy mean and a plausible
interval while its chains peak in different places, which is visible here and
invisible in a row of numbers. The prior is drawn alongside because "the
posterior sits at x" means nothing without knowing where the prior already
put it.

The prior lookup is the caller's, since only the model knows which name
carries which prior and which of them depend on run settings. Everything else
is shared.

Built with raw matplotlib rather than `mg.line_plot`, because the x axis is a
parameter grid and mgplot requires a PeriodIndex or RangeIndex. It still goes
through `mg.finalise_plot`, so styling, footers and filenames match every
other chart in the directory.
"""

import math
from collections.abc import Callable
from typing import Any

import arviz as az
import matplotlib.pyplot as plt
import mgplot as mg
import numpy as np
import xarray as xr
from scipy.stats import gaussian_kde

_CHAIN_COLOURS = ("tab:blue", "tab:orange", "tab:green", "tab:red")

# A scalar parameter's posterior array is (chain, draw); a vector latent's is
# (chain, draw, time). This is how the two are told apart.
_SCALAR_NDIM = 2

# How far either side of the prior's own centre the grid reaches, before it is
# widened to cover the posterior draws.
_GRID_SD = 3.5

# A prior is one of three things: an analytic (kind, mu, sd) with kind
# "normal" or "half"; an array of prior-predictive draws, for models that
# sampled theirs; or a density callable evaluated on a grid, for anything the
# first two cannot say, such as a Beta. A callable is given a grid spanning
# the posterior draws, so it must return zero outside its own support.
Prior = tuple[str, float, float] | np.ndarray | Callable[[np.ndarray], np.ndarray]
PriorLookup = Callable[[str], Prior | None]


def halfnormal_moments(sigma: float) -> tuple[float, float]:
    """Return the mean and sd of HalfNormal(sigma)."""
    return sigma * math.sqrt(2.0 / math.pi), sigma * math.sqrt(1.0 - 2.0 / math.pi)


def _prior_pdf(kind: str, mu: float, sd: float, grid: np.ndarray) -> np.ndarray:
    """Return the prior density on `grid` for a Normal, HalfNormal or TruncatedNormal."""
    density = np.exp(-0.5 * ((grid - mu) / sd) ** 2) / (sd * math.sqrt(2 * math.pi))
    if kind == "half":
        density = np.where(grid < 0, 0.0, 2 * density)
    return density


def plot_parameter(
    name: str,
    draws: np.ndarray,
    prior: Prior,
    *,
    footers: dict[str, str],
    reference: float | None = None,
    label: str | None = None,
) -> None:
    """Draw one parameter's posterior against its prior, chains shown separately.

    `label` is the x-axis text, which should say what the parameter is and in
    what units. The bare name is a poor default and only used when a caller
    supplies nothing.

    `draws` is (chain, draw). `prior` is either (kind, mu, sd) with kind
    "normal" or "half", or an array of prior-predictive draws, for models that
    sampled their prior rather than stating it. `reference` draws a vertical
    line, for where another run or another model put the same quantity.
    """
    sampled = isinstance(prior, np.ndarray)
    callable_prior = callable(prior)
    flat = draws.ravel()
    if sampled:
        prior_flat = np.asarray(prior).ravel()
        kind, mu, sd = "sampled", float(prior_flat.mean()), float(prior_flat.std())
    elif callable_prior:
        kind, mu, sd = "curve", float("nan"), float("nan")
    else:
        kind, mu, sd = prior
    if sampled:
        lo, hi = float(np.asarray(prior).min()), float(np.asarray(prior).max())
    elif callable_prior:
        lo, hi = float(flat.min()), float(flat.max())
    else:
        lo = 0.0 if kind == "half" else mu - _GRID_SD * sd
        hi = mu + _GRID_SD * sd
    lo, hi = min(lo, float(flat.min())), max(hi, float(flat.max()))
    pad = 0.08 * (hi - lo)
    grid = np.linspace(lo - pad, hi + pad, 500)

    _, ax = plt.subplots(figsize=(9, 5))
    if sampled:
        label = "Prior, sampled"
        prior_density = gaussian_kde(np.asarray(prior).ravel())(grid)
    elif callable_prior:
        label = "Prior"
        prior_density = np.asarray(prior(grid))
    else:
        label = f"Prior, {'HalfNormal' if kind == 'half' else 'Normal'}"
        label += f"({sd:g})" if kind == "half" else f"({mu:g}, {sd:g})"
        prior_density = _prior_pdf(kind, mu, sd, grid)
    ax.plot(grid, prior_density, color="grey", ls="--", lw=1.8, label=label)
    for i, colour in zip(range(draws.shape[0]), _CHAIN_COLOURS, strict=False):
        ax.plot(grid, gaussian_kde(draws[i])(grid), color=colour, lw=1.0, ls=":", label=f"Chain {i}")
    ax.plot(grid, gaussian_kde(flat)(grid), color="black", lw=2.5, label="Posterior")
    if reference is not None:
        ax.axvline(reference, color="darkred", lw=1.2, ls="-.", label=f"separately estimated, {reference:g}")
    ax.set_xlim(grid[0], grid[-1])

    prior_mean = halfnormal_moments(sd)[0] if kind == "half" else mu
    means = f"posterior mean {flat.mean():.3f}"
    if not math.isnan(prior_mean):
        means += f", prior mean {prior_mean:.3f}"
    note = footers.get("lheader")
    mg.finalise_plot(
        ax,
        title=f"{name}: posterior against prior",
        xlabel=label or name,
        ylabel="Density",
        legend={"loc": "best", "fontsize": "x-small"},
        lheader=f"{note}. {means}" if note else means,
        lfooter=footers.get("lfooter", "") + "Dotted lines are individual chains. ",
        rfooter=footers.get("rfooter", ""),
        show=False,
    )


def plot_all(
    posterior: xr.Dataset | az.InferenceData,
    prior_for: PriorLookup,
    *,
    footers: dict[str, str],
    references: dict[str, float] | None = None,
    labels: dict[str, str] | None = None,
) -> int:
    """Draw a chart for every scalar in `posterior` whose prior `prior_for` knows.

    Returns the number drawn, so a caller can report it and a silent shortfall
    is visible.
    """
    group: Any = getattr(posterior, "posterior", posterior)
    references = references or {}
    labels = labels or {}
    drawn = 0
    for raw in group.data_vars:
        name = str(raw)
        values = np.asarray(group[name])
        prior = prior_for(name)
        if values.ndim != _SCALAR_NDIM or prior is None:
            continue
        plot_parameter(name, values, prior, footers=footers,
                       reference=references.get(name), label=labels.get(name))
        drawn += 1
    return drawn
