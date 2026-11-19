"""Charts for the GDP nowcast summary: each model's mixture density, Q/Q and annual.

Built with raw matplotlib rather than `mg.line_plot`, because the x axis is a
growth-rate grid and mgplot requires a PeriodIndex or RangeIndex. It still goes
through `mg.finalise_plot`, so styling, footers and filenames match every
other chart.
"""

from dataclasses import dataclass

import matplotlib.pyplot as plt
import mgplot as mg
import numpy as np
from mgplot.utilities import get_color_list
from scipy.optimize import brentq
from scipy.stats import norm

from src.models.common.nowcast_core import compute_tty
from src.models.gdp_nowcast_summary.sources import TOLERANCE, Gathered, Nowcast
from src.paths import CHARTS

CHART_DIR = CHARTS / "GDP-Nowcast-Summary"

# The grid runs this many spreads either side of the outermost nowcasts, far
# enough that every curve has fallen to nothing at the edges.
GRID_SPREADS = 4.0
GRID_POINTS = 801

# Opacity of the area under each curve below zero growth.
NEGATIVE_SHADE_ALPHA = 0.25


@dataclass(frozen=True)
class Curve:
    """One model's density on the Q/Q grid, with its summary statistics."""

    label: str
    colour: str
    density: np.ndarray        # on the Q/Q grid
    median_qoq: float
    p_negative: float          # probability of negative Q/Q growth
    p_per_capita_negative: float  # probability Q/Q growth falls short of population growth


def _mixture_arrays(n: Nowcast) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the means, sds and weights of a model's mixture components."""
    return (
        np.array([c.mean for c in n.components]),
        np.array([c.sd for c in n.components]),
        np.array([c.weight for c in n.components]),
    )


def _spread(n: Nowcast) -> float:
    """Return the standard deviation of a model's Q/Q mixture."""
    m, s, w = _mixture_arrays(n)
    variance = float(w @ (s**2 + m**2)) - float(w @ m) ** 2
    return float(np.sqrt(variance))


def _curve(n: Nowcast, colour: str, grid: np.ndarray, population_log: float) -> Curve:
    """Mixture density on the grid, its median, P(<0) and P(per capita < 0)."""
    m, s, w = _mixture_arrays(n)

    def cdf(x: float) -> float:
        return float(w @ norm.cdf(x, m, s))

    # Median by root-finding on the CDF, bracketed far into both tails.
    reach = GRID_SPREADS * _spread(n)
    centre = n.mixture_mean
    median = float(brentq(lambda x: cdf(x) - 0.5, centre - 2 * reach, centre + 2 * reach))
    return Curve(
        label=n.label,
        colour=colour,
        density=norm.pdf(grid[:, None], m, s) @ w,
        median_qoq=median,
        p_negative=cdf(0.0),
        p_per_capita_negative=cdf(population_log),
    )


def _qoq_grid(nowcasts: list[Nowcast], thresholds: list[float]) -> np.ndarray:
    """Build a Q/Q grid wide enough for every curve, with each threshold on it exactly.

    The thresholds are zero and population growth: on the grid, the shaded area below
    each ends exactly at its line.
    """
    lo = min(n.mixture_mean - GRID_SPREADS * _spread(n) for n in nowcasts)
    hi = max(n.mixture_mean + GRID_SPREADS * _spread(n) for n in nowcasts)
    grid = np.linspace(lo, hi, GRID_POINTS)
    return np.union1d(grid, [t for t in thresholds if lo < t < hi])


def _plot(
    x: np.ndarray,
    curves: list[Curve],
    densities: list[np.ndarray],
    medians: list[float],
    labels: list[str],
    *,
    shade_negative: bool,
    title: str,
    xlabel: str,
    lheader: str,
) -> None:
    """Draw each density with a dashed line at its median, then finalise."""
    _, ax = plt.subplots()
    for c, density, label in zip(curves, densities, labels, strict=True):
        ax.plot(x, density, color=c.colour, lw=2.0, label=label)
        if shade_negative:
            below = x <= 0.0
            ax.fill_between(x[below], density[below], color=c.colour, alpha=NEGATIVE_SHADE_ALPHA, lw=0)
    ax.set_xlim(x[0], x[-1])
    ax.set_ylim(bottom=0)

    vlines: list[dict[str, object]] = [
        {"x": m, "color": c.colour, "linestyle": "--", "linewidth": 1.0}
        for c, m in zip(curves, medians, strict=True)
    ]
    if shade_negative:
        vlines.append({"x": 0.0, "color": "black", "linewidth": 0.8})

    mg.finalise_plot(
        ax,
        title=title,
        xlabel=xlabel,
        ylabel="Density",
        axvline=vlines,
        legend={"loc": "best", "fontsize": "x-small"},
        lheader=lheader,
        rfooter="Built using: ABS",
        lfooter="Australia. Dashed lines are medians. ",
        show=False,
    )


def plot_qoq(x: np.ndarray, curves: list[Curve], target: str) -> None:
    """Q/Q growth: every model's density, with the area below zero shaded."""
    _plot(
        x,
        curves,
        [c.density for c in curves],
        [c.median_qoq for c in curves],
        [f"{c.label}: median {c.median_qoq:+.2f}, P(<0) {c.p_negative:.0%}" for c in curves],
        shade_negative=True,
        title=f"GDP nowcasts for {target}: quarterly growth",
        xlabel="Q/Q growth, per cent",
        lheader="Shaded: probability of negative growth",
    )


def plot_per_capita(x: np.ndarray, curves: list[Curve], g: Gathered, population_log: float) -> None:
    """Q/Q growth per capita: every density shifted down by population growth.

    Population growth is the compounded quarterly rate of the forward proxy for annual ERP
    growth, taken as known; in logs, like the nowcasts, per capita growth is the difference.
    """
    _plot(
        x - population_log,
        curves,
        [c.density for c in curves],
        [c.median_qoq - population_log for c in curves],
        [
            f"{c.label}: median {c.median_qoq - population_log:+.2f}, P(<0) {c.p_per_capita_negative:.0%}"
            for c in curves
        ],
        shade_negative=True,
        title=f"GDP per capita nowcasts for {g.target}: quarterly growth",
        xlabel="Q/Q growth per capita, per cent",
        lheader=f"Population growth {g.population_qoq:.2f}% (compounded quarterly rate)",
    )


def plot_annual(x_qoq: np.ndarray, curves: list[Curve], g: Gathered) -> None:
    """Annual growth: the same distributions carried through compute_tty.

    The map from Q/Q to annual growth is monotonic, so medians carry across
    directly and each density is rescaled by dq/d(annual) = 100 / (100 + annual).
    Earlier quarters are held at their published values: the curves carry no
    uncertainty from revisions to them.
    """
    x_tty = np.array([compute_tty(q, g.gdp, g.target) for q in x_qoq])
    jacobian = 100.0 / (100.0 + x_tty)
    medians = [compute_tty(c.median_qoq, g.gdp, g.target) for c in curves]
    _plot(
        x_tty,
        curves,
        [c.density * jacobian for c in curves],
        medians,
        [f"{c.label}: median {m:.2f}" for c, m in zip(curves, medians, strict=True)],
        shade_negative=False,
        title=f"GDP nowcasts for {g.target}: annual growth",
        xlabel="Through-the-year growth, per cent",
        lheader="Earlier quarters as published: excludes their revisions",
    )


def run_analyse(g: Gathered) -> None:
    """Chart every usable nowcast, Q/Q and annual, and print the numbers."""
    mg.set_chart_dir(str(CHART_DIR))
    mg.clear_chart_dir()

    # The nowcasts' Q/Q growth is a log difference, so population growth goes into logs too.
    population_log = float(100 * np.log1p(g.population_qoq / 100))
    grid = _qoq_grid(g.nowcasts, [0.0, population_log])
    colours = get_color_list(len(g.nowcasts))
    curves = [
        _curve(n, colour, grid, population_log) for n, colour in zip(g.nowcasts, colours, strict=True)
    ]
    for c, n in zip(curves, g.nowcasts, strict=True):
        if abs(c.median_qoq - n.point_qoq) > TOLERANCE:
            msg = f"{n.label}: charted median {c.median_qoq:.6f} is not the model's own {n.point_qoq:.6f}"
            raise ValueError(msg)

    plot_qoq(grid, curves, str(g.target))
    plot_per_capita(grid, curves, g, population_log)
    plot_annual(grid, curves, g)

    width = max(len(label) for label in [*(c.label for c in curves), *g.omitted])
    print(f"\nGDP nowcast summary for {g.target}")
    print(f"  population growth: {g.population_qoq:.3f}% Q/Q (compounded), {g.population_annual:.3f}% annual")
    print(f"  {'model':<{width}} {'median Q/Q':>11} {'P(Q/Q<0)':>9} {'median TTY':>11}"
          f" {'pc Q/Q':>8} {'P(pc<0)':>8} {'pc TTY':>8}")
    for c in curves:
        tty = compute_tty(c.median_qoq, g.gdp, g.target)
        pc_tty = ((1 + tty / 100) / (1 + g.population_annual / 100) - 1) * 100
        print(f"  {c.label:<{width}} {c.median_qoq:>+10.2f}% {c.p_negative:>9.1%} {tty:>10.2f}%"
              f" {c.median_qoq - population_log:>+7.2f}% {c.p_per_capita_negative:>8.1%} {pc_tty:>+7.2f}%")
    for label, why in g.omitted.items():
        print(f"  {label:<{width}} not charted: {why}")
    print(f"\nCharts in {CHART_DIR}")
