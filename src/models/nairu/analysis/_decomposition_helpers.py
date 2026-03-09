"""Shared helpers for inflation decomposition plots."""

import arviz as az
import mgplot as mg
import pandas as pd
from matplotlib.axes import Axes

from src.models.common.extraction import get_scalar_var
from src.models.nairu.config import REGIME_COVID_START, REGIME_GFC_START


def get_regime_gamma(trace: az.InferenceData, obs_index: pd.PeriodIndex, prefix: str) -> pd.Series:
    """Extract regime-switching or single gamma as a time series."""
    if f"{prefix}_pre_gfc" in trace.posterior:
        gamma = pd.Series(index=obs_index, dtype=float)
        for var, mask in [
            (f"{prefix}_pre_gfc", obs_index < REGIME_GFC_START),
            (f"{prefix}_gfc", (obs_index >= REGIME_GFC_START) & (obs_index < REGIME_COVID_START)),
            (f"{prefix}_covid", obs_index >= REGIME_COVID_START),
        ]:
            gamma.loc[mask] = get_scalar_var(var, trace).median()
        return gamma
    return pd.Series(get_scalar_var(prefix, trace).median(), index=obs_index)


# --- LaTeX equation builders ---

def _supply_terms(has_import: bool, has_gscpi: bool) -> str:
    parts = []
    if has_import:
        parts.append(r"\lambda\Delta\rho^m_t")
    if has_gscpi:
        parts.append(r"\xi\cdot GSCPI^2")
    if not parts:
        return ""
    inner = " + ".join(parts)
    return r" + \underbrace{" + inner + r"}_{\mathrm{blue}}"


def eq_unscaled(has_import: bool, has_gscpi: bool) -> str:
    return (r"$\pi_t = \underbrace{\pi^e_t}_{\mathrm{grey}}"
            r" + \underbrace{\gamma\frac{U_t - U^*_t}{U_t}}_{\mathrm{orange}}"
            + _supply_terms(has_import, has_gscpi)
            + r" + \underbrace{\varepsilon_t}_{\mathrm{light\ blue}}$")


def wage_eq_unscaled(has_phi: bool, has_exp: bool) -> str:
    eq = (r"$\Delta ulc_t = \underbrace{\alpha + \pi^e_t}_{\mathrm{grey}}" if has_exp
          else r"$\Delta ulc_t = \underbrace{\alpha}_{\mathrm{grey}}")
    eq += r" + \underbrace{\gamma\frac{U_t - U^*_t}{U_t} + \lambda\frac{\Delta U_{t-1}}{U_t}}_{\mathrm{orange}}"
    if has_phi:
        eq += r" + \underbrace{\phi\Delta_4 dfd_t}_{\mathrm{blue}}"
    return eq + r" + \underbrace{\varepsilon_t}_{\mathrm{light\ blue}}$"


def hcoe_eq_unscaled(has_phi: bool, has_exp: bool) -> str:
    eq = (r"$\Delta hcoe_t = \underbrace{\alpha + \pi^e_t}_{\mathrm{grey}}" if has_exp
          else r"$\Delta hcoe_t = \underbrace{\alpha}_{\mathrm{grey}}")
    eq += r" + \underbrace{\gamma\frac{U_t - U^*_t}{U_t} + \lambda\frac{\Delta U_{t-1}}{U_t}}_{\mathrm{orange}}"
    if has_phi:
        eq += r" + \underbrace{\phi\Delta_4 dfd_t}_{\mathrm{blue}}"
    eq += r" + \underbrace{\psi \cdot mfp_t}_{\mathrm{green}}"
    return eq + r" + \underbrace{\varepsilon_t}_{\mathrm{light\ blue}}$"


# --- Shared plotting ---

def add_equation_box(ax: Axes, equation: str, x: float = 0.5, y: float = 0.02) -> None:
    """Add a LaTeX equation in a text box to the axes."""
    ax.text(x, y, equation, transform=ax.transAxes, fontsize=9,
            va="bottom", ha="center", usetex=True,
            bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.9, "edgecolor": "grey"})


def plot_decomposition_bars(
    bar_data: pd.DataFrame,
    observed: pd.Series,
    colors: list[str],
    title: str,
    equation: str,
    *,
    observed_color: str = "indigo",
    observed_label: str = "Observed (quarterly annualised)",
    target_line: float | None = 2.5,
    lheader: str = "",
    lfooter: str = "Australia. Decomposition based on augmented Phillips curve.",
    rfooter: str = "",
    eq_x: float = 0.4,
    eq_y: float = 0.75,
    show: bool = False,
) -> None:
    """Plot stacked decomposition bars with observed line overlay."""
    ax = mg.bar_plot(bar_data, stacked=True, color=colors)

    obs_plot = observed.copy()
    obs_plot.name = observed_label
    mg.line_plot(obs_plot, ax=ax, color=observed_color, width=1.5, zorder=10)

    add_equation_box(ax, equation, x=eq_x, y=eq_y)

    finalise_kwargs = {
        "title": title, "ylabel": "% p.a.",
        "lheader": lheader, "rheader": rfooter,
        "lfooter": lfooter,
        "legend": {"loc": "best", "fontsize": "x-small"},
        "y0": True, "show": show,
    }
    if target_line is not None:
        finalise_kwargs["axhline"] = {
            "y": target_line, "color": "darkred", "linestyle": "--",
            "linewidth": 1, "label": f"{target_line}% Target",
        }

    mg.finalise_plot(ax, **finalise_kwargs)
