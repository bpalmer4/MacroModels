"""Inflation decomposition analysis.

Decomposes observed inflation into demand and supply components using
the estimated Phillips curve, enabling policy-relevant diagnostics.

The key question this module answers:
    "Is above-target inflation driven by demand (tight labor market)
     or supply (import prices, energy costs)?"

This distinction matters for monetary policy:
    - Demand-driven inflation: Rate rises are effective
    - Supply-driven inflation: Rate rises are costly and less effective

Components
----------
The Phillips curve decomposes quarterly inflation as:

    π = quarterly(π_anchor) + γ_π·u_gap + ρ_π·Δ4ρm + ξ_π·GSCPI² + ε

Where:
    - quarterly(π_anchor): Baseline from expectations/target (neutral)
    - γ_π·u_gap: Demand component (unemployment gap)
    - ρ_π·Δ4ρm: Supply component - import prices
    - ξ_π·GSCPI²: Supply component - global supply chain pressure (COVID-era)
    - ε: Residual (unexplained)

Usage
-----
    from src.models.nairu.analysis import decompose_inflation, plot_inflation_decomposition

    # Get decomposition
    decomp = decompose_inflation(trace, obs, obs_index)

    # Plot stacked contributions
    plot_inflation_decomposition(decomp)

    # Get policy summary for recent period
    summary = inflation_policy_summary(decomp, periods=4)

"""

from dataclasses import dataclass

import arviz as az
import mgplot as mg
import numpy as np
import pandas as pd

from src.models.nairu.analysis.extraction import get_scalar_var, get_vector_var
from src.models.nairu.equations import REGIME_COVID_START, REGIME_GFC_START
from src.utilities.rate_conversion import annualize, quarterly

# LaTeX equation strings for each chart type
# Full Phillips curve: π_t = π^e_t + γ((U_t - U*_t)/U_t) + λΔρ^m_t + ξ·GSCPI² + ε_t
EQ_DEMAND_SUPPLY = (
    r"$\pi_t = \pi^e_t + \underbrace{\gamma\frac{U_t - U^*_t}{U_t}}_{\mathrm{orange}}"
    r" + \underbrace{\lambda\Delta\rho^m_t + \xi\cdot GSCPI^2}_{\mathrm{blue}}"
    r" + \varepsilon_t$"
)
EQ_PROPORTIONAL = (
    r"$\pi_t = \underbrace{\pi^e_t}_{\mathrm{grey}}"
    r" + \underbrace{\gamma\frac{U_t - U^*_t}{U_t}}_{\mathrm{orange}}"
    r" + \underbrace{\lambda\Delta\rho^m_t + \xi\cdot GSCPI^2}_{\mathrm{blue}}"
    r" + \varepsilon_t$"
)
EQ_UNSCALED = (
    r"$\pi_t = \underbrace{\pi^e_t}_{\mathrm{grey}}"
    r" + \underbrace{\gamma\frac{U_t - U^*_t}{U_t}}_{\mathrm{orange}}"
    r" + \underbrace{\lambda\Delta\rho^m_t + \xi\cdot GSCPI^2}_{\mathrm{blue}}"
    r" + \underbrace{\varepsilon_t}_{\mathrm{light\ blue}}$"
)


def add_equation_box(ax, equation: str, x: float = 0.5, y: float = 0.02) -> None:
    """Add a LaTeX equation in a text box to the axes.

    Args:
        ax: Matplotlib axes
        equation: LaTeX equation string
        x: x position in axes coordinates (0-1)
        y: y position in axes coordinates (0-1)

    """
    ax.text(
        x,
        y,
        equation,
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment="bottom",
        horizontalalignment="center",
        usetex=True,
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.9, "edgecolor": "grey"},
    )


def _plot_decomposition_bars(
    bar_data: pd.DataFrame,
    observed: pd.Series,
    colors: list[str],
    title: str,
    equation: str,
    observed_color: str = "indigo",
    observed_label: str = "Observed (quarterly annualised)",
    target_line: float | None = 2.5,
    lheader: str = "",
    lfooter: str = "Australia. Decomposition based on augmented Phillips curve.",
    rfooter: str = "NAIRU + Output Gap Model",
    eq_x: float = 0.4,
    eq_y: float = 0.75,
    show: bool = False,
) -> None:
    """Core plotting function for inflation decomposition charts.

    Args:
        bar_data: DataFrame with columns to stack as bars
        observed: Series with observed values for line overlay
        colors: List of colors for each bar column
        title: Chart title
        equation: LaTeX equation string for text box
        observed_color: Color for observed line
        observed_label: Label for observed line in legend
        target_line: Y-value for horizontal target line (None to skip)
        lheader: Left header text (e.g., acronym definitions)
        lfooter: Left footer text
        rfooter: Right footer text
        eq_x: X position for equation box (0-1)
        eq_y: Y position for equation box (0-1)
        show: If True, display interactively

    """
    ax = mg.bar_plot(bar_data, stacked=True, color=colors)

    observed_plot = observed.copy()
    observed_plot.name = observed_label
    mg.line_plot(observed_plot, ax=ax, color=observed_color, width=1.5, zorder=10)

    add_equation_box(ax, equation, x=eq_x, y=eq_y)

    finalise_kwargs = {
        "title": title,
        "ylabel": "% p.a.",
        "lheader": lheader,
        "rheader": rfooter,
        "lfooter": lfooter,
        "legend": {"loc": "best", "fontsize": "x-small"},
        "y0": True,
        "show": show,
    }
    if target_line is not None:
        finalise_kwargs["axhline"] = {
            "y": target_line, "color": "darkred", "linestyle": "--",
            "linewidth": 1, "label": f"{target_line}% Target"
        }

    mg.finalise_plot(ax, **finalise_kwargs)


@dataclass
class InflationDecomposition:
    """Container for inflation decomposition results.

    Attributes:
        observed: Observed quarterly inflation (%)
        anchor: Contribution from inflation anchor/target
        demand: Contribution from unemployment gap (demand pressure)
        supply_import: Contribution from import prices
        supply_gscpi: Contribution from GSCPI (COVID supply chain pressure)
        residual: Unexplained component
        fitted: Fitted values (sum of components excl. residual)
        index: Time period index

    All components are in percentage points (quarterly).

    Note:
        Oil and coal price effects were tested but found to be statistically
        indistinguishable from zero - their effects are already captured
        through the import price channel. GSCPI captures COVID-era supply
        chain disruptions.

    """

    observed: pd.Series
    anchor: pd.Series
    demand: pd.Series
    supply_import: pd.Series
    supply_gscpi: pd.Series
    residual: pd.Series
    fitted: pd.Series
    index: pd.PeriodIndex

    @property
    def supply_total(self) -> pd.Series:
        """Total supply contribution (import prices + GSCPI)."""
        return self.supply_import + self.supply_gscpi

    @property
    def above_target(self) -> pd.Series:
        """Inflation above target (quarterly equivalent of 2.5% annual)."""
        target_quarterly = quarterly(2.5)
        return self.observed - target_quarterly

    @property
    def demand_share(self) -> pd.Series:
        """Share of above-anchor inflation attributable to demand.

        Returns fraction in [0, 1] when demand and supply push same direction,
        can be outside this range when they offset.
        """
        above_anchor = self.observed - self.anchor
        # Avoid division by zero
        with np.errstate(divide="ignore", invalid="ignore"):
            share = self.demand / above_anchor
            share = share.replace([np.inf, -np.inf], np.nan)
        return share

    @property
    def supply_share(self) -> pd.Series:
        """Share of above-anchor inflation attributable to supply."""
        above_anchor = self.observed - self.anchor
        with np.errstate(divide="ignore", invalid="ignore"):
            share = self.supply_total / above_anchor
            share = share.replace([np.inf, -np.inf], np.nan)
        return share

    def to_dataframe(self) -> pd.DataFrame:
        """Convert to DataFrame with all components."""
        return pd.DataFrame(
            {
                "observed": self.observed,
                "anchor": self.anchor,
                "demand": self.demand,
                "supply_import": self.supply_import,
                "supply_gscpi": self.supply_gscpi,
                "supply_total": self.supply_total,
                "residual": self.residual,
                "fitted": self.fitted,
            },
            index=self.index,
        )


def decompose_inflation(
    trace: az.InferenceData,
    obs: dict[str, np.ndarray],
    obs_index: pd.PeriodIndex,
    use_median: bool = True,
) -> InflationDecomposition:
    """Decompose inflation into demand and supply components.

    Uses the estimated Phillips curve parameters to attribute observed
    inflation to its underlying drivers.

    Args:
        trace: PyMC inference data with posterior samples
        obs: Observation dictionary from model
        obs_index: Time period index
        use_median: If True, use posterior median for decomposition.
                   If False, returns decomposition for each posterior sample.

    Returns:
        InflationDecomposition with component time series

    Note:
        The decomposition uses posterior median parameters by default.
        For uncertainty quantification, set use_median=False to get
        the full posterior distribution of each component.

    """
    # Extract parameters (posterior median)
    # Regime-specific price Phillips curve slopes
    gamma_pi_pre_gfc = get_scalar_var("gamma_pi_pre_gfc", trace).median()
    gamma_pi_gfc = get_scalar_var("gamma_pi_gfc", trace).median()
    gamma_pi_covid = get_scalar_var("gamma_pi_covid", trace).median()

    rho_pi = get_scalar_var("rho_pi", trace).median()
    xi_gscpi = get_scalar_var("xi_gscpi", trace).median()

    # Build regime-specific gamma series
    gamma_pi = pd.Series(index=obs_index, dtype=float)
    gamma_pi.loc[obs_index < REGIME_GFC_START] = gamma_pi_pre_gfc
    gamma_pi.loc[(obs_index >= REGIME_GFC_START) & (obs_index < REGIME_COVID_START)] = gamma_pi_gfc
    gamma_pi.loc[obs_index >= REGIME_COVID_START] = gamma_pi_covid

    # Extract NAIRU (vector, use median across samples) as Series
    # Note: .values needed to extract numpy array from xarray DataArray
    nairu = pd.Series(
        get_vector_var("nairu", trace).median(axis=1).values, index=obs_index, name="nairu"
    )

    # Convert observed data to Series
    U = pd.Series(obs["U"], index=obs_index, name="U")
    pi_anchor = pd.Series(obs["π_anchor"], index=obs_index, name="pi_anchor")
    pi_observed = pd.Series(obs["π"], index=obs_index, name="observed")

    # Import prices (lagged) - may not be present
    delta_4_pm_lag1 = pd.Series(
        obs.get("Δ4ρm_1", np.zeros(len(obs_index))), index=obs_index, name="delta_4_pm"
    )

    # GSCPI (lagged) - may not be present
    gscpi = pd.Series(
        obs.get("ξ_2", np.zeros(len(obs_index))), index=obs_index, name="gscpi"
    )

    # Compute components (all operations on Series)
    # 1. Anchor contribution (quarterly, using compound conversion)
    anchor = quarterly(pi_anchor)
    anchor.name = "anchor"

    # 2. Demand contribution (unemployment gap)
    u_gap = (U - nairu) / U
    demand = gamma_pi * u_gap
    demand.name = "demand"

    # 3. Supply - import prices (includes oil effect via import price channel)
    supply_import = rho_pi * delta_4_pm_lag1
    supply_import.name = "supply_import"

    # 4. Supply - GSCPI (non-linear: squared with sign preservation)
    supply_gscpi = xi_gscpi * gscpi ** 2 * np.sign(gscpi)
    supply_gscpi.name = "supply_gscpi"

    # Fitted values (excluding residual)
    fitted = anchor + demand + supply_import + supply_gscpi
    fitted.name = "fitted"

    # Residual
    residual = pi_observed - fitted
    residual.name = "residual"

    return InflationDecomposition(
        observed=pi_observed,
        anchor=anchor,
        demand=demand,
        supply_import=supply_import,
        supply_gscpi=supply_gscpi,
        residual=residual,
        fitted=fitted,
        index=obs_index,
    )


def plot_demand_contribution(
    decomp: InflationDecomposition,
    start: str | None = None,
    end: str | None = None,
    annual: bool = True,
    rfooter: str = "NAIRU + Output Gap Model",
    show: bool = False,
) -> None:
    """Plot demand contribution to inflation (unemployment gap effect).

    Args:
        decomp: InflationDecomposition from decompose_inflation()
        start: Start period (e.g., "2015Q1")
        end: End period
        annual: If True, annualize quarterly rates (multiply by 4)
        rfooter: Right footer text
        show: If True, display interactively

    """
    df = decomp.to_dataframe()

    if start:
        df = df[df.index >= pd.Period(start)]
    if end:
        df = df[df.index <= pd.Period(end)]

    unit = "% p.a." if annual else "% p.q."

    demand = annualize(df["demand"]) if annual else df["demand"]
    demand.name = "Demand Contribution"

    mg.line_plot_finalise(
        demand,
        color="darkred",
        width=1.5,
        title=f"Price Inflation - Demand Contribution ({unit})",
        ylabel=unit,
        rfooter=rfooter,
        y0=True,
        show=show,
    )


def plot_supply_contribution(
    decomp: InflationDecomposition,
    start: str | None = None,
    end: str | None = None,
    annual: bool = True,
    rfooter: str = "NAIRU + Output Gap Model",
    show: bool = False,
) -> None:
    """Plot supply contributions to inflation (import prices + GSCPI).

    Args:
        decomp: InflationDecomposition from decompose_inflation()
        start: Start period (e.g., "2015Q1")
        end: End period
        annual: If True, annualize quarterly rates (multiply by 4)
        rfooter: Right footer text
        show: If True, display interactively

    """
    df = decomp.to_dataframe()

    if start:
        df = df[df.index >= pd.Period(start)]
    if end:
        df = df[df.index <= pd.Period(end)]

    unit = "% p.a." if annual else "% p.q."

    if annual:
        supply_import = annualize(df["supply_import"])
        supply_gscpi = annualize(df["supply_gscpi"])
        supply_total = annualize(df["supply_total"])
    else:
        supply_import = df["supply_import"]
        supply_gscpi = df["supply_gscpi"]
        supply_total = df["supply_total"]
    supply_import.name = "Import Prices"
    supply_gscpi.name = "GSCPI (Supply Chain)"
    supply_total.name = "Total Supply"

    ax = mg.line_plot(supply_import, color="blue", width=1.5)
    mg.line_plot(supply_gscpi, ax=ax, color="purple", width=1.5)
    mg.line_plot(supply_total, ax=ax, color="black", width=2)

    mg.finalise_plot(
        ax,
        title=f"Price Inflation - Supply Contributions ({unit})",
        ylabel=unit,
        rfooter=rfooter,
        legend={"loc": "best", "fontsize": "x-small"},
        y0=True,
        show=show,
    )


def plot_inflation_drivers(
    decomp: InflationDecomposition,
    start: str | None = None,
    end: str | None = None,
    rfooter: str = "NAIRU + Output Gap Model",
    show: bool = False,
) -> None:
    """Plot price inflation decomposition with stacked bars for demand vs supply."""
    df = decomp.to_dataframe()
    if start:
        df = df[df.index >= pd.Period(start)]
    if end:
        df = df[df.index <= pd.Period(end)]
    df = annualize(df)

    bar_data = pd.DataFrame(
        {"Demand": df["demand"], "Supply": df["supply_total"]},
        index=df.index,
    )

    _plot_decomposition_bars(
        bar_data=bar_data,
        observed=df["observed"],
        colors=["orange", "darkblue"],
        title="Price Inflation Decomposition: Demand vs Supply",
        equation=EQ_DEMAND_SUPPLY,
        observed_label="Observed Inflation (quarterly annualised)",
        rfooter=rfooter,
        eq_x=0.5,
        eq_y=0.65,
        show=show,
    )


def plot_inflation_drivers_proportional(
    decomp: InflationDecomposition,
    start: str | None = "1998Q1",
    end: str | None = None,
    rfooter: str = "NAIRU + Output Gap Model",
    show: bool = False,
) -> None:
    """Plot inflation with proportionalized demand vs supply bars.

    Shows annualized quarterly inflation decomposed into proportional
    contributions from demand and supply:
    - Red bars: Demand share (unemployment gap)
    - Blue bars: Supply share (import prices + coal)

    Bars are scaled so that when both are positive, they sum to observed
    inflation. Proportions calculated as:
        demand_share = demand / (|demand| + |supply|)
        supply_share = supply / (|demand| + |supply|)

    Caveats:
    - Anchor and residual are absorbed into the proportionalization
    - Shows relative attribution, not absolute model estimates
    - When demand and supply have opposite signs, bars show offsetting
      contributions and won't sum to observed

    Args:
        decomp: InflationDecomposition from decompose_inflation()
        start: Start period (e.g., "2015Q1")
        end: End period
        rfooter: Right footer text
        show: If True, display interactively

    """
    df = decomp.to_dataframe()

    if start:
        df = df[df.index >= pd.Period(start)]
    if end:
        df = df[df.index <= pd.Period(end)]

    # Annualize (compound conversion)
    df = annualize(df)

    # Calculate scaled contributions that sum to (observed - anchor)
    # This shows deviation from baseline, removing anchor's distortion
    demand = df["demand"]
    supply = df["supply_total"]
    observed = df["observed"]
    anchor = df["anchor"]

    # Deviation from anchor = what demand + supply + residual explain
    deviation = observed - anchor
    total = demand + supply

    # Scale so demand_scaled + supply_scaled = deviation
    # Use where to avoid division issues when total is near zero
    scale = (deviation / total).where(total.abs() > 0.01, 0)

    # Only valid when scale is not extreme (contributions nearly cancel)
    max_scale = 5.0
    valid = scale.abs() <= max_scale

    demand_prop = (demand * scale).where(valid, 0)
    supply_prop = (supply * scale).where(valid, 0)

    # Create DataFrame for stacked bar plot
    # Anchor as baseline, then demand and supply stack on top
    bar_data = pd.DataFrame(
        {
            "Inflation expectations / inflation target": anchor,
            "Demand": demand_prop,
            "Supply": supply_prop,
        },
        index=df.index,
    )

    # Plot stacked bars (mgplot requires string colors)
    ax = mg.bar_plot(
        bar_data,
        stacked=True,
        color=["#cccccc", "orange", "darkblue"],  # gray anchor, orange demand, darkblue supply
    )

    # Add observed inflation line on top
    observed_line = observed.copy()
    observed_line.name = "Observed Inflation (quarterly annualised)"
    mg.line_plot(observed_line, ax=ax, color="indigo", width=1.5, zorder=10)

    # Add equation text box
    add_equation_box(ax, EQ_PROPORTIONAL)

    mg.finalise_plot(
        ax,
        title="Inflation: Proportional Demand vs Supply Attribution",
        ylabel="% p.a.",
        rheader=rfooter,
        lfooter="Australia. Decomposition based on augmented Phillips curve.",
        legend={"loc": "best", "fontsize": "x-small"},
        axhline={"y": 2.5, "color": "darkred", "linestyle": "--", "linewidth": 1, "label": "2.5% Target"},
        y0=True,
        show=show,
    )


def plot_inflation_drivers_unscaled(
    decomp: InflationDecomposition,
    start: str | None = None,
    end: str | None = None,
    rfooter: str = "NAIRU + Output Gap Model",
    show: bool = False,
) -> None:
    """Plot price inflation with unscaled components including residual noise."""
    df = decomp.to_dataframe()
    if start:
        df = df[df.index >= pd.Period(start)]
    if end:
        df = df[df.index <= pd.Period(end)]
    df = annualize(df)

    bar_data = pd.DataFrame(
        {
            "Inflation expectations / inflation target": df["anchor"],
            "Demand": df["demand"],
            "Supply": df["supply_total"],
            "Noise": df["residual"],
        },
        index=df.index,
    )

    _plot_decomposition_bars(
        bar_data=bar_data,
        observed=df["observed"],
        colors=["#cccccc", "orange", "darkblue", "lightblue"],
        title="Price Inflation Decomposition: Components (Unscaled)",
        equation=EQ_UNSCALED,
        observed_label="Observed Inflation (quarterly annualised)",
        eq_x=0.5,
        eq_y=0.02,
        rfooter=rfooter,
        show=show,
    )


def plot_inflation_decomposition(
    decomp: InflationDecomposition,
    start: str | None = None,
    end: str | None = None,
    annual: bool = True,
    rfooter: str = "NAIRU + Output Gap Model",
    show: bool = False,
) -> None:
    """Generate inflation decomposition charts.

    Creates four chart files:
    1. Demand Contribution (unemployment gap effect)
    2. Supply Contributions (import prices + coal)
    3. Inflation drivers (stacked bars showing demand + supply)
    4. Unscaled components with residual noise

    Args:
        decomp: InflationDecomposition from decompose_inflation()
        start: Start period (e.g., "2015Q1")
        end: End period
        annual: If True, annualize quarterly rates (multiply by 4)
        rfooter: Right footer text
        show: If True, display interactively

    """
    plot_demand_contribution(decomp, start, end, annual, rfooter, show)
    plot_supply_contribution(decomp, start, end, annual, rfooter, show)
    plot_inflation_drivers(decomp, start, end, rfooter, show)
    plot_inflation_drivers_unscaled(decomp, start, end, rfooter, show)


# --- Wage Inflation Decomposition ---

EQ_WAGE_DEMAND_SUPPLY = (
    r"$\Delta ulc_t = \alpha"
    r" + \underbrace{\gamma\frac{U_t - U^*_t}{U_t} + \lambda\frac{\Delta U_{t-1}}{U_t}}_{\mathrm{orange}}"
    r" + \underbrace{\phi\Delta_4 dfd_t}_{\mathrm{blue}}"
    r" + \theta\pi^e_t + \varepsilon_t$"
)
EQ_WAGE_UNSCALED = (
    r"$\Delta ulc_t = \underbrace{\alpha + \theta\pi^e_t}_{\mathrm{grey}}"
    r" + \underbrace{\gamma\frac{U_t - U^*_t}{U_t} + \lambda\frac{\Delta U_{t-1}}{U_t}}_{\mathrm{orange}}"
    r" + \underbrace{\phi\Delta_4 dfd_t}_{\mathrm{blue}}"
    r" + \underbrace{\varepsilon_t}_{\mathrm{light\ blue}}$"
)


@dataclass
class WageInflationDecomposition:
    """Container for wage inflation (ULC growth) decomposition results.

    Note: Persistence term (rho_wg) was tested but posterior not different from zero;
    removed for parsimony.
    """

    observed: pd.Series
    anchor: pd.Series          # α + θ×π_anchor
    demand: pd.Series          # γ×u_gap + λ×ΔU/U
    price_passthrough: pd.Series  # φ×Δ4dfd
    residual: pd.Series
    fitted: pd.Series
    index: pd.PeriodIndex

    def to_dataframe(self) -> pd.DataFrame:
        """Convert to DataFrame with all components."""
        return pd.DataFrame(
            {
                "observed": self.observed,
                "anchor": self.anchor,
                "demand": self.demand,
                "price_passthrough": self.price_passthrough,
                "residual": self.residual,
                "fitted": self.fitted,
            },
            index=self.index,
        )


def decompose_wage_inflation(
    trace: az.InferenceData,
    obs: dict[str, np.ndarray],
    obs_index: pd.PeriodIndex,
) -> WageInflationDecomposition:
    """Decompose wage inflation (ULC growth) into demand and price components."""
    # Extract parameters (posterior median)
    alpha_wg = get_scalar_var("alpha_wg", trace).median()
    lambda_wg = get_scalar_var("lambda_wg", trace).median()
    phi_wg = get_scalar_var("phi_wg", trace).median()
    theta_wg = get_scalar_var("theta_wg", trace).median()

    # Regime-specific wage Phillips curve slopes
    gamma_wg_pre_gfc = get_scalar_var("gamma_wg_pre_gfc", trace).median()
    gamma_wg_gfc = get_scalar_var("gamma_wg_gfc", trace).median()
    gamma_wg_covid = get_scalar_var("gamma_wg_covid", trace).median()
    gamma_wg = pd.Series(index=obs_index, dtype=float)
    gamma_wg.loc[obs_index < REGIME_GFC_START] = gamma_wg_pre_gfc
    gamma_wg.loc[(obs_index >= REGIME_GFC_START) & (obs_index < REGIME_COVID_START)] = gamma_wg_gfc
    gamma_wg.loc[obs_index >= REGIME_COVID_START] = gamma_wg_covid

    nairu = pd.Series(
        get_vector_var("nairu", trace).median(axis=1).values, index=obs_index
    )

    # Convert observed data to Series
    U = pd.Series(obs["U"], index=obs_index)
    pi_anchor = pd.Series(obs["π_anchor"], index=obs_index)
    ulc_observed = pd.Series(obs["Δulc"], index=obs_index)
    delta_u_over_u = pd.Series(obs["ΔU_1_over_U"], index=obs_index)
    dfd_growth = pd.Series(obs["Δ4dfd"], index=obs_index)

    # Compute components
    anchor = alpha_wg + theta_wg * quarterly(pi_anchor)
    u_gap = (U - nairu) / U
    demand = gamma_wg * u_gap + lambda_wg * delta_u_over_u
    price_passthrough = phi_wg * dfd_growth
    fitted = anchor + demand + price_passthrough
    residual = ulc_observed - fitted

    return WageInflationDecomposition(
        observed=ulc_observed,
        anchor=anchor,
        demand=demand,
        price_passthrough=price_passthrough,
        residual=residual,
        fitted=fitted,
        index=obs_index,
    )


def plot_wage_drivers(
    decomp: WageInflationDecomposition,
    start: str | None = None,
    end: str | None = None,
    rfooter: str = "NAIRU + Output Gap Model",
    show: bool = False,
) -> None:
    """Plot wage inflation decomposition with stacked bars for demand vs price pressure."""
    df = decomp.to_dataframe()
    if start:
        df = df[df.index >= pd.Period(start)]
    if end:
        df = df[df.index <= pd.Period(end)]
    df = annualize(df)

    bar_data = pd.DataFrame(
        {"Demand (labor market)": df["demand"], "Price pass-through": df["price_passthrough"]},
        index=df.index,
    )

    _plot_decomposition_bars(
        bar_data=bar_data,
        observed=df["observed"],
        colors=["orange", "darkblue"],
        title="Wage-ULC Inflation Decomposition: Demand vs Price Pressure",
        equation=EQ_WAGE_DEMAND_SUPPLY,
        observed_color="darkorange",
        observed_label="Observed ULC Growth (quarterly annualised)",
        target_line=None,
        lheader="ULC = Unit Labour Costs.",
        lfooter="Australia. Decomposition based on augmented wage Phillips curve.",
        rfooter=rfooter,
        eq_x=0.5,
        eq_y=0.75,
        show=show,
    )


def plot_wage_drivers_unscaled(
    decomp: WageInflationDecomposition,
    start: str | None = None,
    end: str | None = None,
    rfooter: str = "NAIRU + Output Gap Model",
    show: bool = False,
) -> None:
    """Plot wage inflation with unscaled components including residual."""
    df = decomp.to_dataframe()
    if start:
        df = df[df.index >= pd.Period(start)]
    if end:
        df = df[df.index <= pd.Period(end)]
    df = annualize(df)

    bar_data = pd.DataFrame(
        {
            "Anchor (intercept + expectations)": df["anchor"],
            "Demand (labor market)": df["demand"],
            "Price pass-through": df["price_passthrough"],
            "Noise": df["residual"],
        },
        index=df.index,
    )

    _plot_decomposition_bars(
        bar_data=bar_data,
        observed=df["observed"],
        colors=["#cccccc", "orange", "darkblue", "lightblue"],
        title="Wage-ULC Inflation Decomposition: Components (Unscaled)",
        equation=EQ_WAGE_UNSCALED,
        observed_color="darkorange",
        observed_label="Observed ULC Growth (quarterly annualised)",
        target_line=None,
        lheader="ULC = Unit Labour Costs.",
        lfooter="Australia. Decomposition based on augmented wage Phillips curve.",
        rfooter=rfooter,
        eq_x=0.5,
        eq_y=0.02,
        show=show,
    )


def plot_wage_decomposition(
    decomp: WageInflationDecomposition,
    start: str | None = None,
    end: str | None = None,
    rfooter: str = "NAIRU + Output Gap Model",
    show: bool = False,
) -> None:
    """Generate wage-ULC inflation decomposition charts."""
    plot_wage_drivers(decomp, start, end, rfooter, show)
    plot_wage_drivers_unscaled(decomp, start, end, rfooter, show)


# --- Hourly COE Wage Inflation Decomposition ---

EQ_HCOE_DEMAND_SUPPLY = (
    r"$\Delta hcoe_t = \alpha"
    r" + \underbrace{\gamma\frac{U_t - U^*_t}{U_t} + \lambda\frac{\Delta U_{t-1}}{U_t}}_{\mathrm{orange}}"
    r" + \underbrace{\phi\Delta_4 dfd_t}_{\mathrm{blue}}"
    r" + \underbrace{\psi \cdot mfp_t}_{\mathrm{green}}"
    r" + \theta\pi^e_t + \varepsilon_t$"
)
EQ_HCOE_UNSCALED = (
    r"$\Delta hcoe_t = \underbrace{\alpha + \theta\pi^e_t}_{\mathrm{grey}}"
    r" + \underbrace{\gamma\frac{U_t - U^*_t}{U_t} + \lambda\frac{\Delta U_{t-1}}{U_t}}_{\mathrm{orange}}"
    r" + \underbrace{\phi\Delta_4 dfd_t}_{\mathrm{blue}}"
    r" + \underbrace{\psi \cdot mfp_t}_{\mathrm{green}}"
    r" + \underbrace{\varepsilon_t}_{\mathrm{light\ blue}}$"
)


@dataclass
class HCOEInflationDecomposition:
    """Container for hourly COE growth decomposition results.

    Note: Persistence term (rho_hcoe) was tested but posterior not different from zero;
    removed for parsimony.
    """

    observed: pd.Series
    anchor: pd.Series          # α + θ×π_anchor
    demand: pd.Series          # γ×u_gap + λ×ΔU/U
    price_passthrough: pd.Series  # φ×Δ4dfd
    productivity: pd.Series    # ψ×mfp_growth
    residual: pd.Series
    fitted: pd.Series
    index: pd.PeriodIndex

    def to_dataframe(self) -> pd.DataFrame:
        """Convert to DataFrame with all components."""
        return pd.DataFrame(
            {
                "observed": self.observed,
                "anchor": self.anchor,
                "demand": self.demand,
                "price_passthrough": self.price_passthrough,
                "productivity": self.productivity,
                "residual": self.residual,
                "fitted": self.fitted,
            },
            index=self.index,
        )


def decompose_hcoe_inflation(
    trace: az.InferenceData,
    obs: dict[str, np.ndarray],
    obs_index: pd.PeriodIndex,
) -> HCOEInflationDecomposition:
    """Decompose hourly COE growth into demand, price, and productivity components."""
    # Extract parameters (posterior median)
    alpha_hcoe = get_scalar_var("alpha_hcoe", trace).median()
    lambda_hcoe = get_scalar_var("lambda_hcoe", trace).median()
    phi_hcoe = get_scalar_var("phi_hcoe", trace).median()
    theta_hcoe = get_scalar_var("theta_hcoe", trace).median()
    psi_hcoe = get_scalar_var("psi_hcoe", trace).median()

    # Regime-specific hourly COE Phillips curve slopes
    gamma_hcoe_pre_gfc = get_scalar_var("gamma_hcoe_pre_gfc", trace).median()
    gamma_hcoe_gfc = get_scalar_var("gamma_hcoe_gfc", trace).median()
    gamma_hcoe_covid = get_scalar_var("gamma_hcoe_covid", trace).median()
    gamma_hcoe = pd.Series(index=obs_index, dtype=float)
    gamma_hcoe.loc[obs_index < REGIME_GFC_START] = gamma_hcoe_pre_gfc
    gamma_hcoe.loc[(obs_index >= REGIME_GFC_START) & (obs_index < REGIME_COVID_START)] = gamma_hcoe_gfc
    gamma_hcoe.loc[obs_index >= REGIME_COVID_START] = gamma_hcoe_covid

    nairu = pd.Series(
        get_vector_var("nairu", trace).median(axis=1).values, index=obs_index
    )

    # Convert observed data to Series
    U = pd.Series(obs["U"], index=obs_index)
    pi_anchor = pd.Series(obs["π_anchor"], index=obs_index)
    hcoe_observed = pd.Series(obs["Δhcoe"], index=obs_index)
    delta_u_over_u = pd.Series(obs["ΔU_1_over_U"], index=obs_index)
    dfd_growth = pd.Series(obs["Δ4dfd"], index=obs_index)
    mfp_growth = pd.Series(obs["mfp_growth"], index=obs_index)

    # Compute components
    anchor = alpha_hcoe + theta_hcoe * quarterly(pi_anchor)
    u_gap = (U - nairu) / U
    demand = gamma_hcoe * u_gap + lambda_hcoe * delta_u_over_u
    price_passthrough = phi_hcoe * dfd_growth
    productivity = psi_hcoe * mfp_growth
    fitted = anchor + demand + price_passthrough + productivity
    residual = hcoe_observed - fitted

    return HCOEInflationDecomposition(
        observed=hcoe_observed,
        anchor=anchor,
        demand=demand,
        price_passthrough=price_passthrough,
        productivity=productivity,
        residual=residual,
        fitted=fitted,
        index=obs_index,
    )


def plot_hcoe_drivers(
    decomp: HCOEInflationDecomposition,
    start: str | None = None,
    end: str | None = None,
    rfooter: str = "NAIRU + Output Gap Model",
    show: bool = False,
) -> None:
    """Plot hourly COE decomposition with stacked bars for demand vs price pressure."""
    df = decomp.to_dataframe()
    if start:
        df = df[df.index >= pd.Period(start)]
    if end:
        df = df[df.index <= pd.Period(end)]
    df = annualize(df)

    bar_data = pd.DataFrame(
        {
            "Demand (labor market)": df["demand"],
            "Price pass-through": df["price_passthrough"],
            "Productivity": df["productivity"],
        },
        index=df.index,
    )

    _plot_decomposition_bars(
        bar_data=bar_data,
        observed=df["observed"],
        colors=["orange", "darkblue", "limegreen"],
        title="Wage-HCOE Inflation Decomposition: Demand, Price & Productivity",
        equation=EQ_HCOE_DEMAND_SUPPLY,
        observed_color="darkorange",
        observed_label="Observed Hourly COE Growth (quarterly annualised)",
        target_line=None,
        lheader="HCOE = Hourly Compensation of Employees.",
        lfooter="Australia. Decomposition based on hourly compensation Phillips curve.",
        rfooter=rfooter,
        eq_x=0.5,
        eq_y=0.75,
        show=show,
    )


def plot_hcoe_drivers_unscaled(
    decomp: HCOEInflationDecomposition,
    start: str | None = None,
    end: str | None = None,
    rfooter: str = "NAIRU + Output Gap Model",
    show: bool = False,
) -> None:
    """Plot hourly COE with unscaled components including residual."""
    df = decomp.to_dataframe()
    if start:
        df = df[df.index >= pd.Period(start)]
    if end:
        df = df[df.index <= pd.Period(end)]
    df = annualize(df)

    bar_data = pd.DataFrame(
        {
            "Anchor (intercept + expectations)": df["anchor"],
            "Demand (labor market)": df["demand"],
            "Price pass-through": df["price_passthrough"],
            "Productivity": df["productivity"],
            "Noise": df["residual"],
        },
        index=df.index,
    )

    _plot_decomposition_bars(
        bar_data=bar_data,
        observed=df["observed"],
        colors=["#cccccc", "orange", "darkblue", "limegreen", "lightblue"],
        title="Wage-HCOE Inflation Decomposition: Components (Unscaled)",
        equation=EQ_HCOE_UNSCALED,
        observed_color="darkorange",
        observed_label="Observed Hourly COE Growth (quarterly annualised)",
        target_line=None,
        lheader="HCOE = Hourly Compensation of Employees.",
        lfooter="Australia. Decomposition based on hourly compensation Phillips curve.",
        rfooter=rfooter,
        eq_x=0.5,
        eq_y=0.02,
        show=show,
    )


def plot_hcoe_decomposition(
    decomp: HCOEInflationDecomposition,
    start: str | None = None,
    end: str | None = None,
    rfooter: str = "NAIRU + Output Gap Model",
    show: bool = False,
) -> None:
    """Generate hourly COE wage inflation decomposition charts."""
    plot_hcoe_drivers(decomp, start, end, rfooter, show)
    plot_hcoe_drivers_unscaled(decomp, start, end, rfooter, show)


def inflation_policy_summary(
    decomp: InflationDecomposition,
    periods: int = 4,
    annual: bool = True,
) -> pd.DataFrame:
    """Generate policy-relevant summary of recent inflation drivers.

    Answers the key question: Is current above-target inflation
    demand-driven or supply-driven?

    Args:
        decomp: InflationDecomposition from decompose_inflation()
        periods: Number of recent periods to summarize
        annual: If True, report annualized rates

    Returns:
        DataFrame with summary statistics for recent period

    """
    df = decomp.to_dataframe().tail(periods)
    unit = "% p.a." if annual else "% p.q."

    # Target (annual or quarterly)
    target = 2.5 if annual else quarterly(2.5)

    # Compute averages (annualize if needed)
    avg = annualize(df.mean()) if annual else df.mean()

    # Above target
    above_target = avg["observed"] - target

    # Demand vs supply attribution
    demand_contrib = avg["demand"]
    supply_contrib = avg["supply_import"] + avg["supply_gscpi"]

    # Attribution shares (of deviation from anchor)
    deviation_from_anchor = avg["observed"] - avg["anchor"]
    if abs(deviation_from_anchor) > 0.01:
        demand_share = demand_contrib / deviation_from_anchor * 100
        supply_share = supply_contrib / deviation_from_anchor * 100
    else:
        demand_share = supply_share = np.nan

    # Build summary
    summary = pd.DataFrame(
        {
            "Value": [
                f"{avg['observed']:.2f}",
                f"{target:.2f}",
                f"{above_target:+.2f}",
                f"{avg['anchor']:.2f}",
                f"{demand_contrib:+.2f}",
                f"{avg['supply_import']:+.2f}",
                f"{avg['supply_gscpi']:+.2f}",
                f"{supply_contrib:+.2f}",
                f"{avg['residual']:+.2f}",
                f"{demand_share:.0f}%" if not np.isnan(demand_share) else "N/A",
                f"{supply_share:.0f}%" if not np.isnan(supply_share) else "N/A",
            ],
            "Interpretation": [
                f"Average inflation over last {periods} quarters",
                "RBA target",
                "Positive = above target" if above_target > 0 else "Negative = below target",
                "Expectations/target baseline",
                "Tight labor market" if demand_contrib > 0 else "Slack in labor market",
                "Import price pressure" if avg["supply_import"] > 0 else "Import price relief",
                "Supply chain pressure" if avg["supply_gscpi"] > 0 else "Supply chains easing",
                "Total supply-side effect",
                "Unexplained",
                "Share of above-anchor inflation from demand",
                "Share of above-anchor inflation from supply",
            ],
        },
        index=[
            f"Observed Inflation ({unit})",
            f"Target ({unit})",
            f"Above Target ({unit})",
            f"Anchor Contribution ({unit})",
            f"Demand Contribution ({unit})",
            f"Supply: Import Prices ({unit})",
            f"Supply: GSCPI ({unit})",
            f"Supply: Total ({unit})",
            f"Residual ({unit})",
            "Demand Share (%)",
            "Supply Share (%)",
        ],
    )

    return summary


def get_policy_diagnosis(
    decomp: InflationDecomposition,
    periods: int = 4,
) -> str:
    """Get a concise policy diagnosis for recent inflation.

    Args:
        decomp: InflationDecomposition from decompose_inflation()
        periods: Number of recent periods to analyze

    Returns:
        String with policy-relevant diagnosis

    """
    df = decomp.to_dataframe().tail(periods)

    # Annualize (compound conversion)
    avg = annualize(df.mean())
    target = 2.5

    above_target = avg["observed"] - target
    demand = avg["demand"]
    supply = avg["supply_import"] + avg["supply_gscpi"]

    # Determine regime
    if abs(above_target) < 0.25:
        regime = "AT TARGET"
        detail = "Inflation is close to the 2.5% target."
    elif above_target > 0:
        regime = "ABOVE TARGET"
        if demand > 0 and supply > 0:
            if abs(demand) > abs(supply):
                detail = (
                    f"Primarily DEMAND-driven (+{demand:.1f}pp from tight labor market). "
                    f"Supply adds +{supply:.1f}pp. Rate rises likely effective."
                )
            else:
                detail = (
                    f"Primarily SUPPLY-driven (+{supply:.1f}pp from import/energy prices). "
                    f"Demand adds +{demand:.1f}pp. Rate rises less effective, costly."
                )
        elif demand > 0:
            detail = (
                f"DEMAND-driven (+{demand:.1f}pp from tight labor market). "
                f"Supply actually subtracting {supply:.1f}pp. Rate rises appropriate."
            )
        else:
            detail = (
                f"SUPPLY-driven (+{supply:.1f}pp). Demand subtracting {demand:.1f}pp. "
                "Rate rises may be counterproductive."
            )
    else:
        regime = "BELOW TARGET"
        detail = (
            f"Inflation {above_target:.1f}pp below target. "
            f"Demand contribution: {demand:+.1f}pp, Supply: {supply:+.1f}pp."
        )

    return f"{regime}: {detail}"
