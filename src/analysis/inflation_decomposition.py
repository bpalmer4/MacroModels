"""Inflation decomposition analysis.

Decomposes observed inflation into demand and supply components using
the estimated Phillips curve, enabling policy-relevant diagnostics.

The key question this module answers:
    "Is above-target inflation driven by demand (tight labor market)
     or supply (import prices, disruptions)?"

This distinction matters for monetary policy:
    - Demand-driven inflation: Rate rises are effective
    - Supply-driven inflation: Rate rises are costly and less effective

Components
----------
The Phillips curve decomposes quarterly inflation as:

    π = quarterly(π_anchor) + γ_π·u_gap + ρ_π·Δ4ρm + ξ_π·GSCPI + ε

Where:
    - quarterly(π_anchor): Baseline from expectations/target (neutral)
    - γ_π·u_gap: Demand component (unemployment gap)
    - ρ_π·Δ4ρm: Supply component - import prices
    - ξ_π·GSCPI: Supply component - global supply chain
    - ε: Residual (unexplained)

Usage
-----
    from src.analysis import decompose_inflation, plot_inflation_decomposition

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

from src.analysis.extraction import get_scalar_var, get_vector_var
from src.analysis.rate_conversion import annualize, quarterly

# LaTeX equation strings for each chart type
# Full Phillips curve: π_t = π^e_t + γ(U_t - U*_t) + λΔρ^m_t + φξ_t + ε_t
EQ_DEMAND_SUPPLY = (
    r"$\pi_t = \pi^e_t + \underbrace{\gamma(U_t - U^*_t)}_{\mathrm{orange}}"
    r" + \underbrace{\lambda\Delta\rho^m_t + \phi\xi_t}_{\mathrm{blue}} + \varepsilon_t$"
)
EQ_PROPORTIONAL = (
    r"$\pi_t = \underbrace{\pi^e_t}_{\mathrm{grey}}"
    r" + \underbrace{\gamma(U_t - U^*_t)}_{\mathrm{orange}}"
    r" + \underbrace{\lambda\Delta\rho^m_t + \phi\xi_t}_{\mathrm{blue}} + \varepsilon_t$"
)
EQ_UNSCALED = (
    r"$\pi_t = \underbrace{\pi^e_t}_{\mathrm{grey}}"
    r" + \underbrace{\gamma(U_t - U^*_t)}_{\mathrm{orange}}"
    r" + \underbrace{\lambda\Delta\rho^m_t + \phi\xi_t}_{\mathrm{blue}}"
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


@dataclass
class InflationDecomposition:
    """Container for inflation decomposition results.

    Attributes:
        observed: Observed quarterly inflation (%)
        anchor: Contribution from inflation anchor/target
        demand: Contribution from unemployment gap (demand pressure)
        supply_import: Contribution from import prices
        supply_gscpi: Contribution from global supply chain pressures
        residual: Unexplained component
        fitted: Fitted values (sum of components excl. residual)
        index: Time period index

    All components are in percentage points (quarterly).

    Note:
        A direct oil price component was tested but removed as oil's effect
        on inflation is already captured through the import price channel.
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
    gamma_pi = get_scalar_var("gamma_pi", trace).median()
    rho_pi = get_scalar_var("rho_pi", trace).median()
    xi_2sq_pi = get_scalar_var("xi_2sq_pi", trace).median()

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

    # GSCPI - may not be present
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

    # 4. Supply - GSCPI (with quadratic/signed transformation)
    supply_gscpi = xi_2sq_pi * (gscpi**2) * gscpi.apply(np.sign)
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
        title=f"Inflation - Demand Contribution ({unit})",
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
    supply_gscpi.name = "GSCPI"
    supply_total.name = "Total Supply"

    ax = mg.line_plot(supply_import, color="blue", width=1.5)
    mg.line_plot(supply_gscpi, ax=ax, color="green", width=1.5)
    mg.line_plot(supply_total, ax=ax, color="black", width=2)

    mg.finalise_plot(
        ax,
        title=f"Inflation - Supply Contributions ({unit})",
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
    """Plot inflation decomposition with stacked bars for demand vs supply.

    Shows annualized quarterly inflation decomposed into:
    - Red bars: Demand contribution (unemployment gap)
    - Blue bars: Supply contribution (import prices + GSCPI)

    Positive bars push inflation up, negative bars push it down.
    Bars stack from zero.

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

    # Create DataFrame for stacked bar plot
    # Demand (red/warm) and Supply (blue/cool)
    bar_data = pd.DataFrame(
        {
            "Demand": df["demand"],
            "Supply": df["supply_total"],
        },
        index=df.index,
    )

    # Plot stacked bars (mgplot requires string colors)
    ax = mg.bar_plot(
        bar_data,
        stacked=True,
        color=["orange", "darkblue"],  # orange for demand, darkblue for supply
    )

    # Add observed inflation line on top
    observed = df["observed"].copy()
    observed.name = "Observed Inflation (quarterly annualised)"
    mg.line_plot(observed, ax=ax, color="indigo", width=1.5, zorder=10)

    # Add equation text box
    add_equation_box(ax, EQ_DEMAND_SUPPLY)

    mg.finalise_plot(
        ax,
        title="Inflation Decomposition: Demand vs Supply",
        ylabel="% p.a.",
        rheader=rfooter,
        lfooter="Australia. Decomposition based on augmented Phillips curve.",
        legend={"loc": "best", "fontsize": "x-small"},
        axhline={"y": 2.5, "color": "darkred", "linestyle": "--", "linewidth": 1, "label": "2.5% Target"},
        y0=True,
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
    - Blue bars: Supply share (import prices + GSCPI)

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
    start: str | None = "1998Q1",
    end: str | None = None,
    rfooter: str = "NAIRU + Output Gap Model",
    show: bool = False,
) -> None:
    """Plot inflation with unscaled components including residual noise.

    Shows annualized quarterly inflation decomposed into its raw
    model components (no scaling/proportionalizing):
    - Grey bars: Anchor (inflation expectations/target)
    - Orange bars: Demand (unemployment gap)
    - Dark blue bars: Supply (import prices + GSCPI)
    - Light blue bars: Residual/noise

    Bars sum exactly to observed inflation.

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

    # Create DataFrame for stacked bar plot
    # Anchor + Demand + Supply + Residual = Observed
    bar_data = pd.DataFrame(
        {
            "Inflation expectations / inflation target": df["anchor"],
            "Demand": df["demand"],
            "Supply": df["supply_total"],
            "Noise": df["residual"],
        },
        index=df.index,
    )

    # Plot stacked bars
    ax = mg.bar_plot(
        bar_data,
        stacked=True,
        color=["#cccccc", "orange", "darkblue", "lightblue"],
    )

    # Add observed inflation line on top
    observed = df["observed"].copy()
    observed.name = "Observed Inflation (quarterly annualised)"
    mg.line_plot(observed, ax=ax, color="indigo", width=1.5, zorder=10)

    # Add equation text box
    add_equation_box(ax, EQ_UNSCALED)

    mg.finalise_plot(
        ax,
        title="Inflation Decomposition: Components (Unscaled)",
        ylabel="% p.a.",
        rheader=rfooter,
        lfooter="Australia. Decomposition based on augmented Phillips curve.",
        legend={"loc": "best", "fontsize": "x-small"},
        axhline={"y": 2.5, "color": "darkred", "linestyle": "--", "linewidth": 1, "label": "2.5% Target"},
        y0=True,
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
    2. Supply Contributions (import prices + GSCPI)
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
                    f"Primarily SUPPLY-driven (+{supply:.1f}pp from import/supply chain). "
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
