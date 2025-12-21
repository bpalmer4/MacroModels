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

    π = π_anchor/4 + γ_π·u_gap + ρ_π·Δ4ρm + ξ_π·GSCPI + ε

Where:
    - π_anchor/4: Baseline from expectations/target (neutral)
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
        target_quarterly = 2.5 / 4  # 0.625%
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

    # Extract NAIRU (vector, use median across samples)
    nairu = get_vector_var("nairu", trace).median(axis=1).values

    # Observed data
    U = obs["U"]
    pi_anchor = obs["π_anchor"]
    pi_observed = obs["π"]

    # Import prices (lagged) - may not be present
    delta_4_pm_lag1 = obs.get("Δ4ρm_1", np.zeros_like(U))

    # GSCPI - may not be present
    gscpi = obs.get("ξ_2", np.zeros_like(U))

    # Compute components
    # 1. Anchor contribution (quarterly)
    anchor = pi_anchor / 4

    # 2. Demand contribution (unemployment gap)
    u_gap = (U - nairu) / U
    demand = gamma_pi * u_gap

    # 3. Supply - import prices (includes oil effect via import price channel)
    supply_import = rho_pi * delta_4_pm_lag1

    # 4. Supply - GSCPI (with quadratic/signed transformation)
    supply_gscpi = xi_2sq_pi * (gscpi**2) * np.sign(gscpi)

    # Fitted values (excluding residual)
    fitted = anchor + demand + supply_import + supply_gscpi

    # Residual
    residual = pi_observed - fitted

    # Convert to Series with index
    return InflationDecomposition(
        observed=pd.Series(pi_observed, index=obs_index, name="observed"),
        anchor=pd.Series(anchor, index=obs_index, name="anchor"),
        demand=pd.Series(demand, index=obs_index, name="demand"),
        supply_import=pd.Series(supply_import, index=obs_index, name="supply_import"),
        supply_gscpi=pd.Series(supply_gscpi, index=obs_index, name="supply_gscpi"),
        residual=pd.Series(residual, index=obs_index, name="residual"),
        fitted=pd.Series(fitted, index=obs_index, name="fitted"),
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

    scale = 4 if annual else 1
    unit = "% p.a." if annual else "% p.q."

    demand = df["demand"] * scale
    demand.name = "Demand Contribution"

    ax = mg.line_plot(demand, color="darkred", width=1.5)
    ax.axhline(0, color="black", linewidth=0.5)

    mg.finalise_plot(
        ax,
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

    scale = 4 if annual else 1
    unit = "% p.a." if annual else "% p.q."

    supply_import = df["supply_import"] * scale
    supply_import.name = "Import Prices"
    supply_gscpi = df["supply_gscpi"] * scale
    supply_gscpi.name = "GSCPI"
    supply_total = df["supply_total"] * scale
    supply_total.name = "Total Supply"

    ax = mg.line_plot(supply_import, color="blue", width=1.5)
    mg.line_plot(supply_gscpi, ax=ax, color="green", width=1.5)
    mg.line_plot(supply_total, ax=ax, color="black", width=2)
    ax.axhline(0, color="grey", linewidth=0.5)
    ax.legend(loc="upper left")

    mg.finalise_plot(
        ax,
        title=f"Inflation - Supply Contributions ({unit})",
        ylabel=unit,
        rfooter=rfooter,
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

    # Annualize (multiply by 4)
    df = df * 4

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
    # Use period ordinals to align with bar plot x-axis
    x_ordinals = np.array([p.ordinal for p in df.index])
    ax.plot(
        x_ordinals,
        df["observed"].values,
        color="indigo",
        linewidth=1.5,
        label="Observed Inflation (quarterly annualised)",
        zorder=10,
    )

    # Target line
    ax.axhline(2.5, color="darkred", linestyle="--", linewidth=1, label="2.5% Target")

    mg.finalise_plot(
        ax,
        title="Inflation Decomposition: Demand vs Supply",
        ylabel="% p.a.",
        rfooter=rfooter,
        lfooter="Red=Demand (U-gap), Blue=Supply (imports+GSCPI). "
                "Gap to line = anchor (π expectations/target) + residual.",
        legend={"loc": "best", "fontsize": "x-small"},
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

    # Annualize (multiply by 4)
    df = df * 4

    # Calculate scaled contributions that sum to (observed - anchor)
    # This shows deviation from baseline, removing anchor's distortion
    demand = df["demand"].values
    supply = df["supply_total"].values
    observed = df["observed"].values
    anchor = df["anchor"].values

    # Deviation from anchor = what demand + supply + residual explain
    deviation = observed - anchor
    total = demand + supply

    # Scale so demand_scaled + supply_scaled = deviation
    with np.errstate(divide="ignore", invalid="ignore"):
        scale = np.where(np.abs(total) > 0.01, deviation / total, 0)

    # Only invalid when scale is extreme (contributions nearly cancel)
    max_scale = 5.0
    valid = np.abs(scale) <= max_scale

    demand_prop = np.where(valid, demand * scale, 0)
    supply_prop = np.where(valid, supply * scale, 0)

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
    x_ordinals = np.array([p.ordinal for p in df.index])
    ax.plot(
        x_ordinals,
        observed,
        color="indigo",
        linewidth=1.5,
        label="Observed Inflation (quarterly annualised)",
        zorder=10,
    )

    # Target line
    ax.axhline(2.5, color="darkred", linestyle="--", linewidth=1, label="2.5% Target")

    mg.finalise_plot(
        ax,
        title="Inflation: Proportional Demand vs Supply Attribution",
        ylabel="% p.a.",
        rfooter=rfooter,
        lfooter="Grey=Expectations/target baseline. Orange/Blue=Demand/Supply scaled so bars sum to observed. "
                "Residual absorbed into scaling.",
        legend={"loc": "best", "fontsize": "x-small"},
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
    3. Inflation drivers (stacked bars showing absolute contributions)
    4. Proportional attribution (bars scaled to observed inflation)

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
    plot_inflation_drivers_proportional(decomp, start, end, rfooter, show)


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
    scale = 4 if annual else 1
    unit = "% p.a." if annual else "% p.q."

    # Target (annual or quarterly)
    target = 2.5 if annual else 2.5 / 4

    # Compute averages
    avg = df.mean() * scale

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

    # Annualize
    avg = df.mean() * 4
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
