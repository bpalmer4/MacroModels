"""Inflation decomposition bar charts."""

import pandas as pd

from src.models.nairu.analysis._decomposition_helpers import (
    eq_unscaled, hcoe_eq_unscaled, plot_decomposition_bars, wage_eq_unscaled,
)
from src.models.nairu.analysis.decomposition_types import (
    HCOEInflationDecomposition,
    InflationDecomposition,
    WageInflationDecomposition,
)
from src.utilities.rate_conversion import annualize

_CONFIG = {
    InflationDecomposition: {
        "kind": "Price Inflation",
        "obs_color": "indigo",
        "obs_label": "Observed Inflation (quarterly annualised)",
        "target_line": 2.5,
        "lheader": "",
        "lfooter": "Australia. Decomposition based on augmented Phillips curve.",
        "eq": lambda d: eq_unscaled(d.has_import_price, d.has_gscpi),
    },
    WageInflationDecomposition: {
        "kind": "Wage-ULC Inflation",
        "obs_color": "darkorange",
        "obs_label": "Observed ULC Growth (quarterly annualised)",
        "target_line": None,
        "lheader": "ULC = Unit Labour Costs.",
        "lfooter": "Australia. Decomposition based on augmented wage Phillips curve.",
        "eq": lambda d: wage_eq_unscaled(d.has_price_passthrough, d.has_expectations),
    },
    HCOEInflationDecomposition: {
        "kind": "Wage-HCOE Inflation",
        "obs_color": "darkorange",
        "obs_label": "Observed Hourly COE Growth (quarterly annualised)",
        "target_line": None,
        "lheader": "HCOE = Hourly Compensation of Employees.",
        "lfooter": "Australia. Decomposition based on hourly compensation Phillips curve.",
        "eq": lambda d: hcoe_eq_unscaled(d.has_price_passthrough, d.has_expectations),
    },
}


def plot_decomposition(decomp, *, rfooter="", show=False):
    """Plot inflation decomposition as stacked bars with all components."""
    cfg = _CONFIG[type(decomp)]
    df = annualize(decomp.to_dataframe())

    # Anchor
    if isinstance(decomp, InflationDecomposition):
        cols = {"Inflation expectations": df["anchor"]}
    else:
        label = "Baseline (α + expectations)" if decomp.has_expectations else "Baseline (α)"
        cols = {label: df["anchor"]}
    colors = ["#cccccc", "orange"]

    # Demand
    cols["Demand" if isinstance(decomp, InflationDecomposition) else "Demand (labor market)"] = df["demand"]

    # Type-specific middle columns
    if isinstance(decomp, InflationDecomposition):
        if decomp.has_import_price or decomp.has_gscpi:
            cols["Supply"] = df["supply_total"]
            colors.append("darkblue")
    else:
        if decomp.has_price_passthrough:
            cols["Price pass-through"] = df["price_passthrough"]
            colors.append("darkblue")
        if isinstance(decomp, HCOEInflationDecomposition):
            cols["Productivity"] = df["productivity"]
            colors.append("limegreen")

    # Residual
    cols["Noise"] = df["residual"]
    colors.append("lightblue")

    plot_decomposition_bars(
        bar_data=pd.DataFrame(cols, index=df.index),
        observed=df["observed"],
        colors=colors,
        title=f"{cfg['kind']} Decomposition",
        equation=cfg["eq"](decomp),
        observed_color=cfg["obs_color"],
        observed_label=cfg["obs_label"],
        target_line=cfg["target_line"],
        lheader=cfg["lheader"],
        lfooter=cfg["lfooter"],
        eq_x=0.5, eq_y=0.02,
        rfooter=rfooter,
        show=show,
    )
