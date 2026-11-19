"""Run each GDP nowcast and keep its Q/Q distribution for the target quarter.

Every distribution is one normal, centred on the model's headline with the sigma
behind its own bands. The mixture form stays general, but no model needs more than
one component. Nothing is re-estimated here.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.data.gdp import get_gdp
from src.data.population import (
    get_population_growth_annual,
    get_population_growth_proxy,
    get_population_growth_qrtly,
)
from src.models.common.nowcast_core import compute_tty, value_at
from src.models.gdp_nowcast_bridge.model import nowcast as bridge_nowcast
from src.models.gdp_nowcast_bvar.model import nowcast as bvar_nowcast
from src.models.gdp_nowcast_components.model import nowcast as components_nowcast
from src.models.gdp_nowcast_dfm.model import nowcast as dfm_nowcast

# How far a value this package derives may sit from the model's own figure:
# both are the same arithmetic on the same data, so beyond float noise they
# would mean the summary and the model disagree.
TOLERANCE = 1e-6


@dataclass(frozen=True)
class Component:
    """One normal in a mixture."""

    mean: float
    sd: float
    weight: float


@dataclass(frozen=True)
class Nowcast:
    """One model's Q/Q nowcast distribution, as a weighted mixture of normals."""

    label: str
    point_qoq: float           # the model's own headline Q/Q nowcast (%)
    reported_tty: float        # the model's own annual growth figure (%)
    components: list[Component]

    @property
    def mixture_mean(self) -> float:
        """Mean of the mixture."""
        return sum(c.weight * c.mean for c in self.components)


@dataclass(frozen=True)
class Gathered:
    """Every usable nowcast, plus what the annual conversion needs."""

    target: pd.Period
    gdp: pd.Series             # published CVM GDP, ending the quarter before target
    nowcasts: list[Nowcast]
    omitted: dict[str, str]    # label -> why it is not charted
    population_qoq: float      # population growth in the target quarter (%, compounded quarterly rate)
    population_annual: float   # population growth over the year to the target quarter (%)


def gather(verbose: bool = True) -> Gathered:
    """Run the four nowcasts and collect their distributions."""
    quiet = not verbose
    bridge = bridge_nowcast(quiet=quiet)
    dfm = dfm_nowcast(quiet=quiet)
    bvar = bvar_nowcast(quiet=quiet)
    components = components_nowcast()

    targets = {bridge.target_quarter, dfm.target_quarter, bvar.target_quarter, components.target}
    if len(targets) != 1:
        msg = f"nowcasts disagree on the target quarter: {sorted(str(t) for t in targets)}"
        raise ValueError(msg)
    target = bridge.target_quarter

    gdp = get_gdp(gdp_type="CVM", seasonal="SA").data.dropna()
    gdp = gdp.loc[gdp.index < target]

    nowcasts = [
        Nowcast("Bridge", bridge.gdp_qoq, bridge.gdp_tty,
                [Component(bridge.gdp_qoq, bridge.qoq_sigma, 1.0)]),
        Nowcast("DFM", dfm.gdp_qoq, dfm.gdp_tty, [Component(dfm.gdp_qoq, dfm.qoq_sigma, 1.0)]),
        Nowcast("BVAR", bvar.gdp_qoq, bvar.gdp_tty, [Component(bvar.gdp_qoq, bvar.qoq_sigma, 1.0)]),
    ]
    omitted: dict[str, str] = {}

    # Its band measures only the statistical discrepancy, so while a component
    # is still unpublished the curve would claim a certainty the model lacks.
    pending = [name for name, value in components.contributions.items() if np.isnan(value)]
    if pending:
        omitted["Components"] = f"still pending: {', '.join(pending)}"
    else:
        nowcasts.append(Nowcast(
            "Components", components.gdp_qoq, components.gdp_tty,
            [Component(components.gdp_qoq, components.disc_sigma, 1.0)],
        ))

    for n in nowcasts:
        derived = compute_tty(n.point_qoq, gdp, target)
        if abs(derived - n.reported_tty) > TOLERANCE:
            msg = (
                f"{n.label}: annual growth derived here ({derived:.6f}) differs from the "
                f"model's own ({n.reported_tty:.6f}); the two used different GDP data."
            )
            raise ValueError(msg)

    # Population for per capita growth: the forward proxy for total (ERP) growth, which
    # must rest on a complete Labour Force quarter, not extrapolated months.
    last_complete = get_population_growth_proxy().last_complete
    if target > last_complete:
        msg = f"population: last complete Labour Force quarter is {last_complete}, target is {target}"
        raise ValueError(msg)
    population_qoq = value_at(get_population_growth_qrtly(), target)
    population_annual = value_at(get_population_growth_annual(), target)

    return Gathered(
        target=target,
        gdp=gdp,
        nowcasts=nowcasts,
        omitted=omitted,
        population_qoq=population_qoq,
        population_annual=population_annual,
    )
