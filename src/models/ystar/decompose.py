"""Post-modelling accounting split of potential growth into hours and productivity.

This does **not** change the model. It takes the estimated potential output
path `y*` exactly as the model produced it and splits its growth rate into a
labour-input part and a labour-productivity part, using an identity that holds
in logs:

    log Y*  =  log H*  +  log (Y/H)*
    g_Y*    =  g_H*    +  g_LP*

and, one level down, the demographic identity for hours:

    log H*  =  log POP*  +  log PR*  +  log HPP*

with HPP = hours per labour-force participant. That third term absorbs the
unemployment margin, which keeps the identity closed in three terms without
requiring a NAIRU (the same choice `equations/trend_hours.py` makes).

**Trend productivity here is a residual, not an estimate.** Trend hours is a
filter of measured data; trend productivity is whatever is left of `y*` after
trend hours is subtracted, computed draw by draw so it carries the whole of the
model's uncertainty about `y*`. Any error in the hours trend lands entirely on
productivity. The decomposition is therefore an accounting statement about the
model's own potential path, not independent evidence about productivity.

Hours, population and participation come from the LFS (ABS 6202.0), matching
`observations.py`. Because the LFS hours concept differs in scope from the
National Accounts hours index, GDP per LFS hour is not the ABS published
productivity measure; the level of `lp*` is not interpretable and only its
growth is.
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from statsmodels.tsa.filters.hp_filter import hpfilter

from src.data.labour_force import (
    get_civilian_population_qrtly,
    get_hours_worked_qrtly,
    get_participation_rate_qrtly,
)
from src.models.common.sources import SourceSet
from src.models.ystar.observations import smooth_log_level
from src.models.ystar.results import PotentialResults

# Henderson terms applied to log population. 7 matches the treatment
# `observations.py` already gives it, and is the ABS convention for quarterly
# series.
DEFAULT_TERMS = 7

# HP smoothing applied to the two cyclical margins. 1600 is the standard
# quarterly setting; see `_trend_hours_components` for why a Henderson MA
# cannot be used here and why population does not get this treatment.
DEFAULT_LAMBDA = 1600.0

# Additivity is exact in logs, so the reconciliation check should fail on
# anything above floating-point noise rather than on a tolerance chosen to pass.
_ADDITIVITY_TOL = 1e-9

# Period averages for the contributions chart. Blocks rather than single
# quarters because the question the split answers ("why did the speed limit
# fall?") is a low-frequency one; 2020-2026 is kept separate because the
# pandemic swing in participation and hours does not belong in a decade mean.
CONTRIBUTION_BLOCKS: tuple[tuple[str, str, str], ...] = (
    ("1994-1999", "1994Q1", "1999Q4"),
    ("2000-2009", "2000Q1", "2009Q4"),
    ("2010-2019", "2010Q1", "2019Q4"),
    ("2020-2026", "2020Q1", "2026Q4"),
)

COMPONENTS = ("Population", "Participation", "Hours per participant", "Productivity")


@dataclass
class GrowthDecomposition:
    """Year-ended potential growth split into additive components.

    Attributes:
        components: Median contributions, one column per element of
            `COMPONENTS`, in year-ended percentage points. They sum to
            `potential_growth` exactly.
        potential_growth: The model's own year-ended potential growth, median.
        hours_growth: Population + participation + hours per participant.
        productivity_posterior: Trend productivity growth, time x draw, so the
            residual component can be banded.
        terms: Henderson terms used on log population.
        lamb: HP smoothing used on the two cyclical margins.
        sources: Everything behind these numbers, the run's own inputs plus the
            labour series loaded here, for the chart footers.

    """

    components: pd.DataFrame
    potential_growth: pd.Series
    hours_growth: pd.Series
    productivity_posterior: pd.DataFrame
    terms: int
    lamb: float
    sources: SourceSet = field(default_factory=SourceSet)

    @property
    def productivity_growth(self) -> pd.Series:
        """Median trend productivity growth, year-ended per cent."""
        return self.productivity_posterior.median(axis=1)

    def block_means(self) -> pd.DataFrame:
        """Average contributions over `CONTRIBUTION_BLOCKS`, blocks as rows."""
        rows = {}
        for label, start, end in CONTRIBUTION_BLOCKS:
            window = self.components.loc[
                (self.components.index >= pd.Period(start, "Q"))
                & (self.components.index <= pd.Period(end, "Q"))
            ]
            if not window.empty:
                rows[label] = window.mean()
        return pd.DataFrame(rows).T


def _hp_trend(log_level: pd.Series, lamb: float) -> pd.Series:
    """Return the HP trend of a log level series."""
    clean = log_level.dropna()
    _, trend = hpfilter(clean, lamb=lamb)
    return pd.Series(np.asarray(trend), index=clean.index)


def _trend_hours_components(
    index: pd.PeriodIndex,
    terms: int,
    lamb: float,
    sources: SourceSet,
) -> pd.DataFrame:
    """Return trend log population, participation and hours per participant.

    Each is trended separately, so the three sum to trend hours exactly. Units
    are log x 100 throughout, matching the model's states. Trending runs on the
    full history from 1978 and is reindexed to the model sample afterwards, so
    the first in-sample quarter gets a full symmetric window.

    **The three components do not get the same filter, and that is deliberate.**

    Population keeps the Henderson treatment `observations.py` already gives it
    (ARIMA-extended tail, `terms`-term MA). Population is measured and
    acyclical, so its large swings are genuine movements in labour supply:
    growth ran from 0.20% in 2021Q2 to 2.95% in 2023Q3 on the border closure
    and the migration rebound. An HP filter treats that as cycle and smooths it
    away, which would be wrong — it is exactly the supply shock the
    decomposition is supposed to show.

    Participation and hours per participant are the cyclical margins, and they
    need a genuine low-pass. A Henderson MA cannot supply one at any width: on
    hours, a 7-term filter books a −6.2 to +7.3 per cent pandemic swing as
    trend, and even a 31-term filter still books −0.5 to +4.5. HP(1600) leaves
    trend hours growth running 0.7% in 2021 and 3.7% in 2023, which is the
    border closure and the migration rebound rather than the lockdown.

    Note what this makes trend hours: an HP-filtered trend of measured labour
    input, plus a Henderson trend of population. That is what a route-A
    accounting split is. It is not an estimate of labour supply consistent with
    inflation at target, and nothing here identifies one.
    """
    log_hours = np.log(sources.take(get_hours_worked_qrtly())) * 100
    log_pop = np.log(sources.take(get_civilian_population_qrtly())) * 100
    log_pr = np.log(sources.take(get_participation_rate_qrtly())) * 100

    raw = pd.DataFrame({"pop": log_pop, "pr": log_pr, "hours": log_hours}).dropna()
    # Hours per labour-force participant, by the identity. Hours is a level in
    # '000 and participation a per cent, so hpp carries an arbitrary additive
    # constant; only its growth is used.
    raw["hpp"] = raw["hours"] - raw["pop"] - raw["pr"]

    trends = pd.DataFrame({
        "pop": smooth_log_level(raw["pop"], terms),
        "pr": _hp_trend(raw["pr"], lamb),
        "hpp": _hp_trend(raw["hpp"], lamb),
    })
    return trends.reindex(index)


def decompose_potential_growth(
    results: PotentialResults,
    terms: int = DEFAULT_TERMS,
    lamb: float = DEFAULT_LAMBDA,
) -> GrowthDecomposition:
    """Split the model's potential growth into hours and productivity parts.

    Args:
        results: A loaded trace. Any specification with a `potential_output`
            state works; `labour` already has its own decomposition and should
            use that instead.
        terms: Henderson terms applied to log population.
        lamb: HP smoothing for participation and hours per participant.

    Returns:
        A `GrowthDecomposition`. Components are year-ended percentage points
        and sum to the model's own potential growth to floating-point accuracy.

    Raises:
        ValueError: if the components do not reconcile, which would mean the
            identity has been broken somewhere upstream.

    """
    # The labour series loaded here are the decomposition's own: they are not in
    # the model's observations, so the charts drawn from it name a wider set of
    # sources than the charts drawn from the run.
    recorded = SourceSet.from_records(results.constants.get("sources"))
    sources = recorded or SourceSet()
    trends = _trend_hours_components(results.obs_index, terms, lamb, sources)
    trend_hours = trends.sum(axis=1)

    potential = results.potential_posterior()
    # lp* = y* - h*, draw by draw: the residual inherits all of y*'s uncertainty.
    productivity = potential.sub(trend_hours, axis=0)

    components = pd.DataFrame({
        "Population": trends["pop"].diff(4),
        "Participation": trends["pr"].diff(4),
        "Hours per participant": trends["hpp"].diff(4),
        "Productivity": productivity.diff(4).median(axis=1),
    })

    potential_growth = results.potential_growth_posterior().median(axis=1)
    hours_growth = trend_hours.diff(4)

    # The three hours columns sum to `hours_growth` by construction, so that is
    # not worth checking. What can fail is the join to the model: the columns
    # must sum to the *model's own* potential growth, which is computed
    # independently from the y* path. That holds only if the sample indices
    # align and trend hours has no missing quarters inside the sample, so a
    # mismatch here is a real defect rather than arithmetic.
    reconciled = (components[list(COMPONENTS)].sum(axis=1) - potential_growth).abs().max()
    if pd.notna(reconciled) and reconciled > _ADDITIVITY_TOL:
        raise ValueError(f"Potential growth components do not sum to their total: max error {reconciled:.3e}")

    return GrowthDecomposition(
        components=components,
        potential_growth=potential_growth,
        hours_growth=hours_growth,
        productivity_posterior=productivity.diff(4),
        terms=terms,
        lamb=lamb,
        # Empty where the run recorded nothing, so a chart drawn from an older
        # run falls back to its module's constant rather than reporting the
        # three labour series as if they were the whole input set.
        sources=sources if recorded is not None else SourceSet(),
    )


def print_decomposition(decomposition: GrowthDecomposition) -> None:
    """Print the block averages and the latest quarter's split."""
    print("\nPotential growth decomposition (year-ended %, period averages)")
    print("-" * 78)
    blocks = decomposition.block_means()
    blocks["Total"] = blocks.sum(axis=1)
    print(blocks.round(2).to_string())

    last = decomposition.components.dropna().index[-1]
    row = decomposition.components.loc[last]
    print(f"\nLatest quarter ({last})")
    print("-" * 78)
    for name in COMPONENTS:
        print(f"  {name:<28} {row[name]:6.2f}")
    print(f"  {'Total (potential growth)':<28} {row.sum():6.2f}")
    print(
        f"\n  Memo: model potential growth {decomposition.potential_growth.loc[last]:.2f}"
        f", of which trend hours {decomposition.hours_growth.loc[last]:.2f}"
        f" and trend productivity {decomposition.productivity_growth.loc[last]:.2f}",
    )
    print(
        f"  Population trended with a {decomposition.terms}-term Henderson MA;"
        f" participation and hours per participant with HP({decomposition.lamb:g})."
        " Productivity is the residual.",
    )
