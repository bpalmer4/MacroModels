"""Observation assembly for the rstar model.

Two series drive the estimation:

- AU indexed (real) 10-year yield, %   (RBA F2, via `bonds`)
- world r*, %                          (NY Fed HLW estimates, via `world_rstar`)

Everything else is carried for the derived series and the charts, and is
deliberately *not* allowed to shorten the estimation sample:

- corporate A 5y credit spread, ppt    (RBA F3 less F2) — 2005Q1 onwards
- trimmed mean inflation, annual %     (ABS 6401.0)
- the output gap                       (a completed `ystar` run) — 1993Q1 onwards
- the unemployment gap                 (a completed `ustar` run) — optional
- the cash rate, %                     (RBA F1)

The indexed yield is used rather than the nominal one because it is a direct
real-rate observation: no expected-inflation subtraction, and no inflation risk
premium to strip. The cost is that indexed AGS are thin, so the yield carries a
liquidity premium the nominal series does not. That premium is part of what
`mu_tp` absorbs.
"""

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from src.data.bonds import get_corporate_spread, get_indexed_yield_filled
from src.data.cash_rate import get_cash_rate_qrtly
from src.data.dataseries import DataSeries
from src.data.expectations_model import get_model_expectations_unanchored
from src.data.inflation import get_trimmed_mean_annual
from src.data.world_rstar import get_world_rstar
from src.models.common.sources import SourceSet

if TYPE_CHECKING:
    # Type only. The runtime import stays inside `_joint_results`, so a missing
    # joint run costs the Taylor-rule charts rather than the whole module.
    from src.models.ystar_ustar.results import JointResults

_NAME_WIDTH = 26


def _world_series(source: str, sources: SourceSet) -> pd.Series:
    """Return the chosen world r* series on a quarterly PeriodIndex."""
    # The published HLW estimates arrive as a plain DataFrame, so the provider
    # is named here rather than read off a `DataSeries`.
    sources.add("NY Fed")
    df = get_world_rstar()
    index = df.index
    if not isinstance(index, pd.PeriodIndex):
        index = pd.PeriodIndex(index, freq="Q")
    df.index = index

    series = df.mean(axis=1) if source == "mean" else df[source]
    return series.dropna().astype(float)


def _optional(name: str, loader: object, sources: SourceSet) -> pd.Series:
    """Load a series for charting, returning an empty one if it is unavailable.

    The derived Taylor rule and business rate depend on other models' saved
    output. A missing one should cost the affected chart, not the whole run:
    r* itself needs neither. A loader that returns a `DataSeries` has its
    provider recorded here; one that fails records nothing, so the footer names
    only what the charts could actually draw.
    """
    try:
        loaded = loader()  # type: ignore[operator]
    except (FileNotFoundError, KeyError, ValueError) as exc:
        print(f"  note: {name} unavailable ({type(exc).__name__}); dependent charts will be skipped")
        return pd.Series(dtype=float)
    series = sources.take(loaded) if isinstance(loaded, DataSeries) else loaded
    return series.astype(float)


def _record_parent(results: object, sources: SourceSet) -> None:
    """Record the inputs of a completed run whose output is read as data here."""
    constants = getattr(results, "constants", None)
    if not isinstance(constants, dict):
        return
    recorded = SourceSet.from_records(constants.get("sources"))
    if recorded is not None:
        for source, cat in recorded.records:
            sources.add(source, cat)


def _joint_results(prefix: str, sources: SourceSet) -> JointResults:
    """Load a completed joint y*/u* run.

    All three Taylor-rule inputs come from one model here, so they share a
    potential output, a u* and an estimate of `c`. Read from the two parents
    instead, they do not: `ustar` takes `ystar`'s gap as data, so its
    unemployment gap is conditional on a gap `ystar` may since have revised.
    """
    from src.models.ystar_ustar.results import load_results  # noqa: PLC0415 — optional dependency

    results = load_results(prefix=prefix)
    _record_parent(results, sources)
    return results


def _ystar_gap(prefix: str, sources: SourceSet) -> pd.Series:
    """Return the median output gap from a completed ystar run."""
    from src.models.ystar.results import load_results  # noqa: PLC0415 — optional dependency

    results = load_results(prefix=prefix)
    _record_parent(results, sources)
    return results.output_gap_median()


def _ustar_gap(prefix: str, sources: SourceSet) -> pd.Series:
    """Return the median unemployment gap from a completed ustar run."""
    from src.models.ustar.results import load_results  # noqa: PLC0415 — optional dependency

    results = load_results(prefix=prefix)
    _record_parent(results, sources)
    return results.ugap_median()


def _supply_annual(supply: pd.Series) -> pd.Series:
    """Put a quarterly supply contribution on a four-quarter basis.

    A rolling sum, not `annualize()`: the decomposition chart uses the latter,
    but it is a compounding transform and not additive across components, and
    the Taylor rule needs the terms to add up.
    """
    return supply.rolling(4).sum()


def _ustar_supply(prefix: str, sources: SourceSet) -> pd.Series:
    """Return the supply contribution to inflation, on a four-quarter basis.

    `ustar`'s Phillips decomposition isolates `rho·d4pm + xi·GSCPI^2·sign` as
    the supply term. It is a *quarterly* contribution, and the Taylor rule runs
    on four-quarter inflation, so the annual equivalent is a rolling four-
    quarter sum — not `annualize()`, which the decomposition chart uses but
    which is a compounding transform and not additive across components.
    """
    from src.models.ustar.results import load_results  # noqa: PLC0415 — optional dependency

    results = load_results(prefix=prefix)
    _record_parent(results, sources)
    return _supply_annual(results.inflation_decomposition()["supply"])


def build_observations(
    start: str | None = "1986Q3",
    end: str | None = None,
    *,
    world_source: str = "mean",
    input_source: str = "joint",
    joint_prefix: str = "ystar_ustar",
    ystar_prefix: str = "ystar",
    ustar_prefix: str = "ustar",
    verbose: bool = False,
) -> tuple[dict[str, np.ndarray], pd.PeriodIndex, pd.DataFrame, SourceSet]:
    """Build observation arrays for rstar estimation.

    The estimation sample is the intersection of the yield and world r* only.
    The extras are reindexed onto it and may carry NaN — the corporate spread
    begins in 2005 and the output gap in 1993, and neither should truncate a
    sample that starts in 1986.

    Returns:
        Tuple of:
          - obs: dict of numpy arrays for the two estimation series
          - obs_index: the aligned PeriodIndex
          - chart_obs: DataFrame of everything, including the ragged extras
          - sources: the providers behind those series, for the chart footers

    """
    sources = SourceSet()
    yield_real = sources.take(get_indexed_yield_filled()).astype(float).dropna()
    world = _world_series(world_source, sources)

    core = pd.DataFrame({"y": yield_real, "w": world}).dropna()
    index = core.index
    if not isinstance(index, pd.PeriodIndex):
        index = pd.PeriodIndex(index, freq="Q")
    core.index = index.asfreq("Q")

    if start:
        core = core.loc[core.index >= pd.Period(start, "Q")]
    if end:
        core = core.loc[core.index <= pd.Period(end, "Q")]
    if core.empty:
        raise ValueError(f"No observations remain for start={start!r}, end={end!r}")

    obs_index = core.index
    if not isinstance(obs_index, pd.PeriodIndex):
        raise TypeError("aligned observations must carry a PeriodIndex")

    extras = {
        "spread": _optional("corporate spread", get_corporate_spread, sources),
        "pi": _optional("trimmed mean inflation", get_trimmed_mean_annual, sources),
        "pi_exp": _optional(
            "inflation expectations", get_model_expectations_unanchored, sources,
        ),
        "cash_rate": _optional("cash rate", get_cash_rate_qrtly, sources),
        # The Taylor rule's three inputs. From one joint run by default, so
        # they are mutually consistent; see `ModelConfig.input_source`.
        "ygap": _optional(
            f"{input_source} output gap",
            (lambda: _joint_results(joint_prefix, sources).output_gap_median())
            if input_source == "joint" else (lambda: _ystar_gap(ystar_prefix, sources)),
            sources,
        ),
        "ugap": _optional(
            f"{input_source} unemployment gap",
            (lambda: _joint_results(joint_prefix, sources).ugap_median())
            if input_source == "joint" else (lambda: _ustar_gap(ustar_prefix, sources)),
            sources,
        ),
        "supply": _optional(
            f"{input_source} supply contribution",
            (lambda: _supply_annual(
                _joint_results(joint_prefix, sources).inflation_decomposition()["supply"]))
            if input_source == "joint" else (lambda: _ustar_supply(ustar_prefix, sources)),
            sources,
        ),
    }

    chart_obs = core.copy()
    for name, series in extras.items():
        chart_obs[name] = series.reindex(obs_index) if len(series) else np.nan

    if verbose:
        print("Input series coverage:")
        listed = {
            "y (indexed real yield)": yield_real,
            f"w (world r*, {world_source})": world,
            **extras,
        }
        for name, series in listed.items():
            clean = series.dropna()
            if len(clean):
                print(f"  {name:<{_NAME_WIDTH}} {clean.index.min()} -> {clean.index.max()}  n={len(clean)}")
            else:
                print(f"  {name:<{_NAME_WIDTH}} unavailable")
        print(f"\nEstimation sample: {obs_index.min()} to {obs_index.max()}  ({len(obs_index)} quarters)")

    obs = {"y": core["y"].to_numpy(dtype=float), "w": core["w"].to_numpy(dtype=float)}
    return obs, obs_index, chart_obs, sources
