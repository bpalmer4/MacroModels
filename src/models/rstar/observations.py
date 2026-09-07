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

import numpy as np
import pandas as pd

from src.data.bonds import get_corporate_spread, get_indexed_yield_filled
from src.data.cash_rate import get_cash_rate_qrtly
from src.data.expectations_model import get_model_expectations_unanchored
from src.data.inflation import get_trimmed_mean_annual
from src.data.world_rstar import get_world_rstar

_NAME_WIDTH = 26


def _world_series(source: str) -> pd.Series:
    """Return the chosen world r* series on a quarterly PeriodIndex."""
    df = get_world_rstar()
    index = df.index
    if not isinstance(index, pd.PeriodIndex):
        index = pd.PeriodIndex(index, freq="Q")
    df.index = index

    series = df.mean(axis=1) if source == "mean" else df[source]
    return series.dropna().astype(float)


def _optional(name: str, loader: object) -> pd.Series:
    """Load a series for charting, returning an empty one if it is unavailable.

    The derived Taylor rule and business rate depend on other models' saved
    output. A missing one should cost the affected chart, not the whole run:
    r* itself needs neither.
    """
    try:
        return loader()  # type: ignore[operator]
    except (FileNotFoundError, KeyError, ValueError) as exc:
        print(f"  note: {name} unavailable ({type(exc).__name__}); dependent charts will be skipped")
        return pd.Series(dtype=float)


def _joint_results(prefix: str) -> object:
    """Load a completed joint y*/u* run.

    All three Taylor-rule inputs come from one model here, so they share a
    potential output, a u* and an estimate of `c`. Read from the two parents
    instead, they do not: `ustar` takes `ystar`'s gap as data, so its
    unemployment gap is conditional on a gap `ystar` may since have revised.
    """
    from src.models.ystar_ustar.results import load_results  # noqa: PLC0415 — optional dependency

    return load_results(prefix=prefix)


def _ystar_gap(prefix: str) -> pd.Series:
    """Return the median output gap from a completed ystar run."""
    from src.models.ystar.results import load_results  # noqa: PLC0415 — optional dependency

    return load_results(prefix=prefix).output_gap_median()


def _ustar_gap(prefix: str) -> pd.Series:
    """Return the median unemployment gap from a completed ustar run."""
    from src.models.ustar.results import load_results  # noqa: PLC0415 — optional dependency

    return load_results(prefix=prefix).ugap_median()


def _supply_annual(supply: pd.Series) -> pd.Series:
    """Put a quarterly supply contribution on a four-quarter basis.

    A rolling sum, not `annualize()`: the decomposition chart uses the latter,
    but it is a compounding transform and not additive across components, and
    the Taylor rule needs the terms to add up.
    """
    return supply.rolling(4).sum()


def _ustar_supply(prefix: str) -> pd.Series:
    """Return the supply contribution to inflation, on a four-quarter basis.

    `ustar`'s Phillips decomposition isolates `rho·d4pm + xi·GSCPI^2·sign` as
    the supply term. It is a *quarterly* contribution, and the Taylor rule runs
    on four-quarter inflation, so the annual equivalent is a rolling four-
    quarter sum — not `annualize()`, which the decomposition chart uses but
    which is a compounding transform and not additive across components.
    """
    from src.models.ustar.results import load_results  # noqa: PLC0415 — optional dependency

    return _supply_annual(load_results(prefix=prefix).inflation_decomposition()["supply"])


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
) -> tuple[dict[str, np.ndarray], pd.PeriodIndex, pd.DataFrame]:
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

    """
    yield_real = get_indexed_yield_filled().data.astype(float).dropna()
    world = _world_series(world_source)

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
        "spread": _optional("corporate spread", lambda: get_corporate_spread().data.astype(float)),
        "pi": _optional("trimmed mean inflation", lambda: get_trimmed_mean_annual().data.astype(float)),
        "pi_exp": _optional(
            "inflation expectations",
            lambda: get_model_expectations_unanchored().data.astype(float),
        ),
        "cash_rate": _optional("cash rate", lambda: get_cash_rate_qrtly().data.astype(float)),
        # The Taylor rule's three inputs. From one joint run by default, so
        # they are mutually consistent; see `ModelConfig.input_source`.
        "ygap": _optional(
            f"{input_source} output gap",
            (lambda: _joint_results(joint_prefix).output_gap_median())
            if input_source == "joint" else (lambda: _ystar_gap(ystar_prefix)),
        ),
        "ugap": _optional(
            f"{input_source} unemployment gap",
            (lambda: _joint_results(joint_prefix).ugap_median())
            if input_source == "joint" else (lambda: _ustar_gap(ustar_prefix)),
        ),
        "supply": _optional(
            f"{input_source} supply contribution",
            (lambda: _supply_annual(
                _joint_results(joint_prefix).inflation_decomposition()["supply"]))
            if input_source == "joint" else (lambda: _ustar_supply(ustar_prefix)),
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
    return obs, obs_index, chart_obs
