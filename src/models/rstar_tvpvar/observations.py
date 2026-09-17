"""Observation assembly for the TVP-VAR.

Three series, and deliberately only three:

- inflation, %            trimmed mean (ABS 6401.0)
- GDP growth, %           chain volume, seasonally adjusted (ABS 5206.0)
- real policy rate, %     RBA F1 cash rate less the chosen deflator

That is the Lubik-Matthes variable set. It is small on purpose: every extra
variable adds `lags x n` coefficients to a state that already drifts, and on
135 Australian quarters the binding constraint is how much time variation the
data can support, not how much information another series would add.

NO BOND YIELD, and no output gap. Those are the two things the other r* models
in this repo depend on and the two places their estimates are most fragile.
Leaving them out is what makes this model an independent reading rather than a
fourth variation on the same conditioning.

THE REAL POLICY RATE IS BUILT THE SAME WAY `rstar_bonds` BUILDS ITS `r`, cash
less a medium-horizon expectations measure, so the two models' answers are on
the same scale and the comparison in `rstar_summary` is like for like.
"""

import numpy as np
import pandas as pd

from src.data.cash_rate import get_cash_rate_qrtly
from src.data.commodity_prices import get_icp_aud_change_qrtly
from src.data.expectations_model import get_model_expectations_unanchored
from src.data.gdp import get_gdp_growth
from src.data.inflation import get_trimmed_mean_annual, get_trimmed_mean_qrtly
from src.data.twi import get_twi_change_annual, get_twi_change_qrtly
from src.models.common.sources import SourceSet
from src.models.rstar_tvpvar.config import ModelConfig

# Defaults are READ FROM ModelConfig, never restated. Restating them is how the
# sigma_q ensemble silently swept a FIVE-variable model while the config said
# three: `ensemble.py` called this function without the two variable switches
# and picked up `True` from this signature. `config.py` imports nothing from
# this package, so the dependency is one-way.
_DEFAULTS = ModelConfig()

# The order matters: `A` is lower triangular, so it asserts that inflation does
# not respond within the quarter to growth or the policy rate, and that growth
# does not respond within the quarter to the policy rate. That is the
# conventional monetary-VAR ordering, with the policy rate last because it is
# the variable assumed to see everything else contemporaneously.
# The full ordering. `A` is lower triangular, so position asserts what responds
# to what WITHIN a quarter. Commodity prices first because world prices for
# Australian exports respond to nothing here; the policy rate near the end
# because it is assumed to see the real economy; the TWI LAST because it is a
# fast financial variable that responds within the quarter to everything
# including the cash rate. That is the small-open-economy convention (Kim and
# Roubini; Dungey and Pagan for Australia).
VARIABLES = ("commodities", "inflation", "growth", "real_rate", "twi")
# The original Lubik-Matthes set, kept so it stays reproducible.
VARIABLES_NO_COMMODITIES = ("inflation", "growth", "real_rate")


def ordering(*, include_commodities: bool = True, include_twi: bool = True) -> tuple[str, ...]:
    """Return the VAR ordering for a given set of switches."""
    names = ["inflation", "growth", "real_rate"]
    if include_commodities:
        names.insert(0, "commodities")
    if include_twi:
        names.append("twi")
    return tuple(names)

# Positions read by `results`, which must not hardcode them: adding commodity
# prices at the front moves inflation from index 0 to index 1, and the
# projection overwrites the INFLATION element when it conditions on the anchor.
# Getting that wrong would silently condition the wrong variable.
def variable_index(name: str, variables: tuple[str, ...] = VARIABLES) -> int:
    """Return the position of `name` in the VAR ordering."""
    return variables.index(name)


# The ICP's quarterly change has a standard deviation of 6.6 against about 1.0
# for inflation, 1.6 for growth and 1.8 for the real rate. That matters here in
# a way it would not in an ordinary VAR: `sigma_q` is SHARED across all
# coefficients, so a variable on a much larger scale would effectively receive
# far more coefficient drift than the others for the same `sigma_q`. Dividing by
# four brings its sd to about 1.7, in line with the rest. The scale is
# arbitrary and harmless: commodity prices enter as an information variable and
# nothing downstream reads their level.
_ICP_SCALE = 4.0

# The TWI's four-quarter change has an sd of 7.69, so it is divided by five to
# land near 1.5, in line with the other four. Same reasoning as the ICP: with a
# SHARED `sigma_q`, a variable on a larger scale would quietly receive more
# coefficient drift than the rest.
_TWI_SCALE = 5.0

_NAME_WIDTH = 22


def _inflation(basis: str, sources: SourceSet) -> pd.Series:
    """Return trimmed mean inflation on the chosen basis, in per cent."""
    if basis == "annual":
        return sources.take(get_trimmed_mean_annual()).astype(float).dropna()
    # Annualised, so the two bases are on the same scale and `sigma_q` means the
    # same thing under both.
    quarterly = sources.take(get_trimmed_mean_qrtly()).astype(float).dropna()
    return quarterly * 4.0


def _growth(basis: str, sources: SourceSet) -> pd.Series:
    """Return GDP growth on the chosen basis, in per cent."""
    periods = 4 if basis == "annual" else 1
    growth = sources.take(get_gdp_growth(periods=periods)).astype(float).dropna()
    # ALREADY IN PER CENT. `get_log_gdp` scales the log by 100, so its difference
    # is a percentage change, not a proportion. Multiplying by 100 here put GDP
    # growth at 211% and blew up the VAR's second equation; the smoke run caught
    # it. Quarterly still needs annualising to match the other two series.
    return growth if basis == "annual" else growth * 4.0


def _commodities(sources: SourceSet) -> pd.Series:
    """Return the RBA commodity price index change, scaled, in per cent.

    WHY IT IS HERE. Without it the VAR shows the price puzzle: the real rate's
    coefficient in the inflation equation comes back +0.04, significantly
    POSITIVE in 29 quarters, because the RBA raises rates on information the
    three-variable model cannot see and inflation then rises anyway. The 1994-95
    tightening is the clearest case, hiking against 2.2% inflation on a forecast
    that proved right. Sims (1992) named the pathology; adding a variable
    carrying the central bank's information is the standard remedy.

    The RBA's ICP in AUD terms is the right one for Australia and arguably a
    stronger case than the US literature it comes from: commodity prices are a
    terms-of-trade INCOME shock here, so they drive both the inflation the Bank
    is reacting to and the demand that produced it. `commodity_prices.py` also
    argues it is more upstream than the terms of trade and more exogenous than
    net exports.
    """
    change = sources.take(get_icp_aud_change_qrtly()).astype(float).dropna()
    return change / _ICP_SCALE


def _twi(basis: str, sources: SourceSet) -> pd.Series:
    """Return the trade-weighted index change, scaled, in per cent.

    WHY IT IS HERE. Australia is small and open, and the exchange rate channel
    is FAST: a higher policy rate appreciates the currency, which cuts import
    prices within a quarter or two. The demand channel takes years and this VAR
    cannot see it at these lags; the FX channel operates on a timescale it can.
    It also completes the chain the ICP starts, since commodity prices drive the
    AUD and the AUD drives import prices.

    NOMINAL, not real: the real TWI is a level near 127 with an sd of 20, which
    a VAR cannot take without differencing it back to roughly this series.

    Ordered LAST, so `A` lets it respond within the quarter to everything
    including the cash rate, which is what a financial variable does.
    """
    change = (
        get_twi_change_annual() if basis == "annual" else get_twi_change_qrtly()
    )
    return sources.take(change).astype(float).dropna() / _TWI_SCALE


def _deflator(choice: str, sources: SourceSet) -> pd.Series:
    """Return the series that turns the nominal cash rate real.

    The UNANCHORED expectations measure, matching `rstar_bonds`' deflator. What
    matters for a real policy rate is the inflation people actually expected,
    not the target: this is the rate savers and borrowers faced, and the VAR is
    being asked what rate the economy settled toward under it.
    """
    if choice == "expectations":
        return sources.take(get_model_expectations_unanchored()).astype(float).dropna()
    if choice == "trimmed":
        return sources.take(get_trimmed_mean_annual()).astype(float).dropna()
    raise ValueError(f"deflator must be one of {DEFLATORS_MSG}, got {choice!r}")


DEFLATORS_MSG = "'expectations' or 'trimmed'"


def _report_coverage(listed: dict[str, pd.Series], index: pd.PeriodIndex) -> None:
    """Print the span of every series loaded, and the estimation sample."""
    print("Input series coverage:")
    for name, series in listed.items():
        clean = series.dropna()
        if len(clean):
            print(f"  {name:<{_NAME_WIDTH}} {clean.index.min()} -> {clean.index.max()}  n={len(clean)}")
        else:
            print(f"  {name:<{_NAME_WIDTH}} unavailable")
    print(f"\nEstimation sample: {index.min()} to {index.max()}  ({len(index)} quarters)")


def _blank(frame: pd.DataFrame, blank_quarters: tuple[str, ...]) -> None:
    """Blank the chosen quarters IN PLACE, leaving the calendar intact.

    Dropping rows would break the lag structure, since the VAR needs y_{t-1} and
    y_{t-2} to be the actual previous quarters. They are blanked instead and the
    likelihood skips them, which keeps the drifting coefficients connected
    across the gap. `estimate` reads the mask off the NaNs.
    """
    if not blank_quarters:
        return
    drop = [pd.Period(q, "Q") for q in blank_quarters]
    missing = [q for q, p in zip(blank_quarters, drop, strict=True) if p not in frame.index]
    if missing:
        # Silently blanking nothing would report a successful exclusion run that
        # was really the unexcluded one.
        raise ValueError(f"quarters to blank are outside the sample: {', '.join(missing)}")
    frame.loc[frame.index.isin(drop), :] = np.nan


def build_observations(
    start: str | None = _DEFAULTS.start,
    end: str | None = _DEFAULTS.end,
    *,
    basis: str = _DEFAULTS.basis,
    deflator: str = _DEFAULTS.deflator,
    include_commodities: bool = _DEFAULTS.include_commodities,
    include_twi: bool = _DEFAULTS.include_twi,
    # ONE list of quarters to drop from the likelihood, not a flag plus a list.
    # Callers pass `ModelConfig.blanked_quarters`, which folds the COVID switch
    # and any explicit `--exclude-quarters` together, so no caller can honour
    # one and forget the other.
    blank_quarters: tuple[str, ...] = (),
    verbose: bool = False,
) -> tuple[np.ndarray, pd.PeriodIndex, pd.DataFrame, SourceSet]:
    """Build the VAR's data matrix.

    Returns:
        Tuple of:
          - data: (T, 3) array, columns in `VARIABLES` order
          - index: the aligned quarterly PeriodIndex
          - frame: the same data as a DataFrame, for charting
          - sources: the providers behind the series, for chart footers

    """
    sources = SourceSet()
    columns: dict[str, pd.Series] = {}
    if include_commodities:
        columns["commodities"] = _commodities(sources)
    columns["inflation"] = _inflation(basis, sources)
    columns["growth"] = _growth(basis, sources)
    if include_twi:
        columns["twi"] = _twi(basis, sources)
    cash = sources.take(get_cash_rate_qrtly()).astype(float).dropna()
    columns["real_rate"] = (cash - _deflator(deflator, sources)).dropna()

    frame = pd.DataFrame(columns).dropna()
    index = frame.index
    if not isinstance(index, pd.PeriodIndex):
        index = pd.PeriodIndex(index, freq="Q")
    frame.index = index.asfreq("Q")

    if start:
        frame = frame.loc[frame.index >= pd.Period(start, "Q")]
    if end:
        frame = frame.loc[frame.index <= pd.Period(end, "Q")]

    _blank(frame, blank_quarters)

    if frame.dropna().empty:
        raise ValueError(f"No observations remain for start={start!r}, end={end!r}")

    obs_index = frame.index
    if not isinstance(obs_index, pd.PeriodIndex):
        raise TypeError("aligned observations must carry a PeriodIndex")

    order = ordering(include_commodities=include_commodities, include_twi=include_twi)
    if verbose:
        _report_coverage({name: frame[name] for name in order}, obs_index)

    data = frame.loc[:, list(order)].to_numpy(dtype=float)
    return data, obs_index, frame, sources


def design_matrix(data: np.ndarray, lags: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the regressor matrix, the left-hand side, and the usable-row mask.

    Row t of `X` is [1, y_{t-1}, ..., y_{t-lags}] and row t of `Y` is y_t, both
    running over the quarters for which a full set of lags exists. The mask is
    False wherever the row or any of its lags is missing, which is how excluded
    COVID quarters leave the likelihood without breaking the calendar.

    Lives here rather than in `estimate` because `results` needs it too, to fit
    the constant-coefficient baseline, and importing PyMC to build a design
    matrix would make `--analyse-only` pay for the whole sampler.
    """
    n_obs = data.shape[0]
    rows, targets, usable = [], [], []
    for t in range(lags, n_obs):
        lagged = np.concatenate([data[t - lag] for lag in range(1, lags + 1)])
        row = np.concatenate([[1.0], lagged])
        rows.append(row)
        targets.append(data[t])
        usable.append(bool(np.isfinite(row).all() and np.isfinite(data[t]).all()))
    design = np.asarray(rows, dtype=float)
    target = np.asarray(targets, dtype=float)
    mask = np.asarray(usable, dtype=bool)
    # NaNs would poison the dot product even on masked rows, since the mask is
    # applied to the log-likelihood and not to the arithmetic.
    return np.nan_to_num(design), np.nan_to_num(target), mask


def ols_fit(
    data: np.ndarray,
    lags: int,
    training_quarters: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return constant-coefficient OLS coefficients and the residual log variances.

    `training_quarters` limits the fit to the first N quarters of the sample,
    which is the training-sample prior a Primiceri TVP-VAR starts from. None
    fits on every usable row, which centres the prior on the same
    constant-coefficient answer the time variation is meant to be tested
    against. See `ModelConfig.training_sample_quarters`.

    Raises if the training window leaves too few usable rows to identify the
    coefficients, rather than returning a rank-deficient fit.
    """
    design, target, mask = design_matrix(data, lags)
    if training_quarters is not None:
        window = np.zeros_like(mask)
        window[: max(training_quarters - lags, 0)] = True
        mask = mask & window
    n_rows, n_coef = int(mask.sum()), design.shape[1]
    if n_rows <= n_coef:
        raise ValueError(
            f"OLS needs more usable rows than coefficients: {n_rows} rows for {n_coef} "
            f"coefficients (training_quarters={training_quarters}, lags={lags})",
        )
    coefficients, *_ = np.linalg.lstsq(design[mask], target[mask], rcond=None)
    resid = target[mask] - design[mask] @ coefficients
    return coefficients, np.log(resid.var(axis=0).clip(min=1e-6))
