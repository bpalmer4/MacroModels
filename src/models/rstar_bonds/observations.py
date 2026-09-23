"""Observation assembly for the rstar model.

Three series drive the estimation:

- AU indexed (real) 10-year yield, %   (RBA F2, via `bonds`)
- world r*, %                          (NY Fed HLW estimates, via `world_rstar`)
- AU real cash rate, %                 (RBA F1 less the chosen deflator)

The real cash rate is the second window on the same state. A long real yield is
roughly the average expected real short rate over its term plus a premium, so
the short rate carries level information about r* that the long yield alone
cannot supply: with one series, the starting point of a random walk and the mean
of a stationary premium are not separable. Two series do not make the level
free either, one of the two means must still be asserted, but they make the
assertion checkable against the other window.

Everything else is carried for the derived series and the charts, and is
deliberately *not* allowed to shorten the estimation sample:

- corporate A 5y credit spread, ppt    (RBA F3 less F2) — 2005Q1 onwards
- trimmed mean inflation, annual %     (ABS 6401.0)
- the output gap                       (a completed `ystar` run) — 1993Q1 onwards
- the unemployment gap                 (a completed `ustar` run) — optional

The indexed yield is used rather than the nominal one because it is a direct
real-rate observation: no expected-inflation subtraction, and no inflation risk
premium to strip. The cost is that indexed AGS are thin, so the yield carries a
liquidity premium the nominal series does not. That premium is part of what
`mu_tp` absorbs.

WINDOW ONE IS A SWITCH. Under `nominal_window` the long-end observable is not
the indexed yield but the AOFM's RISK-NEUTRAL nominal 10-year yield, deflated:
a yield with the term premium already removed by someone else. The core column
is called `y` either way, because it is whatever window one observes and every
downstream reader wants the series the model was actually fitted on. Which one
it holds is recorded in the run's constants, and under the nominal window the
indexed yield is still carried as a chart extra so the two can be compared.
"""

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from src.data.acm_loader import get_acm_term_premium
from src.data.aofm_loader import (
    get_aofm_5y5y_forward,
    get_aofm_risk_neutral_yield,
    get_aofm_term_premium,
)
from src.data.bonds import get_corporate_spread, get_indexed_yield_filled
from src.data.cash_rate import get_cash_rate_qrtly
from src.data.dataseries import DataSeries
from src.data.expectations_model import get_model_expectations_unanchored
from src.data.fred_loader import get_fred_series
from src.data.inflation import get_trimmed_mean_annual
from src.data.rba_loader import get_bank_bill_rate, get_cgs_yield, get_lending_rate
from src.data.world_rstar import get_world_rstar
from src.models.common.sources import SourceSet
from src.models.rstar_bonds.config import (
    KIM_WRIGHT_SERIES,
    MARKET_WORLD_SOURCES,
    US_PREMIUM_SOURCES,
    WORLD_REAL_SERIES,
)
from src.models.ustar.results import load_results as load_ustar_results
from src.models.ystar.results import load_results as load_ystar_results
from src.models.ystar_ustar.results import load_results as load_joint_results

if TYPE_CHECKING:
    # The class itself, for annotations only; the loader it comes from is
    # imported above as `load_joint_results`.
    from src.models.ystar_ustar.results import JointResults

_NAME_WIDTH = 26


def _us_premium(premium_source: str, sources: SourceSet) -> pd.Series:
    """Return the published US 10-year term premium on a quarterly PeriodIndex.

    The same series serves both places a US premium enters, the `market` world
    anchor and the `--us-premium` pin, so one setting governs both. See
    `config.US_PREMIUM_SOURCES` for what the two providers disagree about.
    """
    if premium_source == "acm":
        series = sources.take(get_acm_term_premium())
    elif premium_source == "kim-wright":
        series = sources.take(get_fred_series(KIM_WRIGHT_SERIES))
    else:
        raise ValueError(
            f"Unknown US premium source {premium_source!r}; expected one of "
            f"{', '.join(repr(s) for s in US_PREMIUM_SOURCES)}",
        )
    quarterly = series.groupby(pd.PeriodIndex(series.index, freq="Q")).mean()
    return quarterly.dropna().astype(float)


def _au_premium(source: str, maturity: int, sources: SourceSet) -> pd.Series:
    """Return the AOFM Australian term premium on a quarterly PeriodIndex.

    A quarterly mean of a daily series, matching how the US premium and the
    market world anchor are handled: it is being aligned against a quarterly
    state and the daily series is noisy.

    This is a NOMINAL premium and the `tp` it pins is real, so the inflation
    risk premium stays in the residual spread. That is the whole residual now,
    though, which is the gain over `--us-premium`: there the spread also carried
    a liquidity difference against TIPS and a currency risk premium.
    """
    series = sources.take(get_aofm_term_premium(maturity, source)).astype(float).dropna()
    quarterly = series.groupby(pd.PeriodIndex(series.index, freq="Q")).mean()
    return quarterly.dropna().astype(float)


def _risk_neutral_real_yield(
    maturity: int,
    source: str,
    deflator_series: pd.Series,
    sources: SourceSet,
) -> pd.Series:
    """Return the AOFM risk-neutral nominal yield, deflated, quarterly.

    The average expected nominal short rate over the tenor with the term premium
    already stripped, less expected inflation: the real expected policy path the
    long-end equation wants, arrived at without a latent premium.

    Two things it is not. It is not premium-free in truth, only premium-free by
    AOFM's model, so their specification error lands in `r*` and is no longer
    visible as a fitted residual. And the deflator is a medium-to-long horizon
    measure being asked to deflate a ten-year yield, which is a smaller horizon
    mismatch than the same series makes at the overnight end but is not zero.
    """
    series = sources.take(get_aofm_risk_neutral_yield(maturity, source)).astype(float).dropna()
    quarterly = series.groupby(pd.PeriodIndex(series.index, freq="Q")).mean()
    return (quarterly - deflator_series).dropna().astype(float)


def _real_forward(source: str, deflator_series: pd.Series, sources: SourceSet) -> pd.Series:
    """Return the AOFM 5y5y risk-neutral forward, deflated, quarterly.

    A THIRD WINDOW, and the only one that speaks to the LEVEL of r* directly.
    The other two cannot: from the indexed yield and the real cash rate the
    model can pin their SUM but not the split between r* and the premium, which
    is the -0.87 correlation between `wedge_0` and `mu_tp` that the notes have
    always flagged. `mu_spread` then carries the level, and its posterior moves
    5% off its prior.

    The 5y5y forward is a market price for where the cash rate settles over
    years five to ten with the term premium removed by AOFM, less long-run
    expected inflation. At 2026Q2 it reads 1.31 real against this model's 0.84.

    BE CLEAR THAT IT IS LESS INDEPENDENT HERE THAN IN `rstar_rba`. That model
    observes only the cash rate, so a bond price was genuinely new information.
    This model already reads the indexed 10-year yield, so a nominal
    risk-neutral forward is a second look at a curve it is watching. The two
    instruments differ, and the AOFM has stripped the premium from one and not
    the other, but they are not independent.
    """
    nominal = sources.take(get_aofm_5y5y_forward(source)).astype(float).dropna()
    quarterly = nominal.groupby(pd.PeriodIndex(nominal.index, freq="Q")).mean()
    return (quarterly - deflator_series).dropna().astype(float)


def _world_series(source: str, sources: SourceSet, premium_source: str = "kim-wright") -> pd.Series:
    """Return the chosen world r* series on a quarterly PeriodIndex.

    Either a Holston-Laubach-Williams estimate or a market price. The market
    options exist because HLW is a model output identified through the IS and
    Phillips curves, so it cannot respond to a bond selloff and did not respond
    to this one: see `config.WORLD_SOURCES`.
    """
    if source == "market":
        # The term premium comes out explicitly rather than being absorbed into
        # a shrunken loading. See `config.WORLD_REAL_SERIES`.
        real = sources.take(get_fred_series(WORLD_REAL_SERIES)).pipe(
            lambda s: s.groupby(pd.PeriodIndex(s.index, freq="Q")).mean(),
        )
        return (real - _us_premium(premium_source, sources)).dropna().astype(float)

    if source in MARKET_WORLD_SOURCES:
        # Quarterly mean rather than the quarter-end value: it is being matched
        # against a quarterly state, and the daily series is noisy.
        return sources.take(
            get_fred_series(MARKET_WORLD_SOURCES[source]),
        ).pipe(lambda s: s.groupby(pd.PeriodIndex(s.index, freq="Q")).mean()).dropna().astype(float)

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
    results = load_joint_results(prefix=prefix)
    _record_parent(results, sources)
    return results


def _ystar_gap(prefix: str, sources: SourceSet) -> pd.Series:
    """Return the median output gap from a completed ystar run."""
    results = load_ystar_results(prefix=prefix)
    _record_parent(results, sources)
    return results.output_gap_median()


def _ustar_gap(prefix: str, sources: SourceSet) -> pd.Series:
    """Return the median unemployment gap from a completed ustar run."""
    results = load_ustar_results(prefix=prefix)
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
    results = load_ustar_results(prefix=prefix)
    _record_parent(results, sources)
    return _supply_annual(results.inflation_decomposition()["supply"])


def _deflator(choice: str, sources: SourceSet) -> pd.Series:
    """Return the inflation series used to turn the nominal cash rate real.

    An identification choice rather than a detail, so it is a switch and not a
    hardcoded series. `expectations` matches what `is_curve` uses, which keeps
    the two packages comparable, but it is a spliced medium-to-long horizon
    measure (a 10-year bond reading before 1991Q2, survey-based after) being
    asked to deflate an overnight rate. `trimmed` is backward-looking instead,
    with the opposite bias: it lags turning points rather than anticipating
    them.

    Neither is right. The point of the switch is that the answer can be shown
    under both.
    """
    if choice == "expectations":
        return sources.take(get_model_expectations_unanchored()).astype(float).dropna()
    if choice == "trimmed":
        return sources.take(get_trimmed_mean_annual()).astype(float).dropna()
    raise ValueError(f"deflator must be 'expectations' or 'trimmed', got {choice!r}")


def _short_rate(choice: str, sources: SourceSet) -> pd.Series:
    """Return the nominal short rate the second window is built on.

    `cash` is the overnight rate. It cannot move before the RBA moves it, so
    when the market prices a tightening the Bank has not yet delivered, the
    anticipation shows up in every other rate on the curve and not in this one.
    The model then has to put that difference somewhere, and the only places
    available are r* and the premium.

    `bill` is the 90-day bank-accepted bill, which covers the next quarter's
    expected path and does move first. The two differ by +0.16 on average since
    1993 (sd 0.21), and the gap opens at exactly the turning points: +0.90 in
    1994Q4, +0.68 in 2008Q1, +0.57 in 2018Q2, +0.87 in 2022Q2.

    Nothing downstream changes. `g` is the real short rate less r* with a
    coefficient of one, and a 90-day rate is the one-quarter point on the same
    expected-path curve the 3-year and 10-year equations read further along, so
    it belongs at H = 1 exactly as the overnight rate did.

    3-month OIS would be the cleaner measure of the same thing and is not
    available: `FIRMMOIS3` ends in November 2022.
    """
    if choice == "cash":
        return sources.take(get_cash_rate_qrtly()).astype(float).dropna()
    if choice == "bill":
        monthly = sources.take(get_bank_bill_rate(90)).astype(float).dropna()
        index = monthly.index
        if not isinstance(index, pd.PeriodIndex):
            index = pd.PeriodIndex(index, freq="M")
        monthly.index = index
        return monthly.groupby(monthly.index.asfreq("Q")).last()
    raise ValueError(f"short_rate must be 'cash' or 'bill', got {choice!r}")


def _mortgage_rate(sources: SourceSet, kind: str = "housing_oo") -> pd.Series:
    """Return a variable owner-occupier mortgage rate, quarterly.

    The household analogue of the corporate spread the package already carries.
    Monthly at source (RBA F5), so it is converted here rather than arriving
    quarterly like the ABS series.

    `housing_oo` is the discounted rate, what a borrower actually pays, and it
    begins 2004Q2. `housing_oo_standard` is the advertised rate and reaches back
    to 1959, which is the only way to say anything about the 1990s.
    """
    monthly = sources.take(get_lending_rate(kind)).astype(float).dropna()
    index = monthly.index
    if not isinstance(index, pd.PeriodIndex):
        index = pd.PeriodIndex(index, freq="M")
    monthly.index = index
    return monthly.groupby(monthly.index.asfreq("Q")).last()


def _chart_extras(
    cash: pd.Series,
    *,
    input_source: str,
    joint_prefix: str,
    ystar_prefix: str,
    ustar_prefix: str,
    au_premium_source: str,
    au_premium_maturity: int,
    indexed_yield: pd.Series | None,
    sources: SourceSet,
) -> dict[str, pd.Series]:
    """Return the ragged series carried for the derived results and the charts.

    None of these enters the estimation, so each is loaded through `_optional`
    and a missing one costs its chart rather than the run. The exception is the
    cash rate, which is an estimation input loaded by the caller and passed in
    here only so the charts can draw it.
    """
    joint = input_source == "joint"
    return {
        "spread": _optional("corporate spread", get_corporate_spread, sources),
        # The household analogue of the corporate spread. Carried, never
        # estimated on: `g` is the risk-free stance and this is what borrowers
        # actually paid, and between 2009 and 2022 the two diverge by up to
        # 1.75 points. Begins 2004Q2.
        "mortgage": _optional("mortgage rate", lambda: _mortgage_rate(sources), sources),
        # The advertised rate, carried because the discounted one begins only in
        # 2004Q2 and the model sample starts in 1993. They are not splice-able:
        # the discount off the standard rate is not a constant, running 0.58
        # over 2004-07 and 1.40 over 2020-26, so both are charted instead.
        "mortgage_std": _optional(
            "standard mortgage rate", lambda: _mortgage_rate(sources, "housing_oo_standard"), sources,
        ),
        # The AOFM's published Australian term premium. Carried on EVERY run,
        # including ones that do not pin to it, because the comparison against
        # the model's own fitted `tp` is the external check this package has
        # never had: `tp` is stationary about a constant by assertion, and this
        # is the first Australian series that can say whether it should be.
        # Both AOFM methods decompose the same curve, so the one named here is
        # the one the run would pin to, keeping the chart and the pin consistent.
        "aofm_tp": _optional(
            "AOFM term premium",
            lambda: _au_premium(au_premium_source, au_premium_maturity, sources),
            sources,
        ),
        # Present only under `nominal_window`, where the indexed yield is not an
        # estimation input but is still the natural comparator for the series
        # that replaced it.
        **({"y_indexed": indexed_yield} if indexed_yield is not None else {}),
        "pi": _optional("trimmed mean inflation", get_trimmed_mean_annual, sources),
        # The UNANCHORED series, and it stays unanchored: this is the deflator
        # for actual borrowing and policy rates, where what matters is what
        # people expect rather than what the target says.
        "pi_exp": _optional(
            "inflation expectations", get_model_expectations_unanchored, sources,
        ),
        "cash_rate": cash,
        # The Taylor rule's three inputs. From one joint run by default, so
        # they are mutually consistent; see `ModelConfig.input_source`.
        "ygap": _optional(
            f"{input_source} output gap",
            (lambda: _joint_results(joint_prefix, sources).output_gap_median())
            if joint else (lambda: _ystar_gap(ystar_prefix, sources)),
            sources,
        ),
        "ugap": _optional(
            f"{input_source} unemployment gap",
            (lambda: _joint_results(joint_prefix, sources).ugap_median())
            if joint else (lambda: _ustar_gap(ustar_prefix, sources)),
            sources,
        ),
        "supply": _optional(
            f"{input_source} supply contribution",
            (lambda: _supply_annual(
                _joint_results(joint_prefix, sources).inflation_decomposition()["supply"]))
            if joint else (lambda: _ustar_supply(ustar_prefix, sources)),
            sources,
        ),
    }


def _report_coverage(listed: dict[str, pd.Series], obs_index: pd.PeriodIndex) -> None:
    """Print the span of every series loaded, and the estimation sample.

    Worth printing rather than inferring: the estimation sample is the
    intersection of the core series, and a ragged extra that quietly arrives
    short is easier to see here than in a chart with a gap in it.
    """
    print("Input series coverage:")
    for name, series in listed.items():
        clean = series.dropna()
        if len(clean):
            print(f"  {name:<{_NAME_WIDTH}} {clean.index.min()} -> {clean.index.max()}  n={len(clean)}")
        else:
            print(f"  {name:<{_NAME_WIDTH}} unavailable")
    print(f"\nEstimation sample: {obs_index.min()} to {obs_index.max()}  ({len(obs_index)} quarters)")


def _real_medium_yield(maturity: int, deflator_series: pd.Series, sources: SourceSet) -> pd.Series:
    """Return a real nominal-CGS yield at `maturity` years, on a quarterly index.

    The belly of the curve is here because the overnight rate is a poor summary
    of the expected policy path exactly when it matters. In 2022Q1 the real
    two-year sat 1.24 points above the real overnight rate, the market having
    priced a normalisation the RBA had not yet begun. A model that reads the
    expected path off the cash rate alone must push that difference into r* or
    the premium, and the wedge's largest move in the sample lands in the very
    next quarter.

    Deflated by the same series as the cash rate, which is a horizon mismatch:
    a three-year nominal yield wants three-year expected inflation. It is a
    smaller mismatch than the one at the overnight end, since the expectations
    series is itself a medium-horizon measure, but it is not zero.
    """
    sources.add("RBA")
    series = get_cgs_yield(maturity=maturity).data.copy()
    index = series.index
    if not isinstance(index, pd.PeriodIndex):
        index = pd.PeriodIndex(index, freq="M")
    series.index = index
    quarterly = series.groupby(series.index.asfreq("Q")).last().astype(float)
    return (quarterly - deflator_series).dropna()


def _core_columns(
    yield_real: pd.Series,
    world: pd.Series,
    real_cash: pd.Series,
    deflator_series: pd.Series,
    sources: SourceSet,
    *,
    us_premium_anchor: bool,
    us_premium_source: str,
    use_curve: bool,
    curve_maturity: int,
    au_premium_anchor: bool,
    au_premium_source: str,
    au_premium_maturity: int,
    nominal_window: bool,
    use_forward: bool,
) -> dict[str, pd.Series]:
    """Return the series that define the estimation sample.

    Core membership is the decision about what may truncate the run, so the
    optional ones are added only when the specification actually uses them: the
    3-year real yield begins 1992Q2, the Kim-Wright premium 1990Q1, the ACM one
    1961Q2 and the AOFM decomposition 1992Q3, and none should shorten a sample
    that does not need it.

    `y` is window one whichever series that is. Under `nominal_window` the
    indexed yield is not in the estimation at all, so it must not sit in core
    constraining the sample; it moves to the chart extras instead.
    """
    window = (
        _risk_neutral_real_yield(au_premium_maturity, au_premium_source, deflator_series, sources)
        if nominal_window
        else yield_real
    )
    columns = {"y": window, "w": world, "r": real_cash}
    if us_premium_anchor:
        columns["us_tp"] = _us_premium(us_premium_source, sources)
    if au_premium_anchor:
        columns["au_tp"] = _au_premium(au_premium_source, au_premium_maturity, sources)
    if use_forward:
        columns["f"] = _real_forward(au_premium_source, deflator_series, sources)
    if use_curve:
        columns["m"] = _real_medium_yield(curve_maturity, deflator_series, sources)
    return columns


def build_observations(
    start: str | None = "1986Q3",
    end: str | None = None,
    *,
    world_source: str = "mean",
    deflator: str = "expectations",
    short_rate: str = "cash",
    us_premium_anchor: bool = False,
    us_premium_source: str = "kim-wright",
    use_curve: bool = False,
    curve_maturity: int = 3,
    au_premium_anchor: bool = False,
    au_premium_source: str = "bc",
    au_premium_maturity: int = 10,
    nominal_window: bool = False,
    use_forward: bool = False,
    input_source: str = "joint",
    joint_prefix: str = "ystar_ustar",
    ystar_prefix: str = "ystar",
    ustar_prefix: str = "ustar",
    verbose: bool = False,
) -> tuple[dict[str, np.ndarray], pd.PeriodIndex, pd.DataFrame, SourceSet]:
    """Build observation arrays for rstar estimation.

    The estimation sample is the intersection of the yield, world r* and the
    real cash rate. The extras are reindexed onto it and may carry NaN — the
    corporate spread begins in 2005 and the output gap in 1993, and neither
    should truncate the estimation sample.

    Returns:
        Tuple of:
          - obs: dict of numpy arrays for the three estimation series
          - obs_index: the aligned PeriodIndex
          - chart_obs: DataFrame of everything, including the ragged extras
          - sources: the providers behind those series, for the chart footers

    """
    sources = SourceSet()
    yield_real = sources.take(get_indexed_yield_filled()).astype(float).dropna()
    world = _world_series(world_source, sources, us_premium_source)

    # The short rate and its deflator are estimation inputs now, not chart
    # extras, so a failure to load either is fatal rather than a skipped chart.
    # The cash rate is loaded separately and unconditionally: whatever the
    # second window is built on, the charts and the Taylor rule compare against
    # the rate the RBA actually sets.
    cash = sources.take(get_cash_rate_qrtly()).astype(float).dropna()
    short = _short_rate(short_rate, sources)
    deflator_series = _deflator(deflator, sources)
    real_cash = (short - deflator_series).dropna()

    core = pd.DataFrame(_core_columns(
        yield_real, world, real_cash, deflator_series, sources,
        us_premium_anchor=us_premium_anchor,
        us_premium_source=us_premium_source,
        use_curve=use_curve,
        curve_maturity=curve_maturity,
        au_premium_anchor=au_premium_anchor,
        au_premium_source=au_premium_source,
        au_premium_maturity=au_premium_maturity,
        nominal_window=nominal_window,
        use_forward=use_forward,
    )).dropna()
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

    extras = _chart_extras(
        cash,
        input_source=input_source,
        joint_prefix=joint_prefix,
        ystar_prefix=ystar_prefix,
        ustar_prefix=ustar_prefix,
        au_premium_source=au_premium_source,
        au_premium_maturity=au_premium_maturity,
        indexed_yield=yield_real if nominal_window else None,
        sources=sources,
    )

    chart_obs = core.copy()
    for name, series in extras.items():
        chart_obs[name] = series.reindex(obs_index) if len(series) else np.nan

    if verbose:
        _report_coverage(
            {
                (
                    f"y (real risk-neutral {au_premium_maturity}y, AOFM {au_premium_source})"
                    if nominal_window
                    else "y (indexed real yield)"
                ): core["y"],
                f"w (world r*, {world_source})": world,
                f"r (real {short_rate}, {deflator})": real_cash,
                **({f"m (real {curve_maturity}y CGS)": core["m"]} if use_curve else {}),
                **(
                    {f"au_tp (AOFM {au_premium_maturity}y, {au_premium_source})": core["au_tp"]}
                    if au_premium_anchor else {}
                ),
                **extras,
            },
            obs_index,
        )

    # Built from whatever `_core_columns` decided, so an optional core series
    # cannot be added there and silently dropped here.
    obs = {name: core[name].to_numpy(dtype=float) for name in core.columns}
    return obs, obs_index, chart_obs, sources
