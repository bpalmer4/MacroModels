"""The series the regime reading needs, and the regime each quarter falls in."""

import numpy as np
import pandas as pd

from src.data.expectations_model import get_model_expectations
from src.data.gscpi_live import get_gscpi_qrtly_live
from src.data.import_prices import get_import_price_growth_lagged_annual
from src.data.inflation import get_trimmed_mean_annual
from src.data.labour_force import (
    get_unemployment_rate_qrtly,
    get_unemployment_speed_limit_qrtly,
)
from src.data.long_cpi import get_long_headline_annual
from src.data.tot import get_tot_change_qrtly
from src.data.ulc import get_ulc_growth_qrtly
from src.models.common.sources import SourceSet
from src.models.regime_ustar.config import ModelConfig


def regime_index(index: pd.PeriodIndex, breaks: tuple[str, ...]) -> np.ndarray:
    """Which regime each quarter belongs to, as 0-based integers.

    Each break is the FIRST quarter of the regime it opens, so a quarter sits
    in regime k when it is at or after break k-1 and before break k.
    """
    cuts = [pd.Period(b, freq="Q") for b in breaks]
    return np.searchsorted(np.asarray(cuts), np.asarray(index), side="right").astype(int)


def regime_labels(index: pd.PeriodIndex, regimes: np.ndarray, n_regimes: int) -> list[str]:
    """Return a "1974Q1-1983Q2" label per regime, taken from the quarters it covers.

    Built from the data rather than from the break dates, so a regime whose
    span is trimmed by `--start` or by the end of the sample is named for what
    it actually contains.
    """
    labels = []
    for k in range(n_regimes):
        span = index[regimes == k]
        labels.append(f"{span[0]}-{span[-1]}" if len(span) else "empty")
    return labels


def salience_expectation(
    pi: pd.Series, config: ModelConfig,
) -> pd.Series:
    """Return expectations formed by attention that switches on above a threshold.

        theta_t  =  theta_lo   if pi_t <= threshold
                 =  theta_hi   if pi_t >  threshold
        pi_e_t   =  pi_e_{t-1} + theta_t x (pi_{t-1} - pi_e_{t-1})

    **Attention is not constant, and that is the whole idea.** Below the
    threshold almost nobody is thinking about inflation, which is most of what
    anchoring is; above it every conversation starts with how expensive things
    have become, and expectations track prints closely. One switch replaces
    both the asserted 2.5 anchor and its asserted end date: the 1960s come out
    anchored because inflation was under 6, not because a date says so.

    **Why a hard switch rather than a ramp.** A learning rate proportional to
    the level of inflation was tried and is worse in the 1960s: at 2.5% it
    still leaves theta near 0.12, so expectations drift with prints and the
    1967-69 reading comes out at 1.54 against `long_run_ustar`'s 1.82. The hard
    switch gives 1.85. Smoothing the switch over a 1.5-point width gives 1.61,
    for the same reason: any leakage into the calm decade un-anchors it.

    **It also removes the overshoot a trailing average cannot avoid.** A
    3-year moving average lags the climb to 17% and then overshoots the
    descent, giving mean surprises of +4.71 across 1973-75 and -2.23 across
    1976-79, which drew a visible kink in u* and put the implied reading at
    -2.22 in 1977Q4. Here the same figures are +1.01 and -0.48, because at 13%
    inflation attention is still on and expectations come down nearly as fast
    as they went up.

    **The threshold is asserted, not estimated, and cannot be estimated.** The
    only era with a measured expectations series, 1983 onwards, is one where
    every high-inflation quarter is also an anchored one, so the level effect
    and the anchor effect are perfectly confounded there. Scored on that
    window alone the data prefer expectations that respond MORE when inflation
    undershoots than when it overshoots, the reverse of the mechanism here.
    That is a fact about a period in which a central bank was establishing and
    then defending credibility, and it does not speak to 1974. Sweep the
    threshold rather than defending 6.
    """
    e = pd.Series(index=pi.index, dtype=float)
    prev = config.anchor_level
    lagged = pi.shift(1)
    for t in pi.index:
        e[t] = prev
        observed = lagged[t]
        if pd.isna(observed):
            continue
        theta = config.theta_hi if observed > config.salience_threshold else config.theta_lo
        prev = prev + theta * (observed - prev)
    return e


def regime_expectation(
    pi: pd.Series, config: ModelConfig, sources: SourceSet,
) -> tuple[pd.Series, pd.Series]:
    """Return the expectation each quarter's Phillips curve is written against, and which kind.

    Two phases. Before `measured_from` the expectation is formed by
    `salience_expectation`; from there the expectations model's own series
    takes over.

    **The measured series takes over as soon as it exists**, in 1983Q1, rather
    than at some later date, because it carries things no rule would: the 1980s
    plateau, where expectations sat between 6.6 and 8.0 for eight years while
    inflation swung from 11 to 2.6 and back; and the 1993-97 disbelief, where
    they averaged 3.15 against a 2.5 target that was not yet credible, settling
    only in 1998. Handing over early imposes no anchoring, because the series
    is not anchored there.
    """
    measured = sources.take(get_model_expectations(), "inflation expectations (model)", key="pi_exp")
    pi_e = salience_expectation(pi, config)

    switch = pd.Period(config.measured_from, freq="Q")
    usable = measured.reindex(pi_e.index)
    from_switch = pd.Series(pi_e.index >= switch, index=pi_e.index) & usable.notna()
    return pi_e.where(~from_switch, usable), from_switch


def _attach_optional(frame: pd.DataFrame, u: pd.Series, config: ModelConfig, sources: SourceSet) -> pd.DataFrame:
    """Attach the columns that must not drive the sample's start date.

    Both of these would otherwise trim the front of the sample through the
    `dropna` that builds the frame: the supply series do not begin until 1984Q3
    and 1998Q1, and `u_move` needs twelve quarters of history. Including them
    in the main frame silently cost every run the first eleven quarters,
    including runs that never read them.
    """
    if config.supply_control:
        # Zero-filled outside their spans rather than truncating the sample to
        # them, which would cost 1959-1984 entirely. See
        # `ModelConfig.supply_control` for why zero is neutral for GSCPI and is
        # not neutral for import price growth.
        d4pm = sources.take(get_import_price_growth_lagged_annual(), "import price growth", key="d4pm")
        gscpi = sources.take(get_gscpi_qrtly_live(), "GSCPI (lagged)", key="gscpi").astype(float)
        frame["d4pm"] = d4pm.reindex(frame.index).fillna(0.0)
        frame["gscpi"] = gscpi.shift(2).reindex(frame.index).fillna(0.0)

    # How much the labour market has actually shifted lately, for
    # `adaptive_sigma`. The one-year average rather than the raw rate, so a
    # single quarter like 2020Q2 does not register as structural change; two
    # years apart, so it measures a shift rather than a wobble.
    frame["u_move"] = (u.rolling(4).mean() - u.rolling(4).mean().shift(8)).abs().reindex(frame.index)
    if config.adaptive_sigma:
        frame = frame.dropna(subset=["u_move"])
    return frame


def build_observations(config: ModelConfig) -> tuple[pd.DataFrame, np.ndarray, list[str], SourceSet]:
    """Return the aligned frame, the regime index, the regime labels and the sources.

    The frame carries:
      `pi`   year-ended inflation, the measure `config.inflation` names
      `pi_e`       the expectation the regime says was held: anchored, then a
                   trailing average, then measured. See `regime_expectation`
      `surprise`   `pi - pi_e`, which is what the Phillips curve explains
      `measured`   1.0 where `pi_e` is measured rather than asserted
      `u`    the unemployment rate, already shifted by `config.lag`
      `tot`  terms of trade growth, only when the control is on
    """
    sources = SourceSet()
    if config.inflation == "trimmed":
        pi = sources.take(get_trimmed_mean_annual(), "trimmed mean CPI, year-ended", key="pi")
    else:
        pi = sources.take(get_long_headline_annual(), "headline CPI, year-ended", key="pi")
    u = sources.take(get_unemployment_rate_qrtly(), "unemployment rate", key="u")

    pi_e, from_switch = regime_expectation(pi, config, sources)

    columns = {
        "pi": pi,
        "pi_e": pi_e,
        # Inflation less the expectation the regime says was held. Negative is
        # a slack signal: prices came in under what was expected.
        "surprise": pi - pi_e,
        "measured": from_switch.astype(float),
        "u": u.shift(config.lag),
    }
    if config.tot_control:
        columns["tot"] = sources.take(get_tot_change_qrtly(), "terms of trade growth", key="tot")

    if config.wage_equation:
        # Year-ended from the quarterly growth rate, to match the price
        # equation's units. Both reach the sample: ULC growth begins 1959Q4 and
        # the speed limit 1960Q1, so the wage block costs the first few
        # quarters and nothing else.
        ulc_q = sources.take(get_ulc_growth_qrtly(), "ULC growth", key="ulc")
        columns["ulc"] = ulc_q.rolling(4).sum()
        columns["speed"] = sources.take(
            get_unemployment_speed_limit_qrtly(), "unemployment speed limit", key="speed",
        )

    frame = pd.DataFrame(columns).dropna()

    frame = _attach_optional(frame, u, config, sources)

    if config.start is not None:
        frame = frame.loc[frame.index >= pd.Period(config.start, freq="Q")]
    if config.end is not None:
        frame = frame.loc[frame.index <= pd.Period(config.end, freq="Q")]
    if frame.empty:
        raise ValueError("no overlapping quarters of inflation and unemployment")

    index = frame.index
    if not isinstance(index, pd.PeriodIndex):
        raise TypeError("every series must carry a quarterly PeriodIndex")

    regimes = regime_index(index, config.breaks)
    n_regimes = len(config.breaks) + 1
    counts = np.bincount(regimes, minlength=n_regimes)
    if (counts == 0).any():
        empty = [config.breaks[k - 1] if k else "start" for k in np.flatnonzero(counts == 0)]
        raise ValueError(f"these regimes contain no quarters of the sample: {empty}")

    return frame, regimes, regime_labels(index, regimes, n_regimes), sources
