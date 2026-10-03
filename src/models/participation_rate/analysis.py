"""Event study and local projections of labour market outcomes on RBA decisions.

Outcome throughout: Y(t+h) - Y(t-1), the change from the month before a
decision to h months after it, for Y the participation rate (the question
asked) or the unemployment rate (the cycle it is tested against).
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.regression.linear_model import RegressionResultsWrapper

from src.models.participation_rate import AGE_BANDS, BASELINE_EXCLUDE, BASELINE_START, COVID, HMAX, PRE
from src.models.participation_rate.data import CONTROLS

RAW_HORIZONS = [0, 3, 6, 12, 18]  # horizons in the raw-means table
LP_HORIZONS = [0, 3, 6, 9, 12, 15, 18]  # horizons printed from the projections
DOSE_HORIZONS = [12, 18]
ERA_HORIZONS = [6, 12, 18]
DOSE_WINDOW = 12  # months of cash rate moves summed for the dose test
CYCLE_WINDOW = 12  # months either side of a first hike in the cycle table

# Household debt rose a long way relative to income through the 1990s and
# early 2000s, so a budget channel should be stronger after this split.
ERA_SPLIT = pd.Period("2005-01", "M")

# Controls for the pre-trend test, by outcome: the outcome's own lags are
# dropped because the pre-trend IS the lagged change in the outcome.
PRE_TREND_CONTROLS = {
    "pr": ["dur12_lag", "demp12_lag", "ur_lag", "dcr12_lag"],
    "ur": ["dpr12_lag", "demp12_lag", "dcr12_lag"],
}


def outcome(d: pd.DataFrame, months: pd.PeriodIndex, col: str, h: int) -> pd.Series:
    """Y(t+h) - Y(t-1) for Y = d[col] at each sample month; NaN where the window touches COVID."""
    y = d[col].shift(-h) - d[col].shift(1)
    idx = d.index
    if not isinstance(idx, pd.PeriodIndex):
        msg = "frame must have a PeriodIndex"
        raise TypeError(msg)
    touches = (idx.shift(h) >= COVID[0]) & (idx.shift(-1) <= COVID[1])
    y[touches] = np.nan
    return y.loc[months]


def _ur_path(d: pd.DataFrame, months: pd.PeriodIndex, h: int) -> pd.Series:
    """Unemployment change over the same window as the outcome."""
    return (d["ur"].shift(-h) - d["ur"].shift(1)).loc[months]


def _fit(y: pd.Series, x: pd.DataFrame, maxlags: int) -> RegressionResultsWrapper:
    """OLS with a constant and Newey-West errors, on rows with no NaN."""
    x = sm.add_constant(x)
    ok = y.notna() & x.notna().all(axis=1)
    fitted = sm.OLS(y[ok], x[ok]).fit(cov_type="HAC", cov_kwds={"maxlags": maxlags})
    if not isinstance(fitted, RegressionResultsWrapper):
        msg = "unexpected statsmodels result type"
        raise TypeError(msg)
    return fitted


def raw_means(d: pd.DataFrame, months: pd.PeriodIndex, col: str) -> pd.DataFrame:
    """Mean outcome after each kind of decision, against all months."""
    rows = []
    for h in RAW_HORIZONS:
        y = outcome(d, months, col, h)
        row: dict[str, float] = {"h": h, "all months": y.mean()}
        for ev in ["hike", "first_hike", "cut", "first_cut"]:
            hit = d.loc[months, ev] == 1
            row[ev] = y[hit].mean()
            row[f"n_{ev}"] = int(y[hit].notna().sum())
        rows.append(row)
    return pd.DataFrame(rows).set_index("h")


def local_projection(
    d: pd.DataFrame,
    months: pd.PeriodIndex,
    col: str,
    events: tuple[str, str],
    *,
    ur_path: bool,
) -> pd.DataFrame:
    """Outcome on a (hike, cut) dummy pair and the controls, h = 0..HMAX.

    With ur_path, the unemployment change over the same window is also a
    regressor, so the hike coefficient is participation beyond what the
    unemployment cycle explains. That conditions on an outcome of the
    decision, so it is a decomposition, not a cleaner estimate.
    """
    hike, cut = events
    res = []
    for h in range(HMAX + 1):
        x = d.loc[months, [hike, cut, *CONTROLS]].copy()
        if ur_path:
            x["dur_path"] = _ur_path(d, months, h)
        f = _fit(outcome(d, months, col, h), x, h + 1)
        res.append(
            {
                "h": h,
                "hike_b": f.params[hike],
                "hike_se": f.bse[hike],
                "cut_b": f.params[cut],
                "cut_se": f.bse[cut],
                "n": int(f.nobs),
            }
        )
    out = pd.DataFrame(res).set_index("h")
    out["hike_t"] = out["hike_b"] / out["hike_se"]
    out["cut_t"] = out["cut_b"] / out["cut_se"]
    return out


def cycle_table(d: pd.DataFrame, months: pd.PeriodIndex) -> pd.DataFrame:
    """For each first hike: participation and unemployment either side of it."""
    w = CYCLE_WINDOW

    def change(col: str, p: pd.Period, start: int, end: int) -> float:
        return float(d[col].get(p + end, np.nan) - d[col].get(p + start, np.nan))

    rows = [
        {
            "first hike": str(p),
            f"PR {w}m before": change("pr", p, -w - 1, -1),
            f"PR {w}m after": change("pr", p, -1, w - 1),
            f"PR_f {w}m after": change("pr_f", p, -1, w - 1),
            f"PR_m {w}m after": change("pr_m", p, -1, w - 1),
            f"UR {w}m after": change("ur", p, -1, w - 1),
            f"cash rate {w}m after": change("cr", p, -1, w - 1),
        }
        for p in months[d.loc[months, "first_hike"] == 1]
    ]
    return pd.DataFrame(rows).set_index("first hike")


def dose_test(d: pd.DataFrame, months: pd.PeriodIndex) -> pd.DataFrame:
    """Outcome on the size of rate rises and falls over the next DOSE_WINDOW months.

    The forward cash rate path is itself driven by how the labour market
    evolves, so this is weaker evidence than its t-statistics suggest.
    """
    fwd = (d["cr"].shift(-(DOSE_WINDOW - 1)) - d["cr"].shift(1)).loc[months]
    rows = []
    for ur_path in (False, True):
        for h in DOSE_HORIZONS:
            x = d.loc[months, CONTROLS].copy()
            x["cr_up"] = fwd.clip(lower=0)
            x["cr_down"] = fwd.clip(upper=0)
            if ur_path:
                x["dur_path"] = _ur_path(d, months, h)
            f = _fit(outcome(d, months, "pr", h), x, h + 1)
            rows.append(
                {
                    "h": h,
                    "ur_path": ur_path,
                    "per_1pp_up": f.params["cr_up"],
                    "t_up": f.tvalues["cr_up"],
                    "per_1pp_down": f.params["cr_down"],
                    "t_down": f.tvalues["cr_down"],
                }
            )
    return pd.DataFrame(rows).set_index(["ur_path", "h"])


def era_split(d: pd.DataFrame, months: pd.PeriodIndex) -> pd.DataFrame:
    """Hike effect before and after ERA_SPLIT."""
    late = pd.Series((months >= ERA_SPLIT).astype(float), index=months)
    rows = []
    for h in ERA_HORIZONS:
        x = d.loc[months, ["cut", *CONTROLS]].copy()
        x["hike_early"] = d.loc[months, "hike"] * (1 - late)
        x["hike_late"] = d.loc[months, "hike"] * late
        x["late"] = late
        f = _fit(outcome(d, months, "pr", h), x, h + 1)
        rows.append(
            {
                "h": h,
                "early_b": f.params["hike_early"],
                "early_t": f.tvalues["hike_early"],
                "late_b": f.params["hike_late"],
                "late_t": f.tvalues["hike_late"],
            }
        )
    return pd.DataFrame(rows).set_index("h")


def pre_trend(d: pd.DataFrame, months: pd.PeriodIndex, col: str) -> pd.Series:
    """Test whether the outcome is already moving in the year before a first move."""
    y = (d[col].shift(1) - d[col].shift(PRE + 1)).loc[months]
    f = _fit(y, d.loc[months, ["first_hike", "first_cut", *PRE_TREND_CONTROLS[col]]], PRE + 1)
    return pd.Series(
        {
            "first_hike_b": f.params["first_hike"],
            "first_hike_t": f.tvalues["first_hike"],
            "first_cut_b": f.params["first_cut"],
            "first_cut_t": f.tvalues["first_cut"],
        }
    )


def event_paths(d: pd.DataFrame, months: pd.PeriodIndex, ev: str, col: str) -> pd.DataFrame:
    """d[col] from PRE months before to HMAX after each event, 0 at t-1.

    One column per event; months inside the COVID window are blanked.
    """
    rel = pd.RangeIndex(-PRE, HMAX + 1)
    cols = {}
    for p in months[d.loc[months, ev] == 1]:
        base = d[col].get(p - 1, np.nan)
        path = pd.Series([d[col].get(p + k, np.nan) - base for k in rel], index=rel)
        path[[COVID[0] <= p + k <= COVID[1] for k in rel]] = np.nan
        cols[str(p)] = path
    return pd.DataFrame(cols)


def drift(d: pd.DataFrame, months: pd.PeriodIndex, col: str) -> pd.Series:
    """Average the same rebased path over every sample month: the trend."""
    rel = pd.RangeIndex(-PRE, HMAX + 1)
    return pd.Series([(d[col].shift(-k) - d[col].shift(1)).loc[months].mean() for k in rel], index=rel)


def _age_band(age: str) -> str:
    """Map an LMS2 age label ('25-29 years', '65 years and over') to its AGE_BANDS band."""
    lower = int(age[:2])
    return max((b for b in AGE_BANDS.items() if b[1] <= lower), key=lambda b: b[1])[0]


def age_comparison(
    status: pd.DataFrame, base: pd.Period, end: pd.Period
) -> tuple[pd.DataFrame, pd.DataFrame, list[int]]:
    """Participation change base -> end by sex and age band, against normal years.

    status is labour force status indexed by (month, sex, age), in Original
    terms, so the change is compared with the change over the same calendar
    window in each normal year, which removes seasonality.

    Returns (change, contribution, normal_years). change is the base -> end
    participation change minus the normal-year average, by band x sex.
    contribution weights it by each band's share of that sex's population at
    end, with a Total row: an approximate decomposition of the sex's change
    that ignores shifts in the age mix.
    """
    banded = status.rename(index=_age_band, level="age").groupby(level=["month", "sex", "age"]).sum()
    lf = banded[["eft", "ept", "uft", "upt"]].sum(axis=1)
    pop = lf + banded["nilf"]
    pr = 100 * lf / pop

    span = end.year - base.year

    def change(end_year: int) -> pd.Series:
        to = pd.Period(year=end_year, month=end.month, freq="M")
        frm = pd.Period(year=end_year - span, month=base.month, freq="M")
        return pr.xs(to, level="month") - pr.xs(frm, level="month")

    def by_sex(s: pd.Series) -> pd.DataFrame:
        """(sex, age) Series -> age x sex frame."""
        return pd.DataFrame({sex: s.xs(sex, level="sex") for sex in s.index.unique(level="sex")})

    normal_years = [y for y in range(BASELINE_START, end.year) if y not in BASELINE_EXCLUDE]
    normal = pd.concat([change(y) for y in normal_years]).groupby(level=["sex", "age"]).mean()
    diff = by_sex(change(end.year) - normal)

    end_pop = pop.xs(end, level="month")
    contribution = diff * by_sex(end_pop / end_pop.groupby(level="sex").transform("sum"))
    contribution.loc["Total"] = contribution.sum()
    return diff, contribution, normal_years
