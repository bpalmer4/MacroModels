"""Components (expenditure-identity) GDP nowcast — run the day before GDP.

Nowcasts quarter-on-quarter GDP growth as the sum of expenditure components'
contributions to growth, each read from its own source release the day before
the National Accounts (T-1):

    GDP growth (ppt) =  Household consumption  +  Government consumption
                     +  Investment (GFCF, private + public)
                     +  Inventories  +  Net exports   [ + statistical discrepancy ]

Four pieces are *accounting-exact* — real $m CVM levels that map straight onto
the GDP identity (government consumption & GFCF from GFS Table 15; inventories
from 5676.0; net exports from 5302.0). Two are *bridged* from indicators that
only partially cover their NA aggregate (household consumption from the 5682.0
CVM index; private GFCF from 5625.0 capex + 8755.0 construction) via a small OLS
fitted on the ABS-published contribution history. The statistical discrepancy —
the residual that makes the components sum exactly to headline GDP — is unknown
at T-1; it is zero in the central nowcast and sized into the uncertainty band.

The contribution math is a single as-of-parameterised path (:func:`_contribute`)
shared by the live run and the backtest, so there is exactly one place where a
component becomes a number.

Usage:
    uv run python -m src.models.gdp_nowcast_components.model

    from src.models.gdp_nowcast_components.model import nowcast
    result = nowcast(target_quarter=pd.Period("2025Q4", "Q-DEC"))
"""

import logging
from dataclasses import dataclass, field

import mgplot as mg
import numpy as np
import pandas as pd

from src.models.gdp_nowcast_components import data as cd

logger = logging.getLogger(__name__)

CHART_DIR = "./charts/GDP-Nowcast-Components/"
CHART_START = "2022Q2"
SHOW = False

# Chart stack order, names, and colours. GFCF is split into private (bridged,
# the noisy AI-capex-driven piece) and public (GFS, accounting-exact) so their
# very different T-1 reliability is legible.
_STACK = [
    ("Household consumption", "skyblue"),
    ("Government consumption", "lightgreen"),
    ("Private investment", "lightcoral"),
    ("Public investment", "firebrick"),
    ("Inventories", "orchid"),
    ("Net exports", "orange"),
    ("Statistical discrepancy", "silver"),
]

# Component (stack name) → published-contribution column in `pub`.
_COMPONENT_PUB = {
    "Household consumption": "household_consumption",
    "Government consumption": "government_consumption",
    "Private investment": "private_gfcf",
    "Public investment": "public_gfcf",
    "Inventories": "inventories",
    "Net exports": "net_exports",
}

# Number of recent quarters used to size the statistical-discrepancy band.
_BAND_WINDOW = 40
_MIN_BAND_OBS = 2          # need more than this to take a std
_DEFAULT_SIGMA = 0.2       # fallback discrepancy sigma (ppt) when history is too short

# COVID quarters excluded from the consumption bridge — the HSI→HFCE relationship
# broke down through the lockdowns (HFCE −12% vs a bridged −4%), so training on
# them contaminates the coefficient.
_COVID_START = pd.Period("2020Q1", "Q-DEC")
_COVID_END = pd.Period("2021Q2", "Q-DEC")
_MIN_CONS_OBS = 8          # minimum ex-COVID training quarters for the consumption bridge


# --- As-of information set -----------------------------------------------------


@dataclass
class AsOf:
    """Everything visible the day before GDP for ``target`` is published.

    ``gdp`` runs through ``target - 1`` (last published quarter); every component
    level runs through ``target`` (its source is out a day early); ``pub`` (the
    ABS-published contributions used for bridge fitting and the chart history)
    runs through ``target - 1``.
    """

    target: pd.Period
    gdp: pd.Series
    gov_c: pd.Series
    gov_gfcf: pd.Series
    inv: pd.Series
    nx: pd.Series
    hh: pd.Series          # 5682 Household Spending Indicator (CVM index) — the predictor
    hfce: pd.Series        # NA household final consumption (CVM $m) — bridge target & prior level
    capex: pd.Series
    constr: pd.Series
    pub: pd.DataFrame


@dataclass
class NowcastResult:
    """Output of a single components nowcast."""

    target: pd.Period
    contributions: dict[str, float]  # the six stacked identity components (ppt)
    gdp_qoq: float                   # sum of identity components (central nowcast)
    private_gfcf: float              # GFCF split (private), ppt
    public_gfcf: float               # GFCF split (public), ppt
    disc_sigma: float                # std of the statistical-discrepancy contribution (ppt)
    band_70: float
    band_90: float
    pub: pd.DataFrame = field(repr=False, default_factory=pd.DataFrame)


# --- Contribution primitives (shared) -----------------------------------------


def _level_contribution(level: pd.Series, target: pd.Period, gdp_tm1: float) -> float:
    """(level_T - level_{T-1}) / GDP_{T-1} * 100 — for flow components (C, G, NX)."""
    prev = target - 1
    if target in level.index and prev in level.index:
        return float((level[target] - level[prev]) / gdp_tm1 * 100)
    return float("nan")


def _inventory_contribution(level: pd.Series, target: pd.Period, gdp_tm1: float) -> float:
    """(Δlevel_T - Δlevel_{T-1}) / GDP_{T-1} * 100 — inventories enter GDP as a flow."""
    flow = level.diff()
    prev = target - 1
    if target in flow.index and prev in flow.index:
        return float((flow[target] - flow[prev]) / gdp_tm1 * 100)
    return float("nan")


def _growth(level: pd.Series) -> pd.Series:
    """100 × log-difference of a CVM level (a quarterly growth rate in %)."""
    return (np.log(level) * 100).diff()


def _bridge_predict(y: pd.Series, proxies: dict[str, pd.Series], target: pd.Period) -> float:
    """Predict a component's contribution at ``target`` from indicator growth.

    Fits ``y ~ 1 + Σ proxies`` by OLS over the overlap where ``y`` is published
    (≤ target-1 by construction) and predicts at ``target`` using the proxies'
    target-quarter growth. Returns NaN if the target proxy values or the fit are
    unavailable.
    """
    names = list(proxies)
    x_target = [proxies[n].get(target, np.nan) for n in names]
    if any(np.isnan(x_target)):
        return float("nan")

    df = pd.concat([y.rename("y"), *[proxies[n].rename(n) for n in names]], axis=1).dropna()
    df = df.loc[df.index < target]
    if len(df) < len(names) + 5:  # need a usable training window
        return float("nan")

    x_mat = np.column_stack([np.ones(len(df)), *[df[n].to_numpy() for n in names]])
    coef, *_ = np.linalg.lstsq(x_mat, df["y"].to_numpy(), rcond=None)
    return float(coef[0] + np.dot(coef[1:], x_target))


def _consumption_contribution(asof: AsOf, gdp_tm1: float) -> float:
    """Consumption contribution via the exact level path (not a contribution regression).

    Fit HFCE *growth* on HSI growth over the ex-COVID history before ``target``,
    predict the target-quarter HFCE growth from the published HSI, roll the last
    HFCE level forward by it, and take ``ΔHFCE / GDP_{t-1} × 100`` — the same
    formula the accounting-exact components use. This avoids regressing the
    (rounded, average-share-laden) published contribution directly.
    """
    t = asof.target
    hsi_g = _growth(asof.hh)
    x_t = hsi_g.get(t, np.nan)
    prev = t - 1
    if np.isnan(x_t) or prev not in asof.hfce.index:
        return float("nan")

    df = pd.concat([_growth(asof.hfce).rename("y"), hsi_g.rename("x")], axis=1).dropna()
    df = df.loc[df.index < t]
    df = df.loc[(df.index < _COVID_START) | (df.index > _COVID_END)]  # ex-COVID training
    if len(df) < _MIN_CONS_OBS:
        return float("nan")

    b1, b0 = np.polyfit(df["x"].to_numpy(), df["y"].to_numpy(), 1)
    pred_growth = b0 + b1 * x_t
    hfce_prev = float(asof.hfce[prev])
    d_hfce = hfce_prev * (np.exp(pred_growth / 100) - 1)  # predicted ΔHFCE in CVM $m
    return d_hfce / gdp_tm1 * 100


def _contribute(asof: AsOf) -> tuple[dict[str, float], float, float]:
    """Turn an as-of information set into the six stacked contributions.

    Returns ``(contributions, private_gfcf, public_gfcf)`` where ``contributions``
    holds the six stacked identity components (ppt) with the statistical
    discrepancy set to zero (the central nowcast).
    """
    t = asof.target
    prev = t - 1
    gdp_tm1 = float(asof.gdp[prev]) if prev in asof.gdp.index else float(asof.gdp.iloc[-1])

    # Accounting-exact components (direct from real $m levels).
    gov_c = _level_contribution(asof.gov_c, t, gdp_tm1)
    gov_gfcf = _level_contribution(asof.gov_gfcf, t, gdp_tm1)
    inv = _inventory_contribution(asof.inv, t, gdp_tm1)
    nx = _level_contribution(asof.nx, t, gdp_tm1)

    # Consumption: exact level path — predict HFCE growth from the HSI, then Δlevel/GDP_lag.
    hh = _consumption_contribution(asof, gdp_tm1)
    # Private GFCF: still a contribution bridge (capex + construction are partial proxies).
    priv_gfcf = _bridge_predict(
        asof.pub["private_gfcf"],
        {"capex": _growth(asof.capex), "constr": _growth(asof.constr)},
        t,
    )

    contributions = {
        "Household consumption": hh,
        "Government consumption": gov_c,
        "Private investment": priv_gfcf,
        "Public investment": gov_gfcf,
        "Inventories": inv,
        "Net exports": nx,
        "Statistical discrepancy": 0.0,
    }
    return contributions, priv_gfcf, gov_gfcf


def _nan_sum(*xs: float) -> float:
    """Sum treating NaN as 0, but NaN if every term is NaN."""
    vals = [x for x in xs if not np.isnan(x)]
    return float(sum(vals)) if vals else float("nan")


# --- As-of builders ------------------------------------------------------------


def _build_live(target_quarter: pd.Period | None) -> AsOf:
    """Assemble the live (latest-vintage) as-of set, with the re-referencing guard."""
    gdp = cd.gdp_level()
    target = target_quarter if target_quarter is not None else gdp.index[-1] + 1

    # Accounting-exact source levels, each down-weighted onto the GDP vintage's
    # basis if the September re-referencing straddle is active (else factor 1).
    return AsOf(
        target=target,
        gdp=gdp,
        gov_c=cd.government_consumption_level(),
        gov_gfcf=cd.government_gfcf_level(),
        inv=cd.inventories_reref(gdp),
        nx=cd.net_exports_reref(gdp),
        hh=cd.household_spending_cvm_level(cd.month_tag(target)),
        hfce=cd.household_consumption_level(),
        capex=cd.private_capex_level(),
        constr=cd.private_construction_level(),
        pub=cd.published_contributions(),
    )


def build_asof(target: pd.Period, *, live_vintage: bool = False) -> AsOf:
    """Assemble a truncated, single-vintage as-of set for ``target`` (backtest path).

    GDP is truncated to ``target - 1`` and every component to ``target``; the
    published contributions to ``target - 1``. No re-referencing guard (single
    vintage shares a reference basis). Set ``live_vintage`` only when you want the
    point-in-time household snapshot fetched via ``history=`` for ``target``.
    """
    prev = target - 1
    hh_hist = cd.month_tag(target) if live_vintage else cd.month_tag(cd.gdp_level().index[-1] + 1)
    hh = cd.household_spending_cvm_level(hh_hist)
    return AsOf(
        target=target,
        gdp=cd.gdp_level().loc[:prev],
        gov_c=cd.government_consumption_level().loc[:target],
        gov_gfcf=cd.government_gfcf_level().loc[:target],
        inv=cd.inventories_level().loc[:target],
        nx=cd.net_exports_level().loc[:target],
        hh=hh.loc[:target] if len(hh) else hh,
        hfce=cd.household_consumption_level().loc[:prev],
        capex=cd.private_capex_level().loc[:target],
        constr=cd.private_construction_level().loc[:target],
        pub=cd.published_contributions().loc[:prev],
    )


# --- Uncertainty band ----------------------------------------------------------


def _discrepancy_sigma(pub: pd.DataFrame) -> float:
    """Std (ppt) of the recent statistical-discrepancy contribution.

    The discrepancy = published GDP growth − sum of the five stacked expenditure
    contributions. Its recent standard deviation is the irreducible part of the
    nowcast error: GDP is the average of the I/E/P measures, so no expenditure
    build-up can be more precise than this (bridge error is measured separately
    by the backtest).
    """
    five = pub[["household_consumption", "government_consumption", "investment_gfcf",
                "inventories", "net_exports"]].sum(axis=1)
    disc = (pub["gdp"] - five).dropna().tail(_BAND_WINDOW)
    return float(disc.std()) if len(disc) > _MIN_BAND_OBS else _DEFAULT_SIGMA


# --- Public API ----------------------------------------------------------------


def nowcast(target_quarter: pd.Period | None = None) -> NowcastResult:
    """Run the components nowcast for ``target_quarter`` (default: next unpublished)."""
    asof = _build_live(target_quarter)
    contributions, priv_gfcf, pub_gfcf = _contribute(asof)
    gdp_qoq = _nan_sum(*(v for k, v in contributions.items() if k != "Statistical discrepancy"))
    sigma = _discrepancy_sigma(asof.pub)
    return NowcastResult(
        target=asof.target,
        contributions=contributions,
        gdp_qoq=gdp_qoq,
        private_gfcf=priv_gfcf,
        public_gfcf=pub_gfcf,
        disc_sigma=sigma,
        band_70=1.04 * sigma,
        band_90=1.645 * sigma,
        pub=asof.pub,
    )


# --- Output: text + chart ------------------------------------------------------


def print_summary(result: NowcastResult) -> None:
    """Print the per-component contributions and the summed nowcast."""
    t = result.target
    pending = [k for k, v in result.contributions.items()
               if k != "Statistical discrepancy" and np.isnan(v)]
    print(f"\nComponents GDP nowcast — {t} (q/q, CVM, seasonally adjusted)")
    print("=" * 64)
    for name, _ in _STACK:
        if name == "Statistical discrepancy":
            continue
        val = result.contributions[name]
        shown = f"{val:+6.2f} ppt" if not np.isnan(val) else "   pending release"
        print(f"  {name:<26} {shown}")
    print("-" * 64)
    status = "PARTIAL — sources pending" if pending else "complete (all sources in)"
    print(f"  {'GDP growth (nowcast)':<26} {result.gdp_qoq:+6.2f} %   [{status}]")
    print(f"  {'70% band':<26} ±{result.band_70:.2f} ppt   90% band ±{result.band_90:.2f} ppt")
    if pending:
        print(f"  Not yet released: {', '.join(pending)}.")
    print("  Statistical discrepancy assumed 0 in the central nowcast.\n")


def _chart_frame(result: NowcastResult) -> tuple[pd.DataFrame, pd.Series]:
    """Build the stacked-contributions frame (history + nowcast bar) and GDP dots."""
    pub = result.pub
    identity = ["household_consumption", "government_consumption", "private_gfcf",
                "public_gfcf", "inventories", "net_exports"]
    hist = pd.DataFrame({
        "Household consumption": pub["household_consumption"],
        "Government consumption": pub["government_consumption"],
        "Private investment": pub["private_gfcf"],
        "Public investment": pub["public_gfcf"],
        "Inventories": pub["inventories"],
        "Net exports": pub["net_exports"],
        "Statistical discrepancy": pub["gdp"] - pub[identity].sum(axis=1),
    }).dropna()
    gdp_dots = pub["gdp"].reindex(hist.index)

    # Append the nowcast quarter. Pending (not-yet-released) components render as
    # zero-height segments; the PARTIAL status flag in the text carries the caveat.
    now_row = {name: (0.0 if np.isnan(result.contributions[name]) else result.contributions[name])
               for name, _ in _STACK}
    comp = pd.concat([hist, pd.DataFrame(now_row, index=[result.target])])
    gdp_dots.loc[result.target] = result.gdp_qoq

    comp = comp.loc[CHART_START:]
    gdp_dots = gdp_dots.loc[CHART_START:]
    gdp_dots.name = "GDP growth"
    return comp, gdp_dots


def plot_contributions(result: NowcastResult) -> None:
    """Stacked contributions chart with the nowcast quarter appended."""
    comp, gdp_dots = _chart_frame(result)
    ax = mg.bar_plot(comp, stacked=True, color=[c for _, c in _STACK])
    mg.line_plot(gdp_dots, ax=ax, width=0, marker="o", markersize=5, color="black", annotate=False)
    forecast_pos = result.target.ordinal  # mgplot positions bars by Period ordinal
    mg.finalise_plot(
        ax,
        title="Contributions to Quarterly GDP Growth",
        ylabel="Percentage points (q/q)",
        y0=True,
        legend={"loc": "best", "fontsize": 8, "ncol": 4},
        axvspan={"xmin": forecast_pos - 0.5, "xmax": forecast_pos + 0.5,
                 "color": "goldenrod", "alpha": 0.25, "zorder": 0},
        rheader=f"Statistical discrepancy zeroed in nowcast, σ={result.disc_sigma:.2f}ppt",
        rfooter="ABS 5206.0/5519.0/5676.0/5302.0/5682.0/5625.0/8755.0",
        lfooter=(
            f"Australia. SA. CVM. Final bar = T-1 components nowcast for {result.target} "
            f"({result.gdp_qoq:+.2f}%). Dots = GDP growth. "
        ),
        show=SHOW,
    )


def plot_component_contributions(result: NowcastResult) -> None:
    """One bar chart per component: 5206 published history + nowcast as the final bar.

    History runs from CHART_START (same window as the combined chart); the final
    bar is the model's nowcast for the target quarter, behind a goldenrod span.
    Each chart uses the component's stack colour.
    """
    pub = result.pub
    colours = dict(_STACK)
    forecast_pos = result.target.ordinal
    span = {"xmin": forecast_pos - 0.5, "xmax": forecast_pos + 0.5,
            "color": "goldenrod", "alpha": 0.25, "zorder": 0}
    for name, col in _COMPONENT_PUB.items():
        series = pub[col].loc[CHART_START:].dropna().copy()
        now = result.contributions[name]
        series.loc[result.target] = 0.0 if np.isnan(now) else now
        series.name = name
        mg.bar_plot_finalise(
            series,
            title=f"{name} Contribution to GDP Growth",
            ylabel="Percentage points (q/q)",
            color=colours[name],
            annotate=True, rounding=1, y0=True, axvspan=span,
            rfooter="ABS 5206.0 + source",
            lfooter=(f"Australia. SA. CVM. History = 5206 published; "
                     f"final bar = nowcast for {result.target}. "),
            show=SHOW,
        )


def run_nowcast() -> NowcastResult:
    """Live entry point: nowcast the next quarter, print the table, draw the charts."""
    result = nowcast()
    print_summary(result)
    mg.set_chart_dir(CHART_DIR)
    mg.clear_chart_dir()
    plot_contributions(result)
    plot_component_contributions(result)
    return result


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    run_nowcast()
