"""Components (expenditure-identity) GDP nowcast — run the day before GDP.

Nowcasts quarter-on-quarter GDP growth as the sum of expenditure components'
contributions to growth, each read from its own source release the day before
the National Accounts (T-0):

    GDP growth (ppt) =  Household consumption  +  Government consumption
                     +  Investment (GFCF, private + public)
                     +  Inventories  +  Net exports   [ + statistical discrepancy ]

Four pieces are *accounting-exact* — real $m CVM levels that map straight onto
the GDP identity (government consumption & GFCF from GFS Table 15; inventories
from 5676.0; net exports from 5302.0). Two are *bridged* from indicators that
only partially cover their NA aggregate (household consumption from the 5682.0
CVM Quarterly Household Spending index; private GFCF from 5625.0 capex + 8755.0
construction) via a small OLS fitted on the ABS-published contribution history.
The statistical discrepancy —
the residual that makes the components sum exactly to headline GDP — is unknown
at T-0; it is zero in the central nowcast and sized into the uncertainty band.

T-0 is the run timing — the complete information set the day before GDP is
published, matching the sibling nowcast models' month-indexed cycle (T-3m … T-0).
Inside the contribution formulas, T instead indexes the *target quarter* and T-1
its predecessor (e.g. ``GDP_{T-1}`` is the last published quarter).

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

from src.models.common.nowcast_charts import NowcastChartSpec, plot_nowcast_charts
from src.models.common.nowcast_core import compute_tty, detect_target_quarter
from src.models.gdp_nowcast_components import data as cd

logger = logging.getLogger(__name__)

CHART_DIR = "./charts/GDP-Nowcast-Components/"
CHART_START = "2022Q2"
SHOW = False

# Chart stack order, names, and colours. GFCF is split into private (bridged,
# the noisy AI-capex-driven piece) and public (GFS, accounting-exact) so their
# very different T-0 reliability is legible.
_STACK = [
    ("Household consumption", "skyblue"),
    ("Government consumption", "lightgreen"),
    ("Private investment", "lightcoral"),
    ("Public investment", "firebrick"),
    ("Inventories", "orchid"),
    ("Net exports", "gold"),
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
    band_70: float                   # symmetric ±band on the Q/Q nowcast (ppt)
    band_90: float
    gdp_tty: float                   # through-the-year (annual) growth implied by the Q/Q nowcast
    gdp_qoq_70: tuple[float, float]  # Q/Q 70% interval (gdp_qoq ∓ band_70)
    gdp_qoq_90: tuple[float, float]  # Q/Q 90% interval
    gdp_tty_70: tuple[float, float]  # annual 70% interval (Q/Q bounds rolled through compute_tty)
    gdp_tty_90: tuple[float, float]  # annual 90% interval
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
    target = target_quarter if target_quarter is not None else detect_target_quarter(gdp)

    # Inventories and net exports take the re-referencing guard (down-weighted
    # onto the GDP vintage's basis when the September straddle is active, else
    # factor 1). Government levels come from GFS on their own basis, so the guard
    # doesn't apply to them.
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
    hh_hist = cd.month_tag(target) if live_vintage else cd.month_tag(detect_target_quarter(cd.gdp_level()))
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
    band_70, band_90 = 1.04 * sigma, 1.645 * sigma

    # Re-express the central Q/Q nowcast and its symmetric band as annual (TTY)
    # growth, using the same Q/Q→annual roll-forward as the sibling models.
    tty = compute_tty(gdp_qoq, asof.gdp, asof.target)
    qoq_70 = (gdp_qoq - band_70, gdp_qoq + band_70)
    qoq_90 = (gdp_qoq - band_90, gdp_qoq + band_90)
    tty_70 = (compute_tty(qoq_70[0], asof.gdp, asof.target),
              compute_tty(qoq_70[1], asof.gdp, asof.target))
    tty_90 = (compute_tty(qoq_90[0], asof.gdp, asof.target),
              compute_tty(qoq_90[1], asof.gdp, asof.target))

    return NowcastResult(
        target=asof.target,
        contributions=contributions,
        gdp_qoq=gdp_qoq,
        private_gfcf=priv_gfcf,
        public_gfcf=pub_gfcf,
        disc_sigma=sigma,
        band_70=band_70,
        band_90=band_90,
        gdp_tty=tty,
        gdp_qoq_70=qoq_70,
        gdp_qoq_90=qoq_90,
        gdp_tty_70=tty_70,
        gdp_tty_90=tty_90,
        pub=asof.pub,
    )


# --- Output: text + chart ------------------------------------------------------


def _pending_components(result: NowcastResult) -> list[str]:
    """Identity components not yet released for the nowcast (statistical discrepancy excluded)."""
    return [name for name, _ in _STACK
            if name != "Statistical discrepancy" and np.isnan(result.contributions[name])]


def _pending_header(result: NowcastResult) -> str:
    """Chart lheader listing the components still awaiting release."""
    pending = _pending_components(result)
    return f"Still pending: {', '.join(pending)}" if pending else "All component sources in"


def print_summary(result: NowcastResult) -> None:
    """Print the per-component contributions and the summed nowcast."""
    t = result.target
    pending = _pending_components(result)
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
    print(f"  {'GDP growth Q/Q (nowcast)':<26} {result.gdp_qoq:+6.2f} %   [{status}]")
    print(f"  {'70% band':<26} ±{result.band_70:.2f} ppt   90% band ±{result.band_90:.2f} ppt")
    print(f"  {'GDP growth TTY (annual)':<26} {result.gdp_tty:+6.2f} %")
    print(f"  {'70% interval':<26} [{result.gdp_tty_70[0]:+.2f}%, {result.gdp_tty_70[1]:+.2f}%]   "
          f"90% [{result.gdp_tty_90[0]:+.2f}%, {result.gdp_tty_90[1]:+.2f}%]")
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
        title=f"Contributions to Quarterly GDP Growth — Nowcast {result.target}",
        ylabel="Percentage points (q/q)",
        y0=True,
        legend={"loc": "best", "fontsize": 8, "ncol": 4},
        axvspan={"xmin": forecast_pos - 0.5, "xmax": forecast_pos + 0.5,
                 "color": "goldenrod", "alpha": 0.25, "zorder": 0},
        lheader=_pending_header(result),
        rheader=f"Statistical discrepancy zeroed in nowcast, σ={result.disc_sigma:.2f}ppt",
        rfooter="ABS 5206.0/5519.0/5676.0/5302.0/5682.0/5625.0/8755.0",
        lfooter=(
            f"Australia. SA. CVM. Final bar = T-0 components nowcast for {result.target} "
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


_AVG_WINDOW_YEARS = 30  # rolling look-back window for one "normal" benchmark
_PRE_COVID = (pd.Period("2000Q1", "Q-DEC"), pd.Period("2019Q4", "Q-DEC"))  # the other "normal"


def _split_label(label: str) -> str:
    """Break a label onto two lines at the space nearest its middle (for axis ticks)."""
    spaces = [i for i, ch in enumerate(label) if ch == " "]
    if not spaces:
        return label
    i = min(spaces, key=lambda j: abs(j - len(label) / 2))
    return f"{label[:i]}\n{label[i + 1:]}"


def plot_average_vs_nowcast(result: NowcastResult) -> None:
    """Horizontal stacked bars: two definitions of "normal" vs the nowcast.

    Three bars, top to bottom — the mean published contribution of each component
    over (1) the last ``_AVG_WINDOW_YEARS`` years and (2) the pre-COVID 2000–2019
    window, then (3) the nowcast quarter. Same stack colours as the contributions
    chart; a black dot marks each bar's net (GDP growth). mgplot has no horizontal
    stacked bar, so this layers matplotlib ``barh`` and finalises through mgplot.
    """
    import matplotlib.pyplot as plt  # noqa: PLC0415

    pub = result.pub
    n = _AVG_WINDOW_YEARS * 4
    identity = list(_COMPONENT_PUB.values())
    pre_covid = (pub.index >= _PRE_COVID[0]) & (pub.index <= _PRE_COVID[1])

    def window_avg(frame: pd.DataFrame) -> dict[str, float]:
        """Mean contribution per component over the rows of ``frame``."""
        out = {name: float(frame[col].mean()) for name, col in _COMPONENT_PUB.items()}
        out["Statistical discrepancy"] = float((frame["gdp"] - frame[identity].sum(axis=1)).mean())
        return out

    avg_30 = window_avg(pub.tail(n))
    avg_pc = window_avg(pub[pre_covid])
    now = {name: (0.0 if np.isnan(result.contributions[name]) else result.contributions[name])
           for name, _ in _STACK}
    bars = [
        (f"{_AVG_WINDOW_YEARS}-year average", avg_30),
        (f"{_PRE_COVID[0].year}–{_PRE_COVID[1].year} average", avg_pc),
        (f"Nowcast {result.target}", now),
    ]

    _fig, ax = plt.subplots(figsize=(9, 5.5))
    ypos = list(range(len(bars) - 1, -1, -1))  # first bar on top
    top = ypos[0]
    data_min = data_max = 0.0
    for y, (_label, comps) in zip(ypos, bars, strict=True):
        pos = neg = 0.0
        for name, colour in _STACK:
            v = comps[name]
            left = pos if v >= 0 else neg
            ax.barh(y, v, height=0.6, left=left, color=colour, zorder=2,
                    label=name if y == top else None)
            pos, neg = (pos + v, neg) if v >= 0 else (pos, neg + v)
        data_min, data_max = min(data_min, neg), max(data_max, pos)
        ax.plot(sum(comps.values()), y, "o", color="black", markersize=6, zorder=3,
                label="GDP growth" if y == top else None)

    ax.set_yticks(ypos)
    ax.set_yticklabels([_split_label(label) for label, _ in bars])
    pad = 0.01 * (data_max - data_min)  # 1% breathing room at each end

    mg.finalise_plot(
        ax,
        title=f"Contributions to GDP Growth: Two Benchmarks vs Nowcast {result.target}",
        xlabel="Percentage points (q/q)",
        ylim=(-0.5, top + 1.1),  # reserve a band above the top bar for the legend (inside axes)
        xlim=(data_min - pad, data_max + pad),
        axvline={"x": 0, "color": "black", "linewidth": 0.6},
        axhspan={"ymin": ypos[-1] - 0.5, "ymax": ypos[-1] + 0.5,  # goldenrod backdrop on the nowcast row
                 "color": "goldenrod", "alpha": 0.25, "zorder": 0},
        legend={"loc": "upper center", "ncol": 4, "fontsize": 8},
        lheader=_pending_header(result),
        rheader=f"Statistical discrepancy zeroed in nowcast, σ={result.disc_sigma:.2f}ppt",
        rfooter="ABS 5206.0 + component sources",
        lfooter=(f"Australia. SA. CVM. Averages = mean published contribution "
                 f"(30-year rolling; {_PRE_COVID[0].year}–{_PRE_COVID[1].year}). Dot = GDP growth. "),
        show=SHOW,
    )


def plot_input_distributions(result: NowcastResult) -> None:
    """Boxplots of each input's published-contribution distribution vs the nowcast.

    One horizontal box per stack component (statistical discrepancy included),
    summarising the published-contribution distribution over the same trailing
    ``_AVG_WINDOW_YEARS``-year window as the benchmark chart's 30-year average bar
    — a contemporaneous economy, free of the high-inflation 1970s–80s volatility.
    A diamond overlays the current nowcast value so it reads at a glance whether
    each input sits inside or outside its normal range. COVID-era extremes fall
    outside the whiskers and are suppressed, keeping the boxes robust. Same stack
    colours, headers and footers as the contributions charts. mgplot has no
    boxplot, so this layers matplotlib and finalises through mgplot.
    """
    import matplotlib.pyplot as plt  # noqa: PLC0415

    pub = result.pub.tail(_AVG_WINDOW_YEARS * 4)  # trailing 30 years, matching the benchmark bar
    identity = list(_COMPONENT_PUB.values())

    # Published-contribution distribution per component over the window, with the
    # statistical discrepancy as published GDP minus the identity stack.
    history = {name: pub[col].dropna() for name, col in _COMPONENT_PUB.items()}
    history["Statistical discrepancy"] = (pub["gdp"] - pub[identity].sum(axis=1)).dropna()

    # Central nowcast value per component (discrepancy is zeroed in the nowcast).
    now = {name: (0.0 if np.isnan(result.contributions[name]) else result.contributions[name])
           for name, _ in _STACK}

    names = [name for name, _ in _STACK]
    colours = dict(_STACK)
    ypos = list(range(len(names) - 1, -1, -1))  # first component on top

    _fig, ax = plt.subplots(figsize=(9, 5.5))
    bp = ax.boxplot(
        [history[name].to_numpy() for name in names],
        positions=ypos, orientation="horizontal", widths=0.6,
        patch_artist=True, showfliers=False, zorder=2,
    )
    for patch, name in zip(bp["boxes"], names, strict=True):
        patch.set_facecolor(colours[name])
        patch.set_alpha(0.75)
    for median in bp["medians"]:
        median.set_color("black")

    for y, name in zip(ypos, names, strict=True):
        ax.plot(now[name], y, marker="D", color="black", markeredgecolor="white",
                markersize=8, linestyle="none", zorder=4,
                label=f"Nowcast {result.target}" if y == ypos[0] else None)

    ax.set_yticks(ypos)
    ax.set_yticklabels([_split_label(name) for name in names])

    mg.finalise_plot(
        ax,
        title=f"Input Distribution Boxplots vs Nowcast {result.target}",
        xlabel="Percentage points (q/q)",
        axvline={"x": 0, "color": "black", "linewidth": 0.6},
        legend={"loc": "best", "fontsize": 8},
        lheader=_pending_header(result),
        rheader=f"Statistical discrepancy zeroed in nowcast, σ={result.disc_sigma:.2f}ppt",
        rfooter="ABS 5206.0 + sources",
        lfooter=(f"Australia. SA. CVM. Boxes = {_AVG_WINDOW_YEARS}-year published-contribution "
                 f"history (COVID extremes suppressed). Diamond = nowcast. "),
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
    plot_average_vs_nowcast(result)
    plot_input_distributions(result)
    plot_nowcast_charts(NowcastChartSpec(
        model_label="Components",
        target_quarter=result.target,
        gdp=cd.gdp_level(),
        gdp_qoq=result.gdp_qoq,
        gdp_tty=result.gdp_tty,
        gdp_qoq_70=result.gdp_qoq_70,
        gdp_qoq_90=result.gdp_qoq_90,
        gdp_tty_70=result.gdp_tty_70,
        gdp_tty_90=result.gdp_tty_90,
        accent="goldenrod",
        show=SHOW,
    ))
    # Eight estimate-vs-NA scatter checks, drawn into the chart dir set above.
    from src.models.gdp_nowcast_components import diagnostics  # noqa: PLC0415
    diagnostics.plot_all_checks()
    return result


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    run_nowcast()
