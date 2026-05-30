"""Diagnostics for the components GDP nowcast — regenerable, never hand-rendered.

Six validation charts, one per component, each of which should sit on the 1:1:

  * Bridged series (household, private investment): the **bridge-adjusted**
    indicator vs NA — by the OLS calibration identity it lands on the 1:1
    (slope 1.00) on the ex-COVID sample it was fit on.
  * Exact series (government, public, inventories, net exports): the **source**
    vs NA — slope ~1 because the source *is* the NA input.

Everything is a change measure: growth (% q/q) for the strictly-positive levels;
$m quarterly change for the flows that cross zero (inventories, net exports),
where a growth rate is undefined. Estimate on x, National Accounts on y.

Run fresh:
    uv run python -m src.models.gdp_nowcast_components.diagnostics
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import mgplot as mg
import numpy as np
import pandas as pd
import readabs as ra
from readabs import metacol as mc

from src.models.gdp_nowcast_components import data as cd
from src.models.gdp_nowcast_components.model import CHART_DIR

if TYPE_CHECKING:
    from collections.abc import Callable

logger = logging.getLogger(__name__)

COVID = (pd.Period("2020Q1", "Q-DEC"), pd.Period("2021Q2", "Q-DEC"))
_MIN_FIT_OBS = 2


def _growth(level: pd.Series) -> pd.Series:
    """Log-difference growth (% q/q) of a CVM level."""
    return (np.log(level) * 100).diff()


def _ex_covid(df: pd.DataFrame) -> pd.DataFrame:
    return df.loc[(df.index < COVID[0]) | (df.index > COVID[1])]


def _na_lookup() -> Callable[[str], pd.Series]:
    """Return a lookup(did) → CVM SA level series from the 5206 expenditure table."""
    tab = "5206002_Expenditure_Volume_Measures"
    d, m = ra.read_abs_cat("5206.0", single_excel_only=tab, verbose=False)
    m = m[m[mc.table] == tab]

    def lookup(did: str) -> pd.Series:
        r = m[(m[mc.did] == did) & (m[mc.stype] == "Seasonally Adjusted")].iloc[0]
        s = d[tab][r[mc.id]].dropna()
        s.index = pd.PeriodIndex(s.index, freq="Q-DEC")
        return s

    return lookup


def _na_inventory_changes() -> pd.Series:
    """NA private non-farm changes in inventories, CVM SA ($m flow), from 5206009."""
    tab = "5206009_Changes_In_Inventories"
    d, m = ra.read_abs_cat("5206.0", single_excel_only=tab, verbose=False)
    m = m[m[mc.table] == tab]
    r = m[(m[mc.did] == "Private ;  Non-farm ;  Chain volume measures ;")
          & (m[mc.stype] == "Seasonally Adjusted")].iloc[0]
    s = d[tab][r[mc.id]].dropna()
    s.index = pd.PeriodIndex(s.index, freq="Q-DEC")
    return s


def _scatter(title: str, estimate: pd.Series, na: pd.Series,
             xlabel: str, ylabel: str, note: str) -> tuple[float, float, int]:
    """Scatter estimate (x) vs NA (y); 45° = match. Returns ex-COVID (slope, R², n)."""
    both = pd.DataFrame({"x": estimate, "y": na}).dropna()
    nm = _ex_covid(both)
    cv = both.loc[(both.index >= COVID[0]) & (both.index <= COVID[1])]
    slope, intc = np.polyfit(nm["x"], nm["y"], 1) if len(nm) > _MIN_FIT_OBS else (np.nan, np.nan)
    r2 = nm["x"].corr(nm["y"]) ** 2 if len(nm) > _MIN_FIT_OBS else np.nan

    _fig, ax = plt.subplots()
    ax.scatter(nm["x"], nm["y"], s=20, color="cornflowerblue", alpha=0.7, label="Quarterly (ex-COVID)")
    if len(cv):
        ax.scatter(cv["x"], cv["y"], s=20, color="orange", alpha=0.8, label="COVID")
    lo, hi = both.min().min(), both.max().max()
    ax.plot([lo, hi], [lo, hi], color="darkred", lw=1, ls="--", label="45 deg (1:1)")
    xs = np.array([nm["x"].min(), nm["x"].max()])
    ax.plot(xs, slope * xs + intc, color="green", lw=1.2, label=f"ex-COVID fit (slope {slope:.2f})")
    mg.finalise_plot(
        ax, title=title, xlabel=xlabel, ylabel=ylabel,
        legend={"loc": "upper left", "fontsize": 8}, rfooter="ABS 5206.0 + source",
        lfooter=(f"Australia. SA, CVM. {note}. ex-COVID slope={slope:.2f}, R^2={r2:.2f}, n={len(nm)}. "),
        pre_tag="check", show=False,
    )
    return slope, r2, len(nm)


def plot_one_to_one_checks() -> pd.DataFrame:
    """Six per-component charts that should each sit on the 1:1."""
    na = _na_lookup()

    # Bridged: household — adjusted = fitted line of HFCE growth on HSI growth (ex-COVID).
    hsi = _growth(cd.household_spending_cvm_level())
    hfce_g = _growth(na("Households ;  Final consumption expenditure ;"))
    hh = _ex_covid(pd.DataFrame({"x": hsi, "y": hfce_g}).dropna())
    hb1, hb0 = np.polyfit(hh["x"], hh["y"], 1)
    hsi_adj = hb0 + hb1 * hsi

    # Bridged: private investment — adjusted = fitted line on capex + construction growth.
    capex, constr = _growth(cd.private_capex_level()), _growth(cd.private_construction_level())
    pgfcf_g = _growth(na("Private ;  Gross fixed capital formation ;"))
    pi = _ex_covid(pd.DataFrame({"c": capex, "k": constr, "y": pgfcf_g}).dropna())
    pc, *_ = np.linalg.lstsq(
        np.column_stack([np.ones(len(pi)), pi["c"], pi["k"]]), pi["y"].to_numpy(), rcond=None)
    pi_adj = pc[0] + pc[1] * capex + pc[2] * constr

    specs = [
        ("Household consumption — bridge-adjusted vs NA", hsi_adj, hfce_g,
         "Bridge-adjusted HSI growth (% q/q)", "HFCE growth, 5206 (% q/q)", "growth, ex-COVID bridge"),
        ("Private investment — bridge-adjusted vs NA", pi_adj, pgfcf_g,
         "Bridge-adjusted indicators (% q/q)", "Private GFCF growth, 5206 (% q/q)", "growth, ex-COVID bridge"),
        ("Government consumption — source vs NA", _growth(cd.government_consumption_level()),
         _growth(na("General government ;  Final consumption expenditure ;")),
         "GFS growth (% q/q)", "NA govt consumption growth (% q/q)", "growth, GFS source"),
        ("Public investment — source vs NA", _growth(cd.government_gfcf_level()),
         _growth(na("Public ;  Gross fixed capital formation ;")),
         "GFS growth (% q/q)", "NA public GFCF growth (% q/q)", "growth, GFS source"),
        ("Inventories — source vs NA", cd.inventories_level().diff(), _na_inventory_changes(),
         "5676 change in inventories ($m)", "NA change in inventories, PNF ($m)", "change, 5676 source"),
        ("Net exports — source vs NA", cd.net_exports_level().diff(),
         (na("Exports of goods and services ;") - na("Imports of goods and services ;")).diff(),
         "5302 change in net exports ($m)", "NA change in net exports ($m)", "change, 5302 source"),
    ]
    out = []
    for title, est, na_series, xlabel, ylabel, note in specs:
        slope, r2, n = _scatter(title, est, na_series, xlabel, ylabel, note)
        out.append({"chart": title, "slope": slope, "r2_ex_covid": r2, "n": n})
    return pd.DataFrame(out).set_index("chart")


def main() -> None:
    """Regenerate the six 1:1 check charts and print the fit table."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    mg.set_chart_dir(CHART_DIR)
    logger.info("One-to-one checks (estimate vs NA — should be slope ~1):\n%s",
                plot_one_to_one_checks().round(2).to_string())
    logger.info("\nCharts written to %s", CHART_DIR)


if __name__ == "__main__":
    main()
