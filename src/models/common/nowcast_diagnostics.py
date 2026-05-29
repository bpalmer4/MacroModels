"""Cross-model diagnostics for the GDP nowcast suite.

Provides post-hoc checks that any of the three nowcast models (bridge, DFM,
BVAR) can append to their output without changing how they estimate.

Currently:
    capex_imports_hotness() — quantifies the gap between the latest equipment
    capex surge and the goods imports that should be offsetting it under the
    GDP identity. Expressed as deviation from the long-run average using the
    BVAR's estimation frame (1997Q4 onwards) so the diagnostic is consistent
    with the cyclical history the models are trained on.

    Useful during import-funded investment cycles (e.g. AI / data-centre
    buildout) where the bridge-style models can't enforce the I↑/M↑
    cancellation that the national accounts will eventually impose.
"""

from dataclasses import dataclass

import pandas as pd
import readabs as ra
from readabs import metacol as mc

from src.data.gdp import get_gdp

# Use the BVAR estimation frame for the long-run average — spans mining boom,
# GFC, COVID, and aligns with the panel one of the models is fit on.
HISTORY_START = pd.Period("1997Q4", freq="Q-DEC")


@dataclass
class CapexImportsHotness:
    """Capex-vs-imports diagnostic for the latest available quarter."""

    quarter: pd.Period
    history_start: pd.Period
    n_history: int
    capex_qoq_pct_gdp_now: float
    capex_qoq_pct_gdp_mean: float
    capex_qoq_pct_gdp_std: float
    imports_qoq_pct_gdp_now: float
    imports_qoq_pct_gdp_mean: float
    imports_qoq_pct_gdp_std: float
    capex_deviation_pp: float
    imports_deviation_pp: float
    hotness_pp: float


def _equipment_capex_qrtly() -> pd.Series:
    """Total Equipment, Plant and Machinery capex (CVM, SA, all industries)."""
    data, _ = ra.read_abs_cat(
        "5625.0",
        single_excel_only="07_volume_measures_seasonally_adjusted_capex",
        verbose=False,
    )
    return data["07_volume_measures_seasonally_adjusted_capex"]["A124797536J"].dropna()


def _goods_imports_qrtly() -> pd.Series:
    """Total goods debits, SA, quarterly sum of monthly $M from 5368.0."""
    data, meta = ra.read_abs_cat("5368.0", single_excel_only="536801", verbose=False)
    m = meta[meta[mc.table] == "536801"]
    gd = m[(m[mc.did] == "Debits, Total goods ;") & (m[mc.stype] == "Seasonally Adjusted")]
    debits_monthly = data["536801"][gd[mc.id].iloc[0]]
    return ra.monthly_to_qtly(debits_monthly, q_ending="DEC", f="sum").abs().dropna()


def capex_imports_hotness() -> CapexImportsHotness:
    """Compute the capex-imports hotness diagnostic.

    For each historical quarter from HISTORY_START onward, compute the QoQ
    change in equipment capex and goods imports as a percentage of that
    quarter's GDP. The "hotness" is the extent to which the latest quarter's
    (capex deviation − imports deviation) is unusual relative to that
    historical pattern.

    Positive hotness ⇒ capex is surging by more than typical, *more* than
    goods imports are surging by more than typical. The nowcast bridges will
    see this as a positive GDP signal without the offsetting M absorbing it.
    """
    capex = _equipment_capex_qrtly()
    imports = _goods_imports_qrtly()
    gdp = get_gdp(gdp_type="CVM", seasonal="SA").data.dropna()

    # Capex and goods imports may run a quarter ahead of published GDP — use
    # the capex/imports overlap as the spine and forward-fill GDP for any
    # nowcast quarters beyond the latest GDP print.
    spine = capex.index.intersection(imports.index)
    capex = capex.loc[spine]
    imports = imports.loc[spine]
    gdp_aligned = gdp.reindex(spine).ffill()

    capex_pct = (capex.diff() / gdp_aligned) * 100
    imports_pct = (imports.diff() / gdp_aligned) * 100

    hist_mask = capex_pct.index >= HISTORY_START
    capex_hist = capex_pct.loc[hist_mask].dropna()
    imports_hist = imports_pct.loc[hist_mask].dropna()

    capex_now = capex_pct.iloc[-1]
    imports_now = imports_pct.iloc[-1]

    # Exclude the latest reading from the mean so we compare against the
    # past, not against a window that already includes the surprise.
    capex_mean = capex_hist.iloc[:-1].mean()
    capex_std = capex_hist.iloc[:-1].std()
    imports_mean = imports_hist.iloc[:-1].mean()
    imports_std = imports_hist.iloc[:-1].std()

    capex_dev = capex_now - capex_mean
    imports_dev = imports_now - imports_mean
    hotness = capex_dev - imports_dev

    return CapexImportsHotness(
        quarter=capex_pct.index[-1],
        history_start=HISTORY_START,
        n_history=len(capex_hist) - 1,
        capex_qoq_pct_gdp_now=float(capex_now),
        capex_qoq_pct_gdp_mean=float(capex_mean),
        capex_qoq_pct_gdp_std=float(capex_std),
        imports_qoq_pct_gdp_now=float(imports_now),
        imports_qoq_pct_gdp_mean=float(imports_mean),
        imports_qoq_pct_gdp_std=float(imports_std),
        capex_deviation_pp=float(capex_dev),
        imports_deviation_pp=float(imports_dev),
        hotness_pp=float(hotness),
    )


def print_capex_imports_hotness(
    target_quarter: pd.Period | None = None,
    h: CapexImportsHotness | None = None,
) -> None:
    """Print the diagnostic block — call from a model's _print_summary.

    If target_quarter is provided and capex is not yet published for it (i.e.
    the latest capex print is for an earlier quarter), the diagnostic is
    suppressed with a short note — it would otherwise be comparing the wrong
    quarter against the nowcast.
    """
    if h is None:
        h = capex_imports_hotness()

    if target_quarter is not None and h.quarter < target_quarter:
        print("\n  Capex-imports hotness diagnostic")
        print(f"    Capex (5625.0) not yet published for {target_quarter} "
              f"— latest is {h.quarter}.")
        print("    Diagnostic suppressed: would compare wrong quarter to nowcast.")
        return

    if h.hotness_pp > 0.10:
        interp = "model may be HOT"
    elif h.hotness_pp < -0.10:
        interp = "model may be COLD"
    else:
        interp = "negligible"
    print("\n  Capex-imports hotness diagnostic")
    print(f"    Reference quarter:  {h.quarter}    "
          f"history: {h.history_start}–prior ({h.n_history} obs)")
    print(f"    Equipment capex QoQ as % GDP:  now {h.capex_qoq_pct_gdp_now:+.2f}pp  "
          f"vs hist mean {h.capex_qoq_pct_gdp_mean:+.2f}pp "
          f"(σ {h.capex_qoq_pct_gdp_std:.2f})  → dev {h.capex_deviation_pp:+.2f}pp")
    print(f"    Goods imports QoQ as % GDP:    now {h.imports_qoq_pct_gdp_now:+.2f}pp  "
          f"vs hist mean {h.imports_qoq_pct_gdp_mean:+.2f}pp "
          f"(σ {h.imports_qoq_pct_gdp_std:.2f})  → dev {h.imports_deviation_pp:+.2f}pp")
    print(f"    Hotness (capex dev − imports dev):  {h.hotness_pp:+.2f}pp   → {interp}")
