"""Data layer for the components (expenditure-identity) GDP nowcast.

Every series returned here is a quarterly ``pd.Series`` on a ``Q-DEC``
``PeriodIndex``. The model assembles a GDP nowcast as the sum of expenditure
components' contributions to quarter-on-quarter growth, each read the day before
the National Accounts (T-0) from its source release:

    Component             Source @ T-0                     Form
    --------------------- -------------------------------- --------------------
    Household consumption 5682.0 t.5682015 (CVM index)     bridged
    Govt consumption      GFS Table 15 (CVM $m)            accounting-exact
    Private GFCF          5625.0 capex + 8755.0 constr     bridged
    Govt GFCF             GFS Table 15 (CVM $m)            accounting-exact
    Inventories           5676.0 t.5676001 (CVM $m level)  accounting-exact
    Net exports           5302.0 t.530205 (CVM $m)         accounting-exact

The "accounting-exact" components are real $m levels that map straight onto the
GDP expenditure identity, so their contribution is computed directly. The
"bridged" components come from indicators that only partially cover their NA
aggregate (a CVM index for consumption; surveys missing IP products / some
industries for private GFCF), so they are mapped to a contribution via a small
OLS bridge fitted on the ABS-published contribution history.

Published ABS "Contributions to growth" series (5206.0 t.5206002) are used both
as the historical bars of the output chart and as the bridge-fit targets.

Re-referencing guard (`reref_factor`): in the annual September re-referencing
quarter the source release is on the new reference year while the last published
GDP is still on the old one. A CVM level is a clean scalar under re-referencing,
so we down-weight the source onto the GDP vintage's basis using the median ratio
of the current series to the vintage published with the latest GDP. This is a
*live-only* adjustment — the backtest uses single-vintage data, so every series
already shares a reference basis and the factor is 1.
"""

from functools import cache

import pandas as pd
import readabs as ra
from readabs import metacol as mc

from src.data.business_indicators import get_inventories_qrtly
from src.data.capex import get_total_capex_qrtly
from src.data.construction import get_private_construction_qrtly
from src.data.gdp import get_gdp
from src.data.gov_finance import get_gov_consumption_gfs_qrtly, get_public_investment_gfs_qrtly
from src.data.household_spending import get_household_spending_cvm_qrtly

# --- Series identifiers -------------------------------------------------------

_INV_TABLE = "5676001"
_INV_DID = (
    "Inventories ;  Total (State) ;  Total (Industry) ;"
    "  Chain Volume Measures ;  TOTAL (SCP_SCOPE) ;"
)

_NX_TABLE = "530205"  # BoP goods & services account, chain volume measures
_NX_CREDITS_DID = "Chain Volume Measures ;  Goods and Services credits ;"
_NX_DEBITS_DID = "Chain Volume Measures ;  Goods and Services debits ;"  # stored negative

_CONTRIB_TABLE = "5206002_Expenditure_Volume_Measures"
# ABS-published contributions to q/q GDP growth (CVM, SA). Keys are our short
# names; values are the exact Data Item Descriptions.
_CONTRIB_DIDS = {
    "household_consumption": "Households ;  Final consumption expenditure: Contributions to growth ;",
    "government_consumption": "General government ;  Final consumption expenditure: Contributions to growth ;",
    "private_gfcf": "Private ;  Gross fixed capital formation: Contributions to growth ;",
    "public_gfcf": "Public ;  Gross fixed capital formation: Contributions to growth ;",
    "investment_gfcf": "All sectors ;  Gross fixed capital formation: Contributions to growth ;",
    "inventories": "Changes in inventories: Contributions to growth ;",
    "exports": "Exports of goods and services: Contributions to growth ;",
    "imports": "Imports of goods and services: Contributions to growth ;",
    "gdp": "GROSS DOMESTIC PRODUCT: Contributions to growth ;",
}

_STYPE = "Seasonally Adjusted"


def _to_qdec(s: pd.Series) -> pd.Series:
    """Coerce a quarterly series onto a Q-DEC PeriodIndex, sorted, NaN-dropped."""
    s = s.dropna().sort_index()
    if isinstance(s.index, pd.PeriodIndex) and s.index.freqstr != "Q-DEC":
        s.index = s.index.asfreq("Q-DEC")
    return s


# --- GDP level and published contributions -----------------------------------


@cache
def gdp_level() -> pd.Series:
    """Real GDP, chain volume measures, seasonally adjusted (5206.0 $m level)."""
    return _to_qdec(get_gdp(gdp_type="CVM", seasonal="SA").data)


@cache
def published_contributions() -> pd.DataFrame:
    """ABS-published contributions to q/q GDP growth (CVM, SA), as a DataFrame.

    Columns are the short names in ``_CONTRIB_DIDS``. ``net_exports`` is added as
    exports + imports (imports already enter as a negative contribution).
    """
    d, m = ra.read_abs_cat("5206.0", single_excel_only=_CONTRIB_TABLE, verbose=False)
    m = m[m[mc.table] == _CONTRIB_TABLE]

    cols = {}
    for name, did in _CONTRIB_DIDS.items():
        row = m[(m[mc.did] == did) & (m[mc.stype] == _STYPE)]
        if len(row):
            cols[name] = d[_CONTRIB_TABLE][row[mc.id].iloc[0]]
    df = pd.DataFrame(cols)
    df.index = pd.PeriodIndex(df.index, freq="Q-DEC")
    df["net_exports"] = df["exports"] + df["imports"]
    return df.sort_index()


# --- Accounting-exact component levels (source releases, T-0) -----------------


@cache
def household_consumption_level() -> pd.Series:
    """Household final consumption expenditure, CVM SA (5206.0 $m level).

    The actual NA consumption aggregate — published only with GDP, so available
    through ``target - 1`` at T-0. Used as the bridge target (predict its growth
    from the HSI) and to convert that growth back to a contribution via the exact
    ``Δlevel / GDP_lag`` formula, the same path the accounting-exact components use.
    """
    d, m = ra.read_abs_cat("5206.0", single_excel_only=_CONTRIB_TABLE, verbose=False)
    m = m[m[mc.table] == _CONTRIB_TABLE]
    row = m[(m[mc.did] == "Households ;  Final consumption expenditure ;")
            & (m[mc.stype] == _STYPE)].iloc[0]
    return _to_qdec(d[_CONTRIB_TABLE][row[mc.id]])


@cache
def inventories_level() -> pd.Series:
    """Private non-farm inventory level, CVM SA (5676.0 t.5676001 $m)."""
    return _to_qdec(get_inventories_qrtly().data)


@cache
def net_exports_level() -> pd.Series:
    """Net exports balance, CVM SA (5302.0 t.530205 $m) = G&S credits + debits.

    Debits are stored negative, so the sum is the net-exports balance. Returned
    alongside its credit/debit parts (see :func:`net_exports_parts`) so the
    re-referencing guard can down-weight each leg separately.
    """
    exports, imports = net_exports_parts()
    return _to_qdec(exports + imports)


@cache
def net_exports_parts() -> tuple[pd.Series, pd.Series]:
    """Return the (exports, imports) legs of the BoP G&S balance, CVM SA (imports negative)."""
    d, m = ra.read_abs_cat("5302.0", single_excel_only=_NX_TABLE, verbose=False)
    m = m[m[mc.table] == _NX_TABLE]

    def leg(did: str) -> pd.Series:
        row = m[(m[mc.did] == did) & (m[mc.stype] == _STYPE)].iloc[0]
        return _to_qdec(d[_NX_TABLE][row[mc.id]])

    return leg(_NX_CREDITS_DID), leg(_NX_DEBITS_DID)


@cache
def government_consumption_level() -> pd.Series:
    """General government final consumption expenditure, CVM SA (GFS Table 15 $m)."""
    return _to_qdec(get_gov_consumption_gfs_qrtly().data)


@cache
def government_gfcf_level() -> pd.Series:
    """Total public gross fixed capital formation, CVM SA (GFS Table 15 $m)."""
    return _to_qdec(get_public_investment_gfs_qrtly().data)


# --- Bridged component proxies (indicators, T-0) ------------------------------


def household_spending_cvm_level(history_month: str | None = None) -> pd.Series:
    """Total household spending, CVM SA (5682.0 t.5682015 index), as a Q-DEC series.

    Delegates to the shared loader, passing ``history_month`` straight through.
    The quarterly CVM table only ships in the quarter-end-month snapshot, so the
    target quarter's end month (e.g. ``"dec-2025"``) pulls the snapshot covering
    it — a single targeted call, separate from any monthly 5682.0 fetch. Empty
    series if the quarter isn't available (caller degrades gracefully).

    Args:
        history_month: quarter-end month as ``"mmm-yyyy"``, or ``None`` for the
            most recent quarter-end month.

    """
    return _to_qdec(get_household_spending_cvm_qrtly(history_month).data)


@cache
def private_capex_level() -> pd.Series:
    """Total private new capex, CVM SA (5625.0 $m)."""
    return _to_qdec(get_total_capex_qrtly().data)


@cache
def private_construction_level() -> pd.Series:
    """Private-sector construction work done, CVM SA (8755.0 $m)."""
    return _to_qdec(get_private_construction_qrtly().data)


# --- Re-referencing guard (live only) -----------------------------------------


def month_tag(period: pd.Period) -> str:
    """Quarter-end month tag (``"mmm-yyyy"``, lowercase) for a quarterly Period."""
    mon = {1: "mar", 2: "jun", 3: "sep", 4: "dec"}[period.quarter]
    return f"{mon}-{period.year}"


def gdp_vintage_month(gdp: pd.Series) -> str:
    """Quarter-end month tag of the latest published GDP quarter."""
    return month_tag(gdp.index[-1])


def inventories_reref(gdp: pd.Series) -> pd.Series:
    """Inventory level, down-weighted onto the GDP vintage's reference basis (live)."""
    inv = inventories_level()
    return inv / reref_factor(inv, "5676.0", _INV_TABLE, _INV_DID, gdp)


def net_exports_reref(gdp: pd.Series) -> pd.Series:
    """Net exports balance, with exports/imports each down-weighted onto GDP's basis (live)."""
    exports, imports = net_exports_parts()
    f_exp = reref_factor(exports, "5302.0", _NX_TABLE, _NX_CREDITS_DID, gdp)
    f_imp = reref_factor(imports, "5302.0", _NX_TABLE, _NX_DEBITS_DID, gdp)
    return _to_qdec(exports / f_exp + imports / f_imp)


def reref_factor(current: pd.Series, cat: str, table: str, did: str, gdp: pd.Series) -> float:
    """Scalar to divide a CVM source by, putting it on the GDP vintage's basis.

    Returns 1.0 outside the re-referencing straddle (when GDP has caught up to
    the source, everything was re-referenced together). Inside the straddle,
    fetches the source vintage that was current when the latest GDP printed and
    returns the median ratio current/vintage. Returns 1.0 on any failure.
    """
    if gdp.index[-1] >= current.index[-1]:
        return 1.0
    try:
        tag = gdp_vintage_month(gdp)
        hd, hm = ra.read_abs_cat(cat, single_excel_only=table, history=tag, verbose=False)
        hm = hm[hm[mc.table] == table]
        row = hm[(hm[mc.did] == did) & (hm[mc.stype] == _STYPE)].iloc[0]
        hist = _to_qdec(hd[table][row[mc.id]])
        j = current.index.intersection(hist.index)
        if not len(j):
            return 1.0
        return float((current.loc[j] / hist.loc[j]).median())
    except (KeyError, ValueError, OSError, IndexError):
        return 1.0
