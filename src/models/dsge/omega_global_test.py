"""Does the FA-NK financial shock ω track global QE / financial conditions?

Tests the thesis that AU imported global financial conditions (the post-GFC
"great divergence"). Takes the smoothed financial shock ω from the FA-NK run
(model_outputs/fa_nk_states.csv) and correlates / overlays it with global proxies
fetched from FRED.

Finding: ω tracks the global financial CYCLE (Chicago Fed FCI, VIX) far more than
the QE *quantity* (Fed balance sheet, which is GFC-contaminated) or the safe-real-
rate channel (US 10y real yield ≈ 0). So "imported global financial conditions",
not "global QE free money" narrowly.

Run: uv run python -m src.models.dsge.omega_global_test
"""

from pathlib import Path

import numpy as np
import pandas as pd

FRED_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={}"
ROOT = Path(__file__).parent.parent.parent.parent


def _fred(sid: str) -> pd.Series | None:
    """Fetch a FRED series, quarterly-averaged, with a PeriodIndex."""
    try:
        df = pd.read_csv(FRED_URL.format(sid))
        df.columns = ["date", sid]
        df["date"] = pd.to_datetime(df["date"])
        s = pd.to_numeric(df.set_index("date")[sid], errors="coerce").resample("QE").mean()
        s.index = pd.PeriodIndex(s.index, freq="Q")
        return s.dropna()
    except Exception as e:  # noqa: BLE001 — network/format failures are reported, not fatal
        print(f"  ({sid} fetch failed: {e})")
        return None


def load_omega() -> pd.Series:
    """Smoothed FA-NK financial shock ω (data-pinned from 2005, where the spread exists)."""
    st = pd.read_csv(ROOT / "model_outputs" / "fa_nk_states.csv", index_col=0)
    st.index = pd.PeriodIndex(st.index, freq="Q")
    return st["omega"].dropna()


def run() -> None:
    omega = load_omega()

    proxies = {
        "WALCL": "Fed assets YoY growth (QE qty)",
        "DFII10": "US 10y real yield (TIPS)",
        "VIXCLS": "VIX (risk)",
        "NFCI": "Chicago Fed FCI",
    }
    fetched = {k: _fred(k) for k in proxies}
    fetched = {k: v for k, v in fetched.items() if v is not None}
    if "WALCL" in fetched:
        fetched["WALCL"] = fetched["WALCL"].pct_change(4) * 100  # QE intensity

    print("\ncorr( omega , X )   [thesis: QE/free money -> compressed AU spreads -> lower omega]")
    qe_lo, qe_hi = pd.Period("2010Q1"), pd.Period("2019Q4")
    for k in ["WALCL", "DFII10", "VIXCLS", "NFCI"]:
        if k in fetched:
            d = pd.DataFrame({"o": omega, "x": fetched[k]}).dropna()
            dp = d[(d.index >= qe_lo) & (d.index <= qe_hi)]
            print(f"  {proxies[k]:28s}: full={d['o'].corr(d['x']):+.2f} (n={len(d)})   "
                  f"QE-era 2010-19={dp['o'].corr(dp['x']):+.2f}")

    # Chart: omega vs the strongest correlate (Chicago Fed FCI), both z-scored
    import mgplot as mg

    def z(s: pd.Series) -> pd.Series:
        return (s - s.mean()) / s.std()

    if "NFCI" in fetched:
        d = pd.DataFrame({"AU financial shock ω (FA-NK)": omega,
                          "Chicago Fed financial conditions": fetched["NFCI"]}).dropna()
        d = d.apply(z)
        full = pd.period_range(d.index.min(), d.index.max(), freq="Q")
        mg.set_chart_dir(str(ROOT / "charts" / "dsge-fa-nk"))
        mg.line_plot_finalise(
            d.reindex(full), width=2, color=["firebrick", "navy"], dropna=False,
            title="FA-NK financial shock vs global financial conditions",
            ylabel="Standardised (z-score)", y0=True, legend={"loc": "best"},
            rfooter="FA-NK DSGE; FRED",
            lfooter=f"Australia. omega vs Chicago Fed FCI, corr={d.corr().iloc[0, 1]:+.2f}. ",
        )
        print(f"\nChart written to {ROOT / 'charts' / 'dsge-fa-nk'}/")


if __name__ == "__main__":
    run()
