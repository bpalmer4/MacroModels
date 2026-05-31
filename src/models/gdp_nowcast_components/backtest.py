"""Backtest for the components (expenditure-identity) GDP nowcast.

Replays the nowcast across historical quarters and evaluates it against the
ABS-published outcome. Pseudo real-time: inputs and evaluation both use latest-
vintage data, truncated to the as-of information set (GDP ≤ target-1; component
sources ≤ target; published contributions ≤ target-1). The September re-
referencing guard is a live-only adjustment and is not exercised here — single-
vintage data already shares a reference basis.

Two things are measured:
  * headline error — summed nowcast vs published GDP growth (MAE, RMSE, bias),
    reported overall and excluding the COVID quarters (2020Q1–2021Q2);
  * per-component error — each component's contribution vs the ABS-published
    contribution, so it is clear which components carry the error (the bridged
    household / private-GFCF pieces and the noisy inventories proxy, typically).

Usage:
    uv run python -m src.models.gdp_nowcast_components.backtest
    uv run python -m src.models.gdp_nowcast_components.backtest --start 2017Q1 --end 2025Q3
"""

import argparse
import logging
from pathlib import Path

import mgplot as mg
import numpy as np
import pandas as pd

from src.models.gdp_nowcast_components import data as cd
from src.models.gdp_nowcast_components.model import _contribute, _nan_sum, build_asof

logger = logging.getLogger(__name__)

BACKTEST_CHART_DIR = "./charts/GDP-Nowcast-Components-Backtest/"
BACKTEST_OUTPUT_DIR = "./model_outputs/gdp_nowcast_components/"
COVID = (pd.Period("2020Q1", "Q-DEC"), pd.Period("2021Q2", "Q-DEC"))

# Component → published-contribution column, for the per-component error table.
_COMPONENT_PUB = {
    "Household consumption": "household_consumption",
    "Government consumption": "government_consumption",
    "Inventories": "inventories",
    "Net exports": "net_exports",
}


def run_backtest(start: str = "2015Q1", end: str | None = None, gov_fallback: bool = True) -> pd.DataFrame:
    """Replay the nowcast over [start, end] and report headline + component error.

    ``gov_fallback`` (default True) substitutes the ABS-published government
    consumption / public-GFCF contribution for quarters before the GFS workbook's
    Table 15 history reaches (currently 2022Q4). Government is accounting-exact at
    T-0, so this is a faithful stand-in that unlocks the long household / GFCF /
    inventories evaluation window. Affected quarters are flagged ``gov_source``.
    """
    pub = cd.published_contributions()
    gdp_c = pub["gdp"].dropna()

    first = pd.Period(start, "Q-DEC")
    last = pd.Period(end, "Q-DEC") if end else gdp_c.index[-1]
    targets = [t for t in gdp_c.index if first <= t <= last]

    rows = []
    for t in targets:
        asof = build_asof(t)
        contrib, priv, pubg = _contribute(asof)

        gov_source = "GFS"
        gov_missing = np.isnan(contrib["Government consumption"]) or np.isnan(pubg)
        if gov_fallback and gov_missing and t in pub.index and not np.isnan(pub.loc[t, "government_consumption"]):
            contrib["Government consumption"] = float(pub.loc[t, "government_consumption"])
            pubg = float(pub.loc[t, "public_gfcf"])
            contrib["Public investment"] = pubg
            gov_source = "ABS-published"

        identity = {k: v for k, v in contrib.items() if k != "Statistical discrepancy"}
        nowcast = _nan_sum(*identity.values())
        if np.isnan(nowcast) or any(np.isnan(v) for v in identity.values()):
            continue  # incomplete source data for this quarter — skip
        actual = float(gdp_c[t])
        row = {
            "quarter": t, "nowcast": nowcast, "actual": actual, "error": nowcast - actual,
            "private_gfcf": priv, "public_gfcf": pubg, "gov_source": gov_source,
        }
        for name, val in identity.items():
            row[f"contrib_{name}"] = val
        rows.append(row)

    df = pd.DataFrame(rows).set_index("quarter")
    _report(df, pub, start, end)
    return df


def _metrics(err: pd.Series) -> dict[str, float]:
    """MAE, RMSE and bias for an error series."""
    return {
        "n": len(err),
        "MAE": float(err.abs().mean()),
        "RMSE": float(np.sqrt((err ** 2).mean())),
        "bias": float(err.mean()),
    }


def _report(df: pd.DataFrame, pub: pd.DataFrame, start: str, end: str | None) -> None:
    """Print headline + per-component summaries, save parquet/txt, draw charts."""
    ex_covid = df.loc[(df.index < COVID[0]) | (df.index > COVID[1])]

    lines = [
        f"Components GDP nowcast backtest — {start} to {end or 'latest'}",
        "=" * 60,
        "Headline (summed nowcast vs published GDP growth, ppt):",
    ]
    for label, sub in [("  all quarters ", df), ("  ex-COVID     ", ex_covid)]:
        m = _metrics(sub["error"])
        lines.append(f"{label} n={m['n']:>3}  MAE={m['MAE']:.3f}  RMSE={m['RMSE']:.3f}  bias={m['bias']:+.3f}")

    lines += ["", "Per-component MAE vs ABS-published contribution (ex-COVID, ppt):"]
    for name, col in _COMPONENT_PUB.items():
        mine = df[f"contrib_{name}"]
        comp = (mine - pub[col].reindex(mine.index)).loc[ex_covid.index].dropna()
        if len(comp):
            lines.append(f"  {name:<24} MAE={comp.abs().mean():.3f}  bias={comp.mean():+.3f}")
    # GFCF is reported via its private/public split (chart shows the combined bar).
    for label, col in [("Investment: private GFCF", "private_gfcf"),
                       ("Investment: public GFCF", "public_gfcf")]:
        pub_col = "private_gfcf" if "private" in col else "public_gfcf"
        comp = (df[col] - pub[pub_col].reindex(df.index)).loc[ex_covid.index].dropna()
        if len(comp):
            lines.append(f"  {label:<24} MAE={comp.abs().mean():.3f}  bias={comp.mean():+.3f}")

    report = "\n".join(lines)
    print("\n" + report + "\n")

    out = Path(BACKTEST_OUTPUT_DIR)
    out.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out / "backtest_results.parquet")
    (out / "backtest_summary.txt").write_text(report + "\n")
    logger.info("Saved backtest to %s", out)

    _plot(df)


def _plot(df: pd.DataFrame) -> None:
    """Nowcast vs actual scatter/line and the error series."""
    mg.set_chart_dir(BACKTEST_CHART_DIR)
    mg.clear_chart_dir()

    both = df[["nowcast", "actual"]].rename(columns={"nowcast": "Nowcast", "actual": "Actual (ABS)"})
    mg.line_plot_finalise(
        both, title="Components Nowcast vs Actual GDP Growth",
        ylabel="Percentage points (q/q)", width=[2, 2], style=["-", "--"],
        y0=True, legend={"loc": "best", "fontsize": "small"},
        rfooter="ABS 5206.0", lfooter="Australia. Pseudo real-time backtest. ", show=False,
    )
    err = df["error"].rename("Nowcast − Actual")
    mg.bar_plot_finalise(
        err, title="Components Nowcast Error", ylabel="Percentage points",
        y0=True, rfooter="ABS 5206.0",
        lfooter=f"Australia. MAE={err.abs().mean():.2f}, bias={err.mean():+.2f} ppt. ", show=False,
    )


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Components GDP nowcast backtest")
    parser.add_argument("--start", default="2015Q1", help="First quarter (default: 2015Q1)")
    parser.add_argument("--end", default=None, help="Last quarter (default: latest)")
    args = parser.parse_args()
    run_backtest(start=args.start, end=args.end)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
