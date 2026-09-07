"""Charts and printed diagnostics for the rstar model."""

from pathlib import Path  # noqa: TC003 — used at runtime in function signatures
from typing import Any

import mgplot as mg
import pandas as pd

from src.models.rstar.results import DEFAULT_CHART_BASE, RStarResults, load_results

CHART_DIR = DEFAULT_CHART_BASE / "RStar"

_RFOOTER = "Source: RBA F1/F2/F3; NY Fed HLW; ABS 6401.0"
_LFOOTER = "Australia. r* model. "
# Deliberately terse: the full sentence ran into the source line on the right.
_LFOOTER_BAND = "Australia. r* model. Band conditional on the imposed sigma_walk. "

_BAND_KWARGS: dict[str, Any] = {
    "color": "cornflowerblue",
    "alpha": 0.25,
    "label": "90% credible interval",
}


def _band(posterior: pd.DataFrame) -> pd.DataFrame:
    """Return a two-column DataFrame of 5th and 95th posterior percentiles."""
    return pd.DataFrame({
        "lower": posterior.quantile(0.05, axis=1),
        "upper": posterior.quantile(0.95, axis=1),
    })


def print_diagnostics(results: RStarResults) -> None:
    """Print the parameters and the checks that would show the model failing."""
    print("\nPosterior summary")
    print("-" * 70)
    print(results.summary().to_string())

    print("\nIs r* placed by the data or by the prior?")
    print("-" * 70)
    for key, value in results.rstar_variation().items():
        print(f"  {key:<34} {value:6.2f}")

    print("\nWhat the decomposition attributes where")
    print("-" * 70)
    for key, value in results.variance_shares().items():
        print(f"  {key:<34} {value:6.1%}" if "share" in key else f"  {key:<34} {value:6.2f}")

    jumps = results.jumps()
    if not jumps.empty:
        print("\nThe Australian wedge: estimated jump at each asserted break (pp)")
        print("-" * 70)
        print(jumps.round(3).to_string())
        straddling = jumps[(jumps["5%"] < 0) & (jumps["95%"] > 0)]
        for label in straddling.index:
            print(f"  note: the {label} jump straddles zero — that break may not be earning its place")

    check = results.end_break_check()
    print("\nHas a new break opened since the last asserted one?")
    print("-" * 70)
    window = int(check["window"])
    print(f"  term premium, last {window} quarters   {check['recent tp mean']:6.2f}")
    print(f"  {'its long-run mean (mu_tp)':<34} {check['mu_tp']:6.2f}")
    print(f"  {'deviation, in stationary sds':<34} {check['in stationary sds']:6.2f}")
    if abs(check["in stationary sds"]) > 1.0:
        print("  *** The premium has drifted from its mean. A new break may be needed:")
        print("  *** the wedge can only move at asserted dates, so a genuine shift in")
        print("  *** Australia's spread over world r* has nowhere to go but here.")

    recent = pd.DataFrame({
        "yield": results.real_yield(),
        "r*": results.rstar_median(),
        "tp": results.term_premium_posterior().median(axis=1),
        "world": results.world_rstar(),
        "r*_bus": results.business_rstar(),
    }).tail(8)
    print("\nRecent estimates (%)")
    print("-" * 70)
    print(f"  {'':<10}{'real yld':>10}{'r*':>8}{'term prm':>10}{'world r*':>10}{'r* bus':>9}")
    for period, row in recent.iterrows():
        bus = f"{row['r*_bus']:9.2f}" if pd.notna(row["r*_bus"]) else f"{'n/a':>9}"
        print(f"  {period!s:<10}{row['yield']:10.2f}{row['r*']:8.2f}{row['tp']:10.2f}{row['world']:10.2f}{bus}")

    prescribed = results.policy_change()
    if prescribed.notna().any():
        both = pd.DataFrame({
            "pi": results._extra("pi"),  # noqa: SLF001 — same package
            "supply": results.supply_contribution(),
            "core": results.rule_inflation(),
            "gap": results._extra("ygap"),  # noqa: SLF001 — same package
            "rule": prescribed,
            "actual": results.policy_change_delivered(),
        }).dropna(subset=["rule"]).tail(6)
        print("\nPolicy rule: change in the cash rate, points per quarter")
        print("-" * 70)
        print(f"  {'':<10}{'pi':>7}{'supply':>8}{'core':>7}{'gap':>7}{'rule':>8}{'actual':>8}")
        for period, row in both.iterrows():
            print(
                f"  {period!s:<10}{row['pi']:7.2f}{row['supply']:8.2f}{row['core']:7.2f}"
                f"{row['gap']:7.2f}{row['rule']:8.2f}{row['actual']:8.2f}",
            )
        print(f"\n  Latest prescription: {'tighten' if prescribed.iloc[-1] > 0 else 'ease'} "
              f"{abs(prescribed.iloc[-1]):.2f} points this quarter.")
        last4 = prescribed.tail(4).sum()
        act4 = results.policy_change_delivered().tail(4).sum()
        print(f"  Over four quarters the rule wanted {last4:+.2f}; {act4:+.2f} was delivered.")


def plot_rstar(results: RStarResults) -> None:
    """r* against the real yield it is extracted from and the world anchor."""
    rstar = results.rstar_posterior()

    ax = mg.fill_between_plot(_band(rstar), **_BAND_KWARGS)
    mg.line_plot(
        pd.DataFrame({
            "Indexed real 10y yield": results.real_yield(),
            "World r*": results.world_rstar(),
            "r*": rstar.median(axis=1),
        }),
        ax=ax,
        color=["black", "darkgreen", "darkorange"],
        width=[1.0, 1.5, 2.5],
        style=["-", ":", "--"],
        annotate=True,
        rounding=2,
    )
    mg.finalise_plot(
        ax,
        title="r* and the real bond yield",
        ylabel="Per cent",
        y0=True,
        legend={"loc": "best", "fontsize": "small"},
        lheader="r* is the permanent component of the real yield, anchored on world r*",
        rfooter=_RFOOTER,
        lfooter=_LFOOTER_BAND,
        show=False,
    )


def plot_term_premium(results: RStarResults) -> None:
    """Plot the transitory half: everything in the yield that is not r*."""
    premium = results.term_premium_posterior()

    ax = mg.fill_between_plot(_band(premium), **_BAND_KWARGS)
    mg.line_plot(
        premium.median(axis=1).rename("Term premium"),
        ax=ax,
        color=["darkorange"],
        width=2,
        annotate=True,
        rounding=2,
    )
    mg.finalise_plot(
        ax,
        title="Term premium",
        ylabel="Percentage points",
        y0=True,
        legend={"loc": "best", "fontsize": "small"},
        lheader="The transitory component, including whatever liquidity premium indexed bonds carry",
        rfooter=_RFOOTER,
        lfooter=_LFOOTER_BAND,
        show=False,
    )


def plot_business_rstar(results: RStarResults) -> None:
    """r* for firms: the risk-free rate plus the observed credit spread."""
    business = results.business_rstar().dropna()
    if business.empty:
        print("  skipping business r* chart: no corporate spread available")
        return

    mg.line_plot_finalise(
        pd.DataFrame({
            "r* (risk-free)": results.rstar_median().reindex(business.index),
            "r* + credit spread": business,
        }),
        color=["darkorange", "darkblue"],
        width=[2, 2],
        style=["--", "-"],
        annotate=True,
        rounding=2,
        title="r* for firms",
        ylabel="Per cent",
        y0=True,
        legend={"loc": "best", "fontsize": "small"},
        lheader="What the bond market prices, and what investment actually faces",
        rfooter=_RFOOTER,
        lfooter=_LFOOTER,
        show=False,
    )



def plot_taylor_level(results: RStarResults) -> None:
    """Plot the level Taylor rule against the cash rate, using this model's r*."""
    level = results.taylor_level()
    cash = results.cash_rate()
    if level.dropna().empty:
        print("  skipping Taylor level chart: inputs unavailable (run ystar first)")
        return

    frame = pd.DataFrame({
        "Taylor rule on this model's r*": level,
        "Cash rate": cash,
        "Nominal r* (r* + 2.5% target)": results.nominal_rstar(),
        "Real r*": results.rstar_median(),
    }).dropna(subset=["Taylor rule on this model's r*"])

    mg.line_plot_finalise(
        frame,
        color=["darkorange", "black", "darkblue", "darkgreen"],
        width=[2.5, 1.5, 2, 1.5],
        style=["-", "-", "--", ":"],
        annotate=True,
        rounding=2,
        title="Where the cash rate should be, on a Taylor rule",
        ylabel="Per cent",
        y0=True,
        legend={"loc": "best", "fontsize": "small"},
        lheader="Taylor 0.5/0.5, supply looked through. Cash rate above nominal r* is restrictive",
        rfooter=_RFOOTER,
        lfooter=_LFOOTER,
        show=False,
    )


def run_analysis(
    output_dir: Path | str | None = None,
    prefix: str = "rstar",
    chart_dir: Path | str | None = None,
) -> RStarResults:
    """Load a saved run, print the diagnostics, write the charts."""
    results = load_results(output_dir=output_dir, prefix=prefix)

    print_diagnostics(results)

    mg.set_chart_dir(str(chart_dir if chart_dir is not None else CHART_DIR))
    mg.clear_chart_dir()

    plot_rstar(results)
    plot_term_premium(results)
    plot_business_rstar(results)
    plot_taylor_level(results)

    print(f"\nCharts written to: {chart_dir if chart_dir is not None else CHART_DIR}")
    return results
