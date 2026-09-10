"""Charts and printed diagnostics for the rstar model."""

from pathlib import Path  # noqa: TC003 — used at runtime in function signatures
from typing import Any

import mgplot as mg
import pandas as pd

from src.models.rstar_bonds.results import DEFAULT_CHART_BASE, RStarResults, load_results

CHART_DIR = DEFAULT_CHART_BASE / "RStarBonds"

# Used only for runs saved before `build_observations` began recording where its
# series came from. A current run carries its own records and `_rfooter` reads
# those instead, including the inputs of the y*/u* run the Taylor rule reads.
_RFOOTER = "Built using: RBA F1/F2/F3; NY Fed HLW; ABS 6401.0"
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
        # The rule's inputs can end a quarter before the state does, since the
        # market anchors run to the current quarter and `ystar_ustar` does not.
        # Report the last quarter the rule actually covers, and name it.
        latest = prescribed.dropna()
        print(f"\n  Latest prescription ({latest.index[-1]}): "
              f"{'tighten' if latest.iloc[-1] > 0 else 'ease'} "
              f"{abs(latest.iloc[-1]):.2f} points this quarter.")
        last4 = prescribed.tail(4).sum()
        act4 = results.policy_change_delivered().tail(4).sum()
        print(f"  Over four quarters the rule wanted {last4:+.2f}; {act4:+.2f} was delivered.")


def _rfooter(results: RStarResults) -> str:
    """Return the source line this run recorded, falling back for older runs."""
    return results.source_footer or _RFOOTER


def plot_rstar(results: RStarResults) -> None:
    """r* against the real yield it is extracted from and the world anchor."""
    rstar = results.rstar_posterior()

    ax = mg.fill_between_plot(_band(rstar), **_BAND_KWARGS)
    mg.line_plot(
        pd.DataFrame({
            "Indexed real 10y yield": results.real_yield(),
            "World r*": results.world_rstar(),
            "Australian r*": rstar.median(axis=1),
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
        rfooter=_rfooter(results),
        lfooter=_LFOOTER_BAND,
        show=False,
    )


def plot_rstar_real_nominal(results: RStarResults) -> None:
    """Real and nominal r*, each with a 90% highest-density interval."""
    real = results.rstar_hdi(0.90)
    nominal = results.nominal_rstar_hdi(0.90)
    real_median = results.rstar_median()
    nominal_median = results.nominal_rstar()
    last = real.index[-1]
    ends = " | ".join(
        f"{name} {band.loc[last, 'lower']:.2f} to {band.loc[last, 'upper']:.2f} "
        f"(median {median.loc[last]:.2f})"
        for name, band, median in (
            ("real", real, real_median),
            ("nominal", nominal, nominal_median),
        )
    )

    ax = mg.fill_between_plot(nominal, color="darkblue", alpha=0.15, label="Nominal 90% HDI")
    mg.fill_between_plot(real, ax=ax, color="darkorange", alpha=0.20, label="Real 90% HDI")
    mg.line_plot(
        pd.DataFrame({
            "Nominal r* (r* + 2.5% target)": results.nominal_rstar(),
            "Real r*": results.rstar_median(),
        }),
        ax=ax,
        color=["darkblue", "darkorange"],
        width=[2.5, 2.5],
        style=["-", "-"],
        annotate=True,
        rounding=2,
    )
    mg.finalise_plot(
        ax,
        title="Australia's real and nominal r-star",
        ylabel="Per cent",
        y0=True,
        legend={"loc": "best", "fontsize": "small"},
        lheader="Nominal is real plus the 2.5% target, so it carries the same uncertainty",
        rheader=f"{last} 90% HDI: {ends}",
        rfooter=_rfooter(results),
        lfooter=_LFOOTER_BAND,
        show=False,
    )


def plot_stance(results: RStarResults) -> None:
    """Plot the neutral nominal rate against the cash rate: the policy stance.

    Distinct from `plot_taylor_level`, which carries the same two lines but is
    about the rule. That chart is confined to the quarters where the rule's
    inputs exist, which is 1993Q1 onwards, and shows no uncertainty. This one
    runs over the whole estimation sample and keeps the band, because the
    question it answers is how much of the stance is inside the noise.

    Two neutral lines, because they answer different questions and reading the
    wrong one against `plot_policy_gap` makes the two charts contradict each
    other. `r* + 2.5` is steady-state neutral: where the cash rate belongs once
    inflation is at target. `r* + expected inflation` is neutral given what
    people actually expected at the time, and *that* is the line whose distance
    from the cash rate is the stance the model estimates. The two diverge
    whenever expectations sit away from the target, which through 2015-19 they
    did, averaging 2.25.

    Worked through on that window, which is where the difference matters most.
    The target-anchored line reads 0.67 points of looseness (cash 1.58 against
    nominal r* of 2.24) where `plot_policy_gap` reads -0.41. The 0.26 between
    them is expectations undershooting the 2.5 anchor, averaging 2.25 over those
    five years. Under `--short-rate bill` a second difference opens on top, since
    the model's window is then the bill and this chart still draws the cash rate.
    """
    nominal = results.nominal_rstar_hdi(0.90)
    nominal_median = results.nominal_rstar()
    cash = results.cash_rate()
    if cash.dropna().empty:
        print("  skipping stance chart: the cash rate is unavailable")
        return

    # Headline the stance against the expectations line, because that is the one
    # the model actually estimates and the one `plot_policy_gap` draws. The
    # target-anchored reading follows it, since the two differ by the anchor
    # shortfall and a reader comparing the charts needs both numbers named.
    anchored = (cash - nominal_median).dropna()
    stance = (cash - results.nominal_rstar(on_expectations=True)).dropna()
    last = stance.index[-1]
    direction = "restrictive" if stance.loc[last] > 0 else "expansionary"

    ax = mg.fill_between_plot(nominal, color="darkblue", alpha=0.15, label="Nominal r* 90% HDI")
    mg.line_plot(
        pd.DataFrame({
            "Nominal r* (r* + 2.5% target)": nominal_median,
            "Nominal r* (r* + expected inflation)": results.nominal_rstar(on_expectations=True),
            "Cash rate": cash,
        }),
        ax=ax,
        color=["darkblue", "darkorange", "black"],
        width=[2.5, 2.0, 1.5],
        style=["-", "--", "-"],
        annotate=True,
        rounding=2,
    )
    mg.finalise_plot(
        ax,
        title="The policy stance: the cash rate against nominal r-star",
        ylabel="Per cent",
        y0=True,
        legend={"loc": "best", "fontsize": "small"},
        # Both headers stay short: the two together span one line, and the long
        # form of either runs into the other.
        lheader="Stance = cash rate less the expectations line, not the target line",
        rheader=f"{last}: {abs(stance.loc[last]):.2f}pp {direction} "
                f"({abs(anchored.loc[last]):.2f} anchored)",
        rfooter=_rfooter(results),
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
        rfooter=_rfooter(results),
        lfooter=_LFOOTER_BAND,
        show=False,
    )


def plot_policy_gap(results: RStarResults) -> None:
    """Plot the policy gap, the second window's read on the stance."""
    gap = results.policy_gap_posterior()
    # Named from the run, not hardcoded: the default short rate is the 90-day
    # bill, and a chart labelled "cash rate" while the model is fitted on the
    # bill is how the stance charts came to disagree by 0.26 without saying so.
    short = "90-day bank bill" if results.constants.get("short_rate_is_bill") else "cash rate"

    ax = mg.fill_between_plot(_band(gap), **_BAND_KWARGS)
    mg.line_plot(
        gap.median(axis=1).rename(f"Real {short} less r*"),
        ax=ax,
        color=["darkgreen"],
        width=2,
        annotate=True,
        rounding=2,
    )
    mg.finalise_plot(
        ax,
        title="Policy gap",
        ylabel="Percentage points",
        y0=True,
        legend={"loc": "best", "fontsize": "small"},
        lheader="Positive is restrictive. An output of the model, not an input: "
                "nothing here asks the gap to move output",
        rfooter=_rfooter(results),
        lfooter=_LFOOTER_BAND,
        show=False,
    )


def plot_wedge(results: RStarResults) -> None:
    """Plot the wedge, the model's only latent state.

    Worth its own chart rather than being left as the gap between two lines on
    the r* chart. With `b_world` estimated the wedge is `r* - b_world·world`,
    so that distance is now the wrong series; and the wedge is where the
    package's claim lives, since world r* is imported and this is the part
    that is Australia's.
    """
    wedge = results.wedge_posterior()

    ax = mg.fill_between_plot(_band(wedge), **_BAND_KWARGS)
    mg.line_plot(
        wedge.median(axis=1).rename("Australian wedge over world r*"),
        ax=ax,
        color=["darkorange"],
        width=2,
        annotate=True,
        rounding=2,
    )
    loading = float(results.posterior["b_world"].mean()) if "b_world" in results.posterior else 1.0
    # Read the largest move off the run rather than describing the one the
    # default happens to produce. Under the HLW anchor it is +0.82 at 2022Q2;
    # under a market anchor that repricing is imported and the largest move is
    # a quarter of the size, so a hardcoded "one leap at liftoff" would be false.
    steps = results.wedge_median().diff().dropna()
    biggest = steps.abs().idxmax()
    mg.finalise_plot(
        ax,
        title="The Australian wedge over world r*",
        ylabel="Percentage points",
        y0=True,
        legend={"loc": "best", "fontsize": "small"},
        lheader=f"r* less {loading:.2f} x world r*. Largest quarterly move "
                f"{steps.loc[biggest]:+.2f} at {biggest}",
        rfooter=_rfooter(results),
        lfooter=_LFOOTER_BAND,
        show=False,
    )


def plot_borrower_stance(results: RStarResults) -> None:
    """Plot the risk-free stance against the one households actually faced.

    `g` is the real cash rate less r*, and it is what the model estimates. The
    second line is the real mortgage rate less r*. The distance between them is
    the pass-through story, and it is not a constant: it widened by 1.75 points
    between 2004-07 and 2015-19 and has since given back about one.
    """
    gap = results.policy_gap_posterior().median(axis=1)
    advertised = results.borrower_stance(discounted=False).dropna()
    borrower = results.borrower_stance().dropna()
    if advertised.empty and borrower.empty:
        print("  skipping borrower stance chart: the mortgage rate is unavailable")
        return

    # The advertised rate carries the history and the discounted one carries the
    # truth, so both are drawn: the distance between them is the discount, which
    # is not a constant and so cannot be spliced away.
    ax = mg.line_plot(
        pd.DataFrame({
            "Risk-free stance (real cash rate less r*)": gap,
            "Borrower stance (advertised rate)": advertised,
            "Borrower stance (discounted rate)": borrower,
        }),
        color=["darkgreen", "indianred", "darkred"],
        width=[2.0, 1.5, 2.5],
        style=["--", ":", "-"],
        annotate=True,
        rounding=2,
    )
    mg.finalise_plot(
        ax,
        title="The stance the RBA sets and the stance households faced",
        ylabel="Percentage points",
        y0=True,
        legend={"loc": "best", "fontsize": "small"},
        lheader="Levels are not comparable: the red lines carry a credit spread. "
                "Read the changes, and mind the widening discount",
        rfooter=_rfooter(results),
        lfooter=_LFOOTER,
        show=False,
    )


def plot_premium_correction(results: RStarResults) -> None:
    """Plot the term premium with and without the `k·g` correction.

    The one-window model set `k` to zero by construction, so its premium was
    really `true tp + k·g`. This is the check on whether the negative premium
    under QE was a finding about bond purchases or the model reading a floored
    cash rate through a missing coefficient.
    """
    corrected = results.term_premium_posterior().median(axis=1)
    one_window = results.term_premium_one_window().median(axis=1)

    combined = pd.DataFrame({
        "Corrected (less k·g)": corrected,
        "One window (y less r*)": one_window,
    })
    mg.line_plot_finalise(
        combined,
        color=["darkorange", "dimgrey"],
        style=["-", "--"],
        width=[2, 1.5],
        annotate=True,
        rounding=2,
        title="Term premium, corrected for policy stance",
        ylabel="Percentage points",
        y0=True,
        legend={"loc": "best", "fontsize": "small"},
        lheader="The gap between the two is what the long yield carries from the cash rate",
        rfooter=_rfooter(results),
        lfooter=_LFOOTER,
        show=False,
    )


def plot_premium_curve(results: RStarResults) -> None:
    """Plot the two premia together: the slope is what the third window adds.

    Neither level is identified, but their difference is, since both are read
    off the same r*. The QE window is where to look: the RBA's yield target was
    on the three-year bond, so any suppression it caused belongs in the shorter
    premium rather than the ten-year one.
    """
    combined = pd.DataFrame({
        "10-year premium": results.term_premium_posterior().median(axis=1),
        "3-year premium": results.medium_premium_posterior().median(axis=1),
    })
    mg.line_plot_finalise(
        combined,
        color=["darkorange", "teal"],
        width=[2, 2],
        annotate=True,
        rounding=2,
        title="Term premium by maturity",
        ylabel="Percentage points",
        y0=True,
        legend={"loc": "best", "fontsize": "small"},
        lheader="Levels are not identified; the gap between them is",
        rfooter=_rfooter(results),
        lfooter=_LFOOTER,
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
        rfooter=_rfooter(results),
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
    }).dropna(subset=["Taylor rule on this model's r*"])

    mg.line_plot_finalise(
        frame,
        color=["darkorange", "black", "darkblue"],
        width=[2.5, 1.5, 2],
        style=["-", "-", "--"],
        annotate=True,
        rounding=2,
        title="The cash rate and the Taylor rule",
        ylabel="Per cent",
        y0=True,
        legend={"loc": "best", "fontsize": "small"},
        lheader="Taylor 0.5/0.5, supply looked through. Cash rate above nominal r* is restrictive",
        rfooter=_rfooter(results),
        lfooter=_LFOOTER,
        show=False,
    )


def run_analysis(
    output_dir: Path | str | None = None,
    prefix: str = "rstar_bonds",
    chart_dir: Path | str | None = None,
) -> RStarResults:
    """Load a saved run, print the diagnostics, write the charts.

    A non-default prefix gets its own chart directory. `clear_chart_dir()`
    empties whatever it is pointed at, so without this a second run under a
    different prefix silently deletes the first run's charts, which is exactly
    what the comparison runs are for.
    """
    results = load_results(output_dir=output_dir, prefix=prefix)

    print_diagnostics(results)

    if chart_dir is None:
        chart_dir = CHART_DIR if prefix == "rstar_bonds" else CHART_DIR.parent / f"RStarBonds_{prefix}"
    mg.set_chart_dir(str(chart_dir))
    mg.clear_chart_dir()

    plot_rstar(results)
    plot_rstar_real_nominal(results)
    plot_wedge(results)
    plot_stance(results)
    plot_term_premium(results)
    if results.has_short_window():
        plot_policy_gap(results)
        plot_borrower_stance(results)
        plot_premium_correction(results)
    if results.has_curve():
        plot_premium_curve(results)
    plot_business_rstar(results)
    plot_taylor_level(results)

    print(f"\nCharts written to: {chart_dir if chart_dir is not None else CHART_DIR}")
    return results
