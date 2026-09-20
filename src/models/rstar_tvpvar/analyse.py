"""Charts and printed diagnostics for the TVP-VAR."""

from pathlib import Path

import mgplot as mg
import pandas as pd

from src.models.common.diagnostics import save_diagnostics
from src.models.common.inflation_scale import to_nominal
from src.models.rstar_tvpvar.ensemble import load_ensemble, print_ensemble
from src.models.rstar_tvpvar.observations import VARIABLES
from src.models.rstar_tvpvar.results import DEFAULT_CHART_BASE, TvpVarResults, load_results
from src.models.rstar_tvpvar.standard_charts import plot_standard_suite

CHART_DIR = DEFAULT_CHART_BASE / "RStarTVPVAR"

_RFOOTER = "Built using: ABS 5206.0, 6401.0; RBA F1"
_LFOOTER = "Australia. TVP-VAR r* model. "
_BAND_KWARGS = {"color": "cornflowerblue", "alpha": 0.25, "label": "90% credible interval"}

# A posterior/prior ratio inside this band means `sigma_q` learned essentially
# nothing from the data, which makes the r* path a picture of the prior. The
# window is deliberately wide: this is a flag to the reader, not a test.
_PRIOR_RATIO_LOW = 0.8
_PRIOR_RATIO_HIGH = 1.25
# Above this share of explosive draw-quarters the long-horizon projection stops
# being a forecast in enough of the posterior to distrust the band.
_EXPLOSIVE_WARN = 0.05
# Above this share of quarters with the constant-coefficient baseline inside the
# credible band, the drift is not distinguishable from no drift at all. Two
# thirds is a reader's flag rather than a test: there is no null distribution
# for "a point path inside an interval" being quoted here.
_BASELINE_INSIDE_WARN = 0.667


def _rfooter(results: TvpVarResults) -> str:
    """Return the source line this run recorded, falling back for older runs."""
    return results.source_footer or _RFOOTER


def print_baseline(results: TvpVarResults) -> None:
    """Print the constant-coefficient comparison, which asks whether the drift does anything.

    Kept separate from `print_diagnostics` because it is the one check that is
    about whether the MODEL earns its name, rather than about whether this run
    of it converged.
    """
    baseline = results.baseline_report()
    print("\nDoes the coefficient drift do anything? (against a constant-coefficient VAR)")
    print("-" * 70)
    if not baseline:
        print("  No overlapping quarters to compare.")
        return
    for key, value in baseline.items():
        print(f"  {key:<34} {value:8.3f}")
    if baseline["share of quarters inside the band"] > _BASELINE_INSIDE_WARN:
        print("  *** The constant-coefficient path sits inside the band in most quarters.")
        print("  *** The time variation is then not doing visible work, and r* is a")
        print("  *** constant-coefficient projection with several thousand extra states.")


def print_diagnostics(results: TvpVarResults) -> None:
    """Print the parameters and the checks that would show the model failing."""
    print("\nPosterior summary (scalar parameters only)")
    print("-" * 70)
    print(results.summary().to_string())

    drift = results.prior_vs_posterior()
    print("\nIs the time variation placed by the data or by the prior?")
    print("-" * 70)
    if not drift:
        print("  sigma_q was IMPOSED, so there is nothing to compare. Sweep it with --sigma-q.")
    else:
        for key, value in drift.items():
            print(f"  {key:<34} {value:8.4f}")
        ratio = drift["ratio, posterior / prior"]
        if _PRIOR_RATIO_LOW < ratio < _PRIOR_RATIO_HIGH:
            print("  *** The posterior is sitting on its prior. How much the coefficients")
            print("  *** drift is then an assumption, and so is the r* path that follows")
            print("  *** from it. This is the test that removed rstar_hlw from the summary.")

    stability = results.stability_report()
    print("\nIs the projection a forecast, or an extrapolation?")
    print("-" * 70)
    for key, value in stability.items():
        print(f"  {key:<34} {value:8.3f}")
    if stability["share of draw-quarters explosive"] > _EXPLOSIVE_WARN:
        print("  *** A material share of draws is explosive, so the H-step projection")
        print("  *** runs away in those draws and the band is not trustworthy.")

    print_baseline(results)

    horizon = results.rstar_median()
    limit = results.unconditional_mean()
    gap = (horizon - limit).dropna()
    print(f"\nr*, {results.horizon} quarters ahead, against the steady state it tends to")
    print("-" * 70)
    print(f"  {'r* latest':<34} {horizon.iloc[-1]:8.2f}")
    print(f"  {'unconditional mean, latest':<34} {limit.dropna().iloc[-1]:8.2f}")
    if len(gap):
        print(f"  {'mean absolute gap':<34} {gap.abs().mean():8.2f}")
        print(f"  {'max absolute gap':<34} {gap.abs().max():8.2f}")
    print(f"  {'r* sample mean':<34} {horizon.mean():8.2f}")
    print(f"  {'r* range':<34} {horizon.max() - horizon.min():8.2f}")

    recent = pd.DataFrame({
        "r*": horizon,
        "steady state": limit,
        "real rate": results.observed("real_rate"),
        "inflation": results.observed("inflation"),
        "growth": results.observed("growth"),
    }).tail(8)
    print("\nRecent estimates (%)")
    print("-" * 70)
    print(f"  {'':<10}{'r*':>8}{'steady':>9}{'real r':>9}{'infl':>8}{'growth':>8}")
    for period, row in recent.iterrows():
        def cell(value: float) -> str:
            return f"{value:8.2f}" if pd.notna(value) else f"{'n/a':>8}"
        print(f"  {period!s:<10}{cell(row['r*'])}{cell(row['steady state']):>9}"
              f"{cell(row['real rate']):>9}{cell(row['inflation'])}{cell(row['growth'])}")


def plot_rstar(results: TvpVarResults) -> None:
    """Plot r* against the real policy rate it is derived from.

    THE BAND IS 50% UNDER THE STEADY-STATE DEFINITION, not the usual 90%. The
    resting point is `(I - F)^-1 d`, a ratio whose denominator is one minus the
    system's persistence, and that denominator is about 0.03. Draws near a unit
    root therefore have a resting point amplified a hundredfold, so the median
    is well behaved and the tails mean nothing: the 90% band is 12.4 points wide
    against 2.7 for the 50%. Reporting the wide one would be honest about the
    arithmetic and misleading about the estimate.
    """
    steady = results.definition_is_steady
    prob = 0.50 if steady else 0.90
    band = results.rstar_hdi(prob)
    label = f"{prob:.0%} credible interval"
    ax = mg.fill_between_plot(band, color="cornflowerblue", alpha=0.25, label=label)
    mg.line_plot(
        pd.DataFrame({
            ("r* (steady state)" if steady else f"r* ({results.horizon}q projection)"):
                results.rstar_median(),
            # The null: the same projection with coefficients that never drift.
            # Where it sits inside the band, the drift is buying nothing.
            "Constant-coefficient VAR": results.constant_coefficient_rstar(),
            "Real cash rate": results.observed("real_rate"),
        }),
        ax=ax,
        color=["darkblue", "darkorange", "darkgrey"],
        width=[2.5, 1.5, 1.5],
        style=["-", "--", "-"],
        annotate=True,
        rounding=2,
    )
    mg.finalise_plot(
        ax,
        title="r-star from a TVP-VAR",
        ylabel="Per cent, real",
        y0=True,
        legend={"loc": "best", "fontsize": "small"},
        lheader=(
            "r* is where the VAR comes to rest, not a 20q forecast"
            if steady
            else "No IS curve and no term premium: r* is where the VAR says rates settle"
        ),
        rheader=(
            "50% band: near-unit-root draws make the tails meaningless"
            if steady else ""
        ),
        rfooter=_rfooter(results),
        lfooter=_LFOOTER,
        show=False,
    )


def plot_rstar_nominal(results: TvpVarResults) -> None:
    """Real and nominal r*, on the package's shared conversion."""
    real = results.rstar_median()
    mg.line_plot_finalise(
        pd.DataFrame({
            "Nominal r* (r* + long-run expectations)": to_nominal(real),
            "Real r*": real,
        }),
        color=["darkblue", "darkorange"],
        width=[2.5, 2.5],
        annotate=True,
        rounding=2,
        title="TVP-VAR r-star, real and nominal",
        ylabel="Per cent",
        y0=True,
        legend={"loc": "best", "fontsize": "small"},
        lheader="Nominal carries the same uncertainty as real",
        rfooter=_rfooter(results),
        lfooter=_LFOOTER,
        show=False,
    )


def _projection_median(results: TvpVarResults) -> pd.Series:
    """Return the H-quarter projection even when the steady state is the default.

    `rstar_posterior` short-circuits to the steady state under the shipped
    definition, so the comparator has to be obtained by temporarily asking for
    the other one. Restored afterwards so nothing downstream sees the change.
    """
    saved = results.constants.get("rstar_is_steady")
    results.constants["rstar_is_steady"] = 0.0
    try:
        return results.rstar_median()
    finally:
        if saved is None:
            results.constants.pop("rstar_is_steady", None)
        else:
            results.constants["rstar_is_steady"] = saved


def plot_horizon_check(results: TvpVarResults) -> None:
    """Plot the H-step projection against the steady state it converges to.

    If these two disagree, the reported r* depends on the choice of horizon as
    much as on the data, which is worth seeing rather than asserting.
    """
    # Explicitly the two DEFINITIONS, not r* against itself: under the steady
    # default `rstar_median` already IS the steady state, so comparing it with
    # `unconditional_mean` plotted the same line twice.
    mg.line_plot_finalise(
        pd.DataFrame({
            f"{results.horizon}q projection (CBA's definition)":
                results.rstar_median(conditioned=None) if not results.definition_is_steady
                else _projection_median(results),
            "Steady state (shipped)": results.unconditional_mean(),
        }),
        color=["darkblue", "firebrick"],
        style=["-", "--"],
        width=[2.5, 1.8],
        annotate=True,
        rounding=2,
        title="Does the projection converge?",
        ylabel="Per cent, real",
        y0=True,
        legend={"loc": "best", "fontsize": "small"},
        lheader="A gap between these means the horizon is doing the work",
        rfooter=_rfooter(results),
        lfooter=_LFOOTER,
        show=False,
    )


def plot_inputs(results: TvpVarResults) -> None:
    """Plot the three series the VAR is fitted to."""
    frame = pd.DataFrame({name: results.observed(name) for name in VARIABLES})
    mg.line_plot_finalise(
        frame.rename(columns={
            "inflation": "Trimmed mean inflation",
            "growth": "GDP growth",
            "real_rate": "Real cash rate",
        }),
        color=["darkorange", "seagreen", "darkslategrey"],
        width=[2, 2, 2],
        annotate=True,
        rounding=1,
        title="What the VAR is fitted to",
        ylabel="Per cent",
        y0=True,
        legend={"loc": "best", "fontsize": "small"},
        rfooter=_rfooter(results),
        lfooter=_LFOOTER,
        show=False,
    )


def plot_sigma_q_ensemble(results: TvpVarResults, ensemble: dict[str, pd.DataFrame]) -> None:
    """Plot r* across the assumed drift, with the real cash rate behind it.

    The cash rate is on this chart deliberately. The question the sweep exists to
    answer is whether r* is anything more than a smoothed policy rate, and that
    is a comparison the reader should be able to make by eye rather than from a
    correlation in a table.
    """
    paths = ensemble.get("paths")
    if paths is None or paths.empty:
        return

    ax = mg.line_plot(
        paths.rename(columns=lambda c: f"sigma_q = {c}"),
        width=[1.8] * paths.shape[1],
        annotate=True,
        rounding=2,
    )
    mg.line_plot(
        results.observed("real_rate").rename("Real cash rate"),
        ax=ax,
        color=["black"],
        style=["--"],
        width=[1.5],
        annotate=False,
    )
    mg.finalise_plot(
        ax,
        title="r-star across the assumed coefficient drift",
        ylabel="Per cent, real",
        y0=True,
        legend={"loc": "best", "fontsize": "x-small", "ncol": 2},
        lheader="Spread between lines is the drift assumption, not sampling error",
        rfooter=_rfooter(results),
        lfooter=_LFOOTER,
        show=False,
    )




def run_analysis(
    output_dir: Path | str | None = None,
    prefix: str = "rstar_tvpvar",
    chart_dir: Path | str | None = None,
) -> TvpVarResults:
    """Load a saved run, print the diagnostics, write the charts."""
    results = load_results(output_dir=output_dir, prefix=prefix)
    print_diagnostics(results)

    if chart_dir is None:
        chart_dir = CHART_DIR if prefix == "rstar_tvpvar" else CHART_DIR.parent / f"RStarTVPVAR_{prefix}"
    mg.set_chart_dir(str(chart_dir))
    mg.clear_chart_dir()
    save_diagnostics(results.trace, chart_dir, prefix, model="rstar_tvpvar")

    plot_rstar(results)
    plot_rstar_nominal(results)
    plot_horizon_check(results)
    plot_inputs(results)
    # The four charts every other r* package draws, so this model can be put
    # beside them rather than read on its own terms.
    plot_standard_suite(results, _rfooter(results), _LFOOTER)

    # Drawn only when the sweep has been run. It is the chart that matters most
    # for this model, so its absence is announced rather than silent.
    ensemble = load_ensemble(output_dir=output_dir, prefix=prefix)
    if ensemble is not None:
        plot_sigma_q_ensemble(results, ensemble)
        print_ensemble(ensemble["table"])
    else:
        print("\n  note: no sigma_q sweep on file. Run with --ensemble: the level is")
        print("  conditional on that parameter and a single run cannot show it.")

    print(f"\nCharts written to: {chart_dir}")
    return results
