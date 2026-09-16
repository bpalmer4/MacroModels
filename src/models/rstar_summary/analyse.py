"""The summary chart: every r* this repo produces, on one nominal scale.

The point is the DISAGREEMENT. Each model anchors r* to a different thing and
each returns roughly what its anchor implies, so the spread between these lines
is not sampling error, it is four different structural assumptions. Nothing
here picks a winner.

The nominal cash rate is drawn behind them because it is the comparison anyone
reading an r* chart is actually making, and because a line that tracks it is
reporting policy back rather than measuring neutral.
"""

from pathlib import Path

import mgplot as mg
import pandas as pd

from src.data.cash_rate import get_cash_rate_qrtly
from src.models.common.inflation_scale import scale_label
from src.models.rstar_summary.sources import DEFAULT_SCALE

CHART_DIR = Path(__file__).parent.parent.parent.parent / "charts" / "rstar-summary"

# Chosen for contrast against each other AND against the grey cash rate, which
# matters more than matching each model's own suite colours.
_COLOURS = ("darkblue", "crimson", "darkgreen", "darkorange", "purple", "teal")


def _cash_rate(index: pd.PeriodIndex) -> pd.Series:
    """Return the nominal cash rate on the chart's index."""
    cash = get_cash_rate_qrtly().data.astype(float)
    if isinstance(cash.index, pd.DatetimeIndex):
        cash.index = cash.index.to_period("Q")
    return cash.reindex(index)


def plot_summary(
    frame: pd.DataFrame,
    notes: dict[str, str],
    start: str | None = "1993Q1",
    scale: str = DEFAULT_SCALE,
) -> None:
    """Plot every model's nominal r* together, with the cash rate behind them.

    `scale` names the convention in the header only: the frame arrives already
    nominal, converted by `sources.gather`, so this must be passed the same
    value that built it rather than deciding for itself.
    """
    data = frame.loc[frame.index >= pd.Period(start, freq="Q")] if start else frame
    index = data.index
    if not isinstance(index, pd.PeriodIndex):
        index = pd.PeriodIndex(index, freq="Q")

    ax = mg.line_plot(
        _cash_rate(index).rename("Nominal cash rate"),
        color=["darkgrey"], width=1.5, style="--", annotate=False,
    )
    mg.line_plot(
        data,
        ax=ax,
        color=list(_COLOURS[:len(data.columns)]),
        width=2.0,
        annotate=True,
        rounding=2,
    )
    # NO rheader. It used to list every model's latest value, which ran into the
    # lheader once a fourth model joined, and it was redundant anyway: each line
    # is annotated with its own latest value at the right-hand end.
    mg.finalise_plot(
        ax,
        title="Australian nominal r*: the models that identify one",
        ylabel="Per cent, nominal",
        y0=True,
        legend={"loc": "best", "fontsize": "x-small"},
        lheader=f"All converted to nominal: real estimates plus {scale_label(scale)}",
        rfooter="Built using: ABS; RBA; NY Fed",
        lfooter=(
            f"Australia. Spread between lines is {len(data.columns)} structural "
            "assumptions, not error. "
        ),
        show=False,
    )
    print("\nWhat each line is anchored to:")
    for label, note in notes.items():
        print(f"  {label:<40} {note}")


def plot_spread(frame: pd.DataFrame, start: str | None = "1993Q1") -> None:
    """Plot the range across models, which is the disagreement itself.

    A single number is only as good as the narrowest this gets. Where the band
    is wide the models are saying different things about the same quarter, and
    no amount of within-model precision closes it.
    """
    data = frame.loc[frame.index >= pd.Period(start, freq="Q")] if start else frame
    usable = data.dropna(how="any")
    if usable.empty or len(usable.columns) < 2:  # noqa: PLR2004 — a range needs two
        print("  note: fewer than two models overlap; skipping the spread chart")
        return

    band = pd.DataFrame({"lower": usable.min(axis=1), "upper": usable.max(axis=1)})
    ax = mg.fill_between_plot(band, color="crimson", alpha=0.18, label="Range across models")
    # MEAN, not median. With three series the median is whichever model happens
    # to sit in the middle that quarter, so it switches identity wherever the
    # lines cross (around 2001, 2010 and 2019) and picks up kinks that say
    # nothing about r*. The mean uses all three and moves smoothly.
    #
    # Neither is an estimate. An average across structural assumptions is a
    # value no model produces, which is the objection `rstar_hlw`'s notes make
    # to its own blended median. It describes where the models sit.
    mg.line_plot(
        usable.mean(axis=1).rename("Mean across models"),
        ax=ax, color=["crimson"], width=2.5, annotate=True, rounding=2,
    )
    # The cash rate behind, as on the levels chart. Without it the band is a
    # picture of disagreement with nothing to judge it against: whether a
    # 1.07pp spread matters depends on where policy actually sat relative to it.
    index = usable.index
    if not isinstance(index, pd.PeriodIndex):
        index = pd.PeriodIndex(index, freq="Q")
    mg.line_plot(
        _cash_rate(index).rename("Nominal cash rate"),
        ax=ax, color=["darkgrey"], width=1.0, style="--", annotate=False,
    )
    widest = (band["upper"] - band["lower"])
    mg.finalise_plot(
        ax,
        title="How much the models disagree about nominal r*",
        ylabel="Per cent, nominal",
        y0=True,
        legend={"loc": "best", "fontsize": "small"},
        lheader=f"Across {len(usable.columns)} models, on their common quarters",
        rheader=f"Widest {widest.max():.2f}pp in {widest.idxmax()}; "
                f"latest {widest.iloc[-1]:.2f}pp",
        rfooter="Built using: ABS; RBA; NY Fed",
        lfooter="Australia. The mean is a description of where the models sit, not an estimate. ",
        show=False,
    )


def run_analyse(
    frame: pd.DataFrame,
    notes: dict[str, str],
    chart_dir: Path | str | None = None,
    start: str | None = "1993Q1",
    scale: str = DEFAULT_SCALE,
) -> None:
    """Produce the summary charts."""
    directory = Path(chart_dir) if chart_dir else CHART_DIR
    mg.set_chart_dir(str(directory))
    mg.clear_chart_dir()

    plot_summary(frame, notes, start=start, scale=scale)
    plot_spread(frame, start=start)
    plot_stance(frame, start=start)
    plot_stance_against_inflation(frame, start=start)
    print(f"\nCharts saved to: {directory}")


def _demand_gap(index: pd.PeriodIndex, target: float = 2.5) -> pd.Series:
    """Return the inflation gap with the supply contribution netted out.

    `inflation - target - supply`, where supply is the joint model's Phillips
    decomposition on a four-quarter basis, the same series and annualisation
    `rstar_bonds` uses for its Taylor rule.
    """
    from src.data.inflation import get_trimmed_mean_annual  # noqa: PLC0415
    from src.models.ystar_ustar.results import load_results  # noqa: PLC0415 — optional dependency

    inflation = get_trimmed_mean_annual().data.astype(float)
    inflation.index = pd.PeriodIndex(inflation.index, freq="Q")
    supply = load_results(prefix="ystar_ustar").inflation_decomposition()["supply"].rolling(4).sum()
    return (inflation - target - supply).reindex(index)


def _stances(frame: pd.DataFrame) -> pd.DataFrame:
    """Return each model's implied policy stance, on identical arithmetic.

    `nominal cash rate - nominal r*`, which is the same number as real cash less
    real r*. Every model gets the same subtraction rather than its own stance
    variable: three of the four estimate one natively (`rstar_bonds.g`,
    `rstar_rba.stance`, `rstar_invert.stance`) and the TVP-VAR has none, and
    the native ones are better estimates of each model's own concept but worse
    for comparing across them.
    """
    index = frame.index
    if not isinstance(index, pd.PeriodIndex):
        index = pd.PeriodIndex(index, freq="Q")
    cash = _cash_rate(index)
    return frame.apply(lambda column: cash - column)


def plot_stance(frame: pd.DataFrame, start: str | None = "1993Q1") -> None:
    """Plot how tight each model says policy was.

    What a neutral rate is FOR. A level nobody can pin down is still useful if
    the gap to the actual policy rate tracks what policy was doing.
    """
    data = frame.loc[frame.index >= pd.Period(start, freq="Q")] if start else frame
    stances = _stances(data).dropna(how="all")
    if stances.empty:
        return
    ax = mg.line_plot(
        stances,
        color=list(_COLOURS[:len(stances.columns)]),
        width=2.0,
        annotate=True,
        rounding=2,
    )
    mg.finalise_plot(
        ax,
        title="How tight was policy? Each model's answer",
        ylabel="Percentage points",
        y0=True,
        legend={"loc": "best", "fontsize": "x-small"},
        lheader="Cash rate less nominal r*, same arithmetic for every model. Positive = restrictive",
        rfooter="Built using: ABS; RBA; NY Fed",
        lfooter="Australia. Spread between lines is structural assumptions, not error. ",
        show=False,
    )


def plot_stance_against_inflation(frame: pd.DataFrame, start: str | None = "1993Q1") -> None:
    """Plot each stance against the contemporaneous inflation gap.

    BUILT TO TEST SOMETHING ELSE AND REFUTED THE TEST. The idea was that tight
    policy should coincide with below-target inflation, so stance and gap should
    sit on opposite sides of zero. They do not: they move TOGETHER, in every
    model.

    That is the reaction function, not a failure. Tight policy lowers inflation
    LATER; contemporaneously, high inflation CAUSES tight policy. Reading the
    two at the same date measures the RBA responding, which is what
    `is_curve`'s notes warn about.

    Lagged, the relationship is absent: corr(stance_t, gap_{t+h}) peaks at -0.08
    across `rstar_bonds`, `rstar_rba` and `rstar_tvpvar` at every horizon out to
    20 quarters. `rstar_invert` alone reaches -0.56 at h=8, and that is
    circular, since its stance is built from an asserted negative IS slope.

    Kept because the comovement is the clearest picture the package has of why a
    stance cannot be validated against inflation outcomes on Australian data.
    """
    data = frame.loc[frame.index >= pd.Period(start, freq="Q")] if start else frame
    stances = _stances(data).dropna(how="all")
    if stances.empty:
        return
    try:
        gap = _demand_gap(stances.index)
    except (FileNotFoundError, KeyError, ValueError) as exc:
        print(f"  note: demand gap unavailable ({type(exc).__name__}); chart skipped")
        return
    if gap.dropna().empty:
        return

    ax = mg.line_plot(
        stances,
        color=list(_COLOURS[:len(stances.columns)]),
        width=1.8,
        annotate=False,
    )
    mg.line_plot(
        gap.rename("Inflation gap, supply netted out"),
        ax=ax, color=["black"], width=[2.6], style=[":"], annotate=False,
    )
    mg.finalise_plot(
        ax,
        title="Stance moves with inflation, not against it",
        ylabel="Percentage points",
        y0=True,
        legend={"loc": "best", "fontsize": "x-small", "ncol": 2},
        lheader="Comovement here is the RBA reacting, not policy transmitting",
        rheader="Lagged, the relationship is absent: peak corr -0.08 to h=20",
        rfooter="Built using: ABS; RBA; NY Fed",
        lfooter="Australia. Direction only: the two are not on a common scale. ",
        show=False,
    )
