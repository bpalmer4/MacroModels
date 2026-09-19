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

from src.data.aofm_loader import get_aofm_5y5y_forward
from src.data.cash_rate import get_cash_rate_qrtly
from src.models.common.inflation_scale import scale_label, to_real
from src.models.rstar_summary.sources import DEFAULT_SCALE, SOURCES

CHART_DIR = Path(__file__).parent.parent.parent.parent / "charts" / "rstar-summary"

# Chosen for contrast against each other AND against the grey cash rate, which
# matters more than matching each model's own suite colours.
_COLOURS = ("darkblue", "crimson", "darkgreen", "darkorange", "purple", "teal")
# Since `rstar_tvpvar` was removed the chart carries two models, and those two
# share an observable. At this count the spread chart says so in its header.
TWO_MODELS = 2

# A quarter counts as complete if it has at least this share of the median
# number of trading days. The AOFM series runs to the current month, so the
# last quarter is otherwise an average over a partial window.
_COMPLETE_QUARTER = 0.8


# This is not a model and collects almost nothing of its own: the r* paths come
# from the models, and the only series loaded here are the cash rate, inflation
# and one model's supply term. Crediting ABS, NY Fed, Bloomberg and the rest
# would claim data these charts never touched, and each model's own footer
# already names its providers. So the footer names the MODELS plus what the
# summary reads itself.
def _footer(*, inflation: bool = False, supply: bool = False) -> str:
    """Return the "built from" line for one chart.

    PER CHART, not one constant for all four. The charts do not share inputs:
    only the stance-against-inflation chart loads inflation, and only that chart
    nets out a supply contribution, which it takes from a THIRD model that never
    appears in `SOURCES` because `_demand_gap` imports it inside the function.
    A single shared footer silently omitted `ystar_ustar` while its output was a
    plotted line on that chart.

    Args:
        inflation: the chart reads trimmed-mean inflation (ABS 6401.0)
        supply: the chart nets out `ystar_ustar`'s Phillips supply term, which
            brings that model's imposed `sigma_okun` and gap definition with it

    """
    models = [*(source.prefix for source in SOURCES), "expectations"]
    if supply:
        models.append("ystar_ustar")
    data = ["RBA F1"]
    if inflation:
        data.append("ABS 6401.0")
    return f"Models: {', '.join(models)}; {'; '.join(data)}"


def _forward_quarterly() -> pd.Series:
    """Return the AOFM 5y5y risk-neutral forward as quarterly means, complete quarters only.

    Daily, averaged to quarters the same way `rstar_bonds` does, so the series
    on this chart is the one the models read rather than a second version of
    it. The BC method for the same reason: it is what both models default to.

    THE PART-QUARTER IS DROPPED. The AOFM series runs to the current month, so
    the final quarter is an average over however many trading days have
    happened, 42 against a typical 63 at the time of writing. Plotting it puts
    a point on the chart that will move for reasons that have nothing to do
    with the market changing its mind, beside a cash rate that is complete.
    """
    daily = get_aofm_5y5y_forward("bc").data.astype(float).dropna()
    grouped = daily.groupby(pd.PeriodIndex(daily.index, freq="Q"))
    counts, means = grouped.count(), grouped.mean()
    if len(counts) and counts.iloc[-1] < _COMPLETE_QUARTER * counts.median():
        means = means.iloc[:-1]
    return means


def plot_forward_against_cash(start: str | None = "1993Q1") -> None:
    """Chart the market's 5y5y forward against the cash rate it is a forecast of.

    NO MODEL OUTPUT ON THIS CHART. It is the raw pair the r* models are built
    on: a market price that is close to the expected average policy rate five
    to ten years out, and the policy rate itself. The point is to let a reader
    see how much of each model's r* is already in the observable before any
    estimation happens, given both models read this series.

    It is not a neutral rate. `get_aofm_5y5y_forward` says why: the forward
    still carries whatever the market believes about the cycle over years five
    to ten, plus whatever premium AOFM's model did not strip.
    """
    forward = _forward_quarterly().rename("AOFM 5y5y risk-neutral forward")
    index = forward.index
    frame = pd.DataFrame({
        forward.name: forward,
        "Cash rate": _cash_rate(index),
    })
    if start:
        frame = frame.loc[pd.Period(start, freq="Q"):]

    spread = (frame.iloc[:, 0] - frame.iloc[:, 1]).dropna()
    mg.line_plot_finalise(
        frame,
        title="The 5y5y forward and the cash rate",
        ylabel="Per cent, nominal",
        color=["darkblue", "darkgrey"],
        style=["-", "--"],
        width=[2.0, 1.5],
        annotate=True,
        rounding=2,
        legend={"loc": "best", "fontsize": "small"},
        lheader=(
            f"Forward less cash rate: latest {spread.iloc[-1]:+.2f}, "
            f"mean {spread.mean():+.2f}pp"
        ),
        lfooter="Australia. Quarterly averages of daily data. Part-quarter dropped. ",
        rfooter="AOFM risk-neutral curve (BC); RBA F1",
        show=False,
    )


def plot_real_cash_rate(
    frame: pd.DataFrame,
    start: str | None = "1993Q1",
    scale: str = DEFAULT_SCALE,
) -> None:
    """Chart the real cash rate on two deflators, against the models' real r*.

    The nominal chart can be read as tightening while the real stance loosens,
    which is what happened across 2025: the cash rate returned to 4.35, the
    same level as 2024Q4, while trimmed mean inflation rose from 2.7 to 3.6,
    so the realised real rate fell from 1.15 to 0.75.

    TWO DEFLATORS BECAUSE THEY ANSWER DIFFERENT QUESTIONS, and over a period
    when inflation moves they part company. Long-run expectations is the
    package's own convention, the one every r* here is converted with, so the
    gap between that line and a model's real r* IS that model's stance.
    Realised trimmed mean is what a borrower actually paid, and is the series
    that erodes when inflation rises.

    The models are plotted in real terms by the inverse of the conversion
    `sources.gather` applied, so nothing is deflated twice.
    """
    from src.data.inflation import get_trimmed_mean_annual  # noqa: PLC0415 — one chart

    data = frame.loc[frame.index >= pd.Period(start, freq="Q")] if start else frame
    index = data.index
    if not isinstance(index, pd.PeriodIndex):
        index = pd.PeriodIndex(index, freq="Q")

    cash = _cash_rate(index)
    inflation = get_trimmed_mean_annual().data.astype(float)
    if isinstance(inflation.index, pd.DatetimeIndex):
        inflation.index = inflation.index.to_period("Q")

    out = pd.DataFrame({
        f"Cash rate less {scale_label(scale)}": to_real(cash, scale=scale),
        "Cash rate less realised trimmed mean": cash - inflation.reindex(index),
    })
    for column in data.columns:
        out[f"{column}, real"] = to_real(data[column], scale=scale)

    models = len(data.columns)
    mg.line_plot_finalise(
        out,
        title="The real cash rate, and real r*",
        ylabel="Per cent, real",
        color=["black", "darkgrey", *_COLOURS[:models]],
        style=["-", "--", *["-"] * models],
        width=[1.8, 1.5, *[2.0] * models],
        annotate=True,
        rounding=2,
        y0=True,
        legend={"loc": "best", "fontsize": "x-small"},
        lheader=(
            "A rising nominal rate can be a falling real one: "
            "the two deflators diverge whenever inflation moves"
        ),
        lfooter="Australia. Real r* is each model's own, deflated back. ",
        rfooter=_footer(inflation=True),
        show=False,
    )



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
        rfooter=_footer(),
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
    # MEAN, not median. With an odd number of series the median is whichever
    # model happens to sit in the middle that quarter, so it switches identity
    # wherever the lines cross and picks up kinks that say nothing about r*. The
    # mean uses every model and moves smoothly.
    #
    # SINCE `rstar_tvpvar` WAS REMOVED (2026-09-17) THERE ARE TWO, so this line
    # is the midpoint of the band drawn above it and carries no information the
    # band does not already show. It is kept because the chart's whole subject
    # is the spread, and a centre makes the spread readable. Do not read it as a
    # consensus: with n = 2 it is an arithmetic midpoint between two models that
    # share the AOFM 5y5y forward.
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
        # With two models left, the shared observable is the thing a reader most
        # needs to know: `rstar_bonds` and `rstar_rba` both read the AOFM 5y5y
        # forward, so part of any agreement is one series counted twice. Said
        # here rather than in a footer because the footers are already full.
        lheader=(
            "Both lines read the same AOFM 5y5y forward"
            if len(usable.columns) == TWO_MODELS
            else f"Across {len(usable.columns)} models, on their common quarters"
        ),
        rheader=f"Widest {widest.max():.2f}pp in {widest.idxmax()}; "
                f"latest {widest.iloc[-1]:.2f}pp",
        rfooter=_footer(),
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
    plot_forward_against_cash(start=start)
    plot_real_cash_rate(frame, start=start, scale=scale)
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
    variable: both remaining models estimate one natively (`rstar_bonds.g` and
    `rstar_rba.stance`), and the native ones are better estimates of each
    model's own concept but worse for comparing across them. The removed
    `rstar_invert` also had one; `rstar_tvpvar` never did.
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
        rfooter=_footer(),
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

    Lagged, the relationship is absent. `corr(stance_t, gap_{t+h})` starts at
    +0.65 (bonds) and +0.55 (rba), decays monotonically, and only crosses zero
    near h=20, reaching -0.047 and -0.029. Transmission would need a NEGATIVE
    correlation at a lag of a year or two; there is none at any horizon.

    The removed `rstar_invert` alone reached -0.56 at h=8, and that was
    circular, since its stance was built from an asserted negative IS slope.

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
        annotate=True,
        rounding=2,
    )
    mg.line_plot(
        gap.rename("Trimmed mean inflation gap (TTY), supply netted out"),
        ax=ax, color=["black"], width=[2.6], style=[":"], annotate=True, rounding=2,
    )
    mg.finalise_plot(
        ax,
        title="Policy tightness stance vs the inflation gap",
        ylabel="Percentage points",
        y0=True,
        # Applies to the DOTTED line only: the stance lines share the axis but
        # are a different quantity, so the band says nothing about them.
        axhspan={
            "ymin": -0.5,
            "ymax": 0.5,
            "color": "green",
            "alpha": 0.10,
            "zorder": 0,
            "label": "Inflation gap within +/-0.5pp of policy target",
        },
        legend={"loc": "best", "fontsize": "x-small", "ncol": 2},
        # The legend names the MODELS, because `_stances` carries their labels
        # through, so without this nothing on the chart says the coloured lines
        # are a rate gap rather than a rate.
        lheader="The cash rate less each model's nominal r*",
        rfooter=_footer(inflation=True, supply=True),
        # Kept SHORT: this chart's rfooter is the longest in the package, since
        # it credits four models plus two series, and a wordier left footer
        # runs straight into it.
        lfooter="Australia. Different quantities: compare shape, not levels. ",
        show=False,
    )
