"""Charts of what the cash rate actually costs a borrower and pays a saver.

The `rstar` model measures the policy stance as the real cash rate less r*.
That is only the stance the economy faces if the cash rate summarises the price
of credit, and after the GFC it stopped doing so: between 2004-07 and 2015-19
the discounted mortgage rate went from 1.24 above the cash rate to 2.99 above,
while banks' term deposit rates went from 1.68 *below* it to 0.39 above.

The second number is larger than the first, which is the point of these charts.
The widening in mortgage spreads is not margin and it is not wholesale funding
(the 90-day bill spread barely moves). It is the marginal funding dollar
repricing: before the GFC banks funded cheaply offshore and paid retail
depositors poorly, and afterwards, with liquidity rules pushing them toward
stable deposits, they had to compete for them.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import mgplot as mg
import pandas as pd

from src.data.cash_rate import get_cash_rate_qrtly
from src.data.rba_loader import get_bank_bill_rate, get_deposit_rate, get_lending_rate

CHART_DIR = Path(__file__).parent.parent.parent.parent / "charts" / "BankCosts"

_RFOOTER = "Built using: RBA F1/F4/F5"
_LFOOTER = "Australia. Bank funding and lending rates. "

# The eras the note in the module docstring compares. Pre-GFC starts 2004Q2
# because that is where the discounted mortgage series begins.
ERAS = {
    "2004-2007": ("2004Q2", "2007Q4"),
    "2012-2014": ("2012Q1", "2014Q4"),
    "2015-2019": ("2015Q1", "2019Q4"),
    "2020-2021": ("2020Q1", "2021Q4"),
    "2022-": ("2022Q1", None),
}


def _quarterly(series: pd.Series) -> pd.Series:
    """Put a monthly RBA series on a quarterly PeriodIndex, end of quarter."""
    index = series.index
    if not isinstance(index, pd.PeriodIndex):
        index = pd.PeriodIndex(index, freq="M")
    series = series.copy()
    series.index = index
    return series.groupby(series.index.asfreq("Q")).last().astype(float)


def build_rates() -> pd.DataFrame:
    """Return the cash rate with the lending and deposit rates beside it."""
    frame = pd.DataFrame({
        "Cash rate": get_cash_rate_qrtly().data.astype(float),
        "Mortgage (discounted, owner-occupier)": _quarterly(get_lending_rate("housing_oo").data),
        "Mortgage (investor)": _quarterly(get_lending_rate("housing_investor").data),
        "Term deposit (average of terms)": _quarterly(get_deposit_rate("term_deposit").data),
        "Online saver": _quarterly(get_deposit_rate("online_saver").data),
        "90-day bank bill": _quarterly(get_bank_bill_rate(90).data),
    })
    return frame.dropna(subset=["Cash rate"])


def spreads_to_cash(rates: pd.DataFrame) -> pd.DataFrame:
    """Return every rate as a spread to the cash rate."""
    others = [c for c in rates.columns if c != "Cash rate"]
    return rates[others].sub(rates["Cash rate"], axis=0)


def era_table(spreads: pd.DataFrame) -> pd.DataFrame:
    """Return era means of the spreads, the numbers the charts are drawn from."""
    rows = {}
    for label, (start, end) in ERAS.items():
        window = spreads.loc[start:] if end is None else spreads.loc[start:end]
        rows[label] = window.mean()
    return pd.DataFrame(rows).T


def plot_levels(rates: pd.DataFrame) -> None:
    """Plot the three rates that matter, in levels."""
    data = rates[[
        "Cash rate",
        "Mortgage (discounted, owner-occupier)",
        "Term deposit (average of terms)",
    ]].dropna()
    mg.line_plot_finalise(
        data,
        title="The cash rate, what borrowers pay, and what savers get",
        ylabel="Per cent",
        color=["black", "darkred", "darkgreen"],
        width=[1.5, 2.5, 2.5],
        style=["-", "-", "--"],
        annotate=True,
        rounding=2,
        legend={"loc": "best", "fontsize": "small"},
        lheader="The gap between the black and red lines is what the cash rate does not tell you",
        rfooter=_RFOOTER,
        lfooter=_LFOOTER,
        show=False,
    )


def plot_spreads(spreads: pd.DataFrame) -> None:
    """Every rate as a spread to the cash rate: the whole argument on one axes."""
    data = spreads[[
        "Mortgage (discounted, owner-occupier)",
        "Term deposit (average of terms)",
        "Online saver",
        "90-day bank bill",
    ]].dropna(how="all")
    # The bill spread runs back to 1976 and its pre-1985 deregulation swings
    # span -3.2 to +4.1, which flattens everything this chart is about. Start
    # where the lending series does.
    first = data["Mortgage (discounted, owner-occupier)"].first_valid_index()
    if first is not None:
        data = data.loc[first:]
    mg.line_plot_finalise(
        data,
        title="Spreads to the cash rate",
        ylabel="Percentage points",
        color=["darkred", "darkgreen", "seagreen", "grey"],
        width=[2.5, 2.5, 1.5, 1.5],
        style=["-", "--", ":", "-"],
        annotate=True,
        rounding=2,
        y0=True,
        legend={"loc": "best", "fontsize": "small"},
        lheader="The bill spread barely moves, so this is not wholesale funding",
        rfooter=_RFOOTER,
        lfooter=_LFOOTER,
        show=False,
    )


def plot_funding_against_lending(spreads: pd.DataFrame) -> None:
    """Plot the core claim: deposits repriced by more than mortgages did."""
    data = spreads[[
        "Mortgage (discounted, owner-occupier)",
        "Term deposit (average of terms)",
    ]].dropna()
    mg.line_plot_finalise(
        data,
        title="Mortgage and deposit spreads to the cash rate",
        ylabel="Percentage points",
        color=["darkred", "darkgreen"],
        width=[2.5, 2.5],
        style=["-", "--"],
        annotate=True,
        rounding=2,
        y0=True,
        legend={"loc": "best", "fontsize": "small"},
        lheader="2004-07 to 2015-19: mortgages +1.74, term deposits +2.07. "
                "The funding cost rose by more than the loan rate",
        rfooter=_RFOOTER,
        lfooter=_LFOOTER,
        show=False,
    )


def plot_investor_gap(rates: pd.DataFrame) -> None:
    """Investor over owner-occupier: the macroprudential period, priced."""
    gap = (
        rates["Mortgage (investor)"] - rates["Mortgage (discounted, owner-occupier)"]
    ).dropna().rename("Investor less owner-occupier")
    if gap.empty:
        print("  skipping investor chart: the investor series is unavailable")
        return
    mg.line_plot_finalise(
        gap,
        title="What investors paid over owner-occupiers",
        ylabel="Percentage points",
        color=["darkorange"],
        width=2.5,
        annotate=True,
        rounding=2,
        y0=True,
        legend={"loc": "best", "fontsize": "small"},
        lheader="APRA's investor and interest-only caps ran from December 2014 to 2019",
        rfooter=_RFOOTER,
        lfooter=_LFOOTER,
        show=False,
    )


def plot_spread_against_cash_level(rates: pd.DataFrame, spreads: pd.DataFrame) -> None:
    """Plot the mortgage spread against the *level* of the cash rate.

    The test of whether the constraint was a boundary rather than a trend. If
    the widening were a post-GFC time trend, the tightening since 2022 could
    not retrace it, because a trend does not go backwards. It does retrace,
    which is what a constraint binding near zero and releasing as rates rise
    looks like.

    Raw matplotlib for the scatter because mgplot has no scatter function;
    `finalise_plot` still does the styling and the save.
    """
    frame = pd.DataFrame({
        "cash": rates["Cash rate"],
        "spread": spreads["Mortgage (discounted, owner-occupier)"],
    }).dropna()
    if frame.empty:
        print("  skipping the level chart: no overlapping data")
        return

    # Three groups, not two. Splitting only on the direction of rates hides the
    # 2008-09 level shift: at a cash rate of 4 to 4.5 the spread was about 1.2
    # before the GFC and about 2.3 after it, so the same x maps to two different
    # regimes. The post-2022 points land on the post-GFC cloud, not the pre-GFC
    # one, which says the structural repricing is permanent and only the extra
    # widening near zero has retraced.
    groups = (
        (frame.loc[:"2008Q2"], "darkred", "o", "2004-2008, before the GFC"),
        (frame.loc["2008Q3":"2021Q4"], "darkorange", "s", "2008-2021, rates falling"),
        (frame.loc["2022Q1":], "darkblue", "^", "2022-, rates rising"),
    )
    _, ax = plt.subplots()
    for data, colour, marker, label in groups:
        ax.scatter(data["cash"], data["spread"], s=26, color=colour,
                   alpha=0.8, marker=marker, label=label)
    for period in ("2004Q2", "2011Q4", "2021Q2", "2026Q3"):
        if period in frame.index:
            row = frame.loc[period]
            ax.annotate(str(period), (row["cash"], row["spread"]),
                        fontsize="x-small", xytext=(4, 4), textcoords="offset points")
    ax.set_xlabel("Cash rate, per cent")

    mg.finalise_plot(
        ax,
        title="The mortgage spread against the level of the cash rate",
        ylabel="Mortgage spread to cash, percentage points",
        legend={"loc": "best", "fontsize": "small"},
        lheader="Two things at once: a permanent level shift at the GFC, and on top of it "
                "a widening near zero that has since retraced",
        rfooter=_RFOOTER,
        lfooter=_LFOOTER,
        show=False,
    )


def run_analysis(chart_dir: Path | str | None = None) -> pd.DataFrame:
    """Print the era table and write the charts. Returns the spreads."""
    rates = build_rates()
    spreads = spreads_to_cash(rates)

    print("\nSpreads to the cash rate, era means")
    print("-" * 70)
    print(era_table(spreads).round(2).to_string())

    mortgage = "Mortgage (discounted, owner-occupier)"
    deposit = "Term deposit (average of terms)"
    early, late = ERAS["2004-2007"], ERAS["2015-2019"]
    shift = {
        name: (
            spreads.loc[late[0]:late[1], name].mean()
            - spreads.loc[early[0]:early[1], name].mean()
        )
        for name in (mortgage, deposit)
    }
    print(f"\n2004-07 to 2015-19: mortgage spread {shift[mortgage]:+.2f}, "
          f"deposit spread {shift[deposit]:+.2f}")
    print("The funding cost rose by more than the loan rate, so the widening in "
          "mortgage\nspreads is not margin: it is the marginal funding dollar repricing.")

    mg.set_chart_dir(str(chart_dir or CHART_DIR))
    mg.clear_chart_dir()
    plot_levels(rates)
    plot_spreads(spreads)
    plot_funding_against_lending(spreads)
    plot_investor_gap(rates)
    plot_spread_against_cash_level(rates, spreads)
    print(f"\nCharts written to: {chart_dir or CHART_DIR}")
    return spreads
