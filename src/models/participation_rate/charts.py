"""Charts: participation around first rate moves, raw and with controls."""

from typing import Any

import mgplot as mg
import pandas as pd

from src.models.participation_rate import CYCLE_GAP, HMAX, PRE
from src.models.participation_rate.analysis import drift, event_paths

Z90 = 1.645  # two-sided 90% band
TICK_STEP = 3  # months between x ticks
RFOOTER = "ABS 6202, RBA"
YLABEL = "pp change from month before"

COMMON: dict[str, Any] = {
    "xlabel": "Months from decision (0 = decision month)",
    "xticks": list(range(-PRE, HMAX + 1, TICK_STEP)),
    "rfooter": RFOOTER,
    "axvline": {"x": -1, "color": "grey", "linestyle": ":", "linewidth": 1},
    "y0": True,
    "show": False,
}


def _cycles_chart(paths: pd.DataFrame, average: pd.Series, name: str, label: str) -> None:
    """Each cycle in light grey, the average in dark blue."""
    n = paths.shape[1]
    ax = mg.line_plot(paths, color=["lightgrey"] * n, style=["-"] * n, width=1, annotate=False, dropna=False)
    mg.line_plot(average.rename("Average"), ax=ax, color="darkblue", width=2.5, annotate=False)
    lines = ax.get_lines()
    mg.finalise_plot(
        ax,
        title=f"{name} around first RBA {label}",
        ylabel=YLABEL,
        legend={"handles": [lines[0], lines[n]], "labels": [f"Each cycle ({n})", "Average"], "loc": "best"},
        lfooter=f"Australia. First move: none that way in prior {CYCLE_GAP}m. ",
        **COMMON,
    )


def _band(t: pd.DataFrame, prefix: str) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "lo": t[f"{prefix}_b"] - Z90 * t[f"{prefix}_se"],
            "hi": t[f"{prefix}_b"] + Z90 * t[f"{prefix}_se"],
        }
    )


def latest_cycle_by_sex_chart(d: pd.DataFrame, months: pd.PeriodIndex) -> None:
    """Participation by sex around first hikes: the earlier cycles' average against the latest cycle."""
    lines = {}
    for col, sex in [("pr_f", "Women"), ("pr_m", "Men")]:
        paths = event_paths(d, months, "first_hike", col)
        latest = paths.columns[-1]
        lines[f"{sex}, typical"] = paths.drop(columns=latest).mean(axis=1)
        lines[f"{sex}, {latest[:4]}"] = paths[latest]
    n_earlier = paths.shape[1] - 1
    ax = mg.line_plot(
        pd.DataFrame(lines),
        color=["darkorange", "darkorange", "darkblue", "darkblue"],
        style=["--", "-", "--", "-"],
        width=[1.5, 2.5, 1.5, 2.5],
        annotate=False,
    )
    mg.finalise_plot(
        ax,
        title="Participation by sex around first RBA hikes",
        ylabel=YLABEL,
        legend={"loc": "best"},
        lfooter=f"Australia. Typical: average of {n_earlier} earlier cycles. ",
        **COMMON,
    )


SEXES = {"Males": "Men", "Females": "Women"}  # LMS2 label -> chart label, in plot order


def _years_label(years: list[int]) -> str:
    """Compress a sorted list of years into ranges: [2010..2019, 2023..2025] -> '2010-2019, 2023-2025'."""
    runs: list[list[int]] = []
    for y in years:
        if runs and y == runs[-1][-1] + 1:
            runs[-1].append(y)
        else:
            runs.append([y])
    return ", ".join(f"{r[0]}-{r[-1]}" if len(r) > 1 else str(r[0]) for r in runs)


def age_charts(
    diff: pd.DataFrame,
    contribution: pd.DataFrame,
    base: pd.Period,
    end: pd.Period,
    normal_years: list[int],
) -> None:
    """Participation change vs normal by age and sex, as rates and as contributions."""
    window = f"{base.strftime('%b %Y')}-{end.strftime('%b %Y')}"
    footer = f"Australia. Original. Normal: {_years_label(normal_years)}. "
    common: dict[str, Any] = {
        "xlabel": "Age",
        "color": ["darkblue", "darkorange"],
        "legend": {"loc": "best"},
        "y0": True,
        "rfooter": "ABS 6202 LMS2",
        "show": False,
    }
    mg.bar_plot_finalise(
        diff[list(SEXES)].rename(columns=SEXES),
        title=f"Participation by age: {window} vs normal year",
        ylabel="pp, change minus normal-year change",
        annotate=True,
        rounding=1,
        lfooter=footer,
        **common,
    )
    mg.bar_plot_finalise(
        contribution[list(SEXES)].rename(columns=SEXES),
        title=f"Contributions to participation change: {window}",
        ylabel="pp of each sex's rate, vs normal year",
        annotate=True,
        rounding=2,
        lfooter=f"{footer}Change x pop share. ",
        **common,
    )


def make_charts(d: pd.DataFrame, months: pd.PeriodIndex, lp_first: pd.DataFrame, col: str, name: str) -> None:
    """Write the four charts for outcome d[col], titled by name, into the current chart directory."""
    averages = {}
    for ev, label in [("first_hike", "hikes"), ("first_cut", "cuts")]:
        paths = event_paths(d, months, ev, col)
        averages[label] = paths.mean(axis=1)
        _cycles_chart(paths, averages[label], name, label)

    ax = mg.line_plot(
        pd.DataFrame(
            {
                "After first hikes": averages["hikes"],
                "After first cuts": averages["cuts"],
                "All months": drift(d, months, col),
            }
        ),
        color=["darkblue", "darkorange", "grey"],
        style=["-", "-", "--"],
        width=[2.5, 2.5, 1.5],
        annotate=False,
    )
    mg.finalise_plot(
        ax,
        title=f"{name}: first hikes vs first cuts",
        ylabel=YLABEL,
        legend={"loc": "best"},
        lfooter="Australia. Simple averages, no controls. ",
        **COMMON,
    )

    ax = mg.fill_between_plot(_band(lp_first, "hike"), color="darkblue", alpha=0.15, label="Hikes 90% band")
    mg.fill_between_plot(_band(lp_first, "cut"), ax=ax, color="darkorange", alpha=0.15, label="Cuts 90% band")
    mg.line_plot(
        pd.DataFrame({"First hike": lp_first["hike_b"], "First cut": lp_first["cut_b"]}),
        ax=ax,
        color=["darkblue", "darkorange"],
        width=2.5,
        annotate=False,
    )
    mg.finalise_plot(
        ax,
        title=f"{name} after first moves, with controls",
        ylabel="pp vs month before, relative to normal",
        xlabel="Months after decision",
        xticks=list(range(0, HMAX + 1, TICK_STEP)),
        legend={"loc": "best"},
        lfooter="Australia. Local projection, HAC SEs. ",
        rfooter=RFOOTER,
        y0=True,
        show=False,
    )
