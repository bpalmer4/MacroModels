"""Chart u* read off the inverted Phillips curve. No estimation."""

import argparse

import mgplot as mg
import pandas as pd

from src.models.regime_ustar.config import DEFAULT_BREAKS, ModelConfig
from src.models.regime_ustar.inversion import (
    DEFAULT_WINDOW,
    DEFAULT_WINDOWS,
    build,
    information_share,
    resolve_beta,
)
from src.models.regime_ustar.observations import build_observations

_LFOOTER = "Australia. Phillips curve inverted quarter by quarter, then smoothed. "

# The readings `long_run_ustar` derives with no estimation and no Phillips
# curve, used here as the outside check. Plotted as points rather than a line:
# they exist only on the quarters its rule selected, and joining them up would
# assert a path between them that the method never claims.
_PLATEAUS: tuple[tuple[str, str, float], ...] = (
    ("1967Q2", "1969Q3", 1.82),
    ("1988Q4", "1989Q4", 6.20),
    ("2002Q4", "2005Q2", 5.57),
    ("2013Q1", "2014Q1", 5.68),
    ("2015Q4", "2019Q4", 5.45),
)


def _plateau_series(index: pd.PeriodIndex) -> pd.Series:
    """Return the long_run_ustar plateau readings, drawn across their own spans."""
    out = pd.Series(float("nan"), index=index)
    for lo, hi, value in _PLATEAUS:
        out.loc[pd.Period(lo, freq="Q"):pd.Period(hi, freq="Q")] = value
    return out.rename("Flat-inflation reading")


def plot_inversion(data: pd.DataFrame, window: int, footers: dict[str, str]) -> None:
    """Chart the smoothed inversion against the unemployment rate and the raw readings."""
    mg.line_plot_finalise(
        pd.DataFrame({
            "Implied, quarter by quarter": data["implied"],
            "Unemployment rate": data["u"],
            f"u* (Henderson {window})": data[f"hma{window}"],
        }),
        color=["lightgrey", "black", "darkorange"],
        width=[1.0, 1.2, 2.8],
        alpha=[0.8, 1.0, 1.0],
        annotate=[False, False, True],
        rounding=2,
        title="u* from the inverted Phillips curve",
        ylabel="Per cent",
        legend={"loc": "best", "fontsize": "small"},
        lheader="Nothing estimated: each quarter's inflation surprise read as a gap",
        rfooter=footers["rfooter"],
        lfooter=footers["lfooter"],
        show=False,
    )


def plot_windows(data: pd.DataFrame, windows: tuple[int, ...], footers: dict[str, str]) -> None:
    """Chart every smoothing window together, because the window is the whole choice."""
    frame = pd.DataFrame({f"Henderson {w} ({w / 4:.1f} yrs)": data[f"hma{w}"] for w in windows})
    frame["Unemployment rate"] = data["u"]
    mg.line_plot_finalise(
        frame,
        color=["darkorange", "seagreen", "purple", "black"],
        width=[2.5, 2.0, 2.0, 1.0],
        style=["-", "-", "-", "-"],
        annotate=True,
        rounding=2,
        title="How much the smoothing decides",
        ylabel="Per cent",
        legend={"loc": "best", "fontsize": "small"},
        lheader="Smoothing harder pulls u* towards a moving average of unemployment",
        rfooter=footers["rfooter"],
        lfooter=footers["lfooter"],
        show=False,
    )


def plot_against_plateaus(
    data: pd.DataFrame, index: pd.PeriodIndex, window: int, footers: dict[str, str],
) -> None:
    """Chart the outside check: a method with no Phillips curve and nothing estimated."""
    mg.line_plot_finalise(
        pd.DataFrame({
            f"u* (Henderson {window})": data[f"hma{window}"],
            "Flat-inflation reading": _plateau_series(index),
        }),
        color=["darkorange", "purple"],
        width=[2.5, 3.5],
        style=["-", "-"],
        annotate=[True, False],
        rounding=2,
        title="Against the flat-inflation readings",
        ylabel="Per cent",
        legend={"loc": "best", "fontsize": "small"},
        lheader="Purple spans are long_run_ustar's rule: no Phillips curve, nothing estimated",
        rfooter=footers["rfooter"],
        lfooter=footers["lfooter"],
        show=False,
    )


def parse_args() -> argparse.Namespace:
    """Read the smoothing and the borrowed slope off the command line."""
    parser = argparse.ArgumentParser(description="u* from the inverted Phillips curve")
    parser.add_argument("--window", type=int, default=DEFAULT_WINDOW, help="Henderson terms, odd")
    parser.add_argument("--windows", nargs="*", type=int, default=list(DEFAULT_WINDOWS))
    parser.add_argument("--beta", type=float, default=None,
                        help="Phillips slope to divide by; default is the saved trace's median")
    parser.add_argument("--breaks", nargs="*", default=list(DEFAULT_BREAKS))
    parser.add_argument("--lag", type=int, default=0)
    parser.add_argument("--inflation", choices=("headline", "trimmed"), default="headline")
    parser.add_argument("--start", default=None)
    parser.add_argument("--end", default=None)
    return parser.parse_args()


def main() -> None:
    """Invert, smooth, chart and report."""
    args = parse_args()
    config = ModelConfig(
        breaks=tuple(args.breaks), lag=args.lag, inflation=args.inflation,
        start=args.start, end=args.end, chart_dir_name="RegimeUStar-inversion",
        prefix="regime_ustar_inversion",
    )
    frame, _regimes, _labels, sources = build_observations(config)
    index = frame.index
    if not isinstance(index, pd.PeriodIndex):
        raise TypeError("the observation frame must carry a quarterly PeriodIndex")
    beta, provenance = resolve_beta(ModelConfig(), args.beta)

    windows = tuple(sorted({*args.windows, args.window}))
    data = build(frame, beta, windows)

    print(f"Sample: {frame.index[0]}-{frame.index[-1]}, {len(frame)} quarters")
    print(f"beta:   {provenance}")
    print("\nIs this more than a smoothed unemployment rate?")
    print(information_share(frame, beta, windows).to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    print("\nAgainst long_run_ustar's flat-inflation readings")
    rows = []
    for lo, hi, value in _PLATEAUS:
        row = {"episode": f"{lo}-{hi}", "reading": value}
        for w in windows:
            row[f"hma{w}"] = float(data[f"hma{w}"].loc[lo:hi].mean())
        rows.append(row)
    print(pd.DataFrame(rows).to_string(index=False, float_format=lambda v: f"{v:.2f}"))

    latest = data[[f"hma{w}" for w in windows]].dropna().iloc[-1]
    print(f"\nLatest ({data.index[-1]}): " + ", ".join(f"{k} {v:.2f}" for k, v in latest.items()))

    mg.set_chart_dir(str(config.chart_dir))
    mg.clear_chart_dir()
    footers = {"rfooter": sources.footer(), "lfooter": _LFOOTER}
    plot_inversion(data, args.window, footers)
    plot_windows(data, windows, footers)
    plot_against_plateaus(data, index, args.window, footers)
    print(f"\nCharts written to: {config.chart_dir}")


if __name__ == "__main__":
    main()
