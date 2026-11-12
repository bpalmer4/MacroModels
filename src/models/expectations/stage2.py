"""Inflation expectations Stage 2: Diagnostics and plotting.

This module handles:
- Loading saved results from Stage 1
- Running diagnostics
- Generating all analysis plots

Run with: uv run python -m src.models.expectations.stage2
"""

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypedDict

import arviz as az
import mgplot as mg
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib.axes import Axes
from mgplot.utilities import get_color_list

from src.data.cash_rate import get_cash_rate_monthly
from src.data.inflation import get_trimmed_mean_annual
from src.data.rba_loader import get_inflation_expectations
from src.models.common.diagnostics import check_model_diagnostics
from src.models.common.results import posterior_summary
from src.models.common.timeseries import plot_posterior_timeseries
from src.models.expectations.common import CHART_DIR, LEGEND_LABELS, MODEL_NAMES, MODEL_TYPES, OUTPUT_DIR
from src.models.expectations.stage1 import MODEL_CONFIGS

# The unanchored model's lineage. Only on charts where it is the sole model
# drawn: on the comparison charts it would be read as covering the short and
# market lines too, which share the random walk but not the biased survey panel.
UNANCHORED_LFOOTER = (
    "Australia. Trend inflation after Chan, Clark & Koop (2018); "
    "biased surveys, bonds, inflation, wages; no target anchor."
)
THREE_MEASURES_LFOOTER = "Australia."

# --- Survey benchmark ---

BENCHMARK_CHART_START = "1996Q1"
# PIE_RBAQ ends 2019Q1; the comparison starts once inflation targeting is settled.
PIE_WINDOW = ("1998Q1", "2019Q1")
TARGET_BAND = (2.0, 3.0)
BAND_LOWER, BAND_UPPER = 0.05, 0.95
QUARTERS_PER_YEAR = 4
BENCHMARK_DISCLAIMER = "Reconstruction after Cusbert (2017); not an RBA series."


# The survey benchmark's mean-reset window, from its config.
RESET_START = pd.Period(MODEL_CONFIGS["cusbert"]["mean_reset"]["start"], freq="Q")
RESET_END = pd.Period(MODEL_CONFIGS["cusbert"]["mean_reset"]["end"], freq="Q")
RESET_LABEL = f"mean reset to inflation over {RESET_START}-{RESET_END}"
# --- Results Container ---


@dataclass
class ExpectationsResults:
    """Container for loaded results."""

    trace: xr.DataTree
    measures: pd.DataFrame
    inflation: pd.Series
    index: pd.PeriodIndex
    model_type: str
    monthly: bool = False

    @property
    def is_monthly(self) -> bool:
        """Whether the model was estimated at monthly frequency."""
        return self.monthly or self.index.freqstr.startswith("M")

    @property
    def last_complete(self) -> pd.Period:
        """Last period with published inflation, where charts end.

        The sample runs on to the latest survey reading, which is a quarter
        still in progress with only some of its inputs; charts stop before it.
        """
        last = self.inflation.dropna().index[-1]
        if not isinstance(last, pd.Period):
            raise TypeError("inflation needs a PeriodIndex")
        return last

    def expectations_posterior(self, var: str = "pi_exp") -> pd.DataFrame:
        """Get full posterior samples at native frequency."""
        samples = self.trace.posterior[var].to_numpy()
        n_chains, n_draws, n_time = samples.shape
        flat = samples.reshape(n_chains * n_draws, n_time).T
        return pd.DataFrame(flat, index=self.index)

    def expectations_posterior_quarterly(self) -> pd.DataFrame:
        """Get posterior samples at quarterly frequency (Mar/Jun/Sep/Dec).

        For quarterly models this is the same as expectations_posterior().
        For monthly models, extracts quarter-end months and relabels to quarterly.
        """
        post = self.expectations_posterior()
        if not self.is_monthly:
            return post
        if not isinstance(post.index, pd.PeriodIndex):
            raise TypeError("expectations posterior needs a PeriodIndex")
        qtr_mask = post.index.month.isin([3, 6, 9, 12])
        post_q = post.loc[qtr_mask].copy()
        post_q.index = post.index[qtr_mask].to_timestamp().to_period("Q")
        return post_q

    def expectations_median(self) -> pd.Series:
        """Get posterior median."""
        return self.expectations_posterior().median(axis=1)

    def expectations_hdi(self, prob: float = 0.9) -> pd.DataFrame:
        """Get HDI bounds."""
        post = self.expectations_posterior()
        alpha = (1 - prob) / 2
        return pd.DataFrame({
            "lower": post.quantile(alpha, axis=1),
            "median": post.median(axis=1),
            "upper": post.quantile(1 - alpha, axis=1),
        }, index=self.index)

    def expectations_hdi_quarterly(self, prob: float = 0.9) -> pd.DataFrame:
        """Get HDI bounds at quarterly frequency."""
        post = self.expectations_posterior_quarterly()
        alpha = (1 - prob) / 2
        return pd.DataFrame({
            "lower": post.quantile(alpha, axis=1),
            "median": post.median(axis=1),
            "upper": post.quantile(1 - alpha, axis=1),
        }, index=post.index)


# --- Loading ---


def load_results(model_type: str, output_dir: Path | None = None) -> ExpectationsResults:
    """Load saved results from stage1."""
    output_dir = output_dir or OUTPUT_DIR

    # Load trace
    trace = az.from_netcdf(output_dir / f"expectations_{model_type}_trace.nc")

    # Load metadata
    meta_path = output_dir / f"expectations_{model_type}_metadata.pkl"
    with meta_path.open("rb") as f:
        metadata = pickle.load(f)

    return ExpectationsResults(
        trace=trace,
        measures=metadata["measures"],
        inflation=metadata["inflation"],
        index=metadata["index"],
        model_type=model_type,
        monthly=metadata.get("monthly", False),
    )


def load_all_results(output_dir: Path | None = None) -> dict[str, ExpectationsResults]:
    """Load all available model results, skipping missing ones."""
    results = {}
    for model_type in MODEL_TYPES:
        try:
            results[model_type] = load_results(model_type, output_dir)
        except FileNotFoundError:
            print(f"  Skipping {model_type}: no saved results found")
    return results


# --- Diagnostics ---


def run_diagnostics(results: ExpectationsResults, verbose: bool = True) -> None:
    """Run and print diagnostics for a model."""
    if verbose:
        print(f"\n{'=' * 60}")
        print(f"Diagnostics: {MODEL_NAMES[results.model_type]}")
        print("=" * 60)

        check_model_diagnostics(results.trace)

        # Parameter estimates
        trace_vars = list(results.trace.posterior.data_vars)
        summary_vars = [
            var for var in ["alpha", "lambda_bias", "sigma_obs", "sigma_inflation", "b_inflation",
                            "sigma_headline", "b_headline", "sigma_early", "sigma_late"]
            if var in trace_vars
        ]

        if summary_vars:
            print("\nParameter Estimates:")
            print(posterior_summary(results.trace, var_names=summary_vars))
        else:
            print("\nNo sampled parameters (all fixed).")

        # Latest values
        if results.is_monthly:
            hdi = results.expectations_hdi_quarterly()
            print("\nLatest Expectations (90% HDI, quarterly extract):")
        else:
            hdi = results.expectations_hdi()
            print("\nLatest Expectations (90% HDI):")
        print(hdi.tail(4))


def _to_index_freq(df: pd.DataFrame, target_index: pd.PeriodIndex) -> pd.DataFrame:
    """Put a DataFrame on the frequency of `target_index` for comparison plots.

    Unchanged when the frequencies already match. A quarterly frame going onto
    a monthly index is placed at quarter-end months and interpolated between
    them, for smooth curves.
    """
    source = df.index
    if not isinstance(source, pd.PeriodIndex):
        raise TypeError("expected a PeriodIndex")
    if source.freqstr[0] == target_index.freqstr[0]:
        return df
    df = df.copy()
    df.index = source.asfreq("M", how="E")
    return df.reindex(target_index).interpolate(method="linear")


# --- Plotting ---

# Common plot styling
class _LevelPlotKwargs(TypedDict):
    """The finalise_plot settings every expectations-level chart shares."""

    ylabel: str
    axhspan: dict[str, Any]
    axhline: dict[str, Any]
    legend: dict[str, Any]


PLOT_KWARGS: _LevelPlotKwargs = {
    "ylabel": "Per cent",
    "axhspan": {"ymin": TARGET_BAND[0], "ymax": TARGET_BAND[1], "color": "red", "alpha": 0.1, "zorder": -1},
    "axhline": {"y": 2.5, "color": "black", "linestyle": "dashed", "linewidth": 0.75},
    "legend": {"loc": "best", "fontsize": "x-small"},
}


def _match_freq(series: pd.Series, target_index: pd.PeriodIndex) -> pd.Series:
    """Convert overlay series to match the target index frequency."""
    if not isinstance(series.index, pd.PeriodIndex):
        return series
    target_freq = target_index.freqstr[0]  # "M" or "Q"
    source_freq = series.index.freqstr[0]
    if source_freq == target_freq:
        return series
    if target_freq == "Q" and source_freq == "M":
        # Monthly → quarterly: take end-of-quarter values
        s = series.copy()
        s.index = series.index.to_timestamp().to_period("Q")
        return s.groupby(s.index).last()
    if target_freq == "M" and source_freq == "Q":
        # Quarterly → monthly: place at quarter-end month
        s = series.copy()
        s.index = series.index.asfreq("M", how="E")
        return s.reindex(target_index)
    return series


def _plot_model(
    results: ExpectationsResults,
    title: str,
    lfooter: str,
    legend_stem: str,
    *,
    overlays: list[pd.Series] | None = None,
    axvspan: dict | None = None,  # Empty dict if no span needed
    start: pd.Period | None = None,
) -> None:
    """Plot a single model's posterior with optional overlays, from `start` to the last complete period.

    Colours come from the mgplot palette: the model takes the first, the overlays the rest in order.
    """
    overlays = overlays or []
    model_color, *overlay_colors = get_color_list(1 + len(overlays))
    posterior = results.expectations_posterior()
    first = results.index[0] if start is None else start
    shown = (posterior.index >= first) & (posterior.index <= results.last_complete)
    posterior = posterior.loc[shown]
    ax = plot_posterior_timeseries(data=posterior, legend_stem=legend_stem, color=model_color, finalise=False)
    if not isinstance(ax, Axes):
        raise TypeError("plot_posterior_timeseries returned no Axes")

    if overlays:
        for series, color in zip(overlays, overlay_colors, strict=True):
            matched = _match_freq(series, results.index)
            # Overlays cover the same span as the model, which may begin later
            # than the series it is drawn against.
            matched = matched[(matched.index >= first) & (matched.index <= results.last_complete)]
            mg.line_plot(matched, ax=ax, color=color, width=1.5, annotate=False, zorder=5)

    mg.finalise_plot(
        ax,
        title=title,
        lfooter=lfooter,
        rfooter=f"Sample: {posterior.index[0]} to {posterior.index[-1]}",
        axvspan=axvspan,
        **PLOT_KWARGS,
    )


def plot_real_cash_rate(results: ExpectationsResults) -> None:
    """Chart the cash rate, and the cash rate less two measures of inflation.

    The argument over whether policy is tight is largely an argument over which
    inflation it should be deflated by: what inflation IS (trimmed mean) or what
    people EXPECT (this model's median). The two real rates part company
    whenever inflation runs away from expectations.

    Monthly, because the cash rate is: a quarterly model and trimmed mean are
    each held across their quarter's three months. All three are drawn as
    steps, since the cash rate moves only at decisions.
    """
    expectations = results.expectations_posterior().median(axis=1)
    expectations = expectations[expectations.index <= results.last_complete]
    if not isinstance(expectations.index, pd.PeriodIndex):
        return
    if expectations.index.freqstr.startswith("M"):
        months = expectations.index
    else:
        months = pd.period_range(expectations.index[0].asfreq("M", how="start"),
                                 expectations.index[-1].asfreq("M", how="end"), freq="M")
        expectations = pd.Series(expectations.reindex(months.asfreq("Q")).to_numpy(), index=months)
    months = months[months <= pd.Period(pd.Timestamp.today(), freq="M") - 1]

    cash = get_cash_rate_monthly().data.astype(float)
    if not isinstance(cash.index, pd.PeriodIndex):
        cash.index = pd.PeriodIndex(cash.index, freq="M")
    cash = cash.reindex(months).rename("Cash rate")
    trimmed = get_trimmed_mean_annual().data.astype(float)
    if not isinstance(trimmed.index, pd.PeriodIndex):
        trimmed.index = pd.PeriodIndex(trimmed.index, freq="Q")
    trimmed_held = pd.Series(trimmed.reindex(months.asfreq("Q")).to_numpy(), index=months)

    real = pd.DataFrame({
        "Cash rate less trimmed mean inflation": cash - trimmed_held,
        "Cash rate less trend expectations": cash - expectations.reindex(months),
    })
    ax = mg.line_plot(cash, color="darkgrey", style="--", width=1.5, drawstyle="steps-post",
                      annotate=True, rounding=2)
    mg.line_plot(real, ax=ax, width=2.0, drawstyle="steps-post", annotate=True, rounding=2)
    mg.finalise_plot(
        ax,
        title="The cash rate, nominal and real",
        ylabel="Per cent",
        y0=True,
        legend={"loc": "best", "fontsize": "x-small"},
        lfooter="Australia. Monthly; quarterly series held across their months. ",
        rfooter="RBA F1; ABS 6401.0; expectations model",
        show=False,
    )


def generate_plots(
    all_results: dict[str, ExpectationsResults],
    chart_dir: Path | None = None,
) -> None:
    """Generate all analysis plots."""
    chart_dir = chart_dir or CHART_DIR
    mg.set_chart_dir(str(chart_dir))
    mg.clear_chart_dir()

    # Use first available result for shared data
    first_result = next(iter(all_results.values()))
    monthly = first_result.is_monthly

    # Prepare overlay data — reindex quarterly series onto the monthly grid
    # so mgplot sees a contiguous PeriodIndex (NaN at non-quarter months).
    # A copy: the loader's result is cached, and re-indexing it in place would
    # hand every later caller in this run a monthly index.
    trimmed = get_trimmed_mean_annual().data.copy()
    if monthly:
        if not isinstance(trimmed.index, pd.PeriodIndex):
            raise TypeError("trimmed mean needs a PeriodIndex")
        trimmed.index = trimmed.index.asfreq("M", how="E")
        trimmed = trimmed.reindex(first_result.index)
    trimmed = trimmed[trimmed.index >= first_result.index[0]]
    trimmed.name = "Trimmed Mean Inflation"

    market_1y = first_result.measures["market_1y"].dropna()
    market_1y.name = "Market Economists (1yr)"
    breakeven_series = first_result.measures["breakeven"].dropna()
    breakeven_series.name = "Breakeven (10yr)"

    pie_rbaq = get_inflation_expectations().data
    if monthly:
        if not isinstance(pie_rbaq.index, pd.PeriodIndex):
            raise TypeError("PIE_RBAQ needs a PeriodIndex")
        pie_rbaq.index = pie_rbaq.index.asfreq("M", how="E")
        pie_rbaq = pie_rbaq.reindex(first_result.index)
    pie_rbaq = pie_rbaq[pie_rbaq.index >= first_result.index[0]]
    pie_rbaq.name = "RBA PIE_RBAQ"

    # Single-model plot specs: (model_type, title, lfooter, legend_stem, overlays, axvspan_fn)
    # (model_type, title, lfooter, legend_stem, overlays, axvspan_fn, start_fn)
    plot_specs = [
        ("unanchored", "Inflation Expectations",
         UNANCHORED_LFOOTER,
         LEGEND_LABELS["unanchored"], [trimmed], None, None),
        ("unanchored", "Inflation Expectations vs RBA PIE_RBAQ",
         UNANCHORED_LFOOTER,
         LEGEND_LABELS["unanchored"], [pie_rbaq], None, None),
        # Starts with the survey: before it, a fixed early step size meets the
        # headline-CPI proxy's level shift and the walk drops several points at once.
        ("short", "Short Run Inflation Expectations (1 Year)",
         "Australia. Bayesian signal extraction from market economist 1-year expectations.",
         LEGEND_LABELS["short"], [market_1y, trimmed], None,
         lambda r: r.measures["market_1y"].dropna().index[0]),
        # Breakeven-implied: its 2008Q4 fall is bond-market liquidity as much as expectations.
        ("market", "Breakeven-Implied Inflation Expectations (10-Year)",
         "Australia. Bayesian signal extraction from breakeven inflation and nominal bonds.",
         LEGEND_LABELS["market"], [breakeven_series, trimmed],
         lambda r: {"xmin": r.index[0].ordinal,
                    "xmax": r.measures["breakeven"].dropna().index[0].ordinal,
                    "color": "goldenrod", "alpha": 0.2, "label": "Proxy period (nominal bonds)"}, None),
        ("cusbert", "Survey Benchmark Inflation Expectations",
         f"Australia. Cusbert-style, surveys only; {RESET_LABEL}. Not an RBA series.",
         LEGEND_LABELS["cusbert"], [trimmed], None, None),
    ]

    for model_type, title, lfooter, legend_stem, overlays, axvspan_fn, start_fn in plot_specs:
        if model_type in all_results:
            results = all_results[model_type]
            axvspan = axvspan_fn(results) if axvspan_fn else None
            start = start_fn(results) if start_fn else None
            _plot_model(results, title, lfooter, legend_stem, overlays=overlays, axvspan=axvspan, start=start)

    plot_comparisons(all_results)

    if "unanchored" in all_results:
        plot_real_cash_rate(all_results["unanchored"])

    plot_survey_benchmark(all_results)

    print(f"\nCharts saved to: {chart_dir}")


def plot_comparisons(all_results: dict[str, ExpectationsResults]) -> None:
    """Chart trend, short-run and long-run expectations together: distributions, then medians."""
    if not all(k in all_results for k in ("unanchored", "short", "market")):
        return
    shared_index = all_results["unanchored"].index
    posterior_unanchored = all_results["unanchored"].expectations_posterior()
    posterior_short = all_results["short"].expectations_posterior()
    # Market always runs quarterly; match the others' frequency for a shared axis
    posterior_market = _to_index_freq(
        all_results["market"].expectations_posterior(), shared_index,
    )
    # Both charts end at the last complete period, not the part-observed final quarter.
    end = all_results["unanchored"].last_complete
    posterior_unanchored = posterior_unanchored.loc[posterior_unanchored.index <= end]
    posterior_short = posterior_short.loc[posterior_short.index <= end]
    posterior_market = posterior_market.loc[posterior_market.index <= end]
    sample = f"Sample: {posterior_unanchored.index[0]} to {end}"

    # Distributions: a median and one 90% band per model.
    trend_color, short_color, market_color = get_color_list(3)
    ax = plot_posterior_timeseries(data=posterior_unanchored, legend_stem=LEGEND_LABELS["unanchored"],
                                   color=trend_color, cuts=(BAND_LOWER,), alphas=(0.2,), finalise=False)
    ax = plot_posterior_timeseries(data=posterior_short, legend_stem=LEGEND_LABELS["short"],
                                   color=short_color, cuts=(BAND_LOWER,), alphas=(0.2,), ax=ax, finalise=False)
    ax = plot_posterior_timeseries(data=posterior_market, legend_stem=LEGEND_LABELS["market"],
                                   color=market_color, cuts=(BAND_LOWER,), alphas=(0.2,), ax=ax, finalise=False)
    if not isinstance(ax, Axes):
        raise TypeError("plot_posterior_timeseries returned no Axes")
    mg.finalise_plot(
        ax,
        title="Inflation Expectations: Trend, Short Run and Long Run",
        lfooter=THREE_MEASURES_LFOOTER,
        rfooter=sample,
        **PLOT_KWARGS,
    )

    # Medians comparison
    medians = [
        (posterior_unanchored.median(axis=1), LEGEND_LABELS["unanchored"]),
        (posterior_short.median(axis=1), LEGEND_LABELS["short"]),
        (posterior_market.median(axis=1), LEGEND_LABELS["market"]),
    ]
    lfooter = THREE_MEASURES_LFOOTER
    # The survey benchmark joins the medians when it has been run; it starts later than the others.
    if "cusbert" in all_results:
        posterior_benchmark = _to_index_freq(all_results["cusbert"].expectations_posterior(), shared_index)
        posterior_benchmark = posterior_benchmark.loc[posterior_benchmark.index <= end]
        medians.append((posterior_benchmark.median(axis=1), LEGEND_LABELS["cusbert"]))
        lfooter = "Australia. Survey-only series: Cusbert-style reconstruction; not an RBA series."
    ax = None
    for (series, name), color in zip(medians, get_color_list(len(medians)), strict=True):
        series.name = name
        ax = mg.line_plot(series, ax=ax, color=color, width=2, annotate=False)
    if not isinstance(ax, Axes):
        raise TypeError("line_plot returned no Axes")
    mg.finalise_plot(
        ax,
        title="Inflation Expectations: Median Comparison",
        lfooter=lfooter,
        rfooter=sample,
        **PLOT_KWARGS,
    )


def plot_survey_benchmark(all_results: dict[str, ExpectationsResults]) -> None:
    """Chart the survey benchmark, its reset to date, trend expectations and PIE_RBAQ."""
    if not all(k in all_results for k in ("cusbert", "unanchored")):
        return
    bench = all_results["cusbert"]
    start = pd.Period(BENCHMARK_CHART_START, freq="Q")
    end = bench.last_complete
    post = bench.expectations_posterior()
    band = pd.DataFrame({"lower": post.quantile(BAND_LOWER, axis=1), "upper": post.quantile(BAND_UPPER, axis=1)})
    label = LEGEND_LABELS["cusbert"]
    # Trend expectations first, so the main model takes the palette's first, darkest colour.
    lines = pd.DataFrame({
        LEGEND_LABELS["unanchored"]: all_results["unanchored"].expectations_posterior_quarterly().median(axis=1),
        label: post.median(axis=1),
        f"Surveys only, level set to {RESET_START.year}-{end} inflation":
            bench.expectations_posterior("pi_exp_todate").median(axis=1),
    })
    pie = get_inflation_expectations().data.rename("RBA PIE_RBAQ")

    ax = mg.fill_between_plot(band[(band.index >= start) & (band.index <= end)], color="lightgrey", alpha=0.6,
                              label=f"{label} 90% band")
    mg.line_plot(pie[pie.index >= start], ax=ax, color="darkgrey", width=1.5, annotate=False)
    mg.line_plot(lines[(lines.index >= start) & (lines.index <= end)], ax=ax, width=2, annotate=False)
    mg.finalise_plot(
        ax,
        title="Inflation expectations: trend and survey benchmark",
        lheader=(f"Survey-only level set to mean inflation over {RESET_START}-{RESET_END}, "
                 f"and over {RESET_START}-{end}"),
        lfooter=f"Australia. {BENCHMARK_DISCLAIMER}",
        rfooter="RBA G3, F2; ABS 6401.0; MARTIN",
        **PLOT_KWARGS,
    )


# --- Survey benchmark validation ---


def print_benchmark_validation(all_results: dict[str, ExpectationsResults]) -> None:
    """Print the survey benchmark's validation table. Reports; does not judge, bar the PIE_RBAQ test."""
    if "cusbert" not in all_results:
        return
    bench = all_results["cusbert"]
    # Short names: the legend labels are too long to sit inside a sentence.
    bench_label, trend_label = "Survey-only", "Trend"
    post = bench.expectations_posterior()
    median = post.median(axis=1)
    in_window = (post.index >= RESET_START) & (post.index <= RESET_END)
    reset_sample = f"{RESET_START} to {RESET_END}"
    quantiles = [BAND_LOWER, 0.5, BAND_UPPER]
    rows: list[tuple[str, str, str]] = []

    summary = posterior_summary(bench.trace, decimals=None)
    rows.append(("Sampler gate", "whole run",
                 (f"max r_hat {summary['r_hat'].max():.3f}, min bulk ESS {summary['ess_bulk'].min():.0f}, "
                  f"divergences {int(bench.trace.sample_stats['diverging'].sum())}")))

    reset_gap = (post.loc[in_window].mean(axis=0) - bench.inflation.loc[in_window].mean()).abs().max()
    rows.append(("Mean pi_exp less mean inflation", reset_sample,
                 f"{reset_gap:.2f} at most, over every draw ({reset_gap:.1e})"))

    # Each series against PIE_RBAQ: correlation in levels and in four-quarter
    # changes, RMSE, and mean difference (series less PIE_RBAQ).
    pie_window = pd.period_range(*PIE_WINDOW, freq="Q")
    pie = get_inflation_expectations().data.reindex(pie_window)
    production = pd.read_parquet(OUTPUT_DIR / "expectations_unanchored_hdi_quarterly.parquet")["median"]
    pie_fit: list[tuple[float, float]] = []  # (RMSE, level correlation): benchmark, then trend
    for label, series in ((bench_label, median), (f"{trend_label} (production file)", production)):
        on_window = series.reindex(pie_window)
        gap = on_window - pie
        rmse, corr = float(np.sqrt((gap**2).mean())), float(on_window.corr(pie))
        changes = float(on_window.diff(QUARTERS_PER_YEAR).corr(pie.diff(QUARTERS_PER_YEAR)))
        pie_fit.append((rmse, corr))
        rows.append((f"{label} vs PIE_RBAQ", f"{PIE_WINDOW[0]} to {PIE_WINDOW[1]}",
                     (f"corr levels {corr:.2f}, corr 4q changes {changes:.2f}, RMSE {rmse:.2f}, "
                      f"mean diff {gap.mean():+.2f}")))

    versus = f"{bench_label} less {trend_label.lower()}"
    diff = (median - production).reindex(median.index[in_window])
    rows.append((versus, reset_sample,
                 f"mean {diff.mean():+.2f}, max abs {diff.abs().max():.2f} ({diff.abs().idxmax()})"))

    # Last complete quarter: the business survey and a lagged inflation reading both observed.
    observed = bench.measures["business"].notna() & bench.inflation.shift(1).notna()
    last, final = observed[observed].index[-1], bench.index[-1]
    note = "" if last == final else f" (final quarter {final} is part-observed; {last} used)"
    rows.append((versus, f"{last}{note}", f"{median.get(last, np.nan) - production.get(last, np.nan):+.2f}"))

    shift = bench.expectations_posterior("pi_exp_todate") - post
    spread = float((shift.max(axis=0) - shift.min(axis=0)).max())
    lo, med, hi = np.quantile(shift.iloc[0].to_numpy(), quantiles)
    rows.append(("pi_exp_todate less pi_exp", "constant per draw",
                 f"{med:.2f} [{lo:.2f}, {hi:.2f}] (max spread across quarters {spread:.1e})"))

    since = median[median.index >= pd.Period(BENCHMARK_CHART_START, freq="Q")]
    outside = since[(since < TARGET_BAND[0]) | (since > TARGET_BAND[1])]
    rows.append((f"Quarters outside {TARGET_BAND[0]:.0f} to {TARGET_BAND[1]:.0f}", f"from {since.index[0]}",
                 ", ".join(f"{q} {v:.2f}" for q, v in outside.items()) or "none"))

    surveys = MODEL_CONFIGS["cusbert"].get("survey_series", [])
    for param in ("alpha", "lambda_bias"):
        draws = bench.trace.posterior[param].stack(sample=("chain", "draw")).to_numpy()
        q = np.quantile(draws, quantiles, axis=1)  # rows: 5th, median, 95th; one column per survey
        rows.extend((f"{param} {survey}", "median [90%]", f"{q[1, i]:.2f} [{q[0, i]:.2f}, {q[2, i]:.2f}]")
                    for i, survey in enumerate(surveys))

    print(f"\n{'=' * 60}\nVALIDATION: {MODEL_NAMES['cusbert']}\n{'=' * 60}")
    width = max(len(r[0]) for r in rows)
    sample_width = max(len(r[1]) for r in rows)
    for check, sample, value in rows:
        print(f"{check:<{width}}  {sample:<{sample_width}}  {value}")

    (bench_rmse, bench_corr), (trend_rmse, trend_corr) = pie_fit
    verdict = "DOES" if bench_rmse <= trend_rmse and bench_corr >= trend_corr else "DOES NOT"
    print(f"\nThe {bench_label.lower()} {verdict} track PIE_RBAQ at least as closely as "
          f"{trend_label.lower()} (rule: RMSE no larger and level correlation no smaller).")


# --- CLI ---

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Expectations Stage 2: Diagnostics & Plots")
    parser.add_argument("--no-plots", action="store_true", help="Skip plot generation")

    args = parser.parse_args()

    print("=" * 60)
    print("EXPECTATIONS STAGE 2: DIAGNOSTICS & PLOTS")
    print("=" * 60)

    # Load all results
    print("\nLoading saved results...")
    all_results = load_all_results()

    # Run diagnostics
    for results in all_results.values():
        run_diagnostics(results)

    # Generate plots
    if not args.no_plots:
        print("\n" + "=" * 60)
        print("GENERATING PLOTS")
        print("=" * 60)
        generate_plots(all_results)

    print_benchmark_validation(all_results)

    print("\n" + "=" * 60)
    print("STAGE 2 COMPLETE")
    print("=" * 60)
