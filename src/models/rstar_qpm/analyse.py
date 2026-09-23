"""Charts and printed diagnostics for the rstar_qpm model."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import arviz as az
import mgplot as mg
import numpy as np
import pandas as pd
import xarray as xr

from src.models.common import prior_posterior
from src.models.common.charts import excluded_span_style
from src.models.common.diagnostics import diagnostics_header, save_diagnostics
from src.models.common.sources import footer_from_constants
from src.models.common.timeseries import last_complete_quarter, plot_posterior_timeseries
from src.models.rstar_qpm.config import IS_PARAMETERS, ModelConfig, Prior
from src.models.rstar_qpm.estimate import load_results, parameter_names, posterior_params
from src.models.rstar_qpm.neutral import rate_response
from src.paths import CHARTS

CHART_DIR = CHARTS / "RStarQPM"

_LFOOTER = "Australia. "

# Posterior sd over prior sd, below which a parameter counts as informed by
# the data. Above it, the posterior is largely the prior restated.
INFORMED_RATIO = 0.5

# A recorded prior is [kind, mu, sd].
PRIOR_FIELDS = 3

# Share of the most negative prior multiplier draws left off the multiplier
# chart, which would otherwise stretch the axis past -15.
PRIOR_TAIL_TRIM = 0.05


def _posterior(trace: az.InferenceData) -> xr.Dataset:
    group = getattr(trace, "posterior", None)
    if not isinstance(group, xr.Dataset):
        raise TypeError("trace has no posterior group")
    return group


def identification_table(trace: az.InferenceData, priors: dict[str, Prior]) -> pd.DataFrame:
    """Prior against posterior for every parameter: the test of what the data decided."""
    posterior = _posterior(trace)
    rows = {}
    for name, prior in priors.items():
        draws = np.asarray(posterior[name].values).ravel()
        prior_mean, prior_sd = prior.moments()
        rows[name] = {
            "prior mean": prior_mean,
            "prior sd": prior_sd,
            "post mean": float(draws.mean()),
            "post sd": float(draws.std()),
            "sd ratio": float(draws.std()) / prior_sd,
            "shift (prior sds)": (float(draws.mean()) - prior_mean) / prior_sd,
        }
    table = pd.DataFrame(rows).T
    table["informed"] = table["sd ratio"] < INFORMED_RATIO
    return table


def _latest(paths: pd.DataFrame, at: pd.Period) -> str:
    row = paths.loc[paths.index == at].iloc[0]
    return f"{row.median():.2f} [{row.quantile(0.05):.2f}, {row.quantile(0.95):.2f}]"


def print_summary(trace: az.InferenceData, frame: pd.DataFrame, states: dict[str, Any],
                  priors: dict[str, Prior]) -> list[str]:
    """Print the identification table and the headline numbers; return report notes."""
    table = identification_table(trace, priors)
    print("\nPrior against posterior (sd ratio < "
          f"{INFORMED_RATIO:g} = the data informed it):")
    print(table.round(3).to_string())

    at = min(last_complete_quarter(), frame.index.max())
    paths = states["paths"]
    multiplier = states["multiplier"]
    real_rate = frame.loc[at, "r"]
    lines = [
        f"Latest complete quarter {at}, real, median [90%]:",
        f"  trend r*            {_latest(paths['rstar'], at)}",
        f"  short-run neutral   {_latest(paths['srn'], at)}   (horizon {states['horizon']}q)",
        f"  real cash rate      {real_rate:.2f}",
        f"  output gap          {_latest(paths['ygap'], at)}",
        f"  real TWI gap        {_latest(paths['qgap'], at)}",
        f"  potential growth    {_latest(paths['g'], at)}",
        (f"Multiplier B (gap at {states['horizon']}q per 1pp rate gap held): median "
         f"{np.median(multiplier):.3f} [{np.quantile(multiplier, 0.05):.3f}, "
         f"{np.quantile(multiplier, 0.95):.3f}]"),
    ]
    print("\n" + "\n".join(lines))
    informed = ", ".join(table.index[table["informed"]]) or "none"
    lines.append(f"Parameters the data informed (sd ratio < {INFORMED_RATIO:g}): {informed}")
    return lines


def _footers(constants: dict[str, Any]) -> dict[str, str]:
    return {"lfooter": _LFOOTER, "rfooter": footer_from_constants(constants) or ""}


def _span(constants: dict[str, Any]) -> list[dict[str, Any]]:
    lo, hi = constants.get("exclude_start"), constants.get("exclude_end")
    if not lo or not hi:
        return []
    style = excluded_span_style()
    return [{"xmin": pd.Period(lo, freq="Q"), "xmax": pd.Period(hi, freq="Q"), **style,
             "label": f"GDP excluded from fit, {lo}-{hi}"}]


@dataclass(frozen=True)
class ChartContext:
    """What every chart in a run needs besides its own data."""

    constants: dict[str, Any]
    header: str      # sampling issues, if any; empty when the run was clean

    def lheader(self, text: str = "") -> str:
        """Put the sampling warning first, ahead of a chart's own header text."""
        if not self.header:
            return text
        return f"{self.header} | {text}" if text else self.header


# Reference series are data the estimate is read against, not model output,
# and are drawn in grey: the cash rate dashed, the market and world rates
# dotted, so the two stay apart when they share a chart.
REFERENCE_STYLES: dict[str, dict[str, Any]] = {
    "cash rate": {"color": "darkgrey", "style": "--"},
    "forward": {"color": "darkgrey", "style": ":"},
    "world real rate": {"color": "darkgrey", "style": ":"},
}
# The palette for one to five lines is darkblue, darkorange, cornflowerblue,
# brown, gray. The band is the first line, so it takes darkblue and a model
# line beside it takes the second.
BAND_COLOUR = "darkblue"
OVERLAY_COLOUR = "darkorange"


def _reference_style(column: object) -> dict[str, Any] | None:
    """Return the grey style for a reference series, or None for a model line."""
    name = str(column).lower()
    for key, style in REFERENCE_STYLES.items():
        if key in name:
            return style
    return None


def _band_chart(paths: pd.DataFrame, overlays: pd.DataFrame | None, *, stem: str,
                ctx: ChartContext, finalise: dict[str, Any]) -> None:
    """One posterior band, optional overlay lines, finalised.

    Overlays named as a reference series (see `REFERENCE_STYLES`) are grey;
    any other overlay is a model line and takes `OVERLAY_COLOUR`.
    """
    end = last_complete_quarter()
    paths = paths.loc[:end]
    ax = plot_posterior_timeseries(data=paths, legend_stem=stem, color=BAND_COLOUR,
                                   cuts=(0.05, 0.25), alphas=(0.15, 0.3), finalise=False)
    if overlays is not None:
        for column in overlays.columns:
            style = _reference_style(column) or {"color": OVERLAY_COLOUR, "style": "-"}
            ax = mg.line_plot(overlays.loc[:end, [column]], ax=ax, width=1.5, annotate=True, **style)
    if ax is None:
        return
    footers = _footers(ctx.constants)
    finalise = {**finalise, "lheader": ctx.lheader(str(finalise.get("lheader", "")))}
    mg.finalise_plot(ax, lfooter=footers["lfooter"], rfooter=footers["rfooter"],
                     legend={"loc": "best", "fontsize": "x-small"}, show=False, **finalise)


def plot_charts(frame: pd.DataFrame, states: dict[str, Any], ctx: ChartContext, *, has_is: bool) -> None:
    """Draw the state charts: r*, short-run neutral, the gaps and potential growth."""
    paths = states["paths"]
    horizon = states["horizon"]
    rstar_median = paths["rstar"].median(axis=1)
    span = _span(ctx.constants)

    _band_chart(
        paths["rstar"],
        pd.DataFrame({"Real cash rate": frame["r"], "Real 5y5y forward": frame["f"]}),
        stem="Trend r*", ctx=ctx,
        finalise={"title": "Trend r*: world real rate plus the Australian wedge",
                  "ylabel": "Per cent, real", "y0": True},
    )
    if has_is:
        _band_chart(
            paths["srn"],
            pd.DataFrame({"Trend r* (median)": rstar_median, "Real cash rate": frame["r"]}),
            stem="Short-run neutral", ctx=ctx,
            finalise={"title": f"Short-run neutral: the real rate closing the gap in {horizon} quarters",
                      "ylabel": "Per cent, real", "y0": True,
                      "lheader": f"Multiplier B median {np.median(states['multiplier']):.3f}"},
        )
    _band_chart(paths["ygap"], None, stem="Output gap", ctx=ctx,
                finalise={"title": "Output gap", "ylabel": "Per cent of potential", "y0": True,
                          "axvspan": span})
    _band_chart(paths["qgap"], None, stem="Real TWI gap", ctx=ctx,
                finalise={"title": "Real exchange rate gap", "ylabel": "Per cent, + = overvalued",
                          "y0": True})
    _band_chart(paths["g"], None, stem="Potential growth", ctx=ctx,
                finalise={"title": "Potential growth", "ylabel": "Per cent per year", "axvspan": span})


def plot_rstar_scales(frame: pd.DataFrame, states: dict[str, Any], ctx: ChartContext) -> None:
    """Trend r* real and nominal, nominal on measured expectations."""
    rstar = states["paths"]["rstar"]
    nominal = rstar.add(frame["pie"], axis=0)
    _band_chart(
        nominal,
        pd.DataFrame({"Real trend r* (median)": rstar.median(axis=1), "Cash rate": frame["i"]}),
        stem="Nominal trend r*", ctx=ctx,
        finalise={"title": "Australia's trend r*, real and nominal",
                  "ylabel": "Per cent", "y0": True,
                  "lheader": "Nominal = real + measured inflation expectations"},
    )


def plot_wedge(frame: pd.DataFrame, states: dict[str, Any], ctx: ChartContext) -> None:
    """Chart the Australian wedge, with the world real rate it sits on."""
    _band_chart(
        states["paths"]["wedge"],
        pd.DataFrame({"World real rate": frame["rw"]}),
        stem="Australian wedge", ctx=ctx,
        finalise={"title": "The Australian wedge over the world real rate",
                  "ylabel": "Percentage points", "y0": True},
    )


def plot_stances(frame: pd.DataFrame, states: dict[str, Any], ctx: ChartContext, *, has_is: bool) -> None:
    """Chart the real cash rate against each neutral: positive is restrictive."""
    paths = states["paths"]
    neutrals = [("rstar", "trend r*")] + ([("srn", "short-run neutral")] if has_is else [])
    for key, label in neutrals:
        stance = -paths[key].sub(frame["r"], axis=0)
        _band_chart(
            stance, None, stem="Stance", ctx=ctx,
            finalise={"title": f"Policy stance: the real cash rate against {label}",
                      "ylabel": "Percentage points, + = restrictive", "y0": True},
        )


def plot_headwinds(states: dict[str, Any], ctx: ChartContext) -> None:
    """Short-run neutral less trend r*: the only thing short-run neutral adds."""
    paths = states["paths"]
    _band_chart(
        paths["srn"] - paths["rstar"], None, stem="Short-run less trend", ctx=ctx,
        finalise={"title": "Headwinds and tailwinds: short-run neutral less trend r*",
                  "ylabel": "Percentage points, + = tailwind", "y0": True},
    )


def plot_rule(frame: pd.DataFrame, states: dict[str, Any], draws: list[dict[str, float]],
              ctx: ChartContext) -> None:
    """Chart the rate the estimated rule wants, before smoothing, against the cash rate."""
    paths = states["paths"]
    target = float(ctx.constants["target"])
    columns = []
    for k, p in enumerate(draws):
        columns.append(paths["rstar"][k] + frame["pie"] + p["phi_pi"] * (frame["pi4"] - target)
                       + p["phi_y"] * paths["ygap"][k])
    desired = pd.concat(columns, axis=1)
    _band_chart(
        desired, pd.DataFrame({"Cash rate": frame["i"]}),
        stem="Rule's desired rate", ctx=ctx,
        finalise={"title": "The cash rate and the estimated Taylor rule",
                  "ylabel": "Per cent, nominal", "y0": True,
                  "lheader": "Desired rate, before the rule's smoothing"},
    )


def plot_transmission(draws: list[dict[str, float]], ctx: ChartContext, horizon: int,
                      exposure: float) -> None:
    """Chart the output gap's response to a 1pp real rate gap held, by channel.

    At the latest quarter's debt exposure: the cash-flow channel scales with it.
    """
    periods = 20
    responses = [rate_response(p, periods, exposure) for p in draws]
    index = pd.RangeIndex(1, periods + 1)
    frame = pd.DataFrame({
        "Total": np.median([r["total"] for r in responses], axis=0),
        "Direct rate channel": np.median([r["direct"] for r in responses], axis=0),
        "Exchange rate channel": np.median([r["exchange_rate"] for r in responses], axis=0),
        "Cash-flow channel": np.median([r["cash_flow"] for r in responses], axis=0),
    }, index=index)
    total = pd.DataFrame(np.column_stack([r["total"] for r in responses]), index=index)
    band = pd.DataFrame({"lower": total.quantile(0.05, axis=1), "upper": total.quantile(0.95, axis=1)})
    # The band belongs to the total, so it takes the total's colour: the first
    # of mgplot's palette, which is what mgplot gives the first line.
    ax = mg.fill_between_plot(band, color=BAND_COLOUR, alpha=0.15, label="Total, 90% interval")
    ax = mg.line_plot(frame, ax=ax, width=[2.0, 1.5, 1.5, 1.5])
    footers = _footers(ctx.constants)
    mg.finalise_plot(
        ax, title="How a rate rise works through: the output gap's response",
        xlabel="Quarters after the real rate is raised 1pp and held",
        ylabel="Percentage points of potential", y0=True,
        lheader=ctx.lheader(f"B = response at {horizon}q; debt exposure {exposure:.2f}x income"),
        axvline={"x": horizon, "color": "grey", "linestyle": "--"},
        legend={"loc": "best", "fontsize": "x-small"},
        lfooter=footers["lfooter"], rfooter=footers["rfooter"], show=False,
    )


def _prior_multiplier(priors: dict[str, Prior], horizon: int, n: int, seed: int, *,
                      exposure: float) -> np.ndarray:
    """B under the priors alone, by sampling each transmission parameter's prior."""
    rng = np.random.default_rng(seed)
    grid = np.linspace(0.0, 1.0, 20_001)
    samples: dict[str, np.ndarray] = {}
    for name in ("b1", "b2", "b3", "b4", "kappa", "rho_q", "delta"):
        prior = priors[name]
        support = grid if prior.kind == "beta" else grid * (prior.mu + 6.0 * prior.sd)
        density = prior.pdf(support)
        cdf = np.cumsum(density)
        samples[name] = np.interp(rng.uniform(size=n), cdf / cdf[-1], support)
    return np.array([
        rate_response({k: float(v[i]) for k, v in samples.items()}, horizon, exposure)["total"][-1]
        for i in range(n)
    ])


def plot_multiplier(states: dict[str, Any], priors: dict[str, Prior], ctx: ChartContext,
                    exposure: float) -> None:
    """Chart B under the priors against B under the posterior, to show whether the data decided it."""
    horizon = states["horizon"]
    draws = np.asarray(states["multiplier"])[np.newaxis, :]
    prior = _prior_multiplier(priors, horizon, n=4_000, seed=0, exposure=exposure)
    # The prior's left tail runs past -15 and squeezes the posterior into a
    # sliver, so its most negative draws are left off the chart and the header
    # says so. The prior mean in the header is of the trimmed draws.
    shown = prior[prior >= np.quantile(prior, PRIOR_TAIL_TRIM)]
    prior_posterior.plot_parameter(
        f"Multiplier B at {horizon}q", draws, shown,
        footers={"lfooter": "", "rfooter": "",
                 "lheader": ctx.lheader(f"Prior's lowest {PRIOR_TAIL_TRIM:.0%} not drawn")},
        label="Output gap response to a 1pp real rate gap held, pp",
    )


def plot_priors(trace: az.InferenceData, priors: dict[str, Prior], ctx: ChartContext) -> int:
    """One prior-against-posterior chart per parameter.

    No left footer: the shared chart appends its own note about the chains
    there, and anything longer collides with the sources on the right.
    """
    return prior_posterior.plot_all(
        trace,
        lambda name: priors[name].pdf if name in priors else None,
        footers={"lfooter": "", "rfooter": _footers(ctx.constants)["rfooter"], "lheader": ctx.lheader()},
    )


def run_priors(constants: dict[str, Any]) -> dict[str, Prior]:
    """Return the priors the run was sampled under, as recorded with it.

    A run saved before the priors were recorded falls back to the current
    config, and says so, since the two may differ.
    """
    recorded = constants.get("priors")
    if not isinstance(recorded, dict):
        print("  note: this run did not record its priors; using the current config's")
        return ModelConfig().priors
    return {
        str(name): Prior(str(spec[0]), float(spec[1]), float(spec[2]))
        for name, spec in recorded.items()
        if isinstance(spec, list | tuple) and len(spec) == PRIOR_FIELDS
    }


def run_analysis(prefix: str = "rstar_qpm", output_dir: Path | None = None,
                 chart_dir: Path | None = None) -> None:
    """Load a saved run, print the summary, write the charts and diagnostics."""
    trace, frame, constants, states = load_results(prefix=prefix, output_dir=output_dir)
    priors = run_priors(constants)
    notes = print_summary(trace, frame, states, priors)

    chart_dir = chart_dir or (CHART_DIR if prefix == "rstar_qpm" else CHART_DIR.parent / f"RStarQPM_{prefix}")
    mg.set_chart_dir(str(chart_dir))
    mg.clear_chart_dir()
    save_diagnostics(trace, chart_dir, prefix, model="rstar_qpm", notes=notes)
    ctx = ChartContext(constants=constants, header=diagnostics_header(chart_dir, prefix))
    # The same evenly spaced draws the state paths were built from, so each
    # parameter set lines up with its own state path column.
    has_is = bool(constants.get("use_is", 1.0))
    fixed = {} if has_is else dict.fromkeys(IS_PARAMETERS, 0.0)
    draws = [{**p, **fixed} for p in
             posterior_params(trace, parameter_names(priors), states["paths"]["rstar"].shape[1])]

    plot_charts(frame, states, ctx, has_is=has_is)
    plot_rstar_scales(frame, states, ctx)
    plot_wedge(frame, states, ctx)
    plot_stances(frame, states, ctx, has_is=has_is)
    plot_rule(frame, states, draws, ctx)
    if has_is:
        # Short-run neutral, and everything built on it, needs an IS curve.
        plot_headwinds(states, ctx)
        exposure = float(frame["exposure"].iloc[-1])
        plot_transmission(draws, ctx, states["horizon"], exposure)
        plot_multiplier(states, priors, ctx, exposure)
    drawn = plot_priors(trace, priors, ctx)
    print(f"\n{drawn} prior/posterior charts. Charts written to: {chart_dir}")
