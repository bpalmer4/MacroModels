"""Tables and charts for the regime u* reading."""

from typing import Any

import arviz as az
import mgplot as mg
import numpy as np
import pandas as pd
import xarray as xr

from src.models.common.diagnostics import save_diagnostics
from src.models.regime_ustar.config import ModelConfig

_BAND_KWARGS: dict[str, Any] = {"color": "darkorange", "alpha": 0.18, "label": "90% credible interval"}
# Kept short deliberately: these sit beside a "Built using: ..." right footer
# that grows with the number of sources, and the two collide if either runs on.
_LFOOTER_BY_STATE = {
    "spline": "Australia. u* is a natural cubic spline with knots at the regime dates. ",
    "attractor": "Australia. u* converges to a regime-specific attractor. ",
}


def _posterior(trace: az.InferenceData) -> xr.Dataset:
    """Return the trace's posterior group, narrowed at runtime."""
    posterior = getattr(trace, "posterior", None)
    if not isinstance(posterior, xr.Dataset):
        raise TypeError("trace has no posterior group — was it loaded from a completed run?")
    return posterior


def _flat(trace: az.InferenceData, name: str) -> np.ndarray:
    """Return draws x dimension for `name`, with chains stacked."""
    values = np.asarray(_posterior(trace)[name])
    return values.reshape(-1, *values.shape[2:])


def regime_table(trace: az.InferenceData, frame: pd.DataFrame, regimes: np.ndarray, labels: list[str]) -> pd.DataFrame:
    """One row per regime: where it pulled u* towards, and where u* actually got to.

    `eq` is the attractor and `u* end` is the level the state reached by the
    last quarter of the regime. The distance between them is how much of the
    move was still outstanding when the regime ended, which is the thing a step
    function cannot express and this model exists to.
    """
    posterior = _posterior(trace)
    eq = _flat(trace, "eq") if "eq" in posterior else None
    path = _flat(trace, "ustar")
    rows = []
    for k, label in enumerate(labels):
        here = np.flatnonzero(regimes == k)
        row = {
            "regime": label,
            "quarters": len(here),
            "mean u": float(frame["u"].iloc[here].mean()),
            "mean surprise": float(frame["surprise"].iloc[here].mean()),
            "u* start": float(np.median(path[:, here[0]])),
            "u* end": float(np.median(path[:, here[-1]])),
            "u* min": float(np.median(path[:, here], axis=0).min()),
            "u* max": float(np.median(path[:, here], axis=0).max()),
        }
        if eq is not None:
            row |= {
                "eq": float(np.median(eq[:, k])),
                "eq 5%": float(np.percentile(eq[:, k], 5)),
                "eq 95%": float(np.percentile(eq[:, k], 95)),
            }
        rows.append(row)
    table = pd.DataFrame(rows)
    if "eq" in table:
        table["outstanding"] = table["eq"] - table["u* end"]
    return table


def ustar_path(trace: az.InferenceData, frame: pd.DataFrame) -> pd.DataFrame:
    """Return the u* state as a quarterly frame of 5th, 50th and 95th percentiles."""
    q = np.percentile(_flat(trace, "ustar"), [5, 50, 95], axis=0)
    return pd.DataFrame({"lower": q[0], "median": q[1], "upper": q[2]}, index=frame.index)


def beta_per_quarter(trace: az.InferenceData, regimes: np.ndarray, config: ModelConfig) -> np.ndarray:
    """Return the posterior median slope facing each quarter.

    A scalar repeated, or, when `config.beta_groups` gives the regimes their
    own slopes, that regime's own median. The inversion divides by this, so
    using a pooled median where the model fitted several would misstate every
    quarter outside the largest group.
    """
    draws = _flat(trace, "beta")
    if draws.ndim == 1:
        return np.full(len(regimes), float(np.median(draws)))
    by_group = np.median(draws, axis=0)
    return by_group[np.asarray(config.beta_groups, dtype=int)[regimes]]


def implied_ustar(frame: pd.DataFrame, beta: np.ndarray | float) -> pd.Series:
    """Return the u* each quarter's inflation would need, taken on its own.

    Invert the Phillips curve with the residual set to zero and solve for u*.
    With `pi_t - pi^e_t = -beta x (u_t - u*_t) / u_t`,

        u*_t = u_t x (1 + (pi_t - pi^e_t) / beta)

    so the surprise is scaled by `u/beta`, which shrinks where unemployment is
    low. That is the convexity, and it is why this is not simply the inflation
    gap shifted by a constant.

    Not an estimator. It is the diagnostic that makes the state law visible: it
    is what the data would say about u* with no smoothness prior and no regime
    structure at all, so plotting the fitted path against it shows whether the
    law is filtering the series or ignoring it. This is the chart the model is
    judged on.

    Expect it to be wild: it divides a noisy residual by a coefficient the same
    equation had trouble identifying.

    `beta` comes from `beta_per_quarter`, so where the regimes carry their own
    slopes the inversion changes scale at each boundary along with them.
    """
    return frame["u"] * (1.0 + frame["surprise"] / beta)


def _breaks_as_lines(frame: pd.DataFrame, regimes: np.ndarray) -> list[dict[str, Any]]:
    """Return a faint vertical rule at each regime boundary present in the sample."""
    starts = frame.index[np.flatnonzero(np.diff(regimes, prepend=regimes[0]) != 0)]
    return [{"x": s, "color": "grey", "linestyle": ":", "linewidth": 0.8} for s in starts]


def plot_ustar(path: pd.DataFrame, frame: pd.DataFrame, regimes: np.ndarray, footers: dict[str, str]) -> None:
    """Chart the step-function u* against the unemployment rate it is read from."""
    ax = mg.fill_between_plot(path[["lower", "upper"]], **_BAND_KWARGS)
    mg.line_plot(frame["u"].rename("Unemployment rate"), ax=ax, color=["black"], width=1.2, annotate=False)
    mg.line_plot(
        path["median"].rename("u*"),
        ax=ax,
        color=["darkorange"],
        width=2.2,
        annotate=True,
        rounding=2,
    )
    mg.finalise_plot(
        ax,
        title=footers["title"],
        ylabel="Per cent",
        legend={"loc": "best", "fontsize": "small"},
        axvline=_breaks_as_lines(frame, regimes),
        lheader=f"{footers['lheader_note']} {footers['lheader']}",
        rfooter=footers["rfooter"],
        lfooter=footers["lfooter"],
        show=False,
    )


def plot_gap(path: pd.DataFrame, frame: pd.DataFrame, regimes: np.ndarray, footers: dict[str, str]) -> None:
    """Chart the implied unemployment gap, u less the regime level."""
    gap = pd.DataFrame({
        "lower": frame["u"] - path["upper"],
        "median": frame["u"] - path["median"],
        "upper": frame["u"] - path["lower"],
    })
    ax = mg.fill_between_plot(gap[["lower", "upper"]], **_BAND_KWARGS)
    mg.line_plot(gap["median"].rename("u - u*"), ax=ax, color=["darkorange"], width=2, annotate=True, rounding=2)
    mg.finalise_plot(
        ax,
        title="Unemployment gap implied by the regimes",
        ylabel="Percentage points",
        y0=True,
        legend={"loc": "best", "fontsize": "small"},
        axvline=_breaks_as_lines(frame, regimes),
        rfooter=footers["rfooter"],
        lfooter=footers["lfooter"],
        show=False,
    )


def _switch_span(frame: pd.DataFrame, config: ModelConfig) -> list[dict[str, Any]]:
    """Shade the stretch where the expectation is asserted rather than measured.

    The distinction is the model's weakest join and is invisible otherwise: to
    the left the expectation is a constant and then a trailing average of
    inflation itself, so it adds no information beyond `pi`. To the right it is
    a separate series built from surveys and bond markets.
    """
    switch = pd.Period(config.measured_from, freq="Q")
    if frame.index[0] >= switch:
        return []
    return [{
        "xmin": frame.index[0],
        "xmax": switch,
        "color": "grey",
        "alpha": 0.12,
        "label": f"Expectations asserted, to {config.measured_from}",
    }]


def plot_expectations(frame: pd.DataFrame, config: ModelConfig, footers: dict[str, str]) -> None:
    """Chart inflation against the expectation the regime says was held.

    The two lines ARE the model: their difference is the whole dependent
    variable, and everything the Phillips curve attributes to the labour market
    is the vertical distance between them divided by `beta`.
    """
    mg.line_plot_finalise(
        pd.DataFrame({
            "Inflation": frame["pi"],
            "Expected inflation": frame["pi_e"],
        }),
        color=["black", "darkorange"],
        width=[1.4, 2.0],
        style=["-", "-"],
        annotate=True,
        rounding=2,
        title="Inflation and the expectation it is measured against",
        ylabel="Per cent, year-ended",
        legend={"loc": "best", "fontsize": "small"},
        axvspan=_switch_span(frame, config),
        lheader=footers["lheader_note"],
        rfooter=footers["rfooter"],
        lfooter=footers["lfooter"],
        show=False,
    )


def plot_surprise(frame: pd.DataFrame, regimes: np.ndarray, config: ModelConfig, footers: dict[str, str]) -> None:
    """Chart the inflation surprise, which is what the Phillips curve explains.

    Positive reads as a tight labour market, negative as slack, and the model
    has nothing else to attribute either to. The largest excursions are
    therefore the places to distrust it: +2.94 at 1973Q4 and +5.17 at 2022Q4
    are both world price shocks arriving as statements about Australian slack.
    """
    mg.line_plot_finalise(
        frame["surprise"].rename("Inflation less expected inflation"),
        color=["darkorange"],
        width=1.8,
        annotate=True,
        rounding=2,
        y0=True,
        title="The inflation surprise the Phillips curve explains",
        ylabel="Percentage points",
        legend={"loc": "best", "fontsize": "small"},
        axvline=_breaks_as_lines(frame, regimes),
        axvspan=_switch_span(frame, config),
        lheader="Positive reads as tight, negative as slack; nothing else can absorb either",
        rfooter=footers["rfooter"],
        lfooter=footers["lfooter"],
        show=False,
    )


def plot_implied(
    path: pd.DataFrame, implied: pd.Series, frame: pd.DataFrame, regimes: np.ndarray, footers: dict[str, str],
) -> None:
    """Draw the acceptance test: what inflation alone says u* is, against the fitted path.

    The grey line owes nothing to the regimes or to the state law. If the
    orange path is a plausible reading of it, the structure is filtering the
    series; if the grey line is formless, the structure is inventing the answer
    and the path should not be quoted.
    """
    fitted = path["median"]
    ratio = implied.diff().std() / fitted.diff().std()

    ax = mg.fill_between_plot(path[["lower", "upper"]], **_BAND_KWARGS)
    mg.line_plot(
        pd.DataFrame({"Implied by inflation alone": implied, "u*": fitted}),
        ax=ax,
        color=["grey", "darkorange"],
        width=[1.0, 2.5],
        alpha=[0.6, 1.0],
        annotate=True,
        rounding=2,
    )
    mg.finalise_plot(
        ax,
        title="What inflation alone says u* is, quarter by quarter",
        ylabel="Per cent",
        legend={"loc": "best", "fontsize": "small"},
        axvline=_breaks_as_lines(frame, regimes),
        lheader=(
            f"Implied series moves {ratio:.0f}x as much quarter to quarter; "
            f"correlation with u* {implied.corr(fitted):.2f}"
        ),
        rfooter=footers["rfooter"],
        lfooter="Australia. Phillips curve inverted at the posterior median. ",
        show=False,
    )


def plot_levels(table: pd.DataFrame, footers: dict[str, str]) -> None:
    """Each regime's attractor beside its mean unemployment and the level u* reached.

    The gap between `eq` and `u* end` is the part of the move the regime did
    not finish, which is the asymmetry the model is built to carry.
    """
    columns = [c for c in ("mean u", "eq", "u* end") if c in table]
    data = table.set_index("regime")[columns]
    mg.bar_plot_finalise(
        data,
        color=["silver", "darkorange", "saddlebrown"][: len(data.columns)],
        stacked=False,
        annotate=True,
        rounding=2,
        label_rotation=30,
        title="Each regime's attractor, and where u* actually got to",
        ylabel="Per cent",
        legend={"loc": "best", "fontsize": "small"},
        rfooter=footers["rfooter"],
        lfooter=footers["lfooter"],
        show=False,
    )


def analyse(
    trace: az.InferenceData,
    frame: pd.DataFrame,
    regimes: np.ndarray,
    labels: list[str],
    config: ModelConfig,
    source_footer: str,
) -> pd.DataFrame:
    """Print the table, draw the charts, write this run's diagnostics."""
    table = regime_table(trace, frame, regimes, labels)
    path = ustar_path(trace, frame)
    implied = implied_ustar(frame, beta_per_quarter(trace, regimes, config))

    posterior = _posterior(trace)
    names = ["beta", "sigma"]
    names += [n for n in ("rho_pi", "xi_gscpi", "gamma_wage", "lambda_wage", "alpha_wage",
                          "sigma_wage", "phi", "kappa", "coef") if n in posterior]
    summary = az.summary(trace, var_names=names)
    print("\nRegime levels")
    print(table.to_string(index=False, float_format=lambda v: f"{v:.2f}"))
    print("\nEquation")
    print(summary.to_string())

    mg.set_chart_dir(str(config.chart_dir))
    mg.clear_chart_dir()
    lag_note = f" Lag {config.lag}." if config.lag else ""
    spline = config.state == "spline"
    footers = {
        "rfooter": source_footer,
        "lfooter": _LFOOTER_BY_STATE[config.state],
        "lheader_note": f"{config.inflation.title()} CPI.{lag_note}",
        "title": "u* as a spline with knots at the regime dates" if spline
                 else "u* with a regime-specific attractor",
        "lheader": "Knots imposed; the shape within each period is fitted" if spline
                   else "Regime dates imposed; u* is one continuous state throughout",
    }
    plot_ustar(path, frame, regimes, footers)
    plot_gap(path, frame, regimes, footers)
    plot_implied(path, implied, frame, regimes, footers)
    plot_expectations(frame, config, footers)
    plot_surprise(frame, regimes, config, footers)
    plot_levels(table, footers)

    save_diagnostics(
        trace,
        config.chart_dir,
        config.prefix,
        model="regime u*",
        notes=[
            f"{len(labels)} regimes, breaks at {', '.join(config.breaks)}",
            f"{config.inflation} CPI, unemployment at lag {config.lag}",
            f"sample {frame.index[0]}-{frame.index[-1]}, {len(frame)} quarters",
        ],
    )
    return table
