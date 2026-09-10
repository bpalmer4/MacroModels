"""Charts and printed diagnostics for the rstar_rba model."""

from math import lgamma
from pathlib import Path

import arviz as az
import matplotlib.pyplot as plt
import mgplot as mg
import numpy as np
import pandas as pd
import xarray as xr

from src.models.common.sources import footer_from_constants
from src.models.rstar_rba.estimate import load_results, posterior_median

CHART_DIR = Path(__file__).parent.parent.parent.parent / "charts" / "RStarRBA"

_LFOOTER = "Australia. r* from the RBA's response to inflation. "
_ERAS = {
    "1994-2007": ("1994Q1", "2007Q4"),
    "2008-2011": ("2008Q1", "2011Q4"),
    "2012-2015": ("2012Q1", "2015Q4"),
    "2016-2019": ("2016Q1", "2019Q4"),
    "2020-2021": ("2020Q1", "2021Q4"),
    "2022-": ("2022Q1", None),
}


def _group(trace: az.InferenceData, name: str) -> xr.Dataset:
    """Return one group of the trace, narrowed at runtime.

    `InferenceData` exposes its groups dynamically, so a static checker cannot
    see `.posterior`. Checking the type here is both the narrowing and a real
    guard: a trace loaded from an incomplete run has no posterior.
    """
    group = getattr(trace, name, None)
    if not isinstance(group, xr.Dataset):
        raise TypeError(f"trace has no {name} group - was it loaded from a completed run?")
    return group


def _period_index(frame: pd.DataFrame) -> pd.PeriodIndex:
    """Return the frame's index as a PeriodIndex, narrowed at runtime."""
    index = frame.index
    if not isinstance(index, pd.PeriodIndex):
        index = pd.PeriodIndex(index, freq="Q")
    return index


def _era(series: pd.Series, start: str, end: str | None) -> pd.Series:
    return series.loc[start:] if end is None else series.loc[start:end]


# Quarters at or below this cash rate are treated as at the effective lower
# bound for *display*, whether or not the run excluded them from the likelihood.
_FLOOR_DISPLAY = 0.5

# Plain-English axis labels for the prior/posterior charts. A parameter name is
# not a label: nobody outside this file knows what `base_0` is measured in.
_PARAM_LABEL = {
    "lambda": "lambda: cash rate response per band-width of inflation gap",
    "lambda_late": "lambda_late: the same response from the split quarter on",
    "lambda_2": "lambda_2: extra response per unit, squared",
    "rho": "rho: decay of the weights on past inflation (0 = latest quarter only)",
    "sigma_u": "sigma_u: sd of the cash rate around the rule, percentage points",
    "base_0": "base_0: the base trend at 1993Q1, per cent (not r*)",
}


def floor_quarters(frame: pd.DataFrame) -> pd.PeriodIndex:
    """Return the quarters where the cash rate was at the effective lower bound."""
    index = _period_index(frame)
    return index[frame["r"].to_numpy(dtype=float) <= _FLOOR_DISPLAY]


def _floor_note(frame: pd.DataFrame, constants: dict) -> str:
    """Return the one-line caveat about the bound, for a chart footer."""
    pinned = floor_quarters(frame)
    if not len(pinned):
        return ""
    excluded = float(constants.get("floor", -1.0)) > 0
    treatment = "excluded from the likelihood" if excluded else "kept in the likelihood"
    # The dates are readable off the shading, so they are left out: with them the
    # footer runs into the right-hand one at the default figure width.
    return f"Shaded: {len(pinned)} quarters at the lower bound, {treatment}. "


def _floor_span(frame: pd.DataFrame) -> dict | None:
    """Return the axvspan kwargs shading the lower-bound quarters, or None."""
    pinned = floor_quarters(frame)
    if not len(pinned):
        return None
    return {"xmin": pinned[0], "xmax": pinned[-1], "color": "grey", "alpha": 0.18}


def equation(trace: az.InferenceData, constants: dict) -> str:
    """Return the estimated equation as a string, for the chart headers.

    Read off the run rather than hardcoded: the response term and whether the
    base walks both change with the specification, and a chart that states the
    linear form while fitting the quadratic one is worse than no header.

    `b_t` is the base, the slow part. r* is `b_t` plus the response, so the
    first line is both the rule and the definition of r*.
    """
    anchor = float(constants.get("anchor", 2.5))
    response = (
        "lambda_1 x g_t + lambda_2 x g_t x |g_t|"
        if "lambda_2" in _group(trace, "posterior")
        else "lambda x g_t"
    )
    state = "b_t = b_{t-1} + sigma_r x e_t" if constants.get("walk") else "b constant"
    # No default for `band`: a run saved before the scaling existed has no such
    # key, and filling one in would state a scaling the run did not use. That
    # exact silent default printed "(pi - 2.5)/0.5" over a run with no scaling,
    # which changes `lambda`'s units by a factor of two and inverts whether the
    # Taylor principle is satisfied.
    gap = (
        f"g_t = (pi_t - {anchor:g})/{float(constants['band']):g}"
        if "band" in constants
        else f"g_t = pi_t - {anchor:g}"
    )
    return f"r*_t = b_t + {response},  {gap},  {state}"


def _normal_pdf(xs: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    """Return the Normal density. Written out because scipy ships no stubs."""
    z = (xs - mu) / sigma
    return np.exp(-0.5 * z * z) / (sigma * np.sqrt(2.0 * np.pi))


def _beta_pdf(xs: np.ndarray, a: float, b: float) -> np.ndarray:
    """Return the Beta density on (0, 1), zero outside it."""
    log_beta = lgamma(a) + lgamma(b) - lgamma(a + b)
    inside = (xs > 0.0) & (xs < 1.0)
    safe = np.where(inside, xs, 0.5)
    density = np.exp((a - 1.0) * np.log(safe) + (b - 1.0) * np.log1p(-safe) - log_beta)
    return np.where(inside, density, 0.0)


def _prior_curve(name: str, constants: dict, xs: np.ndarray) -> np.ndarray | None:
    """Return the prior density for one parameter, or None if unknown."""
    if name in ("lambda", "lambda_late"):
        return _normal_pdf(
            xs, float(constants.get("lambda_mu", 0.5)), float(constants.get("lambda_sigma", 1.0)),
        )
    if name == "lambda_2":
        return _normal_pdf(
            xs, float(constants.get("lambda2_mu", 0.0)), float(constants.get("lambda2_sigma", 0.5)),
        )
    if name == "rho":
        return _beta_pdf(xs, float(constants.get("rho_a", 2.0)), float(constants.get("rho_b", 2.0)))
    if name == "base_0":
        return _normal_pdf(
            xs, float(constants.get("base_mu", 4.0)), float(constants.get("base_sigma", 3.0)),
        )
    if name == "sigma_u":
        scale = float(constants.get("sigma_u_sigma", 2.0))
        return np.where(xs >= 0.0, 2.0 * _normal_pdf(xs, 0.0, scale), 0.0)
    return None


def plot_prior_posterior(trace: az.InferenceData, constants: dict) -> None:
    """One chart per estimated parameter: posterior against its own prior.

    The point is to see how much of each answer is data. A posterior sitting on
    its prior means the data said nothing, which is exactly what happened to the
    Dirichlet weights and nearly happened to `lambda_2`.
    """
    for name in ("lambda", "lambda_late", "lambda_2", "rho", "sigma_u", "base_0"):
        if name not in _group(trace, "posterior"):
            continue
        draws = np.asarray(_group(trace, "posterior")[name].values).ravel()
        lo, hi = float(np.min(draws)), float(np.max(draws))
        pad = 0.5 * (hi - lo) if hi > lo else 1.0
        xs = np.linspace(lo - pad, hi + pad, 400)
        if name == "rho":
            xs = np.linspace(0.0, 1.0, 400)
        if name == "sigma_u":
            xs = np.linspace(0.0, hi + pad, 400)

        _, ax = plt.subplots()
        ax.hist(draws, bins=60, density=True, color="darkblue", alpha=0.55,
                label="posterior")
        prior = _prior_curve(name, constants, xs)
        if prior is not None:
            ax.plot(xs, prior, color="darkred", lw=2, ls="--", label="prior")
        ax.set_xlabel(_PARAM_LABEL.get(name, name))
        mg.finalise_plot(
            ax,
            title=f"Prior and posterior: {name}",
            ylabel="Density",
            legend={"loc": "best", "fontsize": "small"},
            lheader=equation(trace, constants),
            rfooter=footer_from_constants(constants) or "Built using: RBA F1; ABS 6401.0",
            lfooter=_LFOOTER,
            show=False,
        )


def print_diagnostics(
    trace: az.InferenceData,
    frame: pd.DataFrame,
    constants: dict,
) -> None:
    """Print the parameters and the checks that would show the model failing."""
    scalars = [
        name for name in ("lambda", "lambda_late", "lambda_2", "rho", "sigma_u", "base_0")
        if name in _group(trace, "posterior")
    ]
    print("\nEquation")
    print("-" * 70)
    print(f"  {equation(trace, constants)}")

    print("\nPosterior summary")
    print("-" * 70)
    print(az.summary(trace, var_names=scalars)[
        ["mean", "sd", "hdi_3%", "hdi_97%", "ess_bulk", "r_hat"]
    ].round(3).to_string())
    diverging = _group(trace, "sample_stats").get("diverging")
    if diverging is not None:
        print(f"divergences: {int(diverging.sum())} of {int(diverging.size)}")

    lam = float(_group(trace, "posterior")["lambda"].mean())
    base = posterior_median(trace, "base", _period_index(frame))
    rstar = posterior_median(trace, "nominal_rstar", _period_index(frame))
    stance = posterior_median(trace, "stance", _period_index(frame))
    gap = posterior_median(trace, "inflation_gap", _period_index(frame))

    print("\nDoes the identification hold?")
    print("-" * 70)
    # Against the BASE, which is the gap the rule is written on. Against r*
    # itself this is the residual and tells you nothing.
    print(f"  correlation of the two gaps        {(frame['r'] - base).corr(gap):6.2f}")
    # Both units. Quoting the band-width figure alone reads as a much weaker
    # response than it is, since a band-width is half a percentage point.
    band = float(constants.get("band", 1.0))
    print(f"  lambda, per band-width             {lam:6.2f}")
    print(f"  lambda, per pp of inflation        {lam / band:6.2f}   (Taylor is 1.50)")
    print(f"  r* (neutral nominal cash rate)     {rstar.iloc[-1]:6.2f}")
    print(f"  implied real neutral (less anchor) {rstar.iloc[-1] - float(constants.get('anchor', 2.5)):6.2f}")
    print(f"  base b_t (the slow part alone)     {base.iloc[-1]:6.2f}")
    print(f"  residual sd against cash rate sd   {stance.std():6.2f} vs {frame['r'].std():.2f}")
    # The tell. If the BASE is just the cash rate smoothed, the identification
    # has collapsed into the state and lambda is decoration.
    print(f"  corr(base, cash rate)              {base.corr(frame['r']):6.2f}")

    print("\nResidual by era (cash rate less what the rule implies)")
    print("-" * 70)
    # Only the WHOLE-SAMPLE mean is pinned, by the free starting level of the
    # base. Era means are estimated and are not zero, so these can say policy
    # sat away from the rule. What weakens with duration is how much of a
    # departure survives here rather than being taken up by the base.
    print("  The longer a departure lasts, the more of it the base absorbs.")
    for label, (start, end) in _ERAS.items():
        window = _era(stance, start, end)
        infl = _era(gap, start, end)
        print(f"  {label:<11} stance {window.mean():6.2f}   inflation gap {infl.mean():6.2f}")


def plot_decomposition(trace: az.InferenceData, frame: pd.DataFrame, constants: dict) -> None:
    """Split r* into its two parts: the slow base and the inflation response.

    r*_t = b_t + lambda x g_t, so this is the whole model on one chart. The base
    is the hidden trend that anchors neutral; the response is what the RBA adds
    or subtracts for inflation being off target.

    The point it makes is why this r* can turn quickly without the base ever
    jumping. Over the COVID exit, 2021Q4 to 2023Q2, r* rises 2.41 points: 1.74
    of that is the response and 0.67 the base. Fast movement is available
    through lambda by construction, which is why permitting the base to jump
    buys little (see `ModelConfig.jumps`).

    Both series are in percentage points on one axis, so the base can be read as
    a level and the response as a deviation around zero.
    """
    index = _period_index(frame)
    base = posterior_median(trace, "base", index)
    response = posterior_median(trace, "nominal_rstar", index) - base

    mg.line_plot_finalise(
        pd.DataFrame({
            "Base b_t: the slow trend anchoring neutral": base,
            "Response lambda x g_t: added for inflation off target": response,
        }),
        title="What r* is made of: a slow base and a response to inflation",
        ylabel="Per cent, and percentage points",
        color=["darkblue", "darkred"],
        width=[2.5, 2.0],
        style=["-", "-"],
        annotate=True,
        rounding=2,
        y0=True,
        legend={"loc": "best", "fontsize": "small"},
        lheader=equation(trace, constants),
        rheader="The two add to r*: the base sets the level, the response the turns",
        axvspan=_floor_span(frame),
        rfooter=footer_from_constants(constants) or "Built using: RBA F1; ABS 6401.0",
        lfooter=_LFOOTER + _floor_note(frame, constants),
        show=False,
    )


def plot_rule(trace: az.InferenceData, frame: pd.DataFrame, constants: dict) -> None:
    """Plot the cash rate against what the two-gaps rule implies."""
    rstar = posterior_median(trace, "nominal_rstar", _period_index(frame))
    base = posterior_median(trace, "base", _period_index(frame))
    mg.line_plot_finalise(
        pd.DataFrame({
            "Cash rate": frame["r"],
            "r* (neutral nominal): base + lambda x inflation gap": rstar,
            "Base b_t (the slow part alone, not r*)": base,
        }),
        title="The cash rate, the RBA's estimated rule, and neutral",
        ylabel="Cash rate, per cent a year",
        color=["black", "darkred", "darkblue"],
        width=[1.5, 2.5, 2.0],
        style=["-", "-", "--"],
        annotate=True,
        rounding=2,
        legend={"loc": "best", "fontsize": "small"},
        lheader=equation(trace, constants),
        rheader="The rule is clouded at the lower bound: see the shading",
        axvspan=_floor_span(frame),
        rfooter=footer_from_constants(constants) or "Built using: RBA F1; ABS 6401.0",
        lfooter=_LFOOTER + _floor_note(frame, constants),
        show=False,
    )


# The residual chart, `plot_stance`, was built and removed. It plotted
# `u_t = r_t - r*_t` under the title "How far the cash rate sat from the RBA's
# estimated reaction function", which reads as a stance measure and is not one.
#
# The base is a random walk fitted to the same cash rate, so the longer a
# departure lasts the more of it is taken up by neutral rather than left here.
# The line therefore understates any lasting stance, by an amount that grows
# with its duration and that nobody has quantified. It is not zero by
# construction, and the era means are not zero: what it cannot do is tell you
# how much of a decade-long stance has already been absorbed. The title invited
# readers to take it as a stance measure anyway.
#
# Nothing is lost: the same quantity is the gap between the cash rate and r* on
# the rule chart, and the size of the residual is reported as `sigma_u` and in
# the printed era table, where the caveat can travel with the number.
#
# For a level judgement on policy, the neutral rate has to come from outside the
# RBA's own behaviour. That is `rstar_bonds`.


def plot_two_gaps(trace: az.InferenceData, frame: pd.DataFrame, constants: dict) -> None:
    """Plot the two gaps against each other, which is the whole assumption.

    If they are proportional the cloud has a slope. If it does not, `lambda` is
    not identified and no amount of state machinery will rescue it. Raw
    matplotlib because mgplot has no scatter; `finalise_plot` still styles it.

    The rate gap here is measured against the BASE, not against r*. The base is
    the hidden trend the rule reacts around; the gap against r* is the residual
    by construction and would show nothing.
    """
    base = posterior_median(trace, "base", _period_index(frame))
    gap = posterior_median(trace, "inflation_gap", _period_index(frame))
    rate_gap = frame["r"] - base
    lam = float(_group(trace, "posterior")["lambda"].mean())

    # The bound quarters are marked rather than shaded: on a scatter there is no
    # time axis to shade, and these are the points that most distort curvature.
    # The rate gap is truncated there, so a large inflation gap meets a response
    # the RBA could not deliver, which reads as a weak or concave reaction.
    pinned = floor_quarters(frame)
    at_bound = gap.index.isin(pinned)

    _, ax = plt.subplots()
    ax.scatter(gap[~at_bound], rate_gap[~at_bound], s=24, color="darkblue",
               alpha=0.7, label="quarters")
    if at_bound.any():
        ax.scatter(gap[at_bound], rate_gap[at_bound], s=42, facecolors="none",
                   edgecolors="crimson", linewidths=1.4,
                   label=f"at the lower bound ({int(at_bound.sum())})")
    # The fitted response, including the quadratic term when there is one:
    # drawing only `lambda x g` on a non-linear run shows the wrong curve.
    xs = pd.Series(np.linspace(gap.min(), gap.max(), 100))
    fitted = lam * xs
    label = f"fitted, lambda = {lam:.2f}"
    if "lambda_2" in _group(trace, "posterior"):
        lam2 = float(_group(trace, "posterior")["lambda_2"].mean())
        fitted = fitted + lam2 * xs * xs.abs()
        label = f"fitted, lambda_1 = {lam:.2f}, lambda_2 = {lam2:.2f}"
    ax.plot(xs, fitted, color="darkred", lw=2, label=label)
    units = ("band-widths (1.0 = the edge of the 2-3% band)" if "band" in constants
             else "percentage points")
    ax.set_xlabel(f"Inflation less the 2.5% target, {units}")
    ax.axhline(0, color="grey", lw=0.8)
    ax.axvline(0, color="grey", lw=0.8)

    mg.finalise_plot(
        ax,
        title="How the cash rate responded to inflation",
        ylabel="Cash rate less the base trend, percentage points",
        legend={"loc": "best", "fontsize": "small"},
        lheader=equation(trace, constants),
        rheader="Circled quarters cannot inform the response: the rate gap is truncated",
        rfooter=footer_from_constants(constants) or "Built using: RBA F1; ABS 6401.0",
        lfooter=_LFOOTER + _floor_note(frame, constants).replace("Shaded:", "Circled:"),
        show=False,
    )



def posterior_draws(trace: az.InferenceData, name: str, index: pd.PeriodIndex) -> pd.DataFrame:
    """Return every draw of a vector quantity, quarters down, draws across."""
    stacked = _group(trace, "posterior")[name].stack(sample=("chain", "draw"))  # noqa: PD013
    return pd.DataFrame(np.asarray(stacked.values), index=index)


def band_of(draws: pd.DataFrame, prob: float = 0.90) -> pd.DataFrame:
    """Return the equal-tailed credible interval of a draw matrix."""
    tail = (1.0 - prob) / 2.0
    return pd.DataFrame({
        "lower": draws.quantile(tail, axis=1),
        "upper": draws.quantile(1.0 - tail, axis=1),
    })


def posterior_band(
    trace: az.InferenceData, name: str, index: pd.PeriodIndex, prob: float = 0.90,
) -> pd.DataFrame:
    """Return the equal-tailed credible interval of a vector latent."""
    return band_of(posterior_draws(trace, name, index), prob)


def plot_rstar_real_nominal(trace: az.InferenceData, frame: pd.DataFrame, constants: dict) -> None:
    """Real and nominal r*, each with a 90% credible interval.

    The same chart `rstar_bonds` draws, for this model's estimate.

    r* here is the model's COMPLETE estimate of the neutral cash rate,
    `rstar` = b_t + lambda x g_t, not the base `base` alone. The base is the
    hidden trend that roughly anchors neutral; the reaction function is the
    part tied directly to the inflation gap. What the model says the neutral
    cash rate is at a point in time is the two together.

    Real is that less the target, which is what `rstar_bonds` and `rstar_hlw`
    also do, so the three are comparable and the two lines here have the same
    shape. Deflating by realised inflation instead was tried and is wrong for a
    neutral rate: a neutral rate is defined at target inflation, and subtracting
    what inflation actually did makes the "real neutral" swing with the cycle
    (it hit -2.2 in 2022 purely because inflation peaked).

    The band is conditional on `sigma_r`, which is imposed. It is the
    uncertainty given the smoothness assumption, not the whole of it.
    """
    index = _period_index(frame)
    anchor = float(constants.get("anchor", 2.5))
    nominal_draws = posterior_draws(trace, "nominal_rstar", index)
    real_draws = posterior_draws(trace, "real_rstar", index)

    nominal_median = nominal_draws.median(axis=1)
    real_median = real_draws.median(axis=1)
    nominal = band_of(nominal_draws)
    real = band_of(real_draws)

    last = index[-1]
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
            "Nominal r* (base trend + inflation response)": nominal_median,
            f"Real r* (less the {anchor:g}% target)": real_median,
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
        title="Australia's real and nominal r-star, from the RBA's reaction function",
        ylabel="Per cent",
        y0=True,
        legend={"loc": "best", "fontsize": "small"},
        lheader="r*_t = b_t + lambda x g_t",
        rheader=f"{last} 90%: {ends}",
        axvspan=_floor_span(frame),
        rfooter=footer_from_constants(constants) or "Built using: RBA F1; ABS 6401.0",
        lfooter=_LFOOTER + "Band conditional on the imposed sigma_r. ",
        show=False,
    )



def _taylor_inputs(prefix: str = "rstar_bonds") -> tuple[pd.Series, pd.Series]:
    """Return supply-adjusted inflation and the output gap for a Taylor rule.

    Both come from the joint y*/u* run, read here through the saved
    `rstar_bonds` results, which already assemble them. Chart-only: a missing
    run costs this chart and nothing else.
    """
    from src.models.rstar_bonds.results import load_results as load_bonds  # noqa: PLC0415

    try:
        bonds = load_bonds(prefix=prefix)
    except (FileNotFoundError, KeyError, ValueError) as exc:
        print(f"  note: Taylor inputs unavailable ({type(exc).__name__}); "
              "the Taylor chart will be skipped")
        return pd.Series(dtype=float), pd.Series(dtype=float)
    return bonds.rule_inflation(), bonds._extra("ygap")  # noqa: SLF001 — same repo


def plot_taylor(trace: az.InferenceData, frame: pd.DataFrame, constants: dict) -> None:
    """Plot a Taylor rule built on this model's r*.

        i* = real r* + pi_core + 0.5 (pi_core - 2.5) + 0.5 ygap

    On R*, the complete estimate `b_t + lambda x g_t` less the target. Not on
    the base. The base is one term inside r*, and a Taylor rule fed the base is
    a Taylor rule on something this model never calls neutral.

    `pi_core` and `ygap` come from the joint y*/u* run, because a Taylor rule
    wants a broad core measure and an activity term, neither of which this model
    carries. So the blue line is not built purely from this model.

    Known and deliberate: r* already contains the RBA's own response to
    inflation, and Taylor's rule adds a second one, so the inflation gap is
    counted twice. That makes the level of the prescription an overstatement
    whenever inflation is away from target. It is recorded on the chart rather
    than fixed, because the alternative, substituting the base, is the error
    this docstring exists to prevent.
    """
    pi_core, ygap = _taylor_inputs()
    if pi_core.empty:
        return

    index = _period_index(frame)
    anchor = float(constants.get("anchor", 2.5))
    rstar_real = posterior_median(trace, "real_rstar", index)

    frame_t = pd.DataFrame({
        "rstar_real": rstar_real,
        "pi_core": pi_core.reindex(index),
        "ygap": ygap.reindex(index),
    }).dropna()
    taylor = (
        frame_t["rstar_real"] + frame_t["pi_core"]
        + 0.5 * (frame_t["pi_core"] - anchor) + 0.5 * frame_t["ygap"]
    )

    mg.line_plot_finalise(
        pd.DataFrame({
            "Cash rate": frame["r"],
            "Taylor rule on this model's r*": taylor,
            "This model's r* (nominal)": posterior_median(trace, "nominal_rstar", index),
        }).dropna(how="all"),
        title="A Taylor rule on the neutral rate the RBA's behaviour implies",
        ylabel="Cash rate, per cent a year",
        color=["black", "darkblue", "darkred"],
        width=[1.5, 2.5, 2.0],
        style=["-", "-", "--"],
        annotate=True,
        rounding=2,
        legend={"loc": "best", "fontsize": "small"},
        lheader=f"i* = r* + pi_core + 0.5(pi_core - {anchor:g}) + 0.5 x output gap",
        rheader="Counts the inflation gap twice: r* already carries the RBA's own response",
        axvspan=_floor_span(frame),
        rfooter=footer_from_constants(constants) or "Built using: RBA F1; ABS 6401.0",
        lfooter=_LFOOTER + _floor_note(frame, constants),
        show=False,
    )


def run_analysis(
    output_dir: Path | str | None = None,
    prefix: str = "rstar_rba",
    chart_dir: Path | str | None = None,
) -> None:
    """Load a saved run, print the diagnostics, write the charts."""
    trace, frame, constants = load_results(output_dir=output_dir, prefix=prefix)
    print_diagnostics(trace, frame, constants)

    if chart_dir is None:
        chart_dir = CHART_DIR if prefix == "rstar_rba" else CHART_DIR.parent / f"RStarRBA_{prefix}"
    mg.set_chart_dir(str(chart_dir))
    mg.clear_chart_dir()
    plot_rstar_real_nominal(trace, frame, constants)
    plot_rule(trace, frame, constants)
    plot_decomposition(trace, frame, constants)
    plot_two_gaps(trace, frame, constants)
    plot_taylor(trace, frame, constants)
    plot_prior_posterior(trace, constants)
    print(f"\nCharts written to: {chart_dir}")
