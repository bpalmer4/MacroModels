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
from src.models.rstar_rba.ensemble import load_ensemble, print_ensemble
from src.models.rstar_rba.estimate import load_results, posterior_median
from src.models.rstar_rba.injection import load_injection, print_injection

CHART_DIR = Path(__file__).parent.parent.parent.parent / "charts" / "RStarRBA"

_LFOOTER = "Australia. Neutral inferred from the RBA's response to inflation. "
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

# One style per ensemble member, in the order the values are run, so the two
# ensemble charts agree line for line. No brown or orange: those vanish into the
# envelope fill on the neutral chart. The default member carries a heavy solid line
# and everything else is dashed, so the shipped value is readable at a glance.
_ENSEMBLE_COLORS = ["darkgreen", "darkblue", "darkred", "purple"]
_ENSEMBLE_WIDTHS = [1.5, 2.5, 1.5, 1.5]
_ENSEMBLE_STYLES = ["--", "-", ":", "-."]

# Plain-English axis labels for the prior/posterior charts. A parameter name is
# not a label: nobody outside this file knows what `base_0` is measured in.
_PARAM_LABEL = {
    "lambda": "lambda: cash rate response per band-width of inflation gap",
    "lambda_late": "lambda_late: the same response from the split quarter on",
    "lambda_2": "lambda_2: extra response per unit, squared",
    "rho": "rho: decay of the weights on past inflation (0 = latest quarter only)",
    "sigma_eps": "sigma_eps: sd of the cash rate around the rule, percentage points",
    "base_0": "base_0: neutral at 1993Q1, per cent (before the inflation response)",
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

    `b_t` is neutral, the slow part. Adding the response gives the rule's
    prescribed rate, so the first line is the rule, not the definition of
    neutral.
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
    return f"d_t = b_t + {response},  {gap},  {state}"


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
    if name == "sigma_eps":
        scale = float(constants.get("sigma_eps_sigma", 2.0))
        return np.where(xs >= 0.0, 2.0 * _normal_pdf(xs, 0.0, scale), 0.0)
    return None


def plot_prior_posterior(trace: az.InferenceData, constants: dict) -> None:
    """One chart per estimated parameter: posterior against its own prior.

    The point is to see how much of each answer is data. A posterior sitting on
    its prior means the data said nothing, which is exactly what happened to the
    Dirichlet weights and nearly happened to `lambda_2`.
    """
    for name in ("lambda", "lambda_late", "lambda_2", "rho", "sigma_eps", "base_0"):
        if name not in _group(trace, "posterior"):
            continue
        draws = np.asarray(_group(trace, "posterior")[name].values).ravel()
        lo, hi = float(np.min(draws)), float(np.max(draws))
        pad = 0.5 * (hi - lo) if hi > lo else 1.0
        xs = np.linspace(lo - pad, hi + pad, 400)
        if name == "rho":
            xs = np.linspace(0.0, 1.0, 400)
        if name == "sigma_eps":
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
        name for name in ("lambda", "lambda_late", "lambda_2", "phi", "rho", "sigma_eps", "base_0")
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
    neutral = posterior_median(trace, "neutral", _period_index(frame))
    prescribed = posterior_median(trace, "prescribed", _period_index(frame))
    residual = posterior_median(trace, "rule_residual", _period_index(frame))
    gap = posterior_median(trace, "inflation_gap", _period_index(frame))

    print("\nDoes the identification hold?")
    print("-" * 70)
    # Against NEUTRAL, which is the gap the rule is written on. Against the
    # prescribed rate this is the residual and tells you nothing.
    print(f"  correlation of the two gaps        {(frame['r'] - neutral).corr(gap):6.2f}")
    # Both units. Quoting the band-width figure alone reads as a much weaker
    # response than it is, since a band-width is half a percentage point.
    band = float(constants.get("band", 1.0))
    print(f"  lambda, per band-width             {lam:6.2f}")
    # No Taylor comparison here any more. It is not the same object: `lambda`
    # per pp is the Fisher pass-through plus the real response, one coefficient
    # doing two jobs, and Taylor's 1.5 is the threshold only under full
    # pass-through. Separating them needs an expectations series this package
    # deliberately does not import. See MODEL_NOTES.md.
    print(f"  lambda, per pp of inflation        {lam / band:6.2f}   (nominal)")
    anchor = float(constants.get("anchor", 2.5))
    print(f"  neutral b_t, nominal               {neutral.iloc[-1]:6.2f}")
    print(f"  neutral b_t, real (less anchor)    {neutral.iloc[-1] - anchor:6.2f}")
    print(f"  prescribed rate b_t + lambda.g     {prescribed.iloc[-1]:6.2f}")
    print(f"  stance, cash rate less neutral     {frame['r'].iloc[-1] - neutral.iloc[-1]:+6.2f}")
    print(f"  rule residual sd vs cash rate sd   {residual.std():6.2f} vs {frame['r'].std():.2f}")
    # The tell. If NEUTRAL is just the cash rate smoothed, the identification
    # has collapsed into the state and lambda is decoration.
    print(f"  corr(neutral, cash rate)           {neutral.corr(frame['r']):6.2f}")

    print("\nRule residual by era (cash rate less the rule's prescribed rate)")
    print("-" * 70)
    # Only the WHOLE-SAMPLE mean is pinned, by the free starting level of
    # neutral. Era means are estimated and are not zero, so these can say policy
    # sat away from the rule. What weakens with duration is how much of a
    # departure survives here rather than being taken up by neutral.
    print("  The longer a departure lasts, the more of it neutral absorbs.")
    for label, (start, end) in _ERAS.items():
        window = _era(residual, start, end)
        infl = _era(gap, start, end)
        print(f"  {label:<11} residual {window.mean():6.2f}   inflation gap {infl.mean():6.2f}")


def plot_decomposition(trace: az.InferenceData, frame: pd.DataFrame, constants: dict) -> None:
    """Split the prescribed rate into neutral and the inflation response.

    d_t = b_t + lambda x g_t, so this is the whole model on one chart. Neutral
    is the slow trend; the response is what the RBA adds or subtracts for
    inflation being away from target.

    The point it makes is why the prescribed rate can turn quickly without
    neutral ever jumping. Over the COVID exit, 2021Q4 to 2023Q2, it rises 2.41
    points: 1.74 of that is the response and 0.67 neutral. Fast movement is
    available through lambda by construction, which is why permitting neutral to
    jump buys little (see `ModelConfig.jumps`).

    Both series are in percentage points on one axis, so the base can be read as
    a level and the response as a deviation around zero.
    """
    index = _period_index(frame)
    base = posterior_median(trace, "neutral", index)
    response = posterior_median(trace, "prescribed", index) - base

    mg.line_plot_finalise(
        pd.DataFrame({
            "Base b_t: the slow trend anchoring neutral": base,
            "Response lambda x g_t: added for inflation off target": response,
        }),
        title="What the rule prescribes: neutral plus a response to inflation",
        ylabel="Per cent, and percentage points",
        color=["darkblue", "darkred"],
        width=[2.5, 2.0],
        style=["-", "-"],
        annotate=True,
        rounding=2,
        y0=True,
        legend={"loc": "best", "fontsize": "small"},
        lheader=equation(trace, constants),
        rheader="Neutral sets the level, the response supplies the turns",
        axvspan=_floor_span(frame),
        rfooter=footer_from_constants(constants) or "Built using: RBA F1; ABS 6401.0",
        lfooter=_LFOOTER + _floor_note(frame, constants),
        show=False,
    )


def plot_rule(trace: az.InferenceData, frame: pd.DataFrame, constants: dict) -> None:
    """Plot the cash rate against what the two-gaps rule implies."""
    rstar = posterior_median(trace, "prescribed", _period_index(frame))
    base = posterior_median(trace, "neutral", _period_index(frame))
    mg.line_plot_finalise(
        pd.DataFrame({
            "Cash rate": frame["r"],
            "Prescribed rate: b_t + lambda x inflation gap": rstar,
            "Neutral b_t (the slow base)": base,
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
# `rule_residual`, the cash rate less the rule's prescribed rate, under the
# title "How far the cash rate sat from the RBA's estimated reaction function",
# which reads as a stance measure and is not one.
#
# Neutral is a random walk fitted to the same cash rate, so the longer a
# departure lasts the more of it is taken up by neutral rather than left here.
# The line therefore understates any lasting stance, by an amount that grows
# with its duration and that the injection test has now quantified: a known
# stance shows up here at 0.78 of its size after a year, 0.44 after four and
# 0.25 after ten, so the understatement is severe well inside the horizons
# people want to read it over. It is not zero by
# construction, and the era means are not zero: what it cannot do is tell you
# how much of a decade-long stance has already been absorbed. The title invited
# readers to take it as a stance measure anyway.
#
# Nothing is lost: the same quantity is the gap between the cash rate and the
# prescribed rate on the rule chart, and its size is reported as `sigma_eps` and in
# the printed era table, where the caveat can travel with the number.
#
# For a level judgement on policy, the neutral rate has to come from outside the
# RBA's own behaviour. That is `rstar_bonds`.


def plot_two_gaps(trace: az.InferenceData, frame: pd.DataFrame, constants: dict) -> None:
    """Plot the two gaps against each other, which is the whole assumption.

    If they are proportional the cloud has a slope. If it does not, `lambda` is
    not identified and no amount of state machinery will rescue it. Raw
    matplotlib because mgplot has no scatter; `finalise_plot` still styles it.

    The rate gap here is measured against NEUTRAL, not against the prescribed
    rate. Neutral is the trend the rule reacts around; the gap against the
    prescribed rate is the residual by construction and would show nothing.
    """
    base = posterior_median(trace, "neutral", _period_index(frame))
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
    """Real and nominal neutral, each with a 90% credible interval.

    The same chart `rstar_bonds` draws, for this model's estimate.

    The line is `b_t`, the slow base, which is what this package now calls the
    neutral rate. The complete systematic term `b_t + lambda x g_t` is the rule's
    prescribed cash rate, not neutral: the inflation response is a departure
    FROM neutral, which is the conventional reading of a policy rule and the one
    MODEL_NOTES.md now adopts.

    This chart used to plot the complete term. Building it on the base lowers
    today's real reading from 0.98 to 0.49 and makes it comparable for the first
    time with `rstar_bonds` and `rstar_hlw`, which both estimate an intercept.

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
    # `neutral`, not `prescribed`. The deflator is applied to the draws rather
    # than reading `neutral_real`, so the two bands come from one draw matrix.
    nominal_draws = posterior_draws(trace, "neutral", index)
    real_draws = nominal_draws - anchor

    nominal_median = nominal_draws.median(axis=1)
    real_median = real_draws.median(axis=1)
    nominal = band_of(nominal_draws)
    real = band_of(real_draws)

    last = index[-1]
    ends = " | ".join(
        f"{name} {band['lower'].iloc[-1]:.2f} to {band['upper'].iloc[-1]:.2f} "
        f"(median {median.iloc[-1]:.2f})"
        for name, band, median in (
            ("real", real, real_median),
            ("nominal", nominal, nominal_median),
        )
    )

    ax = mg.fill_between_plot(nominal, color="darkblue", alpha=0.15, label="Nominal 90% HDI")
    mg.fill_between_plot(real, ax=ax, color="darkorange", alpha=0.20, label="Real 90% HDI")
    mg.line_plot(
        pd.DataFrame({
            "Nominal neutral b_t (the slow base)": nominal_median,
            f"Real neutral (less the {anchor:g}% target)": real_median,
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
        title="Australia's neutral cash rate, from the RBA's reaction function",
        ylabel="Per cent",
        y0=True,
        legend={"loc": "best", "fontsize": "small"},
        lheader="Neutral = b_t. The rule prescribes b_t + lambda x g_t on top of it",
        rheader=f"{last} 90%: {ends}",
        axvspan=_floor_span(frame),
        rfooter=footer_from_constants(constants) or "Built using: RBA F1; ABS 6401.0",
        lfooter=_LFOOTER + "Band conditional on the imposed sigma_r. ",
        show=False,
    )



def plot_sigma_r_ensemble(
    trace: az.InferenceData,
    frame: pd.DataFrame,
    constants: dict,
    paths: pd.DataFrame,
) -> None:
    """Real neutral across defensible `sigma_r`, against the conditional band.

    Neutral is `b_t`, so this is the base less the target, matching the headline
    chart. The companion chart is the same object in nominal terms drawn against
    the cash rate; the two differ in units and in what they are compared with,
    which is a thin distinction and a candidate for merging.

    The point of the chart is the COMPARISON, so both are drawn. The narrow
    fill is the 90% credible interval from the default run, which is the
    uncertainty the data supplies once the smoothness is granted. The wide fill
    is the spread of posterior medians across `sigma_r` in 0.05 to 0.15, which
    is the smoothness assumption moving. Where the second is wider than the
    first, the headline interval is understating what is not known, and the
    level should be quoted as a range.

    NOT ADDITIVE. These are uncertainty of two different kinds and stacking
    them would double-count: every ensemble member has a band of its own, and
    the band drawn here is one member's. Read the envelope as where the line
    could sit, and the band as how sharply any one choice pins it.

    `lambda` deliberately does not appear. It is stable across this ensemble, so
    the chart is about the quantity that is not. Read that stability narrowly
    though: it holds across `sigma_r` while policy smoothing is held at zero,
    and not across `phi`. See MODEL_NOTES.md point 6.
    """
    index = _period_index(frame)
    anchor = float(constants.get("anchor", 2.5))
    default = float(constants.get("sigma_r", 0.10))

    # On the base less the target, matching the ensemble paths: neutral is b_t.
    band = posterior_band(trace, "neutral", index) - anchor
    envelope = pd.DataFrame({"lower": paths.min(axis=1), "upper": paths.max(axis=1)})
    spread = float((envelope["upper"] - envelope["lower"]).iloc[-1])
    width = float((band["upper"] - band["lower"]).iloc[-1])

    ax = mg.fill_between_plot(
        envelope, color="darkorange", alpha=0.22,
        label=f"Across sigma_r {paths.columns[0]} to {paths.columns[-1]}",
    )
    mg.fill_between_plot(
        band, ax=ax, color="darkblue", alpha=0.18,
        label=f"90% credible interval at sigma_r = {default:g}",
    )
    mg.line_plot(
        paths.rename(columns={col: f"sigma_r = {col}" for col in paths.columns}),
        ax=ax,
        # Not a brown for the third line: it disappears into the orange
        # envelope fill it is meant to be read against.
        color=_ENSEMBLE_COLORS[: paths.shape[1]],
        width=_ENSEMBLE_WIDTHS[: paths.shape[1]],
        style=_ENSEMBLE_STYLES[: paths.shape[1]],
        annotate=True,
        rounding=2,
    )
    mg.finalise_plot(
        ax,
        title="How much of Australia's neutral rate is the smoothness we imposed",
        ylabel="Per cent",
        y0=True,
        legend={"loc": "best", "fontsize": "small"},
        lheader=f"Ensemble test. Real neutral = b_t - {anchor:g}",
        rheader=f"{index[-1]}: assumption spread {spread:.2f}pp vs credible interval {width:.2f}pp",
        axvspan=_floor_span(frame),
        rfooter=footer_from_constants(constants) or "Built using: RBA F1; ABS 6401.0",
        lfooter=_LFOOTER + "sigma_r is imposed, never estimated. ",
        show=False,
    )


def plot_base_ensemble(
    frame: pd.DataFrame, constants: dict, bases: pd.DataFrame, table: pd.DataFrame,
) -> None:
    """Plot the BASE across `sigma_r`, with the cash rate it is being read off.

    This is `b_t`, the slow trend on its own, and the companion chart is
    `b_t + lambda x g_t`. Which of those two deserves the name "neutral" is an
    open question in this package, set out at the top of MODEL_NOTES.md, so the
    chart names the line it draws and gives the other one in the header rather
    than ruling on it. What is not in doubt is that they are different lines and
    give materially different answers downstream, so a number taken from here
    has to say which one it is.

    Why it is worth drawing. The companion chart shows the assumption
    spread on the published number; this shows where that spread comes from, and
    the two are not the same size. `sigma_r` decides how much of the cash rate's
    movement counts as drift in the trend rather than response to inflation, so
    raising it moves the base bodily while `lambda` shrinks to compensate, and
    the sum moves less than the base does, 0.95 against 1.10 points. The gap
    between the two spreads is the response absorbing what the base gives up.

    The cash rate is drawn behind because it is what the failure mode looks
    like: `corr(base, cash rate)` runs 0.80, 0.88, 0.93 across the ensemble, so
    the upper member is visibly the cash rate smoothed, and at some larger
    `sigma_r` the base would be nothing else and `lambda` would be decoration.
    That is the reason not to read the top of the range as equally defensible
    with the bottom, and it is easier to see than to assert.

    Nominal, like the base figures in the notes and on the rule chart, and so
    directly comparable with the cash rate drawn beside it. The companion chart
    is real, so the two are not on the same level.
    """
    columns = {f"Base b_t at sigma_r = {col}": bases[col] for col in bases.columns}
    ax = mg.line_plot(
        pd.DataFrame({"Cash rate": frame["r"], **columns}),
        color=["black", *_ENSEMBLE_COLORS[: bases.shape[1]]],
        width=[1.2, *_ENSEMBLE_WIDTHS[: bases.shape[1]]],
        style=["-", *_ENSEMBLE_STYLES[: bases.shape[1]]],
        annotate=True,
        rounding=2,
    )
    # The other half of the trade, and the reason the base moving this much does
    # not move the sum as much. Written on the chart because the base lines
    # alone look like pure disagreement about the level, and they are not: each
    # is paired with a different response.
    lines = ["lambda, per pp of inflation"] + [
        f"  sigma_r {value:g}:   {row['lambda_pp']:.2f}" for value, row in table.iterrows()
    ]
    ax.text(
        0.985, 0.97, "\n".join(lines), transform=ax.transAxes,
        ha="right", va="top", fontsize="small", family="monospace",
        bbox={"boxstyle": "round", "facecolor": "white", "edgecolor": "grey", "alpha": 0.85},
    )
    mg.finalise_plot(
        ax,
        title="The slow base across the smoothness assumption",
        ylabel="Per cent a year, nominal",
        legend={"loc": "best", "fontsize": "small"},
        lheader="Ensemble test. b_t alone; the complete systematic term is b_t + lambda x g_t",
        rheader="A higher sigma_r buys a base that tracks the cash rate more closely",
        axvspan=_floor_span(frame),
        rfooter=footer_from_constants(constants) or "Built using: RBA F1; ABS 6401.0",
        lfooter=_LFOOTER + "b_t is the slow trend, before the inflation response. ",
        show=False,
    )


def plot_injection(saved: dict, constants: dict) -> None:
    """How much of a known stance comes back, against how long it lasted.

    A known +1pp was added to the cash rate over windows of several lengths and
    the model re-estimated. The bars split that point in two: what the residual
    reports, which is the model getting it right, and what the base takes
    instead, which the model reports as neutral having moved.

    This is the model's honesty curve, and it is the thing the notes previously
    inferred from `sigma_r x sqrt(T)` rather than measured. Read the crossover:
    to the left of it a departure from the rule is mostly seen, to the right it
    is mostly reported as a change in neutral.

    The two bars sum to slightly under the whole because a little of the
    injected point is taken up by `lambda` shifting, which is reported in the
    printed table and is small.
    """
    table = saved["table"]
    size = float(saved["size"])
    shares = pd.DataFrame({
        "Reported as a departure from the rule": table["recovered"],
        "Absorbed: reported as neutral having moved": 1.0 - table["recovered"],
    })
    shares.index = pd.Index([f"{int(years)}y" for years in table.index], name="Length of the imposed stance")

    ax = mg.bar_plot(
        shares, stacked=True, color=["darkgreen", "darkred"], annotate=True, rounding=2,
    )
    mg.finalise_plot(
        ax,
        title="How much of a known policy stance this model still sees",
        ylabel="Share of the imposed stance",
        xlabel="Length of the imposed stance",
        # Stacked shares fill the axis to 1.0, so there is no gap for a legend
        # to find. The headroom is what makes "best" a real choice here.
        ylim=(0.0, 1.22),
        legend={"loc": "best", "fontsize": "small"},
        lheader=f"+{size:.2f}pp added to the cash rate, windows ending {saved['end']}",
        rheader="Inflation held at what it actually did",
        rfooter=footer_from_constants(constants) or "Built using: RBA F1; ABS 6401.0",
        lfooter=_LFOOTER + "Measured, not inferred from the prior. ",
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
    """Plot a Taylor rule built on this model's neutral rate.

        i* = real neutral + pi_core + 0.5 (pi_core - 2.5) + 0.5 ygap

    On `b_t` less the target, which is the intercept a Taylor rule wants.

    THE DOUBLE-COUNT IS GONE. This chart used to be built on the complete term
    `b_t + lambda x g_t`, which already contains the RBA's own response to
    inflation, so Taylor's `0.5 (pi_core - 2.5)` added a second one and the
    prescription overstated whenever inflation was away from target. That was a
    stated bias on a chart. Feeding the intercept removes it rather than
    documenting it, and is the clearest practical gain from the naming change.

    `pi_core` and `ygap` come from the joint y*/u* run, because a Taylor rule
    wants a broad core measure and an activity term, neither of which this model
    carries. So the blue line is not built purely from this model.
    """
    pi_core, ygap = _taylor_inputs()
    if pi_core.empty:
        return

    index = _period_index(frame)
    anchor = float(constants.get("anchor", 2.5))
    rstar_real = posterior_median(trace, "neutral", index) - anchor

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
            "Taylor rule on this model's neutral rate": taylor,
            "This model's neutral rate b_t (nominal)": posterior_median(trace, "neutral", index),
        }).dropna(how="all"),
        title="A Taylor rule on the neutral rate the RBA's behaviour implies",
        ylabel="Cash rate, per cent a year",
        color=["black", "darkblue", "darkred"],
        width=[1.5, 2.5, 2.0],
        style=["-", "-", "--"],
        annotate=True,
        rounding=2,
        legend={"loc": "best", "fontsize": "small"},
        lheader=f"i* = b_t - {anchor:g} + pi_core + 0.5(pi_core - {anchor:g}) + 0.5 x output gap",
        rheader="Core inflation and the output gap come from the joint y*/u* model",
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

    # Optional: only present once --sigma-r-ensemble has been run. Absent, the
    # rest of the analysis is unaffected, so a saved run from before the
    # ensemble existed still charts.
    ensemble = load_ensemble(output_dir=output_dir, prefix=prefix)
    if ensemble is not None:
        print_ensemble(ensemble["table"])
    injection = load_injection(output_dir=output_dir, prefix=prefix)
    if injection is not None:
        print_injection(injection)

    if chart_dir is None:
        chart_dir = CHART_DIR if prefix == "rstar_rba" else CHART_DIR.parent / f"RStarRBA_{prefix}"
    mg.set_chart_dir(str(chart_dir))
    mg.clear_chart_dir()
    plot_rstar_real_nominal(trace, frame, constants)
    if ensemble is not None:
        plot_sigma_r_ensemble(trace, frame, constants, ensemble["paths"])
        bases = ensemble.get("bases")
        if bases is not None:
            plot_base_ensemble(frame, constants, bases, ensemble["table"])
    if injection is not None:
        plot_injection(injection, constants)
    plot_rule(trace, frame, constants)
    plot_decomposition(trace, frame, constants)
    plot_two_gaps(trace, frame, constants)
    plot_taylor(trace, frame, constants)
    plot_prior_posterior(trace, constants)
    print(f"\nCharts written to: {chart_dir}")
