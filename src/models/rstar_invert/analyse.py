"""Charts for the conditional inversion.

Every chart carries what was asserted, in its header or footer. A reader who
takes one of these paths away without that has taken away an estimate of r*,
which is not what this package produces.
"""

import math
from pathlib import Path
from typing import Any

import arviz as az
import matplotlib.pyplot as plt
import mgplot as mg
import numpy as np
import pandas as pd

from src.models.common import prior_posterior
from src.models.common.diagnostics import save_diagnostics
from src.models.common.sources import footer_from_constants
from src.models.is_curve.observations import DEFAULT_WINDOWS
from src.models.rstar_invert.config import DEFAULT_CHART_DIR, PAIR_LAGS
from src.models.rstar_invert.ensemble import load_ensemble
from src.models.rstar_invert.estimate import (
    load_results,
    posterior_band,
    posterior_median,
    scalar_draws,
)

_LFOOTER = "Australia. r* by conditional inversion: the IS slope is ASSERTED, not measured. "

# The inflation target, used only to put r* on a nominal scale. A neutral rate
# is defined at TARGET inflation, not at whatever inflation happened to be,
# which is the convention `rstar_rba` uses for the same conversion.
TARGET = 2.5


def _chart_dir(prefix: str = "rstar_invert") -> Path:
    """Return the chart folder for a run, one per prefix.

    Derived from the prefix so a variant run (a different lag, say) charts
    beside the default rather than overwriting it.
    """
    return DEFAULT_CHART_DIR / prefix.replace("_", "-")


def _asserted(constants: dict[str, Any]) -> str:
    """Return the one-line statement of what the run asserted."""
    mu = float(constants.get("is_slope_mu", float("nan")))
    sd = float(constants.get("is_slope_sigma", float("nan")))
    if bool(constants.get("rstar_constant", 0.0)):
        speed = "r* FIXED"
    elif bool(constants.get("rstar_linear", 0.0)):
        speed = "r* a straight drift"
    elif bool(constants.get("free_sigma_rstar", 0.0)):
        speed = "r* walk, sigma_rstar estimated"
    else:
        speed = f"r* walk, sigma_rstar = {float(constants.get('sigma_rstar', float('nan'))):g}"
    return f"ASSERTED: is_slope ~ TruncNormal({mu:g}, {sd:g}), {speed}"


def plot_rstar(trace: az.InferenceData, frame: pd.DataFrame, constants: dict[str, Any]) -> None:
    """Plot the inverted r* against the real cash rate it should be neutral to.

    Both on the same axes deliberately. r* here is a smoothed version of
    `r_{t-lag} - x_t/is_slope`, which contains the real rate undiluted, so the
    two tracking each other is the expected failure and the chart should show
    it rather than hide it in a separate panel.
    """
    index = frame.index
    if not isinstance(index, pd.PeriodIndex):
        index = pd.PeriodIndex(index, freq="Q")

    rstar = posterior_median(trace, "rstar", index)
    band = posterior_band(trace, "rstar", index)
    correlation = rstar.corr(frame["real_cash"])

    ax = mg.fill_between_plot(band, color="teal", alpha=0.18, label="r* 90% band (conditional)")
    mg.line_plot(
        pd.DataFrame({
            "Inverted r* (real)": rstar,
            "Real cash rate": frame["real_cash"],
        }),
        ax=ax,
        color=["teal", "darkorange"],
        width=[2.5, 1.5],
        style=["-", "--"],
        annotate=True,
        rounding=2,
    )
    mg.finalise_plot(
        ax,
        title="Australian r* implied by an asserted IS curve",
        ylabel="Per cent",
        y0=True,
        legend={"loc": "best", "fontsize": "small"},
        lheader=_asserted(constants),
        rheader=f"corr(r*, real cash) = {correlation:+.3f}",
        rfooter=footer_from_constants(constants) or "Built using: RBA F1; ABS 5206.0",
        lfooter="Australia. IS slope ASSERTED. Band conditional on that. ",
        show=False,
    )


_SQRT_2PI = math.sqrt(2.0 * math.pi)

# What each estimated scalar is, for the x-axis label.
_PARAM_LABEL = {
    "is_slope": "is_slope: per cent of potential per pp of rate gap",
    "sigma_e": "sigma_e: sd of the IS residual, per cent of potential",
    "rstar_0": "rstar_0: starting level of r*, per cent",
    "lag_weight": "w: weight on the shorter lag",
    "rstar_trend": "rstar_trend: drift in r*, pp per quarter",
    "sigma_rstar": "sigma_rstar: sd of r*'s quarterly innovation, pp",
}


def _normal_pdf(xs: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    """Return the normal density, written out to avoid a scipy dependency."""
    return np.exp(-0.5 * ((xs - mu) / sigma) ** 2) / (sigma * _SQRT_2PI)


def _prior_curve(
    name: str, constants: dict[str, Any], xs: np.ndarray,
) -> np.ndarray | None:
    """Return the prior density for one parameter, or None if not known here."""
    if name == "is_slope":
        mu = float(constants.get("is_slope_mu", -0.30))
        sd = float(constants.get("is_slope_sigma", 0.10))
        # Truncated above at zero and renormalised by the mass below it.
        mass = 0.5 * (1.0 + math.erf((0.0 - mu) / (sd * math.sqrt(2.0))))
        return np.where(xs <= 0.0, _normal_pdf(xs, mu, sd) / mass, 0.0)
    if name == "rstar_0":
        return _normal_pdf(
            xs, float(constants.get("rstar_0_mu", 1.5)),
            float(constants.get("rstar_0_sigma", 2.0)),
        )
    if name == "rstar_trend":
        return _normal_pdf(xs, 0.0, float(constants.get("rstar_trend_sigma", 0.03)))
    if name in {"sigma_e", "sigma_rstar"}:
        key = "sigma_e_prior" if name == "sigma_e" else "sigma_rstar_prior"
        sd = float(constants.get(key, 1.0))
        # HalfNormal: twice the normal density on the positive half line.
        return np.where(xs >= 0.0, 2.0 * _normal_pdf(xs, 0.0, sd), 0.0)
    if name == "lag_weight":
        a = float(constants.get("lag_weight_a", 2.0))
        b = float(constants.get("lag_weight_b", 2.0))
        log_norm = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
        inside = (xs > 0.0) & (xs < 1.0)
        density = np.zeros_like(xs)
        with np.errstate(divide="ignore", invalid="ignore"):
            density[inside] = np.exp(
                log_norm + (a - 1) * np.log(xs[inside]) + (b - 1) * np.log(1.0 - xs[inside]),
            )
        return density
    return None


def _grid_for(name: str, draws: np.ndarray, constants: dict[str, Any]) -> np.ndarray:
    """Return an x grid wide enough to show prior and posterior together."""
    if name == "lag_weight":
        return np.linspace(0.0, 1.0, 400)
    lo, hi = float(draws.min()), float(draws.max())
    # Reach out to the prior as well, so a posterior that has moved a long way
    # is visibly a long way rather than filling the frame on its own.
    if name == "is_slope":
        mu = float(constants.get("is_slope_mu", -0.30))
        sd = float(constants.get("is_slope_sigma", 0.10))
        lo, hi = min(lo, mu - 3 * sd), min(0.0, max(hi, 0.0))
    elif name == "rstar_0":
        mu = float(constants.get("rstar_0_mu", 1.5))
        sd = float(constants.get("rstar_0_sigma", 2.0))
        lo, hi = min(lo, mu - 3 * sd), max(hi, mu + 3 * sd)
    pad = 0.1 * (hi - lo) if hi > lo else 1.0
    lo = max(0.0, lo - pad) if name.startswith("sigma") else lo - pad
    return np.linspace(lo, hi + pad, 400)


def plot_prior_posterior(trace: az.InferenceData, constants: dict[str, Any]) -> int:
    """One chart per estimated parameter: posterior against its own prior.

    The point is how much of each answer is data. A posterior sitting on its
    prior means the data said nothing, and this model's notes turn on that.

    Drawing is `common.prior_posterior`, shared with every Bayesian model
    here. What stays is this model's own priors and labels, and the reference
    line at `is_curve`'s slope, since whether `is_slope` sits on its bound is
    the finding.
    """
    drawn = prior_posterior.plot_all(
        getattr(trace, "posterior", trace),
        lambda name: (
            (lambda xs: _prior_curve(name, constants, xs))
            if name in _PARAM_LABEL and _prior_curve(name, constants, np.zeros(1)) is not None
            else None
        ),
        footers={
            "lheader": "Posterior on top of the prior means the data said nothing",
            "rfooter": footer_from_constants(constants) or "Built using: RBA F1; ABS 5206.0",
            "lfooter": "Australia. r* by conditional inversion. ",
        },
        labels=_PARAM_LABEL,
        references={"is_slope": -0.108},
    )
    _plot_dirichlet_weights(trace, constants)
    return drawn


def _plot_dirichlet_weights(trace: az.InferenceData, constants: dict[str, Any]) -> None:
    """One prior-posterior chart per Dirichlet stick, when a run has three lags.

    THE MARGINAL PRIOR of one component of a Dirichlet with k sticks and a
    common concentration c is Beta(c, c(k-1)), so each stick is drawn against
    that rather than against the joint density, which cannot be plotted on one
    axis. It is the right comparison anyway: the question per chart is whether
    the data moved THAT lag's share off its prior.
    """
    posterior = getattr(trace, "posterior", {})
    if "lag_weights" not in posterior:
        return
    draws = np.asarray(posterior["lag_weights"])
    draws = draws.reshape(-1, draws.shape[-1])
    count = draws.shape[1]
    lags = _recorded_lags(constants)
    conc = float(constants.get("lag_weight_conc", 2.0))
    # Beta(c, c(k-1)) as the marginal, expressed in the keys `_prior_curve` reads.
    marginal = {"lag_weight_a": conc, "lag_weight_b": conc * (count - 1)}
    grid = np.linspace(0.0, 1.0, 400)
    # Same marginal on every stick, so it is built once. `_prior_curve` can
    # return None for a parameter it does not know, which this one is not, but
    # the check is cheap and keeps the chart honest if that ever changes.
    prior = _prior_curve("lag_weight", marginal, grid)

    for position in range(count):
        column = draws[:, position]
        lag = lags[position] if position < len(lags) else position + 1
        fig, ax = plt.subplots()
        ax.hist(column, bins=60, density=True, color="teal", alpha=0.55, label="Posterior")
        if isinstance(prior, np.ndarray):
            ax.plot(grid, prior, color="darkorange", lw=2, ls="--",
                    label="Prior (Beta marginal)")
        ax.axvline(float(np.median(column)), color="teal", ls=":", lw=1.5,
                   label=f"Posterior median {np.median(column):.3f}")
        ax.set_xlabel(f"Share of the stance carried by the rate at t-{lag}")
        ax.set_ylabel("Density")
        mg.finalise_plot(
            ax,
            title=f"Prior and posterior: lag weight t-{lag}",
            legend={"loc": "best", "fontsize": "small"},
            lheader="Posterior on top of the prior means the data said nothing",
            rfooter=footer_from_constants(constants) or "Built using: RBA F1; ABS 5206.0",
            lfooter="Australia. Weights sum to one across the lags. ",
            show=False,
        )
        plt.close(fig)


def plot_rstar_real_nominal(
    trace: az.InferenceData, frame: pd.DataFrame, constants: dict[str, Any],
) -> None:
    """Real and nominal r*, each with a 90% band.

    NOMINAL IS REAL PLUS THE TARGET, because a neutral rate is defined at
    target inflation rather than at whatever inflation happened to be. Both
    scales are drawn so nothing downstream has to guess the deflator, which is
    the convention `rstar_rba` settled on after getting it wrong.
    """
    index = frame.index
    if not isinstance(index, pd.PeriodIndex):
        index = pd.PeriodIndex(index, freq="Q")

    real = posterior_median(trace, "rstar", index)
    real_band = posterior_band(trace, "rstar", index)
    nominal = real + TARGET
    nominal_band = real_band + TARGET

    ax = mg.fill_between_plot(nominal_band, color="darkblue", alpha=0.15,
                              label="Nominal 90% band")
    mg.fill_between_plot(real_band, ax=ax, color="darkorange", alpha=0.20,
                         label="Real 90% band")
    mg.line_plot(
        pd.DataFrame({
            f"Nominal r* (real + {TARGET:g}% target)": nominal,
            "Real r*": real,
        }),
        ax=ax,
        color=["darkblue", "darkorange"],
        width=[2.5, 2.5],
        annotate=True,
        rounding=2,
    )
    mg.finalise_plot(
        ax,
        title="r* on both scales",
        ylabel="Per cent",
        y0=True,
        legend={"loc": "best", "fontsize": "small"},
        lheader=_asserted(constants),
        rheader=f"{index[-1]}: real {real.iloc[-1]:.2f}, nominal {nominal.iloc[-1]:.2f}",
        rfooter=footer_from_constants(constants) or "Built using: RBA F1; ABS 5206.0",
        lfooter="Australia. Bands conditional on what was asserted. ",
        show=False,
    )


def _recorded_lags(constants: dict[str, Any]) -> list[int]:
    """Return the rate lags a run used, in the order the model applied them.

    The unused slots are recorded as NaN (`rate_lag_2` on a single-lag run,
    `rate_lag_3` on a pair), and NaN fails every comparison, so they are
    filtered rather than compared.
    """
    recorded = [constants.get(key) for key in ("rate_lag", "rate_lag_2", "rate_lag_3")]
    numeric = [float(value) for value in recorded if isinstance(value, (int, float))]
    return [int(lag) for lag in numeric if math.isfinite(lag)]


def _longest_lag(constants: dict[str, Any]) -> int:
    """Return the longest rate lag a run used, so the empty leading rows drop."""
    lags = _recorded_lags(constants)
    return max(lags) if lags else 0


def _lag_weights(trace: az.InferenceData, count: int) -> list[float]:
    """Return the posterior median weight on each lag, summing to one.

    Two lags carry the scalar `lag_weight` (the second takes 1 - w); three
    carry the Dirichlet vector `lag_weights`. A fixed-weight run has neither,
    and the lags share equally.
    """
    posterior = getattr(trace, "posterior", {})
    if count < PAIR_LAGS:
        return [1.0]
    if count == PAIR_LAGS and "lag_weight" in posterior:
        weight = float(np.median(scalar_draws(trace, "lag_weight")))
        return [weight, 1.0 - weight]
    if "lag_weights" in posterior:
        draws = np.asarray(posterior["lag_weights"])
        medians = np.median(draws.reshape(-1, draws.shape[-1]), axis=0)
        return [float(value) for value in medians]
    return [1.0 / count] * count


def plot_raw_scatter(
    trace: az.InferenceData, frame: pd.DataFrame, constants: dict[str, Any],
) -> None:
    """Plot the gap against the RAW lagged real rate, with nothing fitted on either axis.

    THE UNFITTED PICTURE. The companion scatter puts the model's own stance on
    the x-axis, and r* was chosen to make that line fit, so it cannot be
    evidence for the slope. Here the horizontal coordinate is observed data
    alone. The fitted line is a plain OLS with a FREE intercept, and where it
    crosses zero gap is the r* the raw data implies, read off rather than
    imposed.

    Lockdown quarters are dropped, matching the likelihood, and the leading
    quarters with no lag available go too.
    """
    index = frame.index
    if not isinstance(index, pd.PeriodIndex):
        index = pd.PeriodIndex(index, freq="Q")

    # The same regressor the model uses, but built from the rate alone: the
    # weighted average of the lagged REAL CASH RATE, with no r* subtracted.
    lags = _recorded_lags(constants)
    weights = _lag_weights(trace, len(lags))

    rate = frame["real_cash"]
    # Accumulated explicitly rather than with sum(), whose zero start makes the
    # result Series-or-int and loses the Series methods below.
    x = rate.shift(lags[0]) * weights[0]
    for extra_weight, extra_lag in zip(weights[1:], lags[1:], strict=True):
        x = x + rate.shift(extra_lag) * extra_weight
    keep = x.notna() & frame["gap"].notna()
    for first, last in DEFAULT_WINDOWS:
        keep &= ~((index >= pd.Period(first, freq="Q")) & (index <= pd.Period(last, freq="Q")))
    x, y = x[keep], frame["gap"][keep]

    # Plain OLS with a free intercept. Its zero-gap crossing is r* as the
    # unfitted data sees it.
    slope, intercept = np.polyfit(x.to_numpy(dtype=float), y.to_numpy(dtype=float), 1)
    implied_rstar = -intercept / slope if slope else float("nan")
    asserted = float(constants.get("is_slope_mu", -0.30))

    fig, ax = plt.subplots()
    years = np.asarray([period.year for period in x.index])
    points = ax.scatter(x, y, c=years, cmap="viridis", s=28, alpha=0.85, zorder=3)
    fig.colorbar(points, ax=ax, label="Year")

    grid = np.linspace(float(x.min()), float(x.max()), 50)
    ax.plot(grid, slope * grid + intercept, color="crimson", linewidth=2.5, zorder=4,
            label=f"OLS, free intercept: slope {slope:+.3f}")
    # The asserted slope anchored at the OLS crossing, so only the steepness
    # differs and the comparison is about the slope rather than the level.
    ax.plot(grid, asserted * (grid - implied_rstar), color="darkorange",
            linestyle="--", linewidth=2, zorder=4,
            label=f"Asserted slope {asserted:+.3f}, same crossing")
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.axvline(implied_rstar, color="grey", linestyle=":", linewidth=1.5,
               label=f"Zero-gap crossing: r* = {implied_rstar:+.2f}%")
    ax.set_xlabel("Weighted LAGGED REAL CASH RATE, per cent (no r* subtracted)")
    ax.set_ylabel("Output gap, per cent of potential")
    mg.finalise_plot(
        ax,
        title="The IS curve before the model touches it",
        legend={"loc": "best", "fontsize": "small"},
        lheader="Nothing on either axis is fitted. The intercept is free",
        rfooter=footer_from_constants(constants) or "Built using: RBA F1; ABS 5206.0",
        lfooter="Australia. Lockdown quarters dropped, matching the likelihood. ",
        show=False,
    )
    plt.close(fig)


def plot_is_scatter(
    trace: az.InferenceData, frame: pd.DataFrame, constants: dict[str, Any],
) -> None:
    """Plot the IS curve itself: the output gap against the stance, with the line.

    THE CHART THE PACKAGE WAS MISSING. Everything else here is a time series or
    a density, none of which shows the relationship the model is about. This
    draws it: one point per quarter, the fitted slope through the origin, and
    the asserted prior slope beside it for scale.

    The line goes through the origin by construction. There is no free
    intercept: `x = is_slope . (r - rstar)` means zero stance implies zero gap,
    which is what defines rstar as neutral.
    """
    index = frame.index
    if not isinstance(index, pd.PeriodIndex):
        index = pd.PeriodIndex(index, freq="Q")

    stance = posterior_median(trace, "stance", index)
    gap = frame["gap"]
    slope = float(np.median(scalar_draws(trace, "is_slope")))
    mu = float(constants.get("is_slope_mu", -0.30))

    # The first quarters have no stance (the lags are not yet available) and
    # were never in the likelihood, so they are not in the picture either.
    x, y = stance.iloc[_longest_lag(constants):], gap.iloc[_longest_lag(constants):]

    fig, ax = plt.subplots()
    # Colour by period so era clusters are visible: the is_curve bench found a
    # convincing line can be manufactured from two clusters that individually
    # disagree, and a single-colour scatter would hide that here too.
    years = np.asarray([period.year for period in x.index])
    points = ax.scatter(x, y, c=years, cmap="viridis", s=28, alpha=0.85, zorder=3)
    fig.colorbar(points, ax=ax, label="Year")

    grid = np.linspace(float(x.min()), float(x.max()), 50)
    ax.plot(grid, slope * grid, color="crimson", linewidth=2.5, zorder=4,
            label=f"Fitted: slope {slope:+.3f}")
    ax.plot(grid, mu * grid, color="darkorange", linestyle="--", linewidth=2, zorder=4,
            label=f"Asserted prior: slope {mu:+.3f}")
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.axvline(0.0, color="black", linewidth=0.8)
    ax.set_xlabel("Stance: weighted (r - r*) at the model's lags, pp")
    ax.set_ylabel("Output gap, per cent of potential")
    mg.finalise_plot(
        ax,
        title="The IS curve, drawn",
        legend={"loc": "best", "fontsize": "small"},
        lheader="No free intercept: the line goes through the origin by construction",
        rfooter=footer_from_constants(constants) or "Built using: RBA F1; ABS 5206.0",
        lfooter="Australia. The stance uses the model's own r*, so it is not data. ",
        show=False,
    )
    plt.close(fig)


def plot_stance(trace: az.InferenceData, frame: pd.DataFrame, constants: dict[str, Any]) -> None:
    """Plot the policy stance, the one the equation actually uses."""
    index = frame.index
    if not isinstance(index, pd.PeriodIndex):
        index = pd.PeriodIndex(index, freq="Q")

    lag = int(float(constants.get("rate_lag", 2)))
    stance = posterior_median(trace, "stance", index)
    mg.line_plot_finalise(
        stance,
        title="Policy stance against the inverted r*",
        ylabel="Percentage points",
        color=["darkred"],
        width=2.0,
        annotate=True,
        rounding=2,
        y0=True,
        legend=False,
        lheader=_asserted(constants),
        rheader=f"measured at t-{lag}, as the IS curve uses it",
        rfooter=footer_from_constants(constants) or "Built using: RBA F1; ABS 5206.0",
        lfooter=_LFOOTER + "Positive is restrictive. ",
        show=False,
    )


def plot_gap_fit(trace: az.InferenceData, frame: pd.DataFrame, constants: dict[str, Any]) -> None:
    """Plot the gap the IS curve reproduces, against the gap it was given."""
    index = frame.index
    if not isinstance(index, pd.PeriodIndex):
        index = pd.PeriodIndex(index, freq="Q")

    fitted = posterior_median(trace, "gap_fitted", index)
    mg.line_plot_finalise(
        pd.DataFrame({
            "Output gap (given, from ystar_ustar)": frame["gap"],
            "Fitted by the asserted IS curve": fitted,
        }),
        title="What the asserted IS curve reproduces",
        ylabel="Per cent of potential",
        color=["black", "teal"],
        width=[2.0, 1.6],
        style=["-", "--"],
        annotate=True,
        rounding=2,
        y0=True,
        legend={"loc": "best", "fontsize": "small"},
        lheader=_asserted(constants),
        rfooter=footer_from_constants(constants) or "Built using: RBA F1; ABS 5206.0",
        lfooter=_LFOOTER + "The gap is a model output taken as data. ",
        show=False,
    )


def plot_ensemble_paths(paths: pd.DataFrame, constants: dict[str, Any]) -> None:
    """Plot every r* path across the sigma_rstar sweep."""
    mg.line_plot_finalise(
        paths,
        title="How slow is r*: one path per assumption",
        ylabel="Per cent",
        width=1.5,
        annotate=False,
        y0=True,
        legend={"loc": "best", "fontsize": "small"},
        lheader="Each line asserts a different speed for r*. None is an estimate",
        rfooter=footer_from_constants(constants) or "Built using: RBA F1; ABS 5206.0",
        lfooter="Australia. Spread is the assumption moving, not sampling error. ",
        show=False,
    )


def plot_ensemble_with_band(
    trace: az.InferenceData,
    frame: pd.DataFrame,
    constants: dict[str, Any],
    paths: pd.DataFrame,
) -> None:
    """Draw the sweep over the headline run's band, so the two uncertainties show.

    THEY ARE DIFFERENT IN KIND AND MUST NOT BE ADDED. The shaded band is
    sampling uncertainty GIVEN one assumed speed for r*. The spread between
    the lines is the assumption itself moving. Reported together they say:
    this is what the data can pin down, and this is what the modeller chose.
    `rstar_rba` draws the same comparison for the same reason.
    """
    index = frame.index
    if not isinstance(index, pd.PeriodIndex):
        index = pd.PeriodIndex(index, freq="Q")

    band = posterior_band(trace, "rstar", index)
    ax = mg.fill_between_plot(
        band, color="teal", alpha=0.20,
        label="90% band at the headline assumption",
    )
    mg.line_plot(paths.reindex(index), ax=ax, width=1.4, alpha=0.9, annotate=False)
    mg.finalise_plot(
        ax,
        title="Two kinds of uncertainty about r*",
        ylabel="Per cent",
        y0=True,
        legend={"loc": "best", "fontsize": "x-small"},
        lheader="Shading: sampling error within one assumption. Lines: the assumption moving",
        rfooter=footer_from_constants(constants) or "Built using: RBA F1; ABS 5206.0",
        lfooter="Australia. The two are different in kind; do not add them. ",
        show=False,
    )


def _print_lag_weight(trace: az.InferenceData, constants: dict[str, Any]) -> None:
    """Print the lag weights against their prior, when there are any."""
    posterior = getattr(trace, "posterior", {})
    lags = _recorded_lags(constants)

    if "lag_weights" in posterior:
        draws = np.asarray(posterior["lag_weights"])
        draws = draws.reshape(-1, draws.shape[-1])
        # Every stick shares one concentration, so the Dirichlet's marginal
        # mean is 1/k. The per-stick charts draw the Beta marginal itself.
        count = draws.shape[1]
        print(f"\n  THE LAG WEIGHTS (Dirichlet across lags {lags}, summing to one)")
        print(f"    prior mean:      {1.0 / count:.3f} on each")
        for position in range(count):
            column = draws[:, position]
            label = f"lag {lags[position]}" if position < len(lags) else f"stick {position + 1}"
            print(f"    {label:>10}:      {np.median(column):.3f}  "
                  f"[{np.percentile(column, 5):.3f}, {np.percentile(column, 95):.3f}]")
        print("    an interval spanning the prior means the data cannot tell the lags apart")
        return

    if "lag_weight" not in posterior:
        return
    w = scalar_draws(trace, "lag_weight")
    a = float(constants.get("lag_weight_a", 2.0))
    b = float(constants.get("lag_weight_b", 2.0))
    first = int(float(constants.get("rate_lag", 4)))
    second = int(float(constants.get("rate_lag_2", 8)))
    print(f"\n  THE LAG WEIGHT (on lag {first}; lag {second} takes 1 - w)")
    print(f"    prior mean:      {a / (a + b):.3f}")
    print(f"    posterior:       {np.median(w):.3f}  "
          f"[{np.percentile(w, 5):.3f}, {np.percentile(w, 95):.3f}]")
    print("    on the prior means the data cannot tell the two lags apart")


def run_analyse(
    prefix: str = "rstar_invert",
    chart_dir: Path | str | None = None,
    *,
    verbose: bool = False,
) -> None:
    """Load a completed run and produce every chart."""
    directory = Path(chart_dir) if chart_dir else _chart_dir(prefix)
    mg.set_chart_dir(str(directory))
    mg.clear_chart_dir()

    trace, frame, constants = load_results(prefix=prefix)
    save_diagnostics(trace, directory, prefix, model="rstar_invert")
    index = frame.index
    if not isinstance(index, pd.PeriodIndex):
        index = pd.PeriodIndex(index, freq="Q")

    rstar = posterior_median(trace, "rstar", index)
    slope = scalar_draws(trace, "is_slope")
    sigma_e = scalar_draws(trace, "sigma_e")
    correlation = rstar.corr(frame["real_cash"])
    mu = float(constants.get("is_slope_mu", float("nan")))

    print(f"\nLoaded: {prefix}")
    print(f"  {_asserted(constants)}")
    print(f"  Sample:            {index[0]} to {index[-1]}")
    print("\n  THE SLOPE")
    print(f"    prior mu:        {mu:+.3f}")
    print(f"    posterior:       {np.median(slope):+.3f}  "
          f"[{np.percentile(slope, 5):+.3f}, {np.percentile(slope, 95):+.3f}]")
    print(f"    travelled:       {np.median(slope) - mu:+.3f} from the prior")
    _print_lag_weight(trace, constants)
    print("\n  THE NEUTRAL RATE")
    print(f"    r* latest:       {rstar.iloc[-1]:+.2f}%")
    print(f"    r* mean:         {rstar.mean():+.2f}%  "
          f"(real cash mean {frame['real_cash'].mean():+.2f}%)")
    print(f"    r* range:        [{rstar.min():+.2f}, {rstar.max():+.2f}]%, sd {rstar.std():.2f}")
    print(f"    corr(r*, r):     {correlation:+.3f}   <- near 1 means a smoothed cash rate")
    print(f"\n  sigma_e:           {np.median(sigma_e):.3f}  "
          f"(gap sd {frame['gap'].std():.3f}; rstar_hlw's equivalent is 0.70)")
    if verbose:
        print(az.summary(trace, var_names=["is_slope", "sigma_e", "rstar_0"]))

    plot_raw_scatter(trace, frame, constants)
    plot_is_scatter(trace, frame, constants)
    plot_prior_posterior(trace, constants)
    plot_rstar(trace, frame, constants)
    plot_rstar_real_nominal(trace, frame, constants)
    plot_stance(trace, frame, constants)
    plot_gap_fit(trace, frame, constants)

    try:
        paths, table = load_ensemble(prefix=prefix)
    except FileNotFoundError:
        print("\n  note: no saved ensemble; run with --ensemble for the sweep charts")
        print(f"\nCharts saved to: {directory}")
        return

    plot_ensemble_paths(paths, constants)
    plot_ensemble_with_band(trace, frame, constants, paths)
    print(f"\n  ensemble: {len(table)} members on file")
    print(f"\nCharts saved to: {directory}")
