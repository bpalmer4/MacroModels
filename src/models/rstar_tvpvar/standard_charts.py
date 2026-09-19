"""The charts every r* package in this repo produces, for the TVP-VAR.

`rstar_bonds`, `rstar_rba` and `rstar_invert` all draw the same four things, and
a model that does not is hard to put beside them:

1. PRIOR AGAINST POSTERIOR, one chart per estimated scalar. The point is how
   much of each answer is data. `rstar_hlw` was excluded from `rstar_summary`
   on exactly this test, so every model here has to face it.
2. THE POLICY STANCE, the cash rate against nominal r*. What a neutral rate is
   for.
3. THE TAYLOR RULE, what a standard reaction function would prescribe on this
   model's r*. Needs a completed `ystar_ustar` run for the output gap; a
   missing one costs this chart and nothing else.
4. TWO KINDS OF UNCERTAINTY: the credible band within one drift assumption,
   against the envelope across assumptions. They are different in kind and the
   chart exists so nobody adds them.
"""

import mgplot as mg
import numpy as np
import pandas as pd
from scipy import stats

from src.models.common import prior_posterior
from src.models.common.inflation_scale import to_nominal
from src.models.rstar_tvpvar.ensemble import load_ensemble
from src.models.rstar_tvpvar.results import TvpVarResults

# A scalar parameter's posterior array is (chain, draw); a vector's carries
# a third axis, which is pooled across before charting.
_SCALAR_NDIM = 2

_PARAM_LABEL = {
    "sigma_q": "sigma_q: per-quarter sd of the drift in every VAR coefficient",
    "sigma_h": "sigma_h: per-quarter sd of the drift in each log variance",
    "a_free": "a_free: the contemporaneous structure, lower triangle of A",
    "h_0": "h_0: initial log variance, per variable",
}

# The Taylor rule this package reports, matching `rstar_bonds` so the two
# prescriptions are comparable. Taylor's originals, unswept.
_RULE_PI = 0.5
_RULE_GAP = 0.5
_TARGET = 2.5


def _scalar_draws(results: TvpVarResults, name: str) -> np.ndarray:
    """Return flattened posterior draws for a scalar or small-vector parameter."""
    return np.asarray(results.posterior[name].values).reshape(-1)


def _halfnormal_curve(sigma: float, grid: np.ndarray) -> np.ndarray:
    """Return the HalfNormal(sigma) density on `grid`."""
    return stats.halfnorm.pdf(grid, scale=sigma)


def plot_prior_posterior(results: TvpVarResults, footer: str, lfooter: str) -> int:
    """One chart per estimated scalar: posterior against its own prior.

    A posterior sitting on its prior means the data said nothing about that
    parameter. For `sigma_q` that is the whole ballgame, since it governs how
    much the coefficients move and therefore the entire r* path.

    Drawing is `common.prior_posterior`, shared with every Bayesian model
    here. The named list stays because `sigma_h` and `a_free` are vectors of
    three, pooled into one distribution each rather than charted element by
    element, which the shared scalar sweep would skip.
    """
    constants = results.constants
    priors: dict[str, tuple[str, float, float]] = {
        "sigma_q": ("half", 0.0, float(constants.get("sigma_q_prior", 0.02))),
        "sigma_h": ("half", 0.0, float(constants.get("sigma_h_prior", 0.2))),
        "a_free": ("normal", 0.0, 1.0),
    }

    drawn = 0
    for name, prior in priors.items():
        if name not in results.posterior:
            continue
        values = np.asarray(results.posterior[name])
        draws = values if values.ndim == _SCALAR_NDIM else values.reshape(values.shape[0], -1)
        prior_posterior.plot_parameter(
            name, draws, prior,
            footers={
                "lheader": "Posterior on top of the prior means the data said nothing",
                "rfooter": footer,
                "lfooter": lfooter,
            },
            label=_PARAM_LABEL.get(name),
        )
        drawn += 1
    return drawn


def plot_policy_stance(results: TvpVarResults, footer: str, lfooter: str) -> None:
    """Plot the cash rate against nominal r*: what a neutral rate is for."""
    # The NOMINAL cash rate, since the line it is compared against is nominal.
    from src.data.cash_rate import get_cash_rate_qrtly  # noqa: PLC0415 — local to keep imports light

    nominal = to_nominal(results.rstar_median())
    nominal_cash = get_cash_rate_qrtly().data.astype(float)
    nominal_cash.index = pd.PeriodIndex(nominal_cash.index, freq="Q")

    stance = (nominal_cash - nominal).dropna()
    last = stance.index[-1]
    direction = "restrictive" if stance.loc[last] > 0 else "expansionary"
    mg.line_plot_finalise(
        pd.DataFrame({
            "Nominal r* (r* + long-run expectations)": nominal,
            "Cash rate": nominal_cash,
        }).dropna(how="all"),
        color=["darkblue", "black"],
        width=[2.5, 1.5],
        annotate=True,
        rounding=2,
        title="The policy stance: the cash rate against nominal r-star",
        ylabel="Per cent",
        y0=True,
        legend={"loc": "best", "fontsize": "small"},
        lheader="Above the blue line policy is restrictive, below it expansionary",
        rheader=f"{last}: {abs(stance.loc[last]):.2f}pp {direction}",
        rfooter=footer,
        lfooter=lfooter,
        show=False,
    )


def _output_gap() -> pd.Series:
    """Return the joint model's output gap, for the Taylor rule."""
    from src.models.ystar_ustar.results import load_results  # noqa: PLC0415 — optional dependency

    return load_results(prefix="ystar_ustar").output_gap_median()


def plot_taylor_rule(results: TvpVarResults, footer: str, lfooter: str) -> None:
    """Plot what a standard Taylor rule would prescribe on this model's r*.

    Skipped rather than fatal when the joint run is missing: r* itself does not
    depend on it. Read the prescription as what a conventional reaction function
    WOULD say on this r*, not as a forecast: this repo's standing finding is
    that the rate does not visibly move the gap, which is why there is no IS
    curve in this model either.
    """
    try:
        gap = _output_gap()
    except (FileNotFoundError, KeyError, ValueError) as exc:
        print(f"  note: output gap unavailable ({type(exc).__name__}); Taylor chart skipped")
        return

    from src.data.cash_rate import get_cash_rate_qrtly  # noqa: PLC0415
    from src.data.inflation import get_trimmed_mean_annual  # noqa: PLC0415

    cash = get_cash_rate_qrtly().data.astype(float)
    cash.index = pd.PeriodIndex(cash.index, freq="Q")
    inflation = get_trimmed_mean_annual().data.astype(float)
    inflation.index = pd.PeriodIndex(inflation.index, freq="Q")

    real = results.rstar_median()
    prescribed = (real + inflation + _RULE_PI * (inflation - _TARGET) + _RULE_GAP * gap).dropna()
    frame = pd.DataFrame({"Taylor prescription": prescribed, "Cash rate": cash}).dropna()
    if frame.empty:
        print("  note: no overlap between the rule's inputs and r*; Taylor chart skipped")
        return

    mg.line_plot_finalise(
        frame,
        color=["firebrick", "black"],
        style=["--", "-"],
        width=[2.0, 1.8],
        annotate=True,
        rounding=2,
        title="The cash rate and the Taylor rule",
        ylabel="Per cent",
        y0=True,
        legend={"loc": "best", "fontsize": "small"},
        lheader=f"i* = r* + pi + {_RULE_PI:g}(pi - {_TARGET:g}) + {_RULE_GAP:g} x output gap",
        rfooter=footer,
        lfooter=lfooter,
        show=False,
    )


def plot_two_uncertainties(results: TvpVarResults, footer: str, lfooter: str) -> None:
    """Plot the credible band within one assumption against the envelope across them.

    The two are different in kind and must not be added. The shading is sampling
    uncertainty given a drift assumption; the lines are that assumption moving.
    For this model the second is much the smaller, which is the surprise: the
    LEVEL is robust to `sigma_q` and it is the stance that is not.
    """
    ensemble = load_ensemble()
    if ensemble is None or ensemble.get("paths") is None:
        print("  note: no sigma_q sweep on file; two-uncertainties chart skipped")
        return

    band = results.rstar_hdi(0.90)
    ax = mg.fill_between_plot(
        band, color="cornflowerblue", alpha=0.25, label="90% band, given sigma_q",
    )
    paths = ensemble["paths"]
    mg.line_plot(
        paths.rename(columns=lambda c: f"sigma_q = {c}"),
        ax=ax,
        width=[1.5] * paths.shape[1],
        annotate=False,
    )
    mg.finalise_plot(
        ax,
        title="Two kinds of uncertainty about r-star",
        ylabel="Per cent, real",
        y0=True,
        legend={"loc": "best", "fontsize": "x-small", "ncol": 2},
        lheader="Shading: sampling error within one assumption. Lines: the assumption moving",
        rfooter=footer,
        lfooter=lfooter,
        show=False,
    )


def plot_standard_suite(results: TvpVarResults, footer: str, lfooter: str) -> None:
    """Draw every chart the other r* packages draw."""
    plot_prior_posterior(results, footer, lfooter)
    plot_policy_stance(results, footer, lfooter)
    plot_taylor_rule(results, footer, lfooter)
    plot_two_uncertainties(results, footer, lfooter)
