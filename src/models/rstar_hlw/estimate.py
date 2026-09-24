"""Build, sample, and persist the HLW Bayesian r-star model."""

import pickle
from pathlib import Path
from typing import Any

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt

from src.models.common.model_constants import attach, get_dictionary, record_constant
from src.models.nairu.base import (
    SamplerConfig,
    add_scalar_priors,
    sample_model,
    set_model_coefficients,
)
from src.models.rstar_hlw.equations.indexed_bond import indexed_bond_equation
from src.models.rstar_hlw.equations.is_curve import is_curve_equation
from src.models.rstar_hlw.equations.phillips import phillips_curve_equation
from src.models.rstar_hlw.equations.potential import potential_output_equation
from src.models.rstar_hlw.equations.r_star import r_star_equation
from src.models.rstar_hlw.equations.r_star_blended_z import r_star_blended_z_equation
from src.models.rstar_hlw.equations.r_star_tv_alpha import r_star_tv_alpha_equation
from src.models.rstar_hlw.equations.trend_growth import trend_growth_equation
from src.models.rstar_hlw.equations.z_star import z_star_equation
from src.models.rstar_hlw.observations import (
    DEFAULT_G_ANCHOR,
    DEFAULT_START,
    GAnchor,
    build_observations,
)
from src.paths import MODEL_OUTPUTS

DEFAULT_OUTPUT_DIR = MODEL_OUTPUTS

# The lockdown quarters, dropped from the IS and Phillips likelihoods by
# default. Same window `ystar` uses, and for the same reason: with it in,
# potential takes part of the 2020 collapse as a fall in potential and never
# catches back up, so the recovery reads as output seven per cent above
# capacity through 2022-23. Pass exclude_window=None to keep them.
DEFAULT_EXCLUDE_WINDOW: tuple[str, str] | None = ("2020Q2", "2021Q3")

# The single rate-gap lag on the IS curve. None would be HLW's own averaged
# (t-1, t-2) shape; t-6 matches the lag the `is_curve` bench and
# `rstar_invert` use, so the three stay comparable on timing.
DEFAULT_RATE_LAG: int | None = 6

# Potential output's innovation sd, imposed rather than estimated. See
# `equations/potential.py` for why a prior alone could not hold it.
DEFAULT_SIGMA_YSTAR_FIXED: float | None = 0.078


# lambda_g is HLW's signal-to-noise ratio for trend growth, sigma_g /
# sigma_ystar, which they fix by Stock-Watson median-unbiased estimation rather
# than estimating. It is OFF by default here, and the sweep is why.
#
# `lambda_g_sweep.py` ran Resolution A at 0.053 (HLW's US value), 0.15, 0.34
# and free. On that sweep HLW's own value is rejected on every diagnostic: it
# holds g nearly flat, puts the 2005-19 economy 4.35% above capacity, and
# collapses the Phillips slope to 0.054 against its 0.02 lower bound. At 0.34,
# the ratio Australia's growth slowdown implies, the constraint is not binding
# at all: that run and the free run agree to the third decimal on every
# diagnostic, including r* latest (1.71 vs 1.73).
#
# READ THAT NARROWLY. The sweep leaves `sigma_ystar` at its imposed default, so
# every column constrains BOTH variances, sigma_ystar at 0.078 and sigma_g at
# 4 x lambda_g x 0.078. HLW impose the ratio and estimate the level. The
# rejection is therefore established for a specification HLW do not estimate,
# and the one they do has not been run here.
#
# The ratio with a free level is also the only device that disciplines both
# variances without nailing either, which the default still needs: imposing
# sigma_ystar alone repaired potential and moved the variance into g, where
# sigma_g returns 0.124 against a prior scale of 0.04 and g carries a
# pandemic-shaped trough. Freeing sigma_ystar with no ratio does not sample.
#
# ASSUMPTION, NOT VERIFIED HERE: that 0.053 is HLW (2017)'s US estimate is
# carried from memory, not checked against the paper. It only ever served as
# the sweep's low end, and nothing downstream rests on the exact figure.
DEFAULT_LAMBDA_G: float | None = None

# g is annualised in this implementation (the y* equation uses g/4), while
# HLW's g is the quarterly growth of potential. A ratio quoted on their g
# therefore scales by 4 on ours.
_QUARTERS_PER_YEAR = 4


def _sigma_ystar_scalar(
    model: pm.Model,
    *,
    prior: float,
    fixed: float | None,
) -> float | pt.TensorVariable:
    """Create potential output's innovation sd ahead of the state equations.

    It lives here rather than inside `potential_output_equation` because
    lambda_g ties sigma_g to it and trend growth is built first. Both
    equations then consume this one object, which is what lets the ratio be
    imposed while the level stays free: HLW's own device, and the
    configuration the equation ordering used to make unreachable.

    Returns the imposed float when one is given, otherwise the random
    variable.
    """
    constant = {} if fixed is None else {"sigma_ystar": fixed}
    with model:
        mc = set_model_coefficients(
            model, {"sigma_ystar": {"sigma": prior}}, constant,
        )
    return mc["sigma_ystar"]


def _sigma_g_from_lambda(
    lambda_g: float | None,
    sigma_ystar: float | pt.TensorVariable,
) -> float | pt.TensorVariable | None:
    """Turn the lambda_g ratio into a sigma_g in annualised units.

    Returns None when no ratio is imposed. Otherwise returns a float if
    sigma_ystar was imposed, or a tensor expression if it is being estimated,
    in which case sigma_g is a derived quantity rather than a parameter and
    only their RATIO is pinned. That second case is HLW's device.
    """
    if lambda_g is None:
        return None
    if lambda_g == 0.0:
        # g does not move at all, whatever the level of sigma_ystar. Returned
        # as a float rather than the expression `0 * sigma_ystar`, which is a
        # tensor and would build a degenerate random walk instead of the
        # constant state HLW's first stage calls for.
        return 0.0
    return _QUARTERS_PER_YEAR * lambda_g * sigma_ystar


def _check_given_g(
    given_g: np.ndarray | None,
    lambda_g: float | None,
    n_periods: int,
) -> None:
    """Reject a prescribed g that cannot mean what it says.

    A ratio imposed on g's innovation is meaningless once g is data, and a g
    of the wrong length would broadcast silently against the sample rather
    than fail.
    """
    if given_g is None:
        return
    if lambda_g is not None:
        raise ValueError(
            "lambda_g scales g's innovation, and a prescribed g has none. "
            "Pass one or the other",
        )
    if len(given_g) != n_periods:
        raise ValueError(
            f"given_g covers {len(given_g)} quarters, the sample has {n_periods}",
        )


def _keep_mask(
    exclude_window: tuple[str, str] | None,
    obs_index: pd.PeriodIndex | None,
) -> np.ndarray | None:
    """Resolve an excluded window to a boolean keep-mask over the sample.

    Returns None when nothing is excluded. Raises rather than quietly
    excluding nothing if the window misses the sample: a silent no-op would
    leave the run log claiming an exclusion the model does not have.
    """
    if exclude_window is None:
        return None
    if obs_index is None:
        raise ValueError("exclude_window needs obs_index; pass it to build_model")

    lo, hi = exclude_window
    dropped = (obs_index >= pd.Period(lo, freq="Q")) & (obs_index <= pd.Period(hi, freq="Q"))
    if not dropped.any():
        raise ValueError(
            f"exclude_window {lo} to {hi} covers no quarter of the estimation sample "
            f"{obs_index.min()} to {obs_index.max()}",
        )
    if dropped.all():
        raise ValueError(f"exclude_window {lo} to {hi} would drop the whole sample")

    return ~np.asarray(dropped)


# SOE-block regressors, used by Resolutions D and F only.
_SOE_KEYS = ("tot_change_1", "twi_change_1", "icp_change_1")

# Which r* identity each resolution uses. A, B and D are absent because they
# are canonical HLW (r* = g + z), which needs potential built first.
_RSTAR_EQUATIONS = {
    "C": r_star_equation,
    "G": r_star_equation,
    "H": r_star_tv_alpha_equation,
    "E": r_star_blended_z_equation,
    "F": r_star_blended_z_equation,
}


def _resolution_obs(
    obs: dict[str, np.ndarray],
    resolution: str,
) -> dict[str, np.ndarray]:
    """Drop the observation series a resolution does not use."""
    if resolution in ("A", "B"):
        # Textbook canonical: no fiscal impulse, no soft anchor on g, no SOE.
        return {
            k: v for k, v in obs.items()
            if k not in ("trend_growth_obs", "fiscal_impulse_1", *_SOE_KEYS)
        }
    if resolution in ("C", "E", "G", "H"):
        # Blend (with persistent deviation in E; hierarchical alpha in G;
        # time-varying alpha in H): keeps fiscal + soft anchor on g; drops
        # the SOE block, which only D and F use.
        return {k: v for k, v in obs.items() if k not in _SOE_KEYS}
    # Resolutions D and F keep everything: soft anchor on g, fiscal, full SOE.
    return obs


def _state_equations(
    obs: dict[str, np.ndarray],
    model: pm.Model,
    latents: dict[str, Any],
    *,
    constants: dict[str, Any],
    resolution: str,
    sigma_ystar: float | pt.TensorVariable,
    sigma_g_value: float | pt.TensorVariable | None,
    sigma_z_prior: float,
    given_rstar: np.ndarray | None,
    given_g: np.ndarray | None,
) -> list[str]:
    """Add the state equations, in the order each resolution needs them."""
    if given_g is None:
        descriptions = [
            "Trend growth: " + trend_growth_equation(
                obs, model, latents,
                constant=constants.get("trend_growth"),
                sigma_g_value=sigma_g_value,
            ),
        ]
    else:
        # g supplied from outside as DATA. The whole equation is skipped, not
        # just its state: with g fixed there is nothing for sigma_g to scale
        # and nothing for the soft anchor to pull on, so leaving either in
        # would sample a parameter the likelihood cannot see.
        with model:
            latents["trend_growth"] = pm.Deterministic(
                "trend_growth", pt.as_tensor_variable(np.asarray(given_g, dtype=float)),
            )
        descriptions = ["Trend growth: GIVEN as data, not estimated"]

    def potential() -> str:
        return potential_output_equation(
            obs, model, latents,
            constant=constants.get("potential"),
            sigma_ystar=sigma_ystar,
        )

    if given_rstar is not None:
        # r* supplied from outside as DATA. Not an estimate and not a
        # resolution: a diagnostic that asks what the rest of the model does
        # when the rate gap is allowed to turn positive. See
        # `given_rstar_test.py`. g is still a state, since potential needs it.
        with model:
            latents["r_star"] = pm.Deterministic(
                "r_star", pt.as_tensor_variable(np.asarray(given_rstar, dtype=float)),
            )
        descriptions.append("r-star:       GIVEN as data, not estimated")
        descriptions.append(f"Potential:    {potential()}")
        return descriptions

    r_star_fn = _RSTAR_EQUATIONS.get(resolution)
    if r_star_fn is not None:
        # trend_growth -> r_star (uses g and indexed_10y) -> potential.
        descriptions.append(
            "r-star:       " + r_star_fn(obs, model, latents, constant=constants.get("r_star")),
        )
        descriptions.append(f"Potential:    {potential()}")
    else:
        # Canonical A, B, D: trend_growth -> potential -> z_star, r* = g + z.
        descriptions.append(f"Potential:    {potential()}")
        descriptions.append(
            "z-star:       " + z_star_equation(
                obs, model, latents,
                constant=constants.get("z_star"),
                sigma_z_prior=sigma_z_prior,
            ),
        )

    return descriptions


def build_model(
    obs: dict[str, np.ndarray],
    *,
    constants: dict[str, Any] | None = None,
    resolution: str = "A",
    rate_lag: int | None = DEFAULT_RATE_LAG,
    sigma_ystar_prior: float = 0.12,
    sigma_ystar_fixed: float | None = DEFAULT_SIGMA_YSTAR_FIXED,
    lambda_g: float | None = DEFAULT_LAMBDA_G,
    sigma_z_prior: float = 0.10,
    given_rstar: np.ndarray | None = None,
    given_g: np.ndarray | None = None,
    exclude_window: tuple[str, str] | None = DEFAULT_EXCLUDE_WINDOW,
    sign_prior_only: bool = False,
    obs_index: pd.PeriodIndex | None = None,
) -> pm.Model:
    """Build the HLW r-star PyMC model.

    Equation order matters for NUTS efficiency: state equations first, then
    observation equations.

    Resolution A: r* = g + z (canonical HLW with AR(1) reparameterised z).
    Resolution B: A + indexed-bond observation equation (term-structure pin).
    Resolution C: r* = α·g + (1-α)·(indexed_10y - k) + ε (the blend).
    Resolution D: A's r* identity + an SOE-block IS curve (fiscal + ToT +
                  TWI change + ICP growth) + the soft linear-trend anchor on g.
                  Tests whether enough external regressors firm a_r so that
                  canonical HLW's z latent identifies.
    Resolution E: r* = α·g + (1-α)·(indexed_10y - k) + z, with z AR(1) and
                  σ_z fixed. Generalises C (z=0) and A (α=1) by keeping
                  C's blend as the structural anchor and adding an
                  IS-curve-identifiable persistent deviation z. Tests
                  whether the IS curve has anything to say above and beyond
                  the blend that C suppresses with its small i.i.d. ε_t.
    Resolution F: E's r* identity + the SOE-block IS curve from D. Tests
                  whether giving the IS curve more external regressors lets
                  it identify z (which it could not in E with the
                  fiscal-only IS curve).
    Resolution G: same as C but with hierarchical Beta(a, b) on alpha
                  (a, b ~ Uniform(0.25, 2)) — let the data pick the prior
                  shape itself.
    Resolution H: same r* identity as C but with time-varying alpha_t —
                  logit-scale RW on alpha, sigma_a fixed. Tests whether the
                  data wants alpha to drift over time (e.g. toward the bond
                  anchor in recent years, consistent with Bullock's
                  "shifts in r*" framing).

    Two settings repair the trend/cycle decomposition and are on by default;
    both were off for the eight resolutions written up in MODEL_NOTES.md, so
    results from this build are not directly comparable with those.

    `sigma_ystar_fixed` imposes potential's innovation sd rather than
    estimating it, which is what stops potential absorbing the cycle.
    `exclude_window` drops the lockdown quarters from the IS and Phillips
    likelihoods, which is what stops the 2020 collapse being read as a fall in
    potential. `lambda_g` imposes trend growth's innovation sd as a ratio to
    potential's, which is HLW's device, and it now works whether
    `sigma_ystar_fixed` is set or not: with it set both variances are pinned,
    with it None only the ratio is, which is the form the papers use. It is
    OFF by default, but read the note above `DEFAULT_LAMBDA_G` before treating
    that as settled. See `equations/potential.py`, `equations/exclusion.py`
    and `lambda_g_sweep.py`.
    """
    if constants is None:
        constants = {}
    if resolution not in ("A", "B", "C", "D", "E", "F", "G", "H"):
        raise ValueError(
            f"resolution must be 'A', 'B', 'C', 'D', 'E', 'F', 'G' or 'H', "
            f"got {resolution!r}",
        )
    _check_given_g(given_g, lambda_g, len(obs["log_gdp"]))

    obs = _resolution_obs(obs, resolution)

    # Resolution G: inject the hierarchical-Beta flag into the r_star constants
    # so r_star_equation samples a, b ~ HalfNormal(1) and alpha ~ Beta(a, b).
    if resolution == "G":
        constants = dict(constants)  # shallow copy so we don't mutate caller's dict
        r_star_const = dict(constants.get("r_star", {}))
        r_star_const.setdefault("alpha_hierarchical", True)
        constants["r_star"] = r_star_const

    keep = _keep_mask(exclude_window, obs_index)

    model = pm.Model()
    latents: dict[str, Any] = {}
    descriptions: list[str] = []

    # Potential's innovation sd is built before any state equation, because
    # lambda_g ties sigma_g to it and trend growth comes first. With the level
    # free and the ratio imposed, this is HLW's own device.
    sigma_ystar = _sigma_ystar_scalar(
        model, prior=sigma_ystar_prior, fixed=sigma_ystar_fixed,
    )
    sigma_g_value = _sigma_g_from_lambda(lambda_g, sigma_ystar)

    # --- State equations ---
    descriptions.extend(
        _state_equations(
            obs, model, latents,
            constants=constants,
            resolution=resolution,
            sigma_ystar=sigma_ystar,
            sigma_g_value=sigma_g_value,
            sigma_z_prior=sigma_z_prior,
            given_rstar=given_rstar,
            given_g=given_g,
        ),
    )

    # --- Observation equations ---
    desc = is_curve_equation(
        obs, model, latents,
        constant=constants.get("is_curve"),
        rate_lag=rate_lag,
        keep=keep,
        sign_prior_only=sign_prior_only,
    )
    descriptions.append(f"IS curve:     {desc}")

    desc = phillips_curve_equation(
        obs, model, latents, constant=constants.get("phillips"), keep=keep,
        sign_prior_only=sign_prior_only,
    )
    descriptions.append(f"Phillips:     {desc}")

    if sign_prior_only:
        # Recorded so a run that carries HLW's minimal priors rather than this
        # repo's informative ones says so wherever imposed settings are shown.
        attach(model, {"sign_prior_only": True})

    if exclude_window is not None:
        # Travels with the run's imposed settings so `analyse.py` can shade the
        # window rather than draw states through it as though they were fitted.
        record_constant(model, "exclude_window", exclude_window)

    if resolution == "B":
        desc = indexed_bond_equation(obs, model, latents, constant=constants.get("indexed_bond"))
        descriptions.append(f"Indexed bond: {desc}")

    print("\nHLW r-star model equations:")
    for d in descriptions:
        print(f"  {d}")

    fixed = get_dictionary(model)
    if fixed:
        print("\nFixed constants:")
        for name, value in fixed.items():
            print(f"  {name} = {value}")

    model._descriptions = descriptions
    return model


def save_results(
    trace: az.InferenceData,
    obs: dict[str, np.ndarray],
    obs_index: pd.PeriodIndex,
    *,
    constants: dict[str, Any] | None = None,
    chart_obs: pd.DataFrame | None = None,
    output_dir: Path | str | None = None,
    prefix: str = "rstar_hlw",
) -> Path:
    """Persist trace + observations + metadata to disk."""
    if output_dir is None:
        output_dir = DEFAULT_OUTPUT_DIR
    if constants is None:
        constants = {}
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    trace_path = output_dir / f"{prefix}_trace.nc"
    trace.to_netcdf(str(trace_path))
    print(f"Saved trace to: {trace_path}")

    obs_path = output_dir / f"{prefix}_obs.pkl"
    with obs_path.open("wb") as f:
        pickle.dump(
            {
                "obs": obs,
                "obs_index": obs_index,
                "constants": constants,
                "chart_obs": chart_obs,
            },
            f,
        )
    print(f"Saved observations to: {obs_path}")

    return output_dir


def run_estimate(
    *,
    start: str | None = DEFAULT_START,
    end: str | None = None,
    sampler_config: SamplerConfig | None = None,
    output_dir: Path | str | None = None,
    prefix: str = "rstar_hlw",
    resolution: str = "A",
    rate_lag: int | None = DEFAULT_RATE_LAG,
    sigma_ystar_prior: float = 0.12,
    sigma_ystar_fixed: float | None = DEFAULT_SIGMA_YSTAR_FIXED,
    lambda_g: float | None = DEFAULT_LAMBDA_G,
    exclude_window: tuple[str, str] | None = DEFAULT_EXCLUDE_WINDOW,
    sign_prior_only: bool = False,
    g_anchor: GAnchor = DEFAULT_G_ANCHOR,
    verbose: bool = False,
    seed: int | None = None,
) -> tuple[az.InferenceData, dict[str, np.ndarray], pd.PeriodIndex]:
    """Build observations, sample posterior, save results."""
    if sampler_config is None:
        sampler_config = SamplerConfig(
            draws=10_000,
            tune=3_500,
            chains=5,
            cores=5,
            target_accept=0.90,
        )
    if seed is not None:
        sampler_config.random_seed = seed
    print(f"Sampler seed: {sampler_config.random_seed}")

    print("Building observations...")
    obs, obs_index, chart_obs = build_observations(
        start=start, end=end, verbose=verbose, g_anchor=g_anchor,
    )

    print(f"Building model (Resolution {resolution})...")
    model = build_model(
        obs, resolution=resolution, rate_lag=rate_lag,
        sigma_ystar_prior=sigma_ystar_prior,
        sigma_ystar_fixed=sigma_ystar_fixed,
        lambda_g=lambda_g,
        exclude_window=exclude_window,
        sign_prior_only=sign_prior_only,
        obs_index=obs_index,
    )

    print("\nSampling...")
    trace = sample_model(model, sampler_config)
    print()

    # Prior draws for the free scalars, saved alongside the posterior so
    # `analyse.py` can chart each parameter against its own prior.
    sampled = add_scalar_priors(model, trace, random_seed=sampler_config.random_seed)
    print(f"Sampled priors for {len(sampled)} free scalar parameters\n")

    constants = get_dictionary(model)
    if "sigma_trend_obs" in constants:
        # Travels with the run so the anchor diagnostic can name the series it
        # is plotting. Keyed off the measurement sd that `trend_growth_equation`
        # records only when it actually builds the anchor: `obs` here is still
        # unfiltered, so testing it would claim an anchor for A and B, which
        # drop the series.
        constants["g_anchor"] = g_anchor
    save_results(
        trace, obs, obs_index,
        constants=constants,
        chart_obs=chart_obs,
        output_dir=output_dir,
        prefix=prefix,
    )

    return trace, obs, obs_index
