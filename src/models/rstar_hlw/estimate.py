"""Build, sample, and persist the HLW Bayesian r-star model."""

import pickle
from pathlib import Path
from typing import Any

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt

from src.models.nairu.base import (
    SamplerConfig,
    add_scalar_priors,
    get_fixed_constants,
    sample_model,
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
from src.models.rstar_hlw.observations import DEFAULT_START, build_observations
from src.paths import MODEL_OUTPUTS

DEFAULT_OUTPUT_DIR = MODEL_OUTPUTS

# The lockdown quarters, dropped from the IS and Phillips likelihoods by
# default. Same window `ystar` uses, and for the same reason: with it in,
# potential takes part of the 2020 collapse as a fall in potential and never
# catches back up, so the recovery reads as output seven per cent above
# capacity through 2022-23. Pass exclude_window=None to keep them.
DEFAULT_EXCLUDE_WINDOW: tuple[str, str] | None = ("2020Q2", "2021Q3")


# lambda_g is HLW's signal-to-noise ratio for trend growth, sigma_g /
# sigma_ystar, which they fix by Stock-Watson median-unbiased estimation rather
# than estimating. It is OFF by default here, and the sweep is why.
#
# `lambda_g_sweep.py` ran Resolution A at 0.053 (HLW's US value), 0.15, 0.34
# and free. HLW's own value is rejected by Australian data on every diagnostic:
# it holds g nearly flat, puts the 2005-19 economy 4.35% above capacity, and
# collapses the Phillips slope to 0.054 against its 0.02 lower bound. At 0.34,
# the ratio Australia's growth slowdown implies, the constraint is not binding
# at all: that run and the free run agree to the third decimal on every
# diagnostic, including r* latest (1.71 vs 1.73).
#
# So imposing the ratio here would add a declared setting that changes nothing,
# which is a cost with no benefit. sigma_ystar is still imposed, so the
# original pile-up cannot come back through potential; if it ever migrates into
# g as data accumulates, the sweep is how to see it, and passing lambda_g=0.34
# is how to stop it.
#
# ASSUMPTION, NOT VERIFIED HERE: that 0.053 is HLW (2017)'s US estimate is
# carried from memory, not checked against the paper. It only ever served as
# the sweep's low end, and nothing downstream rests on the exact figure.
DEFAULT_LAMBDA_G: float | None = None

# g is annualised in this implementation (the y* equation uses g/4), while
# HLW's g is the quarterly growth of potential. A ratio quoted on their g
# therefore scales by 4 on ours.
_QUARTERS_PER_YEAR = 4


def _sigma_g_from_lambda(
    lambda_g: float | None,
    sigma_ystar_fixed: float | None,
) -> float | None:
    """Turn the lambda_g ratio into an imposed sigma_g in annualised units.

    Returns None when no ratio is imposed. Raises when sigma_ystar is being
    estimated, because a ratio to a free parameter is not a fixed number: the
    two would have to be tied inside the model, and the state equations run in
    an order (trend growth before potential) that does not allow it.
    """
    if lambda_g is None:
        return None
    if sigma_ystar_fixed is None:
        raise ValueError(
            "lambda_g fixes sigma_g as a ratio to sigma_ystar, so sigma_ystar must be "
            "imposed too. Either pass sigma_ystar_fixed, or set lambda_g to None to "
            "sample sigma_g under its own prior",
        )
    return _QUARTERS_PER_YEAR * lambda_g * sigma_ystar_fixed


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
    sigma_ystar_prior: float,
    sigma_ystar_fixed: float | None,
    sigma_g_fixed: float | None,
    sigma_z_prior: float,
    given_rstar: np.ndarray | None,
) -> list[str]:
    """Add the state equations, in the order each resolution needs them."""
    descriptions = [
        "Trend growth: " + trend_growth_equation(
            obs, model, latents,
            constant=constants.get("trend_growth"),
            sigma_g_fixed=sigma_g_fixed,
        ),
    ]

    def potential() -> str:
        return potential_output_equation(
            obs, model, latents,
            constant=constants.get("potential"),
            sigma_ystar_prior=sigma_ystar_prior,
            sigma_ystar_fixed=sigma_ystar_fixed,
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
    rate_lag: int | None = 6,
    sigma_ystar_prior: float = 0.12,
    sigma_ystar_fixed: float | None = 0.078,
    lambda_g: float | None = DEFAULT_LAMBDA_G,
    sigma_z_prior: float = 0.10,
    given_rstar: np.ndarray | None = None,
    exclude_window: tuple[str, str] | None = DEFAULT_EXCLUDE_WINDOW,
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
    potential. `lambda_g` would impose trend growth's innovation sd as a ratio
    to potential's, completing HLW's device; it is OFF by default because the
    sweep found the binding values are rejected and the non-binding ones change
    nothing. See `equations/potential.py`, `equations/exclusion.py` and
    `lambda_g_sweep.py`.
    """
    if constants is None:
        constants = {}
    if resolution not in ("A", "B", "C", "D", "E", "F", "G", "H"):
        raise ValueError(
            f"resolution must be 'A', 'B', 'C', 'D', 'E', 'F', 'G' or 'H', "
            f"got {resolution!r}",
        )

    obs = _resolution_obs(obs, resolution)

    # Resolution G: inject the hierarchical-Beta flag into the r_star constants
    # so r_star_equation samples a, b ~ HalfNormal(1) and alpha ~ Beta(a, b).
    if resolution == "G":
        constants = dict(constants)  # shallow copy so we don't mutate caller's dict
        r_star_const = dict(constants.get("r_star", {}))
        r_star_const.setdefault("alpha_hierarchical", True)
        constants["r_star"] = r_star_const

    keep = _keep_mask(exclude_window, obs_index)
    sigma_g_fixed = _sigma_g_from_lambda(lambda_g, sigma_ystar_fixed)

    model = pm.Model()
    latents: dict[str, Any] = {}
    descriptions: list[str] = []

    # --- State equations ---
    descriptions.extend(
        _state_equations(
            obs, model, latents,
            constants=constants,
            resolution=resolution,
            sigma_ystar_prior=sigma_ystar_prior,
            sigma_ystar_fixed=sigma_ystar_fixed,
            sigma_g_fixed=sigma_g_fixed,
            sigma_z_prior=sigma_z_prior,
            given_rstar=given_rstar,
        ),
    )

    # --- Observation equations ---
    desc = is_curve_equation(
        obs, model, latents,
        constant=constants.get("is_curve"),
        rate_lag=rate_lag,
        keep=keep,
    )
    descriptions.append(f"IS curve:     {desc}")

    desc = phillips_curve_equation(
        obs, model, latents, constant=constants.get("phillips"), keep=keep,
    )
    descriptions.append(f"Phillips:     {desc}")

    if exclude_window is not None:
        # Travels with the run's imposed settings so `analyse.py` can shade the
        # window rather than draw states through it as though they were fitted.
        get_fixed_constants(model)["exclude_window"] = exclude_window

    if resolution == "B":
        desc = indexed_bond_equation(obs, model, latents, constant=constants.get("indexed_bond"))
        descriptions.append(f"Indexed bond: {desc}")

    print("\nHLW r-star model equations:")
    for d in descriptions:
        print(f"  {d}")

    fixed = getattr(model, "_fixed_constants", {})
    if fixed:
        print("\nFixed constants:")
        for name, value in fixed.items():
            print(f"  {name} = {value}")

    model._descriptions = descriptions  # noqa: SLF001
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
    rate_lag: int | None = 6,
    sigma_ystar_prior: float = 0.12,
    sigma_ystar_fixed: float | None = 0.078,
    lambda_g: float | None = DEFAULT_LAMBDA_G,
    exclude_window: tuple[str, str] | None = DEFAULT_EXCLUDE_WINDOW,
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
        start=start, end=end, verbose=verbose,
    )

    print(f"Building model (Resolution {resolution})...")
    model = build_model(
        obs, resolution=resolution, rate_lag=rate_lag,
        sigma_ystar_prior=sigma_ystar_prior,
        sigma_ystar_fixed=sigma_ystar_fixed,
        lambda_g=lambda_g,
        exclude_window=exclude_window,
        obs_index=obs_index,
    )

    print("\nSampling...")
    trace = sample_model(model, sampler_config)
    print()

    # Prior draws for the free scalars, saved alongside the posterior so
    # `analyse.py` can chart each parameter against its own prior.
    sampled = add_scalar_priors(model, trace, random_seed=sampler_config.random_seed)
    print(f"Sampled priors for {len(sampled)} free scalar parameters\n")

    constants = get_fixed_constants(model)
    save_results(
        trace, obs, obs_index,
        constants=constants,
        chart_obs=chart_obs,
        output_dir=output_dir,
        prefix=prefix,
    )

    return trace, obs, obs_index
