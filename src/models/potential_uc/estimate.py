"""Build, sample, and persist the potential_uc model."""

import pickle
from pathlib import Path
from typing import Any

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm

from src.models.potential_uc.base import (
    SamplerConfig,
    get_fixed_constants,
    sample_model,
)
from src.models.potential_uc.config import DEFAULT_OUTPUT_DIR, ModelConfig
from src.models.potential_uc.equations.hours import hours_equation
from src.models.potential_uc.equations.inflation_gap import inflation_gap_equation
from src.models.potential_uc.equations.output import output_equation
from src.models.potential_uc.equations.participation import participation_equation
from src.models.potential_uc.equations.phillips import phillips_curve_equation
from src.models.potential_uc.equations.potential import potential_output_equation
from src.models.potential_uc.equations.production import production_potential_equation
from src.models.potential_uc.equations.scale import scale_equation
from src.models.potential_uc.equations.target_consistency import (
    target_consistency_equation,
)
from src.models.potential_uc.equations.trend_hours import trend_hours_equation
from src.models.potential_uc.equations.trend_productivity import (
    trend_productivity_equation,
)
from src.models.potential_uc.observations import build_observations


def _free_sigma_ystar(
    model: pm.Model,
    latents: dict[str, Any],
    config: ModelConfig,
) -> list[str]:
    """Replace the imposed sigma_ystar with an estimated one, if asked.

    The weak HalfNormal is deliberate: the point of freeing it is to let the
    data say how far potential wanders, so the prior should not answer that
    question. See `ModelConfig.free_sigma_ystar` for why this is identified in
    the `inflation` specification but not in the others.
    """
    if not config.free_sigma_ystar:
        return []

    with model:
        latents["sigma_ystar"] = pm.HalfNormal("sigma_ystar", sigma=1.0)

    # scale_equation recorded ratio_ystar as a fixed constant a moment ago, and
    # it is no longer one. Left in place it would show up in the run log and on
    # the charts as an imposed setting that the model is not in fact using.
    get_fixed_constants(model).pop("ratio_ystar", None)

    return ["Scale:        sigma_ystar estimated, not imposed"]


def _inflation_family(
    obs: dict[str, np.ndarray],
    model: pm.Model,
    latents: dict[str, Any],
    config: ModelConfig,
) -> list[str]:
    """Build the potential and gap blocks shared by `inflation` and `production`.

    Both define the gap by inflation and fit GDP around it with a residual, so
    they differ only in where potential's *growth* comes from: a free drift
    state (`potential.py`) or the factor trends (`production.py`).
    """
    if config.spec == "production":
        desc = production_potential_equation(
            obs, model, latents,
            constant={
                "ratio_gk": config.ratio_gk,
                "ratio_gl": config.ratio_gl,
                "ratio_gm": config.ratio_gm,
                "ratio_a": config.ratio_a,
                "mfp_observed": config.mfp_observed,
                "sigma_gm": config.sigma_gm,
            },
        )
    else:
        desc = potential_output_equation(obs, model, latents)

    gap_desc = inflation_gap_equation(
        obs, model, latents,
        constant={
            "anchor": config.anchor,
            "ar1_residual": config.ar1_residual,
            "two_sided_c": config.two_sided_c,
        },
    )
    return [f"Potential:    {desc}", f"Gap:          {gap_desc}"]


def build_model(
    obs: dict[str, np.ndarray],
    config: ModelConfig | None = None,
    verbose: bool = True,
) -> pm.Model:
    """Build the potential_uc PyMC model.

    Equation order matters: the variance scale first, then the state
    equations, then the observation equations (see `equations/__init__.py`).
    """
    if config is None:
        config = ModelConfig()

    model = pm.Model()
    latents: dict[str, Any] = {}
    descriptions: list[str] = []

    # --- Variance scale (all imposed: one fixed sigma, fixed ratios) ---
    desc = scale_equation(obs, model, latents, constant=config.scale_constants)
    descriptions.append(f"Scale:        {desc}")

    descriptions.extend(_free_sigma_ystar(model, latents, config=config))

    # Potential is a state as usual; what differs is that the gap is *defined*
    # by inflation rather than restricted to look like a cycle, and GDP is
    # fitted around the two with a white-noise residual. No Phillips curve, no
    # IS curve, no AR(2).
    if config.spec in ("inflation", "production"):
        descriptions.extend(_inflation_family(obs, model, latents, config))

        if verbose:
            print("\nModel specification:")
            for line in descriptions:
                print(f"  {line}")
            print()
        return model

    # --- State equations ---
    if config.spec in ("core", "target"):
        desc = potential_output_equation(obs, model, latents)
        descriptions.append(f"Potential:    {desc}")
    else:
        desc = trend_hours_equation(obs, model, latents)
        descriptions.append(f"Trend hours:  {desc}")

        desc = trend_productivity_equation(obs, model, latents)
        descriptions.append(f"Trend prod:   {desc}")

    # --- Observation equations ---
    desc = output_equation(obs, model, latents, constant={"cycle_ar": config.cycle_ar})
    descriptions.append(f"Output:       {desc}")

    if config.spec == "labour":
        desc = hours_equation(obs, model, latents)
        descriptions.append(f"Hours:        {desc}")

        desc = participation_equation(obs, model, latents)
        descriptions.append(f"Participation:{desc}")

    if config.spec == "target":
        # No Phillips curve at all: inflation weights the "gap is zero"
        # statement rather than entering as a regressor on the gap.
        desc = target_consistency_equation(
            obs, model, latents,
            constant={
                "anchor": config.anchor,
                "gap_sd_on_target": config.gap_sd_on_target,
                "gap_sd_per_pp": config.gap_sd_per_pp,
                "pi_lag_max": config.pi_lag_max,
            },
        )
        descriptions.append(f"Target:       {desc}")
    else:
        desc = phillips_curve_equation(obs, model, latents, constant={"anchor": config.anchor})
        descriptions.append(f"Phillips:     {desc}")

    if verbose:
        print("\nModel specification:")
        for line in descriptions:
            print(f"  {line}")
        print()

    return model


def save_results(
    trace: az.InferenceData,
    obs: dict[str, np.ndarray],
    obs_index: pd.PeriodIndex,
    constants: dict[str, Any],
    *,
    chart_obs: pd.DataFrame | None = None,
    output_dir: Path | str | None = None,
    prefix: str = "potential_uc",
) -> Path:
    """Persist trace and observations to `model_outputs`."""
    output_dir = Path(output_dir) if output_dir is not None else DEFAULT_OUTPUT_DIR
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
    config: ModelConfig | None = None,
    sampler_config: SamplerConfig | None = None,
    prefix: str = "potential_uc",
    verbose: bool = False,
    seed: int | None = None,
) -> tuple[az.InferenceData, dict[str, np.ndarray], pd.PeriodIndex]:
    """Build observations, sample the posterior, save the results."""
    if config is None:
        config = ModelConfig()
    if sampler_config is None:
        sampler_config = SamplerConfig()
    if seed is not None:
        sampler_config.random_seed = seed

    ratios = ",  ".join(
        f"{key.replace('ratio_', '')}={value:g}"
        for key, value in config.scale_constants.items()
        if key != "sigma_c"
    )
    print(f"Spec:         {config.spec}")
    print(f"Sample:       {config.start} -> {config.end or 'latest'}")
    print(f"Anchor:       {config.anchor}%  (inflation basis: {config.pi_basis})")
    print(f"Supply ctrl:  {config.supply_control or 'none'}")
    print(f"Variances:    sigma_c={config.sigma_c:g} (fixed);  ratios {ratios}")
    print(f"Sampler seed: {sampler_config.random_seed}")

    print("\nBuilding observations...")
    obs, obs_index, chart_obs = build_observations(
        start=config.start, end=config.end, verbose=verbose,
        smooth_pop=config.smooth_pop, spec=config.spec,
        pi_basis=config.pi_basis, supply_control=config.supply_control,
    )

    if config.zero_deviation is not None:
        # Setting pi to the anchor is exactly d = 0 for those quarters, and is
        # done here rather than inside the equation because this is the only
        # place holding both the observations and their period index.
        lo, hi = config.zero_deviation
        mask = (obs_index >= pd.Period(lo, freq="Q")) & (obs_index <= pd.Period(hi, freq="Q"))
        obs["pi"] = np.where(mask, config.anchor, obs["pi"])
        print(f"Zeroed dev:   {lo} to {hi}  ({int(mask.sum())} quarters carry no deviation)")

    print("Building model...")
    model = build_model(obs, config=config)

    print("Sampling...")
    trace = sample_model(model, sampler_config)
    print()

    constants = get_fixed_constants(model)
    save_results(
        trace, obs, obs_index,
        constants=constants,
        chart_obs=chart_obs,
        output_dir=config.output_dir,
        prefix=prefix,
    )

    return trace, obs, obs_index
