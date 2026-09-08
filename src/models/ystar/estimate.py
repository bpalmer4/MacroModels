"""Build, sample, and persist the ystar model."""

import pickle
from pathlib import Path
from typing import Any

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm

from src.models.ystar.base import (
    SamplerConfig,
    get_fixed_constants,
    sample_model,
)
from src.models.ystar.config import DEFAULT_OUTPUT_DIR, ModelConfig
from src.models.ystar.equations.hours import hours_equation
from src.models.ystar.equations.inflation_gap import inflation_gap_equation
from src.models.ystar.equations.output import output_equation
from src.models.ystar.equations.participation import participation_equation
from src.models.ystar.equations.phillips import phillips_curve_equation
from src.models.ystar.equations.potential import potential_output_equation
from src.models.ystar.equations.production import production_potential_equation
from src.models.ystar.equations.scale import scale_equation
from src.models.ystar.equations.target_consistency import (
    target_consistency_equation,
)
from src.models.ystar.equations.trend_hours import trend_hours_equation
from src.models.ystar.equations.trend_productivity import (
    trend_productivity_equation,
)
from src.models.ystar.observations import build_observations


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


# Specs whose potential is a single free level recursion, and so can carry a
# one-off step. `production` builds the level from the factor trends and
# `labour` from trend hours times trend productivity; a step in either would be
# a step in one component, which is a different claim.
_BREAKABLE_SPECS = ("inflation", "core", "target")


def _keep_mask(config: ModelConfig, obs_index: pd.PeriodIndex | None) -> np.ndarray | None:
    """Resolve `config.exclude_window` to a boolean keep-mask over the sample.

    Returns None when no window is excluded. Raises rather than quietly
    excluding nothing if the window misses the sample, and rather than leaving
    the model with too little to fit if it swallows most of it.
    """
    if config.exclude_window is None:
        return None
    if obs_index is None:
        raise ValueError("exclude_window needs obs_index; pass it to build_model")
    if config.spec not in ("inflation", "production"):
        raise ValueError(
            f"exclude_window is not available for the {config.spec} spec: the mask is "
            f"applied in the inflation-gap GDP equation, which that spec does not use",
        )

    lo, hi = config.exclude_window
    dropped = (obs_index >= pd.Period(lo, freq="Q")) & (obs_index <= pd.Period(hi, freq="Q"))
    if not dropped.any():
        raise ValueError(
            f"exclude_window {lo} to {hi} covers no quarter of the estimation sample "
            f"{obs_index.min()} to {obs_index.max()}",
        )
    if dropped.all():
        raise ValueError(f"exclude_window {lo} to {hi} would drop the whole sample")

    return ~np.asarray(dropped)


def _record_exclusion(model: pm.Model, config: ModelConfig) -> None:
    """Record the excluded window with the run's imposed settings.

    It travels in the obs pickle to `analyse.py`, which shades the window rather
    than drawing states through it as though they had been fitted. Call after
    `scale_equation`, which creates the dict this writes into.
    """
    if config.exclude_window is not None:
        get_fixed_constants(model)["exclude_window"] = config.exclude_window


def _break_quarters(config: ModelConfig) -> tuple[str, ...]:
    """Normalise `config.level_break` to a tuple of quarter strings."""
    if config.level_break is None:
        return ()
    if isinstance(config.level_break, str):
        return (config.level_break,)
    return tuple(config.level_break)


def _break_indices(
    config: ModelConfig,
    obs_index: pd.PeriodIndex | None,
) -> tuple[tuple[int, ...], tuple[str, ...]]:
    """Resolve `config.level_break` to positions in the estimation sample.

    Returns empty tuples when no break is asked for. Every quarter must be in
    the sample: a break outside it is silently a no-op otherwise, which would
    leave the run log claiming a break the model does not have.
    """
    quarters = _break_quarters(config)
    if not quarters:
        return (), ()
    if obs_index is None:
        raise ValueError("level_break needs obs_index; pass it to build_model")
    if config.spec not in _BREAKABLE_SPECS:
        raise ValueError(
            f"level_break is not available for the {config.spec} spec: potential's level "
            f"is built from factor or component trends there, so there is no single free "
            f"level recursion to break. Available for {_BREAKABLE_SPECS}",
        )
    if len(set(quarters)) != len(quarters):
        raise ValueError(f"level_break has a repeated quarter: {quarters}")

    indices = []
    for quarter in quarters:
        positions = np.flatnonzero(obs_index == pd.Period(quarter, freq="Q"))
        if positions.size == 0:
            raise ValueError(
                f"level_break {quarter!r} is outside the estimation sample "
                f"{obs_index.min()} to {obs_index.max()}",
            )
        indices.append(int(positions[0]))

    # Sorted so the trace's break coordinate reads chronologically whatever
    # order they were given in.
    order = sorted(range(len(indices)), key=lambda i: indices[i])
    return tuple(indices[i] for i in order), tuple(quarters[i] for i in order)


def _potential_constant(
    break_index: tuple[int, ...],
    break_labels: tuple[str, ...],
) -> dict[str, Any] | None:
    """Constants for `potential_output_equation`, or None when there are none."""
    if not break_index:
        return None
    return {"break_index": break_index, "break_labels": break_labels}


def _inflation_family(
    obs: dict[str, np.ndarray],
    model: pm.Model,
    latents: dict[str, Any],
    config: ModelConfig,
    *,
    potential_constant: dict[str, Any] | None = None,
    keep: np.ndarray | None = None,
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
        desc = potential_output_equation(
            obs, model, latents,
            constant=potential_constant,
        )

    gap_desc = inflation_gap_equation(
        obs, model, latents,
        constant={
            "anchor": config.anchor,
            "ar1_residual": config.ar1_residual,
            "two_sided_c": config.two_sided_c,
            "keep": keep,
        },
    )
    return [f"Potential:    {desc}", f"Gap:          {gap_desc}"]


def build_model(
    obs: dict[str, np.ndarray],
    config: ModelConfig | None = None,
    verbose: bool = True,
    obs_index: pd.PeriodIndex | None = None,
) -> pm.Model:
    """Build the ystar PyMC model.

    Equation order matters: the variance scale first, then the state
    equations, then the observation equations (see `equations/__init__.py`).

    `obs_index` is needed only when `config.level_break` is set, to turn the
    nominated quarter into a position in the sample.
    """
    if config is None:
        config = ModelConfig()

    break_index, break_labels = _break_indices(config, obs_index)
    potential_constant = _potential_constant(break_index, break_labels)
    keep = _keep_mask(config, obs_index)

    model = pm.Model()
    latents: dict[str, Any] = {}
    descriptions: list[str] = []

    # --- Variance scale (all imposed: one fixed sigma, fixed ratios) ---
    desc = scale_equation(obs, model, latents, constant=config.scale_constants)
    descriptions.append(f"Scale:        {desc}")

    _record_exclusion(model, config)

    descriptions.extend(_free_sigma_ystar(model, latents, config=config))

    # Potential is a state as usual; what differs is that the gap is *defined*
    # by inflation rather than restricted to look like a cycle, and GDP is
    # fitted around the two with a white-noise residual. No Phillips curve, no
    # IS curve, no AR(2).
    if config.spec in ("inflation", "production"):
        descriptions.extend(
            _inflation_family(
                obs, model, latents, config,
                potential_constant=potential_constant, keep=keep,
            ),
        )

        if verbose:
            print("\nModel specification:")
            for line in descriptions:
                print(f"  {line}")
            print()
        return model

    # --- State equations ---
    if config.spec in ("core", "target"):
        desc = potential_output_equation(
            obs, model, latents,
            constant=potential_constant,
        )
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
    prefix: str = "ystar",
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
    prefix: str = "ystar",
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
    obs, obs_index, chart_obs, sources = build_observations(
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

    quarters = _break_quarters(config)
    if quarters:
        print(f"Level breaks: {', '.join(quarters)}  (free one-off steps in y*)")

    if config.exclude_window is not None:
        lo, hi = config.exclude_window
        dropped = ((obs_index >= pd.Period(lo, freq="Q")) & (obs_index <= pd.Period(hi, freq="Q")))
        print(f"Excluded:     {lo} to {hi}  ({int(dropped.sum())} quarters carry no likelihood)")

    print("Building model...")
    model = build_model(obs, config=config, obs_index=obs_index)

    print("Sampling...")
    trace = sample_model(model, sampler_config)
    print()

    # The providers behind the observations travel with the run, so the charts
    # name what was actually loaded rather than a separately maintained string.
    constants = {**get_fixed_constants(model), "sources": sources.to_records()}
    save_results(
        trace, obs, obs_index,
        constants=constants,
        chart_obs=chart_obs,
        output_dir=config.output_dir,
        prefix=prefix,
    )

    return trace, obs, obs_index
