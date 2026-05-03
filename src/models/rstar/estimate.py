"""Build, sample, and persist the HLW Bayesian r-star model."""

import pickle
from pathlib import Path
from typing import Any

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm

from src.models.nairu.base import SamplerConfig, get_fixed_constants, sample_model
from src.models.rstar.equations.indexed_bond import indexed_bond_equation
from src.models.rstar.equations.is_curve import is_curve_equation
from src.models.rstar.equations.phillips import phillips_curve_equation
from src.models.rstar.equations.potential import potential_output_equation
from src.models.rstar.equations.r_star import r_star_equation
from src.models.rstar.equations.trend_growth import trend_growth_equation
from src.models.rstar.equations.z_star import z_star_equation
from src.models.rstar.observations import build_observations

DEFAULT_OUTPUT_DIR = Path(__file__).parent.parent.parent.parent / "model_outputs"


def build_model(
    obs: dict[str, np.ndarray],
    constants: dict[str, Any] | None = None,
    resolution: str = "C",
) -> pm.Model:
    """Build the HLW r-star PyMC model.

    Equation order matters for NUTS efficiency: state equations first, then
    observation equations.

    Resolution A: r* = g + z (canonical HLW with AR(1) reparameterised z).
    Resolution C: r* = α·g + (1-α)·(indexed_10y - k) + ε (the blend).
    """
    if constants is None:
        constants = {}
    if resolution not in ("A", "B", "C"):
        raise ValueError(f"resolution must be 'A', 'B', or 'C', got {resolution!r}")

    # Resolutions A and B are textbook canonical: no fiscal impulse, no soft
    # anchor on g. Strip those terms from obs so the equations skip them (the
    # equations already check `if X in obs`).
    # B additionally adds an indexed-bond observation equation; A omits it.
    if resolution in ("A", "B"):
        obs = {k: v for k, v in obs.items() if k not in ("trend_growth_obs", "fiscal_impulse_1")}

    model = pm.Model()
    latents: dict[str, Any] = {}
    descriptions: list[str] = []

    # --- State equations ---
    desc = trend_growth_equation(obs, model, latents, constant=constants.get("trend_growth"))
    descriptions.append(f"Trend growth: {desc}")

    if resolution == "C":
        # Order: trend_growth -> r_star (uses g, indexed_10y) -> potential -> observations
        desc = r_star_equation(obs, model, latents, constant=constants.get("r_star"))
        descriptions.append(f"r-star:       {desc}")

        desc = potential_output_equation(obs, model, latents, constant=constants.get("potential"))
        descriptions.append(f"Potential:    {desc}")
    else:  # Resolution A or B (canonical r* = g + z)
        # Order: trend_growth -> potential -> z_star (uses sigma_ystar via lambda_z)
        desc = potential_output_equation(obs, model, latents, constant=constants.get("potential"))
        descriptions.append(f"Potential:    {desc}")

        desc = z_star_equation(obs, model, latents, constant=constants.get("z_star"))
        descriptions.append(f"z-star:       {desc}")

    # --- Observation equations ---
    desc = is_curve_equation(obs, model, latents, constant=constants.get("is_curve"))
    descriptions.append(f"IS curve:     {desc}")

    desc = phillips_curve_equation(obs, model, latents, constant=constants.get("phillips"))
    descriptions.append(f"Phillips:     {desc}")

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
    start: str | None = "1980Q1",
    end: str | None = None,
    sampler_config: SamplerConfig | None = None,
    output_dir: Path | str | None = None,
    prefix: str = "rstar_hlw",
    resolution: str = "C",
    verbose: bool = False,
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

    print("Building observations...")
    obs, obs_index, chart_obs = build_observations(start=start, end=end, verbose=verbose)

    print(f"Building model (Resolution {resolution})...")
    model = build_model(obs, resolution=resolution)

    print("\nSampling...")
    trace = sample_model(model, sampler_config)
    print()

    constants = get_fixed_constants(model)
    save_results(
        trace, obs, obs_index,
        constants=constants,
        chart_obs=chart_obs,
        output_dir=output_dir,
        prefix=prefix,
    )

    return trace, obs, obs_index
