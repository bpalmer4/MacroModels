"""NAIRU + Output Gap estimation: model building, sampling, and saving.

This module handles:
- Building the PyMC model from a ModelConfig
- Sampling the posterior
- Saving results (trace, observations, config) to disk
"""

import pickle
from pathlib import Path
from typing import Any

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm

from src.models.nairu.base import SamplerConfig, get_fixed_constants, sample_model
from src.models.nairu.config import ModelConfig
from src.models.nairu.equations.employment import employment_equation
from src.models.nairu.equations.exchange_rate import exchange_rate_equation
from src.models.nairu.equations.import_price import import_price_equation
from src.models.nairu.equations.is_curve import is_equation
from src.models.nairu.equations.nairu import nairu_equation, nairu_student_t_equation
from src.models.nairu.equations.net_exports import net_exports_equation
from src.models.nairu.equations.okun import okun_equation, okun_gap_equation
from src.models.nairu.equations.participation import participation_equation
from src.models.nairu.equations.phillips_hcoe import (
    hourly_coe_equation,
    hourly_coe_regime_equation,
)
from src.models.nairu.equations.phillips_price import (
    price_inflation_equation,
    price_inflation_regime_equation,
)
from src.models.nairu.equations.phillips_wage import (
    wage_growth_equation,
    wage_growth_regime_equation,
)
from src.models.nairu.equations.potential import (
    potential_output_equation,
    potential_output_skewnormal_equation,
)
from src.models.nairu.observations import ANCHOR_LABELS, AnchorMode, build_observations

# Default output directory
DEFAULT_OUTPUT_DIR = Path(__file__).parent.parent.parent.parent / "model_outputs"


def build_model(obs: dict[str, np.ndarray], config: ModelConfig) -> pm.Model:  # noqa: C901, PLR0912, PLR0915
    """Build the joint NAIRU + Output Gap model from a ModelConfig.

    Args:
        obs: Observation dictionary from build_observations()
        config: Model configuration specifying equations and options

    Returns:
        PyMC Model ready for sampling

    """
    # Strip supply-side controls from Phillips curve if disabled
    strip_keys = set()
    if not config.include_import_price_control:
        strip_keys.add("Δ4ρm_1")
    if not config.include_gscpi_control:
        strip_keys.add("ξ_2")
    if strip_keys:
        obs = {k: v for k, v in obs.items() if k not in strip_keys}

    model = pm.Model()
    latents: dict[str, Any] = {}
    descriptions: list[str] = []

    # --- State equations ---

    # NAIRU
    nairu_func = nairu_student_t_equation if config.student_t_nairu else nairu_equation
    desc = nairu_func(obs, model, latents, constant=config.nairu_const)
    descriptions.append(f"NAIRU: {desc}")

    # Potential output
    potential_func = (
        potential_output_skewnormal_equation if config.skewnormal_potential
        else potential_output_equation
    )
    desc = potential_func(obs, model, latents, constant=config.potential_const)
    descriptions.append(f"Potential: {desc}")

    # --- Observation equations ---

    # Okun's Law
    if config.include_okun:
        okun_func = okun_gap_equation if config.okun_gap_form else okun_equation
        desc = okun_func(obs, model, latents, constant=config.okun_const)
        descriptions.append(f"Okun: {desc}")

    # Price Phillips curve
    if config.include_price_inflation:
        if config.regime_switching:
            desc = price_inflation_regime_equation(
                obs, model, latents, constant=config.price_inflation_const,
            )
        else:
            desc = price_inflation_equation(
                obs, model, latents, constant=config.price_inflation_const,
            )
        descriptions.append(f"Price Phillips: {desc}")

    # Wage Phillips curve (ULC)
    if config.include_wage_growth:
        if config.regime_switching:
            desc = wage_growth_regime_equation(
                obs, model, latents, constant=config.wage_growth_const,
                wage_expectations=config.wage_expectations,
                wage_price_passthrough=config.wage_price_passthrough,
            )
        else:
            desc = wage_growth_equation(
                obs, model, latents, constant=config.wage_growth_const,
                wage_expectations=config.wage_expectations,
                wage_price_passthrough=config.wage_price_passthrough,
            )
        descriptions.append(f"Wage Phillips (ULC): {desc}")

    # IS curve (ordering matters — must come after wage ULC, before HCOE)
    if config.include_is_curve:
        desc = is_equation(obs, model, latents, constant=config.is_curve_const)
        descriptions.append(f"IS curve: {desc}")

    # Hourly COE Phillips curve
    if config.include_hourly_coe:
        if config.regime_switching:
            desc = hourly_coe_regime_equation(
                obs, model, latents, constant=config.hourly_coe_const,
                wage_expectations=config.wage_expectations,
                wage_price_passthrough=config.wage_price_passthrough,
            )
        else:
            desc = hourly_coe_equation(
                obs, model, latents, constant=config.hourly_coe_const,
                wage_expectations=config.wage_expectations,
                wage_price_passthrough=config.wage_price_passthrough,
            )
        descriptions.append(f"Wage Phillips (HCOE): {desc}")

    # Participation rate
    if config.include_participation:
        desc = participation_equation(obs, model, latents, constant=config.participation_const)
        descriptions.append(f"Participation: {desc}")

    # Employment
    if config.include_employment:
        desc = employment_equation(obs, model, latents, constant=config.employment_const)
        descriptions.append(f"Employment: {desc}")

    # Exchange rate
    if config.include_exchange_rate:
        desc = exchange_rate_equation(obs, model, latents, constant=config.exchange_rate_const)
        descriptions.append(f"Exchange rate: {desc}")

    # Import price pass-through
    if config.include_import_price:
        desc = import_price_equation(obs, model, latents, constant=config.import_price_const)
        descriptions.append(f"Import price: {desc}")

    # Net exports
    if config.include_net_exports:
        desc = net_exports_equation(obs, model, latents, constant=config.net_exports_const)
        descriptions.append(f"Net exports: {desc}")

    # Print model summary
    print(f"\n{config.summary()}")
    print("\nModel equations:")
    for d in descriptions:
        print(f"  {d}")

    constants = getattr(model, "_fixed_constants", {})
    if constants:
        print("\nFixed constants:")
        for name, value in constants.items():
            print(f"  {name} = {value}")

    # Store metadata on model
    model._descriptions = descriptions  # noqa: SLF001 — our own metadata on PyMC model
    model._config = config  # noqa: SLF001

    return model


def save_results(
    trace: az.InferenceData,
    obs: dict[str, np.ndarray],
    obs_index: pd.PeriodIndex,
    config: ModelConfig,
    constants: dict[str, Any] | None = None,
    anchor_label: str | None = None,
    chart_obs: pd.DataFrame | None = None,
    output_dir: Path | str | None = None,
    prefix: str = "nairu_output_gap",
) -> Path:
    """Save model results to disk.

    The ModelConfig is serialized alongside the trace so that analyse.py
    and forecast.py know exactly which variant produced the results.

    Args:
        trace: ArviZ InferenceData from sampling
        obs: Observation dictionary
        obs_index: Period index for observations
        config: ModelConfig used for this run
        constants: Fixed parameter values
        anchor_label: Label describing expectations anchor mode
        chart_obs: Extended observations DataFrame for charting
        output_dir: Directory to save to
        prefix: Filename prefix

    Returns:
        Path to output directory

    """
    if output_dir is None:
        output_dir = DEFAULT_OUTPUT_DIR
    if constants is None:
        constants = {}
    if anchor_label is None:
        anchor_label = ANCHOR_LABELS["expectations"]
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save trace as NetCDF
    trace_path = output_dir / f"{prefix}_trace.nc"
    trace.to_netcdf(str(trace_path))
    print(f"Saved trace to: {trace_path}")

    # Save observations, config, and metadata as pickle
    obs_path = output_dir / f"{prefix}_obs.pkl"
    with obs_path.open("wb") as f:
        pickle.dump({
            "obs": obs,
            "obs_index": obs_index,
            "constants": constants,
            "anchor_label": anchor_label,
            "chart_obs": chart_obs,
            "config": config.to_dict(),
        }, f)
    print(f"Saved observations to: {obs_path}")

    return output_dir


def run_estimate(
    start: str | None = "1980Q1",
    end: str | None = None,
    anchor_mode: AnchorMode = "rba",
    config: ModelConfig | None = None,
    sampler_config: SamplerConfig | None = None,
    output_dir: Path | str | None = None,
    prefix: str = "nairu_output_gap",
    verbose: bool = False,
) -> tuple[az.InferenceData, dict[str, np.ndarray], pd.PeriodIndex, str]:
    """Run estimation: build observations, sample model, save results.

    Args:
        start: Start period
        end: End period
        anchor_mode: How to anchor expectations
        config: Model configuration (uses default if None)
        sampler_config: Sampler configuration
        output_dir: Directory to save results
        prefix: Filename prefix
        verbose: Print progress messages

    Returns:
        Tuple of (trace, obs, obs_index, anchor_label)

    """
    if config is None:
        config = ModelConfig()
    if sampler_config is None:
        sampler_config = SamplerConfig(
            draws=10_000,
            tune=3_500,
            chains=5,
            cores=5,
            target_accept=0.90,
        )

    # Build observations
    print("Building observations...")
    print(f"  Anchor mode: {anchor_mode}")
    obs, obs_index, anchor_label, chart_obs = build_observations(
        start=start, end=end, anchor_mode=anchor_mode, verbose=verbose,
    )

    # Build model
    print("Building model...")
    model = build_model(obs, config)

    # Sample
    print("\nSampling...")
    trace = sample_model(model, sampler_config)
    print("\n")

    # Save results
    constants = get_fixed_constants(model)
    save_results(
        trace, obs, obs_index, config,
        constants=constants,
        anchor_label=anchor_label,
        chart_obs=chart_obs,
        output_dir=output_dir,
        prefix=prefix,
    )

    return trace, obs, obs_index, anchor_label
