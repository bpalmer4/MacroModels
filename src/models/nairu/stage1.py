"""NAIRU + Output Gap Stage 1: Model building and sampling.

This module handles:
- Building the PyMC model
- Sampling the posterior
- Saving results (observations, trace) to disk

Data preparation is handled by src.data.observations.
"""

import pickle
from pathlib import Path
from typing import Any

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm

from src.data.henderson import hma
from src.data.observations import ANCHOR_LABELS, AnchorMode, build_observations
from src.models.nairu.base import SamplerConfig, get_fixed_constants, sample_model
from src.models.nairu.equations import (
    employment_equation,
    exchange_rate_equation,
    hourly_coe_equation,
    import_price_equation,
    is_equation,
    nairu_equation,
    net_exports_equation,
    okun_equation,
    participation_equation,
    potential_output_equation,
    price_inflation_equation,
    wage_growth_equation,
)

# Default output directory
DEFAULT_OUTPUT_DIR = Path(__file__).parent.parent.parent.parent / "model_outputs"


# --- Model Assembly ---


# Default fixed constants for model building
# NAIRU uses Student-t(nu=4) innovations for robustness to occasional large shifts
DEFAULT_NAIRU_CONST: dict[str, Any] = {"nairu_innovation": 0.15}
DEFAULT_POTENTIAL_CONST: dict[str, Any] = {"potential_innovation": 0.3}


def build_model(
    obs: dict[str, np.ndarray],
    nairu_const: dict[str, Any] | None = None,
    potential_const: dict[str, Any] | None = None,
    exchange_rate_const: dict[str, Any] | None = None,
    import_price_const: dict[str, Any] | None = None,
    participation_const: dict[str, Any] | None = None,
    hourly_coe_const: dict[str, Any] | None = None,
    employment_const: dict[str, Any] | None = None,
    net_exports_const: dict[str, Any] | None = None,
    include_exchange_rate: bool = True,
    include_import_price: bool = True,
    include_participation: bool = True,
    include_hourly_coe: bool = True,
    include_employment: bool = True,
    include_net_exports: bool = True,
) -> pm.Model:
    """Build the joint NAIRU + Output Gap model.

    Args:
        obs: Observation dictionary from build_observations()
        nairu_const: Fixed values for NAIRU equation
        potential_const: Fixed values for potential output equation
        exchange_rate_const: Fixed values for exchange rate equation
        import_price_const: Fixed values for import price equation
        participation_const: Fixed values for participation rate equation
        hourly_coe_const: Fixed values for hourly COE equation
        employment_const: Fixed values for employment equation
        net_exports_const: Fixed values for net exports equation
        include_exchange_rate: Whether to include TWI equation (default True)
        include_import_price: Whether to include import price pass-through (default True)
        include_participation: Whether to include participation rate equation (default True)
        include_hourly_coe: Whether to include hourly COE wage equation (default True)
        include_employment: Whether to include employment equation (default True)
        include_net_exports: Whether to include net exports equation (default True)

    Returns:
        PyMC Model ready for sampling (fixed constants stored on model._fixed_constants)

    """
    if nairu_const is None:
        nairu_const = DEFAULT_NAIRU_CONST.copy()
    if potential_const is None:
        potential_const = DEFAULT_POTENTIAL_CONST.copy()

    model = pm.Model()

    # State equations
    nairu = nairu_equation(obs, model, constant=nairu_const)
    potential = potential_output_equation(obs, model, constant=potential_const)

    # Observation equations
    okun_equation(obs, model, nairu, potential)  # Error correction: ΔU = β×OG + α×(U-NAIRU-γ×OG)
    price_inflation_equation(obs, model, nairu)
    wage_growth_equation(obs, model, nairu)
    is_equation(obs, model, potential)

    # Hourly COE wage equation (optional, default on)
    if include_hourly_coe:
        hourly_coe_equation(obs, model, nairu, constant=hourly_coe_const)

    # Labour supply equation (optional)
    if include_participation:
        participation_equation(obs, model, nairu, constant=participation_const)

    # Employment equation (optional) - labour demand with wage channel
    if include_employment:
        employment_equation(obs, model, potential, constant=employment_const)

    # Open economy equations (optional)
    if include_exchange_rate:
        exchange_rate_equation(obs, model, constant=exchange_rate_const)
    if include_import_price:
        import_price_equation(obs, model, constant=import_price_const)
    if include_net_exports:
        net_exports_equation(obs, model, potential, constant=net_exports_const)

    return model


# --- Save/Load Functions ---


def save_results(
    trace: az.InferenceData,
    obs: dict[str, np.ndarray],
    obs_index: pd.PeriodIndex,
    constants: dict[str, Any] | None = None,
    anchor_label: str | None = None,
    output_dir: Path | str | None = None,
    prefix: str = "nairu_output_gap",
) -> Path:
    """Save model results to disk.

    Args:
        trace: ArviZ InferenceData from sampling
        obs: Observation dictionary
        obs_index: Period index for observations
        constants: Fixed parameter values used in model building
        anchor_label: Label describing expectations anchor mode (for chart annotations)
        output_dir: Directory to save to (default: model_outputs/)
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

    # Save observations, index, constants, and anchor label as pickle
    obs_path = output_dir / f"{prefix}_obs.pkl"
    with open(obs_path, "wb") as f:
        pickle.dump({
            "obs": obs,
            "obs_index": obs_index,
            "constants": constants,
            "anchor_label": anchor_label,
        }, f)
    print(f"Saved observations to: {obs_path}")

    return output_dir


# --- Main Entry Point ---


def run_stage1(
    start: str | None = "1980Q1",
    end: str | None = None,
    anchor_mode: AnchorMode = "target",
    config: SamplerConfig | None = None,
    output_dir: Path | str | None = None,
    prefix: str = "nairu_output_gap",
    verbose: bool = False,
) -> tuple[az.InferenceData, dict[str, np.ndarray], pd.PeriodIndex, str]:
    """Run Stage 1: Build observations, sample model, and save results.

    Args:
        start: Start period
        end: End period
        anchor_mode: How to anchor expectations
            - "expectations": Use full estimated expectations series
            - "target": Phase from expectations to 2.5% target (1993-1998)
        config: Sampler configuration
        output_dir: Directory to save results
        prefix: Filename prefix for saved files
        verbose: Print progress messages

    Returns:
        Tuple of (trace, obs, obs_index, anchor_label)

    """
    if config is None:
        config = SamplerConfig(
            draws=10_000,
            tune=3_500,
            chains=5,
            cores=5,
            target_accept=0.90,
        )

    # Build observations
    print("Building observations...")
    print(f"  Anchor mode: {anchor_mode}")
    obs, obs_index, anchor_label = build_observations(
        start=start, end=end, anchor_mode=anchor_mode, verbose=verbose
    )

    # Apply HMA(13) smoothing to labour force growth for potential calculation
    lf_raw = pd.Series(obs["lf_growth"], index=obs_index)
    obs["lf_growth"] = hma(lf_raw, 13).to_numpy()

    # Build model
    print("Building model...")
    model = build_model(obs)

    # Sample
    print("\nSampling...")
    trace = sample_model(model, config)
    print("\n")

    # Save results (including fixed constants from model)
    constants = get_fixed_constants(model)
    save_results(
        trace, obs, obs_index,
        constants=constants,
        anchor_label=anchor_label,
        output_dir=output_dir,
        prefix=prefix,
    )

    return trace, obs, obs_index, anchor_label


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run NAIRU + Output Gap model (Stage 1)")
    parser.add_argument("-v", "--verbose", action="store_true", help="Print detailed output")
    parser.add_argument("--start", type=str, default="1980Q1", help="Start period")
    parser.add_argument("--end", type=str, default=None, help="End period")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory")
    parser.add_argument(
        "--anchor",
        type=str,
        choices=["expectations", "target"],
        default="target",
        help="Expectations anchor mode: 'target' (phase to 2.5%%, default) or 'expectations' (full series)",
    )
    args = parser.parse_args()

    run_stage1(
        start=args.start,
        end=args.end,
        anchor_mode=args.anchor,
        output_dir=args.output_dir,
        verbose=args.verbose,
    )
