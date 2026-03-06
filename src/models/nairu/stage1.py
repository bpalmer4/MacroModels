"""NAIRU + Output Gap Stage 1: Model building and sampling.

This module handles:
- Building the PyMC model
- Sampling the posterior
- Saving results (observations, trace) to disk

Data preparation is handled by src.data.observations.
"""

import inspect
import pickle
import re
from pathlib import Path
from typing import Any, Callable

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm

from src.data.observations import ANCHOR_LABELS, AnchorMode, build_observations
from src.models.nairu.base import SamplerConfig, get_fixed_constants, sample_model
from src.models.nairu.equations import (
    employment_equation,
    exchange_rate_equation,
    hourly_coe_equation,
    hourly_coe_regime_equation,
    import_price_equation,
    is_equation,
    nairu_equation,
    nairu_student_t_equation,
    net_exports_equation,
    okun_equation,
    okun_gap_equation,
    participation_equation,
    potential_output_equation,
    potential_output_skewnormal_equation,
    price_inflation_equation,
    price_inflation_regime_equation,
    wage_growth_equation,
    wage_growth_regime_equation,
)

# Default output directory
DEFAULT_OUTPUT_DIR = Path(__file__).parent.parent.parent.parent / "model_outputs"

_MODEL_LINE_RE = re.compile(r"^\s*Model:\s*(.+)$", re.MULTILINE)


def _get_model_line(func: Callable) -> str | None:
    """Extract the 'Model:' line from a function's docstring."""
    doc = inspect.getdoc(func)
    if doc is None:
        return None
    m = _MODEL_LINE_RE.search(doc)
    return m.group(1).strip() if m else None


def print_model_equations(equations: list[Callable]) -> None:
    """Print model equations extracted from function docstrings."""
    print("\nModel equations:")
    for func in equations:
        line = _get_model_line(func)
        if line:
            # Use the first line of the docstring as the equation name
            name = (inspect.getdoc(func) or "").split("\n")[0].rstrip(".")
            print(f"  {name}")
            print(f"    {line}")


# --- Model Assembly ---


# Default fixed constants for model building
# Default fixed constants for state equations (Gaussian by default; Student-t via complex variant)
DEFAULT_NAIRU_CONST: dict[str, Any] = {"nairu_innovation": 0.15}
DEFAULT_POTENTIAL_CONST: dict[str, Any] = {"potential_innovation": 0.3}

# Model variant configurations (kwargs for build_model, only differences from defaults)
MODEL_VARIANTS: dict[str, dict[str, Any]] = {
    "simple": {
        "student_t_nairu": False,
        "regime_switching": False,
        "include_import_price_control": False,
    },
    "complex": {
        "student_t_nairu": True,
        "regime_switching": True,
        "include_import_price_control": True,
        "include_exchange_rate": True,
        "include_import_price": True,
        "include_participation": True,
        "include_employment": True,
        "include_net_exports": True,
        "okun_gap_form": True,
        "nairu_const": {"nairu_innovation": 0.10},
    },
}


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
    price_inflation_const: dict[str, Any] | None = None,
    wage_growth_const: dict[str, Any] | None = None,
    is_curve_const: dict[str, Any] | None = None,
    include_wage_growth: bool = True,
    include_exchange_rate: bool = False,
    include_import_price: bool = False,
    include_participation: bool = False,
    include_hourly_coe: bool = True,
    include_employment: bool = False,
    include_net_exports: bool = False,
    include_is_curve: bool = True,
    regime_switching: bool = False,
    student_t_nairu: bool = False,
    okun_gap_form: bool = False,
    skewnormal_potential: bool = False,
    include_import_price_control: bool = False,
    include_gscpi_control: bool = True,
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
        include_wage_growth: Whether to include ULC wage Phillips curve (default True)
        include_exchange_rate: Whether to include TWI equation (default True)
        include_import_price: Whether to include import price pass-through (default True)
        include_participation: Whether to include participation rate equation (default True)
        include_hourly_coe: Whether to include hourly COE wage equation (default True)
        include_employment: Whether to include employment equation (default False)
        include_net_exports: Whether to include net exports equation (default False)

    Returns:
        PyMC Model ready for sampling (fixed constants stored on model._fixed_constants)

    """
    if nairu_const is None:
        nairu_const = DEFAULT_NAIRU_CONST.copy()
    if potential_const is None:
        potential_const = DEFAULT_POTENTIAL_CONST.copy()

    # Strip supply-side controls from Phillips curve if disabled
    strip_keys = set()
    if not include_import_price_control:
        strip_keys.add("Δ4ρm_1")
    if not include_gscpi_control:
        strip_keys.add("ξ_2")
    if strip_keys:
        obs = {k: v for k, v in obs.items() if k not in strip_keys}

    model = pm.Model()
    equations: list[Callable] = []

    # State equations
    nairu_func = nairu_student_t_equation if student_t_nairu else nairu_equation
    nairu = nairu_func(obs, model, constant=nairu_const)
    equations.append(nairu_func)
    potential_func = potential_output_skewnormal_equation if skewnormal_potential else potential_output_equation
    potential = potential_func(obs, model, constant=potential_const)
    equations.append(potential_func)

    # Observation equations
    okun_func = okun_gap_equation if okun_gap_form else okun_equation
    okun_func(obs, model, nairu, potential)
    equations.append(okun_func)
    if regime_switching:
        price_inflation_regime_equation(obs, model, nairu, constant=price_inflation_const)
        equations.append(price_inflation_regime_equation)
    else:
        price_inflation_equation(obs, model, nairu, constant=price_inflation_const)
        equations.append(price_inflation_equation)
    if include_wage_growth:
        if regime_switching:
            wage_growth_regime_equation(obs, model, nairu, constant=wage_growth_const)
            equations.append(wage_growth_regime_equation)
        else:
            wage_growth_equation(obs, model, nairu, constant=wage_growth_const)
            equations.append(wage_growth_equation)
    if include_is_curve:
        is_equation(obs, model, potential, constant=is_curve_const)
        equations.append(is_equation)

    # Hourly COE wage equation (optional, default on)
    if include_hourly_coe:
        if regime_switching:
            hourly_coe_regime_equation(obs, model, nairu, constant=hourly_coe_const)
            equations.append(hourly_coe_regime_equation)
        else:
            hourly_coe_equation(obs, model, nairu, constant=hourly_coe_const)
            equations.append(hourly_coe_equation)

    # Labour supply equation (optional)
    if include_participation:
        participation_equation(obs, model, nairu, constant=participation_const)
        equations.append(participation_equation)

    # Employment equation (optional) - labour demand with wage channel
    if include_employment:
        employment_equation(obs, model, potential, constant=employment_const)
        equations.append(employment_equation)

    # Open economy equations (optional)
    if include_exchange_rate:
        exchange_rate_equation(obs, model, constant=exchange_rate_const)
        equations.append(exchange_rate_equation)
    if include_import_price:
        import_price_equation(obs, model, constant=import_price_const)
        equations.append(import_price_equation)
    if include_net_exports:
        net_exports_equation(obs, model, potential, constant=net_exports_const)
        equations.append(net_exports_equation)

    print_model_equations(equations)

    # Print fixed constants
    constants = getattr(model, "_fixed_constants", {})
    if constants:
        print("\nFixed constants:")
        for name, value in constants.items():
            print(f"  {name} = {value}")

    # Store equations on model for dynamic plot generation
    model._equations = equations

    return model


# --- Save/Load Functions ---


def save_results(
    trace: az.InferenceData,
    obs: dict[str, np.ndarray],
    obs_index: pd.PeriodIndex,
    constants: dict[str, Any] | None = None,
    anchor_label: str | None = None,
    chart_obs: pd.DataFrame | None = None,
    model_kwargs: dict[str, Any] | None = None,
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
        chart_obs: Extended observations DataFrame for charting (may extend beyond sampling obs)
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
            "chart_obs": chart_obs,
            "model_kwargs": model_kwargs or {},
        }, f)
    print(f"Saved observations to: {obs_path}")

    return output_dir


# --- Main Entry Point ---


def run_stage1(
    start: str | None = "1980Q1",
    end: str | None = None,
    anchor_mode: AnchorMode = "rba",
    config: SamplerConfig | None = None,
    output_dir: Path | str | None = None,
    prefix: str = "nairu_output_gap",
    verbose: bool = False,
    model_kwargs: dict[str, Any] | None = None,
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
    obs, obs_index, anchor_label, chart_obs = build_observations(
        start=start, end=end, anchor_mode=anchor_mode, verbose=verbose
    )

    # Build model
    print("Building model...")
    model = build_model(obs, **(model_kwargs or {}))

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
        chart_obs=chart_obs,
        model_kwargs=model_kwargs,
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
