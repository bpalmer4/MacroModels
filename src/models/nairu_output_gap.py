"""NAIRU + Output Gap joint estimation model.

Bayesian state-space model that jointly estimates:
- NAIRU (Non-Accelerating Inflation Rate of Unemployment)
- Potential output (via Cobb-Douglas production function)
- Output gap and unemployment gap

Equations:
1. NAIRU: Gaussian random walk (state equation)
2. Potential output: Cobb-Douglas with time-varying drift (state equation)
3. Okun's Law: Links output gap to unemployment changes
4. Phillips Curve: Links inflation to unemployment gap
5. Wage Phillips Curve: Links wage growth to unemployment gap
6. IS Curve: Links output gap to real interest rate gap
7. Participation Rate: Links participation to unemployment gap (discouraged worker)
8. Exchange Rate: UIP-style TWI equation linking to interest rate differential
9. Import Price Pass-Through: Links import prices to TWI changes

Data sources:
- ABS: GDP, unemployment, CPI, hours worked, capital stock, MFP, import prices,
       participation rate
- RBA: Cash rate, inflation expectations, Trade-Weighted Index (TWI)

This module provides a unified interface to the two-stage estimation pipeline:
- Stage 1: Data preparation, model building, sampling, and saving
- Stage 2: Loading results and performing analysis/plotting
"""

import argparse
from pathlib import Path

from src.models.base import SamplerConfig

# Re-export key components for backwards compatibility
from src.data.observations import ALPHA, HMA_TERM, build_observations
from src.data import compute_r_star
from src.models.nairu_output_gap_stage1 import (
    build_model,
    run_stage1,
    save_results,
)
from src.models.nairu_output_gap_stage2 import (
    MODEL_NAME,
    NAIRUResults,
    RFOOTER_OUTPUT,
    load_results,
    plot_all,
    run_stage2,
    test_theoretical_expectations,
)

__all__ = [
    # Constants
    "ALPHA",
    "HMA_TERM",
    "MODEL_NAME",
    "RFOOTER_OUTPUT",
    # Stage 1
    "build_observations",
    "build_model",
    "compute_r_star",
    "save_results",
    "run_stage1",
    # Stage 2
    "NAIRUResults",
    "load_results",
    "test_theoretical_expectations",
    "plot_all",
    "run_stage2",
    # Unified
    "run_model",
    "main",
]


# --- Unified Entry Points ---


def run_model(
    start: str | None = "1980Q1",
    end: str | None = None,
    config: SamplerConfig | None = None,
    verbose: bool = False,
) -> NAIRUResults:
    """Run the full NAIRU + Output Gap estimation.

    This is a convenience function that runs Stage 1 (sampling) and returns
    a NAIRUResults container. For the full pipeline including analysis plots,
    use main() instead.

    Args:
        start: Start period
        end: End period
        config: Sampler configuration
        verbose: Print progress messages

    Returns:
        NAIRUResults with trace and computed series

    """
    if config is None:
        config = SamplerConfig()

    # Run stage 1 (without saving)
    from src.models.nairu_output_gap_stage1 import build_model as _build_model
    from src.models.nairu_output_gap_stage1 import build_observations as _build_obs
    from src.models.base import sample_model

    obs, obs_index = _build_obs(start=start, end=end, verbose=verbose)
    model = _build_model(obs)

    print("Sampling...")
    trace = sample_model(model, config)

    return NAIRUResults(
        trace=trace,
        obs=obs,
        obs_index=obs_index,
        model=model,
    )


# --- CLI Entry Point ---


def main(verbose: bool = False) -> None:
    """Run the full NAIRU + Output Gap estimation pipeline.

    This runs both stages:
    1. Stage 1: Build observations, sample model, save results
    2. Stage 2: Load results, run diagnostics, generate all plots
    """
    # Default output directory
    output_dir = Path(__file__).parent.parent.parent / "model_outputs"

    # Stage 1: Sample and save
    print("=" * 60)
    print("STAGE 1: Sampling")
    print("=" * 60)
    run_stage1(output_dir=output_dir, verbose=verbose)

    print("\n")
    print("=" * 60)
    print("STAGE 2: Analysis")
    print("=" * 60)

    # Stage 2: Load and analyze
    run_stage2(output_dir=output_dir, verbose=verbose)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run NAIRU + Output Gap model")
    parser.add_argument("-v", "--verbose", action="store_true", help="Print detailed output")
    args = parser.parse_args()
    main(verbose=args.verbose)
