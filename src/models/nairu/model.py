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
5. IS Curve: Links output gap to real interest rate gap
7. Participation Rate: Links participation to unemployment gap (discouraged worker)
8. Exchange Rate: UIP-style TWI equation linking to interest rate differential
9. Import Price Pass-Through: Links import prices to TWI changes

Data sources:
- ABS: GDP, unemployment, CPI, hours worked, capital stock, MFP, import prices,
       participation rate
- RBA: Cash rate, inflation expectations, Trade-Weighted Index (TWI)

This module provides a unified interface to the three-stage pipeline:
- Stage 1: Data preparation, model building, sampling, and saving
- Stage 2: Loading results and performing analysis/plotting
- Stage 3: Model-consistent forecasting with policy scenarios
"""

import argparse
from pathlib import Path

import arviz as az
import mgplot as mg
import pandas as pd

from src.data import compute_r_star

# Re-export key components for backwards compatibility
from src.data.observations import ANCHOR_LABELS, AnchorMode, HMA_TERM, build_observations
from src.models.nairu.base import SamplerConfig, sample_model
from src.models.nairu.stage1 import (
    MODEL_VARIANTS,
    build_model,
    run_stage1,
    save_results,
)
from src.models.nairu.stage2 import (
    MODEL_NAME,
    RFOOTER_OUTPUT,
    NAIRUResults,
    load_results,
    plot_all,
    run_stage2,
    test_theoretical_expectations,
)
from src.models.nairu.stage3 import (
    DEFAULT_POLICY_SCENARIOS,
    FORECAST_HORIZON,
    ForecastResults,
    forecast,
    run_scenarios,
    run_stage3,
)
from src.models.nairu.stage3_forward_sampling import (
    BayesianScenarioResults,
    run_stage3_bayesian,
)

__all__ = [
    # Variant configs
    "MODEL_VARIANTS",
    # Constants
    "HMA_TERM",
    "MODEL_NAME",
    "RFOOTER_OUTPUT",
    "FORECAST_HORIZON",
    "DEFAULT_POLICY_SCENARIOS",
    "ANCHOR_LABELS",
    # Types
    "AnchorMode",
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
    # Stage 3 (deterministic)
    "ForecastResults",
    "forecast",
    "run_scenarios",
    "run_stage3",
    # Stage 3 (Bayesian)
    "BayesianScenarioResults",
    "run_stage3_bayesian",
    # Unified
    "run_model",
    "main",
]


# --- Unified Entry Points ---


def run_model(
    start: str | None = "1980Q1",
    end: str | None = None,
    anchor_mode: AnchorMode = "rba",
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
        anchor_mode: How to anchor expectations
            - "expectations": Use full estimated expectations series
            - "target": Phase from expectations to 2.5% target (1993-1998)
            - "rba": Use RBA PIE_RBAQ with same phase-in to target
        config: Sampler configuration
        verbose: Print progress messages

    Returns:
        NAIRUResults with trace and computed series

    """
    if config is None:
        config = SamplerConfig()

    # Run stage 1 (without saving)
    obs, obs_index, anchor_label, chart_obs = build_observations(
        start=start, end=end, anchor_mode=anchor_mode, verbose=verbose
    )
    model = build_model(obs)

    print("Sampling...")
    trace = sample_model(model, config)

    return NAIRUResults(
        trace=trace,
        obs=obs,
        obs_index=obs_index,
        model=model,
        anchor_label=anchor_label,
        chart_obs=chart_obs,
    )


# --- CLI Entry Point ---


def _plot_nairu_comparison(
    output_dir: Path,
    variants: list[str],
    chart_dir: str = "charts/nairu_comparison",
) -> None:
    """Plot NAIRU comparison chart with upper/lower medians and fill_between."""
    import pickle

    from src.models.common.extraction import get_vector_var

    start = pd.Period("2000Q1", freq="Q")
    colors = {"simple": "darkblue", "complex": "darkred"}
    labels = {"simple": "Simple Model", "complex": "Complex Model"}

    medians: dict[str, pd.Series] = {}
    for name in variants:
        prefix = f"nairu_{name}"
        trace_path = output_dir / f"{prefix}_trace.nc"
        obs_path = output_dir / f"{prefix}_obs.pkl"

        trace = az.from_netcdf(str(trace_path))
        with open(obs_path, "rb") as f:
            data = pickle.load(f)
        obs_index = data["obs_index"]

        samples = get_vector_var("nairu", trace)
        samples.index = obs_index
        median = samples.quantile(q=0.5, axis=1)
        median = median[median.index >= start]
        medians[name] = median

    # Plot
    ax = None
    for name in variants:
        s = medians[name]
        s.name = labels[name]
        ax = mg.line_plot(s, ax=ax, color=colors[name], width=1.5, annotate=True, zorder=4)

    if ax is not None and len(medians) == 2:
        keys = list(medians.keys())
        idx = medians[keys[0]].index.intersection(medians[keys[1]].index)
        band = pd.DataFrame(
            {
                "lower": medians[keys[0]][idx].clip(upper=medians[keys[1]][idx]),
                "upper": medians[keys[0]][idx].clip(lower=medians[keys[1]][idx]),
            },
            index=idx,
        )
        mg.fill_between_plot(band, ax=ax, color="purple", alpha=0.15, label="NAIRU range")

    if ax is not None:
        # Unemployment overlay
        with open(output_dir / f"nairu_{variants[0]}_obs.pkl", "rb") as f:
            data = pickle.load(f)
        U = pd.Series(data["obs"]["U"], index=data["obs_index"])
        U = U[U.index >= start]
        U.name = "Unemployment Rate"
        mg.line_plot(U, ax=ax, color="brown", width=1.0, zorder=3)

        chart_path = Path(chart_dir)
        chart_path.mkdir(parents=True, exist_ok=True)
        mg.set_chart_dir(str(chart_path))
        mg.finalise_plot(
            ax,
            title="NAIRU Estimate Range for Australia",
            ylabel="Per cent",
            legend={"loc": "best", "fontsize": "x-small"},
            lfooter="Australia. Simple: core Phillips/Okun/IS. Complex: adds regime-switching, open economy, labour market.",
            rfooter="Joint NAIRU + Output Gap Model",
            axisbelow=True,
        )
        print(f"\nComparison chart saved to: {chart_path}")


def _run_pipeline(
    anchor_mode: AnchorMode,
    verbose: bool,
    skip_forecast: bool,
    output_dir: Path,
    prefix: str = "nairu_output_gap",
    chart_dir: str | None = None,
    model_kwargs: dict | None = None,
    label: str | None = None,
) -> None:
    """Run the three-stage pipeline for a single model configuration."""
    tag = f" [{label}]" if label else ""

    # Stage 1: Sample and save
    print("=" * 60)
    print(f"STAGE 1: Sampling{tag}")
    print("=" * 60)
    run_stage1(
        anchor_mode=anchor_mode,
        output_dir=output_dir,
        prefix=prefix,
        verbose=verbose,
        model_kwargs=model_kwargs,
    )

    print("\n")
    print("=" * 60)
    print(f"STAGE 2: Analysis{tag}")
    print("=" * 60)

    # Stage 2: Load and analyze
    run_stage2(output_dir=output_dir, prefix=prefix, chart_dir=chart_dir, verbose=verbose)

    if not skip_forecast:
        print("\n")
        print("=" * 60)
        print(f"STAGE 3a: Scenario Analysis (Deterministic){tag}")
        print("=" * 60)

        # Stage 3a: Deterministic scenario analysis
        run_stage3(output_dir=output_dir, prefix=prefix, chart_dir=chart_dir, verbose=verbose)

        print("\n")
        print("=" * 60)
        print(f"STAGE 3b: Scenario Analysis (Bayesian){tag}")
        print("=" * 60)

        # Stage 3b: Bayesian forward sampling
        run_stage3_bayesian(output_dir=output_dir, prefix=prefix, chart_dir=chart_dir, verbose=verbose)


def main(
    anchor_mode: AnchorMode = "rba",
    verbose: bool = False,
    skip_forecast: bool = False,
    variant: str = "default",
) -> None:
    """Run the full NAIRU + Output Gap estimation pipeline.

    This runs all three stages:
    1. Stage 1: Build observations, sample model, save results
    2. Stage 2: Load results, run diagnostics, generate all plots
    3. Stage 3: Model-consistent forecasting with policy scenarios

    Args:
        anchor_mode: How to anchor expectations
            - "expectations": Use full estimated expectations series
            - "target": Phase from expectations to 2.5% target (1993-1998)
            - "rba": Use RBA PIE_RBAQ with same phase-in to target
        verbose: Print detailed output
        skip_forecast: Skip Stage 3 (scenario analysis)
        variant: Model variant to run ("default", "upper", "lower", or "both")
    """
    output_dir = Path(__file__).parent.parent.parent.parent / "model_outputs"

    if variant == "both":
        variants = ["simple", "complex"]
    elif variant in MODEL_VARIANTS:
        variants = [variant]
    else:
        # default: run with no overrides
        _run_pipeline(
            anchor_mode=anchor_mode,
            verbose=verbose,
            skip_forecast=skip_forecast,
            output_dir=output_dir,
        )
        return

    for name in variants:
        print("\n" + "#" * 60)
        print(f"# VARIANT: {name.upper()}")
        print("#" * 60 + "\n")
        _run_pipeline(
            anchor_mode=anchor_mode,
            verbose=verbose,
            skip_forecast=skip_forecast,
            output_dir=output_dir,
            prefix=f"nairu_{name}",
            chart_dir=f"charts/nairu_{name}",
            model_kwargs=MODEL_VARIANTS[name],
            label=name,
        )

    # Comparison chart when both variants have been run
    if len(variants) == 2:
        print("\n" + "#" * 60)
        print("# COMPARISON CHART")
        print("#" * 60 + "\n")
        _plot_nairu_comparison(output_dir, variants)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run NAIRU + Output Gap model")
    parser.add_argument("-v", "--verbose", action="store_true", help="Print detailed output")
    parser.add_argument(
        "--skip-forecast",
        action="store_true",
        help="Skip Stage 3 (scenario analysis)",
    )
    parser.add_argument(
        "--anchor",
        type=str,
        choices=["expectations", "target", "rba"],
        default="rba",
        help="Expectations anchor mode: 'rba' (default), 'target' (model series phased), or 'expectations' (full model series)",
    )
    parser.add_argument(
        "--variant",
        type=str,
        choices=["default", "simple", "complex", "both"],
        default="default",
        help="Model variant: 'simple' (core equations), 'complex' (all features), 'both', or 'default'",
    )
    args = parser.parse_args()
    main(
        anchor_mode=args.anchor,
        verbose=args.verbose,
        skip_forecast=args.skip_forecast,
        variant=args.variant,
    )
