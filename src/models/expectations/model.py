"""Inflation expectations signal extraction model.

Extracts latent inflation expectations from multiple survey and market-based
measures, following the approach in Cusbert (2017).

This module provides a unified interface to run both stages:
- Stage 1: Model building and MCMC sampling
- Stage 2: Diagnostics and plotting

Run with: uv run python -m src.models.expectations.model

Reference:
    Cusbert T (2017), "Estimating the NAIRU and the Unemployment Gap",
    RBA Bulletin, June, pp 13-22.
"""

from src.models.expectations.common import (
    DEFAULT_CHAINS,
    DEFAULT_DRAWS,
    DEFAULT_TUNE,
    MODEL_NAMES,
    MODEL_TYPES,
)
from src.models.expectations.stage1 import run_model, save_results
from src.models.expectations.stage2 import (
    ExpectationsResults,
    generate_plots,
    load_all_results,
    load_results,
    run_diagnostics,
)

# Re-export for backwards compatibility
__all__ = [
    "run_model",
    "save_results",
    "load_results",
    "load_all_results",
    "run_diagnostics",
    "generate_plots",
    "ExpectationsResults",
    "MODEL_NAMES",
    "MODEL_TYPES",
]


if __name__ == "__main__":
    import argparse

    from src.models.common.diagnostics import check_model_diagnostics

    parser = argparse.ArgumentParser(description="Run expectations model (stages 1 & 2)")
    parser.add_argument("--start", default="1983Q1", help="Start period")
    parser.add_argument("--draws", type=int, default=DEFAULT_DRAWS)
    parser.add_argument("--tune", type=int, default=DEFAULT_TUNE)
    parser.add_argument("--chains", type=int, default=DEFAULT_CHAINS)
    parser.add_argument("-q", "--quiet", action="store_true", help="Suppress progress bar")
    parser.add_argument("--no-plots", action="store_true", help="Skip plot generation")

    args = parser.parse_args()

    # === STAGE 1: SAMPLING ===
    print("=" * 60)
    print("STAGE 1: SAMPLING")
    print("=" * 60)

    for model_type in MODEL_TYPES:
        print("\n" + "=" * 60)
        print(f"Running {MODEL_NAMES[model_type]} model")
        print("=" * 60)

        trace, measures, inflation, index = run_model(
            model_type=model_type,
            start=args.start,
            draws=args.draws,
            tune=args.tune,
            chains=args.chains,
            verbose=not args.quiet,
        )

        print(f"\nMCMC Diagnostics ({MODEL_NAMES[model_type]}):")
        check_model_diagnostics(trace)

        print(f"\nSaving {model_type} results...")
        save_results(model_type, trace, measures, inflation, index)

    # === STAGE 2: DIAGNOSTICS & PLOTS ===
    print("\n" + "=" * 60)
    print("STAGE 2: DIAGNOSTICS & PLOTS")
    print("=" * 60)

    all_results = load_all_results()

    for model_type, results in all_results.items():
        run_diagnostics(results)

    if not args.no_plots:
        print("\n" + "=" * 60)
        print("GENERATING PLOTS")
        print("=" * 60)
        generate_plots(all_results)

    print("\n" + "=" * 60)
    print("COMPLETE")
    print("=" * 60)
