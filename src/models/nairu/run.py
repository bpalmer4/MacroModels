"""Run the NAIRU + Output Gap estimation pipeline.

Usage:
    uv run python -m src.models.nairu.run -v --variant complex
    uv run python -m src.models.nairu.run -v --variant simple complex
    uv run python -m src.models.nairu.run --estimate-only --variant simple
"""

import argparse
from pathlib import Path

from src.data.observations import AnchorMode
from src.models.nairu.config import PRESETS, ModelConfig


# Default output directory
DEFAULT_OUTPUT_DIR = Path(__file__).parent.parent.parent.parent / "model_outputs"


def _run_variant(
    config: ModelConfig,
    anchor_mode: AnchorMode = "rba",
    verbose: bool = False,
    estimate: bool = True,
    validate: bool = True,
    analyse: bool = True,
    forecast: bool = True,
) -> None:
    """Run the pipeline for a single model variant."""
    label = config.label
    prefix = f"nairu_{label}"

    if estimate:
        print("=" * 60)
        print(f"ESTIMATE [{label}]")
        print("=" * 60)

        from src.models.nairu.estimate import run_estimate

        run_estimate(
            anchor_mode=anchor_mode,
            config=config,
            prefix=prefix,
            verbose=verbose,
        )
        print()

    if validate:
        print("=" * 60)
        print(f"VALIDATE [{label}]")
        print("=" * 60)

        from src.models.nairu.validate import run_validate

        run_validate(prefix=prefix, verbose=verbose)
        print()

    if analyse:
        print("=" * 60)
        print(f"ANALYSE [{label}]")
        print("=" * 60)

        from src.models.nairu.analyse import run_analyse

        run_analyse(prefix=prefix, verbose=verbose)
        print()

    if forecast:
        print("=" * 60)
        print(f"FORECAST [{label}]")
        print("=" * 60)

        from src.models.nairu.forecast import run_forecast

        run_forecast(prefix=prefix, verbose=verbose)
        print()


def main(
    anchor_mode: AnchorMode = "rba",
    verbose: bool = False,
    variants: list[str] | None = None,
    estimate_only: bool = False,
    skip_estimate: bool = False,
    skip_forecast: bool = False,
) -> None:
    """Run the NAIRU + Output Gap pipeline.

    Args:
        anchor_mode: Expectations anchor mode
        verbose: Print detailed output
        variants: Preset name(s) to run
        estimate_only: Only run estimation (skip validate/analyse/forecast)
        skip_estimate: Skip estimation (use saved results)
        skip_forecast: Skip forecast scenarios

    """
    if variants is None:
        variants = ["default"]
    variants = sorted(set(variants))

    for name in variants:
        if name not in PRESETS:
            print(f"Unknown variant: {name}")
            print(f"Available: {', '.join(PRESETS.keys())}")
            return

        config = PRESETS[name]

        if len(variants) > 1:
            print("\n" + "#" * 60)
            print(f"# VARIANT: {name.upper()}")
            print("#" * 60 + "\n")

        _run_variant(
            config=config,
            anchor_mode=anchor_mode,
            verbose=verbose,
            estimate=not skip_estimate,
            validate=not estimate_only,
            analyse=not estimate_only,
            forecast=not estimate_only and not skip_forecast,
        )

    if len(variants) > 1 and not estimate_only:
        from src.models.nairu.analysis import plot_nairu_comparison, plot_output_gap_comparison
        from src.models.nairu.results import load_results

        print("\n" + "#" * 60)
        print("# COMPARISON CHARTS")
        print("#" * 60 + "\n")

        all_results = [
            load_results(prefix=f"nairu_{name}", rebuild_model=False)
            for name in variants
        ]
        plot_nairu_comparison(all_results)
        plot_output_gap_comparison(all_results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run NAIRU + Output Gap model")
    parser.add_argument("-v", "--verbose", action="store_true", help="Print detailed output")
    parser.add_argument(
        "--anchor",
        type=str,
        choices=["expectations", "target", "rba"],
        default="rba",
        help="Expectations anchor mode (default: rba)",
    )
    parser.add_argument(
        "--variant",
        type=str,
        nargs="+",
        choices=list(PRESETS.keys()),
        default=["default"],
        help="Model variant(s) to run (default: default)",
    )
    parser.add_argument(
        "--estimate-only",
        action="store_true",
        help="Only run estimation (skip validate/analyse)",
    )
    parser.add_argument(
        "--skip-estimate",
        action="store_true",
        help="Skip estimation (use saved results for validate/analyse)",
    )
    parser.add_argument(
        "--skip-forecast",
        action="store_true",
        help="Skip forecast scenarios",
    )
    args = parser.parse_args()
    main(
        anchor_mode=args.anchor,
        verbose=args.verbose,
        variants=args.variant,
        estimate_only=args.estimate_only,
        skip_estimate=args.skip_estimate,
        skip_forecast=args.skip_forecast,
    )
