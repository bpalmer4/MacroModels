"""Run the HLW Bayesian r-star estimation pipeline.

Usage:
    uv run python -m src.models.rstar.run
    uv run python -m src.models.rstar.run -v
    uv run python -m src.models.rstar.run --estimate-only
    uv run python -m src.models.rstar.run --skip-estimate
"""

import argparse

def main(
    verbose: bool = False,
    estimate: bool = True,
    analyse: bool = True,
    start: str = "1980Q1",
    end: str | None = None,
    resolution: str = "C",
) -> None:
    """Run estimation and/or analysis stages."""
    prefix = f"rstar_hlw_{resolution}"

    if estimate:
        print("=" * 60)
        print(f"ESTIMATE [HLW r-star, Resolution {resolution}, start={start}]")
        print("=" * 60)
        from src.models.rstar.estimate import run_estimate  # noqa: PLC0415
        run_estimate(
            start=start, end=end, verbose=verbose,
            prefix=prefix, resolution=resolution,
        )
        print()

    if analyse:
        print("=" * 60)
        print(f"ANALYSE [HLW r-star, Resolution {resolution}]")
        print("=" * 60)
        from src.models.rstar.analyse import run_analyse  # noqa: PLC0415
        run_analyse(prefix=prefix, resolution=resolution, verbose=verbose)
        print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run HLW Bayesian r-star model")
    parser.add_argument("-v", "--verbose", action="store_true", help="Print detailed output")
    parser.add_argument(
        "--estimate-only",
        action="store_true",
        help="Only run estimation (skip analysis)",
    )
    parser.add_argument(
        "--skip-estimate",
        action="store_true",
        help="Skip estimation (use saved results for analysis)",
    )
    parser.add_argument(
        "--start",
        type=str,
        default="1980Q1",
        help="Sample start period (default 1980Q1; try 1993Q1 to drop pre-target era)",
    )
    parser.add_argument(
        "--end",
        type=str,
        default=None,
        help="Sample end period (default: latest available)",
    )
    parser.add_argument(
        "--resolution",
        type=str,
        choices=["A", "B", "C"],
        default="C",
        help="r* identity: A (canonical HLW, r* = g + z), B (canonical + indexed bond observation), or C (blend, default)",
    )
    args = parser.parse_args()
    main(
        verbose=args.verbose,
        estimate=not args.skip_estimate,
        analyse=not args.estimate_only,
        start=args.start,
        end=args.end,
        resolution=args.resolution,
    )
