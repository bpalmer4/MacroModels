"""Command-line entry point for the ustar model."""

import argparse

from src.models.ustar.analyse import run_analysis
from src.models.ustar.config import GAP_SOURCES, ModelConfig
from src.models.ustar.estimate import run_estimate
from src.models.ystar.base import SamplerConfig


def main() -> None:
    """Estimate u*, then chart it."""
    parser = argparse.ArgumentParser(description="Estimate u* from a given output gap")
    parser.add_argument("--start", default="1993Q1", help="Sample start (default 1993Q1)")
    parser.add_argument("--end", default=None, help="Sample end (default: latest)")
    parser.add_argument(
        "--gap-source", default="defined", choices=list(GAP_SOURCES),
        help="Which ystar gap feeds Okun (default: defined)",
    )
    parser.add_argument(
        "--gap-prefix", default="ystar",
        help="Filename prefix of the ystar run to read (default: ystar)",
    )
    parser.add_argument(
        "--no-gap-error", action="store_true",
        help="Treat the gap's mean path as known, understating the u* band",
    )
    parser.add_argument(
        "--no-output-gap", action="store_true",
        help="Diagnostic: zero the output gap, keeping the Okun equation's structure",
    )
    parser.add_argument(
        "--no-phillips", action="store_true",
        help="Okun only: drop the Phillips curve and the expectations dependency",
    )
    parser.add_argument(
        "--one-sided-beta", action="store_true",
        help="Truncate beta at zero, asserting Okun's law rather than testing it",
    )
    parser.add_argument("--sigma-ustar", type=float, default=0.040, help="Imposed u* innovation sd")
    parser.add_argument(
        "--free-sigma-ustar", action="store_true",
        help="Estimate the drift under a constrained prior instead of imposing it",
    )

    parser.add_argument("--draws", type=int, default=2_000)
    parser.add_argument("--tune", type=int, default=2_000)
    parser.add_argument("--chains", type=int, default=4)
    parser.add_argument("--seed", type=int, default=None)

    parser.add_argument("--prefix", default="ustar", help="Output filename prefix")
    parser.add_argument("--analyse-only", action="store_true", help="Skip estimation")
    parser.add_argument("--no-analyse", action="store_true", help="Estimate without charting")
    parser.add_argument("-v", "--verbose", action="store_true", help="Print detailed output")
    args = parser.parse_args()

    if not args.analyse_only:
        config = ModelConfig(
            start=args.start,
            end=args.end,
            gap_source=args.gap_source,
            gap_prefix=args.gap_prefix,
            gap_measurement_error=not args.no_gap_error,
            use_output_gap=not args.no_output_gap,
            include_phillips=not args.no_phillips,
            two_sided_beta=not args.one_sided_beta,
            sigma_ustar=args.sigma_ustar,
            free_sigma_ustar=args.free_sigma_ustar,
        )
        sampler_config = SamplerConfig(draws=args.draws, tune=args.tune, chains=args.chains)
        run_estimate(
            config=config,
            sampler_config=sampler_config,
            prefix=args.prefix,
            verbose=args.verbose,
            seed=args.seed,
        )

    if not args.no_analyse:
        run_analysis(prefix=args.prefix)


if __name__ == "__main__":
    main()
