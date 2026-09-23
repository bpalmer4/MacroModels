"""The ustar command line: its flags, and how a set of flags becomes a run.

Shared by the default run and by --compare, which defines each comparison
specification as the flags it would be run with. Kept out of `run.py` so the
comparison can use it without importing the entry point that imports it.
"""

import argparse

from src.models.common.cli import add_run_args, add_sampler_args
from src.models.ustar.analyse import run_analysis
from src.models.ustar.config import GAP_SOURCES, USTAR_STRUCTURES, ModelConfig
from src.models.ustar.estimate import run_estimate
from src.models.ystar.base import SamplerConfig


def build_parser() -> argparse.ArgumentParser:
    """Return the command-line parser, shared by the default run and --compare."""
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
    parser.add_argument("--ustar-structure", choices=USTAR_STRUCTURES,
                        default=ModelConfig.ustar_structure,
                        help=f"The structure imposed on u* (default: {ModelConfig.ustar_structure})")
    parser.add_argument("--knots", nargs="*", default=list(ModelConfig.spline_knots), metavar="DATE",
                        help="Interior knot dates for --ustar-structure spline, e.g. 2013Q1")
    parser.add_argument("--quadratic-gap", action="store_true",
                        help="Add delta x g x |g| to the Phillips curve, so wide gaps pull harder")
    parser.add_argument("--okun", action="store_true",
                        help="Include the Okun equation, which is off by default; see config")
    parser.add_argument("--sigma-ustar", type=float, default=0.020, help="Imposed u* innovation sd")
    parser.add_argument("--ustar-init", type=float, default=None,
                        help="Prior mean for u* in the first quarter (default: that quarter's unemployment rate)")
    parser.add_argument(
        "--ustar-drift", action="store_true",
        help="Let u* drift down while inflation expectations sit above target, "
             "instead of being a driftless random walk. See ModelConfig.ustar_drift",
    )
    parser.add_argument(
        "--lambda-prior-sd", type=float, default=0.1,
        help="Prior sd for lambda_ustar (default 0.1); widen to test whether it binds",
    )
    parser.add_argument(
        "--ustar-drift-end", default="2000Q1",
        help="Quarter from which the u* drift term is zero (default 2000Q1)",
    )
    parser.add_argument(
        "--free-sigma-ustar", action="store_true",
        help="Estimate the drift under a constrained prior instead of imposing it",
    )

    parser.add_argument(
        "--compare", action="store_true",
        help="Instead of the default run, re-estimate the comparison specifications that are "
             "not from today and chart them together; with --analyse-only, chart the saved "
             "runs as they stand. See MODEL_NOTES, 'Comparing specifications'",
    )
    add_sampler_args(parser)
    add_run_args(parser, prefix="ustar")
    return parser


def config_from_args(args: argparse.Namespace) -> ModelConfig:
    """Build the model configuration a set of parsed flags describes."""
    return ModelConfig(
        start=args.start,
        end=args.end,
        gap_source=args.gap_source,
        gap_prefix=args.gap_prefix,
        gap_measurement_error=not args.no_gap_error,
        use_output_gap=not args.no_output_gap,
        include_phillips=not args.no_phillips,
        include_okun=args.okun,
        quadratic_gap=args.quadratic_gap,
        ustar_structure=args.ustar_structure,
        spline_knots=tuple(args.knots),
        two_sided_beta=not args.one_sided_beta,
        sigma_ustar=args.sigma_ustar,
        ustar_init_mu=args.ustar_init,
        free_sigma_ustar=args.free_sigma_ustar,
        ustar_drift=args.ustar_drift,
        ustar_drift_end=args.ustar_drift_end,
        lambda_prior_sd=args.lambda_prior_sd,
    )


def run_from_args(args: argparse.Namespace) -> None:
    """Estimate and/or chart one run, as the parsed flags say."""
    if not args.analyse_only:
        sampler_config = SamplerConfig(draws=args.draws, tune=args.tune, chains=args.chains)
        run_estimate(
            config=config_from_args(args),
            sampler_config=sampler_config,
            prefix=args.prefix,
            verbose=args.verbose,
            seed=args.seed,
        )
    if not args.no_analyse:
        run_analysis(prefix=args.prefix)
