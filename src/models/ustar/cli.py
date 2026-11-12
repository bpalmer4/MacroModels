"""The ustar command line: its flags, and how a set of flags becomes a run.

Shared by the default run and by --compare, which defines each comparison
specification as the flags it would be run with. Kept out of `run.py` so the
comparison can use it without importing the entry point that imports it.
"""

import argparse
from collections.abc import Sequence

import pandas as pd

from src.models.common.cli import add_run_args, add_sampler_args
from src.models.ustar.analyse import run_analysis
from src.models.ustar.config import (
    FLAT_WALK_INIT_MU,
    FLAT_WALK_INIT_SD,
    FLAT_WALK_START,
    GAP_SOURCES,
    STEPPED_WALK_END,
    USTAR_STRUCTURES,
    ModelConfig,
)
from src.models.ustar.estimate import run_estimate
from src.models.ystar.base import SamplerConfig

# Where the named walks write unless --prefix is given, so neither overwrites the default run.
FLAT_WALK_PREFIX = "ustar_flat_walk"
STEPPED_WALK_PREFIX = "ustar_stepped_walk"


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
        "--no-inflation-lag", action="store_true",
        help="Drop backward-looking expectations (last quarter's inflation) from the Phillips curve. "
             "See ModelConfig.inflation_lag",
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
                        help="Prior mean for u* in the first quarter (default: that quarter's unemployment "
                             f"rate under decay, {ModelConfig.taper_init_mu:g} under taper)")
    parser.add_argument("--taper-sigma-early", type=float, default=ModelConfig.taper_sigma_early,
                        help=f"Taper: u* innovation sd at the start (default {ModelConfig.taper_sigma_early:g})")
    parser.add_argument("--taper-sigma-late", type=float, default=ModelConfig.taper_sigma_late,
                        help=f"Taper: u* innovation sd from --taper-end on (default {ModelConfig.taper_sigma_late:g})")
    parser.add_argument("--taper-end", default=ModelConfig.taper_end,
                        help=f"Taper: quarter the sd reaches its late value (default {ModelConfig.taper_end})")
    parser.add_argument("--taper-init-sd", type=float, default=ModelConfig.taper_init_sd,
                        help=f"Taper: prior sd on u* in the first quarter (default {ModelConfig.taper_init_sd:g})")
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

    walks = parser.add_mutually_exclusive_group()
    walks.add_argument(
        "--flat-walk", action="store_true",
        help=f"The random walk from {FLAT_WALK_START} with one flat step size, start prior "
             f"N({FLAT_WALK_INIT_MU:g}, {FLAT_WALK_INIT_SD:g}), prefix {FLAT_WALK_PREFIX}; "
             "flags given explicitly still win. See MODEL_NOTES, 'The flat walk'",
    )
    walks.add_argument(
        "--stepped-walk", action="store_true",
        help="The random walk with no taper: the early step size to --stepped-walk-end, the late one "
             f"after, prefix {STEPPED_WALK_PREFIX}. See MODEL_NOTES, 'The stepped walk'",
    )
    parser.add_argument("--stepped-walk-end", default=STEPPED_WALK_END, metavar="QUARTER",
                        help=f"Stepped walk: last quarter at the early step size (default {STEPPED_WALK_END})")

    parser.add_argument(
        "--compare", action="store_true",
        help="Instead of the default run, re-estimate the comparison specifications that are "
             "not from today and chart them together; with --analyse-only, chart the saved "
             "runs as they stand. See MODEL_NOTES, 'Comparing specifications'",
    )
    add_sampler_args(parser)
    add_run_args(parser, prefix="ustar")
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the flags, applying the --flat-walk or --stepped-walk settings as defaults.

    As defaults, so any flag given explicitly overrides them. The stepped walk only
    needs its own prefix; the model builds its schedule from the taper's step sizes.
    The flat walk drops the taper: the early step and the taper end are derived after
    parsing, from the late step and the start, so that overriding either carries
    through rather than leaving a one-quarter taper.
    """
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.stepped_walk:
        parser.set_defaults(prefix=STEPPED_WALK_PREFIX)
        return parser.parse_args(argv)
    if not args.flat_walk:
        return args
    parser.set_defaults(
        start=FLAT_WALK_START,
        ustar_init=FLAT_WALK_INIT_MU,
        taper_init_sd=FLAT_WALK_INIT_SD,
        taper_sigma_early=None,
        taper_end=None,
        prefix=FLAT_WALK_PREFIX,
    )
    args = parser.parse_args(argv)
    if args.taper_sigma_early is None:
        args.taper_sigma_early = args.taper_sigma_late
    if args.taper_end is None:
        args.taper_end = str(pd.Period(args.start, freq="Q") + 1)
    return args


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
        # The lag is a Phillips curve term, so it goes when the curve does.
        inflation_lag=not (args.no_inflation_lag or args.no_phillips),
        include_okun=args.okun,
        quadratic_gap=args.quadratic_gap,
        ustar_structure=args.ustar_structure,
        spline_knots=tuple(args.knots),
        two_sided_beta=not args.one_sided_beta,
        sigma_ustar=args.sigma_ustar,
        ustar_init_mu=args.ustar_init,
        taper_sigma_early=args.taper_sigma_early,
        taper_sigma_late=args.taper_sigma_late,
        taper_end=args.taper_end,
        taper_init_sd=args.taper_init_sd,
        stepped_walk=args.stepped_walk,
        stepped_walk_end=args.stepped_walk_end,
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
