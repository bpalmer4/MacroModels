"""Command-line entry point for the rstar_qpm model."""

import argparse

from src.models.common.cli import add_run_args, add_sampler_args
from src.models.rstar_qpm.analyse import run_analysis
from src.models.rstar_qpm.config import ModelConfig
from src.models.rstar_qpm.estimate import load_results, run_estimate
from src.models.rstar_qpm.recovery import run_recovery
from src.models.ystar.base import SamplerConfig


def _sigma_w(arg: str | None, defaults: ModelConfig) -> float | None:
    """Read --sigma-w: absent keeps the default, 'free' estimates, a number imposes."""
    if arg is None:
        return defaults.sigma_w_fixed
    if arg.lower() == "free":
        return None
    return float(arg)


def main() -> None:
    """Estimate the semi-structural model, chart it, and optionally test recovery."""
    defaults = ModelConfig()
    parser = argparse.ArgumentParser(
        description="Trend r* and short-run neutral from a semi-structural open-economy model",
    )
    parser.add_argument("--start", default=defaults.start)
    parser.add_argument("--end", default=None)
    parser.add_argument(
        "--horizon", type=int, default=defaults.horizon,
        help=f"Quarters ahead at which short-run neutral closes the gap (default {defaults.horizon})",
    )
    parser.add_argument(
        "--state-draws", type=int, default=defaults.state_draws,
        help="Posterior draws pushed through the simulation smoother",
    )
    parser.add_argument(
        "--no-forward", action="store_true",
        help="Drop the 5y5y forward, so trend r*'s level must come from the rest of the system. "
             "Use with --prefix so the default run is not overwritten",
    )
    parser.add_argument(
        "--no-is", action="store_true",
        help="Switch the IS curve off: rates no longer move the output gap (b2, b3, b4 fixed "
             "at zero). No short-run neutral. Use with --prefix",
    )
    parser.add_argument(
        "--sigma-w", default=None, metavar="VALUE|free",
        help=f"The wedge's innovation sd: imposed at {defaults.sigma_w_fixed} by default, another "
             "value to impose that, or 'free' to estimate it. Use with --prefix",
    )
    parser.add_argument(
        "--recovery", action="store_true",
        help="After the run, simulate economies at known parameters and re-estimate: "
             "the test of whether the data can find transmission at all",
    )
    add_sampler_args(parser, draws=1_000, tune=1_000)
    add_run_args(parser, prefix="rstar_qpm")
    args = parser.parse_args()

    config = ModelConfig(start=args.start, end=args.end, horizon=args.horizon,
                         state_draws=args.state_draws, use_forward=not args.no_forward,
                         use_is=not args.no_is, sigma_w_fixed=_sigma_w(args.sigma_w, defaults))
    sampler_config = SamplerConfig(draws=args.draws, tune=args.tune, chains=args.chains)
    if args.seed is not None:
        sampler_config.random_seed = args.seed

    if not args.analyse_only:
        run_estimate(config=config, sampler_config=sampler_config, prefix=args.prefix, verbose=args.verbose)
    if not args.no_analyse:
        run_analysis(prefix=args.prefix)
    if args.recovery:
        trace, frame, _, _ = load_results(prefix=args.prefix)
        run_recovery(trace, frame, config, sampler_config)


if __name__ == "__main__":
    main()
