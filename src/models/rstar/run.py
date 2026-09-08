"""Command-line entry point for the rstar model."""

import argparse

from src.models.rstar.analyse import run_analysis
from src.models.rstar.config import WORLD_SOURCES, ModelConfig
from src.models.rstar.estimate import run_estimate
from src.models.ystar.base import SamplerConfig


def main() -> None:
    """Estimate r*, then chart it."""
    parser = argparse.ArgumentParser(description="Estimate r* from the bond market")
    parser.add_argument("--start", default="1986Q3", help="Sample start (default 1986Q3)")
    parser.add_argument("--end", default=None, help="Sample end (default: latest)")
    parser.add_argument(
        "--world-source", default="mean", choices=list(WORLD_SOURCES),
        help="Which published r* anchors the state (default: mean of the three)",
    )
    parser.add_argument(
        "--no-world", action="store_true",
        help="Diagnostic: drop the world equation, leaving one observable for two components",
    )
    parser.add_argument(
        "--breaks", nargs="*", default=None,
        help="Quarters at which the wedge may jump (default: the five in ModelConfig)",
    )
    parser.add_argument(
        "--steps", action="store_true",
        help="Use the asserted step-break wedge instead of the free Student-t walk",
    )
    parser.add_argument(
        "--input-source", default="joint", choices=["joint", "separate"],
        help="Where the Taylor rule's inputs come from: one joint ystar_ustar run "
             "(default) or separate ystar and ustar runs",
    )
    parser.add_argument("--sigma-walk", type=float, default=0.08, help="Imposed wedge innovation sd")
    parser.add_argument(
        "--nu-walk", type=float, default=None,
        help="fix the StudentT degrees of freedom instead of estimating them "
             "(removes the model's one funnel; see ModelConfig.nu_walk)",
    )
    parser.add_argument(
        "--wedge-drift", type=float, default=0.0,
        help="Background drift between breaks (0 = pure step function)",
    )

    parser.add_argument("--anchor", type=float, default=2.5, help="Inflation target for the Taylor rule")
    parser.add_argument("--rule-pi", type=float, default=0.125, help="Rule coefficient on the inflation gap")
    parser.add_argument("--rule-gap", type=float, default=0.125, help="Rule coefficient on the output gap")
    parser.add_argument(
        "--taylor-ugap", action="store_true",
        help="Use the ustar unemployment gap in the rule instead of the ystar output gap",
    )
    parser.add_argument(
        "--no-look-through", action="store_true",
        help="Respond to headline inflation rather than looking through supply shocks",
    )
    parser.add_argument(
        "--supply-symmetric", action="store_true",
        help="Look through supply shocks in both directions, not only positive ones",
    )

    parser.add_argument("--draws", type=int, default=2_000)
    parser.add_argument("--tune", type=int, default=2_000)
    parser.add_argument("--chains", type=int, default=4)
    parser.add_argument("--seed", type=int, default=None)

    parser.add_argument("--prefix", default="rstar", help="Output filename prefix")
    parser.add_argument("--analyse-only", action="store_true", help="Skip estimation")
    parser.add_argument("--no-analyse", action="store_true", help="Estimate without charting")
    parser.add_argument("-v", "--verbose", action="store_true", help="Print detailed output")
    args = parser.parse_args()

    if not args.analyse_only:
        config = ModelConfig(
            start=args.start,
            end=args.end,
            world_source=args.world_source,
            use_world=not args.no_world,
            wedge_drift=args.wedge_drift,
            free_wedge=not args.steps,
            input_source=args.input_source,
            sigma_walk=args.sigma_walk,
            nu_walk=args.nu_walk,
            anchor=args.anchor,
            rule_pi=args.rule_pi,
            rule_gap=args.rule_gap,
            taylor_use_ugap=args.taylor_ugap,
            look_through_supply=not args.no_look_through,
            supply_positive_only=not args.supply_symmetric,
        )
        # Set after construction rather than unpacked into the call: the default
        # is a meaningful set of dates, so an absent --breaks must leave it
        # alone, and a conditional `**{...}` inside the call is untypeable.
        if args.breaks:
            config.break_quarters = tuple(args.breaks)
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
