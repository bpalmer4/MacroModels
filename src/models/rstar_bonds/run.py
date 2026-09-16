"""Command-line entry point for the rstar model."""

import argparse

from src.models.rstar_bonds.analyse import run_analysis
from src.models.rstar_bonds.config import (
    AU_PREMIUM_SOURCES,
    DEFLATORS,
    SHORT_RATES,
    US_PREMIUM_SOURCES,
    WORLD_SOURCES,
    ModelConfig,
)
from src.models.rstar_bonds.estimate import run_estimate
from src.models.ystar.base import SamplerConfig


def main() -> None:
    """Estimate r*, then chart it."""
    parser = argparse.ArgumentParser(description="Estimate r* from the bond market")
    parser.add_argument("--start", default="1993Q1", help="Sample start (default 1993Q1)")
    parser.add_argument("--end", default=None, help="Sample end (default: latest)")
    parser.add_argument(
        "--world-source", default="market", choices=list(WORLD_SOURCES),
        help="What anchors the state (default: market, the Cleveland Fed 10y expected "
             "real rate LESS the published US term premium, so both sides of the "
             "comparison are premium-free). 'cleveland' is the raw yield; mean, US, "
             "Euro Area and Canada are HLW model estimates",
    )
    parser.add_argument(
        "--no-world", action="store_true",
        help="Diagnostic: drop the world equation, leaving one observable for two components",
    )
    parser.add_argument(
        "--no-short", action="store_true",
        help="Drop the real cash rate equation, leaving the one-window model in which "
             "mu_tp carries the level on its own",
    )
    parser.add_argument(
        "--deflator", default="expectations", choices=list(DEFLATORS),
        help="Inflation series used to make the cash rate real (default: expectations, "
             "matching is_curve)",
    )
    parser.add_argument(
        "--short-rate", default="cash", choices=list(SHORT_RATES),
        help="Which short rate the second window uses: the overnight cash rate (default, "
             "risk-free) or the 90-day bank bill, which prices the expected policy path "
             "but carries a bank credit spread with it (see ModelConfig.short_rate)",
    )
    parser.add_argument(
        "--us-premium", action="store_true",
        help="Pin the term premium to the published US one and estimate only "
             "the Australian spread over it, instead of letting the level rest on mu_tp",
    )
    parser.add_argument(
        "--premium-source", default="kim-wright", choices=list(US_PREMIUM_SOURCES),
        help="Whose US term premium to subtract, in BOTH --world-source market and "
             "--us-premium (default kim-wright, the Fed Board three-factor model; "
             "acm is Adrian-Crump-Moench from the NY Fed)",
    )
    parser.add_argument(
        "--au-premium", action=argparse.BooleanOptionalAction, default=True,
        help="Pin the term premium to the AOFM's published AUSTRALIAN one and estimate "
             "only the real-nominal spread (default on). --no-au-premium restores the free "
             "mu_tp version, whose flat premium the AOFM series contradicts",
    )
    parser.add_argument(
        "--au-premium-source", default="bc", choices=list(AU_PREMIUM_SOURCES),
        help="Which AOFM decomposition, for both --au-premium and --nominal-window "
             "(default bc, bias-corrected; ols is plain ACM, the estimator this package "
             "already rejected for the US)",
    )
    parser.add_argument(
        "--nominal-window", action="store_true",
        help="Read window one off the AOFM risk-neutral nominal yield, deflated, instead "
             "of the indexed real yield. Removes the term premium from the model entirely, "
             "and with it the wedge_0/mu_tp trade-off that is the level problem",
    )
    parser.add_argument(
        "--forward", action=argparse.BooleanOptionalAction, default=ModelConfig().use_forward,
        help="Add the AOFM 5y5y risk-neutral forward, deflated, as a third window loading "
             "directly on r*. The only observable here that speaks to the LEVEL",
    )
    parser.add_argument(
        "--impose-world-loading", action=argparse.BooleanOptionalAction, default=True,
        help="Impose one-for-one pass-through of world r* rather than estimating it "
             "(default on). Free, it is not identified against the wedge once the premium "
             "is pinned: it collapses to 0.015. --no-impose-world-loading estimates it",
    )
    parser.add_argument(
        "--curve", action="store_true",
        help="Add a medium-maturity real CGS yield as a third window. Off by default: "
             "it identifies the premium curve's slope, at the cost of r* absorbing the "
             "policy stance (see ModelConfig.use_curve)",
    )
    parser.add_argument(
        "--curve-maturity", type=int, default=3,
        help="Maturity in years for the third window (default 3)",
    )
    parser.add_argument(
        "--assert-stance", action="store_true",
        help="Assert the average policy stance and report the implied term premium, "
             "instead of asserting mu_tp and reporting the implied stance",
    )
    parser.add_argument(
        "--mu-g-sigma", type=float, default=0.5,
        help="Prior sd on the asserted stance (only used with --assert-stance)",
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
    parser.add_argument(
        "--sigma-walk", type=float, default=0.12,
        help="Imposed wedge innovation sd (default 0.12; at 0.08 the estimated nu falls to "
             "3.64, inside the infinite-kurtosis regime, which is the model straining)",
    )
    parser.add_argument(
        "--nu-walk", type=float, default=ModelConfig().nu_walk,
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

    parser.add_argument("--prefix", default="rstar_bonds", help="Output filename prefix")
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
            free_world_loading=not args.impose_world_loading,
            use_short=not args.no_short,
            deflator=args.deflator,
            short_rate=args.short_rate,
            us_premium_anchor=args.us_premium,
            us_premium_source=args.premium_source,
            au_premium_anchor=args.au_premium,
            au_premium_source=args.au_premium_source,
            nominal_window=args.nominal_window,
            use_forward=args.forward,
            use_curve=args.curve,
            curve_maturity=args.curve_maturity,
            curve_horizon_quarters=args.curve_maturity * 4,
            assert_stance=args.assert_stance,
            mu_g_sigma=args.mu_g_sigma,
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
