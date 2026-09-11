"""Command-line entry point for the rstar_rba model."""

import argparse

from src.models.rstar_rba.analyse import run_analysis
from src.models.rstar_rba.config import WEIGHT_SCHEMES, ModelConfig
from src.models.rstar_rba.ensemble import DEFAULT_SIGMA_R_VALUES, run_sigma_r_ensemble
from src.models.rstar_rba.estimate import run_estimate
from src.models.rstar_rba.injection import INJECTION_YEARS, run_injection_test
from src.models.ystar.base import SamplerConfig


def main() -> None:
    """Estimate r* from the RBA's response to inflation, then chart it."""
    parser = argparse.ArgumentParser(
        description="r* from two proportional gaps: cash-to-neutral and inflation-to-target",
    )
    parser.add_argument("--start", default="1993Q1", help="Sample start (default 1993Q1)")
    parser.add_argument("--end", default=None, help="Sample end (default: latest)")
    parser.add_argument(
        "--weights", default="geometric", choices=list(WEIGHT_SCHEMES),
        help="How the past is weighted: geometric (default, one decay parameter), "
             "dirichlet (free weights), or flat (the imposed window comparator)",
    )
    parser.add_argument(
        "--max-lag", type=int, default=4,
        help="Quarters of inflation carried (default 4). Not estimable: the mean lag "
             "scales with this while the fit does not, so it is a judgement",
    )
    parser.add_argument("--window", type=int, default=4, help="Flat window, only with --weights flat")
    parser.add_argument("--anchor", type=float, default=2.5, help="Inflation target")
    parser.add_argument(
        "--nonlinear", action="store_true",
        help="Add a lambda_2 x g x |g| term, so the response rises faster than linearly "
             "in the size of the inflation gap. lambda_2 = 0 is the linear model",
    )
    parser.add_argument(
        "--no-walk", action="store_true",
        help="Hold neutral constant instead of letting it drift. The regression version, and "
             "it is rejected: its residual is a trending near-unit-root series rather than "
             "the noise the likelihood assumes. Kept because that rejection is a result",
    )
    parser.add_argument("--sigma-r", type=float, default=0.10, help="Imposed walk sd (with --walk)")
    parser.add_argument(
        "--sigma-r-ensemble", default=",".join(f"{v:g}" for v in DEFAULT_SIGMA_R_VALUES),
        metavar="VALUES",
        help="Comma-separated sigma_r values to re-estimate across (default "
             f"{','.join(f'{v:g}' for v in DEFAULT_SIGMA_R_VALUES)}). ON BY DEFAULT: sigma_r "
             "is imposed and decides the level, so the spread across defensible values is "
             "structural uncertainty the conditional band does not show, and it is as wide "
             "as that band. A primary result, not a robustness check",
    )
    parser.add_argument(
        "--no-sigma-r-ensemble", action="store_true",
        help="Skip the sigma_r ensemble. The saved one, if any, is left in place and still "
             "charted, so it can then be older than the trace beside it",
    )
    parser.add_argument(
        "--injection-test", default=",".join(str(y) for y in INJECTION_YEARS),
        metavar="YEARS",
        help="Comma-separated window lengths in years for the injection test (default "
             f"{','.join(str(y) for y in INJECTION_YEARS)}). ON BY DEFAULT: adds a known "
             "+1pp stance over each window, re-estimates, and reports how much comes back "
             "as residual against how much the base absorbs. It is what sets the horizon "
             "over which any era residual can be read at all",
    )
    parser.add_argument(
        "--no-injection-test", action="store_true",
        help="Skip the injection test. The saved one, if any, is left in place and still "
             "charted, so it can then be older than the trace beside it",
    )
    parser.add_argument(
        "--jumps", action="store_true",
        help="Permit a wider tail in the base innovation in the few quarters the economy "
             "moved abruptly, which the model may or may not use. Off by default: offered "
             "the licence the model barely takes it, and taking it costs the ABS GDP "
             "dependency. A sensitivity test, not the headline",
    )
    parser.add_argument(
        "--jump-source", default="gdp", choices=("world", "gdp", "gdp4"),
        help="What times the jump flags: GDP q/q (default), GDP 4q demeaned, or the world real rate",
    )
    parser.add_argument(
        "--jump-percentile", type=float, default=95.0,
        help="Percentile of |GDP q/q| at which a quarter is flagged (default 95)",
    )
    parser.add_argument(
        "--jump-nu", type=float, default=3.0,
        help="Imposed StudentT degrees of freedom at flagged quarters (default 3)",
    )
    parser.add_argument(
        "--partial-adjustment", action="store_true",
        help="Interest-rate smoothing: r_t = phi x r_{t-1} + (1-phi) x (b_t + lambda x g_t) "
             "+ eps. The fix for the residual autocorrelation of 0.857. EXPERIMENTAL, and "
             "note lambda becomes the LONG-RUN response, not the same-quarter one, so it is "
             "not comparable with the default run's",
    )
    parser.add_argument(
        "--employment", action="store_true",
        help="Add the second leg of the mandate: lambda_u x (u - u*), with u* read from a "
             "completed ystar_ustar run. Costs the package its self-containment",
    )
    parser.add_argument(
        "--lambda-split", default=None,
        help="Quarter from which a second lambda applies, e.g. 2008Q1 (default: one lambda)",
    )
    parser.add_argument(
        "--floor", type=float, default=None,
        help="Drop quarters with a cash rate at or below this from the likelihood. "
             "Omitted, the config default applies, which keeps every quarter: excluding "
             "them was tried and made the concavity slightly stronger, not weaker",
    )
    parser.add_argument("--draws", type=int, default=2_000)
    parser.add_argument("--tune", type=int, default=2_000)
    parser.add_argument("--chains", type=int, default=4)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--prefix", default="rstar_rba", help="Output filename prefix")
    parser.add_argument("--analyse-only", action="store_true", help="Skip estimation")
    parser.add_argument("--no-analyse", action="store_true", help="Estimate without charting")
    parser.add_argument("-v", "--verbose", action="store_true", help="Print detailed output")
    args = parser.parse_args()

    if not args.analyse_only:
        config = ModelConfig(
            start=args.start,
            end=args.end,
            weights=args.weights,
            max_lag=args.max_lag,
            window=args.window,
            anchor=args.anchor,
            nonlinear=args.nonlinear,
            **({} if args.floor is None else {"floor": args.floor}),
            walk=not args.no_walk,
            sigma_r=args.sigma_r,
            lambda_split=args.lambda_split,
            employment=args.employment,
            partial_adjustment=args.partial_adjustment,
            jumps=args.jumps,
            jump_source=args.jump_source,
            jump_percentile=args.jump_percentile,
            jump_nu=args.jump_nu,
        )
        sampler_config = SamplerConfig(draws=args.draws, tune=args.tune, chains=args.chains)
        run_estimate(
            config=config, sampler_config=sampler_config,
            prefix=args.prefix, verbose=args.verbose, seed=args.seed,
        )
        # Both on by default. They are not robustness checks any more: the
        # ensemble is how the level is quoted at all, and the injection test is
        # what says how far back an era residual can be read. Running them with
        # the estimate keeps all three in step, so a chart cannot show a spread
        # computed from an older trace.
        if not args.no_injection_test:
            spans = tuple(int(part) for part in args.injection_test.split(","))
            run_injection_test(
                config=config, sampler_config=sampler_config,
                years=spans, prefix=args.prefix, seed=args.seed,
            )
        if not args.no_sigma_r_ensemble:
            values = tuple(float(part) for part in args.sigma_r_ensemble.split(","))
            run_sigma_r_ensemble(
                config=config, sampler_config=sampler_config,
                values=values, prefix=args.prefix, seed=args.seed,
            )

    if not args.no_analyse:
        run_analysis(prefix=args.prefix)


if __name__ == "__main__":
    main()
