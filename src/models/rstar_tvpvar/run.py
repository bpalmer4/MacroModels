"""Command-line entry point for the TVP-VAR r* model."""

import argparse

from src.models.common.cli import add_run_args, add_sampler_args
from src.models.rstar_tvpvar.analyse import run_analysis
from src.models.rstar_tvpvar.config import (
    ANCHOR_SOURCES,
    BASES,
    DEFLATORS,
    RSTAR_DEFINITIONS,
    ModelConfig,
)
from src.models.rstar_tvpvar.ensemble import (
    DEFAULT_SIGMA_Q_VALUES,
    print_ensemble,
    run_sigma_q_ensemble,
)
from src.models.rstar_tvpvar.estimate import run_estimate
from src.models.ystar.base import SamplerConfig


def _build_parser() -> argparse.ArgumentParser:
    """Build the CLI, split from `main` to keep each function reviewable."""
    # Every argparse default is READ FROM ModelConfig, never restated. Stating
    # them twice is how the shipped run silently stayed at five variables after
    # the config was set to three: `BooleanOptionalAction(default=True)` beat
    # `include_commodities = False` and nothing complained.
    d = ModelConfig()
    parser = argparse.ArgumentParser(description="r* from a TVP-VAR with stochastic volatility")
    parser.add_argument("--start", default=d.start, help="Sample start (default 1993Q1)")
    parser.add_argument("--end", default=None, help="Sample end (default: latest)")
    parser.add_argument(
        "--lags", type=int, default=d.lags,
        help="VAR lag order (default 2). More lags means more drifting coefficients on the "
             "same 135 quarters, so this trades dynamics against time variation",
    )
    parser.add_argument(
        "--basis", default=d.basis, choices=list(BASES),
        help="Whether inflation and growth are four-quarter or annualised one-quarter "
             "changes (default quarterly, the canonical choice; annual is quieter but "
             "puts an MA(3) in every variable for the coefficients to absorb)",
    )
    parser.add_argument(
        "--deflator", default=d.deflator, choices=list(DEFLATORS),
        help="What turns the cash rate real (default expectations, matching rstar_bonds)",
    )
    parser.add_argument(
        "--horizon", type=int, default=d.horizon_quarters,
        help="Quarters ahead that DEFINE r* (default 20, the five years CBA describe). "
             "Changing this needs no re-estimation if a trace already exists",
    )
    parser.add_argument(
        "--rstar-definition", default=d.rstar_definition, choices=list(RSTAR_DEFINITIONS),
        help="What r* means: 'projection' (default) is the H-quarter forecast, the "
             "published Lubik-Matthes definition; 'steady' is the rate the VAR comes to "
             "rest at; 'forward' averages quarters 20-40 and requires --anchor-projection",
    )
    parser.add_argument(
        "--training-sample", type=int, default=d.training_sample_quarters,
        help="Quarters of the sample the theta_0 OLS prior is fitted on (default 40). "
             "0 fits it on the full sample, which centres the prior on the same "
             "constant-coefficient answer the drift is being tested against",
    )
    parser.add_argument(
        "--sigma-q", type=float, default=None,
        help="IMPOSE the coefficient drift sd instead of estimating it. This is the "
             "parameter that decides the answer, so sweep it before believing a level",
    )
    parser.add_argument(
        "--sigma-q-prior", type=float, default=d.sigma_q_prior,
        help="HalfNormal scale on the drift sd when it is estimated (default 0.02)",
    )
    parser.add_argument(
        "--anchor-projection", action=argparse.BooleanOptionalAction, default=d.anchor_projection,
        help="Force inflation back toward its anchor along the projection (default OFF "
             "since the strip-back). It IMPOSES an anchor the VAR cannot infer, which "
             "turns r* into the rate this VAR's policy equation prescribes at that "
             "anchor, close to what rstar_rba already estimates from a written-down rule",
    )
    parser.add_argument(
        "--anchor-source", default=d.anchor_source, choices=list(ANCHOR_SOURCES),
        help="What inflation is anchored to: long-run expectations (default) or a flat 2.5",
    )
    parser.add_argument(
        "--anchor-return", type=float, default=d.anchor_return,
        help="Return speed phi in pi_h = anchor + (pi_0 - anchor)*phi^h "
             "(default 0.85, a half-life of about 4.3 quarters)",
    )
    parser.add_argument(
        "--commodities", action=argparse.BooleanOptionalAction, default=d.include_commodities,
        help="Include the RBA commodity price index as a fourth variable (default OFF: "
             "it addresses the price puzzle but failed to converge, R-hat 1.060)",
    )
    parser.add_argument(
        "--twi", action=argparse.BooleanOptionalAction, default=d.include_twi,
        help="Include the trade-weighted index as a fifth variable (default OFF: the best "
             "remaining candidate for a missing channel, but 42 divergences and R-hat 1.02)",
    )
    parser.add_argument(
        "--exclude-covid", action="store_true",
        help="Blank the lockdown quarters. Off by default: absorbing that spike in the "
             "volatility states rather than the coefficients is what SV is for, so "
             "excluding removes the model's best chance to show it works",
    )

    parser.add_argument(
        "--exclude-quarters", nargs="*", default=list(d.exclude_quarters), metavar="QUARTER",
        help="Blank arbitrary quarters from the likelihood, e.g. --exclude-quarters 2008Q4 "
             "2009Q1. They stay in the state, so the calendar and the drifting coefficients "
             "remain connected; only the likelihood skips them. Combines with --exclude-covid. "
             "A quarter outside the sample is an error, not a silent no-op",
    )

    add_sampler_args(parser, draws=1_000, tune=1_000)
    parser.add_argument(
        "--target-accept", type=float, default=None,
        help=f"NUTS acceptance target (default {SamplerConfig().target_accept:g}, or 0.9 with "
             "--smoke). Raise it to buy smaller steps and fewer divergences. It will not fix "
             "a low ESS or the explosive-draw share: those are the posterior's geometry, not "
             "the step size. Note max_tree_depth is 10 and raising it was tried and does "
             "nothing (see ystar/base.py)",
    )
    parser.add_argument(
        "--smoke", action="store_true",
        help="A fast, low-draw run to check the model builds and samples at all",
    )

    parser.add_argument(
        "--ensemble", action="store_true",
        help="Sweep sigma_q and save the range. This is the parameter that decides the "
             "answer, so the sweep is the honest report rather than a single level",
    )
    parser.add_argument(
        "--ensemble-values", type=float, nargs="*", default=None,
        help="sigma_q values for the sweep (default: 0 0.002 0.005 0.01 0.02 0.05)",
    )
    add_run_args(parser, prefix="rstar_tvpvar")
    return parser


def main() -> None:
    """Estimate the TVP-VAR, then chart its r*."""
    parser = _build_parser()
    args = parser.parse_args()

    # An EXPLICIT --target-accept wins over --smoke, so that a quick check of a
    # sampling change is possible; absent the flag, smoke keeps its cheaper 0.9
    # and everything else takes the SamplerConfig default rather than a restated
    # literal. Resolved once and shared by the estimate and the sweep, so the
    # sweep cannot sample on different settings from the run it is sweeping.
    if args.target_accept is None:
        target_accept = 0.9 if args.smoke else SamplerConfig().target_accept
    else:
        target_accept = args.target_accept
    if not 0.0 < target_accept < 1.0:
        parser.error(f"--target-accept must be strictly between 0 and 1, got {target_accept}")

    if not args.analyse_only:
        config = ModelConfig(
            start=args.start,
            end=args.end,
            lags=args.lags,
            basis=args.basis,
            deflator=args.deflator,
            horizon_quarters=args.horizon,
            rstar_definition=args.rstar_definition,
            # 0 means "no training sample", i.e. fit on everything. argparse has
            # no clean way to spell None for an int flag.
            training_sample_quarters=args.training_sample or None,
            sigma_q=args.sigma_q,
            sigma_q_prior=args.sigma_q_prior,
            exclude_covid=args.exclude_covid,
            exclude_quarters=tuple(args.exclude_quarters),
            include_commodities=args.commodities,
            include_twi=args.twi,
            anchor_projection=args.anchor_projection,
            anchor_source=args.anchor_source,
            anchor_return=args.anchor_return,
        )
        sampler = SamplerConfig(
            draws=100 if args.smoke else args.draws,
            tune=100 if args.smoke else args.tune,
            chains=2 if args.smoke else args.chains,
            # The state is thousands of non-centred normals, so the geometry is
            # awkward and 0.95 still leaves divergences on the table. --target-accept
            # raises it.
            target_accept=target_accept,
        )
        run_estimate(
            config=config,
            sampler_config=sampler,
            prefix=args.prefix,
            verbose=args.verbose,
            seed=args.seed,
        )

    if args.ensemble:
        values = tuple(args.ensemble_values) if args.ensemble_values else DEFAULT_SIGMA_Q_VALUES
        print("\n" + "=" * 70)
        print(f"SIGMA_Q SWEEP [{len(values)} members]")
        print("=" * 70)
        _paths, table = run_sigma_q_ensemble(
            config=ModelConfig(
                start=args.start, end=args.end, lags=args.lags, basis=args.basis,
                deflator=args.deflator, horizon_quarters=args.horizon,
                rstar_definition=args.rstar_definition,
                anchor_projection=args.anchor_projection,
                training_sample_quarters=args.training_sample or None,
                # The sweep must be OF the run being swept. These two default to
                # False on both sides today, so omitting them was harmless by
                # luck; with --commodities the estimate would carry four
                # variables and every sweep member three.
                include_commodities=args.commodities,
                include_twi=args.twi,
                exclude_covid=args.exclude_covid,
                exclude_quarters=tuple(args.exclude_quarters),
            ),
            sampler_config=SamplerConfig(
                draws=100 if args.smoke else args.draws,
                tune=100 if args.smoke else args.tune,
                chains=2 if args.smoke else args.chains,
                target_accept=target_accept,
            ),
            values=values,
            prefix=args.prefix,
            seed=args.seed,
        )
        print_ensemble(table)

    if not args.no_analyse:
        run_analysis(prefix=args.prefix)


if __name__ == "__main__":
    main()
