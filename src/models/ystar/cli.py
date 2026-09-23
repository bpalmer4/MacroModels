"""The ystar command line: its flags, and how a set of flags becomes a run.

Shared by the default run and by --compare, which defines each comparison
specification as the flags it would be run with. Kept out of `run.py` so the
comparison can use it without importing the entry point that imports it.
"""

import argparse

from src.models.common.cli import add_run_args, add_sampler_args
from src.models.ystar.analyse import run_analysis
from src.models.ystar.base import SamplerConfig
from src.models.ystar.config import (
    ANCHOR_PHASES,
    DEFAULT_EXCLUDE_WINDOW,
    PI_BASES,
    SPECS,
    SUPPLY_CONTROLS,
    YSTAR_SPLINE_KNOTS,
    YSTAR_STRUCTURES,
    ModelConfig,
)
from src.models.ystar.estimate import run_estimate


def build_parser() -> argparse.ArgumentParser:
    """Return the command-line parser, shared by the default run and --compare."""
    parser = argparse.ArgumentParser(description="Run the ystar model")

    parser.add_argument("--start", default="1993Q1", help="Sample start (default 1993Q1)")
    parser.add_argument("--end", default=None, help="Sample end (default: latest)")
    parser.add_argument("--anchor", type=float, default=2.5, help="Inflation anchor, annual %%")
    parser.add_argument(
        "--anchor-phase", default="none", choices=list(ANCHOR_PHASES),
        help="'glide' uses measured expectations before 1993Q1 and phases to the anchor "
             "across 1993Q1-1998Q4, which is what a pre-1993 sample start requires. "
             "See ModelConfig.anchor_phase",
    )
    parser.add_argument("--spec", default="inflation", choices=SPECS, help="Specification")
    parser.add_argument(
        "--ystar-structure", default="walk", choices=list(YSTAR_STRUCTURES),
        help="Structure imposed on potential output (default: walk). "
             "'spline' replaces the random walk with a natural cubic in time, "
             "which a two-year recession cannot bend. See ModelConfig.ystar_structure",
    )
    parser.add_argument(
        "--ystar-free-ends", action="store_true",
        help="Drop the natural end conditions on the y* spline. With no knots this is "
             "the slowing-growth form: a global cubic in the level, quadratic in growth",
    )
    parser.add_argument(
        "--ystar-adjust", type=float, default=0.0, metavar="RATIO",
        help="Add a slow-moving random walk to the polynomial trend, with innovation sd "
             "RATIO x sigma_c (0 = off). See ModelConfig.ratio_ystar_adjust",
    )
    parser.add_argument(
        "--ystar-degree", type=int, default=3,
        help="Polynomial degree of the y* basis (default 3). Growth is one degree lower",
    )
    parser.add_argument(
        "--ystar-knots", nargs="*", default=list(YSTAR_SPLINE_KNOTS), metavar="QUARTER",
        help="Interior knots for the y* spline. None gives a straight line, so "
             "constant potential growth across the sample",
    )
    parser.add_argument(
        "--pi-basis", default="annual", choices=PI_BASES,
        help="Trimmed mean basis: 'annual' (four-quarter, the default) or "
             "'quarterly' (annualised, non-overlapping; required for --spec core)",
    )
    parser.add_argument(
        "--supply-control", default=None, choices=[c for c in SUPPLY_CONTROLS if c],
        help="Cost-push regressor in the Phillips curve (default: none)",
    )

    parser.add_argument(
        "--gap-sd-on-target", type=float, default=0.50,
        help="target spec: sd of the 'gap is zero' claim when inflation is at target",
    )
    parser.add_argument(
        "--gap-sd-per-pp", type=float, default=1.00,
        help="target spec: extra sd per pp of inflation deviation from target",
    )
    parser.add_argument(
        "--no-cycle-ar", action="store_true",
        help="drop the AR(2) cycle restriction; gap becomes the bare identity",
    )

    parser.add_argument(
        "--ar1-residual", action="store_true",
        help="inflation spec: let the GDP residual e_c be AR(1) rather than white noise",
    )

    parser.add_argument(
        "--two-sided-c", action="store_true",
        help="inflation spec: give c a Normal(0,2) prior so its sign is estimated, not imposed",
    )

    parser.add_argument(
        "--zero-deviation", nargs=2, metavar=("FROM", "TO"), default=None,
        help="inclusive quarter range whose inflation deviation is set to zero, e.g. 2020Q2 2021Q1",
    )

    parser.add_argument(
        "--level-break", nargs="+", default=None, metavar="QUARTER",
        help="quarters at which y* takes a free one-off step, e.g. 2020Q2 2021Q4 "
             "(inflation, core and target specs)",
    )

    parser.add_argument(
        "--exclude-window", nargs=2, metavar=("FROM", "TO"), default=DEFAULT_EXCLUDE_WINDOW,
        help="inclusive quarter range dropped from the likelihood entirely "
             f"(default: {' '.join(DEFAULT_EXCLUDE_WINDOW)})",
    )
    parser.add_argument(
        "--no-exclude-window", action="store_true",
        help="fit the pandemic quarters like any other, recovering the continuous-sample model",
    )

    parser.add_argument("--sigma-c", type=float, default=0.60, help="Fixed cycle innovation sd")
    parser.add_argument("--ratio-ystar", type=float, default=0.13, help="core: potential level")
    parser.add_argument("--ratio-g", type=float, default=0.025, help="core: trend growth")
    parser.add_argument("--ratio-gk", type=float, default=0.05,
                        help="production: capital trend/obs sd ratio (= 1/sqrt(HP lambda))")
    parser.add_argument("--ratio-gl", type=float, default=0.0125,
                        help="production: hours trend/obs sd ratio")
    parser.add_argument("--ratio-gm", type=float, default=0.025,
                        help="production: MFP trend/obs sd ratio")
    parser.add_argument("--ratio-a", type=float, default=0.00625,
                        help="production: capital share trend/obs sd ratio")
    parser.add_argument("--mfp-degree", type=int, default=0,
                        help="production: polynomial degree for the MFP trend (0 = random walk)")
    parser.add_argument("--no-mfp-observation", action="store_true",
                        help="production: drop the MFP observation equation, which double-counts GDP")
    parser.add_argument("--sigma-gm", type=float, default=0.015,
                        help="production: imposed MFP trend innovation sd when MFP is not observed")
    parser.add_argument("--ratio-pr-star", type=float, default=0.10)
    parser.add_argument("--ratio-hpp-star", type=float, default=0.10)
    parser.add_argument("--ratio-lp-star", type=float, default=0.10)
    parser.add_argument("--ratio-g-lp", type=float, default=0.025)

    add_sampler_args(parser)
    parser.add_argument("--max-tree-depth", type=int, default=SamplerConfig.max_tree_depth,
                        help="NUTS trajectory cap, as a power of two")

    add_run_args(parser, prefix="ystar")
    parser.add_argument(
        "--no-decompose", action="store_true",
        help="Skip the hours/productivity accounting split (avoids loading labour force data). "
             "Always skipped for the labour and production specs, which split potential internally",
    )

    parser.add_argument(
        "--compare", action="store_true",
        help="Instead of the default run, re-estimate the comparison specifications that are "
             "not from today and chart them together; with --analyse-only, chart the saved "
             "runs as they stand. See MODEL_NOTES, 'Comparing specifications'",
    )
    return parser


def run_from_args(args: argparse.Namespace) -> None:
    """Estimate and/or chart one run, as the parsed flags say."""
    config = ModelConfig(
        spec=args.spec,
        start=args.start,
        end=args.end,
        anchor=args.anchor,
        anchor_phase=args.anchor_phase,
        ystar_structure=args.ystar_structure,
        ystar_spline_knots=tuple(args.ystar_knots),
        ystar_spline_natural=not args.ystar_free_ends,
        ystar_spline_degree=args.ystar_degree,
        ratio_ystar_adjust=args.ystar_adjust,
        pi_basis=args.pi_basis,
        supply_control=args.supply_control,
        gap_sd_on_target=args.gap_sd_on_target,
        gap_sd_per_pp=args.gap_sd_per_pp,
        cycle_ar=not args.no_cycle_ar,
        ar1_residual=args.ar1_residual,
        zero_deviation=tuple(args.zero_deviation) if args.zero_deviation else None,
        two_sided_c=args.two_sided_c,
        level_break=tuple(args.level_break) if args.level_break else None,
        exclude_window=None if args.no_exclude_window else tuple(args.exclude_window),
        sigma_c=args.sigma_c,
        ratio_ystar=args.ratio_ystar,
        ratio_g=args.ratio_g,
        ratio_gk=args.ratio_gk,
        ratio_gl=args.ratio_gl,
        ratio_gm=args.ratio_gm,
        ratio_a=args.ratio_a,
        mfp_observed=not args.no_mfp_observation,
        mfp_degree=args.mfp_degree,
        sigma_gm=args.sigma_gm,
        ratio_pr_star=args.ratio_pr_star,
        ratio_hpp_star=args.ratio_hpp_star,
        ratio_lp_star=args.ratio_lp_star,
        ratio_g_lp=args.ratio_g_lp,
    )

    if not args.analyse_only:
        sampler_config = SamplerConfig(
            draws=args.draws,
            tune=args.tune,
            chains=args.chains,
            cores=args.chains,
            max_tree_depth=args.max_tree_depth,
        )
        run_estimate(
            config=config,
            sampler_config=sampler_config,
            prefix=args.prefix,
            verbose=args.verbose,
            seed=args.seed,
        )

    if not args.no_analyse:
        run_analysis(
            output_dir=config.output_dir,
            prefix=args.prefix,
            decompose=not args.no_decompose,
        )
