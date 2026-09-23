"""Command-line arguments for the joint y*/u* model, and a run from them.

Kept apart from `run.py` so `compare.py` can re-estimate its specifications
through the same parser without importing the entry point that imports it.
"""

import argparse

from src.models.common.cli import add_run_args, add_sampler_args
from src.models.ystar.base import SamplerConfig
from src.models.ystar_ustar.analyse import run_analysis
from src.models.ystar_ustar.config import (
    ANCHOR_PHASES,
    DEFAULT_EXCLUDE_WINDOW,
    EXCLUDE_SCOPES,
    GAP_PI_BASES,
    GAP_SPECS,
    OKUN_FORMS,
    USTAR_STRUCTURES,
    ModelConfig,
)
from src.models.ystar_ustar.estimate import run_estimate


def build_parser() -> argparse.ArgumentParser:
    """Return the command-line parser."""
    parser = argparse.ArgumentParser(
        description="Estimate y* and u* jointly, with a partly free output gap",
    )
    parser.add_argument("--start", default="1993Q1", help="Sample start (default 1993Q1)")
    parser.add_argument("--end", default=None, help="Sample end (default: latest)")
    parser.add_argument("--anchor", type=float, default=2.5, help="Inflation target, per cent")
    parser.add_argument(
        "--anchor-phase", default="none", choices=list(ANCHOR_PHASES),
        help="Whether the anchor is constant ('none'), expectations until 1998Q1 then the "
             "target ('step'), or a linear glide from 1993Q1 to 1998Q4 ('glide'). See "
             "ModelConfig.anchor_phase",
    )

    parser.add_argument(
        "--gap-spec", default="defined", choices=list(GAP_SPECS),
        help="How the gap is specified: 'defined' (c x (pi - anchor) + v) or "
             "'cycle' (a free AR(1) latent that inflation observes). See "
             "ModelConfig.gap_spec",
    )
    parser.add_argument(
        "--gap-pi-basis", default="quarterly", choices=list(GAP_PI_BASES),
        help="Which trimmed mean horizon defines the gap (default: quarterly). "
             "'quarterly' makes the gap an exact multiple of the Phillips curve's "
             "dependent variable; see ModelConfig.gap_pi_basis",
    )

    # --- The free gap component ---
    parser.add_argument(
        "--no-free-gap", action="store_true",
        help="Drop v, recovering ystar's identity gap_t = c x (pi - anchor)",
    )
    parser.add_argument(
        "--sigma-v", type=float, default=None,
        help="Impose sigma_v instead of estimating it (0 is equivalent to --no-free-gap)",
    )
    parser.add_argument(
        "--sigma-v-prior", type=float, default=1.0,
        help="Prior sd for the HalfNormal on sigma_v (default 1.0)",
    )

    # --- Imposed variances ---
    parser.add_argument("--sigma-c", type=float, default=0.60, help="Cycle scale (ystar's)")
    parser.add_argument("--ratio-ystar", type=float, default=0.13, help="sigma_ystar / sigma_c")
    parser.add_argument("--ratio-g", type=float, default=0.025, help="sigma_g / sigma_c")
    parser.add_argument("--sigma-ustar", type=float, default=0.020, help="Imposed u* innovation sd")

    # --- The pandemic window ---
    parser.add_argument(
        "--exclude-window", nargs=2, metavar=("FROM", "TO"), default=DEFAULT_EXCLUDE_WINDOW,
        help=f"inclusive quarter range dropped from the likelihood "
             f"(default: {' '.join(DEFAULT_EXCLUDE_WINDOW)})",
    )
    parser.add_argument(
        "--no-exclude-window", action="store_true",
        help="fit the pandemic quarters like any other",
    )
    parser.add_argument(
        "--exclude-scope", default="all", choices=list(EXCLUDE_SCOPES),
        help="Apply the window to all equations (default) or to the GDP equation only",
    )

    # --- Diagnostics ---
    parser.add_argument(
        "--no-phillips", action="store_true",
        help="Drop the Phillips curve. u* is then a trend, not a NAIRU, but sigma_v is "
             "measured with inflation out of the likelihood as a dependent variable",
    )
    parser.add_argument(
        "--no-okun", action="store_true",
        help="Drop the Okun equation. sigma_v should return its prior: this is the "
             "control that shows the covariance is what identifies it",
    )
    parser.add_argument(
        "--two-sided-c", action="store_true",
        help="Normal(0, 2) prior on c instead of HalfNormal, so the sign is tested",
    )
    parser.add_argument(
        "--beta-prior-sd", type=float, default=0.5,
        help="Prior sd for beta_okun (default 0.5, inherited from ustar). Widen it to "
             "test whether the textbook-Okun prior mean is pulling the slope",
    )
    parser.add_argument(
        "--sigma-okun", type=float, default=0.20,
        help="Imposed Okun residual sd (default 0.20). See ModelConfig.sigma_okun",
    )
    parser.add_argument(
        "--okun-form", default="gap", choices=list(OKUN_FORMS),
        help="'ec' uses the error-correction form, changes against growth relative to "
             "potential plus a level pull toward u*. See ModelConfig.okun_form",
    )
    parser.add_argument(
        "--ustar-structure", dest="ustar_structure", default="spline", choices=USTAR_STRUCTURES,
        help="The law u* follows (default spline). See ModelConfig.ustar_structure",
    )
    parser.add_argument(
        "--knots", nargs="+", default=["2013Q1"], metavar="QUARTER",
        help="Interior knot dates for the spline state (default 2013Q1)",
    )
    parser.add_argument(
        "--ustar-drift", action="store_true",
        help="Let u* drift down while inflation expectations sit above target, "
             "instead of being a driftless random walk. See ModelConfig.ustar_drift",
    )
    parser.add_argument(
        "--free-sigma-okun", action="store_true",
        help="Sample the Okun residual sd instead of imposing it. Restores the "
             "original specification; expect r_hat ~1.08 on it and 24,000 draws needed",
    )
    parser.add_argument(
        "--one-sided-beta", action="store_true",
        help="Truncate beta at zero, asserting Okun's law rather than testing it",
    )

    # 2,500 x 4 chains = 10,000 draws, which is enough because sigma_okun is
    # imposed. Free, it left a ridge against sigma_e that needed 24,000 draws
    # and still returned r_hat 1.08; imposed, the same model reports 0
    # divergences and a minimum ess of 2,553. Use --free-sigma-okun with
    # --draws 6000 to reproduce the original.
    add_sampler_args(parser, draws=2_500)
    parser.add_argument(
        "--target-accept", type=float, default=0.95,
        help="NUTS target acceptance rate; raise it if the run reports divergences",
    )

    add_run_args(parser, prefix="ystar_ustar")
    parser.add_argument(
        "--chart-dir", default=None,
        help="Where to write charts (default charts/YStarUStar). Use a separate "
             "directory for variant runs: charting clears its directory first",
    )
    parser.add_argument(
        "--compare", action="store_true",
        help="Chart eight specifications side by side (u* structure x gap definition), "
             "re-estimating any not run today; with --analyse-only, chart the saved runs "
             "as they stand. See MODEL_NOTES, 'Comparing specifications'",
    )
    return parser


def run_from_args(args: argparse.Namespace) -> None:
    """Estimate the joint model, then chart it."""
    if not args.analyse_only:
        config = ModelConfig(
            start=args.start,
            end=args.end,
            anchor=args.anchor,
            anchor_phase=args.anchor_phase,
            gap_spec=args.gap_spec,
            gap_pi_basis=args.gap_pi_basis,
            sigma_c=args.sigma_c,
            ratio_ystar=args.ratio_ystar,
            ratio_g=args.ratio_g,
            sigma_ustar=args.sigma_ustar,
            free_gap_component=not args.no_free_gap,
            sigma_v=args.sigma_v,
            sigma_v_prior=args.sigma_v_prior,
            two_sided_c=args.two_sided_c,
            two_sided_beta=not args.one_sided_beta,
            beta_okun_prior_sd=args.beta_prior_sd,
            # The identity gap carries GDP's own noise, which leaves the Okun
            # residual as the only place for it, so the sd is estimated there
            # whether or not --free-sigma-okun was passed.
            sigma_okun=(
                None
                if args.free_sigma_okun or args.gap_spec == "identity"
                else args.sigma_okun
            ),
            ustar_drift=args.ustar_drift,
            ustar_structure=args.ustar_structure,
            spline_knots=tuple(args.knots),
            exclude_window=None if args.no_exclude_window else tuple(args.exclude_window),
            exclude_scope=args.exclude_scope,
            include_phillips=not args.no_phillips,
            include_okun=not args.no_okun,
            okun_form=args.okun_form,
        )
        sampler_config = SamplerConfig(
            draws=args.draws, tune=args.tune, chains=args.chains,
            target_accept=args.target_accept,
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
            prefix=args.prefix,
            sigma_v_prior=args.sigma_v_prior,
            chart_dir=args.chart_dir,
        )
