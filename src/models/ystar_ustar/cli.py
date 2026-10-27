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
    CHART_DIR,
    DEFAULT_EXCLUDE_WINDOW,
    DEFAULT_PREFIX,
    EXCLUDE_SCOPES,
    GAP_PI_BASES,
    GAP_SPECS,
    OKUN_FORMS,
    SLACK_MEASURES,
    STEPPED_WALK_END,
    USTAR_STRUCTURES,
    WALK_SCALES,
    ModelConfig,
)
from src.models.ystar_ustar.estimate import run_estimate


def _scale_help(setting: str) -> str:
    """Quote one walk setting's default for each slack measure, from WALK_SCALES."""
    defaults = ", ".join(f"{slack} {getattr(scale, setting):g}" for slack, scale in WALK_SCALES.items())
    return f"(default by --slack: {defaults})"


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
        "--slack", default=ModelConfig.slack, choices=list(SLACK_MEASURES),
        help=f"The slack measure Okun and the Phillips curve read (default {ModelConfig.slack}); "
             "underutilisation adds the underemployed, so set the u* priors and steps to its scale",
    )
    parser.add_argument("--share-sigma-early", type=float, default=ModelConfig.share_sigma_early,
                        help=f"Slack split: step of unemployment's share (logit) at the start "
                             f"(default {ModelConfig.share_sigma_early:g})")
    parser.add_argument("--share-sigma-late", type=float, default=ModelConfig.share_sigma_late,
                        help=f"Slack split: step of unemployment's share (logit) from --taper-end on "
                             f"(default {ModelConfig.share_sigma_late:g})")
    parser.add_argument(
        "--gap-pi-basis", default="quarterly", choices=list(GAP_PI_BASES),
        help="Which trimmed mean horizon defines the gap (default: quarterly). "
             "'quarterly' makes the gap a transform of the Phillips curve's "
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
        "--inflation-lag", action="store_true",
        help="Put last quarter's inflation in the Phillips curve, so inflation is persistent "
             "and returning it to target can need u above u*. See ModelConfig.inflation_lag",
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
        "--beta-prior-mu", type=float, default=ModelConfig.beta_okun_prior_mu,
        help=f"Prior mean for beta_okun (default {ModelConfig.beta_okun_prior_mu:g}); scale it with the slack measure",
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
        "--ustar-structure", dest="ustar_structure", default=ModelConfig.ustar_structure, choices=USTAR_STRUCTURES,
        help=f"The law u* follows (default {ModelConfig.ustar_structure}). See ModelConfig.ustar_structure",
    )
    parser.add_argument(
        "--knots", nargs="+", default=list(ModelConfig.spline_knots), metavar="QUARTER",
        help=f"Interior knot dates for the spline state (default {' '.join(ModelConfig.spline_knots)})",
    )
    parser.add_argument("--taper-sigma-early", type=float, default=None,
                        help=f"Taper: u* innovation sd at the start {_scale_help('sigma_early')}")
    parser.add_argument("--taper-sigma-late", type=float, default=None,
                        help=f"Taper: u* innovation sd from --taper-end on {_scale_help('sigma_late')}")
    parser.add_argument("--taper-end", default=ModelConfig.taper_end,
                        help=f"Taper: quarter the sd reaches its late value (default {ModelConfig.taper_end})")
    parser.add_argument("--taper-init-mu", type=float, default=None,
                        help=f"Taper: prior mean for u* in the first quarter {_scale_help('init_mu')}")
    parser.add_argument("--taper-init-sd", type=float, default=None,
                        help=f"Taper: prior sd on u* in the first quarter {_scale_help('init_sd')}")
    parser.add_argument("--stepped-walk", action="store_true",
                        help="Taper: no taper, the early step size to --stepped-walk-end and the late one after")
    parser.add_argument("--stepped-walk-end", default=STEPPED_WALK_END, metavar="QUARTER",
                        help=f"Stepped walk: last quarter at the early step size (default {STEPPED_WALK_END})")
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

    add_run_args(parser, prefix=DEFAULT_PREFIX)
    parser.add_argument(
        "--chart-dir", default=None,
        help=f"Where to write charts (default {CHART_DIR.name} for the default prefix, "
             f"{CHART_DIR.name}-<prefix> otherwise, since charting clears its directory first)",
    )
    parser.add_argument(
        "--compare", action="store_true",
        help="Chart ten specifications side by side (u* structure x gap definition, plus the two walks), "
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
            slack=args.slack,
            share_sigma_early=args.share_sigma_early,
            share_sigma_late=args.share_sigma_late,
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
            beta_okun_prior_mu=args.beta_prior_mu,
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
            taper_sigma_early=args.taper_sigma_early,
            taper_sigma_late=args.taper_sigma_late,
            taper_end=args.taper_end,
            taper_init_mu=args.taper_init_mu,
            taper_init_sd=args.taper_init_sd,
            stepped_walk=args.stepped_walk,
            stepped_walk_end=args.stepped_walk_end,
            exclude_window=None if args.no_exclude_window else tuple(args.exclude_window),
            exclude_scope=args.exclude_scope,
            include_phillips=not args.no_phillips,
            include_okun=not args.no_okun,
            okun_form=args.okun_form,
            inflation_lag=args.inflation_lag,
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
