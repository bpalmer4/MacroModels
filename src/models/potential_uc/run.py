"""Entry point for the potential_uc model.

Usage::

    uv run python -m src.models.potential_uc.run
    uv run python -m src.models.potential_uc.run --start 1993Q1 --draws 4000
    uv run python -m src.models.potential_uc.run --analyse-only
"""

import argparse

from src.models.potential_uc.analyse import run_analysis
from src.models.potential_uc.base import SamplerConfig
from src.models.potential_uc.config import PI_BASES, SPECS, SUPPLY_CONTROLS, ModelConfig
from src.models.potential_uc.estimate import run_estimate


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run the potential_uc model")

    parser.add_argument("--start", default="1993Q1", help="Sample start (default 1993Q1)")
    parser.add_argument("--end", default=None, help="Sample end (default: latest)")
    parser.add_argument("--anchor", type=float, default=2.5, help="Inflation anchor, annual %%")
    parser.add_argument("--spec", default="inflation", choices=SPECS, help="Specification")
    parser.add_argument(
        "--pi-basis", default="annual", choices=PI_BASES,
        help="Trimmed mean basis: 'annual' (four-quarter) or 'quarterly' (annualised)",
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

    parser.add_argument("--sigma-c", type=float, default=0.60, help="Fixed cycle innovation sd")
    parser.add_argument("--ratio-ystar", type=float, default=0.13, help="core: potential level")
    parser.add_argument("--ratio-g", type=float, default=0.025, help="core: trend growth")
    parser.add_argument("--ratio-pr-star", type=float, default=0.10)
    parser.add_argument("--ratio-hpp-star", type=float, default=0.10)
    parser.add_argument("--ratio-lp-star", type=float, default=0.10)
    parser.add_argument("--ratio-g-lp", type=float, default=0.025)

    parser.add_argument("--draws", type=int, default=2_000)
    parser.add_argument("--tune", type=int, default=2_000)
    parser.add_argument("--chains", type=int, default=4)
    parser.add_argument("--seed", type=int, default=None)

    parser.add_argument("--prefix", default="potential_uc", help="Output filename prefix")
    parser.add_argument("--analyse-only", action="store_true", help="Skip estimation")
    parser.add_argument("--no-analyse", action="store_true", help="Estimate without charting")
    parser.add_argument(
        "--no-decompose", action="store_true",
        help="Skip the hours/productivity accounting split (avoids loading labour force data)",
    )
    parser.add_argument("--verbose", action="store_true")

    return parser.parse_args()


def main() -> None:
    """Estimate the model and produce the charts."""
    args = parse_args()

    config = ModelConfig(
        spec=args.spec,
        start=args.start,
        end=args.end,
        anchor=args.anchor,
        pi_basis=args.pi_basis,
        supply_control=args.supply_control,
        gap_sd_on_target=args.gap_sd_on_target,
        gap_sd_per_pp=args.gap_sd_per_pp,
        cycle_ar=not args.no_cycle_ar,
        sigma_c=args.sigma_c,
        ratio_ystar=args.ratio_ystar,
        ratio_g=args.ratio_g,
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


if __name__ == "__main__":
    main()
