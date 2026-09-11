"""Run the HLW Bayesian r-star estimation pipeline.

Usage:
    uv run python -m src.models.rstar_hlw.run
    uv run python -m src.models.rstar_hlw.run -v
    uv run python -m src.models.rstar_hlw.run --estimate-only
    uv run python -m src.models.rstar_hlw.run --skip-estimate
"""

import argparse


def main(
    *,
    verbose: bool = False,
    estimate: bool = True,
    analyse: bool = True,
    start: str = "1980Q1",
    end: str | None = None,
    resolution: str = "A",
    rate_lag: int | None = 6,
    sigma_ystar_prior: float = 0.12,
    seed: int | None = None,
) -> None:
    """Run estimation and/or analysis stages."""
    prefix = f"rstar_hlw_{resolution}"

    if estimate:
        lag_desc = "t-1,t-2 averaged" if rate_lag is None else f"t-{rate_lag}"
        print("=" * 60)
        print(f"ESTIMATE [HLW r-star, Resolution {resolution}, start={start}, "
              f"rate lag {lag_desc}]")
        print("=" * 60)
        from src.models.rstar_hlw.estimate import run_estimate  # noqa: PLC0415
        run_estimate(
            start=start, end=end, verbose=verbose,
            prefix=prefix, resolution=resolution, rate_lag=rate_lag,
            sigma_ystar_prior=sigma_ystar_prior, seed=seed,
        )
        print()

    if analyse:
        print("=" * 60)
        print(f"ANALYSE [HLW r-star, Resolution {resolution}]")
        print("=" * 60)
        from src.models.rstar_hlw.analyse import run_analyse  # noqa: PLC0415
        run_analyse(prefix=prefix, resolution=resolution, verbose=verbose)
        print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run HLW Bayesian r-star model")
    parser.add_argument("-v", "--verbose", action="store_true", help="Print detailed output")
    parser.add_argument(
        "--estimate-only",
        action="store_true",
        help="Only run estimation (skip analysis)",
    )
    parser.add_argument(
        "--skip-estimate",
        action="store_true",
        help="Skip estimation (use saved results for analysis)",
    )
    parser.add_argument(
        "--start",
        type=str,
        default="1980Q1",
        help="Sample start period (default 1980Q1; try 1993Q1 to drop pre-target era)",
    )
    parser.add_argument(
        "--end",
        type=str,
        default=None,
        help="Sample end period (default: latest available)",
    )
    parser.add_argument(
        "--rate-lag",
        type=int,
        default=6,
        help=(
            "single lag on the IS curve's rate gap (default 6, matching the is_curve "
            "bench and rstar_invert; pass 0 for HLW's own averaged t-1, t-2 shape). "
            "A longer lag carries less of the RBA's reaction to the economy"
        ),
    )
    parser.add_argument(
        "--sigma-ystar-prior",
        type=float,
        default=0.12,
        help=(
            "HalfNormal scale on potential output's innovation (default 0.12, median "
            "0.081, matching ystar's imposed 0.078). Was 0.55, which let the posterior "
            "pile up at 1.11 and made potential more volatile than GDP. Pass 0.55 to "
            "reproduce the old behaviour"
        ),
    )
    parser.add_argument(
        "--resolution",
        type=str,
        choices=["A", "B", "C", "D", "E", "F", "G", "H"],
        default="A",
        help=(
            "r* identity: A (default; canonical HLW, r* = g + z), "
            "B (canonical + indexed bond observation), "
            "C (blend, fixed alpha prior), "
            "D (canonical r* + open-economy IS curve: fiscal + ToT + TWI + ICP), "
            "E (blend + AR(1) z: r* = alpha*g + (1-alpha)*(indexed-k) + z), "
            "F (E's r* identity + open-economy IS curve), "
            "G (C with hierarchical Beta(a, b) on alpha; a, b ~ Uniform(0.25, 2)), "
            "H (blend with time-varying alpha_t via logit-RW)"
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override the sampler random seed (default: SamplerConfig default = 42)",
    )
    args = parser.parse_args()
    main(
        verbose=args.verbose,
        estimate=not args.skip_estimate,
        analyse=not args.estimate_only,
        start=args.start,
        end=args.end,
        resolution=args.resolution,
        # 0 means HLW's own averaged (t-1, t-2) shape rather than a single lag.
        rate_lag=None if args.rate_lag == 0 else args.rate_lag,
        sigma_ystar_prior=args.sigma_ystar_prior,
        seed=args.seed,
    )
