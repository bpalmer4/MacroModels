"""Run the HLW Bayesian r-star estimation pipeline.

Usage:
    uv run python -m src.models.rstar_hlw.run
    uv run python -m src.models.rstar_hlw.run -v
    uv run python -m src.models.rstar_hlw.run --estimate-only
    uv run python -m src.models.rstar_hlw.run --skip-estimate
"""

import argparse
from typing import Final, Literal

# Estimation defaults live in `estimate.py`, which is imported lazily (it pulls
# in PyMC). This sentinel stands in for them until that import happens, so the
# values are not duplicated here.
USE_ESTIMATE_DEFAULT: Final = "default"

# What the CLI may hand `main`: an explicit setting, None for "switch it off",
# or the sentinel above for "whatever estimate.py defaults to".
ExcludeWindow = tuple[str, str] | Literal["default"] | None
LambdaG = float | Literal["default"] | None

# --exclude-window is START:END, so two pieces once split.
WINDOW_PARTS = 2


def main(
    *,
    verbose: bool = False,
    estimate: bool = True,
    analyse: bool = True,
    start: str = "1993Q1",
    end: str | None = None,
    resolution: str = "A",
    rate_lag: int | None = 6,
    sigma_ystar_prior: float = 0.12,
    sigma_ystar_fixed: float | None = 0.078,
    lambda_g: LambdaG = USE_ESTIMATE_DEFAULT,
    exclude_window: ExcludeWindow = USE_ESTIMATE_DEFAULT,
    seed: int | None = None,
) -> None:
    """Run estimation and/or analysis stages."""
    prefix = f"rstar_hlw_{resolution}"

    if estimate:
        from src.models.rstar_hlw.estimate import (  # noqa: PLC0415
            DEFAULT_EXCLUDE_WINDOW,
            DEFAULT_LAMBDA_G,
            run_estimate,
        )
        # isinstance rather than a comparison with the sentinel, so the types
        # narrow here as well as at runtime.
        window: tuple[str, str] | None = (
            DEFAULT_EXCLUDE_WINDOW if isinstance(exclude_window, str) else exclude_window
        )
        ratio: float | None = (
            DEFAULT_LAMBDA_G if isinstance(lambda_g, str) else lambda_g
        )

        lag_desc = "t-1,t-2 averaged" if rate_lag is None else f"t-{rate_lag}"
        window_desc = (
            "none" if window is None else f"{window[0]}-{window[1]}"
        )
        sigma_desc = (
            f"HalfNormal({sigma_ystar_prior})" if sigma_ystar_fixed is None
            else f"{sigma_ystar_fixed} imposed"
        )
        ratio_desc = "free sigma_g" if ratio is None else f"lambda_g {ratio}"
        print("=" * 60)
        print(f"ESTIMATE [HLW r-star, Resolution {resolution}, start={start}, "
              f"rate lag {lag_desc}]")
        print(f"         [sigma_ystar {sigma_desc}, {ratio_desc}, "
              f"excluded {window_desc}]")
        print("=" * 60)
        run_estimate(
            start=start, end=end, verbose=verbose,
            prefix=prefix, resolution=resolution, rate_lag=rate_lag,
            sigma_ystar_prior=sigma_ystar_prior,
            sigma_ystar_fixed=sigma_ystar_fixed,
            lambda_g=ratio,
            exclude_window=window,
            seed=seed,
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
        default="1993Q1",
        help=(
            "Sample start period (default 1993Q1, the inflation-targeting era). Pass "
            "1980Q1 for the old default, which the indexed bond yield pins to an "
            "effective 1986Q3: on that sample the output gap barely moves through the "
            "2022-26 disinflation, which unemployment contradicts"
        ),
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
        "--sigma-ystar",
        type=str,
        default="0.078",
        help=(
            "potential output's innovation sd, IMPOSED (default 0.078, matching ystar's "
            "ratio_ystar x sigma_c). This is HLW's own lambda_g device. Pass 'free' to "
            "estimate it under --sigma-ystar-prior instead, which is what the eight "
            "resolutions in MODEL_NOTES.md were run on and which let the posterior land "
            "at 0.862, seven sd into the prior's tail"
        ),
    )
    parser.add_argument(
        "--lambda-g",
        type=str,
        default=USE_ESTIMATE_DEFAULT,
        help=(
            "sigma_g / sigma_ystar, HLW's ratio device, imposed. DEFAULT IS 'free': "
            "lambda_g_sweep found HLW's own US value of 0.053 is rejected by AU data "
            "(it flattens trend growth, puts 2005-19 4.35%% above capacity and collapses "
            "the Phillips slope), and at 0.34, the ratio AU's growth slowdown implies, "
            "the constraint is not binding. Pass a float to impose one anyway"
        ),
    )
    parser.add_argument(
        "--exclude-window",
        type=str,
        default=USE_ESTIMATE_DEFAULT,
        help=(
            "quarters dropped from the IS and Phillips likelihoods, as START:END "
            "(default 2020Q2:2021Q3, the lockdowns, matching ystar). The states still "
            "run through the window under their priors; what goes is the claim that "
            "potential plus a cyclical gap should account for output and inflation "
            "while the economy was shut. Pass 'none' to fit them"
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

    if args.lambda_g == USE_ESTIMATE_DEFAULT:
        cli_lambda_g: LambdaG = USE_ESTIMATE_DEFAULT
    elif args.lambda_g.lower() == "free":
        cli_lambda_g = None
    else:
        cli_lambda_g = float(args.lambda_g)

    if args.exclude_window == USE_ESTIMATE_DEFAULT:
        cli_window: ExcludeWindow = USE_ESTIMATE_DEFAULT
    elif args.exclude_window.lower() == "none":
        cli_window = None
    else:
        parts = args.exclude_window.split(":")
        if len(parts) != WINDOW_PARTS:
            parser.error(
                f"--exclude-window must be START:END or 'none', got {args.exclude_window!r}",
            )
        cli_window = (parts[0], parts[1])

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
        sigma_ystar_fixed=(
            None if args.sigma_ystar.lower() == "free" else float(args.sigma_ystar)
        ),
        lambda_g=cli_lambda_g,
        exclude_window=cli_window,
        seed=args.seed,
    )
