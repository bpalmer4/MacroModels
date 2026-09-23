"""Run the HLW Bayesian r-star estimation pipeline.

Usage:
    uv run python -m src.models.rstar_hlw.run
    uv run python -m src.models.rstar_hlw.run -v
    uv run python -m src.models.rstar_hlw.run --estimate-only
    uv run python -m src.models.rstar_hlw.run --skip-estimate
"""

import argparse
from typing import Final, Literal

from src.models.rstar_hlw.analyse import run_analyse
from src.models.rstar_hlw.estimate import (
    DEFAULT_EXCLUDE_WINDOW,
    DEFAULT_LAMBDA_G,
    DEFAULT_RATE_LAG,
    DEFAULT_SIGMA_YSTAR_FIXED,
    run_estimate,
)
from src.models.rstar_hlw.observations import DEFAULT_G_ANCHOR, GAnchor
from src.models.rstar_hlw.results import DEFAULT_CHART_BASE
from src.models.rstar_hlw.stepwise import RESOLUTION as STEPWISE_RESOLUTION
from src.models.rstar_hlw.stepwise import main as run_stepwise

# Estimation defaults live in `estimate.py`. This sentinel means "whatever that
# module defaults to", which None cannot say: None is a real setting here,
# meaning switch the feature off.
USE_ESTIMATE_DEFAULT: Final = "default"

# What the CLI may hand `main`: an explicit setting, None for "switch it off",
# or the sentinel above for "whatever estimate.py defaults to".
ExcludeWindow = tuple[str, str] | Literal["default"] | None
LambdaG = float | Literal["default"] | None
RateLag = int | Literal["default"] | None
SigmaYstar = float | Literal["default"] | None

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
    rate_lag: RateLag = USE_ESTIMATE_DEFAULT,
    sigma_ystar_prior: float = 0.12,
    sigma_ystar_fixed: SigmaYstar = USE_ESTIMATE_DEFAULT,
    lambda_g: LambdaG = USE_ESTIMATE_DEFAULT,
    exclude_window: ExcludeWindow = USE_ESTIMATE_DEFAULT,
    g_anchor: GAnchor = DEFAULT_G_ANCHOR,
    canonical: bool = False,
    seed: int | None = None,
) -> None:
    """Run estimation and/or analysis stages."""
    if resolution == STEPWISE_RESOLUTION:
        # S is not one model, so it cannot go through `build_model`: it is
        # three models in serial, each locking a variance for the next. It
        # runs its own pipeline and writes its own prefixes and charts.
        if not estimate:
            raise ValueError(
                f"resolution {STEPWISE_RESOLUTION} estimates three models in "
                f"sequence, each conditioned on the one before, so there is "
                f"nothing to analyse without re-estimating. Drop "
                f"--skip-estimate",
            )
        run_stepwise()
        return

    # Anything that is not the production specification gets its own prefix
    # and chart directory, so one variant never overwrites another's trace or
    # charts.
    parts = [
        name for name, on in
        (("canonical", canonical), (g_anchor, g_anchor != DEFAULT_G_ANCHOR))
        if on
    ]
    prefix = "_".join(["rstar_hlw", resolution, *parts])
    chart_dir = (
        None if not parts
        else DEFAULT_CHART_BASE / "-".join(["rstar-hlw", resolution, *parts])
    )

    # `--canonical` supplies a value only where the CLI left the sentinel, so
    # an explicit flag always beats the bundle. Every canonical value here is
    # None, because each is a device this repo added that LW/HLW do not have:
    # they average the rate gap over t-1 and t-2 rather than take a single
    # lag, they estimate potential's innovation sd rather than impose one, and
    # they have no lockdown exclusion. isinstance rather than a comparison
    # with the sentinel, so the types narrow here as well as at runtime.
    lag: int | None = (
        (None if canonical else DEFAULT_RATE_LAG)
        if isinstance(rate_lag, str) else rate_lag
    )
    sigma_ystar: float | None = (
        (None if canonical else DEFAULT_SIGMA_YSTAR_FIXED)
        if isinstance(sigma_ystar_fixed, str) else sigma_ystar_fixed
    )
    window: tuple[str, str] | None = (
        (None if canonical else DEFAULT_EXCLUDE_WINDOW)
        if isinstance(exclude_window, str) else exclude_window
    )
    ratio: float | None = (
        DEFAULT_LAMBDA_G if isinstance(lambda_g, str) else lambda_g
    )

    if estimate:
        lag_desc = "t-1,t-2 averaged" if lag is None else f"t-{lag}"
        window_desc = "none" if window is None else f"{window[0]}-{window[1]}"
        sigma_desc = (
            f"HalfNormal({sigma_ystar_prior})" if sigma_ystar is None
            else f"{sigma_ystar} imposed"
        )
        ratio_desc = "free sigma_g" if ratio is None else f"lambda_g {ratio}"
        priors_desc = "HLW sign-only" if canonical else "repo informative"
        print("=" * 60)
        print(f"ESTIMATE [HLW r-star, Resolution {resolution}, start={start}, "
              f"rate lag {lag_desc}]")
        print(f"         [sigma_ystar {sigma_desc}, {ratio_desc}, "
              f"excluded {window_desc}, g-anchor {g_anchor}]")
        print(f"         [a_r/b_y priors {priors_desc}"
              f"{', CANONICAL' if canonical else ''}]")
        print("=" * 60)
        run_estimate(
            start=start, end=end, verbose=verbose,
            prefix=prefix, resolution=resolution, rate_lag=lag,
            sigma_ystar_prior=sigma_ystar_prior,
            sigma_ystar_fixed=sigma_ystar,
            lambda_g=ratio,
            exclude_window=window,
            sign_prior_only=canonical,
            g_anchor=g_anchor,
            seed=seed,
        )
        print()

    if analyse:
        print("=" * 60)
        print(f"ANALYSE [HLW r-star, Resolution {resolution}]")
        print("=" * 60)
        run_analyse(
            prefix=prefix, chart_dir=chart_dir, resolution=resolution, verbose=verbose,
        )
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
        "--analyse-only",
        dest="skip_estimate",
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
        type=str,
        default=USE_ESTIMATE_DEFAULT,
        help=(
            "single lag on the IS curve's rate gap (default 6, matching the is_curve "
            "bench and rstar_invert; pass 0 for HLW's own averaged t-1, t-2 shape, "
            "which is what --canonical selects). A longer lag carries less of the "
            "RBA's reaction to the economy"
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
        default=USE_ESTIMATE_DEFAULT,
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
        choices=["A", "B", "C", "D", "E", "F", "G", "H", "S"],
        default="A",
        help=(
            "r* identity: A (default; canonical HLW, r* = g + z), "
            "B (canonical + indexed bond observation), "
            "C (blend, fixed alpha prior), "
            "D (canonical r* + open-economy IS curve: fiscal + ToT + TWI + ICP), "
            "E (blend + AR(1) z: r* = alpha*g + (1-alpha)*(indexed-k) + z), "
            "F (E's r* identity + open-economy IS curve), "
            "G (C with hierarchical Beta(a, b) on alpha; a, b ~ Uniform(0.25, 2)), "
            "H (blend with time-varying alpha_t via logit-RW), "
            "S (NOT an identity: A's r* estimated in three serial stages after "
            "HLW, each locking a variance for the next. Its bands treat those "
            "locks as known, so they are not comparable with A-H's)"
        ),
    )
    parser.add_argument(
        "--g-anchor",
        type=str,
        choices=["linear", "cagr40"],
        default=DEFAULT_G_ANCHOR,
        help=(
            "soft anchor on trend growth, for the resolutions that carry one "
            "(C through H; A and B drop it). 'linear' (default) regresses "
            "year-on-year growth on time across the sample: unmoved by a single "
            "shock, but monotone by construction and fitted on quarters later "
            "than each date it describes. 'cagr40' is the 40-quarter trailing "
            "compound annual growth rate: one-sided and free to flatten out, at "
            "the cost of dating a change in trend about five years late. A "
            "non-default anchor writes to its own prefix and chart directory"
        ),
    )
    parser.add_argument(
        "--canonical",
        action="store_true",
        help=(
            "run the LW/HLW reference specification rather than this repo's repaired "
            "one: HLW's averaged t-1, t-2 rate gap, sign-only priors on a_r and b_y "
            "(the papers impose the sign and no magnitude), sigma_ystar estimated, and "
            "no lockdown exclusion. Each of those is a device this repo added, so the "
            "canonical run is expected to decompose WORSE, and it exists to show by "
            "how much. Writes to its own prefix and chart directory. Any explicit flag "
            "beats this bundle. Does NOT yet include HLW's two Stock-Watson ratios, "
            "lambda_g and lambda_z"
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

    if args.rate_lag == USE_ESTIMATE_DEFAULT:
        cli_rate_lag: RateLag = USE_ESTIMATE_DEFAULT
    else:
        # 0 means HLW's own averaged (t-1, t-2) shape rather than a single lag.
        cli_rate_lag = None if int(args.rate_lag) == 0 else int(args.rate_lag)

    if args.sigma_ystar == USE_ESTIMATE_DEFAULT:
        cli_sigma_ystar: SigmaYstar = USE_ESTIMATE_DEFAULT
    elif args.sigma_ystar.lower() == "free":
        cli_sigma_ystar = None
    else:
        cli_sigma_ystar = float(args.sigma_ystar)

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
        rate_lag=cli_rate_lag,
        sigma_ystar_prior=args.sigma_ystar_prior,
        sigma_ystar_fixed=cli_sigma_ystar,
        lambda_g=cli_lambda_g,
        exclude_window=cli_window,
        g_anchor=args.g_anchor,
        canonical=args.canonical,
        seed=args.seed,
    )
