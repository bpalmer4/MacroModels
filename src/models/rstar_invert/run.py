"""CLI for the rstar_invert model.

Two axes, both gaps, and a line through the origin:

    (1)  rstar_t = a slow-moving series
    (2)  x_t     = is_slope . (rbar_t - rstarbar_t) + e_t

The gap and the real cash rate are given. There is no constant term: a zero
rate gap means the economy is at potential, which is what neutral means.

The sign of `is_slope` is asserted (NK transmission), its magnitude is a prior
the data can argue with, and `sigma_rstar`, HOW SLOW r* IS, is asserted
outright because nothing measures it. `--ensemble` sweeps that last number,
which is the only genuinely open question here.

`--rstar-form linear` is the version that can be wrong: r* still moves over
time, but with two numbers rather than one per quarter, so the points cannot
slide along the x axis to meet the line.
"""

import argparse

from src.models.rstar_invert.analyse import run_analyse
from src.models.rstar_invert.config import RSTAR_FORMS, ModelConfig
from src.models.rstar_invert.ensemble import (
    DEFAULT_LAGS,
    DEFAULT_SIGMA_RSTAR,
    print_lag_table,
    print_table,
    run_ensemble,
    run_lag_sweep,
)
from src.models.rstar_invert.estimate import run_estimate
from src.models.ystar.base import SamplerConfig


def _parse_lags(text: str) -> tuple[int, ...]:
    """Parse a comma-separated lag list, e.g. '5' or '4,8'."""
    lags = tuple(int(part) for part in text.split(",") if part.strip())
    if not lags:
        raise argparse.ArgumentTypeError(f"no lags parsed from {text!r}")
    return lags



# Defaults are read from ModelConfig, never restated. Restating them is how a
# config change silently fails to reach a run: the value here wins.
_D = ModelConfig()


def main(
    is_slope_mu: float = _D.is_slope_mu,
    is_slope_sigma: float = _D.is_slope_sigma,
    sigma_rstar: float = _D.sigma_rstar,
    rstar_form: str = _D.rstar_form,
    rate_lags: tuple[int, ...] = _D.rate_lags,
    *,
    fix_lag_weight: bool = False,
    free_sigma_rstar: bool = False,
    exclude_qe: bool = False,
    estimate: bool = True,
    analyse: bool = True,
    ensemble: bool = False,
    lag_sweep: bool = False,
    verbose: bool = False,
    seed: int | None = None,
    prefix: str = "rstar_invert",
) -> None:
    """Run the estimation, the sweeps and/or the analysis stages."""
    config = ModelConfig(
        is_slope_mu=is_slope_mu,
        is_slope_sigma=is_slope_sigma,
        sigma_rstar=sigma_rstar,
        free_sigma_rstar=free_sigma_rstar,
        rstar_form=rstar_form,
        rate_lags=rate_lags,
        lag_weight_free=not fix_lag_weight,
        exclude_qe=exclude_qe,
    )

    if estimate:
        print("=" * 70)
        print(f"ESTIMATE [r* by conditional inversion; is_slope ~ N({is_slope_mu:g}, "
              f"{is_slope_sigma:g}), intercept {rstar_form}]")
        print("=" * 70)
        run_estimate(config, prefix=prefix, verbose=True, seed=seed)

    if lag_sweep:
        print()
        print("=" * 70)
        print("LAG SWEEP [does the mechanism appear where theory says it should?]")
        print("=" * 70)
        _, lag_table = run_lag_sweep(config, prefix=prefix, seed=seed)
        print_lag_table(lag_table)

    if ensemble:
        print()
        print("=" * 70)
        print("SWEEP [sigma_rstar: how slow is r*, the one open question]")
        print("=" * 70)
        _, table = run_ensemble(
            config,
            sampler_config=SamplerConfig(draws=2_000, tune=2_000, log_likelihood=False),
            prefix=prefix,
            seed=seed,
        )
        print_table(table)

    if analyse:
        print()
        print("=" * 70)
        print("ANALYSE [r* by conditional inversion]")
        print("=" * 70)
        run_analyse(prefix=prefix, verbose=verbose)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="r* by conditional inversion of an ASSERTED IS curve",
    )
    parser.add_argument(
        "--is-slope-mu", type=float, default=_D.is_slope_mu,
        help=(
            "prior mean of the IS slope, per cent of potential per pp of stance. "
            "Must be negative. Default -0.30, deliberately stronger than anything "
            "measured: a high slope is the FAVOURABLE case, so failure here is "
            "failure everywhere. The is_curve bench measures -0.108 at best"
        ),
    )
    parser.add_argument(
        "--is-slope-sigma", type=float, default=_D.is_slope_sigma,
        help="prior sd of the IS slope (default 0.10; smaller is closer to an assertion)",
    )
    parser.add_argument(
        "--sigma-rstar", type=float, default=_D.sigma_rstar,
        help=(
            "HOW SLOW r* IS: the asserted quarterly sd of its innovation, in "
            "percentage points (default 0.15). The one open question in the model, "
            "and nothing in the data settles it. Only used with --rstar-form walk"
        ),
    )
    parser.add_argument(
        "--rstar-form", type=str, default="walk", choices=list(RSTAR_FORMS),
        help=(
            "how much freedom r* gets: 'walk' is one value per quarter, so every "
            "point can slide along the x axis until it meets the line and the fit "
            "cannot fail; 'linear' is a straight drift, so r* still moves but the "
            "points cannot slide; 'constant' fixes r*, the reference case"
        ),
    )
    parser.add_argument(
        "--rate-lags", type=_parse_lags, default=(4, 8),
        help=(
            "one, two or three lags of the real cash rate, comma separated (default "
            "'4,8', a weighted pair whose effective mean lag is about 6.3 quarters, "
            "so it is comparable with the lag 6 used by the is_curve bench and "
            "rstar_hlw). A single lag, e.g. '6', drops the weight; three, e.g. "
            "'1,4,7', share a Dirichlet"
        ),
    )
    parser.add_argument(
        "--fix-lag-weight", action="store_true",
        help="fix the weight on the first lag at 0.5 instead of estimating it",
    )
    parser.add_argument(
        "--free-sigma-rstar", action="store_true",
        help="estimate sigma_rstar instead of asserting it (expect the prior back)",
    )
    parser.add_argument(
        "--exclude-qe", action="store_true",
        help="also drop 2008Q4-2021Q3, the cut that manufactures a negative IS slope",
    )
    parser.add_argument(
        "--lag-sweep", action="store_true",
        help=f"sweep the rate lag over {DEFAULT_LAGS}, where transmission should live",
    )
    parser.add_argument(
        "--ensemble", action="store_true",
        help=f"sweep sigma_rstar over {DEFAULT_SIGMA_RSTAR} (pp per quarter)",
    )
    parser.add_argument("--estimate-only", action="store_true", help="skip the charts")
    parser.add_argument(
        "--skip-estimate", "--analyse-only", dest="skip_estimate",
        action="store_true", help="chart a saved run",
    )
    parser.add_argument(
        "--prefix", type=str, default="rstar_invert",
        help=(
            "output prefix; charts go to charts/<prefix with hyphens>/, so a variant "
            "run (e.g. --rate-lag 5 --prefix rstar_invert_lag5) sits beside the default"
        ),
    )
    parser.add_argument("--seed", type=int, default=None, help="override the sampler seed")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    main(
        is_slope_mu=args.is_slope_mu,
        is_slope_sigma=args.is_slope_sigma,
        sigma_rstar=args.sigma_rstar,
        rstar_form=args.rstar_form,
        rate_lags=args.rate_lags,
        fix_lag_weight=args.fix_lag_weight,
        free_sigma_rstar=args.free_sigma_rstar,
        exclude_qe=args.exclude_qe,
        estimate=not args.skip_estimate,
        analyse=not args.estimate_only,
        ensemble=args.ensemble,
        lag_sweep=args.lag_sweep,
        verbose=args.verbose,
        seed=args.seed,
        prefix=args.prefix,
    )
