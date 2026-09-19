"""CLI for the regime u* model."""

import argparse

import pandas as pd

from src.models.regime_ustar.analyse import analyse
from src.models.regime_ustar.config import DEFAULT_BREAKS, ModelConfig
from src.models.regime_ustar.estimate import estimate, load_trace, save_trace
from src.models.regime_ustar.observations import build_observations, regime_index, regime_labels
from src.models.ystar.base import SamplerConfig


def parse_args() -> argparse.Namespace:
    """Read the regimes and the equation off the command line."""
    parser = argparse.ArgumentParser(description="u* as a step function over imposed regimes")
    parser.add_argument("--breaks", nargs="*", default=list(DEFAULT_BREAKS),
                        help="first quarter of each regime after the first, e.g. 1974Q1 1983Q3")
    parser.add_argument("--lag", type=int, default=0, help="quarters unemployment leads the inflation change")
    parser.add_argument("--inflation", choices=("headline", "trimmed", "spliced"),
                        default=ModelConfig.inflation,
                        help="'spliced' is headline until the trimmed mean starts at 1983Q1, trimmed after")
    parser.add_argument("--tot-control", action="store_true", help="add terms of trade growth to the equation")
    parser.add_argument("--okun", action="store_true",
                        help="add the output error-correction equation as a second observation on u*")
    parser.add_argument("--wage", action="store_true",
                        help="add the ULC wage equation as a second observation on u*")
    parser.add_argument("--no-supply", action="store_true",
                        help="drop the import-price and GSCPI supply terms")
    parser.add_argument("--normal", action="store_true", help="Normal residuals instead of Student-t")
    parser.add_argument("--ar1", action="store_true", help="AR(1) error in the Phillips curve")
    parser.add_argument("--salience", type=float, default=6.0,
                        help="inflation rate above which attention switches on")
    parser.add_argument("--theta-lo", type=float, default=0.10, help="learning rate below the threshold")
    parser.add_argument("--theta-hi", type=float, default=0.60, help="learning rate above it")
    parser.add_argument("--sigma-ustar", type=float, default=0.05, help="imposed u* innovation sd")
    parser.add_argument("--beta-prior-sd", type=float, default=ModelConfig.beta_prior_sd,
                        help="HalfNormal scale on the Phillips slope")
    parser.add_argument("--intercept-regimes", nargs="*", type=int, default=None, metavar="K",
                        help="regimes given a free constant in the Phillips curve, e.g. 2")
    parser.add_argument("--beta-groups", nargs="*", type=int, default=None, metavar="G",
                        help="which slope each regime uses, one entry per regime, e.g. 0 1 2 0 0 0. "
                             "Omit for one slope across the sample")
    parser.add_argument("--free-phi", action="store_true", help="a separate convergence speed per regime")
    parser.add_argument("--adaptive-sigma", action="store_true",
                        help="let u* innovate more where unemployment has moved over the past two years")
    parser.add_argument("--knot-multiplicity", nargs="*", default=None, metavar="DATE:N",
                        help="repeat a knot to drop continuity there, e.g. 1974Q1:2; "
                             "3 matches the level only, 1 is the usual C2. Omit to keep the default")
    parser.add_argument("--expectations", choices=("model", "spliced"), default=ModelConfig.expectations_source,
                        help="'spliced' carries the measured expectation back to 1970 with PIE_RBAQ; "
                             "'model' asserts the salience rule before 1983")
    parser.add_argument("--splice-offset-quarters", type=int, default=ModelConfig.splice_offset_quarters,
                        help="overlap quarters the PIE_RBAQ level offset is measured on; 0 splices raw")
    parser.add_argument("--start", default=ModelConfig.start,
                        help="first quarter of the sample; 1959Q3 reaches the start of the unemployment rate, "
                             "at the price of asserting the expectation before 1970")
    parser.add_argument("--end", default=None)
    parser.add_argument("--draws", type=int, default=2000)
    parser.add_argument("--tune", type=int, default=2000)
    parser.add_argument("--analyse-only", action="store_true", help="chart a saved trace without re-sampling")
    return parser.parse_args()


def parse_multiplicity(pairs: list[str] | None) -> dict[str, int] | None:
    """Read `DATE:N` arguments into the knot-multiplicity mapping, or None to keep the default."""
    if pairs is None:
        return None
    parsed = {}
    for pair in pairs:
        date, _, count = pair.partition(":")
        if not count.isdigit():
            raise ValueError(f"knot multiplicity must be written DATE:N, got {pair!r}")
        parsed[date] = int(count)
    return parsed


def main() -> None:
    """Estimate, or re-chart, and report."""
    args = parse_args()
    multiplicity = parse_multiplicity(args.knot_multiplicity)
    config = ModelConfig(
        **({} if multiplicity is None else {"knot_multiplicity": multiplicity}),
        breaks=tuple(args.breaks),
        lag=args.lag,
        inflation=args.inflation,
        tot_control=args.tot_control,
        supply_control=not args.no_supply,
        wage_equation=args.wage,
        okun_equation=args.okun,
        student_t=not args.normal,
        ar1_error=args.ar1,
        salience_threshold=args.salience,
        theta_lo=args.theta_lo,
        theta_hi=args.theta_hi,
        sigma_ustar=args.sigma_ustar,
        beta_prior_sd=args.beta_prior_sd,
        beta_groups=tuple(args.beta_groups or ()),
        intercept_regimes=tuple(args.intercept_regimes or ()),
        free_phi_per_regime=args.free_phi,
        adaptive_sigma=args.adaptive_sigma,
        expectations_source=args.expectations,
        splice_offset_quarters=args.splice_offset_quarters,
        start=args.start,
        end=args.end,
    )

    if args.analyse_only:
        trace, frame, labels = load_trace(config)
        index = frame.index
        if not isinstance(index, pd.PeriodIndex):
            raise TypeError("the saved frame has lost its quarterly PeriodIndex")
        regimes = regime_index(index, config.breaks)
        if not labels:
            labels = regime_labels(index, regimes, len(config.breaks) + 1)
        source_footer = ""
    else:
        frame, regimes, labels, sources = build_observations(config)
        source_footer = sources.footer()
        print(f"Sample: {frame.index[0]}-{frame.index[-1]}, {len(frame)} quarters")
        for k, label in enumerate(labels):
            print(f"  regime {k}: {label}  ({int((regimes == k).sum())} quarters)")
        trace = estimate(frame, regimes, labels, config, SamplerConfig(draws=args.draws, tune=args.tune))
        print(f"Saved trace to: {save_trace(trace, frame, labels, config)}")

    analyse(trace, frame, regimes, labels, config, source_footer)


if __name__ == "__main__":
    main()
