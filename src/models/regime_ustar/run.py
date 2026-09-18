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
    parser.add_argument("--inflation", choices=("headline", "trimmed"), default="headline")
    parser.add_argument("--tot-control", action="store_true", help="add terms of trade growth to the equation")
    parser.add_argument("--wage", action="store_true",
                        help="add the ULC wage equation as a second observation on u*")
    parser.add_argument("--no-supply", action="store_true",
                        help="drop the import-price and GSCPI supply terms")
    parser.add_argument("--normal", action="store_true", help="Normal residuals instead of Student-t")
    parser.add_argument("--salience", type=float, default=6.0,
                        help="inflation rate above which attention switches on")
    parser.add_argument("--theta-lo", type=float, default=0.10, help="learning rate below the threshold")
    parser.add_argument("--theta-hi", type=float, default=0.60, help="learning rate above it")
    parser.add_argument("--sigma-ustar", type=float, default=0.05, help="imposed u* innovation sd")
    parser.add_argument("--free-phi", action="store_true", help="a separate convergence speed per regime")
    parser.add_argument("--adaptive-sigma", action="store_true",
                        help="let u* innovate more where unemployment has moved over the past two years")
    parser.add_argument("--start", default=None)
    parser.add_argument("--end", default=None)
    parser.add_argument("--draws", type=int, default=2000)
    parser.add_argument("--tune", type=int, default=2000)
    parser.add_argument("--analyse-only", action="store_true", help="chart a saved trace without re-sampling")
    return parser.parse_args()


def main() -> None:
    """Estimate, or re-chart, and report."""
    args = parse_args()
    config = ModelConfig(
        breaks=tuple(args.breaks),
        lag=args.lag,
        inflation=args.inflation,
        tot_control=args.tot_control,
        supply_control=not args.no_supply,
        wage_equation=args.wage,
        student_t=not args.normal,
        salience_threshold=args.salience,
        theta_lo=args.theta_lo,
        theta_hi=args.theta_hi,
        sigma_ustar=args.sigma_ustar,
        free_phi_per_regime=args.free_phi,
        adaptive_sigma=args.adaptive_sigma,
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
