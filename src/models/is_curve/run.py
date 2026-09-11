"""Command-line entry point for the IS-curve scatter."""

import argparse

from src.models.is_curve.analyse import run_analysis
from src.models.is_curve.fit import DEFAULT_LAG
from src.models.is_curve.observations import (
    DEFAULT_START,
    DEFAULT_WINDOWS,
    GFC_TO_PANDEMIC_WINDOW,
)


def main() -> None:
    """Plot the output gap against the real rate, under three r* treatments."""
    parser = argparse.ArgumentParser(
        description="Plot the IS curve from the data: output gap vs the real rate, with the OLS line",
    )
    parser.add_argument("--start", default=DEFAULT_START, help=f"Sample start (default {DEFAULT_START})")
    parser.add_argument("--end", default=None, help="Sample end (default: latest)")
    parser.add_argument(
        "--lag", type=int, default=DEFAULT_LAG,
        help=f"Quarters by which the rate is lagged (default {DEFAULT_LAG}, as in nairu)",
    )
    parser.add_argument(
        "--joint-prefix", default="ystar_ustar",
        help="Prefix of the saved joint y*/u* run supplying the output gap",
    )
    parser.add_argument(
        "--rstar-prefix", default="rstar_bonds",
        help="Prefix of the saved rstar run supplying r*",
    )
    parser.add_argument(
        "--rule-prefix", default="rstar_rba",
        help="Prefix of the rstar_rba run supplying the reaction-function neutral b_t",
    )
    parser.add_argument(
        "--keep-all", action="store_true",
        help="Keep every quarter, including the lockdowns excluded by default",
    )
    parser.add_argument(
        "--drop-gfc-pandemic", action="store_true",
        help=(
            f"Drop {GFC_TO_PANDEMIC_WINDOW[0]} to {GFC_TO_PANDEMIC_WINDOW[1]} as one stretch, on the "
            "ground that with QE and the lower bound the cash rate is not the stance"
        ),
    )
    parser.add_argument(
        "--exclude-window", nargs=2, metavar=("FIRST", "LAST"), action="append", default=None,
        help="Quarters to exclude; repeat the flag for more than one window",
    )
    args = parser.parse_args()

    if args.keep_all:
        windows: tuple[tuple[str, str], ...] | None = None
    elif args.exclude_window is not None:
        windows = tuple((first, last) for first, last in args.exclude_window)
    elif args.drop_gfc_pandemic:
        windows = (GFC_TO_PANDEMIC_WINDOW,)
    else:
        windows = DEFAULT_WINDOWS

    run_analysis(
        start=args.start,
        end=args.end,
        lag=args.lag,
        joint_prefix=args.joint_prefix,
        rstar_prefix=args.rstar_prefix,
        rule_prefix=args.rule_prefix,
        exclude_windows=windows,
    )


if __name__ == "__main__":
    main()
