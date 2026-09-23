"""CLI for the r* summary.

Gathers every r* this repo estimates onto one nominal scale and charts them
together. Any model whose saved trace was not written today is re-run first,
which takes minutes and overwrites that model's own outputs and charts, so it
is announced before it happens and `--no-refresh` turns it off.
"""

import argparse

from src.models.common.inflation_scale import SCALES, TARGET
from src.models.rstar_summary.analyse import run_analyse
from src.models.rstar_summary.sources import DEFAULT_SCALE, gather


def main(
    *,
    allow_refresh: bool = True,
    start: str | None = "1993Q1",
    verbose: bool = True,
    scale: str = DEFAULT_SCALE,
) -> None:
    """Gather the r* estimates and chart them."""
    print("=" * 70)
    print("R* SUMMARY [every model, one nominal scale]")
    print("=" * 70)
    frame, notes = gather(allow_refresh=allow_refresh, verbose=verbose, scale=scale)
    run_analyse(frame, notes, start=start, scale=scale)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare every r* model on one scale")
    parser.add_argument(
        "--no-refresh", action="store_true",
        help="read saved runs as they stand instead of re-running stale ones",
    )
    parser.add_argument(
        "--start", type=str, default="1993Q1",
        help="first quarter to chart (default 1993Q1, the start of inflation targeting)",
    )
    parser.add_argument(
        "--nominal-on", default=DEFAULT_SCALE, choices=list(SCALES),
        help=f"how real r* converts to nominal: 'expectations' (default) adds inflation "
             f"expectations, matching the RBA and CBA; 'target' adds {TARGET:g}%%",
    )
    parser.add_argument("-q", "--quiet", action="store_true")
    args = parser.parse_args()

    main(
        allow_refresh=not args.no_refresh,
        start=args.start,
        verbose=not args.quiet,
        scale=args.nominal_on,
    )
