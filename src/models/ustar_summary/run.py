"""CLI for the u* summary.

Gathers six u* specifications from `ustar` and `ystar_ustar` and charts them
together. Any specification whose saved trace is not from today is re-run
first, which also redraws its own charts. `--no-refresh` reads the saved runs
as they stand.
"""

import argparse

from src.models.ustar_summary.analyse import run_analyse
from src.models.ustar_summary.sources import gather, load_unemployment


def main(
    *,
    allow_refresh: bool = True,
    start: str | None = "1993Q1",
    verbose: bool = True,
) -> None:
    """Gather the u* estimates and chart them."""
    print("=" * 70)
    print("U* SUMMARY [six specifications from two models]")
    print("=" * 70)
    frame, notes = gather(allow_refresh=allow_refresh, verbose=verbose)
    run_analyse(frame, notes, load_unemployment(), start=start)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare u* across the ustar and ystar_ustar specifications",
    )
    parser.add_argument(
        "--no-refresh", action="store_true",
        help="read saved runs as they stand instead of re-running stale ones",
    )
    parser.add_argument(
        "--start", type=str, default="1993Q1",
        help="first quarter to chart (default 1993Q1, the start of inflation targeting)",
    )
    parser.add_argument("-q", "--quiet", action="store_true")
    args = parser.parse_args()
    main(allow_refresh=not args.no_refresh, start=args.start, verbose=not args.quiet)
