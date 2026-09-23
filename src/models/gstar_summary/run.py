"""CLI for the potential-growth summary.

Gathers g* from every model in the repo that produces one and charts them
together. Any source whose saved trace is not from today is re-run first, with
the arguments that reproduce it, which also redraws its own charts.
`--no-refresh` reads the saved runs as they stand.
"""

import argparse

from src.models.gstar_summary.analyse import run_analyse
from src.models.gstar_summary.sources import gather


def main(
    *,
    allow_refresh: bool = True,
    start: str | None = "1993Q1",
    verbose: bool = True,
) -> None:
    """Gather the potential-growth estimates and chart them."""
    print("=" * 70)
    print("G* SUMMARY [potential growth, every model that estimates one]")
    print("=" * 70)
    frame, notes = gather(allow_refresh=allow_refresh, verbose=verbose)
    run_analyse(frame, notes, start=start)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare potential growth across every model that estimates it",
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
