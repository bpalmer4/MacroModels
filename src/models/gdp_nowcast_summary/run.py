"""CLI for the GDP nowcast summary.

Runs every GDP nowcast for the next unpublished quarter and charts their
distributions together, Q/Q and annual. The nowcasts save nothing, so each is
run afresh every time.
"""

import argparse

from src.models.gdp_nowcast_summary.analyse import run_analyse
from src.models.gdp_nowcast_summary.sources import gather


def main(*, verbose: bool = True) -> None:
    """Run the nowcasts and chart their distributions."""
    print("=" * 70)
    print("GDP NOWCAST SUMMARY [every nowcast's distribution, one chart]")
    print("=" * 70)
    run_analyse(gather(verbose=verbose))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chart every GDP nowcast's distribution together")
    parser.add_argument("-q", "--quiet", action="store_true", help="suppress each model's own printout")
    args = parser.parse_args()
    main(verbose=not args.quiet)
