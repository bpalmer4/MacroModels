"""CLI for the y* specification summary."""

import argparse

from src.models.ystar_summary.analyse import run_analysis
from src.models.ystar_summary.sources import load_all


def main() -> None:
    """Refresh anything stale, then chart and tabulate."""
    parser = argparse.ArgumentParser(
        description="Five specifications of the y* model on one chart",
    )
    parser.add_argument("--no-refresh", action="store_true",
                        help="Chart the saved runs as they stand, however old")
    args = parser.parse_args()

    loaded = load_all(refresh=not args.no_refresh)
    if not loaded:
        print("Nothing to chart.")
        return
    run_analysis(loaded)


if __name__ == "__main__":
    main()
