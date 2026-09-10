"""Command-line entry point for the bank funding and lending charts."""

import argparse

from src.models.bank_costs.analyse import run_analysis


def main() -> None:
    """Chart bank lending and deposit rates against the cash rate."""
    parser = argparse.ArgumentParser(
        description="Chart what the cash rate costs a borrower and pays a saver",
    )
    parser.add_argument("--chart-dir", default=None, help="Override the chart directory")
    args = parser.parse_args()
    run_analysis(chart_dir=args.chart_dir)


if __name__ == "__main__":
    main()
