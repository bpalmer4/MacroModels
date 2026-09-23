"""Command-line entry point for the ustar model."""

from src.models.ustar.cli import build_parser, run_from_args
from src.models.ustar.compare import load_all
from src.models.ustar.compare_charts import run_comparison


def main() -> None:
    """Estimate u*, then chart it; or, with --compare, the comparison specifications."""
    args = build_parser().parse_args()
    if args.compare:
        loaded = load_all(refresh=not args.analyse_only)
        if loaded:
            for item in loaded:
                item.spec.chart()
            run_comparison(loaded)
        else:
            print("Nothing to chart.")
        return
    run_from_args(args)


if __name__ == "__main__":
    main()
