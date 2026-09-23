"""Command-line entry point for the joint y*/u* model."""

from src.models.ystar_ustar.cli import build_parser, run_from_args
from src.models.ystar_ustar.compare import load_all
from src.models.ystar_ustar.compare_charts import run_comparison


def main() -> None:
    """Estimate and chart the joint model, or compare its specifications."""
    args = build_parser().parse_args()
    if args.compare:
        loaded = load_all(refresh=not args.analyse_only)
        if loaded:
            run_comparison(loaded)
        else:
            print("Nothing to chart.")
        return
    run_from_args(args)


if __name__ == "__main__":
    main()
