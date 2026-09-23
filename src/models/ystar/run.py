"""Entry point for the ystar model.

Usage::

    uv run python -m src.models.ystar.run
    uv run python -m src.models.ystar.run --start 1993Q1 --draws 4000
    uv run python -m src.models.ystar.run --analyse-only
    uv run python -m src.models.ystar.run --compare
"""

from src.models.ystar.cli import build_parser, run_from_args
from src.models.ystar.specs import load_all
from src.models.ystar.specs_charts import run_comparison


def main() -> None:
    """Estimate the model and chart it; or, with --compare, the comparison specifications."""
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
