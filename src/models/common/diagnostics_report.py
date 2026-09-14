"""Check every saved trace and print the result.

Each run writes its own diagnostics file into its chart directory, beside the
charts it describes, and that is the only diagnostics FILE there is. This is
the whole-package view, printed to the terminal and written nowhere:

    uv run python -m src.models.common.diagnostics_report            # all traces
    uv run python -m src.models.common.diagnostics_report --only ystar

Nothing is re-sampled. The traces on disk are read as they stand, so this
describes the runs that produced them, not today's data.
"""

import argparse
from pathlib import Path

import arviz as az

from src.models.common.diagnostics import DRAGONS, check_model_diagnostics

DEFAULT_DIR = Path("model_outputs")
EXTRA_DIRS = (Path("output/expectations"),)

# Longest first, so "rstar_hlw" wins over "rstar" and "ystar_ustar" over "ystar".
KNOWN_MODELS = (
    "gdp_nowcast_components",
    "gdp_nowcast_bvar",
    "gdp_nowcast_dfm",
    "ystar_ustar",
    "long_run_ustar",
    "rstar_bonds",
    "rstar_invert",
    "rstar_hlw",
    "rstar_rba",
    "expectations",
    "fa_nk_bayes",
    "cobb_douglas",
    "nairu",
    "ustar",
    "ystar",
    "rstar",
)


def model_of(prefix: str) -> str:
    """Guess which model a trace prefix belongs to, for the report title."""
    for model in KNOWN_MODELS:
        if prefix == model or prefix.startswith(f"{model}_"):
            return model
    return prefix.split("_", maxsplit=1)[0]


def find_traces(directory: Path) -> list[Path]:
    """Every NetCDF file in `directory` that could hold a posterior.

    `*_loglik.nc` files are pointwise log likelihood only, saved for LOO/WAIC
    comparison, with no posterior group and nothing to diagnose.
    """
    return sorted(
        path
        for path in directory.glob("*.nc")
        if not path.stem.endswith("_loglik")
    )


def prefix_of(path: Path) -> str:
    """Return the name a model saved under: `ystar_trace.nc` -> `ystar`."""
    return path.stem.removesuffix("_trace")


def report_one(path: Path, *, verbose: bool = True) -> tuple[str, list[str]] | None:
    """Run the checks on one trace. None if it holds no posterior."""
    prefix = prefix_of(path)
    try:
        trace = az.from_netcdf(str(path))
    except (OSError, ValueError) as exc:
        if verbose:
            print(f"  skipped {path.name}: cannot read ({exc})")
        return None

    if not hasattr(trace, "posterior"):
        if verbose:
            print(f"  skipped {path.name}: no posterior group")
        return None

    return prefix, check_model_diagnostics(trace, verbose=False)


def print_summary(results: dict[str, list[str]]) -> None:
    """One line per trace, failures first, so the package reads at a glance.

    Printed, not written: the only diagnostics FILE is the one each run puts
    in its own chart directory, beside the charts it describes.
    """
    failed = {name: issues for name, issues in results.items() if issues}
    clean = [name for name, issues in results.items() if not issues]

    print(f"\nTraces checked: {len(results)}    with issues: {len(failed)}")
    if failed:
        print("\nWITH ISSUES")
        width = max(len(name) for name in failed)
        for name, issues in sorted(failed.items()):
            print(f"  FAIL  {name:<{width}}  {'; '.join(issues)}")
    print("\nCLEAN")
    for name in sorted(clean):
        print(f"  PASS  {name}")
    if not clean:
        print("  none")


def main() -> None:
    """Check every trace and write the roll-up."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dir",
        type=Path,
        default=None,
        help=f"one directory of saved traces (default: {DEFAULT_DIR} and {EXTRA_DIRS[0]})",
    )
    parser.add_argument(
        "--only",
        default="",
        help="only traces whose prefix contains this string",
    )
    args = parser.parse_args()

    # expectations saves its traces somewhere else, so the whole-package view
    # has to look in both places or it would quietly miss that model.
    directories = [args.dir] if args.dir is not None else [DEFAULT_DIR, *EXTRA_DIRS]
    traces = [path for directory in directories for path in find_traces(directory)]
    if args.only:
        traces = [path for path in traces if args.only in path.stem]
    if not traces:
        print(f"No traces found in {', '.join(str(d) for d in directories)}")
        return

    print(f"Checking {len(traces)} traces ...")
    results: dict[str, list[str]] = {}
    for path in traces:
        outcome = report_one(path)
        if outcome is None:
            continue
        prefix, issues = outcome
        results[prefix] = issues
        flag = f"{DRAGONS}{'; '.join(issues)}" if issues else "ok"
        print(f"  {prefix:<45} {flag}")

    if not results:
        print("No posteriors to report on.")
        return

    print_summary(results)


if __name__ == "__main__":
    main()
