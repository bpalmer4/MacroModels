"""CLI for the long-run u* reading."""

import argparse
import pickle

from src.models.long_run_ustar.analyse import run_analysis
from src.models.long_run_ustar.config import FLATNESS_RULES, ModelConfig
from src.models.long_run_ustar.model import episode_frame, find_episodes
from src.models.long_run_ustar.observations import build_observations


def parse_args() -> argparse.Namespace:
    """Parse the command line."""
    parser = argparse.ArgumentParser(
        description="Read u* off the stretches where inflation was flat, back to 1959",
    )
    parser.add_argument("--start", default=None, help="Sample start (default: all data)")
    parser.add_argument("--end", default=None, help="Sample end (default: latest)")
    parser.add_argument(
        "--smooth", type=int, default=4,
        help="Quarters of centred moving average on inflation before judging flatness",
    )
    parser.add_argument(
        "--window", type=int, default=8, help="Quarters the flatness test spans",
    )
    parser.add_argument(
        "--tolerance", type=float, default=1.0, help="How flat is flat, in percentage points",
    )
    parser.add_argument(
        "--flatness-rule", default="range", choices=list(FLATNESS_RULES),
        help="'range' (high minus low, rejects V shapes) or 'slope'",
    )
    parser.add_argument(
        "--min-quarters", type=int, default=2, help="Drop episodes shorter than this",
    )
    parser.add_argument(
        "--require-flat-u", action="store_true",
        help="Also require unemployment to be flat across the window",
    )
    parser.add_argument(
        "--u-tolerance", type=float, default=0.5,
        help="How flat unemployment must be, with --require-flat-u",
    )
    parser.add_argument(
        "--require-target", action="store_true",
        help="From 1993, also require inflation to sit near the 2.5%% target",
    )
    parser.add_argument("--prefix", default="long_run_ustar", help="Output filename prefix")
    parser.add_argument("--no-analyse", action="store_true", help="Skip tables and charts")
    return parser.parse_args()


def main() -> None:
    """Build the observations, find the episodes, report."""
    args = parse_args()
    config = ModelConfig(
        start=args.start,
        end=args.end,
        smooth=args.smooth,
        window=args.window,
        tolerance=args.tolerance,
        flatness_rule=args.flatness_rule,
        min_quarters=args.min_quarters,
        require_flat_u=args.require_flat_u,
        u_tolerance=args.u_tolerance,
        require_target=args.require_target,
        prefix=args.prefix,
    )

    frame, sources = build_observations(config)
    episodes = find_episodes(frame, config)

    config.output_dir.mkdir(parents=True, exist_ok=True)
    path = config.output_dir / f"{config.prefix}_episodes.pkl"
    with path.open("wb") as handle:
        pickle.dump(
            {
                "episodes": episode_frame(episodes, config),
                "frame": frame,
                "constants": config.constants,
                "sources": sources.to_records(),
            },
            handle,
        )
    print(f"Saved episodes to: {path}")

    if not args.no_analyse:
        run_analysis(frame, episodes, config, sources)


if __name__ == "__main__":
    main()
