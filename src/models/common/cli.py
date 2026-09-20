"""The command-line arguments every sampled model shares.

Only the ones that mean the same thing everywhere. A model's own switches stay
in its own `run.py`, where they are the readable statement of what the model can
be asked to do, and a generic CLI framework would bury them.

`--draws` and `--tune` take their defaults from the caller because the models
genuinely disagree: the joint model wants 2500 draws, the TVP-VAR 1000. The rest
are the same number everywhere and are settled here.
"""

import argparse

DEFAULT_DRAWS = 2_000
DEFAULT_TUNE = 2_000
DEFAULT_CHAINS = 4


def add_sampler_args(
    parser: argparse.ArgumentParser,
    *,
    draws: int = DEFAULT_DRAWS,
    tune: int = DEFAULT_TUNE,
    chains: int = DEFAULT_CHAINS,
) -> argparse.ArgumentParser:
    """Add `--draws`, `--tune`, `--chains` and `--seed`."""
    parser.add_argument("--draws", type=int, default=draws)
    parser.add_argument("--tune", type=int, default=tune)
    parser.add_argument("--chains", type=int, default=chains)
    parser.add_argument("--seed", type=int, default=None)
    return parser


def add_run_args(parser: argparse.ArgumentParser, *, prefix: str) -> argparse.ArgumentParser:
    """Add `--prefix`, the two stage switches, and `--verbose`.

    `prefix` is the model's own output prefix, which is the one thing here that
    differs between them.
    """
    parser.add_argument("--prefix", default=prefix, help="Output filename prefix")
    parser.add_argument("--analyse-only", action="store_true", help="Skip estimation")
    parser.add_argument("--no-analyse", action="store_true", help="Estimate without charting")
    parser.add_argument("-v", "--verbose", action="store_true", help="Print detailed output")
    return parser
