"""Move the rule's three numbers, one at a time, and see what the reading does.

The model has no posterior, so it has no band. What stands in for one is this:
the answer is only worth as much as it is stable across the choices that produced
it. A reading that holds from a 6-quarter window to a 12-quarter one is telling
you something about the data; one that moves a point is telling you about the
rule.
"""

from dataclasses import replace

import pandas as pd

from src.models.long_run_ustar.config import ModelConfig
from src.models.long_run_ustar.model import Episode, find_episodes, reading

# The grids. Each moves one number and leaves the rest at the run's own values.
SWEEP_GRIDS: dict[str, tuple[float, ...]] = {
    "smooth": (1, 2, 4, 6),
    "window": (6, 8, 10, 12),
    "tolerance": (0.75, 1.0, 1.5, 2.0),
    # Only bites when `require_flat_u` is on, and it is the number with no
    # defensible prior value: 0.5pp over two years is tight enough to leave two
    # episodes, 1.5pp loose enough to be no constraint. Swept rather than chosen.
    "u_tolerance": (0.5, 0.75, 1.0, 1.5),
}


def _with_value(config: ModelConfig, param: str, value: float) -> ModelConfig:
    """Return `config` with one swept parameter replaced.

    Written out rather than `replace(config, **{param: value})`: the fields have
    different types, and a dict of them is untyped by construction.
    """
    if param == "smooth":
        return replace(config, smooth=int(value))
    if param == "window":
        return replace(config, window=int(value))
    if param == "tolerance":
        return replace(config, tolerance=float(value))
    if param == "u_tolerance":
        # The tolerance is inert unless the requirement it governs is on, so the
        # sweep turns it on. Otherwise every row of the grid is the base run.
        return replace(config, u_tolerance=float(value), require_flat_u=True)
    raise ValueError(f"no grid for {param!r}; choose from {tuple(SWEEP_GRIDS)}")


def _decade_readings(episodes: list[Episode], lag: int) -> dict[str, float]:
    """Return the length-weighted reading within each decade the episodes touch."""
    out: dict[str, float] = {}
    for decade in sorted({e.decade for e in episodes}):
        chosen = [e for e in episodes if e.decade == decade]
        out[f"{decade}s"] = reading(chosen, lag=lag)
    return out


def sweep(frame: pd.DataFrame, config: ModelConfig, param: str) -> pd.DataFrame:
    """Return one row per grid value of `param`: episodes found and what they read."""
    if param not in SWEEP_GRIDS:
        raise ValueError(f"no grid for {param!r}; choose from {tuple(SWEEP_GRIDS)}")

    lag = config.lags[0]
    rows = []
    for value in SWEEP_GRIDS[param]:
        episodes = find_episodes(frame, _with_value(config, param, value))
        rows.append({
            param: value,
            "episodes": len(episodes),
            "quarters": sum(e.quarters for e in episodes),
            "earliest": str(episodes[0].start) if episodes else "-",
            "reading": reading(episodes, lag=lag),
            **_decade_readings(episodes, lag),
        })
    return pd.DataFrame(rows).set_index(param)


def sweep_all(frame: pd.DataFrame, config: ModelConfig) -> dict[str, pd.DataFrame]:
    """Return every sweep, keyed by the parameter moved."""
    return {param: sweep(frame, config, param) for param in SWEEP_GRIDS}
