"""Re-estimate across `sigma_q`, so the imposed amount of drift is visible as a range.

`sigma_q` is the number that decides this model's answer. It sets how far the VAR
coefficients may move per quarter, and r* is a long-horizon projection built from
those coefficients, so the whole path is downstream of it. The credible band on
the headline chart is conditional on it: that is the uncertainty GIVEN the drift
assumption, not the whole of it. This module supplies the other half.

THE DIAGNOSTIC THIS SWEEP EXISTS FOR is `corr(r*, real cash rate)`. The default
run puts `sigma_q` at 0.0036 and produces an r* that tracks the real cash rate
closely, which is the failure `rstar_rba`'s notes name: if neutral is the policy
rate smoothed, the identification has collapsed into the state. The question the
sweep answers is whether that is a property of this model at every defensible
drift, or an artefact of the one value the posterior happened to pick.

    sigma_q = 0      a constant-coefficient VAR. NOT a degenerate case: r* still
                     moves, because the H-step projection depends on the current
                     state as well as the coefficients. It is the honest "no
                     time variation" baseline and the cleanest thing to compare
                     the drifting versions against.
    0.002 - 0.01     the neighbourhood the posterior chose.
    0.02             the prior's centre.
    0.05             deliberately past defensible: enough drift that the
                     coefficients refit every few years. Carried to SHOW the
                     boundary, not as a candidate answer.

The two uncertainties are different in kind and must not be added. The band is
sampling uncertainty within one assumption; the envelope here is the assumption
moving.

The observations are built ONCE and shared across members. `sigma_q` enters the
model and not the data, so re-fetching per run would only risk members differing
by a data revision landing mid-loop.
"""

import pickle
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.models.common.model_constants import get_dictionary
from src.models.rstar_tvpvar.config import DEFAULT_OUTPUT_DIR, ModelConfig
from src.models.rstar_tvpvar.estimate import build_model
from src.models.rstar_tvpvar.observations import build_observations
from src.models.rstar_tvpvar.results import TvpVarResults
from src.models.ystar.base import SamplerConfig, sample_model

DEFAULT_SIGMA_Q_VALUES = (0.0, 0.002, 0.005, 0.01, 0.02, 0.05)

# The era every other r* model in this repo is judged on, so the comparison is
# like for like: `rstar_bonds` and `rstar_invert` disagree about its sign.
_WATCH_ERA = ("2016Q1", "2019Q4")

# Correlation with the real cash rate above which r* is just that rate smoothed.
_CASH_RATE_CORR = 0.9


def _summarise(results: TvpVarResults, sigma_q: float, real_rate: pd.Series) -> dict[str, Any]:
    """Return one row of the ensemble table."""
    rstar = results.rstar_median()
    rstar_values: np.ndarray = np.asarray(rstar.to_list(), dtype=np.float64)
    band = results.rstar_hdi(0.90)
    # [lower, upper] of the final quarter, as plain floats for the same reason.
    band_last: np.ndarray = np.asarray(band.iloc[-1].to_list(), dtype=np.float64)
    stability = results.stability_report()
    start, end = _WATCH_ERA
    watch = rstar.loc[pd.Period(start, "Q"):pd.Period(end, "Q")]
    aligned = pd.concat({"r": rstar, "cash": real_rate}, axis=1).dropna()
    return {
        "sigma_q": sigma_q,
        "r*_latest": float(rstar.iloc[-1]),
        "r*_mean": float(rstar.mean()),
        # Via an explicitly float64 array: pandas' stubs type Series.max() as a
        # union wide enough to include Timestamp, so subtracting two of them is
        # 150 mypy errors about Timedelta.
        "r*_range": float(rstar_values.max()) - float(rstar_values.min()),
        # THE diagnostic. Near one means r* is the real cash rate smoothed and
        # the model is describing policy rather than measuring neutral.
        "corr_cash": float(aligned["r"].corr(aligned["cash"])),
        "r*_2016_19": float(watch.mean()) if len(watch) else float("nan"),
        "band_width": float(band_last[1]) - float(band_last[0]),
        "pct_explosive": float(stability["share of draw-quarters explosive"]),
    }


def run_sigma_q_ensemble(
    config: ModelConfig | None = None,
    sampler_config: SamplerConfig | None = None,
    values: tuple[float, ...] = DEFAULT_SIGMA_Q_VALUES,
    prefix: str = "rstar_tvpvar",
    *,
    seed: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Sample once per `sigma_q` and return the r* paths and the summary table."""
    config = config or ModelConfig()
    sampler_config = sampler_config or SamplerConfig()
    sampler_config.log_likelihood = False
    if seed is not None:
        sampler_config.random_seed = seed

    data, obs_index, frame, _sources = build_observations(
        start=config.start,
        end=config.end,
        basis=config.basis,
        deflator=config.deflator,
        # Explicit, though the signature now defaults to the same values: the
        # sweep must build the SAME data as the run it is sweeping, and these
        # two were the ones that silently differed.
        include_commodities=config.include_commodities,
        include_twi=config.include_twi,
        blank_quarters=config.blanked_quarters,
        verbose=False,
    )
    real_rate = frame["real_rate"]

    paths: dict[str, pd.Series] = {}
    rows: list[dict[str, Any]] = []
    for value in values:
        print(f"\nsigma_q = {value:g}")
        member = replace(config, sigma_q=value)
        model = build_model(data, member, verbose=False)
        trace = sample_model(model, sampler_config)
        results = TvpVarResults(
            trace=trace,
            data=data,
            obs_index=obs_index,
            constants=get_dictionary(model),
            frame=frame,
        )
        paths[f"{value:g}"] = results.rstar_median()
        rows.append(_summarise(results, value, real_rate))
        print(f"  r* latest {rows[-1]['r*_latest']:6.2f}   corr with real cash "
              f"{rows[-1]['corr_cash']:5.2f}   explosive {rows[-1]['pct_explosive']:5.1%}")

    path_frame = pd.DataFrame(paths)
    table = pd.DataFrame(rows).set_index("sigma_q")
    _save(path_frame, table, output_dir=config.output_dir, prefix=prefix)
    return path_frame, table


def _save(
    paths: pd.DataFrame,
    table: pd.DataFrame,
    output_dir: Path | str | None = None,
    prefix: str = "rstar_tvpvar",
) -> None:
    """Persist the ensemble beside the trace, so charting need not re-sample."""
    directory = Path(output_dir) if output_dir else DEFAULT_OUTPUT_DIR
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / f"{prefix}_sigma_q_ensemble.pkl"
    with target.open("wb") as handle:
        pickle.dump({"paths": paths, "table": table}, handle)
    print(f"\nSaved sigma_q ensemble to: {target}")


def load_ensemble(
    output_dir: Path | str | None = None,
    prefix: str = "rstar_tvpvar",
) -> dict[str, pd.DataFrame] | None:
    """Return a saved ensemble, or None if the sweep has not been run."""
    directory = Path(output_dir) if output_dir else DEFAULT_OUTPUT_DIR
    target = directory / f"{prefix}_sigma_q_ensemble.pkl"
    if not target.exists():
        return None
    with target.open("rb") as handle:
        saved = pickle.load(handle)
    if not isinstance(saved, dict):
        raise TypeError(f"{target} does not hold a sigma_q ensemble")
    return saved


def print_ensemble(table: pd.DataFrame) -> None:
    """Print the table the notes should quote instead of a single conditional level."""
    print("\nsigma_q ensemble: what the assumed drift decides")
    print("-" * 78)
    print("  corr_cash near 1 means r* is the real cash rate smoothed, which is this")
    print("  model's characteristic failure rather than a finding about neutral.")
    print(table.round(3).to_string())
    if "corr_cash" in table:
        worst = table["corr_cash"].max()
        best = table["corr_cash"].min()
        print(f"\n  corr with the real cash rate runs {best:.2f} to {worst:.2f} across the sweep.")
        if best > _CASH_RATE_CORR:
            print(f"  *** It is above {_CASH_RATE_CORR} at EVERY value, so no choice of drift rescues it.")
