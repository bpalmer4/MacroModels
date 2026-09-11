"""Re-estimate across `sigma_r`, so the imposed smoothness is visible as a range.

`sigma_r` is the one imposed number that decides the split between the base and
the inflation response, and hence the LEVEL of r* and the era residuals. The
posterior band on the headline chart is conditional on it: it is the uncertainty
GIVEN the smoothness assumption, not the whole of it. This module supplies the
other half, by re-sampling at each of several defensible values and keeping the
paths.

The two uncertainties are different in kind and should not be added. The band is
sampling uncertainty within one assumption; the envelope here is the assumption
moving. Reported together they say: this is what the data can pin down, and this
is what the modeller chose.

The observations are built ONCE and shared. `sigma_r` enters the model and not
the data, so re-fetching per run would only risk the members differing by a data
revision landing mid-loop.
"""

import pickle
from dataclasses import replace
from pathlib import Path
from typing import Any

import arviz as az
import numpy as np
import pandas as pd
import xarray as xr

from src.models.rstar_rba.config import DEFAULT_OUTPUT_DIR, ModelConfig
from src.models.rstar_rba.estimate import build_model, build_observations, posterior_median
from src.models.ystar.base import SamplerConfig, get_fixed_constants, sample_model

# 0.05 to 0.15 are values a reasonable person could defend, which is what makes
# the spread between them structural uncertainty rather than a demonstration
# that the model is unidentified. 0.20 is deliberately past that: it is where
# the method visibly fails, `corr(base, cash rate)` reaching 0.96 so the base is
# little more than a smoothed cash rate and `lambda` is close to decoration.
# It is carried to SHOW the boundary, not as a candidate answer, and the notes
# quote the range over the first three. Zero is not included: that is the
# fixed-neutral regression, available as `--no-walk`.
DEFAULT_SIGMA_R_VALUES = (0.05, 0.10, 0.15, 0.20)

# The era the absorption question turns on: a sustained stance there is the one
# result the notes lean on, so it is the one to watch across the ensemble.
_WATCH_ERA = ("2016Q1", "2019Q4")


def _summarise(
    trace: az.InferenceData, frame: pd.DataFrame, sigma_r: float, band: float, anchor: float,
) -> dict[str, Any]:
    """Return one row of the ensemble table.

    The reported level is `neutral` and its deflated counterpart, never
    `prescribed`, which carries the inflation response on top.
    """
    index = frame.index
    if not isinstance(index, pd.PeriodIndex):
        index = pd.PeriodIndex(index, freq="Q")
    posterior = getattr(trace, "posterior", None)
    if not isinstance(posterior, xr.Dataset):
        raise TypeError("trace has no posterior group - did sampling complete?")
    lam = np.asarray(posterior["lambda"].values).ravel()
    neutral = posterior_median(trace, "neutral", index)
    residual = posterior_median(trace, "rule_residual", index)
    start, end = _WATCH_ERA
    return {
        "sigma_r": sigma_r,
        "lambda": float(lam.mean()),
        "lambda_pp": float(lam.mean()) / band,
        "neutral_nom": float(neutral.iloc[-1]),
        "neutral_real": float(neutral.iloc[-1]) - anchor,
        "resid_2016_19": float(residual.loc[start:end].mean()),
        "corr_neut_cash": float(neutral.corr(frame["r"])),
    }


def run_sigma_r_ensemble(
    config: ModelConfig | None = None,
    sampler_config: SamplerConfig | None = None,
    values: tuple[float, ...] = DEFAULT_SIGMA_R_VALUES,
    prefix: str = "rstar_rba",
    *,
    seed: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Sample the model once per `sigma_r` and return the paths and the table.

    The paths are posterior MEDIANS of real r*, one column per value. Medians
    rather than bands because the chart draws the envelope across assumptions;
    the within-assumption band comes from the default run and is drawn from the
    saved trace, not from here.
    """
    config = config or ModelConfig()
    sampler_config = sampler_config or SamplerConfig()
    if seed is not None:
        sampler_config.random_seed = seed

    frame, _sources = build_observations(config, verbose=False)
    index = frame.index
    if not isinstance(index, pd.PeriodIndex):
        index = pd.PeriodIndex(index, freq="Q")

    paths: dict[str, pd.Series] = {}
    bases: dict[str, pd.Series] = {}
    rows: list[dict[str, Any]] = []
    for value in values:
        print(f"\nsigma_r = {value:g}")
        member = replace(config, sigma_r=value, walk=True)
        model = build_model(frame, member, verbose=False)
        trace = sample_model(model, sampler_config)
        band = float(get_fixed_constants(model).get("band", 1.0))
        # `paths` is neutral in real terms and `bases` in nominal, so the two
        # charts differ only in units and in what they are drawn against.
        # Neither is `prescribed`, which carries the inflation response.
        bases[f"{value:g}"] = posterior_median(trace, "neutral", index)
        paths[f"{value:g}"] = bases[f"{value:g}"] - config.anchor
        rows.append(_summarise(trace, frame, value, band, config.anchor))

    path_frame = pd.DataFrame(paths, index=index)
    base_frame = pd.DataFrame(bases, index=index)
    table = pd.DataFrame(rows).set_index("sigma_r")
    _save(path_frame, base_frame, table, output_dir=config.output_dir, prefix=prefix)
    return path_frame, table


def _save(
    paths: pd.DataFrame,
    bases: pd.DataFrame,
    table: pd.DataFrame,
    output_dir: Path | str | None = None,
    prefix: str = "rstar_rba",
) -> None:
    """Persist the ensemble beside the trace, so charting need not re-sample."""
    directory = Path(output_dir) if output_dir else DEFAULT_OUTPUT_DIR
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / f"{prefix}_sigma_r_ensemble.pkl"
    with target.open("wb") as handle:
        pickle.dump({"paths": paths, "bases": bases, "table": table}, handle)
    print(f"\nSaved sigma_r ensemble to: {target}")


def load_ensemble(
    output_dir: Path | str | None = None,
    prefix: str = "rstar_rba",
) -> dict[str, pd.DataFrame] | None:
    """Return a saved ensemble, or None if the run has not been done.

    `bases` is absent from files written before the base paths were kept, so
    read it with `.get`: an older ensemble still charts r*, it just cannot draw
    the base alongside it.
    """
    directory = Path(output_dir) if output_dir else DEFAULT_OUTPUT_DIR
    target = directory / f"{prefix}_sigma_r_ensemble.pkl"
    if not target.exists():
        return None
    with target.open("rb") as handle:
        saved = pickle.load(handle)  # noqa: S301 — our own file
    if not isinstance(saved, dict):
        raise TypeError(f"{target} does not hold a sigma_r ensemble")
    return saved


def print_ensemble(table: pd.DataFrame) -> None:
    """Print the table the notes quote instead of a single conditional level."""
    print("\nsigma_r ensemble: what the imposed smoothness decides")
    print("-" * 70)
    print("  lambda is stable across it; the LEVEL and the era residual are not.")
    print(table.round(3).to_string())
