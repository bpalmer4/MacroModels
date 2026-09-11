"""Inject a known policy stance and see how much of it comes back.

The model's central limitation is that a long enough departure from its own rule
is absorbed into neutral, so it cannot tell "the RBA chose to be tight" from "the
RBA thought neutral had risen". The notes inferred the horizon of that from
`sigma_r x sqrt(T)`, which is how far the base COULD wander under its prior. That
is not the same as how much of an imposed stance it WOULD take, because the
likelihood decides the second. Reading the first as the second is what produced
the 2016-19 claim the `sigma_r` ensemble withdrew.

This module measures the second. Add a known constant to the cash rate over a
window of L years, re-estimate, and see where it lands: in the residual, where
it belongs, or in the base, where it is reported as neutral having moved.

NOT AN ECONOMIC SIMULATION, and the distinction matters. Inflation is held at
what it actually did while the rate is moved, so this is not "what if the RBA
had been tighter" - real tightening would have pulled inflation down, and the
model would then see a rate rise beside a falling inflation gap, which is a
different and easier signal. What is measured here is narrower and is the thing
actually in question: given a rate path that departs from the rule by a known
amount for a known time, can the estimator still see the departure?

THE NORMALISATION HAS TO BE HANDLED. Only the whole-sample mean of the residual
is pinned, by the base's free starting level. Adding a positive shift over part
of the sample raises the whole-sample mean of the cash rate, so `base_0` moves to
absorb some of it whatever the duration, and the raw within-window residual
understates what was recovered. The contrast - within-window change less
outside-window change - is the measure that is not contaminated by that, and is
the one to read. Both are reported.
"""

import pickle
from pathlib import Path
from typing import Any

import arviz as az
import numpy as np
import pandas as pd
import xarray as xr

from src.models.rstar_rba.config import DEFAULT_OUTPUT_DIR, ModelConfig
from src.models.rstar_rba.estimate import build_model, build_observations, posterior_median
from src.models.ystar.base import SamplerConfig, sample_model

# Durations tested, in years. 4 is the one that matters most: it is the 2016-19
# window whose result the sigma_r ensemble withdrew.
INJECTION_YEARS = (1, 2, 4, 6, 10)

# Size of the injected stance, in percentage points. A full point is large
# enough to be unmistakable if the estimator can see anything at all, and is the
# size the notes' own discussion used.
INJECTION_SIZE = 1.00

# Every window ends here and extends backwards. Pre-COVID, so no window touches
# the effective-lower-bound quarters, and the four-year case lands exactly on
# 2016-19.
INJECTION_END = "2019Q4"


def _index_of(frame: pd.DataFrame) -> pd.PeriodIndex:
    index = frame.index
    if not isinstance(index, pd.PeriodIndex):
        index = pd.PeriodIndex(index, freq="Q")
    return index


def _posterior(trace: az.InferenceData) -> xr.Dataset:
    group = getattr(trace, "posterior", None)
    if not isinstance(group, xr.Dataset):
        raise TypeError("trace has no posterior group - did sampling complete?")
    return group


def _window(index: pd.PeriodIndex, years: int, end: str) -> np.ndarray:
    """Return a boolean mask for the L-year window ending at `end`."""
    last = pd.Period(end, freq="Q")
    first = last - (years * 4 - 1)
    return np.asarray((index >= first) & (index <= last))


def _fit(
    frame: pd.DataFrame, config: ModelConfig, sampler_config: SamplerConfig,
) -> tuple[pd.Series, pd.Series, float]:
    """Sample once and return the base, the residual and `lambda`."""
    index = _index_of(frame)
    model = build_model(frame, config, verbose=False)
    trace = sample_model(model, sampler_config)
    lam = float(np.asarray(_posterior(trace)["lambda"].values).mean())
    return posterior_median(trace, "neutral", index), posterior_median(trace, "rule_residual", index), lam


def run_injection_test(
    config: ModelConfig | None = None,
    sampler_config: SamplerConfig | None = None,
    years: tuple[int, ...] = INJECTION_YEARS,
    *,
    size: float = INJECTION_SIZE,
    end: str = INJECTION_END,
    prefix: str = "rstar_rba",
    seed: int | None = None,
) -> pd.DataFrame:
    """Add a known stance over windows of several lengths and see what returns."""
    config = config or ModelConfig()
    sampler_config = sampler_config or SamplerConfig()
    if seed is not None:
        sampler_config.random_seed = seed

    frame, _sources = build_observations(config, verbose=False)
    index = _index_of(frame)

    print(f"\nBaseline (no injection), then +{size:.2f}pp over windows ending {end}")
    base_0, stance_0, lambda_0 = _fit(frame, config, sampler_config)

    rows: list[dict[str, Any]] = []
    for span in years:
        mask = _window(index, span, end)
        if not mask.any():
            raise ValueError(f"the {span}-year window ending {end} falls outside the sample")
        injected = frame.copy()
        injected.loc[mask, "r"] = injected.loc[mask, "r"] + size
        print(f"  {span:>2}y: {index[mask][0]} to {index[mask][-1]}")
        base_1, stance_1, lambda_1 = _fit(injected, config, sampler_config)

        d_base = base_1 - base_0
        d_stance = stance_1 - stance_0
        # The contrast. Within-window less outside-window, so the level shift
        # that `base_0` takes whatever the duration cancels out.
        inside = float(d_stance[mask].mean())
        outside = float(d_stance[~mask].mean())
        rows.append({
            "years": span,
            "recovered": (inside - outside) / size,
            "recovered_raw": inside / size,
            "absorbed_neut": float(d_base[mask].mean()) / size,
            "leak_outside": outside / size,
            "lambda": lambda_1,
            "lambda_shift": lambda_1 - lambda_0,
        })

    table = pd.DataFrame(rows).set_index("years")
    _save(table, lambda_0, size=size, end=end, output_dir=config.output_dir, prefix=prefix)
    return table


def _save(
    table: pd.DataFrame,
    baseline_lambda: float,
    *,
    size: float,
    end: str,
    output_dir: Path | str | None = None,
    prefix: str = "rstar_rba",
) -> None:
    """Persist the result beside the trace, so charting need not re-sample."""
    directory = Path(output_dir) if output_dir else DEFAULT_OUTPUT_DIR
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / f"{prefix}_injection.pkl"
    with target.open("wb") as handle:
        pickle.dump(
            {"table": table, "baseline_lambda": baseline_lambda, "size": size, "end": end},
            handle,
        )
    print(f"\nSaved injection test to: {target}")


def load_injection(
    output_dir: Path | str | None = None,
    prefix: str = "rstar_rba",
) -> dict[str, Any] | None:
    """Return a saved injection test, or None if it has not been run."""
    directory = Path(output_dir) if output_dir else DEFAULT_OUTPUT_DIR
    target = directory / f"{prefix}_injection.pkl"
    if not target.exists():
        return None
    with target.open("rb") as handle:
        saved = pickle.load(handle)  # noqa: S301 — our own file
    if not isinstance(saved, dict):
        raise TypeError(f"{target} does not hold an injection test")
    return saved


def print_injection(saved: dict[str, Any]) -> None:
    """Print how much of the imposed stance came back as a stance."""
    table = saved["table"]
    print(f"\nInjection test: +{saved['size']:.2f}pp over windows ending {saved['end']}")
    print("-" * 70)
    print("  recovered = what the rule residual reports, as a fraction of what was imposed.")
    print("  absorbed  = what neutral took instead, and so reports as neutral having moved.")
    print(table.round(3).to_string())
