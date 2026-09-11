"""Sweep `sigma_rstar`: HOW SLOW r* IS, the one genuinely open question.

Everything else in this model is asserted or estimated. This number is neither:
it decides how much of the output gap gets called "the neutral rate moved" and
how much gets called "something else happened", and nothing in the data can
settle it. The fit improves monotonically the faster r* is allowed to move,
with no interior optimum, because a faster r* slides each point along the x
axis until it meets the line. So the answer is a range across defensible
values, never a point.

UNITS: percentage points of r* per quarter. `rstar_rba` defends 0.05 to 0.15
for the same kind of number in a different model, which is the only external
reference available and a weak one.

WHAT TO READ OFF IT.

  `is_slope`  THE COUPLING TO WATCH. With the prior on r* in rate units the
      slope enters twice, so a bigger slope buys a more movable line and the
      likelihood inflates it. Measured previously: -0.032 at sigma_rstar 0.02
      against -0.383 at 0.10, with almost the same fit. Expect that pattern
      here, and do not read the slope from a single member.

  `corr_rstar_cash`  whether r* has become a smoothed policy rate. `rstar_rba`
      treats 0.96 as the point where its own method visibly fails.

  `rstar_sd`, `rstar_max_step`  whether the thing is still a NEUTRAL RATE. One
      that swings by points a quarter is not.

  `sigma_e`  the residual. Expect it WIDE: the equation claims output moves
      only with the rate gap, when fiscal policy, the terms of trade, world
      demand and mismeasured potential are all in there too. `rstar_hlw` puts
      the equivalent at 0.70 and the gap's own sd is 0.42, so anything near
      0.42 means the line explains nothing.
"""

import pickle
from dataclasses import replace
from pathlib import Path
from typing import Any

import arviz as az
import numpy as np
import pandas as pd

from src.models.rstar_invert.config import DEFAULT_OUTPUT_DIR, ModelConfig
from src.models.rstar_invert.estimate import build_model, posterior_median, scalar_draws
from src.models.rstar_invert.observations import build_observations
from src.models.ystar.base import SamplerConfig, sample_model

# PERCENTAGE POINTS of r* per quarter. 0.05 to 0.15 is the range `rstar_rba`
# defends for the same kind of number. 0.30 and 0.50 are deliberately past it,
# to show the branch where r* swallows the residual and stops being a trend at
# all. 0.02 is the other end: near enough to a fixed r* to show what the slope
# can do unaided. Zero is not included; that is `--rstar-form constant`.
DEFAULT_SIGMA_RSTAR = (0.02, 0.05, 0.10, 0.15, 0.30, 0.50)

# The era the stance question turns on, matching `rstar_rba`'s watch window so
# the two models' era statements are about the same quarters.
_WATCH_ERA = ("2016Q1", "2019Q4")

# Lags to sweep. The `is_curve` bench measures its own slope turning negative
# around lag 4-5 on the full sample (+0.081 at 0, falling to -0.015 by 6), and
# on the sample that drops 2008Q4-2021Q3 it strengthens out to -0.139 at lag 5.
# So this is where the theory says to look: transmission should be weak on
# impact and build over three to six quarters.
DEFAULT_LAGS = (1, 2, 3, 4, 5)


# A posterior whose upper tail reaches this close to zero is pressed against
# the truncation, which means the likelihood wanted a positive slope and the
# theory constraint stopped it. Such a row is a REJECTION of the mechanism at
# that specification, not a measurement of a small negative slope.
BOUNDARY_TOL = 0.005


def _summarise(
    trace: az.InferenceData,
    index: pd.PeriodIndex,
    real_cash: pd.Series,
    ident: dict[str, Any],
) -> dict[str, Any]:
    """Return one row of a sweep, identified by `ident`."""
    rstar = posterior_median(trace, "rstar", index)
    stance = posterior_median(trace, "stance", index)
    slope = scalar_draws(trace, "is_slope")
    sigma_e = scalar_draws(trace, "sigma_e")
    start, end = _WATCH_ERA
    slope_q95 = float(np.percentile(slope, 95))

    return {
        **ident,
        "is_slope": float(np.median(slope)),
        # How close the posterior's near-zero tail gets to the bound, and
        # whether it is effectively touching it.
        "slope_q95": slope_q95,
        "on_boundary": bool(slope_q95 > -BOUNDARY_TOL),
        "rstar_latest": float(rstar.iloc[-1]),
        "rstar_mean": float(rstar.mean()),
        "rstar_sd": float(rstar.std()),
        "rstar_range": float(rstar.max() - rstar.min()),
        "rstar_max_step": float(rstar.diff().abs().max()),
        "corr_rstar_cash": float(rstar.corr(real_cash)),
        "sigma_e": float(np.median(sigma_e)),
        "stance_2016_19": float(stance.loc[start:end].mean()),
    }


def run_ensemble(
    config: ModelConfig | None = None,
    sampler_config: SamplerConfig | None = None,
    values: tuple[float, ...] = DEFAULT_SIGMA_RSTAR,
    prefix: str = "rstar_invert",
    *,
    seed: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Sample once per `sigma_rstar` and return the paths and the table.

    The observations are built ONCE and shared: `sigma_rstar` enters the model and
    not the data, so re-fetching per member would only risk the members
    differing by a revision landing mid-loop.
    """
    config = config or ModelConfig()
    sampler_config = sampler_config or SamplerConfig(draws=2_000, tune=2_000, log_likelihood=False)
    if seed is not None:
        sampler_config.random_seed = seed

    print("Building observations once, shared across the sweep...")
    data = build_observations(config, verbose=True)
    index = data.index
    real_cash = data.frame["real_cash"]

    paths: dict[str, pd.Series] = {}
    rows: list[dict[str, Any]] = []

    for position, sigma_rstar in enumerate(values, start=1):
        print()
        print("=" * 70)
        print(f"[{position}/{len(values)}]  sigma_rstar = {sigma_rstar:g}  (pp per quarter)")
        print("=" * 70)

        member = replace(
            config, sigma_rstar=sigma_rstar, free_sigma_rstar=False, rstar_form="walk",
        )
        model = build_model(data, member, verbose=False)
        trace = sample_model(model, sampler_config)

        paths[f"sigma_rstar = {sigma_rstar:g}"] = posterior_median(trace, "rstar", index)
        row = _summarise(trace, index, real_cash, {"sigma_rstar": sigma_rstar})
        rows.append(row)
        print(f"  is_slope {row['is_slope']:+.3f}, r* sd {row['rstar_sd']:.2f}, "
              f"corr with cash {row['corr_rstar_cash']:+.3f}, sigma_e {row['sigma_e']:.3f}")

    path_frame = pd.DataFrame(paths, index=index)
    table = pd.DataFrame(rows)

    directory = Path(config.output_dir) if config.output_dir else DEFAULT_OUTPUT_DIR
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / f"{prefix}_ensemble.pkl").open("wb") as handle:
        pickle.dump({"paths": path_frame, "table": table}, handle)
    print(f"\nSaved ensemble to: {directory / f'{prefix}_ensemble.pkl'}")

    return path_frame, table


def run_lag_sweep(
    config: ModelConfig | None = None,
    sampler_config: SamplerConfig | None = None,
    lags: tuple[int, ...] = DEFAULT_LAGS,
    prefix: str = "rstar_invert",
    *,
    seed: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Sample once per rate lag and return the paths and the table.

    WHY THIS SWEEP EXISTS. Monetary transmission should be weak on impact and
    build over three to six quarters, while a policy REACTION to the economy is
    strongest contemporaneously. So the short lags are where simultaneity lives
    and the long lags are where transmission should appear if it is there at
    all. If the theory holds anywhere in Australian data, it holds at lag 4 or
    5, and if the posterior is still on the truncation boundary out there, no
    lag rescues it.

    Unlike the smoothness sweep, the observations are rebuilt per member: the
    lag is applied when the rate series is shifted, so it is part of the data
    construction rather than of the model.
    """
    config = config or ModelConfig()
    sampler_config = sampler_config or SamplerConfig(draws=2_000, tune=2_000, log_likelihood=False)
    if seed is not None:
        sampler_config.random_seed = seed

    paths: dict[str, pd.Series] = {}
    rows: list[dict[str, Any]] = []

    for position, lag in enumerate(lags, start=1):
        print()
        print("=" * 70)
        print(f"[{position}/{len(lags)}]  rate lag = {lag}")
        print("=" * 70)

        member = replace(config, rate_lags=(lag,))
        data = build_observations(member, verbose=False)
        model = build_model(data, member, verbose=False)
        trace = sample_model(model, sampler_config)

        index = data.index
        paths[f"lag {lag}"] = posterior_median(trace, "rstar", index)
        row = _summarise(trace, index, data.frame["real_cash"], {"rate_lag": lag})
        rows.append(row)
        flag = "ON BOUNDARY (wrong sign wanted)" if row["on_boundary"] else "off the boundary"
        print(f"  is_slope {row['is_slope']:+.3f}, q95 {row['slope_q95']:+.4f}  {flag}")
        print(f"  r* sd {row['rstar_sd']:.2f}, corr with cash {row['corr_rstar_cash']:+.3f}, "
              f"sigma_e {row['sigma_e']:.3f}")

    path_frame = pd.DataFrame(paths)
    table = pd.DataFrame(rows)

    directory = Path(config.output_dir) if config.output_dir else DEFAULT_OUTPUT_DIR
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / f"{prefix}_lagsweep.pkl").open("wb") as handle:
        pickle.dump({"paths": path_frame, "table": table}, handle)
    print(f"\nSaved lag sweep to: {directory / f'{prefix}_lagsweep.pkl'}")

    return path_frame, table


def print_lag_table(table: pd.DataFrame) -> None:
    """Print the lag sweep in the order it should be read."""
    print("\n" + "=" * 94)
    print("THE LAG SWEEP: does the mechanism appear anywhere the theory says it should?")
    print("=" * 94)
    print(f"{'lag':>4} {'is_slope':>9} {'q95':>9} {'boundary':>10} {'r* sd':>7} "
          f"{'corr cash':>10} {'sigma_e':>8} {'r* last':>9}")
    for _, row in table.iterrows():
        flag = "TOUCHING" if row["on_boundary"] else "clear"
        print(f"{int(row['rate_lag']):>4} {row['is_slope']:>9.3f} {row['slope_q95']:>9.4f} "
              f"{flag:>10} {row['rstar_sd']:>7.2f} {row['corr_rstar_cash']:>10.3f} "
              f"{row['sigma_e']:>8.3f} {row['rstar_latest']:>9.2f}")
    print("=" * 94)
    print("TOUCHING: the likelihood wanted a POSITIVE slope and the theory constraint")
    print("          stopped it, so the number is the bound, not a measurement.")
    print("          THIS IS NOT EVIDENCE AGAINST NK TRANSMISSION. It says this")
    print("          specification cannot see it. The RBA raises rates when the gap")
    print("          is positive, which induces positive covariation whatever the")
    print("          true structural slope, so a boundary result is equally")
    print("          consistent with a strong channel swamped by the reaction")
    print("          function, with the gap being half inflation, with sigma_rstar")
    print("          too tight, or with the lag being wrong.")
    print("clear   : the data supports a negative slope on its own at that lag.")


def load_lag_sweep(
    output_dir: Path | str | None = None,
    prefix: str = "rstar_invert",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load a completed lag sweep: (paths, table)."""
    directory = Path(output_dir) if output_dir else DEFAULT_OUTPUT_DIR
    with (directory / f"{prefix}_lagsweep.pkl").open("rb") as handle:
        saved = pickle.load(handle)  # noqa: S301 — our own file
    return saved["paths"], saved["table"]


def load_ensemble(
    output_dir: Path | str | None = None,
    prefix: str = "rstar_invert",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load a completed ensemble: (paths, table)."""
    directory = Path(output_dir) if output_dir else DEFAULT_OUTPUT_DIR
    with (directory / f"{prefix}_ensemble.pkl").open("rb") as handle:
        saved = pickle.load(handle)  # noqa: S301 — our own file
    return saved["paths"], saved["table"]


def print_table(table: pd.DataFrame) -> None:
    """Print the sweep in the order it should be read."""
    print("\n" + "=" * 92)
    print("HOW SLOW IS r*: what sigma_rstar buys, since nothing measures it")
    print("=" * 92)
    print(f"{'sigma_rstar':>12} {'is_slope':>9} {'r* last':>9} {'r* sd':>7} {'r* range':>9} "
          f"{'max step':>9} {'corr cash':>10} {'sigma_e':>8} {'stance 16-19':>13}")
    for _, row in table.iterrows():
        print(f"{row['sigma_rstar']:>12.3f} {row['is_slope']:>9.3f} {row['rstar_latest']:>9.2f} "
              f"{row['rstar_sd']:>7.2f} {row['rstar_range']:>9.2f} {row['rstar_max_step']:>9.2f} "
              f"{row['corr_rstar_cash']:>10.3f} {row['sigma_e']:>8.3f} "
              f"{row['stance_2016_19']:>13.2f}")
    print("=" * 92)
    print("corr cash near 1 : r* is a smoothed real cash rate, not a neutral rate")
    print("sigma_e near 0.70: the asserted slope absorbed nothing (rstar_hlw's value)")
    print("is_slope moving  : the slope and r*'s speed are coupled. With the prior on")
    print("                   r* in rate units the line's vertical freedom is")
    print("                   |is_slope| x sigma_rstar, so a bigger slope buys a more")
    print("                   movable line and the likelihood inflates it. Measured:")
    print("                   -0.032 at 0.05 against -0.383 at 0.10. Do not read the")
    print("                   slope from a single member.")
    print()
    print("NOTE the switch between 0.05 and 0.10: everything changes at once, so this")
    print("is two regimes rather than a dial. And sigma_e falls monotonically with no")
    print("interior optimum, so the data cannot choose. The 2016-19 stance flips sign")
    print("across the switch, -3.13 to +0.52, which is what the choice actually costs.")
