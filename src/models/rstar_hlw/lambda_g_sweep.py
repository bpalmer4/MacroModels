"""Sensitivity sweep on lambda_g, the sigma_g / sigma_ystar ratio.

WHY THIS EXISTS. HLW do not estimate the state variances; they fix the
signal-to-noise ratios by Stock-Watson median-unbiased estimation. This
implementation originally estimated both, and potential output came out more
volatile than GDP. Imposing sigma_ystar alone moved the volatility into trend
growth (sigma_g posterior 0.105 against a 0.04 prior scale). Imposing both at
HLW's US lambda_g of 0.053 moved it into the output gap instead, which then
read +4.4 across 2005-2019 and flattened the Phillips curve to 0.058 rather
than price that gap into inflation.

So the variance goes somewhere in every configuration, and lambda_g decides
WHERE. That makes it exactly the kind of imposed scalar this package exists to
be sceptical of, and the only honest way to choose one is to vary it and look.

THE GRID. 0.053 is HLW's own (US) value. 0.34 is what Australia's actual
growth slowdown implies: letting g fall ~1.5pp over 159 quarters needs sigma_g
of about 1.5/sqrt(159) = 0.119, which over 4 x 0.078 is 0.38, and the free
posterior of 0.105 gives 0.34. 0.15 sits between the two. `None` samples
sigma_g under its own HalfNormal(0.04) prior, for reference.

WHAT TO READ. Not r* first. The diagnostic that separates these runs is
whether the decomposition is a decomposition at all:

- `gap sd`, but NOT against `ystar_ustar`'s 0.42. An earlier version of this file said
  "anything above ~1.5 is not a cycle", which is wrong: the ystar family DEFINES the gap
  as c x (pi - 2.5) with c small, so its gap is a rescaled inflation deviation and is
  far smaller than an output gap. Checked against unemployment, HLW's amplitude is the
  defensible one (see MODEL_NOTES, "How big should the gap be"). Read `gap_2005_19`
  instead: a fifteen-year average should be near zero whatever the amplitude.
- `b_y`, the Phillips slope. If it collapses toward its 0.02 lower bound, the
  model is declining to explain inflation with the gap it just produced, which
  is a confession rather than a fit.
- `g first -> last`. Australian trend growth did fall over this sample. A run
  that holds g flat has put that decline in the cycle.

WHAT IT FOUND (Resolution A, run 2026-09-12 on the 1993Q1 default; the earlier
1986Q3 run in brackets). The grid splits in two on both samples, and the break is
between 0.053 and 0.15 rather than spread across the range:

| lambda_g | 0.053 | 0.15 | 0.34 | free |
|---|---|---|---|---|
| gap, 2005-19 mean | **2.41** (4.35) | 0.25 (0.78) | 0.00 (0.31) | -0.04 (0.34) |
| b_y | **0.110** (0.054) | 0.256 (0.156) | 0.274 (0.169) | 0.273 (0.169) |
| sigma_pi | **0.764** | 0.642 | 0.576 | 0.556 |
| g, first -> last | 2.96 -> 2.39 | 3.49 -> 1.89 | 3.75 -> 2.09 | 3.82 -> 2.24 |
| r* latest | **3.25** (2.75) | 2.02 (1.70) | 2.19 (1.71) | 2.25 (1.73) |

HLW's own 0.053 is the outlier on every line: the only setting that holds g
roughly flat, the only one putting the 2005-19 economy well above capacity,
and the only one where the Phillips curve gives up (b_y against a 0.02 lower
bound). It also moves r* by a full percentage point, which is worth stating
plainly: the TREND-GROWTH variance ratio decides r* here, before any of the
r* priors the other resolutions argue about.

0.34 and free agree closely on every column, so at 0.34 the constraint is not
binding and the data was already there. The real choice is binary, US-style
ratio or the data's own, not a dial. Hence the default: lambda_g off.

What no lambda_g fixes is the gap's own persistence, a_y1 + a_y2 around 0.94
whatever the ratio. On the old 1986Q3 sample that left the gap stuck near +4.5
through the 2022-26 disinflation; moving the default start to 1993Q1 fixed the
level but not the persistence. See MODEL_NOTES, "The sample start is the
biggest single choice".

Seed held at 42 throughout so lambda_g is the only varying input.

Run:
    uv run python -m src.models.rstar_hlw.lambda_g_sweep
"""

import numpy as np
import pandas as pd

from src.models.nairu.base import SamplerConfig, get_fixed_constants, sample_model
from src.models.rstar_hlw.estimate import build_model, save_results
from src.models.rstar_hlw.observations import build_observations
from src.models.rstar_hlw.results import load_results

# (lambda_g, label used in the prefix and the table)
LAMBDA_G_GRID: list[tuple[float | None, str]] = [
    (0.053, "hlw_us"),
    (0.15, "mid"),
    (0.34, "au_slowdown"),
    (None, "free"),
]

RESOLUTION = "A"


def _diagnostics(prefix: str, label: str) -> dict[str, object]:
    """Pull the numbers that decide whether a run's decomposition is usable."""
    results = load_results(prefix=prefix)
    posterior = results.trace["posterior"]

    def flat(name: str) -> np.ndarray:
        # xarray's .values, not pandas'.
        return np.asarray(posterior[name].values).ravel()

    gap = results.output_gap_median()
    potential = results.potential_median()
    g = results.trend_growth_median()
    r_star = results.r_star_median()

    a_r, a_y1, a_y2 = flat("a_r"), flat("a_y1"), flat("a_y2")
    sigma_is = flat("sigma_IS")
    divisor = 1 - a_y1 - a_y2

    diverging = np.asarray(results.trace["sample_stats"]["diverging"].values)

    return {
        "lambda_g": label,
        "sigma_g": results.constants.get("sigma_g", float("nan")),
        "pot_sd": potential.diff().std(),
        "gap_sd": gap.std(),
        "gap_2005_19": gap.loc["2005Q1":"2019Q4"].mean(),
        "b_y": float(np.median(flat("b_y"))),
        "sigma_pi": float(np.median(flat("sigma_pi"))),
        "g_first": g.iloc[0],
        "g_last": g.iloc[-1],
        "a_r": float(np.median(a_r)),
        "sigma_IS": float(np.median(sigma_is)),
        "signal": float(np.median(np.abs(a_r) / sigma_is)),
        "lr_slope": float(np.median(a_r / divisor)),
        "rstar_last": r_star.iloc[-1],
        "divergences": int(diverging.sum()),
    }


def main() -> None:
    """Re-estimate Resolution A once per lambda_g in the grid, then compare."""
    print("Building observations once (shared across sweep)...")
    obs, obs_index, chart_obs = build_observations(verbose=True)

    sampler_config = SamplerConfig(
        draws=10_000,
        tune=3_500,
        chains=5,
        cores=5,
        target_accept=0.90,
    )

    rows: list[dict[str, object]] = []

    for lambda_g, label in LAMBDA_G_GRID:
        prefix = f"rstar_hlw_{RESOLUTION}_lambda_g_{label}"
        shown = "free" if lambda_g is None else f"{lambda_g:.3f}"

        print()
        print("=" * 70)
        print(f"Resolution {RESOLUTION} with lambda_g = {shown}  (prefix = {prefix})")
        print("=" * 70)

        model = build_model(
            obs,
            resolution=RESOLUTION,
            lambda_g=lambda_g,
            obs_index=obs_index,
        )
        trace = sample_model(model, sampler_config)

        save_results(
            trace,
            obs,
            obs_index,
            constants=get_fixed_constants(model),
            chart_obs=chart_obs,
            prefix=prefix,
        )

        rows.append(_diagnostics(prefix, shown))

    table = pd.DataFrame(rows).set_index("lambda_g")
    print()
    print("=" * 70)
    print("lambda_g sweep — is the decomposition a decomposition?")
    print("=" * 70)
    print(table.T.to_string(float_format=lambda v: f"{v:.3f}"))
    print()
    print("Benchmark: b_y's prior lower bound is 0.02, so a b_y near it is the model")
    print("  declining to explain inflation with the gap it just produced.")


if __name__ == "__main__":
    main()
