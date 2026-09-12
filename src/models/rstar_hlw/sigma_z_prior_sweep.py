"""Prior-sensitivity sweep on sigma_z: does the data say how fast r* moves.

THE TEST. Buncic-Pagan-Robinson (2023): when the latent shocks meet or exceed
the identifying observables, the latent is not point-identified and the
posterior is the prior projected through the structural model. The operational
version is simple. Vary the prior and watch:

- if the r* credible interval scales roughly 1:1 with the prior scale, the
  width is the prior's and the data contributed nothing;
- if it stops growing, the likelihood is binding and the data has a view.

WHY RUN IT AGAIN. MODEL_NOTES.md reports this test on Resolution E, where
sigma_z is a fixed constant, and it scaled ~1:1 (r* CI 1.27pp to 5.35pp as
sigma_z ran 0.05 to 0.50). It has never been run on Resolution A, and never on
any resolution since potential output was repaired (sigma_ystar imposed, the
lockdown quarters excluded). That repair tripled the per-quarter signal
|a_r|/sigma_IS from 0.064 to 0.193, so the test is worth redoing: the old
result was measured on a decomposition the notes themselves disown.

WHAT IS BEING VARIED. In Resolution A sigma_z is a free parameter with a
HalfNormal prior, so what moves here is that prior's SCALE, not sigma_z itself.
0.10 is the shipped default. A HalfNormal(s) has mean 0.798s, so the grid spans
a factor of 12 in how fast r* is a priori allowed to wander.

WHAT TO READ, AND WHAT NOT TO. `ci_per_scale` looks like the answer and is
not: the r* credible interval is dominated by uncertainty in g, so it moves
sublinearly with the sigma_z prior even when z is completely unidentified.
Reading a falling `ci_per_scale` as "the likelihood binds" is a trap, and the
first version of this file fell into it.

The clean reads are:

- `sigma_z_post_mean` against `sigma_z_prior_mean` (the `post_over_prior`
  row). Near 1 means the posterior on sigma_z IS the prior.
- `z_span` against the prior scale, in the follow-up table below. z is the
  only part of r* the IS curve is supposed to identify, so if its wander is
  proportional to its prior, nothing has been identified.
- `rstar_latest` across the grid. If a prior that nothing measures moves the
  answer, the answer is the prior's.

WHAT IT FOUND (run 2026-09-12, Resolution A, 1993Q1 default; the earlier 1986Q3
run in brackets, and the verdict is the same on both):

| prior scale | 0.025 | 0.05 | 0.10 | 0.30 |
|---|---|---|---|---|
| sigma_z post / prior mean | 1.08 (1.04) | 1.13 (1.12) | 1.17 (1.10) | 1.26 (1.58) |
| z span | 0.075 (0.057) | 0.074 (0.105) | 0.269 (0.300) | 1.491 (2.841) |
| corr(r*, g) | 1.000 | 1.000 | 0.998 | 0.745 (0.418) |
| r* latest | 2.21 (1.61) | 2.23 (1.62) | 2.25 (1.73) | 3.10 (3.14) |
| sigma_IS | 0.5664 | 0.5627 | 0.5593 | 0.5559 |

The last row is the one that matters most and the script does not print it: 20x the
wander buys a 1.9% improvement in the IS fit and none at all in the Phillips fit.
The likelihood is flat along the whole ridge, so sigma_z is not choosing how fast
r* moves, it is choosing which answer to report from a set the data cannot separate.

z is not identified. Its posterior scale is its prior scale, its wander is
proportional to the rope it is given, and past 0.10 it is superlinear: 3x the
prior buys 9.5x the wander, because nothing in the likelihood is pulling the
other way. Below that, r* and g correlate at 1.000 and r* is just g shifted
down ~0.30, so the IS curve contributes nothing to the r* PATH.

And the level goes with it: r* latest runs 1.61 to 3.14 across the grid. The
apparent stability at 0.025 to 0.10 is the prior forbidding movement, not the
likelihood pinning it down, which is the opposite of identification.

The repair to potential output tripled the per-quarter IS signal (0.064 to
0.193) and that was not enough. NOTE FOR MODEL_NOTES: its claim that the r*
CI "scales linearly" with the sigma_z prior is not what this measures; the CI
scales sublinearly while z itself scales superlinearly. The conclusion is
unchanged and the mechanism is stated wrongly.

Seed held at 42 throughout so the prior scale is the only varying input.

Run:
    uv run python -m src.models.rstar_hlw.sigma_z_prior_sweep
"""

import numpy as np
import pandas as pd

from src.models.nairu.base import SamplerConfig, get_fixed_constants, sample_model
from src.models.rstar_hlw.estimate import build_model, save_results
from src.models.rstar_hlw.observations import build_observations
from src.models.rstar_hlw.results import load_results

SIGMA_Z_PRIOR_GRID = [0.025, 0.05, 0.10, 0.30]

RESOLUTION = "A"

# HalfNormal(s) has mean s * sqrt(2/pi).
_HALFNORMAL_MEAN = np.sqrt(2 / np.pi)


def _diagnostics(prefix: str, scale: float) -> dict[str, float]:
    """Width of the r* answer, and whether it is the prior's width."""
    results = load_results(prefix=prefix)
    sigma_z = np.asarray(results.trace["posterior"]["sigma_z"].values).ravel()

    r_star = results.r_star_posterior()
    ci_width = (r_star.quantile(0.95, axis=1) - r_star.quantile(0.05, axis=1)).mean()
    median_path = r_star.median(axis=1)

    # z is r* less g, and is the only part of r* the IS curve is supposed to
    # identify. Its wander against the prior scale is the clean diagnostic; the
    # r* CI is not, because g's own uncertainty dominates it.
    g = results.trend_growth_median()
    z = median_path - g

    return {
        "prior_scale": scale,
        "sigma_z_prior_mean": scale * _HALFNORMAL_MEAN,
        "sigma_z_post_mean": float(sigma_z.mean()),
        "post_over_prior": float(sigma_z.mean()) / (scale * _HALFNORMAL_MEAN),
        "z_span": float(z.max() - z.min()),
        "z_span_per_scale": float(z.max() - z.min()) / scale,
        "corr_rstar_g": float(median_path.corr(g)),
        "rstar_ci_width": float(ci_width),
        "rstar_span": float(median_path.max() - median_path.min()),
        "rstar_latest": float(median_path.iloc[-1]),
    }


def main() -> None:
    """Re-estimate Resolution A once per sigma_z prior scale, then compare."""
    print("Building observations once (shared across sweep)...")
    obs, obs_index, chart_obs = build_observations(verbose=True)

    sampler_config = SamplerConfig(
        draws=10_000,
        tune=3_500,
        chains=5,
        cores=5,
        target_accept=0.90,
    )

    rows: list[dict[str, float]] = []

    for scale in SIGMA_Z_PRIOR_GRID:
        suffix = f"{round(scale * 1000):04d}"  # 0.025 -> "0025"
        prefix = f"rstar_hlw_{RESOLUTION}_sigma_z_prior_{suffix}"

        print()
        print("=" * 70)
        print(f"Resolution {RESOLUTION}, sigma_z ~ HalfNormal({scale})  (prefix = {prefix})")
        print("=" * 70)

        model = build_model(
            obs,
            resolution=RESOLUTION,
            sigma_z_prior=scale,
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

        rows.append(_diagnostics(prefix, scale))

    table = pd.DataFrame(rows).set_index("prior_scale")
    print()
    print("=" * 70)
    print("sigma_z prior sweep — is the width of r* the data's, or the prior's?")
    print("=" * 70)
    print(table.T.to_string(float_format=lambda v: f"{v:.3f}"))
    print()
    print("post_over_prior near 1 => the posterior on sigma_z is just the prior.")
    print("z_span_per_scale flat or rising => z's wander is whatever rope its prior")
    print("  gives it, which is non-identification. corr_rstar_g near 1 => r* is g,")
    print("  and the IS curve is contributing nothing to the path.")
    print("rstar_ci_width is NOT the diagnostic: g's uncertainty dominates it, so it")
    print("  moves sublinearly with the prior even when z is wholly unidentified.")


if __name__ == "__main__":
    main()
