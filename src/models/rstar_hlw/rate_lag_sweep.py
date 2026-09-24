"""Sweep the IS curve's rate lag to test whether a_r is weak or endogenous.

THE QUESTION. Moving the sample start from 1986Q3 to 1993Q1 halved `a_r`, from
−0.112 to −0.044, and the per-quarter signal `a_r`/`sigma_IS` from 0.193 to
0.079. There are two explanations and they have opposite implications.

1. LOST VARIATION. The 1980s cash rate reached 18.16%; after 1993 the range is
   0.10 to 7.50. Less movement in the regressor, less precision. On this
   reading a small `a_r` says nothing about the economy.

2. ENDOGENEITY FROM SUCCESSFUL POLICY. The post-1993 sample is exactly the
   period in which the RBA is actively setting the cash rate to lean against
   the output gap. The rate gap is then not exogenous: it moves BECAUSE the
   gap moves, which biases `a_r` toward zero. On this reading a small `a_r` in
   the inflation-targeting era is evidence of effective stabilisation, not of
   weak transmission, and the two are near-impossible to separate in a single
   equation.

THE TEST. A regressor further from t carries less of the RBA's response to
contemporaneous conditions, so under (2) `|a_r|` should STRENGTHEN as the lag
lengthens. Under (1) the lag should not matter much: precision is precision at
any horizon.

This is what `is_curve` found in reduced form, where the strongest relationship
is contemporaneous and POSITIVE, i.e. the reaction function rather than
transmission, and it is why the default lag here is t−6 rather than HLW's
averaged (t−1, t−2). This sweep asks whether 6 is far enough on the current
sample.

WHAT TO READ. `a_r` and `signal` down the row. A monotone strengthening with
the lag is the endogeneity signature. Watch `long_run_slope` too: it is the
quantity a believer in monetary transmission would have a prior about, and
values far beyond −1 are a warning that persistence is doing the work rather
than the rate channel.

CAVEAT. A longer lag is not a free fix for simultaneity. It weakens the
correlation with the RBA's response, but it also weakens the correlation with
everything else, and at long enough lags the regressor is close to noise. Read
a peak in `|a_r|` as informative and a monotone rise out to the end of the grid
as a reason to distrust the grid.

Seed held at 42 throughout so the lag is the only varying input.

Run:
    uv run python -m src.models.rstar_hlw.rate_lag_sweep
"""

import numpy as np
import pandas as pd

from src.models.common.model_constants import get_dictionary
from src.models.nairu.base import SamplerConfig, sample_model
from src.models.rstar_hlw.estimate import build_model, save_results
from src.models.rstar_hlw.observations import build_observations
from src.models.rstar_hlw.results import load_results

# None is HLW's own averaged (t-1, t-2) shape; the integers are single lags.
RATE_LAG_GRID: list[int | None] = [None, 1, 2, 4, 6, 8, 10]

RESOLUTION = "A"


def _diagnostics(prefix: str, label: str) -> dict[str, float]:
    """Pull the rate channel's strength and what it implies for the gap."""
    results = load_results(prefix=prefix)
    post = results.trace["posterior"]

    def flat(name: str) -> np.ndarray:
        return np.asarray(post[name].values).ravel()

    a_r, a_y1, a_y2 = flat("a_r"), flat("a_y1"), flat("a_y2")
    sigma_is = flat("sigma_IS")
    gap = results.output_gap_median()
    r_star = results.r_star_median()

    return {
        "lag": label,
        "a_r": float(np.median(a_r)),
        "a_r_lo": float(np.quantile(a_r, 0.05)),
        "a_r_hi": float(np.quantile(a_r, 0.95)),
        "sigma_IS": float(np.median(sigma_is)),
        "signal": float(np.median(np.abs(a_r) / sigma_is)),
        "persistence": float(np.median(a_y1 + a_y2)),
        "long_run_slope": float(np.median(a_r / (1 - a_y1 - a_y2))),
        "gap_2026Q2": float(gap["2026Q2"]),
        "gap_sd": float(gap.std()),
        "rstar_latest": float(r_star.iloc[-1]),
    }


def main() -> None:
    """Re-estimate Resolution A once per rate lag, then compare."""
    print("Building observations once (shared across sweep)...")
    obs, obs_index, chart_obs = build_observations(verbose=True)

    sampler_config = SamplerConfig(
        draws=10_000, tune=3_500, chains=5, cores=5, target_accept=0.90,
    )

    rows: list[dict[str, float]] = []

    for lag in RATE_LAG_GRID:
        label = "avg(1,2)" if lag is None else f"t-{lag}"
        suffix = "avg" if lag is None else f"{lag:02d}"
        prefix = f"rstar_hlw_{RESOLUTION}_ratelag_{suffix}"

        print()
        print("=" * 70)
        print(f"Resolution {RESOLUTION}, rate lag {label}  (prefix = {prefix})")
        print("=" * 70)

        model = build_model(
            obs, resolution=RESOLUTION, rate_lag=lag, obs_index=obs_index,
        )
        trace = sample_model(model, sampler_config)
        save_results(
            trace, obs, obs_index,
            constants=get_dictionary(model),
            chart_obs=chart_obs,
            prefix=prefix,
        )
        rows.append(_diagnostics(prefix, label))

    table = pd.DataFrame(rows).set_index("lag")
    print()
    print("=" * 70)
    print("rate lag sweep — is a_r weak, or endogenous to the RBA's response?")
    print("=" * 70)
    print(table.T.to_string(float_format=lambda v: f"{v:.3f}"))
    print()
    print("|a_r| strengthening with the lag => the short-lag estimates are")
    print("  contaminated by the RBA leaning against the gap, and a small a_r is")
    print("  evidence of stabilisation rather than of weak transmission.")
    print("|a_r| flat across lags => it really is just lost rate variation.")


if __name__ == "__main__":
    main()
