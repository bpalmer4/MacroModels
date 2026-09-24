"""Test whether the output gap closes when r* is handed to the model as data.

THE QUESTION. On the repaired decomposition, HLW's output gap sat between +3.9
and +5.1 right through the 2022-26 disinflation, while trimmed mean inflation
fell 6.8 -> 2.7 and unemployment rose 3.50 -> 4.35. Okun says a rise in
unemployment of that size should have closed the gap by 1.7pp; HLW closed it by
0.56. Decomposing the IS curve says why:

    the rate gap NEVER TURNS POSITIVE. Real cash peaks at 1.58 while the
    model's own r* is 1.73, so on its own account policy never reached
    neutral, and the rate term ADDS to the gap in every quarter of the
    tightening cycle.

That happens because canonical HLW's r* is g plus a level, and Australian trend
growth is around 1.9. Other packages here disagree: `rstar_bonds` reads 0.87
real at 2026Q2 and `rstar_rba` around 0.5, either of which makes real cash of
1.58 genuinely restrictive.

So this script asks a single question: **is the r* level what keeps the gap
open, or would the gap stay stuck anyway?** It hands the model an external r*
as DATA and re-estimates everything else.

THIS IS NOT AN ESTIMATE OF r*, AND NOT A NINTH RESOLUTION. r* is an input here.
The output that matters is the GAP and `a_y1`, not r*.

WHAT TO READ:

- `gap_2026Q2` and `gap_change_since_2022Q4`. If the gap now closes as the
  economy cools, the r* level was the binding constraint. If it stays near +4,
  the problem is the gap's own persistence and no r* fixes it.
- `a_y1 + a_y2`. The baseline is 0.953, a half-life of 14 quarters. A materially
  lower number means the rate channel was doing the work the AR term had been
  standing in for.
- `rate_term_sd` against the baseline's 0.237. If the rate term is still tiny,
  the channel is too weak to matter whatever r* is.

SAMPLE. `rstar_bonds` starts 1993Q1, so these runs do too, against the
baseline's 1986Q3. The 1993Q1 start is what MODEL_NOTES proposes for NAIRU
integration anyway, and it drops the pre-inflation-target regime. The baseline
is re-run on the same sample so the comparison is like for like.

Run:
    uv run python -m src.models.rstar_hlw.given_rstar_test
"""

import numpy as np
import pandas as pd

from src.models.common.model_constants import get_dictionary
from src.models.nairu.base import SamplerConfig, sample_model
from src.models.rstar_bonds.results import load_results as load_bonds
from src.models.rstar_hlw.estimate import build_model, save_results
from src.models.rstar_hlw.observations import build_observations
from src.models.rstar_hlw.results import load_results

START = "1993Q1"
RESOLUTION = "A"
RATE_LAG = 6


def _external_rstar(obs_index: pd.PeriodIndex) -> pd.Series:
    """Bond-market r*, real, aligned to the estimation sample.

    Raises rather than filling: a silently extrapolated r* would be an
    assumption wearing the costume of an external anchor, which is the whole
    thing this test is meant to avoid.
    """
    series = load_bonds(prefix="rstar_bonds").rstar_median()
    aligned = series.reindex(obs_index)
    if aligned.isna().any():
        missing = obs_index[aligned.isna()]
        raise ValueError(
            f"rstar_bonds does not cover {len(missing)} quarters of the sample "
            f"({missing[0]} to {missing[-1]}). Start the sample later.",
        )
    return aligned


def _diagnostics(prefix: str, label: str, obs: dict, index: pd.PeriodIndex) -> dict:
    """Measure the gap's behaviour through the tightening cycle, and what drives it."""
    results = load_results(prefix=prefix)
    post = results.trace["posterior"]

    def med(name: str) -> float:
        return float(np.median(np.asarray(post[name].values).ravel()))

    gap = results.output_gap_median()
    r_star = results.r_star_median()
    real = pd.Series(obs["cash_rate"] - obs["pi_exp"], index=index)
    rate_gap = real - r_star
    a_y1, a_y2, a_r = med("a_y1"), med("a_y2"), med("a_r")
    persistence = a_y1 + a_y2

    return {
        "run": label,
        "gap_2022Q4": float(gap["2022Q4"]),
        "gap_2026Q2": float(gap["2026Q2"]),
        "gap_change": float(gap["2026Q2"] - gap["2022Q4"]),
        "rate_gap_2026Q2": float(rate_gap["2026Q2"]),
        "quarters_rate_gap_positive_since_2022": int((rate_gap.loc["2022Q1":] > 0).sum()),
        "a_y1": a_y1,
        "persistence": persistence,
        "half_life_qtrs": float(np.log(0.5) / np.log(persistence)) if persistence < 1 else np.inf,
        "a_r": a_r,
        "rate_term_sd": float((a_r * rate_gap.shift(RATE_LAG)).std()),
        "b_y": med("b_y"),
        "sigma_IS": med("sigma_IS"),
    }


def main() -> None:
    """Re-estimate with r* given externally, against a like-for-like baseline."""
    print(f"Building observations from {START}...")
    obs, obs_index, chart_obs = build_observations(start=START, verbose=True)

    external = _external_rstar(obs_index)
    print(f"\nExternal r* (rstar_bonds, real): {external.iloc[0]:.2f} "
          f"-> {external.iloc[-1]:.2f}, mean {external.mean():.2f}")

    sampler_config = SamplerConfig(
        draws=10_000, tune=3_500, chains=5, cores=5, target_accept=0.90,
    )

    runs = [
        ("baseline", None),
        ("given_bonds", external.to_numpy()),
    ]
    rows = []

    for label, given in runs:
        prefix = f"rstar_hlw_{RESOLUTION}_{label}_{START}"
        print()
        print("=" * 70)
        print(f"{label}  (r* {'GIVEN from rstar_bonds' if given is not None else 'estimated'})")
        print("=" * 70)

        model = build_model(
            obs,
            resolution=RESOLUTION,
            rate_lag=RATE_LAG,
            given_rstar=given,
            obs_index=obs_index,
        )
        trace = sample_model(model, sampler_config)
        save_results(
            trace, obs, obs_index,
            constants=get_dictionary(model),
            chart_obs=chart_obs,
            prefix=prefix,
        )
        rows.append(_diagnostics(prefix, label, obs, obs_index))

    table = pd.DataFrame(rows).set_index("run")
    print()
    print("=" * 70)
    print("Does an externally supplied r* let the output gap close?")
    print("=" * 70)
    print(table.T.to_string(float_format=lambda v: f"{v:.3f}"))
    print()
    print("If gap_2026Q2 falls toward the Okun-implied +0.8 to +2.3, the r* level was")
    print("  the binding constraint. If it stays near +4, the gap's own persistence is,")
    print("  and no r* fixes it: the next move is an Okun equation.")


if __name__ == "__main__":
    main()
