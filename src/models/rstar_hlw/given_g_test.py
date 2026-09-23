"""Test what the rest of the model does when trend growth is handed to it as data.

THE QUESTION. In Resolution A, g is a driftless random walk with no
observation equation of its own, and `sigma_z` is free, so r* = g + z with
corr 0.998. g therefore IS the answer, and the free posterior does not look
like a trend: 3.80 at 1993Q1, down to 1.26 at 2019Q4, then back UP to 2.22 by
2026Q2, a 2.6pp range with a 1pp rise across the pandemic. Trend growth does
not move like that. What it looks like instead is the volatility that used to
pile into potential output, now piling into g once `sigma_ystar` was imposed:
the free `sigma_g` comes back at 0.126 against its prior scale of 0.04.

So this script asks: **if g were a credible trend, would anything else in the
model change?** It hands g over as DATA and re-estimates everything else.

THIS IS NOT AN ESTIMATE OF g, AND NOT A NINTH RESOLUTION. g is an input here.
It is the mirror of `given_rstar_test.py`, which supplies the other half of
the r* identity, and the two should be read together.

WHAT TO READ:

- `z_sd` and `sigma_z`. This is the real question. With g free, z has nothing
  to do because g absorbs everything; the counting argument in MODEL_NOTES
  says z has no observation equation and so cannot identify. Pinning g does
  not give z an observation equation, so if z stays flat that argument holds
  at full strength. If z suddenly moves, the free g was hiding something.
- `gap_2026Q2` and `gap_change`. `given_rstar_test.py` moved the gap by
  0.18pp when it changed r*'s LEVEL. g changes r*'s SHAPE as well, so this is
  the stronger version of the same test.
- `a_r` and `rate_term_sd`. If the rate channel is still invisible with a
  sensible r* path, the loading problem is not about which r* you choose.

SAMPLE. 1993Q1, matching the live Resolution A default, and the baseline is
re-run here rather than read off disk so the four rows are like for like.

Run:
    uv run python -m src.models.rstar_hlw.given_g_test
"""

import mgplot as mg
import numpy as np
import pandas as pd

from src.models.nairu.base import SamplerConfig, get_fixed_constants, sample_model
from src.models.rstar_hlw.estimate import build_model, save_results
from src.models.rstar_hlw.observations import G_ANCHOR_LABELS, build_observations
from src.models.rstar_hlw.results import DEFAULT_CHART_BASE, load_results
from src.models.ystar.results import load_results as load_ystar

START = "1993Q1"
RESOLUTION = "A"
RATE_LAG = 6

# The flat g handed to the last run, annualised %. A round number in the
# middle of the range this repo's potential-output work returns, picked so
# that run asks "what if g carried no trend variation at all" rather than to
# reproduce any particular estimate.
FLAT_G = 2.0

CAGR_LABEL = G_ANCHOR_LABELS["cagr40"]

CHART_DIR = DEFAULT_CHART_BASE / "rstar-hlw-given-g"

# One colour per run, kept far apart so the lines stay separable where they
# overlap and against the chart's own gridlines.
RUN_COLOURS = {
    "baseline": "navy",
    "given_ystar": "darkorange",
    "given_cagr40": "crimson",
    f"given_flat{FLAT_G:g}": "seagreen",
}


def _external_g(obs_index: pd.PeriodIndex) -> pd.Series:
    """Trend growth from the ystar model, annualised, aligned to the sample.

    Raises rather than filling: an extrapolated g would be an assumption
    wearing the costume of an external estimate, which is what this test
    exists to avoid.
    """
    series = load_ystar(prefix="ystar").trend_growth_posterior().median(axis=1)
    aligned = series.reindex(obs_index)
    if aligned.isna().any():
        missing = obs_index[aligned.isna()]
        raise ValueError(
            f"ystar does not cover {len(missing)} quarters of the sample "
            f"({missing[0]} to {missing[-1]}). Start the sample later.",
        )
    return aligned


def _diagnostics(prefix: str, label: str, obs: dict, index: pd.PeriodIndex) -> dict:
    """Measure what a prescribed g does to z, to the gap, and to the rate channel."""
    results = load_results(prefix=prefix)
    post = results.trace["posterior"]

    def med(name: str) -> float:
        return float(np.median(np.asarray(post[name].values).ravel()))

    gap = results.output_gap_median()
    g = results.trend_growth_median()
    z = results.z_star_median()
    r_star = results.r_star_median()
    real = pd.Series(obs["cash_rate"] - obs["pi_exp"], index=index)
    rate_gap = real - r_star
    a_y1, a_y2, a_r = med("a_y1"), med("a_y2"), med("a_r")
    persistence = a_y1 + a_y2

    return {
        "run": label,
        "g_1993Q1": float(g.iloc[0]),
        "g_2026Q2": float(g["2026Q2"]),
        "z_2026Q2": float(z["2026Q2"]),
        "z_sd": float(z.std()),
        "sigma_z": med("sigma_z"),
        "r_star_2026Q2": float(r_star["2026Q2"]),
        "gap_2022Q4": float(gap["2022Q4"]),
        "gap_2026Q2": float(gap["2026Q2"]),
        "gap_change": float(gap["2026Q2"] - gap["2022Q4"]),
        "rate_gap_2026Q2": float(rate_gap["2026Q2"]),
        "quarters_rate_gap_positive_since_2022": int((rate_gap.loc["2022Q1":] > 0).sum()),
        "a_r": a_r,
        "persistence": persistence,
        "rate_term_sd": float((a_r * rate_gap.shift(RATE_LAG)).std()),
        "b_y": med("b_y"),
        "sigma_IS": med("sigma_IS"),
    }


def _chart(paths: dict[str, pd.DataFrame]) -> None:
    """One chart per state, a line per run."""
    mg.set_chart_dir(str(CHART_DIR))
    mg.clear_chart_dir()

    specs = (
        ("trend_growth", "Trend growth g", "Annualised %", True),
        ("r_star", "r* (real)", "Annualised %", True),
        ("z_star", "z, the non-growth part of r*", "Annualised %", True),
        ("output_gap", "Output gap", "Per cent of potential", True),
    )
    for key, title, ylabel, zero_line in specs:
        frame = paths[key]
        mg.line_plot_finalise(
            frame,
            title=f"HLW Resolution A: {title} with g given",
            ylabel=ylabel,
            color=[RUN_COLOURS[c] for c in frame.columns],
            width=2,
            y0=zero_line,
            legend={"loc": "best", "fontsize": "small"},
            lfooter=f"Australia. From {START}. g prescribed, not estimated.",
            rfooter="Source: ABS, RBA",
            show=False,
        )
    print(f"Charts saved to: {CHART_DIR}")


def main() -> None:
    """Re-estimate with g given externally, against a like-for-like baseline."""
    print(f"Building observations from {START}...")
    obs, obs_index, chart_obs = build_observations(start=START, verbose=True)

    external = _external_g(obs_index)
    print(f"\nExternal g (ystar, annualised): {external.iloc[0]:.2f} "
          f"-> {external.iloc[-1]:.2f}, mean {external.mean():.2f}")

    # The trailing-CAGR anchor series, taken from `build_observations` rather
    # than recomputed here so the two cannot drift. This is the run that
    # matters: ystar's g is smooth by construction, so a persistence collapse
    # under it could be the smoothing rather than the trend. This g is a
    # window on GDP and nothing has deliberately smoothed it.
    cagr_obs, _, _ = build_observations(start=START, g_anchor="cagr40")
    cagr = pd.Series(cagr_obs["trend_growth_obs"], index=obs_index)
    print(f"External g ({CAGR_LABEL}): {cagr.iloc[0]:.2f} "
          f"-> {cagr.iloc[-1]:.2f}, mean {cagr.mean():.2f}")

    flat = pd.Series(FLAT_G, index=obs_index)

    sampler_config = SamplerConfig(
        draws=10_000, tune=3_500, chains=5, cores=5, target_accept=0.90,
    )

    runs = [
        ("baseline", None),
        ("given_ystar", external.to_numpy()),
        ("given_cagr40", cagr.to_numpy()),
        (f"given_flat{FLAT_G:g}", flat.to_numpy()),
    ]
    rows = []
    paths: dict[str, dict[str, pd.Series]] = {
        "trend_growth": {}, "r_star": {}, "z_star": {}, "output_gap": {},
    }

    for label, given in runs:
        prefix = f"rstar_hlw_{RESOLUTION}_{label}"
        print()
        print("=" * 70)
        print(f"{label}  (g {'GIVEN as data' if given is not None else 'estimated'})")
        print("=" * 70)

        model = build_model(
            obs,
            resolution=RESOLUTION,
            rate_lag=RATE_LAG,
            given_g=given,
            obs_index=obs_index,
        )
        trace = sample_model(model, sampler_config)
        save_results(
            trace, obs, obs_index,
            constants=get_fixed_constants(model),
            chart_obs=chart_obs,
            prefix=prefix,
        )
        rows.append(_diagnostics(prefix, label, obs, obs_index))

        results = load_results(prefix=prefix)
        paths["trend_growth"][label] = results.trend_growth_median()
        paths["r_star"][label] = results.r_star_median()
        paths["z_star"][label] = results.z_star_median()
        paths["output_gap"][label] = results.output_gap_median()

    table = pd.DataFrame(rows).set_index("run")
    print()
    print("=" * 70)
    print("What changes when g is prescribed rather than estimated?")
    print("=" * 70)
    print(table.T.to_string(float_format=lambda v: f"{v:.3f}"))
    print()

    # After the table, not before: the sampling is the expensive part, and a
    # charting error should not cost the numbers it took an hour to produce.
    _chart({k: pd.DataFrame(v) for k, v in paths.items()})
    print()
    print("z_sd near zero in every row means pinning g does not give z anything to")
    print("  do, and the counting argument stands: z has no observation equation.")
    print("  A large z_sd would mean the free g had been absorbing a real signal.")


if __name__ == "__main__":
    main()
