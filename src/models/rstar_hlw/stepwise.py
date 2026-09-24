"""Estimate Resolution A in stages, locking one variance at a time.

WHY. The model cannot pin `sigma_ystar` and `sigma_g` at once. Free both and
the posterior funnels: the canonical run returns ESS 13 and 2,939
divergences. Impose `sigma_ystar` alone and the variance reappears in trend
growth, where `sigma_g` reaches 0.124 against a prior scale of 0.04 and g
takes on a pandemic-shaped trough that r* then inherits, since r* = g + z at
corr 0.998. Neither configuration is satisfactory and there is no third one
inside a single estimation.

LW2001 and HLW2017 do not attempt one. They estimate in three stages, each
holding still whatever the next stage measures, and carry each stage's answer
forward as an imposed setting. No stage ever asks the data to pin two
variances together. That is the device, and it is the part of the papers this
implementation has never had.

THIS IS AN ANALOGUE, NOT A REPLICATION. The papers' staging exists because
maximum likelihood returns exactly zero for these variances, the pile-up
problem, so they recover each ratio from an Andrews-Ploberger break test
mapped through Stock-Watson's tables rather than from the estimate itself.
MCMC does not pile up at zero, so each stage here reads its quantity off a
posterior, with an interval, and no break test is involved. The SEQUENCING is
HLW's. The estimator at each stage is not.

THE STAGES

1. **lambda_g**, with g and z held constant and the rate gap deleted from the
   IS curve. Holding g still is not a simplification, it is the measurement:
   Australian trend growth fell over this sample, and a stage that forbids g
   from moving must put that fall into the drift of the preliminary potential.
   The low-frequency movement in `Dy*` is therefore a reading of how far g
   should have been allowed to travel. See `_lambda_g_from_stage1`.
2. The rate gap returns and `a_r` is identified for the first time, with
   lambda_g imposed and z still constant. It produces **no lock**. HLW's
   second stage yields lambda_z, which needs `sigma_IS` and `a_r`, and those
   are built in an observation equation that runs after z. Until that is
   hoisted too, this stage is a checkpoint.
3. Everything, with lambda_g imposed and z released.

**ONLY THE RATIO TRAVELS.** `sigma_ystar` is free in every stage. Stage 1
estimates it in order to form a ratio, not to pin a level, and stages 2 and 3
estimate it again under the imposed lambda_g. That is HLW's device, and the
hoist in `estimate.py` is what makes it expressible.

THIS IS AN ANALOGUE, NOT A REPLICATION. The papers stage because maximum
likelihood returns exactly zero for these variances, the pile-up problem, so
they recover lambda_g from an Andrews-Ploberger break test mapped through
Stock-Watson's tables. MCMC does not pile up at zero, and no break test is
involved here. The SEQUENCING is HLW's and so is the idea that the suppressed
slowdown is the signal. The estimator is not.

WHAT TO WATCH. lambda_g against LW2001's published 0.039, and the sigma_g it
implies against the 0.124 the single-stage default estimates freely. Those are
two independent routes to the same quantity, and agreement between them is the
strongest evidence available here that the staging is measuring something
rather than manufacturing it.

AND THE OBJECTION, WHICH IS THIS REPO'S OWN. Each stage takes its input from a
model that assumes away what the next stage measures. That is a real cost and
it is declared rather than hidden. What the redesign removes is a second,
avoidable cost: an earlier version locked stage 1's `sigma_ystar` as a level,
which starved sigma_g downstream and halved lambda_g.

Run:
    uv run python -m src.models.rstar_hlw.stepwise
"""

import mgplot as mg
import numpy as np
import pandas as pd

from src.models.common.model_constants import get_dictionary
from src.models.nairu.base import (
    SamplerConfig,
    add_scalar_priors,
    sample_model,
)
from src.models.rstar_hlw.analyse import run_analyse
from src.models.rstar_hlw.estimate import (
    DEFAULT_SIGMA_YSTAR_FIXED,
    build_model,
    save_results,
)
from src.models.rstar_hlw.observations import build_observations
from src.models.rstar_hlw.results import DEFAULT_CHART_BASE, load_results

START = "1993Q1"

# The letter this pipeline answers to on the `--resolution` axis. Not a
# successor to H: S is not one model with a different r* identity, it is
# three models in serial, each locking a variance for the next. A letter out
# of sequence is the point, and "I" would read as a 1.
RESOLUTION = "S"

# Which r* identity the stages are built on. A, so the staging is measured
# against the canonical r* = g + z. Exposed so the same staging can later be
# run over one of the blend resolutions.
DEFAULT_BASE_RESOLUTION = "A"

# g is annualised here and quarterly in the papers, so a ratio quoted on their
# g scales by four on ours. Mirrors `estimate.py:_QUARTERS_PER_YEAR`.
_QUARTERS_PER_YEAR = 4

# Quarters in the centred window used to strip the innovation noise out of the
# preliminary potential's growth rate. Five years: long enough that
# sigma_ystar's contribution to the mean falls by sqrt(20), short enough to
# leave the growth slowdown intact. `_lambda_g_from_stage1` reports the
# sensitivity, because this window is a choice and it moves the answer.
_LOWPASS_WINDOW_QTRS = 20


def _lambda_g_from_stage1(prefix: str, sigma_ystar: float) -> tuple[float, str]:
    """Read lambda_g off the growth slowdown that stage 1 was forced to hide.

    THIS IS WHAT HOLDING g CONSTANT IS FOR. Australian trend growth fell over
    this sample, and a stage that forbids g from moving has to put that fall
    somewhere: it goes into the drift of the preliminary potential. So the
    low-frequency movement in `Dy*` is a measurement of how far g should have
    been allowed to travel, and the misspecification is the signal rather
    than the problem.

    HLW recover it with an Andrews-Ploberger break test mapped through
    Stock-Watson's tables. This is NOT that. The smoothed `Dy*` path is taken
    as the trend growth stage 1 could not have, and a driftless random walk of
    length T has a path dispersion of about sigma x sqrt(T/3), so inverting
    that gives sigma_g. Differencing the smoothed path directly would be the
    obvious alternative and is worse: at sigma_ystar near 0.6 the residual
    noise in a 20-quarter mean is about the same size as the sigma_g being
    measured, while the path's dispersion is dominated by the real slowdown.

    Returns lambda_g and a line describing how it was obtained.
    """
    potential = load_results(prefix=prefix).potential_median()
    growth = potential.diff() * _QUARTERS_PER_YEAR

    def sigma_g_for(window: int) -> float:
        smooth = growth.rolling(window, center=True).mean().dropna()
        return float(smooth.std() * np.sqrt(3.0 / len(smooth)))

    sigma_g = sigma_g_for(_LOWPASS_WINDOW_QTRS)
    lambda_g = sigma_g / (_QUARTERS_PER_YEAR * sigma_ystar)

    sweep = "  ".join(
        f"{w}q {sigma_g_for(w) / (_QUARTERS_PER_YEAR * sigma_ystar):.4f}"
        for w in (12, 20, 28, 40)
    )
    smooth = growth.rolling(_LOWPASS_WINDOW_QTRS, center=True).mean().dropna()
    desc = (
        f"Dy* smoothed over {_LOWPASS_WINDOW_QTRS}q runs "
        f"{smooth.iloc[0]:.2f} -> {smooth.iloc[-1]:.2f} (sd {smooth.std():.3f}), "
        f"implying sigma_g {sigma_g:.4f} and lambda_g {lambda_g:.4f}. "
        f"Across windows: {sweep}"
    )
    return lambda_g, desc

CHART_DIR = DEFAULT_CHART_BASE / f"rstar-hlw-{RESOLUTION}"

# The stage whose charts are S's own, rather than an intermediate model's.
_FINAL_STAGE = "stage3"

# Far apart so the four lines stay separable where they overlap.
RUN_COLOURS = {
    "stage1_lambda_g": "seagreen",
    "stage2_rate_gap": "darkorange",
    "stage3_full": "crimson",
    "default_A": "navy",
}

# Any base identity's single-stage run is drawn in the comparison colour.
_COMPARISON_COLOUR = "navy"


def _prefix(label: str, base_resolution: str) -> str:
    """Build the trace prefix for one stage.

    The base identity appears only when it is not the default, so the usual
    runs keep short names.
    """
    tag = "" if base_resolution == DEFAULT_BASE_RESOLUTION else f"_{base_resolution}"
    return f"rstar_hlw_{RESOLUTION}{tag}_{label}"


def _median(prefix: str, name: str) -> float:
    """Posterior median of a scalar in a saved trace."""
    post = load_results(prefix=prefix).trace["posterior"]
    return float(np.median(np.asarray(post[name].values).ravel()))


def _run(
    label: str,
    sampler_config: SamplerConfig,
    *,
    sigma_ystar_fixed: float | None,
    base_resolution: str,
    lambda_g: float | None = None,
    constants: dict[str, dict[str, float]] | None = None,
) -> str:
    """Build, sample and save one stage, returning its prefix.

    The three arguments are exactly what varies between stages: which
    variances are locked, and which coefficients or states are held still.
    """
    prefix = _prefix(label, base_resolution)
    obs, obs_index, chart_obs = build_observations(start=START)

    print()
    print("=" * 70)
    print(label)
    print("=" * 70)

    model = build_model(
        obs,
        resolution=base_resolution,
        obs_index=obs_index,
        sigma_ystar_fixed=sigma_ystar_fixed,
        lambda_g=lambda_g,
        constants=constants,
    )
    trace = sample_model(model, sampler_config)

    # Prior draws, so each stage gets the prior-against-posterior panels the
    # rest of the package produces. A stage that locks a variance has one
    # fewer free scalar, and those panels are how that shows up on a chart.
    add_scalar_priors(model, trace, random_seed=sampler_config.random_seed)

    save_results(
        trace, obs, obs_index,
        constants=get_dictionary(model),
        chart_obs=chart_obs,
        prefix=prefix,
    )

    # The standard suite, per stage. The LAST stage writes to S's own
    # directory, because stage 3 is what S reports and `charts/rstar-hlw-S/`
    # is where a reader looks for it, exactly as for any other resolution.
    # The earlier stages are intermediate models and get their own.
    stage = label.split("_", maxsplit=1)[0]
    run_analyse(
        prefix=prefix,
        chart_dir=(
            CHART_DIR if stage == _FINAL_STAGE
            else DEFAULT_CHART_BASE / f"rstar-hlw-{RESOLUTION}-{stage}"
        ),
        resolution=base_resolution,
    )
    return prefix


def _diagnostics(prefix: str, label: str) -> dict:
    """Report what each stage produced, tolerating parameters a stage lacks."""
    results = load_results(prefix=prefix)
    post = results.trace["posterior"]

    def med(name: str) -> float:
        if name not in post:
            return float("nan")
        return float(np.median(np.asarray(post[name].values).ravel()))

    gap = results.output_gap_median()
    g = results.trend_growth_median()
    pot = results.potential_median()
    lg = pd.Series(results.obs["log_gdp"], index=results.obs_index)

    return {
        "run": label,
        "divergences": int(results.trace["sample_stats"]["diverging"].to_numpy().sum()),
        "sigma_ystar": med("sigma_ystar"),
        "sigma_g": med("sigma_g"),
        "sd_qtrly_potential": float(pot.diff().std()),
        "sd_qtrly_gdp": float(lg.diff().std()),
        "g_1993Q1": float(g.iloc[0]),
        "g_2019Q4": float(g["2019Q4"]),
        "g_2026Q2": float(g["2026Q2"]),
        "g_range": float(g.max() - g.min()),
        "gap_2026Q2": float(gap["2026Q2"]),
        "gap_sd": float(gap.std()),
        "a_r": med("a_r"),
        "b_y": med("b_y"),
    }


def _chart(prefixes: dict[str, str]) -> None:
    """One chart per state, a line per stage."""
    # No clear_chart_dir here: stage 3's suite is already in this directory
    # and these comparisons are added beside it, not instead of it.
    mg.set_chart_dir(str(CHART_DIR))

    # Each state, and how to pull it off a results object. Potential is
    # differenced because its level is a trend in the hundreds and the
    # comparison is about how fast it moves, which is the whole question.
    states = {
        "trend_growth": (
            "Trend growth g", "Annualised %",
            lambda r: r.trend_growth_median(),
        ),
        "output_gap": (
            "Output gap", "Per cent of potential",
            lambda r: r.output_gap_median(),
        ),
        "potential_growth": (
            "Potential output, quarterly growth", "Log points per quarter",
            lambda r: r.potential_median().diff(),
        ),
        "r_star": (
            "r* (real)", "Annualised %",
            lambda r: r.r_star_median(),
        ),
        "z_star": (
            "z, the non-growth part of r*", "Annualised %",
            lambda r: r.z_star_median(),
        ),
    }
    for title, ylabel, getter in states.values():
        frame = pd.DataFrame({
            label: getter(load_results(prefix=p)) for label, p in prefixes.items()
        })
        mg.line_plot_finalise(
            frame,
            title=f"HLW Resolution {RESOLUTION} (staged): {title}",
            ylabel=ylabel,
            color=[RUN_COLOURS.get(c, _COMPARISON_COLOUR) for c in frame.columns],
            width=2,
            y0=True,
            legend={"loc": "best", "fontsize": "small"},
            lfooter=f"Australia. From {START}. Each stage locks one variance.",
            rfooter="Source: ABS, RBA",
            show=False,
        )
    print(f"Charts saved to: {CHART_DIR}")


def main(base_resolution: str = DEFAULT_BASE_RESOLUTION) -> None:
    """Run the three stages, carrying each lock forward, against the default.

    `base_resolution` is the r* identity the stages are built on, and the
    single-stage run of that same identity is what the comparison column
    reports.
    """
    sampler_config = SamplerConfig(
        draws=10_000, tune=3_500, chains=5, cores=5, target_accept=0.90,
    )

    # Stage 1. lambda_g = 0 holds g still; sigma_z = 0 holds z still; a_r = 0
    # deletes the rate gap, as the papers' first stage does. sigma_ystar is
    # the only variance left, and the only one this stage is for.
    s1 = _run(
        "stage1_lambda_g", sampler_config,
        base_resolution=base_resolution,
        lambda_g=0.0,
        sigma_ystar_fixed=None,
        constants={"is_curve": {"a_r": 0.0}, "z_star": {"sigma_z": 0.0}},
    )
    sigma_ystar = _median(s1, "sigma_ystar")
    lambda_g, how = _lambda_g_from_stage1(s1, sigma_ystar)
    print(f"\nSTAGE 1: sigma_ystar = {sigma_ystar:.4f} with g held constant "
          f"(the default imposes {DEFAULT_SIGMA_YSTAR_FIXED})")
    print(f"STAGE 1 LOCK: lambda_g = {lambda_g:.4f}")
    print(f"  {how}")
    print("  sigma_ystar is NOT locked. Only the ratio travels, which is HLW's")
    print("  device and what stages 2 and 3 impose with the level left free.")

    # Stage 2. Carry the ratio, restore the rate gap, keep z still. This is
    # where a_r is first identified. It produces NO lock: HLW's second stage
    # yields lambda_z, which needs sigma_IS and a_r, and those are built in an
    # observation equation that runs after z. Until that is hoisted too, stage
    # 2 is a checkpoint rather than a source.
    s2 = _run(
        "stage2_rate_gap", sampler_config,
        base_resolution=base_resolution,
        sigma_ystar_fixed=None,
        lambda_g=lambda_g,
        constants={"z_star": {"sigma_z": 0.0}},
    )
    print(f"\nSTAGE 2: a_r = {_median(s2, 'a_r'):+.4f}, "
          f"sigma_ystar = {_median(s2, 'sigma_ystar'):.4f} (free, ratio imposed)")

    # Stage 3. The ratio still imposed, the level still free, z released.
    s3 = _run(
        "stage3_full", sampler_config,
        base_resolution=base_resolution,
        sigma_ystar_fixed=None,
        lambda_g=lambda_g,
    )

    prefixes = {
        "stage1_lambda_g": s1,
        "stage2_rate_gap": s2,
        "stage3_full": s3,
        f"default_{base_resolution}": f"rstar_hlw_{base_resolution}",
    }
    table = pd.DataFrame(
        [_diagnostics(p, label) for label, p in prefixes.items()],
    ).set_index("run")

    print()
    print("=" * 70)
    print("Stepwise estimation, each stage locking one variance")
    print("=" * 70)
    print(table.T.to_string(float_format=lambda v: f"{v:.3f}"))
    print()
    print(f"Lock carried forward: lambda_g = {lambda_g:.4f}. sigma_ystar is free")
    print("  in stages 2 and 3, so only the RATIO is imposed. Compare stage 3's")
    print("  sigma_ystar with stage 1's: the staging is meant to constrain how")
    print("  fast g moves relative to potential, not to pin either level.")

    _chart(prefixes)


if __name__ == "__main__":
    main()
