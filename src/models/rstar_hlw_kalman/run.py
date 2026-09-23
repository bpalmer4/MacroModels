"""Run HLW's three-stage procedure on Australian data and chart the result.

Run:
    uv run python -m src.models.rstar_hlw_kalman.run
"""

import mgplot as mg
import numpy as np
import pandas as pd

from src.models.rstar_hlw.observations import build_observations
from src.models.rstar_hlw.results import DEFAULT_CHART_BASE
from src.models.rstar_hlw.results import load_results as load_mcmc
from src.models.rstar_hlw_kalman import stages
from src.models.rstar_hlw_kalman.state_space import YSTAR, G, Z

CHART_DIR = DEFAULT_CHART_BASE / "rstar-hlw-kalman"

# The MCMC run of the same identity, drawn alongside so the difference between
# the two ESTIMATORS is visible rather than asserted.
MCMC_PREFIX = "rstar_hlw_A"

_QUARTERS_PER_YEAR = 4.0


def _paths(obs: dict[str, np.ndarray], index: pd.PeriodIndex, params: object) -> pd.DataFrame:
    """Smoothed g, z, r* and the output gap for a fitted parameter vector."""
    states = stages.smoothed_states(obs, params)  # type: ignore[arg-type]
    g = pd.Series(states[:, G], index=index)
    z = pd.Series(states[:, Z], index=index)
    ystar = pd.Series(states[:, YSTAR], index=index)
    log_gdp = pd.Series(obs["log_gdp"], index=index)
    return pd.DataFrame({
        "trend_growth": g,
        "z_star": z,
        "r_star": g + z,
        "output_gap": log_gdp - ystar,
    })


def main() -> None:
    """Estimate all three stages, then chart g and r* against the MCMC run."""
    obs, index, _ = build_observations()

    stage1 = stages.fit(obs, 1)
    lambda_g, diagnostics = stages.lambda_g_from_stage1(obs, stage1)
    stage2 = stages.fit(obs, 2, lambda_g=lambda_g)
    # sigma_z imposed rather than taken from lambda_z: HLW's ratio divides by
    # an IS slope this data pins to its own bound, which returns sigma_z near
    # 4 and an r* spanning thirty points, and fits worse besides.
    stage3 = stages.fit(obs, 3, lambda_g=lambda_g, sigma_z_imposed=stages.DEFAULT_SIGMA_Z)

    print(f"lambda_g = {lambda_g:.4f} from stage 1 "
          f"(exp-Wald {diagnostics['exp_wald']:.3f}, "
          f"Dy* {diagnostics['growth_first']:.2f} -> {diagnostics['growth_last']:.2f})")
    print(f"sigma_z  = {stages.DEFAULT_SIGMA_Z} IMPOSED directly; the likelihood is")
    print("           nearly flat in it, so z's SHAPE is identified and its scale is not")
    for result in (stage1, stage2, stage3):
        print(f"  {result.name}: loglik {result.log_likelihood:9.2f}  "
              f"converged {result.converged}")

    paths = _paths(obs, index, stage3.params)

    mcmc = load_mcmc(prefix=MCMC_PREFIX)
    comparison = {
        "trend_growth": mcmc.trend_growth_median(),
        "r_star": mcmc.r_star_median(),
    }

    mg.set_chart_dir(str(CHART_DIR))
    mg.clear_chart_dir()

    specs = (
        ("trend_growth", "Trend growth g", "Annualised %"),
        ("r_star", "r* (real)", "Annualised %"),
    )
    for key, title, ylabel in specs:
        frame = pd.DataFrame({
            "HLW three-stage (Kalman, ML)": paths[key],
            "MCMC Resolution A": comparison[key],
        })
        mg.line_plot_finalise(
            frame,
            title=f"HLW three-stage estimation: {title}",
            ylabel=ylabel,
            color=["crimson", "navy"],
            width=2,
            y0=True,
            legend={"loc": "best", "fontsize": "small"},
            lfooter=(
                f"Australia. lambda_g {lambda_g:.4f} from stage 1 break test; "
                f"sigma_z {stages.DEFAULT_SIGMA_Z} imposed, its scale not identified."
            ),
            rfooter="Source: ABS, RBA",
            show=False,
        )
    print(f"Charts saved to: {CHART_DIR}")


if __name__ == "__main__":
    main()
