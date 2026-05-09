"""Prior-sensitivity sweep on sigma_z for Resolution E.

Buncic-Pagan-Robinson (2023) point: when shocks >= observables, the latent r*
is not point-identified; the posterior is essentially the prior projected
through the model. The corollary is that the answer is only as good as the
prior on the variance ratios — and the only honest way to know how much the
prior is doing is to vary it and watch.

Resolution E fixes sigma_z = 0.15 (the AR(1) innovation scale on z, the
deviation of r* from the blend). This script holds the model fixed and runs E
at sigma_z in {0.05, 0.15, 0.30, 0.50, 1.00}, saving each trace under a
distinct prefix. Comparison happens downstream — chart r* posteriors as small
multiples; if r* moves materially with sigma_z, the answer is prior-driven; if
not, the data is informative about r* ~ blend.

Seed held at 42 throughout so the only varying input is sigma_z.

Run:
    uv run python -m src.models.rstar_hlw.sigma_z_sweep
"""

from src.models.nairu.base import SamplerConfig, get_fixed_constants, sample_model
from src.models.rstar_hlw.estimate import build_model, save_results
from src.models.rstar_hlw.observations import build_observations

SIGMA_Z_GRID = [0.05, 0.15, 0.30, 0.50, 1.00]


def main() -> None:
    print("Building observations once (shared across sweep)...")
    obs, obs_index, chart_obs = build_observations(start="1980Q1", verbose=True)

    sampler_config = SamplerConfig(
        draws=10_000,
        tune=3_500,
        chains=5,
        cores=5,
        target_accept=0.90,
    )

    for sigma_z in SIGMA_Z_GRID:
        suffix = f"{int(round(sigma_z * 100)):03d}"  # 0.05 -> "005"
        prefix = f"rstar_hlw_E_sigma_z_{suffix}"

        print()
        print("=" * 70)
        print(f"Resolution E with sigma_z = {sigma_z:.2f}  (prefix = {prefix})")
        print("=" * 70)

        constants_override = {"r_star": {"sigma_z": sigma_z}}
        model = build_model(obs, constants=constants_override, resolution="E")

        trace = sample_model(model, sampler_config)

        constants = get_fixed_constants(model)
        save_results(
            trace,
            obs,
            obs_index,
            constants=constants,
            chart_obs=chart_obs,
            prefix=prefix,
        )


if __name__ == "__main__":
    main()
