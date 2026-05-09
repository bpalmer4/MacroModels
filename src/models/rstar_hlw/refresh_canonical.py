"""Refresh the canonical C, E, F traces with the new Beta(1, 1) default,
and run C with the hierarchical Beta(a, b) where a, b ~ HalfNormal(1).

Background: the Beta(2, 2) prior on alpha was symmetrically tightening the
posterior toward 0.5 and masking the (weak) data preference for higher
alpha visible under Uniform. The default is now Beta(1, 1) so the posterior
shape reflects the likelihood. The hierarchical run is a sharper test of
that — letting the data choose the Beta shape itself.

Each run is followed by an analyse pass that emits charts into a
resolution-specific subfolder so we don't overwrite the standard
charts/rstar-hlw-{C,E,F} directories.

Run:
    uv run python -m src.models.rstar_hlw.refresh_canonical
"""

from src.models.nairu.base import SamplerConfig, get_fixed_constants, sample_model
from src.models.rstar_hlw.analyse import run_analyse
from src.models.rstar_hlw.estimate import build_model, save_results
from src.models.rstar_hlw.observations import build_observations

RUNS = [
    # (resolution, constants_override, prefix, chart_subdir, label)
    ("C", None,                                         "rstar_hlw_C",                 "rstar-hlw-C",                "C (new default Beta(1,1))"),
    ("E", None,                                         "rstar_hlw_E",                 "rstar-hlw-E",                "E (new default Beta(1,1))"),
    ("F", None,                                         "rstar_hlw_F",                 "rstar-hlw-F",                "F (new default Beta(1,1))"),
    ("C", {"r_star": {"alpha_hierarchical": True}},     "rstar_hlw_C_alpha_hier",      "rstar-hlw-C-alpha-hier",     "C with hierarchical Beta(a,b)"),
]


def main() -> None:
    print("Building observations once...")
    obs, obs_index, chart_obs = build_observations(start="1980Q1", verbose=True)

    sampler_config = SamplerConfig(
        draws=10_000,
        tune=3_500,
        chains=5,
        cores=5,
        target_accept=0.90,
    )

    for resolution, override, prefix, chart_subdir, label in RUNS:
        print()
        print("=" * 70)
        print(f"{label}  (resolution={resolution}, prefix={prefix})")
        print("=" * 70)

        constants_override = override or {}
        model = build_model(obs, constants=constants_override, resolution=resolution)

        trace = sample_model(model, sampler_config)

        constants = get_fixed_constants(model)
        save_results(
            trace, obs, obs_index,
            constants=constants, chart_obs=chart_obs,
            prefix=prefix,
        )

        print()
        print(f"Analysing {prefix} → charts/{chart_subdir}/")
        run_analyse(
            prefix=prefix,
            chart_dir=f"charts/{chart_subdir}",
            resolution=resolution,
            verbose=True,
        )


if __name__ == "__main__":
    main()
