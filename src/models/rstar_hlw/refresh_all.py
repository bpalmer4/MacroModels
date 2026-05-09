"""Re-run all resolutions A-G on the gap-filled observation sample.

After wiring ``get_indexed_yield_filled`` into ``observations.py`` (which
fills the 2013Q3-2014Q3 gap in the indexed 10y bond yield via nominal −
interpolated breakeven), the model sample grows from 153 to 158 contiguous
quarters. This script re-estimates each resolution on the new sample and
runs analyse for each one (charts → ``charts/rstar-hlw-{X}/``).

Run:
    uv run python -m src.models.rstar_hlw.refresh_all
"""

from src.models.nairu.base import SamplerConfig, get_fixed_constants, sample_model
from src.models.rstar_hlw.analyse import run_analyse
from src.models.rstar_hlw.estimate import build_model, save_results
from src.models.rstar_hlw.observations import build_observations

RUNS = [
    # (resolution, prefix, chart_subdir, label)
    ("A", "rstar_hlw_A", "rstar-hlw-A", "A (canonical, r* = g + z)"),
    ("B", "rstar_hlw_B", "rstar-hlw-B", "B (canonical + indexed-bond observation)"),
    ("C", "rstar_hlw_C", "rstar-hlw-C", "C (blend, fixed Beta(1,1))"),
    ("D", "rstar_hlw_D", "rstar-hlw-D", "D (canonical r* + open-economy IS)"),
    ("E", "rstar_hlw_E", "rstar-hlw-E", "E (blend + AR(1) z)"),
    ("F", "rstar_hlw_F", "rstar-hlw-F", "F (E + open-economy IS)"),
    ("G", "rstar_hlw_G", "rstar-hlw-G", "G (default; blend + hierarchical Beta)"),
]


def main() -> None:
    print("Building observations once (shared across all 7 runs)...")
    obs, obs_index, chart_obs = build_observations(start="1980Q1", verbose=True)
    print(f"\nSample length: {len(obs_index)}  (158 contiguous quarters with indexed_10y gap filled)\n")

    sampler_config = SamplerConfig(
        draws=10_000,
        tune=3_500,
        chains=5,
        cores=5,
        target_accept=0.90,
    )

    for resolution, prefix, chart_subdir, label in RUNS:
        print()
        print("=" * 70)
        print(f"{label}  (resolution={resolution}, prefix={prefix})")
        print("=" * 70)

        model = build_model(obs, resolution=resolution)
        trace = sample_model(model, sampler_config)

        constants = get_fixed_constants(model)
        save_results(
            trace, obs, obs_index,
            constants=constants, chart_obs=chart_obs,
            prefix=prefix,
        )

        print(f"\nAnalysing {prefix} → charts/{chart_subdir}/")
        run_analyse(
            prefix=prefix,
            chart_dir=f"charts/{chart_subdir}",
            resolution=resolution,
            verbose=True,
        )


if __name__ == "__main__":
    main()
