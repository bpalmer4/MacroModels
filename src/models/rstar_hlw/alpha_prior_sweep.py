"""Prior-sensitivity sweep on the Beta(a, b) prior for alpha_rstar in Resolution C.

Buncic-Pagan-Robinson 2023 prediction: when the data is uninformative about
a parameter, the posterior is the prior projected through the model. This
script tests that for alpha (the blend weight between trend growth and the
real bond yield) by holding Resolution C fixed and varying the Beta prior
shape over five settings:

- Beta(0.5, 0.5)  Jeffreys-style: bimodal, prefers endpoints
- Beta(1.0, 1.0)  Uniform: flat
- Beta(2.0, 2.0)  Default: mild central preference
- Beta(5.0, 5.0)  Tight central preference
- Beta(4.0, 1.0)  Asymmetric: prefers structural anchor (g)

If the posterior follows the prior shape, alpha is prior-driven (Buncic
finding confirmed for alpha specifically). If posteriors converge to a
single shape regardless of prior, alpha is data-identified.

Seed held at 42 throughout. Resolution C is otherwise unchanged.

Run:
    uv run python -m src.models.rstar_hlw.alpha_prior_sweep
"""

from src.models.nairu.base import SamplerConfig, get_fixed_constants, sample_model
from src.models.rstar_hlw.estimate import build_model, save_results
from src.models.rstar_hlw.observations import build_observations

ALPHA_PRIORS = [
    (0.5, 0.5, "jeffreys"),
    (1.0, 1.0, "uniform"),
    (2.0, 2.0, "default"),
    (5.0, 5.0, "tight"),
    (4.0, 1.0, "skew_g"),
]


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

    for a, b, label in ALPHA_PRIORS:
        prefix = f"rstar_hlw_C_alpha_{label}"

        print()
        print("=" * 70)
        print(f"Resolution C with Beta(a={a}, b={b})  [label = {label}]")
        print(f"  prefix = {prefix}")
        print("=" * 70)

        constants_override = {"r_star": {"alpha_prior": (a, b)}}
        model = build_model(obs, constants=constants_override, resolution="C")

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
