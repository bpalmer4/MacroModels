"""Build, sample, and persist the HLW Bayesian r-star model."""

import pickle
from pathlib import Path
from typing import Any

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm

from src.models.nairu.base import SamplerConfig, get_fixed_constants, sample_model
from src.models.rstar_hlw.equations.indexed_bond import indexed_bond_equation
from src.models.rstar_hlw.equations.is_curve import is_curve_equation
from src.models.rstar_hlw.equations.phillips import phillips_curve_equation
from src.models.rstar_hlw.equations.potential import potential_output_equation
from src.models.rstar_hlw.equations.r_star import r_star_equation
from src.models.rstar_hlw.equations.r_star_blended_z import r_star_blended_z_equation
from src.models.rstar_hlw.equations.r_star_tv_alpha import r_star_tv_alpha_equation
from src.models.rstar_hlw.equations.trend_growth import trend_growth_equation
from src.models.rstar_hlw.equations.z_star import z_star_equation
from src.models.rstar_hlw.observations import build_observations

DEFAULT_OUTPUT_DIR = Path(__file__).parent.parent.parent.parent / "model_outputs"


def build_model(
    obs: dict[str, np.ndarray],
    constants: dict[str, Any] | None = None,
    resolution: str = "G",
) -> pm.Model:
    """Build the HLW r-star PyMC model.

    Equation order matters for NUTS efficiency: state equations first, then
    observation equations.

    Resolution A: r* = g + z (canonical HLW with AR(1) reparameterised z).
    Resolution B: A + indexed-bond observation equation (term-structure pin).
    Resolution C: r* = α·g + (1-α)·(indexed_10y - k) + ε (the blend).
    Resolution D: A's r* identity + an SOE-block IS curve (fiscal + ToT +
                  TWI change + ICP growth) + the soft linear-trend anchor on g.
                  Tests whether enough external regressors firm a_r so that
                  canonical HLW's z latent identifies.
    Resolution E: r* = α·g + (1-α)·(indexed_10y - k) + z, with z AR(1) and
                  σ_z fixed. Generalises C (z=0) and A (α=1) by keeping
                  C's blend as the structural anchor and adding an
                  IS-curve-identifiable persistent deviation z. Tests
                  whether the IS curve has anything to say above and beyond
                  the blend that C suppresses with its small i.i.d. ε_t.
    Resolution F: E's r* identity + the SOE-block IS curve from D. Tests
                  whether giving the IS curve more external regressors lets
                  it identify z (which it could not in E with the
                  fiscal-only IS curve).
    Resolution G: same as C but with hierarchical Beta(a, b) on alpha
                  (a, b ~ Uniform(0.25, 2)) — let the data pick the prior
                  shape itself.
    Resolution H: same r* identity as C but with time-varying alpha_t —
                  logit-scale RW on alpha, sigma_a fixed. Tests whether the
                  data wants alpha to drift over time (e.g. toward the bond
                  anchor in recent years, consistent with Bullock's
                  "shifts in r*" framing).
    """
    if constants is None:
        constants = {}
    if resolution not in ("A", "B", "C", "D", "E", "F", "G", "H"):
        raise ValueError(
            f"resolution must be 'A', 'B', 'C', 'D', 'E', 'F', 'G' or 'H', "
            f"got {resolution!r}",
        )

    # SOE-block regressors are only used by Resolutions D and F.
    SOE_KEYS = ("tot_change_1", "twi_change_1", "icp_change_1")

    if resolution in ("A", "B"):
        # Textbook canonical: no fiscal impulse, no soft anchor on g, no SOE.
        obs = {
            k: v for k, v in obs.items()
            if k not in ("trend_growth_obs", "fiscal_impulse_1", *SOE_KEYS)
        }
    elif resolution in ("C", "E", "G", "H"):
        # Blend (with persistent deviation in E; hierarchical alpha in G;
        # time-varying alpha in H): keeps fiscal + soft anchor on g; drops
        # SOE block (only D and F use it).
        obs = {k: v for k, v in obs.items() if k not in SOE_KEYS}
    # Resolutions D and F: keep everything. soft anchor on g, fiscal, full SOE block.

    # Resolution G: inject the hierarchical-Beta flag into the r_star constants
    # so r_star_equation samples a, b ~ HalfNormal(1) and alpha ~ Beta(a, b).
    if resolution == "G":
        constants = dict(constants)  # shallow copy so we don't mutate caller's dict
        r_star_const = dict(constants.get("r_star", {}))
        r_star_const.setdefault("alpha_hierarchical", True)
        constants["r_star"] = r_star_const

    model = pm.Model()
    latents: dict[str, Any] = {}
    descriptions: list[str] = []

    # --- State equations ---
    desc = trend_growth_equation(obs, model, latents, constant=constants.get("trend_growth"))
    descriptions.append(f"Trend growth: {desc}")

    if resolution in ("C", "G"):
        # Order: trend_growth -> r_star (uses g, indexed_10y) -> potential -> observations.
        # G uses the same equation; the alpha_hierarchical flag was injected above.
        desc = r_star_equation(obs, model, latents, constant=constants.get("r_star"))
        descriptions.append(f"r-star:       {desc}")

        desc = potential_output_equation(obs, model, latents, constant=constants.get("potential"))
        descriptions.append(f"Potential:    {desc}")
    elif resolution == "H":
        # Time-varying alpha via logit-RW.
        desc = r_star_tv_alpha_equation(
            obs, model, latents, constant=constants.get("r_star"),
        )
        descriptions.append(f"r-star:       {desc}")

        desc = potential_output_equation(obs, model, latents, constant=constants.get("potential"))
        descriptions.append(f"Potential:    {desc}")
    elif resolution in ("E", "F"):
        # Blend + AR(1) z: r* = alpha*g + (1-alpha)*(indexed-k) + z
        # F additionally has the SOE block in the IS curve (kept in obs above).
        desc = r_star_blended_z_equation(
            obs, model, latents, constant=constants.get("r_star"),
        )
        descriptions.append(f"r-star:       {desc}")

        desc = potential_output_equation(obs, model, latents, constant=constants.get("potential"))
        descriptions.append(f"Potential:    {desc}")
    else:  # Resolution A, B, or D (canonical r* = g + z)
        # Order: trend_growth -> potential -> z_star (uses sigma_ystar via lambda_z)
        desc = potential_output_equation(obs, model, latents, constant=constants.get("potential"))
        descriptions.append(f"Potential:    {desc}")

        desc = z_star_equation(obs, model, latents, constant=constants.get("z_star"))
        descriptions.append(f"z-star:       {desc}")

    # --- Observation equations ---
    desc = is_curve_equation(obs, model, latents, constant=constants.get("is_curve"))
    descriptions.append(f"IS curve:     {desc}")

    desc = phillips_curve_equation(obs, model, latents, constant=constants.get("phillips"))
    descriptions.append(f"Phillips:     {desc}")

    if resolution == "B":
        desc = indexed_bond_equation(obs, model, latents, constant=constants.get("indexed_bond"))
        descriptions.append(f"Indexed bond: {desc}")

    print("\nHLW r-star model equations:")
    for d in descriptions:
        print(f"  {d}")

    fixed = getattr(model, "_fixed_constants", {})
    if fixed:
        print("\nFixed constants:")
        for name, value in fixed.items():
            print(f"  {name} = {value}")

    model._descriptions = descriptions  # noqa: SLF001
    return model


def save_results(
    trace: az.InferenceData,
    obs: dict[str, np.ndarray],
    obs_index: pd.PeriodIndex,
    constants: dict[str, Any] | None = None,
    chart_obs: pd.DataFrame | None = None,
    output_dir: Path | str | None = None,
    prefix: str = "rstar_hlw",
) -> Path:
    """Persist trace + observations + metadata to disk."""
    if output_dir is None:
        output_dir = DEFAULT_OUTPUT_DIR
    if constants is None:
        constants = {}
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    trace_path = output_dir / f"{prefix}_trace.nc"
    trace.to_netcdf(str(trace_path))
    print(f"Saved trace to: {trace_path}")

    obs_path = output_dir / f"{prefix}_obs.pkl"
    with obs_path.open("wb") as f:
        pickle.dump(
            {
                "obs": obs,
                "obs_index": obs_index,
                "constants": constants,
                "chart_obs": chart_obs,
            },
            f,
        )
    print(f"Saved observations to: {obs_path}")

    return output_dir


def run_estimate(
    start: str | None = "1980Q1",
    end: str | None = None,
    sampler_config: SamplerConfig | None = None,
    output_dir: Path | str | None = None,
    prefix: str = "rstar_hlw",
    resolution: str = "G",
    verbose: bool = False,
    seed: int | None = None,
) -> tuple[az.InferenceData, dict[str, np.ndarray], pd.PeriodIndex]:
    """Build observations, sample posterior, save results."""
    if sampler_config is None:
        kwargs = {
            "draws": 10_000,
            "tune": 3_500,
            "chains": 5,
            "cores": 5,
            "target_accept": 0.90,
        }
        if seed is not None:
            kwargs["random_seed"] = seed
        sampler_config = SamplerConfig(**kwargs)
    elif seed is not None:
        sampler_config.random_seed = seed
    print(f"Sampler seed: {sampler_config.random_seed}")

    print("Building observations...")
    obs, obs_index, chart_obs = build_observations(
        start=start, end=end, verbose=verbose,
    )

    print(f"Building model (Resolution {resolution})...")
    model = build_model(obs, resolution=resolution)

    print("\nSampling...")
    trace = sample_model(model, sampler_config)
    print()

    constants = get_fixed_constants(model)
    save_results(
        trace, obs, obs_index,
        constants=constants,
        chart_obs=chart_obs,
        output_dir=output_dir,
        prefix=prefix,
    )

    return trace, obs, obs_index
