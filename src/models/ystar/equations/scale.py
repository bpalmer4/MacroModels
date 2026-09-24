"""Variance scale for the ystar model.

One imposed scale, three imposed ratios. This is where the model's central
identifying assumption lives, so it is a separate step rather than being
scattered through the state equations.

sigma_c — the cycle innovation sd — is fixed, and every trend innovation sd is
a fixed multiple of it. Fixing the trend/cycle variance split rather than
estimating it is the Kuttner (1994) / HP-filter choice: it avoids the
Stock-Watson pile-up problem (where the likelihood puts a mass point at
sigma_trend = 0 when the signal-to-noise ratio is small) by refusing to ask
the data a question it cannot answer.

sigma_c must be fixed rather than estimated, not merely for parsimony. With it
free, the ratios impose no smoothness: a badly-fitting cycle equation pushes
sigma_c up, which loosens every trend in proportion, which lets the trend chase
the data harder, which worsens the cycle fit again. The feedback runs the wrong
way. Pinning the absolute scale is what makes the restriction bind.

The price is that the trend/cycle split is conditional on the ratios. That is
not hidden — it is what `sigma_sweep.py` measures.
"""

from typing import Any

import numpy as np
import pymc as pm

from src.models.ystar.base import set_model_coefficients


def scale_equation(
    obs: dict[str, np.ndarray],
    model: pm.Model,
    latents: dict[str, Any],
    constant: dict[str, Any] | None = None,
) -> str:
    """Fix sigma_c and derive the fixed-ratio trend innovation sigmas.

    The scale and the three ratios must all be supplied via `constant`
    (ModelConfig.scale_constants); they are assumptions, not parameters, so
    there is no prior to fall back on.
    """
    if constant is None:
        constant = {}

    if "sigma_c" not in constant:
        raise ValueError("scale_equation requires a fixed 'sigma_c' — pass ModelConfig.scale_constants")

    ratios = sorted(key for key in constant if key.startswith("ratio_"))
    if not ratios:
        raise ValueError("scale_equation requires at least one 'ratio_*' — pass ModelConfig.scale_constants")

    with model:
        settings: dict[str, dict[str, float]] = {key: {} for key in ("sigma_c", *ratios)}
        mc = set_model_coefficients(model, settings, constant)

    # All four are fixed constants, so the derived sigmas are plain floats
    # rather than PyMC Deterministics — there is nothing random to record.
    # Each ratio_<name> becomes sigma_<name>, so the same equation serves any
    # specification. All are fixed constants, hence plain floats rather than
    # PyMC Deterministics — there is nothing random to record.
    sigma_c = float(mc["sigma_c"])
    latents["sigma_c"] = sigma_c
    for key in ratios:
        latents[key.replace("ratio_", "sigma_", 1)] = float(mc[key]) * sigma_c

    derived = ",  ".join(
        f"{key.replace('ratio_', 'sigma_', 1)} = {constant[key]:g}·sigma_c" for key in ratios
    )
    return f"sigma_c = {constant['sigma_c']:g} (fixed);  {derived}"
