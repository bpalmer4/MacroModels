"""Phillips curve observation equation.

This is what makes the trend/cycle split economic rather than statistical. The
AR(2) restriction in `output.py` says the gap must look like a cycle; the
Phillips curve says the gap must also be the thing that moves inflation. Take
this equation away and the model is a filter.

Anchored form, on annual trimmed mean inflation, with the anchor fixed at the
RBA's 2.5% target midpoint rather than taken from an expectations series. The
anchor and the 1993Q1 sample start are a matched pair: over the target period
the anchor is a known regime fact, so it costs no parameters, and it is a
sharper restriction than the accelerationist alternative (which would assume
the sum of the lagged-inflation coefficients is exactly one, i.e. that the
anchor never moved).

The reason an anchor is needed at all is arithmetic. Trimmed mean inflation
averaged 6.74% pre-1993 and 2.67% from 1993. The gap is a mean-zero stationary
object, so with no anchor to absorb a level shift of that size, the model would
have to book the regime change as a decade of slack — and deliver it by pushing
potential output up to match.

beta is held away from zero (lower=0.02) for the same reason the project's HLW
Phillips curve does it: at beta = 0 the gap is unidentified and y* can absorb
all of output.

Two things the first version of this equation got wrong, both of which made
inflation more informative about the gap than it is entitled to be:

- **Overlapping observations.** The left-hand side was four-quarter trimmed
  mean inflation observed quarterly, with iid errors. Consecutive observations
  then share three CPI quarters, so the error is MA(3) by construction even
  under a correct model, and the likelihood counts roughly four times as many
  independent observations as exist. `ModelConfig.pi_basis="quarterly"` uses
  the non-overlapping quarterly rate annualised instead; "annual" keeps the
  old behaviour for comparison.
- **No supply side.** With no cost-push term, imported and supply-driven
  inflation has nowhere to go but the demand gap, so a run of supply-driven
  inflation reads as repeated evidence of excess demand and pushes potential
  output down. `ModelConfig.supply_control="import_prices"` adds demeaned
  annual consumption-goods import price growth lagged one quarter, the
  cost-push term the RBA's MARTIN uses (RDP 2019-07 s4.7.3).

The supply control is demeaned in `observations.py` rather than here, and it
must be: with the anchor fixed, an undemeaned regressor shifts the anchor
instead of explaining deviations from it.
"""

from typing import Any

import numpy as np
import pymc as pm

from src.models.ystar.base import set_model_coefficients


def phillips_curve_equation(
    obs: dict[str, np.ndarray],
    model: pm.Model,
    latents: dict[str, Any],
    constant: dict[str, Any] | None = None,
) -> str:
    """Anchored Phillips curve on trimmed mean inflation.

    Model: pi_t = anchor + beta · gap_{t-1} [+ gamma · supply_t] + e_pi

    pi is in annual %, whichever basis `observations.py` supplied; the gap is
    in log x 100 units, so beta translates one log point of output gap into
    percentage points of annual inflation.

    The supply term is optional and is present only when `obs` carries a
    "supply" column, which `build_observations` adds when
    `ModelConfig.supply_control` is set. It is already demeaned.

    `anchor` must be supplied via `constant` (ModelConfig.anchor).
    """
    if constant is None:
        constant = {}

    if "anchor" not in constant:
        raise ValueError("phillips_curve_equation requires a fixed 'anchor' — pass ModelConfig.anchor")

    supply = obs.get("supply")

    # Scalar or one value per quarter. A phased anchor is a series, because
    # before target adoption the level inflation returns to is measured
    # expectations rather than a target that did not exist. Sliced to match
    # the likelihood below, which drops the first quarter for the lagged gap.
    anchor = np.asarray(constant["anchor"], dtype=float)
    anchor_t = anchor if anchor.ndim == 0 else anchor[1:]

    with model:
        settings: dict[str, dict[str, float]] = {
            "beta": {"mu": 0.15, "sigma": 0.10, "lower": 0.02},
            "sigma_pi": {"sigma": 0.40},
        }
        if supply is not None:
            # Pass-through of import prices to consumer prices is a fraction of
            # a per cent per per cent, and it is a cost, so the sign is known.
            settings["gamma"] = {"mu": 0.05, "sigma": 0.05, "lower": 0.0}
        mc = set_model_coefficients(model, settings, {})

        predicted = anchor_t + mc["beta"] * latents["output_gap"][:-1]
        if supply is not None:
            predicted = predicted + mc["gamma"] * supply[1:]

        pm.Normal(
            "observed_pi",
            mu=predicted,
            sigma=mc["sigma_pi"],
            observed=obs["pi"][1:],
        )

    term = " + gamma · supply_t" if supply is not None else ""
    label = f"{anchor.flat[0]:g}" if anchor.ndim == 0 or anchor.min() == anchor.max() else "a_t"
    return f"pi_t = {label} + beta · gap_{{t-1}}{term} + e_pi"
