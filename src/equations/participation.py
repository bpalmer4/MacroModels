"""Participation rate equation linking labor force participation to the cycle.

Models how labor force participation responds to labor market conditions,
capturing the discouraged and added worker effects.

Discouraged Worker Effect
-------------------------
When labor market conditions deteriorate (unemployment rises above NAIRU),
some unemployed workers stop actively seeking work and exit the labor force.
This reduces measured unemployment but represents hidden slack.

The effect is well-documented empirically:
    - Stronger during recessions when job-finding rates are low
    - More pronounced for secondary earners and older workers
    - Creates procyclical participation (participation falls in downturns)

Added Worker Effect
-------------------
When primary earners lose jobs, secondary household members may enter the
labor force to maintain household income. This works opposite to the
discouraged worker effect, creating countercyclical participation.

Empirically, the discouraged worker effect typically dominates, so we
expect net procyclical participation (negative coefficient on unemployment gap).

Implications for NAIRU Estimation
---------------------------------
Participation dynamics matter for NAIRU estimation because:

    1. Measured unemployment understates true slack when workers are discouraged
    2. A shock that raises unemployment also reduces participation,
       dampening the measured unemployment response
    3. Separating demand from supply shocks requires understanding
       how much of labor force change is cyclical vs structural

The participation equation provides additional identifying information
for the NAIRU state by exploiting the cyclical participation response.

Model Specification Notes
-------------------------
A persistence term (ρ·Δpr_{t-1}) was tested but found to be statistically
indistinguishable from zero. This indicates that participation changes are
essentially "one-shot" responses to labor market slack rather than exhibiting
serial correlation beyond what's driven by the unemployment gap itself.
The simplified specification without persistence is preferred for parsimony.

References
----------
- Benati, L. (2001). "Some Empirical Evidence on the 'Discouraged Worker'
  Effect." Economics Letters, 70(3), 387-395.
- Erceg, C. and Levin, A. (2014). "Labor Force Participation and Monetary
  Policy in the Wake of the Great Recession." Journal of Money, Credit
  and Banking, 46(S2), 3-49.

"""

from typing import Any

import numpy as np
import pytensor.tensor as pt
import pymc as pm

from src.models.base import set_model_coefficients


def participation_equation(
    inputs: dict[str, np.ndarray],
    model: pm.Model,
    nairu: pt.TensorVariable,
    constant: dict[str, Any] | None = None,
) -> None:
    """Participation rate equation.

    Cyclical participation response (discouraged worker effect):
        Δpr_t = β_pr·(U_{t-1} - NAIRU_{t-1}) + ε

    Where:
        - Δpr = change in participation rate (pp)
        - U - NAIRU = unemployment gap (lagged)
        - β_pr = discouraged worker coefficient (expected negative)

    Theoretical motivation:
        - When U > NAIRU (slack), discouraged workers exit → Δpr < 0
        - When U < NAIRU (tight), workers drawn in → Δpr > 0

    Args:
        inputs: Must contain:
            - "Δpr": Change in participation rate (current period)
            - "U_1": Lagged unemployment rate
        model: PyMC model context
        nairu: NAIRU tensor variable (provides NAIRU_{t-1} via shift)
        constant: Optional fixed values for coefficients

    Note:
        The coefficient β_pr is expected to be negative: higher unemployment
        gap (more slack) leads to falling participation (discouraged workers).
        We use a truncated prior (upper=0) to enforce this sign constraint.

        The unemployment gap uses lagged values to avoid simultaneity:
        participation decisions respond to observed labor market conditions.

    """
    if constant is None:
        constant = {}

    with model:
        settings = {
            # Discouraged worker effect
            # Negative: unemployment gap > 0 → falling participation
            "beta_pr": {"mu": -0.05, "sigma": 0.03, "upper": 0},
            # Error term
            "epsilon_pr": {"sigma": 0.3},
        }
        mc = set_model_coefficients(model, settings, constant)

        # All inputs are pre-aligned (lags computed in build_observations)
        U_lag1 = inputs["U_1"]

        # NAIRU is a tensor - need lagged value
        # nairu has shape (T,) aligned with observations
        # We use nairu[:-1] padded with first value for lag
        nairu_lag1 = pt.concatenate([[nairu[0]], nairu[:-1]])

        # Unemployment gap (lagged)
        u_gap_lag1 = U_lag1 - nairu_lag1

        # Predicted participation change
        # Δpr_t = β_pr·(U_{t-1} - NAIRU_{t-1})
        predicted_delta_pr = mc["beta_pr"] * u_gap_lag1

        # Observation equation
        pm.Normal(
            "observed_participation",
            mu=predicted_delta_pr,
            sigma=mc["epsilon_pr"],
            observed=inputs["Δpr"],
        )
