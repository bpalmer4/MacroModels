"""Import price pass-through equation linking import prices to TWI.

Implements an exchange rate pass-through (ERPT) equation that models how
exchange rate movements translate into import price changes.

Exchange Rate Pass-Through (ERPT)
---------------------------------
Pass-through measures the percentage change in import prices resulting from
a 1% change in the exchange rate. Under the law of one price, pass-through
should be complete (coefficient = -1.0): a 10% depreciation raises import
prices by 10%.

In practice, pass-through is typically incomplete due to:

    1. Pricing-to-market: Foreign exporters absorb exchange rate changes
       in their margins to maintain market share
    2. Local distribution costs: Retail prices include domestic labour,
       rent, and transport that don't respond to exchange rates
    3. Invoice currency: Trade invoiced in USD or other vehicle currencies
       may respond differently than bilateral rate movements suggest
    4. Menu costs: Firms adjust prices infrequently, smoothing pass-through

Empirical Evidence
------------------
Pass-through varies across countries and has declined over time in many
economies. For Australia:

    - Short-run pass-through to import prices: 0.4-0.6 (incomplete)
    - Long-run pass-through approaches unity for commodity imports
    - Pass-through to CPI is lower (0.05-0.15) due to import share and margins
    - Decline in pass-through attributed to low-inflation credibility

This equation models pass-through from TWI to import prices at the border.
The Phillips curve then captures the subsequent pass-through to CPI.

Sign Convention
---------------
The TWI is measured as an index where higher = AUD appreciation. Therefore:

    - Δ4twi > 0: AUD appreciated → import prices should fall (Δ4ρm < 0)
    - Expected coefficient on Δ4twi is NEGATIVE

Oil Prices
----------
Oil is a significant component of Australia's imports and has direct pass-through
to import prices. We include oil prices converted to AUD (from World Bank USD
prices and RBA exchange rates) as an additional driver. The coefficient on oil
is expected to be positive: higher oil prices raise import costs.

References
----------
- Campa, J. and Goldberg, L. (2005). "Exchange Rate Pass-Through into Import
  Prices." Review of Economics and Statistics, 87(4), 679-690.
- Chung, E., Kohler, M. and Lewis, C. (2011). "The Exchange Rate and Consumer
  Prices." RBA Bulletin, September Quarter, 9-16.

"""

from typing import Any

import numpy as np
import pymc as pm

from src.models.nairu.base import set_model_coefficients


def import_price_equation(
    inputs: dict[str, np.ndarray],
    model: pm.Model,
    constant: dict[str, Any] | None = None,
) -> None:
    """Import price pass-through equation.

    Exchange rate pass-through equation with oil prices:
        Δ4ρm_t = β_pt·Δ4twi_{t-1} + β_oil·Δ4oil_{t-1} + ρ·Δ4ρm_{t-1} + ε

    Where:
        - Δ4ρm = annual import price inflation (4-quarter log diff)
        - Δ4twi = annual TWI change (positive = appreciation)
        - Δ4oil = annual oil price change in AUD
        - β_pt = pass-through coefficient (expected negative)
        - β_oil = oil price coefficient (expected positive)
        - ρ = persistence parameter

    Args:
        inputs: Must contain:
            - "Δ4ρm": Annual import price growth (current)
            - "Δ4ρm_1": Annual import price growth (lagged)
            - "Δ4twi_1": Annual TWI change (lagged)
            - "Δ4oil_1": Annual oil price change in AUD (lagged)
        model: PyMC model context
        constant: Optional fixed values for coefficients

    Note:
        Pass-through coefficient is truncated at 0 from above (upper=0) to
        enforce the theoretical sign: appreciation reduces import prices.
        Oil coefficient is truncated at 0 from below (lower=0) since higher
        oil prices should raise import costs.

        A second TWI lag was tested but found to be negligible (beta ≈ 0),
        indicating pass-through is essentially immediate.

    """
    if constant is None:
        constant = {}

    with model:
        settings = {
            # Pass-through coefficient
            # Negative: TWI appreciation → lower import prices
            "beta_pt": {"mu": -0.3, "sigma": 0.15, "upper": 0},
            # Oil price effect
            # Positive: higher oil prices → higher import prices
            "beta_oil": {"mu": 0.10, "sigma": 0.05, "lower": 0},
            # Persistence: import price inflation is autocorrelated
            "rho_pt": {"mu": 0.4, "sigma": 0.15},
            # Error term
            "epsilon_pt": {"sigma": 2.5},
        }
        mc = set_model_coefficients(model, settings, constant)

        # All inputs are pre-aligned (lags computed in build_observations)
        delta_4_twi_lag1 = inputs["Δ4twi_1"]
        delta_4_oil_lag1 = inputs["Δ4oil_1"]
        delta_4_pm_lag1 = inputs["Δ4ρm_1"]

        # Predicted import price inflation
        predicted_delta_pm = (
            mc["beta_pt"] * delta_4_twi_lag1
            + mc["beta_oil"] * delta_4_oil_lag1
            + mc["rho_pt"] * delta_4_pm_lag1
        )

        # Observation equation
        pm.Normal(
            "observed_import_price",
            mu=predicted_delta_pm,
            sigma=mc["epsilon_pt"],
            observed=inputs["Δ4ρm"],
        )
