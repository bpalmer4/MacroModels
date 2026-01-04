"""Exchange rate equation linking TWI to interest rate differentials.

Implements a UIP-style equation for the Trade-Weighted Index, modified
for empirical realism with persistence and terms of trade effects.

Uncovered Interest Parity (UIP)
-------------------------------
UIP is a fundamental arbitrage condition in international finance. It states
that the expected change in the exchange rate should equal the interest rate
differential between two countries:

    E[Δe] = i - i*

Where:
    - E[Δe] = expected exchange rate change (domestic currency per foreign)
    - i = domestic interest rate
    - i* = foreign interest rate

The intuition: if Australian rates exceed foreign rates, capital should flow
into Australia seeking higher returns. This demand for AUD causes appreciation.
The appreciation continues until expected future depreciation exactly offsets
the interest rate advantage, eliminating the arbitrage opportunity.

The UIP Puzzle (Forward Premium Puzzle)
---------------------------------------
Despite its theoretical elegance, UIP consistently fails empirically. Decades
of research since Fama (1984) have documented that:

    1. The coefficient on interest rate differentials is often near zero,
       sometimes negative, and rarely close to the theoretical value of 1.0
    2. High-interest-rate currencies tend to appreciate (not depreciate as
       UIP predicts), giving rise to the profitable "carry trade" strategy
    3. Exchange rates exhibit excess volatility unexplained by fundamentals

This is one of the most robust puzzles in international finance. Explanations
include time-varying risk premia, peso problems, learning, and limits to
arbitrage.

Implications for This Model
---------------------------
Finding weak or near-zero coefficients on the interest rate gap (beta_er_r)
is not a model specification error - it reflects the well-documented empirical
failure of UIP. A coefficient of 0.05-0.15 is typical in the literature and
indicates that interest rate differentials have modest predictive power for
exchange rate movements.

The large residual variance (epsilon_er) captures the substantial unexplained
volatility in exchange rates that is characteristic of floating rate regimes.

References
----------
- Fama, E. (1984). "Forward and Spot Exchange Rates." Journal of Monetary
  Economics, 14(3), 319-338.
- Engel, C. (1996). "The Forward Discount Anomaly and the Risk Premium:
  A Survey of Recent Evidence." Journal of Empirical Finance, 3(2), 123-192.
- Burnside, C. et al. (2011). "Carry Trade and Momentum in Currency Markets."
  Annual Review of Financial Economics, 3, 511-535.

"""

from typing import Any

import numpy as np
import pymc as pm

from src.models.nairu.base import set_model_coefficients


def exchange_rate_equation(
    inputs: dict[str, np.ndarray],
    model: pm.Model,
    constant: dict[str, Any] | None = None,
) -> None:
    """Exchange rate equation linking TWI changes to fundamentals.

    UIP-style equation with persistence:
        Δe_t = ρ·Δe_{t-1} + β_r·(r_{t-1} - r*) + ε

    Where:
        - Δe = change in log TWI (positive = appreciation)
        - r - r* = real interest rate gap (higher → appreciation)
        - ρ = persistence parameter

    Theoretical motivation:
        - UIP suggests higher domestic rates attract capital → appreciation
        - Persistence captures slow adjustment / momentum trading

    Args:
        inputs: Must contain:
            - "Δtwi": Change in log TWI (current period)
            - "Δtwi_1": Lagged change in log TWI
            - "r_gap_1": Lagged real interest rate gap
        model: PyMC model context
        constant: Optional fixed values for coefficients

    Note:
        The interest rate coefficient is expected to be positive (higher rates
        attract capital, appreciating the currency). This differs from some
        textbook UIP formulations where the sign convention varies.

        A terms-of-trade proxy (import prices) was tested but found
        indistinguishable from zero, so excluded for parsimony.

    """
    if constant is None:
        constant = {}

    with model:
        settings = {
            # Persistence: TWI changes are serially correlated
            "rho_er": {"mu": 0.3, "sigma": 0.15},
            # Interest rate differential effect
            # Positive: higher rates → appreciation (capital inflows)
            # Truncated at 0 to enforce UIP sign constraint
            "beta_er_r": {"mu": 0.3, "sigma": 0.2, "lower": 0},
            # Error term (TWI is volatile)
            "epsilon_er": {"sigma": 3.0},
        }
        mc = set_model_coefficients(model, settings, constant)

        # All inputs are pre-aligned (lags computed in build_observations)
        delta_twi_lag1 = inputs["Δtwi_1"]
        r_gap_lag1 = inputs["r_gap_1"]

        # Predicted TWI change
        # Δe_t = ρ·Δe_{t-1} + β_r·r_gap_{t-1}
        # Note: ToT proxy (import prices) was tested but found indistinguishable
        # from zero, so removed for parsimony.
        predicted_delta_twi = (
            mc["rho_er"] * delta_twi_lag1
            + mc["beta_er_r"] * r_gap_lag1
        )

        # Observation equation
        pm.Normal(
            "observed_twi_change",
            mu=predicted_delta_twi,
            sigma=mc["epsilon_er"],
            observed=inputs["Δtwi"],
        )
