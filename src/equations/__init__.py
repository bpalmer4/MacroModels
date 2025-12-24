"""Reusable equation building blocks for Bayesian state-space models.

Each equation function takes:
- inputs: Dict of observed data arrays
- model: PyMC model context
- constant: Optional dict of fixed parameter values

And adds distributions to the model, returning key latent variables.
"""

import pandas as pd

# Regime boundary periods (Phillips curve slope regimes)
REGIME_GFC_START = pd.Period("2008Q4")  # Start of post-GFC (flat) regime - Lehman Sep 2008
REGIME_COVID_START = pd.Period("2021Q1")  # Start of post-COVID (steep) regime

from src.equations.exchange_rate import exchange_rate_equation
from src.equations.import_price import import_price_equation
from src.equations.is_curve import is_equation
from src.equations.okun import okun_law_equation
from src.equations.participation import participation_equation
from src.equations.phillips import (
    hourly_coe_equation,
    price_inflation_equation,
    wage_growth_equation,
)
from src.equations.production import potential_output_equation
from src.equations.state_space import nairu_equation

__all__ = [
    # Regime constants
    "REGIME_GFC_START",
    "REGIME_COVID_START",
    # Equations
    "exchange_rate_equation",
    "hourly_coe_equation",
    "import_price_equation",
    "is_equation",
    "nairu_equation",
    "okun_law_equation",
    "participation_equation",
    "potential_output_equation",
    "price_inflation_equation",
    "wage_growth_equation",
]
