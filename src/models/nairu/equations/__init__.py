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

from src.models.nairu.equations.employment import employment_equation
from src.models.nairu.equations.exchange_rate import exchange_rate_equation
from src.models.nairu.equations.import_price import import_price_equation
from src.models.nairu.equations.is_curve import is_equation
from src.models.nairu.equations.net_exports import net_exports_equation
from src.models.nairu.equations.okun import okun_law_equation
from src.models.nairu.equations.participation import participation_equation
from src.models.nairu.equations.phillips import (
    hourly_coe_equation,
    price_inflation_equation,
    wage_growth_equation,
)
from src.models.nairu.equations.production import potential_output_equation
from src.models.nairu.equations.state_space import nairu_equation

__all__ = [
    # Regime constants
    "REGIME_GFC_START",
    "REGIME_COVID_START",
    # Equations
    "employment_equation",
    "exchange_rate_equation",
    "hourly_coe_equation",
    "import_price_equation",
    "is_equation",
    "nairu_equation",
    "net_exports_equation",
    "okun_law_equation",
    "participation_equation",
    "potential_output_equation",
    "price_inflation_equation",
    "wage_growth_equation",
]
