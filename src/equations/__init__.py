"""Reusable equation building blocks for Bayesian state-space models.

Each equation function takes:
- inputs: Dict of observed data arrays
- model: PyMC model context
- constant: Optional dict of fixed parameter values

And adds distributions to the model, returning key latent variables.
"""

from src.equations.exchange_rate import exchange_rate_equation
from src.equations.import_price import import_price_equation
from src.equations.is_curve import is_equation
from src.equations.okun import okun_law_equation
from src.equations.participation import participation_equation
from src.equations.phillips import price_inflation_equation, wage_growth_equation
from src.equations.production import potential_output_equation
from src.equations.state_space import nairu_equation

__all__ = [
    "exchange_rate_equation",
    "import_price_equation",
    "is_equation",
    "nairu_equation",
    "okun_law_equation",
    "participation_equation",
    "potential_output_equation",
    "price_inflation_equation",
    "wage_growth_equation",
]
