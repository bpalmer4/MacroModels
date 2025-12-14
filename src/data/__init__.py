"""Data loading and transformation utilities for ABS and RBA data."""

from src.data.abs_loader import (
    ReqsDict,
    ReqsTuple,
    get_abs_data,
    load_series,
)
from src.data.capital import get_capital_growth_qrtly, get_capital_stock_qrtly
from src.data.cash_rate import get_cash_rate_monthly, get_cash_rate_qrtly
from src.data.dataseries import DataSeries
from src.data.gdp import get_gdp, get_gdp_growth, get_log_gdp
from src.data.gscpi import get_gscpi_monthly, get_gscpi_qrtly
from src.data.henderson import hma
from src.data.import_prices import get_import_price_growth_annual, get_import_price_index_qrtly
from src.data.inflation import get_inflation_annual, get_inflation_qrtly
from src.data.labour_force import (
    get_hours_worked_monthly,
    get_hours_worked_qrtly,
    get_labour_force_growth_qrtly,
    get_labour_force_qrtly,
    get_unemployed_qrtly,
    get_unemployment_change_qrtly,
    get_unemployment_rate_qrtly,
)
from src.data.mfp import get_mfp_annual
from src.data.transforms import splice_series
from src.data.ulc import get_ulc_growth_qrtly

__all__ = [
    # Core loaders
    "DataSeries",
    "ReqsDict",
    "ReqsTuple",
    "get_abs_data",
    "load_series",
    # GDP
    "get_gdp",
    "get_gdp_growth",
    "get_log_gdp",
    # Labour force
    "get_hours_worked_monthly",
    "get_hours_worked_qrtly",
    "get_labour_force_growth_qrtly",
    "get_labour_force_qrtly",
    "get_unemployed_qrtly",
    "get_unemployment_change_qrtly",
    "get_unemployment_rate_qrtly",
    # Capital
    "get_capital_growth_qrtly",
    "get_capital_stock_qrtly",
    # MFP
    "get_mfp_annual",
    # Inflation
    "get_inflation_annual",
    "get_inflation_qrtly",
    # Interest rates
    "get_cash_rate_monthly",
    "get_cash_rate_qrtly",
    # Costs
    "get_ulc_growth_qrtly",
    # Trade
    "get_import_price_growth_annual",
    "get_import_price_index_qrtly",
    # Supply shocks
    "get_gscpi_monthly",
    "get_gscpi_qrtly",
    # Utilities
    "hma",
    "splice_series",
]
