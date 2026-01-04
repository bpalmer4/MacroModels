"""Data loading and transformation utilities for ABS and RBA data."""

from src.data.abs_loader import (
    ReqsDict,
    ReqsTuple,
    get_abs_data,
    load_series,
)
from src.data.awe import get_awe_growth_annual, get_awe_growth_qrtly, get_awe_index
from src.data.capital import get_capital_growth_qrtly, get_capital_stock_qrtly
from src.data.capital_share import get_capital_share
from src.data.cash_rate import (
    compute_r_star,
    get_cash_rate_monthly,
    get_cash_rate_qrtly,
    get_real_rate_gap_lagged_qrtly,
    get_real_rate_gap_qrtly,
)
from src.data.dataseries import DataSeries
from src.data.debt_servicing import (
    get_dsr_change_lagged_qrtly,
    get_dsr_change_qrtly,
    get_dsr_qrtly,
    get_mortgage_buffers_qrtly,
)
from src.data.dfd_deflator import get_dfd_deflator_growth_annual
from src.data.energy import (
    get_coal_change_annual,
    get_coal_price_aud_qrtly,
    get_coal_price_usd_monthly,
    get_oil_change_annual,
    get_oil_change_lagged_annual,
    get_oil_price_aud_monthly,
    get_oil_price_aud_qrtly,
    get_oil_price_usd_monthly,
)
from src.data.foreign_demand import (
    get_major_trading_partner_growth_qrtly,
    get_world_gdp_growth_qrtly,
)
from src.data.gdp import get_gdp, get_gdp_growth, get_log_gdp
from src.data.gov_spending import (
    get_fiscal_impulse_lagged_qrtly,
    get_fiscal_impulse_qrtly,
    get_gov_consumption_qrtly,
    get_gov_growth_qrtly,
)
from src.data.gscpi import get_gscpi_covid_lagged_qrtly, get_gscpi_monthly, get_gscpi_qrtly
from src.data.henderson import hma
from src.data.hourly_coe import (
    get_hourly_coe,
    get_hourly_coe_growth_annual,
    get_hourly_coe_growth_lagged_qrtly,
    get_hourly_coe_growth_qrtly,
)
from src.data.house_prices import (
    get_housing_wealth_growth_annual,
    get_housing_wealth_growth_lagged_qrtly,
    get_housing_wealth_growth_qrtly,
    get_housing_wealth_qrtly,
)
from src.data.household import (
    get_saving_ratio_change_qrtly,
    get_saving_ratio_qrtly,
)
from src.data.import_prices import (
    get_import_price_growth_annual,
    get_import_price_growth_lagged_annual,
    get_import_price_index_qrtly,
)
from src.data.inflation import get_inflation_annual, get_inflation_qrtly
from src.data.labour_force import (
    get_employed_qrtly,
    get_employment_growth_lagged_qrtly,
    get_employment_growth_qrtly,
    get_hours_growth_qrtly,
    get_hours_worked_monthly,
    get_hours_worked_qrtly,
    get_labour_force_growth_qrtly,
    get_labour_force_qrtly,
    get_participation_rate_change_qrtly,
    get_participation_rate_monthly,
    get_participation_rate_qrtly,
    get_unemployed_qrtly,
    get_unemployment_change_qrtly,
    get_unemployment_rate_lagged_qrtly,
    get_unemployment_rate_qrtly,
    get_unemployment_speed_limit_qrtly,
)
from src.data.mfp import get_mfp_annual
from src.data.net_exports import (
    get_export_growth_qrtly,
    get_exports_qrtly,
    get_import_growth_qrtly,
    get_imports_qrtly,
    get_net_exports_ratio_change_qrtly,
    get_net_exports_ratio_qrtly,
)
from src.data.productivity import (
    compute_mfp_trend_floored,
    get_labour_productivity_growth,
    get_mfp_growth,
    get_real_wage_gap,
)
from src.data.tot import get_tot_change_qrtly
from src.data.transforms import splice_series
from src.data.twi import (
    get_log_twi_qrtly,
    get_real_twi_qrtly,
    get_twi_change_annual,
    get_twi_change_lagged_annual,
    get_twi_change_lagged_qrtly,
    get_twi_change_qrtly,
    get_twi_monthly,
    get_twi_qrtly,
)
from src.data.ulc import get_ulc_growth_lagged_qrtly, get_ulc_growth_qrtly

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
    "get_employed_qrtly",
    "get_employment_growth_lagged_qrtly",
    "get_employment_growth_qrtly",
    "get_hours_growth_qrtly",
    "get_hours_worked_monthly",
    "get_hours_worked_qrtly",
    "get_labour_force_growth_qrtly",
    "get_labour_force_qrtly",
    "get_participation_rate_change_qrtly",
    "get_participation_rate_monthly",
    "get_participation_rate_qrtly",
    "get_unemployed_qrtly",
    "get_unemployment_change_qrtly",
    "get_unemployment_rate_lagged_qrtly",
    "get_unemployment_rate_qrtly",
    "get_unemployment_speed_limit_qrtly",
    # Capital
    "get_capital_growth_qrtly",
    "get_capital_stock_qrtly",
    "get_capital_share",
    # MFP and productivity (derived)
    "get_mfp_annual",
    "get_labour_productivity_growth",
    "get_mfp_growth",
    "compute_mfp_trend_floored",
    "get_real_wage_gap",
    # Inflation
    "get_inflation_annual",
    "get_inflation_qrtly",
    # Interest rates
    "compute_r_star",
    "get_cash_rate_monthly",
    "get_cash_rate_qrtly",
    "get_real_rate_gap_lagged_qrtly",
    "get_real_rate_gap_qrtly",
    # Wages and labour costs
    "get_ulc_growth_lagged_qrtly",
    "get_ulc_growth_qrtly",
    "get_awe_index",
    "get_awe_growth_qrtly",
    "get_awe_growth_annual",
    "get_hourly_coe",
    "get_hourly_coe_growth_lagged_qrtly",
    "get_hourly_coe_growth_qrtly",
    "get_hourly_coe_growth_annual",
    # Demand deflator
    "get_dfd_deflator_growth_annual",
    # Trade
    "get_import_price_growth_annual",
    "get_import_price_growth_lagged_annual",
    "get_import_price_index_qrtly",
    "get_tot_change_qrtly",
    # Net exports
    "get_exports_qrtly",
    "get_imports_qrtly",
    "get_net_exports_ratio_qrtly",
    "get_net_exports_ratio_change_qrtly",
    "get_export_growth_qrtly",
    "get_import_growth_qrtly",
    # Foreign demand
    "get_major_trading_partner_growth_qrtly",
    "get_world_gdp_growth_qrtly",
    # Energy (oil and coal)
    "get_coal_change_annual",
    "get_coal_price_aud_qrtly",
    "get_coal_price_usd_monthly",
    "get_oil_change_annual",
    "get_oil_change_lagged_annual",
    "get_oil_price_aud_monthly",
    "get_oil_price_aud_qrtly",
    "get_oil_price_usd_monthly",
    # Government spending
    "get_fiscal_impulse_lagged_qrtly",
    "get_fiscal_impulse_qrtly",
    "get_gov_consumption_qrtly",
    "get_gov_growth_qrtly",
    # Household
    "get_saving_ratio_change_qrtly",
    "get_saving_ratio_qrtly",
    # Debt servicing
    "get_dsr_qrtly",
    "get_dsr_change_qrtly",
    "get_dsr_change_lagged_qrtly",
    "get_mortgage_buffers_qrtly",
    # Housing wealth
    "get_housing_wealth_qrtly",
    "get_housing_wealth_growth_qrtly",
    "get_housing_wealth_growth_annual",
    "get_housing_wealth_growth_lagged_qrtly",
    # Exchange rates
    "get_log_twi_qrtly",
    "get_real_twi_qrtly",
    "get_twi_change_annual",
    "get_twi_change_lagged_annual",
    "get_twi_change_lagged_qrtly",
    "get_twi_change_qrtly",
    "get_twi_monthly",
    "get_twi_qrtly",
    # Supply shocks
    "get_gscpi_covid_lagged_qrtly",
    "get_gscpi_monthly",
    "get_gscpi_qrtly",
    # Utilities
    "hma",
    "splice_series",
]
