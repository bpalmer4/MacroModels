"""Common ABS series specifications.

Pre-defined ReqsTuple specifications for frequently used ABS data series.
These can be used directly with load_series() or get_abs_data().

Example:
    >>> from src.data.series_specs import UNEMPLOYMENT_RATE, CPI_TRIMMED_MEAN
    >>> from src.data.abs_loader import get_abs_data
    >>> data = get_abs_data({
    ...     "Unemployment": UNEMPLOYMENT_RATE,
    ...     "Inflation": CPI_TRIMMED_MEAN,
    ... })

"""

from pathlib import Path

from src.data.abs_loader import ReqsTuple

# --- Local data files ---
# TODO: Remove once ABS publishes new quarterly CPI series (late Jan 2026)
_PROJECT_ROOT = Path(__file__).parent.parent.parent
CPI_ZIP_FILE = str(_PROJECT_ROOT / "input_data" / "Qrtly-CPI-Time-series-spreadsheets-all.zip")
HISTORICAL_RATE_FILE = str(_PROJECT_ROOT / "input_data" / "interbank_overnight_rate_historical.parquet")

# --- Labour Force (6202.0) ---

UNEMPLOYMENT_RATE = ReqsTuple(
    cat="6202.0",
    table="6202001",
    did="Unemployment rate ;  Persons ;",
    stype="SA",
    unit="",
    seek_yr_growth=False,
    calc_growth=False,
    zip_file="",
)

PARTICIPATION_RATE = ReqsTuple(
    cat="6202.0",
    table="6202001",
    did="Participation rate ;  Persons ;",
    stype="SA",
    unit="",
    seek_yr_growth=False,
    calc_growth=False,
    zip_file="",
)

EMPLOYMENT_PERSONS = ReqsTuple(
    cat="6202.0",
    table="6202001",
    did="Employed total ;  Persons ;",
    stype="SA",
    unit="",
    seek_yr_growth=False,
    calc_growth=False,
    zip_file="",
)

HOURS_WORKED = ReqsTuple(
    cat="6202.0",
    table="6202019",
    did="Monthly hours worked in all jobs ;  Persons ;",
    stype="SA",
    unit="",
    seek_yr_growth=False,
    calc_growth=False,
    zip_file="",
)

# --- Consumer Price Index (6401.0) ---
# Using local zip file until ABS publishes new quarterly series (late Jan 2026)

CPI_ALL_GROUPS = ReqsTuple(
    cat="",
    table="640101",
    did="All groups CPI ;  Australia ;",
    stype="O",
    unit="Index Numbers",
    seek_yr_growth=False,
    calc_growth=False,
    zip_file=CPI_ZIP_FILE,
)

CPI_HEADLINE_ANNUAL = ReqsTuple(
    cat="",
    table="640101",
    did="All groups CPI ;  Australia ;",
    stype="O",
    unit="",
    seek_yr_growth=True,
    calc_growth=False,
    zip_file=CPI_ZIP_FILE,
)

# Quarterly trimmed mean - percentage change from previous period
CPI_TRIMMED_MEAN_QUARTERLY = ReqsTuple(
    cat="",
    table="640106",
    did="Percentage Change from Previous Period ;  Trimmed Mean ;  Australia ;",
    stype="S",
    unit="",
    seek_yr_growth=False,
    calc_growth=False,
    zip_file=CPI_ZIP_FILE,
)

# Annual trimmed mean - percentage change from corresponding quarter previous year
CPI_TRIMMED_MEAN_ANNUAL = ReqsTuple(
    cat="",
    table="640106",
    did="Percentage Change from Corresponding Quarter of Previous Year ;  Trimmed Mean ;  Australia ;",
    stype="S",
    unit="",
    seek_yr_growth=False,
    calc_growth=False,
    zip_file=CPI_ZIP_FILE,
)

# Legacy alias (returns annual via seek_yr_growth - deprecated)
CPI_TRIMMED_MEAN = ReqsTuple(
    cat="",
    table="640106",
    did="Trimmed Mean ;  Australia ;",
    stype="SA",
    unit="",
    seek_yr_growth=True,
    calc_growth=False,
    zip_file=CPI_ZIP_FILE,
)

CPI_WEIGHTED_MEDIAN = ReqsTuple(
    cat="",
    table="640106",
    did="Weighted Median ;  Australia ;",
    stype="SA",
    unit="",
    seek_yr_growth=True,
    calc_growth=False,
    zip_file=CPI_ZIP_FILE,
)

# --- National Accounts (5206.0) ---

GDP_CVM = ReqsTuple(
    cat="5206.0",
    table="5206001_Key_Aggregates",
    did="Gross domestic product: Chain volume measures ;",
    stype="SA",
    unit="",
    seek_yr_growth=False,
    calc_growth=False,
    zip_file="",
)

HOURS_WORKED_INDEX = ReqsTuple(
    cat="5206.0",
    table="5206001_Key_Aggregates",
    did="Hours worked: Index ;",
    stype="SA",
    unit="",
    seek_yr_growth=False,
    calc_growth=False,
    zip_file="",
)

GDP_CURRENT_PRICES = ReqsTuple(
    cat="5206.0",
    table="5206001_Key_Aggregates",
    did="Gross domestic product: Current prices ;",
    stype="SA",
    unit="",
    seek_yr_growth=False,
    calc_growth=False,
    zip_file="",
)

COMPENSATION_OF_EMPLOYEES = ReqsTuple(
    cat="5206.0",
    table="5206020_Household_Income",
    did="Compensation of employees ;",
    stype="SA",
    unit="",
    seek_yr_growth=False,
    calc_growth=False,
    zip_file="",
)

# Gross Operating Surplus (for capital share calculation)
# From Income from GDP table - total corporations
GROSS_OPERATING_SURPLUS = ReqsTuple(
    cat="5206.0",
    table="5206007_Income_From_GDP",
    did="Total corporations ;  Gross operating surplus ;",
    stype="SA",
    unit="",
    seek_yr_growth=False,
    calc_growth=False,
    zip_file="",
)

# Domestic Final Demand implicit price deflator (quarterly, seasonally adjusted)
DFD_DEFLATOR = ReqsTuple(
    cat="5206.0",
    table="5206005_Expenditure_Implicit_Price_Deflators",
    did="Domestic final demand ;",
    stype="SA",
    unit="Index Numbers",
    seek_yr_growth=False,
    calc_growth=False,
    zip_file="",
)

# --- Modellers Database (1364.0.15.003) ---
# Note: These series cover total population including defence/non-civilian employment
# Unemployment rate must be calculated: (labour_force - employed) / labour_force * 100

CAPITAL_STOCK = ReqsTuple(
    cat="1364.0.15.003",
    table="1364015003",
    did="Non-financial and financial corporations ; Net capital stock (Chain volume measures) ;",
    stype="SA",
    unit="",
    seek_yr_growth=False,
    calc_growth=False,
    zip_file="",
)

LABOUR_FORCE_TOTAL = ReqsTuple(
    cat="1364.0.15.003",
    table="1364015003",
    did="Total labour force ;",
    stype="S",
    unit="",
    seek_yr_growth=False,
    calc_growth=False,
    zip_file="",
)

EMPLOYED_TOTAL = ReqsTuple(
    cat="1364.0.15.003",
    table="1364015003",
    did="Total employed ;",
    stype="S",
    unit="",
    seek_yr_growth=False,
    calc_growth=False,
    zip_file="",
)

UNEMPLOYED_TOTAL = ReqsTuple(
    cat="1364.0.15.003",
    table="1364015003",
    did="Total unemployed ;",
    stype="S",
    unit="",
    seek_yr_growth=False,
    calc_growth=False,
    zip_file="",
)

# --- Trade (6457.0) ---

IMPORT_PRICE_INDEX = ReqsTuple(
    cat="6457.0",
    table="64570016",
    did="All groups, Imports ;",
    stype="O",
    unit="Index Numbers",
    seek_yr_growth=False,
    calc_growth=False,
    zip_file="",
)

# --- Productivity (5204.0) ---

MFP_HOURS_WORKED = ReqsTuple(
    cat="5204.0",
    table="5204013_Productivity",
    did="Multifactor productivity - Hours worked: Percentage changes ;",
    stype="O",
    unit="",
    seek_yr_growth=False,  # Already percentage changes
    calc_growth=False,
    zip_file="",
)

# --- Average Weekly Earnings (6302.0) ---

AWE_FULL_TIME_ADULTS = ReqsTuple(
    cat="6302.0",
    table="6302001",
    did="Earnings; Persons; Full Time; Adult; Ordinary time earnings ;",
    stype="Trend",
    unit="$",
    seek_yr_growth=False,
    calc_growth=False,
    zip_file="",
)

# --- Wages (6345.0) ---

WPI_TOTAL = ReqsTuple(
    cat="6345.0",
    table="634501",
    did="Total hourly rates of pay excluding bonuses ;  Australia ;  All industries ;",
    stype="O",
    unit="Index Numbers",
    seek_yr_growth=False,
    calc_growth=False,
    zip_file="",
)

WPI_TOTAL_GROWTH = ReqsTuple(
    cat="6345.0",
    table="634501",
    did="Total hourly rates of pay excluding bonuses ;  Australia ;  All industries ;",
    stype="O",
    unit="",
    seek_yr_growth=True,
    calc_growth=False,
    zip_file="",
)


# --- Collections for common use cases ---

NAIRU_MODEL_SERIES = {
    "Unemployment Rate": UNEMPLOYMENT_RATE,
    "CPI Trimmed Mean": CPI_TRIMMED_MEAN,
}

OUTPUT_GAP_MODEL_SERIES = {
    "GDP (CVM)": GDP_CVM,
    "Capital Stock": CAPITAL_STOCK,
    "Hours Worked": HOURS_WORKED,
}


# --- Testing ---

if __name__ == "__main__":
    from src.data.abs_loader import load_series

    print("Testing series specifications...\n")

    print("Unemployment Rate:")
    ur = load_series(UNEMPLOYMENT_RATE)
    print(f"{ur}")
    print(ur.data.tail())

    print("\nGDP (CVM):")
    gdp = load_series(GDP_CVM)
    print(f"{gdp}")
    print(gdp.data.tail())
