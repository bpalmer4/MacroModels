"""ABS data loading utilities.

Provides functions for fetching and processing Australian Bureau of Statistics
(ABS) time series data using the readabs library.

Types:
    ReqsTuple: NamedTuple specifying series selection requirements.
    ReqsDict: Type alias for dict[str, ReqsTuple].
    DataSeries: Dataclass containing series data and metadata.

Functions:
    load_series: Load a single ABS series based on selection requirements.
    get_abs_data: Load multiple ABS series into a dictionary.
    get_gdp: Fetch GDP series (current prices or chain volume measures).
    get_population: Fetch population series by state.

Example:
    >>> from src.data.abs_loader import ReqsTuple, load_series, get_abs_data
    >>> cpi = ReqsTuple("6401.0", "640106", "All groups CPI", "S", "", False, False, "")
    >>> result = load_series(cpi)
    >>> print(result.data.tail())
    >>> print(f"Units: {result.units}")

"""

from functools import cache
from typing import Any, NamedTuple, cast

import readabs as ra
from pandas import DataFrame, PeriodIndex
from readabs import metacol as mc

from src.data.dataseries import DataSeries

# --- Type definitions ---

STYPE_CODES = {
    "O": "Original",
    "S": "Seasonally Adjusted",
    "SA": "Seasonally Adjusted",
    "T": "Trend",
}


class ReqsTuple(NamedTuple):
    """NamedTuple for specifying ABS series selection requirements.

    Attributes:
        cat: ABS catalogue number (e.g., "6401.0")
        table: ABS table id (e.g., "640106")
        did: Desired text in Data Item Description
        stype: Series Type code: "O", "S", "SA", or "T"
        unit: Unit of Measure (empty string for any)
        seek_yr_growth: Whether to seek yearly growth series from ABS
        calc_growth: Whether to calculate growth rates after loading
        zip_file: Path to local zip file ("" to fetch from ABS website)

    """

    cat: str
    table: str
    did: str
    stype: str
    unit: str
    seek_yr_growth: bool
    calc_growth: bool
    zip_file: str


type ReqsDict = dict[str, ReqsTuple]


# --- Private caching functions ---


@cache
def _get_zip_table(zip_file: str, table: str) -> tuple[DataFrame, DataFrame]:
    """Get a table from an ABS zip file of all tables (cached)."""
    dictionary, meta = ra.read_abs_cat(
        cat="", zip_file=zip_file, single_excel_only=table
    )
    data = dictionary[table]
    meta = meta[meta[mc.table] == table]
    return (data, meta)


@cache
def _get_table(cat: str, table: str) -> tuple[DataFrame, DataFrame]:
    """Get ABS data table and metadata for a given catalogue and table ID (cached)."""
    dictionary, meta = ra.read_abs_cat(cat, single_excel_only=table, verbose=False)
    data = dictionary[table]
    meta = meta[meta[mc.table] == table]
    return (data, meta)


# --- Public API ---


@cache
def load_series(input_tuple: ReqsTuple, verbose: bool = False) -> DataSeries:
    """Load an ABS data series and return as a DataSeries with metadata.

    Args:
        input_tuple: ReqsTuple with selection requirements
        verbose: Whether to print verbose output

    Returns:
        DataSeries containing the data and metadata

    Raises:
        ValueError: If neither cat nor zip_file is provided
        ValueError: If calc_growth is requested for unsupported periodicity

    """
    cat, table, did, stype, unit, seek_yr_growth, calc_growth, zip_file = input_tuple
    stype_full = stype if stype not in STYPE_CODES else STYPE_CODES[stype]

    if cat:
        data, meta = _get_table(cat, table)
    elif zip_file:
        data, meta = _get_zip_table(zip_file, table)
    else:
        raise ValueError("Either cat or zip_file must be provided.")

    selector: dict[str, str] = {
        did: mc.did,
        stype_full: mc.stype,
    }
    if unit:
        selector[unit] = mc.unit
    if seek_yr_growth:
        # ABS inconsistent capitalisation...
        selector["Percentage"] = mc.did
        selector["revious"] = mc.did
        selector["ear"] = mc.did

    _table, series_id, units = ra.find_abs_id(meta, selector, verbose=verbose)
    series = data[series_id]

    # Get full description from metadata
    series_meta = meta[meta[mc.id] == series_id]
    description = series_meta[mc.did].iloc[0] if len(series_meta) > 0 else did

    if calc_growth:
        periodicity = cast("PeriodIndex", series.index).freqstr[0]
        p_map = {"Q": 4, "M": 12}
        if periodicity not in p_map:
            raise ValueError(
                f"Cannot calculate growth for periodicity '{periodicity}'"
            )
        series = series.pct_change(periods=p_map[periodicity]) * 100.0
        units = "% change"

    return DataSeries(
        data=series,
        source="ABS",
        units=units,
        description=description,
        series_id=series_id,
        table=table,
        cat=cat or None,
        stype=stype_full,
    )


def get_abs_data(wanted: ReqsDict, verbose: bool = False) -> dict[str, DataSeries]:
    """Load all ABS data series specified in the requirements dictionary.

    Args:
        wanted: Dictionary mapping names to ReqsTuple requirements
        verbose: Whether to print verbose output

    Returns:
        Dictionary mapping names to DataSeries objects

    """
    box = {}
    for name, reqs in wanted.items():
        result = load_series(reqs, verbose=verbose)
        box[name] = result
    return box


# --- Convenience functions for common series ---


def get_population(
    state: str = "Australia",
    project: bool = True,
    **kwargs: Any,
) -> DataSeries:
    """Fetch ABS population Series for a given state.

    Args:
        state: State name (default "Australia" for national population)
        project: Whether to project population forward by 2 quarters
        **kwargs: Additional arguments passed to ra.find_abs_id

    Returns:
        DataSeries containing population data and metadata

    """
    cat = "3101.0"
    table = "310104"
    pop_data, pop_meta = ra.read_abs_cat(cat, single_excel_only=table, verbose=False)
    selector = {
        f";  {state} ;": mc.did,
        "Estimated Resident Population ;  Persons ;  ": mc.did,
    }
    _table, series_id, units = ra.find_abs_id(pop_meta, selector, **kwargs)
    pop = pop_data[table][series_id]

    if project:
        # Simple projection for short periods (6 months)
        rate = pop.iloc[-1] / pop.iloc[-2]
        base_period = pop.index[-1]
        for i in range(1, 3):
            pop[base_period + i] = pop[base_period + i - 1] * rate

    return DataSeries(
        data=pop,
        source="ABS",
        units=units,
        description=f"Estimated Resident Population - {state}",
        series_id=series_id,
        table=table,
        cat=cat,
        metadata={"projected": project, "state": state},
    )


def get_abs_catalogue_data(
    cat: str,
    **kwargs: Any,
) -> tuple[dict[str, DataFrame], DataFrame, str, str]:
    """Get ABS data for a specific catalogue number.

    This is a general-purpose loader that returns all tables in a catalogue.

    Args:
        cat: ABS catalogue number (e.g., "6401.0")
        **kwargs: Additional arguments to pass to read_abs_cat

    Returns:
        Tuple of (data dictionary, metadata DataFrame, source string, recent date string)

    """
    abs_dict, meta = ra.read_abs_cat(cat, **kwargs)
    source = f"ABS: {cat}"
    recent = "2020-12-01"
    return abs_dict, meta, source, recent


def collate_summary_data(
    to_get: dict[str, tuple[str, int]],
    abs_data: dict[str, DataFrame],
    md: DataFrame,
    verbose: bool = False,
) -> DataFrame:
    """Construct a summary DataFrame of key ABS data.

    Args:
        to_get: Dict of {label: (series_id, n_periods_growth)}
        abs_data: Dictionary of ABS data from readabs
        md: ABS metadata table from readabs
        verbose: Whether to print verbose output

    Returns:
        DataFrame with collated data, growth rates calculated where specified

    """
    data = DataFrame()
    for label, (code, period) in to_get.items():
        selected = md[md[mc.id] == code].iloc[0]
        table_desc = selected[mc.tdesc]
        table = selected[mc.table]
        did = selected[mc.did]
        stype = selected[mc.stype]
        if verbose:
            print(code, table, table_desc, did, stype)
        series = abs_data[table][code]
        if period:
            series = series.pct_change(periods=period, fill_method=None) * 100
        data[label] = series
    return data


# --- Plotting constants ---

ANNUAL_CPI_TARGET: dict[str, float | str | int] = {
    "y": 2.5,
    "color": "gray",
    "linestyle": "--",
    "linewidth": 0.75,
    "label": "2.5% annual inflation target",
    "zorder": -1,
}

ANNUAL_CPI_TARGET_RANGE: dict[str, float | str | int] = {
    "ymin": 2,
    "ymax": 3,
    "color": "#dddddd",
    "label": "2-3% annual inflation target range",
    "zorder": -1,
}

QUARTERLY_CPI_TARGET: dict[str, float | str | int] = {
    "y": (pow(1.025, 0.25) - 1) * 100,
    "linestyle": "dashed",
    "linewidth": 0.75,
    "color": "darkred",
    "label": "Quarterly growth consistent with 2.5% annual inflation",
}

QUARTERLY_CPI_RANGE: dict[str, float | str | int] = {
    "ymin": (pow(1.02, 0.25) - 1) * 100,
    "ymax": (pow(1.03, 0.25) - 1) * 100,
    "color": "#ffdddd",
    "label": "Quarterly growth consistent with 2-3% annual inflation target",
    "zorder": -1,
}

MONTHLY_CPI_TARGET: dict[str, float | str | int] = {
    "y": (pow(1.025, 1.0 / 12.0) - 1) * 100,
    "color": "darkred",
    "linewidth": 0.75,
    "linestyle": "--",
    "label": "Monthly growth consistent with a 2.5% annual inflation target",
    "zorder": -1,
}


# --- Testing ---

if __name__ == "__main__":
    print("Testing ABS loader...\n")

    # Test single series
    cpi_reqs = ReqsTuple(
        "6401.0", "640106", "All groups CPI, seasonally adjusted", "S", "", True, False, ""
    )
    cpi_result = load_series(cpi_reqs)
    print(f"CPI series: {cpi_result}")
    print(cpi_result.data.tail())

    # Test multiple series
    sought: ReqsDict = {
        "Monthly CPI (SA)": cpi_reqs,
        "Unemployment rate monthly (SA)": ReqsTuple(
            "6202.0", "6202001", "Unemployment rate ;  Persons ;", "S", "", False, False, ""
        ),
    }
    dataset = get_abs_data(sought)
    for name, result in dataset.items():
        print(f"\n{name}: {result}")
        print(result.data.tail())
