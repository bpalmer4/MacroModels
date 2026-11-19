"""Total population growth, forward of the official estimate.

Estimated Resident Population (ERP, 3101.0) is published about six months after its
reference quarter. This builds a forward proxy for its annual change from the monthly
civilian population aged 15+ (6202.0), which is current to the latest Labour Force month:

    ERP change  = net overseas migration (NOM) + natural increase
    NOM proxy   = migration 15+ + child migration
    migration 15+   = civ15 annual growth - 15-year-olds ageing in + deaths
    child migration = observed by cohort survival in June ERP by single year of age,
                      extended past the last June at its median ratio to civ15 growth

Deaths are added back to civ15 growth because in a headcount a death looks like an
emigration. Natural increase (births less deaths) is published with NOM, so past the
last official quarter it is held at its latest four-quarter sum.

All counts are quarterly through-the-year changes ('000 a year), Original. The proxy is an
indication, not a substitute: first-published civ15 months are usually revised, and the
civ15 level steps at each ERP benchmark.
"""

from dataclasses import dataclass
from functools import cache

import pandas as pd
import readabs as ra
from pandas import Period, PeriodIndex
from readabs import metacol as mc

from src.data.abs_loader import load_series
from src.data.series_specs import CIVILIAN_POP_15

ERP_CATALOGUE = "3101.0"
ERP_LEVEL_TABLE = "310104"  # ERP by state and sex, quarterly
ERP_PERSONS = "Estimated Resident Population ;  Persons ;  "
ERP_NATIONAL = ";  Australia ;"  # the bare name would also match every state's "Australian"
ERP_LEVEL_UNIT = "Persons"
COMPONENTS_TABLE = "310101"  # births, deaths, NOM, ERP change; quarterly
COMPONENTS_STYPE = "Original"  # 310101 is published Original only
AGE_TABLE = "3101059"  # ERP by single year of age, Australia, at June
NOM_DID = "Net Overseas Migration ;  Australia ;"
DEATHS_DID = "Deaths ;  Australia ;"
NATURAL_INCREASE_DID = "Natural Increase ;  Australia ;"
ERP_CHANGE_DID = "ERP Change Over Previous Year ;  Australia ;"
ERP_CHANGE_UNIT = "000"  # the count, not the % change published beside it
OPEN_AGE, OPEN_AGE_TEXT = 100, "100 and over"
COHORT_AGE = 15  # civ15's youngest age: the cohort ageing in each year
JUNE_QUARTER = 2
QUARTERS_PER_YEAR = 4
MONTHS_PER_QUARTER = 3
THOUSAND = 1_000.0


@dataclass(frozen=True)
class PopulationGrowth:
    """Forward proxy for annual ERP growth, with the official series beside it."""

    proxy: pd.Series  # NOM proxy + natural increase ('000 a year)
    official: pd.Series  # published ERP change over the previous year ('000 a year)
    last_official: Period  # last quarter of official NOM; natural increase is held after it
    last_complete: Period  # last quarter with all three civ15 months; later ones are extrapolated


# --- 3101.0 components of population change
@cache
def _components() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fetch 3101.0 table 310101 and its metadata (cached; not for mutation)."""
    data, meta = ra.read_abs_cat(ERP_CATALOGUE, single_excel_only=COMPONENTS_TABLE, verbose=False)
    return data[COMPONENTS_TABLE], meta


def _component(did: str, unit: str = "") -> pd.Series:
    """One 310101 series, by description (and unit, where the description is ambiguous)."""
    data, meta = _components()
    selector = {COMPONENTS_TABLE: mc.table, COMPONENTS_STYPE: mc.stype, did: mc.did}
    if unit:
        selector[unit] = mc.unit
    _, series_id, _units = ra.find_abs_id(meta, selector, verbose=False)
    return data[series_id]


@cache
def _nom() -> pd.Series:
    """Official NOM as a four-quarter rolling sum, '000 a year (cached; not for mutation)."""
    nom = _component(NOM_DID).rolling(QUARTERS_PER_YEAR).sum().dropna()
    if nom.empty:
        msg = f"No Net Overseas Migration data in {COMPONENTS_TABLE}"
        raise ValueError(msg)
    return nom


# --- 3101.0 ERP by single year of age, at June
@cache
def _erp_aged(min_age: int) -> pd.Series:
    """ERP at June summed over ages min_age and over, '000 (cached; not for mutation)."""
    data, meta = ra.read_abs_cat(ERP_CATALOGUE, single_excel_only=AGE_TABLE, verbose=False)
    persons = meta[meta[mc.did].str.contains("Persons")]
    ids = []
    for _, row in persons.iterrows():
        age_text = str(row[mc.did]).split(";")[2].strip()
        age = OPEN_AGE if age_text == OPEN_AGE_TEXT else int(age_text)
        if age >= min_age:
            ids.append(row[mc.id])
    return data[AGE_TABLE][ids].sum(axis=1).dropna() / THOUSAND


def interp_june(annual: pd.Series, index: PeriodIndex) -> pd.Series:
    """Spread an annual June-referenced series onto a quarterly or monthly index.

    Each value is anchored at the June quarter (or month), interpolated linearly between
    anchors, and held flat after the latest.
    """
    freq = index.freqstr
    anchors: dict[Period, float] = {}
    for period, value in annual.items():
        if not isinstance(period, Period):
            msg = "Expected a PeriodIndex on the annual June-referenced series"
            raise TypeError(msg)
        anchor = (
            Period(year=period.year, quarter=JUNE_QUARTER, freq=freq)
            if freq.startswith("Q")
            else Period(year=period.year, month=6, freq=freq)
        )
        anchors[anchor] = float(value)
    return pd.Series(anchors, dtype=float).reindex(index).interpolate(limit_area="inside").ffill()


# --- 6202.0 civilian population aged 15+, monthly
def complete_trailing_quarter(level: pd.Series) -> pd.Series:
    """Extend a monthly level so its final quarter is complete.

    A quarterly mean of a level estimates the mid-quarter level, so a part-filled final
    quarter would understate the last quarterly increment. The missing months continue the
    latest month-on-month rate, which is what the ABS's linear interpolation within a
    benchmark segment publishes next; they exist only to centre the quarterly mean.
    Works on a copy.
    """
    last = level.index[-1]
    if not isinstance(last, Period):
        msg = "Expected a monthly PeriodIndex on the population level"
        raise TypeError(msg)
    missing = (MONTHS_PER_QUARTER - last.month % MONTHS_PER_QUARTER) % MONTHS_PER_QUARTER
    if not missing:
        return level
    level = level.copy()
    rate = level.iloc[-1] / level.iloc[-2]
    extension = {}
    value = float(level.iloc[-1])
    for i in range(1, missing + 1):
        value *= rate
        extension[last + i] = value
    return pd.concat([level, pd.Series(extension, dtype=float)])


@cache
def _civ15_split() -> tuple[pd.Series, pd.Series, Period]:
    """Split civ15 year-on-year growth into 15+ natural increase and 15+ migration (cached).

    Returns (civ15 growth, migration 15+, last complete civ15 quarter), the growth series as
    through-the-year counts ('000 a year). Quarters after the last complete one rest on
    extrapolated months.
    """
    civ15 = load_series(CIVILIAN_POP_15).data.dropna()
    months_in_quarter = civ15.resample("Q").count()
    last_complete = months_in_quarter[months_in_quarter.eq(MONTHS_PER_QUARTER)].index[-1]
    filled = complete_trailing_quarter(civ15)
    filled_counts = filled.resample("Q").count()
    level = filled.resample("Q").mean().where(filled_counts.eq(MONTHS_PER_QUARTER))
    growth = level.diff(QUARTERS_PER_YEAR)
    growth_index = growth.index
    if not isinstance(growth_index, PeriodIndex):
        msg = "Expected a quarterly PeriodIndex for civ15 growth"
        raise TypeError(msg)
    if not isinstance(last_complete, Period):
        msg = "Expected a quarterly Period for the last complete civ15 quarter"
        raise TypeError(msg)

    ageing_in = interp_june(_erp_aged(COHORT_AGE) - _erp_aged(COHORT_AGE + 1), growth_index)
    deaths = _component(DEATHS_DID).rolling(QUARTERS_PER_YEAR).sum()
    natural_increase = (ageing_in - deaths).ffill()  # ageing in less deaths
    return growth, growth - natural_increase, last_complete


@cache
def _child_migration() -> pd.Series:
    """Net child migration, quarterly through the year ('000 a year) (cached; not for mutation).

    Observed by cohort survival while ERP by age exists (those aged 1-15 each June less those
    aged 0-14 a June earlier: children 14 and under, net of their few deaths), spread onto
    civ15's quarters between Junes, then extended by its median ratio to civ15 growth.
    """
    growth, _migration_15_plus, _last_complete = _civ15_split()
    growth_index = growth.index
    if not isinstance(growth_index, PeriodIndex):
        msg = "Expected a quarterly PeriodIndex for civ15 growth"
        raise TypeError(msg)
    aged_0_to_14 = _erp_aged(0) - _erp_aged(COHORT_AGE)
    aged_1_to_15 = _erp_aged(1) - _erp_aged(COHORT_AGE + 1)
    yearly = (aged_1_to_15 - aged_0_to_14.shift(1)).dropna()
    observed = interp_june(yearly, growth_index)

    last_june = yearly.index[-1]
    if not isinstance(last_june, Period):
        msg = "Expected a PeriodIndex on ERP by age"
        raise TypeError(msg)
    last_age = Period(year=last_june.year, quarter=JUNE_QUARTER, freq="Q-DEC")
    ratio = (observed.loc[:last_age] / growth.loc[:last_age]).dropna().median()
    child_migration = observed.copy()
    later = child_migration.index > last_age
    child_migration[later] = ratio * growth[later]
    return child_migration


@cache
def _population_growth() -> PopulationGrowth:
    """Build the forward proxy for annual ERP growth (cached; not for mutation)."""
    _growth, migration_15_plus, last_complete = _civ15_split()
    nom = _nom()
    nom_proxy = (migration_15_plus + _child_migration()).dropna()
    if nom_proxy.empty:
        msg = "NOM forward proxy is empty: check the 6202 and 3101 inputs"
        raise ValueError(msg)

    natural_increase = _component(NATURAL_INCREASE_DID).rolling(QUARTERS_PER_YEAR).sum()
    natural_increase = natural_increase.dropna().reindex(nom_proxy.index).ffill()
    proxy = (nom_proxy + natural_increase).dropna()
    if proxy.empty:
        msg = "Population growth proxy is empty: check the NOM proxy and natural increase"
        raise ValueError(msg)

    last_official = nom.index[-1]
    if not isinstance(last_official, Period):
        msg = "Expected a quarterly PeriodIndex on official NOM"
        raise TypeError(msg)
    official = _component(ERP_CHANGE_DID, ERP_CHANGE_UNIT).dropna()
    return PopulationGrowth(proxy, official, last_official, last_complete)


@cache
def _erp_level() -> pd.Series:
    """National ERP, persons, '000 (cached; not for mutation)."""
    data, meta = ra.read_abs_cat(ERP_CATALOGUE, single_excel_only=ERP_LEVEL_TABLE, verbose=False)
    selector = {ERP_NATIONAL: mc.did, ERP_PERSONS: mc.did}
    _, series_id, units = ra.find_abs_id(meta, selector, verbose=False)
    if units != ERP_LEVEL_UNIT:  # the scaling below assumes persons
        msg = f"ERP level: expected units {ERP_LEVEL_UNIT!r}, got {units!r}"
        raise ValueError(msg)
    series = data[ERP_LEVEL_TABLE][series_id].dropna() / THOUSAND
    if series.empty:
        msg = f"ABS {ERP_CATALOGUE} returned no national ERP"
        raise ValueError(msg)
    return series


def get_population_growth_annual() -> pd.Series:
    """Annual ERP growth (%): the forward proxy over ERP four quarters earlier.

    Runs as far as the proxy does wherever ERP four quarters back is published, which with
    ERP's six-month lag covers about two quarters past the last complete civ15 quarter.
    """
    proxy = _population_growth().proxy
    erp = _erp_level()
    erp_year_earlier = pd.Series(erp.to_numpy(), index=erp.index + QUARTERS_PER_YEAR)
    return (proxy / erp_year_earlier.reindex(proxy.index) * 100).dropna()


def get_population_growth_qrtly() -> pd.Series:
    """Quarterly ERP growth (%) that compounds to the annual rate over four quarters.

    (1 + annual)^(1/4) - 1: the geometric quarterly rate, not a quarter of the annual one.
    It is the annual rate's average quarter, so any quarter-to-quarter variation in
    population growth within the year is not in it.
    """
    annual = get_population_growth_annual()
    return ((1 + annual / 100) ** (1 / QUARTERS_PER_YEAR) - 1) * 100


def get_population_growth_proxy() -> PopulationGrowth:
    """Return the forward proxy for annual ERP growth and the official series ('000 a year).

    The proxy runs past the official series to the latest civ15 quarter. Quarters after
    ``last_complete`` rest on extrapolated months; natural increase after
    ``last_official`` is held at its latest value.
    """
    g = _population_growth()
    return PopulationGrowth(g.proxy.copy(), g.official.copy(), g.last_official, g.last_complete)
