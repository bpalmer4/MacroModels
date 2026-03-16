"""Government bond yield data.

Provides quarterly bond yields and derived breakeven inflation:
- Nominal 10-year government bond yields (1969+)
- Indexed bond yields (1986+)
- Breakeven inflation = nominal - indexed (1986+)
"""


from src.data.dataseries import DataSeries
from src.data.rba_loader import get_bond_yield_10y, get_indexed_bond_yield

# --- Public API ---


def get_nominal_10y(*, monthly: bool = False) -> DataSeries:
    """Get 10-year nominal government bond yield.

    Args:
        monthly: If True, return monthly frequency. If False (default), quarterly.

    Returns:
        DataSeries with bond yield (%)

    """
    raw = get_bond_yield_10y()
    series = raw.data.copy()
    freq = "M" if monthly else "Q"
    freq_label = "monthly" if monthly else "quarterly"
    series.index = series.index.to_period(freq)
    series = series.groupby(series.index).last()

    return DataSeries(
        data=series,
        source="RBA",
        units="%",
        description=f"10-year Government Bond Yield ({freq_label})",
        table="F2",
        series_id="FCMYGBAG10",
    )


def get_indexed_yield(*, monthly: bool = False) -> DataSeries:
    """Get indexed (inflation-linked) bond yield.

    Args:
        monthly: If True, return monthly frequency. If False (default), quarterly.

    Returns:
        DataSeries with indexed bond yield (%)

    """
    raw = get_indexed_bond_yield()
    series = raw.data.copy()
    freq = "M" if monthly else "Q"
    freq_label = "monthly" if monthly else "quarterly"
    series.index = series.index.to_period(freq)
    series = series.groupby(series.index).last()

    return DataSeries(
        data=series,
        source="RBA",
        units="%",
        description=f"Indexed Bond Yield ({freq_label})",
        table="F2",
        series_id="FCMYGBAGI",
    )


def get_breakeven_inflation(*, monthly: bool = False) -> DataSeries:
    """Get breakeven inflation (nominal 10y minus indexed).

    This is a market-implied measure of inflation expectations,
    though it includes risk and liquidity premia.

    Args:
        monthly: If True, return monthly frequency. If False (default), quarterly.

    Returns:
        DataSeries with breakeven inflation (%)

    """
    nominal = get_nominal_10y(monthly=monthly).data
    indexed = get_indexed_yield(monthly=monthly).data

    # Align and compute breakeven
    common_idx = nominal.index.intersection(indexed.index)
    breakeven = nominal.loc[common_idx] - indexed.loc[common_idx]
    breakeven = breakeven.dropna()

    freq_label = "monthly" if monthly else "quarterly"
    return DataSeries(
        data=breakeven,
        source="RBA",
        units="%",
        description=f"Breakeven Inflation (10y nominal - indexed, {freq_label})",
        table="F2",
        series_id="derived",
    )


# --- Testing ---

if __name__ == "__main__":
    print("Loading F2 bond yields...")

    nominal = get_nominal_10y()
    print(f"\nNominal 10y: {nominal.data.index[0]} to {nominal.data.index[-1]}")
    print(f"  Latest: {nominal.data.iloc[-1]:.2f}%")

    indexed = get_indexed_yield()
    print(f"\nIndexed: {indexed.data.index[0]} to {indexed.data.index[-1]}")
    print(f"  Latest: {indexed.data.iloc[-1]:.2f}%")

    breakeven = get_breakeven_inflation()
    print(f"\nBreakeven: {breakeven.data.index[0]} to {breakeven.data.index[-1]}")
    print(f"  Latest: {breakeven.data.iloc[-1]:.2f}%")
