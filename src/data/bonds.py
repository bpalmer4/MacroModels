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


def get_indexed_yield_filled(*, monthly: bool = False) -> DataSeries:
    """Indexed 10y yield with missing periods filled via nominal − interpolated breakeven.

    The RBA F2 series for the 10y indexed bond yield has a 5-quarter gap
    (2013Q3–2014Q3) when the benchmark inflation-linked Treasury bond was
    being transitioned between maturities, and the standardised 10y measure
    had no clean underlying instrument. The gap drops 5 quarters from the
    HLW model's sample if used unfilled.

    Fill method:

    1. Compute breakeven inflation (nominal_10y − indexed_10y) wherever both
       series are available.
    2. Linearly interpolate breakeven across missing quarters.
    3. For each missing-indexed quarter where nominal_10y is observed,
       set ``indexed = nominal − breakeven_interpolated``.

    Why this works: breakeven inflation is anchored to inflation expectations
    (which are themselves anchored to the RBA target), so it moves slowly and
    interpolates well. The nominal yield, which IS observed throughout the
    gap, contributes the actual real-rate dynamics of the period (notably the
    2013 taper-tantrum spike). Subtracting an interpolated breakeven from the
    observed nominal cleanly recovers the real component without smoothing
    away the underlying real-rate movement.

    Args:
        monthly: If True, return monthly frequency. If False (default), quarterly.

    Returns:
        DataSeries with indexed bond yield (%), gaps filled.

    """
    nominal = get_nominal_10y(monthly=monthly).data
    indexed = get_indexed_yield(monthly=monthly).data

    # Align both series to a contiguous index covering both — so missing
    # quarters in either series surface as explicit NaN values rather than
    # silently dropping out of the index.
    common_idx = nominal.index.union(indexed.index).sort_values()
    nominal_aligned = nominal.reindex(common_idx)
    indexed_aligned = indexed.reindex(common_idx)

    breakeven = nominal_aligned - indexed_aligned  # NaN where either is missing
    breakeven_interp = breakeven.interpolate(method="linear")

    indexed_filled = indexed_aligned.copy()
    fill_mask = (
        indexed_aligned.isna()
        & nominal_aligned.notna()
        & breakeven_interp.notna()
    )
    indexed_filled.loc[fill_mask] = (
        nominal_aligned.loc[fill_mask] - breakeven_interp.loc[fill_mask]
    )

    freq_label = "monthly" if monthly else "quarterly"
    return DataSeries(
        data=indexed_filled,
        source="RBA F2 (gaps filled)",
        units="%",
        description=(
            f"Indexed Bond Yield ({freq_label}); gaps filled via "
            f"nominal − interpolated breakeven"
        ),
        table="F2",
        series_id="FCMYGBAGI_filled",
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
