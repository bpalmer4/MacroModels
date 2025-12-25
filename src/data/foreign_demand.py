"""Foreign demand data loading.

Provides proxies for trading partner demand/output gap.

Note: Trading partner GDP growth data can be sourced from:
- RBA Statistical Tables (G10)
- OECD (Main Economic Indicators)
- IMF (World Economic Outlook)
- Or constructed from individual country GDP weighted by trade shares
"""

from pathlib import Path

import pandas as pd

from src.data.dataseries import DataSeries


# --- Data file paths ---
_PROJECT_ROOT = Path(__file__).parent.parent.parent
FOREIGN_DEMAND_FILE = _PROJECT_ROOT / "data" / "foreign_demand.parquet"


def get_major_trading_partner_growth_qrtly() -> DataSeries:
    """Get major trading partner GDP growth (trade-weighted).

    Loads from local parquet file. This should contain export-weighted
    GDP growth for Australia's major trading partners (China, Japan,
    Korea, US, EU, etc.)

    The data file should have:
    - PeriodIndex (quarterly)
    - Column with trading partner GDP growth (% yoy)

    Returns:
        DataSeries with trading partner GDP growth (% yoy)

    """
    if FOREIGN_DEMAND_FILE.exists():
        data = pd.read_parquet(FOREIGN_DEMAND_FILE)
        if isinstance(data, pd.DataFrame):
            data = data.iloc[:, 0]
        # Ensure quarterly PeriodIndex
        if not isinstance(data.index, pd.PeriodIndex):
            data.index = pd.PeriodIndex(data.index, freq="Q")

        return DataSeries(
            data=data,
            source="External",
            units="% yoy",
            description="Major trading partner GDP growth (export-weighted)",
        )

    # Return empty series if file not available
    return DataSeries(
        data=pd.Series(dtype=float),
        source="External",
        units="% yoy",
        description="Major trading partner GDP growth (data file not found)",
    )


def get_world_gdp_growth_qrtly() -> DataSeries:
    """Get world GDP growth (quarterly).

    Alternative to trading partner growth - uses global GDP.

    Returns:
        DataSeries with world GDP growth (% yoy)

    """
    # For now, return same as trading partner growth
    # Could be extended with separate world GDP file
    return get_major_trading_partner_growth_qrtly()
