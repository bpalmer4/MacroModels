"""Data types for time series with metadata."""

from dataclasses import dataclass, field
from typing import Any

import pandas as pd


@dataclass
class DataSeries:
    """A time series with metadata from any source.

    Attributes:
        data: The time series data as a pandas Series
        source: Data source identifier ("ABS", "RBA", "World Bank", etc.)
        units: Unit of measurement ("$ Millions", "%", "Index", etc.)
        description: Human-readable description of the series
        series_id: Original source series identifier
        table: Table identifier (ABS table ID or RBA table code)
        cat: ABS catalogue number (ABS-specific)
        stype: Series type - Original, Seasonally Adjusted, Trend (ABS-specific)
        metadata: Extensible dict for additional source-specific metadata

    Example:
        >>> result = load_series(UNEMPLOYMENT_RATE)
        >>> print(result.data.tail())
        >>> print(f"Units: {result.units}")
        >>> print(f"Source: {result.source} {result.cat}")

    """

    # Required
    data: pd.Series
    source: str

    # Common optional
    units: str | None = None
    description: str | None = None
    series_id: str | None = None
    table: str | None = None

    # ABS-specific
    cat: str | None = None
    stype: str | None = None

    # Extensible catch-all
    metadata: dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        """Concise representation showing key info."""
        desc = self.description or self.series_id or "unnamed"
        if len(desc) > 50:
            desc = desc[:47] + "..."
        return f"DataSeries({desc!r}, source={self.source!r}, n={len(self.data)})"

    @property
    def name(self) -> str:
        """Return description or series_id as a name."""
        return self.description or self.series_id or ""
