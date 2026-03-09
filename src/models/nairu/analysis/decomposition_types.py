"""Dataclasses for inflation decomposition results."""

from dataclasses import dataclass

import pandas as pd


@dataclass
class DecompositionBase:
    """Base class for inflation decomposition results.

    All components are in percentage points (quarterly).
    """

    observed: pd.Series
    anchor: pd.Series
    demand: pd.Series
    residual: pd.Series
    fitted: pd.Series
    index: pd.PeriodIndex


@dataclass
class InflationDecomposition(DecompositionBase):
    """Price inflation decomposition into demand and supply components."""

    supply_import: pd.Series
    supply_gscpi: pd.Series
    has_import_price: bool = True
    has_gscpi: bool = True

    @property
    def supply_total(self) -> pd.Series:
        """Total supply contribution (import + GSCPI)."""
        return self.supply_import + self.supply_gscpi

    def to_dataframe(self) -> pd.DataFrame:
        """Convert to DataFrame with all components."""
        return pd.DataFrame({
            "observed": self.observed, "anchor": self.anchor,
            "demand": self.demand, "supply_import": self.supply_import,
            "supply_gscpi": self.supply_gscpi, "supply_total": self.supply_total,
            "residual": self.residual, "fitted": self.fitted,
        }, index=self.index)


@dataclass
class WageInflationDecomposition(DecompositionBase):
    """Wage inflation (ULC growth) decomposition."""

    price_passthrough: pd.Series
    has_price_passthrough: bool = False
    has_expectations: bool = False

    def to_dataframe(self) -> pd.DataFrame:
        """Convert to DataFrame with all components."""
        return pd.DataFrame({
            "observed": self.observed, "anchor": self.anchor,
            "demand": self.demand, "price_passthrough": self.price_passthrough,
            "residual": self.residual, "fitted": self.fitted,
        }, index=self.index)


@dataclass
class HCOEInflationDecomposition(DecompositionBase):
    """Hourly COE growth decomposition."""

    price_passthrough: pd.Series
    productivity: pd.Series
    has_price_passthrough: bool = False
    has_expectations: bool = False

    def to_dataframe(self) -> pd.DataFrame:
        """Convert to DataFrame with all components."""
        return pd.DataFrame({
            "observed": self.observed, "anchor": self.anchor,
            "demand": self.demand, "price_passthrough": self.price_passthrough,
            "productivity": self.productivity,
            "residual": self.residual, "fitted": self.fitted,
        }, index=self.index)
