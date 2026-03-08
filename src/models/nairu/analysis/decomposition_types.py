"""Dataclasses for inflation decomposition results."""

from dataclasses import dataclass

import pandas as pd


@dataclass
class InflationDecomposition:
    """Price inflation decomposition into demand and supply components.

    All components are in percentage points (quarterly).
    """

    observed: pd.Series
    anchor: pd.Series
    demand: pd.Series
    supply_import: pd.Series
    supply_gscpi: pd.Series
    residual: pd.Series
    fitted: pd.Series
    index: pd.PeriodIndex
    has_import_price: bool = True
    has_gscpi: bool = True

    @property
    def supply_total(self) -> pd.Series:
        return self.supply_import + self.supply_gscpi

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame({
            "observed": self.observed, "anchor": self.anchor,
            "demand": self.demand, "supply_import": self.supply_import,
            "supply_gscpi": self.supply_gscpi, "supply_total": self.supply_total,
            "residual": self.residual, "fitted": self.fitted,
        }, index=self.index)


@dataclass
class WageInflationDecomposition:
    """Wage inflation (ULC growth) decomposition."""

    observed: pd.Series
    anchor: pd.Series
    demand: pd.Series
    price_passthrough: pd.Series
    residual: pd.Series
    fitted: pd.Series
    index: pd.PeriodIndex
    has_price_passthrough: bool = False
    has_expectations: bool = False

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame({
            "observed": self.observed, "anchor": self.anchor,
            "demand": self.demand, "price_passthrough": self.price_passthrough,
            "residual": self.residual, "fitted": self.fitted,
        }, index=self.index)


@dataclass
class HCOEInflationDecomposition:
    """Hourly COE growth decomposition."""

    observed: pd.Series
    anchor: pd.Series
    demand: pd.Series
    price_passthrough: pd.Series
    productivity: pd.Series
    residual: pd.Series
    fitted: pd.Series
    index: pd.PeriodIndex
    has_price_passthrough: bool = False
    has_expectations: bool = False

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame({
            "observed": self.observed, "anchor": self.anchor,
            "demand": self.demand, "price_passthrough": self.price_passthrough,
            "productivity": self.productivity,
            "residual": self.residual, "fitted": self.fitted,
        }, index=self.index)
