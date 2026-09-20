"""Scenario definitions and the forecast result container.

Separate from `forecast` and `forecast_plots` because both need these names:
the forecaster produces them and the charts consume them. Holding them here
lets each import the other's module at the top of the file.
"""

from dataclasses import dataclass

import pandas as pd

from src.utilities.rate_conversion import annualize

# Scenario ordering and colors for plotting
SCENARIO_ORDER = ["+200bp", "+100bp", "+50bp", "+25bp", "hold",
                  "-25bp", "-50bp", "-100bp", "-200bp"]
SCENARIO_COLORS = {
    "+200bp": "darkred", "+100bp": "red", "+50bp": "orangered",
    "+25bp": "orange", "hold": "black", "-25bp": "deepskyblue",
    "-50bp": "steelblue", "-100bp": "blue", "-200bp": "darkblue",
}


@dataclass
class ForecastResults:
    """Container for forecast results (one scenario)."""

    scenario_name: str
    cash_rate: float
    forecast_index: pd.PeriodIndex
    obs_index: pd.PeriodIndex

    # Posterior samples (rows=periods, cols=samples)
    nairu_forecast: pd.DataFrame
    potential_forecast: pd.DataFrame
    output_gap_forecast: pd.DataFrame
    unemployment_forecast: pd.DataFrame
    inflation_forecast: pd.DataFrame  # quarterly

    # Context
    nairu_final: float
    potential_final: float
    log_gdp_final: float
    unemployment_final: float
    potential_growth: float
    coefficients: dict

    def _quantiles(self, df: pd.DataFrame, prob: float = 0.90) -> pd.DataFrame:
        lower = (1 - prob) / 2
        upper = 1 - lower
        return pd.DataFrame(
            {"lower": df.quantile(lower, axis=1),
             "median": df.median(axis=1),
             "upper": df.quantile(upper, axis=1)},
            index=df.index,
        )

    def output_gap_hdi(self, prob: float = 0.90) -> pd.DataFrame:
        """Output gap HDI bands."""
        return self._quantiles(self.output_gap_forecast, prob)

    def unemployment_hdi(self, prob: float = 0.90) -> pd.DataFrame:
        """Unemployment rate HDI bands."""
        return self._quantiles(self.unemployment_forecast, prob)

    def unemployment_gap_hdi(self, prob: float = 0.90) -> pd.DataFrame:
        """Unemployment gap (U - NAIRU) HDI bands."""
        return self._quantiles(self.unemployment_forecast - self.nairu_forecast, prob)

    def inflation_hdi(self, prob: float = 0.90) -> pd.DataFrame:
        """Quarterly inflation HDI bands."""
        return self._quantiles(self.inflation_forecast, prob)

    def inflation_annual_hdi(self, prob: float = 0.90) -> pd.DataFrame:
        """Annualised inflation HDI bands."""
        return self._quantiles(annualize(self.inflation_forecast), prob)

    def output_samples(self) -> pd.DataFrame:
        """GDP (log) = potential + output_gap."""
        return self.potential_forecast + self.output_gap_forecast

    def output_hdi(self, prob: float = 0.90) -> pd.DataFrame:
        """GDP (log level) HDI bands."""
        return self._quantiles(self.output_samples(), prob)

    def gdp_growth_forecast(self) -> pd.DataFrame:
        """Quarterly GDP growth = potential_growth + delta(output_gap)."""
        og_change = self.output_gap_forecast.diff()
        og_change.iloc[0] = (
            self.output_gap_forecast.iloc[0]
            - (self.log_gdp_final - self.potential_final)
        )
        return self.potential_growth + og_change

    def gdp_growth_annual_hdi(self, prob: float = 0.90) -> pd.DataFrame:
        """Annualised GDP growth HDI bands."""
        return self._quantiles(annualize(self.gdp_growth_forecast()), prob)

    def summary(self) -> pd.DataFrame:
        """Tabular summary of median forecasts."""
        return pd.DataFrame(
            {
                "Output Gap": self.output_gap_forecast.median(axis=1),
                "U": self.unemployment_forecast.median(axis=1),
                "NAIRU": self.nairu_forecast.median(axis=1),
                "U Gap": (self.unemployment_forecast - self.nairu_forecast).median(axis=1),
                "pi (ann)": annualize(self.inflation_forecast).median(axis=1),
            },
            index=self.forecast_index,
        )

    def print_summary(self) -> None:
        """Print formatted forecast summary to console."""
        print(f"\n{'=' * 70}")
        print(f"FORECAST: {self.scenario_name} (cash rate {self.cash_rate:.2f}%)")
        print(f"{'=' * 70}")
        print(f"\nStarting point ({self.obs_index[-1]}):")
        print(f"  Output gap: {self.log_gdp_final - self.potential_final:+.4f}")
        print(f"  Unemployment: {self.unemployment_final:.2f}%")
        print(f"  NAIRU: {self.nairu_final:.2f}%")
        print(f"\n{self.summary().round(3).to_string()}")
        print(f"{'=' * 70}")
