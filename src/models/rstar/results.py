"""Results container and I/O for the HLW r-star model."""

import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import arviz as az
import numpy as np
import pandas as pd

from src.models.common.extraction import get_vector_var

DEFAULT_OUTPUT_DIR = Path(__file__).parent.parent.parent.parent / "model_outputs"
DEFAULT_CHART_BASE = Path(__file__).parent.parent.parent.parent / "charts"


@dataclass
class RStarResults:
    """Container for HLW r-star posterior + observations."""

    trace: az.InferenceData
    obs: dict[str, np.ndarray]
    obs_index: pd.PeriodIndex
    constants: dict[str, Any] = field(default_factory=dict)
    chart_obs: pd.DataFrame | None = None

    def _vector(self, var_name: str) -> pd.DataFrame:
        samples = get_vector_var(var_name, self.trace)
        samples.index = self.obs_index
        return samples

    # --- Posteriors ---

    def r_star_posterior(self) -> pd.DataFrame:
        return self._vector("r_star")

    def trend_growth_posterior(self) -> pd.DataFrame:
        return self._vector("trend_growth")

    def potential_posterior(self) -> pd.DataFrame:
        return self._vector("potential_output")

    def output_gap_posterior(self) -> pd.DataFrame:
        log_gdp = pd.Series(self.obs["log_gdp"], index=self.obs_index)
        potential = self.potential_posterior()
        return potential.rsub(log_gdp, axis=0)

    # --- Point estimates (posterior median) ---

    def r_star_median(self) -> pd.Series:
        return self.r_star_posterior().median(axis=1)

    def trend_growth_median(self) -> pd.Series:
        return self.trend_growth_posterior().median(axis=1)

    def potential_median(self) -> pd.Series:
        return self.potential_posterior().median(axis=1)

    def output_gap_median(self) -> pd.Series:
        log_gdp = pd.Series(self.obs["log_gdp"], index=self.obs_index)
        return log_gdp - self.potential_median()


def load_results(
    output_dir: Path | str | None = None,
    prefix: str = "rstar_hlw",
) -> RStarResults:
    """Load saved trace + observations from disk."""
    if output_dir is None:
        output_dir = DEFAULT_OUTPUT_DIR
    output_dir = Path(output_dir)

    trace = az.from_netcdf(str(output_dir / f"{prefix}_trace.nc"))
    with (output_dir / f"{prefix}_obs.pkl").open("rb") as f:
        saved = pickle.load(f)

    return RStarResults(
        trace=trace,
        obs=saved["obs"],
        obs_index=saved["obs_index"],
        constants=saved.get("constants", {}),
        chart_obs=saved.get("chart_obs"),
    )
