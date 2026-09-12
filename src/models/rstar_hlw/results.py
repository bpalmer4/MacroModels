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
        """Draws of r*, quarters down the rows."""
        return self._vector("r_star")

    def trend_growth_posterior(self) -> pd.DataFrame:
        """Draws of trend growth g, quarters down the rows."""
        return self._vector("trend_growth")

    def potential_posterior(self) -> pd.DataFrame:
        """Draws of potential output, quarters down the rows."""
        return self._vector("potential_output")

    def output_gap_posterior(self) -> pd.DataFrame:
        """Draws of the output gap: observed log GDP less each potential draw."""
        log_gdp = pd.Series(self.obs["log_gdp"], index=self.obs_index)
        potential = self.potential_posterior()
        return potential.rsub(log_gdp, axis=0)

    # --- Point estimates (posterior median) ---

    def r_star_median(self) -> pd.Series:
        """Posterior median r* path."""
        return self.r_star_posterior().median(axis=1)

    def trend_growth_median(self) -> pd.Series:
        """Posterior median trend growth path."""
        return self.trend_growth_posterior().median(axis=1)

    def potential_median(self) -> pd.Series:
        """Posterior median potential output path."""
        return self.potential_posterior().median(axis=1)

    def output_gap_median(self) -> pd.Series:
        """Output gap against the median potential path.

        Not the median of the gap draws: it is log GDP less the median
        potential, which is the same thing only because the median is
        order-preserving and log GDP is data.
        """
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
        saved = pickle.load(f)  # noqa: S301 — our own file, written by save_results

    return RStarResults(
        trace=trace,
        obs=saved["obs"],
        obs_index=saved["obs_index"],
        constants=saved.get("constants", {}),
        chart_obs=saved.get("chart_obs"),
    )
