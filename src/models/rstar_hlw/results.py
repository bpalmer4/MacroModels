"""Results container and I/O for the HLW r-star model."""

import pickle
from dataclasses import dataclass
from pathlib import Path

import arviz as az
import numpy as np
import pandas as pd

from src.models.common.results import PosteriorResults
from src.paths import CHARTS, MODEL_OUTPUTS

DEFAULT_OUTPUT_DIR = MODEL_OUTPUTS
DEFAULT_CHART_BASE = CHARTS


@dataclass(kw_only=True)
class RStarResults(PosteriorResults):
    """Container for HLW r-star posterior + observations."""

    obs: dict[str, np.ndarray]
    chart_obs: pd.DataFrame | None = None

    # --- Posteriors ---

    def r_star_posterior(self) -> pd.DataFrame:
        """Draws of r*, quarters down the rows."""
        return self._vector("r_star")

    def trend_growth_posterior(self) -> pd.DataFrame:
        """Draws of trend growth g, quarters down the rows."""
        return self._vector("trend_growth")

    def z_star_posterior(self) -> pd.DataFrame:
        """Draws of z, the part of r* that is not trend growth.

        Only the resolutions carrying the canonical r* = g + z identity have
        this state; the blend resolutions raise.
        """
        return self._vector("z_star")

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

    def z_star_median(self) -> pd.Series:
        """Posterior median z path."""
        return self.z_star_posterior().median(axis=1)

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
        saved = pickle.load(f)

    return RStarResults(
        trace=trace,
        obs=saved["obs"],
        obs_index=saved["obs_index"],
        constants=saved.get("constants", {}),
        chart_obs=saved.get("chart_obs"),
    )
