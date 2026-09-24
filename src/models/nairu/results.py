"""Results container and I/O for NAIRU + Output Gap model.

Shared by validate.py, analyse.py, and forecast.py.
"""

import pickle
from dataclasses import dataclass
from pathlib import Path

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm

from src.models.common.results import PosteriorResults
from src.models.nairu.config import ModelConfig
from src.models.nairu.estimate import build_model
from src.paths import CHARTS, MODEL_OUTPUTS

# Default paths
DEFAULT_OUTPUT_DIR = MODEL_OUTPUTS
DEFAULT_CHART_BASE = CHARTS


@dataclass(kw_only=True)
class NAIRUResults(PosteriorResults):
    """Results from NAIRU + Output Gap estimation."""

    obs: dict[str, np.ndarray]
    config: ModelConfig
    anchor_label: str
    model: pm.Model | None = None
    chart_obs: pd.DataFrame | None = None

    def nairu_posterior(self) -> pd.DataFrame:
        """Extract NAIRU posterior as DataFrame."""
        return self._vector("nairu")

    def potential_posterior(self) -> pd.DataFrame:
        """Extract potential output posterior as DataFrame."""
        return self._vector("potential_output")

    def nairu_median(self) -> pd.Series:
        """NAIRU point estimate (posterior median)."""
        return self.nairu_posterior().median(axis=1)

    def potential_median(self) -> pd.Series:
        """Potential output point estimate (posterior median)."""
        return self.potential_posterior().median(axis=1)

    def unemployment_gap(self) -> pd.Series:
        """Unemployment gap = U - NAIRU."""
        U = pd.Series(self.obs["U"], index=self.obs_index)
        return U - self.nairu_median()

    def output_gap(self) -> pd.Series:
        """Output gap = log(GDP) - log(potential)."""
        log_gdp = pd.Series(self.obs["log_gdp"], index=self.obs_index)
        return log_gdp - self.potential_median()


def load_results(
    output_dir: Path | str | None = None,
    prefix: str = "nairu_output_gap",
    rebuild_model: bool = False,
) -> NAIRUResults:
    """Load model results from disk.

    Args:
        output_dir: Directory containing saved results
        prefix: Filename prefix used when saving
        rebuild_model: If True, rebuild the PyMC model (needed for PPC)

    Returns:
        NAIRUResults container

    """
    if output_dir is None:
        output_dir = DEFAULT_OUTPUT_DIR
    output_dir = Path(output_dir)

    # Load trace
    trace_path = output_dir / f"{prefix}_trace.nc"
    trace = az.from_netcdf(str(trace_path))
    print(f"Loaded trace from: {trace_path}")

    # Load observations and config
    obs_path = output_dir / f"{prefix}_obs.pkl"
    with obs_path.open("rb") as f:
        data = pickle.load(f)

    obs = data["obs"]
    obs_index = data["obs_index"]
    constants = data.get("constants", {})
    anchor_label = data.get("anchor_label", "Anchor: Estimated expectations")
    chart_obs = data.get("chart_obs")

    config = ModelConfig.from_dict(data["config"])

    print(f"Loaded observations from: {obs_path}")
    print(f"  Period: {obs_index.min()} to {obs_index.max()} ({len(obs_index)} periods)")
    print(f"  Variant: {config.label}")
    print(f"  {anchor_label}")

    # Optionally rebuild model (needed for posterior predictive checks)
    model = None
    if rebuild_model:
        model = build_model(obs, config)

    return NAIRUResults(
        trace=trace,
        obs=obs,
        obs_index=obs_index,
        config=config,
        anchor_label=anchor_label,
        model=model,
        constants=constants,
        chart_obs=chart_obs,
    )
