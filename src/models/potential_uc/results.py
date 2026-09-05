"""Results container and I/O for the potential_uc model."""

import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import arviz as az
import numpy as np
import pandas as pd
import xarray as xr

from src.models.potential_uc.config import DEFAULT_OUTPUT_DIR

DEFAULT_CHART_BASE = Path(__file__).parent.parent.parent.parent / "charts"


@dataclass
class PotentialResults:
    """Container for potential_uc posterior draws plus the observations."""

    trace: az.InferenceData
    obs: dict[str, np.ndarray]
    obs_index: pd.PeriodIndex
    constants: dict[str, Any] = field(default_factory=dict)
    chart_obs: pd.DataFrame | None = None

    @property
    def posterior(self) -> xr.Dataset:
        """The trace's posterior group, narrowed at runtime.

        `az.InferenceData` builds its groups dynamically, so a static checker
        cannot see `.posterior` and every access reads as an attribute error.
        Narrowing here once means the rest of the class is checkable, and it
        turns a missing group into a clear failure rather than an AttributeError
        several frames deep.
        """
        posterior = getattr(self.trace, "posterior", None)
        if not isinstance(posterior, xr.Dataset):
            raise TypeError("trace has no posterior group — was it loaded from a completed run?")
        return posterior

    def _vector(self, var_name: str) -> pd.DataFrame:
        """Return a time x draw DataFrame for a vector-valued latent."""
        # xarray's .stack, not pandas' — PD013 does not apply.
        stacked = self.posterior[var_name].stack(sample=("chain", "draw"))  # noqa: PD013
        return pd.DataFrame(np.asarray(stacked.values), index=self.obs_index)

    def _scalar(self, var_name: str) -> np.ndarray:
        """Return the flattened posterior draws for a scalar parameter."""
        return np.asarray(self.posterior[var_name].values).ravel()

    @property
    def spec(self) -> str:
        """Which specification produced this trace, read from its contents."""
        if "trend_hours" in self.posterior:
            return "labour"
        # Must precede the `c` test: the production spec keeps the inflation
        # gap, so it carries `c` too, and differs by having the factor trends.
        if "trend_gk" in self.posterior:
            return "production"
        if "c" in self.posterior:
            return "inflation"
        if "inflation_deviation" in self.posterior:
            return "target"
        return "core"

    def _require_labour(self, what: str) -> None:
        if self.spec != "labour":
            raise ValueError(f"{what} is only available for the 'labour' specification")

    def trend_growth_posterior(self, annualised: bool = True) -> pd.DataFrame:
        """Return trend output growth, the g state (core specification only).

        Smooth by construction, being the model's drift state. Use
        `potential_growth_posterior` for the differenced-level equivalent.
        """
        if "trend_growth" not in self.posterior:
            raise ValueError(f"trend_growth is not a state of the {self.spec!r} specification")
        growth = self._vector("trend_growth")
        return growth * 4 if annualised else growth

    # --- Posterior distributions (time x draw) ---

    def potential_posterior(self) -> pd.DataFrame:
        """Return potential output, y* = h* + lp* (log x 100)."""
        return self._vector("potential_output")

    def output_gap_posterior(self) -> pd.DataFrame:
        """Return the output gap, log_gdp - y* (log x 100)."""
        return self._vector("output_gap")

    def _require_production(self, what: str) -> None:
        if self.spec != "production":
            raise ValueError(f"{what} is only available for the 'production' specification")

    def factor_trend_posterior(self, factor: str, annualised: bool = True) -> pd.DataFrame:
        """Return a factor's trend growth (production specification only).

        `factor` is "gk" (capital), "gl" (hours), "gm" (MFP) or "a" (the
        capital share). The first three are growth-rate states, so they are
        smooth by construction and need no differencing, unlike
        `potential_growth_posterior`. "a" is a share, not a growth rate, so
        pass `annualised=False` for it.
        """
        self._require_production(f"factor trend {factor!r}")
        if factor not in ("gk", "gl", "gm", "a"):
            raise ValueError(f"factor must be one of 'gk', 'gl', 'gm', 'a', got {factor!r}")
        trend = self._vector(f"trend_{factor}")
        return trend * 4 if annualised else trend

    def factor_contributions(self) -> pd.DataFrame:
        """Return alpha·g_K*, (1-alpha)·g_L* and g_M*, which sum to potential growth.

        The weights are the model's latent capital share, so the three columns
        add to `trend_growth` exactly at every draw and every quarter.
        """
        self._require_production("factor_contributions")
        alpha = self.factor_trend_posterior("a", annualised=False).median(axis=1)
        capital = self.factor_trend_posterior("gk").median(axis=1) * alpha
        hours = self.factor_trend_posterior("gl").median(axis=1) * (1.0 - alpha)
        mfp = self.factor_trend_posterior("gm").median(axis=1)
        return pd.DataFrame({"Capital": capital, "Hours": hours, "MFP": mfp})

    def trend_hours_posterior(self) -> pd.DataFrame:
        """Return trend hours, h* (log x 100). Labour specification only."""
        self._require_labour("trend_hours")
        return self._vector("trend_hours")

    def trend_productivity_posterior(self) -> pd.DataFrame:
        """Return trend productivity, lp* (log x 100, level not interpretable)."""
        self._require_labour("trend_productivity")
        return self._vector("trend_productivity")

    def trend_participation_posterior(self) -> pd.DataFrame:
        """Return trend participation, pr* (log x 100). Labour spec only."""
        self._require_labour("trend_participation")
        return self._vector("trend_participation")

    def trend_hpp_posterior(self) -> pd.DataFrame:
        """Return trend hours per labour-force participant. Labour spec only."""
        self._require_labour("trend_hours_per_participant")
        return self._vector("trend_hours_per_participant")

    def trend_prod_growth_posterior(self, annualised: bool = True) -> pd.DataFrame:
        """Return trend labour productivity growth, the g_lp state itself.

        This is the model's drift state, so it is smooth by construction. Use
        `trend_prod_growth_level_posterior` for the decomposition charts, where
        the three components must add to potential growth exactly.

        Labour specification only.
        """
        self._require_labour("trend_prod_growth")
        growth = self._vector("trend_prod_growth")
        return growth * 4 if annualised else growth

    def trend_prod_growth_level_posterior(self, year_ended: bool = True) -> pd.DataFrame:
        """Return trend productivity growth differenced from the lp* path.

        Additive with `trend_hours_growth_posterior` on the same basis, since
        y* = h* + lp* holds at every draw.
        """
        return self._growth(self.trend_productivity_posterior(), year_ended)

    @staticmethod
    def _growth(level: pd.DataFrame, year_ended: bool) -> pd.DataFrame:
        """Differenced growth of a level path, year-ended or annualised.

        Year-ended is the default for anything derived by differencing a level.
        The annualised quarterly rate carries the population estimate's
        quarter-to-quarter noise into what is meant to read as a trend; taking
        a four-quarter difference removes 57% of that jitter and is a real
        quantity rather than a filter, so it does not distort the endpoint.
        """
        return level.diff(4) if year_ended else level.diff() * 4

    def trend_hours_growth_posterior(self, year_ended: bool = True) -> pd.DataFrame:
        """Return trend hours growth, differenced from the h* path.

        Time-varying, because h* carries observed population. Differencing the
        level keeps this correct regardless of how h* is specified.
        """
        return self._growth(self.trend_hours_posterior(), year_ended)

    def potential_growth_posterior(self, year_ended: bool = True) -> pd.DataFrame:
        """Return potential output growth, differenced from the y* path."""
        return self._growth(self.potential_posterior(), year_ended)

    def ar_root_posterior(self) -> np.ndarray:
        """Return the modulus of the largest AR(2) root, one value per draw.

        The cycle is restricted to an AR(2) by prior only: `phi_1` and `phi_2`
        are centred on a hump-shaped cycle but carry no hard stationarity
        constraint. Whether the posterior actually respects the unit circle is
        therefore a fact to be checked, not asserted. The gap is stationary in
        a draw iff both roots of 1 - phi_1·z - phi_2·z^2 lie outside the unit
        circle, equivalently iff the companion-matrix eigenvalues
        (the roots of z^2 - phi_1·z - phi_2) lie inside it.
        """
        phi_1 = self._scalar("phi_1")
        phi_2 = self._scalar("phi_2")
        return np.array([
            np.abs(np.roots([1.0, -a, -b])).max() for a, b in zip(phi_1, phi_2, strict=True)
        ])

    def hours_gap_posterior(self) -> pd.DataFrame:
        """Return the share of the gap showing up in hours. Labour spec only."""
        self._require_labour("hours_gap")
        lam = self._scalar("lambda_h")
        gap = self.output_gap_posterior()
        return gap.mul(pd.Series(lam, index=gap.columns), axis=1)

    def productivity_gap_posterior(self) -> pd.DataFrame:
        """Return the residual share showing up in measured productivity."""
        return self.output_gap_posterior() - self.hours_gap_posterior()

    # --- Point estimates (posterior median) ---

    def potential_median(self) -> pd.Series:
        """Return the posterior median of potential output."""
        return self.potential_posterior().median(axis=1)

    def output_gap_median(self) -> pd.Series:
        """Return the posterior median of the output gap."""
        return self.output_gap_posterior().median(axis=1)

    def trend_prod_growth_median(self, annualised: bool = True) -> pd.Series:
        """Return the posterior median of trend productivity growth."""
        return self.trend_prod_growth_posterior(annualised).median(axis=1)

    def potential_growth_median(self, year_ended: bool = True) -> pd.Series:
        """Return the posterior median of potential output growth."""
        return self.potential_growth_posterior(year_ended).median(axis=1)

    # --- Bands ---

    @staticmethod
    def band(posterior: pd.DataFrame, lower: float = 0.05, upper: float = 0.95) -> pd.DataFrame:
        """Return a two-column DataFrame of posterior quantiles."""
        return pd.DataFrame({
            "lower": posterior.quantile(lower, axis=1),
            "upper": posterior.quantile(upper, axis=1),
        })

    def summary(self, var_names: list[str] | None = None) -> pd.DataFrame:
        """ArviZ summary for the scalar parameters."""
        if var_names is None and self.spec == "production":
            var_names = [
                "c", "sigma_e", "rho_e", "sigma_e_total",
                "sigma_obs_gk", "sigma_obs_gl", "sigma_obs_gm",
            ]
        if var_names is None and self.spec == "inflation":
            # rho_e and sigma_e_total exist only under `ar1_residual`;
            # `available` below drops them for the white-noise runs.
            var_names = ["c", "sigma_e", "rho_e", "sigma_e_total", "initial_trend_growth"]
        if var_names is None:
            var_names = (
                # `target` has no beta and no sigma_pi; `available` filters them.
                ["sigma_pi", "phi_1", "phi_2", "beta", "initial_trend_growth"]
                if self.spec in ("inflation", "core", "target")
                else [
                    "sigma_eta", "sigma_pi", "sigma_pr_obs",
                    "g_pr", "g_hpp", "phi_1", "phi_2",
                    "lambda_h", "lambda_pr", "beta",
                    "initial_prod_growth",
                ]
            )
        available = [v for v in var_names if v in self.posterior]
        summary = az.summary(self.trace, var_names=available)
        if not isinstance(summary, pd.DataFrame):
            raise TypeError(f"az.summary returned {type(summary).__name__}, expected a DataFrame")
        return summary


def load_results(
    output_dir: Path | str | None = None,
    prefix: str = "potential_uc",
) -> PotentialResults:
    """Load a saved trace and observations from disk."""
    output_dir = Path(output_dir) if output_dir is not None else DEFAULT_OUTPUT_DIR

    trace = az.from_netcdf(str(output_dir / f"{prefix}_trace.nc"))
    with (output_dir / f"{prefix}_obs.pkl").open("rb") as f:
        saved = pickle.load(f)  # noqa: S301 — loading our own model outputs, not untrusted data

    return PotentialResults(
        trace=trace,
        obs=saved["obs"],
        obs_index=saved["obs_index"],
        constants=saved.get("constants", {}),
        chart_obs=saved.get("chart_obs"),
    )
