"""Results container and I/O for the joint y*/u* model."""

import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import arviz as az
import numpy as np
import pandas as pd
import xarray as xr

from src.models.ystar_ustar.config import DEFAULT_OUTPUT_DIR
from src.utilities.rate_conversion import quarterly


@dataclass
class JointResults:
    """Posterior draws plus the observations that produced them."""

    trace: az.InferenceData
    obs: dict[str, np.ndarray]
    obs_index: pd.PeriodIndex
    constants: dict[str, Any] = field(default_factory=dict)
    chart_obs: pd.DataFrame | None = None

    @property
    def posterior(self) -> xr.Dataset:
        """The trace's posterior group, narrowed at runtime."""
        posterior = getattr(self.trace, "posterior", None)
        if not isinstance(posterior, xr.Dataset):
            raise TypeError("trace has no posterior group — was it loaded from a completed run?")
        return posterior

    def _vector(self, var_name: str) -> pd.DataFrame:
        """Return a time x draw DataFrame for a vector-valued latent."""
        stacked = self.posterior[var_name].stack(sample=("chain", "draw"))  # noqa: PD013
        return pd.DataFrame(np.asarray(stacked.values), index=self.obs_index)

    def _scalar(self, var_name: str) -> np.ndarray:
        """Return the flattened posterior draws for a scalar parameter."""
        return np.asarray(self.posterior[var_name].values).ravel()

    @property
    def has_free_gap(self) -> bool:
        """Whether this run gave the gap a free component."""
        return "v" in self.posterior

    @property
    def has_phillips(self) -> bool:
        """Whether the Phillips curve was in the likelihood."""
        return "gamma_pi" in self.posterior or "kappa_gap" in self.posterior

    @property
    def gap_spec(self) -> str:
        """Which gap specification produced this trace, read from its contents."""
        return "cycle" if "rho_gap" in self.posterior else "defined"

    # --- The three stars ---

    def potential_posterior(self) -> pd.DataFrame:
        """Potential output, log x 100."""
        return self._vector("potential_output")

    def potential_growth_posterior(self, year_ended: bool = True) -> pd.DataFrame:
        """Growth of potential output, per cent."""
        level = self.potential_posterior()
        periods = 4 if year_ended else 1
        scale = 1.0 if year_ended else 4.0
        return level.diff(periods) * scale

    def ustar_posterior(self) -> pd.DataFrame:
        """u*, per cent."""
        return self._vector("ustar")

    def unemployment_gap_posterior(self) -> pd.DataFrame:
        """Return u - u*, in percentage points, not the normalised Phillips form."""
        u = pd.Series(self.obs["u"], index=self.obs_index)
        return self.ustar_posterior().rsub(u, axis=0)

    # --- The gap and its decomposition, which is the point of this model ---

    def output_gap_posterior(self) -> pd.DataFrame:
        """Return the whole gap, c·(pi - anchor) + v."""
        return self._vector("output_gap")

    def defined_gap_posterior(self) -> pd.DataFrame:
        """Return the inflation-defined part alone, c·(pi - anchor).

        This is the object `ystar` reports as its output gap, so it is the
        like-for-like comparison with that model.
        """
        if "defined_gap" not in self.posterior:
            return self.output_gap_posterior()
        return self._vector("defined_gap")

    def free_gap_posterior(self) -> pd.DataFrame:
        """Return the free part alone, v, being cycle that inflation does not see."""
        if not self.has_free_gap:
            raise ValueError("this run has no free gap component (v)")
        return self._vector("v")

    def gap_variance_share(self) -> dict[str, float]:
        """How the gap's variance splits between the defined and free parts.

        Computed on posterior medians rather than per draw, so it describes the
        reported paths. The two parts are not orthogonal by construction, so the
        shares need not sum to one and the covariance term is reported too.

        **Scoped to the fitted quarters.** Inside the excluded window no equation
        carries a likelihood, so `v` there is a prior draw whose median across
        draws is ~0 while its true spread is `sigma_v`. Including those quarters
        puts six near-zero values into the free component's variance and
        overstates the inflation-defined share: 47.2% against 44.9% on the
        current run. `ystar`'s notes record being caught by the same thing.
        """
        if self.gap_spec == "cycle":
            # There is no decomposition to report: the gap is one latent state
            # and inflation observes it rather than defining any part of it.
            # That is the point of the reparameterisation.
            return {}
        if not self.has_free_gap:
            return {"defined": 1.0, "free": 0.0, "covariance": 0.0}

        fitted = self.fitted_mask()
        defined = self.defined_gap_posterior().median(axis=1)[fitted]
        free = self.free_gap_posterior().median(axis=1)[fitted]
        total = float((defined + free).var())
        if total <= 0:
            raise ValueError("the fitted gap has no variance — nothing to decompose")
        return {
            "defined": float(defined.var()) / total,
            "free": float(free.var()) / total,
            "covariance": 2.0 * float(defined.cov(free)) / total,
        }

    # --- Residuals, and the covariance that identifies sigma_v ---

    def gdp_residual_posterior(self) -> pd.DataFrame:
        """e_c = log_gdp - y* - gap."""
        log_gdp = pd.Series(self.obs["log_gdp"], index=self.obs_index)
        fitted = self.potential_posterior() + self.output_gap_posterior()
        return fitted.rsub(log_gdp, axis=0)

    def okun_residual_posterior(self) -> pd.DataFrame:
        """e_o = u - u* + beta·gap."""
        if "beta_okun" not in self.posterior:
            raise ValueError("this run has no Okun equation")
        beta = self._scalar("beta_okun")
        return self.unemployment_gap_posterior() + self.output_gap_posterior() * beta

    def residual_correlation(self) -> float:
        """corr(e_c, e_o) on posterior medians.

        The moment that identifies `sigma_v`. Reported so the diagnostic can be
        read directly rather than inferred from `sigma_v`'s posterior.
        """
        return float(
            self.gdp_residual_posterior().median(axis=1)
            .corr(self.okun_residual_posterior().median(axis=1)),
        )

    # --- The Phillips curve, term by term ---

    def inflation_decomposition(self) -> pd.DataFrame:
        """Split observed inflation into the Phillips curve's own terms.

        Columns are the equation term by term, on the quarterly basis the model
        fits, so they sum to observed inflation exactly:

            pi = target + excess + demand + supply + residual

        `demand` is the only term carrying u*, which makes this the readable
        statement of how much inflation the model attributes to the labour
        market as against anchoring and supply. Taken unchanged from `ustar` so
        the two charts are directly comparable.
        """
        if not self.has_phillips:
            raise ValueError("no Phillips curve in this run: nothing to decompose")

        index = self.obs_index
        median = self.posterior.median(dim=("chain", "draw"))
        pi_exp = pd.Series(self.obs["pi_exp"], index=index)

        anchor = pd.Series(quarterly(float(self.constants["anchor"])), index=index)
        excess = float(median["beta_pi"]) * (quarterly(pi_exp) - anchor)
        # The demand term is the only thing that differs between the two specs:
        # kappa x output gap under "cycle", gamma x unemployment gap under
        # "defined". Everything else in the equation is identical, so the two
        # decomposition charts are directly comparable.
        if self.gap_spec == "cycle":
            demand = float(median["kappa_gap"]) * pd.Series(
                np.asarray(median["output_gap"].values), index=index,
            )
        else:
            demand = float(median["gamma_pi"]) * pd.Series(
                np.asarray(median["ugap"].values), index=index,
            )
        gscpi = pd.Series(self.obs["gscpi"], index=index)
        supply = (
            float(median["rho_pi"]) * pd.Series(self.obs["d4pm"], index=index)
            + float(median["xi_gscpi"]) * gscpi**2 * np.sign(gscpi)
        )
        observed = pd.Series(self.obs["pi_qtr"], index=index)
        fitted = anchor + excess + demand + supply

        return pd.DataFrame({
            "observed": observed,
            "anchor": anchor,
            "excess": excess,
            "demand": demand,
            "supply": supply,
            "residual": observed - fitted,
        })

    # --- Medians, matching the accessor names `rstar` reads on the parents ---

    def output_gap_median(self) -> pd.Series:
        """Return the posterior median of the output gap, matching `ystar`'s name."""
        return self.output_gap_posterior().median(axis=1)

    def ugap_median(self) -> pd.Series:
        """Return the posterior median of u - u*, in percentage points.

        Percentage points, not the scale-invariant `(u - u*)/u` the Phillips
        curve uses, matching `ustar.ugap_median` so `rstar` can read either
        source without knowing which it has.
        """
        return self.unemployment_gap_posterior().median(axis=1)

    # --- Convenience ---

    @property
    def excluded_window(self) -> tuple[str, str] | None:
        """Return the run's excluded window, read from its own recorded settings."""
        window = self.constants.get("exclude_window")
        return window if isinstance(window, tuple) else None

    def fitted_mask(self) -> pd.Series:
        """Return True where the run carried a likelihood term.

        Read from the run's own constants, so a statistic computed on it cannot
        depend on whether some earlier call happened to set a module global.
        """
        window = self.excluded_window
        if window is None:
            return pd.Series(data=True, index=self.obs_index)
        lo, hi = window
        excluded = (self.obs_index >= pd.Period(lo, freq="Q")) & (
            self.obs_index <= pd.Period(hi, freq="Q")
        )
        return pd.Series(data=~np.asarray(excluded), index=self.obs_index)

    @staticmethod
    def band(posterior: pd.DataFrame, lower: float = 0.05, upper: float = 0.95) -> pd.DataFrame:
        """Return lower/median/upper quantile columns for a time x draw frame."""
        return pd.DataFrame(
            {
                "lower": posterior.quantile(lower, axis=1),
                "median": posterior.median(axis=1),
                "upper": posterior.quantile(upper, axis=1),
            },
        )

    def summary(self, var_names: list[str] | None = None) -> pd.DataFrame:
        """ArviZ summary for the scalar parameters."""
        if var_names is None:
            var_names = [
                name for name in
                ("c", "sigma_v", "rho_gap", "kappa_gap", "sigma_e", "beta_okun",
                 "sigma_okun", "gamma_pi", "beta_pi", "rho_pi", "xi_gscpi",
                 "epsilon_pi", "initial_trend_growth")
                if name in self.posterior
            ]
        return az.summary(self.trace, var_names=var_names, hdi_prob=0.9)


def load_results(
    prefix: str = "ystar_ustar",
    output_dir: Path | str | None = None,
) -> JointResults:
    """Load a saved run from `model_outputs`."""
    output_dir = Path(output_dir) if output_dir is not None else DEFAULT_OUTPUT_DIR

    trace_path = output_dir / f"{prefix}_trace.nc"
    obs_path = output_dir / f"{prefix}_obs.pkl"
    if not trace_path.exists():
        raise FileNotFoundError(
            f"No saved run found at {trace_path}. Run ./run-ystar-ustar.sh first.",
        )

    trace = az.from_netcdf(str(trace_path))
    with obs_path.open("rb") as f:
        saved = pickle.load(f)  # noqa: S301 — our own file, written by save_results

    return JointResults(
        trace=trace,
        obs=saved["obs"],
        obs_index=saved["obs_index"],
        constants=saved.get("constants", {}),
        chart_obs=saved.get("chart_obs"),
    )
