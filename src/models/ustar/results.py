"""Container and loader for ustar posterior draws."""

import pickle
from dataclasses import dataclass
from pathlib import Path

import arviz as az
import numpy as np
import pandas as pd

from src.models.common.results import PosteriorResults
from src.paths import CHARTS, MODEL_OUTPUTS
from src.utilities.rate_conversion import quarterly

DEFAULT_OUTPUT_DIR = MODEL_OUTPUTS
DEFAULT_CHART_BASE = CHARTS


@dataclass(kw_only=True)
class UStarResults(PosteriorResults):
    """Container for ustar posterior draws plus the observations."""

    obs: dict[str, np.ndarray]
    chart_obs: pd.DataFrame | None = None

    @property
    def has_phillips(self) -> bool:
        """Whether this run included the Phillips curve, read from the trace."""
        return "gamma_pi" in self.posterior

    # --- The object the model exists to estimate ---

    def ustar_posterior(self) -> pd.DataFrame:
        """Return u*, the unemployment rate consistent with output at potential."""
        return self._vector("ustar")

    def ustar_median(self) -> pd.Series:
        """Return the posterior median of u*."""
        return self.ustar_posterior().median(axis=1)

    def ugap_posterior(self) -> pd.DataFrame:
        """Return the unemployment gap in percentage points, u - u*.

        Computed here rather than read from the trace: the `ugap` recorded by
        the model is the scale-invariant `(u - u*)/u` the Phillips curve needs,
        which is a different number and not what a reader of a chart labelled
        "percentage points" expects.
        """
        return self.ustar_posterior().rsub(self.unemployment(), axis=0)

    def ugap_median(self) -> pd.Series:
        """Return the posterior median of the unemployment gap."""
        return self.ugap_posterior().median(axis=1)

    @property
    def decays(self) -> bool:
        """Whether u* was specified as decaying toward an equilibrium."""
        return "phi_ustar" in self.posterior

    def ustar_change_decomposition(self) -> pd.DataFrame:
        """Split u*'s quarterly change into its two exact components.

            u*_t - u*_{t-1} = phi·(u*_eq - u*_{t-1})  +  e_u,t

        The first term is the specification pulling u* toward its estimated
        equilibrium; the second is what the data added on top. `deterministic`
        is the path u* would have taken from the same starting point with every
        innovation set to zero, so the distance between it and u* is the whole
        of what the data contributed.

        Computed on posterior medians, so it describes the reported path rather
        than integrating over uncertainty.
        """
        if not self.decays:
            raise ValueError("this run has no decay terms to decompose")

        median = self.posterior.median(dim=("chain", "draw"))
        phi = float(median["phi_ustar"])
        eq = float(median["ustar_eq"])
        ustar = self.ustar_median()

        decay = phi * (eq - ustar.shift(1))
        innovation = ustar.diff() - decay

        # The counterfactual path: same start, no innovations at all.
        path = [ustar.iloc[0]]
        for _ in range(len(ustar) - 1):
            path.append(path[-1] + phi * (eq - path[-1]))

        return pd.DataFrame({
            "ustar": ustar,
            "deterministic": pd.Series(path, index=ustar.index),
            "decay": decay,
            "innovation": innovation,
        })

    def unemployment(self) -> pd.Series:
        """Return the observed unemployment rate."""
        return pd.Series(self.obs["u"], index=self.obs_index)

    # --- Diagnostics that would show the model failing ---

    def prob_beta_positive(self) -> float:
        """P(beta > 0): whether the data agree with Okun's law at all.

        The analogue of ystar's P(c > 0). If this is not comfortably
        above 0.9 the Okun channel is not identifying u*'s level, which is the
        whole reason the equation is here.
        """
        if "beta_okun" not in self.trace.posterior.data_vars:
            return float("nan")  # no Okun equation in this run
        return float((self._scalar("beta_okun") > 0).mean())

    def prob_gamma_negative(self) -> float:
        """P(gamma_pi < 0): whether slack disinflates in this sample."""
        return float((self._scalar("gamma_pi") < 0).mean())

    def ustar_variation(self) -> dict[str, float]:
        """How much u* actually moves, against how much the prior lets it move.

        The failure mode this is built to catch is the one that killed
        `ustar_wage`: a u* whose path is set by the smoothness prior rather
        than by the data. `sd of du*` close to the imposed `sigma_ustar` means
        u* is wandering as freely as the prior permits and the data are not
        holding it; far below means the data are binding.
        """
        ustar = self.ustar_median()
        allowed = (
            float(np.median(self._scalar("sigma_ustar")))
            if self.free_sigma_ustar
            else float(self.constants.get("sigma_ustar", np.nan))
        )
        label = "sigma_ustar (posterior median)" if self.free_sigma_ustar else "imposed sigma_ustar"
        return {
            "sd of u*": float(ustar.std()),
            "sd of du*": float(ustar.diff().dropna().std()),
            label: allowed,
            "range of u*": float(ustar.max() - ustar.min()),
        }

    @property
    def free_sigma_ustar(self) -> bool:
        """Whether the drift was estimated under a prior rather than imposed."""
        return "sigma_ustar" in self.posterior

    def inflation_decomposition(self) -> pd.DataFrame:
        """Split observed inflation into the Phillips curve's own terms.

        Columns are the equation term by term, on the quarterly basis the model
        fits, so they sum to observed inflation exactly:

            pi = target + excess + demand + supply + residual

        `target` is the flat 2.5 anchor and `excess` is beta x (expectations -
        target), so the two are distinct objects rather than two readings of
        the same one. `demand` is the only term carrying u*, which is what
        makes this the readable statement of how much the model attributes to
        the labour market as against anchoring and supply.
        """
        if not self.has_phillips:
            raise ValueError("no Phillips curve in this run: nothing to decompose")

        index = self.obs_index
        median = self.posterior.median(dim=("chain", "draw"))
        pi_exp = pd.Series(self.obs["pi_exp"], index=index)

        anchor = pd.Series(quarterly(float(self.constants["anchor"])), index=index)
        excess = float(median["beta_pi"]) * (quarterly(pi_exp) - anchor)
        demand = float(median["gamma_pi"]) * pd.Series(
            np.asarray(median["ugap"].values), index=index,
        )
        gscpi = pd.Series(self.obs["gscpi"], index=index)
        supply = (
            float(median["rho_pi"]) * pd.Series(self.obs["d4pm"], index=index)
            + float(median["xi_gscpi"]) * gscpi**2 * np.sign(gscpi)
        )
        observed = pd.Series(self.obs["pi"], index=index)
        fitted = anchor + excess + demand + supply

        return pd.DataFrame({
            "observed": observed,
            "anchor": anchor,
            "excess": excess,
            "demand": demand,
            "supply": supply,
            "residual": observed - fitted,
        })

    def implied_ustar(self) -> pd.Series:
        """Return the u* each quarter's inflation would need, taken on its own.

        Invert the Phillips curve with the residual set to zero and solve for
        u*. Writing the equation's non-demand terms as `R`,

            R      = pi - q(anchor) - beta_pi x [q(pi_exp) - q(anchor)]
                     - rho_pi x d4pm - xi_gscpi x GSCPI^2 x sign(GSCPI)
            u*_t   = u_t x (1 - R_t / gamma_pi)

        Not an estimator. It is the diagnostic that makes the state law
        visible: it is what the data would say about u* with no smoothness
        prior at all, so plotting the fitted u* against it shows how much of
        the reported path is the prior rather than the likelihood. That
        question is sharper here than in the joint model, because this model
        imposes `sigma_ustar` and its own run output already reports that u*
        "wanders as freely as the prior allows".

        Expect it to be wild: it divides a noisy residual by a coefficient near
        one and multiplies by the unemployment rate.

        Coefficients are posterior medians, matching `inflation_decomposition`.
        """
        if not self.has_phillips:
            raise ValueError("no Phillips curve in this run: nothing to invert")

        index = self.obs_index
        median = self.posterior.median(dim=("chain", "draw"))
        anchor = quarterly(float(self.constants["anchor"]))
        gscpi = pd.Series(self.obs["gscpi"], index=index)

        residual_free = (
            pd.Series(self.obs["pi"], index=index)
            - anchor
            - float(median["beta_pi"]) * (quarterly(pd.Series(self.obs["pi_exp"], index=index)) - anchor)
            - float(median["rho_pi"]) * pd.Series(self.obs["d4pm"], index=index)
            - float(median["xi_gscpi"]) * gscpi**2 * np.sign(gscpi)
        )
        u = pd.Series(self.obs["u"], index=index)
        return u * (1.0 - residual_free / float(median["gamma_pi"]))

    def summary(self, var_names: list[str] | None = None) -> pd.DataFrame:
        """ArviZ summary for the scalar parameters."""
        if var_names is None:
            # Named rather than discovered, so the table keeps its order. The
            # Okun block is absent when that equation is off.
            present = set(self.trace.posterior.data_vars)
            var_names = [v for v in ("beta_okun", "sigma_okun") if v in present]
            if self.free_sigma_ustar:
                var_names.insert(0, "sigma_ustar")
            if self.has_phillips:
                var_names += ["gamma_pi", "beta_pi", "rho_pi", "xi_gscpi", "epsilon_pi"]
        summary = az.summary(self.trace, var_names=var_names)
        if not isinstance(summary, pd.DataFrame):
            raise TypeError("az.summary returned a Dataset — expected the DataFrame form")
        return summary


def load_results(
    output_dir: Path | str | None = None,
    prefix: str = "ustar",
) -> UStarResults:
    """Load a saved trace and observations from disk."""
    output_dir = Path(output_dir) if output_dir is not None else DEFAULT_OUTPUT_DIR

    trace_path = output_dir / f"{prefix}_trace.nc"
    obs_path = output_dir / f"{prefix}_obs.pkl"

    trace = az.from_netcdf(str(trace_path))
    with obs_path.open("rb") as f:
        saved = pickle.load(f)  # noqa: S301 — our own output, written by save_results

    return UStarResults(
        trace=trace,
        obs=saved["obs"],
        obs_index=saved["obs_index"],
        constants=saved.get("constants", {}),
        chart_obs=saved.get("chart_obs"),
    )
