"""Container and loader for rstar posterior draws, plus the derived series."""

import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import arviz as az
import numpy as np
import pandas as pd
import xarray as xr

DEFAULT_OUTPUT_DIR = Path(__file__).parent.parent.parent.parent / "model_outputs"
DEFAULT_CHART_BASE = Path(__file__).parent.parent.parent.parent / "charts"


@dataclass
class RStarResults:
    """Container for rstar posterior draws plus the observations."""

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

    def _extra(self, name: str) -> pd.Series:
        """Return one of the ragged chart series, or an all-NaN series."""
        if self.chart_obs is None or name not in self.chart_obs:
            return pd.Series(np.nan, index=self.obs_index)
        return self.chart_obs[name]

    # --- The object the model estimates ---

    def rstar_posterior(self) -> pd.DataFrame:
        """Return r*, the natural rate, on the HLW scale."""
        return self._vector("r_star")

    def rstar_median(self) -> pd.Series:
        """Return the posterior median of r*."""
        return self.rstar_posterior().median(axis=1)

    def term_premium_posterior(self) -> pd.DataFrame:
        """Return the term premium, everything in the yield that is not r*."""
        return self._vector("tp")

    def real_yield(self) -> pd.Series:
        """Return the observed indexed real 10-year yield."""
        return pd.Series(self.obs["y"], index=self.obs_index)

    def world_rstar(self) -> pd.Series:
        """Return the world r* series the model was anchored on."""
        return pd.Series(self.obs["w"], index=self.obs_index)

    # --- Derived, no parameters ---

    def business_rstar(self) -> pd.Series:
        """Return r* plus the observed corporate spread: the cost of capital to firms.

        The bond market prices the globally-arbitraged risk-free rate; what
        governs investment sits above it by the external finance premium. That
        premium is data here, so this costs no parameters and no identification.
        NaN before 2005Q1, where the spread series begins.
        """
        return self.rstar_median() + self._extra("spread")

    def supply_contribution(self, *, positive_only: bool | None = None) -> pd.Series:
        """Return the supply-driven part of four-quarter inflation.

        From `ustar`'s Phillips decomposition, rolling four-quarter. With
        `positive_only`, negative values are zeroed, so the series is what a
        central bank would decline to tighten into rather than a symmetric
        measure of supply's contribution.
        """
        if positive_only is None:
            positive_only = bool(self.constants.get("supply_positive_only", True))
        supply = self._extra("supply")
        return supply.clip(lower=0.0) if positive_only else supply

    def rule_inflation(self, *, look_through: bool | None = None, positive_only: bool | None = None) -> pd.Series:
        """Return the inflation rate the rule responds to."""
        if look_through is None:
            look_through = bool(self.constants.get("look_through_supply", True))
        pi = self._extra("pi")
        if not look_through:
            return pi
        # Quarters with no supply estimate fall back to headline inflation
        # rather than being dropped: filling with zero leaves the rule running
        # on the unadjusted rate, which is the honest default.
        supply = self.supply_contribution(positive_only=positive_only)
        return pi - supply.reindex(pi.index).fillna(0.0)

    def policy_change(
        self,
        *,
        use_ugap: bool | None = None,
        look_through: bool | None = None,
        positive_only: bool | None = None,
    ) -> pd.Series:
        """Return the prescribed change in the cash rate, in points per quarter.

            d_i = a_pi·(pi - anchor) + a_gap·gap

        A first-difference rule: how far to *move* the cash rate, not where to
        put it. It needs no r*, which is why it is the rule this package uses —
        the level of r* is not identified here (`wedge_0` and `mu_tp` correlate
        at -0.76), and a level rule fed that unidentified level prescribed
        tightening right through 2012-2021, a decade of below-target inflation
        and a negative output gap.

        Positive means tighten, negative means ease. The reading is "what still
        needs to happen to get back to target", not "where the rate belongs".
        """
        constants = self.constants
        anchor = float(constants.get("anchor", 2.5))
        a_pi = float(constants.get("rule_pi", 0.125))
        a_gap = float(constants.get("rule_gap", 0.125))

        if use_ugap is None:
            use_ugap = bool(constants.get("taylor_use_ugap", False))
        # An unemployment gap is the negative of a demand gap: unemployment
        # below u* is excess demand, so the sign flips before it enters.
        gap = -self._extra("ugap") if use_ugap else self._extra("ygap")

        pi = self.rule_inflation(look_through=look_through, positive_only=positive_only)
        return a_pi * (pi - anchor) + a_gap * gap

    def policy_change_delivered(self) -> pd.Series:
        """Return the actual quarterly change in the cash rate."""
        return self.cash_rate().diff()

    def policy_level(
        self,
        *,
        use_ugap: bool | None = None,
        look_through: bool | None = None,
        positive_only: bool | None = None,
    ) -> pd.Series:
        """Return where the cash rate should be: today's rate plus the prescribed move.

        A level, but not one that needs r*. The difference rule says how far to
        move; adding that to the rate that actually prevails gives a level
        anchored on an observed policy rate rather than on an estimated neutral
        rate. It re-anchors every quarter, so it cannot accumulate the drift a
        cumulated path would.

        Read it as "given where the rate is and what inflation and the gap are
        doing, here is where it should be", not as a neutral-rate estimate.
        """
        change = self.policy_change(
            use_ugap=use_ugap, look_through=look_through, positive_only=positive_only,
        )
        return self.cash_rate() + change

    def taylor_level(
        self,
        *,
        use_ugap: bool | None = None,
        look_through: bool | None = None,
        positive_only: bool | None = None,
        a_pi: float = 0.5,
        a_gap: float = 0.5,
    ) -> pd.Series:
        """Return the level Taylor prescription, using this model's r*.

            i* = r* + pi + a_pi·(pi - anchor) + a_gap·gap

        Taylor's original coefficients, because this is a level rule — the
        quarterly-scaled ones belong to the difference rule.

        This was unusable while r* was a smooth random walk anchored on world
        r*: the level was unidentified and the rule prescribed tightening right
        through 2012-2021. With the free Student-t wedge it behaves across the
        whole sample, sitting at or below the actual rate through that decade
        and above it now. The level is still only as good as `sigma_walk`,
        which is imposed.
        """
        anchor = float(self.constants.get("anchor", 2.5))
        if use_ugap is None:
            use_ugap = bool(self.constants.get("taylor_use_ugap", False))
        gap = -self._extra("ugap") if use_ugap else self._extra("ygap")
        pi = self.rule_inflation(look_through=look_through, positive_only=positive_only)
        return self.rstar_median() + pi + a_pi * (pi - anchor) + a_gap * gap

    def nominal_rstar(self, *, on_expectations: bool = False) -> pd.Series:
        """Return the neutral *nominal* cash rate: r* plus expected inflation.

        With `on_expectations` false — the default — the inflation term is the
        2.5% target, giving the steady-state neutral rate: where the cash rate
        would sit with inflation at target and the gap closed. That is the
        number directly comparable to the actual cash rate, and the comparison
        is the stance: above it policy is restrictive, below it expansionary.

        With `on_expectations` true, actual expectations are used instead, which
        gives the neutral rate *for the inflation currently expected* rather
        than for the target. The two differ whenever expectations are away from
        2.5, and the gap between them is a de-anchoring measure in its own right.
        """
        inflation = self._extra("pi_exp") if on_expectations else float(self.constants.get("anchor", 2.5))
        return self.rstar_median() + inflation

    def cash_rate(self) -> pd.Series:
        """Return the observed cash rate."""
        return self._extra("cash_rate")

    # --- Diagnostics ---

    def wedge_median(self) -> pd.Series:
        """Return the Australia-specific wedge over world r*."""
        return self._vector("wedge").median(axis=1)

    def jumps(self) -> pd.DataFrame:
        """Return the estimated jump at each break, with its 90% interval."""
        if "jumps" not in self.posterior:
            return pd.DataFrame(columns=["mean", "5%", "95%"])
        draws = np.asarray(self.posterior["jumps"].stack(sample=("chain", "draw")).values)  # noqa: PD013
        labels = [str(b) for b in self.constants.get("break_labels", range(draws.shape[0]))]
        return pd.DataFrame(
            {
                "mean": draws.mean(axis=1),
                "5%": np.quantile(draws, 0.05, axis=1),
                "95%": np.quantile(draws, 0.95, axis=1),
                "P(non-zero side)": np.maximum((draws > 0).mean(axis=1), (draws < 0).mean(axis=1)),
            },
            index=labels[: draws.shape[0]],
        )

    def end_break_check(self, window: int = 8) -> dict[str, float]:
        """Test whether the term premium has drifted at the end of the sample.

        The breaks are asserted, so a *new* one — a shift in Australia's wedge
        since the last named date — has nowhere to go but the term premium,
        where it would show up as a persistent departure from `mu_tp`. This
        reports that departure in units of the premium's own stationary sd.

        It is deliberately a legible number rather than a formal structural
        break test: a break announces itself, and the judgement of whether one
        has happened is the reader's.
        """
        tp = self.term_premium_posterior().median(axis=1)
        mu = float(self._scalar("mu_tp").mean())
        rho = float(self._scalar("rho_tp").mean())
        sigma = float(self._scalar("sigma_tp").mean())
        stationary_sd = sigma / np.sqrt(1.0 - rho**2)

        recent = tp.tail(window)
        deviation = float(recent.mean() - mu)
        return {
            "recent tp mean": float(recent.mean()),
            "mu_tp": mu,
            "deviation": deviation,
            "in stationary sds": deviation / stationary_sd if stationary_sd else float("nan"),
            "window": float(window),
        }

    def rstar_variation(self) -> dict[str, float]:
        """How much r* moves, against how much the imposed prior lets it move."""
        rstar = self.rstar_median()
        world = self.world_rstar()
        return {
            "sd of r*": float(rstar.std()),
            "sd of dr*": float(rstar.diff().dropna().std()),
            "sd of d world r*": float(world.diff().dropna().std()),
            "range of r*": float(rstar.max() - rstar.min()),
            "range of the wedge": float(self.wedge_median().max() - self.wedge_median().min()),
        }

    def variance_shares(self) -> dict[str, float]:
        """Split the observed real yield's variation between r* and the premium.

        The decomposition's headline claim is that the yield's low-frequency
        movement is r*. If the premium is carrying most of the variance, the
        model is attributing the decline in real yields to something transitory
        and the name on the state is wrong.
        """
        rstar = self.rstar_median()
        premium = self.term_premium_posterior().median(axis=1)
        total = self.real_yield().var()
        return {
            "var share, r*": float(rstar.var() / total),
            "var share, term premium": float(premium.var() / total),
            "corr(r*, world r*)": float(rstar.corr(self.world_rstar())),
        }

    def summary(self, var_names: list[str] | None = None) -> pd.DataFrame:
        """ArviZ summary for the scalar parameters."""
        if var_names is None:
            var_names = ["wedge_0", "mu_tp", "rho_tp", "sigma_tp"]
            if "nu_walk" in self.posterior:
                var_names.append("nu_walk")
            if "jumps" in self.posterior:
                var_names.append("jumps")
        summary = az.summary(self.trace, var_names=var_names)
        if not isinstance(summary, pd.DataFrame):
            raise TypeError("az.summary returned a Dataset — expected the DataFrame form")
        return summary


def load_results(
    output_dir: Path | str | None = None,
    prefix: str = "rstar",
) -> RStarResults:
    """Load a saved trace and observations from disk."""
    output_dir = Path(output_dir) if output_dir is not None else DEFAULT_OUTPUT_DIR

    trace = az.from_netcdf(str(output_dir / f"{prefix}_trace.nc"))
    with (output_dir / f"{prefix}_obs.pkl").open("rb") as f:
        saved = pickle.load(f)  # noqa: S301 — our own output, written by save_results

    return RStarResults(
        trace=trace,
        obs=saved["obs"],
        obs_index=saved["obs_index"],
        constants=saved.get("constants", {}),
        chart_obs=saved.get("chart_obs"),
    )
