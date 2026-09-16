"""Container and loader for rstar posterior draws, plus the derived series."""

import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import arviz as az
import numpy as np
import pandas as pd
import xarray as xr

from src.models.common.inflation_scale import long_run_expectations
from src.models.common.sources import footer_from_constants

DEFAULT_OUTPUT_DIR = Path(__file__).parent.parent.parent.parent / "model_outputs"
DEFAULT_CHART_BASE = Path(__file__).parent.parent.parent.parent / "charts"

# `premium_audit` compares era averages at each end of the sample rather than
# single quarters, so a shift has to persist to register. Four years at each
# end, and at least two years of overlap before the comparison means anything.
_AUDIT_ERA_QUARTERS = 16
_AUDIT_MIN_QUARTERS = 8


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

    @property
    def source_footer(self) -> str | None:
        """The "Built using: ..." line for this run's inputs, or None for an older run.

        Runs saved before `build_observations` began recording where its series
        came from carry no "sources" key, so the charting module falls back to
        its own constant.
        """
        return footer_from_constants(self.constants)

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

    def hdi(self, var_name: str, prob: float = 0.90) -> pd.DataFrame:
        """Return the highest-density interval of a vector latent, as lower/upper.

        Unlike the equal-tailed 5th/95th percentiles used elsewhere in the
        charts, this is the *narrowest* interval holding `prob` of the mass.
        The two coincide for a symmetric posterior and differ for a skewed one.
        """
        interval = az.hdi(self.posterior[var_name], hdi_prob=prob)[var_name]
        values = np.asarray(interval.values)
        return pd.DataFrame(
            {"lower": values[:, 0], "upper": values[:, 1]},
            index=self.obs_index,
        )

    def rstar_hdi(self, prob: float = 0.90) -> pd.DataFrame:
        """Return the highest-density interval of real r*."""
        return self.hdi("r_star", prob)

    def _inflation_term(self, scale: str) -> pd.Series | float:
        """Return the inflation term that puts r* on a nominal scale.

        Three, because they answer three questions:

        "expectations" (default) — TARGET-ANCHORED long-run expectations. The
            nominal neutral rate on the convention the RBA and CBA use, so the
            number is comparable with a published one. See
            `src/models/common/inflation_scale.py`.
        "target" — the flat 2.5% anchor. This package's convention before
            2026-09-16, kept so published numbers stay reproducible. It differs
            from the above by up to a point in the 1990s and barely at all
            after 2000.
        "actual" — the UNANCHORED expectations series, which moves with the
            cycle. Not a scale conversion: it gives the neutral rate for the
            inflation currently expected, and its gap to the anchored line is a
            de-anchoring measure in its own right.
        """
        if scale == "target":
            return float(self.constants.get("anchor", 2.5))
        if scale == "actual":
            # The model's own deflator, so it comes from the run's observations.
            return self._extra("pi_exp")
        if scale == "expectations":
            # Loaded fresh rather than read from the saved observations. A run
            # estimated before `pi_exp_lr` was carried would otherwise return an
            # all-NaN column and draw a blank nominal line with a "nan to nan"
            # header, which is worse than either erroring or being right. The
            # series is data either way, so re-reading it changes nothing except
            # that old traces keep working.
            return long_run_expectations(self.obs_index)
        raise ValueError(f"scale must be 'expectations', 'target' or 'actual', got {scale!r}")

    def nominal_rstar_hdi(self, prob: float = 0.90, *, scale: str = "expectations") -> pd.DataFrame:
        """Return the HDI of nominal r*.

        The inflation term is data, whichever of the three it is, so this is the
        real interval shifted rather than a wider one. Nominal r* carries
        exactly the uncertainty real r* does. See `_inflation_term`.
        """
        return self.rstar_hdi(prob).add(self._inflation_term(scale), axis=0)

    def term_premium_posterior(self) -> pd.DataFrame:
        """Return the term premium, everything in the yield that is not r*."""
        return self._vector("tp")

    def has_premium(self) -> bool:
        """Return whether this run estimates a term premium at all.

        False under `nominal_window`, where the premium was removed from the
        observable by the AOFM before the model saw it, so there is no `tp`,
        no `mu_tp` and nothing for the premium charts and checks to draw.
        """
        return "tp" in self.posterior

    def premium_is_pinned(self) -> bool:
        """Return whether `tp` was pinned to the AOFM's published premium.

        Changes what the audit chart MEANS: unpinned, the comparison is a test
        the model can fail; pinned, the two lines agree by construction and the
        estimated quantity is the spread between them.
        """
        return bool(self.constants.get("au_premium_anchor", 0.0))

    def aofm_premium(self) -> pd.Series:
        """Return the AOFM's published Australian term premium, if it loaded.

        Carried on every run, pinned to or not. The comparison against
        `term_premium_posterior` is the external check this package lacked: the
        model's premium is stationary about a constant by assertion, and this is
        the first Australian series that can say whether it should be. Note it
        is NOMINAL and the model's is real, so the honest comparison is of shape
        and change rather than level.
        """
        return self._extra("aofm_tp")

    def real_yield(self) -> pd.Series:
        """Return the observed long-end yield this run was estimated on.

        The indexed real 10-year yield by default. Under `nominal_window` it is
        the AOFM's deflated risk-neutral yield instead, which is the series the
        model actually fitted and therefore the one the variance decomposition
        and the fit charts want. `window_is_nominal` says which.
        """
        return pd.Series(self.obs["y"], index=self.obs_index)

    def window_is_nominal(self) -> bool:
        """Return whether window one was the risk-neutral nominal yield."""
        return bool(self.constants.get("nominal_window", 0.0))

    def indexed_yield(self) -> pd.Series:
        """Return the indexed real yield, whether or not it was estimated on.

        Under the default it is `real_yield()`. Under `nominal_window` it is
        carried as a chart extra so the two long-end measures can be compared.
        """
        if not self.window_is_nominal():
            return self.real_yield()
        return self._extra("y_indexed")

    def has_curve(self) -> bool:
        """Return whether this run used a medium maturity as a third window."""
        return "tp_m" in self.posterior

    def medium_premium_posterior(self) -> pd.DataFrame:
        """Return the premium on the medium maturity, `m - r* - k_m·g`.

        Its level is no better identified than the ten-year's, but the
        *difference* between them is: both are read off the same r*, so
        `tp_slope` is estimated even though neither mean is.
        """
        return self._vector("tp_m")

    def has_short_window(self) -> bool:
        """Return whether this run used the real cash rate as a second window."""
        return "g" in self.posterior

    def real_cash(self) -> pd.Series:
        """Return the observed real cash rate, the second window's observable."""
        if "r" not in self.obs:
            return pd.Series(np.nan, index=self.obs_index)
        return pd.Series(self.obs["r"], index=self.obs_index)

    def policy_gap_posterior(self) -> pd.DataFrame:
        """Return the policy gap, the real cash rate less r*.

        Positive is restrictive. This is an output of the model rather than an
        input to it: nothing here asks the gap to move output, which is the
        link `is_curve` cannot find in Australian data. It only asks the gap to
        be stationary about a mean.
        """
        return self._vector("g")

    def carried_posterior(self) -> pd.DataFrame:
        """Return `k·g`, the part of the long yield explained by policy stance.

        The one-window model set this to zero by construction, so its term
        premium was really `true tp + k·g`. That matters most through the QE
        window, when `g` was at its most negative in the sample.
        """
        k = self._scalar("k")
        return self.policy_gap_posterior().mul(k, axis=1)

    def term_premium_one_window(self) -> pd.DataFrame:
        """Return the premium the one-window model would have reported.

        `y - r*`, without the `k·g` correction, on this run's r*. The
        comparison against `term_premium_posterior` is the QE check: if the
        negative premium through 2020-22 survives the correction it was a
        finding, and if it does not it was the model reading a floored cash
        rate through a missing coefficient.
        """
        return self.term_premium_posterior() + self.carried_posterior()

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

    def real_mortgage_rate(self, *, discounted: bool = True) -> pd.Series:
        """Return an owner-occupier mortgage rate, deflated.

        Deflated by the same expectations series the policy gap uses, so the
        two stance measures differ only in which nominal rate they start from.

        `discounted` is what a borrower actually pays and is NaN before 2004Q2.
        The advertised rate covers the whole sample. They cannot be spliced:
        the discount runs 0.58 over 2004-07 and 1.40 over 2020-26, so the
        advertised rate increasingly overstates what anyone paid.
        """
        rate = self._extra("mortgage") if discounted else self._extra("mortgage_std")
        return rate - self._extra("pi_exp")

    def borrower_stance(self, *, discounted: bool = True) -> pd.Series:
        """Return the real mortgage rate less r*: the stance households faced.

        The household analogue of `business_rstar`. `policy_gap_posterior` is
        the *risk-free* stance, the real cash rate against a neutral rate read
        off a government bond, and the two are the same thing only when the
        cash rate summarises the price of credit. It stopped doing so after the
        GFC: the discounted mortgage spread to cash went from 1.24 over 2004-07
        to 2.99 over 2015-19 and 3.47 in 2020-21, before compressing to about
        2.45 now. Between 2014 and 2019 the cash rate fell 1.38 and the
        mortgage rate 0.70, so half the easing did not reach borrowers.

        That is not margin. Banks' term deposit rates moved from 1.68 *below*
        the cash rate to 0.39 above over the same window, a larger shift than
        the mortgage spread's, while the 90-day bill spread barely moved. The
        marginal funding dollar repriced when liquidity rules pushed banks from
        cheap offshore wholesale funding toward competing for retail deposits.

        The level is not comparable to `g`: this sets a risky borrowing rate
        against a risk-free neutral rate, so it carries a permanent credit
        spread and sits around 2 to 2.5 rather than near zero. Read the
        changes, not the level. On that reading it is remarkably flat, 2.46 in
        2004-07 against 2.10 in 2015-19, while the risk-free stance swung 2.11
        points over the same comparison.
        """
        return self.real_mortgage_rate(discounted=discounted) - self.rstar_median()

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

    def nominal_rstar(self, *, scale: str = "expectations") -> pd.Series:
        """Return the neutral *nominal* cash rate: r* plus expected inflation.

        The default, "expectations", uses TARGET-ANCHORED long-run expectations.
        That is the steady-state neutral rate on the convention the RBA and CBA
        both use, so it is the number to compare with a published one, and the
        comparison with the actual cash rate is the stance: above it policy is
        restrictive, below it expansionary.

        "target" uses the flat 2.5% anchor instead, which is what this method
        did by default before 2026-09-16. The two agree closely after 2000 and
        differ by up to a point through the 1990s re-anchoring.

        "actual" uses the unanchored expectations series, giving the neutral
        rate *for the inflation currently expected* rather than for the long
        run. Its gap to the anchored line is a de-anchoring measure in its own
        right, which is why `plot_stance` draws both.

        See `_inflation_term` for the full argument.
        """
        return self.rstar_median() + self._inflation_term(scale)

    def cash_rate(self) -> pd.Series:
        """Return the observed cash rate."""
        return self._extra("cash_rate")

    # --- Diagnostics ---

    def wedge_posterior(self) -> pd.DataFrame:
        """Return the Australia-specific wedge over world r*, draw by draw.

        The model's only latent state, and in this package's framing the whole
        local story: world r* is the imported price and the wedge is what
        Australia adds on top. Note that with `free_world_loading` on it is
        `r* - b_world·world`, not `r* - world`, so it cannot be read off the
        distance between the r* and world lines on the r* chart.
        """
        return self._vector("wedge")

    def wedge_median(self) -> pd.Series:
        """Return the posterior median of the wedge."""
        return self.wedge_posterior().median(axis=1)

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

        Returns an empty dict under `nominal_window`: with no premium there is
        nowhere for a late break to accumulate, so the test does not apply
        rather than passing.
        """
        if not self.has_premium():
            return {}
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
        total = self.real_yield().var()
        shares = {"var share, r*": float(rstar.var() / total)}
        # Under `nominal_window` the premium was taken out of the observable
        # before the model saw it, so there is no share to report: the residual
        # `sigma_rn` is the only other thing in the yield, and it is noise
        # rather than a component.
        if self.has_premium():
            premium = self.term_premium_posterior().median(axis=1)
            shares["var share, term premium"] = float(premium.var() / total)
        shares["corr(r*, world r*)"] = float(rstar.corr(self.world_rstar()))
        return shares

    def premium_audit(self) -> dict[str, float]:
        """Compare the model's fitted premium against the AOFM's published one.

        The check this package never had. `tp` is stationary about a constant by
        assertion, which forces any secular decline in the long yield into r*
        and therefore into the wedge. If the published Australian premium fell
        by roughly what the wedge fell, the assertion is doing the work and the
        wedge is partly the premium wearing another label.

        Levels are reported but the CHANGES correlation is the one to weigh: the
        model's premium is real and the AOFM's is nominal, so their levels
        differ by an inflation risk premium, and within the AOFM decomposition
        the premium and the risk-neutral yield both inherit the yield's
        downtrend, which inflates any level correlation.

        Returns an empty dict when there is no premium to audit or the AOFM
        series did not load.
        """
        if not self.has_premium():
            return {}
        aofm = self.aofm_premium().dropna()
        if aofm.empty:
            return {}
        model = self.term_premium_posterior().median(axis=1)
        wedge = self.wedge_median()
        joined = pd.concat({"model": model, "aofm": aofm, "wedge": wedge}, axis=1).dropna()
        if len(joined) < _AUDIT_MIN_QUARTERS:
            return {}
        changes = joined.diff().dropna()
        span = _AUDIT_ERA_QUARTERS
        move = joined.head(span).mean() - joined.tail(span).mean()
        return {
            "corr(model tp, aofm tp)": float(joined["model"].corr(joined["aofm"])),
            "corr changes": float(changes["model"].corr(changes["aofm"])),
            "corr(wedge, aofm tp)": float(joined["wedge"].corr(joined["aofm"])),
            "corr(wedge, aofm tp) changes": float(changes["wedge"].corr(changes["aofm"])),
            "sd, model tp": float(joined["model"].std()),
            "sd, aofm tp": float(joined["aofm"].std()),
            "fall in model tp": float(move["model"]),
            "fall in aofm tp": float(move["aofm"]),
            "fall in wedge": float(move["wedge"]),
            "n": float(len(joined)),
        }

    def summary(self, var_names: list[str] | None = None) -> pd.DataFrame:
        """ArviZ summary for the scalar parameters."""
        if var_names is None:
            # Built from what the run actually has: `nominal_window` drops the
            # three premium parameters and adds an observation error, and
            # `mu_spread` exists only under one of the two pins.
            var_names = ["wedge_0"]
            var_names += [
                name for name in ("mu_tp", "rho_tp", "sigma_tp", "mu_spread", "sigma_rn", "b_world")
                if name in self.posterior
            ]
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
    prefix: str = "rstar_bonds",
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
