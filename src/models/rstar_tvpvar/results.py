"""Container and loader for TVP-VAR draws, and the projection that defines r*.

r* IS COMPUTED HERE, not in the model. At each quarter the VAR's coefficients
are held fixed at their time-t values and the system is iterated forward H
quarters; r* is the real-rate component of that projection. That is the
Lubik-Matthes definition, and it is what CBA describe as "the model's
five-year-ahead projection for the real policy rate".

    z_t      = [y_t; y_{t-1}]                    companion state
    F_t      = [[B1_t, B2_t], [I, 0]]            companion matrix
    d_t      = [c_t; 0]
    z_{t+H}  = (sum_{j<H} F_t^j) d_t + F_t^H z_t
    r*_t     = the real-rate element of z_{t+H}

TWO THINGS THIS MAKES VISIBLE, and both are reported rather than smoothed over.

1. **Explosive draws.** The projection is a matrix power, so a draw whose
   companion matrix has a spectral radius at or above one produces a forecast
   that runs away. `stability_report` counts them. They are NOT dropped: a
   model that needs them dropped is a model whose coefficients drift too freely,
   and hiding that would hide the finding.
2. **Whether the horizon is doing the work.** `unconditional_mean` is the
   H -> infinity limit, `(I - B1 - B2)^{-1} c`. If it differs materially from
   the H-quarter projection, the projection has not converged and the reported
   r* depends on the horizon as much as on the data.

DRAWS ARE THINNED for this. Iterating a 6x6 companion matrix at every quarter
of every draw is a large batched operation, so `max_draws` subsamples. The
posterior median of a path is insensitive to thinning at these numbers; the
tails of the stability count are not, which is why that is reported as a share.
"""

import pickle
from dataclasses import dataclass
from pathlib import Path

import arviz as az
import numpy as np
import pandas as pd

from src.models.common.inflation_scale import get_unanchored_expectations
from src.models.common.results import PosteriorResults
from src.models.rstar_tvpvar.config import TARGET, ModelConfig
from src.models.rstar_tvpvar.observations import ols_fit, ordering, variable_index
from src.paths import CHARTS, MODEL_OUTPUTS

# The shipped default for the inflation conditioning, read from the config so
# the two cannot drift apart.
DEFAULT_ANCHORING = ModelConfig().anchor_projection

DEFAULT_OUTPUT_DIR = MODEL_OUTPUTS
DEFAULT_CHART_BASE = CHARTS

# Enough to pin a median and a 90% band without turning the projection into the
# slow part of the analysis.
DEFAULT_MAX_DRAWS = 1_000
# A companion matrix at or above this spectral radius is treated as explosive.
# Exactly 1.0 is the theoretical line; 0.999 avoids counting the numerically
# indistinguishable cases as stable.
STABILITY_LIMIT = 0.999


@dataclass(kw_only=True)
class TvpVarResults(PosteriorResults):
    """Posterior draws plus the data the VAR was fitted to."""

    data: np.ndarray
    frame: pd.DataFrame | None = None

    @property
    def lags(self) -> int:
        """Lag order the VAR was estimated with."""
        return int(self.constants.get("lags", 2))

    @property
    def horizon(self) -> int:
        """The horizon, in quarters, that defines r*."""
        return int(self.constants.get("horizon_quarters", 20))

    @property
    def order(self) -> tuple[str, ...]:
        """The VAR ordering this run used, inferred from the data's width.

        Read from the run's own constants, not from the column count: three
        variables could be the original set OR a four-variable run with two
        switches off, and the positions differ. A trace saved before either
        switch existed has neither key, and both default to on, so those runs
        must be read with the switches stated explicitly.
        """
        width = self.data.shape[1]
        # Prefer the run's own record, but only when it is CONSISTENT with the
        # data. A trace saved before these switches existed has neither key, and
        # defaulting them to on silently read a three-column run with a
        # five-variable ordering, putting the "real rate" at the index of lagged
        # inflation. So the width is the arbiter and the constants only
        # disambiguate between orderings of the same width.
        candidates = [
            ordering(include_commodities=c, include_twi=w)
            for c in (True, False) for w in (True, False)
        ]
        recorded = ordering(
            include_commodities=bool(self.constants.get("include_commodities", 1.0)),
            include_twi=bool(self.constants.get("include_twi", 1.0)),
        )
        if len(recorded) == width:
            return recorded
        matching = [c for c in candidates if len(c) == width]
        if not matching:
            raise ValueError(
                f"no VAR ordering has {width} variables; the saved run cannot be read",
            )
        return matching[0]

    @property
    def inflation_position(self) -> int:
        """Index of inflation in this run's ordering.

        NOT hardcoded. Adding commodity prices at the front moved inflation from
        0 to 1, and the conditioning overwrites this element: getting it wrong
        would condition commodity prices on the inflation anchor and say nothing.
        """
        return variable_index("inflation", self.order)

    @property
    def rate_position(self) -> int:
        """Index of the real policy rate in this run's ordering."""
        return variable_index("real_rate", self.order)

    @property
    def projection_index(self) -> pd.PeriodIndex:
        """The quarters the VAR has coefficients for: the sample less the lags."""
        return self.obs_index[self.lags:]

    def _theta_draws(self, max_draws: int = DEFAULT_MAX_DRAWS) -> np.ndarray:
        """Return coefficient draws as (draw, time, coefficient, equation)."""
        stacked = self.posterior["theta"].stack(sample=("chain", "draw"))  # noqa: PD013 — xarray, not pandas
        values = np.asarray(stacked.values)
        # xarray puts the stacked dimension last; move it to the front.
        theta = np.moveaxis(values, -1, 0)
        if theta.shape[0] > max_draws:
            step = theta.shape[0] // max_draws
            theta = theta[::step][:max_draws]
        return theta

    def _companion(self, theta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return the companion matrices and intercept vectors for every draw and quarter.

        Args:
            theta: (draw, time, coefficient, equation) coefficient draws.

        Returns:
            (F, d) with shapes (draw, time, m, m) and (draw, time, m), where
            m = lags * n_vars is the companion dimension.

        """
        n_draws, n_periods, _, n_vars = theta.shape
        lags = self.lags
        dim = lags * n_vars

        # theta[..., 1 + lag*n_vars + i, j] is the coefficient on y_{t-1-lag}[i]
        # in equation j, so transposing the last two axes puts it at [j, i] —
        # the orientation a companion block needs.
        blocks = np.swapaxes(theta[:, :, 1:, :], 2, 3)

        companion = np.zeros((n_draws, n_periods, dim, dim))
        companion[:, :, :n_vars, :] = blocks
        if lags > 1:
            rows = np.arange(n_vars, dim)
            companion[:, :, rows, rows - n_vars] = 1.0

        intercept = np.zeros((n_draws, n_periods, dim))
        intercept[:, :, :n_vars] = theta[:, :, 0, :]
        return companion, intercept

    def _initial_state(self) -> np.ndarray:
        """Return the companion state z_t = [y_t; y_{t-1}; ...] for each quarter."""
        lags = self.lags
        n_obs, n_vars = self.data.shape
        states = np.empty((n_obs - lags, lags * n_vars))
        for offset, t in enumerate(range(lags, n_obs)):
            states[offset] = np.concatenate([self.data[t - lag] for lag in range(lags)])
        return states

    def inflation_anchor(self) -> pd.Series:
        """Return the anchor inflation is conditioned back toward, per quarter."""
        if bool(self.constants.get("anchor_is_target", 0.0)):
            return pd.Series(TARGET, index=self.projection_index, dtype=float)
        return get_unanchored_expectations(self.projection_index)

    def steady_state_posterior(self, max_draws: int = DEFAULT_MAX_DRAWS) -> pd.DataFrame:
        """Return the VAR's resting real rate, (I - F)^-1 d, as (time x draw).

        The rate the system comes to rest at under its own coefficients, with no
        trace of where the economy actually was. That is the whole point: the
        H-quarter projection still carries half of today at twenty quarters,
        which is why r* chased the cash rate.

        Explosive draws return NaN rather than an exploded number. They are
        counted in `stability_report`, and the median is taken over the stable
        ones.
        """
        theta = self._theta_draws(max_draws)
        companion, intercept = self._companion(theta)
        return pd.DataFrame(self._steady(companion, intercept).T, index=self.projection_index)

    def _steady(self, companion: np.ndarray, intercept: np.ndarray) -> np.ndarray:
        """Solve (I - F)^-1 d for every draw and quarter, as (draw, time).

        Split out for the same reason as `_project`: the constant-coefficient
        baseline has to be the same estimator as r*, whichever definition the
        run shipped with.
        """
        n_draws, n_periods, dim, _ = companion.shape
        identity = np.eye(dim)
        radius = np.abs(np.linalg.eigvals(companion)).max(axis=-1)

        out = np.full((n_draws, n_periods), np.nan)
        for draw, period in np.argwhere(radius < STABILITY_LIMIT):
            try:
                solved = np.linalg.solve(identity - companion[draw, period], intercept[draw, period])
            except np.linalg.LinAlgError:
                continue
            out[draw, period] = solved[self.rate_position]
        return out

    @property
    def definition_is_steady(self) -> bool:
        """Whether r* is the resting point rather than the H-quarter projection."""
        return bool(self.constants.get("rstar_is_steady", 1.0))

    @property
    def definition_is_forward(self) -> bool:
        """Whether r* is the 5y5y-equivalent window average.

        Absent key means a trace from before the window existed, which is not a
        forward run: those were "steady" or "projection" and must keep reporting
        the way they did.
        """
        return bool(self.constants.get("rstar_is_forward", 0.0))

    @property
    def forward_window(self) -> tuple[int, int]:
        """The window, in quarters, that the forward definition averages over."""
        lo = int(self.constants.get("forward_lo_quarters", 20))
        hi = int(self.constants.get("forward_hi_quarters", 40))
        return lo, hi

    def rstar_posterior(
        self,
        max_draws: int = DEFAULT_MAX_DRAWS,
        *,
        conditioned: bool | None = None,
    ) -> pd.DataFrame:
        """Return the H-quarter-ahead real-rate projection, as (time x draw).

        The real rate is the LAST variable in the VAR ordering, so it is the
        element at `n_vars - 1` of the companion state's leading block.

        WITH `conditioned`, inflation is forced back toward its anchor along the
        projection instead of being left to the VAR's own dynamics. That matters
        because those dynamics do not return: unconditioned, the model projects
        2007Q4 inflation at 3.89% five years out and 4.14% fifteen years out,
        against a sample mean of 2.68. Everything the model then says about
        neutral in 2008 follows from believing that.

        The conditioning is HARD: the inflation element of the state is
        overwritten at each step with `anchor + (pi_t - anchor) * phi^h`, and the
        companion shift carries the overwritten value into the next step's lag,
        so the whole projected path is internally consistent. That is the crude
        form of a conditional forecast, chosen because it is transparent: a
        shock-based version would distribute the conditioning across equations
        in a way that is harder to state.

        `conditioned` defaults to whatever the run was configured with.
        """
        if self.definition_is_steady:
            # The steady state uses no initial condition, so the inflation
            # conditioning has nothing to act on and is silently irrelevant here.
            return self.steady_state_posterior(max_draws)

        if conditioned is None:
            # Absent key means the run predates the flag. Fall back to the CURRENT
            # default rather than to off: the conditioning is post-processing and
            # does not depend on how the model was estimated, so an old trace
            # should report the same way a new one does. Falling back to off
            # would let a stale trace silently publish the unconditioned path
            # while the config said otherwise.
            conditioned = bool(self.constants.get("anchor_projection", float(DEFAULT_ANCHORING)))

        theta = self._theta_draws(max_draws)
        companion, intercept = self._companion(theta)
        return pd.DataFrame(self._project(companion, intercept, conditioned).T, index=self.projection_index)

    def _project(
        self,
        companion: np.ndarray,
        intercept: np.ndarray,
        conditioned: bool,
    ) -> np.ndarray:
        """Iterate the VAR forward and return the real-rate projection as (draw, time).

        Split out of `rstar_posterior` so that `constant_coefficient_rstar` runs
        through the SAME arithmetic. A baseline computed by a second
        implementation would differ from r* for reasons that have nothing to do
        with coefficient drift, which is the one thing the comparison is for.
        """
        state = self._initial_state()[None, :, :].repeat(companion.shape[0], axis=0)

        if conditioned:
            phi = float(self.constants.get("anchor_return", 0.85))
            anchor = self.inflation_anchor().to_numpy(dtype=float)
            # Deviation of each quarter's actual inflation from its anchor.
            pi_pos = self.inflation_position
            start_gap = self.data[self.lags:, pi_pos] - anchor

        # The forward definition averages the projected rate over a WINDOW
        # rather than reading it at a point, because a 5y5y is the average rate
        # over years five to ten, not the rate at year five. Iterate to the top
        # of the window and accumulate the rate inside it.
        lo, hi = self.forward_window
        last_step = hi if self.definition_is_forward else self.horizon
        collected: list[np.ndarray] = []

        # z <- d + F z, H times. Batched over (draw, time).
        for step in range(1, last_step + 1):
            state = intercept + np.einsum("dtij,dtj->dti", companion, state)
            if conditioned:
                state[:, :, pi_pos] = anchor + start_gap * phi**step
            if self.definition_is_forward and lo <= step <= hi:
                collected.append(state[:, :, self.rate_position])

        return np.mean(collected, axis=0) if self.definition_is_forward else state[:, :, self.rate_position]

    def rstar_median(
        self, max_draws: int = DEFAULT_MAX_DRAWS, *, conditioned: bool | None = None,
    ) -> pd.Series:
        """Return the posterior median r* path."""
        return self.rstar_posterior(max_draws, conditioned=conditioned).median(axis=1)

    def rstar_hdi(
        self, prob: float = 0.90, max_draws: int = DEFAULT_MAX_DRAWS, *, conditioned: bool | None = None,
    ) -> pd.DataFrame:
        """Return a credible band for r*, as lower/upper columns.

        Quantiles rather than an arviz HDI: the projection can be heavy-tailed
        when some draws are near-explosive, and a highest-density interval on a
        heavy-tailed sample is both unstable and misleading about the centre.
        """
        draws = self.rstar_posterior(max_draws, conditioned=conditioned)
        lower = (1.0 - prob) / 2.0
        return pd.DataFrame({
            "lower": draws.quantile(lower, axis=1),
            "upper": draws.quantile(1.0 - lower, axis=1),
        })

    def constant_coefficient_rstar(self, *, conditioned: bool | None = None) -> pd.Series:
        """Return r* from a VAR whose coefficients never drift: the null to beat.

        THE POINT OF THIS MODEL is that the coefficients move. This is what the
        same estimator says when they do not: one constant-coefficient OLS fit
        over the whole sample, put through `_project` (or `_steady`) exactly as
        the posterior draws are. It still moves quarter to quarter, because the
        projection starts from each quarter's actual state, so the difference
        between this line and r* is the part of r* that the DRIFT is responsible
        for, and nothing else.

        This is a point path with no parameter uncertainty, so it is not what a
        Bayesian constant-coefficient VAR would report; `ensemble.py` samples
        `sigma_q = 0` for that. Comparing a point against the TVP band errs in
        the safe direction: if this line sits inside the band, the drift is not
        detectable, and no wider baseline would change that.

        The OLS is deliberately the FULL-sample fit even when the model's
        `theta_0` prior used a training sample. The prior is about where the
        sampler starts; the baseline is about the best a constant-coefficient
        VAR can do, and handicapping it to ten years of data would flatter the
        TVP model.
        """
        coefficients, _ = ols_fit(self.data, self.lags, None)
        n_periods = len(self.projection_index)
        # One "draw", the same coefficients at every quarter: that is the null.
        theta = np.broadcast_to(coefficients, (1, n_periods, *coefficients.shape))
        companion, intercept = self._companion(np.ascontiguousarray(theta))
        if self.definition_is_steady:
            path = self._steady(companion, intercept)
        else:
            if conditioned is None:
                conditioned = bool(self.constants.get("anchor_projection", float(DEFAULT_ANCHORING)))
            path = self._project(companion, intercept, conditioned)
        return pd.Series(path[0], index=self.projection_index)

    def baseline_report(self, max_draws: int = DEFAULT_MAX_DRAWS, prob: float = 0.90) -> dict[str, float]:
        """Report whether coefficient drift changes r* by more than the band.

        `share inside the band` is the headline. If the constant-coefficient
        line sits inside the TVP model's own credible interval in most quarters,
        the time variation is not doing visible work and the model is a
        constant-coefficient VAR with several thousand extra states.
        """
        baseline = self.constant_coefficient_rstar()
        median = self.rstar_median(max_draws)
        band = self.rstar_hdi(prob, max_draws)
        aligned = pd.concat(
            {"base": baseline, "median": median, "lo": band["lower"], "hi": band["upper"]},
            axis=1,
        ).dropna()
        if aligned.empty:
            return {}
        inside = (aligned["base"] >= aligned["lo"]) & (aligned["base"] <= aligned["hi"])
        gap = (aligned["median"] - aligned["base"]).abs()
        return {
            "share of quarters inside the band": float(inside.mean()),
            "mean absolute gap to r*": float(gap.mean()),
            "max absolute gap to r*": float(gap.max()),
            "correlation with r*": float(aligned["median"].corr(aligned["base"])),
            "baseline latest": float(aligned["base"].iloc[-1]),
            "baseline range": float(aligned["base"].max() - aligned["base"].min()),
        }

    def unconditional_mean(self, max_draws: int = DEFAULT_MAX_DRAWS) -> pd.Series:
        """Return the median resting real rate, the H -> infinity limit.

        Kept as a name because the charts and diagnostics use it to compare the
        two definitions against each other. It is the median of
        `steady_state_posterior`, which is now also the default r*.
        """
        return self.steady_state_posterior(max_draws).median(axis=1)

    def spectral_radius(self, max_draws: int = DEFAULT_MAX_DRAWS) -> pd.DataFrame:
        """Return the companion matrix's largest eigenvalue modulus, as (time x draw).

        The single number that decides what this model can say. At a radius near
        one the H-quarter projection still carries most of today's state, so r*
        is part nowcast; above one it runs away. `stability_report` summarises
        this, and the per-quarter path is what shows WHERE the persistence sits,
        which is how the 2000-07 era was identified as worse than COVID.
        """
        theta = self._theta_draws(max_draws)
        companion, _ = self._companion(theta)
        radius = np.abs(np.linalg.eigvals(companion)).max(axis=-1)
        return pd.DataFrame(radius.T, index=self.projection_index)

    def stability_report(self, max_draws: int = DEFAULT_MAX_DRAWS) -> dict[str, float]:
        """Report how often the fitted VAR is explosive.

        The projection is a matrix power, so this is the check that decides
        whether the r* path means anything. A few per cent is ordinary for a
        drifting-coefficient VAR on a persistent series; a large share means the
        coefficients are free enough to leave the stationary region routinely
        and the long-horizon forecast is not a forecast.
        """
        radius = self.spectral_radius(max_draws).to_numpy().T
        explosive = radius >= STABILITY_LIMIT
        worst_by_period = explosive.mean(axis=0)
        return {
            "share of draw-quarters explosive": float(explosive.mean()),
            "median spectral radius": float(np.median(radius)),
            "95th pct spectral radius": float(np.quantile(radius, 0.95)),
            "worst quarter's explosive share": float(worst_by_period.max()),
            "draws used": float(radius.shape[0]),
        }

    def prior_vs_posterior(self) -> dict[str, float]:
        """How far the drift scale moved from its prior.

        `sigma_q` decides how much the coefficients may change, and therefore
        how much r* moves. If its posterior sits on its prior, the r* path is an
        assumption rather than a finding — the same test that removed
        `rstar_hlw` from `rstar_summary`, where `sigma_z` posteriored at 0.0657
        against a prior median of 0.0674.

        Returns an empty dict when `sigma_q` was imposed rather than estimated.
        """
        if "sigma_q" not in self.posterior:
            return {}
        draws = np.asarray(self.posterior["sigma_q"].values).ravel()
        prior_sigma = float(self.constants.get("sigma_q_prior", 0.02))
        # Median of a HalfNormal(sigma) is sigma * sqrt(2) * erfinv(0.5).
        prior_median = prior_sigma * 0.6744897501960817
        posterior_median = float(np.median(draws))
        return {
            "sigma_q posterior median": posterior_median,
            "sigma_q prior median": prior_median,
            "ratio, posterior / prior": posterior_median / prior_median if prior_median else float("nan"),
            "posterior sd": float(draws.std()),
            "prior sd": prior_sigma * np.sqrt(1.0 - 2.0 / np.pi),
        }

    def observed(self, name: str) -> pd.Series:
        """Return one of the VAR's input series on the full sample index."""
        if self.frame is None or name not in self.frame:
            return pd.Series(np.nan, index=self.obs_index)
        return self.frame[name]

    def summary(self, var_names: list[str] | None = None) -> pd.DataFrame:
        """ArviZ summary for the scalar parameters, skipping the drifting states."""
        if var_names is None:
            var_names = [n for n in ("sigma_q", "sigma_h", "a_free", "h_0") if n in self.posterior]
        summary = az.summary(self.trace, var_names=var_names)
        if not isinstance(summary, pd.DataFrame):
            raise TypeError("az.summary returned a Dataset — expected the DataFrame form")
        return summary


def load_results(
    output_dir: Path | str | None = None,
    prefix: str = "rstar_tvpvar",
) -> TvpVarResults:
    """Load a saved TVP-VAR run."""
    directory = Path(output_dir) if output_dir is not None else DEFAULT_OUTPUT_DIR
    trace = az.from_netcdf(str(directory / f"{prefix}_trace.nc"))
    with (directory / f"{prefix}_obs.pkl").open("rb") as f:
        saved = pickle.load(f)  # noqa: S301 — our own file, written by save_results
    return TvpVarResults(
        trace=trace,
        data=saved["data"],
        obs_index=saved["obs_index"],
        constants=saved.get("constants", {}),
        frame=saved.get("frame"),
    )
