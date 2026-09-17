"""Configuration for the TVP-VAR r* model.

    y_t      = c_t + B1_t y_{t-1} + B2_t y_{t-2} + e_t      y = [pi, growth, r]
    Theta_t  = Theta_{t-1} + sigma_q · eta_t                coefficients drift
    A e_t    = diag(exp(h_t/2)) u_t                         stochastic volatility
    h_{i,t}  = h_{i,t-1} + sigma_h,i · xi_{i,t}

    r*_t     = the H-quarter-ahead projection of r, from Theta_t held fixed

WHY THIS MODEL EXISTS, given the repo already has three r* routes. It is the
only one that needs neither an IS curve nor a term premium.

- No IS curve. `rstar_hlw`, `nairu` and the `dsge` family all independently
  found that the interest rate does not visibly move the output gap in
  Australian data, and `is_curve` shows the raw scatter does not recover the
  sign. Every structural route to r* runs through that link. A long-horizon
  forecast of the real policy rate does not: it asks what rate the economy has
  been settling toward, not what rate would close a gap.
- No term premium. `rstar_bonds` reads r* off a bond yield, so it has to split
  that yield into a natural rate and a premium, and the 2026-09-16 work showed
  how much rides on that split. Nothing here looks at a yield.

So its conditioning is different IN KIND from the other three, which is what
makes agreement with them informative and disagreement diagnostic.

WHAT SHIPS IS THE PUBLISHED LUBIK-MATTHES MODEL: three variables, quarterly
annualised changes, r* as the 20-quarter projection, no inflation conditioning,
`theta_0` centred on a training sample. Every departure from it is a flag, and
each one should be argued as a restriction rather than adopted because it makes
the answer look better. See MODEL_NOTES.

THE FIRST DIAGNOSTIC TO READ is `results.baseline_report()`, printed under "Does
the coefficient drift do anything?". It runs a constant-coefficient VAR through
the SAME projection and asks whether r* differs from it by more than r*'s own
credible band. If not, the several thousand drifting states are decoration and
the answer is a constant-coefficient VAR's answer.

THE PARAMETER THAT DECIDES THE ANSWER IS `sigma_q`. It is this model's
equivalent of `sigma_z` in `rstar_hlw` and `sigma_r` in `rstar_rba`: the
coefficients are a random walk whose innovation scale nothing in the data
strongly pins, and the long-horizon projection is a highly non-linear function
of those coefficients. At `sigma_q` near zero the VAR is constant and r* is a
flat line; too large and the coefficients chase noise and the projection swings
wildly. `results.prior_vs_posterior()` reports how far the posterior moved from
the prior, because a posterior sitting on its prior means the path is an
assumption. Sweep it with `--sigma-q` before believing any level.

THE SECOND THING TO DISTRUST is near-unit-root draws. r* is computed by
iterating the VAR forward, so a draw whose companion matrix has a root close to
or above one produces an explosive projection. Those draws are counted and
reported rather than silently dropped; see `results.stability_report()`.
"""

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

DEFAULT_OUTPUT_DIR = Path(__file__).parent.parent.parent.parent / "model_outputs"

# Which inflation and growth basis the VAR runs on.
#   "annual"    — four-quarter changes. Smoother, and it matches the horizon the
#                 policy rate is set against. The default.
#   "quarterly" — one-quarter changes, annualised. Closer to Lubik-Matthes, but
#                 on 135 Australian quarters the extra noise lands in the
#                 coefficients and therefore in the projection.
BASES = ("annual", "quarterly")

# How the real policy rate is built. Matches `rstar_bonds` so the two are
# comparable: the cash rate less a medium-horizon expectations measure.
DEFLATORS = ("expectations", "trimmed")

# Where the inflation anchor in the conditioned projection comes from.
ANCHOR_SOURCES = ("expectations", "target")

# What r* means: the resting point, or the H-quarter forecast.
RSTAR_DEFINITIONS = ("forward", "steady", "projection")
TARGET = 2.5


@dataclass
class ModelConfig:
    """Specification, sample, and the priors that decide how much drift is allowed."""

    # --- Sample ---
    # 1993Q1, matching every other model in the package: the start of inflation
    # targeting, before which the policy rule and the inflation process are a
    # different regime that a drifting-coefficient VAR would have to absorb as
    # drift rather than as the break it was.
    start: str | None = "1993Q1"
    end: str | None = None

    # --- The VAR ---
    # Two lags with three variables sampled best of everything tried. See
    # `include_commodities`.
    lags: int = 2
    # Quarterly, the canonical choice. Four-quarter changes embed an MA(3) in
    # every variable, so the VAR's residuals are autocorrelated by construction
    # and the drifting coefficients absorb that before describing anything real.
    # `--basis annual` is quieter to look at and buys the quiet by moving
    # structure into the states.
    basis: str = "quarterly"
    deflator: str = "expectations"

    # --- Optional extra variables, both OFF ---
    # Each addresses a real weakness and neither can be afforded: the state is
    # already 21 drifting coefficients per quarter on 132 quarters, and adding a
    # variable scales it quadratically.
    #
    # `--commodities` puts the RBA commodity price index FIRST, the most
    # exogenous series available: world prices for Australian exports with the
    # exchange rate stripped out. It is the standard remedy for the PRICE
    # PUZZLE, and it half works, moving the real rate's coefficient in the
    # inflation equation from +0.041 to +0.027. But four variables is 4,752
    # drifting states and the run did not converge (R-hat 1.060, ESS 101), so
    # the improvement cannot be verified.
    include_commodities: bool = False

    # `--twi` puts the trade-weighted index LAST, a fast financial variable that
    # responds within the quarter to everything including the cash rate. It is
    # the best candidate for a channel the three-variable model is blind to, and
    # the only driver here to come back right-signed in a majority of draws (60%
    # negative into inflation). But it gave R-hat 1.02, ESS 142 and 42
    # divergences, and near-unit-root draws dragged the steady state to 8.2%
    # nominal in 2006.
    include_twi: bool = False

    # --- The horizon that DEFINES r* ---
    # Five years, following CBA's description of the Lubik-Matthes approach:
    # "the neutral real rate is inferred from the model's five-year-ahead
    # projection for the real policy rate". Long enough that the cyclical
    # dynamics have largely died out, short enough that it is still a forecast
    # the VAR's estimated coefficients actually support rather than an
    # extrapolation to infinity.
    #
    # `results` also reports the unconditional mean, which is the H -> infinity
    # limit. The gap between the two is diagnostic: if they differ a lot the
    # projection has not converged and the horizon is doing real work.
    horizon_quarters: int = 20

    # --- WHAT r* MEANS ---
    #   "projection"  the H-quarter forecast, holding Theta_t fixed. The
    #                 published Lubik-Matthes definition and what ships.
    #   "steady"      (I - F)^-1 d, where the system comes to rest under its own
    #                 coefficients.
    #   "forward"     the mean of the conditioned projection over
    #                 `forward_lo_quarters` to `forward_hi_quarters`, a
    #                 model-implied 5y5y. Requires `anchor_projection`.
    #
    # NONE OF THE THREE ESCAPES THE SPECTRAL RADIUS, which runs about 0.98. The
    # projection is part nowcast, since 0.98^20 leaves two thirds of today in
    # the answer; the steady state divides by one minus that, so its median is
    # set by near-unit-root draws; the forward window sits between them. Choose
    # on what you mean, not on which number looks better, and read MODEL_NOTES
    # before quoting any of them.
    rstar_definition: str = "projection"

    # The window, in quarters, that "forward" averages over. 20 to 40 is years
    # five to ten, matching the 5y5y construction `2 x RNY10 - RNY5`.
    forward_lo_quarters: int = 20
    forward_hi_quarters: int = 40

    # --- Conditioning the projection on an inflation anchor ---
    # THE FIX FOR THE MODEL'S CENTRAL DEFECT. Left alone, the VAR's inflation
    # equation is an autoregression with no target in it, and on 1993-2026
    # Australian data that fits a near-unit root: from 2007Q4 the unconditioned
    # model projects inflation at 3.89% in five years and 4.14% in fifteen,
    # against a sample mean of 2.68%. It infers "inflation is a random walk"
    # from a sample generated by successful inflation targeting, because
    # targeting kept the deviations small.
    #
    # Everything else follows from that. With inflation permanently ~1.4pp
    # above its mean and an estimated long-run policy response of about +2.0
    # per pp, the projected real rate settles ~2.8pp high, which is why the
    # model reads the 2008 tightening as a rise in NEUTRAL rather than as
    # policy responding to inflation.
    #
    # The conditioning imposes the anchor the model cannot infer: inflation is
    # forced back toward it along the projection, and r* becomes "where the
    # rate settles once inflation is back at target". That is a RESTRICTION,
    # equivalent in kind to `rstar_bonds` asserting a stationary term premium,
    # and it should be quoted as one rather than presented as a finding.
    #
    # Growth is NOT conditioned. Neutral is properly defined at target
    # inflation AND output at potential, but potential growth would have to
    # come from `ystar`, and this model exists partly to avoid depending on the
    # output-gap machinery. So the conditioning is half of the definition.
    #
    # --- OFF SINCE 2026-09-17 ---
    # Read the four paragraphs above again as a description of the SHIPPED
    # model and the problem is plain: the conditioning pins inflation to an
    # anchor the VAR could not infer, and by h=20 the deviation is down to
    # 0.85^20 = 3.9% of itself, so across the whole projection inflation is an
    # imposed constant. What comes out is then not "the rate the economy has
    # been settling toward" but "the rate this VAR's policy equation prescribes
    # at the anchor", which is the object `rstar_rba` already estimates from a
    # rule written down on purpose. The independence that justifies this model
    # existing does not survive it.
    #
    # The near-unit-root inflation projection it was added to fix is REAL and is
    # not being denied. The point of turning this off is that it is a finding
    # about the canonical estimator on Australian data, which belongs in the
    # diagnostics, rather than a defect to be patched before publishing.
    # `--anchor-projection` restores it.
    anchor_projection: bool = False
    # "expectations" uses the target-anchored long-run expectations series, the
    # same one the package converts real to nominal with. "target" uses a flat
    # 2.5. They differ mainly before 2000, where expectations were genuinely
    # above target and a flat anchor would assert a re-anchoring that had not
    # happened yet.
    anchor_source: str = "expectations"
    # The return function: pi_{t+h} = anchor + (pi_t - anchor) * phi^h.
    # 0.85 gives a half-life of about 4.3 quarters and ~90% closed in 14, which
    # is the two-to-three years the RBA describes. 0 snaps to the anchor
    # immediately; values near 1 approach the unconditioned model.
    anchor_return: float = 0.85

    # --- The training sample the initial coefficients are centred on ---
    # `theta_0` gets a prior centred on a constant-coefficient OLS fit, which is
    # how Primiceri starts a TVP-VAR. Primiceri fits that OLS on a SEPARATE
    # training sample; until 2026-09-17 this model fitted it on the whole
    # estimation sample, which is the constant-coefficient answer the drift is
    # supposed to be tested against, used as the prior for the test.
    #
    # 40 quarters is ten years, leaving ~94 for estimation on the quarterly
    # basis. None restores the full-sample fit. The training rows stay IN the
    # likelihood either way: this changes where the prior is centred, not what
    # the model sees, so it is not a sample split.
    training_sample_quarters: int | None = 40

    # --- How fast the coefficients may drift ---
    # THE parameter. See the module docstring. A per-quarter innovation sd on
    # every VAR coefficient, shared across them so one number governs the whole
    # amount of time variation and can be swept.
    #
    # The prior is deliberately tight. Coefficients are O(1), so 0.02 per
    # quarter compounds to roughly 0.02*sqrt(135) = 0.23 of cumulative drift
    # over the sample, which is substantial time variation without being enough
    # for the VAR to refit itself every few years.
    sigma_q_prior: float = 0.02
    # Impose it instead of estimating, for the sweep. None estimates it.
    sigma_q: float | None = None

    # --- Stochastic volatility ---
    # Log variances are random walks. 0.2 per quarter on a log variance is a
    # ~22% move in the sd per quarter at one sd, which is loose enough to track
    # the GFC and COVID without the level wandering on its own.
    sigma_h_prior: float = 0.2
    # Initial log variance prior, centred on the log variance of the data in
    # levels, which `estimate` computes rather than hardcoding.
    h0_sigma: float = 1.0

    # --- Contemporaneous structure ---
    # A is lower triangular with a unit diagonal, so |det A| = 1 and the
    # likelihood needs no Jacobian term. Its three free elements are constant
    # over time. Primiceri lets them drift too; that is the least important part
    # of the time variation for r* and it doubles the state, so it is not here.
    a_sigma: float = 1.0

    # --- COVID ---
    # Not excluded. The whole point of stochastic volatility is that it can
    # absorb a variance spike without the coefficients having to move, so
    # excluding the quarters would remove the model's main opportunity to show
    # it works. `--exclude-covid` is available for the comparison.
    exclude_covid: bool = False
    covid_quarters: tuple[str, ...] = ("2020Q2", "2020Q3", "2020Q4", "2021Q3")

    # --- Blanking arbitrary quarters ---
    # The COVID switch above was always a general mechanism with a hardcoded
    # list: it blanks the chosen quarters, leaves them in the state so the
    # calendar and the drifting coefficients stay connected, and drops them from
    # the likelihood only. Nothing about it is specific to 2020.
    #
    # ADDED 2026-09-17 to test the GFC. The largest single-quarter move in r* in
    # the whole sample is 2008Q4, at -3.24pp, more than twice the next largest,
    # and it is larger than the fall in the real cash rate that quarter. That is
    # worth being able to excise without editing the config.
    #
    # Combines with `exclude_covid` rather than replacing it; see
    # `blanked_quarters`.
    exclude_quarters: tuple[str, ...] = ()

    output_dir: Path | None = None

    def __post_init__(self) -> None:
        """Validate the specification switches."""
        if self.basis not in BASES:
            raise ValueError(f"basis must be one of {BASES}, got {self.basis!r}")
        if self.deflator not in DEFLATORS:
            raise ValueError(f"deflator must be one of {DEFLATORS}, got {self.deflator!r}")
        if self.lags < 1:
            raise ValueError(f"lags must be at least 1, got {self.lags}")
        if self.horizon_quarters < 1:
            raise ValueError(f"horizon_quarters must be positive, got {self.horizon_quarters}")
        if self.training_sample_quarters is not None and self.training_sample_quarters <= self.lags:
            raise ValueError(
                f"training_sample_quarters must exceed lags, or the OLS fit has no rows: "
                f"got {self.training_sample_quarters} with {self.lags} lags",
            )
        if self.sigma_q_prior <= 0:
            raise ValueError(f"sigma_q_prior must be positive, got {self.sigma_q_prior}")
        if self.sigma_q is not None and self.sigma_q < 0:
            raise ValueError(f"sigma_q must be non-negative, got {self.sigma_q}")
        self._validate_quarters()
        if self.sigma_h_prior <= 0:
            raise ValueError(f"sigma_h_prior must be positive, got {self.sigma_h_prior}")
        self._validate_definitions()

    def _validate_quarters(self) -> None:
        """Check every quarter to be blanked parses.

        A typo would otherwise blank nothing and report success, so the run
        would silently be the unexcluded one and the comparison meaningless.
        """
        for quarter in (*self.covid_quarters, *self.exclude_quarters):
            try:
                pd.Period(quarter, "Q")
            except (ValueError, TypeError) as exc:
                raise ValueError(f"not a quarter: {quarter!r}") from exc

    def _validate_definitions(self) -> None:
        """Validate the switches that decide what r* means and what anchors it.

        Split out to keep each method's branch count readable; checked on every
        construction exactly as the rest are.
        """
        if self.rstar_definition not in RSTAR_DEFINITIONS:
            raise ValueError(
                f"rstar_definition must be one of {RSTAR_DEFINITIONS}, got {self.rstar_definition!r}",
            )
        if self.anchor_source not in ANCHOR_SOURCES:
            raise ValueError(f"anchor_source must be one of {ANCHOR_SOURCES}, got {self.anchor_source!r}")
        if self.forward_lo_quarters < 1:
            raise ValueError(f"forward_lo_quarters must be positive, got {self.forward_lo_quarters}")
        if self.forward_hi_quarters <= self.forward_lo_quarters:
            raise ValueError(
                f"forward_hi_quarters must exceed forward_lo_quarters, got "
                f"{self.forward_hi_quarters} and {self.forward_lo_quarters}",
            )
        # The "forward" definition is the whole reason the conditioning exists:
        # unconditioned, the window would average a projection whose inflation
        # is drifting to the VAR's own near-unit-root limit, which is the defect
        # `anchor_projection` was added to fix. Refuse rather than quietly
        # publish a 5y5y-equivalent defined at 3.15% inflation.
        if self.rstar_definition == "forward" and not self.anchor_projection:
            raise ValueError(
                "rstar_definition='forward' needs anchor_projection: without it the window "
                "averages a projection whose inflation drifts away from the anchor",
            )
        if not 0.0 <= self.anchor_return < 1.0:
            raise ValueError(
                f"anchor_return must be in [0, 1): 1 would never return to the anchor, "
                f"got {self.anchor_return}",
            )

    @property
    def blanked_quarters(self) -> tuple[str, ...]:
        """Every quarter dropped from the likelihood, COVID and explicit combined.

        One list, so a caller cannot blank one set and forget the other. Order
        preserved and duplicates removed, since asking for 2020Q2 explicitly
        alongside `--exclude-covid` should not blank it twice.
        """
        chosen = list(self.covid_quarters) if self.exclude_covid else []
        chosen.extend(self.exclude_quarters)
        return tuple(dict.fromkeys(chosen))

    @property
    def constants(self) -> dict[str, float]:
        """The imposed settings, recorded on the model for the run log."""
        return {
            "lags": float(self.lags),
            "horizon_quarters": float(self.horizon_quarters),
            "rstar_is_steady": float(self.rstar_definition == "steady"),
            "rstar_is_forward": float(self.rstar_definition == "forward"),
            "forward_lo_quarters": float(self.forward_lo_quarters),
            "forward_hi_quarters": float(self.forward_hi_quarters),
            "training_sample_quarters": (
                float(self.training_sample_quarters) if self.training_sample_quarters is not None else float("nan")
            ),
            "sigma_q_prior": self.sigma_q_prior,
            "sigma_q_imposed": float(self.sigma_q) if self.sigma_q is not None else float("nan"),
            "sigma_h_prior": self.sigma_h_prior,
            "basis_is_annual": float(self.basis == "annual"),
            "exclude_covid": float(self.exclude_covid),
            # The COUNT, not the list: `constants` is floats, and the run log and
            # chart footers carry the quarters themselves.
            "n_blanked_quarters": float(len(self.blanked_quarters)),
            "include_commodities": float(self.include_commodities),
            "include_twi": float(self.include_twi),
            "anchor_projection": float(self.anchor_projection),
            "anchor_return": self.anchor_return,
            "anchor_is_target": float(self.anchor_source == "target"),
        }
