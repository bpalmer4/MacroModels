"""Configuration for the rstar_invert model.

TWO AXES, BOTH GAPS. The chart this model is about has the rate gap on the x
axis and the output gap on the y axis, and the line through it passes through
the ORIGIN: at a zero rate gap the economy sits at potential, which is what
"neutral" means. So there is no constant term. The only intercept in the story
is r* itself, the amount subtracted from the real cash rate to turn the rate
axis into a rate-gap axis.

    (1)  rstar_t = a slow-moving series
    (2)  x_t     = is_slope . (rbar_t - rstarbar_t) + e_t

`x_t` is the output gap and `r_t` the real cash rate. BOTH ARE GIVEN. The line
is asserted: negative slope, through the origin. r* is asserted to exist and to
move slowly. What the model then reports is the r* path consistent with the
observed rates and gaps under that line.

WHAT IS ACTUALLY OPEN. One thing: HOW SLOW r* IS, which is `sigma_rstar`.
Nothing in the data will settle it. The fit improves monotonically the faster
r* is allowed to move, with no interior optimum, because a faster r* can slide
each point onto the line. So the number is a judgement and the answer should be
quoted as a range across defensible values, never as a point.

KNOWN PROPERTY OF THIS PARAMETERISATION, which must be stated when quoting any
result. With the prior on r* in RATE units, the slope enters twice: once as the
steepness and again multiplying r* to position the line. The line's vertical
freedom is therefore |is_slope| x (r*'s freedom), so a BIGGER SLOPE BUYS A MORE
MOVABLE LINE and the likelihood has an incentive to inflate it. Measured: at
sigma_rstar 0.02 the slope came back -0.032, at 0.10 it came back -0.383, with
almost the same fit. Writing the model with the prior on the line's height
instead removes that incentive and gives -0.092, close to the `is_curve`
bench's independent -0.108, but it puts the prior on a quantity nobody can
judge the plausibility of. Both are defensible; neither is neutral.

WHAT r* IS NOT ANCHORED TO. Nothing. HLW ties r* to trend growth, `rstar_bonds`
to a world real rate, `rstar_rba` to a policy rule. Each of those disciplines
r* so it cannot wander to absorb residuals, and each imports the answer it then
reports. This model imports none of them, and pays for it: r* has no second job
and its level is barely identified.
"""

from dataclasses import dataclass
from pathlib import Path

from src.paths import CHARTS, MODEL_OUTPUTS

DEFAULT_OUTPUT_DIR = MODEL_OUTPUTS
DEFAULT_CHART_DIR = CHARTS

# An IS curve has a negative slope. At or above this it is a sign error.
MAX_SLOPE = 0.0

# Two lags share one free weight; three share a Dirichlet. The cap is three
# because that is where it has been looked at, not because four would break.
#
# THE OLD NOTE HERE SAID three lags "would not be identified anyway" because
# the regressors are collinear. That reason is wrong and the measurement is
# recorded so it is not reinstated: on 127 usable quarters the real cash rate
# at t-1, t-4 and t-7 correlates 0.73 to 0.87, but the design's condition
# number is 5.4 and the pairwise differences have sd 0.93 to 1.32pp. There is
# independent variation to read weights off.
#
# THE REAL CONSTRAINT is that the weights reach the likelihood only multiplied
# by the slope: what separates two weight vectors is is_slope x (the difference
# between the lagged rates). At the 4/8 default that is 0.38 x 0.93 = 0.35pp
# against sigma_e 0.24, readable. At 1/2 it is 0.008 x 0.9 = 0.007pp against
# sigma_e 0.43, invisible, and w came back at [0.119, 0.851]. So the weights
# are identified only where the slope survives.
MAX_LAGS = 3

# Two lags are the case that keeps the scalar Beta weight `lag_weight`; three
# switch to the Dirichlet vector `lag_weights`. Named so the branch on it reads
# as a specification choice rather than an arithmetic coincidence.
PAIR_LAGS = 2

# HOW SLOW r* IS, expressed as a shape rather than only a scale.
#   "walk"      a random walk, one innovation per quarter, scaled by
#               sigma_rstar. The usual choice, and the least disciplined: it
#               gives r* one free value per observation, so the line can be
#               repositioned for every point and the fit cannot fail.
#   "linear"    a straight drift. Two numbers for the whole sample, so r* moves
#               over time but the points cannot slide to meet the line.
#   "constant"  r* fixed. Not the story (r* is meant to move), but the
#               reference case: it shows what the slope can do unaided.
RSTAR_FORMS = ("walk", "linear", "constant")


@dataclass
class ModelConfig:
    """The specification, and the numbers it conditions on."""

    # --- The asserted slope ---
    #
    # UNITS. Per cent of potential output per percentage point of RATE GAP. A
    # LEVEL relationship, not an impact coefficient, because this equation has
    # no gap persistence. NOT comparable with HLW's `a_r` (-0.04 in
    # `rstar_hlw`), which is an impact coefficient in an equation that has
    # persistence. IS comparable with the `is_curve` bench's slopes, which are
    # plain regressions of the same form.
    #
    # A PRIOR, NOT A CONSTANT, but truncated negative because a positive slope
    # is not a weak IS curve, it is no IS curve: New Keynesian transmission
    # requires a restrictive stance to contract output. A posterior pressed
    # against that bound is a REJECTION at this specification, not a
    # measurement of a small negative slope.
    #
    # WHY -0.30. Deliberately stronger than anything measured here, because it
    # is the FAVOURABLE case: r* movement scales as 1/|is_slope|, so a big
    # slope keeps the implied neutral rate sane and a small one blows it up.
    #
    # WHAT HAS BEEN MEASURED. Single lags 1 to 5 gave -0.007 to -0.034, all but
    # lag 5 pressed against the bound. A free-intercept OLS on the raw weighted
    # lagged rate gives -0.015. The `is_curve` bench reaches -0.108 only on the
    # sample that drops 2008Q4-2021Q3.
    is_slope_mu: float = -0.30
    # At sigma 0.10 the truncation sits three sd out and costs nothing. The
    # prior also puts almost no mass on |slope| < 0.1, which is where every
    # measurement in this repo lives. That is deliberate.
    is_slope_sigma: float = 0.10

    # --- How slow r* is: THE ONE OPEN QUESTION ---
    rstar_form: str = "walk"

    # The quarterly sd of r*'s innovation, in PERCENTAGE POINTS, used when
    # `rstar_form` is "walk".
    #
    # NOTHING MEASURES THIS, and the fit improves monotonically as it rises, so
    # the data cannot choose. Sweep it and quote a range.
    #
    # 0.15 WAS TRIED ON 2026-09-16 AND REVERTED. Recorded because the argument
    # for it was sound and the result was not, which is worth knowing before
    # anyone makes it again.
    #
    # The case: "is this too smooth" is not answered by the flat-line cliff
    # between 0.05 and 0.10, which only tests whether r* moves AT ALL. The
    # reasonableness test is how fast r* moves against how fast the other models
    # let neutral move, measured as the sd of its quarterly change:
    #
    #     rstar_bonds   0.155      off asset prices
    #     rstar_tvpvar  0.153      off a macro VAR
    #     rstar_rba     0.078      but its sigma_r does the same job, so this
    #                              is not independent evidence
    #
    #     this model    0.094 at sigma_rstar 0.10
    #                   0.135 at 0.15   <- closest to the two independent ones
    #                   0.245 at 0.30
    #
    # Two models built on entirely different data independently put neutral's
    # quarterly volatility near 0.15, and 0.15 here reproduced it (0.135 against
    # 0.094 at 0.10). Sampling was pristine: R-hat 1.00, ESS 4,987, zero
    # divergences. The slope barely moved, -0.383 to -0.389, so the change was
    # cheap where it would have mattered.
    #
    # WHY IT WAS REVERTED ANYWAY: the endpoint. r* latest went 1.53 -> 2.05 real,
    # i.e. 4.06 -> 4.59 NOMINAL, above the 4.35 cash rate and the highest line on
    # the summary chart by half a point. The sample mean barely moved (1.56 ->
    # 1.57), so essentially the whole change landed on the last few quarters —
    # the least reliable point of a random walk whose speed had just been raised.
    #
    # The lesson is about the test, not the number: matching another model's
    # volatility is a real check, and it is not sufficient. A setting can pass
    # on volatility and fail on the level it implies.
    sigma_rstar: float = 0.10

    # Estimate `sigma_rstar` rather than asserting it. OFF by default: with one
    # observation equation and a free walk the likelihood is nearly flat in the
    # trade-off between the walk and the residual, so expect the prior back.
    free_sigma_rstar: bool = False
    sigma_rstar_prior: float = 0.50

    # The starting level of r*, REAL, per cent. Wide: the level is barely
    # identified, and a tight prior here would assume the answer.
    rstar_0_mu: float = 1.50
    rstar_0_sigma: float = 2.00
    # Prior sd on the per-quarter drift when `rstar_form` is "linear", in
    # percentage points per quarter. 0.03 over 134 quarters allows r* to travel
    # about 4 points across the sample, which is generous but not absurd.
    rstar_trend_sigma: float = 0.03

    # --- The lags ---
    # A WEIGHTED TWO-POINT DISTRIBUTED LAG at 4 and 8 quarters:
    #
    #     rbar_t = w . r_{t-4} + (1-w) . r_{t-8}
    #
    # DISTRIBUTED, because the response to policy is spread over time and a
    # single lag measures only a slice of it.
    #
    # LONG, because distance from t is a partial fix for simultaneity. The RBA
    # reacts to conditions with a short lag while output responds to rates with
    # a long one, so a regressor further back carries less of the reaction
    # function. That is very likely why the single-lag sweep found the slope
    # strengthening monotonically from -0.007 at lag 1 to -0.034 at lag 5. It
    # is a weak fix: the real cash rate is persistent, so r_{t-8} stays
    # correlated with recent rates that ARE reacting.
    #
    # KEPT BECAUSE IT WORKS: the slope comes off its sign bound at this pair,
    # which it does not at any single short lag.
    #
    # AND BECAUSE THE MODE IS REAL, which was checked rather than assumed
    # (2026-09-14). At (1, 4, 7) the chains do not mix: they spread across a
    # ridge between the slope and r*'s amplitude, -0.030 to -0.319 with r*'s sd
    # running 0.16 to 1.47, and 227 of 258 divergences land at the steep end.
    # This pair shows none of it. Its four chains agree to a between-chain sd
    # of 0.0011 against a pooled posterior sd of 0.0350, r*'s amplitude agrees
    # to three decimals (1.615 to 1.617), and an independent seed returns
    # -0.3836 against -0.3827. Across 16,000 draws from the two seeds nothing
    # gets closer to zero than -0.20, so the flat-slope basin carries no
    # posterior mass here. Not a stuck chain.
    #
    # WHAT THAT DOES NOT SETTLE: the answer. Changing the lags moves the slope
    # by hundredths; changing sigma_rstar moves it twentyfold, at this pair as
    # much as anywhere. See MODEL_NOTES, "The ridge between the slope and r*'s
    # amplitude".
    #
    # COMPARABLE WITH LAG 6 ANYWAY, which is what `is_curve` and `rstar_hlw`
    # now use. The estimated weight is 0.435 on lag 4, so the effective mean
    # lag is 0.435x4 + 0.565x8 = 6.3 quarters. The pair is a distributed lag
    # centred on six, not a different horizon.
    #
    # The weight itself buys little: 0.5 sits inside its interval, and when the
    # slope is not inflated that interval widens to [0.099, 0.608], close to
    # the Beta prior. Fixing w at 0.5 (`--fix-lag-weight`) costs almost
    # nothing.
    #
    # NOT BECAUSE THE TWO LAGS ARE COLLINEAR, which this comment used to say.
    # They correlate 0.805, the design's condition number is 3.0 and their
    # difference has sd 1.127pp, all mild. The width is the slope again: what
    # separates two weights is |is_slope| x (the difference between the lagged
    # rates), so at -0.383 the signal is 0.43pp against sigma_e 0.239 and the
    # interval is tight at [0.328, 0.527], while at -0.092 it is 0.10pp against
    # 0.225 and the interval opens up. One story, both widths.
    #
    # NOT YET TESTED: whether the response ACCUMULATES across many lags. These
    # weights sum to one, so this measures the response to a SUSTAINED stance,
    # the same object a single lag measures. Testing accumulation needs weights
    # that do not sum to one.
    rate_lags: tuple[int, ...] = (4, 8)

    # The weight on the FIRST lag; the second takes 1 - w by construction.
    # ESTIMATED by default. It has come back near 0.44 with a fairly tight
    # interval when the slope was inflated, and near 0.37 with a very wide one
    # when it was not, so how much the data can see of it depends on the rest
    # of the specification. 0.5 sits inside both intervals.
    lag_weight_free: bool = True
    lag_weight_a: float = 2.0
    lag_weight_b: float = 2.0
    lag_weight: float = 0.5

    # THREE LAGS instead of two: the weights become a Dirichlet, stored as the
    # vector `lag_weights`. The concentration is the same 2.0 on every stick,
    # which for two lags IS Beta(2, 2), so the two-lag default is unchanged in
    # distribution and its saved traces stay comparable.
    #
    # WEIGHTS THAT SUM TO ONE, deliberately, as with the pair. The stance is
    # then the response to a SUSTAINED level, and a constant shift in r* shifts
    # the stance one for one, which is what makes r*'s level mean anything.
    # Three unconstrained coefficients would rescale r* silently.
    #
    # When `lag_weight_free` is off, three lags share equally (1/3 each).
    lag_weight_conc: float = 2.0

    # --- Sample ---
    # Inflation targeting. Before it the cash rate is not set by a reaction
    # function and the gap does not mean the same thing.
    start: str = "1993Q1"
    end: str | None = None

    # Drop 2008Q4-2021Q3 as well as the lockdowns. OFF by default: the
    # `is_curve` notes are explicit that this cut manufactures a negative slope
    # out of two clusters that individually disagree.
    exclude_qe: bool = False

    # --- Priors on the rest ---
    # The residual: everything else that moves output, quarter by quarter.
    # `rstar_hlw` puts the equivalent at 0.70 across all eight of its
    # specifications, and the gap's own sd is 0.42. This reaches past both.
    #
    # NOTE there is no constant term beside it. The line goes through the
    # origin because that is what neutral means, so `e_t` is assumed to average
    # zero. If the measured gap has a constant bias it has nowhere to go, and
    # the level of r* absorbs it: the gap averages +0.036, which at a slope of
    # -0.09 is 0.4pp of r*.
    sigma_e_prior: float = 2.0

    # Prefix of the completed `ystar_ustar` run supplying the output gap.
    # THE GAP IS A MODEL OUTPUT taken as data, so this package inherits that
    # model's conditioning, including its imposed sigma_okun of 0.20. Roughly
    # half that gap is c x (inflation - 2.5) with c = 0.275, so the y axis is
    # partly a rescaling of inflation.
    gap_prefix: str = "ystar_ustar"

    output_dir: Path | None = None

    def __post_init__(self) -> None:
        """Validate the specification."""
        if self.is_slope_mu >= MAX_SLOPE:
            raise ValueError(
                f"is_slope_mu must be negative for this to be an IS curve, got {self.is_slope_mu}",
            )
        if self.is_slope_sigma <= 0:
            raise ValueError(f"is_slope_sigma must be positive, got {self.is_slope_sigma}")
        if self.rstar_form not in RSTAR_FORMS:
            raise ValueError(f"rstar_form must be one of {RSTAR_FORMS}, got {self.rstar_form!r}")
        if self.sigma_rstar <= 0:
            raise ValueError(f"sigma_rstar must be positive, got {self.sigma_rstar}")
        if self.sigma_rstar_prior <= 0:
            raise ValueError(f"sigma_rstar_prior must be positive, got {self.sigma_rstar_prior}")
        self._validate_lags()

    def _validate_lags(self) -> None:
        """Validate the lag structure and its weight."""
        if not self.rate_lags:
            raise ValueError("rate_lags must name at least one lag")
        if len(self.rate_lags) > MAX_LAGS:
            raise ValueError(
                f"rate_lags supports at most {MAX_LAGS} lags (a Beta weight between "
                f"two, a Dirichlet across three), got {self.rate_lags}",
            )
        if self.lag_weight_conc <= 0:
            raise ValueError(f"lag_weight_conc must be positive, got {self.lag_weight_conc}")
        if any(lag < 1 for lag in self.rate_lags):
            raise ValueError(f"rate_lags must all be positive, got {self.rate_lags}")
        if len(set(self.rate_lags)) != len(self.rate_lags):
            raise ValueError(f"rate_lags must be distinct, got {self.rate_lags}")
        if not 0.0 <= self.lag_weight <= 1.0:
            raise ValueError(f"lag_weight must lie in [0, 1], got {self.lag_weight}")

    @property
    def constants(self) -> dict[str, float]:
        """The imposed settings, recorded on the model for the run log."""
        return {
            "is_slope_mu": self.is_slope_mu,
            "is_slope_sigma": self.is_slope_sigma,
            "rstar_walk": float(self.rstar_form == "walk"),
            "rstar_linear": float(self.rstar_form == "linear"),
            "rstar_constant": float(self.rstar_form == "constant"),
            "sigma_rstar": self.sigma_rstar,
            "free_sigma_rstar": float(self.free_sigma_rstar),
            "sigma_rstar_prior": self.sigma_rstar_prior,
            "rstar_0_mu": self.rstar_0_mu,
            "rstar_0_sigma": self.rstar_0_sigma,
            "rstar_trend_sigma": self.rstar_trend_sigma,
            "rate_lag": float(self.rate_lags[0]),
            "rate_lag_2": float(self.rate_lags[1]) if len(self.rate_lags) > 1 else float("nan"),
            "rate_lag_3": (
                float(self.rate_lags[2]) if len(self.rate_lags) > PAIR_LAGS else float("nan")
            ),
            "n_rate_lags": float(len(self.rate_lags)),
            "lag_weight_free": float(self.lag_weight_free),
            "lag_weight": self.lag_weight,
            "lag_weight_a": self.lag_weight_a,
            "lag_weight_b": self.lag_weight_b,
            "lag_weight_conc": self.lag_weight_conc,
            "exclude_qe": float(self.exclude_qe),
            "sigma_e_prior": self.sigma_e_prior,
        }
