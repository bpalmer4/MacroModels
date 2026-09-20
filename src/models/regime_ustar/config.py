"""What this model imposes, collected so a run can state it.

The long-run state-space attempt recorded in `long_run_ustar/MODEL_NOTES.md`
failed on a dial: turn `sigma_ustar` down and u* is a prior line, turn it up
and u* copies the unemployment rate. Nothing measures that dial.

**u* is one continuous state across the whole sample.** What a regime changes
is its LAW OF MOTION, never its level, so every handoff is smooth because
there is only ever one state being carried forward. Within regime k,

    u*_t = u*_{t-1} + phi_k x (eq_k - u*_{t-1}) + e_t

so `eq_k` is where that period was pulling u* towards and `phi_k` is how fast.
The step function is the ATTRACTOR, not u* itself: u* takes time to arrive,
which is the stairs-down asymmetry `long_run_ustar` documents rather than a
jump at a date somebody chose.

This is `ustar`'s own `ustar_converge` law (`ustar/config.py:146`) with one
attractor per regime instead of one for the sample.

**What the reach is for.** `ustar` opens at 1993Q1 with a diffuse prior, so
u* there is placed by the unemployment rate sitting beside it, and no repair
from inside that sample can work: inflation in 1993 was at target, so the
Phillips curve sees no disequilibrium to attribute. Opening in 1970 means u*
ARRIVES at 1993 carrying a level inherited from the 1980s, and the early-1990s
data no longer get to set it. Twenty-three years of run-up is more than
enough for that: ten years gets 1993Q1 to 7.03 where thirty gets 7.14, against
`ustar`'s 10.75.
"""

from dataclasses import dataclass, field, fields, replace
from pathlib import Path

DEFAULT_OUTPUT_DIR = Path("model_outputs")
DEFAULT_CHART_BASE = Path("charts")

# Where one regime ends and the next begins. Each date is the FIRST quarter of
# the new regime. Institutional rather than estimated, so they can be disputed
# on history rather than on fit:
#
#   1970Q1-1973Q4  fixed exchange rate, centralised arbitration, u around 2.
#                  Reaches back to 1959Q3 when the sample does
#   1974Q1-1983Q2  wage indexation and the oil shocks. `long_run_ustar`'s
#                  scarring section dates the profit-share trough, the
#                  capital-for-labour substitution and the hiring stop to here
#   1983Q3-1992Q4  the Accord, then the disinflation
#   1993Q1-2014Q4  inflation targeting, the disinflation and the boom
#   2015Q1-2019Q4  the low-inflation era, inflation persistently under the band
#   2020Q1-        pandemic and after
#
# **2008Q4 was tried and moved.** It marks the GFC, which was a demand shock,
# not a change in the inflation regime. Over 2009-2013 inflation averaged 2.42
# and the surprise -0.20, with 2011 at 3.31: an ordinary stretch inside the
# band. The break is at 2015, where inflation goes 2.47 (2014) to 1.51 (2015)
# to 1.25 (2016) and the surprise goes -0.11 to -1.05 to -1.15 and does not
# recover before the pandemic. With the knot at 2008Q4 one cubic had to span
# both the ordinary and the unusual stretch and split the difference.
DEFAULT_BREAKS: tuple[str, ...] = ("1974Q1", "1983Q3", "1993Q1", "2015Q1", "2020Q1")


@dataclass
class ModelConfig:
    """The regimes, the equation, and the sample they are applied to."""

    # --- Sample ---
    # 1970Q1 is where PIE_RBAQ begins, and with `expectations_source`
    # "spliced" that is the first quarter a measured expectation exists for.
    # Opening there means no quarter of the sample rests on the asserted
    # salience rule. The quarterly unemployment rate itself reaches 1959Q3
    # (1364.0.15.003), so `start=None` runs the longer sample, at the price of
    # eleven years whose expectation is asserted.
    start: str | None = "1970Q1"
    end: str | None = None

    # --- Regimes ---
    breaks: tuple[str, ...] = DEFAULT_BREAKS

    # --- The equation ---
    # --- How the expectation is formed ---
    #
    # Attention switches on above a threshold. Below it almost nobody is
    # thinking about inflation, which is most of what anchoring is; above it
    # every conversation starts with how expensive things have become. See
    # `observations.salience_expectation` for the evidence and the caveats.
    #
    #     theta = theta_lo if pi <= salience_threshold else theta_hi
    #     pi_e_t = pi_e_{t-1} + theta x (pi_{t-1} - pi_e_{t-1})
    #
    # 6 per cent is asserted and cannot be estimated: the only era with a
    # measured expectations series has every high-inflation quarter also
    # anchored, so the level effect and the anchor effect are confounded there.
    # Sweep it.
    salience_threshold: float = 6.0
    theta_lo: float = 0.10
    theta_hi: float = 0.60

    # Where the recursion starts, and what an unanchored 1950s would have
    # believed. Headline inflation averaged 2.46 across 1959Q3-1969Q4.
    anchor_level: float = 2.5

    # Where the measured series takes over. 1983Q1 is where it begins, and
    # handing over there imposes no anchoring: the series itself shows
    # expectations at 6.6-8.0 through the 1980s and averaging 3.15 across
    # 1993-97 against a 2.5 target, settling only in 1998.
    measured_from: str = "1983Q1"

    # Which measured series, and how far back it reaches.
    #   "model"   the expectations model alone, from 1983Q1.
    #   "spliced" PIE_RBAQ before 1983Q1, the model after, which moves the
    #             handoff to 1970Q1 and leaves the salience rule covering only
    #             1959Q3-1969Q4. See `src/data/expectations_spliced.py` for the
    #             overlap evidence and for why the level offset is a switch:
    #             the gap between the two series runs +0.87 in the 1980s down
    #             to +0.01 by the 2010s, so no single offset is right, and 0
    #             here splices them raw and accepts a step at the join.
    #
    # "spliced" by default, so the expectation is measured throughout rather
    # than asserted for the first thirteen years. THE 1970s ARE CONDITIONAL ON
    # THIS CHOICE and the two constructions disagree by about 4 points of u*
    # there: PIE_RBAQ is a trend anchor, it sits 6.52 below year-ended headline
    # across 1974-79, and the Phillips curve reads that gap as a tight labour
    # market, putting u* near 9 against an unemployment rate near 5. The
    # salience rule gives about 5. Nothing in the data chooses between them,
    # because neither expectation is observed.
    expectations_source: str = "spliced"
    splice_offset_quarters: int = 8

    # Separate residual scales either side of the 1983 handoff. Inflation less
    # an asserted expectation and inflation less a measured one are different
    # objects with different noise, and one shared sigma would let the longer
    # block set the weight the other is fitted at.
    #
    # **The 1983 handoff is a measurement change as well as a regime change**,
    # and nothing in the sample can separate them: a level shift in u* there
    # could be either. It remains the model's weakest join.
    regime_sigma: bool = True

    # Quarters by which unemployment is read ahead of the inflation change.
    # Reported rather than chosen: `ystar` items 7, 9 and 12 establish that the
    # lag is not identified on Australian data, so this is a switch to sweep.
    lag: int = 0

    # Which inflation series the Phillips curve explains.
    #   "spliced"  headline until the trimmed mean starts at 1983Q1, trimmed
    #              after. The trimmed mean is the cleaner nominal signal and
    #              this uses it everywhere it exists, without giving up the
    #              1970s. Spliced raw: the two measure the same object, they
    #              correlate 0.905 over 174 overlapping quarters with a mean
    #              difference of -0.04, and an offset fitted on that overlap
    #              would be fitting noise.
    #   "headline" headline CPI throughout, on one measure end to end.
    #   "trimmed"  trimmed mean only, which starts the sample at 1983Q1.
    #
    # **What the splice costs.** Headline is 11.24 at 1983Q1 and the trimmed
    # mean 10.60, so the join puts a -0.64 step into the series against
    # headline's own +0.22 move that quarter. That lands two quarters before
    # the 1983Q3 Accord break, and `expectations_source` "spliced" changes
    # construction at 1983Q1 as well, so both sides of `pi - pi_e` change
    # measure in the same quarter as each other and close to a regime date.
    # A level shift in u* around 1983 could be any of the three.
    inflation: str = "spliced"

    # A SECOND observation equation, on wages:
    #
    #     ulc_yoy - pi_e = alpha + gamma x (u - ustar)/u + lambda x dU/U + v
    #
    # This is the only thing here that attacks the circularity. With the price
    # equation alone, `u - ustar` is algebraically the inflation gap scaled by
    # `u/beta`, so no labour-market observable enters except `u` itself. Unit
    # labour costs are a labour-market price rather than a consumer price, so a
    # second equation on them gives the same `ustar` a second thing to answer
    # to.
    #
    # Form follows `nairu/equations/phillips_wage.py`: the same proportional
    # gap, expected inflation entering with a coefficient of one, a free
    # intercept absorbing trend productivity rather than netting productivity
    # off, and the speed-limit term on the CHANGE in unemployment.
    #
    # **Written year-ended, where `nairu` writes it quarterly.** The price
    # equation here is year-ended, so this keeps `beta` and `gamma` in the same
    # units and makes their agreement a diagnostic rather than a units
    # comparison. The cost is a noisier dependent variable: year-ended ULC
    # growth has sd 4.47 against 1.58 quarterly.
    #
    # **OFF BY DEFAULT, because it makes the path worse.** `gamma_wage` comes
    # back at -4.67 [-5.79, -3.63], strongly identified and about 2.8x the
    # price equation's slope in the same units, and u*'s correlation with a
    # smoothed unemployment rate falls 0.881 to 0.781, which is the largest
    # such fall anything here achieved. So the equation is not weak and it does
    # loosen u* from being a smoothed u.
    #
    # It is off anyway because the resulting PATH is worse everywhere it can be
    # checked against history. The hump flattens from a peak near 7.7 to near
    # 6.5, losing the plateau near 8 that the Accord period is usually read as
    # having; the early-1970s rise, already hard to defend, worsens from 3.14
    # to 4.37 by 1973Q4; the end-of-sample reading rises 6.14 to 6.65; and
    # 1993-2001 acquires a fall-then-rise that no account of the period
    # supports.
    #
    # The 1988-89 benchmark does improve sharply, 7.27 to 6.32 against 6.20.
    # That was initially read as the wage equation working. The likelier
    # explanation is that the whole hump fell by about a point and that
    # benchmark sat under it, since one benchmark improving while the shape
    # deteriorates is not evidence of anything.
    #
    # `nairu`'s own caution points the same way: its notes record the wage
    # block's expectations coefficient straddling zero and call the weak
    # identification structural rather than collinearity, and its NAIRU is
    # still suspiciously flat with five observation equations.
    wage_equation: bool = False
    gamma_wage_prior: tuple[float, float, float] = (-1.5, 1.0, 0.0)
    lambda_wage_prior: tuple[float, float] = (-4.0, 2.0)
    alpha_wage_prior_sd: float = 2.0
    sigma_wage_prior_sd: float = 4.0

    # Import price growth and GSCPI as supply controls, after `ustar`, whose
    # functional form and prior scales these copy rather than reinventing:
    #
    #     + rho x d4pm  +  xi x GSCPI^2 x sign(GSCPI)
    #
    # **They cannot reach the 1970s.** Import price growth starts 1984Q3 and
    # GSCPI 1998Q1, and both are zero-filled before. For GSCPI that is neutral,
    # since it is standardised in deviations from its own mean. For import
    # price growth it is NOT neutral: zero asserts no import price inflation
    # through the oil shocks, which is false. The consequence is that the
    # pre-1984 fit is exactly as it was without the term, so the 1970-73 rise
    # in u* stays uncontrolled and stays attributed to the labour market. The
    # coefficient is at least estimated on 1984-2026 where the data are real.
    supply_control: bool = True
    rho_prior_sd: float = 0.1
    xi_prior_sd: float = 0.1

    # Terms of trade growth as a supply-shock control. OFF by default: this is
    # the exploratory cut, and the point is to see what the bare equation says
    # before anything is held constant. Import prices would be the better
    # control and cannot be used, because they start 1983Q2, after the decade
    # that most needs controlling. Terms of trade reach 1959Q4.
    tot_control: bool = False

    # A SECOND observation on u*, from output and unemployment:
    #
    #     du_t = a + b x dy_t + lambda x (u_{t-1} - u*_{t-1}) + e
    #
    # Written as error correction rather than as `u = u* - b x ygap`, the form
    # used where an output gap is already available. That form would need a
    # gap estimated elsewhere, which both couples this model to another and
    # limits the equation to 1993 onward, where those gaps begin. Real GDP
    # growth reaches 1959Q4, so this covers the whole sample.
    #
    # THIS IS THE ONLY THING HERE THAT ATTACKS THE CIRCULARITY. With the
    # Phillips curve alone, u - u* is the inflation gap scaled by u/beta, so
    # no observable enters except u itself, on both sides. Output is a second
    # observable and its equation can disagree.
    #
    # `lambda` is the rate at which unemployment falls back toward u*, so it
    # should be negative; left two-sided, because a prior truncated at zero
    # would assert the error correction rather than let the posterior report
    # whether the data show one.
    okun_equation: bool = False
    a_okun_prior_sd: float = 0.5
    b_okun_prior: tuple[float, float] = (-0.2, 0.2)
    lambda_okun_prior: tuple[float, float] = (-0.1, 0.1)
    sigma_okun_prior_sd: float = 0.5

    # An AR(1) error in the Phillips curve, e_t = phi x e_{t-1} + eta_t.
    #
    # The static form leaves a residual autocorrelated at +0.63 overall and
    # +0.60 to +0.88 within every regime, so a persistent component was being
    # treated as independent noise. That matters beyond tidiness: independent
    # errors make each quarter a separate piece of evidence about u*, so a
    # run of same-signed surprises looks like many confirmations rather than
    # one, and u* is pulled further and reported more precisely than the data
    # warrant.
    ar1_error: bool = False
    phi_e_prior: tuple[float, float] = (0.0, 0.5)

    # Student-t rather than Normal residuals. The 1970s and the 2020s put large
    # changes in year-ended headline inflation next to unremarkable
    # unemployment, and under a Normal likelihood those quarters set `beta`.
    student_t: bool = True
    nu_prior: float = 5.0

    # --- The state law ---
    # "spline"    u* is a natural cubic spline with knots at the regime dates,
    #             deterministic given its coefficients. No innovation variance
    #             to impose, and each period gets a SHAPE rather than a level,
    #             so a regime spanning a peak and a descent can do both.
    # "attractor" u* is a random walk pulled towards one free constant per
    #             regime. Kept for comparison: it is what the spline replaced,
    #             and it needs `sigma_ustar`, which nothing measures.
    state: str = "spline"

    # Knots repeated to drop continuity there. A cubic knot of multiplicity 3
    # matches the level and frees the slope, which is what a break needs.
    # 1974Q1 is the ratchet: oil, indexation and the pass-through arriving at
    # once. Forced to C2 the spline bends the 1960s upward to meet it.
    knot_multiplicity: dict[str, int] = field(default_factory=lambda: {"1974Q1": 3})

    # Force the curve linear beyond the outer knots. The final segment has no
    # knot after it and sits under the largest inflation surprises in the
    # sample, where a free cubic is at its worst.
    natural_spline: bool = True


    # How fast u* is pulled towards its regime's attractor, per quarter. Free,
    # with a prior that spans "barely moves" to "arrives within a couple of
    # years". One speed shared by every regime by default: letting each regime
    # buy its own speed as well as its own destination is two free numbers
    # describing one path segment, and the second is what a weak likelihood
    # would spend first.
    phi_prior: tuple[float, float, float, float] = (0.05, 0.05, 0.0, 1.0)
    free_phi_per_regime: bool = False

    # Where each regime pulls u* towards. Wide and flat over anything
    # defensible: 0.5 to 12 per cent spans the 1.4 minimum and the 11.1 peak of
    # the unemployment series itself.
    eq_prior: tuple[float, float, float, float] = (5.0, 3.0, 0.5, 12.0)

    # Quarter-to-quarter innovation in u*, IMPOSED. This is the dial that sank
    # the deleted model, and it is imposed rather than estimated for the reason
    # `ustar` imposes its own: a weak Phillips likelihood will spend a free
    # variance on fitting noise, and u* becomes the unemployment rate again.
    # Small here because the regime attractors are what is supposed to move u*.
    # Sweep it; do not trust a single value.
    sigma_ustar: float = 0.05

    # Let u* innovate more in periods when unemployment has actually moved.
    #
    #     rel_t   = u_move_t / mean(u_move)
    #     sigma_t = sigma_ustar x max(floor, rel_t) ** kappa
    #
    # `kappa` = 0 is exactly the constant-innovation model, so the idea is
    # NESTED and the posterior on `kappa` is a test of it rather than an
    # assertion. `kappa` = 1 makes the innovation sd proportional to how far
    # the one-year average of unemployment has moved over two years.
    #
    # **The risk, stated because it is the obvious objection.** This licenses
    # u* to move exactly when u moves, which is the failure mode the whole
    # package guards against. It is a variance rule and not a mean rule, so
    # nothing tells u* which WAY to go and the Phillips curve still decides
    # that. But more freedom during the quarters when u is moving will pull u*
    # towards u in those quarters, and the correlation with a smoothed
    # unemployment rate is the thing to check after every run.
    adaptive_sigma: bool = False
    kappa_prior: tuple[float, float, float, float] = (1.0, 0.5, 0.0, 3.0)
    # Keeps sigma_t away from zero in the quietest stretches, where `rel` can
    # be a few hundredths and a power of it would freeze u* entirely.
    move_floor: float = 0.25

    # Regime indices, 0-based, that share a SECOND Phillips slope. Empty means
    # one slope for the whole sample.
    #
    # **What one slope assumes.** The regimes change only the shape of u*;
    # `beta`, `sigma`, `rho` and `xi` are single numbers from 1970 to 2026, so
    # wage indexation, the Accord and inflation targeting are required to share
    # a slack-to-inflation coefficient. Holding u* at its fitted path, four of
    # the six regimes agree: 1970-73, 1993-2014, 2015-19 and 2020-26 each want
    # 3.36 to 4.38 against a pooled 3.60, with residual means inside 0.18. Two
    # do not. 1974Q1-1983Q2 wants 5.64 and is left with a residual averaging
    # +2.57 across ten years, which is a specification failure rather than
    # noise; 1983Q3-1992Q4 wants 0.20, near enough to no Phillips curve, which
    # is what an incomes policy setting wages would look like.
    #
    # **So the two candidate groups pull opposite ways** and a second slope
    # spanning both averages a steep decade with a flat one.
    # One entry per regime, naming which slope that regime uses. Empty is one
    # slope for the whole sample; (0, 1, 2, 0, 0, 0) gives the pre-Accord and
    # Accord decades their own and pools the rest.
    beta_groups: tuple[int, ...] = ()

    # Regimes given a free constant in the Phillips curve, for a period whose
    # inflation sits persistently off what slack alone implies.
    #
    # The alternative to a regime's own slope, and for some regimes the right
    # one. Fitting 1974Q1-1983Q2 through the origin forces a mean surprise of
    # +5.73 to be explained by a mean gap of -0.878, which needs a slope of
    # 5.64; allow a constant and the slope falls to 3.86, near the pooled
    # value, with a +2.34 intercept. A LEVEL THE MODEL CANNOT ACCOUNT FOR IS
    # NOT A STEEPER PHILLIPS CURVE, and giving it a slope also puts a near-zero
    # or near-vertical coefficient into the inversion's denominator.
    intercept_regimes: tuple[int, ...] = ()
    intercept_prior_sd: float = 2.0

    # --- Priors ---
    # On the PROPORTIONAL gap, so the scale is `ustar`'s, whose `gamma_pi` is
    # about 1.15. A half-normal at 0.5 would put that 2.3 prior sd out and
    # bind; 1.5 leaves it comfortably inside.
    beta_prior_sd: float = 1.5
    sigma_prior_sd: float = 1.0
    tot_prior_sd: float = 0.2

    # --- Output ---
    prefix: str = "regime_ustar"
    output_dir: Path = DEFAULT_OUTPUT_DIR
    chart_base: Path = DEFAULT_CHART_BASE
    chart_dir_name: str = "RegimeUStar"
    labels: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        """Validate the regimes and the equation."""
        if self.lag < 0:
            raise ValueError(f"lag must be non-negative, got {self.lag}")
        if self.state not in ("spline", "attractor"):
            raise ValueError(f"state must be 'spline' or 'attractor', got {self.state!r}")
        if self.expectations_source not in ("model", "spliced"):
            raise ValueError(
                f"expectations_source must be 'model' or 'spliced', got {self.expectations_source!r}",
            )
        if self.splice_offset_quarters < 0:
            raise ValueError(f"splice_offset_quarters must be non-negative, got {self.splice_offset_quarters}")
        if self.inflation not in ("headline", "trimmed", "spliced"):
            raise ValueError(f"inflation must be 'headline', 'trimmed' or 'spliced', got {self.inflation!r}")
        self._validate_knots()
        self._validate_beta_groups()
        self._validate_intercept_regimes()
        if len(set(self.breaks)) != len(self.breaks):
            raise ValueError(f"breaks must be distinct, got {self.breaks}")
        if list(self.breaks) != sorted(self.breaks):
            raise ValueError(f"breaks must be in order, got {self.breaks}")

    def _validate_beta_groups(self) -> None:
        """One group per regime, numbered from zero with no gaps."""
        if not self.beta_groups:
            return
        n_regimes = len(self.breaks) + 1
        if len(self.beta_groups) != n_regimes:
            raise ValueError(
                f"beta groups must give one entry per regime: got {len(self.beta_groups)} for {n_regimes} regimes",
            )
        seen = sorted(set(self.beta_groups))
        if seen != list(range(len(seen))):
            raise ValueError(f"beta groups must be numbered from 0 with no gaps, got {self.beta_groups}")

    def _validate_intercept_regimes(self) -> None:
        """Check that each intercept names a real regime, and that they do not claim every one."""
        n_regimes = len(self.breaks) + 1
        for k in self.intercept_regimes:
            if not 0 <= k < n_regimes:
                raise ValueError(f"intercept regime {k} is outside the {n_regimes} regimes the breaks define")
        if len(set(self.intercept_regimes)) != len(self.intercept_regimes):
            raise ValueError(f"intercept regimes must be distinct, got {self.intercept_regimes}")
        # An intercept in every regime is allowed but barely identified: within
        # a regime the equation is (alpha_k - beta_k) + beta_k x u* x (1/u), so
        # only the variation in 1/u separates u*'s level there from the
        # constant. That variation is thin in 2015-19, where the coefficient of
        # variation of 1/u is 0.060, and u*'s level in such a window falls back
        # on `eq_prior` rather than on the data.

    def _validate_knots(self) -> None:
        """Every repeated knot must sit on a break, and cannot outrank the spline's degree."""
        max_multiplicity = 3  # the spline's degree: above this the basis gains nothing
        for date, count in self.knot_multiplicity.items():
            if date not in self.breaks:
                raise ValueError(f"knot multiplicity names {date}, which is not a break: {self.breaks}")
            if not 1 <= count <= max_multiplicity:
                raise ValueError(f"knot multiplicity must be 1 to {max_multiplicity}, got {count} at {date}")

    @property
    def chart_dir(self) -> Path:
        """Where this model's charts and its diagnostics file live."""
        return self.chart_base / self.chart_dir_name

    @property
    def constants(self) -> dict[str, object]:
        """What was imposed, for recording alongside a run."""
        return {
            "breaks": list(self.breaks),
            "state": self.state,
            "natural_spline": self.natural_spline,
            "knot_multiplicity": dict(self.knot_multiplicity),
            "sigma_ustar": self.sigma_ustar,
            "free_phi_per_regime": self.free_phi_per_regime,
            "anchor_level": self.anchor_level,
            "salience_threshold": self.salience_threshold,
            "theta_lo": self.theta_lo,
            "theta_hi": self.theta_hi,
            "measured_from": self.measured_from,
            "expectations_source": self.expectations_source,
            "splice_offset_quarters": self.splice_offset_quarters,
            "regime_sigma": self.regime_sigma,
            "adaptive_sigma": self.adaptive_sigma,
            "move_floor": self.move_floor,
            "n_regimes": len(self.breaks) + 1,
            "lag": self.lag,
            "inflation": self.inflation,
            "tot_control": self.tot_control,
            "supply_control": self.supply_control,
            "wage_equation": self.wage_equation,
            "okun_equation": self.okun_equation,
            "student_t": self.student_t,
            "ar1_error": self.ar1_error,
            "beta_prior_sd": self.beta_prior_sd,
            "beta_groups": list(self.beta_groups),
            "intercept_regimes": list(self.intercept_regimes),
            "intercept_prior_sd": self.intercept_prior_sd,
        }

    def with_saved_settings(self, constants: dict[str, object]) -> "ModelConfig":
        """Return this config with the settings a saved run was estimated under.

        Charting a saved trace has to use the specification that produced it,
        not whatever the command line defaults to. `beta_groups` is the case
        that bites: a run estimated with two slopes writes a `beta` of width
        two, and analysing it under the default single slope indexes an empty
        array. Everything `constants` records is restored, so a setting added
        there is covered without touching this method.

        Where the output goes is NOT restored. `prefix`, `output_dir` and the
        chart directory say where to read and write, which is the caller's to
        choose; the model settings are the run's own and are not.
        """
        names = {f.name for f in fields(self)}
        restored: dict[str, object] = {}
        for key, value in constants.items():
            if key not in names:
                continue  # derived entries such as n_regimes
            current = getattr(self, key)
            restored[key] = tuple(value) if isinstance(current, tuple) else value
        # `replace` re-runs __post_init__, so a trace whose settings contradict
        # each other is caught here rather than part-way through charting.
        return replace(self, **restored)
