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
Phillips curve sees no disequilibrium to attribute. Running from 1959 means
u* ARRIVES at 1993 carrying a level inherited from the 1980s, and the
early-1990s data no longer get to set it.
"""

from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_OUTPUT_DIR = Path("model_outputs")
DEFAULT_CHART_BASE = Path("charts")

# Where one regime ends and the next begins. Each date is the FIRST quarter of
# the new regime. Institutional rather than estimated, so they can be disputed
# on history rather than on fit:
#
#   1959Q3-1973Q4  fixed exchange rate, centralised arbitration, u around 2
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
    # 1959Q3 is where the quarterly unemployment rate starts (1364.0.15.003).
    start: str | None = None
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

    # Headline CPI throughout, not spliced to the trimmed mean at 1983Q1.
    # The trimmed mean is the cleaner nominal signal, but it begins one quarter
    # before the Accord break, so a spliced series would change measure and
    # regime at the same date and the two could not be told apart. The trimmed
    # mean is instead available as a post-1983 comparison run.
    inflation: str = "headline"

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
        if self.inflation not in ("headline", "trimmed"):
            raise ValueError(f"inflation must be 'headline' or 'trimmed', got {self.inflation!r}")
        if len(set(self.breaks)) != len(self.breaks):
            raise ValueError(f"breaks must be distinct, got {self.breaks}")
        if list(self.breaks) != sorted(self.breaks):
            raise ValueError(f"breaks must be in order, got {self.breaks}")

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
            "sigma_ustar": self.sigma_ustar,
            "free_phi_per_regime": self.free_phi_per_regime,
            "anchor_level": self.anchor_level,
            "salience_threshold": self.salience_threshold,
            "theta_lo": self.theta_lo,
            "theta_hi": self.theta_hi,
            "measured_from": self.measured_from,
            "regime_sigma": self.regime_sigma,
            "adaptive_sigma": self.adaptive_sigma,
            "move_floor": self.move_floor,
            "n_regimes": len(self.breaks) + 1,
            "lag": self.lag,
            "inflation": self.inflation,
            "tot_control": self.tot_control,
            "supply_control": self.supply_control,
            "wage_equation": self.wage_equation,
            "student_t": self.student_t,
        }
