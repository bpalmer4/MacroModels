"""Configuration for the ustar model.

One state, two observation equations, four estimated parameters and one
imposed. The whole specification is:

    u*_t = u*_{t-1} + e_u                        e_u ~ N(0, sigma_ustar)
    u_t  = u*_t - beta·ygap_t + e_o              e_o ~ N(0, sigma_o)
    pi_t = pi_exp_t + gamma·ugap_t + e_p         e_p ~ N(0, sigma_p)
    ugap_t = u_t - u*_t

with `ygap` read from a completed `ystar` run rather than estimated.

**Why both equations.** Okun sets u*'s level: rearranged, u* = u + beta·ygap,
so wherever the gap says output is above potential, u* sits above measured
unemployment. That is the identification the deleted `ustar_wage` experiment
lacked, where the level ended up set by the smoothness prior because a nominal
slope alone was too weak to pin it. The Phillips curve supplies what Okun
cannot: the nominal content that makes the answer a NAIRU rather than an
Okun-implied trend, and the place expectations enter.

**Why sigma_ustar is imposed.** A quarterly wiggle in unemployment can be a
shift in u* or a residual, and the likelihood cannot fully apportion between
them — the Stock-Watson pile-up problem, identically to `sigma_ystar` in
`ystar`. The `nairu` model fixes its own `nairu_innovation` at 0.15 for
the same reason, which is where this default comes from. Sweep it rather than
trusting it.

**What `beta` is not.** It will come out well above a textbook Okun coefficient
of 0.3-0.5. `ystar`'s gap is a shrunk regressor — `c` is a conditional
mean on a signal explaining about a fifth of output's variation — so the slope
compensates and the *level* of the unemployment gap comes out roughly right.
Reading `beta` as an Okun coefficient comparable to the literature is a
mistake; it is a scaling onto this particular gap series.
"""

from dataclasses import dataclass
from pathlib import Path

DEFAULT_OUTPUT_DIR = Path(__file__).parent.parent.parent.parent / "model_outputs"

# Which ystar series feeds the Okun equation.
#   "defined" — c·(pi - anchor), the inflation-defined gap. Demand-only by
#       construction, and the one that keeps the nominal anchoring clean.
#   "actual"  — log_gdp - y*, the full deviation. Larger amplitude, but it
#       carries e_c, so Okun would read the 2020 lockdown supply disruption
#       as a movement in u*.
GAP_SOURCES = ("defined", "actual")


@dataclass
class ModelConfig:
    """Specification, sample, and the one imposed variance."""

    # --- Sample ---
    start: str | None = "1993Q1"
    end: str | None = None

    # --- The given output gap ---
    gap_source: str = "defined"
    gap_prefix: str = "ystar"
    # Enter the gap with its posterior sd as a measurement-error prior. Off, it
    # enters as the posterior mean treated as known, which understates the u*
    # band: beta absorbs the shrinkage in `c` while the interval is computed as
    # though the input were exact.
    gap_measurement_error: bool = True

    # Diagnostic: zero the output gap, keeping the Okun equation's structure.
    #
    # The premise of this package is that a credible output gap from
    # `ystar` is what makes a two-equation u* possible. That premise is
    # testable, and this is the test. With the gap zeroed the Okun equation
    # becomes `u = u* + e_o`, still a trend fitted through unemployment, so the
    # comparison isolates the gap's contribution rather than the equation's.
    #
    # If u* barely moves, the model is an inflation-based filter of
    # unemployment and the output gap is decorative. Kept as a switch so the
    # check is reproducible, in the same spirit as `ystar`'s
    # `free_sigma_ystar`.
    use_output_gap: bool = True

    # A quadratic term in the unemployment gap, so the Phillips curve is convex
    # in BOTH directions:
    #
    #     pi = ... + gamma x g + delta x g x |g| + ...      g = (u - u*)/u
    #
    # The linear form makes the likelihood equally sensitive to u* whatever the
    # gap is. With the quadratic the sensitivity rises with |g|, so the
    # quarters when the labour market was furthest from balance do most to
    # place u*, and the quiet middle of the sample does least.
    #
    # `delta` is centred on zero, so the posterior says whether the data want
    # it. The scale follows from the gap's own range, about -0.43 to +0.55: for
    # the quadratic to matter as much as the linear term at a gap of 0.3,
    # delta must be 3 to 4 times gamma.
    quadratic_gap: bool = False
    delta_pi_prior_sd: float = 4.0

    # --- Equations ---
    include_phillips: bool = True

    # Include the GAP-FORM Okun equation, u - u* = -beta x ygap: the
    # unemployment gap against the output gap, not Okun's original relating
    # the CHANGE in unemployment to output growth. `beta` is therefore not
    # comparable with a textbook Okun coefficient of 0.3 to 0.5.
    #
    # OFF: it carries no information this model does not already have, and it
    # biases the answer. The gap-on-gap structure is exactly why.
    #
    # The "defined" gap is 0.1882 x (pi - 2.5) exactly, to machine precision,
    # so u = u* - beta_okun x ygap is itself a Phillips curve in levels and the
    # two equations read one signal. Three things follow, all measured. The
    # reported band counts that signal twice: dropping Okun widens the mean
    # 90% band from 0.35 to 0.58, and over 1993-98 from 0.36 to 1.15. The
    # fitted u* sits systematically above what inflation alone implies, a mean
    # deviation of -0.13 to -0.23 with Okun against -0.00 without. And because
    # the gap enters with a measurement error of 0.05 while the Phillips
    # residual is worth about 1.5 points of u*, Okun effectively dictates the
    # level: across 1993-98 it holds u* at 8.69 against an unemployment rate
    # of 8.90, calling the deepest slack in the sample equilibrium.
    include_okun: bool = False

    # --- The state law ---
    # "converge" u* is a random walk pulled toward one equilibrium. It can
    #            only draw a monotone approach, so it cannot decline and then
    #            stop, and it needs `sigma_ustar`, which nothing measures.
    # "spline"   u* is a natural cubic spline with knots at `spline_knots`,
    #            deterministic given its coefficients. No innovation variance
    #            to impose, and a segment after a knot is free to be flat while
    #            the one before it is steep.
    #
    # "spline" by default, on the band test: scored against the sign of the
    # unemployment gap over the 64 quarters where quarterly annualised trimmed
    # mean inflation sat outside 2-3%, a knot at 2013Q1 gets 59 right against
    # 53 for the convergence law, and 24 of 24 on the below-band quarters.
    state_law: str = "spline"

    # Interior knot dates for the spline. One knot gives three coefficients
    # after the natural boundary reduction, which is stiff: enough to decline
    # then level off, not enough to invent a cycle.
    #
    # 2013Q1 is where the low-inflation era begins, and it is where the decline
    # in u* stops: the fitted path turns from -0.32 over 2015-2026 under the
    # convergence law to +0.36 here. A knot at 2008Q1 was tried and scores 4
    # quarters worse, though the two give nearly identical endpoints, because
    # three coefficients leave the curve's shape largely determined.
    spline_knots: tuple[str, ...] = ("2013Q1",)

    # Prior on the spline coefficients. The basis is a partition of unity, so
    # these are in unemployment-rate units and a coefficient is roughly the
    # level u* passes through near its knot. Bounds span the sample's own
    # range, 3.5 to 10.9.
    spline_coef_prior: tuple[float, float, float, float] = (6.0, 3.0, 2.0, 14.0)

    # The inflation target, asserted flat across the sample. No phase-in: the
    # sample starts in 1993, inside the inflation-targeting era, so there is
    # nothing to phase from. `nairu` needs one because it starts in 1984;
    # `ystar` asserts the same flat 2.5 from the same 1993Q1 start.
    #
    # This is the Phillips curve's baseline, and the expectations series enters
    # only as a deviation from it. That pairing matters: with a *target*
    # baseline the excess term is the pass-through of de-anchoring, beta = 0
    # meaning the target holds and beta = 1 meaning expectations are what bind.
    # With an *expectations* baseline the same term would be one estimate of
    # expectations minus another, which is not an economic object.
    anchor: float = 2.5
    # A one-sided prior asserts the sign of Okun's law rather than testing it.
    # Two-sided is the honest default; the posterior can then report P(beta > 0)
    # the way ystar reports P(c > 0).
    two_sided_beta: bool = True

    # Let u* drift down while inflation expectations sit above target.
    #
    #     u*_t = u*_{t-1} - lambda·max(0, pi^e_{t-1} - anchor) + e_u
    #
    # Off, u* is a driftless random walk, and that prior is badly at odds with
    # the sample. Fitted, u* falls 8.43 to 4.71: 3.72pp over 134 quarters,
    # where a driftless walk at sigma_ustar = 0.040 puts the sd of the total
    # change at 0.46pp. The fitted path is about 8 standard deviations out, and
    # the increments use most of the allowed per-quarter movement with three
    # quarters of it directed rather than random. The prior is not stretched,
    # it is overwhelmed.
    #
    # The cost shows up at the start of the sample, and Limitation 4 in the
    # notes records the symptom without naming this as the cause. To begin
    # where unemployment actually was in 1993Q1, 10.93, and still reach 4.71
    # needs 6.2pp, which the prior cannot afford, so the posterior starts u* at
    # 8.43 and books the remaining +2.5pp in the Okun residual.
    #
    # **Why expectations rather than a date or a constant.** A constant drift
    # says the NAIRU falls forever, which is wrong at the endpoint. A date break
    # is arbitrary. Excess expectations are an observable measuring the thing
    # that would actually move a NAIRU: a target people do not yet believe, so
    # wage-setting has not adapted to it. The series puts the regime's
    # credibility at about 1998 — excess over target averages +0.64 across
    # 1994-1996 against +0.07 across 2000-2019 — even though realised trimmed
    # mean inflation was already 2.1% in 1993Q1. Credibility and realised
    # inflation are different things and only the first should move u*.
    #
    # **Why it stops in 2000, and why that is a date rather than a rule.** The
    # level of excess expectations cannot tell "the target is not yet believed"
    # from "a supply shock has temporarily lifted expectations": it reads +0.64
    # across 1994-1996 and +0.87 across 2022-23. Left ungated the drift pushed
    # u* from 4.71 to 4.25 at 2026Q2 and flipped the current unemployment gap
    # from -0.36 to +0.10, which is a large claim about today resting on an
    # episode that was not a regime transition.
    #
    # The mechanism is one-off. Wage-setting adapts to a credible target once
    # and does not unadapt, so the term is switched off after
    # `ustar_drift_end`. That is an asserted date, and it is asserted rather
    # than estimated because the claim being made is historical: the Australian
    # inflation-targeting regime became credible during the 1990s. The series
    # supports the timing without being asked to — excess averages +0.64 across
    # 1994-1996 and +0.07 across 2000-2019 — but the date is a judgement and
    # should be swept.
    ustar_drift: bool = False
    # Let u* converge to a new equilibrium instead of drifting on expectations:
    #
    #     u*_t = u*_{t-1} + phi·(u*_eq - u*_{t-1}) + e_u
    #
    # The alternative story for the same fact. `ustar_drift` says u* fell in the
    # 1990s because the target was not yet believed; this says it was moving to
    # a new equilibrium and decelerated as it arrived. Three differences that
    # matter. It needs no cutoff date, because the process stops by arriving
    # rather than by decree. `u*` depends on nothing but its own past, so the
    # unemployment gap cannot become a proxy for inflation expectations, which
    # under `ustar_drift` it partly does (corr with excess expectations goes
    # +0.05 to -0.41 over 1993-99, and `beta_pi` falls from 0.578 to 0.378).
    # And it is stationary, which is a strong claim the drift does not make.
    #
    # That last point is the risk. The fitted u* falls 6.64 to 4.66 after 2000,
    # which is 4.8 sd against a driftless walk, so the sample has a slow
    # secular decline as well as the 1990s transition. A process converging to a
    # constant has to fight that, and may resolve it by putting `u*_eq` low and
    # `phi` small, which would be a transition model masquerading as one.
    #
    # **ON BY DEFAULT.** The driftless walk it replaces is about 8 standard
    # deviations from its own prior over the sample and 17 over 1993-1999, puts
    # u* below unemployment in all 16 quarters of 1994-1997 (see MODEL_NOTES for
    # why that window is weaker evidence than it was once written as: the annual
    # trimmed mean averaged 2.41% there and sat below expectations throughout),
    # and leaves a +2.5pp gap one year after the deepest
    # recession since the 1930s. Convergence fixes all three, fits better
    # (sigma_okun 0.685 -> 0.426) and samples better, without an asserted date.
    #
    # What it does not do is improve the model's external validation. Against
    # wage growth, which is not in this likelihood, the driftless gap correlates
    # -0.523 with WPI and this one -0.485; on hourly compensation over the full
    # sample, -0.205 against -0.059. Both still pass the test — the gap beats
    # raw unemployment at -0.243 — but the fix does not make the gap a better
    # measure of tightness, and on the longer wage series it is worse. That is
    # recorded in MODEL_NOTES and is the honest counterweight to everything
    # above.
    #
    # Mutually exclusive with `ustar_drift`.
    ustar_converge: bool = True

    # Prior mean for u* in the first quarter. None centres it on the
    # unemployment rate of that quarter, which at a 1993Q1 start is 10.85, a
    # recession trough. That choice is not innocuous: `phi_ustar` comes back
    # near 0.04, so 72% of the 1993 starting level is still in the path two
    # years later and 44% after five, which is most of what sets u* across
    # 1993-98. The likelihood has little to say there, inflation being at
    # target, so the posterior stays within 0.03 prior sd of this mean.
    ustar_init_mu: float | None = None

    # Quarters from here on carry no drift. See `ustar_drift`.
    ustar_drift_end: str = "2000Q1"
    # Prior sd for `lambda_ustar`, whose mean is zero so the sign is tested
    # rather than asserted. A switch because 0.1 turned out to sit 2.5 prior
    # standard deviations below the posterior, which is where `beta_okun`'s
    # prior was found to be binding.
    lambda_prior_sd: float = 0.1

    # --- How fast u* is allowed to drift ---
    # **0.020, and the number changed when u* stopped being a random walk.**
    #
    # It used to be 0.040, from three readings: `ystar`'s rule of a trend
    # innovation sd around 8% of the observed variation in the series it trends,
    # so 8% of sd(du) = 0.300 giving 0.024; `nairu`'s realised sd(dNAIRU) of
    # 0.032; and a ceiling from the 2012Q4-2015Q4 inflation-band test, which is
    # what pushed the compromise up from 0.024 to 0.040.
    #
    # All three were derived on the driftless random walk, where the innovation
    # had to carry the whole 3.7pp decline in u* over the sample. Under
    # convergence the mechanism carries that and the innovation carries only
    # deviations from it, so the calibration no longer describes the same job.
    # The model says so itself: at 0.040 the realised innovation sd was 0.021,
    # about half the allowance, and it was spending that allowance in the wrong
    # place — 16% of it across 1993-1999 against 53% across 2000-2019.
    #
    # What 0.040 bought was late wandering: u* drifted +0.53 above its own
    # equilibrium path in 2014 and -0.30 below it recently, which put the
    # endpoint at 4.54 against an estimated equilibrium of 4.68. At 0.020 those
    # deviations fall to +0.16 and -0.09, and the endpoint lands at 4.74 with
    # the equilibrium at 4.78.
    #
    # The trade is visible and should be read before trusting it. A less mobile
    # u* means a larger, more persistent unemployment gap, so `gamma_pi`
    # flattens from -1.48 to -1.15, and in the joint model `sigma_v` rises as
    # work is pushed onto the free gap component. And the tighter u* is, the
    # more of it is the convergence mechanism rather than the data: see the
    # "what moves u*" chart, where the data's share of the total fall drops
    # from 3% to about 1%.
    #
    # Still outstanding: the 2012Q4-2015Q4 ceiling test has not been re-run
    # under convergence, and it is the argument that pushed the old number up.
    # If it no longer binds, 0.024 from `ystar`'s rule is the only remaining
    # anchor and 0.020 sits just below it.
    # Imposed, and the single most consequential setting in the model: u* runs
    # 5.04 to 4.58 across 0.024 to 0.065, with the gap moving -0.69 to -0.23.
    #
    # 0.040 is a compromise between three readings. `ystar`'s rule — a
    # trend innovation sd around 8% of the observed variation in the series it
    # trends, so 8% of sd(du) = 0.300 — implies 0.024. The NAIRU model's
    # realised sd(dNAIRU) of 0.032 implies about the same. Both of those give a
    # u* that is a near-straight glide.
    #
    # The ceiling comes from the inflation-band chart. Across 2012Q4-2015Q4,
    # a stretch when inflation sat *below* the RBA band and so was signalling
    # genuine slack, u* should not be rising. It changes by -0.14, -0.13, -0.07,
    # +0.01 and +0.11 across 0.024 to 0.065, so the sign flips between 0.040 and
    # 0.050 and anything looser books part of the post-mining-boom rise in
    # unemployment as structural. 0.040 is the loosest setting that passes.
    # `beta_pi` also falls monotonically, 0.80 at 0.024 to 0.35 at 0.065, as a
    # freer u* crowds out the de-anchoring term.
    sigma_ustar: float = 0.020

    # Estimate sigma_ustar under a constrained prior instead of fixing it.
    #
    # Not a free estimate, and it cannot be one. `ystar` can free
    # `sigma_ystar` because in its `inflation` spec potential is a residual, so
    # once c and g are known the innovation is directly observable. Here u* is
    # a free state sitting beside a free Okun residual, which is exactly the
    # Stock-Watson pile-up pair: the likelihood cannot apportion a wiggle in
    # unemployment between a shift in u* and a residual.
    #
    # What a constrained prior buys is not identification but honesty about the
    # band. Fixing sigma_ustar reports a posterior for "where is u* given that
    # it drifts at exactly this rate", and the sweep shows that question's
    # answer moving 0.39pp across 0.024-0.05 while the drawn band stays 0.40pp
    # wide at every setting. Integrating over the prior folds that back in.
    #
    # Expect the posterior to be substantially prior-driven. That is the point,
    # not a defect: the prior is where the belief "u* is slow moving" lives.
    free_sigma_ustar: bool = False
    # (mu, sigma, lower, upper) for the TruncatedNormal. Centred between the two
    # anchors available: 0.024 from ystar's 8% rule, and 0.032 from the
    # NAIRU model's realised sd(dNAIRU).
    #
    # The upper bound is doing the work, and it is a bound rather than a belief
    # about the centre. Unbounded (upper=None) the posterior runs to 0.131,
    # some 8 prior sd above the mean, because the likelihood prefers more state
    # variance without limit: a u* that tracks unemployment fits better quarter
    # by quarter. Bounded, expect the posterior to pile up against `upper`.
    # That pile-up is the diagnostic, and it means the bound is the
    # specification. What it buys over a fixed value is a u* band that
    # integrates over the range rather than being conditional on one number.
    sigma_ustar_prior: tuple[float, float, float, float | None] = (0.03, 0.012, 0.005, 0.05)

    # --- Output ---
    output_dir: Path | None = None

    def __post_init__(self) -> None:
        """Validate the specification switches."""
        if self.ustar_drift and self.ustar_converge:
            raise ValueError(
                "ustar_drift and ustar_converge are two stories about the same fact; "
                "pick one",
            )
        if self.gap_source not in GAP_SOURCES:
            raise ValueError(f"gap_source must be one of {GAP_SOURCES}, got {self.gap_source!r}")

    @property
    def constants(self) -> dict[str, float]:
        """The imposed settings, recorded on the model for the run log.

        Empty when `sigma_ustar` is estimated: it is then a parameter with a
        posterior, and listing it as an imposed constant would misreport the
        run in the log and in the diagnostics.
        """
        constants = {"anchor": self.anchor}
        if not self.free_sigma_ustar:
            constants["sigma_ustar"] = self.sigma_ustar
        return constants
