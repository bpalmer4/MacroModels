"""Configuration for the joint y*/u* model.

Three states, three observation equations:

    g_t   = g_{t-1} + e_g                        sigma_g imposed
    y*_t  = y*_{t-1} + g_{t-1} + e_y             sigma_ystar imposed
    u*_t  = u*_{t-1} + e_u                       sigma_ustar imposed

    gap_t = c·(pi_ann_t - anchor) + v_t          v ~ N(0, sigma_v)

    log_gdp_t = y*_t + gap_t + e_c               e_c ~ N(0, sigma_e)
    u_t       = u*_t - beta·gap_t + e_o          e_o ~ N(0, sigma_o)
    pi_q_t    = q(anchor) + beta_pi·[q(pi_exp_t) - q(anchor)]
                + gamma·(u_t - u*_t)/u_t
                + rho·d4pm_t + xi·GSCPI_t²·sign(GSCPI_t) + e_p

Every prior and every imposed variance is taken unchanged from `ystar` and
`ustar`, so that differences in the posterior are attributable to joint
estimation and to `v`, not to a re-tuning.

**Two inflation series, deliberately.** The gap is defined on the four-quarter
trimmed mean, as in `ystar`, where inflation is not a regressor and "at target"
is an annual concept. The Phillips curve is estimated on the quarterly rate, as
in `ustar`, where inflation *is* the dependent variable and overlapping
four-quarter errors would be indefensible. Using one series for both would break
one of those two arguments.
"""

from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_OUTPUT_DIR = Path(__file__).parent.parent.parent.parent / "model_outputs"
CHART_DIR = Path(__file__).parent.parent.parent.parent / "charts" / "YStarUStar"

# Inherited from ystar: the pandemic quarters that carry no likelihood.
DEFAULT_EXCLUDE_WINDOW = ("2020Q2", "2021Q3")

# Which equations the exclusion window is applied to.
#   "all"  — GDP, Okun and Phillips (default)
#   "gdp"  — GDP only, matching what `ystar` does on its own
EXCLUDE_SCOPES = ("all", "gdp")

# Which trimmed mean horizon defines the gap. See `ModelConfig.gap_pi_basis`.
GAP_PI_BASES = ("annual", "quarterly")

# How the output gap is specified. See `ModelConfig.gap_spec`.
#   "defined" — gap = c·(pi - anchor) + v, ystar's identity plus a free part
#   "cycle"   — gap is a free AR(1) latent, and inflation observes it
GAP_SPECS = ("defined", "cycle")


@dataclass
class ModelConfig:
    """Sample, imposed variances, and the switches that make the model testable."""

    # How the gap is specified, and the more important of the two switches here.
    #
    # "defined" is the original: gap = c·(pi_ann - anchor) + v. It keeps ystar's
    # identity and bolts a free component onto it.
    #
    # "cycle" makes the gap a free AR(1) latent that three equations observe:
    # GDP, Okun, and a Phillips curve written on the gap rather than on the
    # unemployment gap. It is the same information, differently parameterised,
    # and it fixes three things the "defined" form got awkward.
    #
    # It removes the circularity outright. Under "defined", inflation both
    # constructs the Phillips curve's regressor (via the gap, via Okun) and is
    # its dependent variable, which manufactures a negative slope. Under
    # "cycle" inflation appears exactly once, as a dependent variable observing
    # a latent it does not define.
    #
    # It removes the two-horizon problem with it. "defined" needs the annual
    # rate for the gap and the quarterly rate for the Phillips curve, because
    # inflation is a regressor in one place and a dependent variable in
    # another. Here it is only ever a dependent variable, so the quarterly rate
    # is right everywhere and `gap_pi_basis` does not apply.
    #
    # And it stops the model implying inflation has a privileged claim on
    # defining the gap. Under "defined" the natural summary is "x% of the gap
    # is explained by inflation", which turned out to be an accounting
    # convention rather than a result: it reads 45% on the annual basis and 32%
    # on the quarterly one while the gap itself barely moves.
    #
    # What motivated the switch empirically: under "defined", `v` comes back
    # with a lag-1 autocorrelation of 0.913 despite an iid prior. The model is
    # already estimating a persistent cycle; "cycle" says so directly.
    gap_spec: str = "defined"

    # --- Sample ---
    start: str | None = "1993Q1"
    end: str | None = None
    anchor: float = 2.5

    # Which trimmed mean series defines the gap: "annual" (four-quarter, as
    # `ystar`) or "quarterly" (the quarterly rate x 4, so it is on the anchor's
    # scale). The Phillips curve is always estimated on the quarterly rate,
    # because there inflation is the dependent variable and four-quarter errors
    # overlap three quarters in four.
    #
    # This is the one specification choice the joint model has to make that
    # neither parent faced. In `ystar` inflation is only ever a regressor, so
    # overlapping observations cost nothing and "at target" is an annual
    # concept. In `ustar` it is only ever a dependent variable, so the
    # quarterly rate is forced. Here it is both, and the two horizons are what
    # keep the Phillips curve's regressor from containing its own dependent
    # variable exactly.
    #
    # Measured on the 1993Q1 sample: corr(pi_q_t, pi_ann_t) = 0.828, with
    # quarterly autocorrelations of 0.73, 0.63 and 0.51. So "annual" reduces
    # the mechanical overlap to 0.83 of what "quarterly" would give, where
    # "quarterly" makes it exactly 1.0 — the gap's inflation content would be
    # the Phillips curve's left-hand side, same series, same quarter. A 17%
    # reduction is a thin margin and not a solution, so it was never much of a
    # defence. The default is "quarterly", matching the basis the Phillips
    # curve is forced onto and the one `ystar` is moving to. The cost is that
    # the overlap goes to 1.0; the measured consequences are that the free
    # component's share of the gap rises from 45% to 65% and `c` falls from
    # 0.376 to 0.283, while the gap itself is unchanged (sd 0.545 against
    # 0.544, and the same 0.354 correlation with NAB business conditions).
    # "annual" remains available and is the comparison to run.
    gap_pi_basis: str = "quarterly"

    # --- Imposed variances, all inherited ---
    # sigma_c and the two ratios come from `ystar`'s scale block, so
    # sigma_ystar = 0.13 x 0.60 = 0.078 and sigma_g = 0.025 x 0.60 = 0.015 at
    # the defaults. sigma_ustar comes from `ustar`, where the sweep across
    # 0.024 to 0.065 moves u* by 0.46pp and 0.040 is the loosest setting that
    # does not book the post-mining-boom rise in unemployment as structural.
    #
    # None of the three can be estimated. That is not a limitation of this
    # package: a free state beside a free residual is the Stock-Watson pile-up
    # pair, and `ustar` demonstrated all three routes failing (free prior runs
    # to 0.131, bounded prior pins to its bound). Imposing them is the
    # operational content of "these things move slowly".
    sigma_c: float = 0.60
    ratio_ystar: float = 0.13
    ratio_g: float = 0.025
    sigma_ustar: float = 0.020

    # --- The free gap component: the reason this model exists ---
    # gap_t = c·(pi_t - anchor) + v_t, with v iid N(0, sigma_v).
    #
    # `sigma_v` is what the joint likelihood buys. In `ystar` alone the model is
    # log_gdp = y* + c·d + v + e_c, where v and e_c are two mean-zero terms in
    # one equation: only Var(v) + Var(e_c) is visible and the split is a flat
    # ridge. Adding the Okun equation puts v in a second place, as -beta·v, so
    #
    #     cov(GDP residual, unemployment residual) = -beta·Var(v)
    #
    # and the split is identified. That single covariance is the entire
    # informational gain from joining the two models.
    #
    # Expect it to be weakly identified even so. Both residuals are large
    # (`ystar` reports sigma_e = 0.508, `ustar` sigma_okun = 0.685) and the
    # moment is a covariance between them, so the posterior may sit close to
    # the prior. `--sweep-sigma-v` exists to tell that case apart from a real
    # answer: if the posterior tracks the prior, it is not identified here
    # either and the joint model has failed at the one thing it was built for.
    #
    # iid rather than AR(1), on purpose and conservatively. A persistent v is
    # more plausible economically, since business cycles persist, but a free
    # AR(1) cycle state beside two free trends is three places for unexplained
    # persistence to hide, which is how a joint model turns into a
    # shock-allocation machine. iid understates the cycle it can find, so a
    # large sigma_v under iid is strong evidence; a small one is not proof of
    # absence.
    free_gap_component: bool = True
    # Fix sigma_v instead of estimating it. Used by the prior sweep, and by
    # sigma_v = 0, which recovers `ystar`'s identity exactly.
    sigma_v: float | None = None
    # Prior sd for the HalfNormal on sigma_v. In log x 100 units, the same
    # units as sigma_e, which `ystar` estimates at 0.508.
    sigma_v_prior: float = 1.0

    # --- Inherited switches ---
    # Give `c` a two-sided Normal(0, 2) prior instead of the HalfNormal, so the
    # posterior can place mass on a negative conversion factor. The default
    # imposes the sign, which means "c is clear of zero" is not evidence for
    # the premise. Same argument as `ystar.ModelConfig.two_sided_c`.
    two_sided_c: bool = False
    # A one-sided prior on beta would assert Okun's law rather than test it.
    two_sided_beta: bool = True

    # Prior sd for beta_okun, whose mean stays at 0.5 (textbook Okun) as
    # inherited from `ustar`. The sd is a switch because 0.5 turned out to be
    # doing work it should not: when `sigma_okun` is large the unemployment
    # data pin the slope loosely and the prior pulls beta toward 0.5, a value
    # this sample rejects at three standard deviations. Measured on the
    # quarterly default, beta_okun runs 1.396 in `sigma_okun`'s bottom quartile
    # against 1.252 in its top, a swing of 0.144 against beta's own posterior
    # sd of 0.21. Widening the prior tests whether that matters.
    beta_okun_prior_sd: float = 0.5

    # The Okun residual sd, IMPOSED rather than sampled. None restores the free
    # version, which is the comparison this value rests on.
    #
    # **Why it is imposed.** Free, it is the model's one bad parameter: ess_bulk
    # of 50 with r_hat 1.08, four chains peaking in four different places, and
    # 45 divergences. It trades off against sigma_e at a correlation of -0.55,
    # and that ridge is what the sampler crawls along. Pinning one end dissolves
    # it: 0 divergences, max r_hat 1.000, minimum ess 2,553, and sigma_e's
    # ess_bulk goes from 280 to 20,839. That is what makes 10,000 draws enough
    # where the free version needed 24,000 and still reported r_hat 1.08.
    #
    # **Why it is safe.** Nothing reported depends on it. Free against fixed at
    # 0.20: c 0.284 -> 0.285, sigma_v 0.492 -> 0.495, beta_okun 1.333 -> 1.343,
    # gamma_pi -1.021 -> -1.024. Doubling the imposed value to 0.40 moves c by
    # +0.005 and sigma_v by -0.009; beta_okun shifts -0.113, about half a
    # posterior sd, and sigma_e absorbs the rest, falling to 0.404 exactly as
    # the -0.55 correlation predicts.
    #
    # **The objection, stated rather than hidden.** 0.20 is this model's own
    # posterior mean, so imposing it is circular in a way the package's other
    # imposed variances are not: `ystar`'s sigma_ystar rests on an 8%-of-variation
    # rule, `ustar`'s sigma_ustar on the 2012-15 inflation-band test, and both
    # are swept. This one has no external anchor. What defends it is not the
    # value but the insensitivity to it, and the honest reading is that
    # sigma_okun is a nuisance parameter the data do not determine and the
    # answers do not need. `ustar`'s 0.685 is not a candidate: it was
    # conditional on a frozen gap and this model's free posterior excludes it.
    sigma_okun: float | None = 0.20

    # Let u* drift while inflation expectations sit above target.
    #
    #     u*_t = u*_{t-1} - lambda·max(0, pi^e_{t-1} - anchor) + e_u
    #
    # Off, u* is a driftless random walk, and that prior is badly at odds with
    # the sample. Fitted, u* falls 8.46 to 4.69, which is 3.77pp over 134
    # quarters; a driftless walk at sigma_ustar = 0.040 puts the sd of the total
    # change at 0.46pp, so the fitted path is **8.2 standard deviations** out.
    # The increments use 92% of the allowed per-quarter movement and three
    # quarters of that is directed rather than random. The prior is not stretched,
    # it is overwhelmed.
    #
    # What it costs is visible at the start of the sample. To begin where
    # unemployment actually was in 1993Q1, 10.93, and still reach 4.69 needs
    # 6.24pp, or 13.5 sd, which the prior cannot afford. So the posterior
    # compromises by starting u* at 8.46 and booking the remaining +2.47pp in the
    # Okun residual. `ustar`'s Limitation 4 records the symptom without naming
    # this as the cause.
    #
    # **Why expectations rather than a date or a constant.** A constant drift
    # says the NAIRU falls forever, which is wrong at the endpoint. A date break
    # is arbitrary. Excess expectations are an observable that measures the thing
    # that would actually move a NAIRU: a target people do not yet believe, so
    # wage-setting has not adapted to it. The series says the regime took until
    # about 1998 to become credible — excess over target averages +0.64 across
    # 1994-1996 against +0.07 across 2000-2019 — even though realised trimmed
    # mean inflation was already 2.1% in 1993Q1. Credibility and realised
    # inflation are different things and only the first should move u*.
    #
    # **The side effect to watch.** Excess expectations also reach +0.87 in
    # 2022-23, so this drift will push u* down then too. That episode was a
    # supply shock rather than a regime transition, and the mechanism does not
    # obviously carry over. Read the post-2021 path with that in mind, and
    # compare against `ustar_drift = None`.
    #
    # `lambda` is estimated under a two-sided prior, so the sign is tested rather
    # than asserted. It is a slope on an observable, not a variance, so it
    # carries no pile-up risk, and being tied to an exogenous series it cannot
    # simply track unemployment.
    ustar_drift: bool = False

    # u* converges to an estimated equilibrium rather than wandering:
    #
    #     u*_t = u*_{t-1} + phi·(u*_eq - u*_{t-1}) + e_u
    #
    # Backported from `ustar`, where the driftless walk it replaces turned out to
    # be about 8 standard deviations from its own fitted path over the sample and
    # 17 over 1993-1999, to put u* below unemployment in all 16 quarters of
    # 1994-1997 while the trimmed mean broke above 3%, and to leave a +2.5pp gap
    # one year after the deepest recession since the 1930s. There it cut
    # `sigma_okun` from 0.685 to 0.426 and improved sampling. Its notes carry the
    # evidence, including the external wage check that does *not* favour it.
    #
    # Mutually exclusive with `ustar_drift`.
    ustar_converge: bool = True

    # --- The pandemic window ---
    # `ystar` drops 2020Q2-2021Q3 from its likelihood on the ground that
    # potential output is not well defined in a lockdown. Here the same window
    # is dropped from all three equations by default, which is the consistent
    # extension of that argument rather than a new one: if inflation in those
    # quarters was moved by free childcare and administered fuel prices, the
    # gap built from it is not a gap, and the Okun equation would be fitting
    # unemployment to a number that means nothing. Unemployment itself was
    # distorted too, by JobKeeper holding measured unemployment far below any
    # reasonable reading of labour market slack.
    #
    # `--exclude-scope gdp` restricts the exclusion to the GDP equation, which
    # is literally what `ystar` does when run alone. That is the comparison to
    # make if you want this model's `c` to be directly comparable with
    # `ystar`'s, and it is not the default because it leaves the Okun equation
    # fitting a meaningless regressor in six quarters.
    exclude_window: tuple[str, str] | None = DEFAULT_EXCLUDE_WINDOW
    exclude_scope: str = "all"

    # --- Diagnostics ---
    # Drop the Phillips curve, leaving GDP and Okun. u* is then a trend through
    # unemployment rather than a NAIRU, so this is not a candidate
    # specification. It is the clean measurement of `sigma_v`: with inflation
    # out of the likelihood as a dependent variable, nothing can be accused of
    # choosing `c` to make inflation explain itself.
    include_phillips: bool = True
    # Drop the Okun equation, leaving GDP and Phillips. Recovers `ystar` plus a
    # free v that is then unidentified, so it is a check that the covariance
    # really is what identifies sigma_v: this run should return the prior.
    include_okun: bool = True

    output_dir: Path = field(default_factory=lambda: DEFAULT_OUTPUT_DIR)

    def __post_init__(self) -> None:
        """Validate the switches that have a fixed set of legal values."""
        if self.exclude_scope not in EXCLUDE_SCOPES:
            raise ValueError(
                f"exclude_scope must be one of {EXCLUDE_SCOPES}, got {self.exclude_scope!r}",
            )
        if self.gap_spec not in GAP_SPECS:
            raise ValueError(f"gap_spec must be one of {GAP_SPECS}, got {self.gap_spec!r}")
        if self.gap_spec == "cycle" and not self.include_phillips:
            raise ValueError(
                "the cycle spec needs the Phillips curve: it is the only equation tying the "
                "gap to inflation, and without it nothing makes u* a NAIRU rather than a "
                "trend through unemployment",
            )
        if self.gap_pi_basis not in GAP_PI_BASES:
            raise ValueError(
                f"gap_pi_basis must be one of {GAP_PI_BASES}, got {self.gap_pi_basis!r}",
            )
        if not self.include_okun and not self.include_phillips:
            raise ValueError(
                "dropping both Okun and Phillips leaves only the GDP equation, which is "
                "`ystar` with an unidentified extra variance — use ystar instead",
            )
        if self.ustar_drift and self.ustar_converge:
            raise ValueError(
                "ustar_drift and ustar_converge are two stories about the same fact; "
                "pick one",
            )
        if self.sigma_okun is not None and self.sigma_okun <= 0:
            raise ValueError(f"sigma_okun must be positive, got {self.sigma_okun}")
        if self.sigma_v is not None and self.sigma_v < 0:
            raise ValueError(f"sigma_v must be non-negative, got {self.sigma_v}")
        if self.sigma_v is not None and not self.free_gap_component:
            raise ValueError("sigma_v is meaningless with free_gap_component off")

    @property
    def scale_constants(self) -> dict[str, float]:
        """The fixed variance scale and ratios, in `ystar`'s scale_equation form."""
        return {
            "sigma_c": self.sigma_c,
            "ratio_ystar": self.ratio_ystar,
            "ratio_g": self.ratio_g,
        }

    @property
    def constants(self) -> dict[str, float]:
        """Imposed values worth recording in the trace metadata."""
        recorded = {"sigma_ustar": self.sigma_ustar, "anchor": self.anchor}
        if self.sigma_v is not None:
            recorded["sigma_v"] = self.sigma_v
        return recorded
