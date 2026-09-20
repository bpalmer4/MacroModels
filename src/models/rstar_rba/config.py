"""Configuration for the rstar_rba model.

    pi_t       = (4/w) · sum_{j<w} q_{t-j}           annualised, w-quarter average
    r_t - b_t  = lambda · (pi_t - anchor) + eps_t    the two gaps, proportional

`b_t` is NEUTRAL, the slow nominal cash rate; real neutral is `b_t - anchor`.
`b_t + lambda · (pi_t - anchor)` is the rule's PRESCRIBED rate, which carries
the inflation response on top of neutral and is not itself neutral.

The target is not cosmetic here. In a specification without it the constant is
absorbed by `b_t` and the two are observationally equivalent, so subtracting it
would be a re-centring. Written as two gaps it defines the point at which both
are simultaneously zero, which is what makes `b_t` interpretable.
"""

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.paths import MODEL_OUTPUTS

DEFAULT_OUTPUT_DIR = MODEL_OUTPUTS

WEIGHT_SCHEMES = ("flat", "geometric", "dirichlet")

# What times the jump flags. See `ModelConfig.jump_source`.
JUMP_SOURCES = ("world", "gdp", "gdp4")
# Cleveland Fed 10-year expected real rate, the world anchor `rstar_bonds`
# also uses. Read from FRED here, not from that model's output.
WORLD_REAL_RATE = "REAINTRATREARAT10Y"

# Below 2 degrees of freedom a StudentT has no variance, so the base would
# stop being a random walk with a scale.
MIN_NU = 2.0
PERCENTILE_MAX = 100.0


@dataclass
class ModelConfig:
    """Specification and sample."""

    # 1993Q1, the start of inflation targeting, and matching the rest of the
    # package. Before that the anchor is not 2.5 and the equation is meaningless.
    start: str | None = "1993Q1"
    end: str | None = None

    # How many quarters of inflation the RBA is taken to respond to.
    #
    # w = 1 is the raw quarterly print annualised, and it is the wrong object:
    # a single print has an annualised sd of 1.12 and correlates 0.09 with the
    # cash rate, because the Bank does not respond to one quarter of noise.
    # Averaging is the specification rather than a smoothing convenience, which
    # is what "long and variable lags" means. The correlation rises to about
    # 0.21 at w = 4 and 0.26 at w = 8.
    #
    # Chosen a priori at four quarters, not by maximising the correlation:
    # picking w off the data would be a specification search over the one number
    # the model is meant to estimate. Only used when `weights` is "flat".
    window: int = 4

    # --- How the past is weighted ---
    # A flat window asserts that the RBA weights the last w quarters equally and
    # everything before them at zero, which nobody believes. These estimate the
    # shape instead.
    #
    #   "flat"       w_j = 1/window for j < window, 0 after. The comparator.
    #   "geometric"  w_j proportional to rho^j, normalised. One parameter, and
    #                `rho` is the memory: the mean lag is rho/(1-rho) quarters,
    #                so 0.5 is one quarter, 0.8 is four, 0.9 is nine.
    #   "dirichlet"  free weights over the lags, summing to one. No shape
    #                imposed at all, and the most likely to be unidentified:
    #                the two gaps correlate about 0.2, which is not much to
    #                recover a whole distribution from.
    #
    # `rho` and `lambda` are entangled either way. A longer memory smooths `pi`
    # and shrinks its variance, so `lambda` scales up to compensate. What
    # separates them is the shape of the covariation across lags rather than its
    # size. If `rho` comes back at its prior, the data could not see the shape.
    weights: str = "geometric"
    # Four quarters, set by judgement and not estimated, because it cannot be.
    #
    # The truncation was 12 and that turned out to be the binding choice rather
    # than a harmless bound. Across max_lag of 4, 6, 8 and 12, `rho` barely
    # moves (0.87 to 0.89) while the mean lag scales with the window (1.72,
    # 2.50, 3.26, 4.10, roughly max_lag/3) and `sigma_eps` is flat to three
    # decimals (0.786, 0.776, 0.768, 0.770). So the data cannot distinguish
    # them, and yet `lambda` runs 0.61 to 0.96 across the same range. The memory
    # was being set by the truncation.
    #
    # The reason is that `rho` = 0.87 implies an untruncated mean lag of
    # rho/(1-rho) = 6.6 quarters, so every window tried cuts the geometric tail
    # and then renormalises. Rather than pretend to estimate it, the window is
    # now a stated judgement: a central bank responding to about a year of
    # inflation. `lambda` should be quoted as a range across this choice, not
    # as a point.
    max_lag: int = 4
    # Beta(2, 2) is flat-ish on (0, 1) with a little pull off the boundaries,
    # so it does not smuggle in a memory length.
    rho_a: float = 2.0
    rho_b: float = 2.0
    # 1.0 is uniform over the simplex: every weighting equally likely a priori.
    dirichlet_alpha: float = 1.0

    # --- The second target ---
    # The RBA's mandate is inflation AND full employment. With only the
    # inflation term, any policy driven by the labour market has nowhere to go:
    # if persistent it lands in the base and is reported as neutral falling, if
    # brief it lands in the residual and is reported as timing. Neither is what
    # it was. On:
    #
    #     r_t = b_t + lambda x g_t + lambda_u x (u_t - u*_t) + eps_t
    #
    # `lambda_u` is expected NEGATIVE: unemployment above u* is slack, and the
    # RBA cuts. Its prior is centred there but wide enough to reach zero and to
    # cross it, so "the labour market added nothing" remains available.
    #
    # THE COST IS REAL. u* is not data, it is the posterior median from
    # `ystar_ustar`, so this model stops being estimable from published series
    # and inherits that model's conditioning, including its imposed sigma_okun.
    # Different in kind from the `jumps` dependency, which only times a
    # variance and is itself off by default now. Default off for that reason.
    #
    # OFF, AND IT STAYS OFF. Built, run, and rejected on collinearity, not on
    # principle: the omission is real and the fix is worse.
    #
    #                   lambda_pi              lambda_u               sigma_eps
    #   inflation only  0.303                  -                      0.775
    #   two targets     -0.011 (-0.14, +0.15)  -1.228 (-1.52, -0.93)   0.629
    #
    # `lambda_u` comes back correctly signed, sharp, and improves the fit more
    # than anything else tried. But `lambda_pi` goes to zero and `rho` falls
    # apart (0.869 to 0.473, spanning 0.06 to 0.95), so the lag weights stop
    # being identified too. Read literally it says the RBA ignored inflation,
    # which is not credible. The two gaps correlate -0.67 in sample and the two
    # coefficients correlate +0.71 across draws: Australian inflation and
    # unemployment move together closely enough over 1993-2026 that the data
    # cannot say which one policy tracked, so the second target takes
    # everything for fitting marginally better.
    #
    # Weighting the two responses by hand would suppress that, and is just
    # choosing both coefficients and letting the sampler decorate them. A tight
    # prior on `lambda_pi` is the same choice in Bayesian dress.
    #
    # THE USEFUL FINDING IS ABOUT NEUTRAL. It barely moves: 3.01 to 3.06, with
    # corr(neutral, cash rate) 0.89 to 0.88. So the inflation-only
    # rule was NOT parking the employment response in neutral, which was the
    # worry that prompted this. What it was doing is loading both responses
    # into `lambda`. Quote `lambda` as the response to inflation AND the labour
    # market together, never to inflation alone.
    employment: bool = False
    lambda_u_mu: float = -0.5
    lambda_u_sigma: float = 1.0
    # Prefix of the completed `ystar_ustar` run supplying u - u*.
    ustar_prefix: str = "ystar_ustar"

    # --- The second window: the market's 5y5y forward ---
    # ON. With the cash rate alone the LEVEL of neutral is the historical
    # average cash rate less the average inflation response — an identity, not a
    # choice, and the reason this model reported 2.99 against CBA's 3.85 while
    # being structurally unable to say the whole level had shifted.
    #
    # The AOFM 5y5y risk-neutral forward is the one series that speaks to
    # neutral's level without being the cash rate's own history. It leads policy
    # by two to three quarters and does not chase it, and it carries half the
    # cash rate's volatility. `--no-forward` restores the one-window model.
    use_forward: bool = True
    forward_method: str = "bc"
    # The bias is what the forward carries that neutral does not: the market's
    # view of the cycle over years five to ten, plus any premium AOFM left in.
    # It now holds the level, so its prior IS the assertion. Centred on zero and
    # deliberately tight: widen it and the level goes back to being unidentified.
    forward_bias_mu: float = 0.0
    forward_bias_sigma: float = 0.50
    sigma_f_sigma: float = 1.0

    anchor: float = 2.5

    # Half-width of the target band, used to scale the gap:
    #
    #     g_t = (pi_t - anchor) / band
    #
    # so |g| = 1 at the edge of the 2-3 per cent band. That is the pivot of the
    # power function: |g|^2 damps the response inside the band and amplifies it
    # outside, with the crossover at the policy boundary rather than at an
    # arbitrary unit. Without the scaling a convex term damps most of the sample,
    # since |pi - 2.5| is below one in the large majority of quarters, so
    # `lambda_2 > 0` would be asked to amplify and damp at the same time.
    #
    # With only one power term and `kappa` fixed the scaling is absorbed into
    # `lambda` and is pure interpretation: `lambda` becomes the response at the
    # band edge. With the linear and quadratic terms both present it is not
    # absorbed, because it changes their relative weight.
    band: float = 0.5

    # --- Is the response linear in the gap? ---
    # Off: r_t - b_t = lambda_1 · g_t.
    # On:  r_t - b_t = lambda_1 · g_t + lambda_2 · g_t · |g_t|.
    #
    # `g·|g|` is the quadratic term with the sign kept, so the response stays
    # odd-symmetric: a two-point overshoot and a two-point undershoot get equal
    # and opposite treatment, but both get more than twice a one-point gap.
    # `lambda_2 = 0` is exactly the linear model, so its posterior is a direct
    # test rather than an assumption.
    #
    # There is a reason to expect it to fit. The linear version is dragged by
    # the many quarters where the gap is small and the RBA did nothing;
    # convexity lets those be ignored and hands the identification to the
    # episodes where the gap was large, which is what a central bank with a
    # tolerance band actually does.
    #
    # Two warnings. The gap exceeds two points in only two episodes in the whole
    # sample, 2008 and 2022-23, so `lambda_2` is identified almost entirely by
    # those: a two-observation finding wearing time-series clothes. And 2022-23
    # begins with the cash rate pinned at 0.10, so the early rate gap is
    # mechanically compressed however large inflation got, which biases
    # `lambda_2` down and may hide real convexity.
    nonlinear: bool = False
    lambda2_mu: float = 0.0
    lambda2_sigma: float = 0.5

    # --- The effective lower bound ---
    # Quarters where the cash rate sits at or below this are dropped from the
    # likelihood. They are not generated by the rule: whatever response the
    # inflation gap called for, the delivered one stopped at the floor.
    #
    # This matters most for curvature. `lambda_2` is identified almost entirely
    # by quarters far from the target, since near g = 0 the linear and quadratic
    # terms are indistinguishable, and in this sample every far-from-target
    # quarter sits in or beside the floor episode. Through 2020-21 the gap ran
    # to -1.78 band-widths with the cash rate pinned at 0.10, so a large gap
    # meets a truncated response, which is exactly the shape a concave fit
    # picks up. `lambda_2` has come back negative in every specification tried,
    # and this is the first test of whether that is behaviour or censoring.
    #
    # Note this is a different exclusion rule from the rest of the package.
    # `ystar` and `ystar_ustar` drop 2020Q2-2021Q3 because potential output is
    # not well defined in a lockdown; this drops on the constraint that binds
    # here, which ran about two quarters longer. The windows overlap but are
    # not the same claim.
    #
    # It does not fix everything. Through 2022-23 the cash rate was free to move
    # but was travelling four points from the floor, so the rate gap is small
    # early in the largest inflation gap of the sample. That is adjustment speed
    # rather than a bound, and dropping quarters cannot repair it. The honest
    # fix is to model the censoring, which this is not.
    #
    # None disables the exclusion, and is the default: dropping the floor
    # quarters was tried and did not do what it was meant to. `lambda_2` went
    # from -0.056 to -0.071, slightly more negative rather than less, so the
    # concavity is not censoring. What remains is that through 2022-23 the cash
    # rate was free to move but was travelling four points from a standing
    # start, so the rate gap is small in exactly the quarters where the
    # inflation gap is largest. That is adjustment speed, not a bound, and
    # dropping quarters cannot repair it: rate smoothing would.
    floor: float | None = None

    # --- How neutral is allowed to move ---
    # False: neutral is one constant, so the model is a regression and `lambda` is
    # identified by covariation alone. Start here. It cannot tell you whether
    # neutral has fallen, only what it averaged.
    #
    # True: neutral `b_t` drifts as a Gaussian random walk. This is what anyone
    # wants from the model, and it is where the identification gets difficult:
    # the walk and `lambda` compete to explain the same downward drift in the
    # cash rate since 1993, and with `sigma_r` free the walk wins outright. In
    # the real-rate version of this model that collapse was total, with
    # `sigma_eps` going to 0.045 and neutral correlating 1.00 with the cash rate.
    #
    # True by default, and the alternative is REJECTED rather than merely
    # disliked. With `walk=False` the residual is a trending near-unit-root
    # series, lag-1 autocorrelation 0.974 against 0.857, a trend of -1.54 a
    # decade, and era means running +2.05 (1994-99) to -1.99 (2016-26), while
    # `sigma_eps` goes 0.786 -> 1.930 and the residual sd of 1.91 is essentially
    # the cash rate's own 1.98. The likelihood says the residual is iid; it
    # plainly is not. See "The fixed-neutral alternative is rejected by its own
    # residual" in MODEL_NOTES.md.
    walk: bool = True
    # Imposed when `walk` is on, for the reason above. Swept, never estimated.
    #
    # 0.10 IS ARBITRARY. It is a round number inside the defensible band of
    # roughly 0.05 to 0.20, chosen so the model has a default, and it is not the
    # value the data prefers because the data does not prefer one. The ensemble
    # still runs by default: the range is the result.
    #
    # 0.125 SINCE 2026-09-16, up from 0.10, BECAUSE THE SECOND WINDOW FREED IT.
    # With the cash rate as the only observable, `sigma_r` did two jobs at once:
    # it set the path's smoothness AND implicitly rationed how much of the cash
    # rate's movement could be called neutral. The level rode on it, moving
    # -0.05 to 1.05 real across the sweep — wider than the credible interval at
    # any single value.
    #
    # With the AOFM 5y5y forward pinning the level, those jobs separate. Across
    # the same sweep real neutral now runs 1.10 to 1.41, a spread of 0.31
    # against 1.10, and from 0.10 upward it is nearly flat: 3.85, 3.91, 3.91
    # nominal at 0.10, 0.15, 0.20. `lambda` settles too, 0.552 -> 0.470 -> 0.449
    # -> 0.446 per pp, converging rather than drifting.
    #
    # So the model is less constrained than it was and can afford a looser walk.
    # The gain is that neutral's quarterly volatility moves toward the 0.155 and
    # 0.153 that `rstar_bonds` and `rstar_tvpvar` independently produce, where
    # at 0.10 this model was the slowest-moving neutral in the package.
    #
    # WHY 0.125 AND NOT 0.15. 0.15 fails two sampling checks: MCSE/sd 0.056
    # against 0.05, and min ESS 1,296. At 0.125 both pass, with MCSE/sd 0.026
    # and ESS 2,229.
    #
    # The culprit is `sigma_f`, the 5y5y measurement error, and the reason is
    # structural: `sigma_f` and `sigma_r` compete to explain the same thing,
    # namely how much of the gap between the forward and the fitted neutral is
    # measurement error and how much is real movement in neutral. Loosen the
    # walk and the two become harder to separate. Every other parameter samples
    # cleanly at both settings (`lambda` ESS 14,141, `rho` 12,714).
    #
    # The substance is unchanged: neutral 3.89 against 3.91 nominal, stance
    # +0.46 against +0.44, `forward_bias` -0.109 at both to three decimals. The
    # only real cost is neutral's quarterly volatility, 0.118 against 0.151, so
    # the three-way match with `rstar_bonds` and `rstar_tvpvar` is looser. That
    # was a nice-to-have, not a result the model rests on.
    sigma_r: float = 0.125
    # --- Discontinuities ---
    # Neutral normally crawls, but the world occasionally turns over and a
    # Gaussian walk cannot follow it. With `jumps` on, the base innovation is
    # StudentT rather than Normal in quarters the economy moved abruptly, so
    # neutral is permitted a step there without loosening `sigma_r` everywhere.
    #
    # The quarters are timed off real seasonally adjusted GDP growth, which is
    # EXTERNAL to this model: output never enters the rule, and timing a state's
    # volatility is not an IS curve. It does cost the package its
    # self-containment, which was deliberate, and it uses a series the RBA did
    # not have in real time and which is later revised. Accepted as the better
    # assumption: neutral genuinely can move discontinuously.
    # PERMISSION, NOT A STEP. A flag widens the tails of that quarter's
    # innovation; it does not insert a jump. At nu = 3 most of the mass is still
    # near zero, so the likelihood has to want the step before one appears. The
    # local data says where a step MAY be needed and the model decides whether
    # to take it, which is why the COVID result carries information: offered the
    # licence, the base moved 0.08 instead of 0.04 and declined the rest.
    #
    # One limit on "the model decides". It decides by likelihood, so where the
    # likelihood cannot separate the base from the response, permission is not
    # neutral: it lets the base take movement that belongs to `lambda`. That is
    # the 2022-24 problem recorded under `jump_source`.
    #
    # OFF BY DEFAULT, ON THE EVIDENCE OF ITS OWN RESULT. Offered the licence the
    # model barely uses it: `lambda` 0.305 -> 0.303, `sigma_eps` 0.786 -> 0.775.
    # Nothing measurable is bought, and three things are paid
    # for: the ABS GDP dependency (the headline otherwise needs two published
    # series and nothing else), a series the RBA did not have in real time, and
    # the admission under `jump_source` that `gdp` is the default because its
    # flags happen to miss 2022-24, which is luck rather than design. A
    # permission that changes nothing is not worth those. Kept as a sensitivity
    # test, `--jumps`, where the "neutral does not jump in Australia" result is
    # what it always was: a finding, not a default.
    jumps: bool = False
    # Which series times the flags.
    #
    #   "world"  the quarterly change in the Cleveland Fed 10-year expected real
    #            rate (FRED REAINTRATREARAT10Y). Tried as the default and
    #            withdrawn: see below.
    #   "gdp"    the absolute quarterly change in real seasonally adjusted GDP.
    #   "gdp4"   the four-quarter change in real GDP, DEMEANED. Demeaned because
    #            over a year trend growth is about 3 per cent, which is large
    #            next to the moves being detected: on the raw figure a boom
    #            scores high and a stall at zero growth scores quiet, which is
    #            backwards. Demeaned, both directions count. A four-quarter
    #            difference also spreads an episode across the quarters it
    #            actually occupied, which a one-quarter rule cannot do: the GFC
    #            played out over four to six quarters and no single-quarter
    #            percentile rule catches it.
    #
    # NO DETECTOR FLAGS THE GFC, AND THAT IS CORRECT, not a calibration
    # failure. Australia had no bank failures, the banks stayed funded, and the
    # mining boom held output up. It was a mild event here next to the US, so
    # nothing in the Australian data looks like a discontinuity in 2008-09:
    #
    #   |GDP q/q|        2008Q4 at the 28th percentile, 2009Q1 at the 73rd
    #   demeaned 4q GDP  worst quarter 2009Q3 at the 86th, never clearing 95
    #   world real rate  the big move is 2007Q4-2008Q1, a YEAR before the
    #                    Australian disinflation, which does not arrive until
    #                    2008Q4
    #
    # Reaching the GFC on any of them means dropping to the 75th-85th
    # percentile, which flags a fifth to a quarter of the sample. That is not a
    # discontinuity rule, it is a licence to wander, and it costs the model:
    # under `world`, `lambda` fell 0.305 to 0.168 and corr(base, cash rate) rose
    # 0.88 to 0.94 as the base shadowed the cash rate more closely.
    #
    # `world` is therefore not the default. It is FRED data rather than
    # `rstar_bonds` output, so nothing circular crosses over, but it dates the
    # step to the wrong quarter for Australia.
    #
    # WHICH QUARTERS ARE FLAGGED MATTERS MORE THAN HOW MANY. A flag inside the
    # 2022-24 inflation surge destroys the identification, because that episode
    # is the one large sustained inflation gap in the sample and is where
    # `lambda` comes from. Let the base jump there and it absorbs the very
    # movement `lambda` needs:
    #
    #   gdp    flags stop at 2021Q4     lambda 0.303, two-gaps corr 0.61
    #   gdp4   flags 2022Q3, 2024Q2-Q3  lambda 0.162 (HDI 0.036-0.312), corr 0.42
    #   world  flags 2022Q2, 2022Q4     lambda 0.168, corr 0.46
    #
    # In both damaging cases corr(base, cash rate) rises to 0.94 and lambda's
    # ESS falls by a factor of five. "gdp" is the default because its flags
    # happen to avoid that window, which is luck rather than design: any future
    # detector must be checked against it.
    jump_source: str = "gdp"
    # A quarter is flagged when the ABSOLUTE quarterly change in real seasonally
    # adjusted GDP, |Δy_t|, sits at or above this PERCENTILE of its own
    # distribution over the sample. Absolute, so a collapse and a violent rebound
    # both count: each is the world moving, and neutral can step either way.
    # Not demeaned, because trend growth is small next to the moves being caught
    # and demeaning would only tilt the flags toward weak quarters.
    #
    # A percentile rather than a multiple of
    # the sd because the sd is inflated by the very quarters being flagged: the
    # 2020 collapse and rebound roughly double it, so an sd rule raises its own
    # bar and misses the smaller upheavals. A percentile is unaffected.
    #
    # Still a JUDGEMENT, and the one free parameter here. 95 flags about one
    # quarter in twenty, which over this sample is the genuine upheavals.
    jump_percentile: float = 95.0
    # IMPOSED, not estimated. Flagged quarters are far too few to identify a
    # tail parameter; a free `nu` here would be the prior wearing a hat. 3.0 has
    # a defined mean and variance but very fat tails against a Gaussian.
    jump_nu: float = 3.0

    # --- Priors ---
    # UNITS: `lambda` is per BAND-WIDTH, because the gap is scaled by `band`.
    # The response per percentage point of inflation is `lambda / band`, so at
    # the default band of 0.5 it is twice `lambda`. Taylor's 1.5 per percentage
    # point is 0.75 here, not 1.5 and not 0.5.
    #
    # Centred deliberately on 0.5, which is 1.0 per percentage point: the cash
    # rate moves one for one with inflation. It is NOT "the Taylor coefficient",
    # which an earlier version of this comment claimed.
    #
    # Nor is one-for-one the point at which the real rate is unchanged, which
    # this comment also used to say. That holds only under full pass-through of
    # inflation into expectations. `lambda/band` is the pass-through and the
    # real response summed, and this model cannot separate them, so it asserts
    # nothing about the real response in either direction. See "The Taylor
    # comparison does not work in nominal terms" in MODEL_NOTES.md.
    #
    # Zero is the strong claim, that there is no response at all, and it would
    # falsify the whole identification. The prior is wide enough to reach it.
    #
    # The width makes all of this nearly moot: against a posterior sd of ~0.05,
    # a prior sd of 1.0 carries well under 1% of the posterior. Moving the
    # centre to 0.75 would shift the answer by less than 0.001.
    lambda_mu: float = 0.5
    lambda_sigma: float = 1.0
    # Quarter from which a SECOND `lambda` applies, or None for one coefficient
    # over the whole sample. Set to "2008Q1" to ask whether the cyclical
    # response broke at the GFC. Both halves get the same prior.
    #
    # Note what this can and cannot see. The base already absorbs the post-GFC
    # decline in neutral, so a break here is a break in the response at
    # cyclical frequency, not the drift. If both halves come back near the
    # pooled value with wide posteriors, the trend has taken the difference and
    # the run is uninformative rather than evidence of no break.
    lambda_split: str | None = None
    # The BASE: the slow part, before the inflation response is added. Centred
    # on the sample mean cash rate, wide.
    base_mu: float = 4.0
    base_sigma: float = 3.0
    # Prior scale for the observation error. `eps` not `u`: this is a
    # state-space model, where the convention is eps_t for the observation
    # error and eta_t for the state disturbance, and `u` is already the
    # unemployment rate here (see `employment` and `lambda_u`).
    sigma_eps_sigma: float = 2.0

    # --- Partial adjustment ---
    # OFF by default, and EXPERIMENTAL. With it on the observation equation is
    #
    #   r_t = phi·r_{t-1} + (1 - phi)·(b_t + lambda·g_t) + eps_t
    #
    # so `b_t + lambda·g_t` becomes the rate the Bank is moving TOWARD rather
    # than the rate it sets, and the Bank closes (1 - phi) of the distance each
    # quarter. It is the fix for the residual autocorrelation of 0.857, which is
    # the largest known defect of the default: the likelihood there assumes an
    # independent error and the error is mostly last quarter's cash rate.
    #
    # UNITS CHANGE, AND THIS HAS TO BE SAID. With `phi` free, `lambda` is the
    # LONG-RUN response; in the default it is the same-quarter one. The two are
    # not comparable and quoting them side by side is an error.
    #
    # `r_{t-1}` is the OBSERVED lagged cash rate, not a latent, so the first
    # quarter drops out of the likelihood. That is conditioning on the initial
    # observation, which is standard and costs one of 134 quarters.
    partial_adjustment: bool = False
    # Beta prior on `phi`. Centred near 0.8 and wide: the measured inertia in
    # the cash rate runs 0.90 to 0.97, but that is an AR(1) on the level, which
    # is not the same object as the adjustment speed, so this prior should not
    # be read as asserting either figure. Reaches 0 (no smoothing, the default
    # model) and stays clear of 1 (no adjustment at all, which is not a model).
    phi_a: float = 4.0
    phi_b: float = 1.5

    output_dir: Path | None = None

    def __post_init__(self) -> None:
        """Validate the specification."""
        if self.weights not in WEIGHT_SCHEMES:
            raise ValueError(f"weights must be one of {WEIGHT_SCHEMES}, got {self.weights!r}")
        if self.max_lag < 1:
            raise ValueError(f"max_lag must be positive, got {self.max_lag}")
        if self.window < 1:
            raise ValueError(f"window must be positive, got {self.window}")
        if self.sigma_r <= 0:
            raise ValueError(f"sigma_r must be positive, got {self.sigma_r}")
        if self.jump_source not in JUMP_SOURCES:
            raise ValueError(f"jump_source must be one of {JUMP_SOURCES}, got {self.jump_source!r}")
        if not 0.0 < self.jump_percentile < PERCENTILE_MAX:
            raise ValueError(f"jump_percentile must lie in (0, 100), got {self.jump_percentile}")
        if self.jump_nu <= MIN_NU:
            raise ValueError(f"jump_nu must exceed 2, got {self.jump_nu}")
        if self.lambda_u_sigma <= 0:
            raise ValueError(f"lambda_u_sigma must be positive, got {self.lambda_u_sigma}")
        if self.lambda_sigma <= 0:
            raise ValueError(f"lambda_sigma must be positive, got {self.lambda_sigma}")

    @property
    def constants(self) -> dict[str, float]:
        """The imposed settings, recorded on the model for the run log."""
        return {
            "window": float(self.window),
            "max_lag": float(self.max_lag),
            "weights_geometric": float(self.weights == "geometric"),
            "weights_dirichlet": float(self.weights == "dirichlet"),
            "anchor": self.anchor,
            "band": self.band,
            "floor": -1.0 if self.floor is None else self.floor,
            "walk": float(self.walk),
            "sigma_r": self.sigma_r,
            "jumps": float(self.jumps),
            "jump_world": float(self.jump_source == "world"),
            "jump_percentile": self.jump_percentile,
            "jump_nu": self.jump_nu,
            "employment": float(self.employment),
            "lambda_u_mu": self.lambda_u_mu,
            "lambda_u_sigma": self.lambda_u_sigma,
            "lambda_mu": self.lambda_mu,
            "lambda_sigma": self.lambda_sigma,
            "lambda_split": float("nan") if self.lambda_split is None
            else float(pd.Period(self.lambda_split, freq="Q").ordinal),
            "nonlinear": float(self.nonlinear),
            "lambda2_mu": self.lambda2_mu,
            "lambda2_sigma": self.lambda2_sigma,
            # Carried so the charts can draw each prior against its posterior
            # without re-reading the config.
            "rho_a": self.rho_a,
            "rho_b": self.rho_b,
            "base_mu": self.base_mu,
            "base_sigma": self.base_sigma,
            "sigma_eps_sigma": self.sigma_eps_sigma,
            "partial_adjustment": float(self.partial_adjustment),
            "phi_a": self.phi_a,
            "phi_b": self.phi_b,
        }
