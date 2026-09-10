"""Configuration for the rstar model.

One latent state, two observation equations:

    r*_t = r*_{t-1} + e_r                          sigma_r imposed
    tp_t = mu_tp + rho·(tp_{t-1} - mu_tp) + e_tp   term premium, stationary
    y_t  = r*_t + tp_t + e_y                       AU indexed real 10y yield
    w_t  = r*_t + e_w                              world r*

**Why no IS curve.** `rstar_hlw` (eight resolutions), `nairu` (the free-alpha
blend probe) and `dsge` (the FA-NK family) all independently found the same
thing: in Australian data the interest rate does not visibly move the output
gap, so the IS curve returns whatever structural assumption it was given. This
package therefore does not contain one. It reads r* off asset prices instead.

**The identifying restriction is the missing offset on the world equation.**
With a free constant in both observation equations the level of r* would be
unidentified — shift the state and absorb it in either constant. Setting the
world equation's offset to zero *defines* AU r* on the same scale as the
Holston-Laubach-Williams estimates, so it is directly comparable to their US,
Euro Area and Canada numbers, and the whole level gap to the bond yield lands
in `mu_tp`, which is where a term premium belongs. That assertion is this
model's equivalent of `ystar`'s 2.5% anchor: one asserted thing, stated plainly.

**Two different rates, and the wedge is observed.** What the bond market prices
is the globally-arbitraged risk-free rate. What governs investment is the cost
of capital to firms, which sits above it by the external finance premium. That
premium is *data* here — the corporate-to-CGS spread — not a latent state, and
that is the one substantive difference from the `dsge` FA-NK models, whose
latent wedge produced an r* the notes call not credible.

The spread starts only in 2005Q1 against the yield's 1986Q3, so it is applied
after estimation rather than inside it: the state is estimated on the long
sample and the business rate is derived where the spread exists. Nothing about
the identification depends on the short series.
"""

from dataclasses import dataclass
from pathlib import Path

DEFAULT_OUTPUT_DIR = Path(__file__).parent.parent.parent.parent / "model_outputs"

# Which published r* series stands in for the world rate.
#   "mean" — the simple average of the three, i.e. "the global financial cycle"
#   "US"   — the marginal pricer, arguably the truer mechanism
# Where the Taylor rule's inputs come from. See `ModelConfig.input_source`.
INPUT_SOURCES = ("joint", "separate")
DEFLATORS = ("expectations", "trimmed")
SHORT_RATES = ("cash", "bill")

# The first four are Holston-Laubach-Williams model estimates. The last two are
# market prices, and they are here because HLW is not one: it is `r* = g + z`
# identified through the IS and Phillips curves, so a global bond selloff cannot
# move it, and it did not. Between the COVID-QE era and the tightening era the
# HLW three-country mean moved -0.03 while the US 10y TIPS yield rose 2.31 and
# the real fed funds rate 2.62. Anchoring a market-read model on a macro-read
# estimate is the package's deepest inconsistency; these let it be tested.
#
#   "cleveland" — Cleveland Fed 10-year expected real rate (REAINTRATREARAT10Y),
#                 1982 onwards, so it covers the whole sample. A long-horizon
#                 expected real SHORT rate, so it is neither mostly term premium
#                 nor mostly cyclical stance, which makes it the closest published
#                 thing to a world price of money over the medium run.
#   "tips"      — US 10-year TIPS yield (DFII10), the purer price, but it starts
#                 in 2003 and so truncates the sample by a decade.
WORLD_SOURCES = ("mean", "US", "Euro Area", "Canada", "market", "cleveland", "tips")
MARKET_WORLD_SOURCES = {"cleveland": "REAINTRATREARAT10Y", "tips": "DFII10"}

# "market" is the one to use, and it is a difference rather than a single series:
#
#     world neutral = 10y real rate - 10y term premium
#                   = REAINTRATREARAT10Y - THREEFYTP10
#
# A bare 10y real yield is the wrong object to anchor on, because it contains a
# term premium and the Australian observable contains one too. Faced with two
# co-moving premia the model damps the loading instead of loading one-for-one:
# `b_world` came back at 0.48 on the raw Cleveland series, which reads as "only
# half of a move in world rates reaches Australia" and is not credible for an
# open economy with a floating currency. It was the model stripping a US term
# premium by shrinking the coefficient.
#
# Removing the premium explicitly leaves an expected average real short rate
# over ten years, which is what "the world price of money" should mean. It also
# behaves: 1.30 over 1993-2007, 0.16 through the QE years, and 1.31 now, so on
# this measure the world neutral rate has fully recovered to its pre-GFC average
# and the rest of today's high yield is premium.
#
# Kim-Wright (`THREEFYTP10`, Federal Reserve Board) is a nominal term premium
# being subtracted from a real rate, so this leaves any inflation risk premium
# behind. That is a smaller error than leaving the whole term premium in.
WORLD_NEUTRAL = ("REAINTRATREARAT10Y", "THREEFYTP10")

# The published US term premium, used to pin the Australian one when
# `ModelConfig.us_premium_anchor` is on. Kim-Wright, Federal Reserve Board.
US_PREMIUM_SERIES = "THREEFYTP10"


@dataclass
class ModelConfig:
    """Specification, sample, and the imposed variance."""

    # --- Sample ---
    # 1993Q1, matching ystar, ustar, ystar_ustar and is_curve, so the packages
    # are comparable and the Taylor-rule inputs cover the whole run.
    #
    # This model used to start at 1986Q3, where the indexed bond series begins,
    # on the ground that a decline is easier to locate when its start is in the
    # sample. That held while the long yield was the only observable and the
    # level rested entirely on `mu_tp`. It does not survive the second window.
    #
    # The policy gap is stationary about a single `mu_g`, and the real cash
    # rate averaged 6.55 over 1986Q3-1992Q4 against 1.37 from 1993, peaking at
    # 11.00. Even granting r* of 3 to 4 then, that is `g` at +3 to +7 for 26
    # straight quarters, which one constant mean cannot describe alongside the
    # targeting era. The model would either drag `mu_g` up, and since
    # mean(r*) = mean(r) - mu_g that pushes r* down across the whole sample, or
    # lift r* through the disinflation to shrink `g`, which is r* absorbing the
    # policy stance. That second failure is the one the whole design exists to
    # avoid.
    #
    # So the long sample is misspecified here rather than merely unnecessary.
    # The pre-1993 block is still available via `--start` for the one-window
    # specification, where it remains informative and harmless.
    start: str | None = "1993Q1"
    end: str | None = None

    # --- The global anchor ---
    # "cleveland", the Cleveland Fed's 10-year expected real rate, not the HLW
    # mean it used to be. Three reasons, in order of weight.
    #
    # 1. It is a price. HLW is `r* = g + z` identified through the IS and
    #    Phillips curves, and this package exists because that identification
    #    does not work on Australian data. Rejecting the method domestically
    #    while importing its output as the global anchor is incoherent.
    # 2. It responds to monetary conditions and HLW does not. Between the
    #    COVID-QE era and the tightening era the HLW three-country mean moved
    #    -0.03 while this series moved +1.65 and the real fed funds rate +2.62.
    # 3. The data prefer it. `b_world` comes back at 0.497 [0.325, 0.680], a
    #    posterior sd of 0.09 against 0.25 for the HLW loading, and `nu_walk`
    #    returns to its prior mean instead of sitting in the infinite-kurtosis
    #    regime, so the wedge stops needing fat tails to import a repricing the
    #    anchor could not carry.
    #
    # Two honest caveats. It is a US series, so "world" is a stretch that the
    # three-country HLW mean at least gestured at. And it is a 10-year real
    # yield, so it contains a US term premium, which makes the residual this
    # model calls the Australian term premium a *relative* quantity rather than
    # an absolute one. `--world-source mean` restores the old anchor.
    world_source: str = "cleveland"
    # Drop the world equation entirely, leaving one observable for two
    # components. Kept because it is the honest test of what the anchor
    # contributes, in the same spirit as ustar's `use_output_gap`.
    use_world: bool = True

    # Let the data choose how much of world r* passes through, rather than
    # imposing one for one:
    #     r*_t = b_world · world_t + wedge_t
    #
    # This is the package's maintained hypothesis put at risk. "r* is largely
    # imported" is currently assumed, not tested: world r* enters as data with
    # a coefficient of exactly 1 and no error term, so the model measures the
    # wedge conditional on the premise and cannot be evidence for it. A free
    # loading asks whether Australian yields actually behave that way.
    #
    # Three outcomes, all informative. `b_world` near 1 with a posterior
    # tighter than the N(1, 1) prior supports the premise. Near 0 contradicts
    # it. A posterior that simply reproduces the prior means the free wedge
    # absorbs whatever the loading does not, so the data cannot tell, and the
    # premise stays a premise that should be labelled as one.
    #
    # It is the first outcome: `b_world` = 0.846 [0.383, 1.303], a posterior sd
    # of 0.245 against the prior's 1.0. The data pull the loading to a quarter
    # of the prior's width and centre it near one. On by default for that
    # reason — the premise the whole package rests on should be estimated where
    # it can be, not imposed and then defended in the notes. `--impose-world-
    # loading` restores the b_world = 1 version, which is the comparator.
    #
    # No separate intercept: `wedge_0` already is one, and adding a second
    # would be exactly collinear with it.
    free_world_loading: bool = True

    # --- The second window: the real cash rate ---
    # Drop the short-rate equation, leaving the one-window model: r* against
    # the long yield alone, with `mu_tp` carrying the level. Kept for the same
    # reason as `use_world`, and because the comparison only decomposes
    # cleanly in one direction. Running the *new* spec on the old 1986Q3
    # sample confounds nothing usefully, since that sample is misspecified for
    # the policy gap (see `start`). The pair that separates the two changes is:
    #
    #   --no-short --start 1993Q1    the sample change, old spec
    #   (default)                    the specification change, against the above
    use_short: bool = True

    # Which inflation series turns the nominal cash rate real. An
    # identification choice, so it is a switch: `expectations` matches
    # `is_curve` and keeps the two comparable, but it is a spliced medium-to-
    # long horizon measure deflating an overnight rate. `trimmed` is
    # backward-looking, with the opposite bias. See `observations._deflator`.
    deflator: str = "expectations"

    # Which short rate the second window is built on.
    #
    # `cash` is the overnight rate, which cannot move until the RBA moves it.
    # When the market prices a tightening the Bank has not delivered, every
    # other rate on the curve reflects it and this one does not, so the model
    # must push the difference into r* or the premium. That is the same defect
    # `use_curve` was added to address, at the other end of the curve.
    #
    # `bill` is the 90-day bank-accepted bill (RBA F1, FIRMMBAB90), which
    # covers the next quarter's expected path. It sits +0.16 above the cash
    # rate on average since 1993 (sd 0.21), and the gap opens at the turning
    # points: +0.90 1994Q4, +0.68 2008Q1, +0.57 2018Q2, +0.87 2022Q2.
    #
    # `g` is the real short rate less r* with a coefficient of one either way:
    # a 90-day rate is the one-quarter point on the same expected-path curve,
    # so it belongs at H = 1 exactly where the overnight rate was.
    #
    # `cash` is the default, after `bill` was tried as the default and reverted.
    #
    # BBSW is a bank credit benchmark, not a duration one: it is the rate on
    # prime bank paper, so lending at 90 days is unsecured exposure to a bank
    # for 90 days, and the spread to OIS is default and liquidity compensation
    # rather than an expected policy path. Decomposed against OIS, bill-minus-
    # cash over 2015-19 was +0.26, of which the expected path (OIS less cash)
    # was -0.05 and bank funding was +0.31. Through 2018 the market expected
    # nothing at all (OIS less cash exactly 0.00 in all four quarters) while the
    # bill sat +0.49 over cash on the US repatriation and T-bill supply squeeze.
    # The bill made the pre-COVID stance look 0.26 tighter for reasons that have
    # nothing to do with monetary policy, in the one window the model's central
    # argument turns on.
    #
    # The 30-day bill is worse, not better: +0.38 of pure funding premium in
    # 2018 against the 90-day's +0.49, but only +0.06 over cash in 2022 against
    # +0.39, so it keeps the contamination and loses the anticipation. Any tenor
    # short enough to be nearly credit-free is too short to price a tightening
    # cycle.
    #
    # It also breaks this package's own rule. The design keeps the risk-free
    # rate and the external finance premium separate, the latter as data applied
    # after estimation, precisely so a credit spread never reaches the state.
    # BBSW as an observation feeds one straight into r* and the stance.
    #
    # 3-month OIS is the clean series and is not usable: `FIRMMOIS3` ends in
    # November 2022. `--short-rate bill` is kept as the sensitivity.
    short_rate: str = "cash"

    # The term of the long yield, in quarters, used to convert the policy
    # gap's persistence into how much of that gap the long yield carries:
    #   k = (1/H)·(1 - rho_g^H)/(1 - rho_g)
    # 40 = ten years, matching the indexed series. `k` is computed from
    # `rho_g`, never asserted, which is what makes the dynamics an
    # overidentifying restriction rather than another free knob.
    horizon_quarters: int = 40

    # --- The third window: the belly of the curve ---
    # The overnight rate is a poor summary of the expected policy path exactly
    # where it matters. In 2022Q1 the real two-year sat 1.24 points above the
    # real overnight rate, the market having priced a normalisation the RBA had
    # not begun; the model, seeing only the cash rate, had to push the
    # difference into r* or the premium, and the wedge's largest move in the
    # sample lands in the next quarter.
    #
    # With a medium maturity the same r* faces three points on one curve, each
    # carrying a different share of the policy gap. Note this does *not* close
    # the level: four levels (`wedge_0`, `mu_tp`, `mu_tp_m`, `mu_g`) against
    # three observables still leaves one flat direction, so one mean is still
    # asserted. What becomes identified is the *slope* of the premium curve,
    # `mu_tp - mu_tp_m`, which nothing before could see: 0.700 [0.026, 1.323],
    # just clear of zero.
    #
    # OFF by default, having been tried as the default and dropped.
    #
    # It works, in the narrow sense: zero divergences, `k` and `k_m` computed
    # from one `rho_g` fitting three maturities at once, `tp_slope` identified.
    # What it costs is r*. Forcing three points of one curve onto one state and
    # one stationary `g` leaves only one affordable home for a persistent
    # stretch of low real rates, and that home is r*. Against the two-window
    # model it drives r* over 2016-19 from -0.36 to -0.74 and over the QE
    # window from -0.38 to -0.86, and it pushes `nu` from 2.73 to 1.87, close
    # enough to Cauchy that a large jump in r* is nearly free. That is r*
    # absorbing the policy stance, which is the one failure this whole design
    # exists to avoid.
    #
    # The problem it was built for, the overnight rate not moving when the
    # market prices a tightening the RBA has not delivered, is better handled
    # at the short end by `short_rate = "bill"`, which is a smaller assumption
    # and does not re-identify the state. `--curve` restores it.
    use_curve: bool = False
    curve_maturity: int = 3
    curve_horizon_quarters: int = 12

    # --- Pinning the premium to a published one ---
    # The level of r* is not identified: `wedge_0` and `mu_tp` trade off at
    # -0.7, so the split between "Australia's r* sits above the world's" and
    # "the average term premium is large" is decided by `mu_tp`'s prior rather
    # than by data. Left free, the answer is not credible. At 2026Q3 it put the
    # Australian real term premium at 1.67 against the US Kim-Wright premium of
    # 0.82, and Australian neutral at 0.91 against a world 1.31: a real yield
    # gap of +0.56 explained by moving the premium +0.84 and neutral -0.40, in
    # opposite directions and by more than the gap itself.
    #
    # With this on, `tp` must track the observed US premium and only the spread
    # over it is estimated. The asserted quantity becomes "the average
    # Australian premium over the US one", which is a liquidity argument about
    # thin indexed AGS against TIPS, small enough to argue about and backed by
    # a published series. That is what `ystar`'s 2.5% anchor is: one asserted
    # number, stated plainly, that pins a level the data cannot.
    #
    # Two mismatches that do not go away. Kim-Wright is a nominal term premium
    # and `tp` is real, so the inflation risk premium stays on the Australian
    # side of the spread. And a US premium is standing in for a global one.
    us_premium_anchor: bool = False
    # Centred on a modest liquidity premium, wide enough to be moved.
    mu_spread_mu: float = 0.25
    mu_spread_sigma: float = 0.5

    # Which of the two means is asserted. One of them must be: with two
    # observables and three free levels (`wedge_0`, `mu_tp`, `mu_g`) the
    # posterior still has a flat direction, since adding d to r* and taking d
    # off both means leaves every fitted value unchanged. Two windows do not
    # make the level free, they make the assertion checkable.
    #
    # False (default): assert `mu_tp` exactly as the one-window model did, and
    # report the implied average policy stance. Directly comparable to the old
    # vintage, and it isolates the `k·g` correction.
    #
    # True: assert the stance instead and report the implied term premium. The
    # more interesting write-up, but it changes two things at once, so it is
    # the second run rather than the first.
    assert_stance: bool = False

    # The asserted stance, used only when `assert_stance` is True. A prior and
    # not a hard zero, so the data can push back. Centred on zero: over the
    # targeting era, policy averaging neutral is a claim about the mandate.
    # Over 1986-92 it is not, that block is a deliberate disinflation with a
    # mean real cash rate of 2.21 against 1.37 from 1993, so asserting this on
    # the longer sample biases r* up by roughly the difference.
    mu_g_mu: float = 0.0
    mu_g_sigma: float = 0.5

    # --- How the Australian wedge is allowed to move ---
    # The wedge is a step function: flat, except at named quarters where it
    # jumps. That is a stronger and more honest restriction than a random walk
    # with fat tails, because it puts the assumption in plain sight — you are
    # asserting *when* Australia's spread over world r* changed, and estimating
    # only *how much*.
    #
    # The dates are not guesses. Each is a large move in the observed spread
    # with a name attached, and the era means step cleanly between them:
    # +1.94 pre-1994, +0.89 to 2008Q3, +0.17 to 2019Q1, then the 2019 easing
    # and COVID take it to about -1.5, and liftoff returns it to +0.75.
    #
    # 2020Q2 rather than Q1 for COVID: in 2020Q1 the spread rose 0.50, because
    # the March dash-for-cash pushed the AU real yield *up*. The suppression
    # from bond purchases and yield curve control shows from Q2.
    break_quarters: tuple[str, ...] = (
        "1994Q1",   # inflation targeting established; the global bond rout
        "2008Q4",   # the GFC
        "2019Q2",   # the RBA easing cycle, the first cuts since 2016
        "2020Q2",   # COVID: bond purchases and yield curve control
        "2022Q2",   # liftoff and the exit from QE
    )
    # Prior sd on each jump. Generous against observed level shifts of 0.3-2.3.
    jump_sigma: float = 1.5
    # Background drift between breaks. Zero gives a pure step function, which
    # is the point; raise it to soften the steps and see what that costs.
    wedge_drift: float = 0.0

    # --- Or: let the wedge move whenever it likes ---
    # The step form asserts that Australia's spread over world r* changed on
    # five dates and was rigid in between. r* is a price and does not behave
    # that way: it drifts, and occasionally moves suddenly. A random walk with
    # Student-t innovations says exactly that — quiet most quarters, with the
    # occasional large move permitted — and it does not require anyone to
    # nominate the dates in advance.
    #
    # Two parameters replace five asserted quarters. `sigma_walk` is imposed
    # and swept, as every smoothness setting in this repo is. `nu` is
    # estimated, and genuinely can be: how often large moves happen is a
    # distributional question the data can answer, unlike the level.
    #
    # The step form is kept as the comparator. If the free walk puts its big
    # innovations on 1994, 2008, 2019 and 2022, the asserted dates were right
    # and are now earned from the data rather than from a reading of the era
    # means. Precedent for the fat tails: `nairu`'s `student_t_nairu`.
    free_wedge: bool = True

    # 0.12, raised from 0.08, because `nu` is an internal check on whether the
    # imposed variance is defensible and at 0.08 it was failing that check.
    #
    # The two substitute for each other: sweeping `sigma_walk` from 0.03 to 0.20
    # takes `nu` from 2.14 to 9.79. A low `nu` is not a finding about Australia,
    # it is the model buying with fat tails the room the walk denies it. Below
    # `nu` = 4 the kurtosis is infinite and single jumps are nearly free, which
    # is the regime where the wedge leaps instead of drifting; below `nu` = 2
    # there is no variance at all and `sigma_walk` is a bare scale rather than a
    # standard deviation, so the sweep stops being comparable to itself.
    #
    # At 0.08 `nu` came back at 3.64, inside the infinite-kurtosis regime, and
    # the wedge had to travel 4.8 standard deviations over 2021Q4-2026Q2 to
    # track the global repricing. At 0.12 `nu` is 5.51, clear of that regime and
    # nearer its prior mean of 6, which is the data saying the walk is no longer
    # straining. The cost is a wider r* path; the gain is that the fat tails
    # stop doing work the variance should have been doing.
    #
    # Still imposed and still swept: the sweep table in MODEL_NOTES.md is the
    # honest report, and the level survives it while the path does not.
    sigma_walk: float = 0.12
    nu_prior_mean: float = 6.0

    # Fix `nu` instead of estimating it. None (default) leaves it free.
    #
    # Free `nu` is the model's one bad geometry. It is a centred Student-t: the
    # innovations are drawn at a degrees-of-freedom that is itself sampled, so
    # the two funnel against each other. The block comment above says the
    # non-centred form leaves no funnel, and that is true of `sigma_walk` and
    # only of `sigma_walk`. The symptoms are 12 divergences in 8,000 draws
    # concentrated at low `nu` (median 1.82 among divergent draws against 2.05
    # overall) and `ess_bulk` of 820 for `nu_walk`, whose `mcse_sd` of 0.191 is
    # 14% of its posterior sd. Fixing `nu` removes that geometry outright.
    #
    # Three things to know before choosing a value.
    #
    # The data do speak here: the prior mean is 6.0 and the posterior comes
    # back at 2.36 [1.11, 3.85]. So fixing at the posterior median of about 2.0
    # is declining to pay for uncertainty the data already narrowed; fixing
    # below that is a judgement of your own, and belongs in the notes as one.
    #
    # Below nu = 2 the Student-t has no variance. `sigma_walk` is then a scale
    # rather than a standard deviation, and it already behaves like one: the
    # realised sd of the quarterly wedge change is 0.139 against a nominal
    # 0.08.
    #
    # Lower `nu` does not simply mean jumpier. With `sigma_walk` held fixed,
    # heavier tails make ordinary quarters quieter and rare ones larger, so the
    # expected effect is a flatter wedge between jumps with 2022Q2 sharpening,
    # and the marginal 0.30-sized moves (1993Q3, 1995Q2, 2012Q2) the first to
    # go. Whether that is right is a view about the wedge, not a sampler
    # question, which is why it is swept rather than set.
    nu_walk: float | None = None

    # Sample the Student-t innovations as a scale mixture of normals rather
    # than directly: `eps = z · sqrt(lam)` with `z ~ N(0,1)` and
    # `lam ~ InverseGamma(nu/2, nu/2)`, which *is* a Student-t(nu). Same
    # posterior, different geometry.
    #
    # OFF, because it was tried and it is much worse: 511 divergences against
    # 12, `r_hat` 1.02 against 1.00, and `nu_walk` ESS halved to 440. Kept only
    # so the test is repeatable.
    #
    # The reason is the standard one. Non-centring pays when the data are weakly
    # informative about the latent and costs when they are not. The yield data
    # pin these innovations well, so making the scale an explicit `lam` builds a
    # funnel between `lam` and `z` rather than removing one — and with
    # `alpha = nu/2 ≈ 1.18` that InverseGamma is heavy-tailed enough to make it
    # a bad funnel. Centred is the right parameterisation here.
    noncentred_wedge: bool = False

    # --- The policy rule: first-difference, not level ---
    # `d_i = a_pi·(pi - target) + a_gap·gap`: how far to move the cash rate,
    # not where to put it. Orphanides and Williams proposed this form precisely
    # because r* is badly measured, and that is why it is the rule here.
    #
    # A level Taylor rule needs r*, and this model does not identify its level:
    # `wedge_0` and `mu_tp` are correlated at -0.76, so the data pin their sum
    # but not the split between "Australia's r* sits above the world's" and
    # "the average term premium is large". Feeding an unidentified level into a
    # level rule produced 1.6pp of spurious tightening across 2012-2021, a
    # decade when inflation averaged 1.91 and the output gap -0.28 and the rule
    # should plainly have been saying ease. The difference rule uses only what
    # this model does identify.
    #
    # Coefficients are Taylor's 0.5 divided by four, because these apply to a
    # *quarterly* change rather than a level. Undivided they over-move
    # fourfold: -17.3pp of cuts across 2012-2021 against the 4.15 delivered.
    anchor: float = 2.5
    rule_pi: float = 0.125
    rule_gap: float = 0.125
    # Use the ustar unemployment gap in place of the ystar output gap.
    taylor_use_ugap: bool = False

    # Look through the supply-driven part of inflation, taken from `ustar`'s
    # Phillips decomposition (rho·d4pm + xi·GSCPI^2·sign) on a four-quarter
    # rolling basis. Tightening into a cost-push shock deepens the output loss
    # without addressing the cause, which is why central banks say they look
    # through them; this makes that explicit rather than leaving it as
    # commentary on the chart.
    look_through_supply: bool = True
    # Apply the look-through only when the supply contribution is positive:
    # decline to tighten into a supply-driven overshoot, but still ease when
    # supply is holding inflation down. That asymmetry is a policy stance
    # rather than a measurement choice, so both lines are charted.
    supply_positive_only: bool = True

    # --- Inputs read from other models ---
    # Where the Taylor rule's inputs come from.
    #
    # "joint" reads the output gap, the unemployment gap and the supply term
    # from one completed `ystar_ustar` run. "separate" reads the first from
    # `ystar` and the other two from `ustar`, which is what this model did
    # before the joint model existed.
    #
    # Joint is the default because the three inputs are then mutually
    # consistent: the same potential output, the same u*, the same estimate of
    # `c`. Read separately they come from two models run in sequence, where
    # `ustar` treats `ystar`'s gap as data, so the unemployment gap is
    # conditional on a gap the other model no longer reports.
    #
    # Note the limits of what this changes. The Taylor rule sits outside this
    # model's likelihood, so `r*`, the wedge and the term premium cannot move:
    # only the prescription does.
    input_source: str = "joint"
    joint_prefix: str = "ystar_ustar"
    ystar_prefix: str = "ystar"
    ustar_prefix: str = "ustar"

    # --- Output ---
    output_dir: Path | None = None

    def __post_init__(self) -> None:
        """Validate the specification switches."""
        if self.world_source not in WORLD_SOURCES:
            raise ValueError(f"world_source must be one of {WORLD_SOURCES}, got {self.world_source!r}")
        if self.input_source not in INPUT_SOURCES:
            raise ValueError(f"input_source must be one of {INPUT_SOURCES}, got {self.input_source!r}")
        if self.deflator not in DEFLATORS:
            raise ValueError(f"deflator must be one of {DEFLATORS}, got {self.deflator!r}")
        if self.short_rate not in SHORT_RATES:
            raise ValueError(f"short_rate must be one of {SHORT_RATES}, got {self.short_rate!r}")
        if self.horizon_quarters < 1:
            raise ValueError(f"horizon_quarters must be positive, got {self.horizon_quarters}")
        if self.curve_horizon_quarters < 1:
            raise ValueError(f"curve_horizon_quarters must be positive, got {self.curve_horizon_quarters}")
        if self.curve_horizon_quarters >= self.horizon_quarters:
            raise ValueError(
                "curve_horizon_quarters must be shorter than horizon_quarters, got "
                f"{self.curve_horizon_quarters} against {self.horizon_quarters}",
            )
        if self.use_curve and not self.use_short:
            raise ValueError("use_curve needs use_short: the curve equations carry the policy gap")
        if self.mu_g_sigma <= 0:
            raise ValueError(f"mu_g_sigma must be positive, got {self.mu_g_sigma}")

    @property
    def constants(self) -> dict[str, float]:
        """The imposed settings, recorded on the model for the run log."""
        return {
            "jump_sigma": self.jump_sigma,
            "wedge_drift": self.wedge_drift,
            "free_wedge": float(self.free_wedge),
            "sigma_walk": self.sigma_walk,
            "use_short": float(self.use_short),
            "short_rate_is_bill": float(self.short_rate == "bill"),
            "horizon_quarters": float(self.horizon_quarters),
            "free_world_loading": float(self.free_world_loading),
            "use_curve": float(self.use_curve),
            "curve_maturity": float(self.curve_maturity),
            "curve_horizon_quarters": float(self.curve_horizon_quarters),
            "assert_stance": float(self.assert_stance),
            "mu_g_mu": self.mu_g_mu,
            "mu_g_sigma": self.mu_g_sigma,
            "anchor": self.anchor,
            "rule_pi": self.rule_pi,
            "rule_gap": self.rule_gap,
            "taylor_use_ugap": float(self.taylor_use_ugap),
            "look_through_supply": float(self.look_through_supply),
            "supply_positive_only": float(self.supply_positive_only),
        }
