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
WORLD_SOURCES = ("mean", "US", "Euro Area", "Canada")


@dataclass
class ModelConfig:
    """Specification, sample, and the imposed variance."""

    # --- Sample ---
    # 1986Q3 is where the indexed bond series begins. Unlike ystar and ustar
    # this model does not start in 1993: it needs the high-real-rate period,
    # since a decline is easier to locate when its start is in the sample.
    start: str | None = "1986Q3"
    end: str | None = None

    # --- The global anchor ---
    world_source: str = "mean"
    # Drop the world equation entirely, leaving one observable for two
    # components. Kept because it is the honest test of what the anchor
    # contributes, in the same spirit as ustar's `use_output_gap`.
    use_world: bool = True

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
    sigma_walk: float = 0.08
    nu_prior_mean: float = 6.0

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
    ystar_prefix: str = "ystar"
    ustar_prefix: str = "ustar"

    # --- Output ---
    output_dir: Path | None = None

    def __post_init__(self) -> None:
        """Validate the specification switches."""
        if self.world_source not in WORLD_SOURCES:
            raise ValueError(f"world_source must be one of {WORLD_SOURCES}, got {self.world_source!r}")

    @property
    def constants(self) -> dict[str, float]:
        """The imposed settings, recorded on the model for the run log."""
        return {
            "jump_sigma": self.jump_sigma,
            "wedge_drift": self.wedge_drift,
            "free_wedge": float(self.free_wedge),
            "sigma_walk": self.sigma_walk,
            "anchor": self.anchor,
            "rule_pi": self.rule_pi,
            "rule_gap": self.rule_gap,
            "taylor_use_ugap": float(self.taylor_use_ugap),
            "look_through_supply": float(self.look_through_supply),
            "supply_positive_only": float(self.supply_positive_only),
        }
