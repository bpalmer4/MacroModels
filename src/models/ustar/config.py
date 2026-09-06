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

    # --- Equations ---
    include_phillips: bool = True

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

    # --- How fast u* is allowed to drift ---
    # Imposed, and the single most consequential setting in the model: u* runs
    # 5.08 to 4.62 across 0.024 to 0.065, with the gap moving -0.72 to -0.27.
    #
    # 0.040 is a compromise between three readings. `ystar`'s rule — a
    # trend innovation sd around 8% of the observed variation in the series it
    # trends, so 8% of sd(du) = 0.300 — implies 0.024. The NAIRU model's
    # realised sd(dNAIRU) of 0.032 implies about the same. Both of those give a
    # u* that is a near-straight glide.
    #
    # The ceiling comes from the inflation-band chart. Across 2012Q4-2015Q4,
    # a stretch when inflation sat *below* the RBA band and so was signalling
    # genuine slack, u* declines at 0.040 and tighter (-0.09), is flat at 0.050
    # (-0.01), and turns positive at 0.065 (+0.09) — the model booking part of
    # the post-mining-boom rise in unemployment as structural. `beta_pi` also
    # falls monotonically, 0.79 at 0.024 to 0.34 at 0.065, as a freer u* crowds
    # out the de-anchoring term.
    sigma_ustar: float = 0.040

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
