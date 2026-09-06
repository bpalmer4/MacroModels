"""Configuration for the ystar model.

Single source of truth for the specification, the sample, the fixed variance
settings, and the output location.

Two specifications:

- **core** (default) — potential output from output and inflation alone, as a
  Gaussian random walk with a random-walk drift. Two latent states, two
  observation equations. This is the model the project set out to build.
- **target** — the same Gaussian random walk potential as `core`, but with no
  Phillips curve. Inflation enters only as a weight on the statement "the gap
  is zero": sharp when inflation sits at target, vacuous when it does not.
  No slope is estimated, so there is nothing for the RBA's reaction function
  to attenuate. See `equations/target_consistency.py` for why a Phillips slope
  cannot be recovered from an inflation-targeting sample.
- **labour** — decomposes potential into trend hours and trend productivity,
  with trend hours built on observed working-age population and a separately
  observed participation rate. Five series, six equations. Kept because the
  hours/productivity attribution is useful in its own right, but secondary:
  the decomposition is close to circular (population cancels out of the hours
  equation, and the resulting trend hours path reproduces an HP(1600) trend of
  hours at a correlation of 0.9972), and its trend productivity estimate is
  inseparable from labour-force composition.

The variance settings are the model's central identifying assumption.
Following Kuttner (1994) and the HP filter's implicit choice, the
signal-to-noise ratio is *imposed* rather than estimated: sigma_c is fixed and
every trend innovation is a fixed multiple of it. This sidesteps the
Stock-Watson pile-up problem, at the cost of making the answer conditional on
the settings. `sigma_sweep.py` exists to show how conditional.

Note that sigma_c is itself fixed, not merely ratioed against. A ratio taken
against a free scale imposes no smoothness at all: if the cycle equation fits
badly sigma_c rises, which loosens every trend in proportion, which lets the
trend chase the data harder, which makes the cycle fit worse still. Pinning the
absolute scale is what makes the restriction bite.

Reference points for the settings:
- HP(1600) on quarterly data corresponds to a trend/cycle innovation sd ratio
  of 1/40 = 0.025 on the *second* difference of the trend. `ratio_g` (core) and
  `ratio_g_lp` (labour) govern that second-difference channel, so 0.025 is the
  HP-1600 analogue.
- The level ratios govern first-difference innovations, which an HP trend does
  not have at all. They must therefore be *small*: the test is that each stays
  a modest fraction — around 8% — of the observed variation in the series it is
  a trend for. Over 1993Q1-2026Q2 the sd of quarterly d(log GDP) is 0.97,
  d(log participation) 0.53, d(log hours per participant) 0.79, and
  d(log productivity) 0.77.
- sigma_c = 0.60 against an sd of quarterly d(log GDP) of 0.97: the cycle
  innovation is a fraction of total output growth variation, the rest being
  trend growth and noise.
"""

from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_OUTPUT_DIR = Path(__file__).parent.parent.parent.parent / "model_outputs"

SPECS = ("inflation", "production", "core", "labour", "target")

PI_BASES = ("quarterly", "annual")

SUPPLY_CONTROLS = (None, "import_prices")


@dataclass
class ModelConfig:
    """Model configuration.

    Attributes:
        spec: "core" (Y and pi) or "labour" (hours x productivity).
        start: First quarter of the estimation sample. Default 1993Q1 — the
            inflation-target regime. The 2.5% anchor in the Phillips curve is
            only valid from target adoption, so the sample start and the
            anchor are a matched pair: change one and you must change the other.
        end: Last quarter (None = latest available).
        anchor: Inflation anchor in annual %, the RBA's target midpoint.
        pi_basis: Which trimmed mean series the Phillips curve is estimated on.
            "quarterly" (default) uses the quarterly rate annualised; "annual"
            uses the four-quarter rate. The four-quarter rate is *overlapping*
            at quarterly frequency, so its error is MA(3) by construction even
            under a correct model. Giving it iid errors, as the first version
            of this model did, treats each of the ~133 observations as
            independent when there are closer to 33 independent annual
            differences, which overstates how much identification inflation is
            entitled to supply. The quarterly rate is non-overlapping, so iid
            errors are defensible and the implied annual relationship still
            holds by summation. For the `target` spec the choice is different:
            inflation is not a regressor there, so there is no overlapping-error
            problem, and "at target" is an annual concept, so "annual" is the
            natural setting.
        supply_control: Optional cost-push regressor in the Phillips curve.
            None (default) is the price-gap-only form. "import_prices" adds
            demeaned annual consumption-goods import price growth, lagged one
            quarter, as in the RBA's MARTIN. Without it, the imported component
            of 2022-24 inflation has nowhere to go but the demand gap, which
            mechanically pushes potential output down.
        sigma_c: Cycle innovation sd, in log x 100 units. Fixed, not estimated.
        ratio_ystar: sigma_{y*} / sigma_c   (core: potential output level)
        ratio_g: sigma_{g}     / sigma_c   (core: trend growth)
        ratio_pr_star: sigma_{pr*}  / sigma_c  (labour: trend participation)
        ratio_hpp_star: sigma_{hpp*} / sigma_c (labour: hours per participant)
        ratio_lp_star: sigma_{lp*}  / sigma_c  (labour: trend productivity)
        ratio_g_lp: sigma_{g_lp}    / sigma_c  (labour: trend prod growth)
        smooth_pop: Henderson MA terms applied to log population, labour spec
            only (0 = off). Population enters h* directly and unsmoothed, so
            h* would otherwise inherit the LFS population estimate's
            quarter-to-quarter estimation noise, which contributed 93% of the
            visible jitter in trend hours growth and carries no information.
            Henderson is used rather than a centred MA because it handles the
            endpoint with asymmetric weights, and the tail is ARIMA-extended
            first so the last real observation gets symmetric weights.
        output_dir: Where traces and metadata are written.
        name: Run name, used in output filenames.

    """

    spec: str = "inflation"
    start: str = "1993Q1"
    end: str | None = None
    anchor: float = 2.5
    # "annual" is the default because the live `inflation` spec uses the
    # four-quarter rate: there inflation is not a regressor, so the overlapping
    # -error problem that motivated "quarterly" does not arise, and "at target"
    # is an annual concept. The `core` spec should be run with pi_basis
    # "quarterly"; see the note on this field below.
    pi_basis: str = "annual"
    supply_control: str | None = None

    sigma_c: float = 0.60

    # target spec: the credibility of "the gap is zero", in log x 100 units.
    # sd(e_t) = gap_sd_on_target + gap_sd_per_pp · |pi_t - anchor|.
    # gap_sd_on_target is how sure we are that on-target inflation means zero
    # gap; gap_sd_per_pp is how fast that claim is withdrawn as inflation
    # moves away. Both imposed, not estimated, and both swept.
    # Estimate sigma_ystar instead of imposing it. The package imposes its
    # variances to dodge the Stock-Watson pile-up problem, but that argument
    # does not apply to the `inflation` spec: there potential is a residual,
    # so once c and g are known the innovation is directly observable and its
    # standard deviation is an ordinary estimation problem. sigma_g must stay
    # imposed, since level and drift variances are the pile-up pair and cannot
    # both be free.
    free_sigma_ystar: bool = False

    # Let the GDP residual e_c be AR(1) rather than white noise (`inflation`
    # spec only). e_c carries about four fifths of the cyclical variation in
    # output, and asserting that four fifths is serially independent is a
    # strong claim: a persistent non-inflationary cycle would look exactly like
    # this, and the white-noise model cannot see it. Freeing rho asks whether
    # c and trend growth depend on that assertion. Note what it does to the
    # estimator: with persistent errors the likelihood weights c like a
    # quasi-differenced GLS regression, and the inflation deviation is itself
    # highly persistent, so quasi-differencing strips out much of the
    # low-frequency variation in d. c is not guaranteed to survive.
    ar1_residual: bool = False

    # Inclusive quarter range whose inflation deviation is set to zero, as
    # ("2020Q2", "2021Q1"). Those quarters keep their GDP observation and are
    # fitted continuously, so nothing is dropped and no dummy is added; what
    # changes is that they carry no deviation, so they imply no gap and supply
    # no identification to `c`. This is the targeted form of "the lockdown is
    # not evidence about the inflation-output relationship": the pandemic
    # quarters have a large negative `x` caused by closure and a negative `d`
    # caused by free childcare and fuel, so regressor and residual share a
    # common cause and the projection's orthogonality condition fails there.
    # See iteration log items 13 and 16.
    zero_deviation: tuple[str, str] | None = None

    # Give `c` a two-sided Normal(0, 2) prior instead of the default HalfNormal,
    # so the posterior can place mass on a negative conversion factor. The
    # default imposes the sign, which means the usual "c is clear of zero"
    # statement is guaranteed by the prior rather than earned from the data.
    # This is how the model's central proposition gets tested rather than
    # assumed. `inflation` spec only.
    two_sided_c: bool = False

    # --- production spec ---------------------------------------------------
    # Potential growth from a Cobb-Douglas production function instead of a
    # free drift: g_Y* = alpha·g_K* + (1-alpha)·g_L* + g_M*. The level and the
    # gap are unchanged from the `inflation` spec, so the inflation fulcrum
    # still positions potential; what changes is where its *growth* comes from.
    #
    # Each factor trend is a random walk observed with noise, and the ratio of
    # trend innovation sd to observation sd is imposed.
    #
    # These are NOT HP lambdas, and an earlier version of this comment said
    # they were. HP(lambda) is the local *linear trend* model, where lambda is
    # the variance ratio against the innovation to the slope of an I(2) trend.
    # These trends are local *level* models — I(1) random walks — so the ratio
    # is against the innovation to the level, and for the same nominal lambda
    # it smooths very much harder. The interpretable quantity is how far the
    # trend can wander over the sample: r x sigma_obs x sqrt(T).
    #
    # They differ deliberately, and this is the whole content of the
    # specification. Over 1978Q4 onward the HP(1600) cycle is 37% of the
    # variation in capital growth but 97% of it in hours growth, so putting
    # both through the same filter under-smooths labour badly. Note the
    # consequence: with a common ratio and a constant alpha the production
    # terms cancel algebraically and potential growth collapses to an HP trend
    # of GDP growth, exactly. The differences below are what stop that.
    ratio_gk: float = 0.05      # capital: mostly trend already, light smoothing
    ratio_gl: float = 0.0125    # hours: almost all cycle, heavy smoothing
    ratio_gm: float = 0.025     # MFP: noisy at quarterly frequency

    # alpha is a fourth latent trend, observed by the published capital share,
    # smoothed inside the model on the same footing as the factor trends. No
    # filter is applied to the data beforehand, so there is no smoothing choice
    # sitting outside the specification.
    #
    # Set so tight that alpha is very nearly constant, 0.335 to 0.338 across
    # the sample against a published range of 0.299 to 0.406. That is
    # deliberate, and rests on two things.
    #
    # The published movement is the terms of trade, not technology. alpha is
    # GOS / (GOS + COE), and it correlates **+0.852** with the terms of trade
    # in levels (+0.459 in changes), while correlating −0.077 with potential
    # growth in changes and −0.033 with an HP cycle of GDP. When iron ore and
    # coal prices rise, mining revenue lands in GOS with no matching rise in
    # COE, because the wage bill does not scale with the price of the ore. The
    # share moves without anything happening to what the economy can produce.
    #
    # And Cobb-Douglas assumes an elasticity of substitution of one, which
    # *implies* constant factor shares. A drifting alpha inside it is
    # internally inconsistent: if shares genuinely move, the functional form
    # should be CES with sigma != 1, and a drifting alpha is a patch on that
    # misspecification rather than a feature.
    #
    # Loosening this to 0.05 lets a 0.036 drift through, which is the mining
    # boom. It moves potential growth by 0.01pp, so nothing rests on the
    # choice numerically; it rests on what alpha is supposed to represent.
    ratio_a: float = 0.00625

    # Drop the MFP observation equation, which double-counts GDP.
    #
    # The Solow residual is `g_Y - a·g_K - (1-a)·g_L`, so GDP growth is inside
    # it, and GDP is separately observed in the gap equation. The model then
    # has five observed vectors of length T drawn from four independent data
    # series: 670 likelihood terms from 536 numbers on the 2026Q2 vintage.
    # Exactly T of them are redundant, and they are the MFP equation.
    #
    # With `mfp_observed = False` that equation goes, leaving four equations
    # for four series. `sigma_obs_gm` then appears nowhere, so `g_M*` cannot
    # take its innovation sd as a ratio to it and needs an imposed absolute
    # one: `sigma_gm` below, defaulting to the `sigma_g` of the main spec.
    # `g_M*` is then identified as whatever reconciles the cumulated factor
    # trends with GDP through the level equation, which is what a Solow
    # residual is.
    #
    # Tested and left on. At sigma_gm = 0.015 the two are the same model for
    # practical purposes (potential growth 2.17 against 2.16, band 0.66 against
    # 0.68, c 0.500 against 0.498), so removing the redundancy buys nothing.
    # The feared failure did not happen either: trend capital growth is +2.32
    # to +2.33 in every variant, so `g_M*` does not absorb the decomposition.
    #
    # What sigma_gm does control is the *shape* of the MFP path once it has no
    # data of its own: trend MFP in 1997Q4 runs 0.78, 1.35, 1.99 at sigma_gm
    # 0.005, 0.015, 0.05, against 1.44 when observed. Since the productivity
    # attribution is this specification's main addition, anchoring it to the
    # measured Solow residual is worth more than removing 134 redundant
    # likelihood terms that change no reported number.
    mfp_observed: bool = True
    sigma_gm: float = 0.015

    gap_sd_on_target: float = 0.50
    gap_sd_per_pp: float = 1.00
    cycle_ar: bool = True
    # Longest lag, in quarters, by which inflation may follow the gap. The
    # restriction uses a weighted average of the deviations at lags 0..pi_lag_max,
    # with the weights *estimated* (Dirichlet, so they are non-negative and sum
    # to one) rather than a single lag being picked by hand. Asserting a
    # contemporaneous relation (pi_lag_max = 0) contradicts the transmission
    # lag: on this sample corr(HP gap_t, pi dev_{t+k}) peaks at k=1 (+0.28) and
    # has turned negative by k=8 (-0.29). The cost of a longer window is that
    # the last pi_lag_max quarters carry no restriction, there being no future
    # inflation to compare them with, so the endpoint leans harder on the
    # smoothness priors.
    pi_lag_max: int = 4

    # core
    ratio_ystar: float = 0.13
    ratio_g: float = 0.025

    # labour
    ratio_pr_star: float = 0.10
    ratio_hpp_star: float = 0.10
    ratio_lp_star: float = 0.10
    ratio_g_lp: float = 0.025
    smooth_pop: int = 7

    output_dir: Path = field(default_factory=lambda: DEFAULT_OUTPUT_DIR)
    name: str = "ystar"

    def __post_init__(self) -> None:
        """Validate the specification, inflation basis and supply control."""
        if self.spec not in SPECS:
            raise ValueError(f"spec must be one of {SPECS}, got {self.spec!r}")
        if self.pi_basis not in PI_BASES:
            raise ValueError(f"pi_basis must be one of {PI_BASES}, got {self.pi_basis!r}")
        if self.supply_control not in SUPPLY_CONTROLS:
            raise ValueError(
                f"supply_control must be one of {SUPPLY_CONTROLS}, got {self.supply_control!r}",
            )

    @property
    def scale_constants(self) -> dict[str, float]:
        """Return the fixed variance scale and the ratios this spec uses."""
        constants = {"sigma_c": self.sigma_c}
        if self.spec == "production":
            # No ratio_ystar or ratio_g: potential's growth comes from the
            # factor trends, so there is no free drift state to smooth.
            constants["ratio_gk"] = self.ratio_gk
            constants["ratio_gl"] = self.ratio_gl
            constants["ratio_gm"] = self.ratio_gm
            constants["ratio_a"] = self.ratio_a
        elif self.spec in ("inflation", "core", "target"):
            constants["ratio_ystar"] = self.ratio_ystar
            constants["ratio_g"] = self.ratio_g
        else:
            constants["ratio_pr_star"] = self.ratio_pr_star
            constants["ratio_hpp_star"] = self.ratio_hpp_star
            constants["ratio_lp_star"] = self.ratio_lp_star
            constants["ratio_g_lp"] = self.ratio_g_lp
        return constants

    @property
    def slug(self) -> str:
        """Filename-safe identifier including the variance settings."""
        ratios = "-".join(
            f"{key.replace('ratio_', '')}{value:g}"
            for key, value in self.scale_constants.items()
            if key != "sigma_c"
        )
        return f"{self.name}-{self.spec}-c{self.sigma_c:g}-{ratios}"
