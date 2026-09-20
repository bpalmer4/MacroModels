"""Build, sample and persist the rstar_rba model."""

import pickle
from pathlib import Path
from typing import Any

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt
import xarray as xr

from src.data.aofm_loader import get_aofm_5y5y_forward
from src.data.cash_rate import get_cash_rate_qrtly
from src.data.fred_loader import get_fred_quarterly
from src.data.gdp import get_gdp_growth
from src.data.inflation import get_trimmed_mean_qrtly
from src.models.common.sources import SourceSet
from src.models.rstar_rba.config import DEFAULT_OUTPUT_DIR, WORLD_REAL_RATE, ModelConfig
from src.models.ystar.base import SamplerConfig, get_fixed_constants, sample_model
from src.models.ystar_ustar.results import load_results as load_joint


def _jump_series(config: ModelConfig, sources: SourceSet) -> pd.Series:
    """Return the series whose abrupt moves flag a discontinuity in neutral.

    Already differenced, so `jump_mask` only has to take absolute values and a
    percentile. See `ModelConfig.jump_source` for why the world real rate is the
    default and GDP is not.
    """
    if config.jump_source == "gdp":
        return sources.take(get_gdp_growth()).astype(float)
    if config.jump_source == "gdp4":
        annual = sources.take(get_gdp_growth(periods=4)).astype(float)
        return annual - annual.mean()
    world = get_fred_quarterly(WORLD_REAL_RATE)
    sources.add("Federal Reserve Bank of Cleveland via FRED", WORLD_REAL_RATE)
    return world.diff()


def _unemployment_gap(config: ModelConfig, sources: SourceSet) -> pd.Series:
    """Return u - u* from a completed `ystar_ustar` run.

    A MODEL OUTPUT, not data. Imported at its posterior median, so this model
    treats as known a quantity the joint model estimates, and inherits its
    conditioning. See `ModelConfig.employment` for why that is accepted.

    Raises rather than warning if the run is missing: with `employment` on, the
    term is the specification, so silently dropping it would estimate a
    different model than the one asked for.
    """
    joint = load_joint(prefix=config.ustar_prefix)
    recorded = SourceSet.from_records(joint.constants.get("sources"))
    if recorded is not None:
        for source, cat in recorded.records:
            sources.add(source, cat)
    sources.add(f"ystar_ustar run ({config.ustar_prefix})", "u*")
    return joint.ugap_median()


def _forward(config: ModelConfig, sources: SourceSet) -> pd.Series:
    """Return the AOFM 5y5y risk-neutral forward, quarterly, in per cent.

    THE SECOND WINDOW. With the cash rate alone this model cannot separate the
    LEVEL of neutral from the level of the stance: taking the sample average of
    `r = b + lambda.g + eps` gives `mean(b) = mean(r) - lambda.mean(g)`, so
    neutral's level was the historical average cash rate, adjusted for whether
    inflation averaged on target. Nothing else pinned it, and no
    reparameterisation could, because one observable cannot identify two levels.

    The 5y5y forward is a market price for where the cash rate settles over
    years five to ten, with AOFM's model stripping the term premium. It is the
    only series available that speaks to the LEVEL of neutral without being the
    cash rate's own history.

    IT LEADS POLICY, IT DOES NOT ECHO IT. Measured 2026-09-16 on quarterly
    changes: corr(d5y5y_t, dcash_{t+k}) peaks at +0.340 at k = +2 and is
    NEGATIVE at k = -1 and -2. So it moves two to three quarters before the cash
    rate and does not chase past decisions. Its sd is 0.99 against the cash
    rate's 1.97, so it carries half the volatility, which is what a forward that
    has stripped the cycle should look like.

    WHAT IT IS NOT. Not neutral itself: it still contains the market's view of
    the cyclical position over years five to ten, and whatever premium AOFM did
    not remove. That is why the observation equation carries a free bias term,
    which is now what holds the level. The assertion has MOVED, not vanished.
    The series is also re-estimated full-sample monthly, so it revises.
    """
    series = sources.take(get_aofm_5y5y_forward(config.forward_method)).astype(float).dropna()
    return series.groupby(pd.PeriodIndex(series.index, freq="Q")).mean()


def build_observations(
    config: ModelConfig,
    *,
    verbose: bool = False,
) -> tuple[pd.DataFrame, SourceSet]:
    """Return the cash rate, the averaged inflation rate and the jump timer.

    The cash rate is nominal, so no deflator is needed: the real-rate version of
    this model had to import an expectations series, which is itself a model
    output, and this avoids it.

    The rule uses two series and nothing else, which is the default. A third is
    loaded only when `jumps` is on, and it never enters the rule: it only says
    which quarters may carry a wider innovation. `--jumps` is a sensitivity test
    and brings that dependency with it.
    """
    sources = SourceSet()
    cash = sources.take(get_cash_rate_qrtly()).astype(float)
    quarterly = sources.take(get_trimmed_mean_qrtly()).astype(float)

    columns = {"r": cash, "q": quarterly}
    if config.use_forward:
        columns["f"] = _forward(config, sources)
    if config.employment:
        columns["ugap"] = _unemployment_gap(config, sources)
    if config.jumps:
        # Only loaded when it is used, so the package stays estimable from two
        # series in its default form. Neither series enters the rule; whichever
        # is loaded times the base's innovations and nothing else.
        columns["jump_series"] = _jump_series(config, sources)
    frame = pd.DataFrame(columns).dropna()
    # The lag matrix, not a precomputed `pi`: with the weights estimated inside
    # the model, `pi_t` is a model quantity. Column j is q_{t-j}. Built from the
    # quarterly series, never the year-ended one, which would average
    # overlapping windows and smear the response across twice the intended span.
    for j in range(config.max_lag + 1):
        frame[f"q_lag{j}"] = frame["q"].shift(j)
    frame = frame.dropna()

    if config.start:
        frame = frame.loc[config.start:]
    if config.end:
        frame = frame.loc[: config.end]

    index = frame.index
    if not isinstance(index, pd.PeriodIndex):
        index = pd.PeriodIndex(index, freq="Q")
    frame.index = index.asfreq("Q")

    if verbose:
        flat = frame[[f"q_lag{j}" for j in range(config.window)]].mean(axis=1) * 4.0
        print(f"Sample: {frame.index.min()} to {frame.index.max()}  ({len(frame)} quarters)")
        print(f"  cash rate                mean {frame['r'].mean():6.2f}  sd {frame['r'].std():5.2f}")
        print(f"  inflation, flat w={config.window}      mean {flat.mean():6.2f}  sd {flat.std():5.2f}")
        print(f"  corr(cash rate, that inflation gap)   {frame['r'].corr(flat - config.anchor):5.2f}")
        print(f"  lags carried: 0 to {config.max_lag}")

    return frame, sources


def lag_matrix(frame: pd.DataFrame, max_lag: int) -> np.ndarray:
    """Return the (n, max_lag+1) matrix of quarterly inflation lags."""
    return frame[[f"q_lag{j}" for j in range(max_lag + 1)]].to_numpy(dtype=float)


def _weights(config: ModelConfig) -> tuple[pt.TensorVariable, str]:
    """Return the lag weights, summing to one, and a description.

    Must be called inside the model context, since two of the three create
    variables.
    """
    n_lags = config.max_lag + 1

    if config.weights == "flat":
        fixed = np.zeros(n_lags)
        fixed[: config.window] = 1.0 / config.window
        return pt.as_tensor_variable(fixed), f"flat over {config.window} quarters (imposed)"

    if config.weights == "geometric":
        rho = pm.Beta("rho", alpha=config.rho_a, beta=config.rho_b)
        powers = pt.as_tensor_variable(np.arange(n_lags, dtype=float))
        raw = rho**powers
        weights = pm.Deterministic("weights", raw / pt.sum(raw))
        pm.Deterministic("mean_lag", pt.sum(weights * powers))
        return weights, f"geometric, rho estimated, {n_lags} lags"

    weights = pm.Dirichlet("weights", a=np.full(n_lags, config.dirichlet_alpha))
    pm.Deterministic(
        "mean_lag", pt.sum(weights * pt.as_tensor_variable(np.arange(n_lags, dtype=float))),
    )
    return weights, f"Dirichlet, free over {n_lags} lags"


def _unconstrained(frame: pd.DataFrame, config: ModelConfig) -> np.ndarray:
    """Return the boolean mask of quarters the rule could actually have set.

    Quarters at or below the effective lower bound are excluded: the response
    the inflation gap called for was not deliverable, so scoring them would
    read a truncated rate gap as a weak reaction. See `ModelConfig.floor`.
    """
    if config.floor is None:
        return np.ones(len(frame), dtype=bool)
    return (frame["r"].to_numpy(dtype=float) > config.floor)


def _split_mask(frame: pd.DataFrame, split: str) -> np.ndarray:
    """Return the boolean mask of quarters at or after `split`."""
    index = frame.index
    if not isinstance(index, pd.PeriodIndex):
        index = pd.PeriodIndex(index, freq="Q")
    return (index >= pd.Period(split, freq="Q")).astype(bool)


# Degrees of freedom standing in for a Normal on the unflagged quarters. At 200
# the StudentT and the Normal differ by well under a thousandth of a standard
# deviation in the tails, which is far below anything this model reports.
_GAUSSIAN_NU = 200.0


def jump_mask(frame: pd.DataFrame, config: ModelConfig) -> np.ndarray:
    """Return the boolean mask of quarters the economy moved abruptly.

    Flagged where the absolute quarterly change in the timing series sits at or
    above `jump_percentile` of its own distribution over the estimation sample.
    See `ModelConfig.jumps` for why an external series times this and what it
    costs, and `jump_source` for which series.
    """
    if not config.jumps or "jump_series" not in frame.columns:
        return np.zeros(len(frame), dtype=bool)
    size = frame["jump_series"].abs().to_numpy(dtype=float)
    cut = float(np.percentile(size, config.jump_percentile))
    return size >= cut


def _innovations(frame: pd.DataFrame, config: ModelConfig, n: int) -> pt.TensorVariable:
    """Return the standardised innovations driving the base.

    Must be called inside the model context. Standard Normal everywhere, except
    that with `jumps` on the flagged quarters draw from a StudentT with `nu`
    imposed, so neutral is permitted a step where the economy stepped.

    One vector with a per-quarter `nu` rather than two variables spliced
    together: splicing would give the sampler a discontinuous parameterisation
    and the flagged quarters would come out of a different geometry from their
    neighbours. A large `nu` off the flags is a Normal to well past the
    precision anything here is quoted at.
    """
    flagged = jump_mask(frame, config)
    if not flagged.any():
        return pm.Normal("z", mu=0.0, sigma=1.0, shape=n)
    nu = np.where(flagged, config.jump_nu, _GAUSSIAN_NU)
    return pm.StudentT("z", nu=pt.as_tensor_variable(nu), mu=0.0, sigma=1.0, shape=n)


def _print_jumps(frame: pd.DataFrame, config: ModelConfig) -> None:
    """Print which quarters were flagged as discontinuities, if any."""
    flagged = jump_mask(frame, config)
    if not flagged.any():
        return
    dates = frame.index[flagged]
    print(f"  Jumps:      {int(flagged.sum())} quarters above the "
          f"{config.jump_percentile:g}th percentile of |d {config.jump_source} q/q|, "
          f"StudentT nu={config.jump_nu:g} imposed there")
    print(f"              {', '.join(str(d) for d in dates)}")


def _lambda(frame: pd.DataFrame, config: ModelConfig) -> pt.TensorVariable:
    """Return the inflation-response coefficient, one value or two.

    Must be called inside the model context. With `lambda_split` set, two
    coefficients share one prior and are selected by quarter. Written as a mask
    rather than two likelihoods so neutral, the weights and `sigma_eps` stay
    pooled: only the response is allowed to break, which is the question.
    """
    if config.lambda_split is None:
        return pm.Normal("lambda", mu=config.lambda_mu, sigma=config.lambda_sigma)

    late = _split_mask(frame, config.lambda_split)
    lam_early = pm.Normal("lambda", mu=config.lambda_mu, sigma=config.lambda_sigma)
    lam_late = pm.Normal("lambda_late", mu=config.lambda_mu, sigma=config.lambda_sigma)
    # The break itself, so the posterior answers "did it change" directly rather
    # than by eyeballing two overlapping intervals.
    pm.Deterministic("lambda_break", lam_late - lam_early)
    return pt.where(pt.as_tensor_variable(late), lam_late, lam_early)


def _print_spec(frame: pd.DataFrame, config: ModelConfig, weights_desc: str) -> None:
    """Print the specification actually built, read off the config."""
    form = "random walk, sigma_r imposed" if config.walk else "constant"
    shape = "lambda_1 x g_t + lambda_2 x g_t x |g_t|" if config.nonlinear else "lambda x g_t"
    if config.employment:
        shape += " + lambda_u x (u_t - u*_t)"

    print("\nModel specification:")
    print(f"  Inflation:  pi_t = 4 x sum_j w_j q_{{t-j}},  weights {weights_desc}")
    # The band division has to be printed. Without it the line reads as though
    # `lambda` were per percentage point, which is the units trap MODEL_NOTES.md
    # warns about: it is per band-width, so per point is twice it.
    print(f"  Two gaps:   r_t - b_t = {shape} + eps_t,  "
          f"g_t = (pi_t - {config.anchor:g})/{config.band:g}")
    print(f"  Prescribed: d_t = b_t + {shape}")
    print(f"  Neutral b_t: {form}")

    kept = _unconstrained(frame, config)
    dropped = frame.index[~kept]
    if len(dropped):
        print(f"  Excluded:   {len(dropped)} quarters at or below the {config.floor:g} floor, "
              f"{dropped.min()} to {dropped.max()}")

    if config.lambda_split is not None:
        late = _split_mask(frame, config.lambda_split)
        print(f"  Split:      lambda breaks at {config.lambda_split}, "
              f"{int((~late).sum())} early quarters, {int(late.sum())} late")

    if config.employment:
        print(f"  Employment: u - u* from the {config.ustar_prefix} run, a MODEL OUTPUT, "
              f"lambda_u ~ N({config.lambda_u_mu:g}, {config.lambda_u_sigma:g})")

    _print_jumps(frame, config)
    print(f"  Priors:     lambda ~ N({config.lambda_mu:g}, {config.lambda_sigma:g}), "
          f"b_0 ~ N({config.base_mu:g}, {config.base_sigma:g})")


def _forward_window(config: ModelConfig, frame: pd.DataFrame, neutral: pt.TensorVariable) -> None:
    """Attach the market's 5y5y forward as a second observation on neutral.

    `f_t = neutral_t + bias + e_t`. The bias is what the forward carries that
    neutral does not: the market's view of the cycle over years five to ten,
    plus whatever premium AOFM did not strip.

    THE BIAS NOW HOLDS THE LEVEL. With the cash rate alone that job fell to the
    cash rate's own sample mean, by the identity
    `mean(b) = mean(r) - lambda.mean(g)`, which is why this model could not
    report that the whole level of neutral had shifted. A free bias with a wide
    prior would hand the level straight back, so the prior IS the assertion and
    should be quoted as one rather than presented as an identification.

    Must be called inside the model context, since it creates variables.
    """
    if not config.use_forward or "f" not in frame:
        return
    bias = pm.Normal("forward_bias", mu=config.forward_bias_mu, sigma=config.forward_bias_sigma)
    sigma_f = pm.HalfNormal("sigma_f", sigma=config.sigma_f_sigma)
    pm.Deterministic("forward_fitted", neutral + bias)
    pm.Normal(
        "obs_forward",
        mu=neutral + bias,
        sigma=sigma_f,
        observed=frame["f"].to_numpy(dtype=float),
    )


def build_model(frame: pd.DataFrame, config: ModelConfig, *, verbose: bool = True) -> pm.Model:
    """Build the two-gaps model.

        r_t - b_t = lambda · (pi_t - anchor) + eps_t
        d_t       = b_t + lambda · (pi_t - anchor)

    `b_t` is NEUTRAL, the slow-moving trend, recorded as `neutral`. Adding the
    inflation response gives the rule's prescribed rate, recorded as
    `prescribed`, which is what the cash rate is compared against in the
    likelihood but is NOT neutral: the response is a departure from it.

    With `walk` off the base is one constant. With it on, `b_t` is a Gaussian
    random walk whose innovation sd is imposed, because the walk and `lambda`
    compete for the same downward drift in the cash rate and a free `sigma_r`
    lets the walk take all of it.
    """
    n = len(frame)
    lags = lag_matrix(frame, config.max_lag)
    rate = frame["r"].to_numpy()

    model = pm.Model()
    with model:
        if not hasattr(model, "_fixed_constants"):
            model._fixed_constants = {}  # noqa: SLF001 — our own metadata, as elsewhere
        model._fixed_constants.update(config.constants)  # noqa: SLF001

        weights, weights_desc = _weights(config)
        # Annualised: the weights sum to one, so this is a weighted average
        # quarterly rate, times four.
        pi = pm.Deterministic("pi", 4.0 * pt.dot(pt.as_tensor_variable(lags), weights))
        # Scaled so |g| = 1 at the band edge; see `ModelConfig.band`.
        gap = pm.Deterministic("inflation_gap", (pi - config.anchor) / config.band)

        lam = _lambda(frame, config)
        # `g·|g|` keeps the sign, so the response stays odd-symmetric while
        # rising faster than linearly in the size of the gap. `lambda_2 = 0` is
        # the linear model, so the posterior is a test rather than an assumption.
        response = lam * gap
        if config.employment:
            # The second leg of the mandate. Negative `lambda_u` means the RBA
            # cuts when unemployment sits above u*, which is the expected sign.
            lam_u = pm.Normal("lambda_u", mu=config.lambda_u_mu, sigma=config.lambda_u_sigma)
            ugap = pt.as_tensor_variable(frame["ugap"].to_numpy(dtype=float))
            pm.Deterministic("employment_response", lam_u * ugap)
            response = response + lam_u * ugap
        if config.nonlinear:
            lam2 = pm.Normal("lambda_2", mu=config.lambda2_mu, sigma=config.lambda2_sigma)
            response = response + lam2 * gap * pt.abs(gap)
        pm.Deterministic("response", response)
        sigma_eps = pm.HalfNormal("sigma_eps", sigma=config.sigma_eps_sigma)
        base_0 = pm.Normal("base_0", mu=config.base_mu, sigma=config.base_sigma)

        # NEUTRAL: the slow-moving piece, which is what this package calls the
        # neutral rate. The inflation response is a departure FROM it, not part
        # of it, so `neutral + response` is the rule's prescribed rate below.
        if config.walk:
            innovations = _innovations(frame, config, n)
            neutral = pm.Deterministic(
                "neutral",
                base_0 + config.sigma_r * pt.concatenate([pt.zeros(1), pt.cumsum(innovations[1:])]),
            )
        else:
            neutral = pm.Deterministic("neutral", base_0 * pt.ones(n))

        # Both scales are recorded, so anything downstream reads the one it
        # wants rather than re-deriving it and guessing at the deflator. Real
        # is nominal less the target, because a neutral rate is defined at
        # target inflation, not at whatever inflation happened to be.
        #
        # NAMES MATTER HERE and these ones were wrong until they were fixed.
        # `neutral` is the slow piece on its own. `prescribed` is what the rule
        # says the cash rate should be today, neutral plus the inflation
        # response, and it is NOT neutral. `stance` is the cash rate against
        # neutral, which is what the word means. `rule_residual` is the cash
        # rate against the rule, which is what the old `stance` actually held.
        prescribed = pm.Deterministic("prescribed", neutral + response)
        pm.Deterministic("neutral_real", neutral - config.anchor)
        pm.Deterministic("prescribed_real", prescribed - config.anchor)
        pm.Deterministic("stance", pt.as_tensor_variable(rate) - neutral)
        pm.Deterministic("rule_residual", pt.as_tensor_variable(rate) - prescribed)
        # Excluded quarters keep their place in the state, so neutral still evolves
        # through the floor years, but carry no likelihood: the rule did not
        # generate them. Same treatment `ystar` gives the lockdown quarters.
        _forward_window(config, frame, neutral)

        kept = _unconstrained(frame, config)
        if not config.partial_adjustment:
            pm.Normal("obs", mu=prescribed[kept], sigma=sigma_eps, observed=rate[kept])
        else:
            # r_t = phi·r_{t-1} + (1 - phi)·d_t + eps_t, with d_t the desired
            # rate the rule is moving toward. The lag is the OBSERVED cash rate,
            # so the first quarter has no predictor and leaves the likelihood.
            #
            # `sigma_eps` means something different here: the sd of the
            # quarterly INNOVATION rather than of the level around the rule, so
            # it is far smaller for that reason alone. Do not compare it with
            # the default run's.
            phi = pm.Beta("phi", alpha=config.phi_a, beta=config.phi_b)
            lagged = pt.as_tensor_variable(np.concatenate([[rate[0]], rate[:-1]]))
            usable = kept.copy()
            usable[0] = False
            pm.Deterministic("adjustment_gap", pt.as_tensor_variable(rate) - prescribed)
            pm.Normal(
                "obs",
                mu=(phi * lagged + (1.0 - phi) * prescribed)[usable],
                sigma=sigma_eps,
                observed=rate[usable],
            )

    if verbose:
        _print_spec(frame, config, weights_desc)

    return model


def save_results(
    trace: az.InferenceData,
    frame: pd.DataFrame,
    constants: dict[str, Any],
    output_dir: Path | str | None = None,
    prefix: str = "rstar_rba",
) -> None:
    """Persist the trace and the observations beside it."""
    directory = Path(output_dir) if output_dir else DEFAULT_OUTPUT_DIR
    directory.mkdir(parents=True, exist_ok=True)
    trace_path = directory / f"{prefix}_trace.nc"
    trace.to_netcdf(str(trace_path))
    with (directory / f"{prefix}_obs.pkl").open("wb") as handle:
        pickle.dump({"frame": frame, "constants": constants}, handle)
    print(f"\nSaved trace to: {trace_path}")


def run_estimate(
    config: ModelConfig | None = None,
    sampler_config: SamplerConfig | None = None,
    prefix: str = "rstar_rba",
    *,
    verbose: bool = False,
    seed: int | None = None,
) -> az.InferenceData:
    """Build the observations, sample, and save."""
    config = config or ModelConfig()
    sampler_config = sampler_config or SamplerConfig()
    if seed is not None:
        sampler_config.random_seed = seed

    frame, sources = build_observations(config, verbose=True)
    model = build_model(frame, config, verbose=verbose)
    print("\nSampling...")
    trace = sample_model(model, sampler_config)

    save_results(
        trace, frame,
        constants={**get_fixed_constants(model), "sources": sources.to_records()},
        output_dir=config.output_dir, prefix=prefix,
    )
    return trace


def load_results(
    output_dir: Path | str | None = None,
    prefix: str = "rstar_rba",
) -> tuple[az.InferenceData, pd.DataFrame, dict[str, Any]]:
    """Load a completed run: trace, observations, constants."""
    directory = Path(output_dir) if output_dir else DEFAULT_OUTPUT_DIR
    trace = az.from_netcdf(str(directory / f"{prefix}_trace.nc"))
    with (directory / f"{prefix}_obs.pkl").open("rb") as handle:
        saved = pickle.load(handle)  # noqa: S301 — our own file
    return trace, saved["frame"], saved["constants"]


def posterior_median(trace: az.InferenceData, name: str, index: pd.PeriodIndex) -> pd.Series:
    """Return the posterior median of a vector latent as a series."""
    posterior = getattr(trace, "posterior", None)
    if not isinstance(posterior, xr.Dataset):
        raise TypeError("trace has no posterior group - was it loaded from a completed run?")
    stacked = posterior[name].stack(sample=("chain", "draw"))  # noqa: PD013
    return pd.Series(np.asarray(stacked.median("sample").values), index=index)
