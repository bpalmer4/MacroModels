"""Build, sample and persist the rstar_invert model.

Two axes, both gaps, and a line through the origin:

    (1)  rstar_t = a slow-moving series
    (2)  x_t     = is_slope . (rbar_t - rstarbar_t) + e_t

The gap `x_t` and the real cash rate `r_t` are both given. The latent is r*.
There is NO constant term: at a zero rate gap the economy sits at potential,
which is what neutral means.
"""

import pickle
from pathlib import Path
from typing import Any

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt
import xarray as xr

from src.models.common.model_constants import attach, get_dictionary
from src.models.rstar_invert.config import DEFAULT_OUTPUT_DIR, PAIR_LAGS, ModelConfig
from src.models.rstar_invert.observations import InversionData, build_observations
from src.models.ystar.base import SamplerConfig, sample_model


def _stance_description(config: ModelConfig) -> tuple[str, str]:
    """Return the stance as written out, and how its weights are set."""
    lags = config.rate_lags
    if len(lags) == 1:
        stance = f"(r_{{t-{lags[0]}}} - rstar_{{t-{lags[0]}}})"
        weight_desc = "single lag, no weight to estimate"
    elif len(lags) == PAIR_LAGS:
        first, second = lags
        stance = (
            f"[w.(r_{{t-{first}}} - rstar_{{t-{first}}}) "
            f"+ (1-w).(r_{{t-{second}}} - rstar_{{t-{second}}})]"
        )
        weight_desc = (
            f"w ~ Beta({config.lag_weight_a:g}, {config.lag_weight_b:g})"
            if config.lag_weight_free else f"w FIXED at {config.lag_weight:g}"
        )
    else:
        terms = " + ".join(
            f"w{position + 1}.(r_{{t-{lag}}} - rstar_{{t-{lag}}})"
            for position, lag in enumerate(lags)
        )
        stance = f"[{terms}]"
        weight_desc = (
            f"w ~ Dirichlet({config.lag_weight_conc:g} on each of {len(lags)}), summing to one"
            if config.lag_weight_free else f"w FIXED at 1/{len(lags)} each"
        )
    return stance, weight_desc


def _print_spec(data: InversionData, config: ModelConfig) -> None:
    """Print the specification, leading with what is asserted."""
    stance, weight_desc = _stance_description(config)

    if config.rstar_form == "constant":
        rstar_desc = "rstar_t = rstar_0, FIXED (the reference case, not the story)"
        freedom = "1 free number"
    elif config.rstar_form == "linear":
        rstar_desc = "rstar_t = rstar_0 + trend.t, a straight drift"
        freedom = "2 free numbers"
    else:
        scale = (
            f"ESTIMATED, HalfNormal({config.sigma_rstar_prior:g})" if config.free_sigma_rstar
            else f"ASSERTED at {config.sigma_rstar:g}"
        )
        rstar_desc = f"rstar_t = rstar_{{t-1}} + eta_t,  sigma_rstar {scale}"
        freedom = f"{len(data.frame)} free numbers, one per quarter"

    print("\nModel specification:")
    print(f"  (1) r*:          {rstar_desc}")
    print(f"  (2) the line:    x_t = is_slope . {stance} + e_t")
    print("                   THROUGH THE ORIGIN: no constant term, because a zero")
    print("                   rate gap means the economy is at potential")
    print(f"  Lag weights:     {weight_desc}")
    print(f"  is_slope prior:  TruncatedNormal(mu={config.is_slope_mu:g}, "
          f"sigma={config.is_slope_sigma:g}, upper=0)")
    print("                   sign is THEORY (NK transmission); a posterior on the")
    print("                   boundary means the data wanted the wrong sign")

    estimated = ["is_slope", "sigma_e", "rstar_0"]
    if config.rstar_form == "linear":
        estimated.append("rstar_trend")
    if len(config.rate_lags) > 1 and config.lag_weight_free:
        # The pair keeps the scalar Beta; three or more share the Dirichlet
        # vector. The names differ, so the run log has to say which one is here.
        estimated.append(
            "lag_weight" if len(config.rate_lags) == PAIR_LAGS else "lag_weights",
        )
    if config.rstar_form == "walk" and config.free_sigma_rstar:
        estimated.append("sigma_rstar")
    print(f"  Estimated:       {', '.join(estimated)}")

    usable = int(data.usable.sum())
    print(f"\n  DISCIPLINE:      {usable} observations against an r* with {freedom}")
    if config.rstar_form == "walk":
        print("                   ONE r* PER POINT: r* sets each quarter's position on the")
        print("                   x axis, so every point can slide along it until it meets")
        print("                   the line. Only sigma_rstar restrains that, and it is a")
        print("                   preference, not an equation.")

    gap_sd = float(np.nanstd(data.frame["gap"].to_numpy(dtype=float)))
    print(f"\n  Amplification:   gap sd {gap_sd:.2f} / |is_slope| {abs(config.is_slope_mu):g} "
          f"= {gap_sd / abs(config.is_slope_mu):.1f}pp of r* movement demanded")
    if len(data.excluded):
        print(f"  Excluded:        {len(data.excluded)} quarters, "
              f"{data.excluded.min()} to {data.excluded.max()} (state continues, no likelihood)")


def _lag_weights(config: ModelConfig) -> list[Any]:
    """Return the weights on each lag, summing to one.

    SUMMING TO ONE is the point, not a convenience. It makes the stance the
    response to a SUSTAINED level, so a constant shift in r* shifts the stance
    one for one and r*'s level keeps its meaning. Free coefficients would
    rescale r* silently.

    Two lags share a Beta, three a Dirichlet. Beta(a, b) is Dirichlet([a, b]),
    so the two cases are the same prior; the pair keeps the scalar
    `lag_weight` so its saved traces and charts do not change.
    """
    count = len(config.rate_lags)
    if count == 1:
        return [pt.as_tensor_variable(1.0)]
    if count == PAIR_LAGS:
        if config.lag_weight_free:
            w = pm.Beta("lag_weight", alpha=config.lag_weight_a, beta=config.lag_weight_b)
        else:
            w = pt.as_tensor_variable(config.lag_weight)
        return [w, 1.0 - w]

    if not config.lag_weight_free:
        return [pt.as_tensor_variable(1.0 / count)] * count
    weights = pm.Dirichlet(
        "lag_weights",
        a=np.full(count, config.lag_weight_conc, dtype=float),
    )
    return [weights[position] for position in range(count)]


def _rstar_path(config: ModelConfig, n: int) -> pt.TensorVariable:
    """Return the r* path, in percentage points.

    The form is the answer to "how slow is r*", which is the one genuinely open
    question in this model. A walk gives r* a free value per quarter and the
    fit can never fail; a constant gives it none; linear sits between.
    """
    rstar_0 = pm.Normal("rstar_0", mu=config.rstar_0_mu, sigma=config.rstar_0_sigma)
    if config.rstar_form == "constant":
        return rstar_0 * pt.ones(n)
    if config.rstar_form == "linear":
        trend = pm.Normal("rstar_trend", mu=0.0, sigma=config.rstar_trend_sigma)
        return rstar_0 + trend * pt.as_tensor_variable(np.arange(n, dtype=float))

    # Non-centred walk: the innovations are standard normal and the scale
    # multiplies them, so an asserted `sigma_rstar` is a constant in the graph
    # rather than a variable NUTS has to move through a funnel.
    if config.free_sigma_rstar:
        sigma_rstar = pm.HalfNormal("sigma_rstar", sigma=config.sigma_rstar_prior)
    else:
        sigma_rstar = pt.as_tensor_variable(config.sigma_rstar)
    eta = pm.Normal("eta", mu=0.0, sigma=1.0, shape=n)
    return rstar_0 + sigma_rstar * pt.concatenate([pt.zeros(1), pt.cumsum(eta[1:])])


def build_model(
    data: InversionData,
    config: ModelConfig | None = None,
    *,
    verbose: bool = True,
) -> pm.Model:
    """Build the line through the origin, with r* as the latent.

    KNOWN PROPERTY, and it must be carried into any reading of the output. With
    the prior on r* in rate units, `is_slope` enters twice: once as the
    steepness and again multiplying r* to position the line. The line's
    vertical freedom is |is_slope| x r*'s freedom, so a larger slope buys a
    more movable line and the likelihood has an incentive to inflate it.
    Measured: sigma_rstar 0.02 gave -0.032, sigma_rstar 0.10 gave -0.383, with
    almost the same fit. See `config.py`.
    """
    config = config or ModelConfig()
    n = len(data.frame)
    gap = data.frame["gap"].to_numpy(dtype=float)
    # Zero-filled: the unusable rows carry no likelihood, but NaNs would still
    # poison the gradient, so the mask does the excluding rather than the NaNs.
    rate_lagged = np.nan_to_num(data.rate_lagged, nan=0.0)
    usable = data.usable

    model = pm.Model()
    with model:
        attach(model, config.constants)

        # THE ASSERTION. Truncated above at zero because a positive slope is
        # not a weak IS curve, it is no IS curve. A posterior piled against the
        # bound means the likelihood wanted the wrong sign and the theory
        # constraint stopped it: a rejection at this specification, never a
        # measurement of a small negative slope.
        is_slope = pm.TruncatedNormal(
            "is_slope",
            mu=config.is_slope_mu,
            sigma=config.is_slope_sigma,
            upper=0.0,
        )
        sigma_e = pm.HalfNormal("sigma_e", sigma=config.sigma_e_prior)

        # (1) r*, real, per cent. No nominal counterpart: the rate side was
        # deflated before it reached this model.
        rstar = pm.Deterministic("rstar", _rstar_path(config, n))

        # (2) The x axis: the RATE GAP, weighted across the lags, with the
        # latent put through the identical lag treatment as the observed rate.
        # Identical matters: if the rate were lagged and r* were not, part of
        # their difference would be a timing artefact rather than a stance.
        #
        # ONE stance in this package: the same object drives the likelihood and
        # appears on the charts.
        weights = _lag_weights(config)
        terms = []
        for position, lag in enumerate(config.rate_lags):
            shifted = (
                pt.concatenate([pt.zeros(lag), rstar[:-lag]]) if lag < n else pt.zeros(n)
            )
            observed = pt.as_tensor_variable(rate_lagged[:, position])
            terms.append(weights[position] * (observed - shifted))
        stance = pm.Deterministic(
            "stance", pt.add(*terms) if len(terms) > 1 else terms[0],
        )
        # The observed half of the x axis on its own, so charts can show the
        # unfitted picture without rebuilding the lag structure.
        pm.Deterministic("rate_bar", pt.add(*[
            weights[position] * pt.as_tensor_variable(rate_lagged[:, position])
            for position in range(len(config.rate_lags))
        ]) if len(config.rate_lags) > 1 else weights[0] * pt.as_tensor_variable(rate_lagged[:, 0]))

        # THROUGH THE ORIGIN. No constant: at a zero rate gap the economy is at
        # potential. A free constant would in any case be the same parameter as
        # r*'s level, since shifting it by d is shifting r* by d/is_slope.
        mu = is_slope * stance
        pm.Deterministic("gap_fitted", mu)
        pm.Normal("obs", mu=mu[usable], sigma=sigma_e, observed=gap[usable])

    if verbose:
        _print_spec(data, config)

    return model


def save_results(
    trace: az.InferenceData,
    data: InversionData,
    constants: dict[str, Any],
    output_dir: Path | str | None = None,
    prefix: str = "rstar_invert",
) -> None:
    """Persist the trace and the observations beside it."""
    directory = Path(output_dir) if output_dir else DEFAULT_OUTPUT_DIR
    directory.mkdir(parents=True, exist_ok=True)
    trace_path = directory / f"{prefix}_trace.nc"
    trace.to_netcdf(str(trace_path))
    with (directory / f"{prefix}_obs.pkl").open("wb") as handle:
        pickle.dump(
            {"frame": data.frame, "excluded": data.excluded, "constants": constants},
            handle,
        )
    print(f"\nSaved trace to: {trace_path}")


def run_estimate(
    config: ModelConfig | None = None,
    sampler_config: SamplerConfig | None = None,
    prefix: str = "rstar_invert",
    *,
    verbose: bool = True,
    seed: int | None = None,
) -> tuple[az.InferenceData, InversionData]:
    """Build the observations, sample, and save."""
    config = config or ModelConfig()
    sampler_config = sampler_config or SamplerConfig()
    if seed is not None:
        sampler_config.random_seed = seed

    data = build_observations(config, verbose=True)
    model = build_model(data, config, verbose=verbose)
    print("\nSampling...")
    trace = sample_model(model, sampler_config)

    save_results(
        trace, data,
        constants={**get_dictionary(model), "sources": data.sources.to_records()},
        output_dir=config.output_dir, prefix=prefix,
    )
    return trace, data


def load_results(
    output_dir: Path | str | None = None,
    prefix: str = "rstar_invert",
) -> tuple[az.InferenceData, pd.DataFrame, dict[str, Any]]:
    """Load a completed run: trace, observations, constants."""
    directory = Path(output_dir) if output_dir else DEFAULT_OUTPUT_DIR
    trace = az.from_netcdf(str(directory / f"{prefix}_trace.nc"))
    with (directory / f"{prefix}_obs.pkl").open("rb") as handle:
        saved = pickle.load(handle)
    return trace, saved["frame"], saved["constants"]


def _posterior(trace: az.InferenceData) -> xr.Dataset:
    """Return the posterior group, narrowed."""
    posterior = getattr(trace, "posterior", None)
    if not isinstance(posterior, xr.Dataset):
        raise TypeError("trace has no posterior group - did sampling complete?")
    return posterior


def scalar_draws(trace: az.InferenceData, name: str) -> np.ndarray:
    """Return every draw of a scalar parameter, flattened."""
    return np.asarray(_posterior(trace)[name].values).ravel()


def posterior_median(trace: az.InferenceData, name: str, index: pd.PeriodIndex) -> pd.Series:
    """Return the posterior median of a vector latent as a series."""
    stacked = _posterior(trace)[name].stack(sample=("chain", "draw"))
    return pd.Series(np.asarray(stacked.median("sample").values), index=index)


def posterior_band(
    trace: az.InferenceData,
    name: str,
    index: pd.PeriodIndex,
    lower: float = 5.0,
    upper: float = 95.0,
) -> pd.DataFrame:
    """Return a percentile band on a vector latent.

    CONDITIONAL, and the charts must say so. This is the uncertainty GIVEN the
    asserted slope prior and the asserted speed of r*, not the uncertainty
    about r*.
    """
    stacked = _posterior(trace)[name].stack(sample=("chain", "draw"))
    values = np.asarray(stacked.values)
    return pd.DataFrame(
        {
            "lower": np.percentile(values, lower, axis=-1),
            "upper": np.percentile(values, upper, axis=-1),
        },
        index=index,
    )
