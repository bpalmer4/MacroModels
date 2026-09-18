"""Build, sample and persist the regime u* model."""

import pickle
from pathlib import Path  # noqa: TC003 — used at runtime in function signatures
from typing import Any

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm
import pytensor
import pytensor.tensor as pt
import xarray as xr

from src.models.regime_ustar.config import ModelConfig
from src.models.regime_ustar.spline import basis
from src.models.ystar.base import SamplerConfig, sample_model


def _ustar_spline(frame: pd.DataFrame, config: ModelConfig) -> Any:  # noqa: ANN401
    """u* as a natural cubic spline with knots at the regime dates.

        ustar_t = sum_j c_j B_j(t)

    Deterministic given the coefficients, so there is no innovation variance
    to impose. That removes `sigma_ustar`, which nothing measured and which
    is what the deleted long-run model died on.

    The basis is a partition of unity, its rows summing to 1 at machine
    precision, so `c_j` are in unemployment-rate units and take the same prior
    the attractors did. A coefficient IS roughly the level u* passes through
    near its knot, rather than an abstract weight.

    Natural, so the curve is linear beyond the outer knots. The last segment
    runs 2020Q1 to the end with no knot after it and the largest inflation
    surprises in the sample under it, and a free cubic extrapolating at a
    boundary is the worst case for this basis.
    """
    index = frame.index
    if not isinstance(index, pd.PeriodIndex):
        raise TypeError("the observation frame must carry a quarterly PeriodIndex")
    design = basis(
        index, config.breaks,
        natural=config.natural_spline,
        multiplicity=config.knot_multiplicity,
    )
    mu0, sd0, lo, hi = config.eq_prior
    coef = pm.TruncatedNormal("coef", mu=mu0, sigma=sd0, lower=lo, upper=hi, shape=design.shape[1])
    return pm.Deterministic("ustar", pt.dot(pt.as_tensor_variable(design), coef))


def _ustar_state(frame: pd.DataFrame, regimes: np.ndarray, config: ModelConfig) -> Any:  # noqa: ANN401
    """u* as ONE continuous state whose law of motion changes at each regime.

        u*_t = u*_{t-1} + phi_k x (eq_k - u*_{t-1}) + e_t,   k = regime(t)

    Level continuity across a regime boundary is structural, not imposed:
    there is one state and it is carried forward, so nothing can jump at a
    date. What changes at the boundary is where u* is being pulled and how
    hard, which is a change in direction rather than in position.

    The innovation sd is imposed. This is the dial that sank the deleted
    long-run model, and leaving it free to a weak Phillips likelihood is how
    u* becomes the unemployment rate again. The attractors are what is meant
    to move u*; `sigma_ustar` is only meant to let it wander a little around
    the path they set.

    With `config.adaptive_sigma` the scale is no longer flat across the
    sample: it is keyed to how far the one-year average of unemployment has
    moved over the preceding two years, so u* may innovate more in 1974-75,
    1990-91 and 2020 than in the quiet stretches. `kappa` = 0 recovers the
    constant-sigma model exactly, so the posterior on `kappa` tests the idea.
    See `ModelConfig.adaptive_sigma` for the objection to it.

    Non-centred innovations, as everywhere else in this package.
    """
    n = len(frame)
    n_regimes = int(regimes.max()) + 1
    mu0, sd0, lo, hi = config.eq_prior
    pmu, psd, plo, phi_hi = config.phi_prior

    eq = pm.TruncatedNormal("eq", mu=mu0, sigma=sd0, lower=lo, upper=hi, shape=n_regimes)
    phi_shape = n_regimes if config.free_phi_per_regime else 1
    phi = pm.TruncatedNormal("phi", mu=pmu, sigma=psd, lower=plo, upper=phi_hi, shape=phi_shape)

    # The innovation sd, constant or keyed to how much unemployment has moved.
    if config.adaptive_sigma:
        move = np.asarray(frame["u_move"], dtype=float)
        rel = np.maximum(config.move_floor, move / np.nanmean(move))
        kmu, ksd, klo, khi = config.kappa_prior
        kappa = pm.TruncatedNormal("kappa", mu=kmu, sigma=ksd, lower=klo, upper=khi)
        sigma_t = config.sigma_ustar * pt.power(pt.as_tensor_variable(rel), kappa)
        pm.Deterministic("sigma_ustar_t", sigma_t)
    else:
        sigma_t = pt.as_tensor_variable(np.full(n, config.sigma_ustar))

    z = pm.Normal("z_ustar", mu=0.0, sigma=1.0, shape=n)
    init = pm.Normal("ustar_init", mu=float(frame["u"].iloc[0]), sigma=3.0)

    # Per-quarter attractor and speed, so the scan carries no regime logic.
    eq_t = eq[regimes]
    phi_t = phi[regimes] if config.free_phi_per_regime else pt.repeat(phi, n)

    def step(target: Any, speed: Any, eps: Any, prev: Any) -> Any:  # noqa: ANN401
        return prev + speed * (target - prev) + eps

    path, _ = pytensor.scan(
        fn=step,
        sequences=[eq_t[1:], phi_t[1:], z[1:] * sigma_t[1:]],
        outputs_info=[init],
    )
    return pm.Deterministic("ustar", pt.concatenate([[init], path]))


def _wage_equation(frame: pd.DataFrame, ustar: Any, u_obs: np.ndarray, config: ModelConfig) -> None:  # noqa: ANN401
    """Add the second observation equation, on unit labour costs.

        ulc_yoy - pi_e = alpha + gamma x (u - ustar)/u + lambda x dU/U + v

    This is the only part of the model that attacks the circularity. The price
    equation alone makes `u - ustar` the inflation gap scaled by `u/beta`, so
    no labour-market observable enters except `u` itself. Unit labour costs are
    a labour-market price, so the same `ustar` now has a second thing to answer
    to, and the two equations can disagree.

    Form follows `nairu/equations/phillips_wage.py`: the same proportional gap,
    expected inflation entering with a coefficient of one (so the dependent
    variable is real ULC growth), a free intercept absorbing trend productivity
    rather than netting it off, and the speed-limit term on the CHANGE in
    unemployment. Written year-ended rather than `nairu`'s quarterly, so
    `gamma` and the price equation's `beta` are in the same units and their
    agreement is a diagnostic rather than a units comparison.

    `gamma` is constrained non-positive, as `nairu` constrains its own: the
    content of a wage Phillips curve is that slack slows wage growth.
    """
    gmu, gsd, gupper = config.gamma_wage_prior
    lmu, lsd = config.lambda_wage_prior

    alpha = pm.Normal("alpha_wage", mu=0.0, sigma=config.alpha_wage_prior_sd)
    gamma = pm.TruncatedNormal("gamma_wage", mu=gmu, sigma=gsd, upper=gupper)
    lam = pm.Normal("lambda_wage", mu=lmu, sigma=lsd)
    sigma_w = pm.HalfNormal("sigma_wage", sigma=config.sigma_wage_prior_sd)

    mu_w = (
        alpha
        + gamma * (u_obs - ustar) / u_obs
        + lam * np.asarray(frame["speed"], dtype=float)
    )
    pm.Normal(
        "ulc_obs",
        mu=mu_w,
        sigma=sigma_w,
        observed=np.asarray(frame["ulc"] - frame["pi_e"], dtype=float),
    )


def build_model(frame: pd.DataFrame, regimes: np.ndarray, config: ModelConfig) -> pm.Model:
    """Return the state law above, observed through one accelerationist Phillips curve.

        pi_t - pi^e_t = -beta x (u_t - u*_t) / u_t + supply_t + e_t

    and, when `config.wage_equation`, a SECOND observation on the same u*
    built from unit labour costs. See `_wage_equation`: without it the
    unemployment gap is algebraically the inflation gap rescaled, and no
    labour-market observable enters except u itself.

    written in LEVELS, with `pi^e` formed by attention that switches on above
    6 per cent inflation until 1983, then the expectations model's own series.
    See `observations.salience_expectation`.

    That is what lets the equation say something about a re-anchoring. The
    accelerationist form cannot: when expectations are collapsing onto a new
    target, inflation stops changing for a nominal reason rather than a real
    one, and `Delta pi` near zero reads as equilibrium whatever unemployment is
    doing. `long_run_ustar/MODEL_NOTES.md` diagnoses exactly that for 1993, and
    it is why every method in the package returns about 10.7 there.

    Residual scale is separate either side of the handoff when
    `config.regime_sigma`, because inflation less an adaptive expectation and
    inflation less a measured one are different objects with different noise.
    """
    with pm.Model() as model:
        ustar = (
            _ustar_spline(frame, config)
            if config.state == "spline"
            else _ustar_state(frame, regimes, config)
        )

        # Positive by construction: the Phillips curve's content is that slack
        # slows inflation. Left free to change sign, a weakly identified
        # equation would report a sign rather than a level, and the level is
        # what is being asked for. The restriction is an assumption, not a
        # finding, and `beta` near zero is how the model says the data did not
        # support it.
        beta = pm.HalfNormal("beta", sigma=config.beta_prior_sd)
        measured = np.asarray(frame["measured"], dtype=int)
        if config.regime_sigma and 0 < measured.sum() < len(measured):
            sigma_both = pm.HalfNormal("sigma", sigma=config.sigma_prior_sd, shape=2)
            sigma = sigma_both[measured]
        else:
            sigma = pm.HalfNormal("sigma", sigma=config.sigma_prior_sd)

        # The PROPORTIONAL gap, (u - u*) / u, as `ustar/estimate.py:212` and
        # `ystar_ustar/estimate.py:411` both use. Convex in u, which matters
        # here more than it does there.
        #
        # Inverting gives u* = u x (1 + (pi - pi^e)/beta), so the factor that
        # turns an inflation surprise into an unemployment statement is u/beta
        # rather than 1/beta. It therefore SCALES WITH THE UNEMPLOYMENT RATE:
        # small when the labour market is tight, large when it is slack. Under
        # the level gap a +0.28 average surprise in the 1960s became +0.60 on
        # u* at an unemployment rate of 1.9, and the 2022-23 surprises of +3.4
        # to +5.2 became gaps of 7 to 11 points. Both are amplifications the
        # level form applies regardless of where unemployment actually was.
        #
        # There is evidence for convexity beyond consistency: `ystar_ustar`'s
        # notes split its fitted sample and find a slope of -1.82 on the tight
        # side against -0.45 on the slack side.
        u_obs = np.asarray(frame["u"], dtype=float)
        mu = -beta * (u_obs - ustar) / u_obs
        if config.supply_control:
            # Same form and prior scales as `ustar/estimate.py:206`. The GSCPI
            # term is squared and sign-preserving, so pressure matters more
            # than proportionally and relief is not treated as its mirror.
            rho = pm.Normal("rho_pi", mu=0.0, sigma=config.rho_prior_sd)
            xi = pm.Normal("xi_gscpi", mu=0.0, sigma=config.xi_prior_sd)
            gscpi = np.asarray(frame["gscpi"], dtype=float)
            mu = mu + rho * np.asarray(frame["d4pm"], dtype=float) + xi * gscpi**2 * np.sign(gscpi)

        if config.tot_control:
            gamma = pm.Normal("gamma_tot", mu=0.0, sigma=config.tot_prior_sd)
            mu = mu + gamma * np.asarray(frame["tot"], dtype=float)

        if config.wage_equation:
            _wage_equation(frame, ustar, u_obs, config)

        observed = np.asarray(frame["surprise"], dtype=float)
        if config.student_t:
            nu = pm.Exponential("nu_minus_two", lam=1.0 / config.nu_prior) + 2.0
            pm.Deterministic("nu", nu)
            pm.StudentT("dpi_obs", nu=nu, mu=mu, sigma=sigma, observed=observed)
        else:
            pm.Normal("dpi_obs", mu=mu, sigma=sigma, observed=observed)

    model._fixed_constants = dict(config.constants)  # noqa: SLF001 — our own metadata, as base.py does
    return model


def estimate(
    frame: pd.DataFrame,
    regimes: np.ndarray,
    labels: list[str],
    config: ModelConfig,
    sampler: SamplerConfig | None = None,
) -> az.InferenceData:
    """Sample the model and attach what is needed to read the trace back."""
    trace = sample_model(build_model(frame, regimes, config), sampler)
    posterior = getattr(trace, "posterior", None)
    if isinstance(posterior, xr.Dataset):
        posterior.attrs["regime_labels"] = labels
    return trace


def save_trace(trace: az.InferenceData, frame: pd.DataFrame, labels: list[str], config: ModelConfig) -> Path:
    """Write the trace and the frame it was fitted to, side by side."""
    config.output_dir.mkdir(parents=True, exist_ok=True)
    path = config.output_dir / f"{config.prefix}_trace.nc"
    trace.to_netcdf(str(path))
    with (config.output_dir / f"{config.prefix}_data.pkl").open("wb") as handle:
        pickle.dump({"frame": frame, "labels": labels, "constants": config.constants}, handle)
    return path


def load_trace(config: ModelConfig) -> tuple[az.InferenceData, pd.DataFrame, list[str]]:
    """Read back a saved run, for charting without re-sampling."""
    trace = az.from_netcdf(str(config.output_dir / f"{config.prefix}_trace.nc"))
    with (config.output_dir / f"{config.prefix}_data.pkl").open("rb") as handle:
        saved = pickle.load(handle)  # noqa: S301 — our own file
    return trace, saved["frame"], saved["labels"]
