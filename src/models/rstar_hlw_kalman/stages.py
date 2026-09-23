"""HLW's three-stage estimation, with their constraints.

Each stage is a constrained maximum likelihood fit of the state space in
`state_space.py`, with the states integrated out by the Kalman filter. Between
stages, a signal-to-noise ratio is read off a structural break test rather
than off the likelihood, because the likelihood cannot see it (see `mue.py`).

THEIR CONSTRAINTS, AS IMPOSED HERE

- `c = 1` on trend growth. Not a bound but an identity: the IS curve is built
  on `real - (g + z)`, so r* is g + z by construction and there is no
  coefficient to estimate. HLW impose unity because it "is not well
  identified in the data"; LW2001 estimated it and got 0.97.
- `a_r < 0` and `b_y > 0`. HLW call these "minimal priors" that "facilitate
  the convergence of the numerical optimization". They are bounds here, which
  means the sign of the IS slope is an assumption in this package exactly as
  it is in theirs.
- `a_y1 + a_y2 < 1`, so the output gap is stationary. A linear inequality, so
  the optimiser is SLSQP rather than L-BFGS-B.
- `lambda_g` and `lambda_z` imposed from the earlier stages.

WHAT IS NOT IMPLEMENTED, AND IT IS STAGE 2's HALF OF THE PROCEDURE. HLW read
`lambda_z` off an exponential Wald test for an intercept shift in the IS
equation of the stage 2 model. The machinery to do it is in `mue.py` and
works, but the exact regression HLW run at that step has not been verified
against their code, and guessing it would put an unchecked construction at
the centre of the result. `lambda_z` is therefore an INPUT here, defaulting to
LW2001's published 0.071, and stage 2 estimates the model without producing
it. That is a declared gap, not a silent one.

Buncic (arXiv 2002.11583) argues the stage 2 step is unsound anyway and
inflates `lambda_z`, so the gap is worth filling carefully rather than fast.
"""

from dataclasses import dataclass, replace

import numpy as np
from scipy import optimize

from src.models.dsge.kalman import kalman_smoother
from src.models.rstar_hlw_kalman import mue
from src.models.rstar_hlw_kalman.state_space import (
    YSTAR,
    Params,
    adjusted_observations,
    log_likelihood,
    observation,
    transition,
)

# A state "held constant" needs a zero innovation sd, but the likelihood
# guards against non-positive scales, so it gets the smallest value that is
# numerically indistinguishable from zero over 134 quarters.
HELD_CONSTANT = 1e-10

# g is annualised here and quarterly in the papers, so sigma_g = 4 x lambda_g
# x sigma_ystar. Matches the MCMC implementation's convention.
_QUARTERS_PER_YEAR = 4.0

# LW2001's published lambda_z, used until stage 2's own extraction is built.
# UNVERIFIED against the paper: carried from this repo's MODEL_NOTES.
DEFAULT_LAMBDA_Z = 0.071

# What sigma_z is imposed at when the lambda_z route is bypassed. Small, so z
# moves slowly, which is both the better fit and the only reading under which
# r* stays inside a plausible range. It is an ASSUMPTION: the likelihood is
# almost flat in this number, so nothing here measures it.
DEFAULT_SIGMA_Z = 0.10

# Stage numbers, so the dispatch reads as stages rather than as integers.
_STAGE_CUT_DOWN = 1
_STAGE_RATE_GAP = 2

# Below this the IS slope is treated as zero and z is held constant rather
# than divided by a number indistinguishable from it.
_A_R_FLOOR = 1e-8

# Starting values for every stage. Near the MCMC posterior, so the optimiser
# begins somewhere the model can represent.
START = Params(
    a_y1=0.90, a_y2=-0.05, a_r=-0.05, b_y=0.25,
    sigma_is=0.55, sigma_pi=0.55, sigma_ystar=0.30,
    sigma_g=0.10, sigma_z=0.06,
)


@dataclass(frozen=True)
class StageResult:
    """What one stage produced."""

    name: str
    params: Params
    log_likelihood: float
    converged: bool
    message: str


def _free_fields(stage: int) -> tuple[str, ...]:
    """Which scalars a stage estimates. The rest are imposed or held constant."""
    if stage == _STAGE_CUT_DOWN:
        # No rate gap, so no a_r; g and z do not move, so neither sd is free.
        return ("a_y1", "a_y2", "b_y", "sigma_is", "sigma_pi", "sigma_ystar")
    if stage == _STAGE_RATE_GAP:
        # The rate gap returns. sigma_g follows from lambda_g, z is still still.
        return ("a_y1", "a_y2", "a_r", "b_y", "sigma_is", "sigma_pi", "sigma_ystar")
    return ("a_y1", "a_y2", "a_r", "b_y", "sigma_is", "sigma_pi", "sigma_ystar")


def _build(
    x: np.ndarray,
    fields: tuple[str, ...],
    stage: int,
    *,
    lambda_g: float | None,
    lambda_z: float | None,
    sigma_z_imposed: float | None = None,
) -> Params:
    """Turn a free-parameter vector into a full `Params` for a given stage."""
    p = replace(START, **dict(zip(fields, (float(v) for v in x), strict=True)))

    if stage == _STAGE_CUT_DOWN:
        return replace(p, a_r=0.0, sigma_g=HELD_CONSTANT, sigma_z=HELD_CONSTANT)

    sigma_g = (
        HELD_CONSTANT if lambda_g is None or lambda_g <= 0
        else _QUARTERS_PER_YEAR * lambda_g * p.sigma_ystar
    )
    if stage == _STAGE_RATE_GAP:
        return replace(p, sigma_g=max(sigma_g, HELD_CONSTANT), sigma_z=HELD_CONSTANT)

    # Stage 3. HLW set sigma_z = lambda_z * sigma_IS / |a_r|, which presumes an
    # identified IS slope. On this data a_r converges to its own lower bound,
    # so that expression divides by 0.0025 and returns sigma_z near 4, an r*
    # spanning thirty percentage points, and a WORSE fit than any small value.
    # `sigma_z_imposed` takes the ratio's place and skips the division.
    if sigma_z_imposed is not None:
        sigma_z = sigma_z_imposed
    else:
        sigma_z = (
            HELD_CONSTANT if lambda_z is None or lambda_z <= 0 or abs(p.a_r) < _A_R_FLOOR
            else lambda_z * p.sigma_is / abs(p.a_r)
        )
    return replace(
        p, sigma_g=max(sigma_g, HELD_CONSTANT), sigma_z=max(sigma_z, HELD_CONSTANT),
    )


def fit(
    obs: dict[str, np.ndarray],
    stage: int,
    *,
    lambda_g: float | None = None,
    lambda_z: float | None = None,
    sigma_z_imposed: float | None = None,
    start: Params = START,
) -> StageResult:
    """Constrained maximum likelihood for one stage.

    `sigma_z_imposed` replaces HLW's lambda_z route to sigma_z. Use it: the
    ratio route divides by an IS slope this data cannot identify. Note that
    NEITHER route measures z's amplitude, because a_r is small enough that
    the likelihood is nearly flat in sigma_z. Over 0.05 to 1.00 the fit moves
    0.06 log points while z's range moves three hundredfold, and the path
    keeps a correlation above 0.996 throughout. The SHAPE of z is identified;
    its scale is a declared assumption whichever way it is set.
    """
    fields = _free_fields(stage)
    x0 = np.array([getattr(start, f) for f in fields], dtype=float)

    def negative_ll(x: np.ndarray) -> float:
        ll = log_likelihood(
            obs,
            _build(
                x, fields, stage, lambda_g=lambda_g, lambda_z=lambda_z,
                sigma_z_imposed=sigma_z_imposed,
            ),
        )
        return -ll if np.isfinite(ll) else 1e12

    # HLW's sign constraints, as bounds. a_r is bounded strictly below zero
    # rather than at zero, as they do, so the slope cannot sit exactly on the
    # boundary and report a rate channel that is not there.
    bounds = []
    for f in fields:
        if f == "a_r":
            bounds.append((None, -0.0025))
        elif f == "b_y":
            bounds.append((0.025, None))
        elif f.startswith("sigma_"):
            bounds.append((1e-4, None))
        else:
            bounds.append((None, None))

    # Output-gap stationarity, a_y1 + a_y2 < 1, as a linear inequality.
    i1, i2 = fields.index("a_y1"), fields.index("a_y2")

    def stationarity(x: np.ndarray) -> float:
        return 1.0 - 1e-4 - (x[i1] + x[i2])

    result = optimize.minimize(
        negative_ll, x0, method="SLSQP", bounds=bounds,
        constraints=[{"type": "ineq", "fun": stationarity}],
        options={"maxiter": 500, "ftol": 1e-9},
    )
    p = _build(
        result.x, fields, stage, lambda_g=lambda_g, lambda_z=lambda_z,
        sigma_z_imposed=sigma_z_imposed,
    )
    return StageResult(
        name=f"stage{stage}",
        params=p,
        log_likelihood=log_likelihood(obs, p),
        converged=bool(result.success),
        message=str(result.message),
    )


def smoothed_states(obs: dict[str, np.ndarray], p: Params) -> np.ndarray:
    """Smoothed state paths, shape (T, 9), for a fitted parameter vector."""
    t_mat, r_mat = transition()
    z_mat, q_mat, h_mat = observation(p)
    y = adjusted_observations(obs, p)

    s0 = np.zeros(t_mat.shape[0])
    s0[:3] = float(np.asarray(obs["log_gdp"], dtype=float)[0])
    p0 = np.eye(t_mat.shape[0]) * 1e4

    out = kalman_smoother(y, t_mat, r_mat, z_mat, q_mat, H=h_mat, s0=s0, P0=p0)
    if out.smoothed_states is None:
        raise RuntimeError("kalman_smoother returned no smoothed states")
    return np.asarray(out.smoothed_states)


def lambda_g_from_stage1(
    obs: dict[str, np.ndarray], result: StageResult,
) -> tuple[float, dict[str, float]]:
    """HLW's stage 1 extraction: break in the mean of the preliminary `Dy*`.

    Trend growth was held constant, so any real movement in it had to go into
    the drift of potential output. The exponential Wald statistic for a break
    at an unknown date in `Dy*` measures how much, and Stock-Watson's table
    turns that into `lambda_g`.
    """
    states = smoothed_states(obs, result.params)
    growth = np.diff(states[:, YSTAR]) * _QUARTERS_PER_YEAR

    stats = mue.break_statistics(growth)
    ew = mue.exponential_wald(stats)
    lam = mue.lambda_from(ew, len(growth), denominator_offset=1)

    return lam, {
        "exp_wald": ew,
        "mean_wald": mue.mean_wald(stats),
        "qlr": mue.quandt_likelihood_ratio(stats),
        "growth_first": float(growth[0]),
        "growth_last": float(growth[-1]),
        "table_exhausted": float(ew > mue.EW_CRITICAL_VALUES[-1]),
    }
