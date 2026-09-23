"""Canonical HLW as a linear-Gaussian state-space model.

WHY THIS EXISTS. Sampling this model's states is what creates its funnels:
`sigma_ystar` against 133 potential innovations, `sigma_z` against 134 z
states, `sigma_g` against 134 trend-growth states. Every divergence fought in
the MCMC implementation comes from that pairing. Conditional on the
parameters the model is linear and Gaussian, so the states can be integrated
out exactly instead, leaving a posterior over NINE scalars that has no funnel
available to it.

THE SHAPE IS HLW'S OWN, AND THAT IS WHAT MAKES IT CHEAP. The IS curve averages
the real rate gap over t-1 and t-2, so the companion form needs two lags of
each state and the vector is nine long. A single t-6 lag would need seven lags
of g and z and run past twenty states, which is why the papers wrote it the
way they did.

STATE VECTOR

    s_t = [ y*_t, y*_{t-1}, y*_{t-2}, g_t, g_{t-1}, g_{t-2}, z_t, z_{t-1}, z_{t-2} ]

TRANSITION

    y*_t = y*_{t-1} + g_{t-1}/4 + e_ystar
    g_t  = g_{t-1} + e_g
    z_t  = z_{t-1} + e_z

with the remaining rows shifting lags along. g is annualised, hence g/4 as the
quarterly drift, matching the MCMC implementation.

OBSERVATIONS. Both equations are rearranged so everything known lands on the
left, because `kalman_filter` has no observation intercept. The left side
depends on the parameters as well as the data, so it is rebuilt on every
likelihood evaluation, which costs one vector subtraction.

    IS:   Y_t - a_y1*Y_{t-1} - a_y2*Y_{t-2} - (a_r/2)(real_{t-1} + real_{t-2})
            = y*_t - a_y1*y*_{t-1} - a_y2*y*_{t-2}
              - (a_r/2)(g_{t-1} + z_{t-1} + g_{t-2} + z_{t-2}) + e_IS

    PC:   pi4_t - pi_exp_t - b_y*Y_{t-1} = -b_y*y*_{t-1} + e_pi

So `Z` is constant given the parameters: no time-varying observation matrix is
needed, only the adjustment above.

VARIANCES. The two observation residuals are measurement error, `H`, and the
three state innovations are `Q`. That mapping is exact rather than a
convention: `sigma_IS` and `sigma_pi` enter the MCMC model as the sd of a
Normal on an observed quantity, which is what `H` is.

INITIALISATION IS DIFFUSE AND EXPLICIT. All three states are random walks, so
`T` has three unit eigenvalues and the unconditional covariance does not
exist. `kalman_filter` would try `solve_discrete_lyapunov` and may return
nonsense rather than raise, so `P0` is always passed in. HLW initialise from a
pre-sample regression instead; this is a departure, and a deliberate one.
"""

from dataclasses import dataclass

import numpy as np

from src.models.dsge.kalman import kalman_filter

# Positions in the state vector, named so the matrix builders read as algebra
# rather than as index arithmetic.
YSTAR, YSTAR_1, YSTAR_2 = 0, 1, 2
G, G_1, G_2 = 3, 4, 5
Z, Z_1, Z_2 = 6, 7, 8
N_STATES = 9
N_SHOCKS = 3
N_OBS = 2

# g is annualised, so a quarter's drift on potential is g/4.
_QUARTERS_PER_YEAR = 4.0

# Diffuse prior on the initial state. Large enough that the first quarters
# carry no information about the level, small enough not to wreck the
# conditioning of the first update.
_DIFFUSE_VARIANCE = 1e4


@dataclass(frozen=True)
class Params:
    """The nine scalars. Everything else in the model is data or algebra."""

    a_y1: float
    a_y2: float
    a_r: float
    b_y: float
    # sigma_is and sigma_pi are the IS and Phillips residual sds, named
    # `sigma_IS` and `sigma_pi` in the MCMC implementation. Lower case here
    # because a dataclass field is class scope, where mixed case is a lint
    # error the economics-notation exemption does not cover.
    sigma_is: float
    sigma_pi: float
    sigma_ystar: float
    sigma_g: float
    sigma_z: float


def transition() -> tuple[np.ndarray, np.ndarray]:
    """Return `T` and `R`. Neither depends on the parameters."""
    # Row i says how s_t[i] is built from s_{t-1}. Position 0 of s_{t-1} holds
    # y*_{t-1} and position G holds g_{t-1}, so the drift reads G, not G_1:
    # G_1 in the PREVIOUS state vector is g_{t-2}.
    t_mat = np.zeros((N_STATES, N_STATES))

    # y*_t = y*_{t-1} + g_{t-1}/4 + e
    t_mat[YSTAR, YSTAR] = 1.0
    t_mat[YSTAR, G] = 1.0 / _QUARTERS_PER_YEAR

    # g_t = g_{t-1} + e,  z_t = z_{t-1} + e
    t_mat[G, G] = 1.0
    t_mat[Z, Z] = 1.0

    # The lag rows shift: each reads the previous period's contemporaneous
    # value, or the previous period's first lag.
    t_mat[YSTAR_1, YSTAR] = 1.0
    t_mat[YSTAR_2, YSTAR_1] = 1.0
    t_mat[G_1, G] = 1.0
    t_mat[G_2, G_1] = 1.0
    t_mat[Z_1, Z] = 1.0
    t_mat[Z_2, Z_1] = 1.0

    r_mat = np.zeros((N_STATES, N_SHOCKS))
    r_mat[YSTAR, 0] = 1.0
    r_mat[G, 1] = 1.0
    r_mat[Z, 2] = 1.0
    return t_mat, r_mat


def observation(p: Params) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return `Z`, `Q` and `H` for one parameter draw."""
    z_mat = np.zeros((N_OBS, N_STATES))
    z_mat[0, YSTAR] = 1.0
    z_mat[0, YSTAR_1] = -p.a_y1
    z_mat[0, YSTAR_2] = -p.a_y2
    half_a_r = p.a_r / 2.0
    for idx in (G_1, G_2, Z_1, Z_2):
        z_mat[0, idx] = -half_a_r
    z_mat[1, YSTAR_1] = -p.b_y

    q_mat = np.diag([p.sigma_ystar**2, p.sigma_g**2, p.sigma_z**2])
    h_mat = np.diag([p.sigma_is**2, p.sigma_pi**2])
    return z_mat, q_mat, h_mat


def adjusted_observations(obs: dict[str, np.ndarray], p: Params) -> np.ndarray:
    """Move every known term to the left-hand side.

    The first two quarters lack the lags both equations need and are returned
    as NaN, which `kalman_filter` treats as missing rather than as zero.
    """
    log_gdp = np.asarray(obs["log_gdp"], dtype=float)
    real = np.asarray(obs["cash_rate"], dtype=float) - np.asarray(obs["pi_exp"], dtype=float)
    pi_4 = np.asarray(obs["pi_4"], dtype=float)
    pi_exp = np.asarray(obs["pi_exp"], dtype=float)

    n = len(log_gdp)
    y = np.full((n, N_OBS), np.nan)

    half_a_r = p.a_r / 2.0
    for t in range(2, n):
        y[t, 0] = (
            log_gdp[t]
            - p.a_y1 * log_gdp[t - 1]
            - p.a_y2 * log_gdp[t - 2]
            - half_a_r * (real[t - 1] + real[t - 2])
        )
    for t in range(1, n):
        y[t, 1] = pi_4[t] - pi_exp[t] - p.b_y * log_gdp[t - 1]
    return y


def log_likelihood(obs: dict[str, np.ndarray], p: Params) -> float:
    """Marginal log-likelihood of the data, states integrated out.

    Returns -inf for a parameter vector the model cannot represent, so a
    sampler or optimiser sees a wall rather than an exception.
    """
    if min(p.sigma_is, p.sigma_pi, p.sigma_ystar, p.sigma_g, p.sigma_z) <= 0:
        return -np.inf

    t_mat, r_mat = transition()
    z_mat, q_mat, h_mat = observation(p)
    y = adjusted_observations(obs, p)

    s0 = np.zeros(N_STATES)
    log_gdp0 = float(np.asarray(obs["log_gdp"], dtype=float)[0])
    s0[[YSTAR, YSTAR_1, YSTAR_2]] = log_gdp0

    p0 = np.eye(N_STATES) * _DIFFUSE_VARIANCE

    out = kalman_filter(y, t_mat, r_mat, z_mat, q_mat, H=h_mat, s0=s0, P0=p0)
    ll = float(out.log_likelihood)
    return ll if np.isfinite(ll) else -np.inf
