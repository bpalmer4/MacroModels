"""The rstar_qpm state space, written once and assembled for two backends.

The formulas below take a parameter mapping whose values are either floats or
pytensor scalars, and data arrays already converted to the matching backend.
`pytensor_loglik` assembles them into the likelihood NUTS samples;
`numpy_matrices` assembles the same formulas for the filter, smoother and
simulation smoother that run after sampling. One set of formulas, so the two
cannot drift apart, and `check_backends` confirms they agree.

STATE VECTOR

    s_t = [ y*_t, g_t, ygap_t, ygap_{t-1}, w_t, q*_t, qgap_t ]

TRANSITION, s_t = T s_{t-1} + c_t + R eta_t

    y*_t   = y*_{t-1} + g_{t-1}/4 + e1
    g_t    = g_{t-1} + e2
    ygap_t = b1 ygap_{t-1} + b2 w_{t-1} - b3 qgap_{t-1}
             - b2 (r_{t-1} - rw_{t-1}) - b4 dDSR_{t-1} + e3
    w_t    = w_{t-1} + e4
    q*_t   = q*_{t-1} + e5
    qgap_t = rho_q qgap_{t-1} - kappa w_{t-1}  + kappa (r_t - rw_t)  - kappa e4 + e6

The IS rate gap is r - r* = (r - rw) - w: the data part goes to the input
`c_t`, the state part to `T`. The exchange rate reads the SAME quarter's rate
gap, so w_t appears through w_{t-1} + e4, hence the -kappa loading of e4 in R.

OBSERVATIONS, known terms moved to the left so Z is constant

    y_t                                                        = y*_t + ygap_t
    q_t                                                        = q*_t + qgap_t
    pi_t - a1 pi_{t-1} - (1-a1) pie_t - a3 (m4_{t-1} - pie_{t-1}) = a2 ygap_{t-1} + e
    i_t - rho_i i_{t-1} - (1-rho_i)(rw_t + pie_t + phi_pi (pi4_t - target))
                                                               = (1-rho_i)(w_t + phi_y ygap_t) + e
    f_t - rw_t - bias                                          = w_t + e
    dDSR_t - delta E_t (i_t - i_{t-1})                         = e

The cash-flow equation loads on no state: it is a regression of debt
servicing on the cash rate, estimated jointly so that `delta` is known when
short-run neutral and the transmission charts need debt servicing to respond
to a held rate. In the IS curve the change in debt servicing is data.

WHY THE CASH RATE CAN BE BOTH DATA AND OBSERVED. `r` enters the transition as
an input and `i` is scored by the rule. The map from the rule and UIP shocks
to (i_t, q_t) is triangular with a unit diagonal, so the Jacobian is one and
the product the filter computes is the joint density. The IS input is lagged
and so predetermined.

MISSING OBSERVATIONS are handled by a 0/1 mask rather than NaN, because in a
compiled graph NaN x 0 is NaN. A masked entry gets a zero innovation and unit
variance, which adds only a parameter-free constant to the log-likelihood.
"""

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import pytensor
import pytensor.tensor as pt
from pytensor.tensor.slinalg import cholesky, solve

from src.models.rstar_qpm.config import ModelConfig

YSTAR, G, YGAP, YGAP_1, W, QSTAR, QGAP = range(7)
N_STATES = 7
STATE_NAMES = ("ystar", "g", "ygap", "ygap_1", "wedge", "qstar", "qgap")

# Shock order: one per stochastic state, in the order below.
SHOCKS = ("sigma_ystar", "sigma_g", "sigma_ygap", "sigma_w", "sigma_qstar", "sigma_qgap")
N_SHOCKS = len(SHOCKS)

OBS_Y, OBS_Q, OBS_PI, OBS_RULE, OBS_F, OBS_DSR = range(6)
N_OBS = 6
OBS_NAMES = ("gdp", "twi", "phillips", "rule", "forward", "cash_flow")

# g is annualised; a quarter's drift on potential is g/4.
QUARTERS_PER_YEAR = 4.0

LOG_2PI = float(np.log(2.0 * np.pi))

# Initial-state spreads: wide against the data's own scale, so the first
# quarters carry no real information about the levels, but finite so the
# filter's first updates stay well conditioned.
_INIT_SD = {"ystar": 5.0, "g": 2.0, "ygap": 3.0, "ygap_1": 3.0, "wedge": 2.0, "qstar": 20.0, "qgap": 10.0}
# Trend growth at the start of the sample, annualised, before any data.
_INIT_G = 3.0
# Quarters of the forward averaged for the wedge's starting guess.
_INIT_WEDGE_QUARTERS = 8

Params = Mapping[str, Any]
Data = Mapping[str, Any]


# ---------------------------------------------------------------------------
# The formulas: backend-agnostic
# ---------------------------------------------------------------------------

def transition_entries(p: Params) -> dict[tuple[int, int], Any]:
    """Non-zero entries of T."""
    return {
        (YSTAR, YSTAR): 1.0,
        (YSTAR, G): 1.0 / QUARTERS_PER_YEAR,
        (G, G): 1.0,
        (YGAP, YGAP): p["b1"],
        (YGAP, W): p["b2"],
        (YGAP, QGAP): -p["b3"],
        (YGAP_1, YGAP): 1.0,
        (W, W): 1.0,
        (QSTAR, QSTAR): 1.0,
        (QGAP, QGAP): p["rho_q"],
        (QGAP, W): -p["kappa"],
    }


def shock_entries(p: Params) -> dict[tuple[int, int], Any]:
    """Non-zero entries of R, states x shocks."""
    return {
        (YSTAR, 0): 1.0,
        (G, 1): 1.0,
        (YGAP, 2): 1.0,
        (W, 3): 1.0,
        (QGAP, 3): -p["kappa"],
        (QSTAR, 4): 1.0,
        (QGAP, 5): 1.0,
    }


def observation_entries(p: Params) -> dict[tuple[int, int], Any]:
    """Non-zero entries of Z, observations x states."""
    one_less = 1.0 - p["rho_i"]
    return {
        (OBS_Y, YSTAR): 1.0,
        (OBS_Y, YGAP): 1.0,
        (OBS_Q, QSTAR): 1.0,
        (OBS_Q, QGAP): 1.0,
        (OBS_PI, YGAP_1): p["a2"],
        (OBS_RULE, W): one_less,
        (OBS_RULE, YGAP): one_less * p["phi_y"],
        (OBS_F, W): 1.0,
    }


def shock_sds(p: Params) -> list[Any]:
    """Innovation sds, in shock order."""
    return [p[name] for name in SHOCKS]


def obs_sds(p: Params, identity_sd: float) -> list[Any]:
    """Observation residual sds, in observation order."""
    return [identity_sd, identity_sd, p["sigma_pi"], p["sigma_i"], p["sigma_f"], p["sigma_d"]]


def adjusted_observations(d: Data, p: Params, target: float) -> list[Any]:
    """Left-hand sides, one array per observation, known terms moved over."""
    one_less = 1.0 - p["rho_i"]
    return [
        d["y"],
        d["q"],
        d["pi"] - p["a1"] * d["pi_lag"] - (1.0 - p["a1"]) * d["pie"]
        - p["a3"] * (d["m4_lag"] - d["pie_lag"]),
        d["i"] - p["rho_i"] * d["i_lag"]
        - one_less * (d["rw"] + d["pie"] + p["phi_pi"] * (d["pi4"] - target)),
        d["f"] - d["rw"] - p["forward_bias"],
        d["d_dsr"] - p["delta"] * d["exposure"] * d["d_i"],
    ]


def inputs(d: Data, p: Params) -> dict[int, Any]:
    """Data-driven intercepts in the transition, by state."""
    return {
        YGAP: -p["b2"] * (d["r_lag"] - d["rw_lag"]) - p["b4"] * d["d_dsr_lag"],
        QGAP: p["kappa"] * (d["r"] - d["rw"]),
    }


# ---------------------------------------------------------------------------
# Data that does not depend on the parameters
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Prepared:
    """The data arrays, NaNs zero-filled, with the mask that says which count."""

    data: dict[str, np.ndarray]
    mask: np.ndarray            # (n, N_OBS), 1.0 where observed
    s0: np.ndarray              # mean of the state BEFORE the first quarter
    p0: np.ndarray              # its covariance
    index: pd.PeriodIndex
    target: float
    identity_sd: float


def prepare(frame: pd.DataFrame, config: ModelConfig) -> Prepared:
    """Zero-fill, build the mask, and set the initial state."""
    index = frame.index
    if not isinstance(index, pd.PeriodIndex):
        index = pd.PeriodIndex(index, freq="Q")
    n = len(frame)

    mask = np.ones((n, N_OBS))
    if config.exclude_start and config.exclude_end:
        lo, hi = pd.Period(config.exclude_start, freq="Q"), pd.Period(config.exclude_end, freq="Q")
        mask[(index >= lo) & (index <= hi), OBS_Y] = 0.0
    mask[frame["i"].to_numpy(dtype=float) < config.rule_floor, OBS_RULE] = 0.0
    mask[frame["f"].isna().to_numpy(), OBS_F] = 0.0
    if not config.use_forward:
        mask[:, OBS_F] = 0.0

    data = {col: np.nan_to_num(frame[col].to_numpy(dtype=float), nan=0.0) for col in frame.columns}

    gap = (frame["f"] - frame["rw"]).dropna()
    s0 = np.zeros(N_STATES)
    s0[YSTAR] = float(frame["y"].iloc[0])
    s0[G] = _INIT_G
    # The forward seeds the wedge only when the forward is in the model.
    s0[W] = float(gap.iloc[:_INIT_WEDGE_QUARTERS].mean()) if config.use_forward and len(gap) else 0.0
    s0[QSTAR] = float(frame["q"].iloc[0])
    p0 = np.diag([_INIT_SD[name] ** 2 for name in STATE_NAMES])

    return Prepared(data=data, mask=mask, s0=s0, p0=p0, index=index,
                    target=config.target, identity_sd=config.identity_sd)


# ---------------------------------------------------------------------------
# pytensor: the likelihood NUTS samples
# ---------------------------------------------------------------------------

def _pt_matrix(entries: dict[tuple[int, int], Any], shape: tuple[int, int]) -> pt.TensorVariable:
    out = pt.zeros(shape)
    for (i, j), value in entries.items():
        out = pt.set_subtensor(out[i, j], value)
    return out


def pytensor_loglik(prep: Prepared, p: Params) -> pt.TensorVariable:
    """Marginal log-likelihood, states integrated out by the Kalman filter."""
    n = len(prep.index)
    d = {k: pt.as_tensor_variable(v) for k, v in prep.data.items()}

    t_mat = _pt_matrix(transition_entries(p), (N_STATES, N_STATES))
    r_mat = _pt_matrix(shock_entries(p), (N_STATES, N_SHOCKS))
    z_mat = _pt_matrix(observation_entries(p), (N_OBS, N_STATES))
    q_diag = pt.stack(shock_sds(p)) ** 2
    rqr = r_mat @ pt.diag(q_diag) @ r_mat.T
    h_diag = pt.stack([pt.as_tensor_variable(s) for s in obs_sds(p, prep.identity_sd)]) ** 2

    ys = pt.stack(adjusted_observations(d, p, prep.target), axis=1)
    feeds = inputs(d, p)
    cs = pt.stack([feeds.get(k, pt.zeros(n)) for k in range(N_STATES)], axis=1)
    mask = pt.as_tensor_variable(prep.mask)

    # Scan hands the step its sequences, recurrent states and non-sequences
    # positionally, nine in all. The matrices must be explicit non-sequences:
    # captured by closure instead, scan's gradient fails ("NominalVariable has
    # no attribute 'shape'"), so NUTS cannot run.
    def step(*scan_args: pt.TensorVariable) -> tuple[pt.TensorVariable, pt.TensorVariable, pt.TensorVariable]:
        y_t, m_t, c_t, s, cov, t_mat, rqr, z_mat, h_diag = scan_args
        s_p = t_mat @ s + c_t
        p_p = t_mat @ cov @ t_mat.T + rqr
        zm = z_mat * m_t[:, None]
        v = m_t * (y_t - z_mat @ s_p)
        f_mat = zm @ p_p @ zm.T + pt.diag(h_diag * m_t + (1.0 - m_t))
        chol = cholesky(f_mat)
        f_inv_v = solve(f_mat, v, assume_a="pos")
        gain = p_p @ zm.T @ solve(f_mat, pt.eye(N_OBS), assume_a="pos")
        s_new = s_p + gain @ v
        cov_new = p_p - gain @ zm @ p_p
        cov_new = 0.5 * (cov_new + cov_new.T)
        ll_t = -0.5 * (2.0 * pt.sum(pt.log(pt.diag(chol))) + v @ f_inv_v + pt.sum(m_t) * LOG_2PI)
        return s_new, cov_new, ll_t

    _, _, ll = pytensor.scan(
        step,
        sequences=[ys, mask, cs],
        outputs_info=[pt.as_tensor_variable(prep.s0), pt.as_tensor_variable(prep.p0), None],
        non_sequences=[t_mat, rqr, z_mat, h_diag],
        return_updates=False,
    )
    return pt.sum(ll)


# ---------------------------------------------------------------------------
# numpy: filter, smoother, simulation smoother
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Matrices:
    """The state space at one parameter draw."""

    t: np.ndarray
    r: np.ndarray
    z: np.ndarray
    q: np.ndarray       # shock covariance, diagonal
    h: np.ndarray       # observation covariance, diagonal
    y: np.ndarray       # (n, N_OBS) adjusted observations, zero where masked
    c: np.ndarray       # (n, N_STATES) inputs


def _np_matrix(entries: dict[tuple[int, int], Any], shape: tuple[int, int]) -> np.ndarray:
    out = np.zeros(shape)
    for (i, j), value in entries.items():
        out[i, j] = float(value)
    return out


def numpy_matrices(prep: Prepared, p: Params) -> Matrices:
    """Assemble the same formulas with floats."""
    n = len(prep.index)
    feeds = inputs(prep.data, p)
    return Matrices(
        t=_np_matrix(transition_entries(p), (N_STATES, N_STATES)),
        r=_np_matrix(shock_entries(p), (N_STATES, N_SHOCKS)),
        z=_np_matrix(observation_entries(p), (N_OBS, N_STATES)),
        q=np.diag(np.asarray(shock_sds(p), dtype=float) ** 2),
        h=np.diag(np.asarray(obs_sds(p, prep.identity_sd), dtype=float) ** 2),
        y=np.column_stack(adjusted_observations(prep.data, p, prep.target)) * prep.mask,
        c=np.column_stack([np.asarray(feeds.get(k, np.zeros(n)), dtype=float) for k in range(N_STATES)]),
    )


@dataclass(frozen=True)
class Filtered:
    """Filter output, kept for the smoother."""

    s_pred: np.ndarray
    p_pred: np.ndarray
    s_filt: np.ndarray
    p_filt: np.ndarray
    loglik: float


def kalman_filter(m: Matrices, mask: np.ndarray, s0: np.ndarray, p0: np.ndarray) -> Filtered:
    """Masked Kalman filter with a transition intercept."""
    n = m.y.shape[0]
    rqr = m.r @ m.q @ m.r.T
    s_pred = np.zeros((n, N_STATES))
    p_pred = np.zeros((n, N_STATES, N_STATES))
    s_filt = np.zeros((n, N_STATES))
    p_filt = np.zeros((n, N_STATES, N_STATES))
    s, cov, ll = s0, p0, 0.0
    h_diag = np.diag(m.h)
    for t in range(n):
        m_t = mask[t]
        s_p = m.t @ s + m.c[t]
        p_p = m.t @ cov @ m.t.T + rqr
        zm = m.z * m_t[:, None]
        v = m_t * (m.y[t] - m.z @ s_p)
        f_mat = zm @ p_p @ zm.T + np.diag(h_diag * m_t + (1.0 - m_t))
        f_inv = np.linalg.inv(f_mat)
        gain = p_p @ zm.T @ f_inv
        s = s_p + gain @ v
        cov = p_p - gain @ zm @ p_p
        cov = 0.5 * (cov + cov.T)
        _, logdet = np.linalg.slogdet(f_mat)
        ll += -0.5 * (logdet + v @ f_inv @ v + m_t.sum() * LOG_2PI)
        s_pred[t], p_pred[t], s_filt[t], p_filt[t] = s_p, p_p, s, cov
    return Filtered(s_pred, p_pred, s_filt, p_filt, float(ll))


def rts_smoother(m: Matrices, f: Filtered) -> np.ndarray:
    """Rauch-Tung-Striebel smoothed state means."""
    n = f.s_filt.shape[0]
    smoothed = f.s_filt.copy()
    for t in range(n - 2, -1, -1):
        gain = f.p_filt[t] @ m.t.T @ np.linalg.pinv(f.p_pred[t + 1])
        smoothed[t] = f.s_filt[t] + gain @ (smoothed[t + 1] - f.s_pred[t + 1])
    return smoothed


def simulate(m: Matrices, mask: np.ndarray, s0: np.ndarray, p0: np.ndarray,
             rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Draw states and observations from the model, holding the inputs as data."""
    n = m.y.shape[0]
    states = np.zeros((n, N_STATES))
    obs = np.zeros((n, N_OBS))
    s = rng.multivariate_normal(s0, p0)
    shock_sd = np.sqrt(np.diag(m.q))
    obs_sd = np.sqrt(np.diag(m.h))
    for t in range(n):
        s = m.t @ s + m.c[t] + m.r @ (shock_sd * rng.standard_normal(N_SHOCKS))
        states[t] = s
        obs[t] = (m.z @ s + obs_sd * rng.standard_normal(N_OBS)) * mask[t]
    return states, obs


def simulation_smoother(prep: Prepared, p: Params, rng: np.random.Generator) -> np.ndarray:
    """One draw of the states given the data, Durbin-Koopman (2002).

    Smooth the data, simulate a fresh data set from the model and smooth that
    too; the simulated states less their own smoothed mean are a draw of the
    smoothing error, which added to the data's smoothed mean is a draw from
    the conditional distribution.
    """
    m = numpy_matrices(prep, p)
    smoothed = rts_smoother(m, kalman_filter(m, prep.mask, prep.s0, prep.p0))
    sim_states, sim_obs = simulate(m, prep.mask, prep.s0, prep.p0, rng)
    m_sim = Matrices(t=m.t, r=m.r, z=m.z, q=m.q, h=m.h, y=sim_obs, c=m.c)
    sim_smoothed = rts_smoother(m_sim, kalman_filter(m_sim, prep.mask, prep.s0, prep.p0))
    return smoothed + (sim_states - sim_smoothed)


def check_backends(prep: Prepared, p: Mapping[str, float]) -> tuple[float, float]:
    """Return the pytensor and numpy log-likelihoods at one parameter point."""
    symbols = {name: pt.dscalar(name) for name in p}
    ll = pytensor_loglik(prep, symbols)
    fn: Callable[..., Any] = pytensor.function(list(symbols.values()), ll, mode="JAX")
    pt_value = float(fn(*[p[name] for name in symbols]))
    np_value = numpy_matrices_loglik(prep, p)
    return pt_value, np_value


def numpy_matrices_loglik(prep: Prepared, p: Params) -> float:
    """Return the numpy filter's log-likelihood at `p`."""
    m = numpy_matrices(prep, p)
    return kalman_filter(m, prep.mask, prep.s0, prep.p0).loglik
