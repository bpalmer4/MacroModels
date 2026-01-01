"""Kalman filter and smoother for state-space models.

For a DSGE model solved via Blanchard-Kahn, the state-space form is:

    State equation:     s_{t+1} = T @ s_t + R @ η_{t+1}
    Observation equation:   y_t = Z @ s_t + measurement_error

Where:
    - s_t: State vector (predetermined + forward-looking variables)
    - y_t: Observable variables
    - T: Transition matrix (from BK solution)
    - R: Shock impact matrix (from BK solution)
    - Z: Selection matrix mapping states to observables
    - η_t ~ N(0, Q): Structural shocks
    - measurement_error ~ N(0, H): Measurement error (often zero)

The filter computes:
    1. Log-likelihood of the data given parameters
    2. Filtered states (E[s_t | y_1:t])
    3. Smoothed states (E[s_t | y_1:T])
"""

from dataclasses import dataclass

import numpy as np
from scipy import linalg


@dataclass
class KalmanOutput:
    """Output from Kalman filter/smoother.

    Attributes:
        log_likelihood: Log-likelihood of the observations
        filtered_states: E[s_t | y_1:t], shape (T, n_states)
        filtered_covs: Var[s_t | y_1:t], shape (T, n_states, n_states)
        smoothed_states: E[s_t | y_1:T], shape (T, n_states) (if smoothed)
        smoothed_covs: Var[s_t | y_1:T], shape (T, n_states, n_states) (if smoothed)
        innovations: y_t - E[y_t | y_1:t-1], shape (T, n_obs)
        innovation_covs: Var[y_t | y_1:t-1], shape (T, n_obs, n_obs)

    """

    log_likelihood: float
    filtered_states: np.ndarray
    filtered_covs: np.ndarray
    smoothed_states: np.ndarray | None = None
    smoothed_covs: np.ndarray | None = None
    innovations: np.ndarray | None = None
    innovation_covs: np.ndarray | None = None


def kalman_filter(
    y: np.ndarray,
    T: np.ndarray,
    R: np.ndarray,
    Z: np.ndarray,
    Q: np.ndarray,
    H: np.ndarray | None = None,
    s0: np.ndarray | None = None,
    P0: np.ndarray | None = None,
) -> KalmanOutput:
    """Run the Kalman filter.

    Args:
        y: Observations, shape (T, n_obs). NaN values are treated as missing.
        T: State transition matrix, shape (n_states, n_states)
        R: Shock impact matrix, shape (n_states, n_shocks)
        Z: Observation matrix, shape (n_obs, n_states)
        Q: Shock covariance matrix, shape (n_shocks, n_shocks)
        H: Measurement error covariance, shape (n_obs, n_obs). Default: zeros.
        s0: Initial state mean, shape (n_states,). Default: zeros.
        P0: Initial state covariance, shape (n_states, n_states).
            Default: unconditional covariance (solution to discrete Lyapunov).

    Returns:
        KalmanOutput with log-likelihood and filtered states

    """
    n_periods, n_obs = y.shape
    n_states = T.shape[0]
    n_shocks = R.shape[1]

    # Default measurement error: none
    if H is None:
        H = np.zeros((n_obs, n_obs))

    # Default initial state: zero mean
    if s0 is None:
        s0 = np.zeros(n_states)

    # Default initial covariance: unconditional variance
    # Solves P = T @ P @ T' + R @ Q @ R' (discrete Lyapunov equation)
    if P0 is None:
        RQR = R @ Q @ R.T
        try:
            P0 = linalg.solve_discrete_lyapunov(T, RQR)
        except linalg.LinAlgError:
            # If Lyapunov fails (e.g., unit roots), use diffuse initialization
            P0 = np.eye(n_states) * 1e6

    # Storage
    filtered_states = np.zeros((n_periods, n_states))
    filtered_covs = np.zeros((n_periods, n_states, n_states))
    innovations = np.zeros((n_periods, n_obs))
    innovation_covs = np.zeros((n_periods, n_obs, n_obs))

    log_likelihood = 0.0

    # Initialize
    s_pred = s0
    P_pred = P0

    for t in range(n_periods):
        # Observation available?
        y_t = y[t, :]
        obs_mask = ~np.isnan(y_t)

        if np.any(obs_mask):
            # Select observed components
            y_obs = y_t[obs_mask]
            Z_obs = Z[obs_mask, :]
            H_obs = H[np.ix_(obs_mask, obs_mask)]

            # Innovation
            y_pred = Z_obs @ s_pred
            innovation = y_obs - y_pred

            # Innovation covariance
            S = Z_obs @ P_pred @ Z_obs.T + H_obs

            # Kalman gain
            try:
                S_inv = linalg.inv(S)
                K = P_pred @ Z_obs.T @ S_inv
            except linalg.LinAlgError:
                # Singular S - use pseudoinverse
                S_inv = linalg.pinv(S)
                K = P_pred @ Z_obs.T @ S_inv

            # Update
            s_filt = s_pred + K @ innovation
            P_filt = P_pred - K @ Z_obs @ P_pred

            # Log-likelihood contribution (multivariate normal)
            n_obs_t = np.sum(obs_mask)
            sign, logdet = np.linalg.slogdet(S)
            if sign <= 0:
                # Numerical issue - penalize heavily
                log_likelihood += -1e10
            else:
                log_likelihood += -0.5 * (
                    n_obs_t * np.log(2 * np.pi)
                    + logdet
                    + innovation @ S_inv @ innovation
                )

            # Store full innovation (with NaN for missing)
            innovations[t, obs_mask] = innovation
            # Store innovation covariance for observed components
            if np.all(obs_mask):
                # All observations present - direct assignment
                innovation_covs[t, :, :] = S
            else:
                # Some missing - use index assignment
                innovation_covs[t, :, :] = np.nan
                obs_idx = np.where(obs_mask)[0]
                for i, ii in enumerate(obs_idx):
                    for j, jj in enumerate(obs_idx):
                        innovation_covs[t, ii, jj] = S[i, j]
        else:
            # No observations - just predict
            s_filt = s_pred
            P_filt = P_pred
            innovations[t, :] = np.nan
            innovation_covs[t, :, :] = np.nan

        # Store filtered values
        filtered_states[t, :] = s_filt
        filtered_covs[t, :, :] = P_filt

        # Predict next period
        s_pred = T @ s_filt
        P_pred = T @ P_filt @ T.T + R @ Q @ R.T

    return KalmanOutput(
        log_likelihood=log_likelihood,
        filtered_states=filtered_states,
        filtered_covs=filtered_covs,
        innovations=innovations,
        innovation_covs=innovation_covs,
    )


def kalman_smoother(
    y: np.ndarray,
    T: np.ndarray,
    R: np.ndarray,
    Z: np.ndarray,
    Q: np.ndarray,
    H: np.ndarray | None = None,
    s0: np.ndarray | None = None,
    P0: np.ndarray | None = None,
) -> KalmanOutput:
    """Run the Kalman filter and smoother.

    The smoother provides E[s_t | y_1:T] - the state estimates using
    all available data, including future observations.

    Args:
        Same as kalman_filter

    Returns:
        KalmanOutput with log-likelihood, filtered states, and smoothed states

    """
    # First run the filter
    kf = kalman_filter(y, T, R, Z, Q, H, s0, P0)

    n_periods, n_states = kf.filtered_states.shape

    # Backward smoothing pass
    smoothed_states = np.zeros((n_periods, n_states))
    smoothed_covs = np.zeros((n_periods, n_states, n_states))

    # Initialize at last period: smoothed = filtered
    smoothed_states[-1, :] = kf.filtered_states[-1, :]
    smoothed_covs[-1, :, :] = kf.filtered_covs[-1, :, :]

    for t in range(n_periods - 2, -1, -1):
        # Predicted state and covariance at t+1 (given info at t)
        s_filt_t = kf.filtered_states[t, :]
        P_filt_t = kf.filtered_covs[t, :, :]

        s_pred_t1 = T @ s_filt_t
        P_pred_t1 = T @ P_filt_t @ T.T + R @ Q @ R.T

        # Smoother gain
        try:
            J = P_filt_t @ T.T @ linalg.inv(P_pred_t1)
        except linalg.LinAlgError:
            J = P_filt_t @ T.T @ linalg.pinv(P_pred_t1)

        # Smoothed estimates
        smoothed_states[t, :] = s_filt_t + J @ (
            smoothed_states[t + 1, :] - s_pred_t1
        )
        smoothed_covs[t, :, :] = P_filt_t + J @ (
            smoothed_covs[t + 1, :, :] - P_pred_t1
        ) @ J.T

    return KalmanOutput(
        log_likelihood=kf.log_likelihood,
        filtered_states=kf.filtered_states,
        filtered_covs=kf.filtered_covs,
        smoothed_states=smoothed_states,
        smoothed_covs=smoothed_covs,
        innovations=kf.innovations,
        innovation_covs=kf.innovation_covs,
    )


def kalman_smoother_tv(
    y: np.ndarray,
    T: np.ndarray,
    R: np.ndarray,
    Q: np.ndarray,
    build_obs_eq: callable,
    H: np.ndarray | None = None,
    s0: np.ndarray | None = None,
    P0: np.ndarray | None = None,
) -> KalmanOutput:
    """Kalman smoother with time-varying observation equation.

    For models where Z and d vary over time (e.g., normalised u-gap).

    Args:
        y: Observations, shape (T, n_obs)
        T: State transition matrix, shape (n_states, n_states)
        R: Shock impact matrix, shape (n_states, n_shocks)
        Q: Shock covariance, shape (n_shocks, n_shocks)
        build_obs_eq: Function(t, s_pred) -> (Z_t, d_t) returning:
            Z_t: Observation matrix at time t, shape (n_obs, n_states)
            d_t: Observation offset at time t, shape (n_obs,)
        H: Measurement error covariance, shape (n_obs, n_obs). Default: zeros.
        s0: Initial state mean, shape (n_states,). Default: zeros.
        P0: Initial state covariance, shape (n_states, n_states). Default: identity.

    Returns:
        KalmanOutput with log-likelihood, filtered and smoothed states
    """
    n_periods, n_obs = y.shape
    n_states = T.shape[0]

    if H is None:
        H = np.zeros((n_obs, n_obs))
    if s0 is None:
        s0 = np.zeros(n_states)
    if P0 is None:
        P0 = np.eye(n_states)

    # Storage
    s_pred_all = np.zeros((n_periods, n_states))
    s_filt_all = np.zeros((n_periods, n_states))
    P_pred_all = np.zeros((n_periods, n_states, n_states))
    P_filt_all = np.zeros((n_periods, n_states, n_states))

    log_likelihood = 0.0
    s = s0.copy()
    P = P0.copy()

    # Forward pass (filter)
    for t in range(n_periods):
        # Prediction
        s_pred = T @ s
        P_pred = T @ P @ T.T + R @ Q @ R.T

        s_pred_all[t] = s_pred
        P_pred_all[t] = P_pred

        # Time-varying observation equation
        Z_t, d_t = build_obs_eq(t, s_pred)

        # Innovation
        y_pred = Z_t @ s_pred + d_t
        v = y[t] - y_pred
        F = Z_t @ P_pred @ Z_t.T + H

        try:
            F_inv = linalg.inv(F)
            sign, logdet = np.linalg.slogdet(F)
            if sign > 0:
                log_likelihood += -0.5 * (n_obs * np.log(2 * np.pi) + logdet + v @ F_inv @ v)
        except linalg.LinAlgError:
            pass

        # Update
        K = P_pred @ Z_t.T @ F_inv
        s = s_pred + K @ v
        P = (np.eye(n_states) - K @ Z_t) @ P_pred

        s_filt_all[t] = s
        P_filt_all[t] = P

    # Backward pass (smoother)
    s_smooth = np.zeros((n_periods, n_states))
    P_smooth = np.zeros((n_periods, n_states, n_states))

    s_smooth[-1] = s_filt_all[-1]
    P_smooth[-1] = P_filt_all[-1]

    for t in range(n_periods - 2, -1, -1):
        try:
            J = P_filt_all[t] @ T.T @ linalg.inv(P_pred_all[t + 1])
        except linalg.LinAlgError:
            J = P_filt_all[t] @ T.T @ linalg.pinv(P_pred_all[t + 1])

        s_smooth[t] = s_filt_all[t] + J @ (s_smooth[t + 1] - s_pred_all[t + 1])
        P_smooth[t] = P_filt_all[t] + J @ (P_smooth[t + 1] - P_pred_all[t + 1]) @ J.T

    return KalmanOutput(
        log_likelihood=log_likelihood,
        filtered_states=s_filt_all,
        filtered_covs=P_filt_all,
        smoothed_states=s_smooth,
        smoothed_covs=P_smooth,
    )


def simulate_states(
    T: np.ndarray,
    R: np.ndarray,
    Q: np.ndarray,
    n_periods: int,
    s0: np.ndarray | None = None,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Simulate states and shocks from the state-space model.

    Args:
        T: State transition matrix
        R: Shock impact matrix
        Q: Shock covariance matrix
        n_periods: Number of periods to simulate
        s0: Initial state. Default: zeros.
        seed: Random seed for reproducibility

    Returns:
        Tuple of (states, shocks) with shapes (n_periods, n_states)
        and (n_periods, n_shocks)

    """
    if seed is not None:
        np.random.seed(seed)

    n_states = T.shape[0]
    n_shocks = R.shape[1]

    if s0 is None:
        s0 = np.zeros(n_states)

    # Cholesky decomposition of shock covariance
    Q_chol = linalg.cholesky(Q, lower=True)

    states = np.zeros((n_periods, n_states))
    shocks = np.zeros((n_periods, n_shocks))

    s = s0
    for t in range(n_periods):
        # Draw shocks
        eta = Q_chol @ np.random.randn(n_shocks)
        shocks[t, :] = eta

        # Update state
        s = T @ s + R @ eta
        states[t, :] = s

    return states, shocks
