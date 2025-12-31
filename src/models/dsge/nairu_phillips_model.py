"""Pure NAIRU-Phillips model (no IS curve or output gap dynamics).

Simpler structure matching the state-space NAIRU model:
- NAIRU is the only latent state (random walk)
- Phillips curves link normalised u_gap to inflation
- No IS curve, no r*, no output gap dynamics

States: [NAIRU, ε_p, ε_w]
Observations: [π, π_w]

Equations:
1. NAIRU dynamics (random walk):
   NAIRU_t = NAIRU_{t-1} + ε_nairu

2. Price Phillips Curve:
   π_t = γ_p × (U_t - NAIRU_t)/U_t + ρ_m × Δpm_t + ε_p

3. Wage Phillips Curve:
   π_w,t = γ_w × (U_t - NAIRU_t)/U_t + ε_w
"""

from dataclasses import dataclass, field
import numpy as np
from scipy import linalg


@dataclass
class NairuPhillipsParameters:
    """Parameters for pure NAIRU-Phillips model."""

    # Phillips curves (γ coefficients - expect negative)
    gamma_p: float = -1.5  # Price Phillips curve slope
    gamma_w: float = -2.0  # Wage Phillips curve slope
    rho_m: float = 0.05    # Import price pass-through
    lambda_w: float = -4.0 # Speed limit: ΔU/U effect on wages (negative: falling U → pressure)
    xi_oil: float = 0.02   # Oil price effect on prices (positive: higher oil → inflation)
    xi_coal: float = 0.02  # Coal price effect on prices (positive: higher coal → inflation)

    # Shock volatilities
    sigma_nairu: float = 0.15  # NAIRU innovation
    sigma_p: float = 0.3       # Price Phillips shock
    sigma_w: float = 0.5       # Wage Phillips shock

    def to_dict(self) -> dict:
        return {
            "gamma_p": self.gamma_p,
            "gamma_w": self.gamma_w,
            "rho_m": self.rho_m,
            "lambda_w": self.lambda_w,
            "xi_oil": self.xi_oil,
            "xi_coal": self.xi_coal,
            "sigma_nairu": self.sigma_nairu,
            "sigma_p": self.sigma_p,
            "sigma_w": self.sigma_w,
        }


@dataclass
class NairuPhillipsModel:
    """Pure NAIRU-Phillips model.

    State vector: s_t = [NAIRU_t, ε_p,t, ε_w,t]
    Observations: [π, π_w]
    """

    params: NairuPhillipsParameters = field(default_factory=NairuPhillipsParameters)

    n_states: int = 3  # [NAIRU, ε_p, ε_w]
    n_obs: int = 2     # [π, π_w]

    def kalman_filter(
        self,
        y: np.ndarray,  # Observations (T × 2): [π, π_w]
        U_obs: np.ndarray,  # Observed unemployment rate (T,)
        import_price_growth: np.ndarray | None = None,
        delta_U_over_U: np.ndarray | None = None,  # Speed limit: (U_{t-1} - U_{t-2})/U_{t-1}
        oil_change: np.ndarray | None = None,  # Oil price change (annual %)
        coal_change: np.ndarray | None = None,  # Coal price change (annual %)
        nairu_prior: float = 5.0,
    ) -> float:
        """Run Kalman filter with normalised u_gap.

        Phillips curves use (U - NAIRU)/U which is time-varying.
        Speed limit term ΔU/U captures rapid unemployment changes.
        Oil and coal price changes capture energy cost shocks.
        """
        p = self.params
        n_periods = len(y)
        n_states = self.n_states
        n_obs = self.n_obs

        # State transition: NAIRU random walk, ε_p and ε_w iid
        T_mat = np.array([
            [1, 0, 0],  # NAIRU: random walk
            [0, 0, 0],  # ε_p: iid
            [0, 0, 0],  # ε_w: iid
        ])

        # Shock impact
        R_mat = np.diag([p.sigma_nairu, p.sigma_p, p.sigma_w])

        # Shock covariance
        Q = np.eye(3)

        # No measurement error beyond what's in the state
        H = np.zeros((n_obs, n_obs))

        # Initial state
        s = np.array([nairu_prior, 0.0, 0.0])

        # Initial covariance - diffuse for NAIRU
        P = np.eye(n_states)
        P[0, 0] = 4.0  # Diffuse prior for NAIRU (std dev ~2pp)

        log_lik = 0.0

        for t in range(n_periods):
            U_t = U_obs[t]

            # Time-varying observation matrix Z
            # π = γ_p × (U - NAIRU)/U + ρ_m × Δpm + ε_p
            #   = γ_p - (γ_p/U) × NAIRU + ρ_m × Δpm + ε_p
            # π_w = γ_w × (U - NAIRU)/U + λ_w × ΔU/U + ε_w
            #     = γ_w - (γ_w/U) × NAIRU + λ_w × ΔU/U + ε_w
            Z = np.array([
                [-p.gamma_p / U_t, 1, 0],  # π: loads on NAIRU and ε_p
                [-p.gamma_w / U_t, 0, 1],  # π_w: loads on NAIRU and ε_w
            ])

            # Time-varying offset
            d = np.array([p.gamma_p, p.gamma_w])
            if import_price_growth is not None:
                d[0] += p.rho_m * import_price_growth[t]
            if delta_U_over_U is not None:
                d[1] += p.lambda_w * delta_U_over_U[t]  # Speed limit in wage equation
            if oil_change is not None:
                d[0] += p.xi_oil * oil_change[t]
            if coal_change is not None:
                d[0] += p.xi_coal * coal_change[t]

            # === Prediction step ===
            s_pred = T_mat @ s
            P_pred = T_mat @ P @ T_mat.T + R_mat @ Q @ R_mat.T

            # === Observation prediction ===
            y_pred = Z @ s_pred + d

            # === Innovation ===
            v = y[t] - y_pred

            # Innovation covariance
            F = Z @ P_pred @ Z.T + H

            # Check for numerical issues
            try:
                F_inv = linalg.inv(F)
                log_det_F = np.log(linalg.det(F))
            except linalg.LinAlgError:
                return -1e10

            if not np.isfinite(log_det_F):
                return -1e10

            # Log-likelihood contribution
            log_lik += -0.5 * (n_obs * np.log(2 * np.pi) + log_det_F + v @ F_inv @ v)

            # === Update step ===
            K = P_pred @ Z.T @ F_inv
            s = s_pred + K @ v
            P = (np.eye(n_states) - K @ Z) @ P_pred

        return log_lik


def compute_nairu_phillips_log_likelihood(
    y_obs: np.ndarray,
    U_obs: np.ndarray,
    params: NairuPhillipsParameters,
    import_price_growth: np.ndarray | None = None,
    delta_U_over_U: np.ndarray | None = None,
    oil_change: np.ndarray | None = None,
    coal_change: np.ndarray | None = None,
    nairu_prior: float = 5.0,
) -> float:
    """Compute log-likelihood for NAIRU-Phillips model."""
    try:
        model = NairuPhillipsModel(params=params)
        return model.kalman_filter(y_obs, U_obs, import_price_growth, delta_U_over_U, oil_change, coal_change, nairu_prior)
    except Exception:
        return -1e10


def extract_nairu_estimates(
    y_obs: np.ndarray,
    U_obs: np.ndarray,
    params: NairuPhillipsParameters,
    import_price_growth: np.ndarray | None = None,
    delta_U_over_U: np.ndarray | None = None,
    oil_change: np.ndarray | None = None,
    coal_change: np.ndarray | None = None,
    nairu_prior: float = 5.0,
    smooth: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract NAIRU estimates using Kalman filter and optional RTS smoother.

    The smoother uses future observations to revise early estimates,
    producing more accurate NAIRU paths especially at the start of the sample.

    Returns:
        nairu: Smoothed (or filtered) NAIRU estimates (T,)
        nairu_std: Standard deviation of NAIRU estimates (T,)
    """
    p = params
    n_periods = len(y_obs)
    n_states = 3

    # State transition
    T_mat = np.array([
        [1, 0, 0],
        [0, 0, 0],
        [0, 0, 0],
    ])
    R_mat = np.diag([p.sigma_nairu, p.sigma_p, p.sigma_w])
    Q = np.eye(3)
    H = np.zeros((2, 2))

    # Initial state
    s = np.array([nairu_prior, 0.0, 0.0])
    P = np.eye(3)
    P[0, 0] = 4.0

    # Storage for forward pass (needed for smoother)
    s_filtered = np.zeros((n_periods, n_states))
    P_filtered = np.zeros((n_periods, n_states, n_states))
    s_predicted = np.zeros((n_periods, n_states))
    P_predicted = np.zeros((n_periods, n_states, n_states))

    # === Forward pass (Kalman filter) ===
    for t in range(n_periods):
        U_t = U_obs[t]

        # Time-varying Z
        Z = np.array([
            [-p.gamma_p / U_t, 1, 0],
            [-p.gamma_w / U_t, 0, 1],
        ])
        d = np.array([p.gamma_p, p.gamma_w])
        if import_price_growth is not None:
            d[0] += p.rho_m * import_price_growth[t]
        if delta_U_over_U is not None:
            d[1] += p.lambda_w * delta_U_over_U[t]
        if oil_change is not None:
            d[0] += p.xi_oil * oil_change[t]
        if coal_change is not None:
            d[0] += p.xi_coal * coal_change[t]

        # Prediction
        s_pred = T_mat @ s
        P_pred = T_mat @ P @ T_mat.T + R_mat @ Q @ R_mat.T

        # Store predictions (needed for smoother)
        s_predicted[t] = s_pred
        P_predicted[t] = P_pred

        # Observation prediction
        y_pred = Z @ s_pred + d
        v = y_obs[t] - y_pred

        # Innovation covariance
        F = Z @ P_pred @ Z.T + H
        F_inv = linalg.inv(F)

        # Update
        K = P_pred @ Z.T @ F_inv
        s = s_pred + K @ v
        P = (np.eye(n_states) - K @ Z) @ P_pred

        # Store filtered estimates
        s_filtered[t] = s
        P_filtered[t] = P

    if not smooth:
        # Return filtered estimates only
        return s_filtered[:, 0], np.sqrt(P_filtered[:, 0, 0])

    # === Backward pass (RTS smoother) ===
    s_smoothed = np.zeros((n_periods, n_states))
    P_smoothed = np.zeros((n_periods, n_states, n_states))

    # Initialize with final filtered values
    s_smoothed[-1] = s_filtered[-1]
    P_smoothed[-1] = P_filtered[-1]

    # Backward recursion
    for t in range(n_periods - 2, -1, -1):
        # Smoother gain
        P_pred_inv = linalg.inv(P_predicted[t + 1])
        J = P_filtered[t] @ T_mat.T @ P_pred_inv

        # Smoothed state and covariance
        s_smoothed[t] = s_filtered[t] + J @ (s_smoothed[t + 1] - s_predicted[t + 1])
        P_smoothed[t] = P_filtered[t] + J @ (P_smoothed[t + 1] - P_predicted[t + 1]) @ J.T

    return s_smoothed[:, 0], np.sqrt(P_smoothed[:, 0, 0])


# Parameter bounds matching state-space model
NAIRU_PHILLIPS_PARAM_BOUNDS = {
    "gamma_p": (-6.0, -0.01),  # Price Phillips slope (allow flatter)
    "gamma_w": (-15.0, -0.01), # Wage Phillips slope (wide)
    "rho_m": (0.0, 0.3),       # Import price pass-through
    "lambda_w": (-10.0, 0.0),  # Speed limit (negative: falling U → wage pressure)
    "xi_oil": (0.0, 0.1),      # Oil price effect (positive: higher oil → inflation)
    "xi_coal": (0.0, 0.1),     # Coal price effect (positive: higher coal → inflation)
    "sigma_nairu": (0.01, 1.0),
    "sigma_p": (0.1, 1.5),
    "sigma_w": (0.1, 3.0),
}


if __name__ == "__main__":
    # Quick test
    params = NairuPhillipsParameters()
    model = NairuPhillipsModel(params=params)

    T = 100
    y = np.random.randn(T, 2) * 0.5
    U = 5 + np.random.randn(T) * 0.5

    ll = model.kalman_filter(y, U)
    print(f"Test log-likelihood: {ll:.2f}")
