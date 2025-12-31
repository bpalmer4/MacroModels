"""Combined HLW + NAIRU-Phillips model.

Combines the best features of both models:
- From HLW: Latent r* (natural rate), IS-style demand linkage
- From NAIRU-Phillips: Latent NAIRU, normalised u-gap, speed limit, energy prices

States: [r*, NAIRU, ε_p, ε_w]
Observables: [π, Δulc, U]

Equations:
1. r* dynamics (random walk): r*_t = r*_{t-1} + ε_rstar
2. NAIRU dynamics (random walk): NAIRU_t = NAIRU_{t-1} + ε_nairu
3. Okun-IS hybrid: U_t = NAIRU_t + β_r×(r_{t-1} - r*_t) + ε_okun
4. Price Phillips: π_t = γ_p×(U_t - NAIRU_t)/U_t + ρ_m×Δpm_t + ξ_oil×Δoil_t + ε_p
5. Wage Phillips: Δulc_t = γ_w×(U_t - NAIRU_t)/U_t + λ_w×(ΔU/U)_t + ε_w

Key insight: We skip output gap estimation entirely. The real rate gap (r - r*)
affects unemployment directly through an Okun-IS hybrid equation.
"""

from dataclasses import dataclass
import numpy as np
from scipy import linalg


@dataclass
class HLWNairuPhillipsParameters:
    """Parameters for combined HLW + NAIRU-Phillips model."""

    # Okun-IS hybrid (β_r: real rate gap effect on unemployment)
    beta_r: float = 0.15  # Positive: higher r-r* → higher U

    # Phillips curves (γ coefficients - expect negative)
    gamma_p: float = -1.5  # Price Phillips curve slope
    gamma_w: float = -2.0  # Wage Phillips curve slope
    rho_m: float = 0.05    # Import price pass-through
    lambda_w: float = -4.0  # Speed limit: ΔU/U effect on wages
    xi_oil: float = 0.02   # Oil price effect on prices
    xi_coal: float = 0.02  # Coal price effect on prices

    # Shock volatilities
    sigma_rstar: float = 0.10   # r* innovation
    sigma_nairu: float = 0.15  # NAIRU innovation
    sigma_okun: float = 0.30   # Okun-IS residual
    sigma_p: float = 0.30      # Price Phillips shock
    sigma_w: float = 0.50      # Wage Phillips shock

    def to_dict(self) -> dict:
        return {
            "beta_r": self.beta_r,
            "gamma_p": self.gamma_p,
            "gamma_w": self.gamma_w,
            "rho_m": self.rho_m,
            "lambda_w": self.lambda_w,
            "xi_oil": self.xi_oil,
            "xi_coal": self.xi_coal,
            "sigma_rstar": self.sigma_rstar,
            "sigma_nairu": self.sigma_nairu,
            "sigma_okun": self.sigma_okun,
            "sigma_p": self.sigma_p,
            "sigma_w": self.sigma_w,
        }


@dataclass
class HLWNairuPhillipsModel:
    """Combined HLW + NAIRU-Phillips model.

    State vector: s_t = [r*_t, NAIRU_t, ε_p,t, ε_w,t]
    Observations: [π, Δulc] (U is exogenous, enters Phillips curves directly)
    """

    params: HLWNairuPhillipsParameters

    n_states: int = 4  # [r*, NAIRU, ε_p, ε_w]
    n_obs: int = 2     # [π, Δulc] - U is exogenous

    def kalman_filter(
        self,
        y: np.ndarray,  # Observations (T × 2): [π, Δulc]
        U_obs: np.ndarray,  # Observed unemployment (T,) - exogenous
        r_lag: np.ndarray,  # Lagged real interest rate (T,)
        import_price_growth: np.ndarray | None = None,
        delta_U_over_U: np.ndarray | None = None,
        oil_change: np.ndarray | None = None,
        coal_change: np.ndarray | None = None,
        rstar_prior: float = 1.0,
        nairu_prior: float = 5.0,
    ) -> float:
        """Run Kalman filter for combined model.

        U is exogenous - enters Phillips curves but is not an observable.
        This forces the model to use Phillips curves to identify NAIRU.
        """
        p = self.params
        n_periods = len(y)
        n_states = self.n_states
        n_obs = self.n_obs

        # State transition: r* and NAIRU are random walks, shocks are iid
        T_mat = np.array([
            [1, 0, 0, 0],  # r*: random walk
            [0, 1, 0, 0],  # NAIRU: random walk
            [0, 0, 0, 0],  # ε_p: iid
            [0, 0, 0, 0],  # ε_w: iid
        ])

        # Shock impact matrix
        R_mat = np.diag([p.sigma_rstar, p.sigma_nairu, p.sigma_p, p.sigma_w])

        # Shock covariance (identity - variances absorbed in R_mat)
        Q = np.eye(4)

        # Initial state
        s = np.array([rstar_prior, nairu_prior, 0.0, 0.0])

        # Initial covariance - tighter prior on NAIRU
        P = np.eye(n_states)
        P[0, 0] = 1.0  # r* prior uncertainty (std dev 1pp)
        P[1, 1] = 0.25  # NAIRU prior uncertainty (std dev 0.5pp) - tighter

        # No measurement error
        H = np.zeros((n_obs, n_obs))

        log_lik = 0.0

        for t in range(n_periods):
            U_t = U_obs[t]  # Exogenous unemployment
            r_t = r_lag[t]  # Lagged real rate

            # === Build observation equation ===
            # Only 2 observations: [π, Δulc]
            # U enters through the normalised u-gap but is not observed
            #
            # Observation 1 (π): γ_p × (U - NAIRU)/U + controls + ε_p
            #   = γ_p - (γ_p/U)×NAIRU + controls + ε_p
            #   Z row: [0, -γ_p/U, 1, 0]
            #
            # Observation 2 (Δulc): γ_w × (U - NAIRU)/U + λ_w×ΔU/U + ε_w
            #   = γ_w - (γ_w/U)×NAIRU + λ_w×ΔU/U + ε_w
            #   Z row: [0, -γ_w/U, 0, 1]

            Z = np.array([
                [0, -p.gamma_p / U_t, 1, 0],      # π
                [0, -p.gamma_w / U_t, 0, 1],      # Δulc
            ])

            # Time-varying offset d
            d = np.array([p.gamma_p, p.gamma_w])

            # Add controls to price Phillips
            if import_price_growth is not None:
                d[0] += p.rho_m * import_price_growth[t]
            if oil_change is not None:
                d[0] += p.xi_oil * oil_change[t]
            if coal_change is not None:
                d[0] += p.xi_coal * coal_change[t]

            # Add speed limit to wage Phillips
            if delta_U_over_U is not None:
                d[1] += p.lambda_w * delta_U_over_U[t]

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


def compute_hlw_nairu_phillips_log_likelihood(
    y_obs: np.ndarray,
    U_obs: np.ndarray,
    r_lag: np.ndarray,
    params: HLWNairuPhillipsParameters,
    import_price_growth: np.ndarray | None = None,
    delta_U_over_U: np.ndarray | None = None,
    oil_change: np.ndarray | None = None,
    coal_change: np.ndarray | None = None,
    rstar_prior: float = 1.0,
    nairu_prior: float = 5.0,
) -> float:
    """Compute log-likelihood for combined model."""
    try:
        model = HLWNairuPhillipsModel(params=params)
        return model.kalman_filter(
            y_obs, U_obs, r_lag, import_price_growth, delta_U_over_U,
            oil_change, coal_change, rstar_prior, nairu_prior
        )
    except Exception:
        return -1e10


def extract_latent_estimates(
    y_obs: np.ndarray,
    U_obs: np.ndarray,
    r_lag: np.ndarray,
    params: HLWNairuPhillipsParameters,
    import_price_growth: np.ndarray | None = None,
    delta_U_over_U: np.ndarray | None = None,
    oil_change: np.ndarray | None = None,
    coal_change: np.ndarray | None = None,
    rstar_prior: float = 1.0,
    nairu_prior: float = 5.0,
    smooth: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Extract r* and NAIRU estimates using Kalman filter/smoother.

    Returns:
        rstar: r* estimates (T,)
        rstar_std: Standard deviation of r* (T,)
        nairu: NAIRU estimates (T,)
        nairu_std: Standard deviation of NAIRU (T,)
    """
    p = params
    n_periods = len(y_obs)
    n_states = 4
    n_obs = 2

    # State transition
    T_mat = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
    ])
    R_mat = np.diag([p.sigma_rstar, p.sigma_nairu, p.sigma_p, p.sigma_w])
    Q = np.eye(4)
    H = np.zeros((n_obs, n_obs))

    # Initial state
    s = np.array([rstar_prior, nairu_prior, 0.0, 0.0])
    P = np.eye(n_states)
    P[0, 0] = 1.0  # r* prior uncertainty
    P[1, 1] = 0.25  # NAIRU prior uncertainty - tighter

    # Storage for forward pass
    s_filtered = np.zeros((n_periods, n_states))
    P_filtered = np.zeros((n_periods, n_states, n_states))
    s_predicted = np.zeros((n_periods, n_states))
    P_predicted = np.zeros((n_periods, n_states, n_states))

    # === Forward pass ===
    for t in range(n_periods):
        U_t = U_obs[t]

        Z = np.array([
            [0, -p.gamma_p / U_t, 1, 0],
            [0, -p.gamma_w / U_t, 0, 1],
        ])

        d = np.array([p.gamma_p, p.gamma_w])
        if import_price_growth is not None:
            d[0] += p.rho_m * import_price_growth[t]
        if oil_change is not None:
            d[0] += p.xi_oil * oil_change[t]
        if coal_change is not None:
            d[0] += p.xi_coal * coal_change[t]
        if delta_U_over_U is not None:
            d[1] += p.lambda_w * delta_U_over_U[t]

        # Prediction
        s_pred = T_mat @ s
        P_pred = T_mat @ P @ T_mat.T + R_mat @ Q @ R_mat.T

        s_predicted[t] = s_pred
        P_predicted[t] = P_pred

        # Update
        y_pred = Z @ s_pred + d
        v = y_obs[t] - y_pred
        F = Z @ P_pred @ Z.T + H
        F_inv = linalg.inv(F)
        K = P_pred @ Z.T @ F_inv
        s = s_pred + K @ v
        P = (np.eye(n_states) - K @ Z) @ P_pred

        s_filtered[t] = s
        P_filtered[t] = P

    if not smooth:
        return (
            s_filtered[:, 0], np.sqrt(P_filtered[:, 0, 0]),
            s_filtered[:, 1], np.sqrt(P_filtered[:, 1, 1]),
        )

    # === Backward pass (RTS smoother) ===
    s_smoothed = np.zeros((n_periods, n_states))
    P_smoothed = np.zeros((n_periods, n_states, n_states))

    s_smoothed[-1] = s_filtered[-1]
    P_smoothed[-1] = P_filtered[-1]

    for t in range(n_periods - 2, -1, -1):
        P_pred_inv = linalg.inv(P_predicted[t + 1])
        J = P_filtered[t] @ T_mat.T @ P_pred_inv
        s_smoothed[t] = s_filtered[t] + J @ (s_smoothed[t + 1] - s_predicted[t + 1])
        P_smoothed[t] = P_filtered[t] + J @ (P_smoothed[t + 1] - P_predicted[t + 1]) @ J.T

    return (
        s_smoothed[:, 0], np.sqrt(P_smoothed[:, 0, 0]),
        s_smoothed[:, 1], np.sqrt(P_smoothed[:, 1, 1]),
    )


# Parameter bounds
HLW_NAIRU_PHILLIPS_PARAM_BOUNDS = {
    "beta_r": (0.0, 0.5),         # Real rate gap effect on U (positive)
    "gamma_p": (-6.0, -0.01),     # Price Phillips slope
    "gamma_w": (-15.0, -0.01),    # Wage Phillips slope
    "rho_m": (0.0, 0.3),          # Import price pass-through
    "lambda_w": (-10.0, 0.0),     # Speed limit
    "xi_oil": (0.0, 0.1),         # Oil price effect
    "xi_coal": (0.0, 0.1),        # Coal price effect
    "sigma_rstar": (0.01, 0.5),   # r* innovation
    "sigma_nairu": (0.01, 1.0),   # NAIRU innovation
    "sigma_okun": (0.1, 1.0),     # Okun residual
    "sigma_p": (0.1, 1.5),        # Price Phillips shock
    "sigma_w": (0.1, 3.0),        # Wage Phillips shock
}


if __name__ == "__main__":
    # Quick test
    params = HLWNairuPhillipsParameters()
    model = HLWNairuPhillipsModel(params=params)

    T = 100
    y = np.column_stack([
        np.random.randn(T) * 0.5,  # π
        np.random.randn(T) * 0.5,  # Δulc
        5 + np.random.randn(T) * 0.5,  # U
    ])
    r_lag = 2 + np.random.randn(T) * 0.5

    ll = model.kalman_filter(y, r_lag)
    print(f"Test log-likelihood: {ll:.2f}")
