"""HLW-style model with NAIRU as latent state.

Restructures the model so Phillips curve uses u_gap directly (like state-space NAIRU model):
- NAIRU is a latent random walk state
- Phillips curve: π = γ × (U - NAIRU) + ε (u_gap drives inflation)
- Okun's law: U = NAIRU - ω × ŷ + ε (links output gap to unemployment level)

States: [ŷ, r*, NAIRU, ε_s, ε_w]
Observations: [π, π_w, U]

Equations:
1. IS Curve (backward-looking):
   ŷ_t = ρ_y × ŷ_{t-1} - β_r × (r_{t-1} - r*_t) + ε_demand

2. r* dynamics (random walk):
   r*_t = r*_{t-1} + ε_rstar

3. NAIRU dynamics (random walk):
   NAIRU_t = NAIRU_{t-1} + ε_nairu

4. Phillips Curve (u_gap form):
   π_t = γ_p × (U_t - NAIRU_t) + ρ_m × Δpm_t + ε_supply

5. Wage Phillips Curve:
   π_w,t = γ_w × (U_t - NAIRU_t) + ε_wage

6. Okun's Law (level form):
   U_t = NAIRU_t - ω × ŷ_t + ε_okun
"""

from dataclasses import dataclass, field
import numpy as np
from scipy import linalg


@dataclass
class HLWNairuParameters:
    """Parameters for HLW model with NAIRU."""

    # IS curve
    rho_y: float = 0.8  # Output gap persistence
    beta_r: float = 0.1  # Real rate gap effect on output

    # Phillips curves (γ coefficients - expect negative)
    # With normalised u_gap = (U-NAIRU)/U, typical values ~-1.5 to -2.5
    gamma_p: float = -1.5  # Price Phillips curve slope on normalised u_gap
    gamma_w: float = -1.5  # Wage Phillips curve slope on normalised u_gap
    rho_m: float = 0.05    # Import price pass-through to inflation

    # Okun's law
    omega: float = 0.3  # Okun coefficient (output gap → unemployment)

    # Shock volatilities
    sigma_demand: float = 0.5
    sigma_supply: float = 0.3
    sigma_wage: float = 0.3
    sigma_rstar: float = 0.1
    sigma_nairu: float = 0.1  # NAIRU innovation volatility
    sigma_okun: float = 0.2  # Measurement error on Okun's law

    def to_dict(self) -> dict:
        return {
            "rho_y": self.rho_y,
            "beta_r": self.beta_r,
            "gamma_p": self.gamma_p,
            "gamma_w": self.gamma_w,
            "rho_m": self.rho_m,
            "omega": self.omega,
            "sigma_demand": self.sigma_demand,
            "sigma_supply": self.sigma_supply,
            "sigma_wage": self.sigma_wage,
            "sigma_rstar": self.sigma_rstar,
            "sigma_nairu": self.sigma_nairu,
            "sigma_okun": self.sigma_okun,
        }


@dataclass
class HLWNairuModel:
    """HLW model with NAIRU as latent state.

    State vector: s_t = [ŷ_t, r*_t, NAIRU_t, ε_s,t, ε_w,t]
    Exogenous: r_t (real rate), U_t (unemployment rate)

    Observations: [π, π_w, U]
    """

    params: HLWNairuParameters = field(default_factory=HLWNairuParameters)

    n_states: int = 5  # [ŷ, r*, NAIRU, ε_s, ε_w]
    n_shocks: int = 5  # [ε_d, ε_r*, ε_nairu, ε_s, ε_w]
    n_obs: int = 3  # [π, π_w, U]

    def state_space_matrices(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Build time-invariant state-space matrices.

        State: s_t = [ŷ, r*, NAIRU, ε_s, ε_w]

        Transition: s_{t+1} = T @ s_t + c_t + R @ η_{t+1}
            where c_t contains exogenous effects (r_lag)

        Observation: y_t = Z @ s_t + d_t
            where d_t contains exogenous effects (U_obs for Phillips, import prices)

        Returns:
            T: State transition (5×5)
            R: Shock impact (5×5)
            Z: Observation matrix (3×5)
            Q: Shock covariance (5×5)
        """
        p = self.params

        # State transition matrix T
        # s = [ŷ, r*, NAIRU, ε_s, ε_w]
        T = np.array([
            [p.rho_y, p.beta_r, 0, 0, 0],  # ŷ: persistence + r* effect
            [0, 1, 0, 0, 0],                # r*: random walk
            [0, 0, 1, 0, 0],                # NAIRU: random walk
            [0, 0, 0, 0, 0],                # ε_s: iid
            [0, 0, 0, 0, 0],                # ε_w: iid
        ])

        # Shock impact matrix R (diagonal with std devs)
        R = np.diag([
            p.sigma_demand,
            p.sigma_rstar,
            p.sigma_nairu,
            p.sigma_supply,
            p.sigma_wage,
        ])

        # Observation matrix Z
        # Observations: [π, π_w, U]
        # π = γ_p × (U_obs - NAIRU) + ρ_m × Δpm + ε_s
        #   = -γ_p × NAIRU + ε_s + [γ_p × U_obs + ρ_m × Δpm]  (last part is d_t)
        # π_w = γ_w × (U_obs - NAIRU) + ε_w
        #     = -γ_w × NAIRU + ε_w + [γ_w × U_obs]
        # U = NAIRU - ω × ŷ + ε_okun
        Z = np.array([
            [0, 0, -p.gamma_p, 1, 0],  # π: loads on NAIRU (negative of γ) and ε_s
            [0, 0, -p.gamma_w, 0, 1],  # π_w: loads on NAIRU and ε_w
            [-p.omega, 0, 1, 0, 0],    # U: loads on ŷ (negative) and NAIRU
        ])

        # Shock covariance (standard normal, scaling in R)
        Q = np.eye(5)

        return T, R, Z, Q

    def kalman_filter_with_exog(
        self,
        y: np.ndarray,  # Observations (T × 3): [π, π_w, U]
        r_lag: np.ndarray,  # Lagged real rate (T,)
        U_obs: np.ndarray,  # Observed unemployment rate (T,)
        import_price_growth: np.ndarray | None = None,  # Import price growth (T,)
        nairu_prior: float = 5.0,  # Prior mean for NAIRU
        normalise_u_gap: bool = True,  # Use (U-NAIRU)/U instead of (U-NAIRU)
    ) -> float:
        """Run Kalman filter with exogenous variables.

        Handles:
        - Exogenous r_lag effect on IS curve (state transition)
        - Time-varying Z matrix for normalised u_gap: (U-NAIRU)/U
        - Exogenous import prices in Phillips curves
        """
        p = self.params
        T_mat, R_mat, _, Q = self.state_space_matrices()

        n_periods = len(y)
        n_states = self.n_states
        n_obs = self.n_obs

        # Measurement error covariance (only on U via Okun)
        H = np.zeros((n_obs, n_obs))
        H[2, 2] = p.sigma_okun**2

        # Initial state: [ŷ=0, r*=0, NAIRU=prior, ε_s=0, ε_w=0]
        s = np.array([0.0, 0.0, nairu_prior, 0.0, 0.0])

        # Initial covariance - diffuse for r* and NAIRU
        P = np.eye(n_states)
        P[1, 1] = 10.0  # Diffuse prior for r*
        P[2, 2] = 4.0   # Diffuse prior for NAIRU (std dev ~2pp)

        log_lik = 0.0

        for t in range(n_periods):
            # === Build time-varying Z matrix ===
            # With normalised u_gap = (U - NAIRU)/U:
            #   π = γ_p × (U - NAIRU)/U + ρ_m × Δpm + ε_s
            #     = γ_p - (γ_p/U) × NAIRU + ρ_m × Δpm + ε_s
            #   π_w = γ_w × (U - NAIRU)/U + ε_w
            #       = γ_w - (γ_w/U) × NAIRU + ε_w
            #   U = NAIRU - ω × ŷ + ε_okun

            U_t = U_obs[t]

            if normalise_u_gap:
                # Coefficients on NAIRU are -γ/U (time-varying)
                Z = np.array([
                    [0, 0, -p.gamma_p / U_t, 1, 0],  # π
                    [0, 0, -p.gamma_w / U_t, 0, 1],  # π_w
                    [-p.omega, 0, 1, 0, 0],          # U
                ])
                # Offsets: π gets γ_p (the "1" part of (U-NAIRU)/U = 1 - NAIRU/U)
                d = np.array([p.gamma_p, p.gamma_w, 0.0])
            else:
                # Un-normalised: coefficients are just -γ
                Z = np.array([
                    [0, 0, -p.gamma_p, 1, 0],
                    [0, 0, -p.gamma_w, 0, 1],
                    [-p.omega, 0, 1, 0, 0],
                ])
                # Offsets include γ × U
                d = np.array([p.gamma_p * U_t, p.gamma_w * U_t, 0.0])

            # Add import price effect to π offset
            if import_price_growth is not None:
                d[0] += p.rho_m * import_price_growth[t]

            # === Prediction step ===
            s_pred = T_mat @ s

            # Add exogenous real rate effect to IS curve
            s_pred[0] -= p.beta_r * r_lag[t]

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


def compute_hlw_nairu_log_likelihood(
    y_obs: np.ndarray,  # [π, π_w, U] (T × 3)
    r_lag: np.ndarray,  # Lagged real rate (T,)
    U_obs: np.ndarray,  # Observed unemployment (T,)
    params: HLWNairuParameters,
    import_price_growth: np.ndarray | None = None,
    nairu_prior: float = 5.0,
    normalise_u_gap: bool = True,
) -> float:
    """Compute log-likelihood for HLW-NAIRU model."""
    try:
        model = HLWNairuModel(params=params)
        return model.kalman_filter_with_exog(
            y_obs, r_lag, U_obs, import_price_growth, nairu_prior, normalise_u_gap
        )
    except Exception:
        return -1e10


# Parameter bounds
# With normalised u_gap = (U-NAIRU)/U, gamma values match state-space model (~-1.5 to -2.5)
HLW_NAIRU_PARAM_BOUNDS = {
    "rho_y": (0.3, 0.99),
    "beta_r": (0.01, 0.5),
    "gamma_p": (-4.0, -0.1),   # Negative: higher u_gap → lower inflation (state-space ~-1.5)
    "gamma_w": (-4.0, -0.1),   # Negative: higher u_gap → lower wage growth
    "rho_m": (0.0, 0.3),       # Import price pass-through
    "omega": (0.1, 1.0),       # Okun coefficient
    "sigma_demand": (0.1, 3.0),
    "sigma_supply": (0.01, 2.0),
    "sigma_wage": (0.01, 2.0),
    "sigma_rstar": (0.01, 1.0),
    "sigma_nairu": (0.01, 0.5),
    "sigma_okun": (0.05, 1.0),
}


if __name__ == "__main__":
    # Quick test
    params = HLWNairuParameters()
    model = HLWNairuModel(params=params)

    # Fake data
    T = 100
    y = np.column_stack([
        np.random.randn(T) * 0.5,  # π
        np.random.randn(T) * 0.5,  # π_w
        5 + np.random.randn(T) * 0.5,  # U around 5%
    ])
    r_lag = np.random.randn(T) * 2
    U_obs = y[:, 2]

    ll = model.kalman_filter_with_exog(y, r_lag, U_obs)
    print(f"Test log-likelihood: {ll:.2f}")
