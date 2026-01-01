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

Usage:
    from hlw_nairu_model import HLW_NAIRU_SPEC, load_hlw_nairu_data
    from estimation import estimate_two_stage

    result = estimate_two_stage(HLW_NAIRU_SPEC, load_hlw_nairu_data)
"""

from dataclasses import dataclass, field
import numpy as np
import pandas as pd
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

    def kalman_smoother(
        self,
        y: np.ndarray,
        r_lag: np.ndarray,
        U_obs: np.ndarray,
        import_price_growth: np.ndarray | None = None,
        nairu_prior: float = 5.0,
        normalise_u_gap: bool = True,
    ) -> dict:
        """Run Kalman smoother to extract smoothed states.

        Note: Has exogenous r_lag effect on state prediction, so can't use
        generic kalman_smoother_tv directly.

        Returns dict with:
            states: Smoothed states (T × n_states)
            log_likelihood: Log-likelihood
            state_names: List of state names
        """
        p = self.params
        T_mat, R_mat, _, Q = self.state_space_matrices()

        n_periods = len(y)
        n_states = self.n_states
        n_obs = self.n_obs

        H = np.zeros((n_obs, n_obs))
        H[2, 2] = p.sigma_okun**2

        # Initial state and covariance
        s0 = np.array([0.0, 0.0, nairu_prior, 0.0, 0.0])
        P0 = np.eye(n_states)
        P0[1, 1] = 10.0  # Diffuse for r*
        P0[2, 2] = 4.0   # Diffuse for NAIRU

        # Storage
        s_pred_all = np.zeros((n_periods, n_states))
        s_filt_all = np.zeros((n_periods, n_states))
        P_pred_all = np.zeros((n_periods, n_states, n_states))
        P_filt_all = np.zeros((n_periods, n_states, n_states))

        s, P = s0.copy(), P0.copy()
        log_lik = 0.0

        # Forward pass
        for t in range(n_periods):
            U_t = U_obs[t]

            # Time-varying Z and d
            if normalise_u_gap:
                Z = np.array([
                    [0, 0, -p.gamma_p / U_t, 1, 0],
                    [0, 0, -p.gamma_w / U_t, 0, 1],
                    [-p.omega, 0, 1, 0, 0],
                ])
                d = np.array([p.gamma_p, p.gamma_w, 0.0])
            else:
                Z = np.array([
                    [0, 0, -p.gamma_p, 1, 0],
                    [0, 0, -p.gamma_w, 0, 1],
                    [-p.omega, 0, 1, 0, 0],
                ])
                d = np.array([p.gamma_p * U_t, p.gamma_w * U_t, 0.0])

            if import_price_growth is not None:
                d[0] += p.rho_m * import_price_growth[t]

            # Prediction with exogenous r_lag effect
            s_p = T_mat @ s
            s_p[0] -= p.beta_r * r_lag[t]
            P_p = T_mat @ P @ T_mat.T + R_mat @ Q @ R_mat.T

            s_pred_all[t], P_pred_all[t] = s_p, P_p

            # Innovation
            v = y[t] - (Z @ s_p + d)
            F = Z @ P_p @ Z.T + H

            try:
                F_inv = linalg.inv(F)
                sign, logdet = np.linalg.slogdet(F)
                if sign > 0:
                    log_lik += -0.5 * (n_obs * np.log(2 * np.pi) + logdet + v @ F_inv @ v)
            except linalg.LinAlgError:
                F_inv = np.eye(n_obs)

            # Update
            K = P_p @ Z.T @ F_inv
            s = s_p + K @ v
            P = (np.eye(n_states) - K @ Z) @ P_p

            s_filt_all[t], P_filt_all[t] = s, P

        # Backward pass
        s_smooth = np.zeros((n_periods, n_states))
        s_smooth[-1] = s_filt_all[-1]

        for t in range(n_periods - 2, -1, -1):
            try:
                J = P_filt_all[t] @ T_mat.T @ linalg.inv(P_pred_all[t + 1])
            except linalg.LinAlgError:
                J = P_filt_all[t] @ T_mat.T @ linalg.pinv(P_pred_all[t + 1])
            s_smooth[t] = s_filt_all[t] + J @ (s_smooth[t + 1] - s_pred_all[t + 1])

        return {
            "states": s_smooth,
            "log_likelihood": log_lik,
            "state_names": ["output_gap", "r_star", "nairu", "eps_supply", "eps_wage"],
        }


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


# =============================================================================
# Data Loading
# =============================================================================


def load_hlw_nairu_data(
    start: str = "1984Q1",
    end: str | None = None,
    anchor_inflation: bool = True,
) -> dict:
    """Load data for HLW-NAIRU estimation.

    Returns dict with:
        y: Observations (T × 3): [π, π_w, U]
        r_lag: Lagged real rate (T,)
        U_obs: Unemployment rate (T,)
        import_price_growth: Year-on-year import price growth (T,)
        nairu_prior: Prior mean for NAIRU
        dates: Period index
    """
    from src.data.abs_loader import load_series
    from src.data.series_specs import CPI_TRIMMED_MEAN_QUARTERLY, UNEMPLOYMENT_RATE
    from src.data.cash_rate import get_cash_rate_qrtly
    from src.data.import_prices import get_import_price_growth_annual
    from src.data.ulc import get_ulc_growth_qrtly
    from src.models.dsge.data_loader import compute_inflation_anchor
    from src.models.dsge.shared import ensure_period_index, filter_date_range

    # Load and process inflation
    inflation_raw = ensure_period_index(load_series(CPI_TRIMMED_MEAN_QUARTERLY).data)
    inflation_annual = ((1 + inflation_raw / 100) ** 4 - 1) * 100

    if anchor_inflation:
        pi_anchor = compute_inflation_anchor(inflation_annual)
        inflation = inflation_annual - pi_anchor
    else:
        inflation = inflation_annual

    # Load ULC growth (for wage inflation)
    ulc_growth = ensure_period_index(get_ulc_growth_qrtly().data)

    # Load unemployment rate
    ur = ensure_period_index(load_series(UNEMPLOYMENT_RATE).data)
    if hasattr(ur.index, "freqstr") and ur.index.freqstr == "M":
        ur = ur.resample("Q").mean()
        ur.index = pd.PeriodIndex(ur.index, freq="Q")

    # Load cash rate for real rate
    cash_rate = ensure_period_index(get_cash_rate_qrtly().data)

    # Load import price growth
    import_price_growth = ensure_period_index(get_import_price_growth_annual().data)

    # Build DataFrame for alignment
    df = pd.DataFrame({
        "inflation": inflation,
        "ulc_growth": ulc_growth,
        "U": ur,
        "cash_rate": cash_rate,
        "inflation_raw": inflation_annual,
        "import_price_growth": import_price_growth,
    })

    # Compute real rate and lag it
    df["real_rate"] = df["cash_rate"] - df["inflation_raw"]
    df["real_rate_lag"] = df["real_rate"].shift(1)

    # Drop NaNs and filter to date range
    df = filter_date_range(df.dropna(), start, end)

    return {
        "y": df[["inflation", "ulc_growth", "U"]].values,
        "r_lag": df["real_rate_lag"].values,
        "U_obs": df["U"].values,
        "import_price_growth": df["import_price_growth"].values,
        "nairu_prior": float(np.mean(df["U"].values)),
        "dates": df.index,
    }


# =============================================================================
# Likelihood Wrapper (for generic estimator)
# =============================================================================


def _hlw_nairu_likelihood(params: HLWNairuParameters, data: dict) -> float:
    """Compute log-likelihood for HLW-NAIRU model."""
    return compute_hlw_nairu_log_likelihood(
        y_obs=data["y"],
        r_lag=data["r_lag"],
        U_obs=data["U_obs"],
        params=params,
        import_price_growth=data.get("import_price_growth"),
        nairu_prior=data.get("nairu_prior", 5.0),
    )


# =============================================================================
# State Extractor (for two-stage estimation)
# =============================================================================


def hlw_nairu_extract_states(params: HLWNairuParameters, data: dict) -> dict:
    """Extract smoothed states using Kalman smoother.

    Args:
        params: Model parameters
        data: Data dict with y, r_lag, U_obs, etc.

    Returns:
        Dict with states DataFrame and log_likelihood
    """
    model = HLWNairuModel(params=params)
    result = model.kalman_smoother(
        y=data["y"],
        r_lag=data["r_lag"],
        U_obs=data["U_obs"],
        import_price_growth=data.get("import_price_growth"),
        nairu_prior=data.get("nairu_prior", 5.0),
    )

    states_df = pd.DataFrame(
        result["states"],
        index=data["dates"],
        columns=result["state_names"],
    )
    states_df["unemployment"] = data["U_obs"]

    return {
        "states": states_df,
        "log_likelihood": result["log_likelihood"],
    }


# =============================================================================
# Model Specification
# =============================================================================

from src.models.dsge.estimation import ModelSpec

HLW_NAIRU_SPEC = ModelSpec(
    name="HLW-NAIRU",
    description="HLW with NAIRU latent state, normalised u-gap",
    param_class=HLWNairuParameters,
    param_bounds=HLW_NAIRU_PARAM_BOUNDS,
    estimate_params=[
        "rho_y", "beta_r", "gamma_p", "gamma_w", "rho_m", "omega",
        "sigma_demand", "sigma_supply", "sigma_wage", "sigma_rstar", "sigma_nairu",
    ],
    fixed_params={},
    likelihood_fn=_hlw_nairu_likelihood,
    state_extractor_fn=hlw_nairu_extract_states,
)


# =============================================================================
# Main
# =============================================================================


if __name__ == "__main__":
    from pathlib import Path
    import mgplot as mg
    from src.models.dsge.estimation import estimate_two_stage, print_single_result
    from src.models.dsge.plot_output_gap import plot_output_gap
    from src.models.dsge.plot_rstar import plot_rstar
    from src.models.dsge.plot_nairu import plot_nairu

    # Chart setup
    CHART_DIR = Path(__file__).parent.parent.parent.parent / "charts" / "dsge-hlw-nairu"
    mg.set_chart_dir(str(CHART_DIR))
    mg.clear_chart_dir()

    print("HLW-NAIRU Model - Two-Stage Estimation")
    print("=" * 60)

    result = estimate_two_stage(HLW_NAIRU_SPEC, load_hlw_nairu_data)
    print_single_result(result.estimation_result, HLW_NAIRU_SPEC)

    plot_output_gap(result.states["output_gap"], model_name="HLW-NAIRU")
    plot_rstar(result.states["r_star"], model_name="HLW-NAIRU")
    plot_nairu(result.states["nairu"], unemployment=result.states["unemployment"], model_name="HLW-NAIRU")

    print(f"\nCharts saved to: {CHART_DIR}")
