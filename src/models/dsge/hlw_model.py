"""HLW-style model with r* instead of Taylor rule.

Replaces the Taylor rule with:
- r* as a latent state (random walk)
- IS curve responds to real rate gap (i - π - r*)
- Interest rate is exogenous (observed data)

Equations:
1. IS Curve (backward-looking):
   ŷ_t = ρ_y × ŷ_{t-1} - β × (r_{t-1} - r*_t) + ε_demand
   where r = i - π (ex-post real rate)

2. Phillips Curve (anchor-adjusted):
   π_t = κ_p × ŷ_t + ε_supply
   (using anchor-adjusted inflation, so no β×E[π'] term)

3. Wage Phillips Curve:
   π_w,t = κ_w × ŷ_t + ε_wage

4. r* dynamics (random walk):
   r*_t = r*_{t-1} + ε_rstar

5. Okun's Law:
   u_gap_t = -ω × ŷ_t + ε_okun

States: [ŷ_{t-1}, r*, ε_supply, ε_wage]
Observables: [ŷ, π, π_w, u_gap] + exogenous [r]

Usage:
    from hlw_model import HLW_SPEC, load_hlw_data
    from estimation import estimate_model, estimate_by_regime

    data = load_hlw_data(start="1984Q1")
    result = estimate_model(HLW_SPEC, data)
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import linalg


@dataclass
class HLWParameters:
    """Parameters for HLW-style model."""

    # IS curve
    rho_y: float = 0.8  # Output gap persistence
    beta_r: float = 0.1  # Real rate gap effect on output

    # Phillips curves
    kappa_p: float = 0.1  # Price Phillips curve slope
    kappa_w: float = 0.1  # Wage Phillips curve slope
    rho_m: float = 0.05  # Import price pass-through to inflation

    # Okun's law
    omega: float = 0.4  # Okun coefficient

    # Shock volatilities
    sigma_demand: float = 0.5
    sigma_supply: float = 0.3
    sigma_wage: float = 0.3
    sigma_rstar: float = 0.1
    sigma_okun: float = 0.1  # Measurement error on Okun's law

    def to_dict(self) -> dict:
        return {
            "rho_y": self.rho_y,
            "beta_r": self.beta_r,
            "kappa_p": self.kappa_p,
            "kappa_w": self.kappa_w,
            "rho_m": self.rho_m,
            "omega": self.omega,
            "sigma_demand": self.sigma_demand,
            "sigma_supply": self.sigma_supply,
            "sigma_wage": self.sigma_wage,
            "sigma_rstar": self.sigma_rstar,
            "sigma_okun": self.sigma_okun,
        }


@dataclass
class HLWModel:
    """HLW-style model with r* and exogenous interest rate.

    State vector: s_t = [ŷ_{t-1}, r*_t, ε_s,t, ε_w,t]
    Exogenous: r_t = i_t - π_t (real rate from data)

    Transition:
        ŷ_t = ρ_y × ŷ_{t-1} - β_r × (r_{t-1} - r*_t) + ε_d,t
        r*_t = r*_{t-1} + ε_r*,t
        ε_s,t = ε_s,t (iid)
        ε_w,t = ε_w,t (iid)

    Observation:
        π_t = κ_p × ŷ_t + ε_s,t
        π_w,t = κ_w × ŷ_t + ε_w,t
        u_gap_t = -ω × ŷ_t + measurement error
    """

    params: HLWParameters = field(default_factory=HLWParameters)

    n_states: int = 4  # [ŷ_{t-1}, r*, ε_s, ε_w]
    n_shocks: int = 4  # [ε_d, ε_r*, ε_s, ε_w]

    def state_space_matrices(
        self,
        r_lag: np.ndarray,  # Lagged real rate (exogenous), length T
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Build state-space matrices.

        Note: This model has time-varying system due to exogenous r.
        We handle this by including r_lag effect in the observation equation.

        State: s_t = [ŷ_t, r*_t, ε_s,t, ε_w,t]

        Transition: s_{t+1} = T @ s_t + R @ η_{t+1}
        Observation: y_t = Z @ s_t + d_t + measurement_error
            where d_t depends on r_lag

        Returns:
            T: State transition (4×4)
            R: Shock impact (4×4)
            Z: Observation matrix (3×4) for [π, π_w, u_gap]
            Q: Shock covariance (4×4)
            H: Measurement error covariance (3×3)

        """
        p = self.params

        # State transition matrix T
        # s = [ŷ, r*, ε_s, ε_w]
        # ŷ_{t+1} = ρ_y × ŷ_t + β_r × r*_t + ε_d (note: r_lag handled separately)
        # r*_{t+1} = r*_t + ε_r*
        # ε_s, ε_w are iid (no persistence in this simplified version)
        T = np.array([
            [p.rho_y, p.beta_r, 0, 0],  # ŷ equation (r* enters positively: higher r* = higher neutral = less drag)
            [0, 1, 0, 0],                # r* random walk
            [0, 0, 0, 0],                # ε_s iid
            [0, 0, 0, 0],                # ε_w iid
        ])

        # Shock impact matrix R
        R = np.diag([p.sigma_demand, p.sigma_rstar, p.sigma_supply, p.sigma_wage])

        # Observation matrix Z
        # Observables: [π, π_w, u_gap]
        # π = κ_p × ŷ + ε_s
        # π_w = κ_w × ŷ + ε_w
        # u_gap = -ω × ŷ
        Z = np.array([
            [p.kappa_p, 0, 1, 0],  # π = κ_p × ŷ + ε_s
            [p.kappa_w, 0, 0, 1],  # π_w = κ_w × ŷ + ε_w
            [-p.omega, 0, 0, 0],   # u_gap = -ω × ŷ
        ])

        # Shock covariance (standard normal, scaling in R)
        Q = np.eye(4)

        # Measurement error on u_gap (Okun's law not exact)
        H = np.zeros((3, 3))
        H[2, 2] = p.sigma_okun**2

        return T, R, Z, Q, H

    def kalman_filter_with_exog(
        self,
        y: np.ndarray,  # Observations (T × 3): [π, π_w, u_gap]
        r_lag: np.ndarray,  # Lagged real rate (T,)
        import_price_growth: np.ndarray | None = None,  # Import price growth (T,)
    ) -> float:
        """Run Kalman filter with exogenous real rate and import prices.

        The IS curve is:
            ŷ_t = ρ_y × ŷ_{t-1} - β_r × (r_{t-1} - r*_{t-1}) + ε_d

        The Phillips curve includes import prices:
            π_t = κ_p × ŷ_t + ρ_m × Δpm_t + ε_s

        We handle exogenous variables by adding them to predictions.
        """
        p = self.params
        T_mat, R_mat, Z, Q, H = self.state_space_matrices(r_lag)

        n_periods = len(y)
        n_states = self.n_states
        n_obs = y.shape[1]

        # Initial state
        s = np.zeros(n_states)
        # Initial covariance - diffuse for r*
        P = np.eye(n_states)
        P[1, 1] = 10.0  # Diffuse prior for r*

        log_lik = 0.0

        for t in range(n_periods):
            # Prediction step
            # ŷ_t = ρ_y × ŷ_{t-1} + β_r × r*_{t-1} - β_r × r_{t-1} + ε_d
            # The -β_r × r_{t-1} term is the exogenous effect
            s_pred = T_mat @ s
            s_pred[0] -= p.beta_r * r_lag[t]  # Add exogenous real rate effect

            P_pred = T_mat @ P @ T_mat.T + R_mat @ Q @ R_mat.T

            # Innovation
            y_pred = Z @ s_pred

            # Add import price effect to inflation prediction
            if import_price_growth is not None:
                y_pred[0] += p.rho_m * import_price_growth[t]

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

            # Update step
            K = P_pred @ Z.T @ F_inv
            s = s_pred + K @ v
            P = (np.eye(n_states) - K @ Z) @ P_pred

        return log_lik

    def kalman_smoother(
        self,
        y: np.ndarray,  # Observations (T × 3): [π, π_w, u_gap]
        r_lag: np.ndarray,  # Lagged real rate (T,)
        import_price_growth: np.ndarray | None = None,
    ) -> dict:
        """Run Kalman smoother to extract smoothed states.

        Returns dict with:
            states: Smoothed states (T × n_states)
            states_filtered: Filtered states (T × n_states)
            log_likelihood: Log-likelihood
            state_names: List of state names
        """
        p = self.params
        T_mat, R_mat, Z, Q, H = self.state_space_matrices(r_lag)

        # Initial state
        s0 = np.zeros(self.n_states)
        P0 = np.eye(self.n_states)
        P0[1, 1] = 10.0  # Diffuse prior for r*

        # Build time-varying observation equation
        # (time-varying due to r_lag effect on state prediction and import prices on obs)
        def build_obs_eq(t, s_pred):
            # Adjust s_pred for exogenous r_lag effect
            s_adj = s_pred.copy()
            s_adj[0] -= p.beta_r * r_lag[t]

            d_t = np.zeros(3)
            if import_price_growth is not None:
                d_t[0] = p.rho_m * import_price_growth[t]
            return Z, d_t

        # Note: kalman_smoother_tv doesn't handle state prediction adjustments
        # So we need a modified version or keep inline. For now, use inline approach
        # but cleaner structure.

        n_periods = len(y)
        n_states = self.n_states

        s_pred_all = np.zeros((n_periods, n_states))
        s_filt_all = np.zeros((n_periods, n_states))
        P_pred_all = np.zeros((n_periods, n_states, n_states))
        P_filt_all = np.zeros((n_periods, n_states, n_states))

        s = s0.copy()
        P = P0.copy()
        log_lik = 0.0

        for t in range(n_periods):
            s_p = T_mat @ s
            s_p[0] -= p.beta_r * r_lag[t]  # Exogenous effect on state
            P_p = T_mat @ P @ T_mat.T + R_mat @ Q @ R_mat.T

            s_pred_all[t] = s_p
            P_pred_all[t] = P_p

            y_p = Z @ s_p
            if import_price_growth is not None:
                y_p[0] += p.rho_m * import_price_growth[t]

            v = y[t] - y_p
            F = Z @ P_p @ Z.T + H

            try:
                F_inv = linalg.inv(F)
                sign, logdet = np.linalg.slogdet(F)
                if sign > 0:
                    log_lik += -0.5 * (3 * np.log(2 * np.pi) + logdet + v @ F_inv @ v)
            except linalg.LinAlgError:
                F_inv = np.eye(3)

            K = P_p @ Z.T @ F_inv
            s = s_p + K @ v
            P = (np.eye(n_states) - K @ Z) @ P_p

            s_filt_all[t] = s
            P_filt_all[t] = P

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
            "states_filtered": s_filt_all,
            "log_likelihood": log_lik,
            "state_names": ["output_gap", "r_star", "eps_supply", "eps_wage"],
        }


def compute_hlw_log_likelihood(
    y_obs: np.ndarray,  # [π, π_w, u_gap] (T × 3)
    r_lag: np.ndarray,  # Lagged real rate (T,)
    params: HLWParameters,
    import_price_growth: np.ndarray | None = None,  # Import price growth (T,)
) -> float:
    """Compute log-likelihood for HLW model."""
    try:
        model = HLWModel(params=params)
        return model.kalman_filter_with_exog(y_obs, r_lag, import_price_growth)
    except Exception:
        return -1e10


# Parameter bounds
HLW_PARAM_BOUNDS = {
    "rho_y": (0.3, 0.99),
    "beta_r": (0.01, 0.5),
    "kappa_p": (0.01, 0.5),
    "kappa_w": (0.01, 0.5),
    "rho_m": (0.0, 0.2),  # Import price pass-through
    "omega": (0.1, 1.0),
    "sigma_demand": (0.1, 2.0),
    "sigma_supply": (0.01, 2.0),
    "sigma_wage": (0.01, 2.0),
    "sigma_rstar": (0.01, 0.5),
    "sigma_okun": (0.01, 0.5),
}


# =============================================================================
# Data Loading
# =============================================================================


def load_hlw_data(
    start: str = "1984Q1",
    end: str | None = None,
    anchor_inflation: bool = True,
) -> dict:
    """Load data for HLW estimation.

    Returns dict with:
        y: Observations (T × 3): [π, π_w, u_gap]
        r_lag: Lagged real rate (T,)
        import_price_growth: Year-on-year import price growth (T,)
        dates: Period index
    """
    from src.data.abs_loader import load_series
    from src.data.cash_rate import get_cash_rate_qrtly
    from src.data.import_prices import get_import_price_growth_annual
    from src.data.series_specs import CPI_TRIMMED_MEAN_QUARTERLY
    from src.models.dsge.data_loader import load_estimation_data
    from src.models.dsge.shared import ensure_period_index

    # Load base data with 5 observables
    df = load_estimation_data(
        start=start, end=end, n_observables=5, anchor_inflation=anchor_inflation
    )

    # Compute real rate using raw inflation
    inflation_raw = ensure_period_index(load_series(CPI_TRIMMED_MEAN_QUARTERLY).data)
    inflation_annual = ((1 + inflation_raw / 100) ** 4 - 1) * 100
    inflation_aligned = inflation_annual.reindex(df.index)

    cash_rate = get_cash_rate_qrtly().data.reindex(df.index)
    real_rate = cash_rate - inflation_aligned
    real_rate_lag = real_rate.shift(1).fillna(real_rate.iloc[0])

    # Load import price growth
    import_price_growth = ensure_period_index(get_import_price_growth_annual().data)
    import_price_aligned = import_price_growth.reindex(df.index).fillna(0)

    # Build observation matrix
    y = df[["inflation", "wage_inflation", "u_gap"]].to_numpy()

    return {
        "y": y,
        "r_lag": real_rate_lag.to_numpy(),
        "import_price_growth": import_price_aligned.to_numpy(),
        "dates": df.index,
    }


# =============================================================================
# Likelihood Wrapper (for generic estimator)
# =============================================================================


def _hlw_likelihood(params: HLWParameters, data: dict) -> float:
    """Compute log-likelihood for HLW model."""
    return compute_hlw_log_likelihood(
        y_obs=data["y"],
        r_lag=data["r_lag"],
        params=params,
        import_price_growth=data.get("import_price_growth"),
    )


# =============================================================================
# Smoother Wrapper
# =============================================================================


def hlw_extract_states(params: HLWParameters, data: dict) -> dict:
    """Extract smoothed states using Kalman smoother.

    Args:
        params: Model parameters
        data: Data dict with y, r_lag, import_price_growth, dates

    Returns:
        Dict with states DataFrame and log_likelihood

    """
    model = HLWModel(params=params)
    result = model.kalman_smoother(
        y=data["y"],
        r_lag=data["r_lag"],
        import_price_growth=data.get("import_price_growth"),
    )

    # Build DataFrame with states
    states_df = pd.DataFrame(
        result["states"],
        index=data["dates"],
        columns=result["state_names"],
    )

    return {
        "states": states_df,
        "log_likelihood": result["log_likelihood"],
    }


# =============================================================================
# Model Specification
# =============================================================================

from src.models.dsge.estimation import ModelSpec

HLW_SPEC = ModelSpec(
    name="HLW",
    description="r* replaces Taylor rule, interest rate exogenous",
    param_class=HLWParameters,
    param_bounds=HLW_PARAM_BOUNDS,
    estimate_params=[
        "rho_y", "beta_r", "kappa_p", "kappa_w", "rho_m", "omega",
        "sigma_demand", "sigma_supply", "sigma_wage", "sigma_rstar",
    ],
    fixed_params={},
    likelihood_fn=_hlw_likelihood,
    state_extractor_fn=hlw_extract_states,
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

    # Chart setup
    CHART_DIR = Path(__file__).parent.parent.parent.parent / "charts" / "dsge-hlw"
    mg.set_chart_dir(str(CHART_DIR))
    mg.clear_chart_dir()

    print("HLW Model - Two-Stage Estimation")
    print("=" * 60)

    result = estimate_two_stage(HLW_SPEC, load_hlw_data)
    print_single_result(result.estimation_result, HLW_SPEC)

    plot_output_gap(result.states["output_gap"], model_name="HLW Model")
    plot_rstar(result.states["r_star"], model_name="HLW Model")

    print(f"\nCharts saved to: {CHART_DIR}")
