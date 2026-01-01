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

Usage:
    from nairu_phillips_model import NAIRU_PHILLIPS_SPEC, load_nairu_phillips_data
    from estimation import estimate_two_stage

    result = estimate_two_stage(NAIRU_PHILLIPS_SPEC, load_nairu_phillips_data)
"""

from dataclasses import dataclass, field
import numpy as np
import pandas as pd
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
    """Extract NAIRU estimates using generic Kalman smoother.

    Returns:
        nairu: Smoothed (or filtered) NAIRU estimates (T,)
        nairu_std: Standard deviation of NAIRU estimates (T,)
    """
    from src.models.dsge.kalman import kalman_smoother_tv, kalman_filter

    p = params

    # State transition (time-invariant)
    T_mat = np.array([
        [1, 0, 0],  # NAIRU: random walk
        [0, 0, 0],  # ε_p: iid
        [0, 0, 0],  # ε_w: iid
    ])
    R_mat = np.diag([p.sigma_nairu, p.sigma_p, p.sigma_w])
    Q = np.eye(3)
    H = np.zeros((2, 2))

    # Initial state
    s0 = np.array([nairu_prior, 0.0, 0.0])
    P0 = np.eye(3)
    P0[0, 0] = 4.0  # Diffuse prior for NAIRU

    # Build time-varying observation equation
    def build_obs_eq(t, s_pred):
        U_t = U_obs[t]
        Z_t = np.array([
            [-p.gamma_p / U_t, 1, 0],
            [-p.gamma_w / U_t, 0, 1],
        ])
        d_t = np.array([p.gamma_p, p.gamma_w])
        if import_price_growth is not None:
            d_t[0] += p.rho_m * import_price_growth[t]
        if delta_U_over_U is not None:
            d_t[1] += p.lambda_w * delta_U_over_U[t]
        if oil_change is not None:
            d_t[0] += p.xi_oil * oil_change[t]
        if coal_change is not None:
            d_t[0] += p.xi_coal * coal_change[t]
        return Z_t, d_t

    result = kalman_smoother_tv(y_obs, T_mat, R_mat, Q, build_obs_eq, H, s0, P0)

    if smooth:
        return result.smoothed_states[:, 0], np.sqrt(result.smoothed_covs[:, 0, 0])
    else:
        return result.filtered_states[:, 0], np.sqrt(result.filtered_covs[:, 0, 0])


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


# =============================================================================
# Data Loading
# =============================================================================


def load_nairu_phillips_data(
    start: str = "1984Q1",
    end: str | None = None,
    anchor_inflation: bool = True,
) -> dict:
    """Load data for NAIRU-Phillips estimation.

    Returns dict with:
        y: Observations (T × 2): [π, π_w]
        U_obs: Unemployment rate (T,)
        import_price_growth: Year-on-year import price growth (T,)
        delta_U_over_U: Speed limit term (U_{t-1} - U_{t-2})/U_{t-1} (T,)
        oil_change: Oil price change (annual %, lagged 1Q) (T,)
        coal_change: Coal price change (annual %) (T,)
        nairu_prior: Prior mean for NAIRU (regime average U)
        dates: Period index
    """
    from src.data.abs_loader import load_series
    from src.data.series_specs import CPI_TRIMMED_MEAN_QUARTERLY, UNEMPLOYMENT_RATE
    from src.data.import_prices import get_import_price_growth_annual
    from src.data.ulc import get_ulc_growth_qrtly
    from src.data.energy import get_oil_change_lagged_annual, get_coal_change_annual
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

    # Load ULC growth
    ulc_growth = ensure_period_index(get_ulc_growth_qrtly().data)

    # Load unemployment rate
    ur = ensure_period_index(load_series(UNEMPLOYMENT_RATE).data)
    if hasattr(ur.index, "freqstr") and ur.index.freqstr == "M":
        ur = ur.resample("Q").mean()
        ur.index = pd.PeriodIndex(ur.index, freq="Q")

    # Load import price growth
    import_price_growth = ensure_period_index(get_import_price_growth_annual().data)

    # Load energy prices
    oil_change = ensure_period_index(get_oil_change_lagged_annual().data)
    coal_change = ensure_period_index(get_coal_change_annual().data).shift(1)

    # Build DataFrame for alignment
    df = pd.DataFrame({
        "inflation": inflation,
        "ulc_growth": ulc_growth,
        "U": ur,
        "import_price_growth": import_price_growth,
        "oil_change": oil_change,
        "coal_change": coal_change,
    })

    # Compute speed limit: (U_{t-1} - U_{t-2})/U_{t-1}
    U_lag1 = df["U"].shift(1)
    U_lag2 = df["U"].shift(2)
    df["delta_U_over_U"] = (U_lag1 - U_lag2) / U_lag1

    # Drop NaNs and filter to date range
    df = filter_date_range(df.dropna(), start, end)

    return {
        "y": df[["inflation", "ulc_growth"]].values,
        "U_obs": df["U"].values,
        "import_price_growth": df["import_price_growth"].values,
        "delta_U_over_U": df["delta_U_over_U"].values,
        "oil_change": df["oil_change"].values,
        "coal_change": df["coal_change"].values,
        "nairu_prior": float(np.mean(df["U"].values)),
        "dates": df.index,
    }


# =============================================================================
# Likelihood Wrapper (for generic estimator)
# =============================================================================


def _nairu_phillips_likelihood(params: NairuPhillipsParameters, data: dict) -> float:
    """Compute log-likelihood for NAIRU-Phillips model.

    Args:
        params: Model parameters
        data: Dict from load_nairu_phillips_data

    Returns:
        Log-likelihood value
    """
    return compute_nairu_phillips_log_likelihood(
        y_obs=data["y"],
        U_obs=data["U_obs"],
        params=params,
        import_price_growth=data.get("import_price_growth"),
        delta_U_over_U=data.get("delta_U_over_U"),
        oil_change=data.get("oil_change"),
        coal_change=data.get("coal_change"),
        nairu_prior=data.get("nairu_prior", 5.0),
    )


# =============================================================================
# State Extractor (for two-stage estimation)
# =============================================================================


def nairu_phillips_extract_states(params: NairuPhillipsParameters, data: dict) -> dict:
    """Extract smoothed NAIRU using Kalman smoother.

    Args:
        params: Model parameters
        data: Data dict with y, U_obs, etc.

    Returns:
        Dict with states DataFrame and log_likelihood
    """
    nairu, nairu_std = extract_nairu_estimates(
        y_obs=data["y"],
        U_obs=data["U_obs"],
        params=params,
        import_price_growth=data.get("import_price_growth"),
        delta_U_over_U=data.get("delta_U_over_U"),
        oil_change=data.get("oil_change"),
        coal_change=data.get("coal_change"),
        nairu_prior=data.get("nairu_prior", 5.0),
        smooth=True,
    )

    states_df = pd.DataFrame({
        "nairu": nairu,
        "nairu_std": nairu_std,
        "unemployment": data["U_obs"],
    }, index=data["dates"])

    return {
        "states": states_df,
        "log_likelihood": _nairu_phillips_likelihood(params, data),
    }


# =============================================================================
# Model Specification
# =============================================================================

# Import here to avoid circular imports
from src.models.dsge.estimation import ModelSpec

NAIRU_PHILLIPS_SPEC = ModelSpec(
    name="NAIRU-Phillips",
    description="Pure Phillips curve, normalised u_gap, ULC for wages",
    param_class=NairuPhillipsParameters,
    param_bounds=NAIRU_PHILLIPS_PARAM_BOUNDS,
    estimate_params=[
        "gamma_p", "gamma_w", "rho_m", "lambda_w", "xi_oil", "xi_coal",
        "sigma_p", "sigma_w",
    ],
    fixed_params={"sigma_nairu": 0.15},
    likelihood_fn=_nairu_phillips_likelihood,
    state_extractor_fn=nairu_phillips_extract_states,
)


# =============================================================================
# Main
# =============================================================================


if __name__ == "__main__":
    from pathlib import Path
    import mgplot as mg
    from src.models.dsge.estimation import estimate_two_stage, print_single_result
    from src.models.dsge.plot_nairu import plot_nairu

    # Chart setup
    CHART_DIR = Path(__file__).parent.parent.parent.parent / "charts" / "dsge-nairu-phillips"
    mg.set_chart_dir(str(CHART_DIR))
    mg.clear_chart_dir()

    print("NAIRU-Phillips Model - Two-Stage Estimation")
    print("=" * 60)

    result = estimate_two_stage(NAIRU_PHILLIPS_SPEC, load_nairu_phillips_data)
    print_single_result(result.estimation_result, NAIRU_PHILLIPS_SPEC)

    # Plot NAIRU with unemployment overlay
    plot_nairu(
        result.states["nairu"],
        unemployment=result.states["unemployment"],
        model_name="NAIRU-Phillips",
    )

    print(f"\nCharts saved to: {CHART_DIR}")
