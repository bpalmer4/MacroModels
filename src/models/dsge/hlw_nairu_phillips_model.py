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

Usage:
    from hlw_nairu_phillips_model import HLW_NAIRU_PHILLIPS_SPEC, load_hlw_nairu_phillips_data
    from estimation import estimate_two_stage

    result = estimate_two_stage(HLW_NAIRU_PHILLIPS_SPEC, load_hlw_nairu_phillips_data)
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
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
    """Extract r* and NAIRU estimates using generic Kalman smoother.

    Returns:
        rstar: r* estimates (T,)
        rstar_std: Standard deviation of r* (T,)
        nairu: NAIRU estimates (T,)
        nairu_std: Standard deviation of NAIRU (T,)

    """
    from src.models.dsge.kalman import kalman_smoother_tv

    p = params
    n_states = 4

    # State transition (time-invariant)
    T_mat = np.array([
        [1, 0, 0, 0],  # r*: random walk
        [0, 1, 0, 0],  # NAIRU: random walk
        [0, 0, 0, 0],  # ε_p: iid
        [0, 0, 0, 0],  # ε_w: iid
    ])
    R_mat = np.diag([p.sigma_rstar, p.sigma_nairu, p.sigma_p, p.sigma_w])
    Q = np.eye(4)
    H = np.zeros((2, 2))

    # Initial state
    s0 = np.array([rstar_prior, nairu_prior, 0.0, 0.0])
    P0 = np.eye(n_states)
    P0[0, 0] = 1.0   # r* prior uncertainty
    P0[1, 1] = 0.25  # NAIRU prior uncertainty - tighter

    # Build time-varying observation equation
    def build_obs_eq(t, s_pred):
        U_t = U_obs[t]
        Z_t = np.array([
            [0, -p.gamma_p / U_t, 1, 0],
            [0, -p.gamma_w / U_t, 0, 1],
        ])
        d_t = np.array([p.gamma_p, p.gamma_w])
        if import_price_growth is not None:
            d_t[0] += p.rho_m * import_price_growth[t]
        if oil_change is not None:
            d_t[0] += p.xi_oil * oil_change[t]
        if coal_change is not None:
            d_t[0] += p.xi_coal * coal_change[t]
        if delta_U_over_U is not None:
            d_t[1] += p.lambda_w * delta_U_over_U[t]
        return Z_t, d_t

    result = kalman_smoother_tv(y_obs, T_mat, R_mat, Q, build_obs_eq, H, s0, P0)

    if smooth:
        return (
            result.smoothed_states[:, 0], np.sqrt(result.smoothed_covs[:, 0, 0]),
            result.smoothed_states[:, 1], np.sqrt(result.smoothed_covs[:, 1, 1]),
        )
    return (
        result.filtered_states[:, 0], np.sqrt(result.filtered_covs[:, 0, 0]),
        result.filtered_states[:, 1], np.sqrt(result.filtered_covs[:, 1, 1]),
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


# =============================================================================
# Data Loading
# =============================================================================


def load_hlw_nairu_phillips_data(
    start: str = "1984Q1",
    end: str | None = None,
    anchor_inflation: bool = True,
) -> dict:
    """Load data for combined HLW + NAIRU-Phillips estimation.

    Returns dict with:
        y: Observations (T × 2): [π, Δulc]
        U_obs: Unemployment rate (T,)
        r_lag: Lagged real interest rate (T,)
        import_price_growth: Year-on-year import price growth (T,)
        delta_U_over_U: Speed limit term (T,)
        oil_change: Oil price change (T,)
        coal_change: Coal price change (T,)
        rstar_prior: Prior mean for r* (regime average real rate)
        nairu_prior: Prior mean for NAIRU (fixed at 5.0)
        dates: Period index
    """
    from src.data.abs_loader import load_series
    from src.data.cash_rate import get_cash_rate_qrtly
    from src.data.energy import get_coal_change_annual, get_oil_change_lagged_annual
    from src.data.import_prices import get_import_price_growth_annual
    from src.data.series_specs import CPI_TRIMMED_MEAN_QUARTERLY, UNEMPLOYMENT_RATE
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

    # Load ULC growth
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

    # Load energy prices
    oil_change = ensure_period_index(get_oil_change_lagged_annual().data)
    coal_change = ensure_period_index(get_coal_change_annual().data).shift(1)

    # Build DataFrame for alignment
    df = pd.DataFrame({
        "inflation": inflation,
        "ulc_growth": ulc_growth,
        "U": ur,
        "cash_rate": cash_rate,
        "inflation_raw": inflation_annual,
        "import_price_growth": import_price_growth,
        "oil_change": oil_change,
        "coal_change": coal_change,
    })

    # Compute speed limit
    U_lag1 = df["U"].shift(1)
    U_lag2 = df["U"].shift(2)
    df["delta_U_over_U"] = (U_lag1 - U_lag2) / U_lag1

    # Compute real rate and lag it
    df["real_rate"] = df["cash_rate"] - df["inflation_raw"]
    df["real_rate_lag"] = df["real_rate"].shift(1)

    # Drop NaNs and filter to date range
    df = filter_date_range(df.dropna(), start, end)

    return {
        "y": df[["inflation", "ulc_growth"]].to_numpy(),
        "U_obs": df["U"].to_numpy(),
        "r_lag": df["real_rate_lag"].to_numpy(),
        "import_price_growth": df["import_price_growth"].to_numpy(),
        "delta_U_over_U": df["delta_U_over_U"].to_numpy(),
        "oil_change": df["oil_change"].to_numpy(),
        "coal_change": df["coal_change"].to_numpy(),
        "rstar_prior": float(np.mean(df["real_rate_lag"].to_numpy())),
        "nairu_prior": 5.0,  # Fixed lower prior
        "dates": df.index,
    }


# =============================================================================
# Likelihood Wrapper (for generic estimator)
# =============================================================================


def _hlw_nairu_phillips_likelihood(params: HLWNairuPhillipsParameters, data: dict) -> float:
    """Compute log-likelihood for combined model."""
    return compute_hlw_nairu_phillips_log_likelihood(
        y_obs=data["y"],
        U_obs=data["U_obs"],
        r_lag=data["r_lag"],
        params=params,
        import_price_growth=data.get("import_price_growth"),
        delta_U_over_U=data.get("delta_U_over_U"),
        oil_change=data.get("oil_change"),
        coal_change=data.get("coal_change"),
        rstar_prior=data.get("rstar_prior", 1.0),
        nairu_prior=data.get("nairu_prior", 5.0),
    )


# =============================================================================
# State Extractor (for two-stage estimation)
# =============================================================================


def hlw_nairu_phillips_extract_states(params: HLWNairuPhillipsParameters, data: dict) -> dict:
    """Extract smoothed r* and NAIRU using Kalman smoother.

    Args:
        params: Model parameters
        data: Data dict

    Returns:
        Dict with states DataFrame and log_likelihood

    """
    rstar, rstar_std, nairu, nairu_std = extract_latent_estimates(
        y_obs=data["y"],
        U_obs=data["U_obs"],
        r_lag=data["r_lag"],
        params=params,
        import_price_growth=data.get("import_price_growth"),
        delta_U_over_U=data.get("delta_U_over_U"),
        oil_change=data.get("oil_change"),
        coal_change=data.get("coal_change"),
        rstar_prior=data.get("rstar_prior", 1.0),
        nairu_prior=data.get("nairu_prior", 5.0),
        smooth=True,
    )

    states_df = pd.DataFrame({
        "r_star": rstar,
        "r_star_std": rstar_std,
        "nairu": nairu,
        "nairu_std": nairu_std,
        "unemployment": data["U_obs"],
    }, index=data["dates"])

    return {
        "states": states_df,
        "log_likelihood": _hlw_nairu_phillips_likelihood(params, data),
    }


# =============================================================================
# Model Specification
# =============================================================================

from src.models.dsge.estimation import ModelSpec

HLW_NAIRU_PHILLIPS_SPEC = ModelSpec(
    name="HLW-NAIRU-Phillips",
    description="Combined r* + NAIRU, normalised u-gap, speed limit",
    param_class=HLWNairuPhillipsParameters,
    param_bounds=HLW_NAIRU_PHILLIPS_PARAM_BOUNDS,
    estimate_params=[
        "gamma_p", "gamma_w", "rho_m", "lambda_w",
        "xi_oil", "xi_coal", "sigma_p", "sigma_w",
    ],
    fixed_params={"sigma_rstar": 0.15, "sigma_nairu": 0.10},
    likelihood_fn=_hlw_nairu_phillips_likelihood,
    state_extractor_fn=hlw_nairu_phillips_extract_states,
)


# =============================================================================
# Main
# =============================================================================


if __name__ == "__main__":
    from pathlib import Path

    import mgplot as mg

    from src.models.dsge.estimation import estimate_two_stage, print_single_result
    from src.models.dsge.plot_nairu import plot_nairu
    from src.models.dsge.plot_rstar import plot_rstar

    # Chart setup
    CHART_DIR = Path(__file__).parent.parent.parent.parent / "charts" / "dsge-hlw-nairu-phillips"
    mg.set_chart_dir(str(CHART_DIR))
    mg.clear_chart_dir()

    print("HLW-NAIRU-Phillips Model - Two-Stage Estimation")
    print("=" * 60)

    result = estimate_two_stage(HLW_NAIRU_PHILLIPS_SPEC, load_hlw_nairu_phillips_data)
    print_single_result(result.estimation_result, HLW_NAIRU_PHILLIPS_SPEC)

    plot_rstar(result.states["r_star"], model_name="HLW-NAIRU-Phillips")
    plot_nairu(result.states["nairu"], unemployment=result.states["unemployment"], model_name="HLW-NAIRU-Phillips")

    print(f"\nCharts saved to: {CHART_DIR}")
