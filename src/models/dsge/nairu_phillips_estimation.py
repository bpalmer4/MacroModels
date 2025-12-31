"""Estimation for pure NAIRU-Phillips model."""

from dataclasses import dataclass
import numpy as np
import pandas as pd
from scipy import optimize

from src.models.dsge.nairu_phillips_model import (
    NairuPhillipsParameters,
    compute_nairu_phillips_log_likelihood,
    extract_nairu_estimates,
    NAIRU_PHILLIPS_PARAM_BOUNDS,
)
from src.models.dsge.data_loader import compute_inflation_anchor
from src.models.dsge.shared import (
    REGIMES,
    ensure_period_index,
    filter_date_range,
    print_regime_results,
)


# Parameters to show in results
PARAMS_TO_SHOW = [
    "gamma_p", "gamma_w", "rho_m", "lambda_w",
    "xi_oil", "xi_coal", "sigma_nairu", "sigma_p", "sigma_w",
]


@dataclass
class NairuPhillipsEstimationResult:
    params: NairuPhillipsParameters
    log_likelihood: float
    convergence: dict
    n_obs: int


def load_nairu_phillips_data(
    start: str = "1984Q1",
    end: str | None = None,
    anchor_inflation: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, pd.PeriodIndex]:
    """Load data for NAIRU-Phillips estimation.

    Returns:
        y: Observations (T × 2): [π, π_w]
        U_obs: Unemployment rate (T,)
        import_price_growth: Year-on-year import price growth (T,)
        delta_U_over_U: Speed limit term (U_{t-1} - U_{t-2})/U_{t-1} (T,)
        oil_change: Oil price change (annual %, lagged 1Q) (T,)
        coal_change: Coal price change (annual %) (T,)
        dates: Period index
    """
    from src.data.abs_loader import load_series
    from src.data.series_specs import CPI_TRIMMED_MEAN_QUARTERLY, UNEMPLOYMENT_RATE
    from src.data.import_prices import get_import_price_growth_annual
    from src.data.ulc import get_ulc_growth_qrtly
    from src.data.energy import get_oil_change_lagged_annual, get_coal_change_annual

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

    # Build observation matrix
    y = df[["inflation", "ulc_growth"]].values

    return (
        y,
        df["U"].values,
        df["import_price_growth"].values,
        df["delta_U_over_U"].values,
        df["oil_change"].values,
        df["coal_change"].values,
        df.index,
    )


def estimate_nairu_phillips(
    y: np.ndarray,
    U_obs: np.ndarray,
    import_price_growth: np.ndarray | None = None,
    delta_U_over_U: np.ndarray | None = None,
    oil_change: np.ndarray | None = None,
    coal_change: np.ndarray | None = None,
    initial_params: NairuPhillipsParameters | None = None,
    nairu_prior: float = 5.0,
) -> NairuPhillipsEstimationResult:
    """Estimate NAIRU-Phillips model via MLE."""
    if initial_params is None:
        initial_params = NairuPhillipsParameters()

    param_dict = initial_params.to_dict()

    # Parameters to estimate (sigma_nairu fixed for identification)
    estimate_params = [
        "gamma_p", "gamma_w", "rho_m", "lambda_w", "xi_oil", "xi_coal",
        "sigma_p", "sigma_w",
    ]
    param_dict["sigma_nairu"] = 0.15

    x0 = np.array([param_dict[name] for name in estimate_params])
    bounds = [NAIRU_PHILLIPS_PARAM_BOUNDS[name] for name in estimate_params]

    def neg_ll(x):
        for i, name in enumerate(estimate_params):
            param_dict[name] = x[i]
        params = NairuPhillipsParameters(**param_dict)
        return -compute_nairu_phillips_log_likelihood(
            y, U_obs, params, import_price_growth, delta_U_over_U,
            oil_change, coal_change, nairu_prior
        )

    result = optimize.minimize(
        neg_ll, x0, method="L-BFGS-B", bounds=bounds, options={"maxiter": 500}
    )

    for i, name in enumerate(estimate_params):
        param_dict[name] = result.x[i]

    return NairuPhillipsEstimationResult(
        params=NairuPhillipsParameters(**param_dict),
        log_likelihood=-result.fun,
        convergence={"success": result.success, "nit": result.nit, "nfev": result.nfev},
        n_obs=len(y),
    )


def estimate_nairu_phillips_regimes(
    start: str = "1984Q1",
    anchor_inflation: bool = True,
) -> dict:
    """Estimate NAIRU-Phillips model separately for each regime."""
    results = {}
    total_ll = 0.0

    for name, reg_start, reg_end in REGIMES:
        actual_start = max(start, reg_start)
        data = load_nairu_phillips_data(start=actual_start, end=reg_end, anchor_inflation=anchor_inflation)
        y, U_obs, import_prices, delta_U_over_U, oil_change, coal_change, dates = data

        if len(y) < 10:
            print(f"  {name}: insufficient data ({len(y)} obs), skipping")
            continue

        regime_nairu_prior = np.mean(U_obs)
        print(f"  Estimating {name} ({dates[0]} to {dates[-1]}, n={len(y)}, U̅={regime_nairu_prior:.1f}%)...")

        result = estimate_nairu_phillips(
            y, U_obs, import_prices, delta_U_over_U, oil_change, coal_change,
            nairu_prior=regime_nairu_prior,
        )
        results[name] = result
        total_ll += result.log_likelihood
        print(f"    LL: {result.log_likelihood:.2f}")

    results["total_ll"] = total_ll
    return results


def print_nairu_phillips_results(results: dict) -> None:
    """Print NAIRU-Phillips estimation results."""
    print_regime_results(
        results,
        model_name="NAIRU-PHILLIPS MODEL",
        model_desc="Pure Phillips curve, normalised u_gap, ULC for wages",
        params_to_show=PARAMS_TO_SHOW,
    )


def estimate_and_extract_nairu_regimes(
    start: str = "1984Q1",
    anchor_inflation: bool = True,
) -> tuple[dict, pd.DataFrame]:
    """Estimate NAIRU-Phillips by regime and extract combined NAIRU path.

    Returns:
        results: Dict of regime estimation results
        nairu_df: DataFrame with NAIRU estimates and uncertainty by regime
    """
    results = {}
    total_ll = 0.0
    nairu_records = []

    for name, reg_start, reg_end in REGIMES:
        actual_start = max(start, reg_start)
        data = load_nairu_phillips_data(start=actual_start, end=reg_end, anchor_inflation=anchor_inflation)
        y, U_obs, import_prices, delta_U_over_U, oil_change, coal_change, dates = data

        if len(y) < 10:
            print(f"  {name}: insufficient data ({len(y)} obs), skipping")
            continue

        regime_nairu_prior = np.mean(U_obs)
        print(f"  Estimating {name} ({dates[0]} to {dates[-1]}, n={len(y)}, U̅={regime_nairu_prior:.1f}%)...")

        result = estimate_nairu_phillips(
            y, U_obs, import_prices, delta_U_over_U, oil_change, coal_change,
            nairu_prior=regime_nairu_prior,
        )
        results[name] = result
        total_ll += result.log_likelihood

        print(f"    LL: {result.log_likelihood:.2f}")
        print(f"    γ_p: {result.params.gamma_p:.3f}, γ_w: {result.params.gamma_w:.3f}, ρ_m: {result.params.rho_m:.3f}")
        print(f"    λ_w: {result.params.lambda_w:.3f}, ξ_oil: {result.params.xi_oil:.4f}, ξ_coal: {result.params.xi_coal:.4f}")

        # Extract NAIRU for this regime
        nairu, nairu_std = extract_nairu_estimates(
            y, U_obs, result.params, import_prices, delta_U_over_U,
            oil_change, coal_change, regime_nairu_prior
        )

        for i, date in enumerate(dates):
            nairu_records.append({
                "date": date,
                "regime": name,
                "nairu": nairu[i],
                "nairu_std": nairu_std[i],
                "U": U_obs[i],
            })

    results["total_ll"] = total_ll
    nairu_df = pd.DataFrame(nairu_records).set_index("date").sort_index()

    return results, nairu_df


if __name__ == "__main__":
    print("Loading data and estimating NAIRU-Phillips model by regime...")
    results, nairu_df = estimate_and_extract_nairu_regimes(start="1984Q1", anchor_inflation=True)
    print_nairu_phillips_results(results)

    print(f"\nNAIRU path extracted: {len(nairu_df)} observations")
    print(f"Current NAIRU: {nairu_df['nairu'].iloc[-1]:.2f}%")
    print(f"Current U-gap: {nairu_df['U'].iloc[-1] - nairu_df['nairu'].iloc[-1]:.2f}pp")
