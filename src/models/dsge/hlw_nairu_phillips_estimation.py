"""Estimation for combined HLW + NAIRU-Phillips model."""

from dataclasses import dataclass
import numpy as np
import pandas as pd
from scipy import optimize

from src.models.dsge.hlw_nairu_phillips_model import (
    HLWNairuPhillipsParameters,
    compute_hlw_nairu_phillips_log_likelihood,
    extract_latent_estimates,
    HLW_NAIRU_PHILLIPS_PARAM_BOUNDS,
)
from src.models.dsge.data_loader import compute_inflation_anchor
from src.models.dsge.shared import (
    REGIMES,
    ensure_period_index,
    filter_date_range,
    print_regime_results,
)


PARAMS_TO_SHOW = [
    "gamma_p", "gamma_w", "rho_m", "lambda_w",
    "xi_oil", "xi_coal", "sigma_rstar", "sigma_nairu", "sigma_p", "sigma_w",
]


@dataclass
class HLWNairuPhillipsEstimationResult:
    params: HLWNairuPhillipsParameters
    log_likelihood: float
    convergence: dict
    n_obs: int


def load_hlw_nairu_phillips_data(
    start: str = "1984Q1",
    end: str | None = None,
    anchor_inflation: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, pd.PeriodIndex]:
    """Load data for combined HLW + NAIRU-Phillips estimation.

    Returns:
        y: Observations (T × 3): [π, Δulc, U]
        r_lag: Lagged real interest rate (T,)
        import_price_growth: Year-on-year import price growth (T,)
        delta_U_over_U: Speed limit term (T,)
        oil_change: Oil price change (T,)
        coal_change: Coal price change (T,)
        dates: Period index
    """
    from src.data.abs_loader import load_series
    from src.data.series_specs import CPI_TRIMMED_MEAN_QUARTERLY, UNEMPLOYMENT_RATE
    from src.data.cash_rate import get_cash_rate_qrtly
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

    # Build observation matrix: [π, Δulc] - U is separate (exogenous)
    y = df[["inflation", "ulc_growth"]].values

    return (
        y,
        df["U"].values,  # U is exogenous, passed separately
        df["real_rate_lag"].values,
        df["import_price_growth"].values,
        df["delta_U_over_U"].values,
        df["oil_change"].values,
        df["coal_change"].values,
        df.index,
    )


def estimate_hlw_nairu_phillips(
    y: np.ndarray,
    U_obs: np.ndarray,
    r_lag: np.ndarray,
    import_price_growth: np.ndarray | None = None,
    delta_U_over_U: np.ndarray | None = None,
    oil_change: np.ndarray | None = None,
    coal_change: np.ndarray | None = None,
    initial_params: HLWNairuPhillipsParameters | None = None,
    rstar_prior: float = 1.0,
    nairu_prior: float = 5.0,
) -> HLWNairuPhillipsEstimationResult:
    """Estimate combined model via MLE."""
    if initial_params is None:
        initial_params = HLWNairuPhillipsParameters()

    param_dict = initial_params.to_dict()

    # Parameters to estimate (no beta_r or sigma_okun - U not observed)
    estimate_params = [
        "gamma_p", "gamma_w", "rho_m", "lambda_w",
        "xi_oil", "xi_coal", "sigma_p", "sigma_w",
    ]
    # Fix sigma_rstar and sigma_nairu for identification
    param_dict["sigma_rstar"] = 0.15
    param_dict["sigma_nairu"] = 0.10  # Tight to keep NAIRU near prior

    x0 = np.array([param_dict[name] for name in estimate_params])
    bounds = [HLW_NAIRU_PHILLIPS_PARAM_BOUNDS[name] for name in estimate_params]

    def neg_ll(x):
        for i, name in enumerate(estimate_params):
            param_dict[name] = x[i]
        params = HLWNairuPhillipsParameters(**param_dict)
        return -compute_hlw_nairu_phillips_log_likelihood(
            y, U_obs, r_lag, params, import_price_growth, delta_U_over_U,
            oil_change, coal_change, rstar_prior, nairu_prior
        )

    result = optimize.minimize(
        neg_ll, x0, method="L-BFGS-B", bounds=bounds, options={"maxiter": 500}
    )

    for i, name in enumerate(estimate_params):
        param_dict[name] = result.x[i]

    return HLWNairuPhillipsEstimationResult(
        params=HLWNairuPhillipsParameters(**param_dict),
        log_likelihood=-result.fun,
        convergence={"success": result.success, "nit": result.nit, "nfev": result.nfev},
        n_obs=len(y),
    )


def estimate_hlw_nairu_phillips_regimes(
    start: str = "1984Q1",
    anchor_inflation: bool = True,
) -> dict:
    """Estimate combined model separately for each regime."""
    results = {}
    total_ll = 0.0

    for name, reg_start, reg_end in REGIMES:
        actual_start = max(start, reg_start)
        data = load_hlw_nairu_phillips_data(start=actual_start, end=reg_end, anchor_inflation=anchor_inflation)
        y, U_obs, r_lag, import_prices, delta_U_over_U, oil_change, coal_change, dates = data

        if len(y) < 10:
            print(f"  {name}: insufficient data ({len(y)} obs), skipping")
            continue

        # Use regime mean for r*, but lower fixed prior for NAIRU
        rstar_prior = np.mean(r_lag)
        nairu_prior = 5.0  # Fixed lower prior (vs mean(U) which is too high)

        print(f"  Estimating {name} ({dates[0]} to {dates[-1]}, n={len(y)}, r̄={rstar_prior:.1f}%, NAIRU prior={nairu_prior:.1f}%)...")

        result = estimate_hlw_nairu_phillips(
            y, U_obs, r_lag, import_prices, delta_U_over_U, oil_change, coal_change,
            rstar_prior=rstar_prior, nairu_prior=nairu_prior,
        )
        results[name] = result
        total_ll += result.log_likelihood
        print(f"    LL: {result.log_likelihood:.2f}")
        print(f"    γ_p: {result.params.gamma_p:.3f}, γ_w: {result.params.gamma_w:.3f}, ρ_m: {result.params.rho_m:.3f}")

    results["total_ll"] = total_ll
    return results


def print_hlw_nairu_phillips_results(results: dict) -> None:
    """Print combined model estimation results."""
    print_regime_results(
        results,
        model_name="HLW-NAIRU-PHILLIPS MODEL",
        model_desc="Combined r* + NAIRU, Okun-IS hybrid, normalised u-gap",
        params_to_show=PARAMS_TO_SHOW,
    )


def estimate_and_extract_latents(
    start: str = "1984Q1",
    anchor_inflation: bool = True,
) -> tuple[dict, pd.DataFrame]:
    """Estimate by regime and extract r* and NAIRU paths.

    Returns:
        results: Dict of regime estimation results
        latents_df: DataFrame with r*, NAIRU estimates and uncertainty
    """
    results = {}
    total_ll = 0.0
    records = []

    for name, reg_start, reg_end in REGIMES:
        actual_start = max(start, reg_start)
        data = load_hlw_nairu_phillips_data(start=actual_start, end=reg_end, anchor_inflation=anchor_inflation)
        y, U_obs, r_lag, import_prices, delta_U_over_U, oil_change, coal_change, dates = data

        if len(y) < 10:
            print(f"  {name}: insufficient data ({len(y)} obs), skipping")
            continue

        rstar_prior = np.mean(r_lag)
        nairu_prior = 5.0  # Fixed lower prior

        print(f"  Estimating {name} ({dates[0]} to {dates[-1]}, n={len(y)}, r̄={rstar_prior:.1f}%, NAIRU prior={nairu_prior:.1f}%)...")

        result = estimate_hlw_nairu_phillips(
            y, U_obs, r_lag, import_prices, delta_U_over_U, oil_change, coal_change,
            rstar_prior=rstar_prior, nairu_prior=nairu_prior,
        )
        results[name] = result
        total_ll += result.log_likelihood

        print(f"    LL: {result.log_likelihood:.2f}")
        print(f"    γ_p: {result.params.gamma_p:.3f}, γ_w: {result.params.gamma_w:.3f}, ρ_m: {result.params.rho_m:.3f}")

        # Extract latent estimates
        rstar, rstar_std, nairu, nairu_std = extract_latent_estimates(
            y, U_obs, r_lag, result.params, import_prices, delta_U_over_U,
            oil_change, coal_change, rstar_prior, nairu_prior
        )

        for i, date in enumerate(dates):
            records.append({
                "date": date,
                "regime": name,
                "rstar": rstar[i],
                "rstar_std": rstar_std[i],
                "nairu": nairu[i],
                "nairu_std": nairu_std[i],
                "r": r_lag[i],
                "U": U_obs[i],
            })

    results["total_ll"] = total_ll
    latents_df = pd.DataFrame(records).set_index("date").sort_index()

    return results, latents_df


if __name__ == "__main__":
    print("Loading data and estimating HLW-NAIRU-Phillips model by regime...")
    results, latents_df = estimate_and_extract_latents(start="1984Q1", anchor_inflation=True)
    print_hlw_nairu_phillips_results(results)

    print(f"\nLatent paths extracted: {len(latents_df)} observations")
    print(f"Current r*: {latents_df['rstar'].iloc[-1]:.2f}%")
    print(f"Current NAIRU: {latents_df['nairu'].iloc[-1]:.2f}%")
    print(f"Current r-r*: {latents_df['r'].iloc[-1] - latents_df['rstar'].iloc[-1]:.2f}pp")
    print(f"Current U-gap: {latents_df['U'].iloc[-1] - latents_df['nairu'].iloc[-1]:.2f}pp")
