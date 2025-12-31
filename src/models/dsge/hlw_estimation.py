"""Estimation for HLW-style model."""

from dataclasses import dataclass
import numpy as np
import pandas as pd
from scipy import optimize

from src.models.dsge.hlw_model import (
    HLWParameters,
    compute_hlw_log_likelihood,
    HLW_PARAM_BOUNDS,
)
from src.models.dsge.data_loader import load_estimation_data
from src.models.dsge.shared import (
    REGIMES,
    ensure_period_index,
    print_regime_results,
)


@dataclass
class HLWEstimationResult:
    params: HLWParameters
    log_likelihood: float
    convergence: dict
    n_obs: int


def load_hlw_data(
    start: str = "1984Q1",
    end: str | None = None,
    anchor_inflation: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.PeriodIndex]:
    """Load data for HLW estimation.

    Returns:
        y: Observations (T × 3): [π, π_w, u_gap]
        r_lag: Lagged real rate (T,)
        import_price_growth: Year-on-year import price growth (T,)
        dates: Period index
    """
    from src.data.abs_loader import load_series
    from src.data.series_specs import CPI_TRIMMED_MEAN_QUARTERLY
    from src.data.cash_rate import get_cash_rate_qrtly
    from src.data.import_prices import get_import_price_growth_annual

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
    y = df[["inflation", "wage_inflation", "u_gap"]].values

    return y, real_rate_lag.values, import_price_aligned.values, df.index


def estimate_hlw(
    y: np.ndarray,
    r_lag: np.ndarray,
    import_price_growth: np.ndarray | None = None,
    initial_params: HLWParameters | None = None,
) -> HLWEstimationResult:
    """Estimate HLW model via MLE."""

    if initial_params is None:
        initial_params = HLWParameters()

    # Parameters to estimate
    estimate_params = [
        "rho_y", "beta_r", "kappa_p", "kappa_w", "rho_m", "omega",
        "sigma_demand", "sigma_supply", "sigma_wage", "sigma_rstar",
    ]

    # Initial values and bounds
    param_dict = initial_params.to_dict()
    x0 = np.array([param_dict[name] for name in estimate_params])
    bounds = [HLW_PARAM_BOUNDS[name] for name in estimate_params]

    def neg_ll(x):
        for i, name in enumerate(estimate_params):
            param_dict[name] = x[i]
        params = HLWParameters(**param_dict)
        ll = compute_hlw_log_likelihood(y, r_lag, params, import_price_growth)
        return -ll

    result = optimize.minimize(
        neg_ll,
        x0,
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": 500},
    )

    # Extract final parameters
    for i, name in enumerate(estimate_params):
        param_dict[name] = result.x[i]
    final_params = HLWParameters(**param_dict)

    return HLWEstimationResult(
        params=final_params,
        log_likelihood=-result.fun,
        convergence={"success": result.success, "nit": result.nit, "nfev": result.nfev},
        n_obs=len(y),
    )


def estimate_hlw_regimes(
    start: str = "1984Q1",
    anchor_inflation: bool = True,
) -> dict:
    """Estimate HLW model separately for each regime."""
    results = {}
    total_ll = 0.0

    for name, reg_start, reg_end in REGIMES:
        # Use max of start and reg_start
        actual_start = max(start, reg_start)

        y, r_lag, import_prices, dates = load_hlw_data(
            start=actual_start, end=reg_end, anchor_inflation=anchor_inflation
        )

        if len(y) < 10:
            print(f"  {name}: insufficient data ({len(y)} obs), skipping")
            continue

        print(f"  Estimating {name} ({dates[0]} to {dates[-1]}, n={len(y)})...")

        result = estimate_hlw(y, r_lag, import_prices)
        results[name] = result
        total_ll += result.log_likelihood

        print(f"    LL: {result.log_likelihood:.2f}")

    results["total_ll"] = total_ll
    return results


PARAMS_TO_SHOW = ["kappa_p", "kappa_w", "rho_m", "omega", "rho_y", "beta_r", "sigma_rstar"]


def print_hlw_results(results: dict) -> None:
    """Print HLW estimation results."""
    print_regime_results(
        results,
        model_name="HLW MODEL",
        model_desc="r* replaces Taylor rule, interest rate exogenous",
        params_to_show=PARAMS_TO_SHOW,
    )


if __name__ == "__main__":
    print("Loading data and estimating HLW model...")
    results = estimate_hlw_regimes(start="1984Q1", anchor_inflation=True)
    print_hlw_results(results)
