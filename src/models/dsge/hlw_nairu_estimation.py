"""Estimation for HLW model with NAIRU as latent state."""

from dataclasses import dataclass
import numpy as np
import pandas as pd
from scipy import optimize

from src.models.dsge.hlw_nairu_model import (
    HLWNairuModel,
    HLWNairuParameters,
    compute_hlw_nairu_log_likelihood,
    HLW_NAIRU_PARAM_BOUNDS,
)
from src.models.dsge.data_loader import compute_inflation_anchor


@dataclass
class HLWNairuEstimationResult:
    params: HLWNairuParameters
    log_likelihood: float
    convergence: dict
    n_obs: int


def load_hlw_nairu_data(
    start: str = "1984Q1",
    end: str | None = None,
    anchor_inflation: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, pd.PeriodIndex]:
    """Load data for HLW-NAIRU estimation.

    Returns:
        y: Observations (T × 3): [π, π_w, U]
        r_lag: Lagged real rate (T,)
        U_obs: Unemployment rate (T,) - same as y[:, 2]
        import_price_growth: Year-on-year import price growth (T,)
        dates: Period index
    """
    from src.data.abs_loader import load_series
    from src.data.series_specs import (
        CPI_TRIMMED_MEAN_QUARTERLY,
        COMPENSATION_OF_EMPLOYEES,
        UNEMPLOYMENT_RATE,
    )
    from src.data.cash_rate import get_cash_rate_qrtly
    from src.data.import_prices import get_import_price_growth_annual

    # Load inflation
    inf_series = load_series(CPI_TRIMMED_MEAN_QUARTERLY)
    inflation_raw = inf_series.data
    if not isinstance(inflation_raw.index, pd.PeriodIndex):
        inflation_raw.index = pd.PeriodIndex(inflation_raw.index, freq="Q")
    inflation_annual = ((1 + inflation_raw / 100) ** 4 - 1) * 100

    # Compute anchor-adjusted inflation if requested
    if anchor_inflation:
        pi_anchor = compute_inflation_anchor(inflation_annual)
        inflation = inflation_annual - pi_anchor
    else:
        inflation = inflation_annual

    # Load wage inflation (Compensation of Employees, YoY growth)
    coe_series = load_series(COMPENSATION_OF_EMPLOYEES)
    coe = coe_series.data
    if not isinstance(coe.index, pd.PeriodIndex):
        coe.index = pd.PeriodIndex(coe.index, freq="Q")
    wage_inflation = coe.pct_change(4) * 100

    # Load unemployment rate (raw, not HP-filtered)
    ur_series = load_series(UNEMPLOYMENT_RATE)
    ur = ur_series.data
    if not isinstance(ur.index, pd.PeriodIndex):
        ur.index = pd.PeriodIndex(ur.index, freq="Q")
    # Convert monthly to quarterly if needed
    if hasattr(ur.index, 'freqstr') and ur.index.freqstr == "M":
        ur = ur.resample("Q").mean()
        ur.index = pd.PeriodIndex(ur.index, freq="Q")

    # Load cash rate for real rate
    cash_rate = get_cash_rate_qrtly().data
    if not isinstance(cash_rate.index, pd.PeriodIndex):
        cash_rate.index = pd.PeriodIndex(cash_rate.index, freq="Q")

    # Load import price growth
    import_price_series = get_import_price_growth_annual()
    import_price_growth = import_price_series.data
    if not isinstance(import_price_growth.index, pd.PeriodIndex):
        import_price_growth.index = pd.PeriodIndex(import_price_growth.index, freq="Q")

    # Build DataFrame for alignment
    df = pd.DataFrame({
        "inflation": inflation,
        "wage_inflation": wage_inflation,
        "U": ur,
        "cash_rate": cash_rate,
        "inflation_raw": inflation_annual,
        "import_price_growth": import_price_growth,
    })

    # Drop NaNs and filter to date range
    df = df.dropna()
    start_period = pd.Period(start, freq="Q")
    if end is not None:
        end_period = pd.Period(end, freq="Q")
        df = df[(df.index >= start_period) & (df.index <= end_period)]
    else:
        df = df[df.index >= start_period]

    # Compute real rate and lag it
    real_rate = df["cash_rate"] - df["inflation_raw"]
    real_rate_lag = real_rate.shift(1).fillna(real_rate.iloc[0])

    # Build observation matrix: [π (anchor-adjusted), π_w, U]
    y = df[["inflation", "wage_inflation", "U"]].values

    return (
        y,
        real_rate_lag.values,
        df["U"].values,
        df["import_price_growth"].values,
        df.index,
    )


def estimate_hlw_nairu(
    y: np.ndarray,
    r_lag: np.ndarray,
    U_obs: np.ndarray,
    import_price_growth: np.ndarray | None = None,
    initial_params: HLWNairuParameters | None = None,
    nairu_prior: float = 5.0,
) -> HLWNairuEstimationResult:
    """Estimate HLW-NAIRU model via MLE."""

    if initial_params is None:
        initial_params = HLWNairuParameters()

    # Initial values and bounds
    param_dict = initial_params.to_dict()

    # Parameters to estimate
    estimate_params = [
        "rho_y", "beta_r", "gamma_p", "gamma_w", "rho_m", "omega",
        "sigma_demand", "sigma_supply", "sigma_wage", "sigma_rstar", "sigma_nairu",
    ]
    x0 = np.array([param_dict[name] for name in estimate_params])
    bounds = [HLW_NAIRU_PARAM_BOUNDS[name] for name in estimate_params]

    def neg_ll(x):
        for i, name in enumerate(estimate_params):
            param_dict[name] = x[i]
        params = HLWNairuParameters(**param_dict)
        ll = compute_hlw_nairu_log_likelihood(
            y, r_lag, U_obs, params, import_price_growth, nairu_prior
        )
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
    final_params = HLWNairuParameters(**param_dict)

    return HLWNairuEstimationResult(
        params=final_params,
        log_likelihood=-result.fun,
        convergence={"success": result.success, "nit": result.nit, "nfev": result.nfev},
        n_obs=len(y),
    )


def estimate_hlw_nairu_regimes(
    start: str = "1984Q1",
    anchor_inflation: bool = True,
    nairu_prior: float = 5.0,
) -> dict:
    """Estimate HLW-NAIRU model separately for each regime."""

    # Define regimes
    regimes = [
        ("Pre-GFC", "1984Q1", "2008Q3"),
        ("GFC-COVID", "2008Q4", "2020Q4"),
        ("Post-COVID", "2021Q1", None),
    ]

    results = {}
    total_ll = 0

    for name, reg_start, reg_end in regimes:
        # Use max of start and reg_start
        actual_start = max(start, reg_start)

        y, r_lag, U_obs, import_prices, dates = load_hlw_nairu_data(
            start=actual_start, end=reg_end, anchor_inflation=anchor_inflation
        )

        if len(y) < 10:
            print(f"  {name}: insufficient data ({len(y)} obs), skipping")
            continue

        # Use mean U as NAIRU prior for this regime
        regime_nairu_prior = np.mean(U_obs)

        print(f"  Estimating {name} ({dates[0]} to {dates[-1]}, n={len(y)}, U̅={regime_nairu_prior:.1f}%)...")

        result = estimate_hlw_nairu(
            y, r_lag, U_obs, import_prices,
            nairu_prior=regime_nairu_prior,
        )
        results[name] = result
        total_ll += result.log_likelihood

        print(f"    LL: {result.log_likelihood:.2f}")

    results["total_ll"] = total_ll
    return results


def print_hlw_nairu_results(results: dict) -> None:
    """Print HLW-NAIRU estimation results."""
    print("=" * 70)
    print("HLW-NAIRU MODEL ESTIMATION RESULTS")
    print("(NAIRU latent, Phillips uses u_gap = U - NAIRU)")
    print("=" * 70)

    print(f"\n{'Param':<14} ", end="")
    for name in results:
        if name != "total_ll":
            print(f"{name:>12} ", end="")
    print()
    print("-" * 70)

    params_to_show = [
        "gamma_p", "gamma_w", "rho_m", "omega",
        "rho_y", "beta_r", "sigma_nairu", "sigma_rstar",
    ]

    for param in params_to_show:
        print(f"{param:<14} ", end="")
        for name, result in results.items():
            if name != "total_ll" and hasattr(result, "params"):
                val = getattr(result.params, param)
                print(f"{val:>12.4f} ", end="")
        print()

    print("=" * 70)
    print(f"Total log-likelihood: {results.get('total_ll', 0):.2f}")


if __name__ == "__main__":
    print("Loading data and estimating HLW-NAIRU model...")
    results = estimate_hlw_nairu_regimes(start="1984Q1", anchor_inflation=True)
    print_hlw_nairu_results(results)
