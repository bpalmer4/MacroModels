"""Regime-switching DSGE estimation with known breakpoints.

Allows parameters to vary across pre-specified regimes while
estimating via maximum likelihood with Kalman filter.
"""

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd
from scipy import optimize

from src.models.dsge.nk_model import NKModel, NKParameters, NKSolution
from src.models.dsge.kalman import kalman_filter
from src.models.dsge.estimation import PARAM_BOUNDS


@dataclass
class Regime:
    """A regime period with its own parameter values."""

    name: str
    start: pd.Period
    end: pd.Period  # inclusive
    params: NKParameters | None = None


@dataclass
class RegimeSwitchingResult:
    """Results from regime-switching DSGE estimation."""

    regimes: list[Regime]
    log_likelihood: float
    convergence: dict
    n_obs: int
    regime_likelihoods: dict[str, float]  # Per-regime contributions


# Default regimes for Australian data
def get_default_regimes() -> list[Regime]:
    """Get default Australian monetary policy regimes."""
    return [
        Regime(
            name="Pre-GFC",
            start=pd.Period("1984Q1", freq="Q"),
            end=pd.Period("2008Q3", freq="Q"),
        ),
        Regime(
            name="GFC-COVID",
            start=pd.Period("2008Q4", freq="Q"),
            end=pd.Period("2020Q4", freq="Q"),
        ),
        Regime(
            name="Post-COVID",
            start=pd.Period("2021Q1", freq="Q"),
            end=pd.Period("2030Q4", freq="Q"),  # Far future
        ),
    ]


# Parameters that can vary by regime
REGIME_VARYING_PARAMS = [
    "kappa_p",  # Phillips curve slope
    "kappa_w",  # Wage Phillips curve slope
    "omega",  # Okun's law coefficient
    "phi_pi",  # Taylor rule inflation response
    "phi_y",  # Taylor rule output response
    "sigma_demand",  # Shock volatilities
    "sigma_supply",
    "sigma_wage",
    "sigma_monetary",
]

# Parameters held constant across regimes
REGIME_FIXED_PARAMS = [
    "sigma",  # IES
    "beta",  # Discount factor
    "rho_i",  # Interest rate smoothing
    "rho_demand",  # Shock persistence
    "rho_supply",
    "rho_wage",
]


def split_data_by_regime(
    y: np.ndarray,
    dates: pd.PeriodIndex,
    regimes: list[Regime],
) -> list[tuple[np.ndarray, pd.PeriodIndex, Regime]]:
    """Split data into regime-specific chunks.

    Returns:
        List of (y_regime, dates_regime, regime) tuples

    """
    splits = []
    for regime in regimes:
        mask = (dates >= regime.start) & (dates <= regime.end)
        if mask.any():
            splits.append((y[mask], dates[mask], regime))
    return splits


def compute_regime_log_likelihood(
    y: np.ndarray,
    params: NKParameters,
    n_observables: int = 4,
) -> float:
    """Compute log-likelihood for a single regime's data."""
    try:
        model = NKModel(params=params)

        is_det, _ = model.check_determinacy()
        if not is_det:
            return -1e10

        solution = model.solve()
        T, R, Z, Q, H = model.state_space_matrices(solution, n_observables=n_observables)
        kf = kalman_filter(y, T, R, Z, Q, H=H)

        return kf.log_likelihood

    except Exception:
        return -1e10


def estimate_single_regime(
    y: np.ndarray,
    initial_params: NKParameters | None = None,
    estimate_params: list[str] | None = None,
    n_observables: int = 4,
) -> tuple[NKParameters, float, dict]:
    """Estimate parameters for a single regime.

    Returns:
        (estimated_params, log_likelihood, convergence_info)

    """
    if initial_params is None:
        initial_params = NKParameters()

    if estimate_params is None:
        estimate_params = REGIME_VARYING_PARAMS + REGIME_FIXED_PARAMS

    # Build parameter vector and bounds
    param_dict = initial_params.to_dict()
    x0 = np.array([param_dict[name] for name in estimate_params])
    bounds = [PARAM_BOUNDS[name] for name in estimate_params]

    def neg_ll(x):
        for i, name in enumerate(estimate_params):
            param_dict[name] = x[i]
        params = NKParameters(**param_dict)
        return -compute_regime_log_likelihood(y, params, n_observables)

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
    final_params = NKParameters(**param_dict)

    return (
        final_params,
        -result.fun,
        {"success": result.success, "nit": result.nit, "nfev": result.nfev},
    )


def estimate_regime_switching(
    y: np.ndarray,
    dates: pd.PeriodIndex,
    regimes: list[Regime] | None = None,
    vary_params: list[str] | None = None,
    fixed_params: list[str] | None = None,
    initial_params: NKParameters | None = None,
    n_observables: int = 4,
    joint: bool = False,
) -> RegimeSwitchingResult:
    """Estimate regime-switching DSGE model.

    Args:
        y: Observations (T × n_obs)
        dates: Period index for observations
        regimes: List of Regime objects (default: Australian policy regimes)
        vary_params: Parameters that vary by regime
        fixed_params: Parameters held constant across regimes
        initial_params: Starting values
        n_observables: Number of observables
        joint: If True, estimate all regimes jointly; if False, estimate separately

    Returns:
        RegimeSwitchingResult with regime-specific parameters

    """
    if regimes is None:
        regimes = get_default_regimes()

    if vary_params is None:
        vary_params = REGIME_VARYING_PARAMS

    if fixed_params is None:
        fixed_params = REGIME_FIXED_PARAMS

    if initial_params is None:
        initial_params = NKParameters()

    # Split data by regime
    splits = split_data_by_regime(y, dates, regimes)

    if not joint:
        # Estimate each regime separately
        total_ll = 0.0
        regime_lls = {}
        estimated_regimes = []

        for y_r, dates_r, regime in splits:
            print(f"  Estimating {regime.name} ({dates_r[0]} to {dates_r[-1]}, n={len(y_r)})...")

            params_r, ll_r, conv_r = estimate_single_regime(
                y_r,
                initial_params=initial_params,
                estimate_params=vary_params + fixed_params,
                n_observables=n_observables,
            )

            regime.params = params_r
            estimated_regimes.append(regime)
            total_ll += ll_r
            regime_lls[regime.name] = ll_r

            print(f"    Log-likelihood: {ll_r:.2f}")

        return RegimeSwitchingResult(
            regimes=estimated_regimes,
            log_likelihood=total_ll,
            convergence={"method": "separate"},
            n_obs=len(y),
            regime_likelihoods=regime_lls,
        )

    else:
        # Joint estimation with shared fixed_params
        # This is more complex - implement later if needed
        raise NotImplementedError("Joint estimation not yet implemented")


def print_regime_results(result: RegimeSwitchingResult) -> None:
    """Print formatted regime-switching results."""
    print("=" * 70)
    print("REGIME-SWITCHING DSGE ESTIMATION RESULTS")
    print("=" * 70)

    print(f"\nTotal observations: {result.n_obs}")
    print(f"Total log-likelihood: {result.log_likelihood:.2f}")
    print(f"Number of regimes: {len(result.regimes)}")

    for regime in result.regimes:
        if regime.params is None:
            continue

        print("\n" + "-" * 70)
        print(f"REGIME: {regime.name} ({regime.start} to {regime.end})")
        ll = result.regime_likelihoods.get(regime.name, 0)
        print(f"Log-likelihood contribution: {ll:.2f}")
        print("-" * 70)

        p = regime.params

        print("\nPhillips curves:")
        print(f"  kappa_p (price):     {p.kappa_p:8.4f}")
        print(f"  kappa_w (wage):      {p.kappa_w:8.4f}")

        print("\nTaylor rule:")
        print(f"  phi_pi:              {p.phi_pi:8.4f}")
        print(f"  phi_y:               {p.phi_y:8.4f}")
        print(f"  rho_i (smoothing):   {p.rho_i:8.4f}")

        print("\nShock persistence:")
        print(f"  rho_demand:          {p.rho_demand:8.4f}")
        print(f"  rho_supply:          {p.rho_supply:8.4f}")
        print(f"  rho_wage:            {p.rho_wage:8.4f}")

        print("\nShock volatilities:")
        print(f"  sigma_demand:        {p.sigma_demand:8.4f}")
        print(f"  sigma_supply:        {p.sigma_supply:8.4f}")
        print(f"  sigma_wage:          {p.sigma_wage:8.4f}")
        print(f"  sigma_monetary:      {p.sigma_monetary:8.4f}")

    # Summary comparison table
    print("\n" + "=" * 70)
    print("PARAMETER COMPARISON ACROSS REGIMES")
    print("=" * 70)

    params_to_compare = ["kappa_p", "kappa_w", "phi_pi", "phi_y", "sigma_demand", "sigma_supply"]

    # Header
    header = f"{'Parameter':<15}"
    for regime in result.regimes:
        if regime.params:
            header += f"{regime.name:>12}"
    print(header)
    print("-" * 70)

    # Values
    for param_name in params_to_compare:
        row = f"{param_name:<15}"
        for regime in result.regimes:
            if regime.params:
                val = getattr(regime.params, param_name)
                row += f"{val:>12.4f}"
        print(row)

    print("=" * 70)


if __name__ == "__main__":
    from src.models.dsge.data_loader import get_estimation_arrays

    print("Loading data from 1984Q1...")
    y, dates = get_estimation_arrays(start="1984Q1", n_observables=4)
    print(f"Data: {len(y)} observations from {dates[0]} to {dates[-1]}")

    print("\nEstimating regime-switching model...")
    result = estimate_regime_switching(y, dates, n_observables=4)

    print_regime_results(result)
